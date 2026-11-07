"""Tests for the ReAct loop, approvals, usage accounting and the ToolRegistry."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from agent.approvals import ApprovalDecision
from agent.core import AgentStep, ApprovalRequest, TextChunk
from agent.planner import PlannerResponse
from agent.tool_router import ToolRegistry
from tests.conftest import USAGE, EchoTool, text_turn, tool_turn


async def collect(agent, goal: str, **kwargs) -> list:
    return [event async for event in agent.run(goal, **kwargs)]


def steps_of(events: list) -> list[AgentStep]:
    return [e for e in events if isinstance(e, AgentStep)]


class TestToolRegistry:
    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(EchoTool())

    def test_execute_unknown_tool(self, registry):
        result = registry.execute("nope", {})
        assert not result.success
        assert "not found" in result.error

    def test_execute_catches_exceptions(self, registry):
        result = registry.execute("boom", {})
        assert not result.success
        assert "tool exploded" in result.error

    def test_schema_validation_rejects_bad_input(self, registry):
        result = registry.execute("echo", {"message": 42})
        assert not result.success
        assert "Invalid input for 'echo'" in result.error
        assert "message" in result.error

    def test_schema_validation_requires_fields(self, registry):
        assert "required" in registry.validate("echo", {})
        assert registry.validate("echo", "not a dict").startswith("Tool input must be a JSON object")
        assert registry.validate("echo", {"message": "ok"}) is None

    def test_needs_approval(self, registry):
        assert registry.needs_approval("writer", {"text": "x"})
        assert not registry.needs_approval("echo", {"message": "x"})
        assert not registry.needs_approval("missing", {})

    def test_schemas_sorted_for_cache_stability(self, registry):
        assert [s["name"] for s in registry.get_all_schemas()] == ["boom", "echo", "writer"]

    async def test_aexecute_many_keeps_order(self, registry):
        results = await registry.aexecute_many([("echo", {"message": "a"}), ("boom", {}), ("echo", {"message": "b"})])
        assert [r.success for r in results] == [True, False, True]
        assert results[2].output == "echo: b"


class TestAgentLoop:
    async def test_direct_answer_streams_text(self, make_agent, store):
        agent = make_agent([text_turn("The answer is 42.")])
        events = await collect(agent, "What is the answer?", run_id="r1")

        chunks = [e.text for e in events if isinstance(e, TextChunk)]
        assert "".join(chunks).strip() == "The answer is 42."
        assert isinstance(events[-1], AgentStep) and events[-1].is_final
        run = await store.get_run("r1")
        assert run.status == "completed"
        assert run.model == "claude-opus-5"

    async def test_tool_then_answer(self, make_agent, stm):
        agent = make_agent([tool_turn(("t1", "echo", {"message": "hi"}), text="Let me check."), text_turn("Done.")])
        steps = steps_of(await collect(agent, "Use the tool", session_id="s1"))

        assert [s.is_final for s in steps] == [False, True]
        assert steps[0].actions[0].result == "SUCCESS: echo: hi"
        assert steps[0].actions[0].approval == "not_required"

        history = await stm.get_messages_for_api("s1")
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert history[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "echo", "input": {"message": "hi"}}
        assert history[2]["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "SUCCESS: echo: hi"}]

    async def test_parallel_tool_calls_share_one_user_message(self, make_agent, stm):
        agent = make_agent([tool_turn(("a", "echo", {"message": "1"}), ("b", "boom", {})), text_turn("ok")])
        steps = steps_of(await collect(agent, "two tools", session_id="s2"))

        assert len(steps[0].actions) == 2
        results = (await stm.get_messages_for_api("s2"))[2]["content"]
        assert [r["tool_use_id"] for r in results] == ["a", "b"]
        assert "is_error" not in results[0]
        assert results[1]["is_error"] is True

    async def test_invalid_tool_input_is_rejected_before_running(self, make_agent):
        agent = make_agent([tool_turn(("t1", "echo", {"message": 123})), text_turn("fixed")])
        action = steps_of(await collect(agent, "goal"))[0].actions[0]
        assert action.approval == "invalid_input"
        assert not action.success and "Invalid input" in action.result

    async def test_max_iterations(self, make_agent, store):
        turns = [tool_turn((f"t{i}", "echo", {"message": str(i)})) for i in range(3)]
        agent = make_agent(turns, max_iterations=3)
        steps = steps_of(await collect(agent, "loop forever", run_id="r2"))

        assert len(steps) == 4
        assert steps[-1].stop_reason == "max_iterations"
        assert (await store.get_run("r2")).status == "failed"

    async def test_refusal_is_reported_and_not_memorised(self, make_agent):
        ltm = AsyncMock()
        ltm.asearch.return_value = []
        agent = make_agent([text_turn("", stop_reason="refusal")], ltm=ltm)
        final = steps_of(await collect(agent, "something disallowed"))[-1]

        assert final.stop_reason == "refusal"
        assert "declined" in final.final_answer
        ltm.astore.assert_not_awaited()

    async def test_planner_error_ends_run(self, make_agent, store):
        agent = make_agent([PlannerResponse(stop_reason="error", error="Claude API error (500).")])
        final = steps_of(await collect(agent, "goal", run_id="r3"))[-1]
        assert final.final_answer == "Claude API error (500)."
        assert (await store.get_run("r3")).status == "failed"

    async def test_long_term_memory_recall_and_store(self, make_agent):
        ltm = AsyncMock()
        ltm.asearch.return_value = [{"text": "Goal: x\nAnswer: y"}]
        agent = make_agent([text_turn("answer")], ltm=ltm)
        await collect(agent, "question")

        assert "Goal: x" in agent._planner.calls[0]["memory"]
        ltm.astore.assert_awaited_once()

    async def test_memory_failure_does_not_break_run(self, make_agent):
        ltm = AsyncMock()
        ltm.asearch.side_effect = RuntimeError("chroma down")
        ltm.astore.side_effect = RuntimeError("chroma down")
        agent = make_agent([text_turn("still works")], ltm=ltm)
        assert steps_of(await collect(agent, "q"))[-1].final_answer == "still works"


class TestUsageAccounting:
    async def test_run_totals_are_summed_and_persisted(self, make_agent, store):
        agent = make_agent([tool_turn(("t1", "echo", {"message": "hi"})), text_turn("done")])
        final = steps_of(await collect(agent, "goal", run_id="u1"))[-1]

        expected = USAGE + USAGE
        assert final.run_usage.input_tokens == expected.input_tokens
        assert final.run_usage.cost_usd == expected.cost_usd
        run = await store.get_run("u1")
        assert (run.input_tokens, run.output_tokens, run.cache_read_tokens) == (2000, 400, 8000)
        assert Decimal(run.cost_usd) == expected.cost_usd
        assert run.steps[0].usage["output_tokens"] == 200


class TestApprovals:
    async def _run_with_decision(self, agent, gate, decision: ApprovalDecision | None, **kwargs):
        """Drive a run; when an approval is requested, submit `decision` (or nothing)."""
        events = []
        async for event in agent.run("write something", **kwargs):
            events.append(event)
            if isinstance(event, ApprovalRequest) and decision is not None:
                assert await gate.submit(event.run_id, event.tool_use_id, decision)
        return events

    async def test_approved_call_runs(self, make_agent, gate, registry):
        agent = make_agent([tool_turn(("w1", "writer", {"text": "hello"})), text_turn("written")])
        events = await self._run_with_decision(agent, gate, ApprovalDecision(decision="approved"))

        request = next(e for e in events if isinstance(e, ApprovalRequest))
        assert (request.tool_name, request.tool_input) == ("writer", {"text": "hello"})
        action = steps_of(events)[0].actions[0]
        assert action.approval == "approved" and action.success
        assert registry.get("writer").calls == ["hello"]

    async def test_denied_call_is_not_run_and_claude_is_told(self, make_agent, gate, registry, stm):
        agent = make_agent([tool_turn(("w1", "writer", {"text": "x"})), text_turn("ok, skipped")])
        events = await self._run_with_decision(
            agent, gate, ApprovalDecision(decision="denied", reason="not now"), session_id="d1"
        )

        action = steps_of(events)[0].actions[0]
        assert action.approval == "denied" and not action.success
        assert "declined" in action.result and "not now" in action.result
        assert registry.get("writer").calls == []
        tool_result = (await stm.get_messages_for_api("d1"))[2]["content"][0]
        assert tool_result["is_error"] is True

    async def test_timeout_counts_as_denial(self, make_agent, gate, registry):
        agent = make_agent([tool_turn(("w1", "writer", {"text": "x"})), text_turn("gave up")], approval_timeout=1)
        events = await self._run_with_decision(agent, gate, None)

        action = steps_of(events)[0].actions[0]
        assert action.approval == "timed_out"
        assert registry.get("writer").calls == []

    async def test_mixed_turn_only_waits_for_flagged_calls(self, make_agent, gate, registry):
        agent = make_agent(
            [tool_turn(("e1", "echo", {"message": "free"}), ("w1", "writer", {"text": "gated"})), text_turn("ok")]
        )
        events = await self._run_with_decision(agent, gate, ApprovalDecision(decision="approved"))
        requests = [e for e in events if isinstance(e, ApprovalRequest)]
        assert [r.tool_use_id for r in requests] == ["w1"]
        assert [a.approval for a in steps_of(events)[0].actions] == ["not_required", "approved"]

    async def test_without_gate_flagged_calls_are_denied(self, make_agent, registry):
        agent = make_agent([tool_turn(("w1", "writer", {"text": "x"})), text_turn("ok")], with_gate=False)
        action = steps_of(await collect(agent, "goal"))[0].actions[0]
        assert action.approval == "denied"
        assert registry.get("writer").calls == []

    async def test_approval_can_be_disabled(self, make_agent, registry):
        agent = make_agent([tool_turn(("w1", "writer", {"text": "x"})), text_turn("ok")], require_approval=False)
        events = await collect(agent, "goal")
        assert not any(isinstance(e, ApprovalRequest) for e in events)
        assert registry.get("writer").calls == ["x"]

    async def test_decision_is_single_use(self, gate):
        await gate.open("r", "t", 5)
        assert await gate.submit("r", "t", ApprovalDecision(decision="approved"))
        assert not await gate.submit("r", "t", ApprovalDecision(decision="denied"))
        assert not await gate.submit("r", "unknown", ApprovalDecision(decision="approved"))


class TestCancellation:
    async def test_disconnect_while_waiting_keeps_session_valid(self, make_agent, stm, store):
        agent = make_agent([tool_turn(("w1", "writer", {"text": "x"}))], approval_timeout=30)
        gen = agent.run("write", session_id="c1", run_id="c1")

        async def consume_until_approval():
            async for event in gen:
                if isinstance(event, ApprovalRequest):
                    break

        await consume_until_approval()
        # The client goes away while the agent waits for a human: the generator is closed.
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        history = await stm.get_messages_for_api("c1")
        assert history[-1]["role"] == "user"
        assert history[-1]["content"][0]["tool_use_id"] == "w1"
        assert history[-1]["content"][0]["is_error"] is True
        assert (await store.get_run("c1")).status == "cancelled"
