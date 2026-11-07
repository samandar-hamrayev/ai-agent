"""Tests for the ReAct loop and the ToolRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.core import AgentStep
from agent.tool_router import ToolRegistry
from tests.conftest import EchoTool, text_turn, tool_turn


async def collect(agent, goal: str, **kwargs) -> list[AgentStep]:
    return [step async for step in agent.run(goal, **kwargs)]


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

    def test_execute_bad_arguments(self, registry):
        result = registry.execute("echo", {"wrong": 1})
        assert not result.success

    def test_schemas_sorted_for_cache_stability(self, registry):
        assert [s["name"] for s in registry.get_all_schemas()] == ["boom", "echo"]

    async def test_aexecute_many_keeps_order(self, registry):
        results = await registry.aexecute_many([("echo", {"message": "a"}), ("boom", {}), ("echo", {"message": "b"})])
        assert [r.success for r in results] == [True, False, True]
        assert results[2].output == "echo: b"


class TestAgentLoop:
    async def test_direct_answer(self, make_agent, store):
        agent = make_agent([text_turn("The answer is 42.")])
        steps = await collect(agent, "What is the answer?", run_id="r1")

        assert len(steps) == 1
        assert steps[0].is_final and steps[0].final_answer == "The answer is 42."
        run = await store.get_run("r1")
        assert run.status == "completed"
        assert run.model == "claude-opus-5"
        assert len(run.steps) == 1

    async def test_tool_then_answer(self, make_agent, stm):
        agent = make_agent([tool_turn(("t1", "echo", {"message": "hi"}), text="Let me check."), text_turn("Done.")])
        steps = await collect(agent, "Use the tool", session_id="s1")

        assert [s.is_final for s in steps] == [False, True]
        assert steps[0].actions[0].result == "SUCCESS: echo: hi"

        # History must hold real blocks: user goal, assistant tool_use, user tool_result, assistant answer.
        history = await stm.get_messages_for_api("s1")
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert history[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "echo", "input": {"message": "hi"}}
        assert history[2]["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "SUCCESS: echo: hi"}]

    async def test_parallel_tool_calls_share_one_user_message(self, make_agent, stm):
        agent = make_agent(
            [
                tool_turn(("a", "echo", {"message": "1"}), ("b", "boom", {})),
                text_turn("ok"),
            ]
        )
        steps = await collect(agent, "two tools", session_id="s2")

        assert len(steps[0].actions) == 2
        results = (await stm.get_messages_for_api("s2"))[2]["content"]
        assert [r["tool_use_id"] for r in results] == ["a", "b"]
        assert "is_error" not in results[0]
        assert results[1]["is_error"] is True

    async def test_planner_receives_previous_tool_results(self, make_agent):
        agent = make_agent([tool_turn(("t1", "echo", {"message": "x"})), text_turn("fin")])
        await collect(agent, "goal", session_id="s3")

        second_call_messages = agent._planner.think.await_args_list[1].args[0]
        assert second_call_messages[-1]["content"][0]["type"] == "tool_result"

    async def test_max_iterations(self, make_agent, store):
        turns = [tool_turn((f"t{i}", "echo", {"message": str(i)})) for i in range(3)]
        agent = make_agent(turns, max_iterations=3)
        steps = await collect(agent, "loop forever", run_id="r2")

        assert len(steps) == 4
        assert steps[-1].stop_reason == "max_iterations"
        assert (await store.get_run("r2")).status == "failed"

    async def test_refusal_is_reported_and_not_memorised(self, make_agent):
        ltm = AsyncMock()
        ltm.asearch.return_value = []
        agent = make_agent([text_turn("", stop_reason="refusal")], ltm=ltm)
        steps = await collect(agent, "something disallowed")

        assert steps[0].stop_reason == "refusal"
        assert "declined" in steps[0].final_answer
        ltm.astore.assert_not_awaited()

    async def test_planner_error_ends_run(self, make_agent, store):
        from agent.planner import PlannerResponse

        agent = make_agent([PlannerResponse(stop_reason="error", error="Claude API error (500).")])
        steps = await collect(agent, "goal", run_id="r3")
        assert steps[0].final_answer == "Claude API error (500)."
        assert (await store.get_run("r3")).status == "failed"

    async def test_long_term_memory_recall_and_store(self, make_agent):
        ltm = AsyncMock()
        ltm.asearch.return_value = [{"text": "Goal: x\nAnswer: y"}]
        agent = make_agent([text_turn("answer")], ltm=ltm)
        await collect(agent, "question")

        memory_context = agent._planner.think.await_args.args[2]
        assert "Goal: x" in memory_context
        ltm.astore.assert_awaited_once()

    async def test_memory_failure_does_not_break_run(self, make_agent):
        ltm = AsyncMock()
        ltm.asearch.side_effect = RuntimeError("chroma down")
        ltm.astore.side_effect = RuntimeError("chroma down")
        agent = make_agent([text_turn("still works")], ltm=ltm)
        steps = await collect(agent, "q")
        assert steps[-1].final_answer == "still works"
