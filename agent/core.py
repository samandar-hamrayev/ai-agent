from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.approvals import ApprovalDecision, ApprovalGate
from agent.memory import LongTermMemory, ShortTermMemory
from agent.planner import Planner, PlannerResponse, TextDelta, ToolCall
from agent.storage import NullRunStore, RunStore
from agent.tool_router import ToolRegistry
from agent.usage import Usage

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 20_000

ApprovalStatus = Literal["not_required", "approved", "denied", "timed_out", "invalid_input"]


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=50)
    save_to_long_term: bool = True
    # When False, tools flagged requires_approval run without asking (e.g. offline evals).
    require_approval: bool = True
    approval_timeout_seconds: int = Field(default=300, ge=1, le=3600)


class ToolAction(BaseModel):
    """A single tool call made during a step and what it returned."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    success: bool
    result: str
    approval: ApprovalStatus = "not_required"


class AgentStep(BaseModel):
    """One iteration of the ReAct loop."""

    type: Literal["step"] = "step"
    run_id: str = ""
    step_number: int
    thought: str | None = None
    actions: list[ToolAction] = Field(default_factory=list)
    is_final: bool = False
    final_answer: str | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    # Set on the final step only: totals for the whole run.
    run_usage: Usage | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __str__(self) -> str:
        if self.is_final:
            return f"Step {self.step_number} [FINAL]: {self.final_answer}"
        calls = ", ".join(f"{a.tool_name}({a.tool_input}) → {a.result[:80]}" for a in self.actions)
        return f"Step {self.step_number}: {calls}"


class TextChunk(BaseModel):
    """Claude's visible text for the current step, streamed as it is generated."""

    type: Literal["text"] = "text"
    run_id: str
    step_number: int
    text: str


class ApprovalRequest(BaseModel):
    """The agent is paused until a human approves or denies this tool call."""

    type: Literal["approval_required"] = "approval_required"
    run_id: str
    step_number: int
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    timeout_seconds: int


AgentEvent = TextChunk | ApprovalRequest | AgentStep


class Agent:
    """ReAct agent: THINK → ACT → OBSERVE until Claude answers or the iteration budget runs out."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory | None,
        planner: Planner,
        run_store: RunStore | None = None,
        config: AgentConfig | None = None,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        self._tools = tool_registry
        self._stm = short_term_memory
        self._ltm = long_term_memory
        self._planner = planner
        self._store = run_store or NullRunStore()
        self._config = config or AgentConfig()
        self._gate = approval_gate
        logger.info("Agent ready: max_iterations=%d, tools=%s", self._config.max_iterations, self._tools.tool_names)

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    async def run(
        self, goal: str, session_id: str | None = None, run_id: str | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute the ReAct loop.

        Yields TextChunk while Claude writes, ApprovalRequest when a tool call needs a human,
        and an AgentStep after every iteration (the last one has is_final=True).
        """
        session_id = session_id or str(uuid.uuid4())
        run_id = run_id or str(uuid.uuid4())
        logger.info("Run %s started (session=%s): %.100s", run_id, session_id, goal)

        await self._store.start_run(run_id, session_id, goal)
        await self._stm.add_message(session_id, role="user", content=goal)
        memory_context = await self._recall(goal)
        tool_schemas = self._tools.get_all_schemas()
        model: str | None = None
        total = Usage()
        # tool_use ids appended to history that don't have a tool_result yet.
        unanswered: list[str] = []

        try:
            for step_num in range(1, self._config.max_iterations + 1):
                messages = await self._stm.get_messages_for_api(session_id)
                response: PlannerResponse | None = None
                async for event in self._planner.stream(messages, tool_schemas, memory_context):
                    if isinstance(event, TextDelta):
                        yield TextChunk(run_id=run_id, step_number=step_num, text=event.text)
                    else:
                        response = event
                if response is None:
                    response = PlannerResponse(stop_reason="error", error="Empty response from Claude.")

                model = response.model or model
                total += response.usage

                if response.error:
                    step = self._final(run_id, step_num, response.error, "error", response.usage, total)
                    await self._finish(run_id, step, "failed", model, total)
                    yield step
                    return

                # Replay Claude's turn exactly as received (thinking + text + tool_use blocks).
                if response.content:
                    await self._stm.add_message(session_id, role="assistant", content=response.content)
                unanswered = [c.id for c in response.tool_calls]

                if response.wants_tool:
                    actions: list[ToolAction] = []
                    async for item in self._act(run_id, step_num, response.tool_calls):
                        if isinstance(item, ApprovalRequest):
                            yield item
                        else:
                            actions = item
                    step = AgentStep(
                        run_id=run_id,
                        step_number=step_num,
                        thought=response.text,
                        actions=actions,
                        usage=response.usage,
                    )
                    await self._stm.add_message(
                        session_id, role="user", content=[self._tool_result_block(a) for a in actions]
                    )
                    unanswered = []
                    await self._store.add_step(run_id, step.model_dump(mode="json"))
                    yield step
                    continue

                answer = response.text or self._fallback_answer(response)
                step = self._final(run_id, step_num, answer, response.stop_reason, response.usage, total)
                await self._finish(run_id, step, "failed" if response.refused else "completed", model, total)
                yield step
                if self._config.save_to_long_term and not response.refused:
                    await self._remember(goal, answer, session_id)
                return

            warning = f"Max iterations ({self._config.max_iterations}) reached without a final answer."
            step = self._final(run_id, self._config.max_iterations + 1, warning, "max_iterations", Usage(), total)
            await self._finish(run_id, step, "failed", model, total)
            yield step
        except BaseException as exc:
            # Client disconnected (CancelledError) or something broke mid-step. Close every
            # open tool_use with a tool_result so the session stays valid for the next request.
            status = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "failed"
            await asyncio.shield(self._abort(session_id, run_id, unanswered, status, model, total))
            raise

    async def _abort(
        self, session_id: str, run_id: str, unanswered: list[str], status: str, model: str | None, total: Usage
    ) -> None:
        try:
            if unanswered:
                await self._stm.add_message(
                    session_id,
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Not executed: the run was {status} before this tool ran.",
                            "is_error": True,
                        }
                        for tool_use_id in unanswered
                    ],
                )
            await self._store.finish_run(run_id, status, None, model, total)
        except Exception:
            logger.exception("Failed to record %s state for run %s", status, run_id)

    async def _act(
        self, run_id: str, step_num: int, calls: list[ToolCall]
    ) -> AsyncGenerator[ApprovalRequest | list[ToolAction], None]:
        """Validate, get approvals for, and execute every tool call of one turn.

        Yields an ApprovalRequest per call that needs a human, then the list of ToolActions.
        All approvals are requested up front and awaited together, so the user sees every
        pending action at once; approved calls then run concurrently.
        """
        timeout = self._config.approval_timeout_seconds
        decisions: dict[str, ApprovalStatus] = {}
        reasons: dict[str, str | None] = {}
        invalid: dict[str, str] = {}
        to_approve: list[ToolCall] = []

        for call in calls:
            if error := self._tools.validate(call.name, call.input):
                invalid[call.id] = error
            elif self._config.require_approval and self._tools.needs_approval(call.name, call.input):
                to_approve.append(call)

        if to_approve:
            if self._gate is None:
                for call in to_approve:
                    decisions[call.id], reasons[call.id] = "denied", "No approval channel is configured."
            else:
                for call in to_approve:
                    await self._gate.open(run_id, call.id, timeout)
                    yield ApprovalRequest(
                        run_id=run_id,
                        step_number=step_num,
                        tool_use_id=call.id,
                        tool_name=call.name,
                        tool_input=call.input,
                        timeout_seconds=timeout,
                    )
                results: list[ApprovalDecision] = await asyncio.gather(
                    *(self._gate.wait(run_id, call.id, timeout) for call in to_approve)
                )
                for call, decision in zip(to_approve, results, strict=True):
                    decisions[call.id], reasons[call.id] = decision.decision, decision.reason

        runnable = [c for c in calls if c.id not in invalid and decisions.get(c.id, "approved") == "approved"]
        outputs = dict(
            zip(
                [c.id for c in runnable],
                await self._tools.aexecute_many([(c.name, c.input) for c in runnable]),
                strict=True,
            )
        )

        actions = []
        for call in calls:
            if call.id in invalid:
                success, result, approval = False, invalid[call.id], "invalid_input"
            elif call.id in outputs:
                success, result = outputs[call.id].success, str(outputs[call.id])
                approval = decisions.get(call.id, "not_required")
            else:
                status = decisions[call.id]
                reason = f" Reason: {reasons[call.id]}" if reasons.get(call.id) else ""
                verb = "did not respond in time to" if status == "timed_out" else "declined"
                success, result, approval = False, f"The user {verb} this action; it was not executed.{reason}", status
            actions.append(
                ToolAction(
                    tool_use_id=call.id,
                    tool_name=call.name,
                    tool_input=call.input,
                    success=success,
                    result=result[:_MAX_TOOL_RESULT_CHARS],
                    approval=approval,
                )
            )
        yield actions

    @staticmethod
    def _tool_result_block(action: ToolAction) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": action.tool_use_id, "content": action.result}
        if not action.success:
            block["is_error"] = True
        return block

    @staticmethod
    def _final(
        run_id: str, step_num: int, answer: str, stop_reason: str | None, usage: Usage, total: Usage
    ) -> AgentStep:
        return AgentStep(
            run_id=run_id,
            step_number=step_num,
            thought=answer,
            is_final=True,
            final_answer=answer,
            stop_reason=stop_reason,
            usage=usage,
            run_usage=total,
        )

    @staticmethod
    def _fallback_answer(response: PlannerResponse) -> str:
        if response.refused:
            return "The model declined to help with this request."
        if response.stop_reason == "max_tokens":
            return "The response hit the max_tokens limit before an answer was produced."
        return "I was unable to determine a response."

    async def _finish(self, run_id: str, step: AgentStep, status: str, model: str | None, total: Usage) -> None:
        await self._store.add_step(run_id, step.model_dump(mode="json"))
        await self._store.finish_run(run_id, status, step.final_answer, model, total)

    async def _recall(self, query: str) -> str:
        if self._ltm is None:
            return ""
        try:
            results = await self._ltm.asearch(query, top_k=3)
        except Exception as e:
            logger.warning("Long-term memory search failed: %s", e)
            return ""
        if not results:
            return ""
        lines = ["Relevant memories from past sessions:"]
        lines += [f"{i}. {entry['text']}" for i, entry in enumerate(results, 1)]
        return "\n".join(lines)

    async def _remember(self, goal: str, answer: str, session_id: str) -> None:
        if self._ltm is None:
            return
        try:
            await self._ltm.astore(
                text=f"Goal: {goal}\nAnswer: {answer}",
                metadata={"session_id": session_id, "type": "goal_completion"},
            )
        except Exception as e:
            logger.warning("Failed to save to long-term memory: %s", e)
