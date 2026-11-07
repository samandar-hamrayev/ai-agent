from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.memory import LongTermMemory, ShortTermMemory
from agent.planner import Planner, PlannerResponse
from agent.storage import NullRunStore, RunStore
from agent.tool_router import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 20_000


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=50)
    save_to_long_term: bool = True


class ToolAction(BaseModel):
    """A single tool call made during a step and what it returned."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    success: bool
    result: str


class AgentStep(BaseModel):
    """One iteration of the ReAct loop."""

    run_id: str = ""
    step_number: int
    thought: str | None = None
    actions: list[ToolAction] = Field(default_factory=list)
    is_final: bool = False
    final_answer: str | None = None
    stop_reason: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __str__(self) -> str:
        if self.is_final:
            return f"Step {self.step_number} [FINAL]: {self.final_answer}"
        calls = ", ".join(f"{a.tool_name}({a.tool_input}) → {a.result[:80]}" for a in self.actions)
        return f"Step {self.step_number}: {calls}"


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
    ) -> None:
        self._tools = tool_registry
        self._stm = short_term_memory
        self._ltm = long_term_memory
        self._planner = planner
        self._store = run_store or NullRunStore()
        self._config = config or AgentConfig()
        logger.info("Agent ready: max_iterations=%d, tools=%s", self._config.max_iterations, self._tools.tool_names)

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    async def run(
        self, goal: str, session_id: str | None = None, run_id: str | None = None
    ) -> AsyncGenerator[AgentStep, None]:
        """Execute the ReAct loop, yielding an AgentStep after every iteration."""
        session_id = session_id or str(uuid.uuid4())
        run_id = run_id or str(uuid.uuid4())
        logger.info("Run %s started (session=%s): %.100s", run_id, session_id, goal)

        await self._store.start_run(run_id, session_id, goal)
        await self._stm.add_message(session_id, role="user", content=goal)
        memory_context = await self._recall(goal)
        tool_schemas = self._tools.get_all_schemas()
        model: str | None = None

        try:
            for step_num in range(1, self._config.max_iterations + 1):
                messages = await self._stm.get_messages_for_api(session_id)
                response = await self._planner.think(messages, tool_schemas, memory_context)
                model = response.model or model

                if response.error:
                    step = self._final(run_id, step_num, response.error, "error")
                    await self._finish(run_id, step, "failed", model)
                    yield step
                    return

                # Replay Claude's turn exactly as received (thinking + text + tool_use blocks).
                if response.content:
                    await self._stm.add_message(session_id, role="assistant", content=response.content)

                if response.wants_tool:
                    step = await self._act(run_id, step_num, response)
                    await self._stm.add_message(
                        session_id, role="user", content=[self._tool_result_block(a) for a in step.actions]
                    )
                    await self._store.add_step(run_id, step.model_dump())
                    yield step
                    continue

                answer = response.text or self._fallback_answer(response)
                step = self._final(run_id, step_num, answer, response.stop_reason)
                await self._finish(run_id, step, "completed" if not response.refused else "failed", model)
                yield step
                if self._config.save_to_long_term and not response.refused:
                    await self._remember(goal, answer, session_id)
                return

            warning = f"Max iterations ({self._config.max_iterations}) reached without a final answer."
            step = self._final(run_id, self._config.max_iterations + 1, warning, "max_iterations")
            await self._finish(run_id, step, "failed", model)
            yield step
        except Exception:
            await self._store.finish_run(run_id, "failed", None, model)
            raise

    async def _act(self, run_id: str, step_num: int, response: PlannerResponse) -> AgentStep:
        """Execute every tool call from this turn concurrently."""
        calls = response.tool_calls
        results = await self._tools.aexecute_many([(c.name, c.input) for c in calls])
        actions = [
            ToolAction(
                tool_use_id=call.id,
                tool_name=call.name,
                tool_input=call.input,
                success=result.success,
                result=str(result)[:_MAX_TOOL_RESULT_CHARS],
            )
            for call, result in zip(calls, results, strict=True)
        ]
        return AgentStep(run_id=run_id, step_number=step_num, thought=response.text, actions=actions)

    @staticmethod
    def _tool_result_block(action: ToolAction) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": action.tool_use_id, "content": action.result}
        if not action.success:
            block["is_error"] = True
        return block

    @staticmethod
    def _final(run_id: str, step_num: int, answer: str, stop_reason: str | None) -> AgentStep:
        return AgentStep(
            run_id=run_id,
            step_number=step_num,
            thought=answer,
            is_final=True,
            final_answer=answer,
            stop_reason=stop_reason,
        )

    @staticmethod
    def _fallback_answer(response: PlannerResponse) -> str:
        if response.refused:
            return "The model declined to help with this request."
        if response.stop_reason == "max_tokens":
            return "The response hit the max_tokens limit before an answer was produced."
        return "I was unable to determine a response."

    async def _finish(self, run_id: str, step: AgentStep, status: str, model: str | None) -> None:
        await self._store.add_step(run_id, step.model_dump())
        await self._store.finish_run(run_id, status, step.final_answer, model)

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
