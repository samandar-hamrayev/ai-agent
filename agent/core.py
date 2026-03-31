from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from pydantic import BaseModel, Field

from agent.memory import LongTermMemory, ShortTermMemory
from agent.planner import Planner
from agent.tool_router import ToolRegistry

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=50)
    model: str = "claude-sonnet-4-20250514"
    save_to_long_term: bool = True


class AgentStep(BaseModel):
    """One iteration of the ReAct loop."""

    step_number: int
    thought: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: str | None = None
    is_final: bool = False
    final_answer: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __str__(self) -> str:
        if self.is_final:
            return f"Step {self.step_number} [FINAL]: {self.final_answer}"
        return f"Step {self.step_number}: {self.tool_name}({self.tool_input}) → {self.tool_result}"


class Agent:
    """
    ReAct agent — loops through THINK → DECIDE → ACT → OBSERVE
    until the goal is achieved or we hit max iterations.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        planner: Planner,
        config: AgentConfig | None = None,
    ) -> None:
        self._tools = tool_registry
        self._stm = short_term_memory
        self._ltm = long_term_memory
        self._planner = planner
        self._config = config or AgentConfig()
        logger.info("Agent ready: max_iterations=%d, tools=%s", self._config.max_iterations, self._tools.tool_names)

    async def run(self, goal: str, session_id: str | None = None) -> AsyncGenerator[AgentStep, None]:
        """Execute the ReAct loop, yielding AgentStep for each iteration."""
        session_id = session_id or str(uuid.uuid4())
        logger.info("Starting run: session=%s, goal='%s'", session_id, goal[:100])

        self._stm.add_message(session_id, role="user", content=goal)
        memory_context = self._get_memory_context(goal)
        tool_schemas = self._tools.get_all_schemas()

        for step_num in range(1, self._config.max_iterations + 1):
            logger.info("--- Step %d ---", step_num)

            messages = self._stm.get_messages_for_api(session_id)
            response = self._planner.think(
                messages=messages,
                tool_schemas=tool_schemas,
                session_id=session_id,
                memory_context=memory_context,
            )

            if response.wants_tool:
                step = AgentStep(
                    step_number=step_num,
                    thought=response.text,
                    tool_name=response.tool_name,
                    tool_input=response.tool_input,
                )

                tool_result = self._tools.execute(response.tool_name, response.tool_input or {})
                step.tool_result = str(tool_result)

                # Save the assistant's tool_use message
                assistant_content = []
                if response.text:
                    assistant_content.append({"type": "text", "text": response.text})
                assistant_content.append({
                    "type": "tool_use",
                    "id": response.tool_use_id,
                    "name": response.tool_name,
                    "input": response.tool_input or {},
                })
                self._stm.add_message(session_id, role="assistant", content=assistant_content)

                # Save the tool result
                self._stm.add_message(session_id, role="user", content=[{
                    "type": "tool_result",
                    "tool_use_id": response.tool_use_id,
                    "content": str(tool_result),
                }])

                yield step

            else:
                final_text = response.text or "I was unable to determine a response."
                step = AgentStep(step_number=step_num, thought=final_text, is_final=True, final_answer=final_text)

                # Yield first so the client gets the response immediately
                yield step

                self._stm.add_message(session_id, role="assistant", content=final_text)
                if self._config.save_to_long_term:
                    self._save_to_long_term(goal, final_text, session_id)
                return

        # Ran out of iterations
        warning = f"Max iterations ({self._config.max_iterations}) reached without a final answer."
        yield AgentStep(step_number=self._config.max_iterations + 1, thought=warning, is_final=True, final_answer=warning)

    def _get_memory_context(self, query: str) -> str:
        try:
            results = self._ltm.search(query, top_k=3)
            if not results:
                return ""
            lines = ["Relevant memories from past sessions:"]
            for i, entry in enumerate(results, 1):
                lines.append(f"  {i}. {entry['text']}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Long-term memory search failed: %s", e)
            return ""

    def _save_to_long_term(self, goal: str, answer: str, session_id: str) -> None:
        try:
            self._ltm.store(
                text=f"Goal: {goal}\nAnswer: {answer}",
                metadata={"session_id": session_id, "type": "goal_completion"},
            )
        except Exception as e:
            logger.warning("Failed to save to long-term memory: %s", e)
