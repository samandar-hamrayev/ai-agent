from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import fakeredis
import pytest

from agent.core import Agent, AgentConfig
from agent.memory import ShortTermMemory
from agent.planner import PlannerResponse, ToolCall
from agent.storage import SqlRunStore
from agent.tool_router import ToolRegistry
from tools.base import BaseTool, ToolResult


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the message back."

    def run(self, message: str) -> ToolResult:
        return ToolResult(success=True, output=f"echo: {message}")

    def get_input_schema(self) -> dict:
        return {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}


class FailingTool(BaseTool):
    name = "boom"
    description = "Always raises."

    def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("tool exploded")

    def get_input_schema(self) -> dict:
        return {"type": "object", "properties": {}}


def text_turn(text: str, stop_reason: str = "end_turn") -> PlannerResponse:
    return PlannerResponse(
        text=text,
        stop_reason=stop_reason,
        content=[{"type": "text", "text": text}] if text else [],
        model="claude-opus-5",
    )


def tool_turn(*calls: tuple[str, str, dict], text: str | None = None) -> PlannerResponse:
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "id": cid, "name": name, "input": data} for cid, name, data in calls]
    return PlannerResponse(
        text=text,
        tool_calls=[ToolCall(id=cid, name=name, input=data) for cid, name, data in calls],
        stop_reason="tool_use",
        content=content,
        model="claude-opus-5",
    )


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture
def stm(redis) -> ShortTermMemory:
    return ShortTermMemory(redis, ttl=60)


@pytest.fixture
async def store(tmp_path) -> AsyncIterator[SqlRunStore]:
    s = SqlRunStore.from_url(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await s.create_tables()
    yield s
    await s.close()


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(EchoTool())
    r.register(FailingTool())
    return r


@pytest.fixture
def make_agent(registry, stm, store):
    """Build an Agent whose planner replays a scripted list of Claude turns."""

    def _make(turns: list[PlannerResponse], max_iterations: int = 5, ltm=None) -> Agent:
        planner = AsyncMock()
        planner.think = AsyncMock(side_effect=turns)
        return Agent(
            tool_registry=registry,
            short_term_memory=stm,
            long_term_memory=ltm,
            planner=planner,
            run_store=store,
            config=AgentConfig(max_iterations=max_iterations),
        )

    return _make
