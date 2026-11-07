from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import fakeredis
import pytest

from agent.approvals import RedisApprovalGate
from agent.core import Agent, AgentConfig
from agent.memory import ShortTermMemory
from agent.planner import PlannerResponse, TextDelta, ToolCall
from agent.storage import SqlRunStore
from agent.tool_router import ToolRegistry
from agent.usage import Usage
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


class WriterTool(BaseTool):
    """Side-effecting test tool: always needs approval."""

    name = "writer"
    description = "Pretends to write something."
    requires_approval = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, text: str) -> ToolResult:
        self.calls.append(text)
        return ToolResult(success=True, output=f"wrote {text}")

    def get_input_schema(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


USAGE = Usage.from_counts("claude-opus-5", input_tokens=1000, output_tokens=200, cache_read=4000, cache_write=0)


def text_turn(text: str, stop_reason: str = "end_turn") -> PlannerResponse:
    return PlannerResponse(
        text=text,
        stop_reason=stop_reason,
        content=[{"type": "text", "text": text}] if text else [],
        model="claude-opus-5",
        usage=USAGE,
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
        usage=USAGE,
    )


class ScriptedPlanner:
    """Stands in for Planner: replays scripted turns, streaming each turn's text first."""

    def __init__(self, turns: list[PlannerResponse]) -> None:
        self._turns = iter(turns)
        self.calls: list[dict[str, Any]] = []

    async def stream(self, messages, tool_schemas, memory_context=""):
        self.calls.append({"messages": messages, "tools": tool_schemas, "memory": memory_context})
        turn = next(self._turns)
        if turn.text:
            for word in turn.text.split(" "):
                yield TextDelta(text=word + " ")
        yield turn


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
    r.register(WriterTool())
    return r


@pytest.fixture
def gate(redis) -> RedisApprovalGate:
    return RedisApprovalGate(redis)


@pytest.fixture
def make_agent(registry, stm, store, gate):
    """Build an Agent whose planner replays a scripted list of Claude turns."""

    def _make(
        turns: list[PlannerResponse],
        max_iterations: int = 5,
        ltm=None,
        approval_timeout: int = 5,
        require_approval: bool = True,
        with_gate: bool = True,
    ) -> Agent:
        return Agent(
            tool_registry=registry,
            short_term_memory=stm,
            long_term_memory=ltm,
            planner=ScriptedPlanner(turns),
            run_store=store,
            config=AgentConfig(
                max_iterations=max_iterations,
                approval_timeout_seconds=approval_timeout,
                require_approval=require_approval,
            ),
            approval_gate=gate if with_gate else None,
        )

    return _make
