"""Tests for request building and response parsing in the Planner (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx

from agent.planner import Planner


def block(type_: str, **fields):
    data = {"type": type_, **fields}
    return SimpleNamespace(**data, model_dump=lambda **_: dict(data))


def make_planner(**kwargs) -> tuple[Planner, MagicMock]:
    client = MagicMock()
    client.beta.messages.create = AsyncMock()
    return Planner(client=client, model="claude-opus-5", **kwargs), client


def test_request_uses_adaptive_thinking_effort_and_cache():
    planner, _ = make_planner(effort="xhigh")
    request = planner.build_request([{"role": "user", "content": "hi"}], [{"name": "t"}], "remember this")

    assert request["model"] == "claude-opus-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "xhigh"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "remember this" in request["system"][0]["text"]
    assert request["fallbacks"] == "default"
    assert request["tools"] == [{"name": "t"}]


def test_request_without_tools_or_fallbacks():
    planner, _ = make_planner(fallbacks=False)
    request = planner.build_request([], [])
    assert "tools" not in request
    assert "fallbacks" not in request and "betas" not in request


def test_parse_collects_all_tool_calls_and_keeps_thinking():
    response = SimpleNamespace(
        stop_reason="tool_use",
        model="claude-opus-5",
        content=[
            block("thinking", thinking="", signature="sig"),
            block("text", text="Checking two things."),
            block("tool_use", id="a", name="calculator", input={"expression": "1+1"}),
            block("tool_use", id="b", name="web_search", input={"query": "x"}),
        ],
    )
    parsed = Planner.parse(response)

    assert parsed.wants_tool
    assert [c.id for c in parsed.tool_calls] == ["a", "b"]
    assert parsed.text == "Checking two things."
    assert parsed.content[0] == {"type": "thinking", "thinking": "", "signature": "sig"}
    assert len(parsed.content) == 4


def test_parse_drops_fallback_marker():
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-4-8",
        content=[block("fallback"), block("text", text="answer")],
    )
    parsed = Planner.parse(response)
    assert [b["type"] for b in parsed.content] == ["text"]
    assert parsed.model == "claude-opus-4-8"


def test_parse_refusal():
    parsed = Planner.parse(SimpleNamespace(stop_reason="refusal", model="m", content=[]))
    assert parsed.refused and not parsed.wants_tool and parsed.text is None


async def test_think_maps_api_errors():
    planner, client = make_planner()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.beta.messages.create.side_effect = anthropic.APIStatusError(
        "boom", response=httpx.Response(500, request=request), body=None
    )
    result = await planner.think([], [])
    assert result.stop_reason == "error"
    assert "500" in result.error


async def test_think_handles_connection_error():
    planner, client = make_planner()
    client.beta.messages.create.side_effect = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com")
    )
    result = await planner.think([], [])
    assert result.error == "Could not reach the Claude API."
