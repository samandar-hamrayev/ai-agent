"""Tests for request building, streaming and response parsing in the Planner (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from agent.planner import Planner, PlannerResponse, TextDelta


def block(type_: str, **fields):
    data = {"type": type_, **fields}
    return SimpleNamespace(**data, model_dump=lambda **_: dict(data))


def usage(**kw):
    base = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    return SimpleNamespace(**{**base, **kw}, iterations=kw.get("iterations"))


def message(content, stop_reason="end_turn", model="claude-opus-5", **usage_kw):
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=model, usage=usage(**usage_kw))


class FakeStream:
    """Mimics the SDK's async message stream context manager."""

    def __init__(self, texts: list[str], final, error: Exception | None = None):
        self._texts, self._final, self._error = texts, final, error

    async def __aenter__(self):
        if self._error:
            raise self._error
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for t in self._texts:
            yield SimpleNamespace(type="text", text=t)
        yield SimpleNamespace(type="message_stop")

    async def get_final_message(self):
        return self._final


def make_planner(streams: list[FakeStream], **kwargs) -> tuple[Planner, MagicMock]:
    client = MagicMock()
    client.beta.messages.stream = MagicMock(side_effect=streams)
    return Planner(client=client, model="claude-opus-5", **kwargs), client


def test_request_uses_adaptive_thinking_effort_and_cache():
    planner, _ = make_planner([], effort="xhigh")
    request = planner.build_request([{"role": "user", "content": "hi"}], [{"name": "t"}], "remember this")

    assert request["model"] == "claude-opus-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "xhigh"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["cache_control"] == {"type": "ephemeral"}  # rolling breakpoint on the conversation
    assert "remember this" in request["system"][0]["text"]
    assert request["fallbacks"] == "default"
    assert request["tools"] == [{"name": "t", "eager_input_streaming": True}]


def test_request_without_tools_or_fallbacks():
    planner, _ = make_planner([], fallbacks=False)
    request = planner.build_request([], [])
    assert "tools" not in request
    assert "fallbacks" not in request and "betas" not in request


async def test_stream_yields_text_then_response_with_usage():
    final = message([block("text", text="Hello world")], cache_read_input_tokens=900)
    planner, _ = make_planner([FakeStream(["Hello ", "world"], final)])

    events = [e async for e in planner.stream([], [])]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hello ", "world"]
    response = events[-1]
    assert isinstance(response, PlannerResponse)
    assert response.text == "Hello world"
    assert response.usage.cache_read_tokens == 900
    assert response.usage.cost_usd > 0


async def test_unparseable_tool_json_is_retried():
    final = message([block("text", text="ok")])
    planner, client = make_planner([FakeStream([], None, ValueError("bad json")), FakeStream([], final)])
    response = await planner.think([], [])
    assert response.text == "ok"
    assert client.beta.messages.stream.call_count == 2


async def test_unparseable_tool_json_gives_up_after_retries():
    planner, _ = make_planner([FakeStream([], None, ValueError("bad json")) for _ in range(3)])
    response = await planner.think([], [])
    assert response.stop_reason == "error"


def test_parse_collects_all_tool_calls_and_keeps_thinking():
    response = message(
        [
            block("thinking", thinking="", signature="sig"),
            block("text", text="Checking two things."),
            block("tool_use", id="a", name="calculator", input={"expression": "1+1"}),
            block("tool_use", id="b", name="web_search", input={"query": "x"}),
        ],
        stop_reason="tool_use",
    )
    parsed = Planner.parse(response)

    assert parsed.wants_tool
    assert [c.id for c in parsed.tool_calls] == ["a", "b"]
    assert parsed.text == "Checking two things."
    assert parsed.content[0] == {"type": "thinking", "thinking": "", "signature": "sig"}
    assert len(parsed.content) == 4


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_truncated_or_refused_tool_calls_are_dropped(stop_reason):
    response = message(
        [block("text", text="Let me"), block("tool_use", id="a", name="file_tool", input={"path": "x"})],
        stop_reason=stop_reason,
    )
    parsed = Planner.parse(response)
    assert not parsed.wants_tool
    assert [b["type"] for b in parsed.content] == ["text"]


def test_parse_drops_fallback_marker_and_partial_work_before_it():
    response = message(
        [
            block("thinking", thinking="", signature="s1"),
            block("text", text="partial "),
            block("fallback"),
            block("text", text="answer"),
        ],
        model="claude-opus-4-8",
    )
    parsed = Planner.parse(response)
    assert [b["type"] for b in parsed.content] == ["text", "text"]
    assert parsed.model == "claude-opus-4-8"


def test_parse_refusal():
    parsed = Planner.parse(message([], stop_reason="refusal"))
    assert parsed.refused and not parsed.wants_tool and parsed.text is None


async def test_api_errors_become_error_responses():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIStatusError("boom", response=httpx.Response(500, request=request), body=None)
    planner, _ = make_planner([FakeStream([], None, error)])
    result = await planner.think([], [])
    assert result.stop_reason == "error"
    assert "500" in result.error


async def test_connection_error():
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    planner, _ = make_planner([FakeStream([], None, error)])
    result = await planner.think([], [])
    assert result.error == "Could not reach the Claude API."
