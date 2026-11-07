from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

import anthropic
from pydantic import BaseModel, Field

from agent.usage import Usage

logger = logging.getLogger(__name__)

# Kept free of per-request values (session ids, timestamps) so the prefix stays cacheable.
_SYSTEM_PROMPT = """You are an autonomous agent that solves the user's goal step by step.

Work in a loop: reason about what is still unknown, call a tool when it gets you closer \
to the answer, read the result, and repeat. Answer directly when you already know enough.

- If a tool fails, read the error and try a different approach instead of repeating the same call.
- If the user declines an action, respect it and continue without it or explain what you need.
- Independent tool calls can be made in parallel in a single turn.
- Finish with a clear, well-structured answer for the user.

Today's date: {today}

{memory_context}"""

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MAX_JSON_RETRIES = 2
# Blocks the model produced before a mid-output fallback that must not be replayed.
_DROP_BEFORE_FALLBACK = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class PlannerResponse(BaseModel):
    """One parsed Claude turn: free text, zero or more tool calls, and the raw content to replay."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = None
    # Raw assistant content blocks (thinking, text, tool_use) to append to history unchanged.
    content: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class TextDelta(BaseModel):
    """A fragment of Claude's visible text, streamed as it is generated."""

    text: str


class Planner:
    """Owns every Claude API call: prompt assembly, request options, streaming and parsing."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        effort: str = "high",
        max_tokens: int = 16_000,
        fallbacks: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._fallbacks = fallbacks
        logger.info("Planner ready: model=%s effort=%s fallbacks=%s", model, effort, fallbacks)

    @property
    def model(self) -> str:
        return self._model

    def build_system_prompt(self, memory_context: str = "") -> str:
        return _SYSTEM_PROMPT.format(
            today=date.today().isoformat(),
            memory_context=memory_context or "No relevant memories from earlier sessions.",
        ).strip()

    def build_request(
        self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]], memory_context: str = ""
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": self.build_system_prompt(memory_context),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
            # Automatic breakpoint on the last block: each loop iteration re-reads the
            # whole conversation so far from cache instead of paying for it again.
            "cache_control": {"type": "ephemeral"},
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
        }
        if tool_schemas:
            # Stream large tool inputs (code, file contents) as they are generated. The API no
            # longer validates them, so the registry checks every input against its schema.
            request["tools"] = [{**schema, "eager_input_streaming": True} for schema in tool_schemas]
        if self._fallbacks:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        memory_context: str = "",
    ) -> AsyncIterator[TextDelta | PlannerResponse]:
        """Stream one Claude turn: yields TextDelta fragments, then exactly one PlannerResponse."""
        request = self.build_request(messages, tool_schemas, memory_context)
        for attempt in range(_MAX_JSON_RETRIES + 1):
            try:
                async with self._client.beta.messages.stream(**request) as stream:
                    async for event in stream:
                        if event.type == "text" and event.text:
                            yield TextDelta(text=event.text)
                    message = await stream.get_final_message()
            except ValueError as e:
                # Tool input JSON the SDK couldn't parse at all; there is no tool_use_id to
                # answer, so the only option is to re-issue the turn (bounded).
                logger.warning("Unparseable tool input (attempt %d): %s", attempt + 1, e)
                if attempt == _MAX_JSON_RETRIES:
                    yield PlannerResponse(stop_reason="error", error="Claude produced an unreadable tool call.")
                    return
                continue
            except anthropic.RateLimitError as e:
                logger.warning("Claude rate limit: %s", e)
                yield PlannerResponse(stop_reason="error", error="Claude API rate limit reached, try again shortly.")
                return
            except anthropic.APIStatusError as e:
                logger.error("Claude API error %s: %s", e.status_code, e.message)
                yield PlannerResponse(stop_reason="error", error=f"Claude API error ({e.status_code}).")
                return
            except anthropic.APIConnectionError as e:
                logger.error("Claude connection error: %s", e)
                yield PlannerResponse(stop_reason="error", error="Could not reach the Claude API.")
                return
            yield self.parse(message, requested_model=self._model)
            return

    async def think(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        memory_context: str = "",
    ) -> PlannerResponse:
        """Non-streaming convenience wrapper: consume the stream and return the parsed turn."""
        async for event in self.stream(messages, tool_schemas, memory_context):
            if isinstance(event, PlannerResponse):
                return event
        return PlannerResponse(stop_reason="error", error="Empty response from Claude.")

    @staticmethod
    def parse(response: Any, requested_model: str | None = None) -> PlannerResponse:
        """Split a Messages API response into text, tool calls and replayable content."""
        blocks = list(response.content)
        fallback_at = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content: list[dict[str, Any]] = []
        for i, block in enumerate(blocks):
            if block.type == "fallback":  # audit marker only, not part of the conversation
                continue
            if i < fallback_at and block.type in _DROP_BEFORE_FALLBACK:
                continue  # the declined model's partial work is not echoed back
            content.append(block.model_dump(mode="json", exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))

        stop_reason = response.stop_reason
        if tool_calls and stop_reason in ("max_tokens", "refusal"):
            # A tool call cut off mid-input must not run, and an unanswered tool_use left in
            # history would make the next request in this session invalid — drop both.
            tool_calls = []
            content = [b for b in content if b["type"] != "tool_use"]

        model = getattr(response, "model", None) or requested_model
        return PlannerResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            content=content,
            model=model,
            usage=Usage.from_api(getattr(response, "usage", None), model),
        )
