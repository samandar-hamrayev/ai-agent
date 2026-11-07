from __future__ import annotations

import logging
from datetime import date
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Kept free of per-request values (session ids, timestamps) so the prefix stays cacheable.
_SYSTEM_PROMPT = """You are an autonomous agent that solves the user's goal step by step.

Work in a loop: reason about what is still unknown, call a tool when it gets you closer \
to the answer, read the result, and repeat. Answer directly when you already know enough.

- If a tool fails, read the error and try a different approach instead of repeating the same call.
- Independent tool calls can be made in parallel in a single turn.
- Finish with a clear, well-structured answer for the user.

Today's date: {today}

{memory_context}"""

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


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
    error: str | None = None

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class Planner:
    """Owns every Claude API call: prompt assembly, request options and response parsing."""

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
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
        }
        if tool_schemas:
            request["tools"] = tool_schemas
        if self._fallbacks:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    async def think(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        memory_context: str = "",
    ) -> PlannerResponse:
        """Send the conversation and tool list to Claude and parse its decision."""
        request = self.build_request(messages, tool_schemas, memory_context)
        try:
            response = await self._client.beta.messages.create(**request)
        except anthropic.RateLimitError as e:
            logger.warning("Claude rate limit: %s", e)
            return PlannerResponse(stop_reason="error", error="Claude API rate limit reached, try again shortly.")
        except anthropic.APIStatusError as e:
            logger.error("Claude API error %s: %s", e.status_code, e.message)
            return PlannerResponse(stop_reason="error", error=f"Claude API error ({e.status_code}).")
        except anthropic.APIConnectionError as e:
            logger.error("Claude connection error: %s", e)
            return PlannerResponse(stop_reason="error", error="Could not reach the Claude API.")
        return self.parse(response)

    @staticmethod
    def parse(response: Any) -> PlannerResponse:
        """Split a Messages API response into text, tool calls and replayable content."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "fallback":  # audit marker only, not part of the conversation
                continue
            content.append(block.model_dump(mode="json", exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))

        return PlannerResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            content=content,
            model=getattr(response, "model", None),
        )
