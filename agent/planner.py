from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an autonomous AI Agent with access to tools.
Your task is to help the user achieve their goal by reasoning step-by-step and using tools when needed.

## Your Approach (ReAct Pattern)
1. THINK — Analyze what needs to be done next
2. DECIDE — Choose which tool to use (or provide the final answer)
3. ACT — Use the selected tool
4. OBSERVE — Review the tool's result
5. REPEAT — Continue until the goal is fully achieved

## Rules
- Always think before acting. Explain your reasoning.
- Use tools only when necessary — if you already know the answer, respond directly.
- If a tool fails, acknowledge the error and try an alternative approach.
- Provide clear, structured final answers.
- Be concise but thorough.

## Available Tools
{tool_descriptions}

## Current Context
- Date/Time: {current_datetime}
- Session: {session_id}

## Relevant Memory
{memory_context}
"""


class PlannerResponse:
    """Parsed Claude API response — either a text reply or a tool call."""

    def __init__(
        self,
        response_type: str,
        text: str | None = None,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        stop_reason: str | None = None,
    ) -> None:
        self.response_type = response_type
        self.text = text
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool_use_id = tool_use_id
        self.stop_reason = stop_reason

    @property
    def wants_tool(self) -> bool:
        return self.response_type == "tool_use" and self.tool_name is not None

    @property
    def is_final(self) -> bool:
        return self.stop_reason == "end_turn" and not self.wants_tool

    def __repr__(self) -> str:
        if self.wants_tool:
            return f"PlannerResponse(tool_use: {self.tool_name})"
        return f"PlannerResponse(text: {(self.text or '')[:80]}...)"


class Planner:
    """Handles all Claude API interactions — prompt building and response parsing."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        logger.info("Planner ready: model=%s", model)

    def _build_system_prompt(self, tool_schemas: list[dict], session_id: str, memory_context: str = "") -> str:
        tool_descriptions = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tool_schemas)
        return _SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions or "(no tools available)",
            current_datetime=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            session_id=session_id,
            memory_context=memory_context or "(no relevant memory)",
        )

    def think(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        session_id: str = "default",
        memory_context: str = "",
    ) -> PlannerResponse:
        """Send conversation + tools to Claude, get back a decision."""
        system = self._build_system_prompt(tool_schemas, session_id, memory_context)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIError as e:
            logger.error("Claude API error: %s", e)
            return PlannerResponse(
                response_type="text",
                text=f"API error: {e}. Please try again.",
                stop_reason="error",
            )

        return self._parse(response)

    def _parse(self, response: Any) -> PlannerResponse:
        """Extract text and tool_use blocks from a Claude response."""
        text_parts: list[str] = []
        tool_block = None

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_block = block

        text = "\n".join(text_parts) if text_parts else None

        if tool_block:
            return PlannerResponse(
                response_type="tool_use",
                text=text,
                tool_name=tool_block.name,
                tool_input=dict(tool_block.input) if tool_block.input else {},
                tool_use_id=tool_block.id,
                stop_reason=response.stop_reason,
            )

        return PlannerResponse(response_type="text", text=text, stop_reason=response.stop_reason)
