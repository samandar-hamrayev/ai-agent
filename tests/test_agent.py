"""Tests for the Agent core and ToolRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.core import Agent, AgentConfig, AgentStep
from agent.planner import PlannerResponse
from agent.tool_router import ToolRegistry
from tools.base import BaseTool, ToolResult


# ──────────────────────────────────────────────────────────────────────
# Helpers — Fake tool for testing
# ──────────────────────────────────────────────────────────────────────

class FakeTool(BaseTool):
    """A minimal tool for testing purposes."""

    name: str = "fake_tool"
    description: str = "A fake tool for testing."

    def run(self, message: str = "default") -> ToolResult:
        return ToolResult(success=True, output=f"Fake: {message}")

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "A test message."},
            },
            "required": ["message"],
        }


class ErrorTool(BaseTool):
    """A tool that always raises an exception."""

    name: str = "error_tool"
    description: str = "Always fails."

    def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("Intentional error for testing")

    def get_input_schema(self) -> dict:
        return {"type": "object", "properties": {}}


# ──────────────────────────────────────────────────────────────────────
# ToolRegistry Tests
# ──────────────────────────────────────────────────────────────────────

class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_tool(self):
        self.registry.register(FakeTool())
        assert "fake_tool" in self.registry
        assert len(self.registry) == 1

    def test_register_duplicate_raises(self):
        self.registry.register(FakeTool())
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(FakeTool())

    def test_execute_tool(self):
        self.registry.register(FakeTool())
        result = self.registry.execute("fake_tool", {"message": "hello"})
        assert result.success is True
        assert result.output == "Fake: hello"

    def test_execute_unknown_tool(self):
        result = self.registry.execute("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_error_tool_safe(self):
        """safe_run should catch the exception."""
        self.registry.register(ErrorTool())
        result = self.registry.execute("error_tool", {})
        assert result.success is False
        assert "Intentional error" in result.error

    def test_get_all_schemas(self):
        self.registry.register(FakeTool())
        schemas = self.registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "fake_tool"
        assert "input_schema" in schemas[0]

    def test_tool_names(self):
        self.registry.register(FakeTool())
        self.registry.register(ErrorTool())
        assert self.registry.tool_names == ["error_tool", "fake_tool"]

    def test_get_tool(self):
        self.registry.register(FakeTool())
        tool = self.registry.get("fake_tool")
        assert tool is not None
        assert tool.name == "fake_tool"

    def test_get_nonexistent_tool(self):
        assert self.registry.get("nothing") is None


# ──────────────────────────────────────────────────────────────────────
# AgentStep Tests
# ──────────────────────────────────────────────────────────────────────

class TestAgentStep:
    """Tests for AgentStep model."""

    def test_step_creation(self):
        step = AgentStep(step_number=1, thought="thinking...")
        assert step.step_number == 1
        assert step.is_final is False

    def test_final_step(self):
        step = AgentStep(step_number=3, is_final=True, final_answer="Done!")
        assert step.is_final is True
        assert "FINAL" in str(step)

    def test_tool_step_str(self):
        step = AgentStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            tool_result="4",
        )
        assert "calculator" in str(step)


# ──────────────────────────────────────────────────────────────────────
# AgentConfig Tests
# ──────────────────────────────────────────────────────────────────────

class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 10

    def test_custom_config(self):
        config = AgentConfig(max_iterations=5, model="claude-haiku-4-20250514")
        assert config.max_iterations == 5
        assert config.model == "claude-haiku-4-20250514"


# ──────────────────────────────────────────────────────────────────────
# Agent Core Tests (with mocked dependencies)
# ──────────────────────────────────────────────────────────────────────

class TestAgent:
    """Tests for the Agent ReAct loop with mocked LLM and memory."""

    def _build_agent(self, planner_responses: list[PlannerResponse]) -> Agent:
        """Build an Agent with mocked dependencies."""
        # Mock tool registry
        registry = ToolRegistry()
        registry.register(FakeTool())

        # Mock short-term memory
        stm = MagicMock()
        stm.add_message = MagicMock()
        stm.get_messages_for_api = MagicMock(return_value=[
            {"role": "user", "content": "test goal"}
        ])

        # Mock long-term memory
        ltm = MagicMock()
        ltm.search = MagicMock(return_value=[])
        ltm.store = MagicMock()

        # Mock planner — return predefined responses sequentially
        planner = MagicMock()
        planner.think = MagicMock(side_effect=planner_responses)

        config = AgentConfig(max_iterations=5)
        return Agent(
            tool_registry=registry,
            short_term_memory=stm,
            long_term_memory=ltm,
            planner=planner,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_direct_answer(self):
        """Agent should yield final answer when LLM responds with text only."""
        agent = self._build_agent([
            PlannerResponse(response_type="text", text="The answer is 42.", stop_reason="end_turn"),
        ])

        steps = []
        async for step in agent.run("What is the answer?"):
            steps.append(step)

        assert len(steps) == 1
        assert steps[0].is_final is True
        assert "42" in steps[0].final_answer

    @pytest.mark.asyncio
    async def test_tool_then_answer(self):
        """Agent should use a tool, then provide a final answer."""
        agent = self._build_agent([
            PlannerResponse(
                response_type="tool_use",
                text="Let me use the fake tool.",
                tool_name="fake_tool",
                tool_input={"message": "hello"},
                tool_use_id="tool_123",
                stop_reason="tool_use",
            ),
            PlannerResponse(
                response_type="text",
                text="Based on the tool result, the answer is: Fake: hello",
                stop_reason="end_turn",
            ),
        ])

        steps = []
        async for step in agent.run("Use the tool"):
            steps.append(step)

        assert len(steps) == 2
        assert steps[0].tool_name == "fake_tool"
        assert steps[0].is_final is False
        assert steps[1].is_final is True

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """Agent should stop gracefully when max iterations are exhausted."""
        # Always request tool use — never provide a final answer
        infinite_tool_calls = [
            PlannerResponse(
                response_type="tool_use",
                text="Using tool again...",
                tool_name="fake_tool",
                tool_input={"message": f"call_{i}"},
                tool_use_id=f"tool_{i}",
                stop_reason="tool_use",
            )
            for i in range(10)
        ]

        agent = self._build_agent(infinite_tool_calls)

        steps = []
        async for step in agent.run("Infinite loop test"):
            steps.append(step)

        # Should have max_iterations (5) tool steps + 1 final warning
        assert steps[-1].is_final is True
        assert "max iterations" in steps[-1].final_answer.lower()
