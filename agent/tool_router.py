from __future__ import annotations

import logging
from typing import Any

from tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Stores registered tools and dispatches execution by name."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, tool_name: str) -> BaseTool | None:
        return self._tools.get(tool_name)

    def execute(self, tool_name: str, input_data: dict[str, Any]) -> ToolResult:
        """Look up a tool by name and run it safely."""
        tool = self._tools.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys())) or "(none)"
            return ToolResult(success=False, output=None, error=f"Tool '{tool_name}' not found. Available: {available}")

        logger.info("Executing tool: %s", tool_name)
        result = tool.safe_run(**input_data)
        logger.info("Tool %s finished: success=%s", tool_name, result.success)
        return result

    def get_all_schemas(self) -> list[dict]:
        """Return Claude-compatible schemas for every registered tool."""
        return [tool.to_schema() for tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools
