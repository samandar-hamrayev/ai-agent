from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result of a tool execution."""

    success: bool
    output: Any
    error: str | None = None

    def __str__(self):
        if self.success:
            return f"SUCCESS: {self.output}"
        return f"ERROR: {self.error}"


class BaseTool(ABC):
    """Base class for all agent tools."""

    name: str
    description: str
    # Side-effecting tools set this; the agent then pauses for a human decision before running them.
    requires_approval: bool = False

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool's main logic. Must be overridden by subclasses."""
        pass

    def to_schema(self) -> dict:
        """Return the tool schema in Claude API tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema(),
        }

    @abstractmethod
    def get_input_schema(self) -> dict:
        """Return JSON Schema describing accepted parameters."""
        pass

    def needs_approval(self, tool_input: dict[str, Any]) -> bool:
        """Whether this particular call must be approved. Override for per-input rules."""
        return self.requires_approval

    def safe_run(self, **kwargs) -> ToolResult:
        """Wrap run() in try/except so the agent never crashes."""
        try:
            return self.run(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
