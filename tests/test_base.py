"""Basic smoke test for BaseTool and ToolResult."""

from tools.base import BaseTool, ToolResult


class TestTool(BaseTool):
    name = "test_tool"
    description = "Test tool"

    def run(self, message: str) -> ToolResult:
        return ToolResult(success=True, output=f"Got: {message}")

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        }


def test_tool_run():
    tool = TestTool()
    result = tool.safe_run(message="Hello")
    assert result.success is True
    assert result.output == "Got: Hello"


def test_tool_schema():
    tool = TestTool()
    schema = tool.to_schema()
    assert schema["name"] == "test_tool"


def test_tool_result_str():
    result = ToolResult(success=True, output="ok")
    assert "SUCCESS" in str(result)

    error_result = ToolResult(success=False, output=None, error="fail")
    assert "ERROR" in str(error_result)