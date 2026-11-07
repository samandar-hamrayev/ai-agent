"""Tests for all agent tools: calculator, code_executor, file_tool, web_search."""

import pytest

from tools.base import ToolResult
from tools.calculator import CalculatorTool
from tools.code_executor import CodeExecutorTool
from tools.file_tool import FileTool
from tools.web_search import WebSearchTool

# ──────────────────────────────────────────────────────────────────────
# Calculator Tool Tests
# ──────────────────────────────────────────────────────────────────────


class TestCalculatorTool:
    """Unit tests for CalculatorTool."""

    def setup_method(self):
        self.tool = CalculatorTool()

    def test_basic_addition(self):
        result = self.tool.run(expression="2 + 3")
        assert result.success is True
        assert result.output == 5

    def test_basic_subtraction(self):
        result = self.tool.run(expression="10 - 4")
        assert result.success is True
        assert result.output == 6

    def test_multiplication(self):
        result = self.tool.run(expression="7 * 8")
        assert result.success is True
        assert result.output == 56

    def test_division(self):
        result = self.tool.run(expression="15 / 4")
        assert result.success is True
        assert result.output == 3.75

    def test_floor_division(self):
        result = self.tool.run(expression="15 // 4")
        assert result.success is True
        assert result.output == 3

    def test_modulo(self):
        result = self.tool.run(expression="17 % 5")
        assert result.success is True
        assert result.output == 2

    def test_power(self):
        result = self.tool.run(expression="2 ** 10")
        assert result.success is True
        assert result.output == 1024

    def test_complex_expression(self):
        result = self.tool.run(expression="(25 * 4) + 100")
        assert result.success is True
        assert result.output == 200

    def test_nested_parentheses(self):
        result = self.tool.run(expression="((2 + 3) * (4 - 1)) ** 2")
        assert result.success is True
        assert result.output == 225

    def test_negative_numbers(self):
        result = self.tool.run(expression="-5 + 3")
        assert result.success is True
        assert result.output == -2

    def test_float_arithmetic(self):
        result = self.tool.run(expression="3.14 * 2")
        assert result.success is True
        assert abs(result.output - 6.28) < 0.001

    def test_division_by_zero(self):
        result = self.tool.run(expression="1 / 0")
        assert result.success is False
        assert result.error is not None

    def test_empty_expression(self):
        result = self.tool.run(expression="")
        assert result.success is False

    def test_invalid_expression(self):
        result = self.tool.run(expression="hello + world")
        assert result.success is False

    def test_unsafe_expression_rejected(self):
        """Ensure function calls and imports are rejected."""
        result = self.tool.run(expression="__import__('os').system('ls')")
        assert result.success is False

    def test_large_exponent_rejected(self):
        result = self.tool.run(expression="2 ** 10000")
        assert result.success is False
        assert "too large" in result.error.lower()

    def test_schema(self):
        schema = self.tool.to_schema()
        assert schema["name"] == "calculator"
        assert "expression" in schema["input_schema"]["properties"]

    def test_safe_run_catches_exceptions(self):
        result = self.tool.safe_run(expression="1 / 0")
        assert result.success is False


# ──────────────────────────────────────────────────────────────────────
# Code Executor Tool Tests
# ──────────────────────────────────────────────────────────────────────


class TestCodeExecutorTool:
    """Unit tests for CodeExecutorTool."""

    def setup_method(self):
        self.tool = CodeExecutorTool()

    def test_simple_print(self):
        result = self.tool.run(code='print("Hello, World!")')
        assert result.success is True
        assert "Hello, World!" in result.output["stdout"]

    def test_math_calculation(self):
        result = self.tool.run(code="print(2 ** 10)")
        assert result.success is True
        assert "1024" in result.output["stdout"]

    def test_multiline_code(self):
        code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(f"Sum: {total}")
"""
        result = self.tool.run(code=code)
        assert result.success is True
        assert "Sum: 15" in result.output["stdout"]

    def test_syntax_error(self):
        result = self.tool.run(code="def foo(")
        assert result.success is False
        assert result.error is not None

    def test_runtime_error(self):
        result = self.tool.run(code="x = 1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_empty_code(self):
        result = self.tool.run(code="")
        assert result.success is False

    def test_no_output(self):
        result = self.tool.run(code="x = 42")
        assert result.success is True
        assert result.output["stdout"] == "(no output)"

    def test_schema(self):
        schema = self.tool.to_schema()
        assert schema["name"] == "code_executor"
        assert "code" in schema["input_schema"]["properties"]

    def test_parent_environment_not_inherited(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        result = self.tool.run(code="import os; print(os.environ.get('ANTHROPIC_API_KEY'))")
        assert result.output["stdout"] == "None"

    def test_app_dependencies_not_importable(self):
        result = self.tool.run(code="import anthropic")
        assert result.success is False
        assert "ModuleNotFoundError" in result.error

    @pytest.mark.slow
    def test_timeout(self):
        """Code that runs too long should time out."""
        result = self.tool.run(code="import time; time.sleep(30)")
        assert result.success is False
        assert "timed out" in result.error.lower()


# ──────────────────────────────────────────────────────────────────────
# File Tool Tests
# ──────────────────────────────────────────────────────────────────────


class TestFileTool:
    """Unit tests for FileTool (each test gets its own temporary workspace)."""

    @pytest.fixture(autouse=True)
    def _workspace(self, tmp_path):
        self.workspace = tmp_path / "ws"
        self.tool = FileTool(workspace=self.workspace)

    def test_write_and_read(self):
        # Write
        write_result = self.tool.run(action="write", path="test_file.txt", content="Hello, Agent!")
        assert write_result.success is True

        # Read
        read_result = self.tool.run(action="read", path="test_file.txt")
        assert read_result.success is True
        assert "Hello, Agent!" in read_result.output["content"]

    def test_write_creates_subdirectories(self):
        result = self.tool.run(
            action="write",
            path="test_subdir/nested/file.txt",
            content="nested content",
        )
        assert result.success is True
        assert (self.workspace / "test_subdir" / "nested" / "file.txt").exists()

    def test_read_nonexistent_file(self):
        result = self.tool.run(action="read", path="nonexistent_file.txt")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_write_no_content(self):
        result = self.tool.run(action="write", path="test_file.txt")
        assert result.success is False
        assert "content" in result.error.lower()

    def test_invalid_action(self):
        result = self.tool.run(action="delete", path="test_file.txt")
        assert result.success is False
        assert "invalid action" in result.error.lower()

    def test_directory_traversal_blocked(self):
        """Paths that escape the workspace must be rejected."""
        result = self.tool.run(action="read", path="../../etc/passwd")
        assert result.success is False
        assert "access denied" in result.error.lower() or "outside" in result.error.lower()

    def test_absolute_path_outside_workspace(self):
        result = self.tool.run(action="read", path="/etc/passwd")
        assert result.success is False

    def test_sibling_directory_with_same_prefix_blocked(self):
        """A string-prefix check would let `<ws>-evil` through; is_relative_to doesn't."""
        sibling = self.workspace.parent / (self.workspace.name + "-evil")
        sibling.mkdir()
        (sibling / "secret.txt").write_text("nope")
        result = self.tool.run(action="read", path=str(sibling / "secret.txt"))
        assert result.success is False
        assert "access denied" in result.error.lower()

    def test_symlink_escape_blocked(self, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (self.workspace / "link.txt").symlink_to(outside)
        result = self.tool.run(action="read", path="link.txt")
        assert result.success is False

    def test_schema(self):
        schema = self.tool.to_schema()
        assert schema["name"] == "file_tool"
        assert "action" in schema["input_schema"]["properties"]
        assert "path" in schema["input_schema"]["properties"]


# ──────────────────────────────────────────────────────────────────────
# Web Search Tool Tests
# ──────────────────────────────────────────────────────────────────────


class TestWebSearchTool:
    """Unit tests for WebSearchTool."""

    def setup_method(self):
        self.tool = WebSearchTool()

    def test_empty_query(self):
        result = self.tool.run(query="")
        assert result.success is False

    def test_schema(self):
        schema = self.tool.to_schema()
        assert schema["name"] == "web_search"
        assert "query" in schema["input_schema"]["properties"]

    @pytest.mark.network
    def test_real_search(self):
        """Integration test — requires network access."""
        result = self.tool.run(query="Python programming language")
        assert result.success is True
        assert "results" in result.output
        # DuckDuckGo should return at least 1 result
        assert len(result.output["results"]) >= 1

    def test_safe_run_on_error(self):
        """safe_run should catch any unexpected errors."""
        result = self.tool.safe_run(query="test query")
        # Should return a ToolResult (success or failure), never crash
        assert isinstance(result, ToolResult)
