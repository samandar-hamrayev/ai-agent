from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from tools.base import BaseTool, ToolResult

_TIMEOUT = 10
_MAX_OUTPUT = 5000


class CodeExecutorTool(BaseTool):
    """Runs Python code in an isolated subprocess with a timeout."""

    name: str = "code_executor"
    description: str = (
        "Executes Python code in a sandboxed subprocess and returns stdout/stderr. "
        "The code runs with a 10-second timeout. Use this to run calculations, "
        "data processing, or any Python logic. Do NOT use for installing packages."
    )

    def run(self, code: str) -> ToolResult:
        if not code or not code.strip():
            return ToolResult(success=False, output=None, error="Empty code provided.")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix="agent_exec_", delete=False,
            ) as tmp:
                tmp.write(code)
                tmp_path = Path(tmp.name)
        except OSError as e:
            return ToolResult(success=False, output=None, error=f"Failed to create temp file: {e}")

        try:
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                env={"PATH": "/usr/bin:/usr/local/bin", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"},
                cwd="/tmp",
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if len(stdout) > _MAX_OUTPUT:
                stdout = stdout[:_MAX_OUTPUT] + "\n... [truncated]"
            if len(stderr) > _MAX_OUTPUT:
                stderr = stderr[:_MAX_OUTPUT] + "\n... [truncated]"

            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    output={"stdout": stdout or "(no output)", "stderr": stderr or None, "return_code": 0},
                )
            else:
                return ToolResult(
                    success=False,
                    output={"stdout": stdout or None, "return_code": result.returncode},
                    error=stderr or f"Process exited with code {result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error=f"Code execution timed out after {_TIMEOUT} seconds.")
        except OSError as e:
            return ToolResult(success=False, output=None, error=f"Execution failed: {e}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                }
            },
            "required": ["code"],
        }
