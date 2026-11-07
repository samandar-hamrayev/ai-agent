from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.base import BaseTool, ToolResult

_TIMEOUT = 10  # wall-clock seconds
_CPU_SECONDS = 10
_MEMORY_BYTES = 512 * 1024 * 1024
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_OPEN_FILES = 64
_MAX_OUTPUT = 5000


def _limit_resources() -> None:  # pragma: no cover - runs in the child process
    """Applied in the child before exec: caps CPU time, memory, file size and open files."""
    import resource

    limits = [
        (resource.RLIMIT_CPU, _CPU_SECONDS),
        (resource.RLIMIT_FSIZE, _MAX_FILE_BYTES),
        (resource.RLIMIT_NOFILE, _MAX_OPEN_FILES),
        (resource.RLIMIT_AS, _MEMORY_BYTES),  # not enforced on macOS; works on Linux/Docker
    ]
    for limit, value in limits:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(limit, (value, value))


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT] + "\n... [truncated]"


class CodeExecutorTool(BaseTool):
    """Runs untrusted Python in a resource-limited child process.

    This is defence in depth, not a security boundary: the child runs in isolated mode
    (-I ignores PYTHON* env vars and the user site dir, -S skips site-packages so the app's
    dependencies and secrets-bearing modules aren't importable), with an empty environment,
    a throwaway working directory and rlimits. For real isolation run the API in its own container
    (see docker-compose.yml: read-only root FS, no new privileges, dropped capabilities).
    """

    name: str = "code_executor"
    description: str = (
        "Executes a self-contained Python 3 script (standard library only) and returns stdout/stderr. "
        "Limits: 10 seconds, 512 MB of memory, no third-party packages. "
        "Use it for calculations, data processing or checking logic; print() what you need back."
    )

    def run(self, code: str) -> ToolResult:
        if not code or not code.strip():
            return ToolResult(success=False, output=None, error="Empty code provided.")

        with tempfile.TemporaryDirectory(prefix="agent_exec_") as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(code, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-S", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT,
                    env={"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
                    cwd=tmp,
                    preexec_fn=_limit_resources if sys.platform != "win32" else None,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False, output=None, error=f"Code execution timed out after {_TIMEOUT} seconds."
                )
            except OSError as e:
                return ToolResult(success=False, output=None, error=f"Execution failed: {e}")

        stdout, stderr = _truncate(result.stdout.strip()), _truncate(result.stderr.strip())
        if result.returncode == 0:
            return ToolResult(
                success=True,
                output={"stdout": stdout or "(no output)", "stderr": stderr or None, "return_code": 0},
            )
        return ToolResult(
            success=False,
            output={"stdout": stdout or None, "return_code": result.returncode},
            error=stderr or f"Process exited with code {result.returncode}",
        )

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python source code to execute.",
                }
            },
            "required": ["code"],
        }
