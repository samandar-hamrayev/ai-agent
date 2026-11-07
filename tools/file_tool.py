from __future__ import annotations

from pathlib import Path

from tools.base import BaseTool, ToolResult

_DEFAULT_WORKSPACE = Path("./data/workspace")
_MAX_READ = 50_000


def resolve_safe_path(workspace: Path, user_path: str) -> Path:
    """Resolve `user_path` inside `workspace`, rejecting anything that escapes it.

    Uses `is_relative_to` on fully resolved paths: a plain string-prefix check would let
    `/tmp/ws-evil` pass for a workspace of `/tmp/ws`, and `resolve()` follows symlinks.
    """
    root = workspace.resolve()
    candidate = Path(user_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Access denied: '{user_path}' resolves outside the workspace.")
    return resolved


class FileTool(BaseTool):
    """Reads and writes files inside a single workspace directory."""

    name: str = "file_tool"
    description: str = (
        "Reads or writes text files in the agent's private workspace directory. "
        "Use action='read' to read a file, action='write' to create/overwrite a file. "
        "All paths are restricted to the workspace directory for security."
    )

    def needs_approval(self, tool_input: dict) -> bool:
        """Reading is harmless; writing changes state, so a human signs off on it."""
        return tool_input.get("action") == "write"

    def __init__(self, workspace: Path = _DEFAULT_WORKSPACE) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, action: str, path: str, content: str | None = None) -> ToolResult:
        if action not in ("read", "write"):
            return ToolResult(
                success=False, output=None, error=f"Invalid action '{action}'. Must be 'read' or 'write'."
            )

        try:
            resolved = resolve_safe_path(self.workspace, path)
        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))

        if action == "read":
            return self._read(resolved)
        return self._write(resolved, content)

    def _read(self, path: Path) -> ToolResult:
        if not path.exists():
            return ToolResult(success=False, output=None, error=f"File not found: {path}")
        if not path.is_file():
            return ToolResult(success=False, output=None, error=f"Not a file: {path}")

        try:
            text = path.read_text(encoding="utf-8")
            truncated = len(text) > _MAX_READ
            if truncated:
                text = text[:_MAX_READ] + "\n... [truncated]"

            return ToolResult(
                success=True,
                output={"path": str(path), "content": text, "size_bytes": path.stat().st_size, "truncated": truncated},
            )
        except UnicodeDecodeError:
            return ToolResult(success=False, output=None, error=f"Cannot read binary file: {path}")
        except OSError as e:
            return ToolResult(success=False, output=None, error=f"Read error: {e}")

    def _write(self, path: Path, content: str | None) -> ToolResult:
        if content is None:
            return ToolResult(success=False, output=None, error="Content is required for 'write' action.")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output={"path": str(path), "size_bytes": path.stat().st_size, "message": f"Written: {path}"},
            )
        except OSError as e:
            return ToolResult(success=False, output=None, error=f"Write error: {e}")

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "File operation: 'read' or 'write'.",
                },
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' action).",
                },
            },
            "required": ["action", "path"],
        }
