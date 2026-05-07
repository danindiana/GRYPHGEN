"""
Tool implementations for the GRYPHGEN agent.

All file operations are sandboxed to a workspace directory.
run_shell uses a restricted command whitelist and a hard timeout.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

# Commands allowed by run_shell (prefix-matched)
_SHELL_WHITELIST = (
    "python3", "python", "pip", "pytest", "ruff", "black", "mypy",
    "cargo", "rustc", "go ", "go\t", "node", "npm", "npx", "deno",
    "gcc", "g++", "make", "cmake",
    "ls", "find", "grep", "cat", "head", "tail", "wc", "diff",
    "git status", "git log", "git diff", "git show",
    "echo",
)

_SHELL_TIMEOUT = 30  # seconds


def _safe_path(workspace: Path, rel: str) -> Path:
    """Resolve path and assert it stays inside workspace."""
    target = (workspace / rel).resolve()
    workspace_resolved = workspace.resolve()
    if not str(target).startswith(str(workspace_resolved)):
        raise PermissionError(f"Path {rel!r} escapes workspace")
    return target


# ── tool implementations ──────────────────────────────────────────────────────

async def read_file(workspace: Path, path: str) -> str:
    p = _safe_path(workspace, path)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    if not p.is_file():
        return f"ERROR: {path} is not a file"
    try:
        return p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR reading {path}: {e}"


async def write_file(workspace: Path, path: str, content: str) -> str:
    p = _safe_path(workspace, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"OK: wrote {len(content)} bytes to {path}"


async def list_dir(workspace: Path, path: str = ".") -> str:
    p = _safe_path(workspace, path)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    if not p.is_dir():
        return f"ERROR: {path} is not a directory"
    entries = []
    for entry in sorted(p.iterdir()):
        kind = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else "-"
        entries.append(f"{kind:4}  {size:>10}  {entry.name}")
    return "\n".join(entries) if entries else "(empty)"


async def run_shell(workspace: Path, command: str) -> str:
    cmd_lower = command.strip().lower()
    if not any(cmd_lower.startswith(w) for w in _SHELL_WHITELIST):
        return f"ERROR: command not permitted: {command!r}"
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "HOME": str(workspace)},
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_SHELL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"ERROR: command timed out after {_SHELL_TIMEOUT}s"
        output = stdout.decode(errors="replace").strip()
        return output or "(no output)"
    except Exception as e:
        return f"ERROR running command: {e}"


# ── Ollama tool schema ────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text content of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from workspace root"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path (default '.')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the workspace directory. "
                "Allowed: python3, pip, pytest, ruff, black, cargo, go, node, "
                "git status/log/diff, grep, find, cat, ls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
]


async def dispatch(name: str, args: dict[str, Any], workspace: Path) -> str:
    """Call the named tool with args, sandboxed to workspace."""
    if name == "read_file":
        return await read_file(workspace, args["path"])
    if name == "write_file":
        return await write_file(workspace, args["path"], args["content"])
    if name == "list_dir":
        return await list_dir(workspace, args.get("path", "."))
    if name == "run_shell":
        return await run_shell(workspace, args["command"])
    return f"ERROR: unknown tool {name!r}"
