"""
Tool implementations for the GRYPHGEN agent.

All file operations are sandboxed to a workspace directory.
run_shell executes via DockerSandbox (falls back to whitelist if Docker unavailable).
web_search queries DuckDuckGo Instant Answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sandbox import DockerSandbox

_sandbox = DockerSandbox()


def _safe_path(workspace: Path, rel: str) -> Path:
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
    return await _sandbox.run(command, workspace)


async def web_search(query: str) -> str:
    try:
        from ..tools.web_search import search as _ddg_search
        results = await _ddg_search(query, max_results=5)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in results
        )
    except Exception as e:
        return f"ERROR: web search failed: {e}"


# ── Ollama tool schemas ───────────────────────────────────────────────────────

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
                "Run a shell command in an isolated Docker sandbox. "
                "Any shell command is accepted; the container has no network access "
                "and is limited to 256 MB RAM and 0.5 CPU."
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
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Returns title, URL, and snippet "
                "for up to 5 results. Use this to look up library docs, APIs, "
                "or patterns before writing code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
]


async def dispatch(name: str, args: dict[str, Any], workspace: Path) -> str:
    """Route a tool call to its implementation."""
    if name == "read_file":
        return await read_file(workspace, args["path"])
    if name == "write_file":
        return await write_file(workspace, args["path"], args["content"])
    if name == "list_dir":
        return await list_dir(workspace, args.get("path", "."))
    if name == "run_shell":
        return await run_shell(workspace, args["command"])
    if name == "web_search":
        return await web_search(args["query"])
    return f"ERROR: unknown tool {name!r}"
