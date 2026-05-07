"""
ReAct loop for devstral via Ollama tool-calling API.

Flow per step:
  1. POST /api/chat with current message history + tool schemas
  2. If response.message.tool_calls → execute each, append results, continue
  3. If no tool_calls → model produced final answer, return
  4. If max_steps reached → return with truncation notice

External MCP servers (optional):
  Pass mcp_servers=[McpServerConfig(...)] to AgentRunner to connect additional
  tool servers via the Model Context Protocol. Their tools are merged into the
  Ollama tool-call session automatically.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .tools import TOOL_SCHEMAS, dispatch


@dataclass
class StepTrace:
    step: int
    tool: str
    args: dict[str, Any]
    result: str
    elapsed_s: float


@dataclass
class AgentResult:
    output: str
    trace: list[StepTrace]
    steps_taken: int
    total_time_s: float
    model: str
    truncated: bool = False


@dataclass
class McpServerConfig:
    """Configuration for an external MCP tool server (stdio transport)."""
    name: str
    command: list[str]  # e.g. ["uvx", "mcp-server-filesystem", "/workspace"]
    transport: str = "stdio"


_SYSTEM_PROMPT = """\
You are an autonomous software engineering agent. You MUST use tools to take every action.
Never describe what you would do — call the tool and do it.

Available tools: list_dir, read_file, write_file, run_shell, web_search.

Mandatory sequence for any coding task:
1. Call list_dir to see the workspace contents.
2. Call write_file to create or modify each file needed.
3. Call run_shell to execute and verify the code.
4. Only after all tool calls succeed, write a brief final summary of what was done.

Critical rules:
- NEVER ask the user a question or request confirmation. Make a decision and act.
- NEVER say "Would you like me to..." — just do the best thing and proceed.
- If a tool call fails or returns an error, try a different approach immediately.
- Do NOT use web_search for standard programming tasks or well-known languages/libraries —
  you already have that knowledge. Only use web_search for genuinely unknown third-party APIs.
- Do NOT write prose explanations or code blocks in your text content.
- Do NOT say "I'll do X" — just call the tool and do X.
- If run_shell fails because a compiler or tool is missing, write the code and note the
  missing dependency in your final summary instead of stopping.
"""


class AgentRunner:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "devstral:24b",
        max_steps: int = 20,
        step_timeout: float = 300.0,
        mcp_servers: list[McpServerConfig] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.mcp_servers = mcp_servers or []

    async def run(
        self,
        task: str,
        workspace: Path | None = None,
        files: dict[str, str] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task:      Natural language task description.
            workspace: Directory to use as the agent's workspace.
                       If None, a fresh temp dir is created and cleaned up.
            files:     Pre-populate workspace with {relative_path: content}.
        """
        cleanup = workspace is None
        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="gryphgen_agent_"))

        try:
            if files:
                for rel, content in files.items():
                    p = workspace / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content)

            # Connect to external MCP servers if configured
            extra_schemas, mcp_sessions = await _connect_mcp_servers(
                self.mcp_servers, workspace
            )

            try:
                return await self._react_loop(
                    task, workspace, extra_schemas, mcp_sessions
                )
            finally:
                for session, proc in mcp_sessions.values():
                    try:
                        await session.__aexit__(None, None, None)
                        proc.terminate()
                    except Exception:
                        pass
        finally:
            if cleanup:
                import shutil
                shutil.rmtree(workspace, ignore_errors=True)

    async def _react_loop(
        self,
        task: str,
        workspace: Path,
        extra_schemas: list[dict],
        mcp_sessions: dict[str, Any],
    ) -> AgentResult:
        all_schemas = TOOL_SCHEMAS + extra_schemas
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Workspace: {workspace}\n\nTask: {task}"},
        ]

        trace: list[StepTrace] = []
        t_start = time.monotonic()

        async with httpx.AsyncClient(timeout=self.step_timeout) as client:
            for step in range(self.max_steps):
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": all_schemas,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})

                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    return AgentResult(
                        output=msg.get("content", "").strip(),
                        trace=trace,
                        steps_taken=step + 1,
                        total_time_s=time.monotonic() - t_start,
                        model=self.model,
                    )

                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    raw_args = fn.get("arguments") or {}
                    if isinstance(raw_args, str):
                        try:
                            import json as _json
                            raw_args = _json.loads(raw_args)
                        except Exception:
                            raw_args = {}
                    tool_args = raw_args if isinstance(raw_args, dict) else {}
                    tool_id = tc.get("id", "")

                    t0 = time.monotonic()
                    result = await _dispatch_with_mcp(
                        tool_name, tool_args, workspace, mcp_sessions
                    )
                    elapsed = time.monotonic() - t0

                    trace.append(StepTrace(
                        step=step + 1,
                        tool=tool_name,
                        args=tool_args,
                        result=result[:2000],
                        elapsed_s=elapsed,
                    ))

                    messages.append({
                        "role": "tool",
                        "content": result,
                        **({"tool_call_id": tool_id} if tool_id else {}),
                    })

        return AgentResult(
            output=f"Reached max_steps ({self.max_steps}) without a final answer.",
            trace=trace,
            steps_taken=self.max_steps,
            total_time_s=time.monotonic() - t_start,
            model=self.model,
            truncated=True,
        )


# ── MCP client helpers ────────────────────────────────────────────────────────

async def _connect_mcp_servers(
    configs: list[McpServerConfig],
    workspace: Path,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Launch each MCP server as a subprocess and list its tools.
    Returns (extra_schemas, {tool_name: (session, proc)}).
    """
    if not configs:
        return [], {}

    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        return [], {}

    extra_schemas: list[dict] = []
    sessions: dict[str, Any] = {}

    for cfg in configs:
        try:
            # Substitute {workspace} placeholder in command args
            cmd = [
                arg.replace("{workspace}", str(workspace)) for arg in cfg.command
            ]
            params = StdioServerParameters(command=cmd[0], args=cmd[1:])
            ctx = stdio_client(params)
            read, write = await ctx.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            tools = await session.list_tools()
            for tool in tools.tools:
                schema = _mcp_tool_to_ollama(cfg.name, tool)
                extra_schemas.append(schema)
                sessions[f"{cfg.name}::{tool.name}"] = (session, None)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to connect to MCP server %r: %s", cfg.name, e
            )

    return extra_schemas, sessions


def _mcp_tool_to_ollama(server_name: str, tool: Any) -> dict:
    """Translate an MCP tool definition to Ollama function-call schema."""
    input_schema = {}
    if hasattr(tool, "inputSchema") and tool.inputSchema:
        if hasattr(tool.inputSchema, "model_dump"):
            input_schema = tool.inputSchema.model_dump(exclude_none=True)
        elif isinstance(tool.inputSchema, dict):
            input_schema = tool.inputSchema

    return {
        "type": "function",
        "function": {
            "name": f"{server_name}__{tool.name}",
            "description": tool.description or "",
            "parameters": input_schema or {"type": "object", "properties": {}},
        },
    }


async def _dispatch_with_mcp(
    tool_name: str,
    tool_args: dict[str, Any],
    workspace: Path,
    mcp_sessions: dict[str, Any],
) -> str:
    """Route a tool call to local dispatch or an MCP session."""
    # Try local tools first
    local_result = await dispatch(tool_name, tool_args, workspace)
    if not local_result.startswith("ERROR: unknown tool"):
        return local_result

    # Try MCP sessions (tool_name format: "servername__toolname")
    for key, (session, _) in mcp_sessions.items():
        server_name, mcp_tool_name = key.split("::", 1)
        expected = f"{server_name}__{mcp_tool_name}"
        if tool_name == expected:
            try:
                result = await session.call_tool(mcp_tool_name, tool_args)
                if result.content:
                    return "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                return "(no output)"
            except Exception as e:
                return f"ERROR calling MCP tool {tool_name}: {e}"

    return f"ERROR: unknown tool {tool_name!r}"
