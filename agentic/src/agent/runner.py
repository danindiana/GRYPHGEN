"""
ReAct loop for devstral via Ollama tool-calling API.

Flow per step:
  1. POST /api/chat with current message history + tool schemas
  2. If response.message.tool_calls → execute each, append results, continue
  3. If no tool_calls → model produced final answer, return
  4. If max_steps reached → return with truncation notice
"""

from __future__ import annotations

import os
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


_SYSTEM_PROMPT = """\
You are Devstral, an expert software engineering agent.
You have access to tools that let you read files, write files, list directories,
and run shell commands. Use them to complete the user's task.

Rules:
- Always explore the workspace with list_dir before writing code.
- After writing code, run it with run_shell to verify it works.
- When the task is complete, summarize what you did and what files were changed.
- Be concise in your final summary.
"""


class AgentRunner:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "devstral:24b",
        max_steps: int = 20,
        step_timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_steps = max_steps
        self.step_timeout = step_timeout

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
            # Pre-populate workspace files
            if files:
                for rel, content in files.items():
                    p = workspace / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content)

            return await self._react_loop(task, workspace)
        finally:
            if cleanup:
                import shutil
                shutil.rmtree(workspace, ignore_errors=True)

    async def _react_loop(self, task: str, workspace: Path) -> AgentResult:
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
                        "tools": TOOL_SCHEMAS,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})

                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    # Final answer
                    return AgentResult(
                        output=msg.get("content", "").strip(),
                        trace=trace,
                        steps_taken=step + 1,
                        total_time_s=time.monotonic() - t_start,
                        model=self.model,
                    )

                # Append assistant turn
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tool_calls,
                })

                # Execute each tool call and collect results
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    tool_args = fn.get("arguments", {})
                    tool_id = tc.get("id", "")

                    t0 = time.monotonic()
                    result = await dispatch(tool_name, tool_args, workspace)
                    elapsed = time.monotonic() - t0

                    trace.append(StepTrace(
                        step=step + 1,
                        tool=tool_name,
                        args=tool_args,
                        result=result[:2000],  # cap stored result size
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
