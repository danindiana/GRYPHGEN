"""
POST /api/v1/agent/run  — agentic ReAct loop via devstral tool use.

The agent receives a task description and an optional file payload, then
reasons and acts (read_file / write_file / list_dir / run_shell) until the
task is complete or max_steps is reached.  The full action trace is returned
alongside the final answer so callers can audit every step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Security, status
from pydantic import BaseModel, Field

from ...agent import AgentResult, AgentRunner
from ...auth.api_keys import require_api_key
from ...auth.rate_limit import agent_limiter

router = APIRouter()

_runner = AgentRunner(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    model=os.environ.get("OLLAMA_MODEL", "devstral:24b"),
    max_steps=int(os.environ.get("AGENT_MAX_STEPS", "20")),
    step_timeout=float(os.environ.get("AGENT_STEP_TIMEOUT", "300")),
)


# ── request / response models ─────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    task: str = Field(..., description="Natural language task for the agent")
    files: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional initial workspace files as {relative_path: content}",
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Override default max_steps for this run",
    )


class StepTraceOut(BaseModel):
    step: int
    tool: str
    args: dict[str, Any]
    result: str
    elapsed_s: float


class AgentRunResponse(BaseModel):
    output: str
    trace: list[StepTraceOut]
    steps_taken: int
    total_time_s: float
    model: str
    truncated: bool


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Run the devstral agent on a task",
    description=(
        "Starts a ReAct loop: the model reasons, calls tools (read_file, write_file, "
        "list_dir, run_shell), observes results, and repeats until the task is complete "
        "or max_steps is reached.  All tool calls and their outputs are returned in the "
        "trace for full auditability."
    ),
)
async def agent_run(
    request: AgentRunRequest,
    api_key: str = Security(require_api_key),
) -> AgentRunResponse:
    # Rate limit per key (5 req / 5 min)
    agent_limiter.check(api_key)

    runner = _runner
    if request.max_steps is not None:
        runner = AgentRunner(
            base_url=_runner.base_url,
            model=_runner.model,
            max_steps=request.max_steps,
            step_timeout=_runner.step_timeout,
        )

    try:
        result: AgentResult = await runner.run(
            task=request.task,
            files=request.files,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent error: {exc}",
        ) from exc

    return AgentRunResponse(
        output=result.output,
        trace=[
            StepTraceOut(
                step=t.step,
                tool=t.tool,
                args=t.args,
                result=t.result,
                elapsed_s=t.elapsed_s,
            )
            for t in result.trace
        ],
        steps_taken=result.steps_taken,
        total_time_s=result.total_time_s,
        model=result.model,
        truncated=result.truncated,
    )
