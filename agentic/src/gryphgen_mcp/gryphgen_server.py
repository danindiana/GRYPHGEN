"""
Outward MCP server for GRYPHGEN.

Exposes three tools to any MCP-compatible client (Claude Code, wizard, etc.):
  - generate_code   → POST /api/v1/code/generate
  - agent_run       → POST /api/v1/agent/run
  - web_search      → GET  /api/v1/tools/search

Transport options:
  - SSE/HTTP: create_sse_app() returns a Starlette app for mounting on FastAPI
  - stdio:    make_server(api_key, base_url).run("stdio")  (used by gryphgen-mcp script)

Usage from Claude Code after registering:
  claude mcp add gryphgen -- python3 /path/to/gryphgen-mcp --api-key KEY
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

_DEFAULT_URL = "https://api.grug.ai"


def make_server(
    api_key: str | None = None,
    base_url: str | None = None,
) -> FastMCP:
    """
    Create and return a configured FastMCP instance.

    api_key and base_url fall back to env vars GRYPHGEN_API_KEY / GRYPHGEN_URL,
    then to the config file (same path as the gryphgen CLI uses).
    """
    resolved_key = api_key or os.environ.get("GRYPHGEN_API_KEY") or _read_config_key()
    resolved_url = (base_url or os.environ.get("GRYPHGEN_URL") or _read_config_url() or _DEFAULT_URL).rstrip("/")

    mcp = FastMCP("gryphgen")

    # ── tools ────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_code(
        prompt: str,
        language: str = "python",
        tier: str = "auto",
    ) -> str:
        """
        Generate code using the GRYPHGEN AI pipeline.

        Args:
            prompt:   Natural language description of the code to generate.
            language: Target programming language (default: python).
            tier:     Routing tier — 'auto' (default), 'fast', 'standard', or 'strong'.
                      'auto' classifies the task automatically. 'fast' is best for single
                      functions. 'standard' for modules/APIs. 'strong' for complex/multi-file work.

        Returns:
            Generated source code as a string.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "language": language,
            "include_docs": True,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        if tier == "fast":
            payload["reasoning_effort"] = "none"
        elif tier == "strong":
            payload["reasoning_effort"] = "high"

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{resolved_url}/api/v1/code/generate",
                json=payload,
                headers={"X-API-Key": resolved_key or "", "User-Agent": "gryphgen-mcp/0.1"},
            )
            resp.raise_for_status()
            data = resp.json()

        code = data.get("code", "")
        tier_used = data.get("tier", "?")
        gen_time = data.get("generation_time", 0)
        tokens = data.get("tokens_used", 0)
        footer = f"\n\n# tier={tier_used}  {gen_time:.2f}s  {tokens} tokens"
        return code + footer

    @mcp.tool()
    async def agent_run(
        task: str,
        max_steps: int = 20,
    ) -> str:
        """
        Run an agentic ReAct task using GRYPHGEN.

        The agent can read/write files, execute shell commands in a Docker sandbox,
        and search the web. It works iteratively until the task is complete.

        Args:
            task:      Natural language description of what to accomplish.
            max_steps: Maximum number of tool-calling iterations (default: 20, max: 50).

        Returns:
            Final answer from the agent, plus a summary of steps taken.
        """
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{resolved_url}/api/v1/agent/run",
                json={"task": task, "max_steps": min(max_steps, 50)},
                headers={"X-API-Key": resolved_key or "", "User-Agent": "gryphgen-mcp/0.1"},
            )
            resp.raise_for_status()
            data = resp.json()

        output = data.get("output", "")
        steps = data.get("steps_taken", 0)
        total_s = data.get("total_time_s", 0)
        truncated = data.get("truncated", False)

        trace_lines = []
        for t in data.get("trace", []):
            tool = t.get("tool", "?")
            elapsed = t.get("elapsed_s", 0)
            trace_lines.append(f"  step {t.get('step','?')}: {tool} ({elapsed:.1f}s)")

        summary = f"\n\n─── {steps} steps, {total_s:.1f}s total" + (" [truncated]" if truncated else "")
        if trace_lines:
            summary += "\n" + "\n".join(trace_lines)

        return output + summary

    @mcp.tool()
    async def web_search(
        query: str,
        max_results: int = 5,
    ) -> str:
        """
        Search the web using DuckDuckGo via GRYPHGEN.

        Returns title, URL, and snippet for each result. No API key required for DDG.

        Args:
            query:       Search query string.
            max_results: Maximum number of results to return (default: 5).

        Returns:
            Numbered list of search results with title, URL, and snippet.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{resolved_url}/api/v1/tools/search",
                params={"q": query, "max_results": max_results},
                headers={"X-API-Key": resolved_key or "", "User-Agent": "gryphgen-mcp/0.1"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{i + 1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results)
        )

    return mcp


def create_sse_app():
    """Return a Starlette app for mounting on FastAPI at /mcp."""
    mcp = make_server()
    return mcp.sse_app()


# ── config helpers ────────────────────────────────────────────────────────────

def _read_config_key() -> str:
    cfg = _load_config()
    return cfg.get("api_key", "")


def _read_config_url() -> str:
    cfg = _load_config()
    return cfg.get("api_url", "")


def _load_config() -> dict[str, str]:
    import tomllib
    from pathlib import Path
    path = Path.home() / ".config" / "gryphgen" / "config.toml"
    try:
        return tomllib.loads(path.read_text())
    except Exception:
        return {}
