"""
Docker-based sandbox for run_shell.

DockerSandbox executes commands inside a python:3.12-slim container with:
  - Workspace directory mounted read-write at /workspace
  - Network disabled
  - Memory capped at 256 MB, CPU at 0.5 core
  - Hard timeout via `timeout` shell command wrapper

Falls back to the original whitelist-based execution if Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path

SANDBOX_IMAGE = "python:3.12-slim"
_SHELL_TIMEOUT = 30

# Whitelist kept for the fallback path (Docker unavailable)
_SHELL_WHITELIST = (
    "python3", "python", "pip", "pytest", "ruff", "black", "mypy",
    "cargo", "rustc", "go ", "go\t", "node", "npm", "npx", "deno",
    "gcc", "g++", "make", "cmake",
    "ls", "find", "grep", "cat", "head", "tail", "wc", "diff",
    "git status", "git log", "git diff", "git show",
    "echo",
)


class DockerSandbox:
    def __init__(self) -> None:
        self._available: bool | None = None
        self._client = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import docker  # lazy import — don't break server startup if missing
                client = docker.from_env()
                client.ping()
                self._client = client
                self._available = True
            except Exception:
                self._available = False
        return self._available  # type: ignore[return-value]

    async def run(
        self, command: str, workspace: Path, timeout: int = _SHELL_TIMEOUT
    ) -> str:
        if not self.is_available():
            return await self._whitelist_fallback(command, workspace, timeout)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_sync, command, workspace, timeout
        )

    def _run_sync(self, command: str, workspace: Path, timeout: int) -> str:
        import docker  # already imported during is_available check
        wrapped = f"timeout {timeout} bash -c {shlex.quote(command)}"
        try:
            output = self._client.containers.run(
                SANDBOX_IMAGE,
                ["bash", "-c", wrapped],
                volumes={str(workspace.resolve()): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,  # 0.5 CPU
                network_disabled=True,
                remove=True,
                stdout=True,
                stderr=True,
            )
            text = output.decode("utf-8", errors="replace").strip()
            return text or "(no output)"
        except docker.errors.ContainerError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
            return stderr or f"ERROR: container exited with code {e.exit_status}"
        except Exception as e:
            return f"ERROR: sandbox execution failed: {e}"

    async def _whitelist_fallback(
        self, command: str, workspace: Path, timeout: int
    ) -> str:
        cmd_lower = command.strip().lower()
        if not any(cmd_lower.startswith(w) for w in _SHELL_WHITELIST):
            return f"ERROR: command not permitted (sandbox unavailable): {command!r}"
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "HOME": str(workspace)},
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"ERROR: command timed out after {timeout}s"
            return stdout.decode(errors="replace").strip() or "(no output)"
        except Exception as e:
            return f"ERROR running command: {e}"
