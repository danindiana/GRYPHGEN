"""Ollama local LLM backends — THINKER (GPU 0, port 11434) and CODER (GPU 1, port 11436)."""

import asyncio
import logging
import os
import time

import httpx

from ..generator import GenerationResult

logger = logging.getLogger(__name__)


class OllamaBackend:
    """HTTP client for a single named Ollama instance."""

    def __init__(
        self,
        base_url: str,
        model: str,
        role: str,
        semaphore_count: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.role = role
        self._sem = asyncio.Semaphore(semaphore_count)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
        keep_alive: str = "30m",
        timeout: float = 300.0,
    ) -> GenerationResult:
        effective_model = model or self.model
        t0 = time.monotonic()

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        async with self._sem:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": effective_model,
                        "prompt": full_prompt,
                        "stream": False,
                        "keep_alive": keep_alive,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()

        elapsed = time.monotonic() - t0
        text = data.get("response", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return GenerationResult(
            text=text,
            model=effective_model,
            tokens_used=tokens,
            generation_time=elapsed,
            backend_used=f"ollama-{self.role}",
        )

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        keep_alive: str = "30m",
    ) -> GenerationResult:
        effective_model = model or self.model
        t0 = time.monotonic()

        async with self._sem:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": effective_model,
                        "messages": messages,
                        "stream": False,
                        "keep_alive": keep_alive,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()

        elapsed = time.monotonic() - t0
        text = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return GenerationResult(
            text=text,
            model=effective_model,
            tokens_used=tokens,
            generation_time=elapsed,
            backend_used=f"ollama-{self.role}",
        )

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def is_alive(self) -> bool:
        """Quick health check — False if instance unreachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


def _build_backends() -> tuple["OllamaBackend", "OllamaBackend"]:
    thinker_url = os.environ.get("OLLAMA_THINKER_URL", "http://127.0.0.1:11434")
    coder_url = os.environ.get("OLLAMA_CODER_URL", "http://127.0.0.1:11436")
    thinker_model = os.environ.get("THINKER_MODEL", "deepseek-r1:14b")
    coder_model = os.environ.get("CODER_MODEL", "qwen2.5-coder:7b")

    thinker = OllamaBackend(
        base_url=thinker_url,
        model=thinker_model,
        role="thinker",
        semaphore_count=2,
    )
    coder = OllamaBackend(
        base_url=coder_url,
        model=coder_model,
        role="coder",
        semaphore_count=3,
    )
    return thinker, coder


THINKER, CODER = _build_backends()
