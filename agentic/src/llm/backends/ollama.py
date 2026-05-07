"""Ollama local LLM backend."""

import json
import time
from typing import AsyncIterator

import httpx

from ..generator import GenerationResult


class OllamaBackend:
    """Calls a local Ollama instance via its HTTP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "devstral:24b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> GenerationResult:
        effective_model = model or self.model
        t0 = time.monotonic()

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": effective_model,
                    "prompt": full_prompt,
                    "stream": False,
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
        )

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        effective_model = model or self.model
        t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": effective_model,
                    "messages": messages,
                    "stream": False,
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
        )

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m["name"] for m in data.get("models", [])]
