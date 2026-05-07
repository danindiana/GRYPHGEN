"""Mistral API backend with reasoning_effort support."""

from __future__ import annotations

import logging
import time

import httpx

from ..generator import GenerationResult

logger = logging.getLogger(__name__)

_MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralAPIBackend:
    """Calls api.mistral.ai with optional chain-of-thought via reasoning_effort."""

    def __init__(self, api_key: str, model: str = "mistral-small-latest") -> None:
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system: str | None = None,
        reasoning_effort: str = "none",
        timeout: float = 60.0,
    ) -> GenerationResult:
        effective_model = model or self.model
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": effective_model,
            "messages": messages,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    _MISTRAL_API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
            except httpx.TimeoutException:
                raise

            if resp.status_code == 401:
                raise RuntimeError("invalid MISTRAL_API_KEY")
            if resp.status_code == 429:
                raise RuntimeError("Mistral rate limited")
            if not resp.is_success:
                logger.error("Mistral API error %s: %s", resp.status_code, resp.text[:200])
                raise RuntimeError(f"Mistral API error {resp.status_code}")

            data = resp.json()

        elapsed = time.monotonic() - t0
        text = data["choices"][0]["message"]["content"] or ""
        tokens = data.get("usage", {}).get("total_tokens", 0)

        return GenerationResult(
            text=text,
            model=effective_model,
            tokens_used=tokens,
            generation_time=elapsed,
        )
