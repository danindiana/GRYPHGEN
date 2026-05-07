"""Anthropic Claude backend with prompt caching."""

import time
from typing import Optional

from ..generator import GenerationResult


class AnthropicBackend:
    """Calls the Anthropic API using the anthropic SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        try:
            import anthropic as sdk
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        self._sdk = sdk
        self.client = sdk.AsyncAnthropic(api_key=api_key)
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

        system_blocks = []
        if system:
            system_blocks = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        resp = await self.client.messages.create(
            model=effective_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks or self._sdk.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )

        elapsed = time.monotonic() - t0
        text = resp.content[0].text if resp.content else ""
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)

        return GenerationResult(
            text=text,
            model=effective_model,
            tokens_used=tokens,
            generation_time=elapsed,
        )

    async def list_models(self) -> list[str]:
        return [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]
