"""OpenAI backend."""

import time

from ..generator import GenerationResult


class OpenAIBackend:
    """Calls the OpenAI API (also works with any OpenAI-compatible endpoint)."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
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

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=effective_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = time.monotonic() - t0
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0

        return GenerationResult(
            text=text,
            model=effective_model,
            tokens_used=tokens,
            generation_time=elapsed,
        )

    async def list_models(self) -> list[str]:
        models = await self.client.models.list()
        return [m.id for m in models.data if "gpt" in m.id]
