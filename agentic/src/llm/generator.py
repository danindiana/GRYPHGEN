"""LLM generator — tandem local pipeline (THINKER+CODER) with graceful Mistral fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CODEGEN_SYSTEM = """\
You are an expert software engineer. When given a code generation request, \
output ONLY the code block with no surrounding explanation unless \
documentation was explicitly requested. Always include a module-level \
docstring and proper type annotations."""


@dataclass
class GenerationResult:
    text: str
    model: str
    tokens_used: int
    generation_time: float
    tier: str = field(default="")
    reasoning_effort: str = field(default="")
    backend_used: str = field(default="")


class LLMGenerator:
    """Routing-aware code generation interface.

    Routes through the local THINKER+CODER tandem pipeline by default.
    Falls back to Mistral API if MISTRAL_API_KEY is set and backend='mistral'.
    """

    def __init__(self) -> None:
        self._mistral = self._build_mistral_backend()

    @staticmethod
    def _build_mistral_backend():
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            return None
        try:
            from .backends.mistral_api import MistralAPIBackend
            return MistralAPIBackend(
                api_key=api_key,
                model=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
            )
        except Exception:
            return None

    async def generate(
        self,
        prompt: str,
        language: str = "python",
        framework: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        include_tests: bool = False,
        include_docs: bool = False,
        style_guide: str | None = None,
        context_files: list[str] | None = None,
        reasoning_effort: str | None = None,
        web_context: bool = False,
    ) -> GenerationResult:
        parts = [f"Generate {language} code"]
        if framework:
            parts.append(f"using {framework}")
        parts.append(f"for the following requirement:\n\n{prompt}")
        if style_guide:
            parts.append(f"\n\nFollow the {style_guide} style guide.")
        if include_tests:
            parts.append("\n\nAlso provide unit tests for the generated code.")
        if include_docs:
            parts.append("\n\nAlso provide clear documentation for each function/class.")

        full_prompt = " ".join(parts)

        # Map legacy reasoning_effort → force_tier
        force_tier: str | None = None
        if reasoning_effort == "none":
            force_tier = "FAST"
        elif reasoning_effort == "high":
            force_tier = "STRONG"

        from .router import get_router
        result = await get_router().route_and_run(
            prompt=full_prompt,
            language=language,
            context_files=context_files,
            force_tier=force_tier,
            web_context=web_context,
        )
        result.reasoning_effort = reasoning_effort or ""
        return result

    async def summarize(self, text: str, model: str | None = None) -> GenerationResult:
        from .backends.ollama import THINKER
        prompt = f"Summarize the following in 3-5 bullet points:\n\n{text}"
        result = await THINKER.generate(prompt=prompt, temperature=0.3)
        result.backend_used = "ollama-thinker"
        return result

    async def list_models(self) -> list[str]:
        from .backends.ollama import CODER, THINKER
        models: set[str] = set()
        try:
            for m in await THINKER.list_models():
                models.add(f"thinker:{m}")
        except Exception:
            models.add(f"thinker:{THINKER.model}")
        try:
            for m in await CODER.list_models():
                models.add(f"coder:{m}")
        except Exception:
            models.add(f"coder:{CODER.model}")
        return sorted(models)


_generator: LLMGenerator | None = None


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator


def reset_generator() -> None:
    """Force re-initialisation of the singleton (useful after env changes in tests)."""
    global _generator
    _generator = None
