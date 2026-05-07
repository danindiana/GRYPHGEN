"""LLM generator abstraction — selects backend from GRYPHGEN_LLM_BACKEND env var."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.ollama import OllamaBackend
    from .backends.anthropic import AnthropicBackend
    from .backends.openai import OpenAIBackend

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


class LLMGenerator:
    """Unified code generation interface across Ollama, Anthropic, and OpenAI backends."""

    def __init__(self) -> None:
        self._backend = self._build_backend()

    @staticmethod
    def _build_backend():
        backend_name = os.environ.get("GRYPHGEN_LLM_BACKEND", "ollama").lower()

        if backend_name == "ollama":
            from .backends.ollama import OllamaBackend
            return OllamaBackend(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                model=os.environ.get("OLLAMA_MODEL", "devstral:24b"),
            )

        if backend_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            from .backends.anthropic import AnthropicBackend
            return AnthropicBackend(
                api_key=api_key,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            )

        if backend_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            from .backends.openai import OpenAIBackend
            return OpenAIBackend(
                api_key=api_key,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

        raise ValueError(f"Unknown GRYPHGEN_LLM_BACKEND: {backend_name!r}")

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

        return await self._backend.generate(
            prompt=full_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=_CODEGEN_SYSTEM if hasattr(self._backend, "generate") else None,
        )

    async def summarize(self, text: str, model: str | None = None) -> GenerationResult:
        prompt = f"Summarize the following in 3-5 bullet points:\n\n{text}"
        return await self._backend.generate(prompt=prompt, model=model, temperature=0.3)

    async def list_models(self) -> list[str]:
        if hasattr(self._backend, "list_models"):
            return await self._backend.list_models()
        return []


_generator: LLMGenerator | None = None


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator
