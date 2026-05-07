"""LLM generator — selects backend from env, routes via TaskRouter when Mistral key present."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.mistral_api import MistralAPIBackend
    from .backends.ollama import OllamaBackend
    from .router import TaskRouter

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
    tier: str = field(default="")             # simple | standard | complex | forced
    reasoning_effort: str = field(default="")  # none | high
    backend_used: str = field(default="")     # mistral-api | ollama


class LLMGenerator:
    """Routing-aware code generation interface.

    If MISTRAL_API_KEY is set the TaskRouter classifies each request and
    dispatches to Mistral (with reasoning_effort) or Ollama.  When the key
    is absent every request goes directly to Ollama.
    """

    def __init__(self) -> None:
        self._ollama = self._build_ollama_backend()
        self._mistral = self._build_mistral_backend()
        self._router: TaskRouter | None = None
        if self._mistral is not None:
            from .router import TaskRouter as _TR
            self._router = _TR(self._mistral)

    # ── backend factories ────────────────────────────────────────────────────

    @staticmethod
    def _build_ollama_backend() -> "OllamaBackend":
        from .backends.ollama import OllamaBackend
        return OllamaBackend(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("OLLAMA_MODEL", "devstral:24b"),
        )

    @staticmethod
    def _build_mistral_backend() -> "MistralAPIBackend | None":
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            return None
        from .backends.mistral_api import MistralAPIBackend
        return MistralAPIBackend(
            api_key=api_key,
            model=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
        )

    # ── public interface ─────────────────────────────────────────────────────

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

        if self._router is not None:
            return await self._generate_routed(
                full_prompt, language, model, temperature, max_tokens,
                context_files, reasoning_effort,
            )

        # No Mistral key — straight to Ollama
        result = await self._ollama.generate(
            prompt=full_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=_CODEGEN_SYSTEM,
        )
        result.backend_used = "ollama"
        return result

    async def _generate_routed(
        self,
        full_prompt: str,
        language: str,
        force_model: str | None,
        temperature: float,
        max_tokens: int,
        context_files: list[str] | None,
        reasoning_effort_override: str | None,
    ) -> GenerationResult:
        assert self._router is not None
        assert self._mistral is not None

        cfg = await self._router.route(
            prompt=full_prompt,
            language=language,
            context_files=context_files,
            force_model=force_model,
            mistral_model=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "devstral:24b"),
        )

        eff_effort = reasoning_effort_override or cfg.reasoning_effort

        if cfg.backend == "mistral-api":
            try:
                result = await self._mistral.generate(
                    prompt=full_prompt,
                    model=cfg.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=_CODEGEN_SYSTEM,
                    reasoning_effort=eff_effort,
                )
                result.tier = cfg.rationale
                result.reasoning_effort = eff_effort
                result.backend_used = "mistral-api"
                return result
            except Exception as exc:
                logger.warning("Mistral backend failed (%s), falling back to Ollama", exc)

        # Ollama path (either forced or fallback from Mistral failure)
        result = await self._ollama.generate(
            prompt=full_prompt,
            model=cfg.model if cfg.backend == "ollama" else None,
            temperature=temperature,
            max_tokens=max_tokens,
            system=_CODEGEN_SYSTEM,
        )
        result.tier = cfg.rationale
        result.reasoning_effort = eff_effort
        result.backend_used = "ollama"
        return result

    async def summarize(self, text: str, model: str | None = None) -> GenerationResult:
        prompt = f"Summarize the following in 3-5 bullet points:\n\n{text}"
        result = await self._ollama.generate(prompt=prompt, model=model, temperature=0.3)
        result.backend_used = "ollama"
        return result

    async def list_models(self) -> list[str]:
        return await self._ollama.list_models()


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
