"""Task complexity classifier and model config selector."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .backends.mistral_api import MistralAPIBackend

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = (
    "You are a task classifier. Reply with exactly one word: "
    "SIMPLE, STANDARD, or COMPLEX.\n"
    "SIMPLE: single function, syntax fix, hello world, trivial snippet\n"
    "STANDARD: complete module, API endpoint, data structure, test suite\n"
    "COMPLEX: multi-file changes, debugging existing code, "
    "architecture, refactor, anything needing codebase context"
)

_MISTRAL_PREFIXES = ("mistral", "codestral", "mixtral", "magistral")


class TaskComplexity(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class ModelConfig(BaseModel):
    backend: str           # "mistral-api" | "ollama"
    model: str
    reasoning_effort: str  # "none" | "high"
    rationale: str         # complexity tier that drove this choice


def _is_ollama_model(name: str) -> bool:
    """Ollama models typically carry a tag (e.g. devstral:24b); Mistral names don't."""
    if ":" in name:
        return True
    return not any(name.lower().startswith(p) for p in _MISTRAL_PREFIXES)


class TaskRouter:
    def __init__(self, fast_backend: "MistralAPIBackend") -> None:
        self._fast = fast_backend  # used only for cheap classification calls

    async def classify(
        self,
        prompt: str,
        language: str,
        context_files: list[str] | None = None,
    ) -> TaskComplexity:
        user_msg = f"Language: {language}\nTask: {prompt}"
        if context_files:
            user_msg += f"\nContext files: {', '.join(context_files)}"
        try:
            result = await self._fast.generate(
                prompt=user_msg,
                system=_CLASSIFIER_SYSTEM,
                reasoning_effort="none",
                max_tokens=10,
                temperature=0.0,
                timeout=3.0,
            )
            word = result.text.strip().split()[0].upper()
            return {
                "SIMPLE": TaskComplexity.SIMPLE,
                "STANDARD": TaskComplexity.STANDARD,
                "COMPLEX": TaskComplexity.COMPLEX,
            }.get(word, TaskComplexity.STANDARD)
        except Exception as exc:
            logger.warning("Task classification failed (%s), defaulting to STANDARD", exc)
            return TaskComplexity.STANDARD

    async def route(
        self,
        prompt: str,
        language: str,
        context_files: list[str] | None = None,
        force_model: str | None = None,
        mistral_model: str = "mistral-small-latest",
        ollama_model: str = "devstral:24b",
    ) -> ModelConfig:
        if force_model:
            if _is_ollama_model(force_model):
                return ModelConfig(
                    backend="ollama",
                    model=force_model,
                    reasoning_effort="none",
                    rationale="forced",
                )
            return ModelConfig(
                backend="mistral-api",
                model=force_model,
                reasoning_effort="none",
                rationale="forced",
            )

        # Presence of context_files implies multi-file / complex work
        if context_files:
            return ModelConfig(
                backend="mistral-api",
                model=mistral_model,
                reasoning_effort="high",
                rationale=TaskComplexity.COMPLEX.value,
            )

        complexity = await self.classify(prompt, language, context_files)

        if complexity == TaskComplexity.COMPLEX:
            return ModelConfig(
                backend="mistral-api",
                model=mistral_model,
                reasoning_effort="high",
                rationale=complexity.value,
            )
        return ModelConfig(
            backend="mistral-api",
            model=mistral_model,
            reasoning_effort="none",
            rationale=complexity.value,
        )
