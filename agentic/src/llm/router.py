"""Tandem pipeline dispatcher: classifies task tier and routes to run_fast/standard/strong."""

from __future__ import annotations

import logging

from .generator import GenerationResult

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = (
    "You are a task classifier. Reply with exactly one word: FAST, STANDARD, or STRONG.\n"
    "FAST: single function, syntax fix, hello world, trivial snippet\n"
    "STANDARD: complete module, API endpoint, data structure, test suite\n"
    "STRONG: multi-file changes, debugging, architecture, refactor, "
    "anything needing codebase context"
)

# Kept for backward compat with any code that imported the old enum
from enum import Enum


class TaskComplexity(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class TandemRouter:
    async def classify(self, prompt: str, language: str) -> str:
        """Ask THINKER to classify the task. Falls back to STANDARD on timeout/error."""
        from .backends.ollama import THINKER

        try:
            result = await THINKER.generate(
                f"Language: {language}\nTask: {prompt}",
                system=_CLASSIFIER_SYSTEM,
                temperature=0.0,
                max_tokens=50,
                timeout=10.0,
            )
            text = result.text.strip()
            # Strip deepseek-r1 thinking blocks
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            word = text.split()[0].upper() if text else "STANDARD"
            return word if word in ("FAST", "STANDARD", "STRONG") else "STANDARD"
        except Exception as exc:
            logger.warning("Classification failed (%s), defaulting to STANDARD", exc)
            return "STANDARD"

    async def route_and_run(
        self,
        prompt: str,
        language: str,
        context_files: list[str] | None = None,
        force_tier: str | None = None,
        web_context: bool = False,
    ) -> GenerationResult:
        from .backends.ollama import CODER
        from .tandem import run_fast, run_standard, run_strong

        # Multi-file always escalates to STRONG
        if context_files:
            tier = "STRONG"
        elif force_tier:
            tier = force_tier.upper()
        else:
            tier = await self.classify(prompt, language)

        logger.info("Tier=%s language=%s web_context=%s", tier, language, web_context)

        if tier == "FAST":
            try:
                return await run_fast(prompt, language)
            except Exception as e:
                logger.warning("FAST tier failed (%s), falling back to CODER direct", e)
                result = await CODER.generate(prompt)
                result.tier = "fast-fallback"
                return result

        if tier == "STANDARD":
            try:
                return await run_standard(prompt, language)
            except Exception as e:
                logger.warning("STANDARD failed (%s), falling back to CODER only", e)
                return await run_fast(prompt, language)

        # STRONG
        try:
            return await run_strong(prompt, language, context_files, web_context)
        except Exception as e:
            logger.warning("STRONG failed (%s), falling back to STANDARD", e)
            try:
                return await run_standard(prompt, language)
            except Exception as e2:
                logger.warning("STANDARD also failed (%s), CODER-only fallback", e2)
                return await run_fast(prompt, language)


_router: TandemRouter | None = None


def get_router() -> TandemRouter:
    global _router
    if _router is None:
        _router = TandemRouter()
    return _router
