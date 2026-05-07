"""Tandem pipeline dispatcher: classifies task tier and routes to run_fast/standard/strong."""

from __future__ import annotations

import logging

from .generator import GenerationResult

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = (
    "Classify this coding task as FAST, STANDARD, or STRONG. Reply with exactly one word.\n\n"
    "FAST — single function, trivial logic: hello world, add two numbers, reverse a string, "
    "check if prime, simple loop, basic math, one-liner utility\n"
    "STANDARD — multiple functions, one module, API endpoint, class with methods, "
    "algorithm, test suite, data structure implementation\n"
    "STRONG — multi-file changes, debugging existing code, architecture, security systems, "
    "anything mentioning context files or cross-module refactoring\n\n"
    "Default to FAST for any single-function task. Reply with one word only."
)

# Kept for backward compat with any code that imported the old enum
from enum import Enum


class TaskComplexity(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class TandemRouter:
    async def classify(self, prompt: str, language: str) -> str:
        """Ask THINKER to classify the task. Falls back to STANDARD on timeout/error.

        Uses chat API so the system prompt is handled natively by deepseek-r1.
        Parses the last word of the response — the model sometimes prepends reasoning.
        """
        from .backends.ollama import THINKER

        messages = [
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": f"Language: {language}\nTask: {prompt}"},
        ]
        try:
            result = await THINKER.chat(
                messages,
                temperature=0.0,
            )
            text = result.text.strip()
            # Take last word — model sometimes leads with a reasoning sentence
            words = [w.upper().rstrip(".,!") for w in text.split() if w.strip()]
            for word in reversed(words):
                if word in ("FAST", "STANDARD", "STRONG"):
                    return word
            logger.warning("Classifier returned unexpected text %r, defaulting to STANDARD", text[:80])
            return "STANDARD"
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
