"""Three-tier tandem pipeline: FAST (CODER only) | STANDARD (THINKER→CODER) | STRONG (T→C→T loop)."""

from __future__ import annotations

import logging

from .generator import GenerationResult

logger = logging.getLogger(__name__)


async def run_fast(prompt: str, language: str) -> GenerationResult:
    """CODER only. Simple codegen. Target: <8s."""
    from .backends.ollama import CODER

    system = (
        f"You are an expert {language} programmer. "
        "Write clean, working code. Return only the code, no explanation."
    )
    result = await CODER.generate(prompt, system=system)
    result.tier = "fast"
    return result


async def run_standard(prompt: str, language: str) -> GenerationResult:
    """THINKER plans, CODER executes. Target: <20s."""
    from .backends.ollama import CODER, THINKER

    brief_prompt = (
        f"Produce a brief coding plan (not code) for this task.\n"
        f"Language: {language}\nTask: {prompt}\n\n"
        "Reply in this exact format:\n"
        "GOAL: one sentence\n"
        "FUNCTIONS: comma separated list of function/class names needed\n"
        "PATTERNS: language-specific patterns or libraries to use\n"
        "EDGE_CASES: comma separated list"
    )
    brief = await THINKER.generate(brief_prompt, temperature=0.3, max_tokens=512)

    system = (
        f"You are an expert {language} programmer. "
        f"Follow this plan exactly:\n{brief.text}\n"
        "Write complete, working, well-commented code."
    )
    result = await CODER.generate(prompt, system=system)
    result.tier = "standard"
    result.tokens_used += brief.tokens_used
    return result


async def run_strong(
    prompt: str,
    language: str,
    context_files: list[str] | None = None,
    web_context: bool = False,
) -> GenerationResult:
    """THINKER→CODER→THINKER review. Acceptable latency: 60-90s."""
    from .backends.ollama import CODER, THINKER

    prefix = ""
    if web_context:
        try:
            from ..tools.web_search import search
            results = await search(f"{language} {prompt[:120]}", max_results=3)
            if results:
                snippets = "\n".join(
                    f"- {r['title']}: {r['snippet']}" for r in results
                )
                prefix = f"Web context:\n{snippets}\n\n"
        except Exception as e:
            logger.warning("Web search skipped: %s", e)

    ctx_note = f"\nContext files involved: {', '.join(context_files)}" if context_files else ""

    # Phase 1 — THINKER: deep analysis
    analysis = await THINKER.generate(
        f"{prefix}Analyze this task deeply. Identify risks and approach:\n{prompt}{ctx_note}",
        temperature=0.3,
        max_tokens=1024,
    )

    # Phase 2 — CODER: implement
    implementation = await CODER.generate(
        prompt,
        system=f"Expert {language} programmer. Plan:\n{analysis.text}",
    )

    # Phase 3 — THINKER: review
    review = await THINKER.generate(
        f"Review this {language} code for bugs, edge cases, improvements:\n"
        f"{implementation.text}\n\nOriginal task: {prompt}",
        temperature=0.2,
        max_tokens=512,
    )

    needs_revision = any(
        kw in review.text.lower()
        for kw in ("bug", "issue", "fix", "error", "problem", "incorrect")
    )

    if needs_revision:
        try:
            implementation = await CODER.generate(
                prompt,
                system=(
                    f"Revise based on this review:\n{review.text}\n"
                    f"Original plan:\n{analysis.text}"
                ),
            )
        except Exception as e:
            logger.warning("Revision pass failed (%s), keeping first implementation", e)

    total_tokens = analysis.tokens_used + implementation.tokens_used + review.tokens_used
    implementation.tier = "strong"
    implementation.tokens_used = total_tokens
    return implementation
