"""DuckDuckGo Instant Answer web search — no API key required."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_DDG_URL = "https://api.duckduckgo.com/"


async def search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo; returns [{title, url, snippet}].

    Falls back to empty list on any error — never crashes the pipeline.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                _DDG_URL,
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                headers={"User-Agent": "gryphgen-agentic/1.0"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("DDG search failed: %s", exc)
        return []

    results: list[dict] = []

    if data.get("AbstractText") and data.get("AbstractURL"):
        results.append({
            "title": data.get("Heading", query),
            "url": data["AbstractURL"],
            "snippet": data["AbstractText"][:300],
        })

    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
            results.append({
                "title": topic.get("Text", "")[:80],
                "url": topic["FirstURL"],
                "snippet": topic.get("Text", "")[:300],
            })

    return results[:max_results]
