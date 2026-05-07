# Session: Mistral Small 4 Router — 2026-05-06

## Summary

Implemented Mistral Small 4 as primary LLM backend with async task routing,
replacing the single-path devstral:24b Ollama bottleneck.

## Changes

| File | Change |
|------|--------|
| `agentic/src/llm/backends/mistral_api.py` | NEW — async httpx client, reasoning_effort support |
| `agentic/src/llm/router.py` | NEW — TaskRouter, classify(), route(), ModelConfig |
| `agentic/src/auth/rate_limit.py` | NEW (dev copy) — copied from production |
| `agentic/src/llm/generator.py` | Refactored to routing-aware, GenerationResult +3 fields |
| `agentic/src/llm/backends/ollama.py` | asyncio.Semaphore(2) for dual-GPU concurrency |
| `agentic/src/services/code_generation/router.py` | context_files, reasoning_effort in request; tier/backend_used in response |
| `agentic/src/common/config.py` | mistral_api_key, mistral_model, gryphgen_worker_concurrency settings |
| `/home/jeb/programs/gryphgen-agentic/.env` | Appended MISTRAL_API_KEY, MISTRAL_MODEL, GRYPHGEN_WORKER_CONCURRENCY |

## Routing logic

- SIMPLE / STANDARD → mistral-api, reasoning_effort=none
- COMPLEX → mistral-api, reasoning_effort=high
- context_files present → forced COMPLEX
- force_model containing ":" or non-Mistral prefix → routed to ollama
- MISTRAL_API_KEY empty → all requests go directly to Ollama (no crash)

## Current state

MISTRAL_API_KEY is not yet set. All traffic routes to devstral:24b via Ollama.
Response shape includes `tier`, `reasoning_effort_used`, `backend_used` fields.
To activate Mistral: set MISTRAL_API_KEY in .env and restart service.

## Definition of Done (no key)
- [x] Service restarts without errors
- [x] All requests fall back to Ollama cleanly
- [x] Response JSON includes tier, reasoning_effort_used, backend_used
- [x] Rate limiting preserved (rate_limit.py synced to dev)
- [x] Ollama semaphore allows 2 concurrent requests

## Definition of Done (with key — pending)
- [ ] Set MISTRAL_API_KEY + GRYPHGEN_LLM_BACKEND=mistral-api in .env
- [ ] SIMPLE task returns in < 8s
- [ ] COMPLEX task shows reasoning_effort_used: "high"
- [ ] Timeout falls back to Ollama without 500 error
