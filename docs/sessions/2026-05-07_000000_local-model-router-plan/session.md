# Session: Local Multi-Model Router — Plan (Cold Storage)
## 2026-05-07

**Status: ARCHIVED — direction change before implementation**

This session produced a complete implementation plan for a 4-tier local model
routing system. Work was halted before any code changes were made. The plan is
preserved here as-is for reference.

---

## Why this was explored

`api.grug.ai` was routing every request to `devstral:24b` — a 24B model split
across both GPUs via PCIe — causing ~65s per request. The Mistral API router built
in the previous session (`2026-05-06_215156_mistral-small4-router`) was never
activated because `MISTRAL_API_KEY` remains unset.

The goal of this session was to build a fully local routing system: a tiny ~1GB
model classifies each request, then dispatches to a purpose-built coder model at
the appropriate tier. No cloud API needed for routing.

---

## Model inventory (as of 2026-05-07)

From `ollama list` on worlock:

| Model | Size | Candidate tier |
|-------|------|----------------|
| `qwen3.5:0.8b` | 1.0 GB | ROUTER — 1-word classifier |
| `qwen3:4b` | 2.5 GB | ROUTER alt |
| `qwen3.5:4b` | 3.4 GB | FAST alt |
| `qwen2.5-coder:7b` | 4.7 GB | **FAST** — dedicated coder |
| `deepseek-r1:8b` | 5.2 GB | FAST alt (reasoning) |
| `ministral-3:8b` | 6.0 GB | STANDARD alt |
| `qwen3.5:9b` | 6.6 GB | STANDARD alt |
| `qwen2.5-coder:14b` | 9.0 GB | **STANDARD** — dedicated coder |
| `deepseek-coder-v2:16b` | 8.9 GB | STANDARD alt |
| `devstral:24b` | 14 GB | **STRONG** — current default |

Selected tiers:
- **ROUTER**: `qwen3.5:0.8b` (1 GB, always hot)
- **FAST**: `qwen2.5-coder:7b` (4.7 GB, always hot)
- **STANDARD**: `qwen2.5-coder:14b` (9 GB, on demand)
- **STRONG**: `devstral:24b` (14 GB, on demand)

VRAM budget on RTX 5080 (16 GB):
- ROUTER + FAST hot: 5.7 GB
- All three: 14.7 GB (fits with ~1.3 GB headroom)
- STRONG alone: 14 GB (evicts others)

---

## Hardware context

- GPU 0: RTX 5080, 16 GB → Ollama target (index confirmed via `nvidia-smi`)
- GPU 1: RTX 3080, 10 GB → desktop/display, must stay free
- Current Ollama: `CUDA_VISIBLE_DEVICES=0,1` → both GPUs, causes PCIe transfer
- Target: `CUDA_VISIBLE_DEVICES=0` → 5080 only

---

## Full implementation plan (archived)

### Step 1 — Ollama systemd override

Edit `/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_HOST=127.0.0.1"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_GPU_OVERHEAD=512MiB"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_KEEP_ALIVE=15m"
```

Then: `sudo systemctl daemon-reload && sudo systemctl restart ollama`

### Step 2 — Rewrite `src/llm/router.py`

New 4-tier local router, dropping Mistral API dependency for classification:

```python
class TaskTier(Enum):
    FAST     = "fast"
    STANDARD = "standard"
    STRONG   = "strong"

TIER_MAP = {
    TaskTier.FAST:     os.getenv("MODEL_FAST",     "qwen2.5-coder:7b"),
    TaskTier.STANDARD: os.getenv("MODEL_STANDARD", "qwen2.5-coder:14b"),
    TaskTier.STRONG:   os.getenv("MODEL_STRONG",   "devstral:24b"),
}

CLASSIFICATION_PROMPT = """Reply with exactly one word: FAST, STANDARD, or STRONG.
FAST: single function, hello world, syntax fix, trivial snippet, explain concept
STANDARD: complete module, API endpoint, test suite, data structure, CLI tool
STRONG: multi-file changes, debug existing code, refactor, architecture, agent task
Language: {language}
Task: {prompt}"""
```

`TaskRouter.classify()`: calls `qwen3.5:0.8b`, `max_tokens=5`, `temperature=0`,
`asyncio.wait_for(timeout=2.0)`, defaults to STANDARD on failure.

`TaskRouter.route()`: context_files present → STRONG; else classify.

### Step 3 — Update `src/llm/backends/ollama.py`

Per-tier semaphores replacing single semaphore:

```python
_SEMAPHORES = {
    "route":    asyncio.Semaphore(4),
    "fast":     asyncio.Semaphore(3),
    "standard": asyncio.Semaphore(2),
    "strong":   asyncio.Semaphore(1),
}
```

Add `tier` and `keep_alive` params to `generate()`/`chat()`.
Keep-alive defaults: route/fast → "30m", standard → "10m", strong → "5m".

Add `preload_hot_models(base_url, router_model, fast_model)` — sends dummy
prompts with `keep_alive="60m"` at startup; failures logged, not raised.

### Step 4 — Update `src/llm/generator.py`

- `GenerationResult` gets `classification_time: float = 0.0`
- `LLMGenerator` always builds a `LocalTaskRouter` (no Mistral key dependency)
- `_generate_routed()` records timing, maps `TaskTier` → model via `TIER_MAP`
- Passes `tier=cfg.tier.value` to `ollama.generate()` for semaphore selection

### Step 5 — Update `src/services/code_generation/router.py`

Request: add `force_tier: Optional[str] = None`
Response: add `classification_ms: int = 0`, `generation_ms: int = 0`

### Step 6 — Add model preloading to `src/api/main.py`

In `lifespan()` before `yield`:
```python
asyncio.create_task(preload_hot_models(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    router_model=os.environ.get("MODEL_ROUTER", "qwen3.5:0.8b"),
    fast_model=os.environ.get("MODEL_FAST", "qwen2.5-coder:7b"),
))
```

### Step 7 — DDG web search tool

New files:
- `src/tools/__init__.py`
- `src/tools/web_search.py` — DDG Instant Answer API, httpx async, timeout=5s
- `src/tools/router.py` — `GET /api/v1/tools/search?q=<query>`, API key auth,
  rate limited to 20/day, returns top-5 results

### Step 8 — `.env` additions

```
MODEL_ROUTER=qwen3.5:0.8b
MODEL_FAST=qwen2.5-coder:7b
MODEL_STANDARD=qwen2.5-coder:14b
MODEL_STRONG=devstral:24b
```

---

## Expected outcomes (had this been implemented)

| Scenario | Expected latency | Tier |
|----------|-----------------|------|
| "write hello world" | < 8s | fast |
| "write a REST API with JWT" | < 20s | standard |
| 3 concurrent simple requests | ~8s total | fast (parallel) |
| "refactor auth module" + context_files | 60–90s | strong |

---

## Files that would have been modified

| File | Change |
|------|--------|
| `/etc/systemd/system/ollama.service.d/override.conf` | Pin GPU 0, MAX_LOADED_MODELS=3 |
| `src/llm/router.py` | Full rewrite — local 4-tier routing |
| `src/llm/backends/ollama.py` | Per-tier semaphores + keep_alive |
| `src/llm/generator.py` | Always-on local router, classification_time |
| `src/services/code_generation/router.py` | force_tier, classification_ms, generation_ms |
| `src/api/main.py` | Preload hot models at startup |
| `.env` | Add MODEL_ROUTER/FAST/STANDARD/STRONG |
| `src/tools/__init__.py` | New |
| `src/tools/web_search.py` | New |
| `src/tools/router.py` | New |

---

## Current production state (unchanged)

- Service: `gryphgen-agentic` running at `api.grug.ai`
- All traffic → `devstral:24b` via Ollama (MISTRAL_API_KEY unset)
- No changes were made to any production files in this session
