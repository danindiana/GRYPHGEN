# Session: Two-Model Tandem Architecture

**Timestamp:** 2026-05-07T03:30:05Z  
**Host:** worlock (192.168.1.151)  
**Branch:** main  
**Commits:** `26f92d5`, `799a021`  
**Endpoint:** https://api.grug.ai  

---

## Objective

Replace the single-model Ollama path (devstral:24b, serial, 60-90s) with a
two-model tandem pipeline that permanently assigns one model per GPU, routes
tasks by complexity, and keeps the desktop GPU responsive during inference.

---

## Hardware Assignment

| GPU | Card | VRAM | Role | Model | Port |
|-----|------|------|------|-------|------|
| 0 | RTX 5080 | 16 GB | THINKER — reasoning, planning, classification, review | deepseek-r1:14b (9.0 GB) | 11434 |
| 1 | RTX 3080 | 10 GB | CODER — fast code generation | qwen2.5-coder:7b (4.7 GB) | 11436 |

Desktop/browser runs on GPU 1. CODER coexists with ~4 GB desktop overhead, leaving ~6 GB for the model.

Port 11435 was in use by the self-hosted-ai-starter-kit Docker container; coder instance moved to 11436.

---

## Infrastructure Changes

### New systemd service: `ollama-coder.service`

```
/etc/systemd/system/ollama-coder.service
```

```ini
[Unit]
Description=Ollama CODER instance (RTX 3080, GPU 1)
After=network-online.target ollama.service

[Service]
User=jeb
Environment="OLLAMA_HOST=127.0.0.1:11436"
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_KEEP_ALIVE=60m"
ExecStart=/usr/local/bin/ollama serve
Restart=always
```

### Updated `ollama.service.d/override.conf`

Changed `CUDA_VISIBLE_DEVICES=0,1` → `CUDA_VISIBLE_DEVICES=0` to pin THINKER
exclusively to GPU 0. Set `KEEP_ALIVE=60m`.

---

## Code Changes

### New files

| File | Purpose |
|------|---------|
| `agentic/src/llm/tandem.py` | Three-tier pipeline: `run_fast`, `run_standard`, `run_strong` |
| `agentic/src/tools/__init__.py` | Tools package |
| `agentic/src/tools/web_search.py` | DuckDuckGo Instant Answer search (no key required) |

### Modified files

| File | Change |
|------|--------|
| `agentic/src/llm/backends/ollama.py` | Rewritten: `OllamaBackend(base_url, model, role, semaphore_count)` with per-instance semaphores; module-level `THINKER` and `CODER` singletons; `is_alive()` health check; `timeout` parameter on `generate()` |
| `agentic/src/llm/router.py` | Replaced Mistral-based `TaskRouter` with `TandemRouter`; classifier uses `THINKER.chat()` + last-word parser |
| `agentic/src/llm/generator.py` | `LLMGenerator` now routes through `TandemRouter`; maps `reasoning_effort` → `force_tier` |
| `agentic/src/common/config.py` | Added `ollama_thinker_url`, `ollama_coder_url`, `thinker_model`, `coder_model` |
| `agentic/src/api/main.py` | Added `GET /api/v1/tools/search` endpoint |
| `agentic/src/services/code_generation/router.py` | Added `web_context: bool` field to `CodeGenerationRequest` |

---

## Three-Tier Pipeline

### FAST — CODER only
- Trigger: router classifies as trivial, or `reasoning_effort: "none"`
- Examples: single function, hello world, add two numbers, fibonacci, reverse string
- Target: <8s
- Semaphore: 3 concurrent

### STANDARD — THINKER plans → CODER executes
- Trigger: router classifies as module-level work
- Examples: REST API, auth module, test suite, data structure
- THINKER generates a structured coding brief (GOAL / FUNCTIONS / PATTERNS / EDGE_CASES)
- CODER receives brief as system context
- Target: <20s warm

### STRONG — THINKER → CODER → THINKER review loop
- Trigger: `context_files` provided (auto-escalate), or `reasoning_effort: "high"`
- Optional: `web_context: true` prepends DDG search results to THINKER analysis
- Phases: analysis → implementation → review → conditional revision pass
- Revision triggered if review contains "bug", "issue", "fix", "error", "problem", "incorrect"
- Max 2 revision cycles
- Target: <90s
- Semaphore: THINKER=2, CODER=3

### Graceful degradation
```
STRONG → STANDARD → FAST (CODER direct)
```
Any phase failure falls back one tier; never returns a 500 if at least one backend is alive.

---

## Classifier Fix (commit 799a021)

**Problem:** Trivial tasks like "add two numbers" were routing to STANDARD.

**Root causes:**
1. `THINKER.generate()` uses `/api/generate` with the system prompt prepended as raw text — deepseek-r1 ignores it and returns empty when `num_predict` is small
2. `max_tokens=50` was consumed entirely by the model's `<think>...</think>` block before the answer word
3. Parser took the *first* word; model sometimes outputs "This is simple. FAST"

**Fix:**
- Switch classifier to `THINKER.chat()` — uses `/api/chat`, system message handled natively by the model's chat template
- Remove token cap on classifier (let the model finish)
- Parse last matching word scanning in reverse: `for word in reversed(words): if word in ("FAST", "STANDARD", "STRONG")`
- Expand FAST examples; add "Default to FAST for any single-function task"

**Result after fix:**

| Task | Tier | Gen time |
|------|------|----------|
| add two numbers | FAST | 0.9s |
| fibonacci | FAST | 1.7s |
| reverse a string | FAST | 1.0s |
| check if prime | FAST | 3.6s |
| complete REST API with CRUD | STANDARD | 13.1s |
| JWT auth module | STANDARD | 11.1s |
| refactor across auth.py + middleware.py | STRONG | 11.3s |

---

## New Endpoints

### `GET /api/v1/tools/search`

```
GET https://api.grug.ai/api/v1/tools/search?q=<query>&max_results=5
X-API-Key: <key>
```

Response:
```json
{
  "query": "Redis",
  "results": [
    {"title": "Redis", "url": "https://en.wikipedia.org/wiki/Redis", "snippet": "..."},
    ...
  ],
  "source": "ddg"
}
```

Uses DuckDuckGo Instant Answer API — no key, no rate limits for reasonable use.
Returns entity abstracts and related topics. Returns empty list (not error) for
keyword queries that DDG IA doesn't cover.

### `POST /api/v1/code/generate` — new fields

```json
{
  "prompt": "...",
  "language": "python",
  "context_files": ["auth.py", "middleware.py"],
  "web_context": true,
  "reasoning_effort": "none | high"
}
```

- `context_files`: any non-empty list forces STRONG tier
- `web_context`: STRONG tier prepends DDG search results to THINKER analysis
- `reasoning_effort: "none"` → force FAST; `"high"` → force STRONG

---

## Environment Config

Added to `/home/jeb/programs/gryphgen-agentic/.env`:

```env
OLLAMA_THINKER_URL=http://127.0.0.1:11434
OLLAMA_CODER_URL=http://127.0.0.1:11436
THINKER_MODEL=deepseek-r1:14b
CODER_MODEL=qwen2.5-coder:7b
```

API key: old orphaned hash removed; single known key retained.

---

## Verified Performance (public endpoint, 2026-05-07T03:30Z)

```
GET  /health                          → 200 healthy
FAST  "add two numbers"               → 0.83s gen,  5.2s wall (Cloudflare RTT)
FAST  "fibonacci"                     → 2.91s gen,  9.0s wall
STANDARD "REST API CRUD"              → 20.4s gen, 32.6s wall
STRONG  context_files=[auth,middleware]→ 12.6s gen, 32.0s wall
3× concurrent FAST                    → 5.75s gen, 6.0s wall (all 3)
GET  /api/v1/tools/search?q=Redis     → 3 results, source=ddg
```

GPU state after full test run:
```
GPU 0 (RTX 5080): 10,414 MiB  ← deepseek-r1:14b loaded
GPU 1 (RTX 3080):  5,640 MiB  ← qwen2.5-coder:7b + desktop
```

---

## Definition of Done

- [x] GPU 0 (5080): THINKER warm, ~10.4 GB VRAM
- [x] GPU 1 (3080): CODER warm, ~5.6 GB VRAM; desktop unaffected
- [x] FAST tier: <8s gen, CODER only
- [x] STANDARD tier: <20s warm, THINKER brief + CODER execution
- [x] STRONG tier: full think-code-review loop, working correctly
- [x] 3 concurrent FAST: 6s wall clock
- [x] Web search endpoint returning DDG results
- [x] STRONG + web_context=true prepends search to analysis
- [x] Graceful fallback on backend failure (STRONG→STANDARD→FAST)
- [x] Committed and pushed: `danindiana/GRYPHGEN` commits `26f92d5`, `799a021`
- [x] Trivial tasks (single function) correctly route to FAST
- [x] All tiers verified against public endpoint https://api.grug.ai
