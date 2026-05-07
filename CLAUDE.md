# GRYPHGEN — Claude Code Context

> **Status:** `api.grug.ai` LIVE as of 2026-05-06T21:05:34-0500  
> Systemd service `gryphgen-agentic` on worlock:8090 → Cloudflare tunnel → https://api.grug.ai  
> Session doc: `~/Documents/claude_creations/2026-05-06_gryphgen-modernization/session.md`

## What this repo is
GRYPHGEN (Grid Resource Prioritization in Heterogeneous Environments) is a Python
agentic framework for automated software production. The three core modules are:

- **SYMORQ** (`gstruct/SYMORQ/`) — async ZeroMQ orchestrator, task queue, worker pool
- **SYMORG** (`gstruct/SYMORG/`) — RAG-based task scheduler with networkx dependency graph  
- **SYMAUG** (`gstruct/SYMAUG/`) — Docker microservice deployer

The **agentic API** (`agentic/`) is a FastAPI service that wraps these components and
exposes them as an HTTP API. This is the api.grug.ai product layer.

## Repository structure

```
GRYPHGEN/
  agentic/               ← FastAPI API (api.grug.ai)
    src/
      api/main.py        ← FastAPI app entry point
      llm/               ← LLM backend abstraction (added 2026-05-06)
        generator.py     ← LLMGenerator, backend selection via GRYPHGEN_LLM_BACKEND
        backends/
          ollama.py      ← Local Ollama (default)
          anthropic.py   ← Anthropic API with prompt caching
          openai.py      ← OpenAI API
      services/
        code_generation/ ← POST /api/v1/code/generate (wired to LLM 2026-05-06)
        automated_testing/
        project_management/
        documentation/
        collaboration/
        self_improvement/
      auth/              ← JWT auth (plumbed, not enforced yet)
      common/config.py   ← pydantic-settings; all env vars documented here
  gstruct/               ← Core Python modules (SYMORQ, SYMORG, SYMAUG)
  MCP_SERVER/            ← Haskell MCP reliability system (separate)
  LLM-Sandbox CLI/       ← ShellGenie CLI (separate)
```

## Running the API locally

```bash
cd agentic

# Ollama backend (default — Ollama must be running on port 11434)
pip install -e .
GRYPHGEN_LLM_BACKEND=ollama OLLAMA_MODEL=devstral:24b \
  uvicorn src.api.main:app --reload --port 8000

# Anthropic backend
pip install -e ".[anthropic]"
GRYPHGEN_LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... \
  uvicorn src.api.main:app --reload --port 8000

# OpenAI backend
pip install -e ".[openai]"
GRYPHGEN_LLM_BACKEND=openai OPENAI_API_KEY=sk-... \
  uvicorn src.api.main:app --reload --port 8000
```

## Key env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `GRYPHGEN_LLM_BACKEND` | `ollama` | Backend: `ollama`, `anthropic`, `openai` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP API base |
| `OLLAMA_MODEL` | `devstral:24b` | Default Ollama model |
| `ANTHROPIC_API_KEY` | — | Required for anthropic backend |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model name |
| `OPENAI_API_KEY` | — | Required for openai backend |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |

## Test the code generation endpoint

```bash
# Generate code (Ollama must be running)
curl -X POST http://localhost:8000/api/v1/code/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a fibonacci function", "language": "python"}'

# List available models from current backend
curl http://localhost:8000/api/v1/code/models

# Health check
curl http://localhost:8000/health
```

## What still needs implementing

| Service | Status | Notes |
|---------|--------|-------|
| code_generation | **Working** | Calls real LLM via GRYPHGEN_LLM_BACKEND |
| automated_testing | Stub | Wire to code_gen output + pytest runner |
| project_management | Stub | Needs SQLite/Postgres task store |
| documentation | Stub | Wire to LLM summarize() |
| collaboration | Stub | WebSocket plumbing exists |
| self_improvement | Stub | RAG feedback loop |
| API key auth | Plumbed | JWT module exists; not enforced on routes |
| Stripe metering | Missing | Per-token billing on generate endpoint |

## api.grug.ai deployment plan

1. VPS (or Hetzner box) running Ubuntu 22.04
2. `GRYPHGEN_LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY` in systemd env
3. Nginx reverse proxy → uvicorn on port 8000
4. Stripe metering: intercept `tokens_used` from `CodeGenerationResponse`, emit usage event
5. API keys: hash stored in Redis, checked on each request via middleware

## Worlock dev environment

Ollama is running at `http://127.0.0.1:11434` with `devstral:24b` and `deepseek-r1:14b`
available. Use `GRYPHGEN_LLM_BACKEND=ollama` for zero-cost local development.

The dev working copy is at `/tmp/gryphgen-dev/`. The session doc is at:
`~/Documents/claude_creations/2026-05-06_gryphgen-modernization/session.md`
