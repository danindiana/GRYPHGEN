# 🤺 ollama-duel

A Linux CLI terminal application that orchestrates **two small Ollama models** in an **actor/critic loop**, with **MCP (Model Context Protocol) as a first-class citizen** for tool use. Built with [Ratatui](https://ratatui.rs/) for a fully native terminal UI.

```
┌─ 🤺 ollama-duel  🎭 qwen2.5:3b ↔ 📊 qwen2.5:3b  MCP:2 ──────────────────┐
├─ MCP ──────────────┬─ Conversation ─────────────────────────────────────────┤
│ Servers            │                                                         │
│ ✓ filesystem (4)   │   ℹ  ollama-duel ready. Press [i] to enter a prompt.  │
│ ✓ brave-search (1) │                                                         │
│                    │   ▶ You                                                 │
│ Tools              │     Find the latest Rust async book and summarise ch1   │
│ ▸ read_file        │                                                         │
│ ▸ write_file       │   🎭 Actor [qwen2.5:3b] round 1                        │
│ ▸ list_dir         │     🔧 Tool call brave-search::brave_web_search         │
│ ▸ brave_web_search │       {"query": "Rust async book 2024"}                 │
│                    │     ✓ Result (brave_web_search)                         │
│                    │       1. "Async Rust" by ...                            │
│                    │                                                         │
│                    │     Chapter 1 introduces the async/await model…         │
│                    │                                                         │
│                    │   📊 Critic [qwen2.5:3b]  Score: 8/10  ✓ accept        │
│                    │     Good summary. Could include more code examples.     │
├────────────────────┴─────────────────────────────────────────────────────── ┤
│ [i] to type, [q] quit, [↑↓/PgUp/PgDn] scroll                               │
├─────────────────────────────────────────────────────────────────────────────┤
│   Idle  actor:qwen2.5:3b critic:qwen2.5:3b                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
User Prompt
    │
    ▼
┌───────────────────────────────────────────┐
│              Actor/Critic Loop             │
│                                           │
│  ┌─────────┐   tools    ┌──────────────┐  │
│  │  Actor  │ ◄────────► │  MCP Servers │  │
│  │ (Ollama)│            │  (stdio)     │  │
│  └────┬────┘            └──────────────┘  │
│       │ response                          │
│       ▼                                   │
│  ┌─────────┐                              │
│  │  Critic │ score + feedback             │
│  │ (Ollama)│                              │
│  └────┬────┘                              │
│       │                                   │
│   score ≥ threshold?                      │
│   ├── yes → Final Response                │
│   └── no  → Revision round (max N)        │
└───────────────────────────────────────────┘
```

### Actor/Critic Pattern

| Role | Responsibility | Default model |
|------|---------------|---------------|
| **Actor** | Generates responses, calls MCP tools | `qwen2.5:3b` |
| **Critic** | Scores 0–10, gives feedback, requests revision | `qwen2.5:3b` |

The critic responds with compact JSON:
```json
{"score": 8, "feedback": "Good but missing error handling examples.", "should_revise": false}
```

If `score < critique_threshold` **and** revision rounds remain, the actor is asked to revise with the critic's feedback injected.

### MCP Integration

All MCP servers run as **child processes** (stdio transport). Tools are:
- Discovered at startup via `tools/list`
- Passed to the actor as Ollama function-calling tool definitions
- Executed synchronously when the actor issues a tool call
- Results injected back into the conversation context

---

## Requirements

| Dependency | Notes |
|-----------|-------|
| Rust ≥ 1.75 | `rustup update stable` |
| Ollama ≥ 0.2.0 | `ollama version is 0.22.1` ✓ |
| A tool-calling small model | `ollama pull qwen2.5:3b` |
| Node.js ≥ 18 (optional) | For `@modelcontextprotocol/server-*` MCP servers |
| Python/uv (optional) | For Python-based MCP servers |

**Recommended models for your system** (RTX 5080 16 GB + RTX 3080 10 GB, contested):

| Model | VRAM | Quality | Tool-calling |
|-------|------|---------|-------------|
| `qwen2.5:3b` | ~2.0 GB | ⭐⭐⭐ | ✅ excellent |
| `qwen2.5:7b` | ~4.5 GB | ⭐⭐⭐⭐ | ✅ excellent |
| `llama3.2:3b` | ~2.0 GB | ⭐⭐⭐ | ✅ good |
| `llama3.1:8b` | ~5.0 GB | ⭐⭐⭐⭐ | ✅ good |
| `phi3.5:mini` | ~2.2 GB | ⭐⭐⭐ | ⚠️ unreliable |
| `gemma2:2b`   | ~1.6 GB | ⭐⭐ | ❌ no |

Pull models before use:
```bash
ollama pull qwen2.5:3b
```

---

## Installation

```bash
git clone <repo> ollama-duel
cd ollama-duel

# Debug build (fast compile)
cargo build

# Optimised release build
cargo build --release

# Install to PATH
cargo install --path .
```

---

## Usage

```bash
# Use defaults from config.toml
ollama-duel

# Override models on the fly
ollama-duel --actor qwen2.5:7b --critic gemma2:2b

# Specify a config file
ollama-duel --config /path/to/config.toml

# Custom Ollama endpoint
ollama-duel --ollama http://192.168.1.10:11434
```

### Keybindings

| Key | Action |
|-----|--------|
| `i` / `Enter` | Start typing a prompt |
| `Enter` (in input) | Submit prompt |
| `Esc` (in input) | Cancel editing |
| `↑` / `k` | Scroll conversation up |
| `↓` / `j` | Scroll conversation down |
| `PgUp` / `PgDn` | Scroll faster |
| `g` / `Home` | Scroll to top |
| `G` / `End` | Scroll to bottom |
| `q` / `Ctrl-C` | Quit |

---

## Configuration

Copy and edit `config.toml`:
```bash
mkdir -p ~/.config/ollama-duel
cp config.toml ~/.config/ollama-duel/config.toml
$EDITOR ~/.config/ollama-duel/config.toml
```

### MCP Server Setup

**Filesystem** (read local files):
```bash
npm install -g @modelcontextprotocol/server-filesystem
```
Then in `config.toml`:
```toml
[[mcp_servers]]
name    = "filesystem"
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "/home/you/projects"]
enabled = true
```

**Brave Search** (web search):
```bash
export BRAVE_API_KEY="your_key_here"   # add to ~/.zshrc
```
```toml
[[mcp_servers]]
name    = "brave-search"
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-brave-search"]
env     = { BRAVE_API_KEY = "${BRAVE_API_KEY}" }
enabled = true
```

**Fetch** (web scraping):
```bash
pip install mcp-server-fetch --break-system-packages
# or: uv tool install mcp-server-fetch
```
```toml
[[mcp_servers]]
name    = "fetch"
command = "uvx"
args    = ["mcp-server-fetch"]
enabled = true
```

---

## GPU Notes

Ollama manages GPU assignment automatically. With your RTX 5080 (16 GB) and RTX 3080 (10 GB):

- Two `qwen2.5:3b` models (~2 GB each) will comfortably fit on either GPU
- For the actor, prefer the 5080 (faster inference): set `CUDA_VISIBLE_DEVICES=0` before launching
- To pin models to specific GPUs, use separate Ollama instances on different ports:

```bash
# Terminal 1 — actor on RTX 5080
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve

# Terminal 2 — critic on RTX 3080
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve

# Then in config.toml, use separate URLs... 
# (future feature: per-model ollama URL)
```

---

## Logs

Logs are written to `~/.local/share/ollama-duel/ollama-duel.YYYY-MM-DD.log`

```bash
tail -f ~/.local/share/ollama-duel/ollama-duel.*.log
```

Enable verbose logging:
```bash
RUST_LOG=debug ollama-duel
```

---

## Project Structure

```
ollama-duel/
├── src/
│   ├── main.rs          # Entry point, terminal setup, CLI
│   ├── app.rs           # Application state + event loop
│   ├── ui.rs            # Ratatui rendering
│   ├── ollama.rs        # Ollama HTTP client (streaming + tool calls)
│   ├── actor_critic.rs  # Actor/critic orchestration + BgEvent bus
│   ├── config.rs        # Config structs + loaders
│   └── mcp/
│       ├── mod.rs       # MCP types (JSON-RPC, tool definitions)
│       └── client.rs    # MCP stdio client
├── config.toml          # Example / default config
└── Cargo.toml
```
