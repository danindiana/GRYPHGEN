# api.grug.ai — API Key Authentication

**Timestamp:** 2026-05-06T21:14:34-0500  
**System:** worlock (Ubuntu/Debian, Linux 6.8.12, jeb)  
**Service started with auth:** 2026-05-06T21:13:xx-0500  
**Git commit:** `73ae853`  
**Parent session:** `../session.md` (2026-05-06_gryphgen-modernization)

---

## What was built

Standalone API key authentication for `POST /api/v1/code/generate` on `api.grug.ai`.  
No database, no JWT, no external service — keys are SHA-256 hashed and stored in `.env`.

---

## Files created / modified

| File | Action | Purpose |
|------|--------|---------|
| `agentic/src/auth/api_keys.py` | **Created** | FastAPI `Security` dependency; validates key against hashes in env |
| `agentic/scripts/keygen.py` | **Created** | CLI to generate new keys and print hash for `.env` |
| `agentic/src/auth/__init__.py` | **Modified** | Gutted eager imports (jwt_handler → database chain); now only exports `require_api_key` |
| `agentic/src/services/code_generation/router.py` | **Modified** | Added `Security(require_api_key)` to `generate_code` signature |
| `agentic/src/common/config.py` | **Modified** | Added `gryphgen_api_keys: str` field so pydantic-settings accepts the env var |
| `/home/jeb/programs/gryphgen-agentic/.env` | **Modified** | Added `GRYPHGEN_API_KEYS=<hash>` |

---

## Design

### Key format
```
grug_<secrets.token_urlsafe(32)>
```
Example: `grug_VUkdgrSHxX5JBudBd7TdQkfHQW7OlNNkAvDuQ33qrF8`

- Prefix `grug_` makes keys recognizable in logs and config
- `token_urlsafe(32)` = 43 URL-safe base64 chars → 256 bits entropy
- Plaintext key shown once at generation; never stored anywhere

### Storage
`GRYPHGEN_API_KEYS` in `/home/jeb/programs/gryphgen-agentic/.env`:
```
GRYPHGEN_API_KEYS=<sha256hex>[,<sha256hex>,...]
```
Multiple keys comma-separated. Each is a `hashlib.sha256(key.encode()).hexdigest()`.

### Auth flow
```
client → X-API-Key: grug_xxx  (or Authorization: Bearer grug_xxx)
       → api_keys.py reads GRYPHGEN_API_KEYS from env
       → SHA-256 hashes the presented key
       → compares against frozenset of stored hashes
       → 401 (no key) / 403 (bad key) / pass-through (match)
```

### Protected vs open endpoints

| Endpoint | Auth required |
|----------|--------------|
| `POST /api/v1/code/generate` | **Yes** |
| `GET /api/v1/code/models` | No |
| `GET /api/v1/code/languages` | No |
| `GET /health` | No |
| `GET /` | No |
| `GET /docs` | No |
| `GET /metrics` | No |

---

## Source: `agentic/src/auth/api_keys.py`

```python
import hashlib
import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


def _load_hashes() -> frozenset[str]:
    raw = os.environ.get("GRYPHGEN_API_KEYS", "")
    return frozenset(h.strip() for h in raw.split(",") if h.strip())


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _check(key: Optional[str]) -> str:
    if not key:
        raise HTTPException(status_code=401, detail="API key required",
                            headers={"WWW-Authenticate": "Bearer"})
    hashes = _load_hashes()
    if not hashes:
        raise HTTPException(status_code=503, detail="No API keys configured on server")
    if _hash(key) not in hashes:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


async def require_api_key(
    x_api_key: Optional[str] = Security(_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    key = x_api_key or (bearer.credentials if bearer else None)
    return _check(key)
```

---

## Source: `agentic/scripts/keygen.py`

```python
#!/usr/bin/env python3
import argparse, hashlib, os, secrets

def generate_key():
    key = "grug_" + secrets.token_urlsafe(32)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return key, digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    key, digest = generate_key()

    # Read existing hashes from .env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    existing = ""
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith("GRYPHGEN_API_KEYS="):
                    existing = line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass

    new_value = f"{existing},{digest}" if existing else digest
    label_note = f"  ({args.label})" if args.label else ""

    print(f"\n{'='*60}")
    print(f"  New API key{label_note}")
    print(f"{'='*60}")
    print(f"\n  KEY (give to client, shown once):\n\n    {key}\n")
    print(f"  HASH (add to .env):\n\n    {digest}\n")
    print(f"  Updated GRYPHGEN_API_KEYS:\n\n    GRYPHGEN_API_KEYS={new_value}\n")
    print(f"  Restart:  sudo systemctl restart gryphgen-agentic")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
```

---

## Bugs fixed en route

### 1. `pydantic_settings` rejecting `GRYPHGEN_API_KEYS`
`Settings` model had `extra = "forbidden"` (Pydantic default). Adding the field:
```python
gryphgen_api_keys: str = Field(default="")
```
to `config.py` resolved the `ValidationError: Extra inputs are not permitted`.

### 2. `auth/__init__.py` eager import chain
Importing `from ...auth.api_keys import require_api_key` in the router triggered
`auth/__init__.py`, which ran:
```python
from .jwt_handler import ...   # → database/models → SQLAlchemy metadata crash
```
Fix: gut `auth/__init__.py` to only export `require_api_key`. The JWT/DB chain
is left importable directly when needed but is not loaded at startup.

---

## Verified live at 2026-05-06T21:14:34-0500

```bash
# 401 — no key
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://api.grug.ai/api/v1/code/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","language":"python"}'
# → 401

# 403 — wrong key
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://api.grug.ai/api/v1/code/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: grug_notarealkey" \
  -d '{"prompt":"hello","language":"python"}'
# → 403

# 200 — valid key via X-API-Key header
curl -s -X POST https://api.grug.ai/api/v1/code/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: grug_VUkdgrSHxX5JBudBd7TdQkfHQW7OlNNkAvDuQ33qrF8" \
  -d '{"prompt":"add two numbers","language":"python","max_tokens":150}'
# → 200, real LLM output

# 200 — valid key via Authorization: Bearer
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://api.grug.ai/api/v1/code/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer grug_VUkdgrSHxX5JBudBd7TdQkfHQW7OlNNkAvDuQ33qrF8" \
  -d '{"prompt":"hello","language":"python","max_tokens":50}'
# → 200
```

---

## Key management runbook

### Issue a new key
```bash
cd /home/jeb/programs/gryphgen-agentic
python3 scripts/keygen.py --label "client-name"
# Copy the KEY line → send to client
# Copy the updated GRYPHGEN_API_KEYS line → paste into .env
sudo systemctl restart gryphgen-agentic
```

### Revoke a key
```bash
# Edit .env, remove the offending hash from GRYPHGEN_API_KEYS
nano /home/jeb/programs/gryphgen-agentic/.env
sudo systemctl restart gryphgen-agentic
```

### Rotate a key
Revoke old hash, issue new key, give client the new plaintext key.

### Check active key count
```bash
grep GRYPHGEN_API_KEYS /home/jeb/programs/gryphgen-agentic/.env | \
  tr ',' '\n' | grep -c .
```

---

## Git commit

```
73ae853  feat: add API key auth to /api/v1/code/generate  2026-05-06
```

Files in commit:
- `agentic/scripts/keygen.py` (new)
- `agentic/src/auth/api_keys.py` (new)
- `agentic/src/auth/__init__.py` (rewritten)
- `agentic/src/common/config.py` (added gryphgen_api_keys field)
- `agentic/src/services/code_generation/router.py` (added Security dep)

---

## Next steps

| Priority | Item |
|----------|------|
| High | Rate limiting — no per-key or per-IP throttle yet; Cloudflare rate rules are a stopgap |
| High | Log key identity per request — currently requests are anonymous in logs |
| Medium | Stripe metering — `tokens_used` in every response; hook up usage event on generate |
| Medium | Key expiry — current keys are permanent; add optional expiry timestamp to hash store |
| Low | Admin endpoint — `GET /admin/keys` (list hashes + labels); requires separate admin key |
