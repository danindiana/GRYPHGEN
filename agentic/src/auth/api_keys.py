"""
API key authentication for GRYPHGEN.

Keys are stored as SHA-256 hashes in the GRYPHGEN_API_KEYS env var
(comma-separated). No database required.

Key format:  grug_<secrets.token_urlsafe(32)>
Store format: comma-separated SHA-256 hex digests in GRYPHGEN_API_KEYS

Usage:
    @router.post("/generate")
    async def generate_code(
        request: CodeGenerationRequest,
        _: str = Security(require_api_key),
    ): ...
"""

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
    """Validate key against stored hashes. Returns the key on success."""
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    hashes = _load_hashes()
    if not hashes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No API keys configured on server",
        )
    if _hash(key) not in hashes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return key


async def require_api_key(
    x_api_key: Optional[str] = Security(_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> str:
    """
    FastAPI Security dependency.  Accepts the key via either:
      X-API-Key: grug_xxx
      Authorization: Bearer grug_xxx
    """
    key = x_api_key or (bearer.credentials if bearer else None)
    return _check(key)
