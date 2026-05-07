"""
Sliding-window rate limiter (in-memory, per API key hash).

Limits are enforced per hashed key so the plaintext key is never stored.
Two independent limits:
  - /generate: 10 req / 60 s
  - /agent/run: 5 req / 300 s  (agent tasks are long-running)
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from threading import Lock
from typing import Deque

from fastapi import HTTPException, status


class _SlidingWindow:
    def __init__(self, max_calls: int, window_s: float) -> None:
        self.max_calls = max_calls
        self.window_s = window_s
        self._buckets: dict[str, Deque[float]] = {}
        self._lock = Lock()

    def _key(self, raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def check(self, raw_key: str) -> None:
        """Raise HTTP 429 if the key has exceeded the rate limit."""
        k = self._key(raw_key)
        now = time.monotonic()
        cutoff = now - self.window_s

        with self._lock:
            if k not in self._buckets:
                self._buckets[k] = deque()
            dq = self._buckets[k]

            # Evict expired timestamps
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= self.max_calls:
                oldest = dq[0]
                retry_after = int(self.window_s - (now - oldest)) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry after {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )

            dq.append(now)


# Module-level limiters — one instance per endpoint class
generate_limiter = _SlidingWindow(max_calls=10, window_s=60.0)
agent_limiter = _SlidingWindow(max_calls=5, window_s=300.0)
