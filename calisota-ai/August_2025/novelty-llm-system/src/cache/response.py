"""Traditional key-value response cache using Redis."""

import hashlib
import json
import time
from typing import Any, Optional, Union

from .models import CacheHit, CacheMiss


class ResponseCache:
    """
    Traditional key-value cache for exact-match responses.
    Uses Redis for high-performance caching.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size_mb: int = 1000,
    ):
        """
        Initialize response cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size_mb: Maximum cache size in MB
        """
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb

        # Placeholder for Redis client
        self.redis_client = None

    def set_redis_client(self, redis_client) -> None:
        """Set the Redis client instance."""
        self.redis_client = redis_client

    def _compute_key(
        self,
        text: str,
        model: Optional[str] = None,
        tenant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Compute cache key from input parameters.

        Args:
            text: Query text
            model: Model name
            tenant_id: Tenant ID
            **kwargs: Additional parameters

        Returns:
            Cache key hash
        """
        # Include all parameters in key
        key_data = {
            "text": text,
            "model": model,
            "tenant_id": tenant_id,
            **kwargs,
        }

        # Serialize to JSON and hash
        key_str = json.dumps(key_data, sort_keys=True)
        return f"response:{hashlib.sha256(key_str.encode()).hexdigest()}"

    async def get(
        self,
        text: str,
        model: Optional[str] = None,
        tenant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[CacheHit, CacheMiss]:
        """
        Get cached response.

        Args:
            text: Query text
            model: Model name
            tenant_id: Tenant ID
            **kwargs: Additional parameters

        Returns:
            CacheHit if found, CacheMiss otherwise
        """
        if not self.redis_client:
            return CacheMiss(key="", reason="cache_not_configured")

        key = self._compute_key(text, model, tenant_id, **kwargs)

        try:
            # Get from Redis
            cached_data = await self.redis_client.get(key)

            if not cached_data:
                return CacheMiss(key=key, reason="not_found")

            # Parse cached data
            data = json.loads(cached_data)

            # Get age
            created_at = data.get("created_at", 0)
            age_seconds = time.time() - created_at

            # Increment hit count
            hit_count = data.get("hit_count", 0)
            data["hit_count"] = hit_count + 1
            await self.redis_client.set(key, json.dumps(data), ex=self.ttl_seconds)

            return CacheHit(
                key=key,
                value=data.get("response"),
                similarity=1.0,  # Exact match
                hit_count=hit_count + 1,
                age_seconds=age_seconds,
            )

        except Exception as e:
            return CacheMiss(key=key, reason=f"error: {str(e)}")

    async def set(
        self,
        text: str,
        response: Any,
        model: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Store response in cache.

        Args:
            text: Query text
            response: Response to cache
            model: Model name
            tenant_id: Tenant ID
            ttl: Custom TTL (uses default if None)
            **kwargs: Additional parameters

        Returns:
            Cache key
        """
        if not self.redis_client:
            raise RuntimeError("Cache not configured")

        key = self._compute_key(text, model, tenant_id, **kwargs)

        # Prepare cache data
        cache_data = {
            "text": text,
            "response": response,
            "model": model,
            "tenant_id": tenant_id,
            "created_at": time.time(),
            "hit_count": 0,
            **kwargs,
        }

        # Store in Redis
        ttl_value = ttl or self.ttl_seconds
        await self.redis_client.set(key, json.dumps(cache_data), ex=ttl_value)

        return key

    async def delete(self, key: str) -> bool:
        """Delete cache entry."""
        if not self.redis_client:
            return False

        result = await self.redis_client.delete(key)
        return result > 0

    async def clear(self, pattern: str = "response:*") -> int:
        """
        Clear cache entries matching pattern.

        Args:
            pattern: Redis key pattern

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        # Scan and delete matching keys
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)

            if keys:
                deleted += await self.redis_client.delete(*keys)

            if cursor == 0:
                break

        return deleted

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.redis_client:
            return {}

        # Get Redis info
        info = await self.redis_client.info("memory")

        return {
            "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
            "max_memory_mb": self.max_size_mb,
            "keys": await self.redis_client.dbsize(),
        }
