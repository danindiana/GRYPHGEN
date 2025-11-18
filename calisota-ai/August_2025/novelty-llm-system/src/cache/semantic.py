"""Semantic cache using vector similarity search."""

import hashlib
import time
from typing import Any, Optional, Union

import numpy as np

from .models import CacheHit, CacheMiss


class SemanticCache:
    """
    Semantic cache using vector embeddings for similarity-based retrieval.
    Uses FAISS or Milvus for efficient similarity search.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_results: int = 5,
        ttl_seconds: int = 3600,
    ):
        """
        Initialize semantic cache.

        Args:
            similarity_threshold: Minimum similarity for cache hit (0-1)
            max_results: Maximum number of similar entries to return
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.ttl_seconds = ttl_seconds

        # Placeholders for vector index and metadata store
        self.vector_index = None
        self.metadata_store = None

    def set_vector_index(self, vector_index) -> None:
        """Set the vector index instance (FAISS/Milvus)."""
        self.vector_index = vector_index

    def set_metadata_store(self, metadata_store) -> None:
        """Set the metadata store instance (Redis/PostgreSQL)."""
        self.metadata_store = metadata_store

    def _compute_key(self, text: str, tenant_id: Optional[str] = None) -> str:
        """Compute cache key from text and tenant."""
        content = f"{tenant_id or 'global'}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        embedding: np.ndarray,
        text: str,
        tenant_id: Optional[str] = None,
    ) -> Union[CacheHit, CacheMiss]:
        """
        Get cached response based on semantic similarity.

        Args:
            embedding: Query embedding vector
            text: Original query text
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            CacheHit if similar entry found, CacheMiss otherwise
        """
        if not self.vector_index or not self.metadata_store:
            return CacheMiss(key=self._compute_key(text, tenant_id), reason="cache_not_configured")

        # Search for similar embeddings
        search_results = await self.vector_index.search(
            embedding=embedding,
            k=self.max_results,
            tenant_id=tenant_id,
        )

        # Filter by similarity threshold and TTL
        current_time = time.time()
        for result in search_results:
            similarity = result.get("similarity", 0.0)

            if similarity < self.similarity_threshold:
                continue

            # Check TTL
            entry_id = result["id"]
            metadata = await self.metadata_store.get(entry_id)

            if not metadata:
                continue

            created_at = metadata.get("created_at", 0)
            age_seconds = current_time - created_at

            if age_seconds > self.ttl_seconds:
                continue

            # Cache hit!
            hit_count = metadata.get("hit_count", 0)

            # Increment hit count
            await self.metadata_store.increment_hit_count(entry_id)

            return CacheHit(
                key=entry_id,
                value=metadata.get("response"),
                similarity=similarity,
                hit_count=hit_count + 1,
                age_seconds=age_seconds,
            )

        # No cache hit
        return CacheMiss(key=self._compute_key(text, tenant_id), reason="no_similar_entry")

    async def set(
        self,
        embedding: np.ndarray,
        text: str,
        response: Any,
        tenant_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store response in semantic cache.

        Args:
            embedding: Query embedding vector
            text: Original query text
            response: Response to cache
            tenant_id: Tenant ID
            metadata: Additional metadata

        Returns:
            Cache entry ID
        """
        if not self.vector_index or not self.metadata_store:
            raise RuntimeError("Cache not configured")

        key = self._compute_key(text, tenant_id)

        # Prepare metadata
        cache_metadata = {
            "text": text,
            "response": response,
            "tenant_id": tenant_id,
            "created_at": time.time(),
            "hit_count": 0,
            **(metadata or {}),
        }

        # Store embedding in vector index
        entry_id = await self.vector_index.insert(
            embedding=embedding,
            metadata={"key": key, "tenant_id": tenant_id},
            tenant_id=tenant_id,
        )

        # Store full metadata in metadata store
        await self.metadata_store.set(entry_id, cache_metadata, ttl=self.ttl_seconds)

        return entry_id

    async def delete(self, entry_id: str) -> bool:
        """Delete cache entry."""
        if not self.vector_index or not self.metadata_store:
            return False

        # Delete from both stores
        await self.vector_index.delete(entry_id)
        await self.metadata_store.delete(entry_id)

        return True

    async def clear(self, tenant_id: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            tenant_id: If provided, clear only entries for this tenant

        Returns:
            Number of entries cleared
        """
        if not self.vector_index or not self.metadata_store:
            return 0

        # Clear vector index
        count = await self.vector_index.clear(tenant_id=tenant_id)

        # Clear metadata store
        await self.metadata_store.clear(tenant_id=tenant_id)

        return count

    async def get_stats(self, tenant_id: Optional[str] = None) -> dict:
        """Get cache statistics."""
        if not self.metadata_store:
            return {}

        stats = await self.metadata_store.get_stats(tenant_id=tenant_id)
        return stats
