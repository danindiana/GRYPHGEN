"""Tests for caching system."""

import pytest
import numpy as np

from src.cache.semantic import SemanticCache
from src.cache.response import ResponseCache
from src.cache.models import CacheHit, CacheMiss


class TestSemanticCache:
    """Test suite for SemanticCache."""

    @pytest.fixture
    def cache(self):
        """Create a semantic cache instance."""
        return SemanticCache(similarity_threshold=0.85, ttl_seconds=3600)

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.similarity_threshold == 0.85
        assert cache.ttl_seconds == 3600
        assert cache.vector_index is None
        assert cache.metadata_store is None

    def test_set_vector_index(self, cache):
        """Test setting vector index."""
        mock_index = object()
        cache.set_vector_index(mock_index)
        assert cache.vector_index is mock_index

    def test_set_metadata_store(self, cache):
        """Test setting metadata store."""
        mock_store = object()
        cache.set_metadata_store(mock_store)
        assert cache.metadata_store is mock_store

    @pytest.mark.asyncio
    async def test_get_without_backend(self, cache):
        """Test get without configured backend."""
        embedding = np.random.rand(384)
        result = await cache.get(embedding, "test query")

        assert isinstance(result, CacheMiss)
        assert result.reason == "cache_not_configured"

    def test_compute_key(self, cache):
        """Test cache key computation."""
        key1 = cache._compute_key("test query", "tenant1")
        key2 = cache._compute_key("test query", "tenant1")
        key3 = cache._compute_key("test query", "tenant2")

        # Same input should produce same key
        assert key1 == key2
        # Different tenant should produce different key
        assert key1 != key3


class TestResponseCache:
    """Test suite for ResponseCache."""

    @pytest.fixture
    def cache(self):
        """Create a response cache instance."""
        return ResponseCache(ttl_seconds=3600, max_size_mb=1000)

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.ttl_seconds == 3600
        assert cache.max_size_mb == 1000
        assert cache.redis_client is None

    def test_set_redis_client(self, cache):
        """Test setting Redis client."""
        mock_client = object()
        cache.set_redis_client(mock_client)
        assert cache.redis_client is mock_client

    @pytest.mark.asyncio
    async def test_get_without_backend(self, cache):
        """Test get without configured backend."""
        result = await cache.get("test query")

        assert isinstance(result, CacheMiss)
        assert result.reason == "cache_not_configured"

    def test_compute_key_consistency(self, cache):
        """Test that identical inputs produce identical keys."""
        key1 = cache._compute_key("test", "model1", "tenant1")
        key2 = cache._compute_key("test", "model1", "tenant1")

        assert key1 == key2

    def test_compute_key_different_params(self, cache):
        """Test that different parameters produce different keys."""
        key1 = cache._compute_key("test", "model1", "tenant1")
        key2 = cache._compute_key("test", "model2", "tenant1")
        key3 = cache._compute_key("test", "model1", "tenant2")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_compute_key_with_kwargs(self, cache):
        """Test key computation with additional kwargs."""
        key1 = cache._compute_key("test", temperature=0.7, top_p=0.9)
        key2 = cache._compute_key("test", temperature=0.7, top_p=0.9)
        key3 = cache._compute_key("test", temperature=0.8, top_p=0.9)

        assert key1 == key2  # Same params
        assert key1 != key3  # Different temperature


class TestCacheModels:
    """Test suite for cache data models."""

    def test_cache_hit_model(self):
        """Test CacheHit model."""
        hit = CacheHit(
            key="test_key",
            value={"response": "test"},
            similarity=0.92,
            hit_count=5,
            age_seconds=120.5,
        )

        assert hit.key == "test_key"
        assert hit.similarity == 0.92
        assert hit.hit_count == 5
        assert hit.age_seconds == 120.5

    def test_cache_miss_model(self):
        """Test CacheMiss model."""
        miss = CacheMiss(key="test_key", reason="not_found")

        assert miss.key == "test_key"
        assert miss.reason == "not_found"

    def test_cache_miss_default_reason(self):
        """Test CacheMiss with default reason."""
        miss = CacheMiss(key="test_key")

        assert miss.reason == "not_found"
