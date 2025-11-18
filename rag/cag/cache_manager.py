"""
Cache manager for CAG - simplified version.

This module provides a wrapper around KVCacheStore for easier use.
Most functionality is in kv_cache.py
"""

from .kv_cache import KVCacheStore, CacheManager as BaseCacheManager

# Re-export for convenience
__all__ = ["KVCacheStore", "BaseCacheManager", "CacheManager"]

# Alias for backward compatibility
CacheManager = BaseCacheManager
