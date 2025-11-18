"""
CAG: Cache-Augmented Generation.

A retrieval-free paradigm for knowledge-intensive tasks that leverages
preloaded contexts and KV caching for faster, more accurate inference.

Reference: https://github.com/hhhuang/CAG
"""

from .cache_manager import CacheManager
from .kv_cache import KVCache, KVCacheStore
from .db_interface import DatabaseInterface
from .cag_engine import CAGEngine

__version__ = "1.0.0"

__all__ = [
    "CacheManager",
    "KVCache",
    "KVCacheStore",
    "DatabaseInterface",
    "CAGEngine",
]
