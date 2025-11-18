"""Data models for caching system."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class CacheEntry(BaseModel):
    """Cache entry model."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(..., description="Cache key")
    value: Any = Field(..., description="Cached value")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    hit_count: int = Field(default=0, description="Number of cache hits")
    metadata: dict = Field(default_factory=dict)


class CacheHit(BaseModel):
    """Cache hit result."""

    key: str
    value: Any
    similarity: Optional[float] = None
    hit_count: int
    age_seconds: float


class CacheMiss(BaseModel):
    """Cache miss result."""

    key: str
    reason: str = "not_found"
