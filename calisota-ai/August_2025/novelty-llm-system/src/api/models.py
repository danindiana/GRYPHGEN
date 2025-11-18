"""API request/response models."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """LLM query request."""

    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    model: Optional[str] = Field(None, description="Model name (optional)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2048, ge=1, le=32768, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Enable streaming response")
    documents: list[str] = Field(default_factory=list, description="Optional context documents")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class QueryResponse(BaseModel):
    """LLM query response."""

    response: str = Field(..., description="Generated response")
    model: str = Field(..., description="Model used")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty score")
    novelty_level: str = Field(..., description="Novelty classification")
    cached: bool = Field(..., description="Whether response was cached")
    cache_hit_similarity: Optional[float] = Field(None, description="Cache hit similarity")
    tokens_used: int = Field(..., description="Number of tokens used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(default_factory=dict, description="Component health status")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
