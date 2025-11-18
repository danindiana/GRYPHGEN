"""Data models for novelty scoring."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class NoveltyLevel(str, Enum):
    """Novelty classification levels."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


class NoveltyMetrics(BaseModel):
    """Detailed metrics for novelty calculation."""

    model_config = ConfigDict(frozen=True)

    semantic_distance: float = Field(..., ge=0.0, le=1.0, description="Cosine distance from nearest neighbor")
    entropy: float = Field(..., ge=0.0, description="Information entropy of the input")
    rarity: float = Field(..., ge=0.0, le=1.0, description="Rarity score based on frequency")
    cluster_distance: float = Field(..., ge=0.0, le=1.0, description="Distance from cluster centroids")
    temporal_decay: float = Field(
        ..., ge=0.0, le=1.0, description="Decay factor based on similar past queries"
    )


class NoveltyScore(BaseModel):
    """Complete novelty score with metadata."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(..., ge=0.0, le=1.0, description="Overall novelty score")
    level: NoveltyLevel = Field(..., description="Classification level")
    metrics: NoveltyMetrics = Field(..., description="Detailed scoring metrics")
    embedding_id: Optional[str] = Field(None, description="Reference to stored embedding")
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    input_hash: str = Field(..., description="Hash of the input for deduplication")

    @classmethod
    def from_metrics(cls, metrics: NoveltyMetrics, input_hash: str) -> "NoveltyScore":
        """Calculate overall score from metrics."""
        # Weighted combination of metrics
        score = (
            metrics.semantic_distance * 0.35
            + metrics.entropy * 0.25
            + metrics.rarity * 0.20
            + metrics.cluster_distance * 0.15
            + metrics.temporal_decay * 0.05
        )

        # Classify novelty level
        if score >= 0.8:
            level = NoveltyLevel.VERY_HIGH
        elif score >= 0.6:
            level = NoveltyLevel.HIGH
        elif score >= 0.4:
            level = NoveltyLevel.MEDIUM
        elif score >= 0.2:
            level = NoveltyLevel.LOW
        else:
            level = NoveltyLevel.VERY_LOW

        return cls(score=score, level=level, metrics=metrics, input_hash=input_hash)


class EmbeddingRecord(BaseModel):
    """Record of an embedding in the vector database."""

    id: str
    embedding: list[float]
    text: str
    metadata: dict
    novelty_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
