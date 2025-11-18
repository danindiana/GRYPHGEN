"""Tests for novelty scoring engine."""

import numpy as np
import pytest

from src.novelty.scorer import NoveltyScorer
from src.novelty.engine import NoveltyEngine
from src.novelty.models import NoveltyMetrics, NoveltyScore, NoveltyLevel


class TestNoveltyScorer:
    """Test suite for NoveltyScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a novelty scorer instance."""
        return NoveltyScorer()

    def test_compute_semantic_distance_empty_neighbors(self, scorer):
        """Test semantic distance with no neighbors."""
        embedding = np.random.rand(384)
        distance = scorer.compute_semantic_distance(embedding, [])
        assert distance == 1.0  # Maximum novelty

    def test_compute_semantic_distance_similar(self, scorer):
        """Test semantic distance with similar embeddings."""
        embedding = np.random.rand(384)
        neighbors = [embedding + np.random.rand(384) * 0.1 for _ in range(3)]
        distance = scorer.compute_semantic_distance(embedding, neighbors)
        assert 0.0 <= distance <= 1.0

    def test_compute_entropy(self, scorer):
        """Test entropy computation."""
        embedding = np.random.rand(384)
        entropy = scorer.compute_entropy(embedding)
        assert 0.0 <= entropy <= 1.0

    def test_compute_rarity_empty(self, scorer):
        """Test rarity with no frequency data."""
        embedding = np.random.rand(384)
        rarity = scorer.compute_rarity(embedding, {}, 0)
        assert rarity == 1.0

    def test_compute_rarity_common(self, scorer):
        """Test rarity for common embedding."""
        embedding = np.random.rand(384)
        # Simulate high frequency
        freq_map = {"test": 100}
        rarity = scorer.compute_rarity(embedding, freq_map, 100)
        assert 0.0 <= rarity <= 1.0

    def test_compute_cluster_distance(self, scorer):
        """Test cluster distance computation."""
        embedding = np.random.rand(384)
        centroids = [np.random.rand(384) for _ in range(5)]
        distance = scorer.compute_cluster_distance(embedding, centroids)
        assert 0.0 <= distance <= 1.0

    def test_compute_temporal_decay_empty(self, scorer):
        """Test temporal decay with no history."""
        decay = scorer.compute_temporal_decay([], 1000.0)
        assert decay == 1.0

    def test_compute_temporal_decay_recent(self, scorer):
        """Test temporal decay with recent activity."""
        current_time = 1000.0
        recent_timestamps = [990.0, 995.0, 998.0]
        decay = scorer.compute_temporal_decay(recent_timestamps, current_time)
        assert 0.0 <= decay <= 1.0

    def test_compute_score_complete(self, scorer):
        """Test complete score computation."""
        embedding = np.random.rand(384)
        text = "Test query about quantum computing"

        score = scorer.compute_score(
            embedding=embedding,
            text=text,
            nearest_embeddings=[np.random.rand(384) for _ in range(5)],
            cluster_centroids=[np.random.rand(384) for _ in range(3)],
            frequency_map={"test": 10},
            total_count=100,
            similar_timestamps=[900.0, 950.0],
            current_time=1000.0,
        )

        assert isinstance(score, NoveltyScore)
        assert 0.0 <= score.score <= 1.0
        assert isinstance(score.level, NoveltyLevel)
        assert isinstance(score.metrics, NoveltyMetrics)

    def test_novelty_level_classification(self, scorer):
        """Test novelty level classification."""
        embedding = np.random.rand(384)

        # Very high novelty (all metrics high)
        score = scorer.compute_score(
            embedding=embedding,
            text="Completely novel query",
            nearest_embeddings=[],  # No similar embeddings
            cluster_centroids=[],
            frequency_map={},
            total_count=0,
            similar_timestamps=[],
            current_time=1000.0,
        )

        # Score should be high due to all max metrics
        assert score.level in [NoveltyLevel.HIGH, NoveltyLevel.VERY_HIGH]


class TestNoveltyEngine:
    """Test suite for NoveltyEngine."""

    @pytest.fixture
    def engine(self):
        """Create a novelty engine instance."""
        return NoveltyEngine(model_name="all-MiniLM-L6-v2", device="cpu")

    def test_generate_embedding(self, engine):
        """Test embedding generation."""
        text = "Test query for embedding"
        embedding = engine.generate_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        assert embedding.dtype == np.float32

    def test_generate_embedding_different_texts(self, engine):
        """Test that different texts produce different embeddings."""
        text1 = "Quantum computing is fascinating"
        text2 = "Machine learning is powerful"

        emb1 = engine.generate_embedding(text1)
        emb2 = engine.generate_embedding(text2)

        # Embeddings should be different
        assert not np.array_equal(emb1, emb2)

    @pytest.mark.asyncio
    async def test_compute_novelty_without_db(self, engine):
        """Test novelty computation without vector database."""
        text = "Test query for novelty scoring"
        score = await engine.compute_novelty(text)

        assert isinstance(score, NoveltyScore)
        assert 0.0 <= score.score <= 1.0

    def test_set_vector_db(self, engine):
        """Test setting vector database."""
        mock_db = object()
        engine.set_vector_db(mock_db)
        assert engine.vector_db is mock_db

    def test_set_metadata_store(self, engine):
        """Test setting metadata store."""
        mock_store = object()
        engine.set_metadata_store(mock_store)
        assert engine.metadata_store is mock_store


class TestNoveltyModels:
    """Test suite for novelty data models."""

    def test_novelty_metrics_validation(self):
        """Test NoveltyMetrics validation."""
        metrics = NoveltyMetrics(
            semantic_distance=0.8,
            entropy=0.6,
            rarity=0.7,
            cluster_distance=0.5,
            temporal_decay=0.9,
        )

        assert metrics.semantic_distance == 0.8
        assert metrics.entropy == 0.6
        assert metrics.rarity == 0.7

    def test_novelty_score_from_metrics(self):
        """Test NoveltyScore creation from metrics."""
        metrics = NoveltyMetrics(
            semantic_distance=0.8,
            entropy=0.6,
            rarity=0.7,
            cluster_distance=0.5,
            temporal_decay=0.9,
        )

        score = NoveltyScore.from_metrics(metrics, "test_hash")

        assert isinstance(score, NoveltyScore)
        assert 0.0 <= score.score <= 1.0
        assert isinstance(score.level, NoveltyLevel)
        assert score.metrics == metrics

    def test_novelty_level_thresholds(self):
        """Test novelty level classification thresholds."""
        # Very low
        metrics1 = NoveltyMetrics(
            semantic_distance=0.1,
            entropy=0.1,
            rarity=0.1,
            cluster_distance=0.1,
            temporal_decay=0.1,
        )
        score1 = NoveltyScore.from_metrics(metrics1, "hash1")
        assert score1.level == NoveltyLevel.VERY_LOW

        # High
        metrics2 = NoveltyMetrics(
            semantic_distance=0.7,
            entropy=0.7,
            rarity=0.7,
            cluster_distance=0.7,
            temporal_decay=0.7,
        )
        score2 = NoveltyScore.from_metrics(metrics2, "hash2")
        assert score2.level in [NoveltyLevel.HIGH, NoveltyLevel.VERY_HIGH]
