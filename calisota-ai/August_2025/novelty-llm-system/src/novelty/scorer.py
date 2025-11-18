"""Core novelty scoring logic."""

import hashlib
import math
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from .models import NoveltyMetrics, NoveltyScore


class NoveltyScorer:
    """Calculates novelty scores for embeddings."""

    def __init__(
        self,
        semantic_weight: float = 0.35,
        entropy_weight: float = 0.25,
        rarity_weight: float = 0.20,
        cluster_weight: float = 0.15,
        temporal_weight: float = 0.05,
    ):
        """
        Initialize the novelty scorer.

        Args:
            semantic_weight: Weight for semantic distance
            entropy_weight: Weight for information entropy
            rarity_weight: Weight for rarity score
            cluster_weight: Weight for cluster distance
            temporal_weight: Weight for temporal decay
        """
        self.weights = {
            "semantic": semantic_weight,
            "entropy": entropy_weight,
            "rarity": rarity_weight,
            "cluster": cluster_weight,
            "temporal": temporal_weight,
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_semantic_distance(
        self, embedding: np.ndarray, nearest_embeddings: list[np.ndarray]
    ) -> float:
        """
        Compute semantic distance to nearest neighbors.

        Args:
            embedding: Query embedding
            nearest_embeddings: List of nearest neighbor embeddings

        Returns:
            Normalized semantic distance (0-1)
        """
        if not nearest_embeddings:
            return 1.0  # Maximum novelty if no neighbors

        # Compute cosine distances to all neighbors
        distances = [cosine(embedding, neighbor) for neighbor in nearest_embeddings]

        # Use minimum distance (closest neighbor)
        min_distance = min(distances)

        # Cosine distance is in [0, 2], normalize to [0, 1]
        return min(min_distance / 2.0, 1.0)

    def compute_entropy(self, embedding: np.ndarray, temperature: float = 1.0) -> float:
        """
        Compute information entropy of the embedding.

        Args:
            embedding: Input embedding
            temperature: Temperature for softmax normalization

        Returns:
            Normalized entropy score (0-1)
        """
        # Apply softmax with temperature
        exp_values = np.exp(embedding / temperature)
        probabilities = exp_values / np.sum(exp_values)

        # Compute Shannon entropy
        ent = entropy(probabilities)

        # Normalize by maximum possible entropy (log of embedding dimension)
        max_entropy = math.log(len(embedding))
        normalized_entropy = ent / max_entropy if max_entropy > 0 else 0.0

        return min(normalized_entropy, 1.0)

    def compute_rarity(
        self, embedding: np.ndarray, frequency_map: dict[str, int], total_count: int
    ) -> float:
        """
        Compute rarity score based on frequency of similar embeddings.

        Args:
            embedding: Input embedding
            frequency_map: Mapping of embedding clusters to frequency counts
            total_count: Total number of embeddings

        Returns:
            Rarity score (0-1), higher means rarer
        """
        if total_count == 0:
            return 1.0

        # Create a hash of the embedding for lookup
        embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()[:8]

        # Get frequency of this embedding cluster
        frequency = frequency_map.get(embedding_hash, 1)

        # Rarity is inverse of frequency (normalized)
        rarity = 1.0 - (frequency / total_count)

        return max(0.0, min(rarity, 1.0))

    def compute_cluster_distance(
        self, embedding: np.ndarray, cluster_centroids: list[np.ndarray]
    ) -> float:
        """
        Compute distance to cluster centroids.

        Args:
            embedding: Input embedding
            cluster_centroids: List of cluster centroid embeddings

        Returns:
            Normalized distance to nearest cluster (0-1)
        """
        if not cluster_centroids:
            return 1.0

        # Compute distances to all centroids
        distances = [cosine(embedding, centroid) for centroid in cluster_centroids]

        # Use minimum distance
        min_distance = min(distances)

        # Normalize to [0, 1]
        return min(min_distance / 2.0, 1.0)

    def compute_temporal_decay(
        self, similar_timestamps: list[float], current_time: float, decay_rate: float = 0.1
    ) -> float:
        """
        Compute temporal decay factor.

        Args:
            similar_timestamps: Timestamps of similar past queries
            current_time: Current timestamp
            decay_rate: Rate of exponential decay

        Returns:
            Temporal decay factor (0-1), higher means less recent similar queries
        """
        if not similar_timestamps:
            return 1.0

        # Compute time differences
        time_diffs = [current_time - ts for ts in similar_timestamps]

        # Apply exponential decay
        decay_scores = [math.exp(-decay_rate * diff) for diff in time_diffs]

        # Average decay (lower means less recent activity)
        avg_decay = sum(decay_scores) / len(decay_scores)

        # Invert so that higher score means more novel (less recent activity)
        return 1.0 - avg_decay

    def compute_score(
        self,
        embedding: np.ndarray,
        text: str,
        nearest_embeddings: Optional[list[np.ndarray]] = None,
        cluster_centroids: Optional[list[np.ndarray]] = None,
        frequency_map: Optional[dict[str, int]] = None,
        total_count: int = 0,
        similar_timestamps: Optional[list[float]] = None,
        current_time: Optional[float] = None,
    ) -> NoveltyScore:
        """
        Compute complete novelty score.

        Args:
            embedding: Input embedding vector
            text: Original text input
            nearest_embeddings: Nearest neighbor embeddings
            cluster_centroids: Cluster centroid embeddings
            frequency_map: Frequency map for rarity calculation
            total_count: Total embedding count
            similar_timestamps: Timestamps of similar queries
            current_time: Current timestamp

        Returns:
            Complete novelty score with metrics
        """
        # Set defaults
        nearest_embeddings = nearest_embeddings or []
        cluster_centroids = cluster_centroids or []
        frequency_map = frequency_map or {}
        similar_timestamps = similar_timestamps or []
        current_time = current_time or 0.0

        # Compute individual metrics
        semantic_distance = self.compute_semantic_distance(embedding, nearest_embeddings)
        ent = self.compute_entropy(embedding)
        rarity = self.compute_rarity(embedding, frequency_map, total_count)
        cluster_distance = self.compute_cluster_distance(embedding, cluster_centroids)
        temporal_decay = self.compute_temporal_decay(similar_timestamps, current_time)

        # Create metrics object
        metrics = NoveltyMetrics(
            semantic_distance=semantic_distance,
            entropy=ent,
            rarity=rarity,
            cluster_distance=cluster_distance,
            temporal_decay=temporal_decay,
        )

        # Compute input hash for deduplication
        input_hash = hashlib.sha256(text.encode()).hexdigest()

        # Create and return complete score
        return NoveltyScore.from_metrics(metrics, input_hash)
