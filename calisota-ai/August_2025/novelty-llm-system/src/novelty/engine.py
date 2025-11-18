"""Novelty engine integrating scoring with vector database."""

import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import NoveltyScore
from .scorer import NoveltyScorer


class NoveltyEngine:
    """
    High-level novelty engine integrating embedding generation,
    vector search, and novelty scoring.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        scorer: Optional[NoveltyScorer] = None,
        k_neighbors: int = 10,
        device: str = "cpu",
    ):
        """
        Initialize the novelty engine.

        Args:
            model_name: Sentence transformer model name
            scorer: Custom novelty scorer (optional)
            k_neighbors: Number of neighbors to consider for scoring
            device: Device for model inference (cpu/cuda)
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.scorer = scorer or NoveltyScorer()
        self.k_neighbors = k_neighbors

        # Placeholders for vector database and storage
        # These should be injected via dependency injection in production
        self.vector_db = None
        self.metadata_store = None

    def set_vector_db(self, vector_db) -> None:
        """Set the vector database instance."""
        self.vector_db = vector_db

    def set_metadata_store(self, metadata_store) -> None:
        """Set the metadata store instance."""
        self.metadata_store = metadata_store

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding

    async def compute_novelty(
        self,
        text: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> NoveltyScore:
        """
        Compute novelty score for input text.

        Args:
            text: Input text to score
            user_id: Optional user ID for tenant-specific scoring
            tenant_id: Optional tenant ID for multi-tenancy

        Returns:
            Complete novelty score with metrics
        """
        # Generate embedding
        embedding = self.generate_embedding(text)

        # Search for nearest neighbors (if vector DB is available)
        nearest_embeddings = []
        cluster_centroids = []
        frequency_map = {}
        total_count = 0
        similar_timestamps = []

        if self.vector_db:
            # Query vector database for similar embeddings
            search_results = await self.vector_db.search(
                embedding=embedding, k=self.k_neighbors, tenant_id=tenant_id
            )

            nearest_embeddings = [result["embedding"] for result in search_results]

            # Get cluster information
            cluster_info = await self.vector_db.get_cluster_info(tenant_id=tenant_id)
            cluster_centroids = cluster_info.get("centroids", [])

            # Get frequency information
            frequency_map = cluster_info.get("frequency_map", {})
            total_count = cluster_info.get("total_count", 0)

        if self.metadata_store:
            # Get timestamps of similar past queries
            similar_queries = await self.metadata_store.get_similar_queries(
                text=text, user_id=user_id, tenant_id=tenant_id, limit=10
            )
            similar_timestamps = [q["timestamp"] for q in similar_queries]

        # Current time
        current_time = time.time()

        # Compute novelty score
        score = self.scorer.compute_score(
            embedding=embedding,
            text=text,
            nearest_embeddings=nearest_embeddings,
            cluster_centroids=cluster_centroids,
            frequency_map=frequency_map,
            total_count=total_count,
            similar_timestamps=similar_timestamps,
            current_time=current_time,
        )

        return score

    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        novelty_score: NoveltyScore,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store embedding and metadata in vector database.

        Args:
            text: Original text
            embedding: Embedding vector
            novelty_score: Computed novelty score
            user_id: User ID
            tenant_id: Tenant ID
            metadata: Additional metadata

        Returns:
            ID of stored embedding
        """
        if not self.vector_db:
            raise RuntimeError("Vector database not configured")

        metadata = metadata or {}
        metadata.update(
            {
                "text": text,
                "novelty_score": novelty_score.score,
                "novelty_level": novelty_score.level.value,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "timestamp": time.time(),
            }
        )

        embedding_id = await self.vector_db.insert(
            embedding=embedding, metadata=metadata, tenant_id=tenant_id
        )

        return embedding_id

    async def process(
        self,
        text: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        store: bool = True,
    ) -> tuple[np.ndarray, NoveltyScore]:
        """
        Complete processing pipeline: embed, score, and optionally store.

        Args:
            text: Input text
            user_id: User ID
            tenant_id: Tenant ID
            store: Whether to store the embedding

        Returns:
            Tuple of (embedding, novelty_score)
        """
        # Generate embedding
        embedding = self.generate_embedding(text)

        # Compute novelty score
        novelty_score = await self.compute_novelty(text, user_id, tenant_id)

        # Store if requested
        if store and self.vector_db:
            await self.store_embedding(
                text=text,
                embedding=embedding,
                novelty_score=novelty_score,
                user_id=user_id,
                tenant_id=tenant_id,
            )

        return embedding, novelty_score
