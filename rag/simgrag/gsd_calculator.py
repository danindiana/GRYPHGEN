"""
Graph Semantic Distance (GSD) calculation for SimGRAG.

Computes the semantic distance between pattern graphs and subgraphs
using entity and relation embeddings.
"""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from .pattern_generator import PatternGraph, Triple


class GSDCalculator:
    """
    Calculates Graph Semantic Distance (GSD) between pattern and subgraphs.

    GSD quantifies the alignment between a pattern graph and a candidate subgraph
    by measuring the semantic distance of corresponding entities and relations.
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda"
    ):
        """
        Initialize GSD calculator.

        Args:
            embedding_model: Pre-loaded embedding model (optional)
            model_name: Name of embedding model if not provided
            device: Device to run model on ("cuda" or "cpu")
        """
        if embedding_model is None:
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, device=device)
        else:
            self.embedding_model = embedding_model

        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

        logger.info("GSDCalculator initialized")

    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for a text, with caching.

        Args:
            text: Input text
            use_cache: Whether to use embedding cache

        Returns:
            Embedding vector
        """
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]

        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        if use_cache:
            self._embedding_cache[text] = embedding

        return embedding

    def embedding_distance(self, text1: str, text2: str) -> float:
        """
        Calculate embedding distance between two texts.

        Uses cosine distance (1 - cosine similarity).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Distance value (0 = identical, 2 = opposite)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2)

        # Convert to distance (0 = identical, 2 = opposite)
        distance = 1.0 - similarity

        return float(distance)

    def calculate_triple_distance(
        self,
        pattern_triple: Triple,
        subgraph_triple: Triple
    ) -> float:
        """
        Calculate distance between two triples.

        Distance is the sum of:
        - Head entity distance
        - Relation distance
        - Tail entity distance

        Args:
            pattern_triple: Triple from pattern graph
            subgraph_triple: Triple from subgraph

        Returns:
            Total distance for the triple
        """
        head_dist = self.embedding_distance(
            pattern_triple.head,
            subgraph_triple.head
        )

        relation_dist = self.embedding_distance(
            pattern_triple.relation,
            subgraph_triple.relation
        )

        tail_dist = self.embedding_distance(
            pattern_triple.tail,
            subgraph_triple.tail
        )

        total_distance = head_dist + relation_dist + tail_dist

        return total_distance

    def calculate_gsd(
        self,
        pattern: PatternGraph,
        subgraph_triples: List[Triple],
        mapping: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Calculate Graph Semantic Distance (GSD) between pattern and subgraph.

        GSD = Σ(distance(pattern_triple, matched_subgraph_triple)) / num_triples

        Args:
            pattern: Pattern graph
            subgraph_triples: List of triples from subgraph
            mapping: Optional entity mapping from pattern to subgraph

        Returns:
            GSD value (lower is better)

        Example:
            >>> calculator = GSDCalculator()
            >>> pattern = PatternGraph(...)
            >>> subgraph = [Triple(...), ...]
            >>> gsd = calculator.calculate_gsd(pattern, subgraph)
        """
        if not pattern.triples:
            return float('inf')

        if not subgraph_triples:
            return float('inf')

        total_distance = 0.0
        num_matched = 0

        # For each triple in pattern, find best match in subgraph
        for pattern_triple in pattern.triples:
            min_distance = float('inf')

            for subgraph_triple in subgraph_triples:
                distance = self.calculate_triple_distance(
                    pattern_triple,
                    subgraph_triple
                )

                if distance < min_distance:
                    min_distance = distance

            total_distance += min_distance
            num_matched += 1

        # Calculate average GSD
        gsd = total_distance / num_matched if num_matched > 0 else float('inf')

        logger.debug(f"Calculated GSD: {gsd:.4f}")

        return gsd

    def calculate_lower_bound_gsd(
        self,
        pattern: PatternGraph,
        partial_subgraph: List[Triple],
        remaining_pattern_size: int
    ) -> float:
        """
        Calculate lower bound of GSD for pruning during DFS.

        This is used to prune branches early during subgraph expansion.
        If the lower bound is already worse than the current best, we can
        skip expanding this branch.

        Args:
            pattern: Pattern graph
            partial_subgraph: Partially constructed subgraph
            remaining_pattern_size: Number of triples left to match

        Returns:
            Lower bound GSD estimate
        """
        if not partial_subgraph:
            return 0.0

        # Calculate GSD for matched triples so far
        current_gsd = self.calculate_gsd(pattern, partial_subgraph)

        # Lower bound assumes perfect matches for remaining triples (distance = 0)
        # So lower bound = current_gsd * (matched / total)
        matched_size = len(partial_subgraph)
        total_size = matched_size + remaining_pattern_size

        if total_size == 0:
            return current_gsd

        lower_bound = (current_gsd * matched_size) / total_size

        return lower_bound

    def batch_calculate_gsd(
        self,
        pattern: PatternGraph,
        subgraphs: List[List[Triple]]
    ) -> List[float]:
        """
        Calculate GSD for multiple subgraphs in batch.

        Args:
            pattern: Pattern graph
            subgraphs: List of subgraph triple lists

        Returns:
            List of GSD values
        """
        logger.info(f"Batch calculating GSD for {len(subgraphs)} subgraphs")

        gsd_values = []
        for subgraph in subgraphs:
            gsd = self.calculate_gsd(pattern, subgraph)
            gsd_values.append(gsd)

        return gsd_values

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._embedding_cache)
