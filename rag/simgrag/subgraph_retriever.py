"""
Subgraph retrieval for SimGRAG.

Implements DFS-based retrieval of top-k subgraphs from knowledge graphs
that are semantically aligned with the pattern graph.
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import heapq
import networkx as nx
from loguru import logger

from .pattern_generator import PatternGraph, Triple
from .gsd_calculator import GSDCalculator


@dataclass(order=True)
class Subgraph:
    """
    Represents a retrieved subgraph with its GSD score.

    Uses @dataclass(order=True) to enable comparison by GSD for heap operations.
    """
    gsd: float
    triples: List[Triple] = field(compare=False)
    mapping: Dict[str, str] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        triples_str = "\n  ".join(str(t) for t in self.triples)
        return f"Subgraph(GSD={self.gsd:.4f}, {len(self.triples)} triples)\n  {triples_str}"


class KnowledgeGraph:
    """
    Simple knowledge graph representation using NetworkX.

    Stores triples and provides efficient querying.
    """

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.graph = nx.MultiDiGraph()
        self.triples: List[Triple] = []
        self._entity_triples: Dict[str, List[Triple]] = {}

    def add_triple(self, triple: Triple) -> None:
        """
        Add a triple to the knowledge graph.

        Args:
            triple: Triple to add
        """
        self.triples.append(triple)

        # Add to graph
        self.graph.add_edge(
            triple.head,
            triple.tail,
            relation=triple.relation,
            triple=triple
        )

        # Index by entities
        if triple.head not in self._entity_triples:
            self._entity_triples[triple.head] = []
        if triple.tail not in self._entity_triples:
            self._entity_triples[triple.tail] = []

        self._entity_triples[triple.head].append(triple)
        self._entity_triples[triple.tail].append(triple)

    def add_triples(self, triples: List[Triple]) -> None:
        """
        Add multiple triples to the knowledge graph.

        Args:
            triples: List of triples to add
        """
        for triple in triples:
            self.add_triple(triple)

        logger.info(f"Added {len(triples)} triples to knowledge graph")

    def get_entity_triples(self, entity: str) -> List[Triple]:
        """
        Get all triples containing an entity.

        Args:
            entity: Entity name

        Returns:
            List of triples
        """
        return self._entity_triples.get(entity, [])

    def get_neighbors(self, entity: str) -> Set[str]:
        """
        Get neighboring entities.

        Args:
            entity: Entity name

        Returns:
            Set of neighboring entity names
        """
        neighbors = set()

        if entity in self.graph:
            # Outgoing edges
            neighbors.update(self.graph.successors(entity))
            # Incoming edges
            neighbors.update(self.graph.predecessors(entity))

        return neighbors

    def __len__(self) -> int:
        """Get number of triples in the graph."""
        return len(self.triples)


class SubgraphRetriever:
    """
    Retrieves top-k subgraphs from knowledge graph using DFS with GSD pruning.

    This implements the Pattern-to-Subgraph Alignment phase of SimGRAG.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        gsd_calculator: GSDCalculator,
        top_k: int = 3,
        max_candidates: int = 10
    ):
        """
        Initialize subgraph retriever.

        Args:
            knowledge_graph: Knowledge graph to retrieve from
            gsd_calculator: GSD calculator for scoring
            top_k: Number of top subgraphs to retrieve
            max_candidates: Maximum candidates to consider per entity/relation
        """
        self.kg = knowledge_graph
        self.gsd_calculator = gsd_calculator
        self.top_k = top_k
        self.max_candidates = max_candidates

        logger.info(f"SubgraphRetriever initialized with top_k={top_k}")

    def get_candidate_entities(
        self,
        pattern_entity: str,
        k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k candidate entities similar to pattern entity.

        Uses embedding similarity to find candidates.

        Args:
            pattern_entity: Entity from pattern graph
            k: Number of candidates (uses self.max_candidates if None)

        Returns:
            List of (entity, similarity_score) tuples
        """
        if k is None:
            k = self.max_candidates

        # Get all entities from KG
        all_entities = list(self.kg.graph.nodes())

        if not all_entities:
            return []

        # Calculate similarity scores
        pattern_emb = self.gsd_calculator.get_embedding(pattern_entity)
        candidates = []

        for entity in all_entities:
            entity_emb = self.gsd_calculator.get_embedding(entity)
            similarity = float(pattern_emb.dot(entity_emb))
            candidates.append((entity, similarity))

        # Sort by similarity and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:k]

    def retrieve_top_k(self, pattern: PatternGraph) -> List[Subgraph]:
        """
        Retrieve top-k subgraphs matching the pattern.

        Uses DFS with GSD-based pruning to efficiently search the KG.

        Args:
            pattern: Pattern graph to match

        Returns:
            List of top-k Subgraph objects sorted by GSD (best first)

        Example:
            >>> retriever = SubgraphRetriever(kg, calculator)
            >>> pattern = PatternGraph(...)
            >>> subgraphs = retriever.retrieve_top_k(pattern)
            >>> print(f"Best GSD: {subgraphs[0].gsd}")
        """
        if not pattern.triples:
            logger.warning("Pattern has no triples")
            return []

        logger.info(f"Retrieving top-{self.top_k} subgraphs for pattern with {len(pattern)} triples")

        # Min-heap to maintain top-k subgraphs (smallest GSD = best)
        # But heapq is a min-heap, and we want to keep largest k, so we negate GSD
        # Actually for GSD, smaller is better, so we want the k subgraphs with smallest GSD
        # So we use a max-heap (negate values) to maintain k smallest
        top_k_heap: List[Tuple[float, Subgraph]] = []

        # Start DFS from candidate entities for first pattern triple
        first_triple = pattern.triples[0]
        start_candidates = self.get_candidate_entities(first_triple.head)

        for start_entity, _ in start_candidates:
            # Initialize mapping
            mapping = {first_triple.head: start_entity}

            # Start DFS expansion
            self._dfs_expand(
                pattern=pattern,
                mapping=mapping,
                triple_idx=0,
                current_triples=[],
                top_k_heap=top_k_heap
            )

        # Extract subgraphs from heap and sort by GSD
        subgraphs = [subgraph for _, subgraph in top_k_heap]
        subgraphs.sort(key=lambda s: s.gsd)

        logger.info(f"Retrieved {len(subgraphs)} subgraphs")

        if subgraphs:
            logger.info(f"Best GSD: {subgraphs[0].gsd:.4f}")

        return subgraphs

    def _dfs_expand(
        self,
        pattern: PatternGraph,
        mapping: Dict[str, str],
        triple_idx: int,
        current_triples: List[Triple],
        top_k_heap: List[Tuple[float, Subgraph]]
    ) -> None:
        """
        Recursively expand subgraph using DFS.

        Args:
            pattern: Pattern graph
            mapping: Current entity mapping (pattern -> KG)
            triple_idx: Current triple index in pattern
            current_triples: Current list of matched triples
            top_k_heap: Heap maintaining top-k subgraphs
        """
        # Base case: all triples matched
        if triple_idx >= len(pattern.triples):
            # Calculate final GSD
            gsd = self.gsd_calculator.calculate_gsd(pattern, current_triples)

            subgraph = Subgraph(
                gsd=gsd,
                triples=current_triples.copy(),
                mapping=mapping.copy()
            )

            # Add to heap (using negative GSD for max-heap of smallest GSDs)
            if len(top_k_heap) < self.top_k:
                heapq.heappush(top_k_heap, (-gsd, subgraph))
            elif gsd < -top_k_heap[0][0]:  # Better than worst in heap
                heapq.heapreplace(top_k_heap, (-gsd, subgraph))

            return

        # Get current pattern triple
        pattern_triple = pattern.triples[triple_idx]

        # Check if head is mapped
        if pattern_triple.head not in mapping:
            # Need to map head entity first
            # For simplicity, skip this triple and try next
            self._dfs_expand(
                pattern, mapping, triple_idx + 1,
                current_triples, top_k_heap
            )
            return

        # Head entity is mapped, find candidates for tail
        head_entity = mapping[pattern_triple.head]

        # Get outgoing triples from head entity
        candidate_triples = self.kg.get_entity_triples(head_entity)

        # Filter and score candidates
        for kg_triple in candidate_triples:
            if kg_triple.head != head_entity:
                continue

            # Check if tail can be mapped
            if pattern_triple.tail in mapping:
                # Tail must match existing mapping
                if kg_triple.tail != mapping[pattern_triple.tail]:
                    continue
            else:
                # Create new mapping for tail
                mapping[pattern_triple.tail] = kg_triple.tail

            # Add triple to current subgraph
            new_triples = current_triples + [kg_triple]

            # Pruning: check if lower bound GSD is acceptable
            if len(top_k_heap) >= self.top_k:
                worst_gsd = -top_k_heap[0][0]
                remaining = len(pattern.triples) - triple_idx - 1

                lower_bound = self.gsd_calculator.calculate_lower_bound_gsd(
                    pattern, new_triples, remaining
                )

                if lower_bound > worst_gsd:
                    # Prune this branch
                    if pattern_triple.tail not in mapping:
                        mapping.pop(pattern_triple.tail, None)
                    continue

            # Recurse to next triple
            self._dfs_expand(
                pattern,
                mapping,
                triple_idx + 1,
                new_triples,
                top_k_heap
            )

            # Backtrack mapping if it was newly created
            if pattern_triple.tail in mapping and mapping[pattern_triple.tail] == kg_triple.tail:
                mapping.pop(pattern_triple.tail, None)

    def batch_retrieve(
        self,
        patterns: List[PatternGraph]
    ) -> List[List[Subgraph]]:
        """
        Retrieve subgraphs for multiple patterns.

        Args:
            patterns: List of pattern graphs

        Returns:
            List of lists of subgraphs (one list per pattern)
        """
        logger.info(f"Batch retrieving for {len(patterns)} patterns")

        all_subgraphs = []
        for pattern in patterns:
            subgraphs = self.retrieve_top_k(pattern)
            all_subgraphs.append(subgraphs)

        return all_subgraphs
