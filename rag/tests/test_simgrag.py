"""Tests for SimGRAG components."""

import pytest
from simgrag.pattern_generator import PatternGenerator, PatternGraph, Triple
from simgrag.gsd_calculator import GSDCalculator
from simgrag.subgraph_retriever import KnowledgeGraph, SubgraphRetriever


def test_triple_creation():
    """Test Triple creation."""
    triple = Triple(head="A", relation="rel", tail="B")
    assert triple.head == "A"
    assert triple.relation == "rel"
    assert triple.tail == "B"


def test_pattern_graph_creation():
    """Test PatternGraph creation."""
    triples = [
        Triple("A", "rel1", "B"),
        Triple("B", "rel2", "C")
    ]
    pattern = PatternGraph(triples=triples, query="test query")
    assert len(pattern) == 2
    assert pattern.query == "test query"


def test_knowledge_graph():
    """Test KnowledgeGraph creation and operations."""
    kg = KnowledgeGraph()
    triple = Triple("Entity1", "relation", "Entity2")
    kg.add_triple(triple)

    assert len(kg) == 1
    assert "Entity1" in kg.graph.nodes()
    assert "Entity2" in kg.graph.nodes()


def test_knowledge_graph_neighbors():
    """Test knowledge graph neighbor retrieval."""
    kg = KnowledgeGraph()
    kg.add_triple(Triple("A", "rel1", "B"))
    kg.add_triple(Triple("A", "rel2", "C"))

    neighbors = kg.get_neighbors("A")
    assert "B" in neighbors
    assert "C" in neighbors


@pytest.mark.skipif(True, reason="Requires embedding model and GPU")
def test_gsd_calculator():
    """Test GSD calculator."""
    calculator = GSDCalculator(device="cpu")

    distance = calculator.embedding_distance("apple", "orange")
    assert isinstance(distance, float)
    assert 0 <= distance <= 2  # Cosine distance range


def test_pattern_graph_representation():
    """Test pattern graph string representation."""
    triples = [Triple("A", "rel", "B")]
    pattern = PatternGraph(triples=triples, query="test")

    repr_str = repr(pattern)
    assert "PatternGraph" in repr_str
    assert "test" in repr_str
