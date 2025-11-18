"""
SimGRAG: Similarity-based Graph Retrieval-Augmented Generation.

A novel method for Knowledge Graph-driven RAG that aligns query texts with
KG structures through query-to-pattern and pattern-to-subgraph alignment.

Reference: https://arxiv.org/pdf/2412.15272
Implementation: https://github.com/YZ-Cai/SimGRAG
"""

from .pattern_generator import PatternGenerator
from .subgraph_retriever import SubgraphRetriever, Subgraph
from .gsd_calculator import GSDCalculator
from .simgrag_engine import SimGRAGEngine

__version__ = "1.0.0"

__all__ = [
    "PatternGenerator",
    "SubgraphRetriever",
    "Subgraph",
    "GSDCalculator",
    "SimGRAGEngine",
]
