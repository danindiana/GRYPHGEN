#!/usr/bin/env python3
"""
Example usage of SimGRAG for knowledge graph question answering.

This example demonstrates:
1. Creating a knowledge graph
2. Generating pattern graphs from queries
3. Retrieving relevant subgraphs
4. Generating answers
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from simgrag import (
    PatternGenerator,
    SubgraphRetriever,
    GSDCalculator,
    SimGRAGEngine
)
from simgrag.subgraph_retriever import KnowledgeGraph, Triple
from common.utils import setup_logging
from common.gpu_utils import optimize_for_rtx4080


def create_sample_kg() -> KnowledgeGraph:
    """
    Create a sample knowledge graph about movies.

    Returns:
        KnowledgeGraph with movie data
    """
    logger.info("Creating sample knowledge graph...")

    kg = KnowledgeGraph()

    # Add movie triples
    triples = [
        # The Matrix
        Triple("The Matrix", "directed_by", "The Wachowskis"),
        Triple("The Matrix", "released_in", "1999"),
        Triple("The Matrix", "starred", "Keanu Reeves"),
        Triple("The Matrix", "starred", "Laurence Fishburne"),
        Triple("The Matrix", "genre", "Science Fiction"),

        # Inception
        Triple("Inception", "directed_by", "Christopher Nolan"),
        Triple("Inception", "released_in", "2010"),
        Triple("Inception", "starred", "Leonardo DiCaprio"),
        Triple("Inception", "genre", "Science Fiction"),

        # The Dark Knight
        Triple("The Dark Knight", "directed_by", "Christopher Nolan"),
        Triple("The Dark Knight", "released_in", "2008"),
        Triple("The Dark Knight", "starred", "Christian Bale"),
        Triple("The Dark Knight", "genre", "Action"),

        # John Wick
        Triple("John Wick", "directed_by", "Chad Stahelski"),
        Triple("John Wick", "released_in", "2014"),
        Triple("John Wick", "starred", "Keanu Reeves"),
        Triple("John Wick", "genre", "Action"),

        # Director info
        Triple("Christopher Nolan", "nationality", "British-American"),
        Triple("Christopher Nolan", "born_in", "1970"),

        # Actor info
        Triple("Keanu Reeves", "nationality", "Canadian"),
        Triple("Keanu Reeves", "born_in", "1964"),
        Triple("Leonardo DiCaprio", "nationality", "American"),
        Triple("Leonardo DiCaprio", "born_in", "1974"),
    ]

    kg.add_triples(triples)

    logger.info(f"Knowledge graph created with {len(kg)} triples")

    return kg


def example_pattern_generation():
    """Example: Generate pattern graphs from queries."""
    logger.info("\n" + "="*80)
    logger.info("Example 1: Pattern Generation")
    logger.info("="*80 + "\n")

    generator = PatternGenerator(llm_model="llama3")

    queries = [
        "Who directed The Matrix?",
        "What movies did Christopher Nolan direct?",
        "Which science fiction movies were released after 2000?"
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        pattern = generator.generate(query)
        logger.info(f"Pattern Graph:")
        for triple in pattern.triples:
            logger.info(f"  {triple}")


def example_subgraph_retrieval():
    """Example: Retrieve subgraphs from knowledge graph."""
    logger.info("\n" + "="*80)
    logger.info("Example 2: Subgraph Retrieval")
    logger.info("="*80 + "\n")

    # Create KG
    kg = create_sample_kg()

    # Create components
    gsd_calculator = GSDCalculator(device="cuda")
    retriever = SubgraphRetriever(kg, gsd_calculator, top_k=3)
    pattern_generator = PatternGenerator()

    # Query
    query = "Who directed The Matrix?"
    logger.info(f"Query: {query}\n")

    # Generate pattern
    pattern = pattern_generator.generate(query)
    logger.info(f"Pattern Graph: {pattern}\n")

    # Retrieve subgraphs
    subgraphs = retriever.retrieve_top_k(pattern)

    logger.info(f"Retrieved {len(subgraphs)} subgraphs:\n")
    for i, subgraph in enumerate(subgraphs, 1):
        logger.info(f"Subgraph {i} (GSD: {subgraph.gsd:.4f}):")
        for triple in subgraph.triples:
            logger.info(f"  {triple}")
        logger.info("")


def example_end_to_end():
    """Example: End-to-end question answering with SimGRAG."""
    logger.info("\n" + "="*80)
    logger.info("Example 3: End-to-End Question Answering")
    logger.info("="*80 + "\n")

    # Create KG
    kg = create_sample_kg()

    # Create engine
    engine = SimGRAGEngine(kg)

    # Questions
    questions = [
        "Who directed The Matrix?",
        "What movies did Keanu Reeves star in?",
        "Which movies were directed by Christopher Nolan?",
        "What genre is Inception?"
    ]

    for question in questions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Question: {question}")
        logger.info('='*60)

        result = engine.answer_query(question, verbose=True)

        logger.info(f"\nAnswer: {result['answer']}")
        logger.info(f"Metadata: {result['metadata']}")
        logger.info("")


def example_batch_processing():
    """Example: Batch processing multiple queries."""
    logger.info("\n" + "="*80)
    logger.info("Example 4: Batch Processing")
    logger.info("="*80 + "\n")

    # Create KG
    kg = create_sample_kg()

    # Create engine
    engine = SimGRAGEngine(kg)

    # Multiple questions
    questions = [
        "Who directed The Matrix?",
        "What movies did Christopher Nolan direct?",
        "Which actors starred in The Matrix?"
    ]

    logger.info(f"Processing {len(questions)} questions in batch...\n")

    results = engine.batch_answer(questions)

    for i, (question, result) in enumerate(zip(questions, results), 1):
        logger.info(f"{i}. Q: {question}")
        logger.info(f"   A: {result['answer']}\n")


def main():
    """Run all examples."""
    # Setup logging
    setup_logging(level="INFO")

    # Optimize for RTX 4080
    logger.info("Optimizing for NVIDIA RTX 4080...")
    gpu_manager = optimize_for_rtx4080()
    gpu_manager.log_memory_stats()

    # Run examples
    try:
        example_pattern_generation()
        example_subgraph_retrieval()
        example_end_to_end()
        example_batch_processing()

        logger.info("\n" + "="*80)
        logger.info("All examples completed successfully!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
