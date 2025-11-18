#!/usr/bin/env python3
"""
Example usage of CAG (Cache-Augmented Generation).

This example demonstrates:
1. Setting up database and cache
2. Adding documents
3. Preloading documents into KV cache
4. Retrieval-free generation using cached context
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from cag import CAGEngine, DatabaseInterface, KVCacheStore, CacheManager
from common.utils import setup_logging
from common.gpu_utils import optimize_for_rtx4080


def example_database_setup():
    """Example: Setup database and add documents."""
    logger.info("\n" + "="*80)
    logger.info("Example 1: Database Setup")
    logger.info("="*80 + "\n")

    db_path = Path("./example_docs.db")

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    # Create database
    with DatabaseInterface(db_path) as db:
        db.create_tables()

        # Sample documents about AI and ML
        documents = [
            (
                "Machine Learning is a subset of Artificial Intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                {"topic": "ML", "source": "encyclopedia"}
            ),
            (
                "Neural Networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.",
                {"topic": "Neural Networks", "source": "textbook"}
            ),
            (
                "Transformers are a type of neural network architecture that has revolutionized natural language processing. They use self-attention mechanisms.",
                {"topic": "Transformers", "source": "research"}
            ),
            (
                "Cache-Augmented Generation (CAG) is a paradigm that eliminates real-time retrieval by preloading documents into the LLM's context window.",
                {"topic": "CAG", "source": "paper"}
            ),
            (
                "Knowledge Graphs are structured representations of information that connect entities through relationships, enabling semantic search and reasoning.",
                {"topic": "Knowledge Graphs", "source": "encyclopedia"}
            )
        ]

        doc_ids = db.add_documents(documents)

        logger.info(f"Added {len(doc_ids)} documents to database")

        # Display documents
        all_docs = db.get_all_documents()
        for doc in all_docs:
            logger.info(f"\nDocument {doc.id}:")
            logger.info(f"  Content: {doc.content[:80]}...")
            logger.info(f"  Metadata: {doc.metadata}")


def example_kv_cache():
    """Example: Create and manage KV caches."""
    logger.info("\n" + "="*80)
    logger.info("Example 2: KV Cache Management")
    logger.info("="*80 + "\n")

    cache_dir = Path("./example_cache")
    cache_dir.mkdir(exist_ok=True)

    # Create cache store
    store = KVCacheStore(
        cache_dir=cache_dir,
        model_name="llama3",
        device="cuda"
    )

    # Sample documents
    documents = [
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Machine Learning algorithms learn patterns from data.",
        "Deep Learning uses neural networks with many layers."
    ]

    logger.info("Computing KV cache for documents...")

    # Compute cache (Note: This requires actual model, may fail in example)
    try:
        cache = store.compute_kv_cache(documents)
        logger.info(f"Cache computed: {cache.metadata}")

        # Save cache
        cache_file = store.save_cache(cache, "example_cache")
        logger.info(f"Cache saved to: {cache_file}")

        # Load cache
        loaded_cache = store.load_cache("example_cache")
        if loaded_cache:
            logger.info("Cache loaded successfully")
            logger.info(f"Cache info: {loaded_cache.metadata}")

    except Exception as e:
        logger.warning(f"KV cache computation failed (expected in example): {e}")
        logger.info("In production, ensure model is available")


def example_end_to_end():
    """Example: End-to-end CAG workflow."""
    logger.info("\n" + "="*80)
    logger.info("Example 3: End-to-End CAG")
    logger.info("="*80 + "\n")

    db_path = Path("./example_docs.db")
    cache_dir = Path("./example_cache")

    # Initialize CAG engine
    engine = CAGEngine(
        db_path=db_path,
        cache_dir=cache_dir
    )

    # Add documents
    documents = [
        "The NVIDIA RTX 4080 has 16GB of GDDR6X memory and 9728 CUDA cores.",
        "Cache-Augmented Generation eliminates retrieval latency by preloading documents.",
        "KV caching stores key-value pairs from transformer attention layers.",
        "SimGRAG uses graph semantic distance to retrieve relevant subgraphs.",
        "Pattern graphs capture the semantic structure of user queries."
    ]

    logger.info("Adding documents to CAG engine...")
    doc_ids = engine.add_documents(documents, preload_cache=False)
    logger.info(f"Added {len(doc_ids)} documents")

    # Preload all documents
    logger.info("\nPreloading all documents into cache...")
    try:
        engine.preload_all_documents(cache_name="all_docs")
        logger.info("Documents preloaded successfully")
    except Exception as e:
        logger.warning(f"Preloading failed (model may not be available): {e}")

    # Generate answers (using Ollama fallback since KV cache may not work in example)
    questions = [
        "What is Cache-Augmented Generation?",
        "How much memory does the RTX 4080 have?",
        "What is a pattern graph?"
    ]

    logger.info("\nGenerating answers...")
    for question in questions:
        logger.info(f"\nQuestion: {question}")

        try:
            result = engine.generate(question, cache_name="all_docs")
            logger.info(f"Answer: {result['answer']}")
            if result.get('cache_info'):
                logger.info(f"Cache info: {result['cache_info']}")
        except Exception as e:
            logger.warning(f"Generation failed: {e}")

    # Statistics
    stats = engine.get_statistics()
    logger.info(f"\nEngine statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Cleanup
    engine.close()


def example_batch_generation():
    """Example: Batch generation with CAG."""
    logger.info("\n" + "="*80)
    logger.info("Example 4: Batch Generation")
    logger.info("="*80 + "\n")

    db_path = Path("./example_docs.db")
    cache_dir = Path("./example_cache")

    with CAGEngine(db_path, cache_dir) as engine:
        # Multiple questions
        questions = [
            "What are CUDA cores?",
            "Explain KV caching",
            "What is semantic distance?"
        ]

        logger.info(f"Batch generating answers for {len(questions)} questions...")

        try:
            results = engine.batch_generate(questions, cache_name="all_docs")

            for i, (question, result) in enumerate(zip(questions, results), 1):
                logger.info(f"\n{i}. Q: {question}")
                logger.info(f"   A: {result['answer'][:150]}...")

        except Exception as e:
            logger.warning(f"Batch generation failed: {e}")


def example_document_search():
    """Example: Search documents in database."""
    logger.info("\n" + "="*80)
    logger.info("Example 5: Document Search")
    logger.info("="*80 + "\n")

    db_path = Path("./example_docs.db")

    with DatabaseInterface(db_path) as db:
        # Search for documents
        search_queries = ["RTX 4080", "cache", "graph"]

        for query in search_queries:
            logger.info(f"\nSearching for: '{query}'")

            results = db.search_documents(query, limit=3)

            if results:
                logger.info(f"Found {len(results)} matching documents:")
                for doc in results:
                    logger.info(f"  - Doc {doc.id}: {doc.content[:80]}...")
            else:
                logger.info("  No matching documents found")


def cleanup():
    """Cleanup example files."""
    logger.info("\n" + "="*80)
    logger.info("Cleanup")
    logger.info("="*80 + "\n")

    # Remove example files
    db_path = Path("./example_docs.db")
    cache_dir = Path("./example_cache")

    if db_path.exists():
        db_path.unlink()
        logger.info(f"Removed {db_path}")

    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        logger.info(f"Removed {cache_dir}")

    logger.info("Cleanup complete")


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
        example_database_setup()
        example_kv_cache()
        example_end_to_end()
        example_batch_generation()
        example_document_search()

        logger.info("\n" + "="*80)
        logger.info("All examples completed successfully!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
