"""
Main CAG (Cache-Augmented Generation) engine.

Provides end-to-end functionality for retrieval-free generation using
preloaded document contexts and KV caching.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import ollama
from loguru import logger

from .db_interface import DatabaseInterface, Document
from .kv_cache import KVCacheStore, CacheManager
from common.config import CAGConfig
from common.utils import chunk_text


class CAGEngine:
    """
    Main engine for Cache-Augmented Generation.

    Implements the CAG workflow:
    1. Document Preloading from database
    2. KV Cache Computation
    3. Retrieval-Free Inference using cached context
    """

    def __init__(
        self,
        db_path: Path,
        cache_dir: Path,
        config: Optional[CAGConfig] = None
    ):
        """
        Initialize CAG engine.

        Args:
            db_path: Path to SQLite database
            cache_dir: Directory for KV cache storage
            config: Configuration (uses defaults if None)

        Example:
            >>> engine = CAGEngine(
            ...     db_path=Path("documents.db"),
            ...     cache_dir=Path("./cache")
            ... )
        """
        self.config = config or CAGConfig()

        # Initialize database interface
        self.db = DatabaseInterface(db_path)
        self.db.connect()
        self.db.create_tables()

        # Initialize KV cache store
        self.cache_store = KVCacheStore(
            cache_dir=cache_dir,
            model_name=self.config.llm_model,
            device="cuda"
        )

        # Initialize cache manager
        self.cache_manager = CacheManager(
            cache_store=self.cache_store,
            auto_update=False
        )

        logger.info("CAG engine initialized successfully")

    def add_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        preload_cache: bool = True,
        cache_name: str = "default"
    ) -> List[int]:
        """
        Add documents to database and optionally preload into cache.

        Args:
            documents: List of document texts
            metadata_list: Optional list of metadata dictionaries
            preload_cache: Whether to immediately preload into KV cache
            cache_name: Name for the cache if preloading

        Returns:
            List of document IDs

        Example:
            >>> engine = CAGEngine(Path("docs.db"), Path("./cache"))
            >>> docs = ["Document 1", "Document 2"]
            >>> ids = engine.add_documents(docs, preload_cache=True)
        """
        logger.info(f"Adding {len(documents)} documents")

        # Prepare documents for database
        if metadata_list is None:
            metadata_list = [None] * len(documents)

        doc_tuples = list(zip(documents, metadata_list))

        # Add to database
        document_ids = self.db.add_documents(doc_tuples)

        # Preload into cache if requested
        if preload_cache:
            logger.info("Preloading documents into KV cache")
            self.cache_manager.preload_documents(
                documents=documents,
                document_ids=document_ids,
                cache_name=cache_name,
                save=True
            )

        logger.info(f"Added {len(document_ids)} documents successfully")

        return document_ids

    def preload_all_documents(
        self,
        cache_name: str = "default",
        chunk_size: Optional[int] = None,
        max_documents: Optional[int] = None
    ) -> None:
        """
        Preload all documents from database into KV cache.

        Args:
            cache_name: Name for the cache
            chunk_size: Optional chunk size for large document sets
            max_documents: Maximum number of documents to preload

        Example:
            >>> engine.preload_all_documents(cache_name="all_docs")
        """
        logger.info("Preloading all documents from database")

        # Get all documents
        documents_objs = self.db.get_all_documents(limit=max_documents)

        if not documents_objs:
            logger.warning("No documents found in database")
            return

        # Extract content and IDs
        documents = [doc.content for doc in documents_objs]
        document_ids = [doc.id for doc in documents_objs]

        # Preload into cache
        self.cache_manager.preload_documents(
            documents=documents,
            document_ids=document_ids,
            cache_name=cache_name,
            save=True
        )

        logger.info(f"Preloaded {len(documents)} documents successfully")

    def generate(
        self,
        query: str,
        cache_name: str = "default",
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Generate answer using cached context (retrieval-free).

        Args:
            query: User query
            cache_name: Name of the cache to use
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with:
                - answer: Generated answer
                - cache_info: Information about cache used
                - metadata: Additional information

        Example:
            >>> result = engine.generate("What is cache-augmented generation?")
            >>> print(result['answer'])
        """
        logger.info(f"Generating answer for query: {query}")

        # Get cache
        kv_cache = self.cache_manager.get_cache(cache_name, load_if_missing=True)

        if kv_cache is None:
            logger.error(f"Cache not found: {cache_name}")
            return {
                "answer": "Error: No cache available. Please preload documents first.",
                "cache_info": None,
                "metadata": {"error": "Cache not found"}
            }

        # For Ollama, we don't directly use the KV cache (Ollama manages this internally)
        # Instead, we provide the documents as context

        # Get documents from cache
        documents = []
        for doc_id in kv_cache.document_ids:
            doc = self.db.get_document(doc_id)
            if doc:
                documents.append(doc.content)

        # Create context
        context = "\n\n".join(documents)

        # Truncate if too long (Ollama/LLM context window)
        max_context_length = self.config.max_context_length
        if len(context) > max_context_length:
            logger.warning(f"Context too long ({len(context)} chars), truncating to {max_context_length}")
            context = context[:max_context_length]

        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer the question concisely based on the provided context.

Answer:"""

        try:
            # Generate answer using Ollama
            response = ollama.generate(
                model=self.config.llm_model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )

            answer = response["response"].strip()

            logger.info(f"Generated answer: {answer[:100]}...")

            return {
                "answer": answer,
                "cache_info": {
                    "cache_name": cache_name,
                    "num_documents": len(kv_cache.document_ids),
                    "total_tokens": kv_cache.metadata.get("total_tokens", "unknown")
                },
                "metadata": {
                    "query": query,
                    "model": self.config.llm_model,
                    "temperature": temperature
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "cache_info": None,
                "metadata": {"error": str(e)}
            }

    def batch_generate(
        self,
        queries: List[str],
        cache_name: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of queries
            cache_name: Name of the cache to use

        Returns:
            List of result dictionaries

        Example:
            >>> queries = ["What is CAG?", "How does KV caching work?"]
            >>> results = engine.batch_generate(queries)
        """
        logger.info(f"Batch generating for {len(queries)} queries")

        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.generate(query, cache_name=cache_name)
            results.append(result)

        return results

    def search_documents(
        self,
        query: str,
        limit: int = 10
    ) -> List[Document]:
        """
        Search documents in database (for hybrid approaches).

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching documents

        Example:
            >>> docs = engine.search_documents("machine learning")
        """
        return self.db.search_documents(query, limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "database": {
                "total_documents": self.db.get_document_count(),
                "db_path": str(self.db.db_path)
            },
            "caches": {
                "available_caches": self.cache_store.list_caches(),
                "cache_dir": str(self.cache_store.cache_dir)
            },
            "config": {
                "llm_model": self.config.llm_model,
                "max_context_length": self.config.max_context_length,
                "chunk_size": self.config.chunk_size
            }
        }

        return stats

    def clear_cache(self, cache_name: str = "default") -> bool:
        """
        Clear a specific cache.

        Args:
            cache_name: Name of the cache

        Returns:
            True if cleared, False if not found
        """
        # Unload from memory
        self.cache_manager.unload_cache(cache_name)

        # Delete from disk
        return self.cache_store.delete_cache(cache_name)

    def close(self) -> None:
        """Close database connection and cleanup."""
        self.db.disconnect()
        logger.info("CAG engine closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
