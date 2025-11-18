"""
GPU-accelerated FAISS vector database manager.
Optimized for NVIDIA RTX 4080 16GB.
"""

import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.calisota.core.config import Settings

logger = logging.getLogger(__name__)


class FAISSManager:
    """
    Manages FAISS vector database with GPU acceleration.

    Features:
    - GPU-accelerated similarity search
    - IVF (Inverted File) indexing for large-scale retrieval
    - Support for adding, searching, and persisting embeddings
    - Optimized for RTX 4080 16GB
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize FAISS manager with GPU support."""
        self.settings = settings
        self.dimension = settings.faiss_dimension
        self.use_gpu = settings.use_gpu_index and torch.cuda.is_available()

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=f"cuda:{settings.cuda_device}" if self.use_gpu else "cpu"
        )

        # Initialize FAISS index
        self.index: Optional[faiss.Index] = None
        self.gpu_resources: Optional[faiss.StandardGpuResources] = None
        self.metadata: list[dict] = []

        logger.info(
            f"Initialized FAISS Manager (GPU: {self.use_gpu}, "
            f"Dimension: {self.dimension})"
        )

    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index optimized for RTX 4080.

        Uses IVF (Inverted File) with Product Quantization for memory efficiency.
        """
        # Create base index with IVF and PQ
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            self.settings.faiss_nlist,  # Number of clusters
            8,  # Number of subquantizers
            8,  # Bits per subquantizer
        )

        if self.use_gpu:
            # Configure GPU resources for RTX 4080
            self.gpu_resources = faiss.StandardGpuResources()

            # Set memory fraction (leave some VRAM for other operations)
            mem_fraction = self.settings.gpu_memory_fraction
            total_mem = torch.cuda.get_device_properties(0).total_memory
            self.gpu_resources.setTempMemory(int(total_mem * mem_fraction * 0.3))

            # Move index to GPU
            index = faiss.index_cpu_to_gpu(
                self.gpu_resources,
                self.settings.cuda_device,
                index
            )
            logger.info(f"FAISS index moved to GPU {self.settings.cuda_device}")

        return index

    def initialize_index(self, force_new: bool = False) -> None:
        """Initialize or load existing FAISS index."""
        index_path = Path(self.settings.faiss_index_path)

        if index_path.exists() and not force_new:
            self.load_index()
        else:
            self.index = self._create_index()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Created new FAISS index")

    def add_embeddings(
        self,
        texts: list[str],
        metadata: Optional[list[dict]] = None
    ) -> None:
        """
        Add text embeddings to the index.

        Args:
            texts: List of text strings to embed and add
            metadata: Optional metadata for each text
        """
        if self.index is None:
            self.initialize_index()

        # Generate embeddings with GPU acceleration
        with torch.cuda.amp.autocast(enabled=self.settings.enable_mixed_precision):
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=self.settings.max_batch_size
            )

        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        # Train index if not already trained
        if not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            logger.info("Index training complete")

        # Add embeddings to index
        self.index.add(embeddings)

        # Store metadata
        if metadata is None:
            metadata = [{"text": text} for text in texts]
        self.metadata.extend(metadata)

        logger.info(f"Added {len(texts)} embeddings to index (Total: {self.index.ntotal})")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Search for similar embeddings.

        Args:
            query: Query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results with metadata and scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized")
            return []

        top_k = top_k or self.settings.rag_top_k
        similarity_threshold = similarity_threshold or self.settings.rag_similarity_threshold

        # Generate query embedding
        with torch.cuda.amp.autocast(enabled=self.settings.enable_mixed_precision):
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )

        query_embedding = query_embedding.astype(np.float32)

        # Set search parameters
        self.index.nprobe = self.settings.faiss_nprobe

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            # Convert distance to similarity score (lower is better for L2)
            similarity = 1.0 / (1.0 + dist)

            if similarity >= similarity_threshold:
                result = {
                    "index": int(idx),
                    "similarity": float(similarity),
                    "distance": float(dist),
                    **self.metadata[idx]
                }
                results.append(result)

        logger.info(f"Found {len(results)} results for query")
        return results

    def save_index(self, path: Optional[Path] = None) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        save_path = path or Path(self.settings.faiss_index_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert GPU index to CPU before saving
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index

        # Save index
        faiss.write_index(index_cpu, str(save_path / "index.faiss"))

        # Save metadata
        import pickle
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved index to {save_path}")

    def load_index(self, path: Optional[Path] = None) -> None:
        """Load FAISS index and metadata from disk."""
        load_path = path or Path(self.settings.faiss_index_path)

        if not (load_path / "index.faiss").exists():
            logger.warning(f"No index found at {load_path}")
            self.initialize_index(force_new=True)
            return

        # Load index
        index_cpu = faiss.read_index(str(load_path / "index.faiss"))

        # Move to GPU if enabled
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_resources,
                self.settings.cuda_device,
                index_cpu
            )
        else:
            self.index = index_cpu

        # Load metadata
        import pickle
        with open(load_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"Loaded index from {load_path} ({self.index.ntotal} vectors)")

    def get_stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {"status": "uninitialized"}

        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "is_trained": self.index.is_trained,
            "gpu_enabled": self.use_gpu,
            "device": f"cuda:{self.settings.cuda_device}" if self.use_gpu else "cpu",
        }
