"""
KV (Key-Value) cache management for CAG.

Handles precomputation, storage, and retrieval of KV caches for efficient
inference without retrieval.
"""

from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import pickle
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class KVCache:
    """
    Represents a KV cache for a document or set of documents.

    The KV cache stores precomputed key-value pairs from the model's
    attention layers, allowing faster inference.
    """
    document_ids: List[int]
    past_key_values: Any  # Tuple of tensors from transformer
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_ids": self.document_ids,
            "past_key_values": self.past_key_values,
            "metadata": self.metadata
        }


class KVCacheStore:
    """
    Manages storage and retrieval of KV caches.

    Provides methods to precompute, save, load, and manage KV caches
    for documents.
    """

    def __init__(
        self,
        cache_dir: Path,
        model_name: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize KV cache store.

        Args:
            cache_dir: Directory to store cache files
            model_name: Model name for loading tokenizer/model
            device: Device to run computations on

        Example:
            >>> store = KVCacheStore(Path("./cache"), model_name="llama3")
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Model and tokenizer (loaded on demand)
        self._tokenizer = None
        self._model = None

        logger.info(f"KVCacheStore initialized at {cache_dir}")

    def _load_model(self) -> Tuple[Any, Any]:
        """
        Load model and tokenizer on demand.

        Returns:
            (tokenizer, model) tuple
        """
        if self._tokenizer is None or self._model is None:
            if self.model_name is None:
                raise ValueError("model_name must be set to load model")

            logger.info(f"Loading model: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )

            # Set padding token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info("Model loaded successfully")

        return self._tokenizer, self._model

    def compute_kv_cache(
        self,
        documents: List[str],
        document_ids: Optional[List[int]] = None,
        max_length: int = 8192
    ) -> KVCache:
        """
        Compute KV cache for documents.

        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
            max_length: Maximum context length

        Returns:
            KVCache object

        Example:
            >>> docs = ["Document 1 content", "Document 2 content"]
            >>> cache = store.compute_kv_cache(docs)
        """
        logger.info(f"Computing KV cache for {len(documents)} documents")

        tokenizer, model = self._load_model()

        # Concatenate documents
        combined_text = "\n\n".join(documents)

        # Tokenize
        inputs = tokenizer(
            combined_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Compute KV cache
        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=True,
                return_dict=True
            )

        past_key_values = outputs.past_key_values

        # Create KVCache object
        if document_ids is None:
            document_ids = list(range(len(documents)))

        metadata = {
            "model_name": self.model_name,
            "num_documents": len(documents),
            "total_tokens": inputs["input_ids"].shape[1],
            "device": str(self.device)
        }

        kv_cache = KVCache(
            document_ids=document_ids,
            past_key_values=past_key_values,
            metadata=metadata
        )

        logger.info(f"KV cache computed: {metadata['total_tokens']} tokens")

        return kv_cache

    def save_cache(
        self,
        kv_cache: KVCache,
        cache_name: str = "default"
    ) -> Path:
        """
        Save KV cache to file.

        Args:
            kv_cache: KVCache object to save
            cache_name: Name for the cache file

        Returns:
            Path to saved cache file

        Example:
            >>> store.save_cache(cache, "my_cache")
        """
        cache_file = self.cache_dir / f"{cache_name}.pkl"

        logger.info(f"Saving KV cache to {cache_file}")

        with open(cache_file, "wb") as f:
            pickle.dump(kv_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"KV cache saved successfully ({cache_file.stat().st_size / 1e6:.2f} MB)")

        return cache_file

    def load_cache(
        self,
        cache_name: str = "default"
    ) -> Optional[KVCache]:
        """
        Load KV cache from file.

        Args:
            cache_name: Name of the cache file

        Returns:
            KVCache object or None if not found

        Example:
            >>> cache = store.load_cache("my_cache")
        """
        cache_file = self.cache_dir / f"{cache_name}.pkl"

        if not cache_file.exists():
            logger.warning(f"Cache file not found: {cache_file}")
            return None

        logger.info(f"Loading KV cache from {cache_file}")

        try:
            with open(cache_file, "rb") as f:
                kv_cache = pickle.load(f)

            logger.info("KV cache loaded successfully")

            return kv_cache

        except Exception as e:
            logger.error(f"Failed to load KV cache: {e}")
            return None

    def delete_cache(self, cache_name: str = "default") -> bool:
        """
        Delete a cache file.

        Args:
            cache_name: Name of the cache file

        Returns:
            True if deleted, False if not found
        """
        cache_file = self.cache_dir / f"{cache_name}.pkl"

        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"Deleted cache: {cache_file}")
            return True
        else:
            logger.warning(f"Cache not found: {cache_file}")
            return False

    def list_caches(self) -> List[str]:
        """
        List all available caches.

        Returns:
            List of cache names
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_names = [f.stem for f in cache_files]

        logger.debug(f"Found {len(cache_names)} caches")

        return cache_names

    def get_cache_info(self, cache_name: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get information about a cache.

        Args:
            cache_name: Name of the cache file

        Returns:
            Dictionary with cache information or None if not found
        """
        cache_file = self.cache_dir / f"{cache_name}.pkl"

        if not cache_file.exists():
            return None

        kv_cache = self.load_cache(cache_name)

        if kv_cache is None:
            return None

        return {
            "cache_name": cache_name,
            "file_size_mb": cache_file.stat().st_size / 1e6,
            "document_ids": kv_cache.document_ids,
            "metadata": kv_cache.metadata
        }


class CacheManager:
    """
    High-level manager for KV caches.

    Coordinates cache computation, storage, and retrieval with database.
    """

    def __init__(
        self,
        cache_store: KVCacheStore,
        auto_update: bool = False
    ):
        """
        Initialize cache manager.

        Args:
            cache_store: KVCacheStore instance
            auto_update: Automatically update cache when documents change
        """
        self.cache_store = cache_store
        self.auto_update = auto_update

        # Active caches (in memory)
        self._active_caches: Dict[str, KVCache] = {}

        logger.info("CacheManager initialized")

    def preload_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[int]] = None,
        cache_name: str = "default",
        save: bool = True
    ) -> KVCache:
        """
        Preload documents and create KV cache.

        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
            cache_name: Name for the cache
            save: Whether to save cache to disk

        Returns:
            KVCache object

        Example:
            >>> manager = CacheManager(store)
            >>> docs = ["Doc 1", "Doc 2"]
            >>> cache = manager.preload_documents(docs)
        """
        logger.info(f"Preloading {len(documents)} documents into cache: {cache_name}")

        # Compute cache
        kv_cache = self.cache_store.compute_kv_cache(
            documents=documents,
            document_ids=document_ids
        )

        # Save to disk if requested
        if save:
            self.cache_store.save_cache(kv_cache, cache_name)

        # Keep in memory
        self._active_caches[cache_name] = kv_cache

        logger.info(f"Documents preloaded successfully into cache: {cache_name}")

        return kv_cache

    def get_cache(
        self,
        cache_name: str = "default",
        load_if_missing: bool = True
    ) -> Optional[KVCache]:
        """
        Get a cache (from memory or disk).

        Args:
            cache_name: Name of the cache
            load_if_missing: Load from disk if not in memory

        Returns:
            KVCache object or None
        """
        # Check if in memory
        if cache_name in self._active_caches:
            return self._active_caches[cache_name]

        # Load from disk if requested
        if load_if_missing:
            kv_cache = self.cache_store.load_cache(cache_name)
            if kv_cache is not None:
                self._active_caches[cache_name] = kv_cache
            return kv_cache

        return None

    def unload_cache(self, cache_name: str = "default") -> bool:
        """
        Unload a cache from memory.

        Args:
            cache_name: Name of the cache

        Returns:
            True if unloaded, False if not found
        """
        if cache_name in self._active_caches:
            del self._active_caches[cache_name]
            logger.debug(f"Unloaded cache from memory: {cache_name}")
            return True

        return False

    def clear_all_caches(self) -> None:
        """Clear all caches from memory."""
        self._active_caches.clear()
        logger.info("All caches cleared from memory")
