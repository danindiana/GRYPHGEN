"""
Common utility functions for GRYPHGEN RAG.

Provides logging setup, model loading, embedding computation, and other
shared functionality.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "100 MB"
) -> None:
    """
    Setup logging with loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
        rotation: Log file rotation size (e.g., "100 MB", "1 GB")

    Example:
        >>> setup_logging(level="DEBUG", log_file=Path("rag.log"))
    """
    # Remove default handler
    logger.remove()

    # Add console handler with formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler if specified
    if log_file is not None:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention="10 days",
            compression="zip"
        )

    logger.info(f"Logging initialized at {level} level")


def load_model(
    model_name: str,
    model_type: str = "embedding",
    device: Optional[torch.device] = None,
    use_gpu: bool = True
) -> Any:
    """
    Load a model (embedding or LLM).

    Args:
        model_name: Name or path of the model
        model_type: Type of model ("embedding" or "llm")
        device: Device to load model on (if None, auto-detect)
        use_gpu: Whether to use GPU if available

    Returns:
        Loaded model

    Example:
        >>> model = load_model("sentence-transformers/all-mpnet-base-v2", "embedding")
    """
    if device is None:
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {model_type} model: {model_name}")

    try:
        if model_type == "embedding":
            model = SentenceTransformer(model_name, device=str(device))
            logger.info(f"Embedding model loaded on {device}")
            return model

        elif model_type == "llm":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None
            )
            logger.info(f"LLM model loaded on {device}")
            return {"model": model, "tokenizer": tokenizer}

        elif model_type == "encoder":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            model.to(device)
            logger.info(f"Encoder model loaded on {device}")
            return {"model": model, "tokenizer": tokenizer}

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def get_embeddings(
    texts: Union[str, List[str]],
    model: SentenceTransformer,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False
) -> np.ndarray:
    """
    Compute embeddings for texts using SentenceTransformer.

    Args:
        texts: Single text or list of texts
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        normalize: Whether to L2-normalize embeddings
        show_progress: Show progress bar

    Returns:
        NumPy array of embeddings

    Example:
        >>> model = load_model("sentence-transformers/all-mpnet-base-v2")
        >>> embeddings = get_embeddings(["Hello world", "How are you?"], model)
        >>> print(embeddings.shape)
        (2, 768)
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )

    return embeddings


def cosine_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (M, D)

    Returns:
        Similarity matrix (N, M)

    Example:
        >>> emb1 = np.random.randn(10, 768)
        >>> emb2 = np.random.randn(5, 768)
        >>> sim = cosine_similarity(emb1, emb2)
        >>> print(sim.shape)
        (10, 5)
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    embeddings1_norm = embeddings1 / (norm1 + 1e-8)
    embeddings2_norm = embeddings2 / (norm2 + 1e-8)

    # Compute cosine similarity
    similarity = np.dot(embeddings1_norm, embeddings2_norm.T)

    return similarity


def euclidean_distance(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance between two sets of embeddings.

    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (M, D)

    Returns:
        Distance matrix (N, M)

    Example:
        >>> emb1 = np.random.randn(10, 768)
        >>> emb2 = np.random.randn(5, 768)
        >>> dist = euclidean_distance(emb1, emb2)
        >>> print(dist.shape)
        (10, 5)
    """
    # Expand dimensions for broadcasting
    # (N, 1, D) - (1, M, D) = (N, M, D)
    diff = embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :]

    # Compute Euclidean distance
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    return distance


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = " "
) -> List[str]:
    """
    Split text into chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (in characters)
        chunk_overlap: Overlap between chunks
        separator: Separator for splitting (default: space)

    Returns:
        List of text chunks

    Example:
        >>> text = "This is a long text " * 100
        >>> chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        >>> print(len(chunks))
    """
    if not text:
        return []

    chunks = []
    words = text.split(separator)
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + len(separator)

        if current_length + word_length > chunk_size and current_chunk:
            # Create chunk
            chunk = separator.join(current_chunk)
            chunks.append(chunk)

            # Calculate overlap
            overlap_size = 0
            overlap_words = []

            for w in reversed(current_chunk):
                overlap_size += len(w) + len(separator)
                if overlap_size >= chunk_overlap:
                    break
                overlap_words.insert(0, w)

            # Start new chunk with overlap
            current_chunk = overlap_words
            current_length = sum(len(w) + len(separator) for w in overlap_words)

        current_chunk.append(word)
        current_length += word_length

    # Add last chunk
    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def save_embeddings(
    embeddings: np.ndarray,
    file_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save embeddings to file with optional metadata.

    Args:
        embeddings: NumPy array of embeddings
        file_path: Path to save file (.npz)
        metadata: Optional metadata dictionary

    Example:
        >>> embeddings = np.random.randn(100, 768)
        >>> save_embeddings(embeddings, Path("embeddings.npz"), {"model": "mpnet"})
    """
    if metadata is None:
        np.savez_compressed(file_path, embeddings=embeddings)
    else:
        np.savez_compressed(file_path, embeddings=embeddings, metadata=metadata)

    logger.info(f"Saved embeddings to {file_path}")


def load_embeddings(file_path: Path) -> Dict[str, Any]:
    """
    Load embeddings from file.

    Args:
        file_path: Path to embeddings file (.npz)

    Returns:
        Dictionary with 'embeddings' and optional 'metadata'

    Example:
        >>> data = load_embeddings(Path("embeddings.npz"))
        >>> print(data['embeddings'].shape)
    """
    data = np.load(file_path, allow_pickle=True)
    result = {"embeddings": data["embeddings"]}

    if "metadata" in data:
        result["metadata"] = data["metadata"].item()

    logger.info(f"Loaded embeddings from {file_path}")
    return result


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")
