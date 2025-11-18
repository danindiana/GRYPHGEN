"""Test RAG functionality."""

import pytest
import numpy as np

from src.calisota.core.config import Settings
from src.calisota.rag.faiss_manager import FAISSManager


@pytest.fixture
def faiss_manager(test_settings: Settings) -> FAISSManager:
    """Create FAISS manager for testing."""
    manager = FAISSManager(test_settings)
    manager.initialize_index(force_new=True)
    return manager


def test_faiss_initialization(faiss_manager: FAISSManager) -> None:
    """Test FAISS manager initialization."""
    assert faiss_manager.index is not None
    stats = faiss_manager.get_stats()
    assert stats["status"] == "ready"


def test_add_and_search(faiss_manager: FAISSManager) -> None:
    """Test adding documents and searching."""
    # Add test documents
    texts = [
        "Python is a high-level programming language",
        "Rust is a systems programming language",
        "Go is designed for concurrent programming"
    ]
    faiss_manager.add_embeddings(texts)

    # Search
    results = faiss_manager.search("programming language", top_k=2)
    assert len(results) > 0
    assert results[0]["similarity"] > 0.5
