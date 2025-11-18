"""Tests for common utilities."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from common.config import Config, GPUConfig, SimGRAGConfig, CAGConfig
from common.utils import chunk_text, cosine_similarity, euclidean_distance


def test_gpu_config_defaults():
    """Test GPU config default values."""
    config = GPUConfig()

    assert config.enabled is True
    assert config.device_id == 0
    assert config.memory_fraction == 0.9
    assert config.mixed_precision is True


def test_simgrag_config():
    """Test SimGRAG configuration."""
    config = SimGRAGConfig(
        top_k=5,
        max_pattern_size=15,
        llm_model="llama3"
    )

    assert config.top_k == 5
    assert config.max_pattern_size == 15
    assert config.llm_model == "llama3"


def test_cag_config():
    """Test CAG configuration."""
    config = CAGConfig(
        db_path="test.db",
        cache_path="test.pkl",
        max_context_length=4096
    )

    assert config.db_path == "test.db"
    assert config.cache_path == "test.pkl"
    assert config.max_context_length == 4096


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "word1 word2 word3 word4 word5"
    chunks = chunk_text(text, chunk_size=15, chunk_overlap=5)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 0


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Create sample embeddings
    emb1 = np.random.randn(10, 128)
    emb2 = np.random.randn(5, 128)

    sim = cosine_similarity(emb1, emb2)

    assert sim.shape == (10, 5)
    assert np.all(sim >= -1.0) and np.all(sim <= 1.0)


def test_euclidean_distance():
    """Test Euclidean distance calculation."""
    emb1 = np.random.randn(10, 128)
    emb2 = np.random.randn(5, 128)

    dist = euclidean_distance(emb1, emb2)

    assert dist.shape == (10, 5)
    assert np.all(dist >= 0)


def test_config_memory_fraction_validation():
    """Test GPU config memory fraction validation."""
    # Valid value
    config = GPUConfig(memory_fraction=0.8)
    assert config.memory_fraction == 0.8

    # Invalid value should raise error
    with pytest.raises(ValueError):
        GPUConfig(memory_fraction=1.5)


def test_full_config():
    """Test full configuration."""
    config = Config(
        gpu=GPUConfig(enabled=True, device_id=0),
        simgrag=SimGRAGConfig(top_k=3),
        cag=CAGConfig(db_path="test.db")
    )

    assert config.gpu.enabled is True
    assert config.simgrag.top_k == 3
    assert config.cag.db_path == "test.db"
