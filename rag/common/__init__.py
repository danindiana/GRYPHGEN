"""
Common utilities for GRYPHGEN RAG.

This package provides shared functionality for SimGRAG and CAG implementations,
including GPU optimization for NVIDIA RTX 4080, configuration management, and
utility functions.
"""

from .config import Config, load_config
from .gpu_utils import GPUManager, optimize_for_rtx4080
from .utils import setup_logging, load_model, get_embeddings

__version__ = "1.0.0"

__all__ = [
    "Config",
    "load_config",
    "GPUManager",
    "optimize_for_rtx4080",
    "setup_logging",
    "load_model",
    "get_embeddings",
]
