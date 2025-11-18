"""
CALISOTA AI Engine - Autonomous Software Generation System

An advanced AI-powered system combining:
- Retrieval-Augmented Generation (RAG) with FAISS
- Actor-Critic reinforcement learning ensembles
- Multi-language code generation and execution
- Self-healing deployment mechanisms
- Human-in-the-loop oversight

Optimized for NVIDIA RTX 4080 16GB
"""

__version__ = "1.0.0"
__author__ = "GRYPHGEN Team"
__license__ = "MIT"

from src.calisota.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
