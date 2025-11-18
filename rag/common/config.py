"""
Configuration management for GRYPHGEN RAG.

Handles configuration loading, validation, and defaults for SimGRAG and CAG.
Optimized for NVIDIA RTX 4080 16GB.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator
import yaml
from loguru import logger


class GPUConfig(BaseModel):
    """GPU configuration optimized for RTX 4080 16GB."""

    enabled: bool = Field(default=True, description="Enable GPU acceleration")
    device_id: int = Field(default=0, description="CUDA device ID")
    memory_fraction: float = Field(
        default=0.9,
        description="Fraction of GPU memory to use (0.9 = 14.4GB for RTX 4080)"
    )
    mixed_precision: bool = Field(
        default=True,
        description="Use mixed precision (FP16/BF16) for faster training"
    )
    torch_compile: bool = Field(
        default=True,
        description="Use torch.compile for optimization"
    )
    flash_attention: bool = Field(
        default=True,
        description="Use Flash Attention 2 for efficient attention computation"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size optimized for RTX 4080"
    )
    num_workers: int = Field(
        default=8,
        description="Number of data loading workers"
    )

    @field_validator('memory_fraction')
    @classmethod
    def validate_memory_fraction(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("memory_fraction must be between 0 and 1")
        return v


class SimGRAGConfig(BaseModel):
    """Configuration for SimGRAG."""

    top_k: int = Field(default=3, description="Number of top subgraphs to retrieve")
    max_pattern_size: int = Field(default=10, description="Maximum pattern graph size")
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Model for entity/relation embeddings"
    )
    llm_model: str = Field(
        default="llama3",
        description="LLM model for pattern generation"
    )
    kg_path: Optional[str] = Field(
        default=None,
        description="Path to knowledge graph file"
    )
    use_vector_cache: bool = Field(
        default=True,
        description="Cache embeddings for faster retrieval"
    )


class CAGConfig(BaseModel):
    """Configuration for Cache-Augmented Generation."""

    db_path: str = Field(
        default="documents.db",
        description="Path to SQLite database"
    )
    cache_path: str = Field(
        default="kv_cache.pkl",
        description="Path to KV cache file"
    )
    llm_model: str = Field(
        default="llama3",
        description="LLM model for generation"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Model for document embeddings"
    )
    max_context_length: int = Field(
        default=8192,
        description="Maximum context window length"
    )
    chunk_size: int = Field(
        default=512,
        description="Document chunk size for processing"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks"
    )
    use_kv_cache: bool = Field(
        default=True,
        description="Enable KV cache for inference"
    )


class Config(BaseModel):
    """Main configuration for GRYPHGEN RAG."""

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    simgrag: SimGRAGConfig = Field(default_factory=SimGRAGConfig)
    cag: CAGConfig = Field(default_factory=CAGConfig)

    log_level: str = Field(default="INFO", description="Logging level")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to configuration YAML file. If None, uses defaults.

    Returns:
        Config object with loaded or default settings.

    Example:
        >>> config = load_config(Path("config.yaml"))
        >>> print(config.gpu.device_id)
        0
    """
    if config_path is None or not config_path.exists():
        logger.info("Using default configuration")
        return Config()

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = Config(**config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Falling back to default configuration")
        return Config()


def save_config(config: Config, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path where to save the configuration

    Example:
        >>> config = Config()
        >>> save_config(config, Path("config.yaml"))
    """
    try:
        config_dict = config.model_dump()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise
