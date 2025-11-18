"""
Configuration management for CALISOTA AI Engine.
Optimized for NVIDIA RTX 4080 16GB.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with GPU optimization for RTX 4080."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "CALISOTA AI Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # GPU Configuration - Optimized for RTX 4080 16GB
    cuda_device: int = 0
    cuda_visible_devices: str = "0"
    gpu_memory_fraction: float = 0.85  # Use 85% of 16GB VRAM
    max_batch_size: int = 32
    enable_mixed_precision: bool = True  # Use FP16/BF16 for efficiency
    enable_flash_attention: bool = True  # Enable Flash Attention 2

    # Model Configuration
    slow_thinker_model: str = "gpt-4"  # Large planning model
    fast_thinker_model: str = "gpt-3.5-turbo"  # Fast code generator
    actor_critic_model: str = "gpt-3.5-turbo"  # Evaluation model
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    local_llm_path: Optional[str] = None  # Optional local model path

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")

    # FAISS Vector Database
    faiss_index_path: str = "./data/faiss_index"
    faiss_dimension: int = 768  # Matches embedding model dimension
    faiss_nlist: int = 100  # Number of clusters for IVF index
    faiss_nprobe: int = 10  # Number of clusters to visit during search
    use_gpu_index: bool = True  # Use GPU for FAISS operations
    max_index_size: int = 1_000_000  # Maximum vectors in index

    # RAG Configuration
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7

    # Code Sandbox
    sandbox_timeout: int = 300  # 5 minutes
    max_concurrent_sandboxes: int = 5
    supported_languages: list[str] = ["python", "rust", "go", "cpp", "perl"]
    docker_network: str = "calisota-net"

    # Actor-Critic Training
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5

    # Human-in-the-Loop
    require_human_approval: bool = True
    approval_timeout: int = 3600  # 1 hour
    auto_approve_threshold: float = 0.95  # Confidence threshold for auto-approval

    # Monitoring & Logging
    log_level: str = "INFO"
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    sentry_dsn: Optional[str] = None

    # Redis/Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Self-Healing
    enable_self_healing: bool = True
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 2.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
