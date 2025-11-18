"""Configuration management for agentic services."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service Configuration
    service_name: str = Field(default="agentic-service")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_reload: bool = Field(default=False)

    # Database Configuration
    database_url: str = Field(
        default="postgresql://agentic:agentic_password@localhost:5432/agentic_db"
    )
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)

    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:29092")
    kafka_topic_code_generation: str = Field(default="code-generation")
    kafka_topic_testing: str = Field(default="automated-testing")
    kafka_topic_documentation: str = Field(default="documentation")
    kafka_topic_project_mgmt: str = Field(default="project-management")
    kafka_topic_collaboration: str = Field(default="collaboration")
    kafka_topic_self_improvement: str = Field(default="self-improvement")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379")
    redis_password: Optional[str] = None

    # MinIO/S3 Configuration
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(default="agentic-artifacts")
    minio_secure: bool = Field(default=False)

    # AI Model Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    code_gen_model: str = Field(default="gpt-4-turbo")
    code_gen_temperature: float = Field(default=0.7)
    code_gen_max_tokens: int = Field(default=4096)

    # GPU Configuration
    cuda_visible_devices: str = Field(default="0")
    gpu_memory_limit: int = Field(default=14)  # GB
    use_gpu: bool = Field(default=True)
    mixed_precision: bool = Field(default=True)

    # Security
    jwt_secret_key: str = Field(default="changeme-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=60)

    # Performance
    max_requests: int = Field(default=1000)
    timeout: int = Field(default=30)

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
