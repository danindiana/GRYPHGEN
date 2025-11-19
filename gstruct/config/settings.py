"""Configuration management for GRYPHGEN framework."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class GPUConfig(BaseModel):
    """GPU configuration settings."""
    enabled: bool = True
    device_id: int = 0
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    allow_growth: bool = True


class ZeroMQConfig(BaseModel):
    """ZeroMQ messaging configuration."""
    orchestrator_port: int = 5555
    scheduler_port: int = 5556
    worker_port: int = 5557
    pub_port: int = 5558
    sub_port: int = 5559


class RedisConfig(BaseModel):
    """Redis configuration for state management."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM configuration for orchestration."""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = Field(default=None, exclude=True)
    temperature: float = 0.7
    max_tokens: int = 2048


class ResourceConfig(BaseModel):
    """Resource management configuration."""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0
    min_free_disk_gb: float = 10.0


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    enabled: bool = True
    prometheus_port: int = 9090
    metrics_interval: int = 30  # seconds
    log_level: str = "INFO"


class GryphgenSettings(BaseSettings):
    """Main GRYPHGEN configuration."""

    # Application settings
    app_name: str = "GRYPHGEN"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Component configurations
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    zeromq: ZeroMQConfig = Field(default_factory=ZeroMQConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Paths
    data_dir: Path = Path("./data")
    logs_dir: Path = Path("./logs")
    cache_dir: Path = Path("./cache")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "GRYPHGEN_"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or environment.

    Args:
        config_path: Optional path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    # Start with default settings
    settings = GryphgenSettings()

    # Load from YAML if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Update settings with YAML values
                    for key, value in yaml_config.items():
                        if hasattr(settings, key):
                            setattr(settings, key, value)

    # Create necessary directories
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # Convert to dict for easier access
    config_dict = settings.model_dump()

    # Add environment-specific overrides
    if os.getenv("GRYPHGEN_LLM_API_KEY"):
        config_dict["llm"]["api_key"] = os.getenv("GRYPHGEN_LLM_API_KEY")

    return config_dict


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return GryphgenSettings().model_dump()
