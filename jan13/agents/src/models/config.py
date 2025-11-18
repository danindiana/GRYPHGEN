"""Configuration models using Pydantic for type safety and validation."""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime


class ServiceStatus(str, Enum):
    """Service status enumeration."""

    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class HealthStatus(BaseModel):
    """Health status model."""

    model_config = ConfigDict(frozen=False)

    service: str
    status: ServiceStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class GPUConfig(BaseModel):
    """GPU configuration for NVIDIA RTX 4080."""

    model_config = ConfigDict(frozen=False)

    enabled: bool = True
    gpu_id: int = Field(default=0, ge=0, description="GPU device ID")
    memory_fraction: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use (0.9 = 90% of 16GB)",
    )
    num_threads: int = Field(default=8, ge=1, description="Number of GPU threads")
    cuda_version: str = "12.0+"
    compute_capability: str = "8.9"
    vram_gb: int = 16

    @field_validator("memory_fraction")
    @classmethod
    def validate_memory_fraction(cls, v: float) -> float:
        """Validate memory fraction is reasonable."""
        if v > 0.95:
            raise ValueError("Memory fraction should not exceed 0.95 to avoid OOM errors")
        return v


class OllamaConfig(BaseModel):
    """Ollama service configuration."""

    model_config = ConfigDict(frozen=False)

    host: str = "127.0.0.1"
    port: int = Field(default=11435, ge=1024, le=65535)
    models: List[str] = Field(
        default_factory=lambda: ["llama2", "mistral", "codellama"],
        description="Models to preload",
    )
    num_gpu: int = Field(default=1, ge=0, description="Number of GPUs to use")
    gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)
    context_length: int = Field(default=4096, ge=512, le=32768)
    batch_size: int = Field(default=32, ge=1, le=512)
    service_file: str = "/etc/systemd/system/ollama.service"
    auto_pull_models: bool = True

    @property
    def url(self) -> str:
        """Get full Ollama URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def env_vars(self) -> Dict[str, str]:
        """Get environment variables for Ollama."""
        return {
            "OLLAMA_HOST": f"{self.host}:{self.port}",
            "OLLAMA_NUM_GPU": str(self.num_gpu),
            "OLLAMA_GPU_MEMORY_FRACTION": str(self.gpu_memory_fraction),
        }


class NginxConfig(BaseModel):
    """Nginx reverse proxy configuration."""

    model_config = ConfigDict(frozen=False)

    host: str = "0.0.0.0"
    port: int = Field(default=11434, ge=1024, le=65535)
    upstream_host: str = "127.0.0.1"
    upstream_port: int = 11435
    server_name: Optional[str] = None
    config_file: str = "/etc/nginx/sites-available/ollama"
    enabled_file: str = "/etc/nginx/sites-enabled/ollama"
    ssl_enabled: bool = False
    ssl_certificate: Optional[str] = None
    ssl_key: Optional[str] = None
    proxy_timeout: int = Field(default=300, ge=30, description="Proxy timeout in seconds")
    client_max_body_size: str = "100M"

    @property
    def upstream_url(self) -> str:
        """Get upstream URL."""
        return f"http://{self.upstream_host}:{self.upstream_port}"

    @property
    def listen_address(self) -> str:
        """Get listen address."""
        return f"{self.host}:{self.port}"


class AgentConfig(BaseModel):
    """Base agent configuration."""

    model_config = ConfigDict(frozen=False)

    name: str
    enabled: bool = True
    auto_recover: bool = True
    health_check_interval: int = Field(default=30, ge=5, description="Health check interval in seconds")
    max_retries: int = Field(default=4, ge=1, le=10)
    retry_delay: float = Field(default=2.0, ge=0.5, description="Initial retry delay in seconds")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class DeploymentConfig(BaseModel):
    """Complete deployment configuration."""

    model_config = ConfigDict(frozen=False)

    agent: AgentConfig = Field(default_factory=lambda: AgentConfig(name="infrastructure-agent"))
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    nginx: NginxConfig = Field(default_factory=NginxConfig)

    prometheus_enabled: bool = True
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)

    metrics_enabled: bool = True
    structured_logging: bool = True

    def validate_ports(self) -> List[str]:
        """Validate port configuration and return any errors."""
        errors = []

        if self.ollama.port == self.nginx.port:
            errors.append(
                f"Ollama port ({self.ollama.port}) conflicts with Nginx port ({self.nginx.port})"
            )

        if self.nginx.upstream_port != self.ollama.port:
            errors.append(
                f"Nginx upstream port ({self.nginx.upstream_port}) "
                f"should match Ollama port ({self.ollama.port})"
            )

        return errors


class MetricsSnapshot(BaseModel):
    """Snapshot of system metrics."""

    model_config = ConfigDict(frozen=False)

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    gpu_temperature: Optional[float] = None


class DeploymentStatus(BaseModel):
    """Status of infrastructure deployment."""

    model_config = ConfigDict(frozen=False)

    ollama: HealthStatus
    nginx: HealthStatus
    overall_status: ServiceStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: Optional[MetricsSnapshot] = None
    errors: List[str] = Field(default_factory=list)
