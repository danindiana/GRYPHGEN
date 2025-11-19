"""Data models for ShellGenie using Pydantic."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ModelBackend(str, Enum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    OPENAI_COMPATIBLE = "openai_compatible"


class SecurityLevel(str, Enum):
    """Command execution security levels."""

    STRICT = "strict"  # Only allow whitelisted commands
    MODERATE = "moderate"  # Block dangerous commands
    PERMISSIVE = "permissive"  # Warn but allow most commands
    DISABLED = "disabled"  # No security checks (dangerous!)


class ModelConfig(BaseModel):
    """Configuration for the LLM backend."""

    backend: ModelBackend = Field(
        default=ModelBackend.OLLAMA, description="LLM backend to use"
    )
    model_name: str = Field(
        default="llama3.2", description="Model name or path"
    )
    model_path: Optional[str] = Field(
        default=None, description="Path to model file (for llama.cpp)"
    )
    api_base: str = Field(
        default="http://localhost:11434", description="API base URL for Ollama/OpenAI-compatible"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key if required"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: int = Field(
        default=512, ge=1, le=32768, description="Maximum tokens to generate"
    )
    context_length: int = Field(
        default=8192, ge=512, le=131072, description="Context window size"
    )
    gpu_layers: int = Field(
        default=-1, description="Number of layers to offload to GPU (-1 = all)"
    )
    threads: int = Field(
        default=8, ge=1, le=128, description="Number of CPU threads"
    )
    batch_size: int = Field(
        default=512, ge=1, le=2048, description="Batch size for prompt processing"
    )

    class Config:
        use_enum_values = True


class CommandRequest(BaseModel):
    """Request to generate a bash command from natural language."""

    prompt: str = Field(..., min_length=1, description="Natural language prompt")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context (cwd, env vars, etc.)"
    )
    history: Optional[List[str]] = Field(
        default=None, description="Command history for context"
    )
    require_confirmation: bool = Field(
        default=True, description="Require user confirmation before execution"
    )


class CommandResponse(BaseModel):
    """Response containing the generated bash command."""

    command: str = Field(..., description="Generated bash command")
    explanation: Optional[str] = Field(
        default=None, description="Explanation of what the command does"
    )
    risk_level: str = Field(
        default="low", description="Risk assessment: low, medium, high, critical"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Security warnings"
    )
    alternatives: List[str] = Field(
        default_factory=list, description="Alternative safer commands"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Model confidence score"
    )


class ExecutionResult(BaseModel):
    """Result of command execution."""

    command: str = Field(..., description="Executed command")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    return_code: int = Field(..., description="Command return code")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    success: bool = Field(..., description="Whether execution was successful")


class SystemInfo(BaseModel):
    """System information for context."""

    os: str
    kernel: str
    shell: str
    cwd: str
    user: str
    hostname: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory: Optional[int] = None
    cuda_version: Optional[str] = None


class AppConfig(BaseModel):
    """Application configuration."""

    security_level: SecurityLevel = Field(
        default=SecurityLevel.MODERATE, description="Security level for command execution"
    )
    auto_execute: bool = Field(
        default=False, description="Auto-execute commands without confirmation"
    )
    log_level: str = Field(
        default="INFO", description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None, description="Log file path"
    )
    history_file: str = Field(
        default="~/.shellgenie_history", description="Command history file"
    )
    max_history: int = Field(
        default=1000, ge=0, description="Maximum history entries"
    )
    enable_telemetry: bool = Field(
        default=False, description="Enable anonymous usage telemetry"
    )
    timeout: int = Field(
        default=300, ge=1, description="Command execution timeout in seconds"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()

    class Config:
        use_enum_values = True
