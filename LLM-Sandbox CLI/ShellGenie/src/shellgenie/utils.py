"""Utility functions for ShellGenie."""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
from loguru import logger

from shellgenie.models import ExecutionResult, SystemInfo


def get_system_info() -> SystemInfo:
    """Get system information for context.

    Returns:
        SystemInfo object with system details
    """
    gpu_available = False
    gpu_name = None
    gpu_memory = None
    cuda_version = None

    # Try to detect NVIDIA GPU
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = mem_info.total // (1024**2)  # Convert to MB
        gpu_available = True

        # Get CUDA version
        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        if cuda_version:
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            cuda_version = f"{major}.{minor}"

        pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    return SystemInfo(
        os=platform.system(),
        kernel=platform.release(),
        shell=os.environ.get("SHELL", "unknown"),
        cwd=os.getcwd(),
        user=os.environ.get("USER", "unknown"),
        hostname=platform.node(),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory=gpu_memory,
        cuda_version=cuda_version,
    )


def execute_command(
    command: str,
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> ExecutionResult:
    """Execute a bash command and capture output.

    Args:
        command: The command to execute
        timeout: Timeout in seconds
        env: Environment variables
        cwd: Working directory

    Returns:
        ExecutionResult with execution details
    """
    import time

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
        )

        execution_time = time.time() - start_time

        return ExecutionResult(
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            execution_time=execution_time,
            success=(result.returncode == 0),
        )

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return ExecutionResult(
            command=command,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            return_code=-1,
            execution_time=execution_time,
            success=False,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return ExecutionResult(
            command=command,
            stdout="",
            stderr=f"Execution error: {str(e)}",
            return_code=-1,
            execution_time=execution_time,
            success=False,
        )


def load_command_history(history_file: str, max_entries: int = 100) -> List[str]:
    """Load command history from file.

    Args:
        history_file: Path to history file
        max_entries: Maximum number of entries to load

    Returns:
        List of historical commands
    """
    history_path = Path(history_file).expanduser()

    if not history_path.exists():
        return []

    try:
        with open(history_path, "r") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-max_entries:] if line.strip()]
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")
        return []


def save_to_history(history_file: str, command: str, max_entries: int = 1000) -> None:
    """Save a command to history file.

    Args:
        history_file: Path to history file
        command: Command to save
        max_entries: Maximum number of entries to keep
    """
    history_path = Path(history_file).expanduser()

    try:
        # Create parent directory if needed
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        existing = load_command_history(str(history_path), max_entries - 1)

        # Add new command
        existing.append(command)

        # Write back
        with open(history_path, "w") as f:
            f.write("\n".join(existing) + "\n")

    except Exception as e:
        logger.warning(f"Failed to save to history: {e}")


def check_ollama_running() -> bool:
    """Check if Ollama service is running.

    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def check_gpu_available() -> Tuple[bool, Optional[str]]:
    """Check if GPU is available and get info.

    Returns:
        Tuple of (is_available, gpu_name)
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        pynvml.nvmlShutdown()
        return True, gpu_name
    except Exception:
        return False, None


def format_bytes(bytes: int) -> str:
    """Format bytes to human-readable string.

    Args:
        bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def get_gpu_info() -> Dict[str, any]:
    """Get detailed GPU information.

    Returns:
        Dictionary with GPU details
    """
    try:
        import pynvml

        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            gpus.append({
                "id": i,
                "name": name,
                "memory_total": format_bytes(mem_info.total),
                "memory_used": format_bytes(mem_info.used),
                "memory_free": format_bytes(mem_info.free),
                "temperature": f"{temperature}°C",
                "gpu_utilization": f"{utilization.gpu}%",
                "memory_utilization": f"{utilization.memory}%",
            })

        pynvml.nvmlShutdown()

        return {
            "available": True,
            "count": device_count,
            "devices": gpus,
        }

    except Exception as e:
        logger.debug(f"Failed to get GPU info: {e}")
        return {
            "available": False,
            "count": 0,
            "devices": [],
            "error": str(e),
        }


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    logger.remove()  # Remove default handler

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed.

    Returns:
        Dictionary of dependency name to availability
    """
    deps = {}

    # Check Ollama
    deps["ollama"] = check_ollama_running()

    # Check GPU libraries
    try:
        import pynvml
        deps["pynvml"] = True
    except ImportError:
        deps["pynvml"] = False

    try:
        import torch
        deps["pytorch"] = True
        deps["cuda"] = torch.cuda.is_available()
    except ImportError:
        deps["pytorch"] = False
        deps["cuda"] = False

    # Check llama-cpp-python
    try:
        import llama_cpp
        deps["llama_cpp"] = True
    except ImportError:
        deps["llama_cpp"] = False

    return deps


def create_prompt_template(
    task: str,
    context: Optional[Dict[str, any]] = None,
    system_info: Optional[SystemInfo] = None,
) -> str:
    """Create a prompt template for the LLM.

    Args:
        task: The user's task description
        context: Additional context
        system_info: System information

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        "You are ShellGenie, an AI assistant that helps users with bash commands.",
        "Your task is to convert natural language requests into safe, efficient bash commands.",
        "",
        "Guidelines:",
        "- Generate only the command, no explanations unless asked",
        "- Prefer safe, non-destructive commands",
        "- Use common flags and options",
        "- Consider the user's current context",
        "",
    ]

    if system_info:
        prompt_parts.extend([
            "System Information:",
            f"- OS: {system_info.os}",
            f"- Current directory: {system_info.cwd}",
            f"- Shell: {system_info.shell}",
            "",
        ])

    if context:
        prompt_parts.append("Context:")
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    prompt_parts.extend([
        f"User request: {task}",
        "",
        "Generate the bash command:",
    ])

    return "\n".join(prompt_parts)
