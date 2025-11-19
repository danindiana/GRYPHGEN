"""GPU optimization utilities for NVIDIA RTX 4080."""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information."""

    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float
    temperature_celsius: int
    power_usage_watts: float
    compute_capability: tuple


class GPUManager:
    """
    GPU Manager for NVIDIA RTX 4080 optimization.

    Provides utilities for GPU monitoring, memory management,
    and performance optimization specific to RTX 4080.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize GPU Manager.

        Args:
            device_id: CUDA device ID (default: 0)
        """
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.handle = None
        else:
            self.handle = None

        # RTX 4080 has compute capability 8.9 (Ada Lovelace)
        self._setup_rtx4080_optimizations()

    def _setup_rtx4080_optimizations(self) -> None:
        """Configure PyTorch for RTX 4080 optimizations."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU optimizations")
            return

        # Enable TensorFloat-32 (TF32) for Ampere/Ada GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN auto-tuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True

        # Enable deterministic algorithms for reproducibility (optional)
        # torch.backends.cudnn.deterministic = True

        # Set memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        logger.info(f"Configured optimizations for RTX 4080 (Device {self.device_id})")
        logger.info(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        logger.info(f"cuDNN benchmark enabled: {torch.backends.cudnn.benchmark}")

    def get_info(self) -> GPUInfo:
        """
        Get current GPU information.

        Returns:
            GPUInfo object with current GPU stats
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        name = torch.cuda.get_device_name(self.device_id)
        props = torch.cuda.get_device_properties(self.device_id)
        total_memory = props.total_memory / (1024 ** 3)  # GB

        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024 ** 3)
        memory_free = total_memory - memory_reserved

        # Get utilization and temperature if pynvml is available
        utilization = 0.0
        temperature = 0
        power_usage = 0.0

        if self.handle:
            try:
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                utilization = util_info.gpu
                temperature = pynvml.nvmlDeviceGetTemperature(
                    self.handle, pynvml.NVML_TEMPERATURE_GPU
                )
                power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Watts
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")

        return GPUInfo(
            name=name,
            total_memory_gb=total_memory,
            free_memory_gb=memory_free,
            used_memory_gb=memory_allocated,
            utilization_percent=utilization,
            temperature_celsius=temperature,
            power_usage_watts=power_usage,
            compute_capability=(props.major, props.minor),
        )

    def optimize_memory(self, target_memory_gb: float = 14.0) -> None:
        """
        Optimize GPU memory usage for RTX 4080 (16GB total).

        Reserves memory and configures allocator for optimal performance.

        Args:
            target_memory_gb: Target memory usage in GB (default: 14GB, leaving 2GB for system)
        """
        if not torch.cuda.is_available():
            return

        # Clear cache
        torch.cuda.empty_cache()

        # Set memory fraction
        memory_fraction = target_memory_gb / 16.0  # RTX 4080 has 16GB
        torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device_id)

        logger.info(f"Set memory fraction to {memory_fraction:.2f} ({target_memory_gb}GB)")

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get detailed memory summary.

        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device_id) / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved(self.device_id) / (1024 ** 3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 3),
            "max_reserved_gb": torch.cuda.max_memory_reserved(self.device_id) / (1024 ** 3),
        }

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)

    def __del__(self):
        """Cleanup on destruction."""
        if PYNVML_AVAILABLE and self.handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def optimize_for_rtx4080(
    mixed_precision: bool = True,
    memory_limit_gb: float = 14.0,
    device_id: int = 0,
) -> GPUManager:
    """
    Configure optimal settings for NVIDIA RTX 4080.

    Args:
        mixed_precision: Enable mixed precision training (bfloat16/float16)
        memory_limit_gb: Memory limit in GB
        device_id: CUDA device ID

    Returns:
        Configured GPUManager instance
    """
    manager = GPUManager(device_id=device_id)

    # Optimize memory
    manager.optimize_memory(target_memory_gb=memory_limit_gb)

    # Enable automatic mixed precision if requested
    if mixed_precision:
        logger.info("Mixed precision training enabled (recommended for RTX 4080)")
        # This should be used with torch.cuda.amp.autocast() in training loops

    # Log GPU info
    try:
        info = manager.get_info()
        logger.info(f"GPU: {info.name}")
        logger.info(f"Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")
        logger.info(f"Total Memory: {info.total_memory_gb:.2f} GB")
        logger.info(f"Available Memory: {info.free_memory_gb:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not retrieve GPU info: {e}")

    return manager


def get_gpu_info() -> Optional[GPUInfo]:
    """
    Get GPU information for the default device.

    Returns:
        GPUInfo object or None if CUDA not available
    """
    if not torch.cuda.is_available():
        return None

    try:
        manager = GPUManager(device_id=0)
        return manager.get_info()
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return None


def get_optimal_batch_size(
    model_size_gb: float,
    available_memory_gb: float = 14.0,
    data_memory_multiplier: float = 2.0,
) -> int:
    """
    Calculate optimal batch size for RTX 4080.

    Args:
        model_size_gb: Estimated model size in GB
        available_memory_gb: Available GPU memory in GB
        data_memory_multiplier: Multiplier for data memory overhead

    Returns:
        Recommended batch size
    """
    # Reserve memory for model and gradients (2x model size)
    model_memory = model_size_gb * 2

    # Available memory for batches
    batch_memory = available_memory_gb - model_memory

    if batch_memory <= 0:
        logger.warning("Insufficient memory for model, returning batch size of 1")
        return 1

    # Estimate memory per sample (rough estimate)
    # This is a heuristic and should be tuned per use case
    memory_per_sample_gb = model_size_gb * 0.1 * data_memory_multiplier

    batch_size = int(batch_memory / memory_per_sample_gb)

    return max(1, batch_size)


def enable_cuda_graphs(enabled: bool = True) -> None:
    """
    Enable CUDA graphs for performance optimization.

    CUDA graphs can significantly reduce CPU overhead for repetitive operations.

    Args:
        enabled: Whether to enable CUDA graphs
    """
    # CUDA graphs are enabled per-use in PyTorch
    # This is a placeholder for future implementation
    logger.info(f"CUDA graphs {'enabled' if enabled else 'disabled'}")
