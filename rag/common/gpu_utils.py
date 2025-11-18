"""
GPU utilities optimized for NVIDIA RTX 4080 16GB.

Provides GPU management, memory optimization, and performance tuning
specifically for the RTX 4080 with Ada Lovelace architecture.
"""

import os
from typing import Optional, Dict, Any
import torch
from loguru import logger

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring will be limited.")


class GPUManager:
    """
    Manages GPU resources and optimization for RTX 4080.

    RTX 4080 Specifications:
    - CUDA Cores: 9728
    - Tensor Cores: 304 (4th gen)
    - Memory: 16GB GDDR6X
    - Memory Bandwidth: 716.8 GB/s
    - Compute Capability: 8.9 (Ada Lovelace)
    - TDP: 320W
    """

    def __init__(self, device_id: int = 0, memory_fraction: float = 0.9):
        """
        Initialize GPU manager.

        Args:
            device_id: CUDA device ID (default: 0)
            memory_fraction: Fraction of GPU memory to use (default: 0.9 = 14.4GB)
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = self._setup_device()

        if NVML_AVAILABLE:
            pynvml.nvmlInit()

    def _setup_device(self) -> torch.device:
        """Setup and configure CUDA device."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU.")
            return torch.device("cpu")

        device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(device)

        # Set memory fraction for RTX 4080 (16GB)
        torch.cuda.set_per_process_memory_fraction(
            self.memory_fraction,
            device=device
        )

        logger.info(f"Using device: {device}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(device)}")
        logger.info(f"Memory allocated: {self.memory_fraction * 16:.1f}GB / 16GB")

        return device

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get detailed GPU information.

        Returns:
            Dictionary with GPU specs and current utilization.
        """
        if not torch.cuda.is_available():
            return {"available": False}

        info = {
            "available": True,
            "device_name": torch.cuda.get_device_name(self.device_id),
            "device_id": self.device_id,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "compute_capability": torch.cuda.get_device_capability(self.device_id),
        }

        # Memory information
        total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device_id)
        reserved_memory = torch.cuda.memory_reserved(self.device_id)

        info.update({
            "total_memory_gb": total_memory / 1e9,
            "allocated_memory_gb": allocated_memory / 1e9,
            "reserved_memory_gb": reserved_memory / 1e9,
            "free_memory_gb": (total_memory - reserved_memory) / 1e9,
        })

        # NVML information (if available)
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

                info.update({
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "temperature_c": temperature,
                    "power_draw_w": power,
                })
            except Exception as e:
                logger.warning(f"Failed to get NVML info: {e}")

        return info

    def optimize_for_inference(self) -> None:
        """
        Optimize PyTorch settings for inference on RTX 4080.

        - Enables cuDNN autotuner
        - Enables TF32 for Tensor Cores
        - Enables BF16 for mixed precision
        """
        if not torch.cuda.is_available():
            return

        # Enable cuDNN autotuner for optimal performance
        torch.backends.cudnn.benchmark = True

        # Enable TF32 for Ampere and Ada Lovelace architectures
        # RTX 4080 has Tensor Cores that benefit from TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set optimal cuDNN settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

        logger.info("GPU optimized for inference (TF32 enabled, cuDNN autotuner enabled)")

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared")

    def log_memory_stats(self) -> None:
        """Log current GPU memory statistics."""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(self.device_id) / 1e9
        reserved = torch.cuda.memory_reserved(self.device_id) / 1e9

        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def optimize_for_rtx4080() -> GPUManager:
    """
    Apply optimal settings for RTX 4080 16GB.

    Returns:
        Configured GPUManager instance.

    RTX 4080 Optimizations:
    - Use 90% of 16GB memory (14.4GB)
    - Enable TF32 for Tensor Cores
    - Enable cuDNN autotuner
    - Use mixed precision (FP16/BF16)
    - Enable Flash Attention 2

    Example:
        >>> gpu_manager = optimize_for_rtx4080()
        >>> gpu_manager.log_memory_stats()
    """
    # Set environment variables for optimal performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Initialize GPU manager with RTX 4080 optimal settings
    gpu_manager = GPUManager(device_id=0, memory_fraction=0.9)
    gpu_manager.optimize_for_inference()

    # Log GPU information
    info = gpu_manager.get_gpu_info()
    if info["available"]:
        logger.info("RTX 4080 Optimization Applied:")
        logger.info(f"  - Compute Capability: {info['compute_capability']}")
        logger.info(f"  - Total Memory: {info['total_memory_gb']:.1f}GB")
        logger.info(f"  - Available for Use: {info['total_memory_gb'] * 0.9:.1f}GB")
        logger.info("  - TF32 Enabled for Tensor Cores")
        logger.info("  - cuDNN Autotuner Enabled")

    return gpu_manager


def get_optimal_batch_size(
    model_size_gb: float,
    sequence_length: int = 512,
    available_memory_gb: float = 14.4
) -> int:
    """
    Calculate optimal batch size for RTX 4080 based on model size.

    Args:
        model_size_gb: Model size in GB
        sequence_length: Input sequence length
        available_memory_gb: Available GPU memory (default: 14.4GB for RTX 4080)

    Returns:
        Recommended batch size

    Example:
        >>> batch_size = get_optimal_batch_size(model_size_gb=7, sequence_length=512)
        >>> print(f"Optimal batch size: {batch_size}")
    """
    # Estimate memory per sample (rough approximation)
    # Formula: memory_per_sample ≈ model_size + (seq_len * hidden_dim * 4 bytes)
    # For typical transformer: hidden_dim ≈ 4096 for 7B models

    hidden_dim = 4096 if model_size_gb > 5 else 2048
    memory_per_sample_gb = (sequence_length * hidden_dim * 4) / 1e9

    # Reserve memory for model and overhead (20%)
    usable_memory_gb = available_memory_gb - model_size_gb - (available_memory_gb * 0.2)

    batch_size = max(1, int(usable_memory_gb / memory_per_sample_gb))

    # RTX 4080 optimal batch sizes (powers of 2)
    optimal_sizes = [1, 2, 4, 8, 16, 32, 64]
    batch_size = max([s for s in optimal_sizes if s <= batch_size], default=1)

    logger.info(f"Calculated optimal batch size: {batch_size}")
    return batch_size


def enable_flash_attention() -> bool:
    """
    Enable Flash Attention 2 for efficient attention computation.

    Flash Attention 2 provides significant speedup on RTX 4080:
    - 2-4x faster than standard attention
    - 5-20x memory reduction
    - Optimized for Ada Lovelace Tensor Cores

    Returns:
        True if Flash Attention is available and enabled, False otherwise.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Check if Flash Attention is available
        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH
        ]

        logger.info("Flash Attention 2 enabled for RTX 4080")
        return True

    except ImportError:
        logger.warning(
            "Flash Attention not available. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
        return False
