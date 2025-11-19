"""GPU utility functions for GRYPHGEN framework.

Optimized for NVIDIA RTX 4080 (16GB VRAM, CUDA 12.x)
"""

import os
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import CUDA libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available. GPU acceleration disabled.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    try:
        import nvidia_smi
        NVML_AVAILABLE = True
    except ImportError:
        NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some GPU features disabled.")


def check_gpu_availability() -> bool:
    """Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise
    """
    # Check CUDA availability through different libraries
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return True
    if CUPY_AVAILABLE:
        try:
            # Try to access GPU
            _ = cp.cuda.Device(0)
            return True
        except Exception:
            pass
    if PYCUDA_AVAILABLE:
        try:
            return cuda.Device.count() > 0
        except Exception:
            pass
    return False


def get_gpu_count() -> int:
    """Get number of available GPUs.

    Returns:
        Number of GPUs
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.device_count()
    if PYCUDA_AVAILABLE:
        try:
            return cuda.Device.count()
        except Exception:
            pass
    return 0


def get_gpu_info(device_id: int = 0) -> Dict[str, any]:
    """Get detailed GPU information.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary containing GPU information
    """
    info = {
        "available": False,
        "device_id": device_id,
        "name": "Unknown",
        "compute_capability": "Unknown",
        "total_memory": 0,
        "free_memory": 0,
        "used_memory": 0,
        "memory_utilization": 0.0,
        "gpu_utilization": 0.0,
        "temperature": 0,
        "power_usage": 0,
        "power_limit": 0,
    }

    if not check_gpu_availability():
        return info

    info["available"] = True

    # Get info using PyTorch
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            info["name"] = torch.cuda.get_device_name(device_id)
            info["compute_capability"] = ".".join(
                map(str, torch.cuda.get_device_capability(device_id))
            )
            info["total_memory"] = torch.cuda.get_device_properties(device_id).total_memory
            info["free_memory"] = torch.cuda.mem_get_info(device_id)[0]
            info["used_memory"] = info["total_memory"] - info["free_memory"]
            info["memory_utilization"] = (info["used_memory"] / info["total_memory"]) * 100
        except Exception as e:
            logger.debug(f"Error getting PyTorch GPU info: {e}")

    # Get additional info using NVML
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Update name if not already set
            if info["name"] == "Unknown":
                info["name"] = pynvml.nvmlDeviceGetName(handle)

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["total_memory"] = mem_info.total
            info["free_memory"] = mem_info.free
            info["used_memory"] = mem_info.used
            info["memory_utilization"] = (mem_info.used / mem_info.total) * 100

            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info["gpu_utilization"] = util.gpu

            # Get temperature
            info["temperature"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # Get power usage
            info["power_usage"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
            info["power_limit"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug(f"Error getting NVML GPU info: {e}")

    return info


def format_gpu_memory(bytes_value: int) -> str:
    """Format GPU memory value in human-readable format.

    Args:
        bytes_value: Memory in bytes

    Returns:
        Formatted string (e.g., "15.2 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def get_optimal_batch_size(
    model_memory_mb: float,
    available_memory_mb: float,
    safety_margin: float = 0.8
) -> int:
    """Calculate optimal batch size based on available GPU memory.

    Args:
        model_memory_mb: Memory required per sample (MB)
        available_memory_mb: Available GPU memory (MB)
        safety_margin: Safety margin (0.8 = use 80% of available memory)

    Returns:
        Optimal batch size
    """
    usable_memory = available_memory_mb * safety_margin
    batch_size = int(usable_memory / model_memory_mb)
    return max(1, batch_size)


def set_gpu_device(device_id: int = 0):
    """Set the active GPU device.

    Args:
        device_id: GPU device ID to use
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    if CUPY_AVAILABLE:
        try:
            cp.cuda.Device(device_id).use()
        except Exception as e:
            logger.warning(f"Failed to set CuPy device: {e}")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if CUPY_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            logger.debug(f"Failed to clear CuPy cache: {e}")


def get_gpu_memory_usage(device_id: int = 0) -> Tuple[int, int]:
    """Get current GPU memory usage.

    Args:
        device_id: GPU device ID

    Returns:
        Tuple of (used_memory, total_memory) in bytes
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated(device_id),
            torch.cuda.get_device_properties(device_id).total_memory
        )
    return (0, 0)


def print_gpu_info():
    """Print detailed GPU information to console."""
    if not check_gpu_availability():
        print("No GPU available")
        return

    gpu_count = get_gpu_count()
    print(f"\n{'='*60}")
    print(f"GPU Information ({gpu_count} device(s) detected)")
    print(f"{'='*60}")

    for i in range(gpu_count):
        info = get_gpu_info(i)
        print(f"\nGPU {i}: {info['name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Total Memory: {format_gpu_memory(info['total_memory'])}")
        print(f"  Free Memory: {format_gpu_memory(info['free_memory'])}")
        print(f"  Used Memory: {format_gpu_memory(info['used_memory'])}")
        print(f"  Memory Utilization: {info['memory_utilization']:.1f}%")
        print(f"  GPU Utilization: {info['gpu_utilization']:.1f}%")
        if info['temperature'] > 0:
            print(f"  Temperature: {info['temperature']}°C")
        if info['power_usage'] > 0:
            print(f"  Power Usage: {info['power_usage']:.1f}W / {info['power_limit']:.1f}W")

    print(f"\n{'='*60}\n")


# RTX 4080 specific optimizations
RTX_4080_SPECS = {
    "name": "NVIDIA GeForce RTX 4080",
    "memory_gb": 16,
    "compute_capability": "8.9",
    "cuda_cores": 9728,
    "tensor_cores": 304,
    "rt_cores": 76,
    "base_clock": 2205,
    "boost_clock": 2505,
    "memory_bandwidth": 716.8,  # GB/s
}


def is_rtx_4080() -> bool:
    """Check if the current GPU is an RTX 4080.

    Returns:
        True if RTX 4080 is detected
    """
    if not check_gpu_availability():
        return False

    info = get_gpu_info(0)
    return "4080" in info.get("name", "")


def get_rtx_4080_optimal_settings() -> Dict[str, any]:
    """Get optimal settings for RTX 4080.

    Returns:
        Dictionary of optimal settings
    """
    return {
        "max_batch_size": 64,  # Depends on model
        "fp16_enabled": True,  # Use mixed precision
        "tensor_cores_enabled": True,
        "cudnn_benchmark": True,
        "memory_fraction": 0.85,  # Use 85% of 16GB
        "num_workers": 8,  # For data loading
        "pin_memory": True,
        "persistent_workers": True,
    }
