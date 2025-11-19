"""Unit tests for GPU utilities."""

import pytest
import torch


@pytest.mark.gpu
def test_gpu_manager_initialization():
    """Test GPU manager initialization."""
    from src.utils.gpu_utils import GPUManager

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    manager = GPUManager(device_id=0)
    assert manager.device_id == 0
    assert manager.device.type == "cuda"


@pytest.mark.gpu
def test_get_gpu_info():
    """Test getting GPU information."""
    from src.utils.gpu_utils import get_gpu_info

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    info = get_gpu_info()
    assert info is not None
    assert info.total_memory_gb > 0
    assert "RTX" in info.name or "GPU" in info.name


def test_get_gpu_info_no_cuda(mock_gpu_not_available):
    """Test getting GPU info when CUDA is not available."""
    from src.utils.gpu_utils import get_gpu_info

    info = get_gpu_info()
    assert info is None


@pytest.mark.gpu
def test_optimize_for_rtx4080():
    """Test RTX 4080 optimization."""
    from src.utils.gpu_utils import optimize_for_rtx4080

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    manager = optimize_for_rtx4080(
        mixed_precision=True,
        memory_limit_gb=14.0,
        device_id=0,
    )

    assert manager is not None
    assert manager.device_id == 0


def test_get_optimal_batch_size():
    """Test optimal batch size calculation."""
    from src.utils.gpu_utils import get_optimal_batch_size

    batch_size = get_optimal_batch_size(
        model_size_gb=2.0,
        available_memory_gb=14.0,
        data_memory_multiplier=2.0,
    )

    assert batch_size >= 1
    assert isinstance(batch_size, int)
