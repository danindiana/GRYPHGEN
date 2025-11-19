#!/usr/bin/env python3
"""
GRYPHGEN GPU Benchmark Suite
Performance benchmarks for NVIDIA RTX 4080
"""

import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available, skipping GPU benchmarks")
    TORCH_AVAILABLE = False
    sys.exit(1)


def benchmark_matmul(sizes=[1000, 2000, 4000, 8000]):
    """Benchmark matrix multiplication"""
    print("\n" + "="*60)
    print("Matrix Multiplication Benchmark (FP32)")
    print("="*60)
    print(f"{'Size':>10} {'Time (ms)':>12} {'TFLOPS':>10} {'Memory (GB)':>12}")
    print("-"*60)

    for size in sizes:
        # Allocate matrices
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')

        # Warm-up
        for _ in range(3):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        time_ms = (elapsed / iterations) * 1000
        tflops = (2 * size**3 * iterations) / (elapsed * 1e12)
        memory_gb = (x.element_size() * x.nelement() * 3) / 1e9

        print(f"{size:>10} {time_ms:>12.2f} {tflops:>10.2f} {memory_gb:>12.2f}")

        # Clean up
        del x, y, z
        torch.cuda.empty_cache()


def benchmark_convolution():
    """Benchmark convolution operations"""
    print("\n" + "="*60)
    print("Convolution Benchmark")
    print("="*60)

    configs = [
        (32, 64, 224, 224, 3),   # (batch, channels, height, width, kernel)
        (64, 128, 112, 112, 3),
        (128, 256, 56, 56, 3),
    ]

    print(f"{'Config':>25} {'Time (ms)':>12} {'Images/sec':>12}")
    print("-"*60)

    for batch, channels, height, width, kernel in configs:
        # Create model and input
        conv = nn.Conv2d(channels, channels, kernel, padding=1).cuda()
        x = torch.randn(batch, channels, height, width, device='cuda')

        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                y = conv(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                y = conv(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        time_ms = (elapsed / iterations) * 1000
        images_per_sec = (batch * iterations) / elapsed

        config_str = f"B{batch}_C{channels}_{height}x{width}_K{kernel}"
        print(f"{config_str:>25} {time_ms:>12.2f} {images_per_sec:>12.1f}")

        # Clean up
        del conv, x, y
        torch.cuda.empty_cache()


def benchmark_transformer():
    """Benchmark transformer operations"""
    print("\n" + "="*60)
    print("Transformer Benchmark")
    print("="*60)

    configs = [
        (1, 512, 768),    # (batch, seq_len, hidden_dim)
        (4, 512, 768),
        (8, 1024, 768),
        (16, 2048, 768),
    ]

    print(f"{'Config':>20} {'Time (ms)':>12} {'Tokens/sec':>12}")
    print("-"*60)

    for batch, seq_len, hidden_dim in configs:
        # Create simple transformer layer
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        ).cuda()

        x = torch.randn(batch, seq_len, hidden_dim, device='cuda')

        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                y = layer(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        iterations = 50
        with torch.no_grad():
            for _ in range(iterations):
                y = layer(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate metrics
        time_ms = (elapsed / iterations) * 1000
        tokens_per_sec = (batch * seq_len * iterations) / elapsed

        config_str = f"B{batch}_S{seq_len}_H{hidden_dim}"
        print(f"{config_str:>20} {time_ms:>12.2f} {tokens_per_sec:>12.1f}")

        # Clean up
        del layer, x, y
        torch.cuda.empty_cache()


def benchmark_memory():
    """Benchmark GPU memory bandwidth"""
    print("\n" + "="*60)
    print("Memory Bandwidth Benchmark")
    print("="*60)

    sizes_gb = [0.5, 1.0, 2.0, 4.0, 8.0]

    print(f"{'Size (GB)':>12} {'Read (GB/s)':>12} {'Write (GB/s)':>12} {'Copy (GB/s)':>12}")
    print("-"*60)

    for size_gb in sizes_gb:
        size = int(size_gb * 1024**3 / 4)  # Size in float32 elements

        # Test read
        x = torch.randn(size, device='cuda')
        torch.cuda.synchronize()

        start = time.time()
        y = x.sum()
        torch.cuda.synchronize()
        read_time = time.time() - start
        read_bw = size_gb / read_time

        # Test write
        start = time.time()
        z = torch.zeros(size, device='cuda')
        torch.cuda.synchronize()
        write_time = time.time() - start
        write_bw = size_gb / write_time

        # Test copy
        start = time.time()
        w = x.clone()
        torch.cuda.synchronize()
        copy_time = time.time() - start
        copy_bw = size_gb / copy_time

        print(f"{size_gb:>12.1f} {read_bw:>12.1f} {write_bw:>12.1f} {copy_bw:>12.1f}")

        # Clean up
        del x, y, z, w
        torch.cuda.empty_cache()


def benchmark_mixed_precision():
    """Benchmark mixed precision (FP16/BF16) performance"""
    print("\n" + "="*60)
    print("Mixed Precision Benchmark")
    print("="*60)

    size = 4096
    iterations = 100

    dtypes = [
        (torch.float32, 'FP32'),
        (torch.float16, 'FP16'),
        (torch.bfloat16, 'BF16'),
    ]

    print(f"{'Precision':>12} {'Time (ms)':>12} {'TFLOPS':>10} {'Speedup':>10}")
    print("-"*60)

    fp32_time = None

    for dtype, name in dtypes:
        x = torch.randn(size, size, device='cuda', dtype=dtype)
        y = torch.randn(size, size, device='cuda', dtype=dtype)

        # Warm-up
        for _ in range(3):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(iterations):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        time_ms = (elapsed / iterations) * 1000
        tflops = (2 * size**3 * iterations) / (elapsed * 1e12)

        if fp32_time is None:
            fp32_time = time_ms
            speedup = 1.0
        else:
            speedup = fp32_time / time_ms

        print(f"{name:>12} {time_ms:>12.2f} {tflops:>10.2f} {speedup:>10.2f}x")

        del x, y, z
        torch.cuda.empty_cache()


def print_gpu_info():
    """Print GPU information"""
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"PyTorch Version: {torch.__version__}")

        # Check for TF32 support
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            print(f"TF32 Support: {torch.backends.cuda.matmul.allow_tf32}")

        # Memory info
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nMemory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available")


def main():
    """Run all benchmarks"""
    if not TORCH_AVAILABLE:
        return 1

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run GPU benchmarks.")
        return 1

    print("="*60)
    print("GRYPHGEN GPU Benchmark Suite")
    print("Target: NVIDIA RTX 4080 16GB")
    print("="*60)

    # Print GPU info
    print_gpu_info()

    try:
        # Run benchmarks
        benchmark_matmul()
        benchmark_convolution()
        benchmark_transformer()
        benchmark_memory()
        benchmark_mixed_precision()

        print("\n" + "="*60)
        print("✓ All benchmarks completed successfully")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
