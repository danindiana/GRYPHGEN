#!/usr/bin/env python3
"""
Performance benchmarking script for CALISOTA.
Tests GPU performance, RAG speed, and API latency.
"""

import asyncio
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calisota.core.config import get_settings
from src.calisota.rag.faiss_manager import FAISSManager


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def benchmark_gpu() -> None:
    """Benchmark GPU performance."""
    print_header("GPU Performance Benchmark")

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    device = torch.device("cuda:0")

    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Matrix multiplication benchmark
    sizes = [1024, 2048, 4096]
    print("\nMatrix Multiplication Benchmark (TFLOPS):")
    print(f"{'Size':<10} {'Time (ms)':<15} {'TFLOPS':<15}")
    print("-" * 40)

    for size in sizes:
        # FP32
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        # Calculate TFLOPS
        ops = 2 * size**3  # Matrix multiplication operations
        tflops = (ops / elapsed) / 1e12

        print(f"{size}x{size:<4} {elapsed*1000:<15.2f} {tflops:<15.2f}")

    # Memory benchmark
    print("\nMemory Bandwidth Benchmark:")
    data_size_gb = 1.0
    data = torch.randn(int(data_size_gb * 1024**3 / 4), device=device)

    start = time.time()
    _ = data.sum()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    bandwidth = data_size_gb / elapsed
    print(f"Bandwidth: {bandwidth:.2f} GB/s")


def benchmark_rag() -> None:
    """Benchmark RAG system performance."""
    print_header("RAG System Benchmark")

    settings = get_settings()
    faiss = FAISSManager(settings)
    faiss.initialize_index()

    # Generate test data
    num_docs = 10000
    print(f"Generating {num_docs} test documents...")

    test_texts = [f"This is test document number {i} about programming" for i in range(num_docs)]

    # Benchmark embedding generation
    print("\nEmbedding Generation:")
    start = time.time()
    faiss.add_embeddings(test_texts)
    elapsed = time.time() - start

    docs_per_sec = num_docs / elapsed
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {docs_per_sec:.0f} docs/sec")

    # Benchmark search
    print("\nSearch Performance:")
    queries = [
        "programming tutorial",
        "how to write code",
        "best practices",
        "algorithm implementation",
        "data structures"
    ]

    search_times = []
    for query in queries:
        start = time.time()
        _ = faiss.search(query, top_k=10)
        elapsed = time.time() - start
        search_times.append(elapsed)

    avg_search_time = np.mean(search_times)
    queries_per_sec = 1.0 / avg_search_time

    print(f"Average search time: {avg_search_time*1000:.2f}ms")
    print(f"Queries per second: {queries_per_sec:.0f}")


async def benchmark_api() -> None:
    """Benchmark API endpoints."""
    print_header("API Endpoint Benchmark")

    try:
        import httpx

        client = httpx.AsyncClient(base_url="http://localhost:8000")

        # Health check
        print("Health Check:")
        times = []
        for _ in range(10):
            start = time.time()
            response = await client.get("/api/health")
            elapsed = time.time() - start
            times.append(elapsed)

        print(f"Average latency: {np.mean(times)*1000:.2f}ms")
        print(f"p95 latency: {np.percentile(times, 95)*1000:.2f}ms")

        await client.aclose()

    except Exception as e:
        print(f"❌ API benchmark failed: {e}")
        print("Make sure the API server is running: make run")


def main() -> None:
    """Run all benchmarks."""
    print("CALISOTA Performance Benchmark Suite")
    print(f"Python: {sys.version}")

    try:
        benchmark_gpu()
    except Exception as e:
        print(f"GPU benchmark failed: {e}")

    try:
        benchmark_rag()
    except Exception as e:
        print(f"RAG benchmark failed: {e}")

    try:
        asyncio.run(benchmark_api())
    except Exception as e:
        print(f"API benchmark failed: {e}")

    print_header("Benchmark Complete")


if __name__ == "__main__":
    main()
