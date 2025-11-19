#!/usr/bin/env python3
"""
GRYPHGEN CUDA Test Suite
Tests NVIDIA GPU, CUDA, and cuDNN installation
"""

import sys
import subprocess
from pathlib import Path


def test_nvidia_smi():
    """Test nvidia-smi availability and output"""
    print("Testing nvidia-smi...")
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ nvidia-smi is available")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ nvidia-smi failed: {e}")
        return False


def test_nvcc():
    """Test CUDA compiler (nvcc) availability"""
    print("\nTesting nvcc...")
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ nvcc is available")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ nvcc failed: {e}")
        return False


def test_cuda_paths():
    """Test CUDA installation paths"""
    print("\nTesting CUDA paths...")
    cuda_paths = [
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda/bin"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/include"),
    ]

    all_exist = True
    for path in cuda_paths:
        if path.exists():
            print(f"✓ {path} exists")
        else:
            print(f"✗ {path} does not exist")
            all_exist = False

    return all_exist


def test_cudnn():
    """Test cuDNN installation"""
    print("\nTesting cuDNN...")
    try:
        result = subprocess.run(
            ['ldconfig', '-p'],
            capture_output=True,
            text=True,
            check=True
        )

        if 'libcudnn' in result.stdout:
            print("✓ cuDNN libraries found")
            # Show cuDNN libraries
            for line in result.stdout.split('\n'):
                if 'libcudnn' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("✗ cuDNN libraries not found")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ ldconfig failed: {e}")
        return False


def test_gpu_info():
    """Test and display GPU information"""
    print("\nGPU Information:")
    try:
        # Get GPU name
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )

        info = result.stdout.strip()
        print(f"  {info}")

        # Check for RTX 4080
        if "RTX 4080" in info:
            print("✓ Target GPU (RTX 4080) detected")
        else:
            print("! Different GPU than target (RTX 4080)")

        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to get GPU info: {e}")
        return False


def main():
    """Run all CUDA tests"""
    print("="*60)
    print("GRYPHGEN CUDA Test Suite")
    print("="*60)

    tests = [
        ("NVIDIA SMI", test_nvidia_smi),
        ("CUDA Compiler (nvcc)", test_nvcc),
        ("CUDA Paths", test_cuda_paths),
        ("cuDNN", test_cudnn),
        ("GPU Information", test_gpu_info),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} raised exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All CUDA tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
