#!/usr/bin/env python3
"""
GRYPHGEN Environment Test Suite
Tests Python environment and package installations
"""

import sys
import os
import subprocess
from pathlib import Path


def test_python_version():
    """Test Python version"""
    print("Testing Python version...")
    version = sys.version_info

    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (3.11+ required)")
        return False


def test_package_import(package_name, display_name=None):
    """Test if a package can be imported"""
    if display_name is None:
        display_name = package_name

    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: not found ({e})")
        return False


def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print("\nTesting PyTorch CUDA...")
    try:
        import torch

        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")

            props = torch.cuda.get_device_properties(0)
            print(f"  Device memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")

            # Test basic computation
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print("✓ GPU computation test passed")
                return True
            except Exception as e:
                print(f"✗ GPU computation failed: {e}")
                return False
        else:
            print("✗ CUDA not available in PyTorch")
            return False

    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False


def test_environment_variables():
    """Test important environment variables"""
    print("\nTesting environment variables...")

    env_vars = {
        'CUDA_HOME': '/usr/local/cuda',
        'GRYPHGEN_HOME': '/opt/gryphgen',
        'GRYPHGEN_DATA': '/data/gryphgen',
        'GRYPHGEN_MODELS': '/models/gryphgen',
    }

    all_set = True
    for var, expected in env_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}: {value}")
        else:
            print(f"! {var}: not set (recommended: {expected})")
            all_set = False

    return all_set


def test_directories():
    """Test GRYPHGEN directories"""
    print("\nTesting GRYPHGEN directories...")

    directories = [
        Path('/opt/gryphgen'),
        Path('/data/gryphgen'),
        Path('/models/gryphgen'),
        Path('/var/log/gryphgen'),
    ]

    all_exist = True
    for directory in directories:
        if directory.exists():
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} does not exist")
            all_exist = False

    return all_exist


def test_virtual_environment():
    """Test if running in virtual environment"""
    print("\nTesting virtual environment...")

    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print(f"✓ Running in virtual environment: {sys.prefix}")
        return True
    else:
        print("! Not running in virtual environment")
        return True  # Not critical


def main():
    """Run all environment tests"""
    print("="*60)
    print("GRYPHGEN Environment Test Suite")
    print("="*60)

    # Test Python version
    print("\nPython Environment:")
    python_ok = test_python_version()
    venv_ok = test_virtual_environment()

    # Test core packages
    print("\nCore Packages:")
    packages_to_test = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('requests', 'Requests'),
        ('zmq', 'PyZMQ'),
        ('fastapi', 'FastAPI'),
        ('pydantic', 'Pydantic'),
    ]

    core_results = []
    for pkg, name in packages_to_test:
        result = test_package_import(pkg, name)
        core_results.append(result)

    # Test ML packages
    print("\nMachine Learning Packages:")
    ml_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('langchain', 'LangChain'),
    ]

    ml_results = []
    for pkg, name in ml_packages:
        result = test_package_import(pkg, name)
        ml_results.append(result)

    # Test PyTorch CUDA
    cuda_ok = test_torch_cuda()

    # Test environment
    env_ok = test_environment_variables()
    dir_ok = test_directories()

    # Print summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)

    total_tests = 1 + len(packages_to_test) + len(ml_packages) + 3
    passed_tests = (
        int(python_ok) +
        sum(core_results) +
        sum(ml_results) +
        int(cuda_ok) +
        int(env_ok) +
        int(dir_ok)
    )

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✓ All environment tests passed!")
        return 0
    elif cuda_ok and sum(ml_results) == len(ml_results):
        print("\n✓ Essential tests passed (ML environment ready)")
        return 0
    else:
        print(f"\n! {total_tests - passed_tests} test(s) failed or warned")
        return 1


if __name__ == "__main__":
    sys.exit(main())
