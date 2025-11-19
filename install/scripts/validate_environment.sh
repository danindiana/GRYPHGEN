#!/bin/bash

################################################################################
# GRYPHGEN Environment Validation Script
# Version: 2.0
# Description: Validates complete GRYPHGEN installation
################################################################################

set -euo pipefail
IFS=$'\n\t'

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration
readonly VENV_PATH="/opt/gryphgen/venv"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNED=0

################################################################################
# Logging Functions
################################################################################

log_section() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""
}

test_pass() {
    echo -e "${GREEN}[✓]${NC} $*"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}[✗]${NC} $*"
    ((TESTS_FAILED++))
}

test_warn() {
    echo -e "${YELLOW}[!]${NC} $*"
    ((TESTS_WARNED++))
}

################################################################################
# System Tests
################################################################################

test_system_requirements() {
    log_section "System Requirements"

    # Check OS
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        if [ "$ID" = "ubuntu" ]; then
            test_pass "Ubuntu detected: $VERSION_ID"
        else
            test_warn "Not Ubuntu: $ID $VERSION_ID"
        fi
    else
        test_fail "Cannot determine OS"
    fi

    # Check kernel
    local kernel_version=$(uname -r)
    test_pass "Kernel: $kernel_version"

    # Check disk space
    local disk_free=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$disk_free" -gt 50 ]; then
        test_pass "Disk space: ${disk_free}GB available"
    else
        test_warn "Low disk space: ${disk_free}GB available (50GB+ recommended)"
    fi

    # Check memory
    local mem_total=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$mem_total" -ge 32 ]; then
        test_pass "Memory: ${mem_total}GB (recommended: 32GB+)"
    elif [ "$mem_total" -ge 16 ]; then
        test_warn "Memory: ${mem_total}GB (recommended: 32GB+)"
    else
        test_fail "Memory: ${mem_total}GB (minimum: 16GB)"
    fi

    # Check CPU cores
    local cpu_cores=$(nproc)
    if [ "$cpu_cores" -ge 8 ]; then
        test_pass "CPU cores: $cpu_cores"
    else
        test_warn "CPU cores: $cpu_cores (recommended: 8+)"
    fi
}

test_nvidia_driver() {
    log_section "NVIDIA Driver"

    if command -v nvidia-smi &> /dev/null; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        test_pass "NVIDIA driver: $driver_version"

        # Check GPU
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        test_pass "GPU: $gpu_name"

        # Check memory
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        test_pass "GPU Memory: $gpu_memory"

        # Check compute capability
        local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        test_pass "Compute Capability: $compute_cap"

        # Check if RTX 4080
        if echo "$gpu_name" | grep -q "RTX 4080"; then
            test_pass "Target GPU (RTX 4080) detected"
        else
            test_warn "Different GPU than target (RTX 4080)"
        fi
    else
        test_fail "NVIDIA driver not found"
    fi
}

test_cuda() {
    log_section "CUDA Toolkit"

    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        test_pass "CUDA compiler (nvcc): $cuda_version"

        # Check CUDA path
        if [ -d "/usr/local/cuda" ]; then
            test_pass "CUDA path: /usr/local/cuda"
        else
            test_warn "CUDA path not found at /usr/local/cuda"
        fi

        # Check cuDNN
        if ldconfig -p | grep -q libcudnn; then
            local cudnn_version=$(ldconfig -p | grep libcudnn | head -1 | awk '{print $1}')
            test_pass "cuDNN: $cudnn_version"
        else
            test_warn "cuDNN not found"
        fi
    else
        test_fail "CUDA compiler (nvcc) not found"
    fi
}

test_docker() {
    log_section "Docker"

    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        test_pass "Docker: $docker_version"

        # Check Docker service
        if systemctl is-active --quiet docker; then
            test_pass "Docker service: running"
        else
            test_fail "Docker service: not running"
        fi

        # Check NVIDIA runtime
        if docker info 2>/dev/null | grep -q "nvidia"; then
            test_pass "NVIDIA Docker runtime: configured"
        else
            test_warn "NVIDIA Docker runtime: not configured"
        fi

        # Test GPU access in Docker
        if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            test_pass "Docker GPU access: working"
        else
            test_warn "Docker GPU access: not working"
        fi
    else
        test_fail "Docker not installed"
    fi

    # Check Docker Compose
    if docker compose version &> /dev/null; then
        local compose_version=$(docker compose version | awk '{print $4}')
        test_pass "Docker Compose: $compose_version"
    else
        test_warn "Docker Compose not found"
    fi
}

test_python() {
    log_section "Python Environment"

    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | awk '{print $2}')
        test_pass "Python: $python_version"

        # Check pip
        if command -v pip3 &> /dev/null; then
            local pip_version=$(pip3 --version | awk '{print $2}')
            test_pass "pip: $pip_version"
        else
            test_fail "pip not found"
        fi

        # Check virtual environment
        if [ -d "$VENV_PATH" ]; then
            test_pass "Virtual environment: $VENV_PATH"

            # Activate and test
            source "$VENV_PATH/bin/activate"

            # Check key packages
            local packages=(
                "numpy"
                "torch"
                "transformers"
                "zmq"
                "fastapi"
            )

            for pkg in "${packages[@]}"; do
                if python -c "import $pkg" 2>/dev/null; then
                    local version=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
                    test_pass "Package $pkg: $version"
                else
                    test_fail "Package $pkg: not found"
                fi
            done
        else
            test_fail "Virtual environment not found"
        fi
    else
        test_fail "Python not installed"
    fi
}

test_pytorch() {
    log_section "PyTorch"

    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"

        # Test PyTorch
        python << 'EOF'
import sys
import torch

try:
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available in PyTorch")
        sys.exit(1)

    print(f"PASS: PyTorch version: {torch.__version__}")
    print(f"PASS: CUDA version: {torch.version.cuda}")
    print(f"PASS: cuDNN version: {torch.backends.cudnn.version()}")
    print(f"PASS: GPU device: {torch.cuda.get_device_name(0)}")
    print(f"PASS: GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print("PASS: GPU computation test")

    # Test TF32
    print(f"PASS: TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
EOF

        if [ $? -eq 0 ]; then
            ((TESTS_PASSED += 7))
        else
            ((TESTS_FAILED += 7))
        fi
    else
        test_fail "Cannot test PyTorch: virtual environment not found"
    fi
}

test_gryphgen_directories() {
    log_section "GRYPHGEN Directories"

    local directories=(
        "/opt/gryphgen"
        "/data/gryphgen"
        "/models/gryphgen"
        "/var/log/gryphgen"
    )

    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            test_pass "Directory exists: $dir"
        else
            test_warn "Directory missing: $dir"
        fi
    done
}

test_environment_variables() {
    log_section "Environment Variables"

    # Source profile files
    [ -f /etc/profile.d/gryphgen.sh ] && source /etc/profile.d/gryphgen.sh
    [ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh

    local vars=(
        "CUDA_HOME"
        "GRYPHGEN_HOME"
        "GRYPHGEN_DATA"
        "GRYPHGEN_MODELS"
    )

    for var in "${vars[@]}"; do
        if [ -n "${!var+x}" ]; then
            test_pass "$var: ${!var}"
        else
            test_warn "$var: not set"
        fi
    done
}

test_network() {
    log_section "Network Connectivity"

    # Test DNS
    if ping -c 1 8.8.8.8 &> /dev/null; then
        test_pass "Internet connectivity"
    else
        test_fail "No internet connectivity"
    fi

    # Test PyPI
    if curl -s https://pypi.org &> /dev/null; then
        test_pass "PyPI accessible"
    else
        test_warn "PyPI not accessible"
    fi

    # Test Hugging Face
    if curl -s https://huggingface.co &> /dev/null; then
        test_pass "Hugging Face accessible"
    else
        test_warn "Hugging Face not accessible"
    fi
}

################################################################################
# Performance Benchmark
################################################################################

benchmark_gpu() {
    log_section "GPU Performance Benchmark"

    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"

        python << 'EOF'
import torch
import time

if not torch.cuda.is_available():
    print("WARN: CUDA not available, skipping benchmark")
    exit(0)

# Matrix multiplication benchmark
sizes = [1000, 2000, 4000, 8000]
print("\nMatrix Multiplication Benchmark (FP32):")
print("-" * 50)

for size in sizes:
    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()

    # Warm-up
    for _ in range(3):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(10):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tflops = (2 * size**3 * 10) / (elapsed * 1e12)
    print(f"Size {size}x{size}: {elapsed/10*1000:.2f} ms, {tflops:.2f} TFLOPS")

print("-" * 50)
EOF

        if [ $? -eq 0 ]; then
            test_pass "GPU benchmark completed"
        else
            test_warn "GPU benchmark failed"
        fi
    fi
}

################################################################################
# Summary
################################################################################

print_summary() {
    log_section "Validation Summary"

    local total_tests=$((TESTS_PASSED + TESTS_FAILED + TESTS_WARNED))

    echo ""
    echo -e "${GREEN}Passed:  $TESTS_PASSED${NC}"
    echo -e "${YELLOW}Warnings: $TESTS_WARNED${NC}"
    echo -e "${RED}Failed:  $TESTS_FAILED${NC}"
    echo -e "Total:   $total_tests"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ GRYPHGEN environment is ready!${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
        return 0
    else
        echo -e "${RED}═══════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  ✗ GRYPHGEN environment has issues${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Please review the failed tests above and:"
        echo "  1. Check installation logs: /var/log/gryphgen_install.log"
        echo "  2. Re-run failed installation scripts"
        echo "  3. Consult docs/troubleshooting.md"
        return 1
    fi
}

################################################################################
# Main
################################################################################

main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                                                       ║${NC}"
    echo -e "${BLUE}║        GRYPHGEN Environment Validation Suite         ║${NC}"
    echo -e "${BLUE}║                   Version 2.0                         ║${NC}"
    echo -e "${BLUE}║                                                       ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"

    # Run all tests
    test_system_requirements
    test_nvidia_driver
    test_cuda
    test_docker
    test_python
    test_pytorch
    test_gryphgen_directories
    test_environment_variables
    test_network

    # Run benchmark
    benchmark_gpu

    # Print summary
    print_summary
}

# Run main function
main "$@"
