#!/bin/bash
# Dependency Validation Script
# Checks for all required dependencies
# Target: Ubuntu 22.04 LTS

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_pass() { echo -e "${GREEN}[✓]${NC} $1"; }
print_fail() { echo -e "${RED}[✗]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

MISSING_DEPS=()

check_command() {
    local cmd=$1
    local package=$2

    if command -v "$cmd" &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -1 || echo "unknown")
        print_pass "$cmd: $version"
        return 0
    else
        print_fail "$cmd not found (install: $package)"
        MISSING_DEPS+=("$package")
        return 1
    fi
}

check_library() {
    local lib=$1
    local package=$2

    if ldconfig -p | grep -q "$lib"; then
        print_pass "$lib library found"
        return 0
    else
        print_fail "$lib not found (install: $package)"
        MISSING_DEPS+=("$package")
        return 1
    fi
}

check_python_package() {
    local pkg=$1

    if python3 -c "import $pkg" 2>/dev/null; then
        local version=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        print_pass "Python $pkg: $version"
        return 0
    else
        print_fail "Python $pkg not found"
        MISSING_DEPS+=("python3-$pkg")
        return 1
    fi
}

echo "=== Dependency Validation ==="
echo

# System info
print_info "System: $(lsb_release -d | cut -f2)"
print_info "Kernel: $(uname -r)"
echo

# Build essentials
echo "=== Build Tools ==="
check_command gcc build-essential
check_command g++ build-essential
check_command make build-essential
check_command cmake cmake
check_command git git
echo

# CUDA
echo "=== CUDA Toolkit ==="
check_command nvcc cuda-toolkit
check_command nvidia-smi nvidia-driver
echo

# CUDA Libraries
echo "=== CUDA Libraries ==="
check_library libcudart cudart
check_library libcublas cublas
check_library libcufft cufft
check_library libcurand curand
check_library libcusparse cusparse
echo

# Python
echo "=== Python Environment ==="
check_command python3 python3
check_command pip3 python3-pip
echo

# Optional Python packages
echo "=== Python Packages (Optional) ==="
check_python_package numpy || true
check_python_package torch || true
check_python_package cuda || true
echo

# System utilities
echo "=== System Utilities ==="
check_command htop htop
check_command nvtop nvtop
check_command perf linux-tools-common
echo

# Monitoring
echo "=== Monitoring Tools ==="
check_command sysstat sysstat
check_command iostat sysstat
check_command numactl numactl
echo

# RDMA/InfiniBand (optional)
echo "=== InfiniBand/RDMA (Optional) ==="
check_library libibverbs libibverbs1 || true
check_library librdmacm librdmacm1 || true
check_command ibstat infiniband-diags || true
echo

# Summary
echo "=== Summary ==="
if [[ ${#MISSING_DEPS[@]} -eq 0 ]]; then
    print_pass "All dependencies satisfied!"
    exit 0
else
    print_fail "${#MISSING_DEPS[@]} missing dependencies"
    echo
    echo "Install missing packages with:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y ${MISSING_DEPS[*]}"
    exit 1
fi
