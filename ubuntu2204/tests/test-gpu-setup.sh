#!/bin/bash
# GPU Setup Validation Script
# Tests NVIDIA GPU setup and CUDA installation
# Target: Ubuntu 22.04 LTS with RTX 4080

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name=$1
    local test_command=$2

    echo -n "Testing: $test_name... "

    if eval "$test_command" > /dev/null 2>&1; then
        print_pass "$test_name"
        ((TESTS_PASSED++))
        return 0
    else
        print_fail "$test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "=== GPU Setup Validation ==="
echo "Date: $(date)"
echo

# Test 1: Check if nvidia-smi is available
print_info "Step 1: Checking NVIDIA Driver"
if run_test "NVIDIA driver installation" "command -v nvidia-smi"; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo
fi

# Test 2: Check CUDA installation
print_info "Step 2: Checking CUDA Installation"
if run_test "CUDA installation" "command -v nvcc"; then
    nvcc --version | grep release
    echo
fi

# Test 3: Check GPU count
print_info "Step 3: Checking GPU Detection"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [[ $GPU_COUNT -gt 0 ]]; then
    print_pass "Detected $GPU_COUNT GPU(s)"
    ((TESTS_PASSED++))
else
    print_fail "No GPUs detected"
    ((TESTS_FAILED++))
fi
echo

# Test 4: Check for RTX 4080
print_info "Step 4: Checking for RTX 4080"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" =~ "4080" ]]; then
    print_pass "RTX 4080 detected: $GPU_NAME"
    ((TESTS_PASSED++))
else
    print_warn "GPU detected: $GPU_NAME (not RTX 4080)"
    print_warn "Tests will continue but performance may differ"
fi
echo

# Test 5: Check compute capability
print_info "Step 5: Checking Compute Capability"
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "$COMPUTE_CAP" == "8.9" ]]; then
    print_pass "Compute capability 8.9 (Ada Lovelace)"
    ((TESTS_PASSED++))
elif [[ "${COMPUTE_CAP%%.*}" -ge 7 ]]; then
    print_warn "Compute capability $COMPUTE_CAP (Tensor Cores supported)"
    print_warn "Optimized for 8.9, but will work on $COMPUTE_CAP"
else
    print_fail "Compute capability $COMPUTE_CAP (Tensor Cores not supported)"
    ((TESTS_FAILED++))
fi
echo

# Test 6: Check memory
print_info "Step 6: Checking GPU Memory"
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ $GPU_MEM -ge 15000 ]]; then  # At least 15GB
    print_pass "GPU memory: ${GPU_MEM} MB"
    ((TESTS_PASSED++))
else
    print_warn "GPU memory: ${GPU_MEM} MB (RTX 4080 has 16384 MB)"
fi
echo

# Test 7: Check CUDA libraries
print_info "Step 7: Checking CUDA Libraries"
run_test "cuBLAS library" "ldconfig -p | grep -q libcublas"
run_test "cuDNN library" "ldconfig -p | grep -q libcudnn"
run_test "cuFFT library" "ldconfig -p | grep -q libcufft"
echo

# Test 8: Compile and run simple CUDA program
print_info "Step 8: Testing CUDA Compilation"

TEMP_DIR=$(mktemp -d)
cat > "$TEMP_DIR/test.cu" << 'EOF'
#include <stdio.h>
__global__ void testKernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}
int main() {
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("CUDA test successful!\n");
    return 0;
}
EOF

if nvcc -arch=sm_89 "$TEMP_DIR/test.cu" -o "$TEMP_DIR/test" 2>/dev/null; then
    print_pass "CUDA compilation successful"
    ((TESTS_PASSED++))

    if "$TEMP_DIR/test" > /dev/null 2>&1; then
        print_pass "CUDA program execution successful"
        ((TESTS_PASSED++))
    else
        print_fail "CUDA program execution failed"
        ((TESTS_FAILED++))
    fi
else
    print_fail "CUDA compilation failed"
    ((TESTS_FAILED++))
fi
rm -rf "$TEMP_DIR"
echo

# Test 9: Check GPU health
print_info "Step 9: Checking GPU Health"
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
if [[ $TEMP -lt 85 ]]; then
    print_pass "GPU temperature: ${TEMP}°C"
    ((TESTS_PASSED++))
else
    print_warn "GPU temperature: ${TEMP}°C (high)"
fi

POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader | head -1)
print_info "GPU power draw: $POWER"

THROTTLE=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader | head -1)
if [[ "$THROTTLE" == "0x0000000000000000" ]]; then
    print_pass "No throttling detected"
    ((TESTS_PASSED++))
else
    print_warn "Throttling detected: $THROTTLE"
fi
echo

# Test 10: Check PCIe
print_info "Step 10: Checking PCIe Configuration"
PCIE_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader | head -1)
PCIE_WIDTH=$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader | head -1)
PCIE_MAX_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.max --format=csv,noheader | head -1)
PCIE_MAX_WIDTH=$(nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader | head -1)

echo "Current: PCIe Gen${PCIE_GEN} x${PCIE_WIDTH}"
echo "Maximum: PCIe Gen${PCIE_MAX_GEN} x${PCIE_MAX_WIDTH}"

if [[ $PCIE_GEN -eq $PCIE_MAX_GEN ]] && [[ $PCIE_WIDTH -eq $PCIE_MAX_WIDTH ]]; then
    print_pass "PCIe running at maximum speed"
    ((TESTS_PASSED++))
else
    print_warn "PCIe not at maximum speed (may affect performance)"
fi
echo

# Test 11: Check persistence mode
print_info "Step 11: Checking GPU Persistence Mode"
PERSISTENCE=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)
if [[ "$PERSISTENCE" == "Enabled" ]]; then
    print_pass "Persistence mode enabled"
    ((TESTS_PASSED++))
else
    print_warn "Persistence mode disabled (may affect startup time)"
    print_info "Enable with: sudo nvidia-smi -pm 1"
fi
echo

# Summary
echo "=== Test Summary ==="
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo "Total:  $((TESTS_PASSED + TESTS_FAILED))"
echo

if [[ $TESTS_FAILED -eq 0 ]]; then
    print_pass "All critical tests passed! ✓"
    print_info "Your GPU is properly configured for HPC/ML workloads"
    exit 0
else
    print_fail "Some tests failed"
    print_info "Please review the failures above"
    exit 1
fi
