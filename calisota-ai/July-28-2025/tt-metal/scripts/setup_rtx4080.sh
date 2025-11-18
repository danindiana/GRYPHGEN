#!/bin/bash
# Setup script for NVIDIA RTX 4080 development environment

set -e

echo "========================================"
echo "Dynamic Cortex - RTX 4080 Setup"
echo "========================================"
echo ""

# Check for root/sudo
if [[ $EUID -eq 0 ]]; then
   echo "This script should NOT be run as root (it will ask for sudo when needed)"
   exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "Cannot detect OS. Please run on Ubuntu 20.04 or 22.04"
    exit 1
fi

echo "Detected OS: $OS $VER"

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."

if command -v nvidia-smi &> /dev/null; then
    echo "Found nvidia-smi. GPU information:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
else
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# Check CUDA version
echo ""
echo "Checking CUDA installation..."

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Found CUDA version: $CUDA_VERSION"

    if [ $(echo "$CUDA_VERSION >= 12.0" | bc) -eq 1 ]; then
        echo "CUDA version is compatible (>= 12.0)"
    else
        echo "WARNING: CUDA version < 12.0. Some features may not work."
        echo "Recommended: CUDA 12.0 or later"
    fi
else
    echo "CUDA not found. Installing CUDA Toolkit..."

    # Ubuntu version-specific CUDA installation
    if [[ "$VER" == "22.04" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-4
        rm cuda-keyring_1.1-1_all.deb
    elif [[ "$VER" == "20.04" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-4
        rm cuda-keyring_1.1-1_all.deb
    else
        echo "Unsupported Ubuntu version for automatic CUDA installation"
        echo "Please install CUDA 12.x manually from:"
        echo "https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi

    # Add CUDA to PATH
    SHELL_RC="$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi

    echo "" >> $SHELL_RC
    echo "# CUDA Toolkit" >> $SHELL_RC
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> $SHELL_RC
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> $SHELL_RC

    echo "Added CUDA to $SHELL_RC"
    echo "Please run: source $SHELL_RC"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    clang-format \
    python3 \
    python3-pip

# Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --user \
    numpy \
    matplotlib \
    pycuda

# Check for cuBLAS
echo ""
echo "Checking for cuBLAS..."
if [ -f "/usr/local/cuda/lib64/libcublas.so" ]; then
    echo "Found cuBLAS"
else
    echo "Installing cuBLAS..."
    sudo apt-get install -y libcublas-12-4 libcublas-dev-12-4
fi

# Verify GPU compute capability
echo ""
echo "Verifying GPU compute capability..."
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "GPU Compute Capability: $GPU_COMPUTE_CAP"

if [ $(echo "$GPU_COMPUTE_CAP >= 8.9" | bc) -eq 1 ]; then
    echo "GPU is compatible (Compute Capability >= 8.9 for RTX 4080)"
else
    echo "WARNING: GPU Compute Capability < 8.9"
    echo "This project is optimized for RTX 4080 (Compute Capability 8.9)"
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build/cuda

# Test CUDA compilation
echo ""
echo "Testing CUDA compilation..."
cat > /tmp/test_cuda.cu << 'EOF'
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU!\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("CUDA test successful!\n");
    return 0;
}
EOF

nvcc /tmp/test_cuda.cu -o /tmp/test_cuda
if /tmp/test_cuda; then
    echo "CUDA compilation test passed!"
else
    echo "WARNING: CUDA compilation test failed"
fi
rm /tmp/test_cuda.cu /tmp/test_cuda

echo ""
echo "========================================"
echo "RTX 4080 setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Source your shell profile: source ~/.bashrc"
echo "  2. Build the project: make cuda"
echo "  3. Run examples: ./build/cuda/dynamic_cortex_demo_cuda"
echo ""
