#!/bin/bash
# Build script for all backends

set -e

echo "========================================"
echo "Dynamic Cortex - Building All Backends"
echo "========================================"
echo ""

# Check for required tools
echo "Checking build requirements..."

MISSING_TOOLS=0

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found"
    MISSING_TOOLS=1
fi

if ! command -v make &> /dev/null; then
    echo "ERROR: make not found"
    MISSING_TOOLS=1
fi

if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found"
    MISSING_TOOLS=1
fi

if [ $MISSING_TOOLS -eq 1 ]; then
    echo ""
    echo "Please install missing tools and try again"
    exit 1
fi

echo "All required tools found!"

# Determine which backends to build
BUILD_GREYSKULL=0
BUILD_CUDA=0

if [ -n "$TT_METAL_HOME" ] && [ -d "$TT_METAL_HOME" ]; then
    echo "TT-Metal SDK found: $TT_METAL_HOME"
    BUILD_GREYSKULL=1
else
    echo "TT-Metal SDK not found (TT_METAL_HOME not set or invalid)"
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "CUDA found: version $CUDA_VERSION"
    BUILD_CUDA=1
else
    echo "CUDA not found"
fi

if [ $BUILD_GREYSKULL -eq 0 ] && [ $BUILD_CUDA -eq 0 ]; then
    echo ""
    echo "ERROR: No backends available to build"
    echo "Please install TT-Metal SDK or CUDA Toolkit"
    exit 1
fi

echo ""
echo "Will build:"
[ $BUILD_GREYSKULL -eq 1 ] && echo "  - Greyskull (TT-Metal)"
[ $BUILD_CUDA -eq 1 ] && echo "  - RTX 4080 (CUDA)"
echo ""

# Build Greyskull
if [ $BUILD_GREYSKULL -eq 1 ]; then
    echo "========================================"
    echo "Building Greyskull backend..."
    echo "========================================"
    make greyskull
    echo ""
fi

# Build CUDA
if [ $BUILD_CUDA -eq 1 ]; then
    echo "========================================"
    echo "Building CUDA backend..."
    echo "========================================"
    make cuda
    echo ""
fi

echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Built artifacts:"
[ $BUILD_GREYSKULL -eq 1 ] && echo "  Greyskull: build/greyskull/"
[ $BUILD_CUDA -eq 1 ] && echo "  CUDA: build/cuda/"
echo ""
echo "To run examples:"
[ $BUILD_GREYSKULL -eq 1 ] && echo "  ./build/greyskull/dynamic_cortex_demo_ttmetal"
[ $BUILD_CUDA -eq 1 ] && echo "  ./build/cuda/dynamic_cortex_demo_cuda"
echo ""
