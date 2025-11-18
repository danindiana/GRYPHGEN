#!/bin/bash
# Setup script for Tenstorrent Greyskull development environment

set -e

echo "========================================"
echo "Dynamic Cortex - Greyskull Setup"
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

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libhwloc-dev \
    pkg-config \
    clang-format \
    doxygen \
    graphviz

# Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --user \
    pybind11 \
    numpy \
    pyyaml \
    matplotlib \
    scipy

# Check for TT-Metal SDK
echo ""
echo "Checking for TT-Metal SDK..."

if [ -z "$TT_METAL_HOME" ]; then
    echo "TT_METAL_HOME not set."
    echo ""
    read -p "Install TT-Metal SDK to ~/tt-metal? [y/N] " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TT_METAL_HOME="$HOME/tt-metal"

        if [ ! -d "$TT_METAL_HOME" ]; then
            echo "Cloning TT-Metal repository..."
            git clone https://github.com/tenstorrent/tt-metal.git $TT_METAL_HOME
        fi

        echo "Installing TT-Metal dependencies..."
        cd $TT_METAL_HOME
        ./install_dependencies.sh

        echo "Building TT-Metal..."
        ./build_metal.sh

        # Add to shell profile
        SHELL_RC="$HOME/.bashrc"
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_RC="$HOME/.zshrc"
        fi

        echo "" >> $SHELL_RC
        echo "# TT-Metal SDK" >> $SHELL_RC
        echo "export TT_METAL_HOME=$TT_METAL_HOME" >> $SHELL_RC
        echo "export PYTHONPATH=\$TT_METAL_HOME:\$PYTHONPATH" >> $SHELL_RC

        echo "Added TT_METAL_HOME to $SHELL_RC"
        echo "Please run: source $SHELL_RC"
    else
        echo "Skipping TT-Metal installation."
        echo "Please install manually and set TT_METAL_HOME"
    fi
else
    echo "Found TT_METAL_HOME: $TT_METAL_HOME"

    if [ ! -d "$TT_METAL_HOME" ]; then
        echo "WARNING: TT_METAL_HOME points to non-existent directory"
    fi
fi

# Check for Greyskull device
echo ""
echo "Checking for Greyskull device..."

if command -v tt-smi &> /dev/null; then
    echo "Running tt-smi..."
    tt-smi
else
    echo "tt-smi not found. Make sure Greyskull driver is installed."
    echo "Visit: https://github.com/tenstorrent/tt-kmd"
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build/greyskull

echo ""
echo "========================================"
echo "Greyskull setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Source your shell profile: source ~/.bashrc"
echo "  2. Build the project: make greyskull"
echo "  3. Run examples: ./build/greyskull/dynamic_cortex_demo_ttmetal"
echo ""
