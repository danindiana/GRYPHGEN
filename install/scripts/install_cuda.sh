#!/bin/bash

################################################################################
# GRYPHGEN CUDA Installation Script
# Version: 2.0
# Target: NVIDIA RTX 4080 16GB (Ada Lovelace Architecture)
# CUDA Version: 12.6
# Description: Installs NVIDIA drivers, CUDA toolkit, and cuDNN
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
readonly LOG_FILE="/var/log/gryphgen_install.log"
readonly CUDA_VERSION="12.6"
readonly DRIVER_VERSION="560"  # Minimum driver version for CUDA 12.6
readonly CUDNN_VERSION="9"

################################################################################
# Logging Functions
################################################################################

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*" | tee -a "$LOG_FILE" >&2
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}  $*${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Error Handling
################################################################################

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "CUDA installation failed with exit code $exit_code"
    fi
    exit $exit_code
}

trap cleanup EXIT ERR

################################################################################
# System Checks
################################################################################

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root or with sudo"
        exit 1
    fi
}

check_nvidia_gpu() {
    log_section "Checking for NVIDIA GPU"

    if ! lspci | grep -i nvidia &> /dev/null; then
        log_error "No NVIDIA GPU detected"
        exit 1
    fi

    local gpu_info=$(lspci | grep -i nvidia | head -1)
    log_success "NVIDIA GPU detected: $gpu_info"

    # Check specifically for RTX 4080
    if lspci | grep -i "RTX 4080" &> /dev/null; then
        log_success "NVIDIA RTX 4080 detected - optimal configuration will be applied"
    else
        log_warn "Target GPU is RTX 4080, but different GPU detected"
        log_warn "Installation will continue with generic settings"
    fi
}

check_nouveau() {
    log_section "Checking for Nouveau Driver"

    if lsmod | grep nouveau &> /dev/null; then
        log_warn "Nouveau driver is loaded and must be disabled"
        disable_nouveau
    else
        log_success "Nouveau driver not loaded"
    fi
}

disable_nouveau() {
    log "Disabling Nouveau driver..."

    cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF

    update-initramfs -u

    log_warn "Nouveau driver blacklisted. A reboot is required."
    log_warn "Please reboot and run this script again."
    exit 0
}

################################################################################
# Installation Functions
################################################################################

install_nvidia_driver() {
    log_section "Installing NVIDIA Driver"

    # Check if driver is already installed
    if command -v nvidia-smi &> /dev/null; then
        local current_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        log "Current NVIDIA driver version: $current_version"

        # Extract major version
        local current_major="${current_version%%.*}"
        if [ "$current_major" -ge "$DRIVER_VERSION" ]; then
            log_success "NVIDIA driver version $current_version is sufficient"
            return 0
        else
            log_warn "NVIDIA driver version $current_version is outdated"
        fi
    fi

    # Add NVIDIA driver PPA
    log "Adding NVIDIA driver PPA..."
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt-get update -qq

    # Install latest NVIDIA driver
    log "Installing NVIDIA driver ${DRIVER_VERSION}..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        nvidia-driver-${DRIVER_VERSION} \
        nvidia-dkms-${DRIVER_VERSION} || {
        log_error "Failed to install NVIDIA driver"
        exit 1
    }

    log_success "NVIDIA driver installed"
    log_warn "A reboot is required for the driver to take effect"
    log_warn "Please reboot and run this script again to continue with CUDA installation"

    read -p "Reboot now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        reboot
    fi
    exit 0
}

install_cuda_toolkit() {
    log_section "Installing CUDA Toolkit ${CUDA_VERSION}"

    # Remove old CUDA keyring if exists
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null || true

    # Download and install CUDA keyring
    log "Adding CUDA repository..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb

    # Update package list
    apt-get update -qq

    # Install CUDA toolkit
    log "Installing CUDA ${CUDA_VERSION} (this may take several minutes)..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        cuda-toolkit-12-6 \
        cuda-drivers || {
        log_error "Failed to install CUDA toolkit"
        exit 1
    }

    log_success "CUDA ${CUDA_VERSION} toolkit installed"
}

install_cudnn() {
    log_section "Installing cuDNN ${CUDNN_VERSION}"

    # Install cuDNN
    log "Installing cuDNN ${CUDNN_VERSION}..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libcudnn${CUDNN_VERSION}-cuda-12 \
        libcudnn${CUDNN_VERSION}-dev-cuda-12 || {
        log_warn "Failed to install cuDNN from repository"
        log_warn "You may need to install cuDNN manually from NVIDIA"
        return 1
    }

    log_success "cuDNN ${CUDNN_VERSION} installed"
}

install_cuda_samples() {
    log_section "Installing CUDA Samples"

    # Install CUDA samples for testing
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        cuda-samples-12-6 || {
        log_warn "Failed to install CUDA samples"
        return 1
    }

    log_success "CUDA samples installed"
}

configure_cuda_environment() {
    log_section "Configuring CUDA Environment"

    local cuda_profile="/etc/profile.d/cuda.sh"

    cat > "$cuda_profile" << 'EOF'
# CUDA Environment Variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# CUDA Architecture for RTX 4080 (Ada Lovelace - sm_89)
export TORCH_CUDA_ARCH_LIST="8.9"

# Performance optimizations
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_CACHE_DISABLE=0
EOF

    chmod +x "$cuda_profile"
    source "$cuda_profile"

    # Create symbolic link to latest CUDA version
    if [ ! -L /usr/local/cuda ]; then
        ln -sf /usr/local/cuda-12.6 /usr/local/cuda
    fi

    log_success "CUDA environment configured"
}

optimize_for_rtx4080() {
    log_section "Applying RTX 4080 Optimizations"

    # Set power limit (default 320W for RTX 4080)
    log "Setting GPU power limit to 320W..."
    nvidia-smi -pl 320 || log_warn "Failed to set power limit"

    # Enable persistence mode
    log "Enabling GPU persistence mode..."
    nvidia-smi -pm 1 || log_warn "Failed to enable persistence mode"

    # Set compute mode to default (allows multiple processes)
    log "Setting compute mode..."
    nvidia-smi -c 0 || log_warn "Failed to set compute mode"

    log_success "RTX 4080 optimizations applied"
}

verify_installation() {
    log_section "Verifying CUDA Installation"

    # Check NVIDIA driver
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA driver installed"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$LOG_FILE"
    else
        log_error "NVIDIA driver not found"
        return 1
    fi

    # Check CUDA compiler
    if command -v nvcc &> /dev/null; then
        local nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_success "CUDA compiler (nvcc) version: $nvcc_version"
    else
        log_error "CUDA compiler (nvcc) not found"
        return 1
    fi

    # Check CUDA libraries
    if [ -d "/usr/local/cuda/lib64" ]; then
        log_success "CUDA libraries found"
    else
        log_error "CUDA libraries not found"
        return 1
    fi

    # Check cuDNN
    if ldconfig -p | grep -q libcudnn; then
        log_success "cuDNN libraries found"
    else
        log_warn "cuDNN libraries not found"
    fi

    log_success "CUDA installation verified"
    return 0
}

################################################################################
# Main Installation
################################################################################

main() {
    log_section "GRYPHGEN CUDA Installation"
    log "Target: NVIDIA RTX 4080 16GB"
    log "CUDA Version: ${CUDA_VERSION}"
    log "Installation started at $(date)"

    # Pre-installation checks
    check_root
    check_nvidia_gpu
    check_nouveau

    # Check if driver is installed
    if ! command -v nvidia-smi &> /dev/null; then
        install_nvidia_driver
    else
        log_success "NVIDIA driver already installed"
    fi

    # Install CUDA components
    install_cuda_toolkit
    install_cudnn
    install_cuda_samples

    # Configure environment
    configure_cuda_environment

    # Optimize for RTX 4080
    if lspci | grep -i "RTX 4080" &> /dev/null; then
        optimize_for_rtx4080
    fi

    # Verify installation
    verify_installation

    log_section "Installation Summary"
    log_success "CUDA installation completed successfully!"
    log ""
    log "Next steps:"
    log "  1. Source the environment: source /etc/profile.d/cuda.sh"
    log "  2. Install Docker: sudo bash scripts/install_docker.sh"
    log "  3. Verify GPU: nvidia-smi"
    log "  4. Test CUDA: nvcc --version"
    log ""
    log "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
}

# Run main function
main "$@"
