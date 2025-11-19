#!/bin/bash

################################################################################
# GRYPHGEN Base System Installation Script
# Version: 2.0
# Target: Ubuntu 22.04/24.04 LTS
# Description: Installs core system dependencies for GRYPHGEN framework
################################################################################

set -euo pipefail  # Exit on error, undefined variables, and pipe failures
IFS=$'\n\t'        # Set Internal Field Separator for safer script execution

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Log file
readonly LOG_FILE="/var/log/gryphgen_install.log"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
        log_error "Installation failed with exit code $exit_code"
        log_error "Check log file: $LOG_FILE"
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

check_ubuntu() {
    if [ ! -f /etc/os-release ]; then
        log_error "Cannot determine OS version"
        exit 1
    fi

    source /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        log_error "This script is designed for Ubuntu. Detected: $ID"
        exit 1
    fi

    local version_major="${VERSION_ID%%.*}"
    if [ "$version_major" -lt 22 ]; then
        log_error "Ubuntu 22.04 or later is required. Detected: $VERSION_ID"
        exit 1
    fi

    log_success "Running on Ubuntu $VERSION_ID"
}

check_disk_space() {
    local required_gb=50
    local available_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        exit 1
    fi

    log_success "Sufficient disk space: ${available_gb}GB available"
}

check_memory() {
    local required_gb=16
    local total_gb=$(free -g | awk '/^Mem:/{print $2}')

    if [ "$total_gb" -lt "$required_gb" ]; then
        log_warn "Low memory detected. Recommended: ${required_gb}GB, Available: ${total_gb}GB"
    else
        log_success "Sufficient memory: ${total_gb}GB"
    fi
}

################################################################################
# Installation Functions
################################################################################

update_system() {
    log_section "Updating System Packages"

    # Update package lists
    apt-get update -qq || {
        log_error "Failed to update package lists"
        exit 1
    }

    # Upgrade existing packages
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq || {
        log_error "Failed to upgrade packages"
        exit 1
    }

    log_success "System packages updated"
}

install_essentials() {
    log_section "Installing Essential Build Tools"

    local packages=(
        build-essential
        curl
        wget
        git
        vim
        htop
        tmux
        ca-certificates
        gnupg
        lsb-release
        software-properties-common
        apt-transport-https
        libssl-dev
        libffi-dev
        zlib1g-dev
        libbz2-dev
        libreadline-dev
        libsqlite3-dev
        llvm
        libncurses5-dev
        libncursesw5-dev
        xz-utils
        tk-dev
        liblzma-dev
        jq
        unzip
        tree
    )

    log "Installing: ${packages[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${packages[@]}" || {
        log_error "Failed to install essential packages"
        exit 1
    }

    log_success "Essential build tools installed"
}

install_zeromq() {
    log_section "Installing ZeroMQ"

    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libzmq3-dev \
        libzmq5 || {
        log_error "Failed to install ZeroMQ"
        exit 1
    }

    log_success "ZeroMQ installed"
}

install_monitoring_tools() {
    log_section "Installing Monitoring Tools"

    # Install Prometheus (if not already installed)
    if ! command -v prometheus &> /dev/null; then
        log "Installing Prometheus..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq prometheus || {
            log_warn "Failed to install Prometheus from apt, will skip"
        }
    else
        log_success "Prometheus already installed"
    fi

    # Install node_exporter for system metrics
    if ! command -v node_exporter &> /dev/null; then
        log "Installing Node Exporter..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq prometheus-node-exporter || {
            log_warn "Failed to install Node Exporter, will skip"
        }
    else
        log_success "Node Exporter already installed"
    fi

    log_success "Monitoring tools configured"
}

configure_system() {
    log_section "Configuring System Settings"

    # Increase file descriptors limit
    if ! grep -q "fs.file-max" /etc/sysctl.conf; then
        echo "fs.file-max = 2097152" >> /etc/sysctl.conf
        sysctl -p &> /dev/null || true
        log_success "File descriptor limit increased"
    fi

    # Configure swap (if needed)
    local swap_size=$(free -g | awk '/^Swap:/{print $2}')
    if [ "$swap_size" -lt 8 ]; then
        log_warn "Swap size is low: ${swap_size}GB. Consider increasing swap space"
    fi

    # Create GRYPHGEN directories
    mkdir -p /opt/gryphgen
    mkdir -p /data/gryphgen
    mkdir -p /models/gryphgen
    mkdir -p /var/log/gryphgen

    log_success "System configured"
}

setup_environment_variables() {
    log_section "Setting Up Environment Variables"

    local env_file="/etc/profile.d/gryphgen.sh"

    cat > "$env_file" << 'EOF'
# GRYPHGEN Environment Variables
export GRYPHGEN_HOME=/opt/gryphgen
export GRYPHGEN_DATA=/data/gryphgen
export GRYPHGEN_MODELS=/models/gryphgen
export GRYPHGEN_LOGS=/var/log/gryphgen

# Add to PATH if needed
if [ -d "$GRYPHGEN_HOME/bin" ]; then
    export PATH="$GRYPHGEN_HOME/bin:$PATH"
fi
EOF

    chmod +x "$env_file"
    source "$env_file"

    log_success "Environment variables configured"
}

################################################################################
# Main Installation
################################################################################

main() {
    log_section "GRYPHGEN Base System Installation"
    log "Installation started at $(date)"
    log "Log file: $LOG_FILE"

    # Pre-installation checks
    check_root
    check_ubuntu
    check_disk_space
    check_memory

    # System updates
    update_system

    # Install packages
    install_essentials
    install_zeromq
    install_monitoring_tools

    # Configure system
    configure_system
    setup_environment_variables

    log_section "Installation Summary"
    log_success "Base system installation completed successfully!"
    log ""
    log "Next steps:"
    log "  1. Install CUDA: sudo bash scripts/install_cuda.sh"
    log "  2. Install Docker: sudo bash scripts/install_docker.sh"
    log "  3. Install Python: bash scripts/install_python.sh"
    log "  4. Install LLM components: bash scripts/install_llm.sh"
    log ""
    log "Note: You may need to log out and log back in for group changes to take effect"
}

# Run main function
main "$@"
