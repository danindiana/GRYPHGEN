#!/bin/bash

################################################################################
# GRYPHGEN Docker Installation Script
# Version: 2.0
# Description: Installs Docker Engine and NVIDIA Container Toolkit
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
readonly DOCKER_COMPOSE_VERSION="2.24.0"

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
        log_error "Docker installation failed with exit code $exit_code"
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

check_nvidia_driver() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "NVIDIA driver not detected"
        log_warn "Please install CUDA first: sudo bash scripts/install_cuda.sh"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "NVIDIA driver detected"
    fi
}

################################################################################
# Installation Functions
################################################################################

remove_old_docker() {
    log_section "Removing Old Docker Installations"

    local old_packages=(
        docker
        docker-engine
        docker.io
        containerd
        runc
    )

    for pkg in "${old_packages[@]}"; do
        if dpkg -l | grep -q "^ii.*$pkg"; then
            log "Removing old package: $pkg"
            apt-get remove -y -qq "$pkg" || true
        fi
    done

    log_success "Old Docker installations removed"
}

install_docker() {
    log_section "Installing Docker Engine"

    # Install prerequisites
    log "Installing Docker prerequisites..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        ca-certificates \
        curl \
        gnupg \
        lsb-release || {
        log_error "Failed to install prerequisites"
        exit 1
    }

    # Add Docker's official GPG key
    log "Adding Docker GPG key..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Set up Docker repository
    log "Setting up Docker repository..."
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Update package list
    apt-get update -qq

    # Install Docker Engine
    log "Installing Docker Engine (this may take a few minutes)..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin || {
        log_error "Failed to install Docker Engine"
        exit 1
    }

    log_success "Docker Engine installed"
}

install_nvidia_docker() {
    log_section "Installing NVIDIA Container Toolkit"

    # Add NVIDIA Docker repository
    log "Adding NVIDIA Container Toolkit repository..."

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Update package list
    apt-get update -qq

    # Install NVIDIA Container Toolkit
    log "Installing NVIDIA Container Toolkit..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        nvidia-container-toolkit \
        nvidia-container-runtime || {
        log_error "Failed to install NVIDIA Container Toolkit"
        exit 1
    }

    log_success "NVIDIA Container Toolkit installed"
}

configure_docker() {
    log_section "Configuring Docker"

    # Configure NVIDIA runtime
    log "Configuring NVIDIA runtime..."
    nvidia-ctk runtime configure --runtime=docker || {
        log_error "Failed to configure NVIDIA runtime"
        exit 1
    }

    # Configure Docker daemon
    log "Configuring Docker daemon..."
    mkdir -p /etc/docker

    cat > /etc/docker/daemon.json << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "features": {
        "buildkit": true
    }
}
EOF

    log_success "Docker daemon configured"
}

start_docker() {
    log_section "Starting Docker Service"

    # Restart Docker to apply configuration
    systemctl restart docker || {
        log_error "Failed to restart Docker"
        exit 1
    }

    # Enable Docker to start on boot
    systemctl enable docker || {
        log_error "Failed to enable Docker service"
        exit 1
    }

    log_success "Docker service started and enabled"
}

configure_user_access() {
    log_section "Configuring User Access"

    # Get the original user (not root)
    local original_user="${SUDO_USER:-$USER}"

    if [ "$original_user" != "root" ]; then
        # Add user to docker group
        usermod -aG docker "$original_user" || {
            log_error "Failed to add user to docker group"
            exit 1
        }
        log_success "User '$original_user' added to docker group"
        log_warn "User '$original_user' needs to log out and back in for group changes to take effect"
    fi
}

verify_installation() {
    log_section "Verifying Docker Installation"

    # Check Docker version
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version)
        log_success "Docker installed: $docker_version"
    else
        log_error "Docker not found"
        return 1
    fi

    # Check Docker Compose
    if docker compose version &> /dev/null; then
        local compose_version=$(docker compose version)
        log_success "Docker Compose installed: $compose_version"
    else
        log_error "Docker Compose not found"
        return 1
    fi

    # Check Docker service
    if systemctl is-active --quiet docker; then
        log_success "Docker service is running"
    else
        log_error "Docker service is not running"
        return 1
    fi

    # Check NVIDIA runtime
    if docker info | grep -q "nvidia"; then
        log_success "NVIDIA runtime configured"
    else
        log_error "NVIDIA runtime not configured"
        return 1
    fi

    # Test Docker with NVIDIA runtime
    log "Testing Docker with NVIDIA GPU access..."
    if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_success "Docker can access NVIDIA GPU"
    else
        log_warn "Failed to access GPU in Docker container"
        log_warn "You may need to restart the system"
    fi

    log_success "Docker installation verified"
    return 0
}

create_docker_scripts() {
    log_section "Creating Helper Scripts"

    # Create GPU test script
    cat > /usr/local/bin/test-docker-gpu << 'EOF'
#!/bin/bash
echo "Testing Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
EOF
    chmod +x /usr/local/bin/test-docker-gpu

    log_success "Helper scripts created: test-docker-gpu"
}

################################################################################
# Main Installation
################################################################################

main() {
    log_section "GRYPHGEN Docker Installation"
    log "Installation started at $(date)"

    # Pre-installation checks
    check_root
    check_nvidia_driver

    # Remove old installations
    remove_old_docker

    # Install Docker
    install_docker
    install_nvidia_docker

    # Configure Docker
    configure_docker
    start_docker
    configure_user_access

    # Create helper scripts
    create_docker_scripts

    # Verify installation
    verify_installation

    log_section "Installation Summary"
    log_success "Docker installation completed successfully!"
    log ""
    log "Docker version: $(docker --version)"
    log "Docker Compose version: $(docker compose version)"
    log ""
    log "Next steps:"
    log "  1. Log out and log back in (for group changes)"
    log "  2. Test Docker: docker run hello-world"
    log "  3. Test GPU: test-docker-gpu"
    log "  4. Install Python: bash scripts/install_python.sh"
    log ""
    log "Quick commands:"
    log "  • docker ps                 - List running containers"
    log "  • docker images             - List images"
    log "  • docker-compose up -d      - Start services"
    log "  • nvidia-smi                - Check GPU status"
}

# Run main function
main "$@"
