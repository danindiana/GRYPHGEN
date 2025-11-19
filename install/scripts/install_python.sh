#!/bin/bash

################################################################################
# GRYPHGEN Python Environment Installation Script
# Version: 2.0
# Python Version: 3.11+
# Description: Installs Python, pip, and sets up virtual environments
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
readonly PYTHON_VERSION="3.11"
readonly VENV_PATH="/opt/gryphgen/venv"

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
        log_error "Python installation failed with exit code $exit_code"
    fi
    exit $exit_code
}

trap cleanup EXIT ERR

################################################################################
# Installation Functions
################################################################################

install_python() {
    log_section "Installing Python ${PYTHON_VERSION}"

    # Add deadsnakes PPA for latest Python versions
    log "Adding deadsnakes PPA..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq

    # Install Python and development tools
    log "Installing Python ${PYTHON_VERSION} and development tools..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        python3-setuptools \
        python3-wheel || {
        log_error "Failed to install Python"
        exit 1
    }

    # Set Python alternatives
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

    log_success "Python ${PYTHON_VERSION} installed"
}

upgrade_pip() {
    log_section "Upgrading pip"

    python3 -m pip install --upgrade pip setuptools wheel || {
        log_error "Failed to upgrade pip"
        exit 1
    }

    log_success "pip upgraded to latest version"
}

install_build_dependencies() {
    log_section "Installing Python Build Dependencies"

    local packages=(
        build-essential
        libssl-dev
        libffi-dev
        libbz2-dev
        libreadline-dev
        libsqlite3-dev
        libncurses5-dev
        libncursesw5-dev
        liblzma-dev
        tk-dev
        llvm
        xz-utils
    )

    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${packages[@]}" || {
        log_error "Failed to install build dependencies"
        exit 1
    }

    log_success "Build dependencies installed"
}

create_virtual_environment() {
    log_section "Creating Virtual Environment"

    # Create virtual environment
    mkdir -p "$(dirname "$VENV_PATH")"

    if [ -d "$VENV_PATH" ]; then
        log_warn "Virtual environment already exists at $VENV_PATH"
        read -p "Remove and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
        else
            log "Keeping existing virtual environment"
            return 0
        fi
    fi

    log "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH" || {
        log_error "Failed to create virtual environment"
        exit 1
    }

    # Activate and upgrade pip in venv
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel

    log_success "Virtual environment created at $VENV_PATH"
}

install_base_packages() {
    log_section "Installing Base Python Packages"

    source "$VENV_PATH/bin/activate"

    local packages=(
        numpy
        scipy
        pandas
        matplotlib
        seaborn
        scikit-learn
        ipython
        jupyter
        notebook
        jupyterlab
        requests
        aiohttp
        fastapi
        uvicorn
        pydantic
        python-dotenv
        pyyaml
        toml
        click
        tqdm
        rich
        pyzmq
    )

    log "Installing base packages..."
    pip install "${packages[@]}" || {
        log_error "Failed to install base packages"
        exit 1
    }

    log_success "Base packages installed"
}

setup_jupyter() {
    log_section "Setting Up Jupyter"

    source "$VENV_PATH/bin/activate"

    # Install Jupyter kernel
    python -m ipykernel install --user --name=gryphgen --display-name="GRYPHGEN" || {
        log_warn "Failed to install Jupyter kernel"
        return 1
    }

    # Create Jupyter configuration
    mkdir -p ~/.jupyter

    cat > ~/.jupyter/jupyter_notebook_config.py << 'EOF'
# Jupyter Notebook Configuration
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = False
EOF

    log_success "Jupyter configured"
}

create_activation_script() {
    log_section "Creating Activation Script"

    cat > /usr/local/bin/gryphgen-activate << EOF
#!/bin/bash
source ${VENV_PATH}/bin/activate
echo "GRYPHGEN Python environment activated"
echo "Python: \$(python --version)"
echo "pip: \$(pip --version)"
echo ""
echo "To deactivate: deactivate"
EOF

    chmod +x /usr/local/bin/gryphgen-activate

    log_success "Activation script created: gryphgen-activate"
}

verify_installation() {
    log_section "Verifying Python Installation"

    # Check Python version
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version)
        log_success "Python installed: $python_version"
    else
        log_error "Python not found"
        return 1
    fi

    # Check pip
    if command -v pip3 &> /dev/null; then
        local pip_version=$(pip3 --version)
        log_success "pip installed: $pip_version"
    else
        log_error "pip not found"
        return 1
    fi

    # Check virtual environment
    if [ -d "$VENV_PATH" ]; then
        log_success "Virtual environment exists: $VENV_PATH"
    else
        log_error "Virtual environment not found"
        return 1
    fi

    # Test imports in venv
    source "$VENV_PATH/bin/activate"

    local test_imports=(
        "numpy"
        "pandas"
        "torch"
        "zmq"
    )

    for pkg in "${test_imports[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            log_success "Package $pkg can be imported"
        else
            log_warn "Package $pkg not available (may need LLM installation)"
        fi
    done

    log_success "Python installation verified"
    return 0
}

################################################################################
# Main Installation
################################################################################

main() {
    log_section "GRYPHGEN Python Environment Installation"
    log "Target Python Version: ${PYTHON_VERSION}"
    log "Virtual Environment: $VENV_PATH"
    log "Installation started at $(date)"

    # Install Python
    install_build_dependencies
    install_python
    upgrade_pip

    # Setup environment
    create_virtual_environment
    install_base_packages
    setup_jupyter
    create_activation_script

    # Verify installation
    verify_installation

    log_section "Installation Summary"
    log_success "Python environment installation completed successfully!"
    log ""
    log "Python version: $(python3 --version)"
    log "pip version: $(pip3 --version)"
    log "Virtual environment: $VENV_PATH"
    log ""
    log "Next steps:"
    log "  1. Activate environment: gryphgen-activate"
    log "  2. Install PyTorch/LLM packages: bash scripts/install_llm.sh"
    log "  3. Test environment: python -c 'import sys; print(sys.version)'"
    log ""
    log "Quick commands:"
    log "  • gryphgen-activate         - Activate virtual environment"
    log "  • deactivate                - Deactivate virtual environment"
    log "  • pip list                  - List installed packages"
    log "  • jupyter lab               - Start JupyterLab"
}

# Run main function
main "$@"
