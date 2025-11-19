#!/bin/bash
# ShellGenie Setup Script
# Automated setup for development environment

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root"
   exit 1
fi

# Print banner
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════╗
║      ShellGenie Setup Script              ║
║      v2.0.0                               ║
╚═══════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check prerequisites
log_info "Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_success "Python $PYTHON_VERSION found"
else
    log_error "Python 3.10+ is required"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    log_success "pip3 found"
else
    log_error "pip3 is required"
    exit 1
fi

# Check NVIDIA GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log_success "NVIDIA GPU found: $GPU_NAME"
    HAS_GPU=true
else
    log_warn "No NVIDIA GPU found. GPU features will be disabled."
    HAS_GPU=false
fi

# Install Python dependencies
log_info "Installing Python dependencies..."
pip3 install -e . || {
    log_error "Failed to install Python dependencies"
    exit 1
}
log_success "Python dependencies installed"

# Install GPU dependencies if GPU is available
if [ "$HAS_GPU" = true ]; then
    log_info "Installing GPU dependencies..."
    pip3 install -e ".[gpu]" || log_warn "Failed to install GPU dependencies"
fi

# Install development dependencies
log_info "Installing development dependencies..."
pip3 install -e ".[dev]" || log_warn "Failed to install dev dependencies"

# Setup Ollama
log_info "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    log_success "Ollama is installed"
else
    log_info "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh || {
        log_warn "Failed to install Ollama automatically"
        log_info "Please install manually from https://ollama.ai"
    }
fi

# Start Ollama service
log_info "Starting Ollama service..."
if pgrep -x ollama &> /dev/null; then
    log_success "Ollama is already running"
else
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
    if pgrep -x ollama &> /dev/null; then
        log_success "Ollama service started"
    else
        log_warn "Failed to start Ollama service"
    fi
fi

# Pull default model
log_info "Pulling default model (llama3.2)..."
ollama pull llama3.2 || log_warn "Failed to pull model. You can do this manually later."

# Create config directory
log_info "Creating configuration directory..."
mkdir -p ~/.shellgenie
if [ ! -f ~/.shellgenie/config.yaml ]; then
    cp config/config.yaml ~/.shellgenie/config.yaml
    log_success "Configuration file created at ~/.shellgenie/config.yaml"
else
    log_info "Configuration file already exists"
fi

# Setup pre-commit hooks (if dev dependencies installed)
if command -v pre-commit &> /dev/null; then
    log_info "Setting up pre-commit hooks..."
    pre-commit install || log_warn "Failed to setup pre-commit hooks"
fi

# Run tests
log_info "Running tests..."
if pytest tests/ -v --maxfail=1 &> /dev/null; then
    log_success "All tests passed"
else
    log_warn "Some tests failed. This is okay for first setup."
fi

# Print summary
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    Setup completed successfully!          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""
log_info "Next steps:"
echo "  1. Start using ShellGenie:"
echo "     $ shellgenie interactive"
echo ""
echo "  2. Or run a single command:"
echo "     $ shellgenie run \"list all PDF files\""
echo ""
echo "  3. Check system info:"
echo "     $ shellgenie info"
echo ""
log_info "For more information, see:"
echo "  - README.md"
echo "  - docs/ARCHITECTURE.md"
echo "  - docs/SECURITY.md"
echo ""

# Check if shellgenie command is available
if command -v shellgenie &> /dev/null; then
    log_success "shellgenie command is available!"
else
    log_warn "shellgenie command not found in PATH"
    log_info "You may need to restart your shell or run:"
    echo "  $ source ~/.bashrc  # or ~/.zshrc"
fi

log_success "Setup complete! Happy genie-ing! 🧞"
