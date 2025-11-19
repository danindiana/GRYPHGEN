#!/bin/bash

################################################################################
# GRYPHGEN LLM Components Installation Script
# Version: 2.0
# PyTorch Version: 2.5+ with CUDA 12.6
# Description: Installs PyTorch, Transformers, and LLM dependencies
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
readonly VENV_PATH="/opt/gryphgen/venv"
readonly MODELS_PATH="/models/gryphgen"

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
        log_error "LLM installation failed with exit code $exit_code"
    fi
    exit $exit_code
}

trap cleanup EXIT ERR

################################################################################
# System Checks
################################################################################

check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA driver not found. Please install CUDA first."
        exit 1
    fi

    local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 || echo "Not found")
    if [ "$cuda_version" != "Not found" ]; then
        log_success "CUDA detected: $cuda_version"
    else
        log_warn "CUDA compiler not found in PATH"
    fi
}

check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        log_error "Virtual environment not found at $VENV_PATH"
        log_error "Please run: bash scripts/install_python.sh"
        exit 1
    fi
    log_success "Virtual environment found"
}

################################################################################
# Installation Functions
################################################################################

install_pytorch() {
    log_section "Installing PyTorch 2.5+ with CUDA 12.6"

    source "$VENV_PATH/bin/activate"

    # Install PyTorch with CUDA 12.6 support
    log "Installing PyTorch (this may take several minutes)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 || {
        log_error "Failed to install PyTorch"
        exit 1
    }

    log_success "PyTorch installed"
}

install_transformers() {
    log_section "Installing Hugging Face Transformers"

    source "$VENV_PATH/bin/activate"

    local packages=(
        transformers
        accelerate
        bitsandbytes
        sentencepiece
        protobuf
        tokenizers
    )

    log "Installing Transformers ecosystem..."
    pip install "${packages[@]}" || {
        log_error "Failed to install Transformers"
        exit 1
    }

    log_success "Transformers installed"
}

install_llm_frameworks() {
    log_section "Installing LLM Frameworks"

    source "$VENV_PATH/bin/activate"

    local packages=(
        langchain
        langchain-community
        langchain-openai
        llama-index
        llama-cpp-python
        chromadb
        faiss-gpu
        sentence-transformers
        einops
        xformers
    )

    log "Installing LLM frameworks..."
    pip install "${packages[@]}" || {
        log_warn "Some LLM framework packages failed to install"
    }

    log_success "LLM frameworks installed"
}

install_vector_databases() {
    log_section "Installing Vector Databases"

    source "$VENV_PATH/bin/activate"

    local packages=(
        chromadb
        pymilvus
        qdrant-client
        weaviate-client
        pinecone-client
    )

    log "Installing vector database clients..."
    pip install "${packages[@]}" || {
        log_warn "Some vector database packages failed to install"
    }

    log_success "Vector database clients installed"
}

install_api_frameworks() {
    log_section "Installing API Frameworks"

    source "$VENV_PATH/bin/activate"

    local packages=(
        flask
        flask-restful
        fastapi
        uvicorn
        gunicorn
        python-multipart
        websockets
        aiofiles
        httpx
    )

    log "Installing API frameworks..."
    pip install "${packages[@]}" || {
        log_error "Failed to install API frameworks"
        exit 1
    }

    log_success "API frameworks installed"
}

install_database_connectors() {
    log_section "Installing Database Connectors"

    source "$VENV_PATH/bin/activate"

    local packages=(
        pymongo
        psycopg2-binary
        redis
        sqlalchemy
        alembic
    )

    log "Installing database connectors..."
    pip install "${packages[@]}" || {
        log_warn "Some database connectors failed to install"
    }

    log_success "Database connectors installed"
}

install_monitoring_tools() {
    log_section "Installing Monitoring Tools"

    source "$VENV_PATH/bin/activate"

    local packages=(
        prometheus-client
        opentelemetry-api
        opentelemetry-sdk
        tensorboard
        wandb
    )

    log "Installing monitoring tools..."
    pip install "${packages[@]}" || {
        log_warn "Some monitoring tools failed to install"
    }

    log_success "Monitoring tools installed"
}

create_model_directories() {
    log_section "Creating Model Directories"

    mkdir -p "$MODELS_PATH"/{llama,mistral,codellama,embeddings,custom}
    mkdir -p /data/gryphgen/{vector_db,cache,logs}

    log_success "Model directories created"
}

create_requirements_files() {
    log_section "Creating Requirements Files"

    source "$VENV_PATH/bin/activate"

    # Export all installed packages
    pip freeze > /opt/gryphgen/requirements-full.txt

    # Create GPU-specific requirements
    cat > /opt/gryphgen/requirements-gpu.txt << 'EOF'
# PyTorch with CUDA 12.6
torch>=2.5.0
torchvision>=0.20.0
torchaudio>=2.5.0

# Transformers ecosystem
transformers>=4.50.0
accelerate>=1.2.0
bitsandbytes>=0.45.0
sentencepiece>=0.2.0
tokenizers>=0.20.0

# LLM frameworks
langchain>=0.3.0
langchain-community>=0.3.0
llama-index>=0.12.0
sentence-transformers>=3.3.0
xformers>=0.0.28

# Vector databases
chromadb>=0.5.0
faiss-gpu>=1.9.0
pymilvus>=2.4.0
qdrant-client>=1.12.0

# API frameworks
fastapi>=0.115.0
uvicorn>=0.32.0
websockets>=14.0

# Database connectors
pymongo>=4.10.0
psycopg2-binary>=2.9.0
redis>=5.2.0
sqlalchemy>=2.0.0

# Utilities
pyzmq>=26.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.67.0
rich>=13.9.0
EOF

    log_success "Requirements files created"
}

configure_pytorch() {
    log_section "Configuring PyTorch for RTX 4080"

    source "$VENV_PATH/bin/activate"

    # Create PyTorch configuration script
    cat > /opt/gryphgen/pytorch_config.py << 'EOF'
"""
PyTorch Configuration for NVIDIA RTX 4080
Optimizes performance for Ada Lovelace architecture
"""

import torch
import os

def configure_pytorch():
    """Configure PyTorch for optimal RTX 4080 performance"""

    # Enable TF32 for better performance on Ampere/Ada GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True

    # Set memory allocator configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("✗ CUDA not available")

    return torch.cuda.is_available()

if __name__ == "__main__":
    configure_pytorch()
EOF

    log_success "PyTorch configuration created"
}

verify_installation() {
    log_section "Verifying LLM Installation"

    source "$VENV_PATH/bin/activate"

    # Test PyTorch installation
    log "Testing PyTorch..."
    python << 'EOF'
import torch
import sys

if not torch.cuda.is_available():
    print("✗ CUDA not available in PyTorch")
    sys.exit(1)

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Test basic operations
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f"✓ GPU computation successful")
EOF

    if [ $? -eq 0 ]; then
        log_success "PyTorch verification passed"
    else
        log_error "PyTorch verification failed"
        return 1
    fi

    # Test Transformers
    log "Testing Transformers..."
    python -c "import transformers; print(f'✓ Transformers version: {transformers.__version__}')" || {
        log_error "Transformers test failed"
        return 1
    }

    log_success "LLM installation verified"
    return 0
}

################################################################################
# Main Installation
################################################################################

main() {
    log_section "GRYPHGEN LLM Components Installation"
    log "PyTorch: 2.5+ with CUDA 12.6"
    log "Target: NVIDIA RTX 4080 16GB"
    log "Installation started at $(date)"

    # Pre-installation checks
    check_cuda
    check_venv

    # Install components
    install_pytorch
    install_transformers
    install_llm_frameworks
    install_vector_databases
    install_api_frameworks
    install_database_connectors
    install_monitoring_tools

    # Setup
    create_model_directories
    create_requirements_files
    configure_pytorch

    # Verify
    verify_installation

    log_section "Installation Summary"
    log_success "LLM components installation completed successfully!"
    log ""
    log "Installed components:"
    log "  • PyTorch 2.5+ with CUDA 12.6"
    log "  • Hugging Face Transformers"
    log "  • LangChain & LlamaIndex"
    log "  • Vector databases (ChromaDB, FAISS)"
    log "  • API frameworks (FastAPI, Flask)"
    log ""
    log "Next steps:"
    log "  1. Activate environment: gryphgen-activate"
    log "  2. Test PyTorch: python /opt/gryphgen/pytorch_config.py"
    log "  3. Download models: huggingface-cli login"
    log "  4. Run validation: bash scripts/validate_environment.sh"
    log ""
    log "Model directories:"
    log "  • LLaMA: $MODELS_PATH/llama"
    log "  • Mistral: $MODELS_PATH/mistral"
    log "  • CodeLLaMA: $MODELS_PATH/codellama"
    log "  • Embeddings: $MODELS_PATH/embeddings"
    log ""
}

# Run main function
main "$@"
