#!/bin/bash
#
# Setup Named Pipes for Multi-Model Communication
# Creates FIFOs for bidirectional IPC between models
#

set -euo pipefail

VERSION="2.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default pipe locations
PIPE_DIR="${PIPE_DIR:-/tmp}"
PIPE_PREFIX="${PIPE_PREFIX:-model}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Print usage
usage() {
    cat <<EOF
Setup Named Pipes for Multi-Model Communication

Usage: $(basename "$0") [OPTIONS]

Options:
    -d, --dir DIR       Pipe directory (default: /tmp)
    -p, --prefix NAME   Pipe name prefix (default: model)
    -n, --num COUNT     Number of models (default: 2)
    -c, --clean         Clean existing pipes first
    -h, --help          Show this help message
    -v, --version       Show version information

Examples:
    # Create default pipes for 2 models
    ./setup_pipes.sh

    # Create pipes for 4 models with custom location
    ./setup_pipes.sh --num 4 --dir /var/run/pipes

    # Clean and recreate pipes
    ./setup_pipes.sh --clean

Environment Variables:
    PIPE_DIR            Override default pipe directory
    PIPE_PREFIX         Override default pipe prefix

Created Pipes:
    For 2 models:
        \${PIPE_DIR}/model1_to_model2
        \${PIPE_DIR}/model2_to_model1

    For N models:
        \${PIPE_DIR}/modelX_to_modelY (for all X->Y combinations where X≠Y)
EOF
}

# Parse command line arguments
NUM_MODELS=2
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            PIPE_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            PIPE_PREFIX="$2"
            shift 2
            ;;
        -n|--num)
            NUM_MODELS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -v|--version)
            echo "Setup Pipes v$VERSION"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate directory
if [[ ! -d "$PIPE_DIR" ]]; then
    log_error "Directory does not exist: $PIPE_DIR"
    exit 1
fi

if [[ ! -w "$PIPE_DIR" ]]; then
    log_error "Directory is not writable: $PIPE_DIR"
    exit 1
fi

# Clean existing pipes if requested
if [[ "$CLEAN" == "true" ]]; then
    log_info "Cleaning existing pipes..."
    rm -f "${PIPE_DIR}/${PIPE_PREFIX}"*_to_* 2>/dev/null || true
fi

# Create pipes
log_info "Creating named pipes for $NUM_MODELS models in $PIPE_DIR"

created_count=0
skipped_count=0

for ((i=1; i<=NUM_MODELS; i++)); do
    for ((j=1; j<=NUM_MODELS; j++)); do
        if [[ $i -ne $j ]]; then
            pipe_name="${PIPE_DIR}/${PIPE_PREFIX}${i}_to_${PIPE_PREFIX}${j}"

            if [[ -p "$pipe_name" ]]; then
                log_warn "Pipe already exists: $pipe_name"
                ((skipped_count++))
            else
                mkfifo "$pipe_name"
                chmod 666 "$pipe_name"  # Allow all users to read/write
                log_info "Created: $pipe_name"
                ((created_count++))
            fi
        fi
    done
done

# Summary
echo ""
log_info "Setup complete!"
echo "  Created: $created_count pipes"
echo "  Skipped: $skipped_count pipes (already existed)"
echo ""

# List created pipes
log_info "Created pipes:"
ls -lh "${PIPE_DIR}/${PIPE_PREFIX}"*_to_* 2>/dev/null | while read -r line; do
    echo "  $line"
done

# Print environment setup
echo ""
log_info "Environment variables for use:"
echo "  export PIPE_TO_MODEL2=\"${PIPE_DIR}/${PIPE_PREFIX}1_to_${PIPE_PREFIX}2\""
echo "  export PIPE_FROM_MODEL2=\"${PIPE_DIR}/${PIPE_PREFIX}2_to_${PIPE_PREFIX}1\""
echo ""

# Print example usage
log_info "Example usage:"
echo "  # Terminal 1 (Model 2 - must start first to wait for message):"
echo "  $SCRIPT_DIR/model2_comm.pl --verbose"
echo ""
echo "  # Terminal 2 (Model 1):"
echo "  echo 'Hello Model 2' | $SCRIPT_DIR/model1_comm.pl --verbose"
echo ""

exit 0
