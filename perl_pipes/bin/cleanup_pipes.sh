#!/bin/bash
#
# Cleanup Named Pipes for Multi-Model Communication
# Removes FIFOs created by setup_pipes.sh
#

set -euo pipefail

VERSION="2.0.0"

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
Cleanup Named Pipes for Multi-Model Communication

Usage: $(basename "$0") [OPTIONS]

Options:
    -d, --dir DIR       Pipe directory (default: /tmp)
    -p, --prefix NAME   Pipe name prefix (default: model)
    -f, --force         Force removal without confirmation
    -h, --help          Show this help message
    -v, --version       Show version information

Examples:
    # Remove default pipes
    ./cleanup_pipes.sh

    # Remove pipes with custom location
    ./cleanup_pipes.sh --dir /var/run/pipes

    # Force removal without confirmation
    ./cleanup_pipes.sh --force

Environment Variables:
    PIPE_DIR            Override default pipe directory
    PIPE_PREFIX         Override default pipe prefix
EOF
}

# Parse command line arguments
FORCE=false

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
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -v|--version)
            echo "Cleanup Pipes v$VERSION"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Find pipes to remove
PIPES=()
while IFS= read -r -d '' pipe; do
    PIPES+=("$pipe")
done < <(find "$PIPE_DIR" -maxdepth 1 -type p -name "${PIPE_PREFIX}*_to_*" -print0 2>/dev/null || true)

# Check if any pipes found
if [[ ${#PIPES[@]} -eq 0 ]]; then
    log_warn "No pipes found matching pattern: ${PIPE_DIR}/${PIPE_PREFIX}*_to_*"
    exit 0
fi

# List pipes to be removed
log_info "Found ${#PIPES[@]} pipe(s) to remove:"
for pipe in "${PIPES[@]}"; do
    echo "  $pipe"
done
echo ""

# Confirm removal unless forced
if [[ "$FORCE" != "true" ]]; then
    read -p "Remove these pipes? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 0
    fi
fi

# Remove pipes
removed_count=0
failed_count=0

for pipe in "${PIPES[@]}"; do
    if rm -f "$pipe"; then
        log_info "Removed: $pipe"
        ((removed_count++))
    else
        log_error "Failed to remove: $pipe"
        ((failed_count++))
    fi
done

# Summary
echo ""
log_info "Cleanup complete!"
echo "  Removed: $removed_count pipes"
if [[ $failed_count -gt 0 ]]; then
    echo "  Failed:  $failed_count pipes"
fi

exit 0
