#!/bin/bash
#
# Find GPU-Compatible Models Example
# Searches for GGUF models that fit in NVIDIA RTX 4080 16GB
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/../bin"

echo "=== GPU-Compatible Model Finder ==="
echo "Target: NVIDIA RTX 4080 (16GB VRAM)"
echo ""

# Search common model directories
SEARCH_DIRS=(
    "$HOME/models"
    "$HOME/.cache/lm-studio"
    "/mnt/models"
    "/opt/models"
)

echo "Searching for GGUF models in common locations..."
echo ""

for dir in "${SEARCH_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "Searching: $dir"
        $BIN_DIR/find_gguf_files.pl \
            --dir "$dir" \
            --max-size 14GB \
            --sort size \
            --gpu-fit \
            2>/dev/null || echo "  (no models found)"
        echo ""
    fi
done

echo "=== Search Complete ==="
echo ""
echo "Tip: Models marked 'Yes' for GPU Fit will run comfortably on RTX 4080"
echo "     Models marked 'Tight' may run but with limited context window"
echo ""
