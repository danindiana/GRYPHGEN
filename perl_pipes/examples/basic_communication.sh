#!/bin/bash
#
# Basic Model Communication Example
# Demonstrates simple message exchange between two models
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/../bin"

echo "=== Basic Model Communication Example ==="
echo ""

# Step 1: Setup pipes
echo "Step 1: Creating named pipes..."
$BIN_DIR/setup_pipes.sh

echo ""
echo "Step 2: Starting communication..."
echo ""

# Start Model 2 in background (must start first to listen)
echo "Starting Model 2 (listener)..."
$BIN_DIR/model2_comm.pl --verbose &
MODEL2_PID=$!

# Give Model 2 time to start listening
sleep 1

# Start Model 1 with a message
echo "Starting Model 1 (sender)..."
echo "Hello from the basic example!" | $BIN_DIR/model1_comm.pl --verbose

# Wait for processes to complete
wait $MODEL2_PID 2>/dev/null || true

echo ""
echo "Step 3: Cleaning up pipes..."
$BIN_DIR/cleanup_pipes.sh --force

echo ""
echo "=== Example Complete ==="
