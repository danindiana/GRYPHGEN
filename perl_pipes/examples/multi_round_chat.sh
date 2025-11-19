#!/bin/bash
#
# Multi-Round Chat Example
# Demonstrates multiple message exchanges between models
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/../bin"

ROUNDS=3

echo "=== Multi-Round Model Chat Example ==="
echo "Performing $ROUNDS rounds of communication"
echo ""

# Setup pipes
echo "Setting up named pipes..."
$BIN_DIR/setup_pipes.sh > /dev/null

echo ""
echo "Starting multi-round communication..."
echo ""

for ((i=1; i<=ROUNDS; i++)); do
    echo "--- Round $i/$ROUNDS ---"

    # Start Model 2 in background
    $BIN_DIR/model2_comm.pl --verbose &
    MODEL2_PID=$!

    sleep 0.5

    # Send message from Model 1
    echo "Round $i message from Model 1" | $BIN_DIR/model1_comm.pl --verbose

    # Wait for communication to complete
    wait $MODEL2_PID 2>/dev/null || true

    echo ""
    sleep 1
done

# Cleanup
echo "Cleaning up..."
$BIN_DIR/cleanup_pipes.sh --force > /dev/null

echo ""
echo "=== Multi-Round Chat Complete ==="
echo "Completed $ROUNDS successful message exchanges"
