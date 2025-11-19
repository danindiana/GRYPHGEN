#!/bin/bash
# GPU Health Monitoring Script
# Monitors NVIDIA RTX 4080 health and performance
# Target: Ubuntu 22.04 LTS

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_section() { echo -e "${BLUE}[====]${NC} $1"; }
print_data() { echo -e "${CYAN}$1${NC}"; }

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check if running in watch mode
WATCH_MODE=false
INTERVAL=2

while getopts "wi:" opt; do
    case $opt in
        w) WATCH_MODE=true ;;
        i) INTERVAL=$OPTARG ;;
        *) echo "Usage: $0 [-w] [-i interval]" >&2; exit 1 ;;
    esac
done

# Function to display GPU health
check_gpu_health() {
    clear
    print_section "NVIDIA GPU Health Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo

    # GPU Count
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    print_info "Detected GPUs: $GPU_COUNT"
    echo

    # Detailed GPU Information
    print_section "GPU Details"
    nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,compute_cap --format=csv
    echo

    # Temperature Monitoring
    print_section "Temperature Status"
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    TEMP_THRESHOLD_WARN=75
    TEMP_THRESHOLD_CRIT=85

    for i in $(seq 0 $((GPU_COUNT - 1))); do
        TEMP_GPU=$(nvidia-smi -i $i --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        echo -n "GPU $i Temperature: "

        if [[ $TEMP_GPU -ge $TEMP_THRESHOLD_CRIT ]]; then
            echo -e "${RED}${TEMP_GPU}°C (CRITICAL)${NC}"
        elif [[ $TEMP_GPU -ge $TEMP_THRESHOLD_WARN ]]; then
            echo -e "${YELLOW}${TEMP_GPU}°C (Warning)${NC}"
        else
            echo -e "${GREEN}${TEMP_GPU}°C (Normal)${NC}"
        fi
    done
    echo

    # Power Usage
    print_section "Power Consumption"
    nvidia-smi --query-gpu=index,power.draw,power.limit,power.max_limit --format=csv
    echo

    # Memory Usage
    print_section "Memory Usage"
    nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv
    echo

    # Utilization
    print_section "GPU Utilization"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv
    echo

    # Clock Speeds
    print_section "Clock Speeds"
    nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory --format=csv
    echo

    # Performance State
    print_section "Performance State"
    nvidia-smi --query-gpu=index,pstate,clocks_throttle_reasons.active --format=csv
    echo

    # Processes Using GPU
    print_section "GPU Processes"
    if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    else
        echo "No processes currently using GPU"
    fi
    echo

    # ECC Errors (if supported)
    print_section "Error Checking"
    ECC_SUPPORT=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | head -1)
    if [[ "$ECC_SUPPORT" != "N/A" ]]; then
        nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
    else
        echo "ECC not supported on this GPU"
    fi
    echo

    # PCIe Information
    print_section "PCIe Status"
    nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
    echo

    # Fan Speed
    print_section "Fan Status"
    nvidia-smi --query-gpu=index,fan.speed --format=csv
    echo

    # Health Warnings
    print_section "Health Warnings"
    WARNINGS=0

    for i in $(seq 0 $((GPU_COUNT - 1))); do
        # Check temperature
        TEMP_GPU=$(nvidia-smi -i $i --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        if [[ $TEMP_GPU -ge $TEMP_THRESHOLD_CRIT ]]; then
            print_error "GPU $i: Critical temperature ($TEMP_GPU°C)"
            ((WARNINGS++))
        fi

        # Check throttling
        THROTTLE=$(nvidia-smi -i $i --query-gpu=clocks_throttle_reasons.active --format=csv,noheader)
        if [[ "$THROTTLE" != "0x0000000000000000" ]]; then
            print_warn "GPU $i: Throttling detected ($THROTTLE)"
            ((WARNINGS++))
        fi

        # Check power limit
        POWER_DRAW=$(nvidia-smi -i $i --query-gpu=power.draw --format=csv,noheader,nounits)
        POWER_LIMIT=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader,nounits)
        POWER_PCT=$(awk "BEGIN {printf \"%.0f\", ($POWER_DRAW/$POWER_LIMIT)*100}")

        if [[ $POWER_PCT -ge 95 ]]; then
            print_warn "GPU $i: Near power limit (${POWER_PCT}%)"
            ((WARNINGS++))
        fi
    done

    if [[ $WARNINGS -eq 0 ]]; then
        print_info "All health checks passed ✓"
    else
        print_warn "Total warnings: $WARNINGS"
    fi
    echo

    # RTX 4080 Specific Information
    print_section "RTX 4080 Optimization Status"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

    if [[ "$GPU_NAME" =~ "4080" ]]; then
        print_info "RTX 4080 detected - Ada Lovelace architecture"
        echo "  Compute Capability: 8.9"
        echo "  CUDA Cores: 9728"
        echo "  Tensor Cores: 304 (4th Gen)"
        echo "  RT Cores: 76 (3rd Gen)"
        echo "  Memory: 16GB GDDR6X"
        echo "  Memory Bandwidth: 716.8 GB/s"
        echo "  Boost Clock: ~2.51 GHz"
    fi
    echo

    # Recommendations
    print_section "Recommendations"
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        TEMP_GPU=$(nvidia-smi -i $i --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        UTIL=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits)

        if [[ $TEMP_GPU -ge 80 ]]; then
            echo "GPU $i: Consider improving cooling (temp: ${TEMP_GPU}°C)"
        fi

        if [[ $UTIL -lt 10 ]]; then
            echo "GPU $i: Low utilization (${UTIL}%) - GPU may be idle"
        fi
    done
    echo
}

# Main execution
if [[ "$WATCH_MODE" == true ]]; then
    print_info "Starting watch mode (Ctrl+C to exit, interval: ${INTERVAL}s)"
    while true; do
        check_gpu_health
        sleep "$INTERVAL"
    done
else
    check_gpu_health
    echo
    print_info "Tip: Use -w flag for continuous monitoring"
    print_info "Example: $0 -w -i 5  (watch mode, 5 second interval)"
fi
