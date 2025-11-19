#!/bin/bash
# Memory Monitoring Script
# Monitors system and GPU memory for ML workloads
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

# Configuration
WATCH_MODE=false
INTERVAL=2
LOG_FILE=""

while getopts "wi:l:" opt; do
    case $opt in
        w) WATCH_MODE=true ;;
        i) INTERVAL=$OPTARG ;;
        l) LOG_FILE=$OPTARG ;;
        *) echo "Usage: $0 [-w] [-i interval] [-l logfile]" >&2; exit 1 ;;
    esac
done

# Function to get memory info
monitor_memory() {
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    if [[ "$WATCH_MODE" == true ]]; then
        clear
    fi

    print_section "Memory Monitor - $TIMESTAMP"
    echo

    # System Memory Overview
    print_section "System Memory Overview"
    free -h
    echo

    # Detailed Memory Statistics
    print_section "Detailed Memory Statistics"
    echo "Total:       $(grep MemTotal /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Free:        $(grep MemFree /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Available:   $(grep MemAvailable /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Buffers:     $(grep Buffers /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Cached:      $(grep '^Cached:' /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Active:      $(grep '^Active:' /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Inactive:    $(grep '^Inactive:' /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Dirty:       $(grep '^Dirty:' /proc/meminfo | awk '{print $2/1024 " MB"}')"
    echo "Writeback:   $(grep Writeback /proc/meminfo | awk '{print $2/1024 " MB"}')"
    echo

    # Memory Usage by Process (Top 10)
    print_section "Top 10 Memory-Consuming Processes"
    ps aux --sort=-%mem | awk 'NR<=11{printf "%-8s %-8s %-6s %-6s %-s\n", $1, $2, $3, $4, $11}'
    echo

    # Swap Usage
    print_section "Swap Usage"
    echo "Total:       $(grep SwapTotal /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Used:        $(grep '^SwapFree' /proc/meminfo | awk -v total="$(grep SwapTotal /proc/meminfo | awk '{print $2}')" '{print (total-$2)/1024/1024 " GB"}')"
    echo "Free:        $(grep SwapFree /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "Cached:      $(grep SwapCached /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo
    swapon --show || echo "No swap devices"
    echo

    # VM Statistics
    print_section "VM Statistics"
    echo "Swappiness:           $(cat /proc/sys/vm/swappiness)"
    echo "Cache Pressure:       $(cat /proc/sys/vm/vfs_cache_pressure)"
    echo "Dirty Ratio:          $(cat /proc/sys/vm/dirty_ratio)%"
    echo "Dirty BG Ratio:       $(cat /proc/sys/vm/dirty_background_ratio)%"
    echo "Min Free kBytes:      $(cat /proc/sys/vm/min_free_kbytes | awk '{print $1/1024 " MB"}')"
    echo

    # Huge Pages Status
    print_section "Huge Pages Status"
    if [[ -f /proc/meminfo ]]; then
        grep -i huge /proc/meminfo
    fi
    echo

    # THP Status
    print_section "Transparent Huge Pages"
    echo -n "THP Enabled: "
    cat /sys/kernel/mm/transparent_hugepage/enabled
    echo -n "THP Defrag:  "
    cat /sys/kernel/mm/transparent_hugepage/defrag
    echo

    # GPU Memory (if NVIDIA GPU present)
    if command -v nvidia-smi &> /dev/null; then
        print_section "GPU Memory Usage"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.memory --format=csv
        echo

        # Detailed GPU memory by process
        print_section "GPU Memory by Process"
        if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        else
            echo "No processes currently using GPU memory"
        fi
        echo
    fi

    # NUMA Information
    if command -v numactl &> /dev/null; then
        print_section "NUMA Memory"
        numactl --hardware | grep -A 3 "available:"
        echo
    fi

    # Memory Pressure and OOM
    print_section "Memory Pressure"
    if [[ -f /proc/pressure/memory ]]; then
        echo "Memory Pressure (PSI):"
        cat /proc/pressure/memory
    else
        echo "PSI metrics not available (kernel < 4.20)"
    fi
    echo

    # Check for recent OOM kills
    print_section "Recent OOM Kills"
    OOM_COUNT=$(dmesg | grep -i "killed process" | tail -5 | wc -l)
    if [[ $OOM_COUNT -gt 0 ]]; then
        print_warn "Found $OOM_COUNT recent OOM kills:"
        dmesg | grep -i "killed process" | tail -5
    else
        print_info "No recent OOM kills detected"
    fi
    echo

    # Memory Fragmentation
    print_section "Memory Fragmentation"
    cat /proc/buddyinfo | head -5
    echo

    # System Load
    print_section "System Load"
    uptime
    echo

    # Memory Health Warnings
    print_section "Health Warnings"
    WARNINGS=0

    # Check available memory
    MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_AVAIL_PCT=$((MEM_AVAIL * 100 / MEM_TOTAL))

    if [[ $MEM_AVAIL_PCT -lt 10 ]]; then
        print_error "Critical: Low available memory (${MEM_AVAIL_PCT}%)"
        ((WARNINGS++))
    elif [[ $MEM_AVAIL_PCT -lt 20 ]]; then
        print_warn "Warning: Available memory is low (${MEM_AVAIL_PCT}%)"
        ((WARNINGS++))
    fi

    # Check swap usage
    SWAP_TOTAL=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
    if [[ $SWAP_TOTAL -gt 0 ]]; then
        SWAP_FREE=$(grep SwapFree /proc/meminfo | awk '{print $2}')
        SWAP_USED=$((SWAP_TOTAL - SWAP_FREE))
        SWAP_USED_PCT=$((SWAP_USED * 100 / SWAP_TOTAL))

        if [[ $SWAP_USED_PCT -gt 50 ]]; then
            print_warn "Warning: High swap usage (${SWAP_USED_PCT}%)"
            ((WARNINGS++))
        fi
    fi

    # Check dirty pages
    DIRTY=$(grep '^Dirty:' /proc/meminfo | awk '{print $2}')
    if [[ $DIRTY -gt 1048576 ]]; then  # > 1GB dirty
        print_warn "Warning: High dirty pages ($(echo "scale=2; $DIRTY/1024/1024" | bc) GB)"
        ((WARNINGS++))
    fi

    if [[ $WARNINGS -eq 0 ]]; then
        print_info "All memory health checks passed ✓"
    else
        print_warn "Total warnings: $WARNINGS"
    fi
    echo

    # Recommendations
    print_section "Recommendations"
    if [[ $MEM_AVAIL_PCT -lt 20 ]]; then
        echo "- Consider freeing memory or adding more RAM"
        echo "- Check for memory leaks in applications"
        echo "- Use: ps aux --sort=-%mem | head to find memory hogs"
    fi

    if [[ $SWAP_USED_PCT -gt 30 ]]; then
        echo "- High swap usage detected, may impact performance"
        echo "- Consider reducing swappiness: sudo sysctl vm.swappiness=10"
        echo "- Add more physical RAM if possible"
    fi

    # Log to file if specified
    if [[ -n "$LOG_FILE" ]]; then
        {
            echo "=== Memory Monitor Log - $TIMESTAMP ==="
            free -h
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
            echo
        } >> "$LOG_FILE"
    fi
}

# Main execution
if [[ "$WATCH_MODE" == true ]]; then
    print_info "Starting watch mode (Ctrl+C to exit, interval: ${INTERVAL}s)"
    [[ -n "$LOG_FILE" ]] && print_info "Logging to: $LOG_FILE"
    echo
    while true; do
        monitor_memory
        sleep "$INTERVAL"
    done
else
    monitor_memory
    echo
    print_info "Tip: Use -w flag for continuous monitoring"
    print_info "Example: $0 -w -i 3 -l memory.log"
fi
