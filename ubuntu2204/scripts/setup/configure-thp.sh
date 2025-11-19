#!/bin/bash
# Transparent Huge Pages (THP) Configuration Script
# Optimized for memory-intensive HPC/ML workloads
# Target: Ubuntu 22.04 LTS

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

print_info "Transparent Huge Pages (THP) Configuration Utility"
print_info "=================================================="
echo

# Display current THP status
print_info "Current THP Status:"
cat /sys/kernel/mm/transparent_hugepage/enabled
echo

# Function to configure THP
configure_thp() {
    local mode=$1

    case $mode in
        "disable")
            print_info "Disabling THP (Recommended for ML/LLM workloads)..."
            echo never | tee /sys/kernel/mm/transparent_hugepage/enabled
            echo never | tee /sys/kernel/mm/transparent_hugepage/defrag
            ;;
        "enable")
            print_info "Enabling THP with always mode..."
            echo always | tee /sys/kernel/mm/transparent_hugepage/enabled
            echo always | tee /sys/kernel/mm/transparent_hugepage/defrag
            ;;
        "madvise")
            print_info "Enabling THP with madvise mode (selective)..."
            echo madvise | tee /sys/kernel/mm/transparent_hugepage/enabled
            echo madvise | tee /sys/kernel/mm/transparent_hugepage/defrag
            ;;
        *)
            print_error "Invalid mode: $mode"
            print_error "Valid modes: disable, enable, madvise"
            exit 1
            ;;
    esac
}

# Parse command line arguments
MODE=${1:-"disable"}

print_info "Configuring THP mode: $MODE"
configure_thp "$MODE"

# Make changes persistent across reboots
print_info "Making changes persistent..."

# Create systemd service
cat > /etc/systemd/system/disable-thp.service << 'EOF'
[Unit]
Description=Disable Transparent Huge Pages (THP)
DefaultDependencies=no
After=sysinit.target local-fs.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never | tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null'
ExecStart=/bin/sh -c 'echo never | tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null'

[Install]
WantedBy=basic.target
EOF

# Enable the service
systemctl daemon-reload
systemctl enable disable-thp.service
systemctl start disable-thp.service

print_info "Systemd service created and enabled"

# Update GRUB for kernel parameter (alternative method)
if ! grep -q "transparent_hugepage=never" /etc/default/grub; then
    print_info "Adding kernel parameter to GRUB..."
    sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="transparent_hugepage=never /' /etc/default/grub
    update-grub
    print_warn "GRUB updated. Changes will take full effect after reboot."
fi

# Verify configuration
echo
print_info "Current THP Configuration:"
echo -n "Enabled: "
cat /sys/kernel/mm/transparent_hugepage/enabled
echo -n "Defrag: "
cat /sys/kernel/mm/transparent_hugepage/defrag

# Additional THP settings
print_info "Configuring additional THP parameters..."
echo 0 | tee /sys/kernel/mm/transparent_hugepage/khugepaged/defrag > /dev/null
echo 0 | tee /sys/kernel/mm/transparent_hugepage/use_zero_page > /dev/null

# Display summary
echo
print_info "THP Configuration Complete!"
print_info "================================"
print_info "Mode: $MODE"
print_info "Persistent: Yes (systemd service + GRUB)"
print_info "Service Status:"
systemctl status disable-thp.service --no-pager | head -5

echo
print_warn "For complete effect, consider rebooting the system."
print_info "To check status: cat /sys/kernel/mm/transparent_hugepage/enabled"
