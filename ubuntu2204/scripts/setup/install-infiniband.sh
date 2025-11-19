#!/bin/bash
# InfiniBand Installation and Configuration Script
# Sets up NVIDIA MLNX_OFED for high-performance networking
# Target: Ubuntu 22.04 LTS

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_section() { echo -e "${BLUE}[====]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

print_section "NVIDIA InfiniBand (MLNX_OFED) Installation"
echo

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
print_info "Detected Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" ]]; then
    print_warn "This script is optimized for Ubuntu 22.04. Detected: $UBUNTU_VERSION"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Update system
print_section "Step 1: System Update"
print_info "Updating package lists..."
apt-get update
apt-get upgrade -y

# Install prerequisites
print_section "Step 2: Installing Prerequisites"
PREREQ_PACKAGES=(
    build-essential
    linux-headers-$(uname -r)
    dkms
    python3
    python3-pip
    libnl-3-dev
    libnl-route-3-dev
    pkg-config
    ethtool
    perl
    pciutils
    kmod
)

print_info "Installing prerequisite packages..."
apt-get install -y "${PREREQ_PACKAGES[@]}"

# Install base RDMA packages
print_section "Step 3: Installing Base RDMA Packages"
RDMA_PACKAGES=(
    rdma-core
    libibverbs1
    ibverbs-utils
    librdmacm1
    libibumad3
    ibverbs-providers
    rdmacm-utils
    infiniband-diags
    perftest
)

print_info "Installing RDMA packages..."
apt-get install -y "${RDMA_PACKAGES[@]}"

# Check for InfiniBand hardware
print_section "Step 4: Detecting InfiniBand Hardware"
if lspci | grep -i mellanox > /dev/null; then
    print_info "Mellanox/NVIDIA InfiniBand adapter detected:"
    lspci | grep -i mellanox
    HAS_IB=true
else
    print_warn "No Mellanox/NVIDIA InfiniBand adapter detected"
    print_warn "Installation will continue but hardware-specific features won't work"
    HAS_IB=false
fi

# Download and install MLNX_OFED
print_section "Step 5: MLNX_OFED Installation"

OFED_VERSION="24.01-0.3.3.1"  # Latest stable as of 2024
OFED_OS="ubuntu22.04"
OFED_ARCH="x86_64"
OFED_DIR="MLNX_OFED_LINUX-${OFED_VERSION}-${OFED_OS}-${OFED_ARCH}"
OFED_TGZ="${OFED_DIR}.tgz"
OFED_URL="https://www.mellanox.com/downloads/ofed/MLNX_OFED-${OFED_VERSION}/${OFED_TGZ}"

print_info "MLNX_OFED version: $OFED_VERSION"

# Ask user if they want to install MLNX_OFED
read -p "Do you want to download and install MLNX_OFED? (Recommended if you have IB hardware) [y/N]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Downloading MLNX_OFED..."
    cd /tmp

    if [[ ! -f "$OFED_TGZ" ]]; then
        print_warn "Automatic download may fail. You may need to download manually from:"
        print_warn "https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/"
        print_info "Attempting download..."

        wget "$OFED_URL" || {
            print_error "Download failed. Please download manually and run:"
            print_error "  tar xzf $OFED_TGZ"
            print_error "  cd $OFED_DIR"
            print_error "  sudo ./mlnxofedinstall --force"
            exit 1
        }
    fi

    print_info "Extracting MLNX_OFED..."
    tar xzf "$OFED_TGZ"
    cd "$OFED_DIR"

    print_info "Installing MLNX_OFED (this may take several minutes)..."
    ./mlnxofedinstall --force --without-fw-update

    print_info "Restarting OFED services..."
    /etc/init.d/openibd restart

    print_info "MLNX_OFED installed successfully"
else
    print_info "Skipping MLNX_OFED installation, using inbox drivers"
fi

# Load kernel modules
print_section "Step 6: Loading Kernel Modules"
MODULES=(
    ib_core
    ib_uverbs
    ib_umad
    rdma_ucm
    mlx5_core
    mlx5_ib
)

print_info "Loading InfiniBand kernel modules..."
for module in "${MODULES[@]}"; do
    if modprobe "$module" 2>/dev/null; then
        print_info "  ✓ Loaded: $module"
    else
        print_warn "  ✗ Failed to load: $module (may not be needed)"
    fi
done

# Make modules load at boot
print_info "Configuring modules to load at boot..."
cat > /etc/modules-load.d/infiniband.conf << 'EOF'
# InfiniBand kernel modules
ib_core
ib_uverbs
ib_umad
rdma_ucm
mlx5_core
mlx5_ib
EOF

# Configure IPoIB (IP over InfiniBand)
print_section "Step 7: IPoIB Configuration (Optional)"

if [[ "$HAS_IB" == true ]]; then
    read -p "Do you want to configure IP over InfiniBand? [y/N]: " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create IPoIB interface configuration
        print_info "Configuring IPoIB interface..."

        # Find IPoIB interfaces
        IB_IFACES=$(ip link | grep ib | awk '{print $2}' | sed 's/://g')

        if [[ -n "$IB_IFACES" ]]; then
            for IFACE in $IB_IFACES; do
                print_info "Found IPoIB interface: $IFACE"

                # Example netplan configuration
                cat >> /etc/netplan/99-infiniband.yaml << EOF
network:
  version: 2
  ethernets:
    $IFACE:
      dhcp4: false
      addresses:
        - 192.168.100.1/24
      mtu: 65520
EOF
            done

            print_info "IPoIB configured. Apply with: sudo netplan apply"
        else
            print_warn "No IPoIB interfaces found"
        fi
    fi
fi

# Verify installation
print_section "Step 8: Verifying Installation"

print_info "Checking RDMA devices..."
if command -v ibstat &> /dev/null; then
    ibstat || print_warn "No InfiniBand devices found"
else
    print_warn "ibstat not available"
fi

print_info "Checking RDMA links..."
rdma link show || print_warn "No RDMA links"

print_info "Checking loaded modules..."
lsmod | grep -E "ib_|mlx|rdma" || print_warn "No IB modules loaded"

# Performance tuning
print_section "Step 9: Performance Tuning"

print_info "Applying InfiniBand performance optimizations..."
cat > /etc/sysctl.d/99-infiniband.conf << 'EOF'
# InfiniBand Performance Tuning
# Generated by install-infiniband.sh

# Network buffer sizes
net.core.rmem_max=268435456
net.core.wmem_max=268435456
net.core.rmem_default=67108864
net.core.wmem_default=67108864
net.core.optmem_max=134217728

# TCP tuning
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728
net.ipv4.tcp_mem=134217728 134217728 134217728

# Connection tracking
net.netfilter.nf_conntrack_max=1048576
net.core.netdev_max_backlog=250000

# ARP cache
net.ipv4.neigh.default.gc_thresh1=8192
net.ipv4.neigh.default.gc_thresh2=32768
net.ipv4.neigh.default.gc_thresh3=65536
EOF

sysctl -p /etc/sysctl.d/99-infiniband.conf

# Create testing script
print_section "Step 10: Creating Test Scripts"

cat > /usr/local/bin/test-infiniband.sh << 'EOF'
#!/bin/bash
echo "===== InfiniBand Status Check ====="
echo
echo "=== Hardware Detection ==="
lspci | grep -i mellanox || echo "No Mellanox adapters found"
echo
echo "=== RDMA Devices ==="
rdma link show || echo "No RDMA devices"
echo
echo "=== InfiniBand Status ==="
ibstat || echo "ibstat not available or no devices"
echo
echo "=== Loaded Modules ==="
lsmod | grep -E "ib_|mlx|rdma"
echo
echo "=== IPoIB Interfaces ==="
ip link show | grep ib || echo "No IPoIB interfaces"
echo
echo "=== OFED Version ==="
ofed_info -s 2>/dev/null || echo "MLNX_OFED not installed"
EOF

chmod +x /usr/local/bin/test-infiniband.sh

# Create benchmark script
cat > /usr/local/bin/benchmark-infiniband.sh << 'EOF'
#!/bin/bash
# InfiniBand Performance Benchmark Script

echo "===== InfiniBand Performance Benchmark ====="
echo
echo "This script will test IB performance between two nodes"
echo "Run in SERVER mode on one node and CLIENT mode on another"
echo

read -p "Run as [S]erver or [C]lient? [S/c]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Cc]$ ]]; then
    read -p "Enter server IP address: " SERVER_IP

    echo
    echo "=== Write Bandwidth Test ==="
    ib_write_bw -d mlx5_0 -i 1 -s 65536 -n 10000 "$SERVER_IP"

    echo
    echo "=== Write Latency Test ==="
    ib_write_lat -d mlx5_0 -i 1 -s 2 -n 10000 "$SERVER_IP"

    echo
    echo "=== Read Bandwidth Test ==="
    ib_read_bw -d mlx5_0 -i 1 -s 65536 -n 10000 "$SERVER_IP"

    echo
    echo "=== Read Latency Test ==="
    ib_read_lat -d mlx5_0 -i 1 -s 2 -n 10000 "$SERVER_IP"
else
    echo "Starting server mode tests..."
    echo "Waiting for client connections..."
    echo
    echo "Run these commands on the CLIENT node:"
    echo "  ib_write_bw -d mlx5_0"
    echo "  ib_write_lat -d mlx5_0"
    echo

    echo "Starting bandwidth server..."
    ib_write_bw -d mlx5_0
fi
EOF

chmod +x /usr/local/bin/benchmark-infiniband.sh

# Summary
print_section "Installation Complete!"
echo
print_info "Summary:"
echo "  ✓ RDMA packages installed"
echo "  ✓ Kernel modules loaded"
echo "  ✓ Performance tuning applied"
echo "  ✓ Test scripts created"
echo
print_info "Test Scripts:"
echo "  - Status check: /usr/local/bin/test-infiniband.sh"
echo "  - Benchmarks: /usr/local/bin/benchmark-infiniband.sh"
echo
print_info "Quick Start:"
echo "  1. Check status: sudo /usr/local/bin/test-infiniband.sh"
echo "  2. View devices: sudo ibstat"
echo "  3. Check links: rdma link show"
echo "  4. Run benchmarks: /usr/local/bin/benchmark-infiniband.sh"
echo
if [[ "$HAS_IB" == true ]]; then
    print_info "InfiniBand hardware detected and configured"
else
    print_warn "No InfiniBand hardware detected"
    print_warn "Drivers installed but will only work when hardware is present"
fi
echo
print_warn "Note: Some changes may require a system reboot to take full effect"
