# Top-Level Makefile for Ubuntu 22.04 HPC/ML Toolkit
# Orchestrates building all CUDA examples and tools
# Target: NVIDIA RTX 4080 (sm_89)

.PHONY: all clean test help examples setup check-cuda

# Default target
all: check-cuda examples

# Help target
help:
	@echo "=== Ubuntu 22.04 HPC/ML Toolkit ==="
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all examples (default)"
	@echo "  examples         - Build all CUDA examples"
	@echo "  test             - Run all tests"
	@echo "  setup            - Run system setup scripts"
	@echo "  check-cuda       - Verify CUDA installation"
	@echo "  clean            - Clean all build artifacts"
	@echo "  install-deps     - Install system dependencies"
	@echo "  profile          - Profile all examples"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Example CUDA programs:"
	@echo "  matrix-multiply  - Build matrix multiplication example"
	@echo "  memory-opt       - Build memory optimization example"
	@echo "  tensor-cores     - Build Tensor Core example"
	@echo ""
	@echo "Usage:"
	@echo "  make             # Build everything"
	@echo "  make test        # Run tests"
	@echo "  make clean       # Clean build files"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found. Please install CUDA toolkit."; exit 1; }
	@command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."; exit 1; }
	@echo "✓ CUDA toolkit found: $$(nvcc --version | grep release | awk '{print $$5}' | sed 's/,//')"
	@echo "✓ NVIDIA driver found: $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
	@echo "✓ GPU detected: $$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
	@echo ""

# Build all examples
examples: matrix-multiply memory-opt tensor-cores
	@echo ""
	@echo "✓ All examples built successfully!"

# Individual example targets
matrix-multiply:
	@echo "Building matrix multiplication example..."
	@$(MAKE) -C examples/cuda/matrix-multiply
	@echo ""

memory-opt:
	@echo "Building memory optimization example..."
	@$(MAKE) -C examples/cuda/memory-optimization
	@echo ""

tensor-cores:
	@echo "Building Tensor Core example..."
	@$(MAKE) -C examples/cuda/tensor-operations
	@echo ""

# Run tests
test:
	@echo "=== Running Test Suite ==="
	@echo ""
	@echo "Test 1: Validating dependencies..."
	@./tests/validate-dependencies.sh || true
	@echo ""
	@echo "Test 2: Validating GPU setup..."
	@./tests/test-gpu-setup.sh || true
	@echo ""
	@echo "=== Test Suite Complete ==="

# System setup
setup:
	@echo "=== Running System Setup ==="
	@echo ""
	@echo "This will optimize your system for HPC/ML workloads."
	@echo "The following scripts will be run (requires sudo):"
	@echo "  1. Kernel optimization"
	@echo "  2. THP configuration"
	@echo "  3. Swappiness tuning"
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		sudo ./scripts/setup/optimize-kernel.sh; \
		sudo ./scripts/setup/configure-thp.sh; \
		sudo ./scripts/setup/configure-swappiness.sh; \
	fi

# Install dependencies
install-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		cuda-toolkit \
		nvidia-driver-545 \
		cmake \
		git \
		htop \
		nvtop \
		linux-tools-common \
		sysstat \
		numactl
	@echo "✓ Dependencies installed"

# Profile all examples
profile: examples
	@echo "=== Profiling Examples ==="
	@echo ""
	@echo "Profiling matrix multiplication..."
	@$(MAKE) -C examples/cuda/matrix-multiply profile || true
	@echo ""
	@echo "Profiling memory optimization..."
	@$(MAKE) -C examples/cuda/memory-optimization profile || true
	@echo ""
	@echo "Profiling Tensor Cores..."
	@$(MAKE) -C examples/cuda/tensor-operations profile || true
	@echo ""
	@echo "✓ Profiling complete"

# Benchmark suite
benchmark: examples
	@echo "=== Running Benchmark Suite ==="
	@echo ""
	@echo "Benchmark 1: Matrix Multiplication"
	@$(MAKE) -C examples/cuda/matrix-multiply run
	@echo ""
	@echo "Benchmark 2: Memory Patterns"
	@$(MAKE) -C examples/cuda/memory-optimization run
	@echo ""
	@echo "Benchmark 3: Tensor Cores"
	@$(MAKE) -C examples/cuda/tensor-operations run
	@echo ""
	@echo "=== Benchmarks Complete ==="

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@$(MAKE) -C examples/cuda/matrix-multiply clean 2>/dev/null || true
	@$(MAKE) -C examples/cuda/memory-optimization clean 2>/dev/null || true
	@$(MAKE) -C examples/cuda/tensor-operations clean 2>/dev/null || true
	@find . -name "*.o" -delete
	@find . -name "*.nsys-rep" -delete
	@find . -name "*.ncu-rep" -delete
	@echo "✓ Clean complete"

# Monitor system
monitor:
	@echo "Starting system monitoring..."
	@./scripts/monitoring/check-gpu-health.sh -w -i 5

# System info
info:
	@echo "=== System Information ==="
	@./scripts/utils/system-info.sh 2>/dev/null || { \
		echo "OS: $$(lsb_release -d | cut -f2)"; \
		echo "Kernel: $$(uname -r)"; \
		echo "CUDA: $$(nvcc --version | grep release | awk '{print $$5}' | sed 's/,//')"; \
		nvidia-smi --query-gpu=name,memory.total,driver_version --format=table; \
	}
