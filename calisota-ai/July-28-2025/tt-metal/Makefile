# Dynamic Cortex Communication Framework
# Top-level Makefile for easy building

.PHONY: all clean greyskull cuda test install help

# Default target
all: help

# Build directories
BUILD_DIR := build
BUILD_GREYSKULL := $(BUILD_DIR)/greyskull
BUILD_CUDA := $(BUILD_DIR)/cuda
BUILD_ALL := $(BUILD_DIR)/all

# CMake options
CMAKE := cmake
CMAKE_BUILD_TYPE := Release
JOBS := $(shell nproc)

help:
	@echo "Dynamic Cortex Communication Framework - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make greyskull     - Build for Tenstorrent Greyskull (TT-Metal)"
	@echo "  make cuda          - Build for NVIDIA RTX 4080 (CUDA)"
	@echo "  make all-backends  - Build both TT-Metal and CUDA backends"
	@echo "  make test          - Run test suite"
	@echo "  make install       - Install libraries and headers"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  TT_METAL_HOME      - Path to TT-Metal SDK installation"
	@echo "  CUDA_HOME          - Path to CUDA Toolkit (optional)"
	@echo ""

# Build for Greyskull
greyskull:
	@echo "Building for Tenstorrent Greyskull..."
	@mkdir -p $(BUILD_GREYSKULL)
	@cd $(BUILD_GREYSKULL) && \
		$(CMAKE) ../.. \
			-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
			-DBUILD_TT_METAL=ON \
			-DBUILD_CUDA=OFF \
			-DBUILD_TESTS=ON \
			-DBUILD_EXAMPLES=ON
	@$(CMAKE) --build $(BUILD_GREYSKULL) -j$(JOBS)
	@echo "Greyskull build complete! Binaries in $(BUILD_GREYSKULL)/"

# Build for CUDA
cuda:
	@echo "Building for NVIDIA RTX 4080 (CUDA)..."
	@mkdir -p $(BUILD_CUDA)
	@cd $(BUILD_CUDA) && \
		$(CMAKE) ../.. \
			-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
			-DBUILD_TT_METAL=OFF \
			-DBUILD_CUDA=ON \
			-DBUILD_TESTS=ON \
			-DBUILD_EXAMPLES=ON
	@$(CMAKE) --build $(BUILD_CUDA) -j$(JOBS)
	@echo "CUDA build complete! Binaries in $(BUILD_CUDA)/"

# Build both backends
all-backends:
	@echo "Building all backends..."
	@mkdir -p $(BUILD_ALL)
	@cd $(BUILD_ALL) && \
		$(CMAKE) ../.. \
			-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
			-DBUILD_TT_METAL=ON \
			-DBUILD_CUDA=ON \
			-DBUILD_TESTS=ON \
			-DBUILD_EXAMPLES=ON
	@$(CMAKE) --build $(BUILD_ALL) -j$(JOBS)
	@echo "All backends build complete! Binaries in $(BUILD_ALL)/"

# Run tests
test: greyskull
	@echo "Running tests..."
	@cd $(BUILD_GREYSKULL) && ctest --output-on-failure

# Install
install:
	@if [ -d "$(BUILD_GREYSKULL)" ]; then \
		echo "Installing Greyskull build..."; \
		cd $(BUILD_GREYSKULL) && $(CMAKE) --install .; \
	fi
	@if [ -d "$(BUILD_CUDA)" ]; then \
		echo "Installing CUDA build..."; \
		cd $(BUILD_CUDA) && $(CMAKE) --install .; \
	fi
	@if [ -d "$(BUILD_ALL)" ]; then \
		echo "Installing all backends..."; \
		cd $(BUILD_ALL) && $(CMAKE) --install .; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

# Development targets
.PHONY: format check-format cloc

# Format code using clang-format
format:
	@echo "Formatting code..."
	@find src include -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" | xargs clang-format -i

# Check code formatting
check-format:
	@echo "Checking code formatting..."
	@find src include -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" | xargs clang-format --dry-run --Werror

# Count lines of code
cloc:
	@echo "Lines of code:"
	@cloc src include --exclude-dir=build

# Quick development builds
.PHONY: dev-greyskull dev-cuda

dev-greyskull:
	@$(MAKE) greyskull CMAKE_BUILD_TYPE=Debug

dev-cuda:
	@$(MAKE) cuda CMAKE_BUILD_TYPE=Debug
