# Makefile for Matrix Multiplication CUDA Example
# Target: NVIDIA RTX 4080 (Ada Lovelace, Compute Capability 8.9)
# CUDA 12.x

# Compiler
NVCC = nvcc

# Target compute capability for RTX 4080
# 8.9 for Ada Lovelace architecture
CUDA_ARCH = -arch=sm_89

# Compiler flags
NVCCFLAGS = -O3 \
            $(CUDA_ARCH) \
            -Xcompiler -Wall \
            -Xcompiler -Wextra \
            --use_fast_math \
            -lineinfo

# For debugging, add:
# DEBUG_FLAGS = -g -G -DDEBUG

# Target executable
TARGET = matrix_multiply

# Source files
SOURCES = matrix_multiply.cu

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SOURCES)
	@echo "Building $@ for NVIDIA RTX 4080 (sm_89)..."
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "Build complete!"

# Debug build
debug: NVCCFLAGS += -g -G -DDEBUG
debug: clean $(TARGET)
	@echo "Debug build complete!"

# Run the program
run: $(TARGET)
	@echo "Running matrix multiplication..."
	./$(TARGET) 2048 2048 2048

# Run with different sizes
run-small: $(TARGET)
	./$(TARGET) 512 512 512

run-medium: $(TARGET)
	./$(TARGET) 1024 1024 1024

run-large: $(TARGET)
	./$(TARGET) 4096 4096 4096

# Profile with NVIDIA Nsight Compute
profile: $(TARGET)
	@echo "Profiling with ncu..."
	ncu --set full --export matrix_mul_profile ./$(TARGET)

# Profile with NVIDIA Nsight Systems
profile-sys: $(TARGET)
	@echo "Profiling with nsys..."
	nsys profile --stats=true ./$(TARGET)

# Check CUDA device properties
deviceQuery:
	@echo "Checking CUDA device properties..."
	nvidia-smi
	@echo ""
	@echo "CUDA Compiler Info:"
	$(NVCC) --version

# Clean
clean:
	rm -f $(TARGET) *.o *.nsys-rep *.ncu-rep

# Clean all including profiles
cleanall: clean
	rm -f *.qdrep *.sqlite

.PHONY: all debug run run-small run-medium run-large profile profile-sys deviceQuery clean cleanall
