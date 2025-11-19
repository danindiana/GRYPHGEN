# Makefile for Tensor Core Examples
# Target: NVIDIA RTX 4080 (sm_89)
# Requires cuBLAS library

NVCC = nvcc
CUDA_ARCH = -arch=sm_89
NVCCFLAGS = -O3 $(CUDA_ARCH) -Xcompiler -Wall --use_fast_math -lineinfo
LIBS = -lcublas

TARGET = tensor_cores
SOURCES = tensor_cores.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "Building Tensor Core examples..."
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

run: $(TARGET)
	@echo "Running Tensor Core benchmark..."
	./$(TARGET) 4096 4096 4096

run-small: $(TARGET)
	./$(TARGET) 1024 1024 1024

run-large: $(TARGET)
	./$(TARGET) 8192 8192 8192

profile: $(TARGET)
	@echo "Profiling Tensor Core usage..."
	ncu --set full --section SpeedOfLight --section Occupancy ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.nsys-rep *.ncu-rep

.PHONY: all run run-small run-large profile clean
