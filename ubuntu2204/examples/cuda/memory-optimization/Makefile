# Makefile for Memory Optimization CUDA Examples
# Target: NVIDIA RTX 4080 (sm_89)

NVCC = nvcc
CUDA_ARCH = -arch=sm_89
NVCCFLAGS = -O3 $(CUDA_ARCH) -Xcompiler -Wall --use_fast_math -lineinfo

TARGET = memory_patterns
SOURCES = memory_patterns.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "Building memory optimization examples..."
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	@echo "Running memory pattern benchmarks..."
	./$(TARGET)

profile: $(TARGET)
	@echo "Profiling with ncu..."
	ncu --set full --section MemoryWorkloadAnalysis ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.nsys-rep *.ncu-rep

.PHONY: all run profile clean
