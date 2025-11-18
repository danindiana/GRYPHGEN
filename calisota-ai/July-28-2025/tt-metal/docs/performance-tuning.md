# Performance Tuning Guide

Optimization strategies for maximizing performance on Tenstorrent Greyskull and NVIDIA RTX 4080.

## Table of Contents

- [General Principles](#general-principles)
- [TT-Metal Optimization](#tt-metal-optimization)
- [CUDA Optimization](#cuda-optimization)
- [Memory Management](#memory-management)
- [Profiling](#profiling)

## General Principles

### 1. Choose Appropriate Precision

**FP32 (Full Precision)**
- Best accuracy
- Highest memory usage
- Lower throughput

**FP16 (Half Precision)**
- 2x memory efficiency
- 2x computational throughput
- Sufficient for most neural computations

**INT8 (Quantized)**
- 4x memory efficiency
- 4x computational throughput
- Requires careful calibration

**Recommendation:**
```cpp
// Use mixed precision for best balance
MatMulOp matmul;
matmul.setMathFidelity(MathFidelity::HiFi2);  // FP16 compute, FP32 accumulation
```

### 2. Minimize Data Movement

Data transfer is typically the bottleneck. Strategies:

1. **Keep data on-device** as long as possible
2. **Overlap computation and communication**
3. **Use compression for sparse data**
4. **Batch operations** to amortize transfer costs

## TT-Metal Optimization

### Core Allocation

Efficiently map computation to Tensix cores.

```cpp
// V1: 32 cores (8x4 grid)
// LM: 16 cores (4x4 grid)

v1->assignCores(core_utils::getCoreRange({0, 0}, {7, 3}));
lm->assignCores(core_utils::getCoreRange({0, 4}, {3, 7}));
```

### NOC Optimization

Greyskull has 2 NOCs - use both to maximize bandwidth:

```cpp
NOCConfig feedforward_noc;
feedforward_noc.noc_id = 0;  // Use NOC 0 for feedforward

NOCConfig feedback_noc;
feedback_noc.noc_id = 1;  // Use NOC 1 for feedback

// Minimize cross-NOC conflicts
stream_manager.optimizeRouting();
```

### Memory Hierarchy

**L1 SRAM (1.2 MB per core):**
- Fastest access
- Use for frequently accessed data
- Buffer intermediate results

**Device DRAM (8/16 GB):**
- Larger capacity
- ~1 TB/s bandwidth
- Use for large tensors

```cpp
// Keep working set in L1
StreamEndpoint v1_endpoint;
v1_endpoint.buffer_addr = L1_BASE_ADDR;
v1_endpoint.buffer_size = 256 * 1024;  // 256 KB
```

### Sparse Activation

Power down inactive cores to save energy:

```cpp
SparseActivation sparse_mgr;
sparse_mgr.setActivationThreshold(0.1f);

auto active_cores = sparse_mgr.computeActiveCores(
    activity_data, num_elements, all_cores);

sparse_mgr.powerDownInactiveCores(inactive_cores);
```

**Energy Savings:** 30-50% for typical sparsity levels (50-70%)

### Channel Rotation

Optimize rotation frequency based on behavioral context:

```cpp
ChannelRotator rotator(8, 121);  // 8 channels, 121ms default

// Fast rotation for rewarded stimuli
if (is_rewarded) {
    rotator.setRotationPeriod(15);  // 15ms → higher throughput
} else {
    rotator.setRotationPeriod(121);  // 121ms → lower power
}
```

### Pipeline Parallelism

Overlap stages for continuous execution:

```cpp
// Stage 1: V1 compute (cores 0-31)
// Stage 2: Feedforward transfer (NOC 0)
// Stage 3: LM compute (cores 32-47)
// Stage 4: Feedback transfer (NOC 1)

// All stages execute concurrently!
```

### Math Fidelity

Trade accuracy for speed:

| Fidelity | Accuracy | Speed | Power |
|----------|----------|-------|-------|
| LoFi     | ~8-bit   | 4x    | Low   |
| HiFi2    | ~12-bit  | 2x    | Med   |
| HiFi3    | ~16-bit  | 1.5x  | Med   |
| HiFi4    | ~24-bit  | 1x    | High  |

```cpp
MatMulOp matmul;
matmul.setMathFidelity(MathFidelity::HiFi2);  // Good balance
```

## CUDA Optimization

### Block/Thread Configuration

**RTX 4080 specs:**
- 76 SMs
- 1536 threads/SM
- 48 KB shared memory/block

**Optimal configurations:**

```cpp
// For small kernels (< 100 registers/thread)
dim3 block(256);  // 8 warps
dim3 grid((n + 255) / 256);

// For large kernels (register pressure)
dim3 block(128);  // 4 warps, more registers available
```

### Shared Memory

Cache frequently accessed data in shared memory (100 GB/s vs 716 GB/s VRAM):

```cpp
__global__ void v1_encode_kernel_shared(
    const float* __restrict__ stimulus,
    /* ... */
) {
    extern __shared__ float shared_stimulus[];

    // Cooperative load
    for (uint32_t i = threadIdx.x; i < stimulus_size; i += blockDim.x) {
        shared_stimulus[i] = stimulus[i];
    }
    __syncthreads();

    // Compute using shared memory
    // ... 100x faster than global memory access!
}
```

### Warp-Level Primitives

Use warp shuffle for fast inter-thread communication:

```cpp
// Warp reduction (no shared memory needed!)
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
}
```

**Performance gain:** 2-3x vs shared memory reduction

### Tensor Cores

Use Tensor Cores for mixed-precision matrix multiplication:

```cpp
#include <cublas_lt.h>

// FP16 input, FP32 accumulation
cublasLtMatmul(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, CUDA_R_16F,  // FP16 input
    B, CUDA_R_16F,
    &beta,
    C, CUDA_R_32F,  // FP32 output
    CUDA_R_32F,     // FP32 compute
    CUBLAS_COMPUTE_32F_FAST_16F);  // Use Tensor Cores!
```

**Throughput:** 97.5 TFLOPS (vs 48.7 TFLOPS FP32)

### Stream Concurrency

Overlap data transfer and computation:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Pipeline:
// Stream 1: H2D copy trial N+1
// Stream 2: Compute trial N
// Stream 3: D2H copy trial N-1

cudaMemcpyAsync(d_input, h_input_next, size, H2D, stream1);
kernel<<<grid, block, 0, stream2>>>(d_input_current, d_output);
cudaMemcpyAsync(h_output_prev, d_output_prev, size, D2H, stream3);
```

### Unified Memory

Simplify memory management (at slight performance cost):

```cpp
float* data;
cudaMallocManaged(&data, size);

// Automatic migration between host and device
kernel<<<grid, block>>>(data);  // GPU access
for (int i = 0; i < n; ++i) {
    printf("%f\n", data[i]);    // CPU access
}
```

**When to use:**
- Prototyping
- Data > GPU memory
- Irregular access patterns

**When NOT to use:**
- Performance-critical kernels
- Streaming workloads

### Kernel Fusion

Combine operations to reduce memory traffic:

```cpp
// Instead of:
// 1. Convolution (read input, write temp)
// 2. ReLU (read temp, write output)

// Fuse into single kernel:
__global__ void conv_relu_fused(/* ... */) {
    float conv_result = convolve(/* ... */);
    float output = fmaxf(0.0f, conv_result);  // Inline ReLU
    // Write once!
}
```

**Memory traffic reduction:** 2x

### Occupancy Optimization

Maximize SM utilization:

```bash
# Compile with occupancy report
nvcc -Xptxas -v kernel.cu

# Output:
# Used 64 registers, 48 KB shared memory
# Theoretical occupancy: 75%
```

**Improve occupancy:**
1. Reduce register usage (`-maxrregcount`)
2. Reduce shared memory
3. Adjust block size

## Memory Management

### TT-Metal Memory Pools

Reuse allocations to avoid overhead:

```cpp
class MemoryPool {
public:
    void* allocate(size_t size) {
        if (auto ptr = free_list_.find(size)) {
            return ptr;
        }
        return tt_metal_malloc(size);
    }

    void deallocate(void* ptr, size_t size) {
        free_list_.insert(size, ptr);  // Reuse later
    }
};
```

### CUDA Memory Pools

```cpp
cudaMemPool_t pool;
cudaMemPoolCreate(&pool, &poolProps);

// Fast allocations from pool
cudaMallocFromPoolAsync(&ptr, size, pool, stream);
cudaFreeAsync(ptr, stream);  // Return to pool
```

### Buffer Reuse

Minimize allocations in hot loops:

```cpp
// BAD: Allocate every iteration
for (int trial = 0; trial < num_trials; ++trial) {
    Tensor temp({64, 64}, DataType::FP32);  // Slow!
    // ...
}

// GOOD: Allocate once, reuse
Tensor temp({64, 64}, DataType::FP32);
for (int trial = 0; trial < num_trials; ++trial) {
    temp.zero();  // Fast!
    // ...
}
```

## Profiling

### TT-Metal Profiling

```cpp
framework->enableProfiling(true);

// Run workload
framework->processStimulus(stimulus, milliseconds(1000));

// Get report
std::string report = framework->getProfilingReport();
std::cout << report << std::endl;
```

**Output:**
```
=== Profiling Report ===
V1 Encode:     12.3 ms (35%)
Feedforward:    2.1 ms (6%)
LM Modulate:    8.7 ms (25%)
Feedback:       3.2 ms (9%)
V1 Update:      9.1 ms (26%)
Total:         35.4 ms
```

### CUDA Profiling

**nvprof:**
```bash
nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency \
       ./dynamic_cortex_demo_cuda
```

**Nsight Compute:**
```bash
ncu --set full --export profile.ncu-rep \
    ./dynamic_cortex_demo_cuda

# Open in GUI for detailed analysis
ncu-ui profile.ncu-rep
```

**Key metrics:**
- **Occupancy:** > 75% is good
- **Memory efficiency:** > 80% is good
- **Compute utilization:** > 60% is good

### Event Timing

```cpp
// TT-Metal
auto start = std::chrono::high_resolution_clock::now();
kernel.execute();
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<microseconds>(end - start);

// CUDA
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>();
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## Performance Targets

### Greyskull e75

| Metric | Target | Notes |
|--------|--------|-------|
| V1 Encode | < 15 ms | 64x64 neurons, 64x64 stimulus |
| Stream Transfer | < 3 ms | 256 KB payload |
| Channel Rotation | < 1 ms | Update routing config |
| End-to-end Latency | < 50 ms | Full trial (1 sec stimulus) |
| Power | < 80 W | Including sparse activation |

### RTX 4080

| Metric | Target | Notes |
|--------|--------|-------|
| V1 Encode | < 2 ms | 64x64 neurons, 64x64 stimulus |
| Stream Transfer | < 0.5 ms | Via PCIe |
| Tensor Core MatMul | < 1 ms | 1024x1024 @ FP16 |
| End-to-end Latency | < 10 ms | Full trial |
| Power | < 300 W | Typical workload |

## Troubleshooting

### Low Throughput

**Symptoms:**
- Low GPU/core utilization
- Long execution times

**Solutions:**
1. Increase batch size
2. Check for CPU bottlenecks
3. Profile to find hot spots
4. Verify async execution

### High Latency

**Symptoms:**
- Long time-to-first-result
- Stuttering

**Solutions:**
1. Reduce data transfer size
2. Use pinned memory (CUDA)
3. Prefetch data
4. Optimize kernel launch overhead

### Memory Issues

**Symptoms:**
- Out-of-memory errors
- Excessive swapping

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing
3. Enable compression
4. Stream large datasets

## Best Practices Summary

1. **Use appropriate precision** (FP16 for most cases)
2. **Minimize data movement** (keep data on-device)
3. **Maximize parallelism** (use all cores/SMs)
4. **Optimize memory access** (coalescing, shared memory)
5. **Profile regularly** (identify bottlenecks)
6. **Batch operations** (amortize overhead)
7. **Use hardware features** (Tensor Cores, NOC, etc.)

## See Also

- [Architecture Guide](architecture.md)
- [API Reference](api-reference.md)
