/**
 * Memory Access Pattern Optimization for RTX 4080
 * Demonstrates various memory access patterns and optimizations
 *
 * Target: Ada Lovelace Architecture (sm_89)
 * CUDA 12.x
 *
 * Topics covered:
 * - Memory coalescing
 * - Shared memory usage
 * - Bank conflicts
 * - Texture memory
 * - Constant memory
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define ARRAY_SIZE (32 * 1024 * 1024)  // 32M elements
#define BLOCK_SIZE 256

// Constant memory example
__constant__ float constParams[256];

/**
 * Uncoalesced memory access (BAD)
 * Each thread accesses strided memory locations
 */
__global__ void uncoalescedAccess(float* input, float* output, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = idx; i < ARRAY_SIZE; i += totalThreads) {
        int stridedIdx = (i * stride) % ARRAY_SIZE;
        output[i] = input[stridedIdx] * 2.0f;
    }
}

/**
 * Coalesced memory access (GOOD)
 * Threads in a warp access consecutive memory locations
 */
__global__ void coalescedAccess(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int i = idx; i < ARRAY_SIZE; i += totalThreads) {
        output[i] = input[i] * 2.0f;
    }
}

/**
 * Shared memory optimization
 * Load data to shared memory first, then process
 */
__global__ void sharedMemoryOptimized(float* input, float* output) {
    __shared__ float sharedData[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory (coalesced)
    if (idx < ARRAY_SIZE) {
        sharedData[tid] = input[idx];
    }
    __syncthreads();

    // Process data from shared memory
    if (idx < ARRAY_SIZE) {
        float value = sharedData[tid];
        // Simulate computation
        value = value * 2.0f + 1.0f;
        value = sqrtf(value);
        output[idx] = value;
    }
}

/**
 * Shared memory with bank conflict (BAD)
 * Threads access same bank causing serialization
 */
__global__ void sharedMemoryBankConflict(float* input, float* output) {
    __shared__ float sharedData[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < ARRAY_SIZE) {
        sharedData[tid] = input[idx];
    }
    __syncthreads();

    // Accessing with stride of 32 causes bank conflicts on 32-bank shared memory
    int conflictIdx = (tid * 32) % BLOCK_SIZE;
    if (idx < ARRAY_SIZE) {
        output[idx] = sharedData[conflictIdx] * 2.0f;
    }
}

/**
 * Shared memory without bank conflict (GOOD)
 * Padding to avoid bank conflicts
 */
#define PADDED_SIZE (BLOCK_SIZE + 1)  // Add padding
__global__ void sharedMemoryNoBankConflict(float* input, float* output) {
    __shared__ float sharedData[PADDED_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < ARRAY_SIZE) {
        sharedData[tid] = input[idx];
    }
    __syncthreads();

    if (idx < ARRAY_SIZE) {
        output[idx] = sharedData[tid] * 2.0f;
    }
}

/**
 * Constant memory usage
 * Good for read-only data accessed uniformly by all threads
 */
__global__ void constantMemoryAccess(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ARRAY_SIZE) {
        float value = input[idx];
        // Use constant memory parameters
        value = value * constParams[idx % 256];
        output[idx] = value;
    }
}

/**
 * Vectorized memory access using float4
 * Loads 128 bits (4 floats) in a single transaction
 */
__global__ void vectorizedAccess(float* input, float* output) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < ARRAY_SIZE) {
        float4* input4 = reinterpret_cast<float4*>(input);
        float4* output4 = reinterpret_cast<float4*>(output);

        float4 data = input4[idx / 4];
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        output4[idx / 4] = data;
    }
}

/**
 * Benchmark a kernel
 */
float benchmarkKernel(void (*kernel)(float*, float*), float* d_input, float* d_output,
                      const char* name, int numBlocks, int numThreads) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    kernel<<<numBlocks, numThreads>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<numBlocks, numThreads>>>(d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iterations;

    // Calculate bandwidth
    float bandwidth = (2.0f * ARRAY_SIZE * sizeof(float)) / (milliseconds / 1000.0f) / 1e9;

    printf("%-35s: %7.3f ms, %7.2f GB/s\n", name, milliseconds, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return bandwidth;
}

/**
 * Wrapper for kernels with stride parameter
 */
__global__ void uncoalescedAccessWrapper(float* input, float* output) {
    uncoalescedAccess(input, output, 32);
}

int main() {
    printf("=== Memory Access Pattern Optimization ===\n");
    printf("Array size: %d elements (%.2f MB)\n\n", ARRAY_SIZE,
           ARRAY_SIZE * sizeof(float) / (1024.0 * 1024.0));

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Peak Memory Bandwidth: %.2f GB/s\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    // Allocate memory
    float* h_input = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_output = (float*)malloc(ARRAY_SIZE * sizeof(float));

    // Initialize input
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, ARRAY_SIZE * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Initialize constant memory
    float params[256];
    for (int i = 0; i < 256; i++) {
        params[i] = 1.0f + i * 0.01f;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(constParams, params, 256 * sizeof(float)));

    // Setup grid and block dimensions
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numThreads = BLOCK_SIZE;

    printf("=== Benchmarks ===\n");
    printf("Grid: %d blocks, Block: %d threads\n\n", numBlocks, numThreads);

    // Run benchmarks
    benchmarkKernel(uncoalescedAccessWrapper, d_input, d_output,
                    "Uncoalesced Access (Bad)", numBlocks, numThreads);
    benchmarkKernel(coalescedAccess, d_input, d_output,
                    "Coalesced Access (Good)", numBlocks, numThreads);
    benchmarkKernel(sharedMemoryOptimized, d_input, d_output,
                    "Shared Memory Optimized", numBlocks, numThreads);
    benchmarkKernel(sharedMemoryBankConflict, d_input, d_output,
                    "Shared Memory w/ Bank Conflict", numBlocks, numThreads);
    benchmarkKernel(sharedMemoryNoBankConflict, d_input, d_output,
                    "Shared Memory No Bank Conflict", numBlocks, numThreads);
    benchmarkKernel(constantMemoryAccess, d_input, d_output,
                    "Constant Memory Access", numBlocks, numThreads);
    benchmarkKernel(vectorizedAccess, d_input, d_output,
                    "Vectorized Access (float4)", numBlocks / 4, numThreads);

    printf("\n=== Summary ===\n");
    printf("Key takeaways:\n");
    printf("1. Coalesced memory access is crucial for performance\n");
    printf("2. Shared memory can reduce global memory traffic\n");
    printf("3. Avoid bank conflicts in shared memory\n");
    printf("4. Use constant memory for read-only uniform data\n");
    printf("5. Vectorized loads can improve bandwidth utilization\n");
    printf("6. RTX 4080 has 716.8 GB/s theoretical peak bandwidth\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
