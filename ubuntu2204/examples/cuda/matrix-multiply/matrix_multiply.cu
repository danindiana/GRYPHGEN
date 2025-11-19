/**
 * Optimized Matrix Multiplication for NVIDIA RTX 4080
 * Target: Ada Lovelace Architecture (Compute Capability 8.9)
 * CUDA 12.x
 *
 * Features:
 * - Shared memory optimization
 * - Memory coalescing
 * - Tile-based algorithm
 * - Optimized for Ada Lovelace L1/shared memory (100KB)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Tile size optimized for RTX 4080 (Ada Lovelace)
// 32x32 tiles fit well in the 100KB L1/shared memory
#define TILE_SIZE 32

/**
 * Naive matrix multiplication kernel (for comparison)
 */
__global__ void matrixMulNaive(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Optimized matrix multiplication kernel using shared memory
 * C = A * B where A is MxK, B is KxN, C is MxN
 */
__global__ void matrixMulShared(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate global row and column
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            tileA[ty][tx] = A[row * K + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile from B
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Initialize matrix with random values
 */
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * Verify matrix multiplication result
 */
bool verifyResult(const float* C_cpu, const float* C_gpu, int M, int N) {
    const float epsilon = 1e-3f;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(C_cpu[i] - C_gpu[i]) > epsilon) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, C_cpu[i], C_gpu[i]);
            return false;
        }
    }
    return true;
}

/**
 * CPU matrix multiplication (for verification)
 */
void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Print device properties
 */
void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        exit(1);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== GPU Information ===\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("=======================\n\n");
}

int main(int argc, char** argv) {
    // Print device info
    printDeviceInfo();

    // Matrix dimensions (default: 2048x2048)
    int M = 2048;  // Rows of A and C
    int N = 2048;  // Cols of B and C
    int K = 2048;  // Cols of A, Rows of B

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Total elements: %d (%.2f MB per matrix)\n\n",
           M * N, (M * N * sizeof(float)) / (1024.0 * 1024.0));

    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    // Initialize matrices
    srand(time(NULL));
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Setup execution configuration
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Grid: (%d, %d), Block: (%d, %d)\n\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Warm-up run
    matrixMulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark optimized kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int numIterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        matrixMulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecShared;
    CUDA_CHECK(cudaEventElapsedTime(&msecShared, start, stop));
    msecShared /= numIterations;

    // Benchmark naive kernel
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecNaive;
    CUDA_CHECK(cudaEventElapsedTime(&msecNaive, start, stop));
    msecNaive /= numIterations;

    // Copy result back to host
    matrixMulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // CPU computation for verification (skip for large matrices)
    bool verify = (M * N * K < 1000000);  // Skip verification for large matrices
    if (verify) {
        printf("Running CPU verification...\n");
        clock_t cpuStart = clock();
        matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);
        clock_t cpuEnd = clock();
        double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

        printf("CPU time: %.2f ms\n", cpuTime);
        printf("Verification: %s\n\n", verifyResult(h_C_ref, h_C, M, N) ? "PASSED" : "FAILED");
    }

    // Performance metrics
    double flops = 2.0 * M * N * K;  // Multiply-add counts as 2 operations
    double gflopsShared = (flops / 1e9) / (msecShared / 1000.0);
    double gflopsNaive = (flops / 1e9) / (msecNaive / 1000.0);

    printf("=== Performance Results ===\n");
    printf("Optimized (Shared Memory):\n");
    printf("  Time: %.3f ms\n", msecShared);
    printf("  Performance: %.2f GFLOPS\n", gflopsShared);
    printf("\nNaive:\n");
    printf("  Time: %.3f ms\n", msecNaive);
    printf("  Performance: %.2f GFLOPS\n", gflopsNaive);
    printf("\nSpeedup: %.2fx\n", msecNaive / msecShared);
    printf("===========================\n");

    // RTX 4080 theoretical peak: 48.74 TFLOPS (FP32)
    printf("\nRTX 4080 Theoretical Peak: 48.74 TFLOPS (FP32)\n");
    printf("Achieved: %.2f%% of peak\n", (gflopsShared / 48740.0) * 100.0);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
