/**
 * Tensor Core Utilization Example for RTX 4080
 * Demonstrates FP16/BF16 mixed-precision matrix multiplication
 * using 4th Generation Tensor Cores (304 cores on RTX 4080)
 *
 * Target: Ada Lovelace Architecture (sm_89)
 * CUDA 12.x with cuBLAS
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Print device info
 */
void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== GPU Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    if (prop.major >= 7) {
        printf("Tensor Cores: Supported (Generation %d)\n",
               prop.major == 7 ? 1 : prop.major == 8 ? (prop.minor == 0 ? 2 : 3) : 4);
    } else {
        printf("Tensor Cores: Not supported\n");
    }
    printf("=======================\n\n");
}

/**
 * Initialize matrix with random FP32 values
 */
void initMatrixFP32(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

/**
 * Convert FP32 to FP16
 */
void convertFP32toFP16(__half* dst, const float* src, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

/**
 * FP32 GEMM (standard precision)
 */
void benchmarkFP32GEMM(cublasHandle_t handle, int M, int N, int K,
                       float* d_A, float* d_B, float* d_C) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warm-up
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,
                            d_A, K,
                            &beta,
                            d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                d_B, N,
                                d_A, K,
                                &beta,
                                d_C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (milliseconds / 1000.0);

    printf("FP32 GEMM (Standard Cores):\n");
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Performance: %.2f TFLOPS\n\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/**
 * FP16 GEMM with Tensor Cores (Mixed Precision)
 */
void benchmarkFP16GEMM(cublasHandle_t handle, int M, int N, int K,
                       __half* d_A_fp16, __half* d_B_fp16, __half* d_C_fp16) {
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // Enable Tensor Core math mode
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Warm-up
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B_fp16, N,
                            d_A_fp16, K,
                            &beta,
                            d_C_fp16, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                d_B_fp16, N,
                                d_A_fp16, K,
                                &beta,
                                d_C_fp16, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (milliseconds / 1000.0);

    printf("FP16 GEMM (Tensor Cores):\n");
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Performance: %.2f TFLOPS\n\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Reset to default math mode
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

/**
 * Mixed Precision GEMM (FP16 compute, FP32 accumulate)
 */
void benchmarkMixedPrecisionGEMM(cublasHandle_t handle, int M, int N, int K,
                                 __half* d_A_fp16, __half* d_B_fp16, float* d_C_fp32) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Enable Tensor Core math mode
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Warm-up
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B_fp16, CUDA_R_16F, N,
                             d_A_fp16, CUDA_R_16F, K,
                             &beta,
                             d_C_fp32, CUDA_R_32F, N,
                             CUBLAS_COMPUTE_32F_FAST_16F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B_fp16, CUDA_R_16F, N,
                                 d_A_fp16, CUDA_R_16F, K,
                                 &beta,
                                 d_C_fp32, CUDA_R_32F, N,
                                 CUBLAS_COMPUTE_32F_FAST_16F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (milliseconds / 1000.0);

    printf("Mixed Precision GEMM (FP16 compute, FP32 accumulate):\n");
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Performance: %.2f TFLOPS\n\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Reset to default math mode
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

int main(int argc, char** argv) {
    printDeviceInfo();

    // Matrix dimensions (default: 4096x4096)
    int M = 4096, N = 4096, K = 4096;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    // Allocate host memory for FP32
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);

    // Initialize matrices
    srand(time(NULL));
    initMatrixFP32(h_A, M, K);
    initMatrixFP32(h_B, K, N);

    // Convert to FP16
    size_t sizeA_fp16 = M * K * sizeof(__half);
    size_t sizeB_fp16 = K * N * sizeof(__half);
    size_t sizeC_fp16 = M * N * sizeof(__half);

    __half* h_A_fp16 = (__half*)malloc(sizeA_fp16);
    __half* h_B_fp16 = (__half*)malloc(sizeB_fp16);

    convertFP32toFP16(h_A_fp16, h_A, M * K);
    convertFP32toFP16(h_B_fp16, h_B, K * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    CUDA_CHECK(cudaMalloc(&d_A_fp16, sizeA_fp16));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, sizeB_fp16));
    CUDA_CHECK(cudaMalloc(&d_C_fp16, sizeC_fp16));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A_fp16, sizeA_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B_fp16, sizeB_fp16, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("=== Performance Comparison ===\n\n");

    // Benchmark FP32
    benchmarkFP32GEMM(handle, M, N, K, d_A, d_B, d_C);

    // Benchmark FP16
    benchmarkFP16GEMM(handle, M, N, K, d_A_fp16, d_B_fp16, d_C_fp16);

    // Benchmark Mixed Precision
    benchmarkMixedPrecisionGEMM(handle, M, N, K, d_A_fp16, d_B_fp16, d_C);

    printf("=== RTX 4080 Theoretical Performance ===\n");
    printf("FP32 (CUDA Cores): 48.74 TFLOPS\n");
    printf("FP16 (Tensor Cores): 389.9 TFLOPS\n");
    printf("INT8 (Tensor Cores): 779.8 TOPS\n\n");

    printf("=== Key Takeaways ===\n");
    printf("1. Tensor Cores provide ~8x speedup for FP16 operations\n");
    printf("2. Mixed precision maintains FP32 accuracy with FP16 speed\n");
    printf("3. Ada Lovelace (4th Gen) Tensor Cores support:\n");
    printf("   - FP16, BF16, TF32, FP8, INT8 operations\n");
    printf("   - Faster sparse matrix operations\n");
    printf("   - Improved FP8 precision for AI training\n");
    printf("4. Use cuBLAS with CUBLAS_TENSOR_OP_MATH for automatic Tensor Core usage\n");

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C_fp16));
    free(h_A);
    free(h_B);
    free(h_A_fp16);
    free(h_B_fp16);

    return 0;
}
