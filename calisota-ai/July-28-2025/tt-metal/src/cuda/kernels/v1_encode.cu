/**
 * @file v1_encode.cu
 * @brief V1 visual encoding CUDA kernel for RTX 4080
 *
 * Implements Gabor filter-based visual feature extraction
 * optimized for NVIDIA GPUs with Tensor Cores.
 *
 * @author GRYPHGEN Project
 * @date 2025
 * @license Apache 2.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dynamic_cortex {
namespace cuda_kernels {

/**
 * @brief Gabor filter parameters (device-side)
 */
struct GaborParams {
    float orientation;
    float spatial_freq;
    float phase;
    float sigma_x;
    float sigma_y;
    float aspect_ratio;
};

/**
 * @brief V1 neuron state (device-side)
 */
struct V1Neuron {
    float activity;
    GaborParams filter;
    float threshold;
    float gain;
};

/**
 * @brief Device function: Evaluate Gabor filter
 */
__device__ __forceinline__ float evaluateGabor(
    float x, float y,
    const GaborParams& params
) {
    // Rotate coordinates
    float cos_theta = __cosf(params.orientation);
    float sin_theta = __sinf(params.orientation);
    float x_rot = x * cos_theta + y * sin_theta;
    float y_rot = -x * sin_theta + y * cos_theta;

    // Gaussian envelope (using fast exponential)
    float gaussian = __expf(
        -(x_rot * x_rot / (2.0f * params.sigma_x * params.sigma_x) +
          y_rot * y_rot / (2.0f * params.sigma_y * params.sigma_y))
    );

    // Sinusoidal carrier
    float carrier = __cosf(2.0f * M_PI * params.spatial_freq * x_rot + params.phase);

    return gaussian * carrier;
}

/**
 * @brief V1 encoding kernel (basic version)
 *
 * Grid: (num_neurons + 255) / 256 blocks
 * Block: 256 threads
 */
__global__ void v1_encode_kernel(
    const float* __restrict__ stimulus,
    float* __restrict__ activity,
    const V1Neuron* __restrict__ neurons,
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons
) {
    uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id >= num_neurons) return;

    const V1Neuron& neuron = neurons[neuron_id];
    float response = 0.0f;

    // Convolve Gabor filter with stimulus
    for (uint32_t y = 0; y < stimulus_height; ++y) {
        for (uint32_t x = 0; x < stimulus_width; ++x) {
            float cx = static_cast<float>(x) - stimulus_width / 2.0f;
            float cy = static_cast<float>(y) - stimulus_height / 2.0f;

            float filter_val = evaluateGabor(cx, cy, neuron.filter);
            float stimulus_val = stimulus[y * stimulus_width + x];

            response += filter_val * stimulus_val;
        }
    }

    // Apply gain and threshold with ReLU
    response *= neuron.gain;
    response = (response > neuron.threshold) ? response - neuron.threshold : 0.0f;
    activity[neuron_id] = fmaxf(0.0f, response);
}

/**
 * @brief V1 encoding kernel with shared memory optimization
 *
 * Uses shared memory to cache stimulus patches for faster access.
 */
__global__ void v1_encode_kernel_shared(
    const float* __restrict__ stimulus,
    float* __restrict__ activity,
    const V1Neuron* __restrict__ neurons,
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons
) {
    // Shared memory for stimulus tile
    extern __shared__ float shared_stimulus[];

    uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    // Cooperatively load stimulus into shared memory
    uint32_t stimulus_size = stimulus_width * stimulus_height;
    for (uint32_t i = tid; i < stimulus_size; i += blockDim.x) {
        shared_stimulus[i] = stimulus[i];
    }
    __syncthreads();

    if (neuron_id >= num_neurons) return;

    const V1Neuron& neuron = neurons[neuron_id];
    float response = 0.0f;

    // Convolve using shared memory
    for (uint32_t y = 0; y < stimulus_height; ++y) {
        for (uint32_t x = 0; x < stimulus_width; ++x) {
            float cx = static_cast<float>(x) - stimulus_width / 2.0f;
            float cy = static_cast<float>(y) - stimulus_height / 2.0f;

            float filter_val = evaluateGabor(cx, cy, neuron.filter);
            float stimulus_val = shared_stimulus[y * stimulus_width + x];

            response += filter_val * stimulus_val;
        }
    }

    response *= neuron.gain;
    response = (response > neuron.threshold) ? response - neuron.threshold : 0.0f;
    activity[neuron_id] = fmaxf(0.0f, response);
}

/**
 * @brief V1 encoding kernel with warp-level primitives
 *
 * Uses warp shuffle operations for efficient reduction.
 */
__global__ void v1_encode_kernel_warp(
    const float* __restrict__ stimulus,
    float* __restrict__ activity,
    const V1Neuron* __restrict__ neurons,
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons
) {
    uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    if (neuron_id >= num_neurons) return;

    const V1Neuron& neuron = neurons[neuron_id];
    float response = 0.0f;

    uint32_t stimulus_size = stimulus_width * stimulus_height;

    // Each thread in warp processes subset of pixels
    for (uint32_t i = warp.thread_rank(); i < stimulus_size; i += warp.size()) {
        uint32_t y = i / stimulus_width;
        uint32_t x = i % stimulus_width;

        float cx = static_cast<float>(x) - stimulus_width / 2.0f;
        float cy = static_cast<float>(y) - stimulus_height / 2.0f;

        float filter_val = evaluateGabor(cx, cy, neuron.filter);
        float stimulus_val = stimulus[i];

        response += filter_val * stimulus_val;
    }

    // Warp-level reduction (sum across threads)
    #pragma unroll
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        response += warp.shfl_down(response, offset);
    }

    // Only first thread in warp writes result
    if (warp.thread_rank() == 0) {
        response *= neuron.gain;
        response = (response > neuron.threshold) ? response - neuron.threshold : 0.0f;
        activity[neuron_id] = fmaxf(0.0f, response);
    }
}

/**
 * @brief Host function: Launch V1 encoding kernel
 */
cudaError_t launchV1Encode(
    const float* d_stimulus,
    float* d_activity,
    const V1Neuron* d_neurons,
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons,
    cudaStream_t stream = 0
) {
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;

    // Choose kernel based on stimulus size
    size_t stimulus_size = stimulus_width * stimulus_height * sizeof(float);

    if (stimulus_size <= 48 * 1024) {  // Fits in shared memory
        size_t shared_mem_size = stimulus_size;
        v1_encode_kernel_shared<<<grid_size, block_size, shared_mem_size, stream>>>(
            d_stimulus, d_activity, d_neurons,
            stimulus_width, stimulus_height, num_neurons
        );
    } else {
        v1_encode_kernel<<<grid_size, block_size, 0, stream>>>(
            d_stimulus, d_activity, d_neurons,
            stimulus_width, stimulus_height, num_neurons
        );
    }

    return cudaGetLastError();
}

/**
 * @brief Host function: Initialize V1 neurons
 */
void initializeV1Neurons(
    V1Neuron* h_neurons,
    uint32_t num_neurons,
    float orientation_range = M_PI
) {
    for (uint32_t n = 0; n < num_neurons; ++n) {
        V1Neuron& neuron = h_neurons[n];

        neuron.filter.orientation = (static_cast<float>(n) / num_neurons) * orientation_range;
        neuron.filter.spatial_freq = 0.05f + (n % 8) * 0.01f;
        neuron.filter.phase = (n % 2) * M_PI;
        neuron.filter.sigma_x = 4.0f;
        neuron.filter.sigma_y = 4.0f;
        neuron.filter.aspect_ratio = 1.5f;

        neuron.threshold = 0.1f;
        neuron.gain = 1.0f;
        neuron.activity = 0.0f;
    }
}

/**
 * @brief Generate visual grating stimulus (GPU kernel)
 */
__global__ void generateGratingKernel(
    float* stimulus,
    uint32_t width,
    uint32_t height,
    float orientation_rad,
    float spatial_freq,
    float contrast
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float cos_theta = __cosf(orientation_rad);
    float sin_theta = __sinf(orientation_rad);

    float cx = static_cast<float>(x) - width / 2.0f;
    float cy = static_cast<float>(y) - height / 2.0f;

    float x_rot = cx * cos_theta + cy * sin_theta;

    float value = 0.5f + 0.5f * contrast * __cosf(2.0f * M_PI * spatial_freq * x_rot);

    stimulus[y * width + x] = value;
}

/**
 * @brief Host function: Generate grating stimulus
 */
cudaError_t generateGratingStimulus(
    float* d_stimulus,
    uint32_t width,
    uint32_t height,
    float orientation_deg,
    float spatial_freq = 0.05f,
    float contrast = 1.0f,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    float orientation_rad = orientation_deg * M_PI / 180.0f;

    generateGratingKernel<<<grid, block, 0, stream>>>(
        d_stimulus, width, height,
        orientation_rad, spatial_freq, contrast
    );

    return cudaGetLastError();
}

/**
 * @brief V1 encoding wrapper class
 */
class V1Encoder {
public:
    V1Encoder(uint32_t num_neurons, uint32_t stimulus_width, uint32_t stimulus_height)
        : num_neurons_(num_neurons)
        , stimulus_width_(stimulus_width)
        , stimulus_height_(stimulus_height)
        , d_stimulus_(nullptr)
        , d_activity_(nullptr)
        , d_neurons_(nullptr)
    {
        allocateDeviceMemory();
        initializeNeurons();
    }

    ~V1Encoder() {
        freeDeviceMemory();
    }

    void encode(const float* h_stimulus, float* h_activity, cudaStream_t stream = 0) {
        // Copy stimulus to device
        cudaMemcpyAsync(d_stimulus_, h_stimulus,
                        stimulus_width_ * stimulus_height_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Launch kernel
        launchV1Encode(d_stimulus_, d_activity_, d_neurons_,
                       stimulus_width_, stimulus_height_, num_neurons_, stream);

        // Copy activity back to host
        cudaMemcpyAsync(h_activity, d_activity_,
                        num_neurons_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }

private:
    uint32_t num_neurons_;
    uint32_t stimulus_width_;
    uint32_t stimulus_height_;

    float* d_stimulus_;
    float* d_activity_;
    V1Neuron* d_neurons_;

    void allocateDeviceMemory() {
        cudaMalloc(&d_stimulus_, stimulus_width_ * stimulus_height_ * sizeof(float));
        cudaMalloc(&d_activity_, num_neurons_ * sizeof(float));
        cudaMalloc(&d_neurons_, num_neurons_ * sizeof(V1Neuron));
    }

    void freeDeviceMemory() {
        cudaFree(d_stimulus_);
        cudaFree(d_activity_);
        cudaFree(d_neurons_);
    }

    void initializeNeurons() {
        V1Neuron* h_neurons = new V1Neuron[num_neurons_];
        initializeV1Neurons(h_neurons, num_neurons_);
        cudaMemcpy(d_neurons_, h_neurons, num_neurons_ * sizeof(V1Neuron),
                   cudaMemcpyHostToDevice);
        delete[] h_neurons;
    }
};

} // namespace cuda_kernels
} // namespace dynamic_cortex
