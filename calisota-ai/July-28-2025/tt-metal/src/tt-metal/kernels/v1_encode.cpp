/**
 * @file v1_encode.cpp
 * @brief V1 visual encoding kernel for TT-Metal
 *
 * Implements Gabor filter-based visual feature extraction
 * optimized for Tensix cores.
 *
 * @author GRYPHGEN Project
 * @date 2025
 * @license Apache 2.0
 */

#include <cstdint>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// TT-Metal kernel programming model
// Note: This uses TT-Metal's compute kernel API
// Actual implementation depends on SDK version

namespace dynamic_cortex {
namespace tt_metal_kernels {

/**
 * @brief Gabor filter parameters
 */
struct GaborParams {
    float orientation;      // Preferred orientation (radians)
    float spatial_freq;     // Spatial frequency (cycles/pixel)
    float phase;            // Phase offset
    float sigma_x;          // Gaussian envelope width (x)
    float sigma_y;          // Gaussian envelope width (y)
    float aspect_ratio;     // Aspect ratio
};

/**
 * @brief V1 neuron state
 */
struct V1Neuron {
    float activity;         // Current firing rate
    GaborParams filter;     // Receptive field parameters
    float threshold;        // Activation threshold
    float gain;             // Response gain
};

/**
 * @brief Gabor filter evaluation
 * @param x X coordinate relative to filter center
 * @param y Y coordinate relative to filter center
 * @param params Gabor filter parameters
 * @return Filter response
 */
inline float evaluateGabor(float x, float y, const GaborParams& params) {
    // Rotate coordinates
    float cos_theta = cosf(params.orientation);
    float sin_theta = sinf(params.orientation);
    float x_rot = x * cos_theta + y * sin_theta;
    float y_rot = -x * sin_theta + y * cos_theta;

    // Gaussian envelope
    float gaussian = expf(
        -(x_rot * x_rot / (2.0f * params.sigma_x * params.sigma_x) +
          y_rot * y_rot / (2.0f * params.sigma_y * params.sigma_y))
    );

    // Sinusoidal carrier
    float carrier = cosf(2.0f * M_PI * params.spatial_freq * x_rot + params.phase);

    return gaussian * carrier;
}

/**
 * @brief V1 encoding kernel (compute kernel)
 *
 * This kernel runs on each Tensix core and processes a subset of V1 neurons.
 * Input: Visual stimulus (image patch)
 * Output: V1 neuron activities
 */
extern "C" void v1_encode_kernel(
    const float* stimulus,      // Input stimulus buffer
    float* activity,            // Output activity buffer
    const V1Neuron* neurons,    // Neuron parameters
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons,
    uint32_t neuron_offset      // Offset for this core's neurons
) {
    // Process neurons assigned to this core
    for (uint32_t n = 0; n < num_neurons; ++n) {
        const V1Neuron& neuron = neurons[neuron_offset + n];
        float response = 0.0f;

        // Convolve Gabor filter with stimulus
        for (uint32_t y = 0; y < stimulus_height; ++y) {
            for (uint32_t x = 0; x < stimulus_width; ++x) {
                // Center coordinates
                float cx = static_cast<float>(x) - stimulus_width / 2.0f;
                float cy = static_cast<float>(y) - stimulus_height / 2.0f;

                float filter_val = evaluateGabor(cx, cy, neuron.filter);
                float stimulus_val = stimulus[y * stimulus_width + x];

                response += filter_val * stimulus_val;
            }
        }

        // Apply gain and threshold
        response *= neuron.gain;
        response = (response > neuron.threshold) ? response - neuron.threshold : 0.0f;

        // Apply rectification (ReLU)
        activity[neuron_offset + n] = fmaxf(0.0f, response);
    }
}

/**
 * @brief Optimized V1 encoding using Tensix SIMD
 *
 * This version uses Tensix's SIMD capabilities for parallel processing.
 */
extern "C" void v1_encode_kernel_simd(
    const float* __restrict__ stimulus,
    float* __restrict__ activity,
    const V1Neuron* __restrict__ neurons,
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons,
    uint32_t neuron_offset
) {
    // Tensix supports 32-wide SIMD operations
    constexpr uint32_t SIMD_WIDTH = 32;

    // Process neurons in SIMD batches
    for (uint32_t n = 0; n < num_neurons; n += SIMD_WIDTH) {
        uint32_t batch_size = fminf(SIMD_WIDTH, num_neurons - n);

        // SIMD vector for accumulating responses
        float responses[SIMD_WIDTH] = {0.0f};

        // Convolve filters with stimulus (outer loops over spatial dimensions)
        for (uint32_t y = 0; y < stimulus_height; ++y) {
            for (uint32_t x = 0; x < stimulus_width; ++x) {
                float cx = static_cast<float>(x) - stimulus_width / 2.0f;
                float cy = static_cast<float>(y) - stimulus_height / 2.0f;
                float stimulus_val = stimulus[y * stimulus_width + x];

                // Process batch of neurons in parallel
                #pragma unroll
                for (uint32_t b = 0; b < batch_size; ++b) {
                    const V1Neuron& neuron = neurons[neuron_offset + n + b];
                    float filter_val = evaluateGabor(cx, cy, neuron.filter);
                    responses[b] += filter_val * stimulus_val;
                }
            }
        }

        // Apply gain, threshold, and rectification
        #pragma unroll
        for (uint32_t b = 0; b < batch_size; ++b) {
            const V1Neuron& neuron = neurons[neuron_offset + n + b];
            float response = responses[b] * neuron.gain;
            response = (response > neuron.threshold) ? response - neuron.threshold : 0.0f;
            activity[neuron_offset + n + b] = fmaxf(0.0f, response);
        }
    }
}

/**
 * @brief Initialize V1 neuron bank
 * @param neurons Output neuron array
 * @param num_neurons Number of neurons to initialize
 * @param orientation_range Range of orientations (radians)
 */
void initializeV1Neurons(
    V1Neuron* neurons,
    uint32_t num_neurons,
    float orientation_range = M_PI
) {
    for (uint32_t n = 0; n < num_neurons; ++n) {
        V1Neuron& neuron = neurons[n];

        // Distribute orientations evenly
        neuron.filter.orientation = (static_cast<float>(n) / num_neurons) * orientation_range;

        // Spatial frequency (typical V1 range)
        neuron.filter.spatial_freq = 0.05f + (n % 8) * 0.01f;  // 0.05 - 0.12 cycles/pixel

        // Phase (0 or π for even/odd symmetric)
        neuron.filter.phase = (n % 2) * M_PI;

        // Gaussian envelope parameters
        neuron.filter.sigma_x = 4.0f;
        neuron.filter.sigma_y = 4.0f;
        neuron.filter.aspect_ratio = 1.5f;

        // Activation parameters
        neuron.threshold = 0.1f;
        neuron.gain = 1.0f;
        neuron.activity = 0.0f;
    }
}

/**
 * @brief Generate visual grating stimulus
 * @param stimulus Output stimulus buffer
 * @param width Stimulus width
 * @param height Stimulus height
 * @param orientation Grating orientation (degrees)
 * @param spatial_freq Spatial frequency (cycles/degree)
 * @param contrast Contrast [0.0, 1.0]
 */
void generateGratingStimulus(
    float* stimulus,
    uint32_t width,
    uint32_t height,
    float orientation_deg,
    float spatial_freq = 0.05f,
    float contrast = 1.0f
) {
    float orientation_rad = orientation_deg * M_PI / 180.0f;
    float cos_theta = cosf(orientation_rad);
    float sin_theta = sinf(orientation_rad);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            // Center coordinates
            float cx = static_cast<float>(x) - width / 2.0f;
            float cy = static_cast<float>(y) - height / 2.0f;

            // Rotate coordinates
            float x_rot = cx * cos_theta + cy * sin_theta;

            // Sinusoidal grating
            float value = 0.5f + 0.5f * contrast * cosf(2.0f * M_PI * spatial_freq * x_rot);

            stimulus[y * width + x] = value;
        }
    }
}

} // namespace tt_metal_kernels
} // namespace dynamic_cortex
