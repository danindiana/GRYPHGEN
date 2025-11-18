/**
 * @file stream_config.hpp
 * @brief TT-Metal specific stream configuration implementation
 *
 * Implements dynamic reconfigurable streams using Tenstorrent's
 * Network-on-Chip (NOC) and streaming fabric.
 *
 * @author GRYPHGEN Project
 * @date 2025
 * @license Apache 2.0
 */

#pragma once

#include <vector>
#include <cstdint>
#include <memory>

// TT-Metal SDK includes (adjust paths based on actual SDK installation)
#ifdef HAS_TT_METAL
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/device/device.hpp"
#endif

namespace dynamic_cortex {
namespace tt_metal_backend {

/**
 * @brief NOC (Network-on-Chip) configuration
 */
struct NOCConfig {
    uint8_t noc_id;              ///< NOC instance (0 or 1 on Greyskull)
    uint32_t src_addr;           ///< Source memory address
    uint32_t dst_addr;           ///< Destination memory address
    uint32_t transfer_size;      ///< Transfer size in bytes
    bool use_multicast;          ///< Enable multicast transfer

    NOCConfig() : noc_id(0), src_addr(0), dst_addr(0),
                  transfer_size(0), use_multicast(false) {}
};

/**
 * @brief Core coordinate for Tensix core addressing
 */
struct CoreCoord {
    uint32_t x;
    uint32_t y;

    CoreCoord() : x(0), y(0) {}
    CoreCoord(uint32_t x_, uint32_t y_) : x(x_), y(y_) {}

    uint32_t linear() const { return y * 12 + x; }  // Greyskull: 12x12 grid

    bool operator==(const CoreCoord& other) const {
        return x == other.x && y == other.y;
    }
};

/**
 * @brief Stream endpoint specification
 */
struct StreamEndpoint {
    std::vector<CoreCoord> cores;  ///< Cores participating in stream
    uint32_t buffer_addr;          ///< L1 buffer address
    uint32_t buffer_size;          ///< Buffer size in bytes

    StreamEndpoint() : buffer_addr(0), buffer_size(0) {}
};

/**
 * @brief TT-Metal stream implementation
 */
class TTMetalStream {
public:
    TTMetalStream();
    ~TTMetalStream();

    // Configuration
    void setSourceEndpoint(const StreamEndpoint& endpoint);
    void setDestEndpoint(const StreamEndpoint& endpoint);
    void setNOCConfig(const NOCConfig& config);

    // Dynamic reconfiguration
    void updateRouting(const std::vector<CoreCoord>& new_src_cores,
                       const std::vector<CoreCoord>& new_dst_cores);

    void enableSparseMode(bool enable);
    void setActiveCores(const std::vector<CoreCoord>& active_cores);

    // Data transfer
    void initiateTransfer();
    void waitForCompletion();
    bool isTransferComplete() const;

    // Bandwidth control
    void setBandwidthLimit(uint32_t bytes_per_cycle);
    void setPriority(uint8_t priority);  // 0-7, higher = higher priority

    // Statistics
    uint64_t totalBytesTransferred() const;
    uint32_t currentLatencyCycles() const;
    float utilizationPercent() const;

#ifdef HAS_TT_METAL
    // TT-Metal specific accessors
    void setDevice(tt::tt_metal::Device* device);
    tt::tt_metal::Device* device() const;
#endif

private:
    StreamEndpoint src_endpoint_;
    StreamEndpoint dst_endpoint_;
    NOCConfig noc_config_;
    bool sparse_mode_enabled_;
    std::vector<CoreCoord> active_cores_;

    uint32_t bandwidth_limit_;
    uint8_t priority_;
    uint64_t bytes_transferred_;
    bool transfer_in_progress_;

#ifdef HAS_TT_METAL
    tt::tt_metal::Device* device_;
#else
    void* device_;  // Placeholder when SDK not available
#endif
};

/**
 * @brief Manages multiple streams with coordination
 */
class StreamManager {
public:
    StreamManager();
    ~StreamManager();

    // Stream lifecycle
    uint32_t createStream();
    void destroyStream(uint32_t stream_id);
    TTMetalStream* getStream(uint32_t stream_id);

    // Coordinated transfers
    void transferAll();                    // Sequential
    void transferAllParallel();            // Concurrent on different NOCs
    void synchronizeAll();                 // Wait for all transfers

    // Routing optimization
    void optimizeRouting();                // Minimize NOC conflicts
    void balanceLoad();                    // Balance traffic across NOCs

    // Event-based triggers
    void setRotationCallback(uint32_t stream_id,
                             std::function<void()> callback);
    void triggerRotation(uint32_t stream_id);

private:
    std::vector<std::unique_ptr<TTMetalStream>> streams_;
    std::vector<std::function<void()>> rotation_callbacks_;

    void* noc0_lock_;  // Mutex for NOC 0
    void* noc1_lock_;  // Mutex for NOC 1
};

/**
 * @brief Channel rotation logic implementation
 */
class ChannelRotator {
public:
    ChannelRotator(uint32_t num_channels, uint32_t rotation_period_ms);

    // Rotation schedule
    void setRotationPeriod(uint32_t period_ms);
    uint32_t getCurrentChannel() const;
    std::vector<CoreCoord> getActiveChannelCores(uint32_t channel_id);

    // Behavioral modulation
    void setRewardedMode(bool rewarded);  // Changes rotation speed
    void updateTimestep(uint32_t elapsed_ms);

    // Rotation patterns
    enum class RotationPattern {
        SEQUENTIAL,      // Round-robin through channels
        RANDOM,          // Random channel selection
        ADAPTIVE,        // Based on activity levels
        CUSTOM           // User-defined pattern
    };

    void setRotationPattern(RotationPattern pattern);
    void setCustomPattern(const std::vector<uint32_t>& channel_sequence);

private:
    uint32_t num_channels_;
    uint32_t rotation_period_ms_;
    uint32_t current_channel_;
    uint32_t time_in_channel_;
    bool rewarded_mode_;
    RotationPattern pattern_;
    std::vector<uint32_t> custom_sequence_;
    size_t custom_sequence_idx_;

    void advanceChannel();
};

/**
 * @brief Sparse activation pattern manager
 */
class SparseActivation {
public:
    SparseActivation();

    // Activation patterns
    void setActivationThreshold(float threshold);
    std::vector<CoreCoord> computeActiveCores(const float* activity_data,
                                              size_t num_elements,
                                              const std::vector<CoreCoord>& all_cores);

    // Energy optimization
    void powerDownInactiveCores(const std::vector<CoreCoord>& inactive_cores);
    void powerUpCores(const std::vector<CoreCoord>& cores);

    // Statistics
    float currentSparsity() const;  // Fraction of inactive cores
    float energySavingsPercent() const;

private:
    float activation_threshold_;
    size_t total_cores_;
    size_t active_cores_;
    float baseline_power_;
    float current_power_;
};

/**
 * @brief Utility functions for core management
 */
namespace core_utils {

/**
 * @brief Map linear core ID to 2D coordinate
 */
CoreCoord linearToCoord(uint32_t linear_id);

/**
 * @brief Map 2D coordinate to linear core ID
 */
uint32_t coordToLinear(const CoreCoord& coord);

/**
 * @brief Get all cores in a rectangular region
 */
std::vector<CoreCoord> getCoreRange(CoreCoord start, CoreCoord end);

/**
 * @brief Compute Manhattan distance between cores
 */
uint32_t coreManhattanDistance(const CoreCoord& a, const CoreCoord& b);

/**
 * @brief Find optimal NOC for a given src->dst transfer
 * @return NOC ID (0 or 1)
 */
uint8_t selectOptimalNOC(const CoreCoord& src, const CoreCoord& dst);

} // namespace core_utils

} // namespace tt_metal_backend
} // namespace dynamic_cortex
