/**
 * @file dynamic_cortex.hpp
 * @brief Main API for Dynamic Cortex Communication Framework
 *
 * Provides high-level abstractions for implementing neuroscience-inspired
 * algorithms on Tenstorrent and NVIDIA hardware accelerators.
 *
 * @author GRYPHGEN Project
 * @date 2025
 * @license Apache 2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <cstdint>

namespace dynamic_cortex {

// Forward declarations
class CorticalArea;
class Stream;
class EventScheduler;
class TensorOps;

// =============================================================================
// Type Definitions
// =============================================================================

using milliseconds = std::chrono::milliseconds;
using microseconds = std::chrono::microseconds;
using Timestamp = std::chrono::high_resolution_clock::time_point;

/**
 * @brief Hardware backend types
 */
enum class HardwareType {
    GREYSKULL_E75,   ///< Tenstorrent Greyskull e75 (8GB)
    GREYSKULL_E150,  ///< Tenstorrent Greyskull e150 (16GB)
    RTX_4080,        ///< NVIDIA RTX 4080 (16GB)
    AUTO             ///< Auto-detect available hardware
};

/**
 * @brief Stream routing modes
 */
enum class RoutingMode {
    STATIC,          ///< Fixed routing (set at initialization)
    DYNAMIC,         ///< Reconfigurable routing (updated via events)
    ADAPTIVE         ///< Hardware-managed adaptive routing
};

/**
 * @brief Stream bandwidth settings
 */
enum class Bandwidth {
    LOW,             ///< Minimal bandwidth (power-saving)
    NORMAL,          ///< Standard bandwidth
    HIGH,            ///< Maximum bandwidth (performance)
    ADAPTIVE         ///< Dynamic bandwidth adjustment
};

/**
 * @brief Tensor data types
 */
enum class DataType {
    FP32,            ///< 32-bit floating point
    FP16,            ///< 16-bit floating point
    BF16,            ///< Brain float 16
    INT8,            ///< 8-bit integer (quantized)
    INT32            ///< 32-bit integer
};

/**
 * @brief Tensor dimensions
 */
struct Dimensions {
    uint32_t rows;
    uint32_t cols;
    uint32_t depth = 1;  ///< Optional third dimension

    size_t size() const { return rows * cols * depth; }
};

/**
 * @brief Behavioral context for modulating processing
 */
struct Context {
    bool is_rewarded;          ///< Reward association flag
    bool is_go_trial;          ///< Go/No-go task indicator
    float attention_level;     ///< Attention modulation [0.0, 1.0]
    uint32_t trial_number;     ///< Current trial index
    Timestamp timestamp;       ///< Context creation time

    Context() : is_rewarded(false), is_go_trial(false),
                attention_level(1.0f), trial_number(0),
                timestamp(std::chrono::high_resolution_clock::now()) {}
};

/**
 * @brief Stimulus input data
 */
struct Stimulus {
    std::vector<float> data;   ///< Raw stimulus values
    Dimensions dims;           ///< Spatial dimensions
    DataType dtype;            ///< Data type
    std::string label;         ///< Optional label/identifier

    Stimulus() : dtype(DataType::FP32) {}
};

/**
 * @brief Tensor data container
 */
class Tensor {
public:
    Tensor() = default;
    Tensor(Dimensions dims, DataType dtype);
    ~Tensor() = default;

    // Data access
    float* data();
    const float* data() const;
    size_t size() const;
    Dimensions dimensions() const;
    DataType dataType() const;

    // Operations
    Tensor clone() const;
    void copyFrom(const Tensor& other);
    void copyTo(Tensor& other) const;
    void fill(float value);
    void randomize(float min = 0.0f, float max = 1.0f);

private:
    std::vector<float> data_;
    Dimensions dims_;
    DataType dtype_;
};

// =============================================================================
// Stream Configuration
// =============================================================================

/**
 * @brief Configuration for inter-area communication streams
 */
class StreamConfig {
public:
    StreamConfig() = default;

    // Builder pattern setters
    StreamConfig& setRoutingMode(RoutingMode mode);
    StreamConfig& setBandwidth(Bandwidth bw);
    StreamConfig& setLatency(milliseconds latency);
    StreamConfig& setRotationPeriod(milliseconds period);
    StreamConfig& enableSparseTransfer(bool enable);
    StreamConfig& setCompressionRatio(float ratio);

    // Getters
    RoutingMode routingMode() const;
    Bandwidth bandwidth() const;
    milliseconds latency() const;
    milliseconds rotationPeriod() const;
    bool isSparseEnabled() const;
    float compressionRatio() const;

    // Dynamic updates
    void updateRouting(const std::vector<uint32_t>& source_cores,
                       const std::vector<uint32_t>& dest_cores);

private:
    RoutingMode routing_mode_ = RoutingMode::STATIC;
    Bandwidth bandwidth_ = Bandwidth::NORMAL;
    milliseconds latency_{2};
    milliseconds rotation_period_{121};  // Default: No-go trial timescale
    bool sparse_enabled_ = false;
    float compression_ratio_ = 1.0f;
    std::vector<uint32_t> active_source_cores_;
    std::vector<uint32_t> active_dest_cores_;
};

// =============================================================================
// Cortical Area
// =============================================================================

/**
 * @brief Represents a cortical processing area (e.g., V1, LM)
 */
class CorticalArea {
public:
    CorticalArea(const std::string& name, Dimensions dims, HardwareType hw_type);
    ~CorticalArea();

    // Identity
    std::string name() const;
    Dimensions dimensions() const;
    HardwareType hardwareType() const;

    // Activity management
    void setActivity(const Tensor& activity);
    Tensor getActivity() const;
    void activate(const Stimulus& stimulus);
    void reset();

    // Connectivity
    void addInput(std::shared_ptr<Stream> input_stream);
    void addOutput(std::shared_ptr<Stream> output_stream);
    std::vector<std::shared_ptr<Stream>> inputs() const;
    std::vector<std::shared_ptr<Stream>> outputs() const;

    // Hardware mapping
    std::vector<uint32_t> assignedCores() const;
    void assignCores(const std::vector<uint32_t>& core_ids);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Communication Stream
// =============================================================================

/**
 * @brief Data stream between cortical areas
 */
class Stream {
public:
    Stream(std::shared_ptr<CorticalArea> source,
           std::shared_ptr<CorticalArea> target);
    ~Stream();

    // Configuration
    StreamConfig& config();
    const StreamConfig& config() const;

    // Data transfer
    void transfer();
    void transferAsync();
    void synchronize();

    // Status
    bool isTransferring() const;
    milliseconds lastTransferTime() const;
    size_t bytesTransferred() const;

    // Source/target access
    std::shared_ptr<CorticalArea> source() const;
    std::shared_ptr<CorticalArea> target() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Event System
// =============================================================================

/**
 * @brief Event for triggering actions at specific times
 */
class Event {
public:
    using Action = std::function<void()>;

    Event() = default;
    explicit Event(const std::string& name);

    // Configuration
    void setName(const std::string& name);
    void setAction(Action action);
    void setPeriod(milliseconds period);
    void setOneShot(bool one_shot);

    // Control
    void trigger();
    void enable();
    void disable();
    bool isEnabled() const;

    // Info
    std::string name() const;
    milliseconds period() const;
    Timestamp lastTriggerTime() const;

private:
    std::string name_;
    Action action_;
    milliseconds period_{0};
    bool one_shot_ = false;
    bool enabled_ = true;
    Timestamp last_trigger_;
};

/**
 * @brief Scheduler for managing timed events
 */
class EventScheduler {
public:
    EventScheduler() = default;
    ~EventScheduler() = default;

    // Event management
    void addEvent(const Event& event);
    void removeEvent(const std::string& name);
    void clearEvents();

    // Execution
    void step(milliseconds timestep);
    void run(milliseconds duration);
    void stop();

    // Status
    bool isRunning() const;
    Timestamp currentTime() const;

private:
    std::vector<Event> events_;
    bool running_ = false;
    Timestamp start_time_;
    Timestamp current_time_;
};

// =============================================================================
// Main Framework Interface
// =============================================================================

/**
 * @brief Main framework class for dynamic cortex simulations
 */
class DynamicCortex {
public:
    /**
     * @brief Factory method to create framework instance
     * @param hw_type Hardware backend to use
     * @return Shared pointer to framework instance
     */
    static std::shared_ptr<DynamicCortex> create(HardwareType hw_type);

    virtual ~DynamicCortex() = default;

    // Cortical area management
    virtual std::shared_ptr<CorticalArea> createCorticalArea(
        const std::string& name, Dimensions dims) = 0;

    virtual std::shared_ptr<CorticalArea> getCorticalArea(
        const std::string& name) = 0;

    // Stream management
    virtual std::shared_ptr<Stream> createStream(
        std::shared_ptr<CorticalArea> source,
        std::shared_ptr<CorticalArea> target) = 0;

    // Event system
    virtual EventScheduler& scheduler() = 0;
    virtual void setContextHandler(std::function<void(const Context&)> handler) = 0;

    // Execution
    virtual void processStimulus(const Stimulus& stimulus, milliseconds duration) = 0;
    virtual void step(milliseconds timestep) = 0;
    virtual void reset() = 0;

    // Hardware info
    virtual HardwareType hardwareType() const = 0;
    virtual std::string hardwareName() const = 0;
    virtual size_t availableMemory() const = 0;
    virtual uint32_t numCores() const = 0;

    // Profiling
    virtual void enableProfiling(bool enable) = 0;
    virtual std::string getProfilingReport() const = 0;
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Load visual grating stimulus
 * @param orientation Grating orientation in degrees
 * @param spatial_freq Spatial frequency (cycles/degree)
 * @param dims Stimulus dimensions
 * @return Generated stimulus
 */
Stimulus loadVisualGrating(float orientation,
                           float spatial_freq = 0.05f,
                           Dimensions dims = {64, 64, 1});

/**
 * @brief Get available hardware backends
 * @return Vector of detected hardware types
 */
std::vector<HardwareType> getAvailableHardware();

/**
 * @brief Get hardware name string
 * @param hw_type Hardware type
 * @return Human-readable name
 */
std::string getHardwareName(HardwareType hw_type);

/**
 * @brief Set global logging level
 * @param level Log level (0=off, 1=error, 2=warning, 3=info, 4=debug)
 */
void setLogLevel(int level);

} // namespace dynamic_cortex
