# API Reference

Complete API reference for the Dynamic Cortex Communication Framework.

## Table of Contents

- [Core API](#core-api)
- [TT-Metal Backend](#tt-metal-backend)
- [CUDA Backend](#cuda-backend)
- [Utilities](#utilities)

## Core API

### DynamicCortex

Main framework class for creating and managing cortical simulations.

```cpp
#include "dynamic_cortex.hpp"

namespace dynamic_cortex {
class DynamicCortex;
}
```

#### Factory Method

```cpp
static std::shared_ptr<DynamicCortex> create(HardwareType hw_type);
```

Creates a new framework instance for the specified hardware backend.

**Parameters:**
- `hw_type`: Hardware backend type (`GREYSKULL_E75`, `GREYSKULL_E150`, `RTX_4080`, `AUTO`)

**Returns:**
- Shared pointer to framework instance

**Example:**
```cpp
auto framework = DynamicCortex::create(HardwareType::GREYSKULL_E75);
```

#### createCorticalArea

```cpp
virtual std::shared_ptr<CorticalArea> createCorticalArea(
    const std::string& name,
    Dimensions dims) = 0;
```

Create a new cortical processing area.

**Parameters:**
- `name`: Unique identifier for the area (e.g., "V1", "LM")
- `dims`: Spatial dimensions of the neuron grid

**Returns:**
- Shared pointer to cortical area

**Example:**
```cpp
auto v1 = framework->createCorticalArea("V1", {64, 64});
auto lm = framework->createCorticalArea("LM", {32, 32});
```

#### createStream

```cpp
virtual std::shared_ptr<Stream> createStream(
    std::shared_ptr<CorticalArea> source,
    std::shared_ptr<CorticalArea> target) = 0;
```

Create a communication stream between two cortical areas.

**Parameters:**
- `source`: Source cortical area
- `target`: Target cortical area

**Returns:**
- Shared pointer to stream

**Example:**
```cpp
auto feedforward = framework->createStream(v1, lm);
auto feedback = framework->createStream(lm, v1);
```

### CorticalArea

Represents a cortical processing area with local computation and connectivity.

```cpp
class CorticalArea {
public:
    CorticalArea(const std::string& name, Dimensions dims, HardwareType hw_type);

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

    // Hardware mapping
    std::vector<uint32_t> assignedCores() const;
    void assignCores(const std::vector<uint32_t>& core_ids);
};
```

#### Example Usage

```cpp
auto v1 = std::make_shared<CorticalArea>("V1", Dimensions{64, 64}, HardwareType::GREYSKULL_E75);

// Set initial activity
Tensor initial_activity({64, 64}, DataType::FP32);
initial_activity.fill(0.0f);
v1->setActivity(initial_activity);

// Process stimulus
Stimulus grating = loadVisualGrating(45.0f);  // 45° orientation
v1->activate(grating);

// Retrieve activity
Tensor response = v1->getActivity();
```

### Stream

Data stream for inter-area communication with dynamic routing.

```cpp
class Stream {
public:
    Stream(std::shared_ptr<CorticalArea> source,
           std::shared_ptr<CorticalArea> target);

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
};
```

#### Example Usage

```cpp
auto stream = std::make_shared<Stream>(v1, lm);

// Configure stream
stream->config()
    .setRoutingMode(RoutingMode::DYNAMIC)
    .setBandwidth(Bandwidth::HIGH)
    .setLatency(milliseconds(2))
    .setRotationPeriod(milliseconds(15));

// Transfer data
stream->transferAsync();
// ... do other work ...
stream->synchronize();
```

### StreamConfig

Configuration for communication streams.

```cpp
class StreamConfig {
public:
    // Builder pattern setters
    StreamConfig& setRoutingMode(RoutingMode mode);
    StreamConfig& setBandwidth(Bandwidth bw);
    StreamConfig& setLatency(milliseconds latency);
    StreamConfig& setRotationPeriod(milliseconds period);
    StreamConfig& enableSparseTransfer(bool enable);
    StreamConfig& setCompressionRatio(float ratio);

    // Dynamic updates
    void updateRouting(const std::vector<uint32_t>& source_cores,
                       const std::vector<uint32_t>& dest_cores);
};
```

### Tensor

Multi-dimensional data container with hardware memory management.

```cpp
class Tensor {
public:
    Tensor();
    Tensor(Dimensions dims, DataType dtype);

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
};
```

### Event

Event for triggering actions at specific times.

```cpp
class Event {
public:
    using Action = std::function<void()>;

    Event();
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
};
```

#### Example Usage

```cpp
Event rotation_event("ChannelRotation");
rotation_event.setPeriod(milliseconds(15));  // 15ms for rewarded
rotation_event.setAction([&]() {
    // Rotate communication channels
    stream->config().updateRouting(new_src_cores, new_dst_cores);
});

scheduler.addEvent(rotation_event);
```

### EventScheduler

Manages timed events with priority-based execution.

```cpp
class EventScheduler {
public:
    EventScheduler();

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
};
```

## TT-Metal Backend

### TTMetalStream

TT-Metal specific stream implementation using NOC.

```cpp
namespace tt_metal_backend {

class TTMetalStream {
public:
    TTMetalStream();

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
};

}
```

### ChannelRotator

Manages rotating communication channels.

```cpp
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

    enum class RotationPattern {
        SEQUENTIAL,
        RANDOM,
        ADAPTIVE,
        CUSTOM
    };

    void setRotationPattern(RotationPattern pattern);
};
```

### Tensor Operations

```cpp
class MatMulOp {
public:
    MatMulOp& setMathFidelity(MathFidelity fidelity);
    MatMulOp& setTransposeA(bool transpose);
    MatMulOp& setTransposeB(bool transpose);

    void execute(const TTTensor& A, const TTTensor& B, TTTensor& C);
};

class CovarianceOp {
public:
    void operator()(const TTTensor& activity, TTTensor& covariance);
    void setNormalized(bool normalized);
};

class PCAOp {
public:
    PCAOp(uint32_t num_components);

    void operator()(const TTTensor& activity,
                    TTTensor& components,
                    std::vector<float>& explained_variance);
};

class RotationOp {
public:
    void operator()(TTTensor& activity,
                    const TTTensor& rotation_matrix,
                    bool preserve_variance = true);
};
```

## CUDA Backend

### V1Encoder

CUDA-based V1 visual encoding.

```cpp
namespace cuda_kernels {

class V1Encoder {
public:
    V1Encoder(uint32_t num_neurons,
              uint32_t stimulus_width,
              uint32_t stimulus_height);

    void encode(const float* h_stimulus,
                float* h_activity,
                cudaStream_t stream = 0);
};

}
```

### CUDA Kernel Functions

```cpp
// V1 encoding kernels
__global__ void v1_encode_kernel(
    const float* __restrict__ stimulus,
    float* __restrict__ activity,
    const V1Neuron* __restrict__ neurons,
    uint32_t stimulus_width,
    uint32_t stimulus_height,
    uint32_t num_neurons);

__global__ void v1_encode_kernel_shared(/* ... */);
__global__ void v1_encode_kernel_warp(/* ... */);

// Stimulus generation
__global__ void generateGratingKernel(
    float* stimulus,
    uint32_t width,
    uint32_t height,
    float orientation_rad,
    float spatial_freq,
    float contrast);
```

## Utilities

### Stimulus Generation

```cpp
Stimulus loadVisualGrating(
    float orientation,
    float spatial_freq = 0.05f,
    Dimensions dims = {64, 64, 1});
```

Generate a sinusoidal grating stimulus.

**Parameters:**
- `orientation`: Grating orientation in degrees
- `spatial_freq`: Spatial frequency (cycles/degree)
- `dims`: Stimulus dimensions

**Returns:**
- Generated stimulus

**Example:**
```cpp
Stimulus grating_45 = loadVisualGrating(45.0f, 0.05f, {64, 64});
Stimulus grating_135 = loadVisualGrating(135.0f, 0.05f, {64, 64});
```

### Hardware Detection

```cpp
std::vector<HardwareType> getAvailableHardware();
std::string getHardwareName(HardwareType hw_type);
```

### Logging

```cpp
void setLogLevel(int level);
```

Set global logging verbosity (0=off, 1=error, 2=warning, 3=info, 4=debug).

## Data Types

### Enumerations

```cpp
enum class HardwareType {
    GREYSKULL_E75,
    GREYSKULL_E150,
    RTX_4080,
    AUTO
};

enum class RoutingMode {
    STATIC,
    DYNAMIC,
    ADAPTIVE
};

enum class Bandwidth {
    LOW,
    NORMAL,
    HIGH,
    ADAPTIVE
};

enum class DataType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT32
};
```

### Structures

```cpp
struct Dimensions {
    uint32_t rows;
    uint32_t cols;
    uint32_t depth = 1;
};

struct Context {
    bool is_rewarded;
    bool is_go_trial;
    float attention_level;
    uint32_t trial_number;
    Timestamp timestamp;
};

struct Stimulus {
    std::vector<float> data;
    Dimensions dims;
    DataType dtype;
    std::string label;
};
```

## Complete Example

```cpp
#include "dynamic_cortex.hpp"

using namespace dynamic_cortex;

int main() {
    // Create framework
    auto framework = DynamicCortex::create(HardwareType::GREYSKULL_E75);

    // Create cortical areas
    auto v1 = framework->createCorticalArea("V1", {64, 64});
    auto lm = framework->createCorticalArea("LM", {32, 32});

    // Create communication streams
    auto feedforward = framework->createStream(v1, lm);
    feedforward->config()
        .setRoutingMode(RoutingMode::DYNAMIC)
        .setLatency(milliseconds(2));

    auto feedback = framework->createStream(lm, v1);
    feedback->config()
        .setRoutingMode(RoutingMode::DYNAMIC)
        .setLatency(milliseconds(3));

    // Configure behavioral modulation
    framework->setContextHandler([&](const Context& ctx) {
        if (ctx.is_rewarded) {
            feedback->config().setRotationPeriod(milliseconds(15));
        } else {
            feedback->config().setRotationPeriod(milliseconds(121));
        }
    });

    // Process stimulus
    Stimulus grating = loadVisualGrating(45.0f);
    framework->processStimulus(grating, milliseconds(1000));

    // Get results
    Tensor v1_response = v1->getActivity();

    return 0;
}
```

## See Also

- [Architecture Guide](architecture.md)
- [Neuroscience Background](neuroscience-background.md)
- [Examples](../src/tt-metal/examples/)
