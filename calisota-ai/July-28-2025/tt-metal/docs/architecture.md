# System Architecture

## Overview

The Dynamic Cortex Communication Framework implements neuroscience-inspired algorithms on hardware accelerators using a modular, backend-agnostic architecture.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        A[User Application]
        B[Python Bindings]
    end

    subgraph "Framework Core"
        C[Dynamic Cortex API]
        D[Hardware Abstraction Layer]
        E[Stream Manager]
        F[Event Scheduler]
        G[Tensor Operations]
    end

    subgraph "Backend Implementations"
        H[TT-Metal Backend]
        I[CUDA Backend]
    end

    subgraph "Hardware Layer"
        J[Greyskull e75/e150]
        K[RTX 4080]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    D --> F
    D --> G
    E --> H
    E --> I
    F --> H
    F --> I
    G --> H
    G --> I
    H --> J
    I --> K

    style C fill:#4ECDC4,stroke:#333,stroke-width:3px
    style H fill:#FF6B6B,stroke:#333,stroke-width:2px
    style I fill:#95E1D3,stroke:#333,stroke-width:2px
    style J fill:#F38181,stroke:#333,stroke-width:2px
    style K fill:#EAFFD0,stroke:#333,stroke-width:2px
```

## Core Components

### 1. Cortical Area Abstraction

Represents a group of processing elements (neurons) with local state and connectivity.

```mermaid
classDiagram
    class CorticalArea {
        +string name
        +Dimensions dims
        +Tensor activity
        +vector~Connection~ inputs
        +vector~Connection~ outputs
        +activate(Stimulus)
        +getActivity()
        +reset()
    }

    class Connection {
        +CorticalArea source
        +CorticalArea target
        +StreamConfig stream
        +Weight weights
    }

    class StreamConfig {
        +RoutingMode mode
        +Bandwidth bandwidth
        +Latency latency
        +updateRouting()
    }

    CorticalArea "1" --> "*" Connection
    Connection --> StreamConfig
```

### 2. Dynamic Communication Streams

Implements reconfigurable data pathways with temporal dynamics.

```mermaid
sequenceDiagram
    participant V1 as V1 Area
    participant Stream as Stream Manager
    participant LM as LM Area
    participant Event as Event System

    V1->>Stream: Send activity (t=0ms)
    Stream->>Stream: Route via NOC/PCIe
    Stream->>LM: Deliver (t=2ms)
    LM->>LM: Process + compute feedback
    LM->>Stream: Send feedback (t=5ms)
    Stream->>V1: Deliver feedback (t=8ms)

    Event->>Stream: Trigger rotation (t=15ms)
    Stream->>Stream: Update routing config
    Stream-->>V1: New channel active
    Stream-->>LM: New channel active
```

### 3. Event-Driven Execution

Millisecond-scale temporal control using hardware events.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: Stimulus Event
    Processing --> Feedforward: V1 Ready
    Feedforward --> Feedback: LM Computed
    Feedback --> ChannelRotation: Timer Event (15ms)
    ChannelRotation --> Processing
    Processing --> Idle: Trial Complete
    Idle --> [*]

    note right of ChannelRotation
        Rewarded: 15ms rotation
        Non-rewarded: 121ms rotation
    end note
```

## TT-Metal Backend Architecture

### Core Mapping

```mermaid
graph LR
    subgraph "Greyskull Die"
        subgraph "V1 Area (32 cores)"
            V1_0[Core 0,0]
            V1_1[Core 0,1]
            V1_N[Core 7,7]
        end

        subgraph "LM Area (16 cores)"
            LM_0[Core 8,0]
            LM_1[Core 8,1]
            LM_N[Core 11,7]
        end

        NOC[Network-on-Chip]

        V1_0 <--> NOC
        V1_1 <--> NOC
        V1_N <--> NOC
        LM_0 <--> NOC
        LM_1 <--> NOC
        LM_N <--> NOC
    end

    DRAM[GDDR6 Memory]
    HOST[Host PCIe]

    NOC <--> DRAM
    NOC <--> HOST

    style NOC fill:#FFD93D,stroke:#333,stroke-width:3px
    style DRAM fill:#6BCB77,stroke:#333,stroke-width:2px
```

### Kernel Execution Model

```mermaid
sequenceDiagram
    participant Host
    participant Device
    participant Core0 as Tensix Core 0
    participant Core1 as Tensix Core 1
    participant NOC

    Host->>Device: Upload kernels + data
    Device->>Core0: Dispatch V1 encode
    Device->>Core1: Dispatch LM modulate

    par Parallel Execution
        Core0->>Core0: Execute V1 kernel
        Core1->>Core1: Execute LM kernel
    end

    Core0->>NOC: Stream output
    NOC->>Core1: Deliver to LM
    Core1->>NOC: Stream feedback
    NOC->>Core0: Deliver to V1

    Core0->>Device: Signal complete
    Device->>Host: Transfer results
```

### Memory Hierarchy

```mermaid
graph TD
    A[Host DRAM]
    B[PCIe Gen4 x16]
    C[Device GDDR6<br/>8/16 GB]
    D[L1 Cache<br/>1MB per core]
    E[Tensix Core SRAM<br/>1.2MB]
    F[Register File<br/>256 entries]

    A <-->|12-16 GB/s| B
    B <-->|64 GB/s| C
    C <-->|~1 TB/s| D
    D <-->|~4 TB/s| E
    E <-->|~16 TB/s| F

    style C fill:#4ECDC4,stroke:#333,stroke-width:2px
    style E fill:#FF6B6B,stroke:#333,stroke-width:2px
    style F fill:#FFA07A,stroke:#333,stroke-width:2px
```

## CUDA Backend Architecture

### Block/Thread Mapping

```mermaid
graph TB
    subgraph "RTX 4080 (76 SMs)"
        subgraph "V1 Processing"
            Grid1[Grid: 64 blocks]
            Block1[Block 0: 256 threads]
            Block2[Block 1: 256 threads]
        end

        subgraph "LM Processing"
            Grid2[Grid: 32 blocks]
            Block3[Block 0: 256 threads]
            Block4[Block 1: 256 threads]
        end

        subgraph "Shared Resources"
            L2[L2 Cache<br/>64 MB]
            VRAM[GDDR6X<br/>16 GB]
        end

        Block1 --> L2
        Block2 --> L2
        Block3 --> L2
        Block4 --> L2
        L2 <--> VRAM
    end

    style Grid1 fill:#95E1D3,stroke:#333,stroke-width:2px
    style Grid2 fill:#EAFFD0,stroke:#333,stroke-width:2px
    style L2 fill:#FFD93D,stroke:#333,stroke-width:2px
```

### Warp Execution Timeline

```mermaid
gantt
    title CUDA Kernel Execution (RTX 4080)
    dateFormat X
    axisFormat %L ms

    section V1 Encode
    Warp 0-7 :active, 0, 5
    Warp 8-15 :active, 0, 5

    section Feedforward
    Transfer V1→LM :crit, 5, 6

    section LM Modulate
    Warp 0-7 :active, 6, 10
    Tensor Core ops :active, 7, 9

    section Feedback
    Transfer LM→V1 :crit, 10, 11

    section V1 Update
    Warp 0-7 :active, 11, 14
```

## Data Flow

### Stimulus Processing Pipeline

```mermaid
graph LR
    A[Input Stimulus<br/>45° Grating] --> B[Preprocessing<br/>Normalization]
    B --> C[V1 Encoding<br/>Gabor Filters]
    C --> D[Feedforward Stream<br/>2ms latency]
    D --> E[LM Integration<br/>Context Modulation]
    E --> F[Feedback Stream<br/>3ms latency]
    F --> G[V1 Restructuring<br/>PC Rotation]
    G --> H[Output Activity<br/>Population Response]

    style A fill:#E3F2FD,stroke:#333,stroke-width:2px
    style C fill:#FFE0B2,stroke:#333,stroke-width:2px
    style E fill:#F8BBD0,stroke:#333,stroke-width:2px
    style G fill:#C8E6C9,stroke:#333,stroke-width:2px
    style H fill:#B2EBF2,stroke:#333,stroke-width:2px
```

### Behavioral Context Modulation

```mermaid
graph TD
    A{Stimulus Type}
    A -->|Rewarded| B[High Bandwidth Mode]
    A -->|Non-rewarded| C[Normal Bandwidth Mode]

    B --> D[Fast Rotation<br/>15ms period]
    C --> E[Slow Rotation<br/>121ms period]

    D --> F[Strong Feedback<br/>Enhanced Coupling]
    E --> G[Weak Feedback<br/>Baseline Coupling]

    F --> H[Rapid PC Rotation<br/>Dynamic Subnetworks]
    G --> I[Slow PC Evolution<br/>Stable Subnetworks]

    style B fill:#4CAF50,stroke:#333,stroke-width:2px
    style C fill:#9E9E9E,stroke:#333,stroke-width:2px
    style D fill:#8BC34A,stroke:#333,stroke-width:2px
    style E fill:#BDBDBD,stroke:#333,stroke-width:2px
```

## Communication Channel Dynamics

### Channel Rotation Mechanism

```mermaid
sequenceDiagram
    autonumber
    participant T as Timer (15ms)
    participant SM as Stream Manager
    participant V1 as V1 Subpop A
    participant V1B as V1 Subpop B
    participant LM as LM Cores

    Note over T,LM: Initial Configuration
    SM->>V1: Activate channel A
    V1->>LM: Feedforward (A→LM)
    LM->>V1: Feedback (LM→A)

    T->>SM: Rotation event
    Note over SM: Compute new routing
    SM->>V1: Deactivate channel A
    SM->>V1B: Activate channel B

    V1B->>LM: Feedforward (B→LM)
    LM->>V1B: Feedback (LM→B)

    T->>SM: Next rotation (t+15ms)
```

### Sparse Activation Pattern

```mermaid
graph TB
    subgraph "Time Window t=0-15ms"
        V1A[V1 Neurons 0-31<br/>ACTIVE]
        V1B[V1 Neurons 32-63<br/>IDLE]
        LMA[LM Neurons 0-15<br/>ACTIVE]
        LMB[LM Neurons 16-31<br/>IDLE]
    end

    subgraph "Time Window t=15-30ms"
        V1C[V1 Neurons 0-31<br/>IDLE]
        V1D[V1 Neurons 32-63<br/>ACTIVE]
        LMC[LM Neurons 0-15<br/>IDLE]
        LMD[LM Neurons 16-31<br/>ACTIVE]
    end

    V1A -.->|Rotation| V1D
    V1B -.->|Rotation| V1C
    LMA -.->|Rotation| LMD
    LMB -.->|Rotation| LMC

    style V1A fill:#4CAF50,stroke:#333,stroke-width:2px
    style V1B fill:#BDBDBD,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style V1D fill:#4CAF50,stroke:#333,stroke-width:2px
    style V1C fill:#BDBDBD,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
```

## Tensor Operations

### Population Geometry Transformation

```mermaid
graph LR
    A[V1 Activity<br/>N x T matrix] --> B[Compute Covariance<br/>C = A·Aᵀ]
    B --> C[PCA Decomposition<br/>C = U·Σ·Uᵀ]
    C --> D[Feedback Rotation<br/>R = LM_feedback]
    D --> E[Apply Transformation<br/>A' = R·A]
    E --> F[Normalize Variance<br/>var(A') = var(A)]
    F --> G[Updated V1 Activity<br/>Rotated PCs]

    style B fill:#E1F5FE,stroke:#333,stroke-width:2px
    style C fill:#F3E5F5,stroke:#333,stroke-width:2px
    style D fill:#FFF3E0,stroke:#333,stroke-width:2px
    style E fill:#E8F5E9,stroke:#333,stroke-width:2px
```

### Matrix Operations Mapping

**TT-Metal:**
```cpp
// Leverages Tensix matrix engine
TensorOp matmul = TensorOp::MATMUL
    .set_input_layout(TileLayout::ROW_MAJOR)
    .set_output_layout(TileLayout::ROW_MAJOR)
    .set_math_fidelity(MathFidelity::HiFi4);

Tensor result = matmul(rotation_matrix, v1_activity);
```

**CUDA:**
```cpp
// Uses Tensor Cores for mixed-precision
cublasLtMatmul(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    rotation_matrix, CUDA_R_16F,  // FP16 input
    v1_activity, CUDA_R_16F,
    &beta,
    result, CUDA_R_32F);  // FP32 accumulation
```

## Performance Optimization

### Greyskull Pipeline

```mermaid
gantt
    title TT-Metal Pipelined Execution (1ms trial)
    dateFormat X
    axisFormat %L µs

    section Stage 1
    V1 Compute :active, 0, 200

    section Stage 2
    FF Transfer :crit, 200, 250
    LM Compute :active, 250, 500

    section Stage 3
    FB Transfer :crit, 500, 550
    V1 Update :active, 550, 800

    section Stage 4
    Rotation :800, 850
    Next Cycle :850, 1000
```

### RTX 4080 Optimization

```mermaid
graph TB
    A[Input Data] --> B{Data Size}
    B -->|< 1MB| C[Use Shared Memory]
    B -->|1-100MB| D[Use L2 Cache]
    B -->|> 100MB| E[Stream from VRAM]

    C --> F[Warp-Level Ops]
    D --> G[Block-Level Reduction]
    E --> H[Grid-Level Batching]

    F --> I{Precision}
    G --> I
    H --> I

    I -->|High| J[FP32 Compute]
    I -->|Mixed| K[Tensor Core FP16→FP32]
    I -->|Low| L[INT8 Quantized]

    J --> M[Output]
    K --> M
    L --> M

    style C fill:#81C784,stroke:#333,stroke-width:2px
    style D fill:#FFB74D,stroke:#333,stroke-width:2px
    style E fill:#E57373,stroke:#333,stroke-width:2px
```

## Scalability

### Multi-Device Configuration

```mermaid
graph TB
    subgraph "Host System"
        H[Host CPU<br/>PCIe Switch]
    end

    subgraph "Device 0"
        G1[Greyskull e150<br/>V1 Processing]
    end

    subgraph "Device 1"
        G2[Greyskull e150<br/>LM Processing]
    end

    subgraph "Device 2"
        R1[RTX 4080<br/>Tensor Ops]
    end

    H <--> G1
    H <--> G2
    H <--> R1
    G1 <-.->|High-speed| G2
    G2 <-.->|Offload| R1

    style H fill:#42A5F5,stroke:#333,stroke-width:3px
    style G1 fill:#FF6B6B,stroke:#333,stroke-width:2px
    style G2 fill:#FF6B6B,stroke:#333,stroke-width:2px
    style R1 fill:#66BB6A,stroke:#333,stroke-width:2px
```

## Error Handling & Debugging

### Event Tracing

```mermaid
sequenceDiagram
    participant App
    participant Debug
    participant V1
    participant LM

    App->>Debug: Enable tracing
    App->>V1: Process stimulus

    V1->>Debug: Log: V1_ACTIVATE (t=0)
    V1->>LM: Send data

    LM->>Debug: Log: LM_RECEIVE (t=2ms)
    LM->>Debug: Log: LM_COMPUTE (t=2-5ms)
    LM->>V1: Send feedback

    V1->>Debug: Log: V1_FEEDBACK (t=8ms)

    Debug->>App: Trace report
```

## Summary

The architecture provides:

1. **Hardware Abstraction**: Unified API across TT-Metal and CUDA backends
2. **Dynamic Reconfiguration**: Event-driven routing updates at millisecond scales
3. **Behavioral Modulation**: Context-dependent execution paths
4. **Efficient Data Movement**: Pipelined streams and sparse activation
5. **Scalable Design**: Multi-device and multi-backend support

See [API Reference](api-reference.md) for detailed programming interfaces.
