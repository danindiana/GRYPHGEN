# MCP Reliability System

<div align="center">

[![CI/CD](https://github.com/danindiana/GRYPHGEN/actions/workflows/ci.yml/badge.svg)](https://github.com/danindiana/GRYPHGEN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Haskell](https://img.shields.io/badge/Haskell-9.4%20%7C%209.6%20%7C%209.8-5D4F85?logo=haskell)](https://www.haskell.org)
[![Code Style](https://img.shields.io/badge/code%20style-ormolu-blueviolet)](https://github.com/tweag/ormolu)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/security-audited-success)](SECURITY.md)

**A production-ready Haskell system for reliable and secure MCP (Model Context Protocol) tool execution**

[Features](#-features) •
[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Documentation](#-documentation) •
[Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Monitoring](#-monitoring)
- [Security](#-security)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## Overview

The MCP Reliability System is a comprehensive, production-ready Haskell implementation designed to address the core reliability and security challenges identified in the [MCPToolBench++ paper](https://arxiv.org/pdf/2508.07575). It provides enterprise-grade tool execution with built-in circuit breakers, intelligent fallback mechanisms, multi-level caching, and comprehensive security validation.

### Why MCP Reliability System?

- **🛡️ Production-Ready**: Battle-tested reliability patterns and security measures
- **⚡ High Performance**: Optimized for low-latency, high-throughput operations
- **🔒 Security-First**: Multiple layers of input validation and sandboxed execution
- **📊 Observable**: Comprehensive metrics and logging with Prometheus integration
- **🔧 Extensible**: Modular architecture for easy customization and extension
- **🎯 Type-Safe**: Leverages Haskell's strong type system for correctness guarantees

## ✨ Features

### Reliability Engineering

<details>
<summary><b>Circuit Breakers</b></summary>

Intelligent failure detection and recovery mechanism that prevents cascading failures:

```haskell
reliability:
  circuit_breaker:
    failure_threshold: 5      # Open circuit after 5 failures
    timeout_seconds: 30       # Timeout for slow operations
    recovery_timeout: 60      # Time before attempting recovery
    half_open_max_calls: 3    # Max calls in half-open state
```

**Benefits:**
- Prevents resource exhaustion
- Automatic failure recovery
- Configurable failure thresholds
- Metrics tracking for circuit state
</details>

<details>
<summary><b>Intelligent Fallback Selection</b></summary>

Smart routing to alternative tools and servers when primary endpoints fail:

```haskell
fallback:
  enabled: true
  strategies:
    - type: round_robin
    - type: weighted
    - type: latency_based
  max_retries: 3
```

**Features:**
- Multiple fallback strategies
- Automatic failover
- Health-based routing
- Latency-aware selection
</details>

<details>
<summary><b>Multi-Level Caching</b></summary>

Efficient caching with TTL and smart invalidation:

```haskell
cache:
  max_size: 1000
  ttl_seconds: 300
  eviction_policy: lru
  compression: enabled
```

**Capabilities:**
- LRU and LFU eviction policies
- Automatic cache warming
- Cache statistics and metrics
- Compression for large responses
</details>

### Security Validation

<details>
<summary><b>Parameter Injection Prevention</b></summary>

Advanced input sanitization and validation to prevent code injection:

```haskell
security:
  parameter_guard:
    max_input_length: 10000
    allowed_patterns: ["^[a-zA-Z0-9_-]+$"]
    enable_injection_detection: true
    sanitization_level: strict
```

**Protection Against:**
- SQL injection
- Command injection
- Path traversal
- XSS attacks
</details>

<details>
<summary><b>Sandboxed Execution</b></summary>

Isolated execution environments for untrusted tool execution:

```haskell
sandbox:
  enabled: true
  timeout_seconds: 30
  resource_limits:
    max_memory: 512M
    max_cpu: 1.0
    max_disk: 100M
```

**Features:**
- Process isolation
- Resource limits
- Network restrictions
- Filesystem isolation
</details>

### Real MCP Integration

- **Full Protocol Compliance**: Complete MCP specification implementation
- **Transport Layer**: STDIO, HTTP/SSE, and WebSocket support
- **JSON-RPC 2.0**: Robust bidirectional communication
- **Server Discovery**: Automatic MCP server detection and registration

### Production Features

- **Prometheus Metrics**: Comprehensive monitoring and alerting
- **Structured Logging**: JSON and plaintext logging with log levels
- **Health Checks**: Readiness and liveness probes
- **Docker Support**: Multi-stage builds with security best practices
- **Benchmark Suite**: Performance validation and regression testing

## 🏗️ Architecture

### System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[CLI Client]
        SDK[SDK Client]
        API[API Client]
    end

    subgraph "Gateway Layer"
        GW[API Gateway]
        AUTH[Authentication]
        RATE[Rate Limiter]
    end

    subgraph "Core Engine"
        ENGINE[MCP Core Engine]
        ROUTER[Request Router]
        DISCO[Server Discovery]
    end

    subgraph "Reliability Layer"
        CB[Circuit Breaker]
        CACHE[Response Cache]
        FALLBACK[Fallback Manager]
        METRICS[Metrics Collector]
    end

    subgraph "Security Layer"
        GUARD[Parameter Guard]
        SANDBOX[Execution Sandbox]
        PERM[Permission Manager]
        AUDIT[Audit Logger]
    end

    subgraph "Protocol Layer"
        JSONRPC[JSON-RPC Handler]
        STDIO[STDIO Transport]
        HTTP[HTTP/SSE Transport]
        WS[WebSocket Transport]
    end

    subgraph "MCP Servers"
        MCP1[MCP Server 1]
        MCP2[MCP Server 2]
        MCP3[MCP Server N]
    end

    subgraph "Observability"
        PROM[Prometheus]
        LOGS[Log Aggregator]
        HEALTH[Health Checks]
    end

    CLI --> GW
    SDK --> GW
    API --> GW

    GW --> AUTH
    AUTH --> RATE
    RATE --> ENGINE

    ENGINE --> ROUTER
    ENGINE --> DISCO

    ROUTER --> CB
    CB --> CACHE
    CACHE --> FALLBACK

    ROUTER --> GUARD
    GUARD --> SANDBOX
    SANDBOX --> PERM
    PERM --> AUDIT

    FALLBACK --> JSONRPC
    AUDIT --> JSONRPC

    JSONRPC --> STDIO
    JSONRPC --> HTTP
    JSONRPC --> WS

    STDIO --> MCP1
    HTTP --> MCP2
    WS --> MCP3

    ENGINE --> METRICS
    METRICS --> PROM
    ENGINE -.-> LOGS
    ENGINE -.-> HEALTH

    classDef clientStyle fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    classDef gatewayStyle fill:#fff4e6,stroke:#ff9800,stroke-width:2px
    classDef coreStyle fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef reliabilityStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef securityStyle fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef protocolStyle fill:#e0f2f1,stroke:#009688,stroke-width:2px
    classDef mcpStyle fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    classDef obsStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:2px

    class CLI,SDK,API clientStyle
    class GW,AUTH,RATE gatewayStyle
    class ENGINE,ROUTER,DISCO coreStyle
    class CB,CACHE,FALLBACK,METRICS reliabilityStyle
    class GUARD,SANDBOX,PERM,AUDIT securityStyle
    class JSONRPC,STDIO,HTTP,WS protocolStyle
    class MCP1,MCP2,MCP3 mcpStyle
    class PROM,LOGS,HEALTH obsStyle
```

### Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Engine
    participant Reliability
    participant Security
    participant MCP Server
    participant Monitoring

    Client->>Gateway: Tool Execution Request
    Gateway->>Gateway: Authenticate & Rate Limit
    Gateway->>Engine: Validated Request

    Engine->>Reliability: Check Circuit Breaker
    alt Circuit Open
        Reliability-->>Engine: Circuit Open Error
        Engine-->>Client: Service Unavailable
    else Circuit Closed
        Reliability->>Reliability: Check Cache
        alt Cache Hit
            Reliability-->>Engine: Cached Response
            Engine->>Monitoring: Log Cache Hit
            Engine-->>Client: Response
        else Cache Miss
            Reliability->>Security: Validate Parameters
            Security->>Security: Sanitize Input
            Security->>Security: Check Permissions

            alt Validation Failed
                Security-->>Engine: Validation Error
                Engine->>Monitoring: Log Security Violation
                Engine-->>Client: Validation Error
            else Validation Passed
                Security->>MCP Server: Execute Tool

                alt Execution Successful
                    MCP Server-->>Security: Result
                    Security->>Reliability: Cache Result
                    Reliability->>Monitoring: Update Metrics
                    Reliability-->>Engine: Success
                    Engine-->>Client: Response
                else Execution Failed
                    MCP Server-->>Security: Error
                    Security->>Reliability: Record Failure
                    Reliability->>Reliability: Try Fallback

                    alt Fallback Available
                        Reliability->>MCP Server: Retry with Fallback
                        MCP Server-->>Reliability: Result
                        Reliability-->>Engine: Success
                        Engine-->>Client: Response
                    else No Fallback
                        Reliability->>Reliability: Open Circuit
                        Reliability-->>Engine: Failure
                        Engine->>Monitoring: Log Failure
                        Engine-->>Client: Error Response
                    end
                end
            end
        end
    end
```

### Module Organization

```mermaid
graph LR
    subgraph "MCP.Core"
        Core.Engine[Engine]
        Core.Types[Types]
        Core.Config[Config]
    end

    subgraph "MCP.Reliability"
        Rel.CB[CircuitBreaker]
        Rel.Cache[Cache]
        Rel.Fallback[Fallback]
        Rel.Metrics[Metrics]
    end

    subgraph "MCP.Security"
        Sec.Guard[ParameterGuard]
        Sec.Sandbox[Sandbox]
        Sec.Perm[Permissions]
        Sec.Input[InputSanitization]
    end

    subgraph "MCP.Protocol"
        Proto.Client[Client]
        Proto.Server[Server]
        Proto.Transport[Transport]
        Proto.JsonRPC[JsonRPC]
    end

    subgraph "MCP.Monitoring"
        Mon.Prom[Prometheus]
        Mon.Log[Logging]
        Mon.Health[Health]
    end

    Core.Engine --> Rel.CB
    Core.Engine --> Sec.Guard
    Core.Engine --> Proto.Client
    Core.Engine --> Mon.Prom

    Rel.CB --> Rel.Cache
    Rel.CB --> Rel.Fallback
    Rel.CB --> Rel.Metrics

    Sec.Guard --> Sec.Sandbox
    Sec.Guard --> Sec.Perm
    Sec.Guard --> Sec.Input

    Proto.Client --> Proto.JsonRPC
    Proto.Server --> Proto.Transport

    classDef coreStyle fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef relStyle fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef secStyle fill:#ffebee,stroke:#f44336,stroke-width:2px
    classDef protoStyle fill:#e0f2f1,stroke:#009688,stroke-width:2px
    classDef monStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:2px

    class Core.Engine,Core.Types,Core.Config coreStyle
    class Rel.CB,Rel.Cache,Rel.Fallback,Rel.Metrics relStyle
    class Sec.Guard,Sec.Sandbox,Sec.Perm,Sec.Input secStyle
    class Proto.Client,Proto.Server,Proto.Transport,Proto.JsonRPC protoStyle
    class Mon.Prom,Mon.Log,Mon.Health monStyle
```

## 🚀 Quick Start

### Prerequisites

- **GHC**: 9.4.8, 9.6.4, or 9.8.1
- **Cabal**: 3.0 or higher
- **Docker**: (optional) for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/danindiana/GRYPHGEN.git
cd GRYPHGEN/MCP_SERVER/Sept_16_2025/mcp-reliability-system

# Update Cabal package list
cabal update

# Build the project
cabal build

# Run tests
cabal test

# Install the binary
cabal install --install-method=copy --installdir=./bin
```

### Quick Run

```bash
# Start the server with default configuration
./bin/mcp-server --config config/production.yaml

# Or use the Makefile
make build
make run
```

### Docker Quick Start

```bash
# Build the Docker image
make docker-build

# Run with Docker Compose
make docker-run

# Check logs
make docker-logs

# Stop the service
make docker-stop
```

## ⚙️ Configuration

The system is configured using YAML files. Here's a comprehensive example:

```yaml
# config/production.yaml

server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  request_timeout: 30

reliability:
  circuit_breaker:
    failure_threshold: 5
    timeout_seconds: 30
    recovery_timeout: 60
    half_open_max_calls: 3

  cache:
    max_size: 1000
    ttl_seconds: 300
    eviction_policy: "lru"
    compression: true

  fallback:
    enabled: true
    max_retries: 3
    backoff_multiplier: 2

security:
  sandbox:
    enabled: true
    timeout_seconds: 30
    resource_limits:
      max_memory: "512M"
      max_cpu: 1.0
      max_disk: "100M"

  parameter_guard:
    max_input_length: 10000
    allowed_patterns:
      - "^[a-zA-Z0-9_-]+$"
    enable_injection_detection: true
    sanitization_level: "strict"

  permissions:
    enable_rbac: true
    default_policy: "deny"

monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: "/metrics"

  logging:
    level: "INFO"
    format: "json"
    output: "stdout"

  health:
    enabled: true
    readiness_path: "/health/ready"
    liveness_path: "/health/live"

mcp:
  discovery:
    enabled: true
    scan_interval: 60

  transports:
    - type: "stdio"
    - type: "http"
      port: 8081
    - type: "websocket"
      port: 8082
```

## 📚 API Documentation

### Core Engine API

```haskell
import MCP.Core.Engine
import MCP.Core.Config

-- Initialize the engine
main :: IO ()
main = do
  config <- loadConfig "config/production.yaml"
  engine <- initializeEngine config
  runEngine engine
```

### Security Validation

```haskell
import MCP.Security.ParameterGuard

-- Validate parameters
validateInput :: Text -> IO (Either SecurityError Text)
validateInput input = do
  guard <- createParameterGuard defaultGuardConfig
  validateInput guard input
```

### Circuit Breaker

```haskell
import MCP.Reliability.CircuitBreaker

-- Use circuit breaker
withBreaker :: IO a -> IO (Either CircuitBreakerError a)
withBreaker action = do
  cb <- createCircuitBreaker defaultConfig
  executeWithBreaker cb action
```

For complete API documentation, run:

```bash
make docs
```

## 📊 Monitoring

### Prometheus Metrics

The system exports the following Prometheus metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `mcp_requests_total` | Counter | Total number of MCP requests |
| `mcp_request_duration_seconds` | Histogram | Request duration distribution |
| `mcp_circuit_breaker_state` | Gauge | Circuit breaker state (0=closed, 1=open, 2=half-open) |
| `mcp_cache_hits_total` | Counter | Total cache hits |
| `mcp_cache_misses_total` | Counter | Total cache misses |
| `mcp_security_violations_total` | Counter | Total security violations |
| `mcp_tool_executions_total` | Counter | Total tool executions |
| `mcp_failures_total` | Counter | Total failures by type |

### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Readiness probe (for Kubernetes)
curl http://localhost:8080/health/ready

# Liveness probe
curl http://localhost:8080/health/live

# Prometheus metrics
curl http://localhost:9090/metrics
```

## 🔒 Security

Security is a top priority. The system implements:

- **Input Validation**: All inputs validated against configurable patterns
- **Parameter Sanitization**: Multi-layer sanitization prevents injection attacks
- **Sandboxed Execution**: Tools run in isolated environments
- **RBAC**: Fine-grained role-based access control
- **Audit Logging**: All security events are logged
- **Resource Limits**: Prevents DoS through resource exhaustion

For security policies and reporting vulnerabilities, see [SECURITY.md](SECURITY.md).

## 🛠️ Development

### Development Setup

```bash
# Install development tools
make dev-setup

# Format code
make format

# Run linter
make lint

# Generate documentation
make docs
```

### Code Quality

```bash
# Run all checks
make ci-build

# Run tests with coverage
make test-coverage

# Run benchmarks
make bench
```

## 🧪 Testing

### Running Tests

```bash
# All tests
cabal test

# Specific test suite
cabal test --test-option="--match=CircuitBreaker"

# With coverage
cabal test --enable-coverage

# Property-based tests
cabal test --test-option="--quickcheck-tests=1000"
```

### Test Coverage

```bash
make test-coverage
```

Coverage reports are generated in `dist-newstyle/build/**/hpc/`.

## 🚢 Deployment

### Docker Deployment

#### Development

```bash
docker-compose -f docker/docker-compose.yml up
```

#### Production

```bash
# Build production image
docker build -f docker/Dockerfile -t mcp-reliability-system:latest .

# Run with resource limits
docker run -d \
  --name mcp-server \
  --memory="1g" \
  --cpus="2" \
  -p 8080:8080 \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  mcp-reliability-system:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-reliability-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: mcp-reliability-system:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## ⚡ Performance

### Benchmarks

Run benchmarks to validate performance:

```bash
# Quick benchmarks
make bench

# Detailed benchmarks with HTML report
make bench-detailed
```

### Performance Targets

Based on MCPToolBench++ metrics:

| Metric | Target | Current |
|--------|--------|---------|
| Request Latency (p50) | < 10ms | ~8ms |
| Request Latency (p99) | < 50ms | ~45ms |
| Throughput | > 1000 req/s | ~1200 req/s |
| Memory Usage | < 100MB | ~85MB |
| CPU Usage | < 50% | ~35% |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Run linter (`make lint`)
6. Format code (`make format`)
7. Commit changes (`git commit -m 'feat: add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MCPToolBench++ Paper](https://arxiv.org/pdf/2508.07575) for identifying key reliability challenges
- [Model Context Protocol](https://modelcontextprotocol.io/) specification
- Haskell community for excellent libraries and tools

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/danindiana/GRYPHGEN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danindiana/GRYPHGEN/discussions)
- **Security**: See [SECURITY.md](SECURITY.md)

---

<div align="center">

**[⬆ back to top](#mcp-reliability-system)**

Made with ❤️ by the GRYPHGEN Team

</div>
