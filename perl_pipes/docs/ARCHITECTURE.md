# GRYPHGEN Perl Pipes Architecture

## Overview

GRYPHGEN Perl Pipes is an Inter-Process Communication (IPC) system designed for orchestrating multi-model Large Language Model (LLM) sessions. The architecture leverages Unix named pipes (FIFOs) for efficient, low-latency communication between model instances.

## Design Philosophy

### Core Principles

1. **Simplicity** - Use proven Unix primitives (named pipes)
2. **Efficiency** - Minimize overhead and latency
3. **Reliability** - Timeout protection and error handling
4. **Scalability** - Support 2+ concurrent models
5. **Observability** - Comprehensive logging and monitoring

### Why Named Pipes?

- **Zero-copy** - Data transferred via kernel buffers
- **Blocking I/O** - Natural synchronization
- **File system** - Easy to create, inspect, and clean up
- **Cross-process** - Language-agnostic communication
- **Local-only** - Security through isolation

## System Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Model 1    │  │   Model 2    │  │   Model N    │  │
│  │  LLM Instance│  │  LLM Instance│  │  LLM Instance│  │
│  └──────┬───────┘  └───────┬──────┘  └───────┬──────┘  │
└─────────┼────────────────────┼─────────────────┼─────────┘
          │                    │                 │
┌─────────┼────────────────────┼─────────────────┼─────────┐
│         │   IPC Layer        │                 │         │
│  ┌──────▼──────┐      ┌──────▼──────┐   ┌──────▼──────┐ │
│  │model1_comm  │      │model2_comm  │   │modelN_comm  │ │
│  │   .pl       │      │   .pl       │   │   .pl       │ │
│  └──────┬──────┘      └──────┬──────┘   └──────┬──────┘ │
└─────────┼────────────────────┼─────────────────┼─────────┘
          │                    │                 │
┌─────────┼────────────────────┼─────────────────┼─────────┐
│         │  Pipe Layer        │                 │         │
│  ┌──────▼──────────────┬─────▼─────────────┬───▼──────┐ │
│  │  Named Pipe 1→2     │  Named Pipe 2→1   │  Pipe N  │ │
│  │  /tmp/model1_to_2   │  /tmp/model2_to_1 │   ...    │ │
│  └─────────────────────┴───────────────────┴──────────┘ │
└─────────────────────────────────────────────────────────┘
          │                    │                 │
┌─────────┼────────────────────┼─────────────────┼─────────┐
│         │   System Layer     │                 │         │
│  ┌──────▼────────────────────▼─────────────────▼──────┐ │
│  │              Linux Kernel (IPC)                     │ │
│  │         Named Pipe Buffers & Scheduling             │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
perl_pipes/
├── bin/                    # Executable scripts
│   ├── model1_comm.pl     # Model 1 communication
│   ├── model2_comm.pl     # Model 2 communication
│   ├── find_gguf_files.pl # Model file scanner
│   ├── monitor_gpu.pl     # GPU monitoring
│   ├── setup_pipes.sh     # Pipe creation
│   └── cleanup_pipes.sh   # Pipe cleanup
│
├── lib/                    # Perl modules (extensible)
│   └── (future modules)
│
├── examples/              # Usage examples
│   ├── basic_communication.sh
│   ├── multi_round_chat.sh
│   └── find_gpu_models.sh
│
├── config/                # Configuration files
│   └── rtx4080_optimized.conf
│
├── tests/                 # Test suite
│   ├── 01-basic.t
│   ├── 02-pipes.t
│   └── 03-find-gguf.t
│
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # This file
│   └── old_readme.md      # Historical reference
│
├── logs/                  # Runtime logs
│   ├── model1.log
│   └── model2.log
│
├── .github/
│   └── workflows/         # CI/CD
│       └── test.yml
│
├── README.md              # Main documentation
├── CONTRIBUTING.md        # Contribution guide
├── LICENSE                # MIT License
├── Makefile               # Build automation
├── cpanfile               # Perl dependencies
├── Dockerfile             # Container definition
└── docker-compose.yml     # Multi-container setup
```

## Communication Protocol

### Message Flow

```
┌─────────┐                                         ┌─────────┐
│ Model 1 │                                         │ Model 2 │
└────┬────┘                                         └────┬────┘
     │                                                    │
     │ 1. Open pipe2→1 for reading (blocking)            │
     │◄───────────────────────────────────────────────────┤
     │                                                    │
     │ 2. Open pipe1→2 for writing                       │
     ├───────────────────────────────────────────────────►│
     │                                                    │
     │ 3. Write message to pipe1→2                        │
     ├───────────────────────────────────────────────────►│
     │                                                    │
     │                  4. Read from pipe1→2              │
     │                  (unblocks Model 2)                │
     │                                                    │
     │                  5. Process message                │
     │                                                    │
     │                  6. Write response to pipe2→1      │
     │◄───────────────────────────────────────────────────┤
     │                                                    │
     │ 7. Read response from pipe2→1                      │
     │◄───────────────────────────────────────────────────┤
     │                                                    │
     │ 8. Process response                                │
     │                                                    │
     │ 9. Close pipes                                     │
     ├─────────────────────────┬──────────────────────────┤
     │                         │                          │
```

### Synchronization

- **Model 2 starts first** - Opens input pipe in read mode (blocks)
- **Model 1 starts second** - Writes message (unblocks Model 2)
- **Automatic flow control** - Kernel handles buffering
- **Timeout protection** - ALRM signal prevents indefinite blocking

## Data Flow Patterns

### 1. Request-Response (Implemented)

```perl
# Model 1 (sender)
open($out, '>', $pipe_to_model2);
print $out $request;
close($out);

open($in, '<', $pipe_from_model2);
$response = <$in>;
close($in);

# Model 2 (receiver)
open($in, '<', $pipe_from_model1);
$request = <$in>;
close($in);

# Process and respond
open($out, '>', $pipe_to_model1);
print $out $response;
close($out);
```

### 2. Streaming (Future)

```perl
# Continuous data flow without closing pipes
open($out, '>', $pipe);
$out->autoflush(1);

while (my $chunk = get_next_chunk()) {
    print $out $chunk;
}
```

### 3. Broadcast (Future)

```perl
# One-to-many communication
for my $model (@models) {
    my $pipe = "/tmp/broadcast_to_$model";
    open(my $out, '>', $pipe);
    print $out $message;
    close($out);
}
```

## GPU Memory Management

### Architecture Integration

The system is optimized for NVIDIA RTX 4080 (16GB VRAM):

```
┌────────────────────────────────────────────────┐
│           RTX 4080 VRAM (16GB)                 │
├────────────────────────────────────────────────┤
│  System Reserved (~1.6GB)                      │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  Model Instance(s)                       │ │
│  │  ┌────────────┐    ┌────────────┐       │ │
│  │  │  Model 1   │    │  Model 2   │       │ │
│  │  │  (~7GB)    │    │  (~7GB)    │       │ │
│  │  └────────────┘    └────────────┘       │ │
│  │                                          │ │
│  │  Context Windows (~500MB each)          │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  Free VRAM Buffer (~1GB)                       │
└────────────────────────────────────────────────┘
```

### Model Selection Strategy

```perl
# Pseudo-code for model selection
sub select_models_for_gpu($available_vram) {
    my @models = find_gguf_files();
    my @compatible;

    my $usable_vram = $available_vram * 0.90;  # 90% usable
    my $context_overhead = 500 * 1024 * 1024;  # 500MB per model

    for my $model (@models) {
        my $total_required = $model->{size} + $context_overhead;

        if ($total_required <= $usable_vram) {
            push @compatible, $model;
        }
    }

    return \@compatible;
}
```

## Error Handling Strategy

### Timeout Protection

```perl
eval {
    local $SIG{ALRM} = sub { die "Timeout\n" };
    alarm $timeout;

    # Risky operation
    open(my $fh, '>', $pipe);
    print $fh $data;
    close($fh);

    alarm 0;
};

if ($@ =~ /Timeout/) {
    log_error("Operation timed out");
    # Handle timeout
}
```

### Pipe Existence Validation

```perl
unless (-p $pipe_path) {
    die "Pipe not found: $pipe_path\n" .
        "Run setup_pipes.sh first\n";
}
```

### Graceful Degradation

```perl
# Attempt operation with fallback
eval {
    primary_method();
};

if ($@) {
    warn "Primary method failed: $@";
    fallback_method();
}
```

## Logging Architecture

### Log Levels

- **DEBUG** - Detailed diagnostic information
- **INFO** - General informational messages
- **WARN** - Warning messages (non-fatal)
- **ERROR** - Error messages (recoverable)
- **FATAL** - Fatal errors (unrecoverable)

### Log Format

```
[TIMESTAMP] [LEVEL] MESSAGE
[2024-11-19 12:34:56] [INFO] Starting Model 1 Communication
[2024-11-19 12:34:56] [INFO] Opening pipe: /tmp/model1_to_model2
[2024-11-19 12:34:57] [ERROR] Timeout reading from pipe
```

### Log Rotation

```bash
# Using logrotate (future implementation)
/opt/gryphgen/perl_pipes/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

## Performance Characteristics

### Latency Profile

| Operation | Typical Latency | Max Latency |
|-----------|----------------|-------------|
| Pipe creation | < 1ms | 5ms |
| Open pipe (non-blocking) | < 1ms | 2ms |
| Open pipe (blocking) | Variable | Timeout |
| Write to pipe | < 1ms | 5ms |
| Read from pipe | < 1ms | 5ms |
| Close pipe | < 1ms | 2ms |
| Round-trip message | 2-5ms | 100ms |

### Throughput

- **Sequential**: ~200-500 messages/second
- **Parallel** (multiple pipe pairs): 1000+ messages/second
- **Bottleneck**: Perl script startup overhead

### Optimization Strategies

1. **Persistent processes** - Keep scripts running
2. **RAM disk** - Use /dev/shm for pipes
3. **Batch operations** - Process multiple messages
4. **Parallel pipes** - Multiple concurrent pairs

## Security Considerations

### Threat Model

- **Local-only** - Named pipes are local to system
- **File permissions** - Standard Unix permissions
- **No encryption** - Data in kernel buffers (plaintext)
- **Process isolation** - Each model in separate process

### Security Measures

1. **Pipe permissions**: 0666 (rw-rw-rw-)
2. **Temporary location**: /tmp (cleared on reboot)
3. **No network exposure**: Pipes are local-only
4. **User isolation**: Run as non-root user

### Future Enhancements

- [ ] Encrypted pipes (GPG wrapper)
- [ ] Access control lists (ACLs)
- [ ] Audit logging
- [ ] SELinux/AppArmor profiles

## Extensibility

### Adding New Communication Scripts

```perl
#!/usr/bin/env perl
use v5.36;
use FindBin qw($RealBin);
use lib "$RealBin/../lib";

# Use common library (future)
use GRYPHGEN::IPC;

my $ipc = GRYPHGEN::IPC->new(
    pipe_to   => '/tmp/custom_pipe',
    pipe_from => '/tmp/custom_response',
    timeout   => 30,
);

$ipc->send_message("Hello");
my $response = $ipc->receive_message();
```

### Plugin Architecture (Future)

```perl
# lib/GRYPHGEN/Plugin/Base.pm
package GRYPHGEN::Plugin::Base;

sub new { ... }
sub on_message { ... }
sub on_error { ... }
```

## Testing Strategy

### Test Pyramid

```
       ┌───────────────┐
       │  Integration  │  ← examples/*.sh
       │     Tests     │
       └───────────────┘
      ┌─────────────────┐
      │   Functional    │  ← tests/02-pipes.t
      │     Tests       │
      └─────────────────┘
    ┌───────────────────────┐
    │    Unit Tests         │  ← tests/01-basic.t
    │                       │     tests/03-find-gguf.t
    └───────────────────────┘
```

### Continuous Integration

GitHub Actions workflow:
1. Syntax validation
2. Unit tests
3. Functional tests
4. Docker build
5. Code quality checks

## Monitoring and Observability

### Metrics Collected

- **Latency**: Message round-trip time
- **Throughput**: Messages per second
- **Errors**: Timeout, permission, not found
- **GPU**: VRAM usage, temperature, utilization

### Monitoring Tools

```bash
# Real-time GPU monitoring
./bin/monitor_gpu.pl --continuous --interval=2

# Log analysis
tail -f logs/*.log | grep ERROR

# Pipe status
ls -la /tmp/model*_to_*
```

## Future Architecture Enhancements

### Planned Improvements

1. **WebSocket Gateway**
   - Remote model communication
   - Web-based dashboard

2. **gRPC Interface**
   - Binary protocol
   - Bi-directional streaming

3. **Load Balancer**
   - Distribute requests across models
   - Health checking

4. **Message Queue**
   - Asynchronous processing
   - Message persistence

5. **Distributed Pipes**
   - Cross-machine communication
   - Network transparency

### Architectural Evolution

```
Current: Named Pipes (Local IPC)
    ↓
Phase 2: WebSocket (Remote IPC)
    ↓
Phase 3: gRPC (High-performance)
    ↓
Phase 4: Message Queue (Async)
    ↓
Phase 5: Kubernetes Operators
```

## References

### Unix Named Pipes

- `man 7 fifo` - Linux FIFO documentation
- Stevens, W. Richard. "Unix Network Programming"

### Perl IPC

- `perlipc` - Perl IPC documentation
- Christiansen, Tom. "Perl Cookbook"

### GPU Computing

- NVIDIA CUDA Documentation
- GGUF Format Specification

---

**Document Version**: 2.0.0
**Last Updated**: 2024-11-19
**Authors**: GRYPHGEN Team
