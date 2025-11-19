# ShellGenie Architecture

## Overview

ShellGenie is built with a modular architecture that separates concerns and allows for easy extension and maintenance.

## Component Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI Layer                         │
│  - Click-based CLI                                  │
│  - Rich terminal UI                                 │
│  - Command parsing                                  │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                 Core Layer                          │
│  - ShellGenieCore                                   │
│  - Request processing                               │
│  - Command generation                               │
│  - Execution management                             │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴─────────┬─────────────┐
        │                   │             │
┌───────▼─────┐  ┌─────────▼──────┐  ┌──▼──────────┐
│   Models    │  │   Security     │  │   Utils     │
│  - Pydantic │  │  - Validator   │  │  - Helpers  │
│  - Types    │  │  - Sandbox     │  │  - GPU      │
└─────────────┘  └────────────────┘  └─────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼─────────┐          ┌───────────▼──────────┐
│  LLM Backends   │          │   Execution Engine   │
│  - Ollama       │          │   - Bash executor    │
│  - llama.cpp    │          │   - Output capture   │
│  - OpenAI API   │          │   - Timeout handling │
└─────────────────┘          └──────────────────────┘
```

## Core Components

### 1. CLI Layer (`cli.py`)

The command-line interface built with Click and Rich.

**Responsibilities:**
- Parse command-line arguments
- Display beautiful terminal UIs
- Handle user interactions
- Route commands to core

**Key Classes:**
- `cli()`: Main CLI group
- `run()`: Execute single command
- `interactive()`: Interactive mode
- `info()`: System information
- `history()`: Command history

### 2. Core Layer (`core.py`)

The heart of ShellGenie that orchestrates all operations.

**Responsibilities:**
- Process user requests
- Generate commands via LLMs
- Manage command execution
- Handle history and state

**Key Class: `ShellGenieCore`**

```python
class ShellGenieCore:
    def __init__(self, model_config, app_config)
    async def generate_command(self, request) -> CommandResponse
    def execute(self, command) -> ExecutionResult
    async def process_request(self, prompt) -> (CommandResponse, ExecutionResult)
```

### 3. Models Layer (`models.py`)

Pydantic models for type-safe data handling.

**Key Models:**
- `ModelConfig`: LLM configuration
- `AppConfig`: Application settings
- `CommandRequest`: Input request
- `CommandResponse`: Generated command
- `ExecutionResult`: Execution outcome
- `SystemInfo`: System details

### 4. Security Layer (`security.py`)

Multi-layered security validation and sandboxing.

**Responsibilities:**
- Validate commands for risks
- Pattern-based threat detection
- Risk level assessment
- Generate safer alternatives

**Key Class: `SecurityValidator`**

```python
class SecurityValidator:
    def validate_command(self, command) -> CommandResponse
    def _check_dangerous_patterns(self, command) -> List[str]
    def _check_critical_command(self, command) -> bool
```

**Security Levels:**
1. **Strict**: Only whitelisted commands
2. **Moderate**: Block dangerous patterns (default)
3. **Permissive**: Warn but allow most
4. **Disabled**: No checks (use with caution!)

### 5. Utilities Layer (`utils.py`)

Helper functions and system utilities.

**Key Functions:**
- `get_system_info()`: Gather system details
- `execute_command()`: Run bash commands
- `check_gpu_available()`: GPU detection
- `load_command_history()`: History management
- `create_prompt_template()`: Prompt engineering

## Data Flow

### 1. Command Generation Flow

```
User Input
    ↓
CLI Parsing
    ↓
Create CommandRequest
    ↓
Add System Context
    ↓
Generate Prompt Template
    ↓
LLM Backend (Ollama/llama.cpp)
    ↓
Parse & Clean Response
    ↓
Security Validation
    ↓
CommandResponse
    ↓
Present to User
```

### 2. Execution Flow

```
CommandResponse
    ↓
User Confirmation
    ↓
Create Sandboxed Environment
    ↓
Execute Command
    ↓
Capture Output (stdout/stderr)
    ↓
Record Execution Time
    ↓
Save to History
    ↓
ExecutionResult
    ↓
Display to User
```

## LLM Backend Integration

### Ollama Backend

```python
async def _generate_with_ollama(self, prompt: str) -> str:
    # HTTP API call to Ollama server
    # POST to /api/generate
    # Returns generated text
```

**Pros:**
- Easy to use
- Model management built-in
- Good performance

**Cons:**
- Requires separate service
- Network dependency

### llama.cpp Backend

```python
async def _generate_with_llama_cpp(self, prompt: str) -> str:
    # Direct library call
    # Load model into memory
    # Generate with GPU acceleration
```

**Pros:**
- No external dependencies
- Direct GPU control
- Maximum performance

**Cons:**
- Manual model management
- More memory usage

## Security Architecture

### Pattern-Based Detection

```python
CRITICAL_COMMANDS = {
    "rm -rf /",
    "dd if=/dev/zero",
    "mkfs",
    # ...
}

DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"dd\s+if=/dev/(zero|random)",
    r"wget.*\|\s*(bash|sh)",
    # ...
]
```

### Risk Assessment

```
Command → Pattern Matching → Risk Score
                            ↓
                    ┌───────┴────────┐
                    │                │
                LOW/MEDIUM       HIGH/CRITICAL
                    │                │
                    ↓                ↓
              Present to User    Block/Warn
```

## Configuration System

### Configuration Sources (Priority Order)

1. Command-line arguments (highest)
2. Environment variables
3. Config file (`~/.shellgenie/config.yaml`)
4. Defaults (lowest)

### Configuration Loading

```python
# Load from file
config = load_config("~/.shellgenie/config.yaml")

# Override with environment
config.update_from_env()

# Override with CLI args
config.update_from_args(args)
```

## GPU Optimization

### CUDA Configuration

```python
# RTX 4080 specific optimizations
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_MAXSIZE=2147483648
```

### Memory Management

- Batch size optimization for 16GB VRAM
- Dynamic layer offloading
- Efficient context window usage

## Error Handling

### Layered Error Handling

```
CLI Layer: User-friendly messages
    ↓
Core Layer: Retry logic with tenacity
    ↓
Backend Layer: Specific error types
    ↓
Logging: Structured logs with loguru
```

### Retry Strategy

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def generate_command(self, request):
    # Will retry up to 3 times with exponential backoff
```

## Testing Architecture

### Test Structure

```
tests/
├── test_cli.py          # CLI tests
├── test_core.py         # Core logic tests
├── test_security.py     # Security validation tests
└── test_utils.py        # Utility function tests
```

### Test Categories

- **Unit Tests**: Individual components
- **Integration Tests**: Component interaction
- **GPU Tests**: GPU-specific functionality
- **Slow Tests**: Long-running tests

## Extension Points

### Adding New LLM Backends

```python
# 1. Add backend enum
class ModelBackend(str, Enum):
    NEW_BACKEND = "new_backend"

# 2. Implement generation method
async def _generate_with_new_backend(self, prompt: str) -> str:
    # Implementation

# 3. Route in generate_command()
if self.model_config.backend == ModelBackend.NEW_BACKEND:
    command = await self._generate_with_new_backend(prompt)
```

### Adding Security Patterns

```python
# Add to SecurityValidator
NEW_PATTERNS = [
    r"dangerous_pattern_1",
    r"dangerous_pattern_2",
]

DANGEROUS_PATTERNS.extend(NEW_PATTERNS)
```

## Performance Considerations

### Optimization Strategies

1. **Async I/O**: Non-blocking operations
2. **Connection Pooling**: Reuse HTTP connections
3. **Model Caching**: Keep models in memory
4. **Batch Processing**: Process multiple requests
5. **GPU Utilization**: Maximize GPU usage

### Monitoring

- Command execution time
- GPU memory usage
- Model inference latency
- Request throughput

## Future Architecture Plans

1. **Plugin System**: Dynamic backend loading
2. **Distributed Processing**: Multi-GPU support
3. **Web Interface**: REST API + Web UI
4. **Database Backend**: SQLite for history
5. **Cloud Integration**: Cloud LLM support
