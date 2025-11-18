# CALISOTA Architecture Documentation

## Overview

This document provides detailed architectural information for the CALISOTA AI Engine.

## System Components

### 1. API Layer (FastAPI)

Entry point for all client interactions. Provides:
- RESTful API endpoints
- Automatic OpenAPI documentation
- Request validation with Pydantic
- CORS handling
- Health checks

### 2. RAG System (FAISS)

GPU-accelerated retrieval system using FAISS for similarity search:
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2
- **Index Type**: IVF (Inverted File) with Product Quantization
- **GPU Acceleration**: CUDA-optimized for RTX 4080
- **Storage**: Persistent index on disk

### 3. Agent System

Three-tier ensemble architecture:

#### Slow Thinker (GPT-4)
- High-level task planning
- Task decomposition
- Context retrieval from RAG
- Clarification requests to frontier models

#### Fast Thinker (GPT-3.5)
- Rapid code generation
- Retrieves code samples from RAG
- Quick iteration cycles
- Multi-language support

#### Actor-Critic (GPT-3.5)
- Code evaluation
- Quality scoring
- Improvement suggestions
- Feedback loops

### 4. Sandbox System

Secure code execution environment:
- **Container Technology**: Docker
- **Supported Languages**: Python, Rust, Go, C/C++, Perl
- **Security**: Network isolation, read-only filesystem, resource limits
- **Monitoring**: Execution time, memory usage, exit codes

### 5. Monitoring & Observability

- **Metrics**: Prometheus
- **Visualization**: Grafana
- **Logging**: Structured logging with structlog
- **Tracing**: Ready for OpenTelemetry integration

## Data Flow

See README.md for detailed sequence diagrams.

## GPU Optimization

Optimizations for RTX 4080 16GB:
- Mixed precision (FP16/BF16)
- Flash Attention 2
- Batch processing
- Memory-mapped FAISS indices
- Efficient VRAM management

## Security

- API key authentication
- Docker container isolation
- No network access in sandboxes
- Resource limits per execution
- Input validation

## Scalability

- Horizontal scaling with Celery workers
- Redis for distributed caching
- Kubernetes-ready architecture
- Stateless API design

## Future Enhancements

See docs/archive/ for original architecture proposals including:
- Episodic Memory Module
- Apache Airflow integration
- Advanced graph processing
- Enhanced self-healing mechanisms
