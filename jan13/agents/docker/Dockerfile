# Multi-stage Dockerfile for GRYPHGEN Infrastructure Agents
# Optimized for NVIDIA RTX 4080

# ═══════════════════════════════════════════════════════════════
# Stage 1: Builder
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim as builder

LABEL maintainer="GRYPHGEN Contributors <dan@gryphgen.dev>"
LABEL description="Infrastructure Agents for Ollama LLM deployment"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ═══════════════════════════════════════════════════════════════
# Stage 2: Runtime
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    nginx \
    curl \
    lsof \
    systemd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN groupadd -r appuser && \
    useradd -r -g appuser -s /bin/bash appuser && \
    mkdir -p /app /var/log/gryphgen && \
    chown -R appuser:appuser /app /var/log/gryphgen

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser pyproject.toml .

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose ports
EXPOSE 8080 11434 11435 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to app user
USER appuser

# Set GPU environment variables (optimized for RTX 4080)
ENV CUDA_VISIBLE_DEVICES=0 \
    OLLAMA_NUM_GPU=1 \
    OLLAMA_GPU_MEMORY_FRACTION=0.9 \
    OLLAMA_NUM_THREAD=8

# Run FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
