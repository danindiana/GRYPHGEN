# GRYPHGEN RAG - Docker Image
# Optimized for NVIDIA RTX 4080 16GB

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.3.1-cudnn9-devel-ubuntu22.04

# Metadata
LABEL maintainer="GRYPHGEN Contributors"
LABEL description="Modern implementation of SimGRAG and CAG for Retrieval-Augmented Generation"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Install GPU-specific packages
RUN pip install \
    faiss-gpu \
    cupy-cuda12x \
    triton \
    flash-attn --no-build-isolation

# Copy package files
COPY pyproject.toml setup.py README.md /workspace/
COPY simgrag/ /workspace/simgrag/
COPY cag/ /workspace/cag/
COPY common/ /workspace/common/
COPY examples/ /workspace/examples/
COPY tests/ /workspace/tests/
COPY docs/ /workspace/docs/
COPY Makefile /workspace/

# Install the package in development mode
RUN pip install -e ".[dev,gpu]"

# Create directories for data and cache
RUN mkdir -p /workspace/data /workspace/cache /workspace/models

# Set up Ollama (optional - uncomment if needed)
# RUN curl -fsSL https://ollama.ai/install.sh | sh

# GPU optimization environment variables for RTX 4080
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TOKENIZERS_PARALLELISM=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["/bin/bash"]

# Usage examples:
# Build: docker build -t gryphgen-rag:latest .
# Run: docker run --rm -it --gpus all -v $(pwd):/workspace gryphgen-rag:latest
# Test: docker run --rm --gpus all gryphgen-rag:latest make test
# Examples: docker run --rm --gpus all gryphgen-rag:latest python examples/simgrag_example.py
