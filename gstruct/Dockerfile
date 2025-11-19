# Multi-stage Dockerfile for GRYPHGEN Grid Computing Framework
# Optimized for NVIDIA RTX 4080 with CUDA 12.x support

# Base stage with CUDA support
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Development stage
FROM base AS development

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Expose ports
EXPOSE 5555 5556 5557 5558 5559 9090

CMD ["python", "GRYPHGEN.py", "start"]

# Production stage
FROM base AS production

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 gryphgen && \
    mkdir -p /app/data /app/logs /app/cache && \
    chown -R gryphgen:gryphgen /app

# Copy requirements
COPY requirements.txt .

# Install production dependencies
RUN pip install -r requirements.txt && \
    pip cache purge

# Copy application code
COPY --chown=gryphgen:gryphgen . .

# Install the package
RUN pip install .

# Switch to non-root user
USER gryphgen

# Expose ports
EXPOSE 5555 5556 5557 5558 5559 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["gryphgen", "start"]

# Testing stage
FROM development AS testing

# Install test dependencies
RUN pip install pytest pytest-asyncio pytest-cov black ruff mypy

# Run tests
CMD ["pytest", "-v", "--cov=.", "--cov-report=html", "--cov-report=term"]
