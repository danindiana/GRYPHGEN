# CALISOTA AI Engine - Production Dockerfile
# Optimized for NVIDIA RTX 4080 16GB

FROM nvidia/cuda:12.6.2-cudnn9-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --no-dev --no-root

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create data directories
RUN mkdir -p /app/data/faiss_index /app/logs

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
CMD ["uvicorn", "src.calisota.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage
FROM base AS development

# Install development dependencies
RUN poetry install --no-root

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "src.calisota.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
