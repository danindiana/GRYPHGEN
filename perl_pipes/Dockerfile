# GRYPHGEN Perl Pipes - Multi-Model IPC System
# Optimized for NVIDIA RTX 4080 16GB

FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

LABEL maintainer="GRYPHGEN Project"
LABEL description="Multi-Model LLM IPC System with Perl Pipes"
LABEL version="2.0.0"

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PERL_MM_USE_DEFAULT=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    perl \
    cpanminus \
    make \
    gcc \
    libc6-dev \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA tools for GPU monitoring
RUN apt-get update && apt-get install -y \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/* || true

# Set working directory
WORKDIR /opt/gryphgen/perl_pipes

# Copy dependency files first for better caching
COPY cpanfile ./

# Install Perl dependencies
RUN cpanm --quiet --notest --installdeps .

# Copy application files
COPY bin/ ./bin/
COPY lib/ ./lib/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY config/ ./config/
COPY Makefile ./
COPY README.md ./

# Make scripts executable
RUN chmod +x bin/*.pl bin/*.sh examples/*.sh

# Create necessary directories
RUN mkdir -p logs /tmp/gryphgen

# Set environment variables
ENV PATH="/opt/gryphgen/perl_pipes/bin:${PATH}"
ENV PIPE_DIR=/tmp/gryphgen
ENV LOG_DIR=/opt/gryphgen/perl_pipes/logs

# Verify Perl version
RUN perl -v

# Run tests during build (optional, comment out for faster builds)
# RUN prove -lr tests/

# Default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD perl -e 'exit 0' || exit 1

# Expose volumes for logs and pipes
VOLUME ["/opt/gryphgen/perl_pipes/logs", "/tmp/gryphgen"]

# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
