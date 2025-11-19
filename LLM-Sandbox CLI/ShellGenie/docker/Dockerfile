# ShellGenie Docker Image - Optimized for NVIDIA RTX 4080
# CUDA 12.4, Python 3.11, Latest Ollama

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

LABEL maintainer="GRYPHGEN Team <dev@gryphgen.ai>"
LABEL description="ShellGenie - AI-powered bash assistant with GPU acceleration"
LABEL version="2.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA and GPU optimizations for RTX 4080
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# RTX 4080 specific optimizations (Ada Lovelace architecture, Compute Capability 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_MAXSIZE=2147483648

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    # Python dependencies
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    # Development tools
    vim \
    nano \
    htop \
    tmux \
    bash-completion \
    # Network tools
    net-tools \
    iputils-ping \
    # System utilities
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==69.0.3 \
    wheel==0.42.0

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create non-root user
RUN useradd -m -s /bin/bash shellgenie && \
    echo "shellgenie:shellgenie" | chpasswd && \
    usermod -aG sudo shellgenie && \
    echo "shellgenie ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /home/shellgenie

# Copy application files
COPY --chown=shellgenie:shellgenie . /home/shellgenie/shellgenie/

# Install Python dependencies
RUN cd /home/shellgenie/shellgenie && \
    python3 -m pip install --no-cache-dir -e ".[gpu,dev]"

# Build llama.cpp with CUDA support for RTX 4080
RUN git clone https://github.com/ggerganov/llama.cpp.git /home/shellgenie/llama.cpp && \
    cd /home/shellgenie/llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DLLAMA_CUDA=ON -DLLAMA_CUDA_F16=ON -DCMAKE_CUDA_ARCHITECTURES=89 && \
    cmake --build . --config Release -j$(nproc) && \
    chown -R shellgenie:shellgenie /home/shellgenie/llama.cpp

# Create necessary directories
RUN mkdir -p /home/shellgenie/.ollama \
    /home/shellgenie/.shellgenie \
    /home/shellgenie/models && \
    chown -R shellgenie:shellgenie /home/shellgenie

# Switch to non-root user
USER shellgenie

# Setup shell environment
RUN echo 'export PATH=/home/shellgenie/.local/bin:$PATH' >> /home/shellgenie/.bashrc && \
    echo 'export OLLAMA_HOST=0.0.0.0:11434' >> /home/shellgenie/.bashrc && \
    echo 'alias sg=shellgenie' >> /home/shellgenie/.bashrc

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in background\n\
ollama serve > /tmp/ollama.log 2>&1 &\n\
sleep 5\n\
\n\
# Pull default model if not exists\n\
if [ ! -d "/home/shellgenie/.ollama/models/manifests/registry.ollama.ai/library/llama3.2" ]; then\n\
    echo "Pulling llama3.2 model..."\n\
    ollama pull llama3.2\n\
fi\n\
\n\
# Keep container running\n\
exec "$@"\n\
' > /home/shellgenie/entrypoint.sh && chmod +x /home/shellgenie/entrypoint.sh

# Expose Ollama port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

ENTRYPOINT ["/home/shellgenie/entrypoint.sh"]
CMD ["/bin/bash"]
