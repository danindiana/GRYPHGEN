
# Multi-stage build for MCP Reliability System
# Use latest GHC for better performance and features
FROM haskell:9.6-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \
    zlib1g-dev \
    libtinfo-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy cabal files first for better caching
COPY mcp-reliability-system.cabal ./
COPY cabal.project ./

# Update cabal and install dependencies
RUN cabal update && cabal build --dependencies-only

# Copy source code
COPY . .

# Build the application
RUN cabal build --enable-optimization=2

# Install the binary
RUN cabal install --install-method=copy --installdir=/app/bin

# Production stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libgmp10 \
    libssl3 \
    libtinfo6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd -m -u 1000 mcpuser

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/bin/mcp-server /app/bin/mcp-server

# Copy configuration files
COPY config/ /app/config/
COPY docker/entrypoint.sh /app/entrypoint.sh

# Set permissions
RUN chmod +x /app/bin/mcp-server /app/entrypoint.sh
RUN chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "/app/config/production.yaml"]
