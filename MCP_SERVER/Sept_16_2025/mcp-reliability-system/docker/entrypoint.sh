
#!/bin/bash
set -e

# Default configuration
CONFIG_PATH=${MCP_CONFIG_PATH:-"/app/config/production.yaml"}
LOG_LEVEL=${MCP_LOG_LEVEL:-"INFO"}

# Create necessary directories
mkdir -p /app/logs /app/cache

# Wait for dependencies (if any)
if [ -n "$WAIT_FOR_SERVICES" ]; then
    echo "Waiting for services: $WAIT_FOR_SERVICES"
    for service in $WAIT_FOR_SERVICES; do
        echo "Waiting for $service..."
        while ! nc -z ${service%:*} ${service#*:}; do
            sleep 1
        done
        echo "$service is ready"
    done
fi

# Print startup information
echo "Starting MCP Reliability Server..."
echo "Config: $CONFIG_PATH"
echo "Log Level: $LOG_LEVEL"
echo "Arguments: $@"

# Execute the main application
exec /app/bin/mcp-server "$@"
