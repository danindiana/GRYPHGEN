"""Advanced deployment example with custom configuration and monitoring."""

import asyncio
import logging
from datetime import datetime
from src.agents.infrastructure import InfrastructureAgent
from src.models.config import DeploymentConfig, AgentConfig
from src.utils.monitoring import SystemMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def monitor_metrics(agent: InfrastructureAgent, duration: int = 60):
    """Monitor and log system metrics.

    Args:
        agent: Infrastructure agent instance
        duration: Monitoring duration in seconds
    """
    logger.info(f"Starting metrics monitoring for {duration}s...")

    monitor = SystemMonitor(gpu_enabled=True, gpu_id=0)

    start_time = datetime.utcnow()

    while (datetime.utcnow() - start_time).total_seconds() < duration:
        # Get metrics snapshot
        snapshot = await monitor.get_snapshot()

        # Log metrics
        logger.info(
            f"Metrics - CPU: {snapshot.cpu_percent:.1f}%, "
            f"Memory: {snapshot.memory_percent:.1f}%, "
            f"Disk: {snapshot.disk_percent:.1f}%"
        )

        if snapshot.gpu_utilization is not None:
            logger.info(
                f"GPU - Utilization: {snapshot.gpu_utilization:.1f}%, "
                f"Memory: {snapshot.gpu_memory_used}/{snapshot.gpu_memory_total} MB, "
                f"Temp: {snapshot.gpu_temperature:.1f}°C"
            )

        # Check agent health
        health = await agent.health_check()
        logger.info(f"Health: {health.status.value}")

        await asyncio.sleep(10)  # Update every 10 seconds


async def main():
    """Advanced deployment with custom configuration."""
    logger.info("=" * 60)
    logger.info("Advanced Deployment Example")
    logger.info("=" * 60)

    # Create custom configuration
    config = DeploymentConfig()

    # Customize agent behavior
    config.agent.auto_recover = True
    config.agent.max_retries = 5
    config.agent.health_check_interval = 15

    # Configure Ollama
    config.ollama.port = 11435
    config.ollama.models = ["llama2", "mistral", "codellama"]
    config.ollama.auto_pull_models = True
    config.ollama.context_length = 4096
    config.ollama.batch_size = 32

    # Configure Nginx
    config.nginx.port = 11434
    config.nginx.upstream_port = 11435
    config.nginx.proxy_timeout = 300

    # Configure GPU (RTX 4080 optimized)
    config.gpu.enabled = True
    config.gpu.gpu_id = 0
    config.gpu.memory_fraction = 0.9
    config.gpu.num_threads = 8

    # Enable monitoring
    config.metrics_enabled = True
    config.prometheus_enabled = True

    # Validate configuration
    errors = config.validate_ports()
    if errors:
        logger.error(f"Configuration errors: {errors}")
        return

    # Create infrastructure agent
    agent = InfrastructureAgent(config)

    try:
        # Deploy infrastructure
        logger.info("\nDeploying infrastructure with custom configuration...")
        success = await agent.deploy()

        if not success:
            logger.error("Deployment failed")
            return

        logger.info("\nDeployment successful!")

        # Get detailed status
        status = await agent.get_status()
        logger.info("\nDeployment Status:")
        logger.info(f"  Overall: {status.overall_status.value}")
        logger.info(f"  Ollama: {status.ollama.status.value}")
        logger.info(f"  Nginx: {status.nginx.status.value}")

        if status.metrics:
            logger.info("\nSystem Metrics:")
            logger.info(f"  CPU: {status.metrics.cpu_percent:.1f}%")
            logger.info(f"  Memory: {status.metrics.memory_percent:.1f}%")
            if status.metrics.gpu_utilization is not None:
                logger.info(f"  GPU: {status.metrics.gpu_utilization:.1f}%")

        # List available models
        models = await agent.ollama_agent.list_models()
        logger.info(f"\nAvailable Models: {models}")

        # Monitor metrics
        logger.info("\n" + "=" * 60)
        logger.info("Starting Monitoring")
        logger.info("=" * 60)

        # Create monitoring tasks
        monitoring_task = asyncio.create_task(monitor_metrics(agent, duration=120))

        # Wait for monitoring to complete
        await monitoring_task

    except KeyboardInterrupt:
        logger.info("\nStopping services...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Clean shutdown
        logger.info("\nShutting down...")
        await agent.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
