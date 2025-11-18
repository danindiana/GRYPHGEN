"""Basic usage example for GRYPHGEN Infrastructure Agents."""

import asyncio
import logging
from src.agents.infrastructure import InfrastructureAgent
from src.models.config import DeploymentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    """Basic deployment example."""
    logger.info("Starting basic deployment example")

    # Create configuration with defaults
    config = DeploymentConfig()

    # Customize if needed
    config.ollama.port = 11435
    config.ollama.models = ["llama2", "mistral"]
    config.nginx.port = 11434
    config.gpu.enabled = True
    config.gpu.gpu_id = 0
    config.gpu.memory_fraction = 0.9

    # Create infrastructure agent
    agent = InfrastructureAgent(config)

    try:
        # Deploy infrastructure
        logger.info("Deploying infrastructure...")
        success = await agent.deploy()

        if not success:
            logger.error("Deployment failed")
            return

        logger.info("Deployment successful!")

        # Get status
        status = await agent.get_status()
        logger.info(f"Overall Status: {status.overall_status.value}")
        logger.info(f"Ollama Status: {status.ollama.status.value}")
        logger.info(f"Nginx Status: {status.nginx.status.value}")

        # Check health
        health = await agent.health_check()
        logger.info(f"Health Check: {health.status.value}")

        # Keep running for a while (monitoring will continue)
        logger.info("Services running. Press Ctrl+C to stop...")
        await asyncio.sleep(300)  # Run for 5 minutes

    except KeyboardInterrupt:
        logger.info("Stopping services...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean shutdown
        await agent.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
