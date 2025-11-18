"""Infrastructure orchestration agent that manages Ollama and Nginx."""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.agents.base import BaseAgent
from src.agents.ollama import OllamaAgent
from src.agents.nginx import NginxAgent
from src.models.config import (
    AgentConfig,
    DeploymentConfig,
    DeploymentStatus,
    HealthStatus,
    ServiceStatus,
    MetricsSnapshot,
)


class InfrastructureAgent(BaseAgent):
    """Main infrastructure orchestration agent."""

    def __init__(self, config: Optional[DeploymentConfig] = None):
        """Initialize infrastructure agent.

        Args:
            config: Deployment configuration (uses defaults if None)
        """
        self.deployment_config = config or DeploymentConfig()

        # Initialize base agent
        super().__init__(self.deployment_config.agent)

        self.logger = self.logger.getChild("infrastructure")

        # Validate configuration
        errors = self.deployment_config.validate_ports()
        if errors:
            self.logger.warning(f"Configuration warnings: {errors}")

        # Initialize child agents
        self.ollama_agent = OllamaAgent(
            config=AgentConfig(name="ollama-agent"),
            ollama_config=self.deployment_config.ollama,
        )

        self.nginx_agent = NginxAgent(
            config=AgentConfig(name="nginx-agent"),
            nginx_config=self.deployment_config.nginx,
        )

        self._monitoring_task: Optional[asyncio.Task] = None

    async def deploy(self) -> bool:
        """Deploy complete infrastructure stack.

        Returns:
            True if deployment successful
        """
        self.logger.info("=" * 60)
        self.logger.info("GRYPHGEN Infrastructure Deployment Starting")
        self.logger.info("=" * 60)

        self.status = ServiceStatus.STARTING

        try:
            # Step 1: Deploy Ollama
            self.logger.info("\n[1/2] Deploying Ollama LLM Service...")
            ollama_success = await self.ollama_agent.deploy()

            if not ollama_success:
                self.logger.error("Ollama deployment failed")
                self.status = ServiceStatus.FAILED
                return False

            # Step 2: Deploy Nginx
            self.logger.info("\n[2/2] Deploying Nginx Reverse Proxy...")
            nginx_success = await self.nginx_agent.deploy()

            if not nginx_success:
                self.logger.error("Nginx deployment failed")
                self.status = ServiceStatus.FAILED
                return False

            # Verify full stack
            self.logger.info("\n[✓] Verifying deployment...")
            if await self._verify_deployment():
                self.status = ServiceStatus.RUNNING
                self.logger.info("\n" + "=" * 60)
                self.logger.info("DEPLOYMENT COMPLETE ✓")
                self.logger.info("=" * 60)
                self._print_deployment_info()

                # Start monitoring if enabled
                if self.deployment_config.metrics_enabled:
                    self._monitoring_task = asyncio.create_task(self.monitor())

                return True
            else:
                self.logger.error("Deployment verification failed")
                self.status = ServiceStatus.FAILED
                return False

        except Exception as e:
            self.logger.error(f"Deployment error: {e}")
            self.status = ServiceStatus.FAILED
            return False

    async def health_check(self) -> HealthStatus:
        """Check health of entire infrastructure.

        Returns:
            Overall health status
        """
        # Check Ollama
        ollama_health = await self.ollama_agent.health_check()

        # Check Nginx
        nginx_health = await self.nginx_agent.health_check()

        # Determine overall status
        if (
            ollama_health.status == ServiceStatus.RUNNING
            and nginx_health.status == ServiceStatus.RUNNING
        ):
            overall_status = ServiceStatus.RUNNING
        elif (
            ollama_health.status == ServiceStatus.FAILED
            or nginx_health.status == ServiceStatus.FAILED
        ):
            overall_status = ServiceStatus.FAILED
        else:
            overall_status = ServiceStatus.UNKNOWN

        return HealthStatus(
            service="infrastructure",
            status=overall_status,
            details={
                "ollama": ollama_health.dict(),
                "nginx": nginx_health.dict(),
            },
        )

    async def stop(self) -> bool:
        """Stop all infrastructure services.

        Returns:
            True if all services stopped successfully
        """
        self.logger.info("Stopping infrastructure...")

        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop services in reverse order
        nginx_stopped = await self.nginx_agent.stop()
        ollama_stopped = await self.ollama_agent.stop()

        success = nginx_stopped and ollama_stopped

        if success:
            self.status = ServiceStatus.STOPPED
            self.logger.info("All services stopped successfully")
        else:
            self.logger.warning("Some services failed to stop")

        return success

    async def get_status(self) -> DeploymentStatus:
        """Get comprehensive deployment status.

        Returns:
            DeploymentStatus with health and metrics
        """
        # Get health status
        ollama_health = await self.ollama_agent.health_check()
        nginx_health = await self.nginx_agent.health_check()

        # Determine overall status
        if (
            ollama_health.status == ServiceStatus.RUNNING
            and nginx_health.status == ServiceStatus.RUNNING
        ):
            overall = ServiceStatus.RUNNING
        elif (
            ollama_health.status == ServiceStatus.FAILED
            or nginx_health.status == ServiceStatus.FAILED
        ):
            overall = ServiceStatus.FAILED
        else:
            overall = ServiceStatus.UNKNOWN

        # Get metrics if enabled
        metrics = None
        if self.deployment_config.metrics_enabled:
            try:
                from src.utils.monitoring import SystemMonitor

                monitor = SystemMonitor(
                    gpu_enabled=self.deployment_config.gpu.enabled,
                    gpu_id=self.deployment_config.gpu.gpu_id,
                )
                metrics = await monitor.get_snapshot()
            except Exception as e:
                self.logger.warning(f"Could not get metrics: {e}")

        return DeploymentStatus(
            ollama=ollama_health,
            nginx=nginx_health,
            overall_status=overall,
            metrics=metrics,
            errors=[],
        )

    async def _verify_deployment(self) -> bool:
        """Verify that the complete deployment is working.

        Returns:
            True if verification successful
        """
        try:
            # Check both services
            health = await self.health_check()

            if health.status != ServiceStatus.RUNNING:
                return False

            # Test end-to-end: Nginx → Ollama
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"http://{self.deployment_config.nginx.host}:{self.deployment_config.nginx.port}"
                response = await client.get(url)

                if response.status_code in [200, 404]:  # 404 is OK for root endpoint
                    self.logger.info("End-to-end verification passed")
                    return True
                else:
                    self.logger.error(f"End-to-end test failed: HTTP {response.status_code}")
                    return False

        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            return False

    def _print_deployment_info(self) -> None:
        """Print deployment information to console."""
        info = f"""
╔══════════════════════════════════════════════════════════════╗
║            GRYPHGEN Infrastructure Deployment Info           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Ollama LLM Service:                                         ║
║    URL:  http://{self.deployment_config.ollama.host}:{self.deployment_config.ollama.port:<5}                           ║
║    GPU:  {self.deployment_config.gpu.gpu_id} ({self.deployment_config.gpu.vram_gb}GB VRAM, {self.deployment_config.gpu.memory_fraction*100:.0f}% allocated)                  ║
║                                                              ║
║  Nginx Reverse Proxy:                                        ║
║    URL:  http://{self.deployment_config.nginx.host}:{self.deployment_config.nginx.port:<5}                            ║
║    Backend: {self.deployment_config.nginx.upstream_url:<45}   ║
║                                                              ║
║  Models Available:                                           ║
"""
        for model in self.deployment_config.ollama.models:
            info += f"║    - {model:<54} ║\n"

        info += """║                                                              ║
║  Test Commands:                                              ║
║    Health:                                                   ║
║      curl http://localhost:11434/health                      ║
║                                                              ║
║    Generate:                                                 ║
║      curl -X POST http://localhost:11434/api/generate \\      ║
║        -H "Content-Type: application/json" \\                 ║
║        -d '{"model": "llama2", "prompt": "Hello!"}'          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(info)


# CLI interface for direct usage
def cli() -> None:
    """Command-line interface for infrastructure agent."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="GRYPHGEN Infrastructure Agent")
    parser.add_argument(
        "action",
        choices=["deploy", "status", "stop", "health"],
        help="Action to perform",
    )
    parser.add_argument("--ollama-port", type=int, default=11435, help="Ollama port")
    parser.add_argument("--nginx-port", type=int, default=11434, help="Nginx port")
    parser.add_argument("--gpu-enabled", action="store_true", help="Enable GPU")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=0.9, help="GPU memory fraction"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create configuration
    from src.models.config import DeploymentConfig, OllamaConfig, NginxConfig, GPUConfig

    config = DeploymentConfig()
    config.ollama.port = args.ollama_port
    config.nginx.port = args.nginx_port
    config.nginx.upstream_port = args.ollama_port
    config.gpu.enabled = args.gpu_enabled
    config.gpu.gpu_id = args.gpu_id
    config.gpu.memory_fraction = args.gpu_memory_fraction

    # Create agent
    agent = InfrastructureAgent(config)

    # Run action
    async def run_action() -> None:
        if args.action == "deploy":
            success = await agent.deploy()
            sys.exit(0 if success else 1)

        elif args.action == "status":
            status = await agent.get_status()
            print(f"\nOverall Status: {status.overall_status.value}")
            print(f"Ollama: {status.ollama.status.value}")
            print(f"Nginx: {status.nginx.status.value}")
            if status.metrics:
                print(f"\nCPU: {status.metrics.cpu_percent:.1f}%")
                print(f"Memory: {status.metrics.memory_percent:.1f}%")
                if status.metrics.gpu_utilization is not None:
                    print(f"GPU: {status.metrics.gpu_utilization:.1f}%")

        elif args.action == "health":
            health = await agent.health_check()
            print(f"Health Status: {health.status.value}")
            if health.error_message:
                print(f"Error: {health.error_message}")

        elif args.action == "stop":
            success = await agent.stop()
            sys.exit(0 if success else 1)

    asyncio.run(run_action())


if __name__ == "__main__":
    cli()
