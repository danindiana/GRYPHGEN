"""
SYMAUG Deployment Module

Manages deployment, scaling, and lifecycle of microservices in the
GRYPHGEN grid computing framework.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import docker
from docker.errors import DockerException
from loguru import logger


class ServiceStatus(Enum):
    """Microservice status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    SCALING = "scaling"


class DeploymentManager:
    """Manages microservice deployment and lifecycle."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize deployment manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.docker_client: Optional[docker.DockerClient] = None
        self.services: Dict[str, Dict[str, Any]] = {}
        self.is_running = False

    async def initialize(self):
        """Initialize deployment manager."""
        logger.info("Initializing Deployment Manager...")

        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()

            # Verify Docker is accessible
            self.docker_client.ping()

            logger.info("✓ Docker client initialized")
        except DockerException as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_client = None

        logger.info("✓ Deployment Manager initialized")

    async def start(self):
        """Start deployment manager."""
        self.is_running = True
        logger.info("Starting Deployment Manager...")

        # Start monitoring loop
        await asyncio.gather(
            self._monitor_services(),
        )

    async def shutdown(self):
        """Shutdown deployment manager."""
        logger.info("Shutting down Deployment Manager...")
        self.is_running = False

        # Optionally stop all services
        if self.config.get("stop_services_on_shutdown", False):
            for service_id in list(self.services.keys()):
                await self.stop_service(service_id)

        if self.docker_client:
            self.docker_client.close()

        logger.info("✓ Deployment Manager shutdown complete")

    async def deploy_service(
        self,
        service_id: str,
        image: str,
        name: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        ports: Optional[Dict[str, int]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        replicas: int = 1,
        gpu: bool = False,
    ) -> Dict[str, Any]:
        """Deploy a microservice.

        Args:
            service_id: Unique service identifier
            image: Docker image name
            name: Service name (optional)
            environment: Environment variables
            ports: Port mappings
            volumes: Volume mounts
            replicas: Number of replicas
            gpu: Enable GPU support

        Returns:
            Deployment result
        """
        if not self.docker_client:
            return {
                "success": False,
                "error": "Docker client not available"
            }

        if service_id in self.services:
            return {
                "success": False,
                "error": f"Service {service_id} already deployed"
            }

        logger.info(f"Deploying service {service_id} from image {image}")

        try:
            containers = []

            for i in range(replicas):
                container_name = f"{name or service_id}_{i}" if replicas > 1 else (name or service_id)

                # Prepare runtime configuration
                runtime = "nvidia" if gpu else None

                # Create and start container
                container = await asyncio.to_thread(
                    self.docker_client.containers.run,
                    image=image,
                    name=container_name,
                    environment=environment or {},
                    ports=ports or {},
                    volumes=volumes or {},
                    detach=True,
                    runtime=runtime,
                )

                containers.append(container)

            # Register service
            self.services[service_id] = {
                "id": service_id,
                "name": name or service_id,
                "image": image,
                "containers": [c.id for c in containers],
                "status": ServiceStatus.RUNNING,
                "replicas": replicas,
                "deployed_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            logger.info(f"✓ Service {service_id} deployed with {replicas} replica(s)")

            return {
                "success": True,
                "service_id": service_id,
                "containers": len(containers),
            }

        except Exception as e:
            logger.error(f"Failed to deploy service {service_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def stop_service(self, service_id: str) -> Dict[str, Any]:
        """Stop a running service.

        Args:
            service_id: Service identifier

        Returns:
            Stop result
        """
        if service_id not in self.services:
            return {
                "success": False,
                "error": f"Service {service_id} not found"
            }

        if not self.docker_client:
            return {
                "success": False,
                "error": "Docker client not available"
            }

        logger.info(f"Stopping service {service_id}")

        try:
            service = self.services[service_id]
            service["status"] = ServiceStatus.STOPPING

            # Stop all containers
            for container_id in service["containers"]:
                try:
                    container = self.docker_client.containers.get(container_id)
                    await asyncio.to_thread(container.stop, timeout=10)
                    await asyncio.to_thread(container.remove)
                except Exception as e:
                    logger.warning(f"Failed to stop container {container_id}: {e}")

            # Update service status
            service["status"] = ServiceStatus.STOPPED
            service["updated_at"] = datetime.now().isoformat()

            logger.info(f"✓ Service {service_id} stopped")

            return {
                "success": True,
                "service_id": service_id
            }

        except Exception as e:
            logger.error(f"Failed to stop service {service_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def scale_service(
        self,
        service_id: str,
        replicas: int
    ) -> Dict[str, Any]:
        """Scale a service to the specified number of replicas.

        Args:
            service_id: Service identifier
            replicas: Target number of replicas

        Returns:
            Scaling result
        """
        if service_id not in self.services:
            return {
                "success": False,
                "error": f"Service {service_id} not found"
            }

        if not self.docker_client:
            return {
                "success": False,
                "error": "Docker client not available"
            }

        service = self.services[service_id]
        current_replicas = service["replicas"]

        if current_replicas == replicas:
            return {
                "success": True,
                "message": f"Service already at {replicas} replicas"
            }

        logger.info(f"Scaling service {service_id} from {current_replicas} to {replicas} replicas")

        try:
            service["status"] = ServiceStatus.SCALING

            if replicas > current_replicas:
                # Scale up
                for i in range(current_replicas, replicas):
                    container_name = f"{service['name']}_{i}"

                    container = await asyncio.to_thread(
                        self.docker_client.containers.run,
                        image=service["image"],
                        name=container_name,
                        detach=True,
                    )

                    service["containers"].append(container.id)

            else:
                # Scale down
                containers_to_remove = service["containers"][replicas:]
                for container_id in containers_to_remove:
                    try:
                        container = self.docker_client.containers.get(container_id)
                        await asyncio.to_thread(container.stop, timeout=10)
                        await asyncio.to_thread(container.remove)
                    except Exception as e:
                        logger.warning(f"Failed to remove container {container_id}: {e}")

                service["containers"] = service["containers"][:replicas]

            service["replicas"] = replicas
            service["status"] = ServiceStatus.RUNNING
            service["updated_at"] = datetime.now().isoformat()

            logger.info(f"✓ Service {service_id} scaled to {replicas} replicas")

            return {
                "success": True,
                "service_id": service_id,
                "replicas": replicas
            }

        except Exception as e:
            logger.error(f"Failed to scale service {service_id}: {e}")
            service["status"] = ServiceStatus.FAILED
            return {
                "success": False,
                "error": str(e)
            }

    async def _monitor_services(self):
        """Monitor service health."""
        while self.is_running:
            try:
                if self.docker_client:
                    for service_id, service in self.services.items():
                        # Check container status
                        running_containers = 0
                        for container_id in service["containers"]:
                            try:
                                container = self.docker_client.containers.get(container_id)
                                if container.status == "running":
                                    running_containers += 1
                            except Exception:
                                pass

                        # Update service status
                        if running_containers == 0 and service["status"] == ServiceStatus.RUNNING:
                            service["status"] = ServiceStatus.FAILED
                            logger.warning(f"Service {service_id} has failed")

                await asyncio.sleep(30)  # Monitor every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring services: {e}")
                await asyncio.sleep(30)

    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service.

        Args:
            service_id: Service identifier

        Returns:
            Service status or None
        """
        if service_id not in self.services:
            return None

        service = self.services[service_id]

        return {
            "id": service["id"],
            "name": service["name"],
            "image": service["image"],
            "status": service["status"].value,
            "replicas": service["replicas"],
            "containers": len(service["containers"]),
            "deployed_at": service["deployed_at"],
            "updated_at": service["updated_at"],
        }

    def get_status(self) -> Dict[str, Any]:
        """Get overall deployment status.

        Returns:
            Status dictionary
        """
        return {
            "running": self.is_running,
            "docker_available": self.docker_client is not None,
            "total_services": len(self.services),
            "services": {
                service_id: self.get_service_status(service_id)
                for service_id in self.services.keys()
            }
        }


def main():
    """Main entry point."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from config.settings import load_config

    async def run():
        config = load_config()
        manager = DeploymentManager(config)

        try:
            await manager.initialize()

            # Example deployment
            if manager.docker_client:
                result = await manager.deploy_service(
                    "test_service",
                    "nginx:latest",
                    name="test_nginx",
                    ports={"80/tcp": 8080},
                    replicas=2
                )
                print(f"\nDeployment result: {result}\n")

                # Print status
                print(f"Deployment Status:\n{manager.get_status()}\n")

                # Wait a bit
                await asyncio.sleep(5)

                # Stop service
                stop_result = await manager.stop_service("test_service")
                print(f"\nStop result: {stop_result}\n")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await manager.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
