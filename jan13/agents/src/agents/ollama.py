"""Ollama LLM service management agent."""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import httpx

from src.agents.base import BaseAgent
from src.models.config import AgentConfig, OllamaConfig, HealthStatus, ServiceStatus


class OllamaAgent(BaseAgent):
    """Agent for managing Ollama LLM service."""

    def __init__(self, config: AgentConfig, ollama_config: OllamaConfig):
        """Initialize Ollama agent.

        Args:
            config: Base agent configuration
            ollama_config: Ollama-specific configuration
        """
        super().__init__(config)
        self.ollama_config = ollama_config
        self.logger = self.logger.getChild("ollama")

    async def deploy(self) -> bool:
        """Deploy Ollama service with configured settings.

        Returns:
            True if deployment successful
        """
        self.logger.info(f"Deploying Ollama on {self.ollama_config.url}")
        self.status = ServiceStatus.STARTING

        try:
            # Check if port is available
            if not await self._check_port_available(self.ollama_config.port):
                self.logger.warning(f"Port {self.ollama_config.port} is in use")
                await self._resolve_port_conflict()

            # Create systemd service file
            await self._create_service_file()

            # Start service
            success = await self._start_service()

            if not success:
                self.status = ServiceStatus.FAILED
                return False

            # Wait for service to be ready
            ready = await self._wait_for_ready()

            if not ready:
                self.status = ServiceStatus.FAILED
                return False

            # Pull models if configured
            if self.ollama_config.auto_pull_models:
                await self._pull_models()

            self.status = ServiceStatus.RUNNING
            self.logger.info("Ollama deployed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.status = ServiceStatus.FAILED
            return False

    async def health_check(self) -> HealthStatus:
        """Check if Ollama service is healthy.

        Returns:
            HealthStatus with current status
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.ollama_config.url)

                if response.status_code == 200:
                    return HealthStatus(
                        service="ollama",
                        status=ServiceStatus.RUNNING,
                        details={"url": self.ollama_config.url, "port": self.ollama_config.port},
                    )
                else:
                    return HealthStatus(
                        service="ollama",
                        status=ServiceStatus.FAILED,
                        error_message=f"HTTP {response.status_code}",
                    )

        except Exception as e:
            return HealthStatus(
                service="ollama",
                status=ServiceStatus.FAILED,
                error_message=str(e),
            )

    async def stop(self) -> bool:
        """Stop Ollama service.

        Returns:
            True if stopped successfully
        """
        self.logger.info("Stopping Ollama service")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "stop", "ollama.service"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.status = ServiceStatus.STOPPED
                self.logger.info("Ollama stopped successfully")
                return True
            else:
                self.logger.error(f"Failed to stop Ollama: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error stopping Ollama: {e}")
            return False

    async def _check_port_available(self, port: int) -> bool:
        """Check if a port is available.

        Args:
            port: Port number to check

        Returns:
            True if port is available
        """
        try:
            result = subprocess.run(
                ["lsof", f"-i:{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # lsof returns 0 if port is in use, 1 if available
            return result.returncode != 0

        except Exception as e:
            self.logger.warning(f"Could not check port availability: {e}")
            return True  # Assume available if check fails

    async def _resolve_port_conflict(self) -> None:
        """Attempt to resolve port conflict."""
        self.logger.info(f"Attempting to resolve port conflict on {self.ollama_config.port}")

        try:
            # First, try to stop any existing Ollama service
            subprocess.run(
                ["sudo", "systemctl", "stop", "ollama.service"],
                capture_output=True,
                timeout=10,
            )
            await asyncio.sleep(2)

            # Check if port is now available
            if await self._check_port_available(self.ollama_config.port):
                self.logger.info("Port conflict resolved")
                return

            # If still in use, find and kill process
            result = subprocess.run(
                ["lsof", "-t", f"-i:{self.ollama_config.port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.stdout:
                pid = result.stdout.strip()
                self.logger.warning(f"Killing process {pid} using port {self.ollama_config.port}")
                subprocess.run(["sudo", "kill", "-9", pid], timeout=5)
                await asyncio.sleep(2)

        except Exception as e:
            self.logger.error(f"Error resolving port conflict: {e}")

    async def _create_service_file(self) -> None:
        """Create systemd service file for Ollama."""
        self.logger.info(f"Creating service file: {self.ollama_config.service_file}")

        service_content = f"""[Unit]
Description=Ollama LLM Service
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST={self.ollama_config.host}:{self.ollama_config.port}"
Environment="OLLAMA_NUM_GPU={self.ollama_config.num_gpu}"
Environment="OLLAMA_GPU_MEMORY_FRACTION={self.ollama_config.gpu_memory_fraction}"

[Install]
WantedBy=multi-user.target
"""

        try:
            # Write service file
            subprocess.run(
                ["sudo", "tee", self.ollama_config.service_file],
                input=service_content.encode(),
                capture_output=True,
                timeout=10,
            )

            # Reload systemd
            subprocess.run(["sudo", "systemctl", "daemon-reload"], timeout=10)

            self.logger.info("Service file created successfully")

        except Exception as e:
            self.logger.error(f"Error creating service file: {e}")
            raise

    async def _start_service(self) -> bool:
        """Start Ollama systemd service.

        Returns:
            True if started successfully
        """
        self.logger.info("Starting Ollama service")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "start", "ollama.service"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.logger.info("Ollama service started")
                return True
            else:
                self.logger.error(f"Failed to start service: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error starting service: {e}")
            return False

    async def _wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for Ollama to be ready to accept connections.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if ready within timeout
        """
        self.logger.info(f"Waiting for Ollama to be ready (timeout: {timeout}s)")

        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            health = await self.health_check()

            if health.status == ServiceStatus.RUNNING:
                self.logger.info("Ollama is ready")
                return True

            await asyncio.sleep(2)

        self.logger.error("Timeout waiting for Ollama to be ready")
        return False

    async def _pull_models(self) -> None:
        """Pull configured models."""
        self.logger.info(f"Pulling models: {self.ollama_config.models}")

        for model in self.ollama_config.models:
            try:
                self.logger.info(f"Pulling model: {model}")
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes per model
                )

                if result.returncode == 0:
                    self.logger.info(f"Model {model} pulled successfully")
                else:
                    self.logger.warning(f"Failed to pull model {model}: {result.stderr}")

            except Exception as e:
                self.logger.warning(f"Error pulling model {model}: {e}")

    async def list_models(self) -> List[str]:
        """List available models.

        Returns:
            List of model names
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Parse output to extract model names
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                models = [line.split()[0] for line in lines if line]
                return models
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
