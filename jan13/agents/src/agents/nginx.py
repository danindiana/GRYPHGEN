"""Nginx reverse proxy management agent."""

import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
import httpx

from src.agents.base import BaseAgent
from src.models.config import AgentConfig, NginxConfig, HealthStatus, ServiceStatus


class NginxAgent(BaseAgent):
    """Agent for managing Nginx reverse proxy."""

    def __init__(self, config: AgentConfig, nginx_config: NginxConfig):
        """Initialize Nginx agent.

        Args:
            config: Base agent configuration
            nginx_config: Nginx-specific configuration
        """
        super().__init__(config)
        self.nginx_config = nginx_config
        self.logger = self.logger.getChild("nginx")

    async def deploy(self) -> bool:
        """Deploy Nginx reverse proxy with configured settings.

        Returns:
            True if deployment successful
        """
        self.logger.info(f"Deploying Nginx on port {self.nginx_config.port}")
        self.status = ServiceStatus.STARTING

        try:
            # Create configuration file
            await self._create_config_file()

            # Enable site
            await self._enable_site()

            # Test configuration
            if not await self._test_config():
                self.status = ServiceStatus.FAILED
                return False

            # Reload Nginx
            success = await self._reload_nginx()

            if not success:
                self.status = ServiceStatus.FAILED
                return False

            # Wait for service to be ready
            ready = await self._wait_for_ready()

            if not ready:
                self.status = ServiceStatus.FAILED
                return False

            self.status = ServiceStatus.RUNNING
            self.logger.info("Nginx deployed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.status = ServiceStatus.FAILED
            return False

    async def health_check(self) -> HealthStatus:
        """Check if Nginx service is healthy.

        Returns:
            HealthStatus with current status
        """
        try:
            # Check if Nginx is running
            result = subprocess.run(
                ["sudo", "systemctl", "is-active", "nginx"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.stdout.strip() != "active":
                return HealthStatus(
                    service="nginx",
                    status=ServiceStatus.FAILED,
                    error_message="Nginx service not active",
                )

            # Test upstream connectivity through Nginx
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"http://{self.nginx_config.host}:{self.nginx_config.port}"
                response = await client.get(url)

                if response.status_code in [200, 301, 302, 404]:  # Accept various valid responses
                    return HealthStatus(
                        service="nginx",
                        status=ServiceStatus.RUNNING,
                        details={
                            "url": url,
                            "port": self.nginx_config.port,
                            "upstream": self.nginx_config.upstream_url,
                        },
                    )
                else:
                    return HealthStatus(
                        service="nginx",
                        status=ServiceStatus.FAILED,
                        error_message=f"HTTP {response.status_code}",
                    )

        except Exception as e:
            return HealthStatus(
                service="nginx",
                status=ServiceStatus.FAILED,
                error_message=str(e),
            )

    async def stop(self) -> bool:
        """Stop Nginx service.

        Returns:
            True if stopped successfully
        """
        self.logger.info("Stopping Nginx service")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "stop", "nginx"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.status = ServiceStatus.STOPPED
                self.logger.info("Nginx stopped successfully")
                return True
            else:
                self.logger.error(f"Failed to stop Nginx: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error stopping Nginx: {e}")
            return False

    async def _create_config_file(self) -> None:
        """Create Nginx configuration file for Ollama reverse proxy."""
        self.logger.info(f"Creating config file: {self.nginx_config.config_file}")

        # Build server_name directive
        server_name = self.nginx_config.server_name or "_"

        # Build SSL configuration if enabled
        ssl_config = ""
        if self.nginx_config.ssl_enabled and self.nginx_config.ssl_certificate:
            ssl_config = f"""
    ssl_certificate {self.nginx_config.ssl_certificate};
    ssl_certificate_key {self.nginx_config.ssl_key};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
"""

        config_content = f"""# Nginx reverse proxy for Ollama LLM Service
# Automatically generated by GRYPHGEN Infrastructure Agents

upstream ollama_backend {{
    server {self.nginx_config.upstream_host}:{self.nginx_config.upstream_port};
    keepalive 32;
}}

server {{
    listen {self.nginx_config.port};
    server_name {server_name};

    client_max_body_size {self.nginx_config.client_max_body_size};
{ssl_config}
    # Proxy settings
    location / {{
        proxy_pass {self.nginx_config.upstream_url};
        proxy_http_version 1.1;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        # Timeouts
        proxy_connect_timeout {self.nginx_config.proxy_timeout}s;
        proxy_send_timeout {self.nginx_config.proxy_timeout}s;
        proxy_read_timeout {self.nginx_config.proxy_timeout}s;

        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;

        # WebSocket support (if needed)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }}

    # Health check endpoint
    location /health {{
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }}

    # Metrics endpoint (optional)
    location /nginx_status {{
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }}
}}
"""

        try:
            # Write config file
            subprocess.run(
                ["sudo", "tee", self.nginx_config.config_file],
                input=config_content.encode(),
                capture_output=True,
                timeout=10,
            )

            self.logger.info("Config file created successfully")

        except Exception as e:
            self.logger.error(f"Error creating config file: {e}")
            raise

    async def _enable_site(self) -> None:
        """Enable Nginx site by creating symlink."""
        self.logger.info("Enabling Nginx site")

        try:
            # Create symlink if it doesn't exist
            subprocess.run(
                [
                    "sudo",
                    "ln",
                    "-sf",
                    self.nginx_config.config_file,
                    self.nginx_config.enabled_file,
                ],
                capture_output=True,
                timeout=10,
            )

            self.logger.info("Site enabled successfully")

        except Exception as e:
            self.logger.error(f"Error enabling site: {e}")
            raise

    async def _test_config(self) -> bool:
        """Test Nginx configuration syntax.

        Returns:
            True if configuration is valid
        """
        self.logger.info("Testing Nginx configuration")

        try:
            result = subprocess.run(
                ["sudo", "nginx", "-t"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                self.logger.info("Configuration test passed")
                return True
            else:
                self.logger.error(f"Configuration test failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error testing configuration: {e}")
            return False

    async def _reload_nginx(self) -> bool:
        """Reload Nginx to apply new configuration.

        Returns:
            True if reloaded successfully
        """
        self.logger.info("Reloading Nginx")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "reload", "nginx"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.logger.info("Nginx reloaded successfully")
                return True
            else:
                self.logger.error(f"Failed to reload Nginx: {result.stderr}")
                # Try starting if reload failed
                return await self._start_nginx()

        except Exception as e:
            self.logger.error(f"Error reloading Nginx: {e}")
            return False

    async def _start_nginx(self) -> bool:
        """Start Nginx service.

        Returns:
            True if started successfully
        """
        self.logger.info("Starting Nginx service")

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "start", "nginx"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.logger.info("Nginx started successfully")
                return True
            else:
                self.logger.error(f"Failed to start Nginx: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error starting Nginx: {e}")
            return False

    async def _wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for Nginx to be ready to accept connections.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if ready within timeout
        """
        self.logger.info(f"Waiting for Nginx to be ready (timeout: {timeout}s)")

        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            health = await self.health_check()

            if health.status == ServiceStatus.RUNNING:
                self.logger.info("Nginx is ready")
                return True

            await asyncio.sleep(2)

        self.logger.error("Timeout waiting for Nginx to be ready")
        return False

    async def get_access_logs(self, lines: int = 50) -> str:
        """Get recent Nginx access logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Log content as string
        """
        try:
            result = subprocess.run(
                ["sudo", "tail", "-n", str(lines), "/var/log/nginx/access.log"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return result.stdout if result.returncode == 0 else ""

        except Exception as e:
            self.logger.error(f"Error getting access logs: {e}")
            return ""

    async def get_error_logs(self, lines: int = 50) -> str:
        """Get recent Nginx error logs.

        Args:
            lines: Number of lines to retrieve

        Returns:
            Log content as string
        """
        try:
            result = subprocess.run(
                ["sudo", "tail", "-n", str(lines), "/var/log/nginx/error.log"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return result.stdout if result.returncode == 0 else ""

        except Exception as e:
            self.logger.error(f"Error getting error logs: {e}")
            return ""
