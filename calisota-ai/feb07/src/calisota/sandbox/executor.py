"""
Multi-language sandbox executor using Docker.
Supports Python, Rust, Go, C/C++, and Perl.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import docker
from docker.models.containers import Container

from src.calisota.core.config import Settings

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """
    Execute code in isolated Docker containers.

    Supports multiple programming languages with resource limits.
    """

    # Language-specific Docker images
    LANGUAGE_IMAGES = {
        "python": "python:3.11-slim",
        "rust": "rust:1.75-slim",
        "go": "golang:1.21-alpine",
        "cpp": "gcc:13-slim",
        "c": "gcc:13-slim",
        "perl": "perl:5.38-slim",
    }

    # Language-specific execution commands
    EXECUTE_COMMANDS = {
        "python": ["python", "/code/main.py"],
        "rust": ["sh", "-c", "rustc /code/main.rs -o /code/main && /code/main"],
        "go": ["go", "run", "/code/main.go"],
        "cpp": ["sh", "-c", "g++ /code/main.cpp -o /code/main && /code/main"],
        "c": ["sh", "-c", "gcc /code/main.c -o /code/main && /code/main"],
        "perl": ["perl", "/code/main.pl"],
    }

    # File extensions
    FILE_EXTENSIONS = {
        "python": ".py",
        "rust": ".rs",
        "go": ".go",
        "cpp": ".cpp",
        "c": ".c",
        "perl": ".pl",
    }

    def __init__(self, settings: Settings) -> None:
        """Initialize sandbox executor."""
        self.settings = settings
        self.docker_client = docker.from_env()
        self._ensure_network()

    def _ensure_network(self) -> None:
        """Ensure Docker network exists."""
        try:
            self.docker_client.networks.get(self.settings.docker_network)
        except docker.errors.NotFound:
            self.docker_client.networks.create(
                self.settings.docker_network,
                driver="bridge"
            )
            logger.info(f"Created Docker network: {self.settings.docker_network}")

    async def execute(
        self,
        code: str,
        language: str,
        timeout: Optional[int] = None,
        memory_limit: str = "512m",
        cpu_quota: int = 50000  # 50% of one CPU
    ) -> dict:
        """
        Execute code in a sandboxed container.

        Args:
            code: Source code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            memory_limit: Memory limit (e.g., "512m", "1g")
            cpu_quota: CPU quota in microseconds per 100ms period

        Returns:
            Execution result with stdout, stderr, exit_code, and execution_time
        """
        if language not in self.LANGUAGE_IMAGES:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "supported_languages": list(self.LANGUAGE_IMAGES.keys())
            }

        timeout = timeout or self.settings.sandbox_timeout

        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / f"main{self.FILE_EXTENSIONS[language]}"
            code_path.write_text(code)

            try:
                # Run code in Docker container
                result = await self._run_container(
                    image=self.LANGUAGE_IMAGES[language],
                    command=self.EXECUTE_COMMANDS[language],
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    timeout=timeout,
                    memory_limit=memory_limit,
                    cpu_quota=cpu_quota
                )

                return result

            except Exception as e:
                logger.error(f"Sandbox execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1
                }

    async def _run_container(
        self,
        image: str,
        command: list[str],
        volumes: dict,
        timeout: int,
        memory_limit: str,
        cpu_quota: int
    ) -> dict:
        """Run Docker container with resource limits."""
        import time

        start_time = time.time()

        container: Optional[Container] = None
        try:
            # Create and start container
            container = self.docker_client.containers.run(
                image=image,
                command=command,
                volumes=volumes,
                network=self.settings.docker_network,
                mem_limit=memory_limit,
                cpu_quota=cpu_quota,
                detach=True,
                remove=False,  # We'll remove it manually
                network_mode="none",  # No network access for security
                security_opt=["no-new-privileges"],
                read_only=True,
                tmpfs={"/tmp": "size=100m,mode=1777"}
            )

            # Wait for container with timeout
            try:
                exit_code = container.wait(timeout=timeout)["StatusCode"]
            except Exception:
                container.kill()
                raise TimeoutError(f"Execution exceeded {timeout} seconds")

            # Get logs
            logs = container.logs()
            stdout = logs.decode("utf-8", errors="replace")

            execution_time = time.time() - start_time

            return {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": "",  # Combined in stdout for Docker
                "exit_code": exit_code,
                "execution_time": execution_time,
                "language": self._get_language_from_image(image)
            }

        finally:
            # Clean up container
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

    def _get_language_from_image(self, image: str) -> str:
        """Get language name from Docker image."""
        for lang, img in self.LANGUAGE_IMAGES.items():
            if img == image:
                return lang
        return "unknown"

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return list(self.LANGUAGE_IMAGES.keys())

    async def cleanup(self) -> None:
        """Cleanup Docker resources."""
        try:
            # Remove stopped containers
            containers = self.docker_client.containers.list(
                all=True,
                filters={"status": "exited", "network": self.settings.docker_network}
            )
            for container in containers:
                container.remove()

            logger.info("Cleaned up sandbox containers")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
