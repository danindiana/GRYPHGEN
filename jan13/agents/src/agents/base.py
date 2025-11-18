"""Base agent class for all infrastructure agents."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from src.models.config import AgentConfig, HealthStatus, ServiceStatus


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: AgentConfig):
        """Initialize base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.logger.setLevel(config.log_level)

        self._status = ServiceStatus.UNKNOWN
        self._last_health_check: Optional[datetime] = None
        self._error_count = 0

    @property
    def status(self) -> ServiceStatus:
        """Get current agent status."""
        return self._status

    @status.setter
    def status(self, value: ServiceStatus) -> None:
        """Set agent status."""
        old_status = self._status
        self._status = value
        if old_status != value:
            self.logger.info(f"Status changed: {old_status} -> {value}")

    @abstractmethod
    async def deploy(self) -> bool:
        """Deploy the agent's managed service.

        Returns:
            True if deployment successful, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Perform health check on managed service.

        Returns:
            HealthStatus object with current status
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """Stop the agent's managed service.

        Returns:
            True if stopped successfully, False otherwise
        """
        pass

    async def retry_operation(
        self, operation: callable, *args: Any, **kwargs: Any
    ) -> Optional[Any]:
        """Retry an operation with exponential backoff.

        Args:
            operation: Async function to retry
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation if successful, None otherwise
        """
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{self.config.max_retries} for {operation.__name__}")
                result = await operation(*args, **kwargs)
                self._error_count = 0  # Reset error count on success
                return result

            except Exception as e:
                self._error_count += 1
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"Max retries ({self.config.max_retries}) exceeded for {operation.__name__}")

        return None

    async def recover(self) -> bool:
        """Attempt to recover from failure state.

        Returns:
            True if recovery successful, False otherwise
        """
        if not self.config.auto_recover:
            self.logger.info("Auto-recovery disabled")
            return False

        self.logger.info("Attempting recovery...")
        self.status = ServiceStatus.RECOVERING

        try:
            # Stop current instance
            await self.stop()
            await asyncio.sleep(2)

            # Redeploy
            success = await self.deploy()

            if success:
                self.logger.info("Recovery successful")
                return True
            else:
                self.logger.error("Recovery failed")
                self.status = ServiceStatus.FAILED
                return False

        except Exception as e:
            self.logger.error(f"Recovery error: {e}")
            self.status = ServiceStatus.FAILED
            return False

    async def monitor(self, interval: Optional[int] = None) -> None:
        """Continuously monitor service health.

        Args:
            interval: Health check interval in seconds (uses config default if None)
        """
        check_interval = interval or self.config.health_check_interval

        self.logger.info(f"Starting health monitoring (interval: {check_interval}s)")

        while True:
            try:
                health = await self.health_check()
                self._last_health_check = datetime.utcnow()

                if health.status == ServiceStatus.FAILED:
                    self.logger.error(f"Health check failed: {health.error_message}")

                    if self.config.auto_recover:
                        self.logger.info("Initiating automatic recovery")
                        await self.recover()

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                self.logger.info("Monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(check_interval)

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "name": self.config.name,
            "status": self.status.value,
            "error_count": self._error_count,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "auto_recover_enabled": self.config.auto_recover,
        }
