"""System and GPU monitoring utilities."""

import psutil
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import nvidia_smi

    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
from src.models.config import MetricsSnapshot


class GPUMonitor:
    """Monitor NVIDIA GPU metrics."""

    def __init__(self, gpu_id: int = 0):
        """Initialize GPU monitor.

        Args:
            gpu_id: GPU device ID to monitor
        """
        self.gpu_id = gpu_id
        self.available = NVIDIA_SMI_AVAILABLE

        if self.available:
            nvidia_smi.nvmlInit()

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get current GPU statistics.

        Returns:
            Dictionary of GPU stats or None if not available
        """
        if not self.available:
            return None

        try:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu_id)

            # Get GPU utilization
            utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)

            # Get memory info
            memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            # Get temperature
            temp = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)

            # Get power usage
            power = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

            # Get clock speeds
            graphics_clock = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_GRAPHICS)
            sm_clock = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_SM)
            memory_clock = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_MEM)

            return {
                "gpu_id": self.gpu_id,
                "utilization": utilization.gpu,
                "memory_utilization": utilization.memory,
                "memory_used": memory.used // (1024 * 1024),  # MB
                "memory_total": memory.total // (1024 * 1024),  # MB
                "memory_free": memory.free // (1024 * 1024),  # MB
                "temperature": temp,
                "power_usage": power,
                "graphics_clock": graphics_clock,
                "sm_clock": sm_clock,
                "memory_clock": memory_clock,
            }

        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None

    def __del__(self):
        """Cleanup NVML on deletion."""
        if self.available:
            try:
                nvidia_smi.nvmlShutdown()
            except:
                pass


class SystemMonitor:
    """Monitor system resources (CPU, memory, disk, network)."""

    def __init__(self, gpu_enabled: bool = True, gpu_id: int = 0):
        """Initialize system monitor.

        Args:
            gpu_enabled: Whether to monitor GPU
            gpu_id: GPU device ID if GPU monitoring enabled
        """
        self.gpu_enabled = gpu_enabled and NVIDIA_SMI_AVAILABLE
        self.gpu_monitor = GPUMonitor(gpu_id) if self.gpu_enabled else None

        # Store initial network counters for delta calculation
        self._net_io_initial = psutil.net_io_counters()

    def get_cpu_percent(self) -> float:
        """Get CPU utilization percentage."""
        return psutil.cpu_percent(interval=1)

    def get_memory_percent(self) -> float:
        """Get memory utilization percentage."""
        return psutil.virtual_memory().percent

    def get_disk_percent(self) -> float:
        """Get disk utilization percentage."""
        return psutil.disk_usage("/").percent

    def get_network_stats(self) -> Dict[str, int]:
        """Get network statistics.

        Returns:
            Dict with bytes_sent and bytes_recv
        """
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }

    async def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of all system metrics.

        Returns:
            MetricsSnapshot with current metrics
        """
        net_stats = self.get_network_stats()

        gpu_stats = None
        if self.gpu_monitor:
            gpu_stats = self.gpu_monitor.get_stats()

        return MetricsSnapshot(
            cpu_percent=self.get_cpu_percent(),
            memory_percent=self.get_memory_percent(),
            disk_percent=self.get_disk_percent(),
            network_bytes_sent=net_stats["bytes_sent"],
            network_bytes_recv=net_stats["bytes_recv"],
            gpu_utilization=gpu_stats["utilization"] if gpu_stats else None,
            gpu_memory_used=gpu_stats["memory_used"] if gpu_stats else None,
            gpu_memory_total=gpu_stats["memory_total"] if gpu_stats else None,
            gpu_temperature=gpu_stats["temperature"] if gpu_stats else None,
        )


class PrometheusMetrics:
    """Prometheus metrics collector for infrastructure agents."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics.

        Args:
            registry: Prometheus registry (creates new one if None)
        """
        self.registry = registry or CollectorRegistry()

        # Define metrics
        self.requests_total = Counter(
            "infrastructure_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.deployment_duration = Histogram(
            "infrastructure_deployment_duration_seconds",
            "Time spent deploying infrastructure",
            ["service"],
            registry=self.registry,
        )

        self.service_status = Gauge(
            "infrastructure_service_status",
            "Service status (1=running, 0=stopped, -1=failed)",
            ["service"],
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "infrastructure_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry,
        )

        self.memory_usage = Gauge(
            "infrastructure_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )

        self.gpu_usage = Gauge(
            "infrastructure_gpu_usage_percent",
            "GPU usage percentage",
            ["gpu_id"],
            registry=self.registry,
        )

        self.gpu_memory = Gauge(
            "infrastructure_gpu_memory_used_mb",
            "GPU memory used in MB",
            ["gpu_id"],
            registry=self.registry,
        )

        self.gpu_temperature = Gauge(
            "infrastructure_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["gpu_id"],
            registry=self.registry,
        )

    def record_request(self, method: str, endpoint: str, status: int) -> None:
        """Record an HTTP request.

        Args:
            method: HTTP method
            endpoint: Endpoint path
            status: HTTP status code
        """
        self.requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()

    def update_service_status(self, service: str, is_running: bool, is_failed: bool = False) -> None:
        """Update service status metric.

        Args:
            service: Service name
            is_running: Whether service is running
            is_failed: Whether service has failed
        """
        if is_failed:
            value = -1
        elif is_running:
            value = 1
        else:
            value = 0

        self.service_status.labels(service=service).set(value)

    def update_system_metrics(self, snapshot: MetricsSnapshot) -> None:
        """Update system metrics from snapshot.

        Args:
            snapshot: Metrics snapshot to update from
        """
        self.cpu_usage.set(snapshot.cpu_percent)
        self.memory_usage.set(snapshot.memory_percent)

        if snapshot.gpu_utilization is not None:
            self.gpu_usage.labels(gpu_id="0").set(snapshot.gpu_utilization)

        if snapshot.gpu_memory_used is not None:
            self.gpu_memory.labels(gpu_id="0").set(snapshot.gpu_memory_used)

        if snapshot.gpu_temperature is not None:
            self.gpu_temperature.labels(gpu_id="0").set(snapshot.gpu_temperature)
