"""
SYMORQ Resource Management Module

Manages grid computing resources including allocation, deallocation, and monitoring.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gpu_utils import get_gpu_info, check_gpu_availability


class ResourceType(Enum):
    """Resource types in the grid."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


class ResourceStatus(Enum):
    """Resource allocation status."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    UNAVAILABLE = "unavailable"


@dataclass
class Resource:
    """Represents a grid resource."""
    id: str
    type: ResourceType
    capacity: float
    allocated: float
    reserved: float
    status: ResourceStatus
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    @property
    def available(self) -> float:
        """Get available capacity."""
        return self.capacity - self.allocated - self.reserved

    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return ((self.allocated + self.reserved) / self.capacity) * 100


class ResourceManager:
    """Manages grid computing resources."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize resource manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.resources: Dict[str, Resource] = {}
        self.allocations: Dict[str, Dict[str, float]] = {}  # task_id -> {resource_id: amount}
        self.is_running = False

    async def initialize(self):
        """Initialize resource manager."""
        logger.info("Initializing Resource Manager...")

        # Discover and register system resources
        await self._discover_resources()

        logger.info(f"✓ Resource Manager initialized with {len(self.resources)} resources")

    async def _discover_resources(self):
        """Discover available system resources."""
        now = datetime.now().isoformat()

        # CPU resources
        cpu_count = psutil.cpu_count(logical=True)
        self.resources["cpu"] = Resource(
            id="cpu",
            type=ResourceType.CPU,
            capacity=float(cpu_count),
            allocated=0.0,
            reserved=0.0,
            status=ResourceStatus.AVAILABLE,
            metadata={
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": cpu_count,
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            },
            created_at=now,
            updated_at=now
        )

        # Memory resources
        memory = psutil.virtual_memory()
        self.resources["memory"] = Resource(
            id="memory",
            type=ResourceType.MEMORY,
            capacity=float(memory.total),
            allocated=0.0,
            reserved=0.0,
            status=ResourceStatus.AVAILABLE,
            metadata={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3)
            },
            created_at=now,
            updated_at=now
        )

        # GPU resources
        if check_gpu_availability():
            gpu_info = get_gpu_info(0)
            self.resources["gpu_0"] = Resource(
                id="gpu_0",
                type=ResourceType.GPU,
                capacity=float(gpu_info.get("total_memory", 0)),
                allocated=0.0,
                reserved=0.0,
                status=ResourceStatus.AVAILABLE,
                metadata=gpu_info,
                created_at=now,
                updated_at=now
            )

        # Disk resources
        disk = psutil.disk_usage('/')
        self.resources["disk"] = Resource(
            id="disk",
            type=ResourceType.DISK,
            capacity=float(disk.total),
            allocated=0.0,
            reserved=0.0,
            status=ResourceStatus.AVAILABLE,
            metadata={
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "mount_point": "/"
            },
            created_at=now,
            updated_at=now
        )

    async def allocate_resources(
        self,
        task_id: str,
        requirements: Dict[str, float]
    ) -> Dict[str, Any]:
        """Allocate resources for a task.

        Args:
            task_id: Task identifier
            requirements: Resource requirements {resource_type: amount}

        Returns:
            Allocation result
        """
        logger.info(f"Allocating resources for task {task_id}: {requirements}")

        # Check if resources are available
        for resource_id, amount in requirements.items():
            if resource_id not in self.resources:
                return {
                    "success": False,
                    "error": f"Resource {resource_id} not found"
                }

            resource = self.resources[resource_id]
            if resource.available < amount:
                return {
                    "success": False,
                    "error": f"Insufficient {resource_id}: requested {amount}, available {resource.available}"
                }

        # Allocate resources
        allocated = {}
        for resource_id, amount in requirements.items():
            resource = self.resources[resource_id]
            resource.allocated += amount
            resource.updated_at = datetime.now().isoformat()
            allocated[resource_id] = amount

        self.allocations[task_id] = allocated

        logger.info(f"✓ Resources allocated for task {task_id}")

        return {
            "success": True,
            "task_id": task_id,
            "allocated": allocated
        }

    async def deallocate_resources(self, task_id: str) -> Dict[str, Any]:
        """Deallocate resources for a task.

        Args:
            task_id: Task identifier

        Returns:
            Deallocation result
        """
        if task_id not in self.allocations:
            return {
                "success": False,
                "error": f"No allocations found for task {task_id}"
            }

        logger.info(f"Deallocating resources for task {task_id}")

        # Deallocate resources
        allocated = self.allocations.pop(task_id)
        for resource_id, amount in allocated.items():
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.allocated = max(0, resource.allocated - amount)
                resource.updated_at = datetime.now().isoformat()

        logger.info(f"✓ Resources deallocated for task {task_id}")

        return {
            "success": True,
            "task_id": task_id,
            "deallocated": allocated
        }

    async def reserve_resources(
        self,
        task_id: str,
        requirements: Dict[str, float],
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """Reserve resources for future allocation.

        Args:
            task_id: Task identifier
            requirements: Resource requirements
            duration_minutes: Reservation duration in minutes

        Returns:
            Reservation result
        """
        logger.info(f"Reserving resources for task {task_id}: {requirements}")

        # Check availability
        for resource_id, amount in requirements.items():
            if resource_id not in self.resources:
                return {
                    "success": False,
                    "error": f"Resource {resource_id} not found"
                }

            resource = self.resources[resource_id]
            if resource.available < amount:
                return {
                    "success": False,
                    "error": f"Insufficient {resource_id} for reservation"
                }

        # Reserve resources
        for resource_id, amount in requirements.items():
            resource = self.resources[resource_id]
            resource.reserved += amount
            resource.updated_at = datetime.now().isoformat()

        logger.info(f"✓ Resources reserved for task {task_id}")

        return {
            "success": True,
            "task_id": task_id,
            "reserved": requirements,
            "expires_at": (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
        }

    async def get_resource_status(self) -> Dict[str, Any]:
        """Get status of all resources.

        Returns:
            Resource status dictionary
        """
        return {
            "resources": {
                resource_id: {
                    "type": resource.type.value,
                    "capacity": resource.capacity,
                    "allocated": resource.allocated,
                    "reserved": resource.reserved,
                    "available": resource.available,
                    "utilization": resource.utilization,
                    "status": resource.status.value,
                }
                for resource_id, resource in self.resources.items()
            },
            "allocations": {
                task_id: allocated
                for task_id, allocated in self.allocations.items()
            },
            "timestamp": datetime.now().isoformat()
        }

    async def update_resource_metrics(self):
        """Update resource metrics from system."""
        # Update CPU usage
        if "cpu" in self.resources:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.resources["cpu"].metadata["current_usage"] = cpu_percent

        # Update memory usage
        if "memory" in self.resources:
            memory = psutil.virtual_memory()
            self.resources["memory"].metadata["current_usage"] = memory.percent
            self.resources["memory"].metadata["available_gb"] = memory.available / (1024**3)

        # Update GPU metrics
        if "gpu_0" in self.resources and check_gpu_availability():
            gpu_info = get_gpu_info(0)
            self.resources["gpu_0"].metadata.update(gpu_info)

        # Update disk usage
        if "disk" in self.resources:
            disk = psutil.disk_usage('/')
            self.resources["disk"].metadata["free_gb"] = disk.free / (1024**3)
            self.resources["disk"].metadata["current_usage"] = disk.percent

    async def start(self):
        """Start resource monitoring."""
        self.is_running = True
        logger.info("Starting Resource Manager monitoring...")

        while self.is_running:
            try:
                await self.update_resource_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating resource metrics: {e}")
                await asyncio.sleep(10)

    async def shutdown(self):
        """Shutdown resource manager."""
        logger.info("Shutting down Resource Manager...")
        self.is_running = False
        logger.info("✓ Resource Manager shutdown complete")


def main():
    """Main entry point."""
    from config.settings import load_config

    async def run():
        config = load_config()
        manager = ResourceManager(config)

        try:
            await manager.initialize()
            status = await manager.get_resource_status()
            print(f"\nResource Status:\n{status}\n")

            # Example allocation
            result = await manager.allocate_resources(
                "test_task",
                {"cpu": 2.0, "memory": 4 * 1024**3}  # 2 CPUs, 4GB memory
            )
            print(f"Allocation result: {result}\n")

            status = await manager.get_resource_status()
            print(f"\nResource Status After Allocation:\n{status}\n")

            await manager.deallocate_resources("test_task")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await manager.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
