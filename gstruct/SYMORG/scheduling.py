"""
SYMORG Scheduling Module

Task scheduling and resource allocation using RAG (Resource Allocation Graph)
and optimization algorithms.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from loguru import logger


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a computational task."""
    id: str
    name: str
    priority: TaskPriority
    resources_required: Dict[str, float]
    dependencies: List[str]
    estimated_duration: float  # seconds
    state: TaskState
    scheduled_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    assigned_worker: Optional[str] = None
    metadata: Dict[str, Any] = None


class Scheduler:
    """Main scheduling component with RAG-based allocation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the scheduler.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.rag: Optional[nx.DiGraph] = None  # Resource Allocation Graph
        self.is_running = False

    async def initialize(self):
        """Initialize scheduler."""
        logger.info("Initializing Scheduler...")

        # Initialize Resource Allocation Graph
        self.rag = nx.DiGraph()

        logger.info("✓ Scheduler initialized")

    async def start(self):
        """Start the scheduler."""
        self.is_running = True
        logger.info("Starting Scheduler...")

        # Start scheduling loop
        await asyncio.gather(
            self._scheduling_loop(),
            self._monitor_tasks(),
        )

    async def shutdown(self):
        """Shutdown the scheduler."""
        logger.info("Shutting down Scheduler...")
        self.is_running = False
        logger.info("✓ Scheduler shutdown complete")

    async def submit_task(
        self,
        task_id: str,
        name: str,
        resources_required: Dict[str, float],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: List[str] = None,
        estimated_duration: float = 60.0,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Submit a task for scheduling.

        Args:
            task_id: Unique task identifier
            name: Task name
            resources_required: Required resources
            priority: Task priority
            dependencies: List of task IDs this task depends on
            estimated_duration: Estimated execution time in seconds
            metadata: Additional task metadata

        Returns:
            Submission result
        """
        if task_id in self.tasks:
            return {
                "success": False,
                "error": f"Task {task_id} already exists"
            }

        task = Task(
            id=task_id,
            name=name,
            priority=priority,
            resources_required=resources_required,
            dependencies=dependencies or [],
            estimated_duration=estimated_duration,
            state=TaskState.PENDING,
            metadata=metadata or {}
        )

        self.tasks[task_id] = task

        # Add to RAG
        self.rag.add_node(task_id, task=task)
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.rag.add_edge(dep_id, task_id)

        logger.info(f"Task {task_id} submitted with priority {priority.name}")

        return {
            "success": True,
            "task_id": task_id,
            "state": task.state.value
        }

    async def _scheduling_loop(self):
        """Main scheduling loop."""
        while self.is_running:
            try:
                # Get schedulable tasks
                schedulable = await self._get_schedulable_tasks()

                if schedulable:
                    # Sort by priority and dependencies
                    sorted_tasks = await self._prioritize_tasks(schedulable)

                    # Schedule tasks
                    for task_id in sorted_tasks[:10]:  # Schedule up to 10 tasks per iteration
                        await self._schedule_task(task_id)

                await asyncio.sleep(1)  # Schedule every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(1)

    async def _get_schedulable_tasks(self) -> List[str]:
        """Get tasks that are ready to be scheduled.

        Returns:
            List of task IDs ready for scheduling
        """
        schedulable = []

        for task_id, task in self.tasks.items():
            if task.state != TaskState.PENDING:
                continue

            # Check if all dependencies are completed
            dependencies_met = all(
                self.tasks[dep_id].state == TaskState.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )

            if dependencies_met:
                schedulable.append(task_id)

        return schedulable

    async def _prioritize_tasks(self, task_ids: List[str]) -> List[str]:
        """Prioritize tasks for scheduling.

        Args:
            task_ids: List of task IDs to prioritize

        Returns:
            Sorted list of task IDs
        """
        def priority_key(task_id: str) -> Tuple[int, float]:
            task = self.tasks[task_id]
            # Sort by priority (descending) and estimated duration (ascending)
            return (-task.priority.value, task.estimated_duration)

        return sorted(task_ids, key=priority_key)

    async def _schedule_task(self, task_id: str):
        """Schedule a specific task.

        Args:
            task_id: Task ID to schedule
        """
        task = self.tasks[task_id]

        # Here you would implement actual resource allocation logic
        # For now, just mark as scheduled
        task.state = TaskState.SCHEDULED
        task.scheduled_at = datetime.now().isoformat()

        logger.info(f"Task {task_id} scheduled")

        # Simulate assigning to a worker
        task.assigned_worker = f"worker_{hash(task_id) % 10}"

        return {
            "success": True,
            "task_id": task_id,
            "worker": task.assigned_worker
        }

    async def _monitor_tasks(self):
        """Monitor task execution."""
        while self.is_running:
            try:
                # Check scheduled tasks
                for task_id, task in self.tasks.items():
                    if task.state == TaskState.SCHEDULED:
                        # Simulate task execution
                        # In real implementation, this would check worker status
                        pass

                await asyncio.sleep(5)  # Monitor every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring tasks: {e}")
                await asyncio.sleep(5)

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task.

        Args:
            task_id: Task ID to cancel

        Returns:
            Cancellation result
        """
        if task_id not in self.tasks:
            return {
                "success": False,
                "error": f"Task {task_id} not found"
            }

        task = self.tasks[task_id]

        if task.state in [TaskState.COMPLETED, TaskState.FAILED]:
            return {
                "success": False,
                "error": f"Cannot cancel task in state {task.state.value}"
            }

        task.state = TaskState.CANCELLED
        task.completed_at = datetime.now().isoformat()

        logger.info(f"Task {task_id} cancelled")

        return {
            "success": True,
            "task_id": task_id
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.

        Args:
            task_id: Task ID

        Returns:
            Task status dictionary or None
        """
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        return {
            "id": task.id,
            "name": task.name,
            "state": task.state.value,
            "priority": task.priority.name,
            "resources_required": task.resources_required,
            "dependencies": task.dependencies,
            "estimated_duration": task.estimated_duration,
            "scheduled_at": task.scheduled_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "assigned_worker": task.assigned_worker,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get overall scheduler status.

        Returns:
            Status dictionary
        """
        state_counts = {}
        for state in TaskState:
            state_counts[state.value] = sum(
                1 for t in self.tasks.values() if t.state == state
            )

        return {
            "running": self.is_running,
            "total_tasks": len(self.tasks),
            "tasks_by_state": state_counts,
            "rag_nodes": self.rag.number_of_nodes() if self.rag else 0,
            "rag_edges": self.rag.number_of_edges() if self.rag else 0,
        }

    def generate_rag_visualization(self) -> str:
        """Generate a DOT representation of the RAG.

        Returns:
            DOT format string
        """
        if not self.rag:
            return ""

        dot_lines = ["digraph RAG {"]
        dot_lines.append("  rankdir=LR;")
        dot_lines.append("  node [shape=box];")

        # Add nodes
        for node_id in self.rag.nodes():
            task = self.tasks.get(node_id)
            if task:
                color = {
                    TaskState.PENDING: "lightgray",
                    TaskState.SCHEDULED: "lightblue",
                    TaskState.RUNNING: "yellow",
                    TaskState.COMPLETED: "lightgreen",
                    TaskState.FAILED: "red",
                    TaskState.CANCELLED: "orange",
                }.get(task.state, "white")

                dot_lines.append(
                    f'  "{node_id}" [label="{task.name}\\n{task.state.value}" fillcolor="{color}" style=filled];'
                )

        # Add edges
        for edge in self.rag.edges():
            dot_lines.append(f'  "{edge[0]}" -> "{edge[1]}";')

        dot_lines.append("}")

        return "\n".join(dot_lines)


def main():
    """Main entry point."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.settings import load_config

    async def run():
        config = load_config()
        scheduler = Scheduler(config)

        try:
            await scheduler.initialize()

            # Submit sample tasks
            await scheduler.submit_task(
                "task1",
                "Data Processing",
                {"cpu": 2.0, "memory": 4 * 1024**3},
                priority=TaskPriority.HIGH,
                estimated_duration=120.0
            )

            await scheduler.submit_task(
                "task2",
                "Model Training",
                {"cpu": 4.0, "gpu_0": 8 * 1024**3},
                priority=TaskPriority.CRITICAL,
                dependencies=["task1"],
                estimated_duration=600.0
            )

            # Print status
            print(f"\nScheduler Status:\n{scheduler.get_status()}\n")
            print(f"\nTask 1 Status:\n{scheduler.get_task_status('task1')}\n")
            print(f"\nTask 2 Status:\n{scheduler.get_task_status('task2')}\n")

            # Generate RAG visualization
            dot = scheduler.generate_rag_visualization()
            print(f"\nRAG Visualization (DOT format):\n{dot}\n")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await scheduler.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
