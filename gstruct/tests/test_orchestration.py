"""Tests for SYMORQ Orchestration module."""

import pytest
import asyncio
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SYMORQ.orchestration import Orchestrator
from config.settings import get_default_config


@pytest.fixture
def config():
    """Get test configuration."""
    return get_default_config()


@pytest.fixture
async def orchestrator(config):
    """Create orchestrator instance."""
    orch = Orchestrator(config)
    yield orch
    # Cleanup
    if orch.is_running:
        await orch.shutdown()


class TestOrchestrator:
    """Test Orchestrator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        await orchestrator.initialize()
        assert orchestrator.command_socket is not None
        assert orchestrator.pub_socket is not None
        assert orchestrator.sub_socket is not None
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator):
        """Test task submission."""
        await orchestrator.initialize()

        task = {
            "id": "test_task_1",
            "name": "Test Task",
            "resources": {"cpu": 2.0}
        }

        result = await orchestrator.submit_task(task)

        assert result["success"] is True
        assert result["task_id"] == "test_task_1"
        assert "test_task_1" in orchestrator.tasks

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_task(self, orchestrator):
        """Test task cancellation."""
        await orchestrator.initialize()

        # Submit a task
        task = {"id": "test_task_2", "name": "Test Task 2"}
        await orchestrator.submit_task(task)

        # Cancel the task
        result = await orchestrator.cancel_task("test_task_2")

        assert result["success"] is True
        assert orchestrator.tasks["test_task_2"]["status"] == "cancelled"

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_register_worker(self, orchestrator):
        """Test worker registration."""
        await orchestrator.initialize()

        worker_info = {
            "id": "worker_1",
            "capabilities": ["cpu", "gpu"],
            "resources": {"cpu": 8, "memory": 16}
        }

        result = await orchestrator.register_worker(worker_info)

        assert result["success"] is True
        assert "worker_1" in orchestrator.workers

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_get_status(self, orchestrator):
        """Test status retrieval."""
        await orchestrator.initialize()

        # Submit some tasks
        await orchestrator.submit_task({"id": "task1", "name": "Task 1"})
        await orchestrator.submit_task({"id": "task2", "name": "Task 2"})

        status = orchestrator.get_status()

        assert status["running"] is False  # Not started yet
        assert status["tasks"]["total"] == 2
        assert status["tasks"]["pending"] == 2

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self, orchestrator):
        """Test concurrent task submission."""
        await orchestrator.initialize()

        # Submit multiple tasks concurrently
        tasks = [
            {"id": f"task_{i}", "name": f"Task {i}"}
            for i in range(10)
        ]

        results = await asyncio.gather(*[
            orchestrator.submit_task(task)
            for task in tasks
        ])

        assert all(r["success"] for r in results)
        assert len(orchestrator.tasks) == 10

        await orchestrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
