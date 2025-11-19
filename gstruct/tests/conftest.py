"""Pytest configuration and fixtures."""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability."""
    from utils import gpu_utils
    monkeypatch.setattr(gpu_utils, "check_gpu_availability", lambda: True)
    monkeypatch.setattr(gpu_utils, "get_gpu_count", lambda: 1)
