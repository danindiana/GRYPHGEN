"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """Mock GPU availability."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    yield


@pytest.fixture
def mock_gpu_not_available(monkeypatch):
    """Mock GPU not available."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
