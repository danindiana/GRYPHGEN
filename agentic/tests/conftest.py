"""Pytest configuration and fixtures."""

import hashlib

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

TEST_API_KEY = "gryphgen-test-tandem-2026"
_TEST_KEY_HASH = hashlib.sha256(TEST_API_KEY.encode()).hexdigest()


@pytest.fixture(autouse=True)
def set_test_api_key(monkeypatch):
    """Inject the test API key hash so auth-required endpoints pass."""
    monkeypatch.setenv("GRYPHGEN_API_KEYS", _TEST_KEY_HASH)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def authed_client(client):
    """TestClient with X-API-Key header pre-set."""
    client.headers.update({"X-API-Key": TEST_API_KEY})
    return client


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
