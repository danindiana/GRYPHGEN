"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from src.calisota.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        debug=True,
        use_gpu_index=False,  # Use CPU for tests
        openai_api_key="test-key",
        redis_url="redis://localhost:6379/15",  # Use separate DB for tests
    )


@pytest.fixture
def api_client(test_settings: Settings) -> TestClient:
    """Create test API client."""
    from src.calisota.api.main import app

    return TestClient(app)
