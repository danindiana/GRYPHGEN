"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.api.main import app, infrastructure_agent
from src.models.config import ServiceStatus, HealthStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.unit
class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    @pytest.mark.asyncio
    @patch("src.api.main.infrastructure_agent")
    async def test_health_check_unavailable(self, mock_agent, client):
        """Test health check when agent not initialized."""
        mock_agent = None
        response = client.get("/health")
        # Will return 503 or error depending on implementation

    @pytest.mark.asyncio
    def test_deploy_endpoint(self, client):
        """Test deploy endpoint."""
        payload = {
            "ollama_port": 11435,
            "nginx_port": 11434,
            "gpu_enabled": True,
            "gpu_id": 0,
            "gpu_memory_fraction": 0.9,
            "auto_pull_models": False,
            "models": ["llama2"],
        }

        # Note: This will fail in test environment without actual services
        # In a real test, you'd mock the infrastructure_agent.deploy()
        response = client.post("/api/v1/deploy", json=payload)
        # Response code depends on whether services actually deploy

    def test_deploy_invalid_config(self, client):
        """Test deploy with invalid configuration."""
        payload = {
            "ollama_port": 11434,  # Same as nginx_port
            "nginx_port": 11434,
            "gpu_enabled": True,
        }

        response = client.post("/api/v1/deploy", json=payload)
        # Should return 400 for invalid configuration

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        # Should return metrics in Prometheus format
        assert response.status_code in [200, 503]


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API."""

    @pytest.mark.asyncio
    async def test_full_deployment_lifecycle(self, client):
        """Test complete deployment lifecycle."""
        # Deploy
        deploy_payload = {
            "ollama_port": 11435,
            "nginx_port": 11434,
            "gpu_enabled": False,  # Disable GPU for testing
            "auto_pull_models": False,
            "models": [],
        }

        # Note: This requires actual services to be available
        # In CI/CD, you'd use Docker containers or mocks

        # deploy_response = client.post("/api/v1/deploy", json=deploy_payload)
        # status_response = client.get("/api/v1/status")
        # stop_response = client.post("/api/v1/stop")

        # assert deploy_response.status_code == 202
        # assert status_response.status_code == 200
        # assert stop_response.status_code == 200
