"""Test API endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(api_client: TestClient) -> None:
    """Test root endpoint."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "CALISOTA AI Engine"
    assert data["version"] == "1.0.0"


def test_health_endpoint(api_client: TestClient) -> None:
    """Test health check endpoint."""
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_supported_languages(api_client: TestClient) -> None:
    """Test supported languages endpoint."""
    response = api_client.get("/api/tasks/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert "python" in data["languages"]
