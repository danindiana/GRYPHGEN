"""Unit tests for API endpoints."""

import pytest
from fastapi import status


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["service"] == "GRYPHGEN Agentic API Gateway"
    assert data["version"] == "0.1.0"
    assert data["status"] == "operational"


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"


def test_readiness_check(client):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "ready"


def test_openapi_docs(client):
    """Test OpenAPI documentation endpoint."""
    response = client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["info"]["title"] == "GRYPHGEN Agentic API Gateway"
    assert data["info"]["version"] == "0.1.0"


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
