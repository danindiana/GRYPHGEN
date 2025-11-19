"""Integration tests for services."""

import pytest
from fastapi import status


@pytest.mark.integration
def test_code_generation_flow(client):
    """Test complete code generation flow."""
    # Generate code
    request_data = {
        "prompt": "Create a function to reverse a string",
        "language": "python",
        "include_tests": True,
    }

    response = client.post("/api/v1/code/generate", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["code"] is not None
    assert data["tests"] is not None


@pytest.mark.integration
def test_testing_service_flow(client):
    """Test testing service flow."""
    # Generate tests
    request_data = {
        "code": "def add(a, b): return a + b",
        "language": "python",
        "framework": "pytest",
    }

    response = client.post("/api/v1/test/generate", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "tests" in data
    assert data["test_count"] > 0


@pytest.mark.integration
def test_documentation_generation_flow(client):
    """Test documentation generation flow."""
    request_data = {
        "code": "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)",
        "language": "python",
        "format": "markdown",
    }

    response = client.post("/api/v1/docs/generate", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "documentation" in data
    assert len(data["sections"]) > 0
