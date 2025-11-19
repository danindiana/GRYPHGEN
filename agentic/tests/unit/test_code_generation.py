"""Unit tests for code generation service."""

import pytest
from fastapi import status


def test_generate_code_endpoint(client):
    """Test code generation endpoint."""
    request_data = {
        "prompt": "Create a Python function to calculate factorial",
        "language": "python",
        "include_tests": True,
        "include_docs": True,
    }

    response = client.post("/api/v1/code/generate", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "request_id" in data
    assert "code" in data
    assert "tests" in data
    assert "documentation" in data
    assert data["language"] == "python"


def test_list_models(client):
    """Test listing available models."""
    response = client.get("/api/v1/code/models")
    assert response.status_code == status.HTTP_200_OK

    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "gpt-4-turbo-preview" in models


def test_list_languages(client):
    """Test listing supported languages."""
    response = client.get("/api/v1/code/languages")
    assert response.status_code == status.HTTP_200_OK

    languages = response.json()
    assert isinstance(languages, list)
    assert "python" in languages
    assert "javascript" in languages
