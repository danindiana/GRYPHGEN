"""Unit tests for agent service router and endpoint."""

import hashlib
import os
import pytest
from fastapi import status
from unittest.mock import AsyncMock, patch

from src.agent.runner import AgentResult, StepTrace


@pytest.fixture
def valid_api_key():
    """Generate a valid test API key and set env var."""
    test_key = "grug_test_key_12345678901234567890"
    key_hash = hashlib.sha256(test_key.encode()).hexdigest()
    
    # Monkeypatch the env var for this test
    old_keys = os.environ.get("GRYPHGEN_API_KEYS")
    os.environ["GRYPHGEN_API_KEYS"] = key_hash
    
    yield test_key
    
    # Restore old value
    if old_keys is not None:
        os.environ["GRYPHGEN_API_KEYS"] = old_keys
    else:
        os.environ.pop("GRYPHGEN_API_KEYS", None)


class TestAgentRunEndpoint:
    """Test /api/v1/agent/run endpoint."""

    def test_agent_run_basic_task(self, client, valid_api_key):
        """Test agent run endpoint with basic task."""
        request_data = {
            "task": "Create a simple Python function",
        }
        headers = {"X-API-Key": valid_api_key}
        
        with patch('src.services.agent.router._runner.run') as mock_run:
            mock_run.return_value = AgentResult(
                output="Function created successfully",
                trace=[],
                steps_taken=1,
                total_time_s=0.5,
                model="test-model",
            )
            response = client.post(
                "/api/v1/agent/run",
                json=request_data,
                headers=headers,
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["output"] == "Function created successfully"
        assert data["steps_taken"] == 1
        assert data["model"] == "test-model"
        assert data["truncated"] is False

    def test_agent_run_with_files(self, client, valid_api_key):
        """Test agent run with pre-populated files."""
        request_data = {
            "task": "Modify the test file",
            "files": {
                "test.py": "def hello(): pass",
                "README.md": "# Test",
            },
        }
        headers = {"X-API-Key": valid_api_key}
        
        with patch('src.services.agent.router._runner.run') as mock_run:
            mock_run.return_value = AgentResult(
                output="Files modified",
                trace=[
                    StepTrace(
                        step=1,
                        tool="read_file",
                        args={"path": "test.py"},
                        result="def hello(): pass",
                        elapsed_s=0.1,
                    ),
                ],
                steps_taken=2,
                total_time_s=1.0,
                model="test-model",
            )
            response = client.post(
                "/api/v1/agent/run",
                json=request_data,
                headers=headers,
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["output"] == "Files modified"
        assert len(data["trace"]) == 1
        assert data["trace"][0]["tool"] == "read_file"



    def test_agent_run_missing_api_key(self, client):
        """Test that request without API key is rejected."""
        request_data = {
            "task": "Some task",
        }
        
        # No valid key configured, request should fail
        response = client.post(
            "/api/v1/agent/run",
            json=request_data,
        )
        
        # Should be 401 (missing key) or 503 (no keys configured)
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_403_FORBIDDEN,
        ]

    def test_agent_run_invalid_max_steps(self, client, valid_api_key):
        """Test that invalid max_steps are rejected."""
        request_data = {
            "task": "Task",
            "max_steps": 100,  # Exceeds le=50
        }
        headers = {"X-API-Key": valid_api_key}
        
        response = client.post(
            "/api/v1/agent/run",
            json=request_data,
            headers=headers,
        )
        
        # Should be validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_agent_run_missing_task(self, client, valid_api_key):
        """Test that request without task is rejected."""
        request_data = {
            # Missing "task" field
        }
        headers = {"X-API-Key": valid_api_key}
        
        response = client.post(
            "/api/v1/agent/run",
            json=request_data,
            headers=headers,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_agent_run_error_handling(self, client, valid_api_key):
        """Test error handling when agent fails."""
        request_data = {"task": "Failing task"}
        headers = {"X-API-Key": valid_api_key}
        
        with patch('src.services.agent.router._runner.run') as mock_run:
            mock_run.side_effect = Exception("LLM connection failed")
            response = client.post(
                "/api/v1/agent/run",
                json=request_data,
                headers=headers,
            )
        
        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        data = response.json()
        assert "detail" in data


class TestAgentRunResponse:
    """Test response model structure."""

    def test_response_includes_trace(self, client, valid_api_key):
        """Response includes full trace."""
        request_data = {"task": "Task"}
        headers = {"X-API-Key": valid_api_key}
        
        with patch('src.services.agent.router._runner.run') as mock_run:
            mock_run.return_value = AgentResult(
                output="Result",
                trace=[
                    StepTrace(step=1, tool="list_dir", args={}, result="(empty)", elapsed_s=0.05),
                    StepTrace(step=2, tool="read_file", args={"path": "test.txt"}, result="content", elapsed_s=0.1),
                ],
                steps_taken=2,
                total_time_s=0.2,
                model="test-model",
            )
            response = client.post(
                "/api/v1/agent/run",
                json=request_data,
                headers=headers,
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["trace"]) == 2
        assert data["trace"][0]["step"] == 1
        assert data["trace"][0]["tool"] == "list_dir"
        assert data["trace"][1]["tool"] == "read_file"

    def test_response_truncation_flag(self):
        """AgentResult can be marked truncated."""
        result = AgentResult(
            output="Max steps reached",
            trace=[],
            steps_taken=20,
            total_time_s=10.0,
            model="test-model",
            truncated=True,
        )
        assert result.truncated is True
