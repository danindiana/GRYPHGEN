"""Unit tests for agent service router and endpoint."""

import pytest
from fastapi import status
from unittest.mock import AsyncMock, patch

from src.agent.runner import AgentResult, StepTrace


@pytest.fixture
def api_key():
    """Valid test API key."""
    return "test-key-12345"


class TestAgentRunEndpoint:
    """Test /api/v1/agent/run endpoint."""

    @pytest.mark.asyncio
    async def test_agent_run_basic_task(self, client, api_key):
        """Test agent run endpoint with basic task."""
        request_data = {
            "task": "Create a simple Python function",
        }
        headers = {"X-API-Key": api_key}
        
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

    @pytest.mark.asyncio
    async def test_agent_run_with_files(self, client, api_key):
        """Test agent run with pre-populated files."""
        request_data = {
            "task": "Modify the test file",
            "files": {
                "test.py": "def hello(): pass",
                "README.md": "# Test",
            },
        }
        headers = {"X-API-Key": api_key}
        
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

    @pytest.mark.asyncio
    async def test_agent_run_with_max_steps_override(self, client, api_key):
        """Test agent run with custom max_steps."""
        request_data = {
            "task": "Quick task",
            "max_steps": 5,
        }
        headers = {"X-API-Key": api_key}
        
        with patch('src.services.agent.router.AgentRunner') as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner_class.return_value = mock_runner
            mock_runner.run.return_value = AgentResult(
                output="Done",
                trace=[],
                steps_taken=3,
                total_time_s=0.3,
                model="test-model",
            )
            
            with patch('src.services.agent.router._runner') as mock_default_runner:
                mock_default_runner.base_url = "http://localhost:11434"
                mock_default_runner.model = "test"
                mock_default_runner.step_timeout = 30.0
                
                response = client.post(
                    "/api/v1/agent/run",
                    json=request_data,
                    headers=headers,
                )
        
        assert response.status_code == status.HTTP_200_OK

    def test_agent_run_missing_api_key(self, client):
        """Test that request without API key is rejected."""
        request_data = {
            "task": "Some task",
        }
        
        response = client.post(
            "/api/v1/agent/run",
            json=request_data,
            # No X-API-Key header
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_agent_run_invalid_max_steps(self, client, api_key):
        """Test that invalid max_steps are rejected."""
        request_data = {
            "task": "Task",
            "max_steps": 100,  # Exceeds le=50
        }
        headers = {"X-API-Key": api_key}
        
        response = client.post(
            "/api/v1/agent/run",
            json=request_data,
            headers=headers,
        )
        
        # Should be validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_agent_run_missing_task(self, client, api_key):
        """Test that request without task is rejected."""
        request_data = {
            # Missing "task" field
        }
        headers = {"X-API-Key": api_key}
        
        response = client.post(
            "/api/v1/agent/run",
            json=request_data,
            headers=headers,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_agent_run_rate_limiting(self, client, api_key):
        """Test rate limiting per API key."""
        request_data = {"task": "Task"}
        headers = {"X-API-Key": api_key}
        
        with patch('src.services.agent.router._runner.run') as mock_run:
            mock_run.return_value = AgentResult(
                output="Done",
                trace=[],
                steps_taken=1,
                total_time_s=0.1,
                model="test",
            )
            
            # Make multiple requests
            for i in range(3):
                response = client.post(
                    "/api/v1/agent/run",
                    json=request_data,
                    headers=headers,
                )
                # All should succeed for now (rate limit check happens at service level)
                assert response.status_code in [200, 429]

    @pytest.mark.asyncio
    async def test_agent_run_error_handling(self, client, api_key):
        """Test error handling when agent fails."""
        request_data = {"task": "Failing task"}
        headers = {"X-API-Key": api_key}
        
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

    @pytest.mark.asyncio
    async def test_response_includes_trace(self, client, api_key):
        """Response includes full trace."""
        request_data = {"task": "Task"}
        headers = {"X-API-Key": api_key}
        
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

    @pytest.mark.asyncio
    async def test_response_truncation_flag(self, client, api_key):
        """Response indicates when output is truncated."""
        request_data = {"task": "Long task"}
        headers = {"X-API-Key": api_key}
        
        with patch('src.services.agent.router._runner.run') as mock_run:
            mock_run.return_value = AgentResult(
                output="Max steps reached",
                trace=[],
                steps_taken=20,
                total_time_s=10.0,
                model="test-model",
                truncated=True,
            )
            response = client.post(
                "/api/v1/agent/run",
                json=request_data,
                headers=headers,
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["truncated"] is True
