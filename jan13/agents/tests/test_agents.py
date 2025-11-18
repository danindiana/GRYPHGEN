"""Tests for infrastructure agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import BaseAgent
from src.agents.ollama import OllamaAgent
from src.agents.nginx import NginxAgent
from src.agents.infrastructure import InfrastructureAgent
from src.models.config import (
    AgentConfig,
    OllamaConfig,
    NginxConfig,
    DeploymentConfig,
    ServiceStatus,
)


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing BaseAgent."""

    async def deploy(self) -> bool:
        self.status = ServiceStatus.RUNNING
        return True

    async def health_check(self):
        from src.models.config import HealthStatus

        return HealthStatus(service="test", status=ServiceStatus.RUNNING)

    async def stop(self) -> bool:
        self.status = ServiceStatus.STOPPED
        return True


@pytest.mark.unit
class TestBaseAgent:
    """Test BaseAgent functionality."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration."""
        return AgentConfig(name="test-agent")

    @pytest.fixture
    def agent(self, agent_config):
        """Create test agent instance."""
        return ConcreteAgent(agent_config)

    def test_agent_initialization(self, agent, agent_config):
        """Test agent initialization."""
        assert agent.config == agent_config
        assert agent.status == ServiceStatus.UNKNOWN
        assert agent._error_count == 0

    @pytest.mark.asyncio
    async def test_deploy(self, agent):
        """Test deploy method."""
        success = await agent.deploy()
        assert success is True
        assert agent.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test health check."""
        health = await agent.health_check()
        assert health.service == "test"
        assert health.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop(self, agent):
        """Test stop method."""
        success = await agent.stop()
        assert success is True
        assert agent.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_retry_operation_success(self, agent):
        """Test successful retry operation."""

        async def successful_operation():
            return "success"

        result = await agent.retry_operation(successful_operation)
        assert result == "success"
        assert agent._error_count == 0

    @pytest.mark.asyncio
    async def test_retry_operation_failure(self, agent):
        """Test retry operation with failures."""

        async def failing_operation():
            raise Exception("Test error")

        result = await agent.retry_operation(failing_operation)
        assert result is None
        assert agent._error_count == agent.config.max_retries

    def test_get_metrics(self, agent):
        """Test metrics retrieval."""
        metrics = agent.get_metrics()
        assert "name" in metrics
        assert "status" in metrics
        assert "error_count" in metrics
        assert metrics["name"] == "test-agent"


@pytest.mark.unit
class TestOllamaAgent:
    """Test OllamaAgent functionality."""

    @pytest.fixture
    def ollama_config(self):
        """Create Ollama configuration."""
        return OllamaConfig(port=11435, models=["llama2"])

    @pytest.fixture
    def agent(self, ollama_config):
        """Create Ollama agent instance."""
        agent_config = AgentConfig(name="ollama-test")
        return OllamaAgent(agent_config, ollama_config)

    def test_initialization(self, agent, ollama_config):
        """Test Ollama agent initialization."""
        assert agent.ollama_config == ollama_config
        assert agent.ollama_config.port == 11435

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_check_port_available(self, mock_run, agent):
        """Test port availability check."""
        mock_run.return_value = MagicMock(returncode=1)
        result = await agent._check_port_available(11435)
        assert result is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_health_check_running(self, mock_get, agent):
        """Test health check when Ollama is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        health = await agent.health_check()
        assert health.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_health_check_failed(self, mock_get, agent):
        """Test health check when Ollama is not running."""
        mock_get.side_effect = Exception("Connection refused")

        health = await agent.health_check()
        assert health.status == ServiceStatus.FAILED


@pytest.mark.unit
class TestNginxAgent:
    """Test NginxAgent functionality."""

    @pytest.fixture
    def nginx_config(self):
        """Create Nginx configuration."""
        return NginxConfig(port=11434, upstream_port=11435)

    @pytest.fixture
    def agent(self, nginx_config):
        """Create Nginx agent instance."""
        agent_config = AgentConfig(name="nginx-test")
        return NginxAgent(agent_config, nginx_config)

    def test_initialization(self, agent, nginx_config):
        """Test Nginx agent initialization."""
        assert agent.nginx_config == nginx_config
        assert agent.nginx_config.port == 11434
        assert agent.nginx_config.upstream_port == 11435

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_test_config_success(self, mock_run, agent):
        """Test successful configuration validation."""
        mock_run.return_value = MagicMock(returncode=0)
        result = await agent._test_config()
        assert result is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_test_config_failure(self, mock_run, agent):
        """Test failed configuration validation."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Syntax error")
        result = await agent._test_config()
        assert result is False


@pytest.mark.unit
class TestInfrastructureAgent:
    """Test InfrastructureAgent functionality."""

    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration."""
        return DeploymentConfig()

    @pytest.fixture
    def agent(self, deployment_config):
        """Create infrastructure agent instance."""
        return InfrastructureAgent(deployment_config)

    def test_initialization(self, agent):
        """Test infrastructure agent initialization."""
        assert agent.deployment_config is not None
        assert agent.ollama_agent is not None
        assert agent.nginx_agent is not None

    def test_configuration_validation(self, deployment_config):
        """Test configuration validation."""
        # Valid configuration
        errors = deployment_config.validate_ports()
        assert len(errors) == 0

        # Invalid configuration (port conflict)
        deployment_config.ollama.port = 11434
        deployment_config.nginx.port = 11434
        errors = deployment_config.validate_ports()
        assert len(errors) > 0

    @pytest.mark.asyncio
    @patch.object(OllamaAgent, "deploy")
    @patch.object(NginxAgent, "deploy")
    @patch.object(InfrastructureAgent, "_verify_deployment")
    async def test_deploy_success(self, mock_verify, mock_nginx_deploy, mock_ollama_deploy, agent):
        """Test successful infrastructure deployment."""
        mock_ollama_deploy.return_value = True
        mock_nginx_deploy.return_value = True
        mock_verify.return_value = True

        success = await agent.deploy()
        assert success is True
        assert agent.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    @patch.object(OllamaAgent, "deploy")
    async def test_deploy_ollama_failure(self, mock_ollama_deploy, agent):
        """Test deployment failure when Ollama fails."""
        mock_ollama_deploy.return_value = False

        success = await agent.deploy()
        assert success is False
        assert agent.status == ServiceStatus.FAILED

    @pytest.mark.asyncio
    @patch.object(OllamaAgent, "health_check")
    @patch.object(NginxAgent, "health_check")
    async def test_health_check(self, mock_nginx_health, mock_ollama_health, agent):
        """Test infrastructure health check."""
        from src.models.config import HealthStatus

        mock_ollama_health.return_value = HealthStatus(
            service="ollama", status=ServiceStatus.RUNNING
        )
        mock_nginx_health.return_value = HealthStatus(
            service="nginx", status=ServiceStatus.RUNNING
        )

        health = await agent.health_check()
        assert health.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    @patch.object(OllamaAgent, "stop")
    @patch.object(NginxAgent, "stop")
    async def test_stop(self, mock_nginx_stop, mock_ollama_stop, agent):
        """Test stopping infrastructure."""
        mock_ollama_stop.return_value = True
        mock_nginx_stop.return_value = True

        success = await agent.stop()
        assert success is True
