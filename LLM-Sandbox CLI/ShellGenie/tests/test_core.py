"""Tests for core functionality."""

import pytest

from shellgenie.core import ShellGenieCore
from shellgenie.models import (
    AppConfig,
    CommandRequest,
    ModelBackend,
    ModelConfig,
    SecurityLevel,
)


class TestShellGenieCore:
    """Test ShellGenieCore class."""

    def test_initialization(self):
        """Test core initialization."""
        core = ShellGenieCore()

        assert core.model_config is not None
        assert core.app_config is not None
        assert core.security is not None
        assert core.system_info is not None
        assert isinstance(core.history, list)

    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        model_config = ModelConfig(
            backend=ModelBackend.OLLAMA,
            model_name="test_model",
            temperature=0.5,
        )
        app_config = AppConfig(
            security_level=SecurityLevel.STRICT,
            auto_execute=True,
        )

        core = ShellGenieCore(model_config=model_config, app_config=app_config)

        assert core.model_config.model_name == "test_model"
        assert core.model_config.temperature == 0.5
        assert core.app_config.security_level == SecurityLevel.STRICT
        assert core.app_config.auto_execute is True

    def test_clean_command(self):
        """Test command cleaning."""
        core = ShellGenieCore()

        # Test various command formats
        assert core._clean_command("$ ls -la") == "ls -la"
        assert core._clean_command("```bash\nls -la\n```") == "ls -la"
        assert core._clean_command("  ls -la  ") == "ls -la"
        assert core._clean_command("bash\nls -la") == "ls -la"

    def test_execute_command(self):
        """Test command execution."""
        core = ShellGenieCore()

        result = core.execute("echo 'test'")

        assert result.success is True
        assert result.return_code == 0
        assert "test" in result.stdout

    def test_execute_with_timeout(self):
        """Test command execution with timeout."""
        core = ShellGenieCore()

        result = core.execute("sleep 10", timeout=1)

        assert result.success is False
        assert "timeout" in result.stderr.lower()

    def test_get_stats(self):
        """Test getting statistics."""
        core = ShellGenieCore()

        stats = core.get_stats()

        assert isinstance(stats, dict)
        assert "commands_in_history" in stats
        assert "security_level" in stats
        assert "model_backend" in stats
        assert "model_name" in stats
        assert "gpu_available" in stats

    def test_set_security_level(self):
        """Test changing security level."""
        core = ShellGenieCore()

        core.set_security_level(SecurityLevel.STRICT)

        assert core.app_config.security_level == SecurityLevel.STRICT
        assert core.security.security_level == SecurityLevel.STRICT

    def test_clear_history(self, tmp_path):
        """Test clearing command history."""
        history_file = tmp_path / "history.txt"
        app_config = AppConfig(history_file=str(history_file))

        core = ShellGenieCore(app_config=app_config)

        # Add some commands
        core.execute("echo 'test1'")
        core.execute("echo 'test2'")

        assert len(core.history) > 0

        # Clear history
        core.clear_history()

        assert len(core.history) == 0


@pytest.mark.slow
@pytest.mark.integration
class TestShellGenieIntegration:
    """Integration tests requiring actual LLM backend."""

    @pytest.mark.skip(reason="Requires running Ollama instance")
    async def test_generate_command_ollama(self):
        """Test command generation with Ollama."""
        model_config = ModelConfig(
            backend=ModelBackend.OLLAMA,
            model_name="llama3.2",
        )
        core = ShellGenieCore(model_config=model_config)

        request = CommandRequest(prompt="list all files in current directory")
        response = await core.generate_command(request)

        assert response.command
        assert "ls" in response.command.lower()

    @pytest.mark.skip(reason="Requires llama.cpp model file")
    async def test_generate_command_llama_cpp(self):
        """Test command generation with llama.cpp."""
        model_config = ModelConfig(
            backend=ModelBackend.LLAMA_CPP,
            model_path="/path/to/model.gguf",
        )
        core = ShellGenieCore(model_config=model_config)

        request = CommandRequest(prompt="show disk usage")
        response = await core.generate_command(request)

        assert response.command
        assert "df" in response.command.lower() or "du" in response.command.lower()
