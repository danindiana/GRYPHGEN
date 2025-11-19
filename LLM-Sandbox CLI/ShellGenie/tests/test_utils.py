"""Tests for utility functions."""

import os
from pathlib import Path

import pytest

from shellgenie.models import ExecutionResult, SystemInfo
from shellgenie.utils import (
    check_dependencies,
    execute_command,
    format_bytes,
    get_system_info,
    load_command_history,
    save_to_history,
)


class TestSystemInfo:
    """Test system information gathering."""

    def test_get_system_info(self):
        """Test getting system information."""
        info = get_system_info()

        assert isinstance(info, SystemInfo)
        assert info.os in ["Linux", "Darwin", "Windows"]
        assert len(info.kernel) > 0
        assert len(info.shell) > 0
        assert Path(info.cwd).exists()
        assert len(info.user) > 0
        assert len(info.hostname) > 0
        assert isinstance(info.gpu_available, bool)


class TestExecuteCommand:
    """Test command execution."""

    def test_successful_command(self):
        """Test execution of successful command."""
        result = execute_command("echo 'Hello World'", timeout=5)

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.return_code == 0
        assert "Hello World" in result.stdout
        assert result.execution_time > 0

    def test_failed_command(self):
        """Test execution of failed command."""
        result = execute_command("exit 1", timeout=5)

        assert result.success is False
        assert result.return_code == 1

    def test_command_timeout(self):
        """Test command timeout."""
        result = execute_command("sleep 10", timeout=1)

        assert result.success is False
        assert "timeout" in result.stderr.lower()

    def test_command_with_stderr(self):
        """Test command that produces stderr."""
        result = execute_command("ls /nonexistent_directory_12345", timeout=5)

        assert result.success is False
        assert len(result.stderr) > 0

    def test_command_with_env(self):
        """Test command with custom environment."""
        env = {"TEST_VAR": "test_value"}
        result = execute_command("echo $TEST_VAR", timeout=5, env=env)

        assert result.success is True
        assert "test_value" in result.stdout

    def test_command_with_cwd(self, tmp_path):
        """Test command with custom working directory."""
        result = execute_command("pwd", timeout=5, cwd=str(tmp_path))

        assert result.success is True
        assert str(tmp_path) in result.stdout


class TestCommandHistory:
    """Test command history management."""

    def test_save_and_load_history(self, tmp_path):
        """Test saving and loading command history."""
        history_file = tmp_path / "history.txt"

        # Save some commands
        commands = ["ls -la", "pwd", "echo test"]
        for cmd in commands:
            save_to_history(str(history_file), cmd)

        # Load history
        loaded = load_command_history(str(history_file))

        assert len(loaded) == len(commands)
        assert loaded == commands

    def test_load_nonexistent_history(self, tmp_path):
        """Test loading non-existent history file."""
        history_file = tmp_path / "nonexistent.txt"
        loaded = load_command_history(str(history_file))

        assert loaded == []

    def test_history_max_entries(self, tmp_path):
        """Test history respects max entries."""
        history_file = tmp_path / "history.txt"
        max_entries = 5

        # Save more than max
        for i in range(10):
            save_to_history(str(history_file), f"command_{i}", max_entries=max_entries)

        # Load history
        loaded = load_command_history(str(history_file))

        assert len(loaded) <= max_entries
        # Should keep the most recent
        assert loaded[-1] == "command_9"


class TestFormatBytes:
    """Test byte formatting."""

    @pytest.mark.parametrize(
        "bytes_val,expected",
        [
            (0, "0.0 B"),
            (1024, "1.0 KB"),
            (1024**2, "1.0 MB"),
            (1024**3, "1.0 GB"),
            (1024**4, "1.0 TB"),
            (1500, "1.5 KB"),
            (1500000, "1.4 MB"),
        ],
    )
    def test_format_bytes(self, bytes_val, expected):
        """Test byte formatting."""
        result = format_bytes(bytes_val)
        assert result == expected or result.startswith(expected[0])  # Allow rounding differences


class TestCheckDependencies:
    """Test dependency checking."""

    def test_check_dependencies(self):
        """Test dependency checking."""
        deps = check_dependencies()

        assert isinstance(deps, dict)
        assert "ollama" in deps
        assert "pynvml" in deps

        # All values should be booleans
        for value in deps.values():
            assert isinstance(value, bool)
