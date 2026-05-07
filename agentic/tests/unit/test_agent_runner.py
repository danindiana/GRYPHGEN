"""Unit tests for agent runner and ReAct loop."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.runner import AgentResult, AgentRunner, StepTrace


@pytest.fixture
def runner():
    """Create an AgentRunner instance."""
    return AgentRunner(
        base_url="http://127.0.0.1:11434",
        model="test-model",
        max_steps=5,
        step_timeout=30.0,
    )


class TestAgentRunnerInit:
    """Test AgentRunner initialization."""

    def test_init_defaults(self):
        """Initialize with default parameters."""
        runner = AgentRunner()
        assert runner.base_url == "http://127.0.0.1:11434"
        assert runner.model == "devstral:24b"
        assert runner.max_steps == 20
        assert runner.step_timeout == 300.0

    def test_init_custom_params(self):
        """Initialize with custom parameters."""
        runner = AgentRunner(
            base_url="http://custom:8000",
            model="custom-model",
            max_steps=10,
            step_timeout=60.0,
        )
        assert runner.base_url == "http://custom:8000"
        assert runner.model == "custom-model"
        assert runner.max_steps == 10
        assert runner.step_timeout == 60.0

    def test_init_url_trailing_slash_removed(self):
        """Trailing slash is removed from base_url."""
        runner = AgentRunner(base_url="http://localhost:11434/")
        assert runner.base_url == "http://localhost:11434"


class TestAgentRunnerRun:
    """Test AgentRunner.run() method."""

    @pytest.mark.asyncio
    async def test_run_creates_temp_workspace(self, runner):
        """Run creates temporary workspace when none provided."""
        with patch.object(runner, '_react_loop') as mock_loop:
            mock_loop.return_value = AgentResult(
                output="done",
                trace=[],
                steps_taken=1,
                total_time_s=0.1,
                model="test-model",
            )
            result = await runner.run("test task")
            
            # Verify temp workspace was created
            assert mock_loop.called
            call_args = mock_loop.call_args
            workspace = call_args[0][1]
            assert isinstance(workspace, Path)
            # Temp dir should be cleaned up
            # (We can't check this directly since cleanup is in finally)

    @pytest.mark.asyncio
    async def test_run_uses_provided_workspace(self, runner):
        """Run uses provided workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with patch.object(runner, '_react_loop') as mock_loop:
                mock_loop.return_value = AgentResult(
                    output="done",
                    trace=[],
                    steps_taken=1,
                    total_time_s=0.1,
                    model="test-model",
                )
                result = await runner.run("test task", workspace=workspace)
                
                call_args = mock_loop.call_args
                assert call_args[0][1] == workspace

    @pytest.mark.asyncio
    async def test_run_prepopulates_files(self, runner):
        """Run prepopulates workspace with initial files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            files = {
                "test.txt": "hello",
                "subdir/nested.txt": "world",
            }
            with patch.object(runner, '_react_loop') as mock_loop:
                mock_loop.return_value = AgentResult(
                    output="done",
                    trace=[],
                    steps_taken=1,
                    total_time_s=0.1,
                    model="test-model",
                )
                result = await runner.run("test task", workspace=workspace, files=files)
                
                # Verify files were created
                assert (workspace / "test.txt").read_text() == "hello"
                assert (workspace / "subdir" / "nested.txt").read_text() == "world"

    @pytest.mark.asyncio
    async def test_run_returns_agent_result(self, runner):
        """Run returns AgentResult from react loop."""
        expected_result = AgentResult(
            output="task complete",
            trace=[],
            steps_taken=3,
            total_time_s=0.5,
            model="test-model",
        )
        with patch.object(runner, '_react_loop') as mock_loop:
            mock_loop.return_value = expected_result
            result = await runner.run("test task")
            
            assert result == expected_result


class TestAgentRunnerReactLoop:
    """Test AgentRunner._react_loop() method - signature is complex, minimal coverage."""

    @pytest.mark.asyncio
    async def test_react_loop_signature(self, runner):
        """Verify _react_loop has correct signature."""
        import inspect
        sig = inspect.signature(runner._react_loop)
        params = list(sig.parameters.keys())
        # Should have task, workspace, extra_schemas, mcp_sessions
        assert "task" in params
        assert "workspace" in params
        assert "extra_schemas" in params
        assert "mcp_sessions" in params


class TestStepTrace:
    """Test StepTrace dataclass."""

    def test_step_trace_creation(self):
        """Create StepTrace instance."""
        trace = StepTrace(
            step=1,
            tool="read_file",
            args={"path": "test.txt"},
            result="file content",
            elapsed_s=0.123,
        )
        assert trace.step == 1
        assert trace.tool == "read_file"
        assert trace.args == {"path": "test.txt"}
        assert trace.result == "file content"
        assert trace.elapsed_s == 0.123


class TestAgentResult:
    """Test AgentResult dataclass."""

    def test_agent_result_creation(self):
        """Create AgentResult instance."""
        trace = [
            StepTrace(
                step=1,
                tool="list_dir",
                args={},
                result="(empty)",
                elapsed_s=0.05,
            ),
        ]
        result = AgentResult(
            output="Task complete",
            trace=trace,
            steps_taken=2,
            total_time_s=0.5,
            model="test-model",
        )
        assert result.output == "Task complete"
        assert len(result.trace) == 1
        assert result.steps_taken == 2
        assert result.total_time_s == 0.5
        assert result.model == "test-model"
        assert not result.truncated

    def test_agent_result_truncated_default(self):
        """AgentResult truncated defaults to False."""
        result = AgentResult(
            output="test",
            trace=[],
            steps_taken=1,
            total_time_s=0.1,
            model="test",
        )
        assert result.truncated is False

    def test_agent_result_truncated_true(self):
        """AgentResult can be marked truncated."""
        result = AgentResult(
            output="test",
            trace=[],
            steps_taken=20,
            total_time_s=10.0,
            model="test",
            truncated=True,
        )
        assert result.truncated is True
