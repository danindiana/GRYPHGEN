"""Unit tests for agent tool implementations and dispatch."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.agent.tools import (
    _safe_path,
    dispatch,
    list_dir,
    read_file,
    run_shell,
    write_file,
    TOOL_SCHEMAS,
)


@pytest.fixture
def workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory(prefix="gryphgen_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def workspace_with_files(workspace):
    """Create workspace with pre-populated test files."""
    (workspace / "test.txt").write_text("Hello, World!")
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "nested.txt").write_text("Nested content")
    return workspace


class TestSafePath:
    """Test path validation and sandbox enforcement."""

    def test_safe_path_valid_relative(self, workspace):
        """Valid relative path within workspace."""
        result = _safe_path(workspace, "test.txt")
        assert result.parent == workspace

    def test_safe_path_valid_subdirectory(self, workspace):
        """Valid path in subdirectory."""
        result = _safe_path(workspace, "subdir/test.txt")
        assert str(result).startswith(str(workspace))

    def test_safe_path_valid_dot(self, workspace):
        """Current directory reference."""
        result = _safe_path(workspace, ".")
        assert result == workspace.resolve()

    def test_safe_path_escape_attempt_parent(self, workspace):
        """Reject path escape via parent directory (..)."""
        with pytest.raises(PermissionError, match="escapes workspace"):
            _safe_path(workspace, "..")

    def test_safe_path_escape_attempt_absolute(self, workspace):
        """Reject absolute path escape."""
        with pytest.raises(PermissionError, match="escapes workspace"):
            _safe_path(workspace, "/etc/passwd")

    def test_safe_path_escape_attempt_symlink(self, workspace):
        """Reject symlink escape (if symlink exists and escapes)."""
        if hasattr(Path, 'symlink_to'):
            try:
                (workspace / "escape").symlink_to("/tmp")
                with pytest.raises(PermissionError, match="escapes workspace"):
                    _safe_path(workspace, "escape")
            except (OSError, NotImplementedError):
                # Symlinks may not be supported on this OS
                pytest.skip("Symlinks not supported on this OS")

    def test_safe_path_normalization(self, workspace):
        """Path normalization resolves correctly."""
        result = _safe_path(workspace, "./subdir/../test.txt")
        # Should resolve to test.txt in workspace
        assert result.name == "test.txt"


class TestReadFile:
    """Test read_file tool."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, workspace_with_files):
        """Successfully read a file."""
        result = await read_file(workspace_with_files, "test.txt")
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_file_nested(self, workspace_with_files):
        """Read file from subdirectory."""
        result = await read_file(workspace_with_files, "subdir/nested.txt")
        assert result == "Nested content"

    @pytest.mark.asyncio
    async def test_read_file_not_exists(self, workspace):
        """Error when file doesn't exist."""
        result = await read_file(workspace, "nonexistent.txt")
        assert "ERROR" in result and "does not exist" in result

    @pytest.mark.asyncio
    async def test_read_file_is_directory(self, workspace_with_files):
        """Error when path is a directory."""
        result = await read_file(workspace_with_files, "subdir")
        assert "ERROR" in result and "not a file" in result

    @pytest.mark.asyncio
    async def test_read_file_binary_with_fallback(self, workspace):
        """Read binary file with error replacement."""
        # Create a file with invalid UTF-8
        bin_path = workspace / "binary.bin"
        bin_path.write_bytes(b"\x80\x81\x82\x83")
        result = await read_file(workspace, "binary.bin")
        # Should not raise, should have replacement chars
        assert isinstance(result, str)
        assert len(result) > 0


class TestWriteFile:
    """Test write_file tool."""

    @pytest.mark.asyncio
    async def test_write_file_new(self, workspace):
        """Create a new file."""
        result = await write_file(workspace, "new.txt", "test content")
        assert "OK" in result and "wrote" in result
        assert (workspace / "new.txt").read_text() == "test content"

    @pytest.mark.asyncio
    async def test_write_file_overwrite(self, workspace_with_files):
        """Overwrite existing file."""
        result = await write_file(workspace_with_files, "test.txt", "new content")
        assert "OK" in result
        assert (workspace_with_files / "test.txt").read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_file_nested_dir(self, workspace):
        """Create nested directories as needed."""
        result = await write_file(
            workspace, "a/b/c/test.txt", "nested content"
        )
        assert "OK" in result
        assert (workspace / "a" / "b" / "c" / "test.txt").read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_write_file_empty(self, workspace):
        """Write empty file."""
        result = await write_file(workspace, "empty.txt", "")
        assert "wrote 0 bytes" in result
        assert (workspace / "empty.txt").exists()

    @pytest.mark.asyncio
    async def test_write_file_large_content(self, workspace):
        """Write large file."""
        large_content = "x" * 1_000_000  # 1MB
        result = await write_file(workspace, "large.txt", large_content)
        assert "1000000 bytes" in result


class TestListDir:
    """Test list_dir tool."""

    @pytest.mark.asyncio
    async def test_list_dir_root(self, workspace_with_files):
        """List root workspace directory."""
        result = await list_dir(workspace_with_files, ".")
        assert "test.txt" in result
        assert "subdir" in result
        assert "dir" in result  # keyword indicating directory

    @pytest.mark.asyncio
    async def test_list_dir_subdirectory(self, workspace_with_files):
        """List subdirectory."""
        result = await list_dir(workspace_with_files, "subdir")
        assert "nested.txt" in result

    @pytest.mark.asyncio
    async def test_list_dir_empty(self, workspace):
        """List empty directory."""
        result = await list_dir(workspace, ".")
        assert "(empty)" in result

    @pytest.mark.asyncio
    async def test_list_dir_not_exists(self, workspace):
        """Error when directory doesn't exist."""
        result = await list_dir(workspace, "nonexistent")
        assert "ERROR" in result and "does not exist" in result

    @pytest.mark.asyncio
    async def test_list_dir_not_directory(self, workspace_with_files):
        """Error when path is not a directory."""
        result = await list_dir(workspace_with_files, "test.txt")
        assert "ERROR" in result and "not a directory" in result


class TestRunShell:
    """Test run_shell tool with whitelist enforcement."""

    @pytest.mark.asyncio
    async def test_run_shell_echo(self, workspace):
        """Execute allowed echo command."""
        result = await run_shell(workspace, "echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_run_shell_ls(self, workspace):
        """Execute allowed ls command."""
        (workspace / "file.txt").write_text("test")
        result = await run_shell(workspace, "ls")
        assert "file.txt" in result

    @pytest.mark.asyncio
    async def test_run_shell_python(self, workspace):
        """Execute allowed python command."""
        result = await run_shell(workspace, "python3 -c 'print(42)'")
        assert "42" in result

    @pytest.mark.asyncio
    async def test_run_shell_git_status(self, workspace):
        """Execute allowed git command."""
        await run_shell(workspace, "git init")
        result = await run_shell(workspace, "git status")
        assert "On branch" in result or "working tree clean" in result

    @pytest.mark.asyncio
    async def test_run_shell_command_not_permitted(self, workspace):
        """Reject command not in whitelist."""
        result = await run_shell(workspace, "rm -rf /")
        assert "ERROR" in result and "not permitted" in result

    @pytest.mark.asyncio
    async def test_run_shell_forbidden_prefix(self, workspace):
        """Case-insensitive whitelist check."""
        result = await run_shell(workspace, "RM nonexistent.txt")
        assert "ERROR" in result and "not permitted" in result

    @pytest.mark.asyncio
    async def test_run_shell_timeout(self, workspace):
        """Command timeout enforcement."""
        result = await run_shell(workspace, "python3 -c 'import time; time.sleep(60)'")
        assert "timed out" in result or "ERROR" in result

    @pytest.mark.asyncio
    async def test_run_shell_no_output(self, workspace):
        """Handle commands with no output."""
        result = await run_shell(workspace, "ls /dev/null")
        # Should not raise, returns empty or "(no output)"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_shell_error_stderr(self, workspace):
        """Capture stderr output."""
        result = await run_shell(workspace, "python3 -c 'import sys; sys.stderr.write(\"error\")'")
        # Stderr is captured as stdout
        assert isinstance(result, str)


class TestDispatch:
    """Test tool dispatch mechanism."""

    @pytest.mark.asyncio
    async def test_dispatch_read_file(self, workspace_with_files):
        """Dispatch to read_file."""
        result = await dispatch("read_file", {"path": "test.txt"}, workspace_with_files)
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_dispatch_write_file(self, workspace):
        """Dispatch to write_file."""
        result = await dispatch(
            "write_file",
            {"path": "test.txt", "content": "test"},
            workspace,
        )
        assert "OK" in result

    @pytest.mark.asyncio
    async def test_dispatch_list_dir(self, workspace_with_files):
        """Dispatch to list_dir."""
        result = await dispatch("list_dir", {"path": "."}, workspace_with_files)
        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_dispatch_list_dir_default_path(self, workspace_with_files):
        """Dispatch list_dir with default path."""
        result = await dispatch("list_dir", {}, workspace_with_files)
        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_dispatch_run_shell(self, workspace):
        """Dispatch to run_shell."""
        result = await dispatch("run_shell", {"command": "echo test"}, workspace)
        assert "test" in result

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, workspace):
        """Unknown tool returns error."""
        result = await dispatch("unknown_tool", {}, workspace)
        assert "ERROR" in result and "unknown tool" in result


class TestToolSchemas:
    """Test TOOL_SCHEMAS export."""

    def test_tool_schemas_format(self):
        """TOOL_SCHEMAS has correct format."""
        assert isinstance(TOOL_SCHEMAS, list)
        assert len(TOOL_SCHEMAS) == 4

    def test_tool_schemas_structure(self):
        """Each schema has required fields."""
        for schema in TOOL_SCHEMAS:
            assert "type" in schema
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_tool_schemas_names(self):
        """All expected tools are present."""
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert names == {"read_file", "write_file", "list_dir", "run_shell"}
