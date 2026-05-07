"""Unit tests for agent/tools.py."""

import pytest
from unittest.mock import AsyncMock

from src.agent.tools import read_file, write_file, list_dir, run_shell, web_search, dispatch


async def test_read_file_exists(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    result = await read_file(tmp_path, "a.txt")
    assert result == "hello"


async def test_read_file_missing(tmp_path):
    result = await read_file(tmp_path, "nope.txt")
    assert "does not exist" in result


async def test_read_file_traversal_blocked(tmp_path):
    with pytest.raises(PermissionError):
        await read_file(tmp_path, "../../etc/passwd")


async def test_write_file(tmp_path):
    result = await write_file(tmp_path, "out.py", "x = 1")
    assert "wrote" in result
    assert (tmp_path / "out.py").read_text() == "x = 1"


async def test_write_file_creates_parent_dirs(tmp_path):
    await write_file(tmp_path, "sub/dir/f.py", "pass")
    assert (tmp_path / "sub/dir/f.py").exists()


async def test_list_dir_shows_files(tmp_path):
    (tmp_path / "a.py").write_text("x")
    result = await list_dir(tmp_path)
    assert "a.py" in result


async def test_list_dir_missing_path(tmp_path):
    result = await list_dir(tmp_path, "no_such_dir")
    assert "does not exist" in result


async def test_list_dir_not_a_directory(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    result = await list_dir(tmp_path, "file.txt")
    assert "not a directory" in result


async def test_run_shell_delegates_to_sandbox(tmp_path, monkeypatch):
    from src.agent import tools
    mock_sandbox = AsyncMock()
    mock_sandbox.run.return_value = "42"
    monkeypatch.setattr(tools, "_sandbox", mock_sandbox)
    result = await run_shell(tmp_path, "echo 42")
    mock_sandbox.run.assert_called_once_with("echo 42", tmp_path)
    assert result == "42"


async def test_web_search_returns_results(monkeypatch):
    import src.tools.web_search as ws_mod
    mock_results = [{"title": "Python", "url": "http://python.org", "snippet": "Language"}]
    monkeypatch.setattr(ws_mod, "search", AsyncMock(return_value=mock_results))
    result = await web_search("python")
    assert "Python" in result
    assert "http://python.org" in result


async def test_web_search_no_results(monkeypatch):
    import src.tools.web_search as ws_mod
    monkeypatch.setattr(ws_mod, "search", AsyncMock(return_value=[]))
    result = await web_search("xyzzy")
    assert "No results" in result


async def test_dispatch_unknown_tool(tmp_path):
    result = await dispatch("unknown_tool", {}, tmp_path)
    assert "unknown tool" in result


async def test_dispatch_read_file(tmp_path):
    (tmp_path / "data.txt").write_text("content")
    result = await dispatch("read_file", {"path": "data.txt"}, tmp_path)
    assert result == "content"


async def test_dispatch_list_dir(tmp_path):
    (tmp_path / "item.py").write_text("")
    result = await dispatch("list_dir", {"path": "."}, tmp_path)
    assert "item.py" in result


async def test_dispatch_web_search(tmp_path, monkeypatch):
    from src.agent import tools
    monkeypatch.setattr(tools, "web_search", AsyncMock(return_value="search results"))
    result = await dispatch("web_search", {"query": "python"}, tmp_path)
    assert result == "search results"
