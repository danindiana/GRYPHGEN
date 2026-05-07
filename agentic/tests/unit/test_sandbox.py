"""Unit tests for agent/sandbox.py (DockerSandbox)."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sandbox():
    from src.agent.sandbox import DockerSandbox
    return DockerSandbox()


def test_is_available_caches_true(sandbox, monkeypatch):
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    import docker
    monkeypatch.setattr(docker, "from_env", lambda: mock_client)
    sandbox._available = None
    assert sandbox.is_available() is True
    assert sandbox._available is True


def test_is_available_false_on_exception(sandbox, monkeypatch):
    def raise_err():
        raise Exception("docker not running")
    import docker
    monkeypatch.setattr(docker, "from_env", raise_err)
    sandbox._available = None
    assert sandbox.is_available() is False
    assert sandbox._available is False


def test_is_available_uses_cache(sandbox):
    sandbox._available = True
    assert sandbox.is_available() is True


async def test_run_dispatches_to_run_sync(sandbox, tmp_path, monkeypatch):
    sandbox._available = True
    monkeypatch.setattr(sandbox, "_run_sync", lambda cmd, ws, to: "hello world")
    result = await sandbox.run("echo hello world", tmp_path)
    assert result == "hello world"


async def test_run_falls_back_when_unavailable(sandbox, tmp_path):
    sandbox._available = False
    result = await sandbox.run("echo hi", tmp_path)
    assert "hi" in result or "not permitted" in result


async def test_whitelist_fallback_blocks_rm(sandbox, tmp_path):
    result = await sandbox._whitelist_fallback("rm -rf /", tmp_path, 5)
    assert "not permitted" in result


async def test_whitelist_fallback_allows_python(sandbox, tmp_path):
    (tmp_path / "hello.py").write_text("print('ok')")
    result = await sandbox._whitelist_fallback("python3 hello.py", tmp_path, 10)
    assert result == "ok"


async def test_whitelist_fallback_allows_echo(sandbox, tmp_path):
    result = await sandbox._whitelist_fallback("echo gryphgen", tmp_path, 5)
    assert "gryphgen" in result
