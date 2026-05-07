"""Unit tests for agent/runner.py (AgentRunner ReAct loop)."""

import pytest


def _chat_resp(content="", tool_calls=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


@pytest.fixture
def runner():
    from src.agent.runner import AgentRunner
    return AgentRunner(base_url="http://mock-ollama", model="test-model", max_steps=5)


def test_mcp_server_config_defaults():
    from src.agent.runner import McpServerConfig
    cfg = McpServerConfig(name="test", command=["uvx", "test-server"])
    assert cfg.transport == "stdio"
    assert cfg.name == "test"
    assert cfg.command == ["uvx", "test-server"]


async def test_run_direct_answer(runner, tmp_path, httpx_mock):
    httpx_mock.add_response(json=_chat_resp("Done!"))
    result = await runner.run("say done", workspace=tmp_path)
    assert result.output == "Done!"
    assert result.steps_taken == 1
    assert not result.truncated
    assert result.model == "test-model"


async def test_run_with_tool_call_then_answer(runner, tmp_path, httpx_mock):
    httpx_mock.add_response(json=_chat_resp(
        tool_calls=[{"function": {"name": "list_dir", "arguments": {}}, "id": "1"}]
    ))
    httpx_mock.add_response(json=_chat_resp("I listed the dir."))
    result = await runner.run("list the workspace", workspace=tmp_path)
    assert result.output == "I listed the dir."
    assert len(result.trace) == 1
    assert result.trace[0].tool == "list_dir"
    assert result.trace[0].step == 1


async def test_run_truncates_at_max_steps(runner, tmp_path, httpx_mock):
    for _ in range(runner.max_steps):
        httpx_mock.add_response(json=_chat_resp(
            tool_calls=[{"function": {"name": "list_dir", "arguments": {}}, "id": "x"}]
        ))
    result = await runner.run("never stop", workspace=tmp_path)
    assert result.truncated is True
    assert result.steps_taken == runner.max_steps


async def test_run_no_mcp_servers(runner, tmp_path, httpx_mock):
    httpx_mock.add_response(json=_chat_resp("ok"))
    result = await runner.run("test", workspace=tmp_path)
    assert result.output == "ok"
    assert result.total_time_s >= 0
    assert isinstance(result.trace, list)


async def test_run_with_files_prepopulates_workspace(runner, tmp_path, httpx_mock):
    httpx_mock.add_response(json=_chat_resp("saw files"))
    await runner.run("check files", workspace=tmp_path, files={"hello.py": "print('hi')"})
    assert (tmp_path / "hello.py").read_text() == "print('hi')"
