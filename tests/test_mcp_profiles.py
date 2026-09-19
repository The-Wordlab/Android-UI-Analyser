"""A smaller opt-in catalogue must preserve schemas, dispatch and the full default."""

import json

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import mcp_server
from android_ui_analyser.capabilities import render_mcp_instructions
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_profiles import WEB_TOOL_NAMES, ToolProfile
from conftest import FakeDevice, make_config


def _engine():
    return Engine(make_config(), device=FakeDevice())


@pytest.mark.parametrize("profile", [None, "full"])
def test_full_profile_preserves_the_entire_catalogue_and_instructions(profile):
    server = mcp_server.build_server(
        _engine(), **({} if profile is None else {"tool_profile": profile})
    )
    assert server.instructions == render_mcp_instructions()

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            assert (await client.list_tools()).tools == mcp_server._tool_definitions()

    anyio.run(run)


def test_web_profile_is_exact_and_preserves_schemas():
    expected = {
        "configure",
        "session_start",
        "session_progress",
        "session_review",
        "session_finish",
        "analyze_screen",
        "has",
        "tap_and_analyze",
        "input_and_analyze",
        "clear_and_analyze",
        "swipe_and_analyze",
        "key_and_analyze",
        "scroll_to_and_analyze",
        "wait_and_analyze",
        "wait_changed_and_analyze",
        "wait_stable_and_analyze",
        "await_and_analyze",
        "expect_and_analyze",
        "screenshot",
        "inspect",
        "open_link_and_analyze",
        "orient",
        "reach",
        "goto",
        "map_find",
        "flow_list",
        "flow_run",
        "flow_save",
        "flow_delete",
        "browser_status",
        "browser_logs",
        "browser_storage",
        "browser_network",
        "browser_cors",
        "browser_proxy",
        "browser_har",
        "browser_mock",
        "browser_pages",
        "browser_trace",
    }
    assert expected == WEB_TOOL_NAMES
    server = mcp_server.build_server(_engine(), tool_profile="web")
    assert "analyze_screen next" not in server.instructions
    assert "do not follow them with analyze_screen" in server.instructions
    assert "a11y_scroll" not in server.instructions
    assert "prepare_start" not in server.instructions

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            listed = (await client.list_tools()).tools
            assert {tool.name for tool in listed} == expected
            assert listed == [t for t in mcp_server._tool_definitions() if t.name in expected]

    anyio.run(run)


def test_web_profile_rejects_cached_hidden_tool_before_dispatch(monkeypatch):
    server = mcp_server.build_server(_engine(), tool_profile="web")
    monkeypatch.setattr(
        mcp_server, "_dispatch", lambda *args: pytest.fail("hidden tool dispatched")
    )

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("database_execute", {})
            assert result.isError
            assert json.loads(result.content[0].text)["error"]["code"] == "tool_profile_unavailable"

    anyio.run(run)


def test_web_profile_still_validates_listed_tool_arguments(monkeypatch):
    server = mcp_server.build_server(_engine(), tool_profile="web")
    monkeypatch.setattr(
        mcp_server, "_dispatch", lambda *args: pytest.fail("invalid tool dispatched")
    )

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("browser_pages", {"action": "invented"})
            assert result.isError
            assert "Input validation error" in result.content[0].text

    anyio.run(run)


def test_web_profile_uses_existing_engine_dispatch_and_capability_errors(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "browser_status", lambda: {"ok": True, "marker": "shared-engine"})
    server = mcp_server.build_server(engine, tool_profile="web")

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            status = await client.call_tool("browser_status", {})
            assert json.loads(status.content[0].text)["marker"] == "shared-engine"
            # The profile does not switch the configured adapter or pretend native targets
            # provide web capabilities. An unsupported listed operation keeps its typed error.
            logs = await client.call_tool("browser_logs", {})
            assert (
                json.loads(logs.content[0].text)["error"]["code"]
                == "platform_capability_unsupported"
            )

    anyio.run(run)


@pytest.mark.parametrize("entry", ["build", "stdio"])
def test_invalid_programmatic_profile_fails_before_runtime_start(monkeypatch, entry):
    engine = _engine()
    monkeypatch.setattr(engine, "capture_service_start", lambda: pytest.fail("started capture"))
    monkeypatch.setattr(
        mcp_server, "build_default_engine", lambda *args: pytest.fail("built engine")
    )
    with pytest.raises(ValueError, match="invalid-profile"):
        if entry == "build":
            mcp_server.build_server(engine, tool_profile="invalid-profile")
        else:
            mcp_server.run_stdio(tool_profile="invalid-profile")


@pytest.mark.parametrize(
    ("args", "env", "expected"),
    [
        ([], {}, ToolProfile.full),
        (["--tool-profile", "web"], {}, ToolProfile.web),
        ([], {"AUA_MCP_TOOL_PROFILE": "web"}, ToolProfile.web),
        (["--tool-profile", "full"], {"AUA_MCP_TOOL_PROFILE": "web"}, ToolProfile.full),
    ],
)
def test_cli_profile_and_environment_are_forwarded(monkeypatch, args, env, expected):
    monkeypatch.delenv("AUA_MCP_TOOL_PROFILE", raising=False)
    calls = []
    monkeypatch.setattr(mcp_server, "run_stdio", lambda config, **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(app, ["mcp", *args], env=env)
    assert result.exit_code == 0, result.output
    assert calls == [{"tool_profile": expected}]


@pytest.mark.parametrize(
    "args,env", [(["--tool-profile", "unknown"], {}), ([], {"AUA_MCP_TOOL_PROFILE": "unknown"})]
)
def test_invalid_cli_profile_fails_without_starting_server(monkeypatch, args, env):
    monkeypatch.setattr(mcp_server, "run_stdio", lambda *a, **kw: pytest.fail("started server"))
    result = CliRunner().invoke(app, ["mcp", *args], env=env)
    assert result.exit_code == 2
    assert "unknown" in result.output
    assert "full" in result.output and "web" in result.output
