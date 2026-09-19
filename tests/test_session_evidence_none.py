"""Explicitly disabling evidence never requires an artifact output directory."""

import json

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser import cli
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import AuaError
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.session import load_session_state
from conftest import FakeDevice, make_config


@pytest.fixture
def engine():
    config = make_config(
        daemon={"enabled": False},
        lease={"enabled": False},
        memory={"enabled": False},
        teardown={"enabled": False},
    )
    instance = Engine(config, device=FakeDevice())
    yield instance
    instance.close()


def _start(surface, engine, monkeypatch, options):
    goal = "Inspect the fictional fixture screen"
    if surface == "engine":
        try:
            return engine.session_start(goal, **options)
        except AuaError as error:
            return error.to_dict()
    if surface == "cli":
        monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
        monkeypatch.setattr(cli.GlobalOpts, "load", lambda self: engine.config)
        argv = ["session", "start", "--goal", goal]
        if "evidence" in options:
            argv.extend(["--evidence", options["evidence"]])
        if options.get("junit"):
            argv.append("--junit")
        result = CliRunner().invoke(cli.app, argv)
        return json.loads(result.stdout if result.exit_code == 0 else result.stderr)

    async def invoke():
        async with create_connected_server_and_client_session(build_server(engine)) as client:
            listed = await client.list_tools()
            schema = next(tool.inputSchema for tool in listed.tools if tool.name == "session_start")
            assert "none" in schema["properties"]["evidence"]["enum"]
            result = await client.call_tool("session_start", {"goal": goal, **options})
            return json.loads(next(block.text for block in result.content if block.type == "text"))

    return anyio.run(invoke)


@pytest.mark.parametrize("surface", ["engine", "cli", "mcp"])
@pytest.mark.parametrize("options", [{}, {"evidence": "failures"}, {"evidence": "none"}])
def test_session_without_artifacts_accepts_no_evidence_or_the_default(
    surface, options, engine, monkeypatch
):
    result = _start(surface, engine, monkeypatch, options)
    assert "error" not in result, result
    state = load_session_state(engine.config.cache.dir, session_id=result["session_id"])
    assert state is not None
    assert state.artifact_dir is None
    assert state.evidence == options.get("evidence", "failures")


@pytest.mark.parametrize("surface", ["engine", "cli", "mcp"])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"evidence": "all"}, "--evidence needs --artifacts-dir"),
        ({"evidence": "none", "junit": True}, "--junit needs --artifacts-dir"),
    ],
)
def test_artifact_outputs_still_require_a_directory_before_observation(
    surface, options, message, engine, monkeypatch
):
    result = _start(surface, engine, monkeypatch, options)
    assert result["error"]["code"] == "usage"
    assert result["error"]["message"] == message
    assert engine._last_analyze_result is None
