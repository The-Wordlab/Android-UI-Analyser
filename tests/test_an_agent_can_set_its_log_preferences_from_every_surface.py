"""One implementation, three doors: CLI, warm daemon, and MCP.

The rule this pins is the repository's, not this feature's: the CLI and MCP must share the same
engine implementation and the same error behaviour, and a warm daemon must answer every name
the CLI can send. A preference surface that drifts between doors is worse than none — the agent
sets a preference through one and the next call, made through another, ignores it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.daemon import _LEASE_FREE_COMMANDS, dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _LEASE_FREE_TOOLS, _tool_definitions
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from conftest import FakeDevice, make_config

APP = "com.example.notes"
runner = CliRunner()


def test_every_surface_reaches_the_same_engine_method(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[dict[str, Any]] = []

    def spy(self: Engine, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"ok": True, "action": "app-log-prefs-set", "package": kwargs.get("app")}

    monkeypatch.setattr(Engine, "app_log_prefs_set", spy)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: FakeDevice(package=APP))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    cli = runner.invoke(
        cli_app,
        [
            "--no-lease",
            "logcat",
            "prefs",
            "set",
            "--app",
            APP,
            "--ignore-tag",
            "ChattyThing",
            "--lines",
            "40",
        ],
    )
    assert cli.exit_code == 0, cli.stderr

    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice(package=APP))
    assert dispatch(
        engine,
        {
            "cmd": "app_log_prefs_set",
            "args": {"app": APP, "ignore_tags": ["ChattyThing"], "limit": 40},
        },
    )["ok"]
    assert (
        mcp_dispatch(
            engine,
            "app_log_prefs_set",
            {"package": APP, "ignore_tags": ["ChattyThing"], "limit": 40},
        )["action"]
        == "app-log-prefs-set"
    )

    assert len(seen) == 3, "one of the three doors is not reaching the engine"
    for call in seen:
        assert call["app"] == APP
        assert list(call["ignore_tags"]) == ["ChattyThing"]
        assert call["limit"] == 40


def test_reading_the_preference_is_reachable_from_every_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def spy(self: Engine, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"ok": True, "action": "app-log-prefs", "package": kwargs.get("app")}

    monkeypatch.setattr(Engine, "app_log_prefs", spy)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: FakeDevice(package=APP))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    cli = runner.invoke(cli_app, ["--no-lease", "logcat", "prefs", "show", "--app", APP])
    assert cli.exit_code == 0, cli.stderr
    assert json.loads(cli.stdout)["package"] == APP

    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice(package=APP))
    assert dispatch(engine, {"cmd": "app_log_prefs", "args": {"app": APP}})["ok"]
    assert mcp_dispatch(engine, "app_log_prefs_get", {"package": APP})["package"] == APP

    assert len(seen) == 3


def test_the_published_tools_take_an_app_and_refuse_anything_else() -> None:
    tools = {tool.name: tool for tool in _tool_definitions()}

    for name in ("app_log_prefs_get", "app_log_prefs_set"):
        assert name in tools, f"{name} is not published to agents"
        schema = tools[name].inputSchema
        assert schema["required"] == ["package"]
        assert schema["additionalProperties"] is False
        assert name in _LEASE_FREE_TOOLS, "a host-side preference must not demand a device lease"

    setter = tools["app_log_prefs_set"].inputSchema["properties"]
    for arg in ("ignore_tags", "unignore_tags", "only_tags", "levels", "limit", "per_tag"):
        assert arg in setter, f"the setter cannot express {arg}"


def test_a_warm_daemon_answers_both_names_without_a_device_lease() -> None:
    for cmd in ("app_log_prefs", "app_log_prefs_set"):
        assert cmd in _LEASE_FREE_COMMANDS


def test_the_session_only_switch_can_still_be_set_without_persisting_anything() -> None:
    # `configure` is the per-turn twin of the persisted preference: same knobs, this session
    # only, every app. An agent chasing one library for three calls should not have to write
    # a preference it will have to remember to undo.
    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice(package=APP))

    result = mcp_dispatch(
        engine,
        "configure",
        {"app_log_limit": 60, "app_log_per_tag": 12, "app_log_only_tags": ["Checkout"]},
    )

    assert result["app_log_limit"] == 60
    assert engine.config.logs.limit == 60
    assert engine.config.logs.per_tag == 12
    assert engine.config.logs.only_tags == ["Checkout"]
