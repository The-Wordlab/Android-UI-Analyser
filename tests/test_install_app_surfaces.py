"""CLI, daemon, MCP, and the capability catalogue all reach the one ``Engine.install_app``.

A missing daemon branch is the failure mode this file exists to prevent: `_route` raises the
daemon's structured error rather than falling back in-process, so a command wired only into the
CLI works when tested cold and answers `unknown_command` for everyone running the warm daemon the
guide recommends. It has shipped that way twice before (`tap_point`, `input_text`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.capabilities import capability_manifest
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import _LONG_POLL_COMMANDS, dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _OBSERVATION_TOOL_NAMES, _tool_definitions
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from android_ui_analyser.schema import ActionResult
from conftest import FakeDevice, make_config

runner = CliRunner()


@pytest.fixture
def spy(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_install(self: Engine, bundle: str, **kwargs: Any) -> ActionResult:
        calls.append({"bundle": bundle, **kwargs})
        return ActionResult(
            ok=True,
            action="app-install",
            detail="com.example.app 2.1.0 (installed)",
            app_install={"package": "com.example.app", "pushed": True},
        )

    monkeypatch.setattr(Engine, "install_app", fake_install)
    return calls


def test_every_surface_calls_the_same_engine_method(spy, tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "example-debug.apk"
    bundle.write_bytes(b"stub")
    monkeypatch.setattr(
        engine_mod, "connect", lambda serial=None: FakeDevice(serial=serial or "emulator-5554")
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cli-cache"))

    cli = runner.invoke(app, ["--no-lease", "install", str(bundle), "--reinstall"])
    assert cli.exit_code == 0, cli.stderr

    engine = Engine(make_config(memory={"enabled": False}), device=FakeDevice())
    assert dispatch(
        engine, {"cmd": "install_app", "args": {"bundle": str(bundle), "mode": "reinstall"}}
    )["ok"]
    assert (
        mcp_dispatch(engine, "install_app", {"bundle": str(bundle), "mode": "reinstall"})["action"]
        == "app-install"
    )

    assert len(spy) == 3
    assert all(call["mode"] == "reinstall" for call in spy)
    assert all(Path(call["bundle"]) == bundle for call in spy)


def test_the_cli_converts_its_seconds_flag_into_the_daemon_s_millisecond_budget(
    spy, tmp_path, monkeypatch
) -> None:
    # The daemon sizes a request's socket from `timeout_ms`. Sending seconds would cap the socket
    # at the 5s default and answer `daemon_outcome_unknown` for an install still in flight.
    bundle = tmp_path / "example-debug.apk"
    bundle.write_bytes(b"stub")
    monkeypatch.setattr(
        engine_mod, "connect", lambda serial=None: FakeDevice(serial=serial or "emulator-5554")
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cli-cache"))

    cli = runner.invoke(
        app, ["--no-lease", "install", str(bundle), "--install-timeout", "90"]
    )

    assert cli.exit_code == 0, cli.stderr
    assert spy[0]["timeout_ms"] == 90_000
    # And a caller that omits it still gets a socket above the default.
    assert "install_app" in _LONG_POLL_COMMANDS


def test_reinstall_and_fresh_are_mutually_exclusive(spy, tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "example-debug.apk"
    bundle.write_bytes(b"stub")
    monkeypatch.setattr(
        engine_mod, "connect", lambda serial=None: FakeDevice(serial=serial or "emulator-5554")
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cli-cache"))

    cli = runner.invoke(app, ["--no-lease", "install", str(bundle), "--reinstall", "--fresh"])

    assert cli.exit_code != 0
    assert spy == []


def test_the_mcp_tool_is_declared_and_trims_its_folded_observation() -> None:
    tools = {tool.name: tool for tool in _tool_definitions()}

    assert "install_app" in tools
    schema = tools["install_app"].inputSchema
    assert schema["required"] == ["bundle"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["mode"]["enum"] == ["if-needed", "reinstall", "fresh"]
    # `confirmed` is defaulted rather than required because the destructive path is opt-in via
    # mode=fresh, and the engine refuses without it. `database_execute` requires it because every
    # call there mutates.
    assert schema["properties"]["confirmed"]["default"] is False
    # `--launch` folds in a screen, so MCP must trim it the way the CLI does.
    assert "install_app" in _OBSERVATION_TOOL_NAMES


def test_the_capability_catalogue_points_at_both_real_surfaces() -> None:
    capability = next(item for item in capability_manifest() if item["id"] == "install")

    assert capability["cli"].startswith("aua install ")
    assert capability["mcp"] == "install_app"
    assert "install_app" in {tool.name for tool in _tool_definitions()}
