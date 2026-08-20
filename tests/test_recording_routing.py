"""Screen recording must share the session daemon's device command fence.

The daemon holds the session's long-lived device-use fence. Running ``record stop``
directly in a second CLI process therefore waits forever on that same fence instead
of stopping or pulling the recording. Both recording commands must use the ordinary
daemon route, which also keeps CLI and MCP on the same Engine implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

from typer.testing import CliRunner

from android_ui_analyser import cli as cli_mod
from android_ui_analyser.cli import app

CLI = Path(cli_mod.__file__).resolve()
runner = CliRunner()


def _function_source(name: str) -> str:
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in cli.py")


def test_recording_cli_commands_route_through_the_session_daemon() -> None:
    for name in ("record_start_cmd", "record_stop_cmd"):
        source = _function_source(name)
        assert "_route(" in source
        assert "engine.record_" not in source


def test_recording_cli_routes_remote_and_caller_resolved_local_paths(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_route(_engine, command: str, **kwargs: object) -> dict[str, object]:
        calls.append((command, kwargs))
        return {"ok": True, "action": command.replace("_", "-")}

    monkeypatch.setattr(cli_mod, "_route", fake_route)

    started = runner.invoke(app, ["record", "start", "--remote", "/sdcard/journey.mp4"])
    destination = tmp_path / "evidence" / "journey.mp4"
    stopped = runner.invoke(app, ["record", "stop", str(destination)])

    assert started.exit_code == 0, started.stderr
    assert stopped.exit_code == 0, stopped.stderr
    assert calls == [
        ("record_start", {"path": "/sdcard/journey.mp4"}),
        ("record_stop", {"local_path": str(destination.resolve())}),
    ]
