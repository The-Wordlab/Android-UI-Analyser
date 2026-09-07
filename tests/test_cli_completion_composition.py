"""Caller-level restart and completion bookkeeping stay one operation, without extra reads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli, daemon, journal
from android_ui_analyser.schema import ActionResult
from android_ui_analyser.session import load_session_state, review_session_events
from test_goal_session_lifecycle import _engine, _observation

runner = CliRunner()


def test_restart_components_reach_the_journal_and_fold_into_one_caller_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path, "composite-fixture")
    engine.config.daemon.enabled = False
    engine.config.lease.enabled = False
    monkeypatch.setattr(engine, "_list_targets", lambda: pytest.fail("test must not discover targets"))
    started = engine.session_start("verify the fixture", observation=_observation(engine.device.serial))
    calls: list[str] = []
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
    monkeypatch.setattr(cli.GlobalOpts, "load", lambda self: engine.config)

    def app(action: str, **kwargs: Any) -> ActionResult:
        calls.append(action)
        return ActionResult(ok=True, action=f"app-{action}")

    monkeypatch.setattr(engine, "app", app)
    result = runner.invoke(cli.app, ["app", "restart-and-analyze", "example.app"])
    assert result.exit_code == 0, result.output
    assert calls == ["stop", "launch"]
    rows = journal.read_since(engine.config.cache.dir, engine.device.serial, limit=10)
    assert len(rows) == 2
    assert rows[0]["invocation_id"] == rows[1]["invocation_id"]
    assert [(row["extra"]["component"], row["extra"]["component_index"]) for row in rows] == [
        ("stop", 0), ("launch", 1)
    ]
    state = load_session_state(engine.config.cache.dir, session_id=started["session_id"])
    assert state is not None
    review = review_session_events(state, rows)
    assert not review["patterns"].get("ambiguous_invocation")
    assert review["commands"] == {"app_restart": 1}
    assert review["accounting"]["top_level_calls"] == 1


def test_daemon_transport_keeps_component_metadata_outside_engine_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []

    class Socket:
        def settimeout(self, value: float) -> None:
            pass

        def connect(self, path: str) -> None:
            pass

        def sendall(self, payload: bytes) -> None:
            requests.append(json.loads(payload))

        def recv(self, count: int) -> bytes:
            return b'{"ok":true,"result":{"ok":true}}\n'

        def close(self) -> None:
            pass

    monkeypatch.setattr(daemon.socket, "socket", lambda *args: Socket())
    component = {
        "parent_command": "app_restart", "component": "launch",
        "component_index": 1, "component_count": 2,
    }
    response = daemon.DaemonClient(
        "fake.sock", invocation_id="one-restart", journal_component=component
    ).call("app", action="launch", package="example.app")
    assert requests[0]["journal_component"] == component
    assert requests[0]["args"] == {"action": "launch", "package": "example.app"}
    engine = _engine(tmp_path, "daemon-composite-fixture")
    daemon._journal_dispatch(engine, requests[0], response, duration_ms=1)
    rows = journal.read_since(engine.config.cache.dir, engine.device.serial, limit=10)
    assert rows[0]["extra"]["component"] == "launch"
    assert rows[0]["extra"]["component_index"] == 1
    assert rows[0]["invocation_id"] == "one-restart"


def test_global_phase_facts_can_finish_a_session_without_analyze_or_device_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path, "finish-facts-fixture")
    engine.config.daemon.enabled = False
    engine.config.lease.enabled = False
    monkeypatch.setattr(engine, "_list_targets", lambda: pytest.fail("test must not discover targets"))
    started = engine.session_start(
        "Explore the app and complete one meaningful non-destructive end-to-end flow",
        observation=_observation(engine.device.serial),
    )
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
    monkeypatch.setattr(cli.GlobalOpts, "load", lambda self: engine.config)
    monkeypatch.setattr(engine, "analyze", lambda **kwargs: pytest.fail("finish must reuse evidence"))
    monkeypatch.setattr(engine, "_connect_target", lambda *a, **kw: pytest.fail("finish connected"))
    result = runner.invoke(
        cli.app,
        [
            "--phase-done",
            "phase_1=Conversation opened; assistant reply appeared; thread persisted after returning",
            "session", "finish", "--session-id", started["session_id"],
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    state = load_session_state(engine.config.cache.dir, session_id=started["session_id"])
    assert state is not None and state.finished_ms is not None
    assert all(phase.status == "completed" for phase in state.phases)
    rows = journal.read_since(engine.config.cache.dir, state.serial, limit=10)
    assert [row["cmd"] for row in rows] == ["session_finish"]
