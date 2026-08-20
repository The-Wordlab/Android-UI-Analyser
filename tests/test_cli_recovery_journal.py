"""Top-level CLI recovery is structured, safe, and visible without touching Android."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser import journal
from android_ui_analyser.cli import app
from android_ui_analyser.config import load_config

runner = CliRunner()


def _events() -> list[dict[str, object]]:
    return journal.read_since(load_config().cache.dir, None, limit=20)


def _error(result: object) -> dict[str, object]:
    stderr = str(getattr(result, "stderr", "") or "")
    return json.loads(stderr.splitlines()[-1])["error"]


def test_absence_only_action_until_has_exact_recovery_and_is_journaled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("safe usage recovery must not connect to Android")
        ),
    )

    result = runner.invoke(
        app,
        [
            "--until",
            "!text:Loading",
            "tap-and-analyze",
            "--rid",
            "continueButton",
        ],
    )

    assert result.exit_code == 2
    error = _error(result)
    assert error["code"] == "usage"
    assert error["recommended_call"] == (
        "aua await-and-analyze '!text:Loading' --observe"
    )
    event = _events()[-1]
    assert event["cmd"] == "cli_usage_error"
    assert event["ok"] is False
    assert event["result"]["recommended_call"] == error["recommended_call"]
    assert event["args"] == {"command": "aua tap-and-analyze"}


def test_unknown_command_recovery_is_structured_and_journaled() -> None:
    result = runner.invoke(app, ["screen"])

    assert result.exit_code == 2
    error = _error(result)
    assert error["recommended_call"] == "aua analyze --help"
    event = _events()[-1]
    assert event["cmd"] == "cli_usage_error"
    assert event["error"]["code"] == "unknown_command"
    assert event["result"]["recommended_call"] == "aua analyze --help"


def test_top_level_help_is_journaled_without_raw_argv() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    event = _events()[-1]
    assert event["cmd"] == "cli_help"
    assert event["ok"] is True
    assert event["args"] == {"command": "aua"}
    assert event["result"]["recommended_call"] == "aua guide --brief"
