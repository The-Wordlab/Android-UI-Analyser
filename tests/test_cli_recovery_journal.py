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
        engine_mod.Engine,
        "_connect_target",
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
    assert event["args"]["command"] == "aua tap-and-analyze"


def test_misspelled_action_until_is_refused_before_android_and_is_journaled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        engine_mod.Engine,
        "_connect_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("safe usage recovery must not connect to Android")
        ),
    )

    result = runner.invoke(
        app,
        [
            "--until",
            "nosuchfield:Loading",
            "tap-and-analyze",
            "--rid",
            "continueButton",
        ],
    )

    assert result.exit_code == 2
    error = _error(result)
    assert error["code"] == "usage"
    assert "unknown field" in str(error["message"])
    event = _events()[-1]
    assert event["cmd"] == "cli_usage_error"
    assert event["ok"] is False
    assert event["args"]["command"] == "aua tap-and-analyze"


def test_unknown_command_recovery_is_structured_and_journaled() -> None:
    result = runner.invoke(app, ["screen"])

    assert result.exit_code == 2
    error = _error(result)
    assert error["recommended_call"] == "aua analyze --help"
    event = _events()[-1]
    assert event["cmd"] == "cli_usage_error"
    assert event["error"]["code"] == "unknown_command"
    assert event["result"]["recommended_call"] == "aua analyze --help"


def test_top_level_help_is_journaled_with_its_shape_and_its_answer() -> None:
    """The row records what was asked and what came back, with values redacted.

    It used to record neither: `args` was the binary name and the result a bare marker, so
    a help call rendered as a content-free row — indistinguishable from help returning
    nothing. Raw argv still never lands here; see `redacted_argv`.
    """
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    event = _events()[-1]
    assert event["cmd"] == "cli_help"
    assert event["ok"] is True
    assert event["args"]["command"] == "aua"
    assert event["args"]["argv"] == ["--help"]
    assert event["result"]["recommended_call"] == "aua guide --brief"
    # The answer the caller actually received is what makes the row worth reading. The
    # index stays small and tailable, so the body lives in the on-demand detail — the same
    # place the dashboard's expandable row fetches it from.
    detail = journal.read_detail(load_config().cache.dir, None, str(event["detail_id"]))
    body = ((detail or {}).get("response") or {}).get("result") or {}
    assert "The loop" in body["response_text"]
    assert body["response_lines"] > 10


def test_host_only_recovery_journal_uses_the_selected_platform(monkeypatch) -> None:
    # Eager top-level help runs before Click binds CLI options, but configuration/environment
    # selection is already authoritative and must still choose the journal namespace.
    monkeypatch.setenv("AUA_PLATFORM", "strict-fake")
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    cfg = load_config()
    events = journal.read_since(
        cfg.cache.dir,
        None,
        limit=20,
        platform="strict-fake",
    )
    assert events[-1]["cmd"] == "cli_help"
    assert events[-1]["platform"] == "strict-fake"
