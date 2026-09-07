"""Prior arrival evidence cannot establish that a later behavior check is redundant."""

from __future__ import annotations

from typing import Any

import pytest

import android_ui_analyser.cli as cli_mod


class _Recorder:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.information: list[str] = []

    def warning(self, msg: str, *args: Any) -> None:
        self.warnings.append(msg % args if args else msg)

    def info(self, msg: str, *args: Any) -> None:
        self.information.append(msg % args if args else msg)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    value = _Recorder()
    monkeypatch.setattr(cli_mod, "logger", value)
    return value


def _fire(monkeypatch: pytest.MonkeyPatch, result: dict, *, waited_for: str | None = None) -> None:
    """Drive the lint against a scripted journal, with no device anywhere."""
    import sys
    import types

    import android_ui_analyser

    events = [{"cmd": "await_predicate", "ok": True, "args": {"predicate": "text:Ready"}, "result": result}]
    fake_journal = types.SimpleNamespace(
        read_since=lambda *a, **k: events,
        review_events=lambda _cache, _serial, rows, **kwargs: rows,
    )
    monkeypatch.setattr(android_ui_analyser, "journal", fake_journal, raising=False)
    monkeypatch.setitem(sys.modules, "android_ui_analyser.journal", fake_journal)
    engine = types.SimpleNamespace(
        config=types.SimpleNamespace(cache=types.SimpleNamespace(dir="/example/none"))
    )
    cli_mod._warn_if_wait_could_have_been_until(engine, waited_for)


def _ready() -> dict:
    return {
        "action": "await",
        "await_outcome": "satisfied",
        "observation": {"elements": [{"id": "el:ready", "text": "Ready"}]},
    }


def test_a_wait_command_without_evidence_is_not_called_settled(monkeypatch, recorder) -> None:
    _fire(monkeypatch, {"action": "await"})

    assert recorder.warnings == []
    assert recorder.information == []


def test_a_later_behavior_wait_is_not_accused_of_repeating_arrival(monkeypatch, recorder) -> None:
    _fire(monkeypatch, _ready())

    assert recorder.warnings == []
    assert recorder.information == []


def test_an_exact_satisfied_predicate_gets_conditional_information(monkeypatch, recorder) -> None:
    _fire(monkeypatch, _ready(), waited_for="Ready")

    assert recorder.warnings == []
    assert len(recorder.information) == 1
    assert "if that is all this check needs" in recorder.information[0]
    assert "later behavior" in recorder.information[0]


def test_a_different_positive_predicate_is_not_a_repeated_check(monkeypatch, recorder) -> None:
    _fire(monkeypatch, _ready(), waited_for="Dismissed")

    assert recorder.warnings == []
    assert recorder.information == []


def test_an_unmet_wait_never_claims_its_requested_state_is_available(monkeypatch, recorder) -> None:
    _fire(monkeypatch, {**_ready(), "settled_unmet": True}, waited_for="Ready")

    assert recorder.warnings == []
    assert recorder.information == []
