"""Waiting twice is the expensive mistake, and the advice has to differ from waiting once.

Run 6 of the fresh-agent series (2026-08-10) did this:

    tap-and-analyze --rid bottomBarAppsHub --until 'text:Search'     # 2042ms, satisfied
    wait-and-analyze --after-change --until '!text:Loading'          # 25256ms

The second call cost 25.3s — more than a third of the whole run — and bought nothing: the search
field it went on to use was already on the screen the first call returned. For comparison, the
entire flow through to the search verdict takes 5.7s when each wait names the element it is about
to act on.

Two distinct mistakes, so two messages:

* Waiting after a plain action → "pass --until to the action".
* Waiting after a call that ALREADY waited → that advice is what they just followed, so saying it
  again is noise. They need to hear that they are re-reading a settled screen, and that a
  screen-wide predicate (`!text:Loading`) waits for the whole page while `rid:<target>` returns as
  soon as the one element they want exists.

The signal is the journal entry kind. `await_outcome` is attached to the emitted result and never
reaches the journal, and a global `--until` is recorded as its own `await_predicate` entry — so
after `tap --until X` the newest entry is the await, not the tap.
"""

from __future__ import annotations

from typing import Any

import pytest

import android_ui_analyser.cli as cli_mod


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, msg: str, *args: Any) -> None:
        self.messages.append(msg % args if args else msg)


@pytest.fixture
def warned(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(cli_mod, "logger", recorder)
    return recorder


def _fire(monkeypatch: pytest.MonkeyPatch, events: list[dict]) -> None:
    """Drive the lint against a scripted journal, with no device anywhere."""
    import sys
    import types

    import android_ui_analyser

    fake_journal = types.SimpleNamespace(read_since=lambda *a, **k: events)
    # `from . import journal` reads the attribute off the package once it has been imported, so
    # patching sys.modules alone passes in isolation and fails in a full run.
    monkeypatch.setattr(android_ui_analyser, "journal", fake_journal, raising=False)
    monkeypatch.setitem(sys.modules, "android_ui_analyser.journal", fake_journal)

    engine = types.SimpleNamespace(
        config=types.SimpleNamespace(cache=types.SimpleNamespace(dir="/tmp/none")),
        device=types.SimpleNamespace(serial="emulator-0"),
    )
    cli_mod._warn_if_wait_could_have_been_until(engine, None)


_ACTION = {"cmd": "tap", "ok": True, "result": {"action": "tap", "observation": {"elements": []}}}
_AWAIT = {"cmd": "await_predicate", "ok": True, "result": {"action": "await"}}


def test_waiting_after_an_until_says_you_already_waited(
    monkeypatch: pytest.MonkeyPatch, warned: _Recorder
) -> None:
    _fire(monkeypatch, [_ACTION, _AWAIT])

    assert warned.messages, "the expensive case must not be the silent one"
    message = warned.messages[0]
    assert "already waited with `--until`" in message, message
    assert "pass it to the action instead" not in message, "that is what they just did"


def test_it_names_the_cheaper_predicate(
    monkeypatch: pytest.MonkeyPatch, warned: _Recorder
) -> None:
    _fire(monkeypatch, [_ACTION, _AWAIT])

    message = warned.messages[0]
    assert "rid:<target>" in message, message
    assert "!text:Loading" in message, "the screen-wide predicate is the thing being corrected"


def test_waiting_after_a_plain_action_still_says_to_fold_it_in(
    monkeypatch: pytest.MonkeyPatch, warned: _Recorder
) -> None:
    _fire(monkeypatch, [_ACTION])

    message = warned.messages[0]
    assert "this wait follows `tap`" in message, message
    assert "pass it to the action instead" in message, message


def test_the_action_is_named_not_the_await_that_followed_it(
    monkeypatch: pytest.MonkeyPatch, warned: _Recorder
) -> None:
    """`--until` journals an `await` entry, which would otherwise be reported as the action."""
    _fire(monkeypatch, [_ACTION, _AWAIT])

    assert "`await`" not in warned.messages[0]


def test_an_empty_journal_says_nothing(
    monkeypatch: pytest.MonkeyPatch, warned: _Recorder
) -> None:
    _fire(monkeypatch, [])

    assert warned.messages == []
