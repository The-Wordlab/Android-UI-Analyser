"""Waiting for a slow surface had no way to say whether it was a hang or a slow backend.

Observed: image-editing round trips run **2-4 minutes** each; one lane watched a GET hang 48
seconds before cancelling, another a 3-minute stall that resolved on retry. With no way to wait on
a *condition*, lanes polled every 15s, waited fixed intervals, or concluded "stuck" from a stale
frame. A coordinator twice had to ask a lane "is that a hang or a slow backend?" — because nothing
in the output distinguished them.

So the outcome is a named field, not something inferred from `ok`, and it has three values:

* ``satisfied`` — every term held.
* ``screen-changed`` — the foreground activity or package moved while waiting and the predicate is
  still unmet. Returns *immediately*: the surface being waited on is gone, so more waiting cannot
  help. This is what separates "an error dialog took over / we got kicked out" from "still working".
* ``timeout`` — budget spent, predicate unmet, same screen.

Two deliberate divergences from the request, both recorded here so nobody re-adds them:

**No `await-network-idle`.** That is what was originally asked for and it must not be built: this
app is never network idle — opening the app hub fires a fire-and-forget play call, analytics post
continuously, chat surfaces stream — so idleness would be a flaky proxy for "this is ready".

**`screen-changed` is keyed on the resumed activity/package, not on the element tree.** A streaming
surface rewrites its tree constantly; a tree-change trigger would abort every legitimate wait on
precisely the screens this exists for.

Per-term results come back either way, because *which* term is unmet is how a reader tells a failed
load from a slow one: spinner gone but results absent is a failure; spinner still present is
progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine, _parse_await_terms
from android_ui_analyser.errors import UsageError
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import Bounds
from conftest import FakeDevice, make_config

PKG = "com.example.app"


class ScriptedScreens(FakeDevice):
    """A device whose visible text and foreground activity change on a schedule.

    `find_text` is driven by a list of per-poll text sets, so a predicate can be watched turning
    true — which is the only way to test a wait without sleeping for real.
    """

    def __init__(self, *, frames: list[set[str]], activities: list[str] | None = None) -> None:
        super().__init__(package=PKG)
        self.frames = frames
        self.activities = activities or []
        self.polls = 0

    def _frame(self) -> set[str]:
        idx = min(self.polls, len(self.frames) - 1)
        return self.frames[idx]

    def find_text(self, text: str, **kwargs: Any) -> Bounds | None:
        self.polls += 1
        return (0, 0, 10, 10) if text in self._frame() else None

    def current_app(self) -> dict[str, str]:
        if not self.activities:
            return {"package": PKG, "activity": ".MainActivity"}
        idx = min(self.polls, len(self.activities) - 1)
        return {"package": PKG, "activity": self.activities[idx]}


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


# --------------------------------------------------------------- the predicate grammar


def test_terms_are_anded_and_negation_is_understood() -> None:
    terms = _parse_await_terms("rid:resultCard,!text:Generating")
    assert [(t.by, t.value, t.negated) for t in terms] == [
        ("rid", "resultCard", False),
        ("text", "Generating", True),
    ]


@pytest.mark.parametrize("field,expected", [("text", "text"), ("rid", "rid"), ("id", "rid"), ("desc", "desc")])
def test_the_field_vocabulary_matches_every_other_selector(field: str, expected: str) -> None:
    assert _parse_await_terms(f"{field}:Thing")[0].by == expected


@pytest.mark.parametrize("bad", ["", "   ", "Generating", "spinner", "when:Done", ":Done", "text:"])
def test_a_malformed_predicate_is_refused_rather_than_guessed(bad: str) -> None:
    """A predicate that quietly means something else is a second place to be wrong about a screen.

    `Generating` with no field is the tempting one — treating it as a text search would work often
    enough to be trusted and then silently mean the wrong thing for `!` terms.
    """
    with pytest.raises(UsageError):
        _parse_await_terms(bad)


# --------------------------------------------------------------- the three outcomes


def test_a_satisfied_predicate_says_so_and_exits_ok(tmp_path: Path) -> None:
    dev = ScriptedScreens(frames=[{"Generating"}, {"Generating"}, {"Result ready"}])
    eng = _engine(tmp_path, dev)

    out = eng.await_predicate("text:Result ready,!text:Generating", timeout_ms=5000, poll_ms=1)

    assert out.ok is True
    assert out.await_outcome == "satisfied"
    assert out.await_terms is not None
    assert all(t["satisfied"] for t in out.await_terms)
    assert out.elapsed_ms is not None


def test_a_timeout_names_the_terms_that_never_held(tmp_path: Path) -> None:
    """"unmet" is the hang-versus-slow-backend discriminator, so it must be in the output."""
    dev = ScriptedScreens(frames=[{"Generating"}])
    eng = _engine(tmp_path, dev)

    out = eng.await_predicate("text:Result ready,!text:Generating", timeout_ms=30, poll_ms=1)

    assert out.ok is False
    assert out.await_outcome == "timeout"
    unmet = [t["term"] for t in out.await_terms or [] if not t["satisfied"]]
    assert unmet == ["text:Result ready", "!text:Generating"]
    assert "unmet" in (out.detail or "")


def test_a_partly_satisfied_timeout_reads_as_a_failed_load(tmp_path: Path) -> None:
    """The spinner going away without results arriving is a different fact from a spinner stuck."""
    dev = ScriptedScreens(frames=[{"Generating"}, set()])
    eng = _engine(tmp_path, dev)

    out = eng.await_predicate("text:Result ready,!text:Generating", timeout_ms=40, poll_ms=1)

    assert out.await_outcome == "timeout"
    by_term = {t["term"]: t["satisfied"] for t in out.await_terms or []}
    assert by_term["!text:Generating"] is True, "the spinner did go away"
    assert by_term["text:Result ready"] is False, "and nothing arrived — a failed load"


def test_the_screen_moving_underneath_is_its_own_outcome(tmp_path: Path) -> None:
    """The surface is gone, so more waiting cannot help — and this is not a timeout."""
    dev = ScriptedScreens(
        frames=[{"Generating"}],
        activities=[".EditorActivity", ".EditorActivity", ".ErrorDialogActivity"],
    )
    eng = _engine(tmp_path, dev)

    out = eng.await_predicate("text:Result ready", timeout_ms=30_000, poll_ms=1)

    assert out.ok is False
    assert out.await_outcome == "screen-changed"
    assert ".ErrorDialogActivity" in (out.detail or "")
    assert ".EditorActivity" in (out.detail or ""), "must say what it was, not only what it is"


def test_screen_changed_returns_early_rather_than_spending_the_budget(tmp_path: Path) -> None:
    """The 2-4 minute budgets these waits carry are exactly why an early exit matters."""
    dev = ScriptedScreens(
        frames=[{"Generating"}], activities=[".EditorActivity", ".SomewhereElse"]
    )
    eng = _engine(tmp_path, dev)

    out = eng.await_predicate("text:Never", timeout_ms=600_000, poll_ms=1)

    assert out.await_outcome == "screen-changed"
    assert (out.elapsed_ms or 0) < 5_000, f"burned the budget anyway: {out.elapsed_ms}ms"


def test_a_satisfied_predicate_wins_over_a_moved_screen(tmp_path: Path) -> None:
    """Arriving somewhere new is how a wait normally succeeds, so satisfaction is checked first.

    Otherwise `await 'text:Settings'` after a navigation would report `screen-changed` for the very
    transition it was waiting for.
    """
    dev = ScriptedScreens(frames=[{"Settings"}], activities=[".SettingsActivity"])
    dev.activities = [".SettingsActivity"]
    eng = _engine(tmp_path, dev)

    out = eng.await_predicate("text:Settings", timeout_ms=5000, poll_ms=1)

    assert out.await_outcome == "satisfied"


def test_a_device_that_cannot_report_its_activity_does_not_fake_a_navigation(
    tmp_path: Path,
) -> None:
    """An unreadable foreground must not read as "the screen moved" — that would abort real waits."""

    class Mute(ScriptedScreens):
        def current_app(self) -> dict[str, str]:
            raise RuntimeError("adb hiccup")

    eng = _engine(tmp_path, Mute(frames=[{"Generating"}]))
    out = eng.await_predicate("text:Result ready", timeout_ms=30, poll_ms=1)

    assert out.await_outcome == "timeout", "a failed read is not a navigation"
