"""Nothing told an agent it had stopped making progress and started going in circles.

A live 12-minute run relaunched the app seven times and walked the same three taps four times,
looking for content the screen did not contain. Every individual call succeeded, so every
individual response looked fine. `aua session review` would have named the churn immediately —
but only if the agent chose to call it, and an agent convinced it is one tap from success does
not stop to audit itself.

So the churn is now named on the responses the agent already reads. Two shapes, both taken from
that run:

* **Relaunching.** A relaunch resets app state — it cost that run its login — and it cannot
  change what a screen contains. Three in one window is a symptom, not a strategy.
* **Repeating.** The same call on the same target three times means the first answer was already
  the true one.

Coaching is additive here: it appends alongside whatever command-specific hint also applies,
because "you are looping" and "reuse that observation" are both worth saying.
"""

from __future__ import annotations

from android_ui_analyser.coaching import looping_advice


def _launch(n: int) -> list[dict[str, object]]:
    return [{"cmd": "app_launch_and_analyze", "args": {"package": "com.example.app"}}] * n


def _tap(label: str, n: int = 1) -> list[dict[str, object]]:
    return [{"cmd": "tap_and_analyze", "args": {"label": label}}] * n


def test_a_quiet_productive_run_is_not_nagged() -> None:
    history = _launch(1) + _tap("Apps") + _tap("My Apps") + _tap("Drafts")
    assert looping_advice(history) is None


def test_three_relaunches_are_named_as_churn() -> None:
    history = _launch(1) + _tap("Apps") + _launch(1) + _tap("Apps") + _launch(1)
    advice = looping_advice(history)
    assert advice is not None
    assert advice["id"] == "session_looping"
    assert "relaunch" in advice["message"]
    # The count is the part that makes it undeniable rather than naggy.
    assert "3" in advice["message"]


def test_two_relaunches_are_still_ordinary_work() -> None:
    # A second launch is a normal recovery. Only a pattern is worth interrupting for.
    assert looping_advice(_launch(1) + _tap("Apps") + _launch(1)) is None


def test_the_same_tap_three_times_is_named() -> None:
    history = _tap("Drafts") + _tap("Create app") + _tap("Drafts") + _tap("Back") + _tap("Drafts")
    advice = looping_advice(history)
    assert advice is not None
    assert advice["id"] == "session_looping"
    assert "Drafts" in advice["message"]


def test_relaunching_outranks_repeating_when_both_hold() -> None:
    # A relaunch destroys state; a repeated tap only wastes a call. Name the worse one.
    history = _launch(3) + _tap("Drafts", 3)
    advice = looping_advice(history)
    assert advice is not None
    assert "relaunch" in advice["message"]


def test_different_targets_are_not_a_repeat() -> None:
    history = _tap("One") + _tap("Two") + _tap("Three") + _tap("Four") + _tap("Five")
    assert looping_advice(history) is None


def test_the_advice_points_at_a_real_next_move() -> None:
    advice = looping_advice(_launch(3))
    assert advice is not None
    assert advice["recommended_call"].strip()
