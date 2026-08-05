"""AUA remembers what a control cost last time, tells the agent, and waits longer itself.

The pre-existing learning was one EMA per *action kind* (tap/swipe/key), app-wide, held in a dict
built fresh in ``Engine.__init__`` — so it reset every run, and it averaged a 40ms same-screen tap
together with a login button that waits on a network round-trip. ``perf.py`` then capped it at
1.6s, while real screens in this app take 18-60s. A per-kind number cannot answer the question an
agent actually has, which is "what will *this* button on *this* screen cost me".

Timing is stored per (screen, control), scoped to the flag context, and used two ways: surfaced in
``meta.slow_controls`` when the agent arrives, and spent as a deadline when it acts.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.memory import AppMemoryStore
from conftest import make_config


def _store(tmp_path: Path) -> AppMemoryStore:
    cfg = make_config(memory={"enabled": True, "dir": str(tmp_path)})
    return AppMemoryStore(cfg.memory)


P = "com.example.app"


def _seeded(tmp_path: Path) -> AppMemoryStore:
    from test_memory import HOME, _elements

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="welcome")
    return store


def test_a_timing_is_remembered_per_screen_and_control(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=5000.0,
                               outcome="changed")

    t = store.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.n == 1
    assert t.ema_ms == 5000.0 and t.max_ms == 5000.0

    # A different control on the same screen is a different cost, not the same one.
    assert store.action_timing(P, screen="welcome", control="buttonCancel") is None


def test_it_survives_a_new_store_instance(tmp_path: Path) -> None:
    """The old profile died with the process; this is the half that makes it *memory*."""
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=5000.0)

    fresh = _store(tmp_path)
    t = fresh.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.ema_ms == 5000.0


def test_repeated_samples_keep_the_worst_case_not_just_the_average(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    for ms in (5000.0, 1000.0, 1000.0):
        store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=ms)

    t = store.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.n == 3
    assert t.max_ms == 5000.0, "a deadline built from the mean is too short half the time"
    assert t.ema_ms < 5000.0, "and the average still tracks the recent trend"


def test_slow_controls_are_offered_worst_first_and_the_fast_ones_are_not(tmp_path: Path) -> None:
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=6000.0)
    store.record_action_timing(P, screen="welcome", control="buttonSlowish", ms=2000.0)
    store.record_action_timing(P, screen="welcome", control="linkFast", ms=40.0)

    rows = store.slow_controls(P, screen="welcome")
    assert [r["control"] for r in rows] == ["buttonContinue", "buttonSlowish"], (
        "worst first, and an ordinary sub-second tap is not a warning"
    )
    assert rows[0]["n"] == 1, "sample count rides along so one observation is not read as ten"


def test_a_timeout_is_recorded_rather_than_discarded(tmp_path: Path) -> None:
    """The coarse EMA refuses timeouts because they poison an app-wide average. Per control,
    "this one timed out" is exactly what the next run needs to be told."""
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=8000.0,
                               outcome="unchanged")

    t = store.action_timing(P, screen="welcome", control="buttonContinue")
    assert t is not None and t.last_outcome == "unchanged"


def test_an_unknown_screen_or_control_records_nothing(tmp_path: Path) -> None:
    # No screen record → no home for the timing. Silently filing it under a name that means
    # something else next run would be worse than not learning it, because it gets spent.
    store = _store(tmp_path)
    store.record_action_timing(P, screen="never-seen", control="buttonContinue", ms=5000.0)
    assert store.action_timing(P, screen="never-seen", control="buttonContinue") is None


def test_timings_do_not_leak_across_flag_contexts(tmp_path: Path) -> None:
    """An experiment arm that adds a network call changes the cost; averaging arms describes
    neither. CLAUDE.md requires learned data be scoped to the exact flag context."""
    store = _seeded(tmp_path)
    store.record_action_timing(P, screen="welcome", control="buttonContinue", ms=5000.0)

    assert store.action_timing(P, screen="welcome", control="buttonContinue",
                               context_id="flags:experiment-on") is None
