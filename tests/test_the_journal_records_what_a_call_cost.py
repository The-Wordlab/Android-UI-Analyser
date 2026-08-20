"""The session journal must answer "how long did that take" after the fact.

A live run produced 40 recorded actions, none of them carrying a timestamp, an elapsed
time or an outcome, and no wait at all — so "how long did that tap cost" could only be
answered from the device's own (rotting) log ring buffer.  These tests pin the three
things that fixed it: an action records what it COST, a wait can be journaled without
entering the replayable journal, and the whole thing renders as an access log.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import StabilityTimeout
from android_ui_analyser.flows import Flow, parse_flow_yaml, render_flow_yaml
from android_ui_analyser.memory import (
    AppMemoryStore,
    CallRecord,
    RouteStep,
    SessionState,
    render_call_log,
)
from conftest import FakeDevice, make_config, make_engine

PACKAGE = "com.example.catalog"
SERIAL = "journal-timing-host-only"
# A fixed instant so rendered lines are assertable (the log renders it in local time,
# so the expectation is derived the same way rather than hard-coded).
STARTED_MS = 1786007643412


def _store(tmp_path: Path) -> AppMemoryStore:
    config = make_config(memory={"dir": str(tmp_path / "memory")})
    return AppMemoryStore(config.memory)


def test_a_recorded_action_carries_what_it_cost(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, SessionState(package=PACKAGE))

    store.observe_action(
        SERIAL,
        RouteStep(kind="tap", label="Continue", package=PACKAGE),
        started_at_ms=STARTED_MS,
        elapsed_ms=412,
        outcome="ok",
    )

    recorded = store.load_session(SERIAL).recent[-1]
    assert recorded.started_at_ms == STARTED_MS
    assert recorded.elapsed_ms == 412
    assert recorded.outcome == "ok"
    # The ask must stay distinguishable from the cost.
    assert recorded.timeout_ms is None


def test_a_recorded_action_also_lands_in_the_access_log(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, SessionState(package=PACKAGE))

    store.observe_action(
        SERIAL,
        RouteStep(kind="tap", label="Continue", package=PACKAGE),
        started_at_ms=STARTED_MS,
        elapsed_ms=412,
        outcome="ok",
    )

    calls = store.load_session(SERIAL).calls
    assert [(c.kind, c.elapsed_ms, c.outcome) for c in calls] == [("tap", 412, "ok")]


def test_a_wait_is_journaled_but_never_becomes_a_replayable_step(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_session(SERIAL, SessionState(package=PACKAGE))

    store.record_call(
        SERIAL,
        CallRecord(
            kind="wait-for",
            target="Welcome",
            started_at_ms=STARTED_MS,
            elapsed_ms=8003,
            outcome="timeout",
            package=PACKAGE,
        ),
    )

    sess = store.load_session(SERIAL)
    assert [(c.kind, c.outcome, c.elapsed_ms) for c in sess.calls] == [
        ("wait-for", "timeout", 8003)
    ]
    # Route learning and `flow save` read these two; a wait in either would replay as
    # navigation or change an edge's identity.
    assert sess.recent == []
    assert sess.pending == []
    # A journaled wait must not consume the arrival-proof watermark either.
    assert sess.next_capture_order == 0


def test_an_old_record_without_timing_still_loads() -> None:
    legacy = (
        '{"package": "com.example.catalog", "current_screen": "home", '
        '"recent": [{"kind": "tap", "label": "Continue", "package": "com.example.catalog"}], '
        '"pending": [], "capture_segment": 0, "next_capture_order": 1}'
    )

    sess = SessionState.model_validate_json(legacy)

    assert len(sess.recent) == 1
    assert sess.recent[0].started_at_ms is None
    assert sess.recent[0].elapsed_ms is None
    assert sess.recent[0].outcome is None
    assert sess.calls == []


def test_timing_never_reaches_a_saved_flow_or_the_durable_map() -> None:
    step = RouteStep(
        kind="tap",
        label="Continue",
        package=PACKAGE,
        started_at_ms=STARTED_MS,
        elapsed_ms=412,
        outcome="ok",
    )

    text = render_flow_yaml(Flow(name="probe", app=PACKAGE, steps=[step]))
    assert "elapsed_ms" not in text
    assert "started_at_ms" not in text
    assert "outcome" not in text
    reparsed = parse_flow_yaml(text, name="probe")
    assert [s.kind for s in reparsed.steps] == ["tap"]
    assert reparsed.steps[0].label == "Continue"

    durable = AppMemoryStore._route_step(step, PACKAGE)
    assert durable.started_at_ms is None
    assert durable.elapsed_ms is None
    assert durable.outcome is None


def test_the_access_log_shows_a_call_and_its_answer() -> None:
    lines = render_call_log(
        [
            CallRecord(
                kind="tap",
                target="'Continue'",
                started_at_ms=STARTED_MS,
                elapsed_ms=412,
                outcome="ok",
                package=PACKAGE,
                screen="details",
            ),
            CallRecord(
                kind="wait-for",
                target="'Welcome'",
                started_at_ms=STARTED_MS + 500,
                elapsed_ms=8003,
                outcome="timeout",
                package=PACKAGE,
            ),
        ]
    )

    assert len(lines) == 2
    # The request half: when, what was called, on what.
    expected_ts = (
        datetime.fromtimestamp(STARTED_MS / 1000).astimezone().isoformat(timespec="milliseconds")
    )
    assert lines[0].startswith(expected_ts)
    assert "tap" in lines[0] and "'Continue'" in lines[0]
    # The response half: what came back and what it cost.
    assert "ok" in lines[0] and "412ms" in lines[0]
    assert "screen=details" in lines[0]
    assert "timeout" in lines[1] and "8003ms" in lines[1]


def test_the_access_log_renders_the_replayable_journal_too() -> None:
    lines = render_call_log(
        [
            RouteStep(
                kind="tap",
                label="Continue",
                package=PACKAGE,
                started_at_ms=STARTED_MS,
                elapsed_ms=412,
                outcome="ok",
            ),
            RouteStep(kind="tap", label="Skip", package=PACKAGE),
        ]
    )

    assert "tap 'Continue'" in lines[0]
    assert "412ms" in lines[0]
    # An untimed legacy record still renders one line rather than crashing the log.
    assert "tap 'Skip'" in lines[1]


# --------------------------------------------------------------------- through the engine
#
# The store above is only half the answer: a journal nobody writes to reports zero forever.
# These drive the real engine against a fake device, so the wiring — not just the schema —
# is what is under test.


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    return make_engine(
        device=device,
        memory={"dir": str(tmp_path / "memory")},
        perf={"settle_profiles": False},
    )


def _tappable() -> FakeDevice:
    return FakeDevice(
        hierarchy_xml=(
            '<hierarchy rotation="0">'
            '<node text="Continue" resource-id="com.example.catalog:id/go" class="Button" '
            'clickable="true" bounds="[0,0][300,120]" />'
            "</hierarchy>"
        ),
        package=PACKAGE,
        text_index={"Continue": (0, 0, 300, 120)},
    )


def test_a_tap_journals_the_cost_the_caller_was_handed(tmp_path: Path) -> None:
    device = _tappable()
    engine = _engine(tmp_path, device)
    engine.analyze()

    result = engine.tap(selector={"text": "Continue"}, observe=False)

    assert result.wall_ms is not None
    store = engine._memory
    assert store is not None
    session = store.load_session(device.serial)
    # The replayable record and the access log must agree with the response: one number for
    # one call, or a latency review argues with the response the agent already read.
    assert session.recent[-1].elapsed_ms == result.wall_ms
    assert session.recent[-1].started_at_ms is not None
    assert [(c.kind, c.elapsed_ms, c.outcome) for c in session.calls] == [
        ("tap", result.wall_ms, "ok")
    ]
    assert session.calls[-1].target == "'Continue'"


def test_a_wait_is_journaled_by_the_engine_without_becoming_a_step(tmp_path: Path) -> None:
    device = _tappable()
    engine = _engine(tmp_path, device)
    engine.analyze()

    result = engine.wait(for_="Never appears", timeout_ms=10)

    assert result.ok is False
    store = engine._memory
    assert store is not None
    session = store.load_session(device.serial)
    line = session.calls[-1]
    assert line.kind == "wait"
    assert line.elapsed_ms is not None  # the whole point: the wait's cost is on record
    assert line.outcome == "timeout"
    # A wait is not navigation and must never be replayed as such.
    assert session.recent == []
    assert session.pending == []


def test_a_wait_that_raises_is_journaled_before_the_exception_leaves(tmp_path: Path) -> None:
    """`wait-stable`/`wait-changed` end by raising — the costliest wait must not vanish."""
    device = _tappable()
    engine = _engine(tmp_path, device)
    engine.analyze()

    with pytest.raises(StabilityTimeout):
        engine.wait_changed(timeout_ms=30, interval_ms=10)

    store = engine._memory
    assert store is not None
    line = store.load_session(device.serial).calls[-1]
    assert line.kind == "wait-changed"
    assert line.outcome == "timeout"
    assert line.elapsed_ms is not None


def test_the_rendered_log_is_readable_from_the_store(tmp_path: Path) -> None:
    device = _tappable()
    engine = _engine(tmp_path, device)
    engine.analyze()
    engine.tap(selector={"text": "Continue"}, observe=False)
    engine.wait(for_="Never appears", timeout_ms=10)

    store = engine._memory
    assert store is not None
    lines = store.call_log(device.serial)

    assert len(lines) == 2
    assert "tap 'Continue' -> ok" in lines[0]
    assert lines[1].split(" -> ")[0].endswith("wait")
    assert all("ms" in line for line in lines)


def test_a_journaled_phrase_never_carries_pii(tmp_path: Path) -> None:
    """The log records prose the tool did not choose — so it redacts like the map does."""
    store = _store(tmp_path)
    store.save_session(SERIAL, SessionState(package=PACKAGE))

    store.record_call_cost(
        SERIAL,
        kind="wait",
        elapsed_ms=8003,
        outcome="timeout",
        target="someone@example.com",
        detail="waited for someone@example.com",
    )

    line = store.load_session(SERIAL).calls[-1]
    assert line.target == "<redacted>"
    assert line.detail == "<redacted>"
    assert "example.com" not in " ".join(render_call_log([line]))


def test_a_session_review_shows_the_per_call_timeline(tmp_path: Path) -> None:
    """The review counts calls; without this it could not say which one was slow."""
    device = _tappable()
    engine = _engine(tmp_path, device)
    engine.session_start("verify the catalog opens")
    engine.tap(selector={"text": "Continue"}, observe=False)
    engine.wait(for_="Never appears", timeout_ms=10)

    review = engine.session_review()

    log = review["call_log"]
    assert [line.split(" -> ")[0].split(" ", 1)[1] for line in log] == ["tap 'Continue'", "wait"]
    assert "-> ok" in log[0]
    assert "-> timeout" in log[1]
