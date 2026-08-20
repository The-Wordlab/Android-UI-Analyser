"""The caller's own think time is measured, reported, and used to size waits — under a 5s cap.

The tests use a rounded synthetic trace whose gaps range from a few seconds to a deliberately slow
turn. Nothing is pending during those gaps; they model what it costs the caller to read a response,
decide, and write the next call.

Two things are done with the measurement, and only two.

**The ceiling moves down, never up.** `perf.max_wait_ms` (5000ms) is a standing decision and
the hard maximum: an agent that needs longer must make another call rather than hold one long
wait. So the adaptive ceiling ranges over ``[wait_ceiling_min_ms, perf.max_wait_ms]`` and the
measurement is only ever allowed to *shorten* it. That is still worth having: a shell script
whose re-call costs ~4s of tool time and no thinking should not inherit the budget that exists
for callers that think, and a caller that has proved it is slow should not be told its 5s
ceiling was picked for it arbitrarily. This file pins the cap on every route into it —
adaptive, cold, config-pinned and env-pinned alike — because a ceiling with an exception in it
reads as a guarantee and is not one.

**The caller is told when its screen is gone.** The synthetic fixture moves to the next screen
during a thinking gap. Comparing the fingerprint stamped when the last call returned against the
one this call already produced answers that for free, with no extra device read.

What is deliberately *not* adopted: an action-bound `--until` still refuses an absence-only
predicate. `absence-satisfied` is honest for a standalone await — "what I left is gone" is a
real, weaker fact — but an action-bound predicate exists to prove the action landed somewhere,
and absence cannot prove arrival. See `TestAbsenceIsHonestButNotArrival`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.caller_latency import (
    IDLE_GAP_MS,
    RECALL_TOOL_MS,
    CallerLatencyStore,
)
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.perf import WAIT_CEILING_ENV, wait_ceiling_ms
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config, make_engine

# Rounded synthetic caller gaps, in seconds.
SYNTHETIC_GAPS_S = [6.0, 11.0, 9.0, 10.0, 7.0, 12.0, 8.0, 35.0, 8.0, 30.0]

#: The standing cap. Not a tuning parameter of this feature: the feature may only move the
#: ceiling *below* it.
CAP_MS = 5_000

HOME_XML = (
    '<hierarchy rotation="0">'
    '<node index="0" text="Start" resource-id="com.example.app:id/start"'
    ' class="android.widget.Button" package="com.example.app" clickable="true"'
    ' bounds="[0,0][200,100]" />'
    "</hierarchy>"
)
NEXT_XML = (
    '<hierarchy rotation="0">'
    '<node index="0" text="Allow notifications" resource-id="com.example.app:id/prompt"'
    ' class="android.widget.Button" package="com.example.app" clickable="true"'
    ' bounds="[0,0][200,100]" />'
    "</hierarchy>"
)


class _Clock:
    """A hand-cranked wall clock: the gap under test is a difference of wall times."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _store(tmp_path: Path, clock: _Clock, key: str = "agent-1") -> CallerLatencyStore:
    return CallerLatencyStore(tmp_path / "state", key, clock=clock)


def _turn(store: CallerLatencyStore, clock: _Clock, think_s: float) -> Any:
    """One caller round trip: think for *think_s*, call, respond."""
    clock.advance(think_s)
    facts = store.open_turn()
    store.close_turn()
    return facts


def _measured(tmp_path: Path, clock: _Clock, gaps: list[float]) -> Any:
    """A store that has seen *gaps*, and its settled profile."""
    store = _store(tmp_path, clock)
    _turn(store, clock, 0.0)
    for gap in gaps:
        _turn(store, clock, gap)
    return store


# --------------------------------------------------------------------- (a) measure the gap


class TestTheCallerGapIsMeasuredNotGuessed:
    def test_the_second_call_of_the_very_first_run_already_knows_the_gap(
        self, tmp_path: Path
    ) -> None:
        """Caller latency is one global property of the caller, not a per-control history.

        A design that waits for a second *run* to learn it throws away the whole first run,
        which is where a fresh agent does its most expensive thinking.
        """
        clock = _Clock()
        store = _store(tmp_path, clock)

        first = store.open_turn()
        assert first.profile.gap_ms is None, "there is no previous call to measure from"
        assert first.profile.samples == 0
        store.close_turn()

        second = _turn(store, clock, 11.0)
        assert second.profile.gap_ms == 11_000
        assert second.profile.samples == 1
        assert second.profile.ema_ms == pytest.approx(11_000, rel=0.01)

    def test_the_estimate_survives_the_process_that_measured_it(self, tmp_path: Path) -> None:
        """Every CLI call is a new process, so an in-memory EMA learns once and forgets."""
        clock = _Clock()
        _turn(_store(tmp_path, clock), clock, 0.0)
        for gap in SYNTHETIC_GAPS_S[:4]:
            _turn(_store(tmp_path, clock), clock, gap)

        reopened = _store(tmp_path, clock).profile()
        assert reopened.samples == 4
        assert reopened.ema_ms is not None and 6_000 < reopened.ema_ms < 14_000

    def test_a_walk_away_gap_does_not_poison_the_estimate(self, tmp_path: Path) -> None:
        """A human leaving for lunch is not the caller generating.

        `IDLE_GAP_MS` sits well past the synthetic slow turn, while an overnight gap can never
        move the ceiling.
        """
        clock = _Clock()
        store = _measured(tmp_path, clock, SYNTHETIC_GAPS_S)
        learned = store.profile()

        idle = _turn(store, clock, IDLE_GAP_MS / 1000.0 + 60.0)

        assert idle.profile.gap_ignored == "idle", idle.profile
        assert idle.profile.samples == learned.samples, "an idle gap became a sample"
        assert idle.profile.ema_ms == pytest.approx(learned.ema_ms)
        assert idle.profile.gap_ms is not None, "the gap is still reported, just not learned from"

    def test_a_clock_that_went_backwards_is_ignored_rather_than_believed(
        self, tmp_path: Path
    ) -> None:
        clock = _Clock()
        store = _store(tmp_path, clock)
        _turn(store, clock, 8.69)
        _turn(store, clock, 8.69)

        clock.advance(-30.0)
        facts = store.open_turn()

        assert facts.profile.gap_ignored == "clock"
        assert facts.profile.samples == 1

    def test_two_agents_on_one_device_do_not_learn_each_others_gaps(self, tmp_path: Path) -> None:
        """Interleaved processes share the device, not the think time.

        A fast script and a slow model can legitimately drive the same emulator at once. Each
        measures from its *own* last response, so keying the record on the caller is what keeps
        the script from dragging the model's ceiling down to its own — and what keeps two
        writers from landing in one file.
        """
        clock = _Clock()
        slow = _store(tmp_path, clock, key="agent-slow")
        fast = _store(tmp_path, clock, key="agent-fast")
        for store in (slow, fast):
            store.open_turn()
            store.close_turn()

        for _round in range(4):
            for tick in range(12):  # one second per fast turn, twelve per slow turn
                clock.advance(1.0)
                fast.open_turn()
                fast.close_turn()
                if tick == 11:
                    slow.open_turn()
                    slow.close_turn()

        assert slow.profile().ema_ms == pytest.approx(12_000, rel=0.05)
        assert fast.profile().ema_ms == pytest.approx(1_000, rel=0.05)
        assert json.loads(slow.path.read_text(encoding="utf-8"))["samples"] == 4
        assert json.loads(fast.path.read_text(encoding="utf-8"))["samples"] == 48

    def test_a_corrupt_file_is_treated_as_no_history_rather_than_crashing_the_call(
        self, tmp_path: Path
    ) -> None:
        clock = _Clock()
        store = _store(tmp_path, clock)
        _turn(store, clock, 9.56)
        store.path.write_text("{not json", encoding="utf-8")

        facts = store.open_turn()

        assert facts.profile.samples == 0
        assert facts.profile.gap_ms is None

    def test_a_warm_engine_does_not_price_one_agent_from_another(self) -> None:
        """The daemon's Engine outlives the client that built it, and adopts the next one.

        A cached record key would survive that hand-off and quote the previous agent's thinking
        speed to the next one — which is the same class of bug the owner reset already exists to
        prevent for ids and prefetched trees.
        """
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine._lease_owner_resolved = "agent-a"  # noqa: SLF001
        engine.open_caller_turn()
        first = engine._caller_latency_store().path  # noqa: SLF001

        engine._reset_owner_transient_state()  # noqa: SLF001
        engine._lease_owner_resolved = "agent-b"  # noqa: SLF001

        assert engine._caller_turn is None, "the previous client's turn was kept"  # noqa: SLF001
        assert engine._caller_latency_store().path != first  # noqa: SLF001

    def test_the_response_carries_the_measured_gap(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine.open_caller_turn()
        engine.close_caller_turn()
        engine._caller_turn = None  # a second process, same caller  # noqa: SLF001
        engine.open_caller_turn()

        report = engine.caller_turn_report()

        assert report is not None
        assert report["gap_ms"] is not None
        assert 0 < report["wait_ceiling_ms"] <= CAP_MS
        assert report["wait_ceiling_mode"] in {"cold", "adaptive", "fixed", "pinned"}


# ------------------------------------------------------------------ (b) adaptive, under a cap


class TestTheCeilingIsAdaptiveButNeverAboveTheCap:
    """`perf.max_wait_ms` is the hard maximum on every route. The measurement may only lower it.

    The worktree this was ported from argued the opposite — a 15s cold default and a 30s
    maximum, on the grounds that a re-call costs a whole caller gap and so a short wait is the
    losing trade. That reasoning is sound about *cost* and is why the gap is measured and
    reported at all; it is overridden here by a standing decision that a single aua call must
    never block a session for more than 5s. An agent that needs longer makes another call.
    """

    def test_a_cold_ceiling_is_the_cap_and_not_a_number_of_its_own(self) -> None:
        """With nothing measured, behaviour is exactly what it was before this feature.

        The cold path deliberately has no default of its own to get wrong: it is
        `perf.max_wait_ms`, so an install that never accumulates samples is unaffected.
        """
        cfg = make_config()
        ceiling, mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg)

        assert mode == "cold"
        assert ceiling == CAP_MS == cfg.perf.max_wait_ms

    def test_a_slow_caller_is_capped_rather_than_indulged(self, tmp_path: Path) -> None:
        """The measured caller would earn ~16s of ceiling if the cost argument decided it.

        It does not. This is the single most important assertion in the file: the adaptive
        estimate is an input to the cap, never a way around it.
        """
        clock = _Clock()
        profile = _measured(tmp_path, clock, SYNTHETIC_GAPS_S).profile()
        cfg = make_config()

        ceiling, mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg, profile)

        assert profile.recall_cost_ms > CAP_MS, "the fixture no longer exercises the cap"
        assert mode == "adaptive"
        assert ceiling == CAP_MS

    def test_a_scripted_caller_gets_a_shorter_ceiling_than_a_thinking_one(
        self, tmp_path: Path
    ) -> None:
        """This is what the measurement buys, now that it can only move the ceiling down.

        A shell script's re-call costs almost nothing but the aua call itself, so its ceiling
        collapses to about that — it does not inherit the budget that exists for callers that
        think between calls.
        """
        clock = _Clock()
        profile = _measured(tmp_path, clock, [0.04] * 6).profile()
        cfg = make_config()

        ceiling, mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg, profile)

        assert mode == "adaptive"
        assert ceiling == pytest.approx(RECALL_TOOL_MS, abs=200)
        assert ceiling < wait_ceiling_ms(cfg.perf.max_wait_ms, cfg)[0], (
            "a script inherited the cold budget"
        )

    def test_the_floor_still_covers_the_devices_own_transition_budget(self, tmp_path: Path) -> None:
        """A caller with no latency at all must not be given a ceiling shorter than a screen.

        `settle_total_max_ms` alone can spend 1.6s before a screen is settled, so the adaptive
        estimate has a floor as well as a cap.
        """
        clock = _Clock()
        profile = _measured(tmp_path, clock, [0.0]).profile()

        cfg = make_config(perf={"wait_ceiling_min_ms": 4_000})
        ceiling, _mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg, profile)

        assert ceiling == 4_000

    def test_a_floor_above_the_cap_does_not_beat_the_cap(self, tmp_path: Path) -> None:
        """The floor is a floor under the cap, not a second way to ask for a longer wait."""
        clock = _Clock()
        profile = _measured(tmp_path, clock, [0.0]).profile()

        cfg = make_config(perf={"wait_ceiling_min_ms": 30_000})
        ceiling, _mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg, profile)

        assert ceiling == CAP_MS

    def test_a_config_pin_cannot_exceed_the_cap(self) -> None:
        """`wait_ceiling_mode: fixed` opts out of adapting, not out of the ceiling."""
        cfg = make_config(perf={"wait_ceiling_mode": "fixed"})

        ceiling, mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg)

        assert (ceiling, mode) == (CAP_MS, "fixed")

    def test_a_pinned_ceiling_is_reproducible_across_days(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An adaptive budget is poison for a measurement run; naming a number opts out."""
        clock = _Clock()
        profile = _measured(tmp_path, clock, [30.0]).profile()
        cfg = make_config()

        monkeypatch.setenv(WAIT_CEILING_ENV, "4500")
        pinned, mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg, profile)

        assert (pinned, mode) == (4_500, "pinned")

    def test_an_env_pin_above_the_cap_is_still_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An env var is a convenience for sweeps, not an escape hatch from the ceiling.

        `perf.max_wait_ms` is the one number that raises the maximum, and it is config — so a
        sweep that wants a longer ceiling has to say so where the ceiling is defined.
        """
        cfg = make_config()
        monkeypatch.setenv(WAIT_CEILING_ENV, "45000")

        ceiling, mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg)

        assert (ceiling, mode) == (CAP_MS, "pinned")

    def test_a_junk_pin_is_ignored_rather_than_crashing_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = make_config()
        monkeypatch.setenv(WAIT_CEILING_ENV, "as-long-as-it-takes")

        assert wait_ceiling_ms(cfg.perf.max_wait_ms, cfg)[1] == "cold"

    def test_raising_the_configured_maximum_raises_the_adaptive_ceiling_with_it(
        self, tmp_path: Path
    ) -> None:
        """One knob, not two. The adaptive range is defined by `perf.max_wait_ms`."""
        clock = _Clock()
        profile = _measured(tmp_path, clock, SYNTHETIC_GAPS_S).profile()
        cfg = make_config(perf={"max_wait_ms": 12_000})

        ceiling, _mode = wait_ceiling_ms(cfg.perf.max_wait_ms, cfg, profile)

        assert ceiling == 12_000


class TestTheOneGateCarriesTheAdaptiveCeiling:
    """There is one clamp point, and the estimate is fed into it rather than beside it."""

    def test_the_single_gate_reports_the_capped_ceiling(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))

        effective, clamped_from, ceiling = engine._bounded_wait_ms(None)  # noqa: SLF001

        assert ceiling == CAP_MS
        assert effective == ceiling
        assert clamped_from is None

    def test_a_request_over_the_ceiling_is_clamped_and_says_so(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))

        effective, clamped_from, ceiling = engine._bounded_wait_ms(600_000)  # noqa: SLF001

        assert effective == ceiling == CAP_MS
        assert clamped_from == 600_000

    def test_a_request_inside_the_ceiling_is_left_alone(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))

        effective, clamped_from, _ceiling = engine._bounded_wait_ms(1_200)  # noqa: SLF001

        assert (effective, clamped_from) == (1_200, None)

    def test_the_gate_honours_a_scripted_callers_shorter_ceiling(self, tmp_path: Path) -> None:
        """The adaptive number has to arrive at the gate, or it is decoration.

        A request of 4800ms is inside the 5s cap and would sail through unclamped — it is only
        cut because this caller has *measured* itself fast enough that waiting that long is a
        worse trade than calling again.
        """
        clock = _Clock()
        store = _measured(tmp_path, clock, [0.04] * 6)
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine._caller_turn = store.open_turn()  # noqa: SLF001

        effective, clamped_from, ceiling = engine._bounded_wait_ms(4_800)  # noqa: SLF001

        assert ceiling == pytest.approx(RECALL_TOOL_MS, abs=200)
        assert effective == ceiling
        assert clamped_from == 4_800

    def test_a_background_job_is_still_exempt(self) -> None:
        """The exemption predates this and survives it: nobody is blocked on a job's wait."""
        import threading

        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine._job_cancel_event = threading.Event()  # noqa: SLF001

        assert engine._bounded_wait_ms(600_000) == (600_000, None, CAP_MS)  # noqa: SLF001

    def test_a_reader_can_tell_a_measured_run_from_an_adaptive_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        monkeypatch.setenv(WAIT_CEILING_ENV, "3000")
        engine.open_caller_turn()

        result = engine.await_predicate("text:NothingHere", timeout_ms=90_000, observe=False)

        assert result.wait_ceiling_ms == 3_000
        assert result.wait_ceiling_mode == "pinned"
        assert result.wait_clamped_from_ms == 90_000
        assert "call again" in (result.note or "").lower()

    def test_await_keeps_no_uncapped_default(self) -> None:
        """`await` is the wait an agent reaches for most often; an uncapped default undid it."""
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        _effective, _clamped, ceiling = engine._bounded_wait_ms(60_000)  # noqa: SLF001

        assert ceiling <= CAP_MS


# ----------------------------------------------------- (c) absence is honest, but not arrival


class TestAbsenceIsHonestButNotArrival:
    """A standalone await may report absence. An action-bound `--until` still may not.

    The worktree relaxed both, arguing that refusing absence pushes a first-time caller onto
    `--after-change`, which is relative-to-now and can miss a change that landed while the
    caller was composing the call. The first half of that is adopted: a standalone await that
    holds on negated terms alone now says `absence-satisfied` instead of claiming `satisfied`,
    which is strictly more honest than what main did.

    The second half is not. An action-bound predicate's entire job is to prove the action
    landed somewhere; "the thing I tapped went away" is compatible with a crash, a blank frame,
    or the wrong screen. Making it accept absence-only evidence would silently weaken every
    recorded arrival — and the same reasoning already guards flow and route arrivals, which
    still call `_parse_await_terms(require_positive=True)`. The refusal keeps a recovery call
    that now leads somewhere better than before, because the standalone await it recommends
    reports the weaker outcome under its own name.
    """

    def test_a_standalone_absence_only_await_is_named_absence_not_satisfied(
        self, tmp_path: Path
    ) -> None:
        dev = FakeDevice(hierarchy_xml=NEXT_XML, package="com.example.app", text_index={})
        cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
        eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

        out = eng.await_predicate("!text:Start", timeout_ms=5, poll_ms=1, observe=False)

        assert out.await_outcome == "absence-satisfied"
        assert out.ok, "the wait did what it was asked to; the caveat is not a failure"
        assert "absence" in (out.note or "").lower()

    def test_a_standalone_positive_term_still_reports_a_positive_match(
        self, tmp_path: Path
    ) -> None:
        dev = FakeDevice(hierarchy_xml=NEXT_XML, package="com.example.app")
        cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
        eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

        out = eng.await_predicate("text:Allow notifications", timeout_ms=5, poll_ms=1)

        assert out.await_outcome == "satisfied"

    def test_a_mixed_predicate_is_arrival_because_something_positive_held(
        self, tmp_path: Path
    ) -> None:
        """One positive term is enough: absence-satisfied is for predicates with none."""
        dev = FakeDevice(hierarchy_xml=NEXT_XML, package="com.example.app", text_index={})
        cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
        eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

        out = eng.await_predicate(
            "text:Allow notifications,!text:Start", timeout_ms=5, poll_ms=1, observe=False
        )

        assert out.await_outcome == "satisfied"

    def test_an_action_bound_absence_only_predicate_is_still_refused(self, tmp_path: Path) -> None:
        """The deliberate refusal survives the port. Absence is not arrival evidence."""
        dev = FakeDevice(package="com.example.app")
        cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
        eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

        with pytest.raises(UsageError):
            eng.await_predicate("!text:Start", timeout_ms=5, poll_ms=1, adopt_action=True)

        assert dev.calls == [], "a usage error must cost zero device calls"
        assert dev.hierarchy_calls == 0

    def test_the_refusal_still_describes_what_it_recommends(self) -> None:
        """The message has to stay accurate: it names a call whose outcome is now different.

        A refusal that recommends a standalone await must say what that await will answer,
        otherwise the caller retries expecting `satisfied` and branches on the wrong field.
        """
        from typer.testing import CliRunner

        from android_ui_analyser.cli import app

        result = CliRunner().invoke(
            app, ["--until", "!text:Start", "tap-and-analyze", "--rid", "nothing"]
        )
        combined = result.output + str(result.stderr or "")

        assert "positive arrival evidence" in combined, combined
        assert "await-and-analyze" in combined, combined
        assert "absence-satisfied" in combined, combined

    def test_a_predicate_with_no_terms_at_all_is_still_refused(self) -> None:
        """Accepting absence is not accepting nothing: an empty predicate waits for nothing."""
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))

        with pytest.raises(UsageError):
            engine.await_predicate("", timeout_ms=5, poll_ms=1)


# --------------------------------------------------------------- (d) the screen moved on


class TestACallerIsToldItsScreenWasReplaced:
    def test_a_replaced_screen_is_named_on_the_next_response(self) -> None:
        """A stale element id raises; a wholly stale *screen* said nothing at all.

        In this synthetic case the next screen arrives during the caller's gap, so the call that
        follows would otherwise reason about a screen that no longer exists.
        """
        device = FakeDevice(hierarchy_xml=HOME_XML)
        engine = make_engine(device=device)
        engine.open_caller_turn()
        engine.analyze()
        engine.close_caller_turn()

        device._xml = NEXT_XML  # noqa: SLF001 - the app moved on during the caller's gap
        engine._caller_turn = None  # noqa: SLF001 - the next call is a new process
        engine.open_caller_turn()
        engine.analyze()
        report = engine.caller_turn_report()

        assert report is not None
        assert report["previous_screen_gone"] is True
        assert report["previous_screen_age_ms"] is not None

    def test_a_screen_that_is_still_up_is_not_flagged(self) -> None:
        device = FakeDevice(hierarchy_xml=HOME_XML)
        engine = make_engine(device=device)
        engine.open_caller_turn()
        engine.analyze()
        engine.close_caller_turn()

        engine._caller_turn = None  # noqa: SLF001
        engine.open_caller_turn()
        engine.analyze()

        report = engine.caller_turn_report()
        assert report is not None
        assert report["previous_screen_gone"] is False

    def test_a_moved_screen_is_reported_not_raised(self) -> None:
        """The caller may legitimately want to act anyway, so this can never block the call."""
        device = FakeDevice(hierarchy_xml=HOME_XML)
        engine = make_engine(device=device)
        engine.open_caller_turn()
        engine.analyze()
        engine.close_caller_turn()

        device._xml = NEXT_XML  # noqa: SLF001
        engine._caller_turn = None  # noqa: SLF001
        engine.open_caller_turn()
        observed = engine.analyze()

        assert observed.elements, "the call still returned a usable screen"

    def test_the_first_call_of_a_session_says_nothing_at_all(self) -> None:
        """No gap and no previous screen means no block, not a block full of nulls.

        This rides on every response, so the cold call must not pay for it: there is nothing to
        compare, and a header of empty fields would also read as "measured, answer nothing".
        """
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine.open_caller_turn()
        engine.analyze()

        assert engine.caller_turn_report() is None

    def test_the_block_stays_sparse_once_there_is_something_to_say(self) -> None:
        """Only measured keys appear. An absent key and a null key do not mean the same thing."""
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine.open_caller_turn()
        engine.analyze()
        engine.close_caller_turn()
        engine._caller_turn = None  # noqa: SLF001
        engine.open_caller_turn()
        engine.analyze()

        report = engine.caller_turn_report()
        assert report is not None
        assert None not in report.values(), report


class TestTheStampSurvivesTheWarmDaemon:
    """The screen-gone comparison has to arm itself in the process that answers the caller.

    Found by running the CLI rather than the engine: through a warm daemon the work happens in
    another process, so the CLI's own engine has no observation cached and the stamp was written
    as None — every call, forever, with `previous_screen_gone` reading `null` and the whole
    feature quietly never arming. It looked fine in-process, which is why it needs its own test.
    """

    def test_a_stamp_can_come_from_the_payload_rather_than_a_local_observation(
        self, tmp_path: Path
    ) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine.open_caller_turn()
        assert engine._last_analyze_result is None, "this engine did no observing"  # noqa: SLF001

        engine.close_caller_turn("fp-from-the-daemons-answer")
        store = engine._caller_latency_store()  # noqa: SLF001
        record = json.loads(store.path.read_text(encoding="utf-8"))

        assert record["fingerprint"] == "fp-from-the-daemons-answer"

    def test_a_missing_stamp_is_not_silently_written_as_no_screen(self, tmp_path: Path) -> None:
        """The bug in one assertion: no observation anywhere must leave the key unset.

        Writing `fingerprint: None` would be worse than writing nothing, because the next call
        cannot tell "the last call showed no screen" from "the last call forgot to say".
        """
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine.open_caller_turn()
        engine.close_caller_turn(None)

        record = json.loads(engine._caller_latency_store().path.read_text("utf-8"))  # noqa: SLF001
        assert "fingerprint" not in record

    @pytest.mark.parametrize(
        "payload",
        [
            {"meta": {"fingerprint": "fp"}},
            {"observation": {"meta": {"fingerprint": "fp"}}},
            {"action": "tap", "observation": {"meta": {"fingerprint": "fp"}}},
        ],
    )
    def test_the_fingerprint_is_found_in_every_payload_shape(self, payload: dict) -> None:
        """`analyze` and an action-with-observation nest it differently; both must be read."""
        from android_ui_analyser.coaching import emitted_fingerprint

        assert emitted_fingerprint(payload) == "fp"

    def test_a_payload_with_no_screen_yields_no_fingerprint(self) -> None:
        from android_ui_analyser.coaching import emitted_fingerprint

        assert emitted_fingerprint({"ok": True, "action": "screenshot"}) is None


class TestTheBlockCostsNothingItHasNotEarned:
    def test_the_serialised_form_carries_no_null_keys(self) -> None:
        """A nested model dumps its own Nones; the enclosing filters cannot see inside it.

        Found the same way: the report is built sparse, then re-inflated to a full block of
        nulls by `CallerTurn`'s dump. That is token cost on every response, and
        `previous_screen_gone: null` reads as "checked, fine" when it means "nothing to check".
        """
        from android_ui_analyser.schema import CallerTurn

        dumped = CallerTurn(gap_ms=41, wait_ceiling_ms=5_000).model_dump(mode="json")

        assert dumped == {"gap_ms": 41, "wait_ceiling_ms": 5_000}

    def test_a_response_that_measured_nothing_carries_no_caller_block(self) -> None:
        """A call that never waits and has nothing measured should not pay for this at all."""
        from android_ui_analyser.coaching import attach_caller_turn
        from android_ui_analyser.schema import ActionResult

        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME_XML))
        engine.open_caller_turn()
        result = ActionResult(ok=True, action="screenshot", detail="out.png")

        attach_caller_turn(engine, result)

        assert result.caller is None
        assert "caller" not in result.model_dump(mode="json", exclude_none=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
