"""Waiting is bounded and deliberate, not open-ended and accidental.

The regression is modeled by this timing sequence:

    20:08:06  CALL app restart-and-analyze
    20:08:11  RESP wall= 5.26s  elements=3 shown=0     <- empty, and reported ok
    20:08:22  CALL wait-and-analyze --after-change --timeout-ms 45000
    20:09:04  RESP wall=42.36s  "changed and confirmed settled after 41072ms"

Three defects compound there:

* an action returned an observation with nothing visible in it, and called that success;
* the agent was allowed to ask for a 45s wait, so one call blocked for 42s;
* ``--after-change`` waits for the *next* change, but the change it wanted had already
  happened while the agent was composing the call, so it blocked until an unrelated tick.

The contract asserted here: a single observation wait can never exceed
``perf.max_wait_ms``; the agent cannot raise that from the command line; and every action
kind gets an explicit, tunable ``perf.stable_delay_ms`` pause so the settle is a knob we
can sweep rather than an emergent property of the poll loop.
"""

from __future__ import annotations

import time

import pytest

from android_ui_analyser.perf import STABLE_DELAY_ENV, clamp_wait_ms, stable_delay_for
from conftest import FakeDevice, make_config, make_engine
from test_memory import HOME


class TestTheCeilingIsRealAndTheAgentCannotRaiseIt:
    def test_a_wait_longer_than_the_ceiling_is_clamped(self) -> None:
        cfg = make_config()
        assert cfg.perf.max_wait_ms == 5000
        clamped, was_clamped = clamp_wait_ms(45_000, cfg)
        assert clamped == 5000
        assert was_clamped is True

    def test_a_wait_inside_the_ceiling_is_left_alone(self) -> None:
        cfg = make_config()
        clamped, was_clamped = clamp_wait_ms(1_200, cfg)
        assert clamped == 1_200
        assert was_clamped is False

    def test_the_ceiling_is_configurable_for_a_genuinely_slow_app(self) -> None:
        cfg = make_config(perf={"max_wait_ms": 9_000})
        clamped, was_clamped = clamp_wait_ms(45_000, cfg)
        assert clamped == 9_000
        assert was_clamped is True

    def test_a_clamped_wait_tells_the_agent_to_call_again(self) -> None:
        """Silently shortening a wait would read as 'nothing arrived'. Say it instead."""
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        result = engine.wait_after_change(timeout_ms=45_000, observe=False)
        assert result.wait_clamped_from_ms == 45_000
        assert result.wait_ceiling_ms == 5_000
        assert "call again" in (result.note or "").lower()


class TestTheSettlePauseIsAKnobNotAnAccident:
    def test_every_action_kind_resolves_a_stable_delay(self) -> None:
        cfg = make_config()
        for kind in ("tap", "input", "swipe", "key", "launch", "open-link"):
            assert stable_delay_for(kind, cfg) >= 0

    def test_an_unknown_kind_falls_back_to_the_default(self) -> None:
        cfg = make_config(perf={"stable_delay_ms": {"default": 350}})
        assert stable_delay_for("no-such-kind", cfg) == 350

    def test_a_kind_specific_delay_beats_the_default(self) -> None:
        cfg = make_config(perf={"stable_delay_ms": {"launch": 1_500}})
        assert stable_delay_for("launch", cfg) == 1_500

    def test_config_overrides_merge_per_key_so_one_slow_kind_can_be_tuned_alone(self) -> None:
        """Naming `launch` must not silently reset `tap` to the default."""
        base = make_config()
        tuned = make_config(perf={"stable_delay_ms": {"launch": 1_500}})
        assert stable_delay_for("tap", tuned) == stable_delay_for("tap", base)

    def test_the_sweep_moves_every_kind_with_one_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-key merging is wrong for a sweep: `default` alone would move nothing."""
        cfg = make_config()
        monkeypatch.setenv(STABLE_DELAY_ENV, "0")
        assert stable_delay_for("tap", cfg) == 0
        assert stable_delay_for("launch", cfg) == 0
        monkeypatch.setenv(STABLE_DELAY_ENV, "750")
        assert stable_delay_for("tap", cfg) == 750
        assert stable_delay_for("launch", cfg) == 750

    def test_a_junk_env_value_is_ignored_rather_than_crashing_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = make_config()
        monkeypatch.setenv(STABLE_DELAY_ENV, "not-a-number")
        assert stable_delay_for("tap", cfg) == stable_delay_for("tap", make_config())

    def test_the_delay_is_actually_spent_before_observing(self) -> None:
        cfg = make_config(perf={"stable_delay_ms": {"default": 0, "tap": 300}})
        engine = make_engine(config=cfg, device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        started = time.monotonic()
        engine.tap(1, observe=True)
        spent_ms = (time.monotonic() - started) * 1000
        assert spent_ms >= 300, f"stable delay was not spent (only {spent_ms:.0f}ms)"


class TestAnObservationIsNeverSilentlyEmpty:
    def test_an_empty_observation_is_not_reported_as_a_settled_screen(self) -> None:
        """`elements=3 shown=0` with ok=true is the bug that cost 42s downstream."""
        # An id-free action, so the assertion is about the empty observation and not about
        # id staleness. A bare FakeDevice draws nothing, exactly as a splash does.
        engine = make_engine()
        result = engine.key("back", observe=True)
        obs = result.observation
        assert obs is not None
        assert not obs.elements, "premise: nothing is on screen"
        assert result.observation_empty is True
        assert "empty" in (result.note or "").lower()

    def test_a_wait_with_nothing_visible_yet_keeps_looking(self) -> None:
        engine = make_engine()
        result = engine.wait_after_change(timeout_ms=2_000, observe=True)
        obs = result.observation
        if obs is not None and not obs.elements:
            assert result.observation_empty is True


class TestAfterChangeDoesNotBlockOnAChangeThatAlreadyHappened:
    def test_an_already_settled_screen_returns_promptly(self) -> None:
        """The 42s call. A settled, non-empty screen is the answer — return it."""
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()  # establish a baseline, as a real agent would have
        started = time.monotonic()
        engine.wait_after_change(timeout_ms=5_000, observe=False)
        spent_ms = (time.monotonic() - started) * 1000
        assert spent_ms < 5_000, f"blocked for the whole ceiling ({spent_ms:.0f}ms)"


class TestEveryObservationWaitObeysTheCeiling:
    """Found in a live pass: one `await-and-analyze` ran 62.26s.

    The ceiling had been wired into `wait_after_change` only. `await_predicate` kept its own
    60s default and nothing capped it — and `await` is the wait an agent should reach for most
    often, so leaving it uncapped undid the ceiling everywhere it mattered.
    """

    def test_await_does_not_keep_its_own_uncapped_default(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        started = time.monotonic()
        engine.await_predicate("text:NothingOnThisScreenEver", observe=False)
        spent_ms = (time.monotonic() - started) * 1000
        ceiling = make_config().perf.max_wait_ms
        assert spent_ms < ceiling * 2, f"await ran {spent_ms:.0f}ms against a {ceiling}ms ceiling"

    def test_await_asked_for_a_minute_is_capped(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        started = time.monotonic()
        engine.await_predicate("text:NothingOnThisScreenEver", timeout_ms=60_000, observe=False)
        spent_ms = (time.monotonic() - started) * 1000
        assert spent_ms < 10_000, f"await ran {spent_ms:.0f}ms after asking for 60_000ms"


class TestABoundedWaitNeverExitsAsAnError:
    """Found on a device: `rc=3`, `wait_timeout: screen did not settle within 748 ms`.

    The outer deadline had been converted to a hand-back, but the inner `wait_stable` still
    raised, so the bounded wait a caller was told to expect to expire exited as a device fault
    and threw away the screen it had already read.
    """

    def test_a_still_moving_screen_returns_rather_than_raising(self) -> None:
        from conftest import make_png

        # Frames that never hold steady, so the inner settle can only ever run out.
        a = make_png(80, 80, color=(255, 255, 255))
        b = make_png(80, 80, color=(0, 0, 0))
        device = FakeDevice(hierarchy_xml=HOME, screenshots=[a, b] * 400)
        engine = make_engine(device=device)
        result = engine.wait_after_change(
            timeout_ms=400, interval_ms=1, settle_ms=50, confirmation_ms=20, observe=False
        )
        assert result.ok is True
        assert result.settled_unmet is True
        assert "still moving" in (result.detail or "")

    def test_the_returned_screen_is_not_discarded_on_expiry(self) -> None:
        from conftest import make_png

        a = make_png(80, 80, color=(255, 255, 255))
        b = make_png(80, 80, color=(0, 0, 0))
        device = FakeDevice(hierarchy_xml=HOME, screenshots=[a, b] * 400)
        engine = make_engine(device=device)
        result = engine.wait_after_change(
            timeout_ms=400, interval_ms=1, settle_ms=50, confirmation_ms=20, observe=True
        )
        assert result.observation is not None
        assert result.observation.elements, "expiry threw away the screen it had already read"


class TestTheWallClockCannotReportSomeoneElsesCall:
    """The engine outlives one command under the warm daemon.

    Found on a device, not in this suite: a 1.82s wait reported ``wall_ms=51302`` — the age of
    the previous action's stamp. A FakeDevice never reproduces it because each unit test builds
    a fresh engine, which is precisely why the stamp has to be consume-once by construction.
    """

    def test_a_second_read_does_not_reuse_the_first_calls_stamp(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        first = engine.key("back", observe=True)
        assert first.wall_ms is not None
        second = engine._wall_ms()
        assert second is None, f"stale stamp leaked into a later call ({second}ms)"

    def test_a_wait_reports_its_own_duration_not_an_earlier_actions(self) -> None:
        engine = make_engine(device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        engine.key("back", observe=True)  # leaves a stamp behind, in the old code
        time.sleep(0.15)
        waited = engine.wait_after_change(timeout_ms=1_000, observe=False)
        assert waited.wall_ms is not None
        assert waited.wall_ms < 1_500, f"wall_ms={waited.wall_ms} — not this call's duration"


class TestTheReportedDurationIsTheWallClock:
    def test_duration_ms_is_not_only_the_tree_parse(self) -> None:
        """aua reported 247ms for a call that took 5.26s. Report the wall."""
        cfg = make_config(perf={"stable_delay_ms": {"default": 250}})
        engine = make_engine(config=cfg, device=FakeDevice(hierarchy_xml=HOME))
        engine.analyze()
        started = time.monotonic()
        result = engine.tap(1, observe=True)
        wall_ms = (time.monotonic() - started) * 1000
        assert result.wall_ms is not None
        assert result.wall_ms >= 250
        assert result.wall_ms <= wall_ms + 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
