"""Handing a flow to the device is a shortcut, and a shortcut must never change the outcome.

Measured on an 8-step Settings flow: 4092ms driven from the host against 606ms run on the
device, so roughly 436ms saved per step. Passing the UiAutomation slot across and back costs
~1.8s, because Android suppresses accessibility services while uiautomator2 holds it — which
is why short runs deliberately stay on the host.

Everything here is about the guarantee rather than the speed: off by default, refuses the
steps it cannot honour, and falls back silently on anything unexpected. A run must produce the
same result whether or not a helper happened to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import DeviceStoodDownError, Engine
from android_ui_analyser.memory import RouteStep
from conftest import FakeDevice, make_config


def _steps(n: int, kind: str = "tap") -> list[RouteStep]:
    return [RouteStep(kind=kind, resource_id=f"row{i}") for i in range(n)]


class _Agent:
    """Stands in for the Android ``device_agent`` capability."""

    def __init__(
        self, *, bound: bool = True, completed: int | None = None, can_root: bool = True
    ) -> None:
        self.bound = bound
        self.can_root = can_root
        self.completed = completed
        self.requests: list[dict[str, Any]] = []
        self.enable_calls = 0

    def is_bound(self, serial: str) -> bool:
        return self.bound

    def is_enabled(self, serial: str) -> bool:
        return self.bound

    def is_installed(self, serial: str) -> bool:
        return True

    def rootable(self, serial: str) -> bool:
        return self.can_root

    def snapshot_state(self, serial: str) -> dict[str, Any]:
        return {
            "enabled_services": [],
            "accessibility_enabled": "0",
            "restricted_settings_appop": "default",
            "adbd_root": False,
        }

    def uiautomation_held(self, serial: str) -> bool:
        return False

    def release_uiautomation(self, serial: str) -> None:
        return None

    def enable(self, serial: str) -> dict[str, Any]:
        self.enable_calls += 1
        self.bound = True
        return {"enabled": True, "bound": True}

    def open_channel(self, serial: str, *, timeout: float = 5.0) -> Any:
        agent = self

        class _Channel:
            def request(self, method: str, params=None, *, timeout: float = 5.0):
                agent.requests.append({"method": method, "params": params})
                sent = (params or {}).get("steps") or []
                done = len(sent) if agent.completed is None else agent.completed
                return {
                    "completed": done,
                    "total": len(sent),
                    "stopped_at": -1 if done == len(sent) else done,
                    "steps": [
                        {"index": i, "kind": s.get("kind"), "ok": True, "ms": 40}
                        for i, s in enumerate(sent[:done])
                    ],
                }

            def close(self) -> None:
                return None

        return _Channel()


def _engine(tmp_path: Path, agent: _Agent | None, **helper: Any) -> Engine:
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, helper=helper or {"enabled": True})
    engine = Engine(cfg, device=FakeDevice())
    if agent is not None:
        engine.platform.capability = lambda name: agent  # type: ignore[method-assign]
    return engine


def test_the_offload_is_off_until_it_is_switched_on(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=False)
    ran = engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True)
    assert ran == 0
    assert agent.requests == [], "a disabled helper must not be contacted at all"


def test_a_platform_without_a_helper_just_runs_on_the_host(tmp_path: Path) -> None:
    """An iOS or web adapter raises from the capability gate; that is not an error here."""

    engine = _engine(tmp_path, None, enabled=True)
    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0


def test_a_single_step_stays_on_the_host(tmp_path: Path) -> None:
    """One step saves ~430ms and cannot repay a ~682ms handover. Two can: measured 1.27x."""

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)  # shipped default
    assert engine._offload_steps_to_device(_steps(1), executed=None, allow_destructive=True) == 0
    assert agent.requests == [], "a lone step cannot cover the handover"


def test_two_steps_already_pay_for_the_handover(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)  # shipped default
    assert engine._offload_steps_to_device(_steps(2), executed=None, allow_destructive=True) == 2


def test_a_long_enough_run_is_handed_over_whole(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=5)
    executed: list[dict[str, Any]] = []
    ran = engine._offload_steps_to_device(_steps(8), executed=executed, allow_destructive=True)
    assert ran == 8
    assert agent.requests[0]["method"] == "flow.run"
    assert [row["ran_on"] for row in executed] == ["device"] * 8, (
        "the report must say where each step ran"
    )


def test_only_the_leading_device_runnable_stretch_is_offered(tmp_path: Path) -> None:
    """A host-only kind ends the prefix — the device never sees it."""

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    steps = [*_steps(3), RouteStep(kind="proxy-start"), *_steps(3)]
    ran = engine._offload_steps_to_device(steps, executed=None, allow_destructive=True)
    assert ran == 3
    assert len(agent.requests[0]["params"]["steps"]) == 3


def test_an_arbitrary_keycode_is_not_offered_to_the_device(tmp_path: Path) -> None:
    """Only back/home/recents exist as accessibility actions."""

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    steps = [*_steps(2), RouteStep(kind="key", arg="enter"), *_steps(4)]
    assert engine._offload_steps_to_device(steps, executed=None, allow_destructive=True) == 2


def test_a_destructive_step_stops_the_prefix_unless_allowed(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    steps = [*_steps(3), RouteStep(kind="tap", label="Delete account"), *_steps(3)]
    assert engine._offload_steps_to_device(steps, executed=None, allow_destructive=False) == 3


def test_a_partial_device_run_leaves_the_rest_to_the_host(tmp_path: Path) -> None:
    agent = _Agent(completed=4)
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 4


def test_auto_setup_can_be_refused_for_a_fleet_that_installs_nothing(tmp_path: Path) -> None:
    agent = _Agent(bound=False)
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2, auto_setup=False)
    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0
    assert agent.enable_calls == 0, "auto_setup is off, so nothing may be installed"


def test_switching_it_on_is_enough__aua_does_the_setup(tmp_path: Path) -> None:
    """One switch. No second flag, no manual install step."""

    agent = _Agent(bound=False)
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 8
    assert agent.enable_calls == 1


def test_a_target_that_cannot_root_is_probed_once_and_then_left_alone(tmp_path: Path) -> None:
    """`adb root` restarts adbd. Paying that on every run of a retail phone is unacceptable."""

    agent = _Agent(bound=False, can_root=False)
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    for _ in range(3):
        assert (
            engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0
        )
    assert agent.enable_calls == 0, "a device that cannot root must never be sent to enable()"


def test_setup_that_throws_is_remembered_rather_than_retried(tmp_path: Path) -> None:
    class _FailsSetup(_Agent):
        def enable(self, serial: str):
            self.enable_calls += 1
            raise RuntimeError("install refused")

    agent = _FailsSetup(bound=False)
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    for _ in range(3):
        engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True)
    assert agent.enable_calls == 1, "a failed setup must not be attempted again every run"


def test_a_helper_that_throws_never_breaks_the_run(tmp_path: Path) -> None:
    class _Broken(_Agent):
        def open_channel(self, serial: str, *, timeout: float = 5.0):
            raise RuntimeError("channel exploded")

    engine = _engine(tmp_path, _Broken(), enabled=True, min_flow_steps=2)
    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0


def test_gesture_steps_are_offered_to_the_device(tmp_path: Path) -> None:
    """Gestures are why the offload is worth having: real flows scroll early.

    Before they were supported the prefix stopped at the first swipe, so a scrolling flow
    handed over nothing at all and the handover could never pay for itself.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=2)
    steps = [
        RouteStep(kind="scroll", arg="up"),
        RouteStep(kind="swipe", arg="down"),
        RouteStep(kind="scroll-to", arg="Settings", by="text"),
        RouteStep(kind="tap-point", arg="10,20"),
        RouteStep(kind="wait-stable", timeout_ms=4000),
        RouteStep(kind="paste"),
    ]
    assert engine._offload_steps_to_device(steps, executed=None, allow_destructive=True) == 6


def test_a_bad_direction_or_point_is_not_offered(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=1)
    for bad in (
        RouteStep(kind="scroll", arg="sideways"),
        RouteStep(kind="swipe", arg=""),
        RouteStep(kind="tap-point", arg="not-a-point"),
        RouteStep(kind="tap-point", arg="-5,10"),
        RouteStep(kind="scroll-to", arg=""),
    ):
        assert engine._offload_steps_to_device([bad], executed=None, allow_destructive=True) == 0, (
            f"{bad.kind} {bad.arg!r} must stay on the host"
        )


def test_hide_keyboard_stays_on_the_host(tmp_path: Path) -> None:
    """Accessibility cannot send KEYCODE_ESCAPE, and Back would finish the Activity.

    The device can only press Back after seeing an input-method window, and uiautomator2's
    headless AdbKeyboard exposes none — so on an AUA-driven device the step would report
    success having done nothing at all.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=1)
    step = RouteStep(kind="hide-keyboard")
    assert engine._offload_steps_to_device([step], executed=None, allow_destructive=True) == 0


def test_scroll_to_with_an_unsupported_matcher_stays_on_the_host(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True, min_flow_steps=1)
    step = RouteStep(kind="scroll-to", arg="Thing", by="regex")
    assert engine._offload_steps_to_device([step], executed=None, allow_destructive=True) == 0


def test_the_offload_does_not_connect_uiautomator2(tmp_path: Path) -> None:
    """The whole saving lives here, and it is one attribute access away from being lost.

    Touching `self.device` connects uiautomator2, and connecting it is what makes the
    handover expensive: the device then has to surrender the UiAutomation slot and wait for
    the helper to bind. Measured 2155ms that way against 16ms when nothing ever attached —
    2839ms total fixed cost against 682ms. That difference is what moved the break-even from
    about nine steps down to two, which is the difference between a feature that fires on
    this repo's flows and one that never does.
    """

    agent = _Agent()
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, helper={"enabled": True})
    cfg.device.serial = "emulator-1234"
    engine = Engine(cfg)  # deliberately no device: nothing is connected yet
    engine.platform.capability = lambda name: agent  # type: ignore[method-assign]

    ran = engine._offload_steps_to_device(_steps(4), executed=None, allow_destructive=True)

    assert ran == 4
    assert engine._device is None, (
        "the offload connected uiautomator2, which costs ~2.1s of handover it did not need"
    )
    assert agent.requests and agent.requests[0]["method"] == "flow.run"


def _journal_lines(engine: Engine) -> list[dict[str, Any]]:
    import json
    from pathlib import Path as _P

    from android_ui_analyser import journal

    out: list[dict[str, Any]] = []
    for path in _P(journal.journal_dir(engine.config.cache.dir)).glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return [e for e in out if str(e.get("cmd", "")).startswith("helper.")]


def test_a_declined_offload_says_why_in_the_journal(tmp_path: Path) -> None:
    """The helper can decline for half a dozen good reasons and the run still succeeds.

    Without a record there is no way to answer "did it fire, and if not why" after the fact,
    which is the question worth asking while its thresholds are being tuned on real flows.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)  # shipped default: min 2 steps
    engine._offload_steps_to_device(_steps(1), executed=None, allow_destructive=True)

    events = _journal_lines(engine)
    assert events, "a declined offload must leave a trace"
    assert events[-1]["cmd"] == "helper.skipped"
    extra = events[-1].get("extra") or {}
    assert extra["reason"] == "run_too_short"
    assert extra["runnable"] == 1 and extra["min_flow_steps"] == 2, (
        "the numbers behind the decision matter more than the verdict"
    )


def test_a_successful_offload_records_what_it_cost(tmp_path: Path) -> None:
    agent = _Agent()
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, helper={"enabled": True})
    cfg.device.serial = "emulator-1234"
    engine = Engine(cfg)  # no device: the real lazy path
    engine.platform.capability = lambda name: agent  # type: ignore[method-assign]
    engine._offload_steps_to_device(_steps(4), executed=None, allow_destructive=True)

    events = [e for e in _journal_lines(engine) if e["cmd"] == "helper.offloaded"]
    assert events, "a successful offload must be visible too"
    extra = events[-1]["extra"]
    assert extra["completed"] == 4 and extra["total"] == 4
    assert isinstance(extra["ms"], (int, float))
    assert extra["u2_was_connected"] is False, (
        "whether uiautomator2 was already attached is the single biggest cost factor "
        "(682ms vs 2839ms), so a disappointing run can be explained rather than guessed at"
    )


def test_enabling_it_on_a_device_that_cannot_root_is_inert_not_broken(tmp_path: Path) -> None:
    """An agent will switch this on while testing against a phone. That must cost nothing.

    Not an error, not a slower run that still works, and above all not an APK left installed
    on somebody's device listed under Accessibility doing nothing. The run simply proceeds on
    the path it would have taken anyway.
    """

    agent = _Agent(bound=False, can_root=False)
    engine = _engine(tmp_path, agent, enabled=True)  # the agent turned it on
    executed: list[dict[str, Any]] = []

    ran = engine._offload_steps_to_device(_steps(8), executed=executed, allow_destructive=True)

    assert ran == 0, "nothing may run on the device"
    assert agent.enable_calls == 0, "nothing may be installed or switched on"
    assert agent.requests == [], "the helper must not even be contacted"
    assert executed == [], "and the run must be left entirely to the host"

    events = [e for e in _journal_lines(engine) if e["cmd"] == "helper.skipped"]
    assert events and (events[-1].get("extra") or {}).get("reason") == "not_rootable", (
        "and it must say why, or the next person debugs a silence"
    )


def test_install_refuses_a_target_that_could_never_run_it() -> None:
    """`aua helper install` used to push 900KB onto any device that would accept it."""

    import pytest

    from android_ui_analyser import device_agent

    calls: list[str] = []
    original_rootable = device_agent.rootable
    original_adb = device_agent._adb
    device_agent.rootable = lambda serial: False  # type: ignore[assignment]
    device_agent._adb = lambda serial, *a, **k: calls.append(a[0]) or None  # type: ignore[assignment]
    try:
        with pytest.raises(device_agent.HelperUnavailableError) as excinfo:
            device_agent.install("emulator-retail")
    finally:
        device_agent.rootable = original_rootable  # type: ignore[assignment]
        device_agent._adb = original_adb  # type: ignore[assignment]

    assert excinfo.value.code == "helper_needs_root"
    assert "install" not in calls, "it must refuse before pushing anything"


def test_a_flow_that_opens_with_launch_app_can_still_offload_the_rest(tmp_path: Path) -> None:
    """The shape of every real flow, and the one a leading-prefix rule could not help.

    Flows open with `launch_app`, which cannot move to the device: it resolves the entry
    Activity, marks a capture boundary and promotes app context into memory. A rule that only
    ever took a *leading* prefix therefore offered nothing on the one shape that matters, and
    the feature sat switched on doing nothing.
    """

    engine = _engine(tmp_path, _Agent(), enabled=True)
    steps = [RouteStep(kind="launch-app", arg="dev.example.app"), *_steps(8)]

    assert engine._pick_offload_start(steps, allow_destructive=True) == 1


def test_a_short_run_after_the_launch_is_not_worth_the_handover(tmp_path: Path) -> None:
    """The regression this policy exists to prevent.

    Two steps repay a 682ms handover, which is what a run starting at index 0 costs while
    nothing is attached. Once the host has driven a step, uiautomator2 holds the slot: it has
    to be taken away and given back, and the same two steps then come out slower than never
    offloading at all. Measured on a five-step flow: 12.2s host-only against 24.6s with a
    two-step run handed over mid-flow.
    """

    engine = _engine(tmp_path, _Agent(), enabled=True)
    steps = [RouteStep(kind="launch-app", arg="dev.example.app"), *_steps(2)]

    assert engine._pick_offload_start(steps, allow_destructive=True) is None


def test_a_leading_run_is_priced_as_cheap_only_while_nothing_is_attached(
    tmp_path: Path,
) -> None:
    """Index 0 is not automatically the cheap case — an unconnected engine is."""

    engine = _engine(tmp_path, _Agent(), enabled=True)
    steps = _steps(2)

    engine._device = None
    assert engine._pick_offload_start(steps, allow_destructive=True) == 0

    engine._device = FakeDevice()
    assert engine._pick_offload_start(steps, allow_destructive=True) is None


def test_picking_where_to_fire_never_touches_the_device(tmp_path: Path) -> None:
    """The decision has to be free, or a flow the helper cannot help with pays anyway.

    An earlier version re-probed at every index after a refusal, and each probe paid a
    release-and-rebind. Choosing from the step list alone is what keeps a declined flow at
    exactly zero device contact.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    steps = [
        RouteStep(kind="launch-app", arg="dev.example.app"),
        *_steps(2),
        RouteStep(kind="flow", arg="other"),
        *_steps(2),
        RouteStep(kind="stop-app", arg="dev.example.app"),
        *_steps(3),
    ]

    assert engine._pick_offload_start(steps, allow_destructive=True) is None
    assert agent.requests == [], "deciding must be a pure host-side decision"
    assert agent.enable_calls == 0, "and must never trigger setup"


def test_a_later_run_is_taken_when_the_first_one_is_too_short(tmp_path: Path) -> None:
    """Short run first, long run later: skip the short one and fire on the long one."""

    engine = _engine(tmp_path, _Agent(), enabled=True)
    steps = [
        RouteStep(kind="launch-app", arg="dev.example.app"),
        *_steps(2),
        RouteStep(kind="stop-app", arg="dev.example.app"),
        *_steps(9),
    ]

    assert engine._pick_offload_start(steps, allow_destructive=True) == 4


def test_offloaded_steps_report_their_real_position_in_the_flow(tmp_path: Path) -> None:
    """The device numbers its own slice; a report that disagrees with the flow misleads."""

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    executed: list[dict[str, Any]] = []

    engine._offload_steps_to_device(
        _steps(3), executed=executed, allow_destructive=True, index_offset=5
    )

    assert [row["index"] for row in executed] == [5, 6, 7], (
        f"positions must be flow-relative, got {[r['index'] for r in executed]}"
    )


def test_a_running_background_job_blocks_the_handover(tmp_path: Path) -> None:
    """A job runs on its own thread inside a warm engine, so there is nobody to ask to wait."""

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    engine._device_is_spoken_for = lambda serial: "job_running"  # type: ignore[method-assign]

    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0
    assert agent.requests == [], "the device must not be handed a run it cannot finish"


def test_the_capture_buffer_is_paused_across_the_handover(tmp_path: Path) -> None:
    """Closing the engine's device handle is not enough, and assuming it was cost real runs.

    ``capture_start`` hands the buffer ``device.screenshot``, so the buffer holds its own
    reference to the same uiautomator2 client. Left sampling, its thread restarts the server
    mid-handover and Android tears the accessibility service down under a run already in
    flight. It must be paused for the duration and resumed afterwards — a daemon left with a
    stopped buffer would quietly break the next ``capture last``.
    """

    class _Buffer:
        running = True
        paused = False
        events: list[str] = []

        def pause(self, reason: str = "manual", *, settle_s: float = 0.0) -> bool:
            type(self).paused = True
            self.events.append(f"pause:{reason}:settle={settle_s}")
            return True

        def resume(self, *, only_if_idle: bool = False) -> None:
            type(self).paused = False
            self.events.append("resume")

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    buffer = _Buffer()
    engine._capture = buffer
    engine._device_is_spoken_for = lambda serial: None  # type: ignore[method-assign]

    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 8

    assert buffer.events == ["pause:handover:settle=2.0", "resume"], buffer.events
    assert agent.requests, "the run should still have reached the device"


def test_the_capture_buffer_resumes_even_when_the_handover_throws(tmp_path: Path) -> None:
    """A failed offload must not leave a warm daemon with its recorder switched off."""

    class _Buffer:
        running = True
        paused = False

        def __init__(self) -> None:
            self.events: list[str] = []

        def pause(self, reason: str = "manual", *, settle_s: float = 0.0) -> bool:
            self.paused = True
            self.events.append("pause")
            return True

        def resume(self, *, only_if_idle: bool = False) -> None:
            self.paused = False
            self.events.append("resume")

    class _Exploding(_Agent):
        def open_channel(self, serial: str, *, timeout: float = 5.0):
            raise RuntimeError("channel refused")

    engine = _engine(tmp_path, _Exploding(), enabled=True)
    buffer = _Buffer()
    engine._capture = buffer
    engine._device_is_spoken_for = lambda serial: None  # type: ignore[method-assign]

    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0
    assert buffer.events == ["pause", "resume"], buffer.events


def test_a_warm_daemon_keeps_the_slot_so_the_offload_stands_down(tmp_path: Path) -> None:
    """The failure that looked like a broken helper and was AUA competing with itself.

    The daemon's whole job is to hold a uiautomator2 connection open for its serial and to
    re-establish it whenever it goes away. Releasing UiAutomation under one does not hand the
    device over, it starts a race the daemon wins about 600ms later — tearing down the
    accessibility service partway through a run the device is already executing. Measured on a
    24-step flow with the helper switched on: 24 of 24 steps in 3.5s with no daemon, 3 or 4
    steps in 22s with one, and a different number every run.

    Declining costs the host path, which is what would have happened anyway.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    engine._device_is_spoken_for = (  # type: ignore[method-assign]
        lambda serial: "another_process_owns_device"
    )

    ran = engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True)

    assert ran == 0
    assert agent.requests == [], "the device must not be handed a run it cannot finish"


def test_without_a_daemon_the_offload_proceeds(tmp_path: Path) -> None:
    """The guard must be about the daemon specifically, not a blanket refusal."""

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    engine._device_is_spoken_for = lambda serial: None  # type: ignore[method-assign]

    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 8


def test_a_capture_buffer_that_will_not_settle_blocks_the_handover(tmp_path: Path) -> None:
    """A frame still in flight lands mid-run and takes the slot straight back.

    This was the last of the flakiness, and it hid as a broken helper: the offload cost
    10-13s and lost the accessibility service partway through, against 1.7s whenever the
    buffer had gone quiet. The pause already reports whether it settled — the bug was
    handing the device over regardless of the answer.
    """

    class _Buffer:
        running = True
        paused = False

        def pause(self, reason: str = "manual", *, settle_s: float = 0.0) -> bool:
            self.paused = True
            return False  # a grab is still in flight

        def resume(self, *, only_if_idle: bool = False) -> None:
            self.paused = False

    agent = _Agent()
    engine = _engine(tmp_path, agent, enabled=True)
    engine._capture = _Buffer()
    engine._device_is_spoken_for = lambda serial: None  # type: ignore[method-assign]

    assert engine._offload_steps_to_device(_steps(8), executed=None, allow_destructive=True) == 0
    assert agent.requests == [], "the device must not be handed a run something will interrupt"


def test_a_stood_down_device_is_never_reconnected_by_the_capture_buffer(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Pausing the buffer is not enough on its own, and the gap is what cost 3 runs in 10.

    ``capture_start`` used to hand the buffer ``device.screenshot`` — a method *bound* to the
    uiautomator2 client. The handover then closes that client and drops the engine's
    reference, but the buffer still holds the bound method, so the very next sampling tick
    calls into a dead connection and uiautomator2 quietly restarts the server to satisfy it.

    That costs twice. It takes the UiAutomation slot back off a helper that is mid-run, and
    the restart itself outlasts the two-second settle budget, so the *following* handover sees
    a buffer that will not go quiet and declines to the slow host path. It only showed up
    back-to-back because only then is the previous handover's teardown still in flight —
    which is exactly the shape of the 3-in-10 refusals on record.

    The buffer must therefore ask the engine for a screenshot rather than hold a device, so a
    tick that lands during a handover fails fast instead of resurrecting the connection the
    handover deliberately tore down.
    """

    agent = _Agent()
    engine = _engine(tmp_path, agent)

    reconnects: list[str] = []

    class _Dev:
        serial = "emulator-1234"

        def screenshot(self) -> Any:
            return object()

        def close(self) -> None:
            return None

    def _connect(self: Engine) -> Any:
        if self._device is None:
            # This is the uiautomator2 reconnect the handover exists to prevent.
            reconnects.append("connect")
            self._device = _Dev()  # type: ignore[assignment]
        return self._device

    monkeypatch.setattr(type(engine), "device", property(_connect))
    engine._device = _Dev()  # type: ignore[assignment]

    # Nothing is being handed over, so sampling works exactly as before.
    assert engine._capture_screenshot() is not None
    assert reconnects == []

    with engine._device_stood_down(), pytest.raises(DeviceStoodDownError):
        engine._capture_screenshot()

    assert reconnects == [], (
        "a capture tick reconnected uiautomator2 during the handover, taking the slot back "
        "off the helper"
    )

    # Once the handover is over the buffer still does not reconnect. Closing the gap only
    # inside the handover would leave the back-to-back case exactly as it was: the tick that
    # fires the instant capture resumes would start the ~2.1s reconnect itself, and the next
    # handover's two-second settle wait would expire in the middle of it. Sampling is a
    # background courtesy, so it waits for the engine to pick the device up again.
    assert engine._capture_screenshot() is None
    assert reconnects == []

    # ...and resumes on its own once the engine's own next analyze has reconnected.
    engine._device = _Dev()  # type: ignore[assignment]
    assert engine._capture_screenshot() is not None
    assert reconnects == []


def test_the_capture_buffer_is_not_handed_a_device_bound_screenshot(tmp_path: Path) -> None:
    """The binding itself is the bug, so pin it rather than only its symptom.

    A buffer holding ``device.screenshot`` keeps the client alive past every teardown the
    engine performs. Whatever it is handed must route through the engine, so that closing the
    device is actually enough to stop it being used.
    """

    engine = _engine(tmp_path, None)
    engine._device = FakeDevice()  # type: ignore[assignment]

    shot = engine._capture_screenshot_fn()

    assert getattr(shot, "__self__", None) is engine, (
        "the capture buffer was handed a device-bound method; closing the device will not "
        "stop it sampling"
    )


def test_a_second_run_is_offered_once_the_host_has_stepped_over_the_gap(
    tmp_path: Path,
) -> None:
    """One handover per flow is not enough, and asserts are why.

    A real QA flow is not a straight line of taps: it acts, checks what it did, acts again.
    Every check is a host step, so the device-runnable stretch ends there — and because the
    engine picked a single offload point for the whole flow, everything past the *first*
    check went back to one host round trip per step no matter how long it was. A 26-step
    journey with a check at step 6 handed over 5 steps and drove the other 21 by hand.

    Nothing about the second stretch is different in kind from the first, so it gets asked
    the same question, from wherever the host has got to.
    """

    agent = _Agent()
    engine = _engine(
        tmp_path, agent, enabled=True, min_flow_steps=2, min_midflow_steps=3
    )
    steps = (
        _steps(4)
        + [RouteStep(kind="launch-app", arg="com.example.placeholder")]
        + _steps(5)
    )

    assert engine._pick_offload_start(steps, allow_destructive=True) == 0
    assert engine._pick_offload_start(steps, allow_destructive=True, start=5) == 5


def test_the_second_run_is_judged_by_the_dearer_mid_flow_floor(tmp_path: Path) -> None:
    """A later run is never the cheap one, so it must not be priced like the first.

    Starting at index 0 with nothing connected costs ~682ms. Every run after that has the
    host's uiautomator2 attached, so the slot has to be taken away and handed back, which is
    several times dearer. Re-asking the question must not quietly re-apply the opening
    discount to a run that cannot have it.
    """

    agent = _Agent()
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        helper={"enabled": True, "min_flow_steps": 2, "min_midflow_steps": 6},
    )
    cfg.device.serial = "emulator-1234"
    engine = Engine(cfg)  # nothing connected, so the opening run is the cheap one
    engine.platform.capability = lambda name: agent  # type: ignore[method-assign]
    steps = (
        _steps(3)
        + [RouteStep(kind="launch-app", arg="com.example.placeholder")]
        + _steps(4)
    )

    assert engine._pick_offload_start(steps, allow_destructive=True) == 0
    assert engine._pick_offload_start(steps, allow_destructive=True, start=4) is None, (
        "a 4-step tail cleared a 6-step mid-flow floor — it was priced as an opening run"
    )


def test_a_refused_handover_stops_the_flow_asking_again(tmp_path: Path) -> None:
    """Re-probing after a refusal is the regression this whole policy exists to prevent.

    Asking is free only until the device has been contacted. Once a handover has been tried
    and declined, trying again at the next gap re-pays the setup cost per gap, which is how an
    earlier version turned a saving into a loss. A flow gets one refusal and then stops asking.
    """

    agent = _Agent(completed=0)  # the device takes the run and reports it did nothing
    engine = _engine(
        tmp_path, agent, enabled=True, min_flow_steps=2, min_midflow_steps=2
    )
    steps = (
        _steps(3)
        + [RouteStep(kind="launch-app", arg="com.example.placeholder")]
        + _steps(3)
        + [RouteStep(kind="launch-app", arg="com.example.placeholder")]
        + _steps(3)
    )

    ran, next_at = engine._offload_from(
        steps, at=0, executed=None, allow_destructive=True
    )

    assert ran == 0
    assert next_at is None, (
        "a refused handover left the flow willing to ask again at the next gap"
    )


def test_a_handover_that_worked_points_at_the_next_one(tmp_path: Path) -> None:
    """The success path is the whole point: keep going, from wherever the device stopped."""

    agent = _Agent()
    engine = _engine(
        tmp_path, agent, enabled=True, min_flow_steps=2, min_midflow_steps=2
    )
    steps = (
        _steps(3)
        + [RouteStep(kind="launch-app", arg="com.example.placeholder")]
        + _steps(3)
    )

    ran, next_at = engine._offload_from(
        steps, at=0, executed=None, allow_destructive=True
    )

    assert ran == 3
    assert next_at == 4, "the tail after the host-only step was not offered to the device"
