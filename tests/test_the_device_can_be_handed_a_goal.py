"""`drive.run` is reachable, and a refusal is loud.

The on-device scoring rule was registered in the APK and called by nothing. It is the only
component of the policy experiment that has ever driven a real device — 17 of 19 reachable
destinations against 0 of 19 for every trained checkpoint — and with no host caller it could not be
run against a device at all, so every change to it shipped unvalidated. These tests are what stop
that happening again.

The other half is the difference from the flow offload, which is the reason both exist. That path is
an optimisation: it swallows refusals and runs on the host, because the host can execute the same
steps. There is no host implementation of the *rule*, so this one has nothing to fall back to — and
"nothing happened, no reason given" is the one answer an explicit request must never get.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import DeviceError, Engine, UsageError
from conftest import FakeDevice, make_config


class _Agent:
    """Stands in for the ``device_agent`` capability, with no Android anywhere in it.

    A platform that is not Android would supply exactly this shape. If the engine reached for adb
    or uiautomator2 itself rather than going through the capability, none of it would be recorded
    here — which is what makes this the platform-neutrality proof.
    """

    def __init__(self, *, bound: bool = True, reply: dict[str, Any] | None = None) -> None:
        self.bound = bound
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.channels_closed = 0
        self.released = 0

    def is_bound(self, serial: str) -> bool:
        return self.bound

    def is_enabled(self, serial: str) -> bool:
        return True

    def is_installed(self, serial: str) -> bool:
        return True

    def rootable(self, serial: str) -> bool:
        return True

    def uiautomation_held(self, serial: str) -> bool:
        return False

    def release_uiautomation(self, serial: str) -> None:
        self.released += 1

    def enable(self, serial: str) -> dict[str, Any]:
        self.bound = True
        return {"enabled": True, "bound": True}

    def open_channel(self, serial: str, *, timeout: float = 5.0) -> Any:
        agent = self

        class _Channel:
            def request(self, method: str, params=None, *, timeout: float = 5.0):
                agent.requests.append({"method": method, "params": params or {}})
                if agent.reply is not None:
                    return agent.reply
                goal = (params or {}).get("goal")
                return {
                    "goal": goal,
                    "stop_reason": "handoff",
                    "step_count": 1,
                    "steps": [
                        {
                            "step": 0,
                            "decision": "tap",
                            "n": "n2",
                            "label": "Display",
                            "score": 1.5,
                            "ok": True,
                            "outcome": "changed",
                        }
                    ],
                }

            def close(self) -> None:
                agent.channels_closed += 1

        return _Channel()


def _engine(tmp_path: Path, agent: _Agent | None, **helper: Any) -> Engine:
    """An engine whose only route to the device is the capability, as the boundary requires.

    ``agent=None`` stands in for a platform that does not claim ``device_agent`` at all — it raises
    from the capability gate, exactly as an iOS or web adapter would. It must not be left unpatched:
    that reaches the real Android module and tests the host's adb instead of the contract.
    """

    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, helper=helper)
    engine = Engine(cfg, device=FakeDevice())

    def _capability(name: str) -> Any:
        if agent is None:
            raise LookupError(f"this platform does not provide {name}")
        return agent

    engine.platform.capability = _capability  # type: ignore[method-assign]
    return engine


# --------------------------------------------------------------------------- it is reachable


def test_a_goal_reaches_the_device_as_drive_run(tmp_path: Path) -> None:
    """The whole point: a host call that ends in `drive.run` on the device."""

    agent = _Agent()
    got = _engine(tmp_path, agent).drive_on_device("open the display settings", budget=5)

    assert [r["method"] for r in agent.requests] == ["drive.run"]
    assert agent.requests[0]["params"] == {"goal": "open the display settings", "budget": 5}
    assert got["ok"] is True
    assert got["ran_on"] == "device"
    assert got["stop_reason"] == "handoff"
    assert got["step_count"] == 1
    assert got["steps"][0]["decision"] == "tap"


def test_the_channel_is_always_closed(tmp_path: Path) -> None:
    agent = _Agent()
    _engine(tmp_path, agent).drive_on_device("open Display")
    assert agent.channels_closed == 1


def test_the_channel_is_closed_even_when_the_device_raises(tmp_path: Path) -> None:
    agent = _Agent()

    class _Boom(_Agent):
        def open_channel(self, serial: str, *, timeout: float = 5.0) -> Any:
            outer = self

            class _Channel:
                def request(self, method, params=None, *, timeout: float = 5.0):
                    raise RuntimeError("device went away")

                def close(self) -> None:
                    outer.channels_closed += 1

            return _Channel()

    agent = _Boom()
    with pytest.raises(RuntimeError):
        _engine(tmp_path, agent).drive_on_device("open Display")
    assert agent.channels_closed == 1, "a failed request must not leak the channel"


def test_the_slot_is_handed_over_before_the_goal_is_sent(tmp_path: Path) -> None:
    """Android suppresses accessibility services while uiautomator2 holds UiAutomation, so a goal
    sent without releasing the slot reaches a helper that cannot see the screen."""

    agent = _Agent()
    _engine(tmp_path, agent).drive_on_device("open Display")
    assert agent.released == 1


# --------------------------------------------------------------------------- refusal is loud


def test_a_refusal_is_raised_with_its_reason_and_never_swallowed(tmp_path: Path) -> None:
    """The flow offload returns 0 here and runs on the host. This has no host path to fall to."""

    engine = _engine(tmp_path, _Agent(bound=False))
    with pytest.raises(DeviceError) as excinfo:
        engine.drive_on_device("open Display")
    assert "not_bound_after_release" in str(excinfo.value)


def test_a_platform_with_no_helper_says_so_rather_than_doing_nothing(tmp_path: Path) -> None:
    """An iOS or web adapter raises from the capability gate. For the offload that is a silent
    fallback; for an explicit request it is an unsupported-capability error."""

    engine = _engine(tmp_path, None)
    with pytest.raises(DeviceError) as excinfo:
        engine.drive_on_device("open Display")
    assert "platform_has_no_helper" in str(excinfo.value)


def test_every_refusal_reason_carries_advice(tmp_path: Path) -> None:
    """A reason with no hint is a dead end for whoever hit it."""

    from android_ui_analyser.engine import _HANDOVER_HINTS

    for reason, hint in _HANDOVER_HINTS.items():
        assert hint and hint[0].isupper(), reason


def test_an_empty_goal_is_a_usage_error_not_a_handover(tmp_path: Path) -> None:
    agent = _Agent()
    for blank in ("", "   "):
        with pytest.raises(UsageError):
            _engine(tmp_path, agent).drive_on_device(blank)
    assert agent.requests == [], "the device must not be touched to reject a blank goal"


# --------------------------------------------------------------------------- no step floor


def test_there_is_no_step_floor_because_there_is_nothing_cheaper_to_lose_to(tmp_path: Path) -> None:
    """`helper.min_flow_steps` exists because the offload competes with the host doing the same
    work. Nothing competes with this, so a one-step budget is a legitimate request."""

    agent = _Agent()
    got = _engine(tmp_path, agent, min_flow_steps=8).drive_on_device("open Display", budget=1)
    assert agent.requests[0]["params"]["budget"] == 1
    assert got["ok"] is True


def test_it_does_not_wait_for_helper_enabled_like_the_automatic_path_does(tmp_path: Path) -> None:
    """`helper.enabled` governs whether AUA reaches for the device on its own. Somebody typing
    `aua helper drive` has already decided, exactly as for `helper tree` and `helper enable`."""

    agent = _Agent()
    got = _engine(tmp_path, agent, enabled=False).drive_on_device("open Display")
    assert got["ok"] is True
    assert [r["method"] for r in agent.requests] == ["drive.run"]


def test_a_budget_below_one_is_clamped_not_rejected(tmp_path: Path) -> None:
    agent = _Agent()
    _engine(tmp_path, agent).drive_on_device("open Display", budget=0)
    assert agent.requests[0]["params"]["budget"] == 1


# --------------------------------------------------------------------------- one handover, shared


def test_the_drive_and_the_offload_share_one_slot_handover(tmp_path: Path) -> None:
    """Both reach the device through `_device_agent_borrowed`, so the sequence that took months of
    real-device diagnosis to get right — quiesce, stand down, release, re-check bound — exists once.

    A second copy would not learn the next fix. This asserts the shape rather than the text: both
    callers must route through the borrow, and neither may open a channel of its own.
    """

    import inspect

    from android_ui_analyser.engine import Engine as E

    borrow = inspect.getsource(E._device_agent_borrowed)
    for step in (
        "_quiesce_background_device_work",
        "_device_is_spoken_for",
        "_device_stood_down",
        "release_uiautomation",
        "is_bound",
        "open_channel",
    ):
        assert step in borrow, f"the borrow lost {step}"

    for caller in (E.drive_on_device, E._offload_steps_to_device):
        src = inspect.getsource(caller)
        assert "_device_agent_borrowed" in src, f"{caller.__name__} bypasses the shared handover"
        assert "open_channel" not in src, f"{caller.__name__} opens its own channel"


# --------------------------------------------------------------------------- the Android adapter


def test_the_android_adapter_really_provides_what_the_borrow_asks_for() -> None:
    """The neutral tests above prove the core is platform-independent. This proves the Android
    implementation actually satisfies the contract they fake, so the two cannot drift apart.
    """

    from android_ui_analyser.platforms.android import AndroidPlatform
    from android_ui_analyser.platforms.services import CAPABILITY_METHODS, DEVICE_AGENT

    platform = AndroidPlatform(make_config())
    assert DEVICE_AGENT in platform.capabilities

    agent = platform.capability(DEVICE_AGENT)
    for method in (
        "is_enabled",
        "is_bound",
        "rootable",
        "enable",
        "release_uiautomation",
        "uiautomation_held",
        "open_channel",
    ):
        assert callable(getattr(agent, method)), method
        assert method in CAPABILITY_METHODS[DEVICE_AGENT], f"{method} is not in the contract"


def test_the_bundled_apk_and_the_host_agree_on_the_protocol() -> None:
    """The helper answers `drive.run` differently since it became reachable — the per-step reply
    carries `tried`/`last`/`outcome`. A protocol-1 helper left on a device from an older AUA would
    answer without them, and a caller cannot tell "never stalled" from "does not report stalls".
    """

    from android_ui_analyser import device_agent

    java = Path("helper/app/src/main/java/dev/aua/helper/InfoFeature.java").read_text()
    assert f"PROTOCOL = {device_agent.PROTOCOL};" in java, (
        "InfoFeature.PROTOCOL and device_agent.PROTOCOL must be bumped together"
    )
    assert device_agent.PROTOCOL >= 2
