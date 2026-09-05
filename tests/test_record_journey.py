"""Recording a human's journey — reached through the platform capability, never through adb.

The device recorder only works while its accessibility service is bound, and Android tears
every accessibility service down while uiautomator2 holds the UiAutomation slot. So the one
thing this path must not do is the thing every other command does first: connect the device.
That is not an optimisation here, it is the difference between recording a journey and
recording nothing at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import device_ledger
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import AuaError
from conftest import make_config


class _Agent:
    """Stands in for the Android ``device_agent`` capability."""

    def __init__(self, *, rows: list[dict[str, Any]] | None = None, enabled: bool = True) -> None:
        self.rows = rows or []
        self.enabled = enabled
        self.calls: list[str] = []
        self.released: list[str] = []
        # Whether the accessibility service actually comes back after the slot is released.
        self.bound_after_release = True
        # Whether the service is still armed when the journey is drained.
        self.still_recording = True
        self.touch_capture_calls: list[str] = []

    def is_bound(self, serial: str) -> bool:  # noqa: F811 - overrides the simple stub below
        return self.enabled and self.bound_after_release

    def is_enabled(self, serial: str) -> bool:
        return self.enabled

    def rootable(self, serial: str) -> bool:
        return True

    def enable(self, serial: str) -> dict[str, Any]:
        self.enabled = True
        return {"enabled": True, "bound": True}

    def release_uiautomation(self, serial: str) -> None:
        self.released.append(serial)

    def open_channel(self, serial: str, *, timeout: float = 5.0) -> Any:
        outer = self

        class _Channel:
            def request(self, method: str, params: Any = None, *, timeout: float = 5.0) -> Any:
                outer.calls.append(method)
                if method == "record.start":
                    return {"recording": True}
                if method == "record.peek":
                    return {
                        "recording": outer.still_recording,
                        "count": len(outer.rows),
                        "steps": outer.rows,
                    }
                return {"recording": False, "count": len(outer.rows), "steps": outer.rows}

            def close(self) -> None:
                return None

        return _Channel()

    def start_touch_capture(self, serial: str) -> dict[str, Any]:
        self.touch_capture_calls.append(f"start:{serial}")
        return {"capturing": True}

    def stop_touch_capture(self, serial: str) -> list[Any]:
        self.touch_capture_calls.append(f"stop:{serial}")
        return []

    def discard_touch_capture(self, serial: str) -> dict[str, Any]:
        self.touch_capture_calls.append(f"discard:{serial}")
        return {"capturing": False}


def _engine(tmp_path: Path, agent: _Agent) -> Engine:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"dir": str(tmp_path / "memory")},
        helper={"enabled": True},
    )
    cfg.device.serial = "emulator-1234"
    engine = Engine(cfg)  # deliberately no device
    engine.platform.capability = lambda name: agent  # type: ignore[method-assign]
    return engine


def test_arming_the_recorder_never_connects_the_device(tmp_path: Path) -> None:
    """Connecting would suppress the accessibility service the recorder depends on."""

    agent = _Agent()
    engine = _engine(tmp_path, agent)

    result = engine.demo_record_start()

    assert result["recording"] is True
    assert engine._device is None, (
        "arming the recorder connected uiautomator2, which suppresses the accessibility "
        "service and guarantees an empty recording"
    )
    assert agent.calls == ["record.start"]
    assert agent.released == ["emulator-1234"], (
        "the UiAutomation slot was not handed back, so the helper's service stays torn down"
    )


def test_touch_capture_is_recorded_before_start_and_forgotten_after_stop(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent)
    original_start = agent.start_touch_capture

    def guarded_start(serial: str) -> dict[str, Any]:
        pending = device_ledger.read_ledger(serial, platform=engine.platform.name)
        assert any(
            entry.key == "device_agent_touch_capture"
            and entry.op == "stop_device_agent_touch_capture"
            for entry in pending
        ), "detached touch capture started before its durable stop record"
        return original_start(serial)

    agent.start_touch_capture = guarded_start  # type: ignore[method-assign]

    assert engine.demo_record_start()["touch_capture"] is True
    assert any(
        entry.key == "device_agent_touch_capture"
        for entry in device_ledger.read_ledger("emulator-1234", platform=engine.platform.name)
    )

    engine.demo_record_stop()

    assert agent.touch_capture_calls == [
        "start:emulator-1234",
        "stop:emulator-1234",
        "discard:emulator-1234",
    ]
    assert not any(
        entry.key == "device_agent_touch_capture"
        for entry in device_ledger.read_ledger("emulator-1234", platform=engine.platform.name)
    )


def test_touch_capture_cleanup_failure_keeps_the_durable_stop_pending(tmp_path: Path) -> None:
    agent = _Agent()
    engine = _engine(tmp_path, agent)
    engine.demo_record_start()

    def refuse_discard(serial: str) -> dict[str, Any]:
        raise RuntimeError(f"capture still running on {serial}")

    agent.discard_touch_capture = refuse_discard  # type: ignore[method-assign]

    result = engine.demo_record_stop()

    assert result["ok"] is True
    assert result["recovered_from_touches"] == 0
    assert any(
        entry.key == "device_agent_touch_capture"
        for entry in device_ledger.read_ledger("emulator-1234", platform=engine.platform.name)
    )


def test_stopping_returns_the_journey_as_steps(tmp_path: Path) -> None:
    agent = _Agent(
        rows=[
            {"kind": "tap", "resource_id": "rowSettings", "label": "Settings"},
            {"kind": "tap", "resource_id": "rowTheme", "label": "Theme"},
        ]
    )
    engine = _engine(tmp_path, agent)

    result = engine.demo_record_stop()

    assert result["count"] == 2
    assert result["complete"] is True
    assert [s["resource_id"] for s in result["steps"]] == ["rowSettings", "rowTheme"]
    assert engine._device is None


def test_a_recording_with_a_hole_says_so_and_says_where(tmp_path: Path) -> None:
    agent = _Agent(
        rows=[
            {"kind": "tap", "resource_id": "rowOne"},
            {"kind": "gap", "reason": "screen_changed_with_no_announced_action"},
            {"kind": "tap", "resource_id": "rowTwo"},
        ]
    )
    engine = _engine(tmp_path, agent)

    result = engine.demo_record_stop()

    assert result["complete"] is False
    assert result["gaps"] == [
        {
            "after_step": 1,
            "reason": "screen_changed_with_no_announced_action",
            "package": None,
        }
    ]


def test_an_incomplete_recording_is_not_saved_as_a_runnable_flow(tmp_path: Path) -> None:
    """A draft that skips a step must never be handed over as if it were finished."""

    agent = _Agent(
        rows=[
            {"kind": "tap", "resource_id": "rowOne"},
            {"kind": "gap", "reason": "screen_changed_with_no_announced_action"},
        ]
    )
    engine = _engine(tmp_path, agent)

    with pytest.raises(AuaError) as raised:
        engine.demo_record_stop(save="onboarding")

    assert "gap" in str(raised.value).lower() or "incomplete" in str(raised.value).lower()
    assert not (tmp_path / "memory" / "flows" / "onboarding.yaml").exists()


def test_a_complete_recording_can_be_saved(tmp_path: Path) -> None:
    agent = _Agent(
        rows=[
            {"kind": "tap", "resource_id": "rowSettings", "label": "Settings"},
            {"kind": "tap", "resource_id": "rowTheme", "label": "Theme"},
        ]
    )
    engine = _engine(tmp_path, agent)

    result = engine.demo_record_stop(save="theme-journey")

    assert result["saved"]
    assert (tmp_path / "memory" / "flows" / "theme-journey.yaml").is_file()


def test_a_platform_with_no_helper_says_so_plainly(tmp_path: Path) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "memory")}
    )
    cfg.device.serial = "emulator-1234"
    engine = Engine(cfg)

    def _no_capability(name: str) -> Any:
        raise LookupError(name)

    engine.platform.capability = _no_capability  # type: ignore[method-assign]

    with pytest.raises(AuaError):
        engine.demo_record_start()


def test_arming_refuses_when_the_service_is_suppressed(tmp_path: Path) -> None:
    """A recorder that is not listening must say so, not hand back an empty journey.

    Android tears every accessibility service down while uiautomator2 holds the UiAutomation
    slot, and a warm daemon holds it just by existing. Arming then "succeeds", the human walks
    the whole journey, and `demo stop` returns zero steps and — because nothing was announced
    and nothing was missed either — cheerfully reports the recording complete. The person has
    no way to tell that from an app that announces nothing.

    This was hit for real while testing: a daemon respawned mid-session and three recordings
    came back empty before the cause was visible.
    """

    agent = _Agent()
    agent.bound_after_release = False  # the slot is still held by something else
    engine = _engine(tmp_path, agent)

    with pytest.raises(AuaError) as raised:
        engine.demo_record_start()

    message = f"{raised.value} {getattr(raised.value, 'hint', '')}".lower()
    assert "daemon" in message or "suppress" in message, (
        f"the refusal does not tell the caller what is holding the device: {message}"
    )


def test_a_recording_that_was_torn_down_mid_journey_is_an_error_not_an_empty_draft(
    tmp_path: Path,
) -> None:
    """Losing the service mid-journey must not look like a journey where nothing happened.

    Arming can succeed and the service still be torn down later — any aua command that
    connects uiautomator2 takes the slot back, and a daemon does it just by warming up. The
    service restarts having forgotten it was recording, so `record.stop` drains an empty list.

    Empty-because-nothing-happened and empty-because-nobody-was-watching are the same JSON,
    and only one of them means "walk it again". Asking whether it is still armed before
    draining is what tells them apart.
    """

    agent = _Agent(rows=[])
    agent.still_recording = False  # the service restarted while the human was walking
    engine = _engine(tmp_path, agent)

    with pytest.raises(AuaError) as raised:
        engine.demo_record_stop()

    message = f"{raised.value} {getattr(raised.value, 'hint', '')}".lower()
    assert "lost" in message or "interrupt" in message or "torn down" in message, message
