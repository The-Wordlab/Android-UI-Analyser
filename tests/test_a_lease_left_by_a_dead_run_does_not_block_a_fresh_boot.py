"""A lease record older than the emulator it now points at belongs to a device that is gone.

Seen live on 2026-09-22. A session start found its preferred emulator leased by another agent,
so it booted a second one, which came up on port 5556. A lease file for ``emulator-5556`` was
still there from a run the day before: its emulator had been stopped for 23 hours, but the
owner was a long-lived agent process, so the record read as live. The fresh boot was refused
as "leased by <owner> (active 83967s ago)", rolled back, and the harness fell back to waiting
for the device it had just tried not to wait for.

A serial the caller has *just booted* had no live device a moment ago, so any record acquired
before that boot cannot be about this device. Only a record acquired after the boot -- another
agent claiming the new emulator in the gap -- is a real holder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import leases
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceLeasedError
from android_ui_analyser.platforms.virtual_targets import (
    OwnedVirtualTargetStopRequest,
    VirtualTargetInstance,
    VirtualTargetProvisionRequest,
    VirtualTargetStopResult,
)
from android_ui_analyser.schema import DeviceInfo
from conftest import make_config


def plant_stale_record(registry: Path, serial: str, *, owner: str, age_s: float) -> None:
    """A record from a run whose device is gone, still live by its owner's lights."""
    assert leases.acquire(registry, serial, owner=owner) is True
    path = leases._lease_path(registry, leases.target_ref(serial))
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["acquired"] = time.time() - age_s
    entry["last_activity"] = time.time()  # the owner is around; only its device is not
    path.write_text(json.dumps(entry), encoding="utf-8")


def engine_that_boots(tmp_path: Path, monkeypatch: Any, *, on_boot=None):
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(tmp_path / "coordination")},
    )
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    engine._lease_needs = ["root"]
    online: list[DeviceInfo] = []
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    monkeypatch.setattr(engine.platform, "probe_target_capabilities",
                        lambda _serial: {"root": True, "headed": True})
    calls: list[str] = []

    class VirtualDevices:
        def provision_virtual_target(self, request: VirtualTargetProvisionRequest) -> VirtualTargetInstance:
            calls.append("provision")
            online.append(DeviceInfo(serial="emulator-5556", model="fresh", android_version="14"))
            if on_boot:
                on_boot()
            return VirtualTargetInstance(target_id="emulator-5556", definition_id="rootable",
                                         instance_token="rootable.p5556")

        def stop_virtual_target_instance(self, request: OwnedVirtualTargetStopRequest) -> VirtualTargetStopResult:
            calls.append("stop")
            return VirtualTargetStopResult(stopped_target_ids=("emulator-5556",))

    monkeypatch.setattr(engine.platform, "capability",
                        lambda name: VirtualDevices() if name == "virtual_targets" else None)
    return engine, cfg, calls


def test_a_record_older_than_the_boot_is_dropped_and_the_fresh_emulator_is_claimed(tmp_path, monkeypatch) -> None:
    engine, cfg, calls = engine_that_boots(tmp_path, monkeypatch)
    plant_stale_record(Path(cfg.lease.registry_dir), "emulator-5556", owner="yesterdays-run", age_s=23 * 3600)

    prepared = engine._prepare_session_target(wait_for_lease_s=0, start_emulator=True, headed=True, audio=False)

    assert prepared["serial"] == "emulator-5556" and prepared["emulator_started"] is True
    assert leases.holder(cfg.lease.registry_dir, "emulator-5556") == "session-agent"
    assert calls == ["provision"], "nothing was rolled back"


def test_a_record_acquired_after_the_boot_is_a_real_holder(tmp_path, monkeypatch) -> None:
    """Another agent that claims the new emulator in the gap keeps it; our boot is rolled back."""
    registry = tmp_path / "coordination"

    def someone_grabs_it() -> None:
        assert leases.acquire(registry, "emulator-5556", owner="quick-agent") is True

    engine, cfg, calls = engine_that_boots(tmp_path, monkeypatch, on_boot=someone_grabs_it)

    with pytest.raises(DeviceLeasedError):
        engine._prepare_session_target(wait_for_lease_s=0, start_emulator=True, headed=True, audio=False)

    assert leases.holder(cfg.lease.registry_dir, "emulator-5556") == "quick-agent"
    assert "stop" in calls


def test_forget_predating_is_the_narrow_rule_it_says(tmp_path) -> None:
    registry = tmp_path / "coordination"
    plant_stale_record(registry, "emulator-5556", owner="old-run", age_s=3600)

    assert leases.forget_predating(registry, "emulator-5556", booted_at=time.time() - 7200) is False, \
        "a record acquired after the boot is kept"
    assert leases.holder(registry, "emulator-5556") == "old-run"
    assert leases.forget_predating(registry, "emulator-5556", booted_at=time.time()) is True
    assert leases.holder(registry, "emulator-5556") is None
    assert leases.forget_predating(registry, "emulator-5556", booted_at=time.time()) is False, "nothing left to drop"
