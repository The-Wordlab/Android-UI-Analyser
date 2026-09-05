"""Session start owns device discovery, provisioning, and process-bound leasing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import daemon as daemon_mod
from android_ui_analyser import emulator, leases, network
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.base import InstalledApp
from android_ui_analyser.platforms.virtual_targets import (
    OwnedVirtualTargetStopRequest,
    VirtualTargetInstance,
    VirtualTargetProvisionRequest,
    VirtualTargetStopResult,
)
from android_ui_analyser.schema import DeviceInfo
from android_ui_analyser.session import create_session_state
from conftest import make_config


def test_two_run_caches_share_one_lease_authority(tmp_path: Path, monkeypatch: Any) -> None:
    registry = tmp_path / "coordination"
    cfg = make_config(
        cache={"dir": str(tmp_path / "worker-b")},
        lease={"registry_dir": str(registry)},
    )
    engine = Engine(cfg)
    probes: list[str] = []
    targets = [
        DeviceInfo(serial="emulator-5554", model="first", android_version="14"),
        DeviceInfo(serial="emulator-5556", model="second", android_version="14"),
    ]
    monkeypatch.setattr(engine, "_list_targets", lambda: targets)
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)

    def capabilities(serial: str) -> dict[str, bool]:
        probes.append(serial)
        return {"root": True, "proxy": True, "play": False}

    monkeypatch.setattr(engine.platform, "probe_target_capabilities", capabilities)
    assert leases.acquire(
        registry,
        "emulator-5554",
        owner="another-live-agent",
    )
    engine._lease_owner = "this-agent"
    engine._lease_needs = ["root"]

    prepared = engine._prepare_session_target(
        wait_for_lease_s=0,
        start_emulator=True,
        headed=False,
        audio=False,
        avd=None,
    )

    assert prepared["serial"] == "emulator-5556"
    assert probes == ["emulator-5554", "emulator-5556"]
    assert leases.holder(registry, "emulator-5554") == "another-live-agent"
    assert leases.holder(registry, "emulator-5556") == "this-agent"
    assert not (Path(cfg.cache.dir) / "leases").exists()


def test_app_bootstrap_provisionally_leases_and_skips_targets_without_app(
    tmp_path: Path, monkeypatch: Any
) -> None:
    registry = tmp_path / "coordination"
    cfg = make_config(lease={"registry_dir": str(registry)})
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    targets = [
        DeviceInfo(serial="emulator-5554", state="device"),
        DeviceInfo(serial="emulator-5556", state="device"),
    ]
    monkeypatch.setattr(engine, "_list_targets", lambda: targets)
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    connected: list[str] = []

    def connect(serial: str | None) -> Any:
        assert serial is not None
        connected.append(serial)
        return SimpleNamespace(serial=serial, close=lambda: None)

    monkeypatch.setattr(engine, "_connect_target", connect)
    monkeypatch.setattr(
        engine.platform,
        "installed_app",
        lambda runtime, app_id: InstalledApp(
            app_id=app_id,
            installed=runtime.serial == "emulator-5556",
        ),
    )

    prepared = engine._prepare_session_target(
        wait_for_lease_s=0,
        start_emulator=True,
        headed=False,
        audio=False,
        avd=None,
        package="com.example.notes",
    )

    assert prepared["serial"] == "emulator-5556"
    assert connected == ["emulator-5554", "emulator-5556"]
    assert leases.holder(registry, "emulator-5554") is None
    assert leases.holder(registry, "emulator-5556") == "session-agent"


def test_busy_app_pool_provisions_without_connecting_to_the_foreign_target(
    tmp_path: Path, monkeypatch: Any
) -> None:
    registry = tmp_path / "coordination"
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(registry)},
    )
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    online = [DeviceInfo(serial="emulator-5554", state="device")]
    assert leases.acquire(registry, "emulator-5554", owner="rename-agent")
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    connected: list[str] = []

    def connect(serial: str | None) -> Any:
        assert serial is not None
        connected.append(serial)
        return SimpleNamespace(serial=serial, close=lambda: None)

    monkeypatch.setattr(engine, "_connect_target", connect)
    monkeypatch.setattr(
        engine.platform,
        "installed_app",
        lambda runtime, app_id: InstalledApp(
            app_id=app_id,
            installed=runtime.serial == "emulator-5556",
        ),
    )

    class VirtualDevices:
        @staticmethod
        def provision_virtual_target(
            request: VirtualTargetProvisionRequest,
        ) -> VirtualTargetInstance:
            assert request.definition_id is None
            assert request.needs == ()
            online.append(DeviceInfo(serial="emulator-5556", state="device"))
            return VirtualTargetInstance(
                target_id="emulator-5556",
                definition_id="test-ready",
                instance_token="test-ready.p5556",
                pid=6556,
            )

        @staticmethod
        def stop_virtual_target_instance(
            request: OwnedVirtualTargetStopRequest,
        ) -> VirtualTargetStopResult:
            raise AssertionError(f"successful provisioning rolled back: {request}")

    virtual = VirtualDevices()
    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: virtual if name == "virtual_targets" else None,
    )

    prepared = engine._prepare_session_target(
        wait_for_lease_s=0,
        start_emulator=True,
        headed=False,
        audio=False,
        avd=None,
        package="com.example.notes",
    )

    assert prepared["serial"] == "emulator-5556"
    assert prepared["emulator_started"] is True
    assert connected == ["emulator-5556"]
    assert leases.holder(registry, "emulator-5554") == "rename-agent"
    assert leases.holder(registry, "emulator-5556") == "session-agent"


def test_app_bootstrap_never_switches_a_sticky_target_missing_the_app(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser.errors import DeviceError

    registry = tmp_path / "coordination"
    cfg = make_config(lease={"registry_dir": str(registry)})
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    assert leases.acquire(registry, "emulator-5554", owner="session-agent")
    targets = [
        DeviceInfo(serial="emulator-5554", state="device"),
        DeviceInfo(serial="emulator-5556", state="device"),
    ]
    monkeypatch.setattr(engine, "_list_targets", lambda: targets)
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    monkeypatch.setattr(
        engine,
        "_connect_target",
        lambda serial: SimpleNamespace(serial=serial, close=lambda: None),
    )
    monkeypatch.setattr(
        engine.platform,
        "installed_app",
        lambda _runtime, app_id: InstalledApp(app_id=app_id, installed=False),
    )

    with pytest.raises(DeviceError) as caught:
        engine._prepare_session_target(
            wait_for_lease_s=0,
            start_emulator=True,
            headed=False,
            audio=False,
            avd=None,
            package="com.example.notes",
        )

    assert caught.value.code == "required_app_not_installed_on_leased_target"
    assert leases.holder(registry, "emulator-5554") == "session-agent"
    assert leases.holder(registry, "emulator-5556") is None


def test_apk_bootstrap_does_not_require_the_app_to_be_preinstalled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    registry = tmp_path / "coordination"
    cfg = make_config(lease={"registry_dir": str(registry)})
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    monkeypatch.setattr(
        engine,
        "_list_targets",
        lambda: [DeviceInfo(serial="emulator-5554", state="device")],
    )
    monkeypatch.setattr(
        engine.platform,
        "installed_app",
        lambda *_args: (_ for _ in ()).throw(AssertionError("checked before install")),
    )

    prepared = engine._prepare_session_target(
        wait_for_lease_s=0,
        start_emulator=True,
        headed=False,
        audio=False,
        avd=None,
        package="com.example.notes",
        app_will_be_installed=True,
    )

    assert prepared["serial"] == "emulator-5554"
    assert leases.holder(registry, "emulator-5554") == "session-agent"


def test_inventory_failure_never_provisions_an_emulator(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from android_ui_analyser.errors import DeviceError

    cfg = make_config(lease={"registry_dir": str(tmp_path / "coordination")})
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"

    def fail_inventory() -> list[DeviceInfo]:
        raise DeviceError("inventory failed", code="target_inventory_unavailable")

    monkeypatch.setattr(engine, "_list_targets", fail_inventory)
    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: (_ for _ in ()).throw(AssertionError(f"provisioned through {name}")),
    )

    with pytest.raises(DeviceError) as caught:
        engine._prepare_session_target(
            wait_for_lease_s=0,
            start_emulator=True,
            headed=False,
            audio=False,
            avd=None,
        )

    assert caught.value.code == "target_inventory_unavailable"


def test_dead_owner_is_reclaimed_during_the_same_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    alive = {111, 222}
    starts = {111: "old", 222: "new"}

    def kill(pid: int, _signal: int) -> None:
        if pid not in alive:
            raise ProcessLookupError

    monkeypatch.setattr(leases.os, "kill", kill)
    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    old = leases.LeaseOwner("old-agent", pid=111, started="old")
    new = leases.LeaseOwner("new-agent", pid=222, started="new")
    assert leases.acquire(tmp_path, "emulator-5554", owner=old)
    alive.remove(111)

    serial, why = leases.choose_device(
        tmp_path,
        owner=new,
        explicit=None,
        candidates=[("emulator-5554", {})],
    )

    assert (serial, why) == ("emulator-5554", "assigned")
    assert leases.holder(tmp_path, serial) == "new-agent"


def test_session_provisions_matching_avd_and_claims_it_automatically(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(tmp_path / "coordination")},
    )
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    engine._lease_needs = ["root", "proxy"]
    online: list[DeviceInfo] = []
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    monkeypatch.setattr(
        engine.platform,
        "probe_target_capabilities",
        lambda _serial: {
            "root": True,
            "proxy": True,
            "play": False,
            "headed": True,
            "audio": True,
        },
    )
    calls: list[tuple[str, Any]] = []

    class VirtualDevices:
        def provision_virtual_target(
            self, request: VirtualTargetProvisionRequest
        ) -> VirtualTargetInstance:
            calls.append(("provision", request))
            online.append(
                DeviceInfo(serial="emulator-5558", model="new", android_version="14")
            )
            return VirtualTargetInstance(
                target_id="emulator-5558",
                definition_id="rootable-api34",
                instance_token="rootable-api34.p5558",
            )

        def stop_virtual_target_instance(
            self, request: OwnedVirtualTargetStopRequest
        ) -> VirtualTargetStopResult:
            calls.append(("stop", request))
            return VirtualTargetStopResult(stopped_target_ids=("emulator-5558",))

    virtual = VirtualDevices()
    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: virtual if name == "virtual_targets" else None,
    )

    prepared = engine._prepare_session_target(
        wait_for_lease_s=0,
        start_emulator=True,
        headed=True,
        audio=True,
        avd=None,
    )

    assert prepared["emulator_started"] is True
    assert prepared["serial"] == "emulator-5558"
    assert calls[0][0] == "provision"
    start = calls[0][1]
    assert start.definition_id is None
    assert start.needs == ("root", "proxy", "headed", "audio")
    assert start.headless is False
    assert start.audio is True
    assert start.animations is False
    assert start.parallel is True
    assert leases.holder(cfg.lease.registry_dir, "emulator-5558") == "session-agent"
    assert all(name != "stop" for name, _detail in calls)


def test_headed_session_does_not_reuse_an_unverifiably_headless_emulator(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(tmp_path / "coordination")},
    )
    engine = Engine(cfg)
    online = [DeviceInfo(serial="emulator-5554", model="old", android_version="14")]
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    monkeypatch.setattr(
        engine.platform,
        "probe_target_capabilities",
        lambda serial: {
            "root": True,
            "proxy": True,
            "headed": serial == "emulator-5556",
            "audio": False,
        },
    )

    class VirtualDevices:
        @staticmethod
        def provision_virtual_target(
            request: VirtualTargetProvisionRequest,
        ) -> VirtualTargetInstance:
            assert request.definition_id is None
            assert request.needs == ("headed",)
            online.append(
                DeviceInfo(serial="emulator-5556", model="new", android_version="14")
            )
            return VirtualTargetInstance(
                target_id="emulator-5556",
                definition_id="visible-image",
                instance_token="visible-image.p5556",
            )

        @staticmethod
        def stop_virtual_target_instance(
            _request: OwnedVirtualTargetStopRequest,
        ) -> VirtualTargetStopResult:
            return VirtualTargetStopResult()

    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: VirtualDevices() if name == "virtual_targets" else None,
    )

    prepared = engine._prepare_session_target(
        wait_for_lease_s=0,
        start_emulator=True,
        headed=True,
        audio=False,
        avd=None,
    )

    assert prepared["serial"] == "emulator-5556"
    assert prepared["emulator_started"] is True


def test_avd_selection_uses_image_capabilities_and_a_stable_default(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        emulator,
        "list_avds",
        lambda: {
            "details": [
                {"name": "play-image", "rootable": False, "play_store": True},
                {"name": "root-image", "rootable": True, "play_store": False},
            ]
        },
    )

    assert emulator.select_avd_for_session() == "play-image"
    assert emulator.select_avd_for_session(needs=["play"]) == "play-image"
    assert emulator.select_avd_for_session(needs=["root", "proxy"]) == "root-image"


def test_session_finish_releases_the_automatic_process_lease(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(tmp_path / "coordination")},
    )
    engine = Engine(cfg)
    owner = leases.resolve_owner("session-agent")
    serial = "emulator-5554"
    assert leases.acquire(cfg.lease.registry_dir, serial, owner=owner)
    engine._lease_owner = owner
    engine._lease_owner_resolved = owner
    engine._lease_serial = serial
    engine._leased_serial_resolved = (True, serial)
    state = create_session_state(
        cfg.cache.dir,
        owner=str(owner),
        serial=serial,
        goal="finish lease test",
        recommended_kind="manual_action",
        recommended_cli="aua analyze",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
        emulator_started=False,
    )
    monkeypatch.setattr(engine, "session_review", lambda _session_id: {"ok": True})

    finished = engine.session_finish(state.session_id, allow_incomplete=True)

    assert finished["ok"] is True
    assert leases.read_lease(cfg.lease.registry_dir, serial) is None
    release = next(item for item in finished["cleanup"] if item["action"] == "lease_release")
    assert release["ok"] is True


def test_daemon_refuses_an_old_runs_cache_before_touching_its_engine(tmp_path: Path) -> None:
    old = Config()
    old.cache.dir = str(tmp_path / "old-run")
    old.lease.registry_dir = str(tmp_path / "coordination")
    new = Config()
    new.cache.dir = str(tmp_path / "new-run")
    new.lease.registry_dir = str(tmp_path / "coordination")
    touched: list[str] = []
    engine = SimpleNamespace(config=old, analyze=lambda: touched.append("analyze"))

    response = daemon_mod.dispatch(
        engine,
        {
            "cmd": "analyze",
            "args": {},
            "runtime_fingerprint": daemon_mod.runtime_config_fingerprint(new),
        },
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "daemon_runtime_mismatch"
    assert touched == []


def test_claim_rollback_never_stops_a_foreign_leased_serial(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The live failure, end to end at the engine seam.

    Provisioning collided with another worker's leased emulator (adb had blinked, so its
    console port looked free); the claim then failed with device_leased — and the rollback
    used to issue `stop(serial=...)`, killing the foreign worker's device mid-run. The
    rollback may reap only the boot it performed (instance + pid), and the foreign lease
    must survive untouched.
    """
    import os

    from android_ui_analyser.errors import DeviceLeasedError

    registry = tmp_path / "coordination"
    foreign = leases.LeaseOwner(
        "foreign-worker", pid=os.getpid(), started=leases._proc_started(os.getpid())
    )
    assert leases.acquire(registry, "emulator-5554", owner=foreign)

    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(registry)},
    )
    engine = Engine(cfg)
    engine._lease_owner = "session-agent"
    online = [DeviceInfo(serial="emulator-5554", model="busy", android_version="14")]
    monkeypatch.setattr(engine, "_list_targets", lambda: list(online))
    monkeypatch.setattr(engine.platform, "target_preference", lambda info: info.serial)
    monkeypatch.setattr(engine.platform, "probe_target_capabilities", lambda _serial: {})
    calls: list[tuple[str, dict[str, Any]]] = []

    class VirtualDevices:
        def provision_virtual_target(
            self, request: VirtualTargetProvisionRequest
        ) -> VirtualTargetInstance:
            calls.append(("provision", {"request": request}))
            # The collision: the "new" boot answers on the foreign worker's serial.
            return VirtualTargetInstance(
                target_id="emulator-5554",
                instance_token="explicit-avd.p5554",
                pid=64064,
                definition_id=request.definition_id,
            )

        def stop_virtual_target_instance(
            self, request: OwnedVirtualTargetStopRequest
        ) -> VirtualTargetStopResult:
            calls.append(("stop_virtual_target_instance", {"request": request}))
            return VirtualTargetStopResult()

    virtual = VirtualDevices()
    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: virtual if name == "virtual_targets" else None,
    )

    with pytest.raises(DeviceLeasedError):
        engine._prepare_session_target(
            wait_for_lease_s=0,
            start_emulator=True,
            headed=False,
            audio=False,
            avd="explicit-avd",
        )

    rollback = dict(calls)["stop_virtual_target_instance"]["request"]
    assert rollback.instance_token == "explicit-avd.p5554"
    assert rollback.expected_pid == 64064
    assert rollback.requested_by == "session-start-claim-rollback"
    assert str(rollback.lease_registry_dir) == str(registry)
    assert leases.holder(registry, "emulator-5554") == "foreign-worker"


def test_session_finish_failure_retains_the_lease_until_cleanup_succeeds(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cfg = make_config(
        cache={"dir": str(tmp_path / "run")},
        lease={"registry_dir": str(tmp_path / "coordination")},
    )
    engine = Engine(cfg)
    owner = leases.resolve_owner("session-agent")
    serial = "emulator-5554"
    assert leases.acquire(cfg.lease.registry_dir, serial, owner=owner)
    engine._lease_owner = owner
    engine._lease_owner_resolved = owner
    engine._lease_serial = serial
    engine._leased_serial_resolved = (True, serial)
    state = create_session_state(
        cfg.cache.dir,
        owner=str(owner),
        serial=serial,
        goal="cleanup failure keeps ownership",
        recommended_kind="manual_action",
        recommended_cli="aua analyze",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
        emulator_started=True,
    )
    monkeypatch.setattr(engine, "session_review", lambda _session_id: {"ok": True})
    backup = network.backup_path(cfg.cache.dir, serial)
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("session-owned", encoding="utf-8")
    restore_results = iter(
        [
            {"ok": False, "detail": "device unreachable"},
            {"ok": True},
        ]
    )
    monkeypatch.setattr(engine, "network_restore", lambda: next(restore_results))

    failed = engine.session_finish(state.session_id, allow_incomplete=True)

    assert failed["ok"] is False
    assert leases.holder(cfg.lease.registry_dir, serial) == str(owner), (
        "failed cleanup must retain the lease so the same process can retry"
    )

    retried = engine.session_finish(state.session_id, allow_incomplete=True)

    assert retried["ok"] is True
    assert leases.read_lease(cfg.lease.registry_dir, serial) is None
    handoff = next(
        item for item in retried["cleanup"] if item["action"] == "owned_emulator_handoff"
    )
    assert handoff["result"] == {
        "ok": True,
        "serial": serial,
        "retained": True,
        "leased": False,
        "auto_stop": True,
        "idle_stop_s": 1200.0,
    }
