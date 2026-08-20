"""Session start owns device discovery, provisioning, and process-bound leasing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from android_ui_analyser import daemon as daemon_mod
from android_ui_analyser import emulator, leases
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
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
        def select_avd_for_session(self, avd: str | None, *, needs: list[str]) -> str:
            calls.append(("select", {"avd": avd, "needs": needs}))
            return "rootable-api34"

        def start(self, avd: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(("start", {"avd": avd, **kwargs}))
            online.append(
                DeviceInfo(serial="emulator-5558", model="new", android_version="14")
            )
            return {"ok": True, "serial": "emulator-5558", "avd": avd}

        def stop(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("stop", kwargs))
            return {"ok": True, "stopped": [kwargs["serial"]]}

    virtual = VirtualDevices()
    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: virtual if name == "virtual_devices" else None,
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
    assert calls[0] == (
        "select",
        {"avd": None, "needs": ["root", "proxy"]},
    )
    start = calls[1][1]
    assert start["avd"] == "rootable-api34"
    assert start["headless"] is False
    assert start["audio"] is True
    assert start["parallel"] is True
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
        def select_avd_for_session(avd: str | None, *, needs: list[str]) -> str:
            assert avd is None
            assert needs == []
            return "visible-image"

        @staticmethod
        def start(avd: str, **_kwargs: Any) -> dict[str, Any]:
            assert avd == "visible-image"
            online.append(
                DeviceInfo(serial="emulator-5556", model="new", android_version="14")
            )
            return {"ok": True, "serial": "emulator-5556", "avd": avd}

        @staticmethod
        def stop(**_kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "stopped": []}

    monkeypatch.setattr(
        engine.platform,
        "capability",
        lambda name: VirtualDevices() if name == "virtual_devices" else None,
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

    finished = engine.session_finish(state.session_id)

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
