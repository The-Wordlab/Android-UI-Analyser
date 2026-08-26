"""The shared Android transport has one crash-safe cold-start coordinator."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser import device, emulator
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_transport
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.schema import DeviceInfo
from conftest import make_config


def _cold_start_worker(
    registry: str,
    ready_path: str,
    starts_path: str,
    results: multiprocessing.Queue,
) -> None:
    ready = Path(ready_path)
    starts = Path(starts_path)
    android_transport.adb_server_endpoint = lambda: ("127.0.0.1", 5037)  # type: ignore[method-assign]
    android_transport.adb_bin = lambda: "/fake/adb"  # type: ignore[assignment]
    android_transport.ensure_adb_on_path = lambda: "/fake/adb"  # type: ignore[assignment]
    android_transport._adb_server_ready = (  # type: ignore[assignment]
        lambda _host, _port: ready.exists()
    )

    def start(_adb: str, *, timeout_s: float) -> subprocess.CompletedProcess[str]:
        del timeout_s
        with starts.open("a", encoding="utf-8") as handle:
            handle.write("start\n")
        time.sleep(0.1)
        ready.write_text("ready\n", encoding="utf-8")
        return subprocess.CompletedProcess(["adb", "start-server"], 0, "", "")

    android_transport._start_adb_server = start  # type: ignore[assignment]
    try:
        android_transport.ensure_adb_server_ready(registry)
        results.put("ok")
    except Exception as exc:  # pragma: no cover - assertion reports child failure
        results.put(f"{type(exc).__name__}: {exc}")


def test_parallel_cold_start_invokes_one_adb_server_start(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    ready = tmp_path / "ready"
    starts = tmp_path / "starts"
    workers = [
        ctx.Process(
            target=_cold_start_worker,
            args=(str(tmp_path / "coordination"), str(ready), str(starts), results),
        )
        for _ in range(5)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    assert [results.get(timeout=1) for _ in workers] == ["ok"] * 5
    assert starts.read_text(encoding="utf-8").splitlines() == ["start"]


def _parallel_provision_worker(
    worker: int,
    cache_dir: str,
    registry_dir: str,
    lock_dir: str,
    ready_path: str,
    starts_path: str,
    failure_path: str,
    online_dir: str,
    results: multiprocessing.Queue,
) -> None:
    ready = Path(ready_path)
    starts = Path(starts_path)
    failure = Path(failure_path)
    online = Path(online_dir)
    android_transport.adb_server_endpoint = lambda: ("127.0.0.1", 5037)  # type: ignore[method-assign]
    android_transport._adb_coordination_dir = lambda: Path(lock_dir)  # type: ignore[assignment]
    android_transport.adb_bin = lambda: "/fake/adb"  # type: ignore[assignment]
    android_transport.ensure_adb_on_path = lambda: "/fake/adb"  # type: ignore[assignment]
    android_transport._adb_server_ready = (  # type: ignore[assignment]
        lambda _host, _port: ready.exists()
    )

    def start_server(_adb: str, *, timeout_s: float) -> subprocess.CompletedProcess[str]:
        del timeout_s
        with starts.open("a", encoding="utf-8") as handle:
            handle.write("start\n")
        time.sleep(0.05)
        ready.write_text("ready\n", encoding="utf-8")
        return subprocess.CompletedProcess(["adb", "start-server"], 0, "", "")

    android_transport._start_adb_server = start_server  # type: ignore[assignment]
    emulator.list_avds = lambda: {  # type: ignore[assignment]
        "ok": True,
        "avds": ["parallel"],
        "count": 1,
        "emulator": "/fake/emulator",
    }
    emulator.emulator_bin = lambda: "/fake/emulator"  # type: ignore[assignment]
    emulator.avd_name_of_serial = lambda _serial: "parallel"  # type: ignore[assignment]
    emulator._wait_for_boot = lambda *_args, **_kwargs: True  # type: ignore[assignment]
    emulator._clear_inherited_blackholed_proxy = (  # type: ignore[assignment]
        lambda *_args, **_kwargs: {"ok": True, "checked": True, "cleared": False}
    )

    def inventory() -> list[DeviceInfo]:
        if not failure.exists():
            failure.write_text("injected Unknown data: b''\n", encoding="utf-8")
            raise DeviceError("could not list devices: Unknown data: b''")
        return [
            DeviceInfo(serial=path.name, state="device")
            for path in sorted(online.glob("emulator-*"))
        ]

    device.list_devices = inventory  # type: ignore[assignment]

    class FakeProcess:
        pid = os.getpid()

        @staticmethod
        def poll() -> None:
            return None

    def popen(cmd: list[str], **_kwargs: object) -> FakeProcess:
        port = int(cmd[cmd.index("-port") + 1])
        (online / f"emulator-{port}").write_text(str(worker), encoding="utf-8")
        return FakeProcess()

    emulator.subprocess.Popen = popen  # type: ignore[assignment]
    try:
        boot = emulator.start(
            "parallel",
            cache_dir=cache_dir,
            lease_registry_dir=registry_dir,
            owner=f"worker-{worker}",
            parallel=True,
            animations=True,
            idle_timeout_s=0,
            wait_s=5,
        )
        results.put({"serial": boot["serial"], "instance": boot["instance"]})
    except Exception as exc:  # pragma: no cover - assertion reports child failure
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def test_parallel_session_provisioning_launches_once_per_worker_under_one_transport_lock(
    tmp_path: Path,
) -> None:
    """Exercise the real multiprocess boot/port/wait path, not only the readiness primitive."""

    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    cache = tmp_path / "cache"
    online = tmp_path / "online"
    online.mkdir()
    workers = [
        ctx.Process(
            target=_parallel_provision_worker,
            args=(
                index,
                str(cache),
                str(tmp_path / f"registry-{index}"),
                str(tmp_path / "endpoint-lock"),
                str(tmp_path / "ready"),
                str(tmp_path / "starts"),
                str(tmp_path / "first-inventory-failure"),
                str(online),
                results,
            ),
        )
        for index in range(5)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0

    outcomes = [results.get(timeout=1) for _ in workers]
    assert all("error" not in item for item in outcomes), outcomes
    assert len({item["serial"] for item in outcomes}) == 5
    assert len(list(online.glob("emulator-*"))) == 5, "no worker retried or leaked a second boot"
    assert (tmp_path / "first-inventory-failure").is_file()
    assert (tmp_path / "starts").read_text(encoding="utf-8").splitlines() == ["start"]


def test_remote_endpoint_is_never_started_locally(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(android_transport, "adb_server_endpoint", lambda: ("10.0.0.5", 5037))
    monkeypatch.setattr(android_transport, "_adb_server_ready", lambda _host, _port: False)
    monkeypatch.setattr(
        android_transport,
        "_start_adb_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("started locally")),
    )

    try:
        android_transport.ensure_adb_server_ready(tmp_path)
    except Exception as exc:
        assert getattr(exc, "code", None) == "adb_server_unavailable"
    else:  # pragma: no cover - explicit assertion keeps the expected type flexible
        raise AssertionError("unavailable remote endpoint was accepted")


def test_coordinated_inventory_publishes_stock_sdk_adb_before_adbutils(
    tmp_path: Path, monkeypatch
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        android_transport,
        "ensure_adb_on_path",
        lambda: order.append("path") or "/stock/android-studio/platform-tools/adb",
    )
    monkeypatch.setattr(android_transport, "adb_server_endpoint", lambda: ("127.0.0.1", 5037))
    monkeypatch.setattr(android_transport, "_adb_server_ready", lambda *_args: True)
    monkeypatch.setattr(android_transport, "_adb_coordination_dir", lambda: tmp_path)

    result = android_transport.run_adb_server_operation(
        tmp_path / "ignored-registry",
        lambda: order.append("adbutils") or ["emulator-5554"],
    )

    assert result == ["emulator-5554"]
    assert order == ["path", "adbutils"]


def test_read_only_inventory_retries_one_transient_locked_failure(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0
    monkeypatch.setattr(android_transport, "ensure_adb_on_path", lambda: "/fake/adb")
    monkeypatch.setattr(android_transport, "adb_server_endpoint", lambda: ("127.0.0.1", 5037))
    monkeypatch.setattr(android_transport, "_adb_server_ready", lambda *_args: True)
    monkeypatch.setattr(android_transport, "_adb_coordination_dir", lambda: tmp_path)
    monkeypatch.setattr(android_transport.time, "sleep", lambda _seconds: None)

    def inventory() -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DeviceError("could not list devices: Unknown data: b''")
        return ["emulator-5554"]

    assert android_transport.run_adb_inventory_operation(tmp_path, inventory) == [
        "emulator-5554"
    ]
    assert calls == 2


def test_non_inventory_adb_operation_is_never_retried(tmp_path: Path, monkeypatch) -> None:
    calls = 0
    monkeypatch.setattr(android_transport, "ensure_adb_on_path", lambda: "/fake/adb")
    monkeypatch.setattr(android_transport, "adb_server_endpoint", lambda: ("127.0.0.1", 5037))
    monkeypatch.setattr(android_transport, "_adb_server_ready", lambda *_args: True)
    monkeypatch.setattr(android_transport, "_adb_coordination_dir", lambda: tmp_path)

    def connect_or_act() -> None:
        nonlocal calls
        calls += 1
        raise DeviceError("outcome unknown")

    with pytest.raises(DeviceError, match="outcome unknown"):
        android_transport.run_adb_server_operation(tmp_path, connect_or_act)
    assert calls == 1


def test_persistently_failed_inventory_stops_after_one_retry(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0
    monkeypatch.setattr(android_transport, "ensure_adb_on_path", lambda: "/fake/adb")
    monkeypatch.setattr(android_transport, "adb_server_endpoint", lambda: ("127.0.0.1", 5037))
    monkeypatch.setattr(android_transport, "_adb_server_ready", lambda *_args: True)
    monkeypatch.setattr(android_transport, "_adb_coordination_dir", lambda: tmp_path)
    monkeypatch.setattr(android_transport.time, "sleep", lambda _seconds: None)

    def inventory() -> None:
        nonlocal calls
        calls += 1
        raise DeviceError("could not list devices: Unknown data: b''")

    with pytest.raises(DeviceError, match="Unknown data"):
        android_transport.run_adb_inventory_operation(tmp_path, inventory)
    assert calls == 2


def test_device_inventory_preserves_offline_and_unauthorized_states(monkeypatch) -> None:
    transports = [
        SimpleNamespace(serial="emulator-5554", state="device", tags={"model": "pixel"}),
        SimpleNamespace(serial="emulator-5556", state="offline", tags={}),
        SimpleNamespace(serial="phone-1", state="unauthorized", tags={"model": "phone"}),
    ]
    online = SimpleNamespace(
        serial="emulator-5554",
        prop=SimpleNamespace(model="Pixel Test"),
        getprop=lambda key: {
            "ro.build.version.release": "14",
            "persist.sys.locale": "en-US",
            "ro.product.locale": "",
        }[key],
    )
    fake_adb = SimpleNamespace(
        list=lambda: transports,
        device=lambda *, serial: online
        if serial == "emulator-5554"
        else (_ for _ in ()).throw(AssertionError(f"enriched {serial}")),
    )
    monkeypatch.setitem(sys.modules, "adbutils", SimpleNamespace(adb=fake_adb))

    inventory = device.list_devices()

    assert [(item.serial, item.state) for item in inventory] == [
        ("emulator-5554", "device"),
        ("emulator-5556", "offline"),
        ("phone-1", "unauthorized"),
    ]
    assert inventory[0].model == "Pixel Test"
    assert inventory[0].locale == "en-US"


def test_android_platform_prepares_transport_through_host_coordinator(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "coordination"
    platform = AndroidPlatform(make_config(lease={"registry_dir": str(registry)}))
    prepared: list[Path] = []
    from android_ui_analyser import emulator

    monkeypatch.setattr(emulator, "ensure_adb_on_path", lambda: "/fake/adb")
    monkeypatch.setattr(
        android_transport,
        "ensure_adb_server_ready",
        lambda path: prepared.append(Path(path)),
    )

    platform.prepare_host()

    assert prepared == [registry]
