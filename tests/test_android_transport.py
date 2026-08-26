"""The shared Android transport has one crash-safe cold-start coordinator."""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from android_ui_analyser import device
from android_ui_analyser.platforms import android_transport
from android_ui_analyser.platforms.android import AndroidPlatform
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
