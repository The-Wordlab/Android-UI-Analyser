"""A proxy start's CA relaunch brings the app back on the Activity that was in front.

A dev build can have two launchers, the product and a developer-tools screen. A QA setup opened
the tools screen, started the proxy, then opened the tools screen again to pick a backend. The
relaunch after the CA install used the default launcher, so the product's splash came up and
landed on top of that second launch: the backend flow found the product's welcome screen instead
of the tools list (2026-09-28).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from conftest import FakeDevice, make_config

DEV_TOOLS = "co.example.devtools.DevToolsActivity"


def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: FakeDevice) -> dict[str, Any]:
    def fake_start(*, cache_dir: Path, port: int | None = None, mode: str = "map",
                   serial: str | None = None) -> tuple[int, int]:
        pm.save_listen_port(cache_dir, 43210)
        return 12345, 43210

    monkeypatch.setattr(pm, "start_mitm", fake_start)
    monkeypatch.setattr(pm, "install_system_ca", lambda *_a, **_k: {"ok": True, "hash": "abc"})
    engine = Engine(make_config(cache={"dir": str(tmp_path / "cache")}), device=device)
    return engine.proxy_start(install_ca=True)


def test_the_relaunch_reopens_the_activity_that_was_in_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(activity=DEV_TOOLS)
    _start(tmp_path, monkeypatch, device)
    launches = [args for name, args in device.calls if name == "launch_app"]
    assert launches == [(device._pkg, DEV_TOOLS)]


def test_a_refused_activity_falls_back_to_the_default_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(activity=".InternalActivity")
    launch = device.launch_app

    def refuse_explicit(package: str, *, activity: str | None = None, arguments: Any = ()) -> None:
        if activity is not None:
            raise DeviceError("am start refused: not exported")
        launch(package)

    device.launch_app = refuse_explicit  # type: ignore[method-assign]
    _start(tmp_path, monkeypatch, device)
    assert [args for name, args in device.calls if name == "launch_app"] == [(device._pkg,)]
