"""A transport must be selected from the lease, never before it."""

from __future__ import annotations

from types import SimpleNamespace

import android_ui_analyser.cli as cli
import android_ui_analyser.daemon as daemon_mod
from android_ui_analyser.config import Config


def test_unpinned_route_uses_the_leased_devices_socket(tmp_path, monkeypatch) -> None:
    cfg = Config()
    cfg.cache.dir = str(tmp_path / "cache")
    cfg.daemon.socket = str(tmp_path / "daemon.sock")
    cfg.device.serial = None
    sockets: list[str] = []

    class Client:
        def __init__(self, socket: str, **_kwargs: object) -> None:
            sockets.append(socket)

        def call(self, _cmd: str, **_kwargs: object) -> dict[str, object]:
            return {"ok": True, "result": {"serial": cfg.device.serial}}

    engine = SimpleNamespace(
        config=cfg,
        _lease_serial=None,
        _lease_owner="agent-a",
        _lease_owner_resolved="agent-a",
        _lease_device=lambda: "emulator-5558",
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _cfg: daemon_mod._aua_version())
    monkeypatch.setattr(
        daemon_mod,
        "running_policy_fingerprint",
        lambda route_cfg: daemon_mod.policy_config_fingerprint(route_cfg),
    )
    monkeypatch.setattr(daemon_mod, "DaemonClient", Client)

    result = cli._route(engine, "analyze")

    assert result["serial"] == "emulator-5558"
    assert sockets == [daemon_mod.socket_path(cfg, serial="emulator-5558")]


def test_two_owners_select_two_device_daemons(tmp_path, monkeypatch) -> None:
    sockets: list[str] = []

    class Client:
        def __init__(self, socket: str, **_kwargs: object) -> None:
            sockets.append(socket)

        def call(self, _cmd: str, **_kwargs: object) -> dict[str, object]:
            return {"ok": True, "result": {}}

    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _cfg: daemon_mod._aua_version())
    monkeypatch.setattr(
        daemon_mod,
        "running_policy_fingerprint",
        lambda route_cfg: daemon_mod.policy_config_fingerprint(route_cfg),
    )
    monkeypatch.setattr(daemon_mod, "DaemonClient", Client)

    for owner, serial in (("agent-a", "emulator-5558"), ("agent-b", "emulator-5560")):
        cfg = Config()
        cfg.cache.dir = str(tmp_path / "cache")
        cfg.daemon.socket = str(tmp_path / "daemon.sock")
        cfg.device.serial = None
        engine = SimpleNamespace(
            config=cfg,
            _lease_serial=None,
            _lease_owner=owner,
            _lease_owner_resolved=owner,
            _lease_device=lambda serial=serial: serial,
        )
        cli._route(engine, "analyze")

    assert sockets[0].endswith(".emulator-5558")
    assert sockets[1].endswith(".emulator-5560")
    assert sockets[0] != sockets[1]
