"""Verified, reversible network isolation across engine and CLI surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.network import backup_path, parse_connectivity
from conftest import FakeDevice, make_config

runner = CliRunner()


class _ShellResponse:
    def __init__(self, output: str) -> None:
        self.output = output


class _StructuredShell:
    def shell(self, command: str) -> _ShellResponse:
        assert command == "settings get global airplane_mode_on"
        return _ShellResponse("1\n")


def test_airplane_readback_accepts_structured_uiautomator_shell_response() -> None:
    device = object.__new__(Uiautomator2Device)
    device._d = _StructuredShell()
    assert device.get_airplane_mode() is True


def test_parse_connectivity_reports_only_the_active_default() -> None:
    raw = """
Active default network: 107
Current Networks:
  NetworkAgentInfo{network{107} nc{[ Transports: WIFI|VPN Capabilities: INTERNET&VALIDATED ]}}
  NetworkAgentInfo{network{108} nc{[ Transports: CELLULAR Capabilities: INTERNET ]}}
"""
    assert parse_connectivity(raw) == {
        "active_network": True,
        "active_network_id": "107",
        "active_transports": ["wifi", "vpn"],
        "internet_validated": True,
    }
    assert parse_connectivity("Active default network: none\n") == {
        "active_network": False,
        "active_network_id": None,
        "active_transports": [],
        "internet_validated": False,
    }


def test_network_offline_is_verified_and_restore_is_reversible(tmp_path: Path) -> None:
    dev = FakeDevice(serial="fake-network")
    cfg = make_config(cache={"dir": str(tmp_path)})
    eng = Engine(cfg, device=dev)

    before = eng.network_status()
    assert before.state.active_transports == ["wifi"]
    assert before.state.offline is False

    offline = eng.network_offline(timeout_ms=0)
    assert offline.ok is True
    assert offline.verified is True
    assert offline.state.offline is True
    assert offline.state.active_transports == []
    assert offline.saved_state is not None
    assert offline.saved_state.wifi_enabled is True
    saved = backup_path(tmp_path, dev.serial)
    assert saved.is_file()

    # Re-entering offline mode must not overwrite the original online restore point.
    repeated = eng.network_offline(timeout_ms=0)
    assert repeated.saved_state is not None
    assert repeated.saved_state.wifi_enabled is True

    restored = eng.network_restore(timeout_ms=0)
    assert restored.ok is True
    assert restored.verified is True
    assert restored.state.airplane_mode is False
    assert restored.state.wifi_enabled is True
    assert restored.state.mobile_data_enabled is True
    assert restored.state.active_transports == ["wifi"]
    assert not saved.exists()


class _WifiOverrideDevice(FakeDevice):
    """Android reports 2 while Wi-Fi is enabled through its airplane-mode override."""

    def shell(self, command: str) -> str:
        if command == "settings get global wifi_on" and self._wifi_enabled:
            self.calls.append(("shell", (command,)))
            return "2"
        return super().shell(command)


def test_wifi_override_state_is_preserved_and_restored(tmp_path: Path) -> None:
    dev = _WifiOverrideDevice(serial="fake-wifi-override")
    eng = Engine(make_config(cache={"dir": str(tmp_path)}), device=dev)

    assert eng.network_status().state.wifi_enabled is True
    assert eng.network_offline(timeout_ms=0).saved_state.wifi_enabled is True
    restored = eng.network_restore(timeout_ms=0)

    assert restored.ok is True
    assert restored.state.wifi_enabled is True
    assert ("shell", ("svc wifi enable",)) in dev.calls
    assert not backup_path(tmp_path, dev.serial).exists()


class _StubbornNetworkDevice(FakeDevice):
    def shell(self, command: str) -> str:
        if command == "dumpsys connectivity":
            return (
                "Active default network: 55\n"
                "NetworkAgentInfo{network{55} nc{[ Transports: ETHERNET "
                "Capabilities: INTERNET&VALIDATED ]}}\n"
            )
        return super().shell(command)


def test_network_offline_does_not_claim_success_while_a_transport_remains(
    tmp_path: Path,
) -> None:
    dev = _StubbornNetworkDevice(serial="fake-ethernet")
    cfg = make_config(cache={"dir": str(tmp_path)})
    result = Engine(cfg, device=dev).network_offline(timeout_ms=0)

    assert result.ok is False
    assert result.verified is False
    assert result.state.active_transports == ["ethernet"]
    assert backup_path(tmp_path, dev.serial).is_file()


class _BootDevice(FakeDevice):
    def __init__(self, *, token: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._token = token

    def instance_token(self) -> str:
        return self._token


def test_restore_refuses_a_restore_point_from_an_older_boot(tmp_path: Path) -> None:
    cfg = make_config(cache={"dir": str(tmp_path)})
    first = _BootDevice(serial="recycled-serial", token="boot-one")
    Engine(cfg, device=first).network_offline(timeout_ms=0)

    replacement = _BootDevice(serial="recycled-serial", token="boot-two")
    with pytest.raises(UsageError, match="previous device boot"):
        Engine(cfg, device=replacement).network_restore(timeout_ms=0)

    # Starting a fresh offline session on the replacement safely supersedes stale state.
    result = Engine(cfg, device=replacement).network_offline(timeout_ms=0)
    assert result.saved_state is not None
    assert result.saved_state.wifi_enabled is True


def test_cli_network_offline_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(serial="fake-cli-network")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    status = runner.invoke(app, ["--serial", dev.serial, "network", "status"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout)["state"]["active_transports"] == ["wifi"]

    offline = runner.invoke(
        app,
        ["--serial", dev.serial, "network", "offline", "--verify", "--timeout", "0"],
    )
    assert offline.exit_code == 0, offline.output
    assert json.loads(offline.stdout)["verified"] is True

    restore = runner.invoke(
        app,
        ["--serial", dev.serial, "network", "restore", "--timeout", "0"],
    )
    assert restore.exit_code == 0, restore.output
    assert json.loads(restore.stdout)["state"]["offline"] is False


def test_daemon_network_offline_and_restore_roundtrip(tmp_path: Path) -> None:
    dev = FakeDevice(serial="fake-daemon-network")
    eng = Engine(make_config(cache={"dir": str(tmp_path)}), device=dev)

    offline = dispatch(
        eng,
        {"cmd": "network_offline", "args": {"verify": True, "timeout_ms": 0}},
    )
    restored = dispatch(eng, {"cmd": "network_restore", "args": {"timeout_ms": 0}})

    assert offline["ok"] is True
    assert offline["result"]["verified"] is True
    assert restored["ok"] is True
    assert restored["result"]["state"]["offline"] is False


def test_cli_network_verification_failure_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = _StubbornNetworkDevice(serial="fake-cli-ethernet")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    result = runner.invoke(
        app,
        ["--serial", dev.serial, "network", "offline", "--timeout", "0"],
    )
    assert result.exit_code != 0
    assert json.loads(result.stdout)["verified"] is False
    assert "network_verification_failed" in result.stderr
