"""Reversible network profiles across pure logic, engine, CLI, daemon, and MCP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.network_profiles import (
    EmulatorShape,
    parse_emulator_shape,
    profile_path,
    profile_restored_verified,
    root_qdisc_kind,
)
from android_ui_analyser.schema import NetworkState
from conftest import FakeDevice, make_config

runner = CliRunner()


def _engine(tmp_path: Path, *, preference: str = "wifi") -> tuple[Engine, FakeDevice]:
    device = FakeDevice(serial="fake-profile", network_preference=preference)
    return Engine(make_config(cache={"dir": str(tmp_path)}), device=device), device


def test_parse_emulator_shape() -> None:
    raw = """
Current network status:
  download speed:     473600 bits/s (57.8 KB/s)
  upload speed:       473600 bits/s (57.8 KB/s)
  minimum latency:  80 ms
  maximum latency:  400 ms
OK
"""
    assert parse_emulator_shape(raw) == EmulatorShape(
        upload_bps=473600,
        download_bps=473600,
        min_latency_ms=80,
        max_latency_ms=400,
    )


def test_root_qdisc_parser_ignores_ingress_and_child_queues() -> None:
    raw = """
qdisc clsact ffff: parent ffff:fff1
qdisc mq 0: root
qdisc pfifo_fast 0: parent :1
"""
    assert root_qdisc_kind(raw) == "mq"


def test_radio_restore_waits_for_the_saved_transport() -> None:
    saved = NetworkState(
        airplane_mode=False,
        wifi_enabled=True,
        mobile_data_enabled=True,
        active_network=True,
        active_transports=["wifi"],
        internet_validated=True,
    )
    transient = saved.model_copy(update={"active_transports": ["cellular"]})
    assert profile_restored_verified(transient, saved) is False
    assert profile_restored_verified(saved, saved) is True


@pytest.mark.parametrize(
    ("profile", "preference", "expected_transport"),
    [
        ("wifi-only", "wifi", "wifi"),
        ("cellular-only", "cellular", "cellular"),
    ],
)
def test_radio_profiles_are_verified_and_reversible(
    tmp_path: Path,
    profile: str,
    preference: str,
    expected_transport: str,
) -> None:
    engine, device = _engine(tmp_path, preference=preference)
    initial = engine.network_status().state

    applied = engine.network_profile_apply(profile, timeout_ms=0)
    assert applied.ok is True
    assert applied.verified is True
    assert applied.profile == profile
    assert applied.state.active_transports == [expected_transport]
    assert profile_path(tmp_path, device.serial).is_file()

    status = engine.network_profile_status()
    assert status.verified is True
    assert status.profile == profile

    with pytest.raises(UsageError, match="already active"):
        engine.network_profile_apply("wifi-only", timeout_ms=0)
    with pytest.raises(UsageError, match="profile .* is active"):
        engine.network_offline(timeout_ms=0)

    restored = engine.network_profile_restore(timeout_ms=0)
    assert restored.ok is True
    assert restored.state.airplane_mode == initial.airplane_mode
    assert restored.state.wifi_enabled == initial.wifi_enabled
    assert restored.state.mobile_data_enabled == initial.mobile_data_enabled
    assert not profile_path(tmp_path, device.serial).exists()


def test_profile_refuses_to_stack_on_verified_offline_mode(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path)
    assert engine.network_offline(timeout_ms=0).ok is True
    with pytest.raises(UsageError, match="offline mode is active"):
        engine.network_profile_apply("wifi-only", timeout_ms=0)
    assert engine.network_restore(timeout_ms=0).ok is True


def test_slow_profile_saves_applies_and_restores_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import network_profiles

    engine, device = _engine(tmp_path)
    original = EmulatorShape(
        upload_bps=0,
        download_bps=0,
        min_latency_ms=0,
        max_latency_ms=0,
    )
    slow = EmulatorShape(
        upload_bps=473600,
        download_bps=473600,
        min_latency_ms=80,
        max_latency_ms=400,
    )
    monkeypatch.setattr(network_profiles, "read_emulator_shape", lambda serial: original)
    monkeypatch.setattr(
        network_profiles,
        "set_emulator_shape",
        lambda serial, speed, delay: slow,
    )
    monkeypatch.setattr(
        network_profiles,
        "restore_emulator_shape",
        lambda serial, shape: original,
    )

    applied = engine.network_profile_apply("slow")
    assert applied.ok is True
    assert applied.shaping is not None
    assert applied.shaping.min_latency_ms == 80

    # Status is independently observed, not inferred from the saved intent.
    monkeypatch.setattr(network_profiles, "read_emulator_shape", lambda serial: slow)
    assert engine.network_profile_status().verified is True

    restored = engine.network_profile_restore()
    assert restored.ok is True
    assert restored.shaping is not None
    assert restored.shaping.upload_bps == 0
    assert not profile_path(tmp_path, device.serial).exists()


def test_lossy_profile_uses_readback_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser import network_profiles
    from android_ui_analyser.schema import NetworkShaping

    engine, device = _engine(tmp_path)
    monkeypatch.setattr(network_profiles, "prepare_loss", lambda serial: ("wlan0", "mq", False))
    monkeypatch.setattr(
        network_profiles,
        "set_loss",
        lambda serial, interface, loss_percent: NetworkShaping(
            mechanism="tc-netem",
            interface=interface,
            loss_percent=loss_percent,
            qdisc="netem",
            root_enabled=True,
        ),
    )
    monkeypatch.setattr(network_profiles, "root_enabled", lambda serial: True)
    monkeypatch.setattr(
        network_profiles,
        "qdisc_evidence",
        lambda serial, interface, root: NetworkShaping(
            mechanism="tc-netem",
            interface=interface,
            loss_percent=12.5,
            qdisc="netem",
            root_enabled=root,
        ),
    )
    monkeypatch.setattr(
        network_profiles,
        "remove_loss",
        lambda serial, backup: (
            NetworkShaping(
                mechanism="tc-netem",
                interface="wlan0",
                qdisc="mq",
                root_enabled=False,
            ),
            True,
        ),
    )

    applied = engine.network_profile_apply("lossy", loss_percent=12.5)
    assert applied.ok is True
    assert applied.shaping is not None
    assert applied.shaping.loss_percent == 12.5
    assert engine.network_profile_status().verified is True
    assert engine.network_profile_restore().ok is True
    assert not profile_path(tmp_path, device.serial).exists()


def test_cli_and_daemon_profile_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice(serial="fake-profile-cli", network_preference="wifi")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: device)

    listed = runner.invoke(app, ["network", "profile", "list"])
    assert listed.exit_code == 0
    assert "wifi-only" in json.loads(listed.stdout)["names"]

    applied = runner.invoke(
        app,
        ["network", "profile", "apply", "wifi-only", "--timeout", "0"],
    )
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.stdout)["verified"] is True

    cfg = make_config(cache={"dir": str(tmp_path)})
    daemon_engine = Engine(cfg, device=FakeDevice(serial="fake-profile-daemon"))
    response = dispatch(
        daemon_engine,
        {
            "cmd": "network_profile_apply",
            "args": {"profile": "wifi-only", "timeout_ms": 0},
        },
    )
    assert response["ok"] is True
    assert response["result"]["verified"] is True
