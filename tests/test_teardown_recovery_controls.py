"""Operator recovery never hides another target or silently widens undo authority."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import device_ledger, leases
from android_ui_analyser.cli import GlobalOpts, app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ConfigError, DeviceError, UsageError
from android_ui_analyser.mcp_server import _dispatch
from android_ui_analyser.platforms import TargetRef
from conftest import FakeDevice, make_config
from test_a_dead_agents_device_changes_are_undone import _Device, _Platform


def test_invalid_ledgers_are_reported_without_blocking_other_targets(tmp_path: Path) -> None:
    malformed = device_ledger.ledger_path("malformed")
    malformed.write_text('{"entries": [', encoding="utf-8")
    invalid = device_ledger.ledger_path("invalid")
    invalid.write_text(json.dumps({"serial": "invalid", "entries": [{"key": "no-op"}]}))
    device_ledger.record("valid", key="proxy", kind="proxy", op="set_http_proxy")
    engine = Engine(make_config(cache={"dir": str(tmp_path)}))
    device = _Device(serial="valid")
    engine._platform = _Platform(device)  # type: ignore[assignment]

    status = engine.teardown_status()
    assert status["ok"] is False
    assert len(status["devices"]) == 3
    assert {row["ledger_path"] for row in status["devices"] if row.get("code")} == {
        str(malformed), str(invalid),
    }
    result = engine.teardown_run(force=True)
    assert result["ok"] is False
    assert len(result["reports"]) == 3
    assert device.calls == [("set_http_proxy", None)]
    assert not device_ledger.read_ledger("valid")
    assert malformed.read_text() == '{"entries": ['
    assert invalid.exists()


def _unavailable_plugin_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Engine:
    cfg = make_config(
        device={"platform": "uninstalled-fixture"},
        cache={"dir": str(tmp_path / "cache")},
        lease={"registry_dir": str(tmp_path / "leases")},
    )
    engine = Engine(cfg)
    monkeypatch.setattr(engine._platform_factory, "create", lambda *_args: pytest.fail("adapter loaded"))
    return engine


def _stale(target: TargetRef, *, key: str = "screen_recording") -> None:
    device_ledger.record(target, key=key, kind=key, op="discard_recording",
                         instance_token="dead-boot", platform_options_fingerprint="lost-identity")


@pytest.mark.parametrize("surface", ["engine", "cli", "mcp"])
def test_discard_archives_only_confirmed_keys_without_loading_any_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    engine = _unavailable_plugin_engine(tmp_path, monkeypatch)
    ref = TargetRef("uninstalled-fixture", "same-id")
    _stale(ref)
    _stale(ref, key="another-change")
    reason = "disposable boot is gone"
    if surface == "engine":
        result = engine.teardown_discard(serial=ref.target_id, keys=["screen_recording"],
                                         reason=reason, confirmed=True)
    elif surface == "cli":
        monkeypatch.setattr(GlobalOpts, "engine", lambda _self: engine)
        output = CliRunner().invoke(app, ["--format", "compact", "teardown", "discard",
            "--serial-target", ref.target_id, "--key", "screen_recording", "--reason", reason,
            "--confirmed"])
        assert output.exit_code == 0, output.output
        result = json.loads(output.stdout)
    else:
        result = _dispatch(engine, "teardown_discard", {"target_id": ref.target_id,
            "keys": ["screen_recording"], "reason": reason, "confirmed": True})
    assert result["device_touched"] is False
    assert result["restored"] is False
    assert result["discarded"] == ["screen_recording"]
    assert [entry.key for entry in device_ledger.read_ledger(ref)] == ["another-change"]
    archive = Path(result["archive_path"])
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    audit = json.loads(archive.read_text())
    assert audit["reason"] == reason
    assert audit["entries"][0]["instance_token"] == "dead-boot"
    assert audit["discarded_keys"] == ["screen_recording"]


@pytest.mark.parametrize("keys,reason,confirmed", [
    (["screen_recording"], "gone", False), ([], "gone", True),
    (["screen_recording"], "", True), (["missing-key"], "gone", True),
])
def test_discard_requires_complete_explicit_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    keys: list[str], reason: str, confirmed: bool,
) -> None:
    engine = _unavailable_plugin_engine(tmp_path, monkeypatch)
    ref = TargetRef("uninstalled-fixture", "same-id")
    _stale(ref)
    with pytest.raises(UsageError):
        engine.teardown_discard(serial=ref.target_id, keys=keys, reason=reason, confirmed=confirmed)
    assert len(device_ledger.read_ledger(ref)) == 1
    assert not (device_ledger.ledger_dir() / "discarded").exists()


def test_a_live_lease_protects_its_undo_from_discard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _unavailable_plugin_engine(tmp_path, monkeypatch)
    ref = TargetRef("uninstalled-fixture", "same-id")
    _stale(ref)
    assert leases.acquire(engine.config.lease.registry_dir, ref, owner="working-agent")
    with pytest.raises(UsageError, match="leased target"):
        engine.teardown_discard(serial=ref.target_id, keys=["screen_recording"],
                                reason="gone", confirmed=True)
    assert device_ledger.read_ledger(ref)


def test_discard_clears_a_stale_recording_gate_without_replaying_it(tmp_path: Path) -> None:
    device = FakeDevice(serial="recording-target")
    engine = Engine(make_config(cache={"dir": str(tmp_path)}, lease={"enabled": False}), device=device)
    device_ledger.record(device.serial, key="screen_recording", kind="screen_recording",
                         op="discard_recording", instance_token="old-boot")
    with pytest.raises(DeviceError) as caught:
        engine.record_start("/sdcard/new.mp4")
    assert caught.value.code == "recording_cleanup_pending"
    assert "teardown discard" in caught.value.hint
    engine.teardown_discard(serial=device.serial, keys=["screen_recording"],
                            reason="disposable old boot ended", confirmed=True)
    assert engine.record_start("/sdcard/new.mp4").ok


def test_rotating_a_referenced_credential_blocks_mixing_new_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_ui_analyser.platforms.options_transport import platform_options_fingerprint

    options = {"token_env": "FIXTURE_RECOVERY_TOKEN"}
    monkeypatch.setenv("FIXTURE_RECOVERY_TOKEN", "original-credential")
    original = platform_options_fingerprint(options, key_dir=device_ledger.ledger_dir())
    device_ledger.record("rotation-target", key="old", kind="proxy", op="set_http_proxy",
                         platform_options_fingerprint=original)
    monkeypatch.setenv("FIXTURE_RECOVERY_TOKEN", "rotated-credential")
    rotated = platform_options_fingerprint(options, key_dir=device_ledger.ledger_dir())
    with pytest.raises(ConfigError) as caught:
        device_ledger.record("rotation-target", key="new", kind="proxy", op="set_http_proxy",
                             platform_options_fingerprint=rotated)
    assert caught.value.code == "platform_options_recovery_mismatch"
    assert [entry.key for entry in device_ledger.read_ledger("rotation-target")] == ["old"]
