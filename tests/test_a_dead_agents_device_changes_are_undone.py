"""A device change outlives the agent that made it; the undo must outlive it too.

The failure this covers, observed on this host: two orphan ``mitmdump`` processes alive, no
ownership record for any of them, three emulators leased by processes that no longer existed.
An agent is SIGKILLed, its lease frees instantly (the pid check runs before the TTL), and the
next agent inherits a device still pointed at a dead proxy port — every app reports "Offline"
for a reason that has nothing to do with the app under test.

Lease expiry is lazy: nothing runs at the moment a lease lapses. So the undo has to be written
down where a stranger can find it, and replayed by someone else.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import device_ledger, teardown
from android_ui_analyser import teardown_watchdog as teardown_watchdog_mod
from android_ui_analyser.errors import ConfigError
from android_ui_analyser.platforms.options_transport import platform_options_fingerprint

_REAL_ENSURE_WATCHDOG = teardown.ensure_watchdog


class _Device:
    def __init__(self, serial: str = "emulator-5554", token: str | None = "boot-1") -> None:
        self.serial = serial
        self._token = token
        self.calls: list[tuple[str, Any]] = []

    def instance_token(self) -> str | None:
        return self._token

    def set_http_proxy(self, host_port: str | None) -> None:
        self.calls.append(("set_http_proxy", host_port))

    def remove_reverse_port(self, port: int) -> None:
        self.calls.append(("remove_reverse_port", port))


class _Platform:
    """A platform with no Android in it, proving the reaper needs none."""

    # The implementation is deliberately fake; the records under test are legacy Android
    # records, so the strategy identity still has to agree with their platform scope.
    name = "android"

    def __init__(self, device: _Device | None) -> None:
        self.device = device

    def connect(self, serial: str | None = None) -> _Device:
        if self.device is None:
            raise RuntimeError("target unreachable")
        return self.device

    def validate_runtime(self, runtime: _Device) -> _Device:
        return runtime

    def runtime_capability(self, name: str, runtime: _Device) -> _Device:
        return runtime

    def capability(self, name: str) -> Any:
        raise RuntimeError(f"no {name} capability in this test")


def _record_proxy(serial: str, *, owner_pid: int, cache_dir: Path, port: int = 49097) -> None:
    device_ledger.record(
        serial,
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        detail=f"device http_proxy set to 127.0.0.1:{port}",
        owner=f"claude-{owner_pid}-abc",
        owner_pid=owner_pid,
        owner_started=None,
        instance_token="boot-1",
        cache_dir=str(cache_dir),
    )
    device_ledger.record(
        serial,
        key=f"reverse_port:{port}",
        kind="reverse_port",
        op="remove_reverse_port",
        args={"port": port},
        owner=f"claude-{owner_pid}-abc",
        owner_pid=owner_pid,
        owner_started=None,
        instance_token="boot-1",
        cache_dir=str(cache_dir),
    )


def _dead_pid() -> int:
    """A pid that is certainly gone: fork a child and reap it."""
    import os

    pid = os.fork()
    if pid == 0:  # pragma: no cover — child exits immediately
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_a_dead_owners_proxy_is_taken_off_the_device(tmp_path: Path) -> None:
    device = _Device()
    _record_proxy("emulator-5554", owner_pid=_dead_pid(), cache_dir=tmp_path)

    report = teardown.reap(
        "emulator-5554", platform=_Platform(device), cache_dir=tmp_path
    )

    assert "is gone" in report["reason"], report
    assert ("set_http_proxy", None) in device.calls, (
        "the device was left pointing at a proxy port nobody serves"
    )
    assert ("remove_reverse_port", 49097) in device.calls
    assert device_ledger.read_ledger("emulator-5554") == [], "a replayed undo must not repeat"


def test_a_live_owners_changes_are_reported_not_undone(tmp_path: Path) -> None:
    """Pulling a proxy out from under a running test is worse than leaving it set."""
    import os

    device = _Device()
    _record_proxy("emulator-5554", owner_pid=os.getpid(), cache_dir=tmp_path)

    report = teardown.reap(
        "emulator-5554", platform=_Platform(device), cache_dir=tmp_path
    )

    assert report["skipped"] == "a live holder still owns these changes"
    assert device.calls == []
    assert len(device_ledger.read_ledger("emulator-5554")) == 2, "the record must survive"


def test_a_live_lease_protects_a_change_whose_owner_pid_is_unknown(tmp_path: Path) -> None:
    """A daemon-transported owner may carry no pid; the lease is then the only signal."""
    from android_ui_analyser import leases

    device_ledger.record(
        "emulator-5554",
        key="wall_clock",
        kind="wall_clock",
        op="set_clock",
        args={"timestamp_ms": 1},
        owner="orchestrator",
        cache_dir=str(tmp_path),
    )
    assert leases.acquire(tmp_path, "emulator-5554", owner="orchestrator")

    device = _Device()
    report = teardown.reap("emulator-5554", platform=_Platform(device), cache_dir=tmp_path)

    assert report["skipped"] == "a live holder still owns these changes"
    assert device.calls == []


def test_an_unknown_owner_is_undone_once_the_grace_period_lapses(tmp_path: Path) -> None:
    device = _Device()
    device_ledger.record(
        "emulator-5554",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        owner="who-knows",
        cache_dir=str(tmp_path),
    )

    fresh = teardown.reap(
        "emulator-5554", platform=_Platform(device), cache_dir=tmp_path, grace_s=600
    )
    assert fresh["skipped"], "a change made seconds ago must be left alone"

    lapsed = teardown.reap(
        "emulator-5554", platform=_Platform(device), cache_dir=tmp_path, grace_s=0
    )
    assert ("set_http_proxy", None) in device.calls, lapsed


@pytest.mark.parametrize("token", ["boot-2", None])
def test_a_changed_or_unreadable_boot_retains_undo_without_touching_the_target(
    tmp_path: Path, token: str | None,
) -> None:
    """Reboot/offline is not evidence that retained settings and files disappeared."""
    rebooted = _Device(token=token)
    _record_proxy("emulator-5554", owner_pid=_dead_pid(), cache_dir=tmp_path)

    report = teardown.reap(
        "emulator-5554", platform=_Platform(rebooted), cache_dir=tmp_path
    )

    assert rebooted.calls == [], report
    assert not report["undone"]
    assert len(report["failed"]) == 2
    assert len(device_ledger.read_ledger("emulator-5554")) == 2


def test_a_corrupt_ledger_cannot_be_overwritten_by_a_new_mutation(tmp_path: Path) -> None:
    path = device_ledger.ledger_path("corrupt-target")
    path.write_text('{"entries": [', encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read pending"):
        device_ledger.record("corrupt-target", key="new", kind="new", op="set_http_proxy")
    assert path.read_text(encoding="utf-8") == '{"entries": ['


@pytest.mark.parametrize("operation", [
    "restore_adbd_root", "disable_device_agent", "remove_device_agent",
    "stop_device_agent_touch_capture",
])
def test_target_services_require_the_same_instance_proof_as_runtime_undos(
    tmp_path: Path, operation: str,
) -> None:
    device_ledger.record(
        "service-target", key=operation, kind=operation, op=operation,
        instance_token="original-boot", cache_dir=tmp_path,
    )
    calls: list[str] = []
    result = device_ledger.replay("service-target", context=device_ledger.UndoContext(
        serial="service-target", instance_token="replacement-boot",
        capability=lambda name: calls.append(name),
    ))
    assert calls == []
    assert result["remaining"] == 1
    assert result["failed"]


@pytest.mark.parametrize("operation", ["restore_developer_settings", "restore_app_prefs"])
def test_a_missing_backup_is_not_a_successful_undo(tmp_path: Path, operation: str) -> None:
    device_ledger.record(
        "backup-target", key=operation, kind=operation, op=operation,
        args={"backup_path": str(tmp_path / "missing.json")},
    )
    result = device_ledger.replay("backup-target", context=device_ledger.UndoContext(
        serial="backup-target", capability=lambda name: object(),
    ))
    assert result["remaining"] == 1
    assert "missing" in result["failed"][0]["error"]


def test_teardown_run_continues_past_an_unavailable_recorded_plugin(tmp_path: Path) -> None:
    from android_ui_analyser.engine import Engine
    from conftest import make_config

    device_ledger.record("gone", platform="aaa-uninstalled", key="proxy", kind="proxy", op="set_http_proxy")
    _record_proxy("emulator-5554", owner_pid=_dead_pid(), cache_dir=tmp_path)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}))
    original = _Platform(_Device())

    def create(name: str) -> Any:
        if name == "aaa-uninstalled":
            raise ConfigError("plugin unavailable")
        assert name == "android"
        return original

    engine._platform_factory.create = create  # type: ignore[method-assign]
    result = engine.teardown_run(force=True)
    assert result["ok"] is False
    assert len(result["reports"]) == 2
    assert result["reports"][0]["skipped"].startswith("no adapter available")
    assert not device_ledger.read_ledger("emulator-5554")
    assert device_ledger.read_ledger("gone", platform="aaa-uninstalled")


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_corrupt_option_key_retains_nonempty_configuration_undos(
    tmp_path: Path, corrupt: bool,
) -> None:
    from android_ui_analyser.config import Config

    cfg = Config.model_validate({"platforms": {"android": {"endpoint": "https://grid.invalid"}}})
    options = cfg.platform_options()
    fingerprint = _options_fingerprint(options)
    device_ledger.record("configured-target", key="proxy", kind="proxy", op="set_http_proxy",
                         platform_options_fingerprint=fingerprint)
    key = device_ledger.ledger_dir() / ".platform-options-hmac-key"
    if corrupt:
        key.write_bytes(b"short")
    else:
        key.unlink()
    platform = _Platform(_Device())
    platform.config = cfg
    result = teardown.reap("configured-target", platform=platform, force=True)
    assert result["code"] in {"platform_options_recovery_mismatch", "platform_options_identity_unavailable"}
    assert result["identity_key"] == str(key)
    assert result["remaining"] == 1
    assert not platform.device.calls


def test_host_residue_is_cleaned_up_even_when_the_target_is_unreachable(tmp_path: Path) -> None:
    """An unplugged phone forgot its settings; the orphan host process did not."""
    import os
    import subprocess
    import time

    victim = subprocess.Popen(  # noqa: S603
        ["sleep", "120"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        device_ledger.record(
            "emulator-5554",
            key="host_proxy_process",
            kind="host_proxy_process",
            op="kill_host_process",
            args={"pid": victim.pid, "match": "sleep"},
            owner_pid=_dead_pid(),
            cache_dir=str(tmp_path),
        )
        report = teardown.reap(
            "emulator-5554", platform=_Platform(None), cache_dir=tmp_path
        )
        assert report["undone"], report
        deadline = time.time() + 5
        while time.time() < deadline and victim.poll() is None:
            time.sleep(0.05)
        assert victim.poll() is not None, "the orphan host process is still holding its port"
    finally:
        if victim.poll() is None:  # pragma: no cover — only on failure
            victim.kill()
        with __import__("contextlib").suppress(Exception):
            os.waitpid(victim.pid, 0)


def test_a_recycled_pid_is_never_signalled(tmp_path: Path) -> None:
    """A bare pid from a file is not evidence: pids are recycled and the next holder is innocent."""
    import os

    device_ledger.record(
        "emulator-5554",
        key="host_proxy_process",
        kind="host_proxy_process",
        op="kill_host_process",
        # This process is alive, but it is emphatically not a mitmdump.
        args={"pid": os.getpid(), "match": "mitmdump"},
        owner_pid=_dead_pid(),
        cache_dir=str(tmp_path),
    )

    report = teardown.reap("emulator-5554", platform=_Platform(None), cache_dir=tmp_path)

    assert report["undone"], report
    assert "left alone" in report["undone"][0]["result"], report


def test_the_sweep_skips_the_callers_own_device(tmp_path: Path) -> None:
    _record_proxy("emulator-5554", owner_pid=_dead_pid(), cache_dir=tmp_path)
    _record_proxy("emulator-5556", owner_pid=_dead_pid(), cache_dir=tmp_path)

    device = _Device()
    reports = teardown.sweep(
        platform=_Platform(device), cache_dir=tmp_path, skip="emulator-5554"
    )

    assert [r["serial"] for r in reports] == ["emulator-5556"]
    assert device_ledger.read_ledger("emulator-5554"), "the caller's own changes are live"


def test_a_long_lived_agent_that_handed_the_device_back_is_still_cleaned_up(
    tmp_path: Path,
) -> None:
    """The owner process outliving the work is the normal case, not the exception.

    An orchestrator, a warm daemon, or a `claude` process lives for hours across many devices.
    Waiting for it to exit would keep the first emulator proxied for the rest of that lifetime,
    so once its lease is gone the changes are fair game — the lease is the ownership signal, the
    process is only a fast path for "provably dead".
    """
    import os

    device = _Device()
    _record_proxy("emulator-5554", owner_pid=os.getpid(), cache_dir=tmp_path)

    report = teardown.reap(
        "emulator-5554", platform=_Platform(device), cache_dir=tmp_path, grace_s=0
    )

    assert "its lease is gone" in report["reason"], report
    assert ("set_http_proxy", None) in device.calls


def test_with_leasing_off_a_live_owner_is_never_reaped(tmp_path: Path) -> None:
    """No lease means no ownership signal, so the process is the only thing left to trust."""
    import os

    device = _Device()
    device_ledger.record(
        "emulator-5554",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        owner="solo-run",
        owner_pid=os.getpid(),
        cache_dir=str(tmp_path),
        leased=False,
    )

    report = teardown.reap(
        "emulator-5554", platform=_Platform(device), cache_dir=tmp_path, grace_s=0
    )

    assert report["skipped"], report
    assert device.calls == [], "a run with leasing off was still in progress"


def test_with_leasing_off_a_dead_owner_is_reaped_at_once(tmp_path: Path) -> None:
    device = _Device()
    device_ledger.record(
        "emulator-5554",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        owner="solo-run",
        owner_pid=_dead_pid(),
        cache_dir=str(tmp_path),
        leased=False,
    )

    report = teardown.reap("emulator-5554", platform=_Platform(device), cache_dir=tmp_path)

    assert "is gone" in report["reason"], report
    assert ("set_http_proxy", None) in device.calls


def _options_fingerprint(options: dict[str, str]) -> str:
    return platform_options_fingerprint(options, key_dir=device_ledger.ledger_dir())


def test_pending_changes_refuse_a_different_nonempty_adapter_identity(tmp_path: Path) -> None:
    first = _options_fingerprint({"endpoint": "https://first.invalid", "token": "private-one"})
    second = _options_fingerprint(
        {"endpoint": "https://second.invalid", "token": "private-two"}
    )
    device_ledger.record(
        "configured-target",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        cache_dir=tmp_path,
        platform_options_fingerprint=first,
    )

    with pytest.raises(ConfigError) as raised:
        device_ledger.record(
            "configured-target",
            key="reverse_port:1234",
            kind="reverse_port",
            op="remove_reverse_port",
            args={"port": 1234},
            cache_dir=tmp_path,
            platform_options_fingerprint=second,
        )

    assert raised.value.code == "platform_options_recovery_mismatch"
    assert [entry.key for entry in device_ledger.read_ledger("configured-target")] == [
        "http_proxy"
    ]
    persisted = device_ledger.ledger_path("configured-target").read_text(encoding="utf-8")
    assert first in persisted
    assert "first.invalid" not in persisted
    assert "private-one" not in persisted


def test_pending_configured_changes_refuse_a_missing_adapter_identity(tmp_path: Path) -> None:
    recorded = _options_fingerprint({"endpoint": "https://original.invalid"})
    device_ledger.record(
        "configured-target",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        cache_dir=tmp_path,
        platform_options_fingerprint=recorded,
    )

    with pytest.raises(ConfigError, match="another adapter configuration"):
        device_ledger.record(
            "configured-target",
            key="reverse_port:1234",
            kind="reverse_port",
            op="remove_reverse_port",
            args={"port": 1234},
            cache_dir=tmp_path,
        )

    assert not device_ledger.options_match(
        device_ledger.read_ledger("configured-target"),
        None,
    )


def test_teardown_refuses_active_adapter_options_that_cannot_replay_the_change(
    tmp_path: Path,
) -> None:
    recorded = _options_fingerprint({"endpoint": "https://original.invalid"})
    active = _options_fingerprint({"endpoint": "https://other.invalid"})
    device_ledger.record(
        "configured-target",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        owner_pid=_dead_pid(),
        cache_dir=tmp_path,
        platform_options_fingerprint=recorded,
    )
    device = _Device(serial="configured-target")

    report = teardown.reap(
        "configured-target",
        platform=_Platform(device),
        cache_dir=tmp_path,
        force=True,
        options_fingerprint=active,
    )

    assert report["code"] == "platform_options_recovery_mismatch"
    assert report["remaining"] == 1
    assert device.calls == []
    assert device_ledger.read_ledger("configured-target")


def test_watchdog_refuses_mismatched_transported_options_and_leaves_the_undo(
    tmp_path: Path,
) -> None:
    original_options = {"endpoint": "https://original.invalid"}
    device_ledger.record(
        "configured-target",
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        cache_dir=tmp_path,
        platform="example-os",
        platform_options_fingerprint=_options_fingerprint(original_options),
    )

    code = teardown_watchdog_mod.run_watchdog(
        serial="configured-target",
        cache_dir=str(tmp_path),
        platform_name="example-os",
        platform_options={"endpoint": "https://other.invalid"},
        max_lifetime_s=0,
    )

    assert code == 1
    assert device_ledger.read_ledger("configured-target", platform="example-os")


def test_ensure_watchdog_replaces_legacy_and_mismatched_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = teardown.watchdog_pid_path("configured-target")
    path.write_text("111\n", encoding="utf-8")
    retired: list[tuple[int, str]] = []
    spawned: list[list[str]] = []
    next_pid = iter((222, 333))

    monkeypatch.setattr(teardown, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(teardown, "_is_target_watchdog", lambda _pid, _ref: True)
    monkeypatch.setattr(
        teardown,
        "_retire_watchdog",
        lambda _ref, pid, fingerprint: retired.append((pid, fingerprint)) or True,
    )

    def popen(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        spawned.append(command)
        return SimpleNamespace(pid=next(next_pid))

    monkeypatch.setattr(teardown.subprocess, "Popen", popen)
    first_options = {"endpoint": "https://first.invalid", "token": "secret-one"}
    second_options = {"endpoint": "https://second.invalid", "token": "secret-two"}

    assert _REAL_ENSURE_WATCHDOG(
        "configured-target",
        cache_dir=tmp_path,
        platform_name="android",
        platform_options=first_options,
    ) == 222
    assert retired == [(111, "")]
    assert _REAL_ENSURE_WATCHDOG(
        "configured-target",
        cache_dir=tmp_path,
        platform_name="android",
        platform_options=first_options,
    ) == 222
    assert len(spawned) == 1

    assert _REAL_ENSURE_WATCHDOG(
        "configured-target",
        cache_dir=tmp_path,
        platform_name="android",
        platform_options=second_options,
    ) == 333
    assert retired[-1][0] == 222
    metadata = path.read_text(encoding="utf-8")
    assert _options_fingerprint(second_options) in metadata
    assert "second.invalid" not in metadata
    assert "secret-two" not in metadata


def test_stale_watchdog_metadata_never_signals_an_unrelated_recycled_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = teardown.target_ref("configured-target")
    teardown._write_watchdog_registration(
        ref, pid=444, options_fingerprint="legacy-fingerprint"
    )
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(teardown, "_is_target_watchdog", lambda _pid, _ref: False)
    monkeypatch.setattr(teardown.os, "kill", lambda pid, sig: signalled.append((pid, sig)))

    assert teardown._retire_watchdog(ref, 444, "legacy-fingerprint") is True
    assert signalled == []
    assert not teardown.watchdog_pid_path(ref).exists()
