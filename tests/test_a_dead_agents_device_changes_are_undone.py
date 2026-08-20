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
from typing import Any

from android_ui_analyser import device_ledger, teardown


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

    name = "fake"

    def __init__(self, device: _Device | None) -> None:
        self.device = device

    def connect(self, serial: str | None = None) -> _Device:
        if self.device is None:
            raise RuntimeError("target unreachable")
        return self.device

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


def test_a_reboot_since_the_change_is_not_replayed_onto_the_fresh_boot(tmp_path: Path) -> None:
    """The device already forgot the setting; re-applying an undo would be a change of its own."""
    rebooted = _Device(token="boot-2")
    _record_proxy("emulator-5554", owner_pid=_dead_pid(), cache_dir=tmp_path)

    report = teardown.reap(
        "emulator-5554", platform=_Platform(rebooted), cache_dir=tmp_path
    )

    assert rebooted.calls == [], report
    assert all("rebooted" in item["result"] for item in report["undone"]), report
    assert device_ledger.read_ledger("emulator-5554") == []


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
