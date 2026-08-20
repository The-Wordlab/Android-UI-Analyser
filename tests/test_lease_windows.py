from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import leases
from android_ui_analyser.errors import DeviceLeasedError


class _FakeMsvcrt:
    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self.busy_once = True

    def locking(self, fd: int, mode: int, length: int) -> None:
        self.calls.append((fd, mode, length))
        if mode == self.LK_NBLCK and self.busy_once:
            self.busy_once = False
            raise OSError(errno.EACCES, "locked by another process")


def test_windows_range_lock_waits_and_releases_through_msvcrt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    msvcrt = _FakeMsvcrt()
    sleeps: list[float] = []
    monkeypatch.setattr(leases, "_windows_lock_module", lambda: msvcrt)
    monkeypatch.setattr(leases.time, "sleep", sleeps.append)
    path = tmp_path / "host.lock"

    with path.open("a+", encoding="utf-8") as handle:
        leases._acquire_windows_file_lock(handle)
        leases._release_windows_file_lock(handle)

    assert path.stat().st_size == 1
    assert [mode for _, mode, _ in msvcrt.calls] == [msvcrt.LK_NBLCK] * 2 + [msvcrt.LK_UNLCK]
    assert sleeps == [leases._WINDOWS_LOCK_POLL_S]


def test_windows_shared_device_use_is_conservatively_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquired: list[tuple[str, bool]] = []
    released: list[tuple[str, str]] = []
    monkeypatch.setattr(leases, "_host_lock_kind", lambda: "windows")

    def acquire(handle: Any, *, exclusive: bool) -> str:
        acquired.append((Path(handle.name).name, exclusive))
        return "windows"

    def release(handle: Any, backend: str) -> None:
        released.append((Path(handle.name).name, backend))

    monkeypatch.setattr(leases, "_acquire_file_lock", acquire)
    monkeypatch.setattr(leases, "_release_file_lock", release)

    with leases.device_use(tmp_path, "emulator-5554"):
        pass

    assert len(acquired) == 2
    assert all(exclusive for _, exclusive in acquired)
    assert sorted(name for name, _ in released) == sorted(name for name, _ in acquired)


def test_windows_process_identity_uses_win32_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(leases, "_is_windows_host", lambda: True)
    monkeypatch.setattr(
        leases, "_windows_process_snapshot", lambda pid: (f"agent-{pid}.exe", pid - 1)
    )
    monkeypatch.setattr(leases, "_windows_proc_started", lambda pid: f"created-{pid}")
    monkeypatch.setattr(leases, "_windows_process_exists", lambda pid: pid == 42)

    assert leases._proc_name(42) == "agent-42.exe"
    assert leases._proc_ppid(42) == 41
    assert leases._proc_started(42) == "created-42"
    assert leases._process_exists(42) is True
    assert leases._process_exists(43) is False


@pytest.mark.parametrize("name", ["python.exe", "uv.exe", "cmd.exe", "PowerShell.EXE"])
def test_windows_command_wrappers_are_transient(name: str) -> None:
    assert leases._is_transient(name)


def test_windows_liveness_check_never_calls_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(leases, "_is_windows_host", lambda: True)
    monkeypatch.setattr(leases, "_windows_process_exists", lambda pid: True)
    monkeypatch.setattr(leases, "_windows_proc_started", lambda pid: "same-start")
    monkeypatch.setattr(
        leases.os,
        "kill",
        lambda pid, signal: (_ for _ in ()).throw(AssertionError("os.kill is unsafe on Windows")),
    )
    entry = {
        "owner_pid": 42,
        "owner_started": "same-start",
        "last_activity": 0,
        "ttl_s": 1,
    }

    assert leases._expired(entry, now=99_999) is False


@pytest.mark.parametrize("explicit", ["emulator-5554", None])
def test_busy_process_bound_lease_hint_never_promises_ttl_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: str | None
) -> None:
    monkeypatch.setattr(leases, "_process_exists", lambda pid: True)
    monkeypatch.setattr(leases, "_proc_started", lambda pid: "source-start")
    source = leases.LeaseOwner("source", pid=42, started="source-start")
    assert leases.acquire(tmp_path, "emulator-5554", owner=source, ttl_s=1)

    with pytest.raises(DeviceLeasedError) as caught:
        leases.choose_device(
            tmp_path,
            owner="receiver",
            explicit=explicit,
            candidates=[("emulator-5554", {})],
        )

    hint = caught.value.hint or ""
    assert "do not expire while their owner is alive" in hint
    assert "free as soon as that process exits" in hint
    assert "expires after 1s idle" not in hint
