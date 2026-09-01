"""`record stop` must let the *device* finish its MP4 before it touches the local adb client.

Measured on emulator-5556 on 2026-09-01: every `aua record stop` returned "screenrecord
produced an incomplete MP4 (the moov box is missing)", for a 4-second clip and for a
two-minute one, while the identical sequence by hand — `adb shell screenrecord`, then
`adb shell pkill -l 2 screenrecord`, then wait — produced a playable 46 KB file every time.

The difference was one line. `stop_recording` sent the on-device SIGINT and then, in a
`finally`, immediately sent SIGINT to its own local `adb shell` client and waited on it.
Killing that client tears down the device-side shell session, and it takes `screenrecord`
with it *while the muxer is still writing the moov box* — so the remote file froze at 3232
bytes (ftyp plus the `free` placeholder the moov box was going to fill) and nothing was
recoverable. Every scenario in the QA suite lists a recording first in its evidence, and
there is no second route to one, so this cost the whole sweep its video.

The device process is the one that has to finish. The local client then exits on its own —
measured return code 0, with no signal sent at all.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.errors import DeviceError

_REMOTE = "/sdcard/aua_recording.mp4"
_PS_IDLE = "ARGS\ninit second_stage\n[kthreadd]\n"
_PS_RECORDING = f"ARGS\ninit second_stage\nscreenrecord --time-limit 180 {_REMOTE}\n"


def _box(kind: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


_FINALIZED = _box(b"ftyp", b"isom") + _box(b"mdat", b"frames") + _box(b"moov", b"index")
# What the device actually held after the client was killed mid-finalize: a header and the
# placeholder the moov box never got to fill.
_TRUNCATED = _box(b"ftyp", b"isom") + _box(b"free", b"\x00" * 32)


class _FakeProc:
    """The local `adb -s <serial> shell screenrecord …` client."""

    def __init__(self) -> None:
        self.signals: list[int] = []
        self.killed = False
        self._alive = True

    def poll(self) -> int | None:
        return None if self._alive else 0

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0


class _FakeU2:
    """A device whose recording needs a few polls to finish writing its moov box."""

    def __init__(self, proc: _FakeProc, *, polls_to_finalize: int = 3) -> None:
        self._proc = proc
        self._polls_to_finalize = polls_to_finalize
        self._interrupted = False
        self._polls_since_interrupt = 0
        self.finalized = False

    def _recording(self) -> bool:
        if not self._interrupted:
            return True
        self._polls_since_interrupt += 1
        if self._polls_since_interrupt >= self._polls_to_finalize:
            # The muxer got to the end only because nobody pulled the session out from
            # under it. A signal to the local client before this point is fatal.
            self.finalized = not self._proc.signals and not self._proc.killed
            return False
        return True

    def interrupt(self) -> None:
        self._interrupted = True

    def shell(self, command: str) -> str:
        if command.startswith("ps "):
            return _PS_RECORDING if self._recording() else _PS_IDLE
        if command.startswith("ls -l"):
            return command.split()[2]
        return ""

    def pull(self, remote: str, local: str) -> None:
        Path(local).write_bytes(_FINALIZED if self.finalized else _TRUNCATED)


def _device(u2: _FakeU2, proc: _FakeProc | None) -> Uiautomator2Device:
    dev = object.__new__(Uiautomator2Device)
    dev.serial = "emulator-5554"
    dev._settle = 0.0
    dev._d = u2
    dev._winsize = None
    dev._recording_remote = _REMOTE
    dev._recording_proc = proc
    return dev


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))


def test_stop_waits_for_the_device_before_signalling_its_own_adb_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The measured failure: the client dies first, so the moov box is never written."""
    proc = _FakeProc()
    u2 = _FakeU2(proc)

    def fake_run(argv: list[str], **kwargs: Any) -> Any:
        assert "pkill" in argv, argv
        u2.interrupt()
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    dev = _device(u2, proc)

    dest = dev.stop_recording(str(tmp_path / "evidence.mp4"))

    assert proc.signals == [], "the local adb client was signalled mid-finalize"
    assert not proc.killed
    assert Path(dest).read_bytes() == _FINALIZED


def test_a_client_that_outlives_the_device_process_is_still_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Waiting is not leaking: once the device is done, a client still up gets its SIGINT."""
    proc = _FakeProc()
    u2 = _FakeU2(proc, polls_to_finalize=1)

    def fake_run(argv: list[str], **kwargs: Any) -> Any:
        u2.interrupt()
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    dev = _device(u2, proc)
    dev.stop_recording(str(tmp_path / "evidence.mp4"))

    assert dev._recording_proc is None
    assert proc.poll() is not None, "the local adb client was left running"
    assert proc.signals in ([], [signal.SIGINT]), proc.signals


def test_a_stale_in_memory_handle_does_not_refuse_the_next_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed stop leaves the handle set; the device says idle, so start must not refuse.

    In the warm-daemon shape `record start` and `record stop` share one Device object, so a
    stop that raised left `_recording_remote` pointing at the old path forever. Every later
    `record start` was refused with "a screen recording is already in progress", and every
    later `record stop` reported on that first path — which is why the second failure in the
    sweep named the *first* call's `--remote` file. The on-disk handle already defers to
    `ps`; the in-memory one has to as well.
    """
    proc = _FakeProc()
    u2 = _FakeU2(proc)
    u2._interrupted = True
    u2._polls_since_interrupt = 99  # the device is idle: nothing is recording
    dev = _device(u2, None)
    dev._recording_remote = "/sdcard/aua_previous.mp4"

    started: list[list[str]] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _FakeProc:
        started.append(argv)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    remote = dev.start_recording("/sdcard/aua_next.mp4", time_limit_s=30)

    assert remote == "/sdcard/aua_next.mp4"
    assert dev._recording_remote == "/sdcard/aua_next.mp4"
    assert started and started[0][-1] == "/sdcard/aua_next.mp4"


def test_a_device_that_never_finishes_still_reports_the_path_it_was_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing here loosens the guard: an unfinalized recording is still refused."""
    proc = _FakeProc()
    u2 = _FakeU2(proc, polls_to_finalize=10**6)  # never finishes

    def fake_run(argv: list[str], **kwargs: Any) -> Any:
        u2.interrupt()
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "android_ui_analyser.device._SCREENRECORD_FINALIZE_TIMEOUT_S", 0.3, raising=False
    )
    dev = _device(u2, proc)

    with pytest.raises(DeviceError) as failure:
        dev.stop_recording(str(tmp_path / "evidence.mp4"))

    assert _REMOTE in str(failure.value.hint or "")
    assert not (tmp_path / "evidence.mp4").exists()
