"""A recording's existence is decided by the device, not by a file keyed to serial+cache dir.

Three symptoms in one sweep, one cause — the on-disk handle written by `record start` is
addressed by two things that move underneath it:

- **the cache directory.** The handle lives under `AUA_CACHE__DIR`. A worker that exports a
  per-worker cache dir *between* its start and its stop looks elsewhere at stop time, so
  `record stop` reported "no screen recording in progress" for a recording that had plainly
  run — and the video was lost rather than truncated.
- **the serial.** Console ports recycle within minutes, so a handle orphaned by a killed
  worker made the *next* worker's first `record start` fail with "already in progress"
  before it had started anything.

`ps` describes this boot of this device, so it settles both. These tests pin that the
device's answer wins, and — the part that makes this safe — that an *unreadable* `ps` is
never mistaken for an idle device, because clearing a live handle would start a second
recorder over the top of the first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.errors import DeviceError

_PS_IDLE = "ARGS\ninit second_stage\n[kthreadd]\n"
_PS_RECORDING = (
    "ARGS\ninit second_stage\nscreenrecord --time-limit 180 /sdcard/aua_recording.mp4\n"
)


class _FakeU2:
    """Just enough uiautomator2 surface for the recording paths: `shell` and `pull`."""

    def __init__(self, ps_output: str | None, *, remote_exists: bool = True) -> None:
        self._ps = ps_output  # None → `ps` unreadable (raises, as an offline device would)
        self._remote_exists = remote_exists
        self.shell_calls: list[str] = []
        self.pulls: list[tuple[str, str]] = []

    def shell(self, command: str) -> str:
        self.shell_calls.append(command)
        if command.startswith("ps "):
            if self._ps is None:
                raise RuntimeError("device offline")
            return self._ps
        if command.startswith("ls -l"):
            return command.split()[2] if self._remote_exists else ""
        return ""

    def pull(self, remote: str, local: str) -> None:
        self.pulls.append((remote, local))
        Path(local).write_bytes(b"\x00fake mp4 payload")


def _device(u2: _FakeU2) -> Uiautomator2Device:
    """A real Uiautomator2Device with a fake connection — `__init__` would need a phone."""
    dev = object.__new__(Uiautomator2Device)
    dev.serial = "emulator-5554"
    dev._settle = 0.0
    dev._d = u2
    dev._winsize = None
    dev._recording_remote = None
    dev._recording_proc = None
    return dev


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real recording handles."""
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))


@pytest.fixture(autouse=True)
def _no_real_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reaching a real `adb` must be an error, not a recording on someone's live device.

    Both recording paths shell out: `stop` runs `adb … pkill -l 2 screenrecord`, `start`
    spawns `adb … screenrecord`. While proving these tests fail without the fix, the
    unfixed `start` fell straight through to the real `Popen` — so a regression here would
    kill or start a recording on whatever device is attached. Same reasoning as
    `test_no_real_device_kills.py`: make the dangerous call an error by default and let a
    test opt in deliberately.
    """
    import subprocess

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"test tried to run a real subprocess: {args!r}")

    monkeypatch.setattr(subprocess, "run", _refuse)
    monkeypatch.setattr(subprocess, "Popen", _refuse)


def test_stop_recovers_a_running_recording_when_the_handle_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cache-dir-changed case: the handle is gone, the recording is real, keep the video."""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    u2 = _FakeU2(_PS_RECORDING)
    dev = _device(u2)
    assert not dev._recording_state_path().exists()  # nothing local to recover from

    dest = dev.stop_recording(str(tmp_path / "evidence.mp4"))

    assert Path(dest).is_file() and Path(dest).stat().st_size > 0
    # It pulled the path the *device* reported, not a guess.
    assert u2.pulls == [("/sdcard/aua_recording.mp4", dest)]


def test_stop_still_reports_nothing_when_the_device_agrees_nothing_runs() -> None:
    """The recovery must not invent a recording; an honest failure is still required."""
    dev = _device(_FakeU2(_PS_IDLE))
    with pytest.raises(DeviceError, match="no screen recording in progress"):
        dev.stop_recording("/tmp/never-written.mp4")


def test_start_clears_an_orphaned_handle_from_a_recycled_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead worker's handle must not block the next worker's first `record start`."""
    u2 = _FakeU2(_PS_IDLE)
    dev = _device(u2)
    state = dev._recording_state_path()
    state.write_text(json.dumps({"remote": "/sdcard/someone_elses.mp4", "time_limit_s": 180}))

    started: dict[str, Any] = {}

    class _FakePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            started["argv"] = argv

        def poll(self) -> None:
            return None

    monkeypatch.setattr("subprocess.Popen", _FakePopen)

    remote = dev.start_recording("/sdcard/mine.mp4")

    assert remote == "/sdcard/mine.mp4"
    assert "screenrecord" in started["argv"]
    assert json.loads(state.read_text())["remote"] == "/sdcard/mine.mp4"


def test_start_refuses_when_the_device_says_one_is_already_running() -> None:
    """With no local handle at all, a live recorder still has to be respected."""
    dev = _device(_FakeU2(_PS_RECORDING))
    assert not dev._recording_state_path().exists()
    with pytest.raises(DeviceError, match="already in progress"):
        dev.start_recording("/sdcard/second.mp4")


def test_start_refuses_on_an_unreadable_ps_rather_than_clearing_a_live_handle() -> None:
    """An unverifiable `ps` must not be read as an idle device.

    Getting this wrong is worse than the bug being fixed: it would clear a handle for a
    recording that *is* running and start a second recorder over the top of it.
    """
    dev = _device(_FakeU2(None))
    dev._recording_state_path().write_text(json.dumps({"remote": "/sdcard/live.mp4"}))
    with pytest.raises(DeviceError, match="already in progress"):
        dev.start_recording("/sdcard/second.mp4")
