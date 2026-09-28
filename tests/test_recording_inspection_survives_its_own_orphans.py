"""Recording inspection must survive a shell that exits while it is being inspected.

Found in a two-lane sweep (2026-09-28). A `record stop` failed with "ambiguous recording
command line", its cleanup stayed pending, and every later session on that emulator failed
`record start` with `recording_cleanup_pending`. On the device sat an earlier inspection whose
toybox `tr` had opened the cmdline of a shell that then exited: it spun at most of a CPU for as
long as the emulator lived, under a parent shell that outlived the transport timeout. Its own
command line is this module's script, `AUA_UNKNOWN` included, so the marker must match exactly.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_recording

ROOT = "/sdcard/aua.aua-recording-8d1f0c2b9a6e4f7d8c3b2a1908f7e6d5"
# What an orphaned inspection shell looks like in /proc after `tr "\000\n" "  "`.
ORPHAN = ("/system/bin/sh -c # AUA_RECORDING_CMDLINES for pid in 166 10334; do if [ -r "
          '/proc/$pid/cmdline ]; then command=$(tr "\\000\\n" "  " < /proc/$pid/cmdline) || '
          '{ echo "$pid AUA_UNKNOWN"; continue; }; fi; done; echo AUA_CMDLINES_COMPLETE')


class _Device:
    def __init__(self, ps: dict[int, str], cmdlines: dict[int, str]) -> None:
        self._ps, self._cmdlines, self.scripts = ps, cmdlines, []

    def shell(self, command: str) -> str:
        if command == "ps -A -w -o PID,ARGS":
            return "PID ARGS\n" + "\n".join(f"{pid} {args}" for pid, args in self._ps.items())
        assert command.startswith("# AUA_RECORDING_CMDLINES"), command
        self.scripts.append(command)
        pids = command.split("for pid in ", 1)[1].split(";", 1)[0].split()
        return "".join(f"{pid} {self._cmdlines[int(pid)]}\n" for pid in pids) + "AUA_CMDLINES_COMPLETE"


def test_an_orphaned_inspection_shell_is_an_ordinary_process_not_an_unknown_one() -> None:
    encoder = f"screenrecord --time-limit 180 {ROOT}/segment-0.mp4"
    dev = _Device({166: "sh", 11148: "sh -c # AUA_RECORDING_CMDLINES", 20001: "screenrecord"},
                  {166: "/system/bin/sh", 11148: ORPHAN, 20001: encoder})

    table = dict(android_recording._process_table(dev))

    assert table[11148] == ORPHAN
    assert android_recording._processes(dev, ROOT) == [(20001, encoder)]


def test_a_cmdline_the_device_could_not_read_still_refuses() -> None:
    dev = _Device({166: "sh"}, {166: "AUA_UNKNOWN"})

    with pytest.raises(DeviceError, match="ambiguous recording command line"):
        android_recording._process_table(dev)


def test_each_cmdline_is_read_with_one_bounded_read() -> None:
    """`tr < /proc/$pid/cmdline` never returns once that process exits after the open."""
    dev = _Device({166: "sh"}, {166: "/system/bin/sh"})

    android_recording._process_table(dev)

    (script,) = dev.scripts
    assert "dd if=/proc/$pid/cmdline bs=65536 count=1" in script
    assert "< /proc/$pid/cmdline" not in script
