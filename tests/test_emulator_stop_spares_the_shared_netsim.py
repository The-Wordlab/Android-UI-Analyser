"""Stopping one emulator must not take down the others.

The first emulator on a host spawns `netsimd`, the network simulator every later emulator
shares, and that child lives in the first emulator's process group. Stopping the emulator by
signalling its whole group killed `netsimd`; each other running emulator then logged
"Netsim Wifi ... is gone due to CANCELLED" and shut itself down. With four parallel QA lanes,
the first lane to finish took a neighbour's device with it in every run.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import android_ui_analyser.emulator as emu
from android_ui_analyser.leases import _proc_started

# Stands in for qemu: a session leader whose child (the shared simulator) joins its group.
EMULATOR = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stopping_an_emulator_spares_the_simulator_in_its_group() -> None:
    emulator = subprocess.Popen(
        [sys.executable, "-c", EMULATOR], stdout=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    assert emulator.stdout is not None
    netsim = int(emulator.stdout.readline())
    try:
        assert os.getpgid(netsim) == emulator.pid, "the stand-in must share the emulator's group"
        started = _proc_started(emulator.pid)

        assert emu._terminate_recorded_process(emulator.pid, started)
        emulator.wait(timeout=10)

        time.sleep(0.2)
        assert _alive(netsim), "the shared simulator died with the emulator that spawned it"
    finally:
        for pid in (netsim, emulator.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_no_emulator_stop_signals_a_whole_process_group() -> None:
    """Every stop path (rollback, owner-scope, spawned instance) signals the emulator alone."""
    source = Path(emu.__file__).read_text(encoding="utf-8")
    assert "os.killpg(" not in source
