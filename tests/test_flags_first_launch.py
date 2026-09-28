"""A set-flags deeplink that is the app's first launch must get to write before it is killed.

Kept apart from ``test_flags.py``, whose autouse fixture zeroes every poll window: this test is
about the length of the real one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser.engine import Engine
from test_flags import PKG, PREFS_FILE, device_with, make_flags_engine

# Measured 2026-09-28 on an SDK 34 emulator, fresh install of a large debug build, flags link as
# the first launch: `Start proc` to `finishAttachApplication` took 2.9 s, so the flag activity's
# onCreate had not run when the 2 s read-back gave up and the restart force-stopped the process.
# Five of five flags were lost on two of two attempts; with the app launched once before, all
# five landed.
FIRST_LAUNCH_WRITE_S = 3.5


def test_a_first_launch_cold_start_writes_its_flags_before_the_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    flags_home = sys.modules[Engine.flags_set.__module__]
    monkeypatch.setattr(flags_home, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))

    device = device_with({"hub": "a"})
    written = device.prefs.pop(PREFS_FILE)
    serve = device._run_as

    def run_as(command: str) -> str:
        if clock.now >= FIRST_LAUNCH_WRITE_S:
            device.prefs[PREFS_FILE] = written
        return serve(command)

    monkeypatch.setattr(device, "_run_as", run_as)
    engine = make_flags_engine(tmp_path, device)

    result = engine.flags_set(PKG, ["hub=a"], observe=False)

    assert result["applied"] == {"hub": "a"}, result.get("detail")
    assert result["ok"] is True
