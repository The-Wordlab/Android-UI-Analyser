"""No unit test may terminate a real emulator.

`stop()` shells out to `adb -s <serial> emu kill`. `test_emulator_stop_guard.py` faked the
device list with the serials a developer actually runs (`emulator-5554`, `emulator-5556`) and
stubbed the kill in one of its four tests — via `monkeypatch.setattr(..., raising=False)` on a
name that did not exist, so it stubbed nothing. Every full-suite run therefore killed the live
emulators, repeatedly, mid-session.

Two properties make that impossible to repeat: the kill is a patchable module-level seam, and
the autouse `_aua_never_kill_a_real_emulator` fixture makes reaching it an error by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser import emulator as emulator_mod
from android_ui_analyser.errors import UsageError


def test_the_kill_is_a_patchable_seam() -> None:
    """If this is inlined back into `stop`, stubbing becomes impossible and the footgun returns."""
    assert callable(getattr(emulator_mod, "_adb_emu_kill", None)), (
        "emulator.stop must route its kill through a module-level _adb_emu_kill seam"
    )


def test_reaching_the_kill_is_an_error_by_default() -> None:
    """The autouse guard must be active for every test, with no opt-in required."""
    with pytest.raises(AssertionError, match="tried to kill emulator"):
        emulator_mod._adb_emu_kill("emulator-9998")


def test_stop_cannot_kill_a_device_without_an_explicit_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop() call in a test that forgot to stub must fail loudly, not kill a device."""
    monkeypatch.setattr(
        emulator_mod,
        "running_emulators",
        lambda: [{"serial": "emulator-9998", "model": "m", "android_version": "16", "state": "device"}],
    )
    with pytest.raises(AssertionError, match="tried to kill emulator"):
        emulator_mod.stop(all_devices=True, cache_dir=tmp_path)


def test_the_untargeted_guard_still_refuses_before_any_kill(tmp_path: Path) -> None:
    """Cheap regression on the other half: no target means refuse, not kill everything."""
    with pytest.raises(UsageError):
        emulator_mod.stop(cache_dir=tmp_path)
