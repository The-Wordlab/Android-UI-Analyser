"""`emulator stop` must be told what to kill.

`targets = running_emulators()` meant a bare `aua emulator stop` killed EVERY running
emulator. The command reads like "stop the one I started", but an emulator can be holding a
signed-in session, seeded data, or belong to something else on the same machine — and
`adb emu kill` is not undoable. `app clear` already requires `--yes` for the same reason;
this makes the two consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser import emulator as emulator_mod
from android_ui_analyser.errors import ExitCode, UsageError

# Serials outside the real emulator port range (5554-5682, even ports only), so that even a
# bug in this file's stubbing cannot reach a device someone is using.
SER_A = "emulator-9998"
SER_B = "emulator-9996"

RUNNING = [
    {"serial": SER_A, "model": "pixel", "android_version": "16", "state": "device"},
    {"serial": SER_B, "model": "pixel", "android_version": "16", "state": "device"},
]


@pytest.fixture(autouse=True)
def _fake_running(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake the device list AND neutralise the kill for every test in this module.

    Autouse and unconditional on purpose: this suite exercises `stop`, whose whole job is to
    terminate emulators. A per-test stub is one forgotten decorator away from `adb emu kill`
    running for real — which is exactly what happened, repeatedly, against a live session.
    """
    monkeypatch.setattr(emulator_mod, "running_emulators", lambda: list(RUNNING))
    killed: list[str] = []
    monkeypatch.setattr(emulator_mod, "_adb_emu_kill", killed.append)
    return killed


def test_untargeted_stop_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as err:
        emulator_mod.stop(cache_dir=tmp_path)
    assert err.value.exit_code == ExitCode.USAGE
    # The message must name what WOULD have died, so the caller can pick one.
    assert SER_A in (err.value.hint or "")
    assert "--all" in str(err.value)


def test_untargeted_stop_kills_nothing(tmp_path: Path, _fake_running: list[str]) -> None:
    """The refusal must happen BEFORE any kill is issued."""
    killed = _fake_running
    with pytest.raises(UsageError):
        emulator_mod.stop(cache_dir=tmp_path)
    assert killed == [], "refused, so nothing may have been signalled"


def test_all_is_accepted_as_an_explicit_choice(tmp_path: Path) -> None:
    """`--all` is the deliberate form — it must NOT raise."""
    try:
        emulator_mod.stop(all_devices=True, cache_dir=tmp_path)
    except UsageError as exc:  # pragma: no cover - the guard must not fire here
        raise AssertionError(f"--all should be an explicit target, got: {exc}") from exc


def test_serial_still_scopes_to_one(tmp_path: Path) -> None:
    out = emulator_mod.stop(serial=SER_A, cache_dir=tmp_path)
    assert out["ok"] is True
    stopped = [s if isinstance(s, str) else s.get("serial") for s in out.get("stopped") or []]
    assert SER_B not in stopped, "a scoped stop must not touch the other emulator"
