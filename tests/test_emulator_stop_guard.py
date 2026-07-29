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

RUNNING = [
    {"serial": "emulator-5554", "model": "pixel", "android_version": "16", "state": "device"},
    {"serial": "emulator-5556", "model": "pixel", "android_version": "16", "state": "device"},
]


@pytest.fixture(autouse=True)
def _fake_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emulator_mod, "running_emulators", lambda: list(RUNNING))


def test_untargeted_stop_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as err:
        emulator_mod.stop(cache_dir=tmp_path)
    assert err.value.exit_code == ExitCode.USAGE
    # The message must name what WOULD have died, so the caller can pick one.
    assert "emulator-5554" in (err.value.hint or "")
    assert "--all" in str(err.value)


def test_untargeted_stop_kills_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal must happen BEFORE any kill is issued."""
    killed: list[str] = []
    monkeypatch.setattr(emulator_mod, "_adb_emu_kill", lambda s: killed.append(s), raising=False)
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
    out = emulator_mod.stop(serial="emulator-5554", cache_dir=tmp_path)
    assert out["ok"] is True
    stopped = [s if isinstance(s, str) else s.get("serial") for s in out.get("stopped") or []]
    assert "emulator-5556" not in stopped, "a scoped stop must not touch the other emulator"
