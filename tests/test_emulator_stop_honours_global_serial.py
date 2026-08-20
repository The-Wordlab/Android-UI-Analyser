"""`aua --serial <s> emulator stop` must stop `<s>`, not report a success that stopped nothing.

Observed across a five-worker sweep: `aua --serial emulator-5556 emulator stop --owner lane3`
returned `{"ok": true, "stopped": []}` and left qemu running. The serial written *before* the
subcommand — the position every other command wants it in — was read into `GlobalOpts`, put into
`config.device.serial`, and then never looked at by this command, which only ever read its own
`--serial`. The call fell through to the owner branch, matched no records, and answered `ok:true`.

`hoist_global_options` is right not to move it (`emulator stop --serial` names the emulator to
kill, so the command "defines the option itself" and the hoist correctly leaves it alone) — which
is exactly why the global one had to be honoured here instead.

Why it is worth a test rather than a doc note: a teardown that silently does nothing is the root
cause of every orphaned instance in that sweep, including one where the coordinator later read a
leftover registry entry as an orphan and killed a **live** worker. Reporting `ok:true` for a stop
that stopped nothing is the failure that produced both.

Deliberately *not* honoured: `device.serial` coming from `$AUA_SERIAL` or a config file. A
coordinator with a stale `AUA_SERIAL` exported running `emulator stop --owner other-lane` would
otherwise silently kill the wrong device — the same accident, in the same direction. Only the
serial typed on this command line targets a kill.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import emulator as emulator_mod
from android_ui_analyser.cli import app
from android_ui_analyser.errors import ExitCode

runner = CliRunner()

# Outside the real emulator port range (5554-5682, even ports only), so a bug in this file's
# stubbing still cannot reach a device somebody is using.
SER_A = "emulator-9998"
SER_B = "emulator-9996"

RUNNING = [
    {"serial": SER_A, "model": "pixel", "android_version": "16", "state": "device"},
    {"serial": SER_B, "model": "pixel", "android_version": "16", "state": "device"},
]


@pytest.fixture(autouse=True)
def _sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Fake the device list, neutralise the kill, and isolate the registry.

    Autouse and unconditional: every test here drives `emulator stop`, whose whole job is to
    terminate emulators, and the serial-scoped path also *unlinks* matching pid records. A
    per-test stub is one forgotten decorator away from doing both for real.
    """
    monkeypatch.setattr(emulator_mod, "running_emulators", lambda: list(RUNNING))
    killed: list[str] = []
    monkeypatch.setattr(emulator_mod, "_adb_emu_kill", killed.append)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("AUA_SERIAL", raising=False)
    return killed


def _stop(*argv: str) -> Any:
    result = runner.invoke(app, list(argv))
    return result


def test_global_serial_before_the_subcommand_stops_that_emulator(_sandbox: list[str]) -> None:
    """The exact call from the sweep: global --serial plus --owner."""
    result = _stop("--serial", SER_A, "emulator", "stop", "--owner", "lane3")
    assert result.exit_code == ExitCode.OK, result.output
    payload = json.loads(result.stdout)
    assert payload["stopped"] == [SER_A], (
        "a stop that reports ok:true with an empty `stopped` list is the bug: "
        f"got {payload!r}"
    )
    assert _sandbox == [SER_A], "the kill must actually be issued, and only for that serial"


def test_global_serial_alone_is_an_explicit_target(_sandbox: list[str]) -> None:
    """`aua --serial X emulator stop` is targeted, so the "needs a target" guard must not fire."""
    result = _stop("--serial", SER_A, "emulator", "stop")
    assert result.exit_code == ExitCode.OK, result.output
    assert json.loads(result.stdout)["stopped"] == [SER_A]
    assert SER_B not in _sandbox, "a scoped stop must not touch the other emulator"


def test_subcommand_serial_still_works(_sandbox: list[str]) -> None:
    """The position that always worked must keep working — this fix only widens what is accepted."""
    result = _stop("emulator", "stop", "--serial", SER_A)
    assert result.exit_code == ExitCode.OK, result.output
    assert json.loads(result.stdout)["stopped"] == [SER_A]


def test_two_different_serials_are_refused_rather_than_guessed(_sandbox: list[str]) -> None:
    """Both positions work, so naming two devices is a real ambiguity — refuse, kill nothing."""
    result = _stop("--serial", SER_A, "emulator", "stop", "--serial", SER_B)
    assert result.exit_code == ExitCode.USAGE, result.output
    assert _sandbox == [], "refused, so nothing may have been signalled"
    assert SER_A in result.output and SER_B in result.output


def test_same_serial_in_both_positions_is_not_a_conflict(_sandbox: list[str]) -> None:
    result = _stop("--serial", SER_A, "emulator", "stop", "--serial", SER_A)
    assert result.exit_code == ExitCode.OK, result.output
    assert json.loads(result.stdout)["stopped"] == [SER_A]


def test_aua_serial_env_does_not_target_a_kill(
    monkeypatch: pytest.MonkeyPatch, _sandbox: list[str]
) -> None:
    """An ambient `$AUA_SERIAL` must not turn an untargeted stop into a kill.

    Every read command resolves `device.serial` through the env, but a kill is not undoable and
    a stale exported serial is how a coordinator would destroy somebody else's device while
    believing it had asked for its own. The refusal is loud and lists the running serials.
    """
    monkeypatch.setenv("AUA_SERIAL", SER_A)
    result = _stop("emulator", "stop")
    assert result.exit_code == ExitCode.USAGE, result.output
    assert _sandbox == [], "nothing may die on an untargeted stop"
