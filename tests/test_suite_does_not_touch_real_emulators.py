"""The suite must not consult whatever is plugged into the developer's machine.

`emulator.start` cross-checks the AVD name before accepting a serial, so a port collision
cannot make one agent drive another agent's device. That check shells out to
`adb -s <serial> emu avd name`, and patching `running_emulators` does not stop it.

The consequence was a suite that passed or failed depending on what was running: a test whose
fake device is "emulator-5554" asked the *real* emulator-5554 what it was, got a genuine AVD
name back, and failed the comparison against its own fixture. Reproduced deterministically —
with the console answering "aua_proxy34" the start times out; with it silent it succeeds.
"""

from __future__ import annotations

from android_ui_analyser import emulator as emulator_mod


def test_avd_name_of_serial_is_neutralised_during_tests() -> None:
    """None is what a console that will not answer returns, and the code treats it as
    "keep waiting" rather than as a mismatch — so it is the honest stand-in."""

    assert emulator_mod.avd_name_of_serial("emulator-5554") is None
    assert emulator_mod.avd_name_of_serial("emulator-9999") is None


def test_a_foreign_avd_name_is_still_rejected_in_production() -> None:
    """The guard must not weaken the real check it is standing in for.

    Answering to somebody else's device is the failure this cross-check exists to prevent, so
    a serial whose console reports a different AVD must never satisfy the wait.
    """

    seen: list[str] = []

    def foreign(serial: str) -> str:
        seen.append(serial)
        return "somebody-elses-avd"

    original = emulator_mod.avd_name_of_serial
    emulator_mod.avd_name_of_serial = foreign  # type: ignore[assignment]
    try:
        got = emulator_mod._wait_for_serial(
            "emulator-5554", timeout_s=0.1, expect_avd="the-one-i-launched"
        )
    finally:
        emulator_mod.avd_name_of_serial = original  # type: ignore[assignment]

    assert got is None, "a device belonging to another AVD must not satisfy the wait"
