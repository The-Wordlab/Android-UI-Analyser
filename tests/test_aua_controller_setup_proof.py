"""A setup precondition a caller can prove, without the engine learning the product.

Some preconditions cannot be read off the screen: a signed-in session and a guest session can
look identical, so a caller that judges the tier from Home will happily record a pass it never
established. This lets the caller name a string that proves it and a label to record, and
returns only whether that string was there.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.run_realapp import capture_setup_proof

PROOF = ("setup_tier", '"tier":"premium"', "premium")
SECRET = '{"email":"someone@example.test","tier":"premium"}'


seen_argv: list[list[str]] = []


def _logcat(monkeypatch, *, lines=None, returncode=0, stdout=None, raises=None):
    seen_argv.clear()
    def fake_run(argv, **kwargs):
        if raises is not None:
            raise raises
        # Filtered on the device: pulling the whole buffer timed out on a longer run.
        assert "logcat" in argv and "--grep" in argv, argv
        assert argv[argv.index("--grep") + 1] == PROOF[1], argv
        seen_argv.append(argv)
        body = stdout if stdout is not None else json.dumps({"lines": lines or []})
        return subprocess.CompletedProcess(argv, returncode, body, "")
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_a_present_pattern_proves_the_precondition(monkeypatch):
    _logcat(monkeypatch, lines=["D/Net: " + SECRET])
    assert capture_setup_proof("aua", PROOF) == {
        "verified": True, "actual": "premium", "source": "logcat"
    }


def test_the_matching_line_is_never_returned(monkeypatch):
    """The body that carries the field carries the account address beside it."""
    _logcat(monkeypatch, lines=["D/Net: " + SECRET])
    assert "someone@example.test" not in json.dumps(capture_setup_proof("aua", PROOF))


def test_a_loose_device_filter_cannot_manufacture_a_positive(monkeypatch):
    """`--grep` is a regex, so the substring is re-checked against what came back."""
    _logcat(monkeypatch, lines=['D/Net: {"tier":"premium-trial-expired"}'.replace("premium", "prem")])
    assert capture_setup_proof("aua", PROOF) == {
        "verified": False, "actual": None, "source": "logcat"
    }


def test_an_absent_pattern_is_an_unproved_precondition_not_a_proved_one(monkeypatch):
    _logcat(monkeypatch, lines=["D/Net: {\"tier\":\"free\"}"])
    assert capture_setup_proof("aua", PROOF) == {
        "verified": False, "actual": None, "source": "logcat"
    }


@pytest.mark.parametrize("broken", [
    {"returncode": 1},
    {"stdout": "not json at all"},
    {"stdout": json.dumps({"lines": "not a list"})},
    {"raises": OSError("aua is not installed")},
    {"raises": subprocess.TimeoutExpired("aua", 90)},
])
def test_a_capture_that_cannot_run_records_nothing(monkeypatch, broken):
    """None, not False: a log we could not read is not a precondition that failed.

    Recording False here would turn every environment where the log is unreadable into a
    confident negative, and recording True would invent a precondition that was never shown.
    Returning None leaves the caller exactly where it was before it asked.
    """
    _logcat(monkeypatch, **broken)
    assert capture_setup_proof("aua", PROOF) is None


def test_the_capture_targets_the_sessions_own_device(monkeypatch):
    """Without a serial the lookup resolves by lease scope and can pick another emulator.

    Seen on 2026-09-15: the session held emulator-5560 while several emulators were up, and the
    capture came back empty on a run whose login had plainly succeeded, so a passing row was
    reported as having no tier proof.
    """
    _logcat(monkeypatch, lines=["D/Net: " + SECRET])
    capture_setup_proof("aua", PROOF, "emulator-5560")
    argv = seen_argv[0]
    assert "--serial" in argv, argv
    assert argv[argv.index("--serial") + 1] == "emulator-5560", argv
    assert argv.index("--serial") < argv.index("logcat"), "--serial is a global option"


def test_no_serial_still_works_for_a_single_target_host(monkeypatch):
    _logcat(monkeypatch, lines=["D/Net: " + SECRET])
    assert capture_setup_proof("aua", PROOF)["verified"] is True
    assert "--serial" not in seen_argv[0]
