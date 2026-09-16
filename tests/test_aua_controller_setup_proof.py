"""A setup precondition a caller can prove, without the engine learning the product.

Some preconditions cannot be read off the screen: a signed-in session and a guest session can
look identical, so a caller that judges the tier from Home will record a pass it never
established. This lets the caller name a string that proves it and a label to record, and
returns only whether that string was there.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller import run_realapp
from experiments.aua_controller.run_realapp import capture_setup_proof

PROOF = ("setup_tier", '"tier":"premium"', "premium")
SECRET = '{"email":"someone@example.test","tier":"premium"}'


def _calls(*payloads):
    """A fake session channel returning each payload in turn (last one repeats)."""
    seen: list[tuple[str, dict]] = []

    async def call(name, arguments, actor):
        seen.append((name, arguments))
        item = payloads[min(len(seen) - 1, len(payloads) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    return call, seen


def _run(call):
    return asyncio.run(capture_setup_proof(call, PROOF))


def test_a_present_pattern_proves_the_precondition():
    call, seen = _calls({"lines": ["D/Net: " + SECRET]})
    assert _run(call) == {"verified": True, "actual": "premium", "source": "logcat"}
    assert seen[0][0] == "logcat_dump"
    assert seen[0][1]["grep"] == re.escape(PROOF[1])


def test_it_reads_over_the_session_channel_not_a_second_process():
    """A subprocess derives its own worker scope and is refused against this session's device.

    An earlier version shelled out to `aua --serial <s> logcat`. It worked when a scenario ran
    alone and returned nothing under the panel's parallel workers -- `device_leased`, against the
    very device the session held -- silently failing rows whose login had succeeded.
    """
    import inspect
    body = inspect.getsource(capture_setup_proof)
    body = body.replace(capture_setup_proof.__doc__ or "", "")
    assert "subprocess" not in body, "the capture must not spawn a second aua process"
    assert "logcat_dump" in body


def test_the_matching_line_is_never_returned():
    call, _ = _calls({"lines": ["D/Net: " + SECRET]})
    assert "someone@example.test" not in repr(_run(call))


def test_an_absent_pattern_is_an_unproved_precondition_not_a_proved_one(monkeypatch):
    monkeypatch.setattr(run_realapp.asyncio, "sleep", _nosleep)
    call, _ = _calls({"lines": ['D/Net: {"tier":"free"}']})
    assert _run(call) == {"verified": False, "actual": None, "source": "logcat"}


def test_a_loose_filter_cannot_manufacture_a_positive(monkeypatch):
    """`grep` is a regex on AUA's side, so the substring is re-checked here."""
    monkeypatch.setattr(run_realapp.asyncio, "sleep", _nosleep)
    call, _ = _calls({"lines": ['D/Net: {"tier":"premium-trial-expired"}'.replace("premium", "prem")]})
    assert _run(call)["verified"] is False


async def _nosleep(_seconds):
    return None


def test_evidence_that_arrives_a_moment_late_is_still_caught(monkeypatch):
    monkeypatch.setattr(run_realapp.asyncio, "sleep", _nosleep)
    call, seen = _calls({"lines": []}, {"lines": []}, {"lines": ["D/Net: " + SECRET]})
    assert _run(call)["verified"] is True
    assert len(seen) == 3, "it must keep looking until the evidence appears"


def test_polling_stops_once_it_is_proved(monkeypatch):
    monkeypatch.setattr(run_realapp.asyncio, "sleep", _nosleep)
    call, seen = _calls({"lines": ["D/Net: " + SECRET]})
    _run(call)
    assert len(seen) == 1, "a proved precondition must not be re-read"


@pytest.mark.parametrize("payload", [
    RuntimeError("tool call failed"),
    {"lines": "not a list"},
    {},
])
def test_a_read_that_cannot_run_records_nothing(monkeypatch, payload):
    """None, not False: a log we could not read is not a precondition that failed."""
    monkeypatch.setattr(run_realapp.asyncio, "sleep", _nosleep)
    call, _ = _calls(payload)
    assert _run(call) is None


@pytest.mark.parametrize("latest,expected", [("premium", True), ("free", False)])
def test_regex_reads_latest_state_in_current_session_without_exposing_capture(latest, expected):
    call, seen = _calls({"lines": [SECRET, '{"tier": "' + latest + '"}']})
    result = asyncio.run(capture_setup_proof(
        call, ("setup_tier", r'"tier"\s*:\s*"(?P<value>[^"]+)"', "premium"),
        regex=True, since="this-session",
    ))
    assert seen[0][1]["since"] == "this-session"
    assert result == {"verified": expected, "actual": "premium" if expected else None,
                      "source": "logcat"}
    assert "someone@example.test" not in repr(result)


def test_literal_pattern_does_not_treat_regex_metacharacters_as_a_filter():
    call, seen = _calls({"lines": ["access [premium]"]})
    result = asyncio.run(capture_setup_proof(call, ("proof", "[premium]", "yes")))
    assert seen[0][1]["grep"] == r"\[premium\]"
    assert result["verified"] is True
