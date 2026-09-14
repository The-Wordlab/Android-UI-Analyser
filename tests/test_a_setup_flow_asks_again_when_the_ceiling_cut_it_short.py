"""A setup flow that ran out of wait budget is re-issued, not abandoned.

Measured on a real run: the guest-entry flow's `wait_for` asked for 60s, got the ~5s
`perf.max_wait_ms` ceiling, and the cold start needed a little more. The harness turned that
into a hard `RunError`, so a healthy app produced `unverified` and spent a device for
nothing. AUA's own answer to a shortened wait is to ask again from the same step - there is
a `resume_from_step` on the failure for exactly that - so the harness does.

What must NOT become a retry loop is every other divergence. A missing element, an unsafe
step or a failed assertion is information about the flow or the app; repeating it spends the
device on the same answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from experiments.aua_controller.run_realapp import SETUP_FLOW_RESUMES, replay_setup_flow

FLOW = "name: enter\nsteps:\n  - wait_for: { id: containerDetail, timeout_ms: 60000 }\n"


class Recorder:
    """Answers `flow_run` from a script and remembers what it was asked."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, name: str, arguments: dict[str, Any], actor: str) -> dict[str, Any]:
        assert name == "flow_run"
        self.calls.append(arguments)
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


def _timed_out(step: int = 0) -> dict[str, Any]:
    return {"ok": False, "code": "wait_timeout", "step_index": step, "resume_from_step": step}


def test_a_shortened_wait_is_re_issued_from_the_step_it_stopped_on() -> None:
    call = Recorder([_timed_out(4), {"ok": True}])

    flow, resumes = asyncio.run(replay_setup_flow(call, FLOW, None))

    assert flow["ok"] is True
    assert resumes == 1
    assert "from_step" not in call.calls[0]
    assert call.calls[1]["from_step"] == 4


def test_the_flow_body_is_resubmitted_unchanged() -> None:
    """The resume repeats the same flow; only the entry point moves."""
    call = Recorder([_timed_out(2), {"ok": True}])

    asyncio.run(replay_setup_flow(call, FLOW, {"user": "guest"}))

    assert call.calls[1]["yaml"] == FLOW
    assert call.calls[1]["params"] == {"user": "guest"}


def test_asking_again_is_bounded() -> None:
    call = Recorder([_timed_out(1)])

    flow, resumes = asyncio.run(replay_setup_flow(call, FLOW, None))

    assert flow["ok"] is False
    assert resumes == SETUP_FLOW_RESUMES
    assert len(call.calls) == SETUP_FLOW_RESUMES + 1


def test_every_other_divergence_is_reported_once() -> None:
    call = Recorder([{"ok": False, "code": "element_not_found", "resume_from_step": 3}])

    flow, resumes = asyncio.run(replay_setup_flow(call, FLOW, None))

    assert flow["code"] == "element_not_found"
    assert resumes == 0
    assert len(call.calls) == 1


def test_a_timeout_with_nowhere_to_resume_is_not_retried() -> None:
    call = Recorder([{"ok": False, "code": "wait_timeout"}])

    _flow, resumes = asyncio.run(replay_setup_flow(call, FLOW, None))

    assert resumes == 0
    assert len(call.calls) == 1
