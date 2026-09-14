"""A flow's `wait_for` failure has to say whether the budget or the app ended it.

Every observation wait is capped by `perf.max_wait_ms`, so a flow that writes
`timeout_ms: 60000` gets the ceiling instead and reports `wait_timeout` either way. "Late"
and "not there" have opposite remedies - ask again, or stop believing the marker exists -
and the QA lessons record a lane reporting a healthy app as broken on exactly that
confusion. Measured here: a real setup flow's `wait_for containerDetail`-shaped step gave up
after ~5s of a 60s request and the run ended `unverified`, with nothing in the payload to
say the wait had been shortened.

The clamp was already on the wait result. This only carries it out to the failure, beside
the `resume_from_step` a caller acts on.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config
from test_memory import HOME, P


def _engine(tmp_path: Path) -> Engine:
    return Engine(
        make_config(
            memory={"enabled": True, "dir": str(tmp_path / "memory")},
            cache={"dir": str(tmp_path / "cache")},
            daemon={"enabled": False},
        ),
        device=FakeDevice(hierarchy_xml=HOME, package=P, serial="wait-budget"),
    )


def _run_waiting_for_an_absent_marker(tmp_path: Path, *, requested_ms: int) -> dict:
    engine = _engine(tmp_path)
    engine.config.perf.max_wait_ms = 50  # a ceiling the test can afford to spend
    engine.config.perf.wait_ceiling_min_ms = 50
    flow_path = tmp_path / "f.yaml"
    flow_path.write_text(
        f"name: f\napp: {P}\nsteps:\n"
        f"  - wait_for: {{ id: noSuchContainer, timeout_ms: {requested_ms} }}\n",
        encoding="utf-8",
    )
    return engine.flow_run(file=str(flow_path))


def test_a_wait_the_ceiling_shortened_says_so(tmp_path: Path) -> None:
    result = _run_waiting_for_an_absent_marker(tmp_path, requested_ms=60000)

    assert result["ok"] is False
    assert result["code"] == "wait_timeout"
    detail = result.get("failure_detail") or ""
    assert "60000" in detail, detail
    assert "perf.max_wait_ms" in detail, detail


def test_the_caller_is_told_where_to_resume(tmp_path: Path) -> None:
    """The remedy for a shortened wait is another call from the same step.

    A flow run from a file gets the command to repeat; one submitted inline - which is how
    the controller replays a setup flow - gets the step index, because there is no command
    to name.
    """
    result = _run_waiting_for_an_absent_marker(tmp_path, requested_ms=60000)

    assert result["step_index"] == 0
    assert "--from-step 0" in result.get("resume_call", "") or result.get(
        "resume_from_step"
    ) == 0, result


def test_a_wait_inside_the_ceiling_does_not_blame_the_budget(tmp_path: Path) -> None:
    """A marker that is genuinely absent must not be excused as a short budget."""
    result = _run_waiting_for_an_absent_marker(tmp_path, requested_ms=10)

    assert result["ok"] is False
    assert result["code"] == "wait_timeout"
    assert "failure_detail" not in result, result.get("failure_detail")
