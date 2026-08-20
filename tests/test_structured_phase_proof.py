"""UI phase completion requires one exact, correlated, conjunctive proof."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.jobs import JobState, _complete_correlated_goal_phase
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import (
    complete_current_ui_phase_from_job,
    complete_current_ui_phase_from_observation,
    create_session_state,
    load_session_state,
)
from conftest import FakeDevice, make_config

SERIAL = "example-structured-proof"
OWNER = "example-agent"
GOAL = "Verify Example summary panel is visible, Loading is absent, and Reply control is enabled"


def _state(tmp_path: Path):
    return create_session_state(
        tmp_path,
        goal=GOAL,
        serial=SERIAL,
        owner=OWNER,
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )


def _element(
    element_id: int,
    text: str,
    *,
    enabled: bool = True,
) -> Element:
    return Element(
        id=element_id,
        type="android.widget.TextView",
        text=text,
        bounds=(0, element_id * 10, 800, element_id * 10 + 80),
        center=(400, element_id * 10 + 40),
        enabled=enabled,
        source=Source.hierarchy,
    )


def _observation(*, loading: bool = False) -> AnalyzeResult:
    elements = [
        _element(1, "Example summary panel"),
        _element(2, "Reply control", enabled=True),
    ]
    if loading:
        elements.append(_element(3, "Loading"))
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="com.example.catalog",
            source="hierarchy",
        ),
        elements=elements,
        meta=Meta(
            duration_ms=8,
            tier_used="hierarchy",
            path="hierarchy",
            device_serial=SERIAL,
            fingerprint="frame-abc123",
            via="hierarchy",
        ),
    )


def _job(state: Any, *, predicate: str | None = None) -> JobState:
    selected = predicate or "text:Example summary panel,!text:Loading,text:Reply control"
    rows = []
    for term in selected.split(","):
        negated = term.startswith("!")
        rows.append(
            {
                "term": term,
                "present": not negated,
                "satisfied": True,
            }
        )
    return JobState(
        job_id="job-structured-proof",
        operation="await",
        args={"predicate": selected, "timeout_ms": 5_000, "observe": True},
        serial=state.serial,
        owner=state.owner,
        session_id=state.session_id,
        status="succeeded",
        created_ms=1,
        started_ms=2,
        finished_ms=3,
        result={
            "ok": True,
            "action": "await",
            "await_outcome": "satisfied",
            "await_terms": rows,
            "observation": _observation().model_dump(mode="json"),
        },
    )


def test_exact_observation_requires_all_facts_and_persists_provenance(tmp_path: Path) -> None:
    state = _state(tmp_path)

    completed = complete_current_ui_phase_from_observation(
        tmp_path,
        state,
        observation=_observation(),
    )

    proof = completed.phases[0].proof
    assert completed.phases[0].status == "completed"
    assert proof is not None
    assert proof.source == "observation"
    assert [(item.terms, item.expected) for item in proof.satisfied_requirements] == [
        (["loading"], "absent"),
        (["control", "reply"], "enabled"),
        (["example", "panel", "summary"], "present"),
    ]
    assert proof.observation is not None
    assert proof.observation.model_dump() == {
        "fingerprint": "frame-abc123",
        "source": "hierarchy",
        "via": "hierarchy",
        "device_serial": SERIAL,
        "package": "com.example.catalog",
    }

    persisted = load_session_state(tmp_path, session_id=state.session_id)
    assert persisted is not None
    assert persisted.phases[0].proof == proof


def test_visible_loading_keeps_the_explicit_negative_fact_unproven(tmp_path: Path) -> None:
    state = _state(tmp_path)

    unchanged = complete_current_ui_phase_from_observation(
        tmp_path,
        state,
        observation=_observation(loading=True),
    )

    assert unchanged == state
    assert unchanged.phases[0].status == "active"


def test_unrelated_observation_does_not_complete_a_ui_phase(tmp_path: Path) -> None:
    state = _state(tmp_path)
    unrelated = _observation()
    unrelated.elements = [_element(9, "Battery ready")]

    unchanged = complete_current_ui_phase_from_observation(
        tmp_path,
        state,
        observation=unrelated,
    )

    assert unchanged == state


def test_correlated_successful_job_requires_its_predicate_to_cover_every_fact(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    job = _job(state)

    completed = complete_current_ui_phase_from_job(tmp_path, state, job=job)

    proof = completed.phases[0].proof
    assert proof is not None
    assert proof.source == "job_result"
    assert proof.job_id == job.job_id
    assert proof.job_operation == "await"
    assert proof.predicate_terms == [
        "text:Example summary panel",
        "!text:Loading",
        "text:Reply control",
    ]


def test_unrelated_successful_job_cannot_borrow_matching_screen_content(tmp_path: Path) -> None:
    state = _state(tmp_path)
    job = _job(state, predicate="text:Battery ready")

    unchanged = complete_current_ui_phase_from_job(tmp_path, state, job=job)

    assert unchanged == state


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("session_id", "another-session"),
        ("serial", "another-serial"),
        ("owner", "another-agent"),
    ],
)
def test_job_proof_requires_exact_session_device_and_owner_correlation(
    tmp_path: Path,
    field: str,
    wrong: str,
) -> None:
    state = _state(tmp_path)
    job = _job(state).model_copy(update={field: wrong})

    unchanged = complete_current_ui_phase_from_job(tmp_path, state, job=job)

    assert unchanged == state


def test_job_worker_hook_persists_completion_and_refreshes_terminal_progress(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    job = _job(state)

    _complete_correlated_goal_phase(tmp_path, job)

    persisted = load_session_state(tmp_path, session_id=state.session_id)
    assert persisted is not None
    assert persisted.phases[0].status == "completed"
    assert job.result["goal_progress"]["done"] is True


def test_bootstrap_proof_recommends_only_session_finish(tmp_path: Path) -> None:
    engine = Engine(
        make_config(
            cache={"dir": str(tmp_path)},
            memory={"enabled": False, "dir": str(tmp_path / "memory")},
        ),
        device=FakeDevice(serial=SERIAL),
    )

    started = engine.session_start(GOAL, observation=_observation())

    assert started["goal_progress"]["done"] is True
    assert started["goal_progress"]["next_call"] is None
    assert started["recommended_call"]["kind"] == "session_finish"
    assert started["recommended_call"]["cli"] == "aua session finish"
    assert started["recommended_call"]["mcp"] == {
        "tool": "session_finish",
        "arguments": {"session_id": started["session_id"]},
    }
