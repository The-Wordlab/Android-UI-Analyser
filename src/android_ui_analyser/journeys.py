"""Authored instructions for an agent, with explicit assertion or review proof.

A journey is not a program for an LLM hidden inside AUA. Its instructions are delivered to
the calling agent; ordinary AUA actions execute them. SessionState remains the sole owner of
progress, device leases and cleanup. The companion file only preserves the authored brief
and unsuccessful checks for review.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .atomic import atomic_write_text
from .errors import UsageError
from .memory import RouteStep
from .session import GoalPhase, SessionState, phase_progress
from .session_contracts import parse_session_contract_yaml

MAX_JOURNEY_BYTES = 1024 * 1024


class JourneyStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    instruction: str = Field(min_length=1, max_length=4000)
    assertions: list[RouteStep] = Field(default_factory=list)
    review: str | None = Field(default=None, min_length=1, max_length=4000)

    @model_validator(mode="after")
    def _one_proof_kind(self) -> JourneyStep:
        if bool(self.assertions) == bool(self.review):
            raise ValueError("each step needs either assertions or a review question")
        if not self.instruction.strip() or (self.review is not None and not self.review.strip()):
            raise ValueError("instructions and review questions must not be blank")
        return self


class Journey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=8000)
    package: str | None = None
    constraints: list[str] = Field(default_factory=list, max_length=50)
    steps: list[JourneyStep] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _version_and_ids(self) -> Journey:
        if self.version != 1:
            raise ValueError("only journey version 1 is supported")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("journey step IDs must be unique")
        if not self.name.strip():
            raise ValueError("journey name must not be blank")
        return self


def load_journey(*, file: str | None = None, yaml_text: str | None = None) -> Journey:
    if (file is None) == (yaml_text is None):
        raise UsageError("provide exactly one journey file or YAML body")
    if file is not None:
        path = Path(file).expanduser()
        try:
            if path.stat().st_size > MAX_JOURNEY_BYTES:
                raise UsageError("journey file exceeds 1 MiB")
            yaml_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise UsageError(f"cannot read journey file: {exc}") from exc
    assert yaml_text is not None
    if len(yaml_text.encode("utf-8")) > MAX_JOURNEY_BYTES:
        raise UsageError("journey YAML exceeds 1 MiB")
    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise UsageError(f"invalid journey YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise UsageError("journey YAML needs a mapping with a steps list")
    # Reuse the existing authored assertion grammar, never an independent selector parser.
    converted = dict(raw)
    converted["steps"] = []
    for index, item in enumerate(raw["steps"]):
        if not isinstance(item, dict):
            raise UsageError(f"journey step {index + 1} must be a mapping")
        step = dict(item)
        if "assertions" in step:
            contract = parse_session_contract_yaml(yaml.safe_dump({
                "version": 1,
                "checkpoints": [{
                    "id": "proof",
                    "description": "Journey checkpoint",
                    "assertions": step["assertions"],
                }],
            }))
            step["assertions"] = contract.checkpoints[0].assertions
        converted["steps"].append(step)
    try:
        return Journey.model_validate(converted)
    except ValidationError as exc:
        raise UsageError(f"invalid journey: {exc}") from exc


def journey_phases(journey: Journey) -> list[GoalPhase]:
    phases = []
    for step in journey.steps:
        deterministic = bool(step.assertions)
        phases.append(GoalPhase(
            id=step.id,
            objective=step.review or step.instruction,
            intent="contract_checkpoint" if deterministic else "ui_verification",
            satisfaction="fresh_assertions" if deterministic else "relevant_evidence",
            proof_mode="fresh_assertions" if deterministic else "manual_or_structured",
            manual_completion_allowed=not deterministic,
            assertions=step.assertions,
            constraints=journey.constraints,
        ))
    phases[0].status = "active"
    return phases


def companion_path(cache_dir: str | Path, state: SessionState) -> Path:
    # No session ID, platform string or authored name becomes a filesystem component.
    identity = hashlib.sha256(f"{state.platform}:{state.session_id}".encode()).hexdigest()
    return Path(cache_dir).expanduser() / "journeys" / f"{identity}.json"


def save_companion(cache_dir: str | Path, state: SessionState, data: dict[str, Any]) -> None:
    atomic_write_text(companion_path(cache_dir, state), json.dumps(data, ensure_ascii=False, indent=2))


def load_companion(cache_dir: str | Path, state: SessionState) -> dict[str, Any]:
    try:
        data = json.loads(companion_path(cache_dir, state).read_text(encoding="utf-8"))
        if data["session_id"] != state.session_id:
            raise ValueError("session identity differs")
        Journey.model_validate(data["journey"])
        return data
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise UsageError("this session has no valid authored journey") from exc


def journey_report(state: SessionState, data: dict[str, Any]) -> dict[str, Any]:
    journey = Journey.model_validate(data["journey"])
    by_id = {phase.id: phase for phase in state.phases}
    steps = []
    for step in journey.steps:
        phase = by_id[step.id]
        steps.append({
            "id": step.id,
            "instruction": step.instruction,
            "review": step.review,
            "assertions": [assertion.model_dump(mode="json", exclude_none=True) for assertion in step.assertions],
            "proof_kind": "assertions" if step.assertions else "agent_assessment",
            "status": phase.status,
            "verified": bool(phase.proof and phase.proof.verified),
            "evidence": phase.evidence,
            "proof": phase.proof.model_dump(mode="json") if phase.proof else None,
        })
    current = next((step for step in steps if step["status"] != "completed"), None)
    complete = current is None
    verified = complete and all(step["verified"] for step in steps)
    latest = (data.get("attempts") or [None])[-1]
    failed = bool(current and latest and latest["step_id"] == current["id"] and not latest["passed"])
    verdict = (
        "asserted_pass" if verified else "agent_assessed_pass" if complete
        else "failed" if failed and state.finished_ms is None else "incomplete"
    )
    next_call = None
    if current and state.finished_ms is None:
        tool = "journey_check" if current["proof_kind"] == "assertions" else "journey_record"
        next_call = {
            "kind": tool,
            "tool": tool,
            "cli": "aua journey check" if tool == "journey_check" else
            "aua journey record --fingerprint '<observed fingerprint>' --evidence '<observed facts>'",
            "mcp": {"tool": tool, "arguments": {"session_id": state.session_id}},
            "executes": False,
            "instruction": current["instruction"],
            "arguments": {"session_id": state.session_id},
            "reason": (
                "Execute the instruction using AUA actions, then check the authored assertions."
                if tool == "journey_check" else
                "Inspect the current screenshot, then record concrete facts and its fingerprint; "
                "this is an agent assessment, not an automated assertion."
            ),
        }
    elif state.finished_ms is None:
        next_call = {
            "kind": "journey_finish", "tool": "journey_finish", "cli": "aua journey finish",
            "mcp": {"tool": "journey_finish", "arguments": {"session_id": state.session_id}},
            "arguments": {"session_id": state.session_id}, "executes": True,
            "reason": "Every authored step is complete; finish once to restore session-owned state and release the lease.",
        }
    return {
        "name": journey.name,
        "session_id": state.session_id,
        "verdict": verdict,
        "complete": complete,
        "verified": verified,
        "finished": state.finished_ms is not None,
        "steps": steps,
        "current_step": current,
        "next_call": next_call,
        "attempts": data.get("attempts", []),
        "constraints": journey.constraints,
        "execution": "calling_agent",
    }


def journey_progress(state: SessionState, data: dict[str, Any]) -> dict[str, Any]:
    """Keep ordinary action progress aligned with the authored agent instruction."""
    progress = phase_progress(state, compact=True)
    report = journey_report(state, data)
    progress["next_call"] = report["next_call"]
    # Generic --phase-done prose has no observation fingerprint. Authored reviews use
    # journey_record so the assessment remains tied to the frame the agent inspected.
    progress["checkpoint"] = None
    progress["journey_step"] = report["current_step"]
    return progress
