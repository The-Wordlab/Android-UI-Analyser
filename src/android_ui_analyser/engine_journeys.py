"""Agent-executed authored journeys on the existing session/evidence lifecycle."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from .assertions import evaluate_assertion_step
from .errors import UsageError
from .journeys import (
    journey_phases,
    journey_progress,
    journey_report,
    load_companion,
    load_journey,
    save_companion,
)
from .projection import Projection, trim_observation_payload
from .schema import AnalyzeResult
from .session import (
    ObservationProvenance,
    PhaseProof,
    mark_phase_complete,
    update_session_state,
)
from .session_artifacts import observation_evidence_id

if TYPE_CHECKING:
    from .engine import Engine


def journey_validate(
    self: Engine, *, file: str | None = None, yaml: str | None = None,
) -> dict[str, Any]:
    journey = load_journey(file=file, yaml_text=yaml)
    return {
        "ok": True,
        "journey": journey.model_dump(mode="json"),
        "execution": "calling_agent",
        "assertion_steps": sum(bool(step.assertions) for step in journey.steps),
        "review_steps": sum(bool(step.review) for step in journey.steps),
        "note": "Instructions are executed by your agent using AUA; validation runs no actions.",
    }


def journey_start(
    self: Engine,
    *,
    file: str | None = None,
    yaml: str | None = None,
    package: str | None = None,
    apk: str | None = None,
    start_emulator: bool = True,
    artifacts_dir: str | None = None,
    headed: bool = False,
    _bootstrap_new_lease: bool = False,
) -> dict[str, Any]:
    journey = load_journey(file=file, yaml_text=yaml)
    # Validate every authored selector before session bootstrap can provision or launch.
    phases = journey_phases(journey)
    result = cast(Any, self).session_start(
        goal=journey.name,
        package=package or journey.package,
        apk=apk,
        start_emulator=start_emulator,
        artifacts_dir=artifacts_dir,
        evidence="all" if artifacts_dir else "failures",
        headed=headed,
        _bootstrap_new_lease=_bootstrap_new_lease,
    )
    if result.get("ok") is False:
        return result
    state = self._session_state()
    state = update_session_state(self.config.cache.dir, state, phases=phases)
    data = {
        "session_id": state.session_id,
        "journey": journey.model_dump(mode="json"),
        "attempts": [],
    }
    save_companion(self.config.cache.dir, state, data)
    result["goal_progress"] = journey_progress(state, data)
    result["journey"] = journey_report(state, data)
    result["recommended_call"] = result["journey"]["next_call"]
    result["cleanup_call"] = {
        "cli": "aua journey finish",
        "mcp": {"tool": "journey_finish", "arguments": {"session_id": state.session_id}},
        "reason": "Finish the authored journey to collect its verdict and restore session-owned state.",
    }
    # Bootstrap advice was inferred from the title before the exact authored phases existed.
    for key in ("policy", "policy_suggestion", "policy_handoff"):
        result.pop(key, None)
    return result


def journey_status(self: Engine, session_id: str | None = None) -> dict[str, Any]:
    state = self._session_state(session_id)
    data = load_companion(self.config.cache.dir, state)
    return {"ok": True, "journey": journey_report(state, data), "goal_progress": journey_progress(state, data)}


def _current(self: Engine, session_id: str | None) -> tuple[Any, dict[str, Any], Any]:
    state = self._session_state(session_id)
    _require_owner(self, state)
    data = load_companion(self.config.cache.dir, state)
    if state.finished_ms is not None:
        raise UsageError("this journey session is already finished")
    current = next((phase for phase in state.phases if phase.status != "completed"), None)
    if current is None:
        raise UsageError("all journey steps are complete; finish the journey")
    return state, data, current


def _require_owner(self: Engine, state: Any) -> None:
    if state.owner and state.owner != self._ledger_identity()["owner"]:
        raise UsageError("this journey belongs to another worker; use its original AUA worker scope")


def _validate_frame(state: Any, frame: AnalyzeResult | None) -> AnalyzeResult:
    if frame is None or not frame.meta.fingerprint:
        raise UsageError("journey evidence needs a fingerprinted AUA observation")
    if frame.meta.stale_risk or frame.meta.device_serial != state.serial:
        raise UsageError("journey evidence must be a non-stale frame from this session's device")
    return frame


def _attempt(
    self: Engine, state: Any, data: dict[str, Any], *, phase_id: str,
    passed: bool, fingerprint: str, kind: str, detail: Any,
) -> None:
    data["attempts"].append({
        "step_id": phase_id,
        "passed": passed,
        "fingerprint": fingerprint,
        "kind": kind,
        "detail": detail,
        "recorded_ms": int(time.time() * 1000),
    })
    save_companion(self.config.cache.dir, state, data)


def journey_check(self: Engine, session_id: str | None = None) -> dict[str, Any]:
    state, data, current = _current(self, session_id)
    if not current.assertions:
        raise UsageError("this step needs an agent review; use journey record with the observed fingerprint")
    frame = _validate_frame(state, self.analyze(source="hierarchy", with_ocr=False, no_cache=True))
    results = []
    for assertion in current.assertions:
        verdict = evaluate_assertion_step(assertion, frame.elements)
        results.append({"kind": assertion.kind, "ok": verdict.ok, "detail": verdict.detail})
    passed = all(result["ok"] for result in results)
    fingerprint = str(frame.meta.fingerprint)
    if passed:
        evidence_id = observation_evidence_id(state.session_id, frame.model_dump(mode="json"))
        proof = PhaseProof(
            source="contract_assertions",
            command="contract_assertions",
            verified=True,
            observation=ObservationProvenance(
                fingerprint=fingerprint,
                source=frame.screen.source.value,
                via=frame.meta.via,
                device_serial=state.serial,
                package=frame.screen.package or "unknown",
            ),
            evidence_id=evidence_id,
            assertions_verified=len(results),
        )
        try:
            state = mark_phase_complete(
                self.config.cache.dir, state, phase_id=current.id,
                evidence=f"All {len(results)} authored assertions passed on {fingerprint}", _proof=proof,
            )
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
    _attempt(self, state, data, phase_id=current.id, passed=passed, fingerprint=fingerprint,
             kind="assertions", detail=results)
    payload = {
        "ok": passed,
        "assertions": results,
        "observation": frame.as_dict("json"),
        "journey": journey_report(state, data),
        "goal_progress": journey_progress(state, data),
    }
    view = Projection.for_observation(
        self.config.output.observation_fields, meta=self.config.output.observation_meta,
    )
    return trim_observation_payload(payload, view)


def journey_record(
    self: Engine,
    evidence: str,
    fingerprint: str,
    *,
    passed: bool = True,
    session_id: str | None = None,
) -> dict[str, Any]:
    state, data, current = _current(self, session_id)
    if current.assertions:
        raise UsageError("manual claims cannot complete assertion steps; use journey check")
    if not isinstance(evidence, str) or not evidence.strip():
        raise UsageError("record concrete observed facts, not an empty assessment")
    # Never attach a claim about a previous image to a new capture taken behind the agent's back.
    try:
        cached = AnalyzeResult.model_validate_json(self._cache_path(state.serial).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UsageError("journey review needs the saved observation the agent inspected") from exc
    frame = _validate_frame(state, cached)
    if not fingerprint or fingerprint != frame.meta.fingerprint:
        raise UsageError("assessment fingerprint does not match the current AUA observation")
    if passed:
        # Existing manual-proof validation requires phase-specific observable facts. Its
        # resulting proof remains unverified; no model assessment becomes machine evidence.
        try:
            state = mark_phase_complete(
                self.config.cache.dir, state, phase_id=current.id, evidence=evidence,
            )
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        phases = [phase.model_copy(deep=True) for phase in state.phases]
        phase = next(phase for phase in phases if phase.id == current.id)
        assert phase.proof is not None
        phase.proof.observation = ObservationProvenance(
            fingerprint=fingerprint,
            source=frame.screen.source.value, via=frame.meta.via,
            device_serial=state.serial, package=frame.screen.package or "unknown",
        )
        phase.proof.evidence_id = observation_evidence_id(state.session_id, frame.model_dump(mode="json"))
        phase.proof.verified = False
        state = update_session_state(self.config.cache.dir, state, phases=phases)
    _attempt(self, state, data, phase_id=current.id, passed=passed, fingerprint=fingerprint,
             kind="agent_assessment", detail=evidence.strip())
    result = {
        "ok": passed,
        "verified": False,
        "assessment": "agent_assessment",
        "fingerprint": fingerprint,
        "journey": journey_report(state, data),
        "goal_progress": journey_progress(state, data),
    }
    if state.artifact_dir:
        result["artifacts_dir"] = state.artifact_dir
    return result


def journey_finish(
    self: Engine,
    session_id: str | None = None,
    *,
    allow_incomplete: bool = False,
    retain_started_target: bool = True,
) -> dict[str, Any]:
    state = self._session_state(session_id)
    _require_owner(self, state)
    data = load_companion(self.config.cache.dir, state)
    report = journey_report(state, data)
    if report["finished"]:
        return {
            "ok": True, "finished": True, "terminated": True,
            "verdict": report["verdict"], "verified": report["verified"],
            "journey": report, "cleanup": [],
        }
    if not report["complete"] and not allow_incomplete:
        return {"ok": False, "finished": False, "code": "journey_incomplete", "journey": report}
    result = self.session_finish(
        state.session_id, allow_incomplete=allow_incomplete,
        retain_started_target=retain_started_target,
    )
    state = self._session_state(state.session_id)
    report = journey_report(state, data)
    result["journey"] = report
    result["finished"] = report["finished"]
    result["verified"] = bool(report["verified"] and result.get("ok"))
    if not report["complete"]:
        result["verdict"] = "incomplete"
    elif result.get("ok"):
        result["verdict"] = report["verdict"]
    return result
