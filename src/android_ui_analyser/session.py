"""Goal-aware session bootstrap and safe navigation recommendations.

This module is deliberately interface agnostic.  CLI and MCP adapters both consume the
same typed plan, while :class:`~android_ui_analyser.engine.Engine` supplies the one screen
observation and the local memory records.  Planning is pure: it never connects to a device
or executes a remembered action.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .atomic import atomic_write_text
from .flows import Flow
from .memory import (
    AppMap,
    RouteEdge,
    _shortest_path,
    is_destructive_step,
    resolve_goal,
    route_step_risks,
    step_display,
)
from .schema import AnalyzeResult
from .selectors import is_back_resource_id


class GoalCall(BaseModel):
    """One exact next call, expressed for both supported agent interfaces."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    cli: str
    mcp: dict[str, Any]
    reason: str
    candidate_id: str | None = None
    executes: bool = True


class GoalCandidate(BaseModel):
    """A ranked navigation option and the evidence/risk behind it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["goto", "flow", "deeplink", "arrived"]
    name: str
    target: str | None = None
    score: int = 0
    safe: bool
    status: str
    risks: list[dict[str, str]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    call: GoalCall


class GoalSessionPlan(BaseModel):
    """Structured result returned by ``session start``."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    goal: str
    package: str | None = None
    current_screen: str | None = None
    observation_note: str = (
        "This is the current settled screen. Reuse it; do not call analyze next."
    )
    observation: AnalyzeResult
    candidates: list[GoalCandidate] = Field(default_factory=list)
    selected_candidate: str | None = None
    recommended_call: GoalCall
    warnings: list[str] = Field(default_factory=list)
    relevant_capabilities: list[dict[str, Any]] = Field(default_factory=list)


class GoalPhase(BaseModel):
    """One durable, ordered checkpoint in an end-to-end verification goal."""

    model_config = ConfigDict(extra="forbid")

    id: str
    objective: str
    kind: Literal["environment", "verify", "cleanup"] = "verify"
    status: Literal["pending", "active", "completed"] = "pending"
    completed_ms: int | None = None
    evidence: str | None = None
    recommended_call: dict[str, Any] | None = None


class SessionState(BaseModel):
    """Small persisted owner/device scope for review and reversible cleanup."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    goal: str
    goal_hash: str
    serial: str
    owner: str | None = None
    started_ms: int
    recommended_kind: str
    recommended_cli: str
    network_backup_preexisting: bool = False
    network_profile_preexisting: bool = False
    emulator_started: bool = False
    phases: list[GoalPhase] = Field(default_factory=list)
    finished_ms: int | None = None


_SEQUENCE_BOUNDARY = re.compile(
    r"\s*(?:[.;]\s+|\bthen\b|\bafter that\b|\bnext\b|\bfinally\b|\bcompare\b)",
    re.IGNORECASE,
)
_RESTORE_GOAL = re.compile(
    r"\b(?:restore|re-enable|reconnect)\b.{0,48}\b(?:network|connectivity|wi-?fi|internet)\b",
    re.IGNORECASE,
)
_OFFLINE_GOAL = re.compile(
    r"\b(?:(?:go|switch|work|test|verify)\s+)?(?:fully\s+)?offline\b(?:\s+(?:mode|state))?"
    r"|\bairplane mode\b",
    re.IGNORECASE,
)


def goal_phases(goal: str) -> list[GoalPhase]:
    """Extract conservative ordered checkpoints without pretending to understand the app.

    Only explicit sequence markers split the user's prose. Ordinary ``and`` stays inside one
    checkpoint, so "open and verify" is not turned into two invented tasks. Offline setup and
    connectivity cleanup are deterministic environment phases; UI checkpoints remain explicit
    evidence contracts owned by the agent using the app.
    """
    cleaned = " ".join(goal.strip().split())
    segments = [part.strip(" ,") for part in _SEQUENCE_BOUNDARY.split(cleaned) if part.strip(" ,")]
    phases: list[GoalPhase] = []
    offline_added = False
    cleanup_added = False
    for segment in segments or [cleaned]:
        cleanup_here = bool(_RESTORE_GOAL.search(segment))
        objective = _RESTORE_GOAL.sub("", segment).strip(" ,;.")
        offline_here = bool(_OFFLINE_GOAL.search(objective))
        if offline_here and not offline_added:
            phases.append(
                GoalPhase(
                    id=f"phase_{len(phases) + 1}",
                    objective="Establish and verify the requested offline network state",
                    kind="environment",
                )
            )
            offline_added = True
        if offline_here:
            transition = _OFFLINE_GOAL.search(objective)
            prefix = transition.group(0).casefold() if transition is not None else ""
            if transition is not None and transition.start() == 0 and (
                prefix.startswith(("go ", "switch ", "airplane "))
                or prefix.startswith(("offline", "fully offline"))
            ):
                objective = _OFFLINE_GOAL.sub("", objective, count=1).strip(" ,;.")
                objective = re.sub(r"^(?:and|before)\s+", "", objective, flags=re.IGNORECASE)
        if objective and objective.casefold() not in {"and", "before finishing"}:
            phases.append(
                GoalPhase(
                    id=f"phase_{len(phases) + 1}",
                    objective=objective,
                    kind="verify",
                )
            )
        if cleanup_here and not cleanup_added:
            phases.append(
                GoalPhase(
                    id=f"phase_{len(phases) + 1}",
                    objective="Restore the session-owned network state",
                    kind="cleanup",
                )
            )
            cleanup_added = True
    if not phases:
        phases.append(GoalPhase(id="phase_1", objective=cleaned, kind="verify"))
    phases[0].status = "active"
    return phases


def phase_progress(state: SessionState, *, compact: bool = False) -> dict[str, Any]:
    """Return goal progress; ordinary results use the compact non-duplicating form."""
    phases = state.phases or goal_phases(state.goal)
    current = next((phase for phase in phases if phase.status != "completed"), None)
    completed = sum(phase.status == "completed" for phase in phases)
    current_payload = current.model_dump(mode="json") if current is not None else None
    if compact and isinstance(current_payload, dict):
        current_payload.pop("recommended_call", None)
    payload: dict[str, Any] = {
        "session_id": state.session_id,
        "completed": completed,
        "total": len(phases),
        "done": current is None,
        "current": current_payload,
        "next_call": current.recommended_call if current is not None else None,
        "checkpoint": (
            None
            if current is None
            else {
                "cli": f"--phase-done {current.id}=\"<evidence from the current result>\"",
                "mcp": {"phase_done": {"id": current.id, "evidence": "<evidence>"}},
                "note": (
                    "Acknowledge this phase on the next AUA call; it adds no extra round trip."
                ),
            }
        ),
    }
    if compact:
        payload["upcoming"] = [
            {"id": phase.id, "objective": phase.objective, "kind": phase.kind}
            for phase in phases
            if phase.status == "pending"
        ]
    else:
        payload["phases"] = [phase.model_dump(mode="json") for phase in phases]
    return payload


def mark_phase_complete(
    cache_dir: str | Path,
    state: SessionState,
    *,
    phase_id: str,
    evidence: str,
) -> SessionState:
    """Complete only the current phase, preserving ordered goal semantics."""
    evidence = " ".join(evidence.strip().split())
    if not evidence:
        raise ValueError("phase evidence must not be empty")
    phases = [phase.model_copy(deep=True) for phase in (state.phases or goal_phases(state.goal))]
    current_index = next(
        (index for index, phase in enumerate(phases) if phase.status != "completed"), None
    )
    if current_index is None:
        raise ValueError("all goal phases are already complete")
    current = phases[current_index]
    if current.id != phase_id:
        raise ValueError(
            f"{phase_id!r} is not the current goal phase; complete {current.id!r} first"
        )
    current.status = "completed"
    current.completed_ms = int(time.time() * 1000)
    current.evidence = evidence[:600]
    if current_index + 1 < len(phases):
        phases[current_index + 1].status = "active"
    updated = state.model_copy(update={"phases": phases})
    _write_state(cache_dir, updated)
    return updated


def update_phase_recommendation(
    cache_dir: str | Path,
    state: SessionState,
    *,
    phase_id: str,
    call: dict[str, Any],
) -> SessionState:
    """Persist a fresh-screen next call for one incomplete phase."""
    phases = [phase.model_copy(deep=True) for phase in (state.phases or goal_phases(state.goal))]
    phase = next((item for item in phases if item.id == phase_id), None)
    if phase is None or phase.status == "completed":
        return state
    phase.recommended_call = call
    updated = state.model_copy(update={"phases": phases})
    _write_state(cache_dir, updated)
    return updated


def complete_environment_phase(
    cache_dir: str | Path,
    state: SessionState,
    *,
    command: str,
    result: dict[str, Any],
) -> SessionState:
    """Advance deterministic offline/cleanup phases from verified tool evidence."""
    phases = state.phases or goal_phases(state.goal)
    current = next((phase for phase in phases if phase.status != "completed"), None)
    if current is None:
        return state
    if current.kind == "environment" and command == "network_offline":
        network_value = result.get("state")
        network_state: dict[str, Any] = network_value if isinstance(network_value, dict) else {}
        if result.get("ok") and result.get("verified") and network_state.get("offline") is True:
            return mark_phase_complete(
                cache_dir,
                state,
                phase_id=current.id,
                evidence="AUA verified no active default network and offline=true",
            )
    return state


def _safe_token(value: str) -> str:
    readable = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return readable[:80] or "unknown"


def _session_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "sessions"


def _session_path(cache_dir: str | Path, session_id: str) -> Path:
    return _session_dir(cache_dir) / f"{_safe_token(session_id)}.json"


def _active_path(cache_dir: str | Path, serial: str, owner: str | None) -> Path:
    identity = hashlib.sha256((owner or "anonymous").encode()).hexdigest()[:12]
    return _session_dir(cache_dir) / f"active-{_safe_token(serial)}-{identity}.txt"


def _write_state(cache_dir: str | Path, state: SessionState) -> None:
    path = _session_path(cache_dir, state.session_id)
    atomic_write_text(path, state.model_dump_json(indent=2))
    if state.finished_ms is None:
        atomic_write_text(_active_path(cache_dir, state.serial, state.owner), state.session_id)


def create_session_state(
    cache_dir: str | Path,
    *,
    goal: str,
    serial: str,
    owner: str | None,
    recommended_kind: str,
    recommended_cli: str,
    network_backup_preexisting: bool,
    network_profile_preexisting: bool,
    emulator_started: bool = False,
) -> SessionState:
    state = SessionState(
        session_id=uuid.uuid4().hex,
        goal=goal,
        goal_hash=hashlib.sha256(goal.encode()).hexdigest()[:16],
        serial=serial,
        owner=owner,
        started_ms=int(time.time() * 1000),
        recommended_kind=recommended_kind,
        recommended_cli=recommended_cli,
        network_backup_preexisting=network_backup_preexisting,
        network_profile_preexisting=network_profile_preexisting,
        emulator_started=emulator_started,
        phases=goal_phases(goal),
    )
    _write_state(cache_dir, state)
    return state


def load_session_state(
    cache_dir: str | Path,
    *,
    session_id: str | None = None,
    serial: str | None = None,
    owner: str | None = None,
) -> SessionState | None:
    if session_id is None:
        if not serial:
            return None
        pointer = _active_path(cache_dir, serial, owner)
        try:
            session_id = pointer.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    try:
        payload = json.loads(_session_path(cache_dir, session_id).read_text(encoding="utf-8"))
        return SessionState.model_validate(payload)
    except (OSError, ValueError, TypeError):
        return None


def active_session_metadata(
    cache_dir: str | Path, serial: str | None, owner: str | None
) -> dict[str, str]:
    """Return journal-safe correlation fields without exposing the natural-language goal."""
    if not serial:
        return {}
    state = load_session_state(cache_dir, serial=serial, owner=owner)
    if state is None or state.finished_ms is not None:
        return {}
    return {"session_id": state.session_id, "goal_hash": state.goal_hash}


def finish_session_state(cache_dir: str | Path, state: SessionState) -> SessionState:
    phases = [phase.model_copy(deep=True) for phase in (state.phases or goal_phases(state.goal))]
    now_ms = int(time.time() * 1000)
    for phase in phases:
        if phase.kind == "cleanup" and phase.status != "completed":
            phase.status = "completed"
            phase.completed_ms = now_ms
            phase.evidence = "session finish restored session-owned reversible state"
    finished = state.model_copy(update={"finished_ms": now_ms, "phases": phases})
    _write_state(cache_dir, finished)
    pointer = _active_path(cache_dir, state.serial, state.owner)
    try:
        if pointer.read_text(encoding="utf-8").strip() == state.session_id:
            pointer.unlink(missing_ok=True)
    except OSError:
        pass
    return finished


_ACTION_COMMANDS = frozenset(
    {
        "tap",
        "long_press",
        "tap_point",
        "input",
        "input_text",
        "clear",
        "swipe",
        "scroll",
        "scroll_to",
        "key",
        "open_link",
    }
)
_WAIT_COMMANDS = frozenset(
    {"wait", "wait_stable", "wait_changed", "wait_after_change", "await_predicate", "await"}
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _base_command(value: Any) -> str:
    command = str(value or "").removesuffix("_and_analyze")
    aliases = {"input_text": "input", "analyze_screen": "analyze"}
    return aliases.get(command, command)


def review_session_events(state: SessionState, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Classify avoidable call patterns without conflating concurrent owners."""
    scoped = [
        event
        for event in events
        if (event.get("session_id") or (event.get("extra") or {}).get("session_id"))
        == state.session_id
    ]
    counts: dict[str, int] = {}
    avoidable: dict[str, list[dict[str, Any]]] = {
        "redundant_analyze": [],
        "wait_after_observed_action": [],
        "consecutive_has": [],
        "manual_path": [],
        "deeplink_over_verified_navigation": [],
        "airplane_for_offline": [],
        "predicate_timeout": [],
        "repeated_back": [],
        "reused_numeric_id": [],
        "ambiguous_invocation": [],
    }
    duration_ms = sum(
        float(event["duration_ms"])
        for event in scoped
        if isinstance(event.get("duration_ms"), (int, float))
    )
    manual_streak: list[dict[str, Any]] = []
    back_streak: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    top_level: list[dict[str, Any]] = []
    invocation_indexes: dict[str, int] = {}
    ambiguous_invocations: dict[str, dict[str, Any]] = {}
    for event in scoped:
        args_value = event.get("args")
        args: dict[str, Any] = args_value if isinstance(args_value, dict) else {}
        result_value = event.get("result")
        result: dict[str, Any] = result_value if isinstance(result_value, dict) else {}
        if _base_command(event.get("cmd")) in _WAIT_COMMANDS and args.get("adopt_action") is True:
            if str(result.get("detail") or "").startswith("timeout after"):
                avoidable["predicate_timeout"].append(
                    {
                        "ts_ms": event.get("ts_ms"),
                        "predicate": args.get("predicate"),
                        "duration_ms": event.get("duration_ms"),
                    }
                )
            continue
        invocation_id = event.get("invocation_id") or (event.get("extra") or {}).get(
            "invocation_id"
        )
        if isinstance(invocation_id, str):
            if invocation_id in invocation_indexes:
                # Historical daemons could time out after executing a mutation, after which
                # `_route` replayed it in-process under the same invocation id. Journal order
                # cannot reveal which response the caller saw: the daemon's delayed success may
                # be written after the CLI failure that was actually returned. Treat that call
                # as unknown, not as success/failure, and never use either observation to accuse
                # the caller's next analyze of being redundant.
                index = invocation_indexes[invocation_id]
                first = top_level[index]
                row = ambiguous_invocations.setdefault(
                    invocation_id,
                    {
                        "invocation_id": invocation_id,
                        "cmd": first.get("cmd"),
                        "outcomes": [bool(first.get("ok"))],
                    },
                )
                outcomes = row["outcomes"]
                if isinstance(outcomes, list):
                    outcomes.append(bool(event.get("ok")))
                top_level[index] = {
                    **first,
                    "ok": True,
                    "result": {},
                    "error": None,
                    "ambiguous_invocation": True,
                }
                continue
            invocation_indexes[invocation_id] = len(top_level)
        top_level.append(event)

    avoidable["ambiguous_invocation"] = list(ambiguous_invocations.values())

    for event in top_level:
        cmd = str(event.get("cmd") or "?")
        base = _base_command(cmd)
        counts[cmd] = counts.get(cmd, 0) + 1
        prior_result = (previous or {}).get("result")
        prior_observed = isinstance(prior_result, dict) and bool(prior_result.get("observation"))
        if base == "analyze" and prior_observed:
            avoidable["redundant_analyze"].append(
                {"ts_ms": event.get("ts_ms"), "after": (previous or {}).get("cmd")}
            )
        if base in _WAIT_COMMANDS and prior_observed:
            avoidable["wait_after_observed_action"].append(
                {"ts_ms": event.get("ts_ms"), "after": (previous or {}).get("cmd")}
            )
        if base == "has" and previous is not None and _base_command(previous.get("cmd")) == "has":
            avoidable["consecutive_has"].append({"ts_ms": event.get("ts_ms")})
        if base in {"open_link", "open"} and state.recommended_kind in {"goto", "flow"}:
            avoidable["deeplink_over_verified_navigation"].append({"ts_ms": event.get("ts_ms")})
        if base.startswith("airplane") and "offline" in state.goal.casefold():
            avoidable["airplane_for_offline"].append({"ts_ms": event.get("ts_ms")})
        event_args = _mapping(event.get("args"))
        previous_args = _mapping(previous.get("args") if previous is not None else None)
        event_result = _mapping(event.get("result"))
        previous_result = _mapping(previous.get("result") if previous is not None else None)

        def observed_screen(value: dict[str, Any]) -> str | None:
            observation = value.get("observation")
            if not isinstance(observation, dict):
                return None
            meta = observation.get("meta")
            if isinstance(meta, str):
                return meta
            if isinstance(meta, dict) and meta.get("known_screen"):
                return str(meta["known_screen"])
            return None

        numeric_id = event_args.get("element_id")
        previous_numeric_id = previous_args.get("element_id")
        if (
            base == "tap"
            and previous is not None
            and _base_command(previous.get("cmd")) == "tap"
            and isinstance(numeric_id, int)
            and numeric_id == previous_numeric_id
            and observed_screen(event_result)
            and observed_screen(event_result) != observed_screen(previous_result)
        ):
            avoidable["reused_numeric_id"].append(
                {
                    "ts_ms": event.get("ts_ms"),
                    "element_id": numeric_id,
                    "from_screen": observed_screen(previous_result),
                    "to_screen": observed_screen(event_result),
                }
            )
        selector = _mapping(event_args.get("selector"))
        semantic_selector = (
            is_back_resource_id(str(selector.get("rid") or ""))
            or str(selector.get("desc") or "").casefold() in {"back", "navigate up", "up"}
            or str(selector.get("text") or "").casefold() == "back"
        )
        is_back = (base == "key" and str(event_args.get("name")).casefold() == "back") or (
            base == "tap" and semantic_selector
        )
        if is_back:
            back_streak.append(event)
        elif base in _ACTION_COMMANDS:
            if len(back_streak) >= 2:
                avoidable["repeated_back"].append(
                    {"calls": len(back_streak), "from_ms": back_streak[0].get("ts_ms")}
                )
            back_streak = []
        if base in _ACTION_COMMANDS:
            manual_streak.append(event)
        else:
            if len(manual_streak) >= 4:
                avoidable["manual_path"].append(
                    {"calls": len(manual_streak), "from_ms": manual_streak[0].get("ts_ms")}
                )
            manual_streak = []
        previous = event
    if len(manual_streak) >= 4:
        avoidable["manual_path"].append(
            {"calls": len(manual_streak), "from_ms": manual_streak[0].get("ts_ms")}
        )
    if len(back_streak) >= 2:
        avoidable["repeated_back"].append(
            {"calls": len(back_streak), "from_ms": back_streak[0].get("ts_ms")}
        )

    patterns = {name: rows for name, rows in avoidable.items() if rows}
    avoidable_calls = sum(
        len(patterns.get(name, []))
        for name in ("redundant_analyze", "wait_after_observed_action", "consecutive_has")
    )
    manual_savings = sum(max(1, int(row["calls"]) - 1) for row in patterns.get("manual_path", []))
    back_savings = sum(max(1, int(row["calls"]) - 1) for row in patterns.get("repeated_back", []))
    # A repeated Back streak is normally contained in the same manual journey. Do not claim
    # both a flow saving and a back-until saving for the same top-level calls.
    avoidable_calls += manual_savings or back_savings
    advice: list[dict[str, str]] = []
    if patterns.get("redundant_analyze"):
        advice.append(
            {
                "id": "reuse_observation",
                "recommended_call": "Use the action response's observation.",
            }
        )
    if patterns.get("wait_after_observed_action"):
        advice.append(
            {"id": "fold_until", "recommended_call": "Put --until on the analyzed action."}
        )
    if patterns.get("consecutive_has"):
        advice.append(
            {
                "id": "combine_assertions",
                "recommended_call": "Use await-and-analyze with comma-separated terms or a suite.",
            }
        )
    if patterns.get("manual_path"):
        advice.append({"id": "save_flow", "recommended_call": "aua flow save <name>"})
    if patterns.get("deeplink_over_verified_navigation"):
        advice.append(
            {"id": "prefer_verified_navigation", "recommended_call": state.recommended_cli}
        )
    if patterns.get("airplane_for_offline"):
        advice.append(
            {"id": "verified_offline", "recommended_call": "aua network offline --verify"}
        )
    if patterns.get("predicate_timeout"):
        advice.append(
            {
                "id": "exact_arrival_predicate",
                "recommended_call": (
                    "Use the exact destination label or rid returned by the prior observation."
                ),
            }
        )
    if patterns.get("repeated_back"):
        advice.append(
            {
                "id": "bounded_back_navigation",
                "recommended_call": "aua back-until-and-analyze '<known_screen>'",
            }
        )
    if patterns.get("reused_numeric_id"):
        advice.append(
            {
                "id": "do_not_reuse_frame_id",
                "recommended_call": (
                    "Use a fresh --rid/stable_key; for nested Back use "
                    "aua back-until-and-analyze '<known_screen>'."
                ),
            }
        )
    if patterns.get("ambiguous_invocation"):
        advice.append(
            {
                "id": "daemon_outcome_unknown",
                "recommended_call": (
                    "Do not infer or replay the action; inspect one fresh screen after the "
                    "daemon responds."
                ),
            }
        )
    high_level = sum(
        counts.get(name, 0) for name in ("session_start", "reach", "goto", "flow_run", "back_until")
    )
    failures = sum(
        1
        for event in top_level
        if not event.get("ok") and _base_command(event.get("cmd")) != "session_review"
    )
    return {
        # The review command succeeded even when the run being reviewed contained recoverable
        # failures. Keeping those two meanings in one `ok` made the review journal itself as a
        # failure, so every later review looked worse merely because an agent inspected it.
        "ok": True,
        "run_ok": None if ambiguous_invocations and failures == 0 else failures == 0,
        "session_id": state.session_id,
        "goal_hash": state.goal_hash,
        "serial": state.serial,
        "started_ms": state.started_ms,
        "finished_ms": state.finished_ms,
        "calls": len(top_level),
        "engine_events": len(scoped),
        "failures": failures,
        "duration_ms": round(duration_ms, 1),
        "commands": counts,
        "high_level_navigation_ratio": (
            round(high_level / len(top_level), 3) if top_level else 0.0
        ),
        "avoidable_calls": avoidable_calls,
        "estimated_calls_saved_next_run": avoidable_calls,
        "patterns": patterns,
        "advice": advice,
    }


_GOAL_WORD = re.compile(r"[a-z0-9]+")
_LOW_SIGNAL = frozenset(
    {
        "a",
        "an",
        "and",
        "android",
        "app",
        "check",
        "do",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "open",
        "reach",
        "screen",
        "test",
        "that",
        "the",
        "this",
        "to",
        "use",
        "verify",
        "navigate",
        "or",
        "then",
        "with",
    }
)


def _goal_terms(goal: str) -> list[str]:
    terms = [word for word in _GOAL_WORD.findall(goal.casefold()) if word not in _LOW_SIGNAL]
    return terms or _GOAL_WORD.findall(goal.casefold())


def _match_score(goal: str, *values: str | None) -> int:
    haystack = " ".join(value for value in values if value).casefold()
    if not haystack:
        return 0
    phrase = goal.casefold().strip()
    terms = _goal_terms(goal)
    haystack_terms = set(_GOAL_WORD.findall(haystack))
    matched = [term for term in terms if term in haystack_terms]
    score = 5 * len(matched)
    if phrase and phrase in haystack:
        score += 40
    if terms and len(matched) == len(terms):
        score += 20
    return score


def _edge_risks(
    path: Sequence[RouteEdge], *, package: str | None, destructive_labels: Sequence[str]
) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    for edge_index, edge in enumerate(path):
        if not edge.steps:
            risks.append(
                {
                    "code": "legacy_route",
                    "reason": "route has no inspectable structured steps",
                    "path": f"route[{edge_index}]",
                }
            )
            continue
        for step_index, step in enumerate(edge.steps):
            for item in route_step_risks(
                step,
                origin_package=package,
                destructive_labels=destructive_labels,
                path=f"route[{edge_index}].steps[{step_index}]",
            ):
                if item not in risks:
                    risks.append(item)
    return risks


def _flow_risks(
    flow: Flow, *, package: str | None, destructive_labels: Sequence[str]
) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    for index, step in enumerate(flow.steps):
        for item in route_step_risks(
            step,
            origin_package=flow.app or package,
            destructive_labels=destructive_labels,
            path=f"steps[{index}]",
        ):
            if item not in risks:
                risks.append(item)
        # Keep the destructive fact explicit even if a future risk classifier treats the
        # step as otherwise safe.
        if is_destructive_step(step, destructive_labels):
            item = {
                "code": "destructive",
                "reason": "label matches the configured destructive-action vocabulary",
                "path": f"steps[{index}]",
            }
            if item not in risks:
                risks.append(item)
    required = sorted(name for name, default in flow.params.items() if not default)
    if required:
        risks.append(
            {
                "code": "required_params",
                "reason": "flow requires parameter values: " + ", ".join(required),
                "path": "params",
            }
        )
    return risks


def _goto_candidate(
    goal: str,
    observation: AnalyzeResult,
    app: AppMap,
    *,
    context_id: str,
    current_screen: str | None,
    destructive_labels: Sequence[str],
) -> GoalCandidate | None:
    target = resolve_goal(app, goal, start=current_screen, context_id=context_id)
    if target is None:
        # Natural goals commonly start with an action verb ("open saved items"), while the
        # map correctly names the destination only "saved_items".  Retry with low-signal
        # orchestration words removed; target resolution and context checks remain canonical.
        destination_terms = " ".join(_goal_terms(goal))
        if destination_terms and destination_terms != goal.casefold().strip():
            target = resolve_goal(
                app,
                destination_terms,
                start=current_screen,
                context_id=context_id,
            )
    if target is None:
        return None
    quoted = shlex.quote(goal)
    if current_screen == target:
        call = GoalCall(
            kind="arrived",
            cli=f"aua reach {quoted}",
            mcp={"tool": "reach", "arguments": {"goal": goal}},
            reason=f"The one observation already identifies the destination as {target}.",
        )
        return GoalCandidate(
            id=f"arrived:{target}",
            kind="arrived",
            name=target,
            target=target,
            score=10_000,
            safe=True,
            status="already_there",
            evidence={"known_screen": observation.meta.known_screen},
            call=call,
        )
    path = _shortest_path(app, target, start=current_screen, context_id=context_id)
    if not path:
        return None
    risks = _edge_risks(path, package=app.package, destructive_labels=destructive_labels)
    safe = not risks
    call = GoalCall(
        kind="goto" if safe else "goto_plan",
        cli=f"aua goto {quoted}" + (" --plan" if not safe else ""),
        mcp={
            "tool": "goto",
            "arguments": {"goal": goal, **({"plan": True} if not safe else {})},
        },
        reason=(
            "A verified route is available in the active app-map context."
            if safe
            else "Review the full route risk preview before authorizing any side effect."
        ),
        executes=safe,
    )
    return GoalCandidate(
        id=f"goto:{target}",
        kind="goto",
        name=target,
        target=target,
        score=8_000 - len(path),
        safe=safe,
        status="verified" if safe else "requires_review",
        risks=risks,
        evidence={
            "context_id": context_id,
            "route": [
                {
                    "from": edge.from_screen,
                    "to": edge.to_screen,
                    "steps": [step_display(step) for step in edge.steps],
                    "status": edge.status,
                }
                for edge in path
            ],
        },
        call=call,
    )


def _flow_candidates(
    goal: str,
    observation: AnalyzeResult,
    flows: Iterable[Flow],
    *,
    destructive_labels: Sequence[str],
) -> list[GoalCandidate]:
    out: list[GoalCandidate] = []
    for flow in flows:
        if flow.app not in (None, observation.screen.package):
            continue
        score = _match_score(
            goal,
            flow.name,
            flow.description,
            flow.arrival,
            *flow.aliases,
        )
        # One incidental shared word is not intent.  In a multi-part test goal it made a
        # destructive account-reset flow outrank the visible controls merely because both
        # descriptions said "online".  A single-term goal still scores 25 (token + all-terms),
        # while longer goals need several terms or an exact phrase.
        if score < 20:
            continue
        risks = _flow_risks(
            flow,
            package=observation.screen.package,
            destructive_labels=destructive_labels,
        )
        safe = not risks
        cli_name = shlex.quote(flow.name)
        call = GoalCall(
            kind="flow" if safe else "flow_preview",
            cli=f"aua flow run {cli_name}" + (" --dry-run" if not safe else ""),
            mcp={
                "tool": "flow_run",
                "arguments": {"name": flow.name, **({"dry_run": True} if not safe else {})},
            },
            reason=(
                "A matching saved journey can perform the setup in one call."
                if safe
                else "Preview this matching journey before supplying parameters or authorizing effects."
            ),
            executes=safe,
        )
        out.append(
            GoalCandidate(
                id=f"flow:{flow.name}",
                kind="flow",
                name=flow.name,
                score=4_000 + score,
                safe=safe,
                status="ready" if safe else "requires_review",
                risks=risks,
                evidence={
                    "description": flow.description,
                    "aliases": flow.aliases,
                    "arrival": flow.arrival,
                    "steps": [step_display(step) for step in flow.steps],
                    "params": sorted(flow.params),
                },
                call=call,
            )
        )
    return sorted(out, key=lambda item: (-item.score, item.name))[:5]


def _deeplink_candidates(goal: str, app: AppMap, *, target: str | None) -> list[GoalCandidate]:
    out: list[GoalCandidate] = []
    for index, link in enumerate(app.deeplinks):
        score = _match_score(goal, link.uri, link.note, link.landed)
        if target and link.landed == target:
            score += 100
        if score < 20:
            continue
        uri = shlex.quote(link.uri)
        call = GoalCall(
            kind="deeplink_preview",
            cli=f"aua open-and-analyze {uri}",
            mcp={"tool": "open_link", "arguments": {"uri": link.uri, "observe": True}},
            reason=(
                "This shortcut was probed before, but intent delivery still does not prove arrival."
                if link.probed
                else "This remembered shortcut has not been probed; inspect it only after routes and flows."
            ),
            executes=False,
        )
        out.append(
            GoalCandidate(
                id=f"deeplink:{index}",
                kind="deeplink",
                name=link.uri,
                target=link.landed,
                score=2_000 + score + (50 if link.probed else 0),
                safe=False,
                status="probed" if link.probed else "unprobed",
                risks=[
                    {
                        "code": "deeplink_effect",
                        "reason": "intent delivery does not prove navigation or exclude state mutation",
                        "path": "deeplink",
                    }
                ],
                evidence={"note": link.note, "probed": link.probed, "landed": link.landed},
                call=call,
            )
        )
    return sorted(out, key=lambda item: (-item.score, item.name))[:5]


def plan_goal_session(
    goal: str,
    observation: AnalyzeResult,
    *,
    app: AppMap | None = None,
    context_id: str = "default",
    current_screen: str | None = None,
    flows: Iterable[Flow] = (),
    destructive_labels: Sequence[str] = (),
    relevant_capabilities: Iterable[dict[str, Any]] = (),
) -> GoalSessionPlan:
    """Rank known ways to reach *goal* without performing another observation or action."""
    if not goal.strip():
        raise ValueError("goal must not be empty")
    screen = observation.meta.known_screen or current_screen
    candidates: list[GoalCandidate] = []
    goto_candidate: GoalCandidate | None = None
    if app is not None and app.package == observation.screen.package:
        goto_candidate = _goto_candidate(
            goal,
            observation,
            app,
            context_id=context_id,
            current_screen=screen,
            destructive_labels=destructive_labels,
        )
        if goto_candidate is not None:
            candidates.append(goto_candidate)
    candidates.extend(
        _flow_candidates(
            goal,
            observation,
            flows,
            destructive_labels=destructive_labels,
        )
    )
    if app is not None and app.package == observation.screen.package:
        candidates.extend(
            _deeplink_candidates(
                goal,
                app,
                target=goto_candidate.target if goto_candidate is not None else None,
            )
        )
    # Tier is intentional, not score-only: goto → flow → deeplink.  Scores rank peers.
    order = {"arrived": 0, "goto": 0, "flow": 1, "deeplink": 2}
    candidates.sort(key=lambda item: (order[item.kind], -item.score, item.name))
    selected = next((candidate for candidate in candidates if candidate.safe), None)
    warnings: list[str] = []
    if "offline" in goal.casefold() or "airplane" in goal.casefold():
        warnings.append(
            "Use `aua network offline --verify`, not airplane mode. Because this is a goal "
            "session, end with its exact `cleanup_call`; one `aua session finish` restores "
            "session-owned connectivity and returns the review."
        )
    if any(candidate.kind == "deeplink" for candidate in candidates):
        warnings.append(
            "Deeplinks rank after verified routes and saved flows and require explicit unsafe "
            "authorization plus arrival evidence."
        )
    if "offline" in goal.casefold():
        recommendation = GoalCall(
            kind="network_offline",
            cli="aua network offline --verify",
            mcp={"tool": "network_offline", "arguments": {"verify": True}},
            reason=(
                "The goal requires a proven offline state. AUA saves the current controls so "
                "session finish can restore them."
            ),
        )
    elif selected is not None:
        recommendation = selected.call.model_copy(update={"candidate_id": selected.id})
    elif candidates:
        candidate = candidates[0]
        recommendation = candidate.call.model_copy(update={"candidate_id": candidate.id})
        warnings.append("No automatically safe navigation option matched; review before acting.")
    else:
        recommendation = GoalCall(
            kind="map_find",
            cli=f"aua map --find {shlex.quote(goal)}",
            mcp={"tool": "map_find", "arguments": {"goal": goal}},
            reason="No context-compatible route, matching flow, or relevant deeplink is known.",
            executes=False,
        )
        warnings.append(
            "Continue from the returned observation with semantic selectors; repeated manual "
            "navigation can be saved as a flow."
        )
    return GoalSessionPlan(
        goal=goal,
        package=observation.screen.package,
        current_screen=screen,
        observation=observation,
        candidates=candidates,
        selected_candidate=selected.id if selected is not None else None,
        recommended_call=recommendation,
        warnings=warnings,
        relevant_capabilities=list(relevant_capabilities),
    )
