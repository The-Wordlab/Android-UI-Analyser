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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from .atomic import atomic_write_text
from .errors import UsageError
from .flows import Flow
from .memory import (
    AppMap,
    RouteEdge,
    RouteStep,
    ScreenRecord,
    _shortest_path,
    is_destructive_step,
    resolve_goal,
    route_step_risks,
    step_display,
    target_arrival_evidence,
)
from .schema import AnalyzeResult
from .selectors import is_back_resource_id
from .session_contracts import (
    SessionContract,
    parse_session_contract_yaml,
    render_session_contract_yaml,
)

if TYPE_CHECKING:
    from .jobs import JobState


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


PhaseIntent: TypeAlias = Literal[
    "ui_verification",
    "alternative",
    "network_observation",
    "offline_transition",
    "cleanup_finalizer",
    "contract_checkpoint",
    "contract_cleanup",
]
PhaseSatisfaction: TypeAlias = Literal[
    "relevant_evidence",
    "verified_network_status",
    "verified_offline",
    "session_cleanup",
    "fresh_assertions",
]
RequirementExpected: TypeAlias = Literal[
    "present",
    "absent",
    "enabled",
    "disabled",
    "checked",
    "unchecked",
    "selected",
    "unselected",
]


class PhaseRequirement(BaseModel):
    """One explicit observable state whose polarity must survive evidence summarization."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    terms: list[str] = Field(default_factory=list)
    expected: RequirementExpected


class ObservationProvenance(BaseModel):
    """Identity of the exact settled frame used as automatic phase evidence."""

    model_config = ConfigDict(extra="forbid")

    fingerprint: str
    source: Literal["hierarchy", "vision", "mixed"]
    via: str | None = None
    device_serial: str
    package: str


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
    # These fields are additive so sessions persisted before semantic phase compilation still
    # validate. New sessions record what a phase means and what kind of proof can satisfy it.
    intent: PhaseIntent | None = None
    source_span: tuple[int, int] | None = None
    branches: list[GoalBranch] = Field(default_factory=list)
    satisfaction: PhaseSatisfaction | None = None
    terminal: bool = False
    proof: PhaseProof | None = None
    requirements: list[PhaseRequirement] = Field(default_factory=list)
    # A policy such as "never use a deeplink" constrains a checkpoint but is not separately
    # completable work. Keep it typed instead of compiling an impossible verify phase.
    constraints: list[str] = Field(default_factory=list)
    # Authored contracts keep their exact flow assertions on the phase. Natural-language phases
    # preserve the legacy proof policy through these defaults.
    assertions: list[RouteStep] = Field(default_factory=list)
    proof_mode: Literal["manual_or_structured", "fresh_assertions"] = "manual_or_structured"
    manual_completion_allowed: bool = True


class GoalBranch(BaseModel):
    """One mutually exclusive way to satisfy an alternative checkpoint."""

    model_config = ConfigDict(extra="forbid")

    id: str
    condition: str
    objective: str


class PhaseProof(BaseModel):
    """Structured provenance for the evidence that completed a phase."""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "manual_evidence",
        "verified_event",
        "session_cleanup",
        "observation",
        "job_result",
        "contract_assertions",
    ]
    matched_terms: list[str] = Field(default_factory=list)
    satisfied_requirements: list[PhaseRequirement] = Field(default_factory=list)
    branch_id: str | None = None
    command: str | None = None
    verified: bool | None = None
    observation: ObservationProvenance | None = None
    job_id: str | None = None
    job_operation: str | None = None
    predicate_terms: list[str] = Field(default_factory=list)
    # Contract proof is one atomic assertion-set verdict against one fresh observation. The
    # evidence id links the phase to the session bundle; capture_order anchors candidate-flow
    # insertion to the action that produced that frame.
    evidence_id: str | None = None
    assertions_verified: int = Field(default=0, ge=0)
    capture_order: int | None = Field(default=None, ge=0)


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
    animations_enabled: bool = False
    animation_backup_path: str | None = None
    phases: list[GoalPhase] = Field(default_factory=list)
    contract: SessionContract | None = None
    contract_yaml: str | None = None
    artifact_dir: str | None = None
    evidence: Literal["none", "failures", "all"] = "failures"
    junit: bool = False
    # The navigation journal is rolling, so candidate capture starts from an absolute action
    # watermark inside the segment active when the session began.
    capture_package: str | None = None
    capture_context_id: str | None = None
    capture_segment: int | None = Field(default=None, ge=0)
    capture_start_order: int | None = Field(default=None, ge=0)
    # Prevent two ordered authored checkpoints from being completed by one unchanged frame.
    last_contract_fingerprint: str | None = None
    finished_ms: int | None = None


_SEQUENCE_BOUNDARY = re.compile(
    r"\s*(?:[.;]\s+|\bthen\b|\bafter that\b|\bnext\b|\bfinally\b|\bcompare\b)",
    re.IGNORECASE,
)
_RETURN_SEQUENCE = re.compile(
    r",\s+(?=(?:then\s+)?(?:open|verify|inspect|select|tap|check|confirm)\b)",
    re.IGNORECASE,
)
_RESTORE_GOAL = re.compile(
    r"\b(?:restore|restoring|re-enable|reconnect)\b.{0,48}"
    r"\b(?:network(?:ing)?|connectivity|connections?|wi-?fi|internet)\b"
    r"|\breturn\b\s+(?:(?:the|its)\s+)?"
    r"(?:network(?:ing)?|connectivity|connections?|internet)\b",
    re.IGNORECASE,
)
_FINISH_CLEANUP_GOAL = re.compile(
    r"\b(?:finish|complete|finali[sz]e|end)\b.{0,36}\b(?:session\s+)?cleanup\b",
    re.IGNORECASE,
)
_CONNECTIVITY_RESTORED_GOAL = re.compile(
    r"\b(?:network(?:ing)?|connectivity|connections?|wi-?fi|internet)\b.{0,24}"
    r"\b(?:restored|re-enabled|reconnected|online)\b",
    re.IGNORECASE,
)
_NETWORK_STATUS_GOAL = re.compile(
    r"\b(?:record|capture|observe|report|read)\b.{0,48}?\b"
    r"(?:(?:active|current|default|initial|verified)\s+)*"
    r"(?:(?:network|connections?|internet)\s+(?:status|state|transport)"
    r"|connectivity(?:\s+(?:status|state))?|transport)\b",
    re.IGNORECASE,
)
_OFFLINE_GOAL = re.compile(
    r"\b(?:make|take|put|set|ensure)\b.{0,32}?\b"
    r"(?:emulator|device|phone|tablet)\b.{0,24}?\boffline\b"
    r"|\b(?:(?:go|switch|enter|work|test|verify)\s+)?(?:fully\s+)?offline\b"
    r"(?:\s+(?:mode|state))?"
    r"|\b(?:(?:enter|enable|switch\s+to|turn\s+on)\s+)?airplane mode\b",
    re.IGNORECASE,
)
_CONDITIONAL_BRANCH = re.compile(
    r"\bif\b[^.]*?\botherwise\b[^.;]*",
    re.IGNORECASE,
)
_CONDITIONAL_MASK = "\ue000"
_PRESENT_BRANCH = re.compile(
    r"\bif\s+(?:(?:it|they|one|threads?|items?|fixtures?)\s+)?"
    r"(?:(?:is|are)\s+)?(?:already\s+)?(?:present|available|existing|found|exists)\b",
    re.IGNORECASE,
)
_MISSING_BRANCH = re.compile(
    r"\b(?:only\s+)?if\s+(?:(?:it|they|one|threads?|items?|fixtures?)\s+)?"
    r"(?:(?:is|are)\s+)?(?:missing|absent|unavailable|not\s+found)\b",
    re.IGNORECASE,
)
_CONSTRAINT_GOAL = re.compile(
    r"^(?:never|do\s+not|don't|must\s+not|avoid|without)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _GoalClause:
    text: str
    start: int
    end: int


def _protect_conditional_branches(goal: str) -> str:
    """Keep ``if … then … otherwise …`` as one checkpoint.

    ``then`` normally and usefully denotes an ordered checkpoint. Inside an explicit
    alternative it instead connects mutually exclusive branches. Splitting there turns both
    branches into mandatory work, so mask only that token until ordinary sequencing is parsed.
    Sentence/semicolon boundaries remain available around the conditional group.
    """

    protected = list(goal)
    for match in _CONDITIONAL_BRANCH.finditer(goal):
        branch = match.group(0)
        for token in re.finditer(r"\bthen\b", branch, flags=re.IGNORECASE):
            start = match.start() + token.start()
            protected[start : start + len(token.group(0))] = _CONDITIONAL_MASK * len(token.group(0))
        for token in re.finditer(r";(?=\s*otherwise\b)", branch, flags=re.IGNORECASE):
            protected[match.start() + token.start()] = _CONDITIONAL_MASK
    return "".join(protected)


def _goal_clauses(goal: str) -> list[_GoalClause]:
    """Split explicit sequencing while retaining source ownership for every checkpoint."""
    protected = _protect_conditional_branches(goal)
    boundaries = list(_SEQUENCE_BOUNDARY.finditer(protected))
    clauses: list[_GoalClause] = []
    start = 0
    for boundary in [*boundaries, None]:
        end = boundary.start() if boundary is not None else len(goal)
        clause_start = start
        clause_end = end
        while clause_start < clause_end and (
            goal[clause_start].isspace() or goal[clause_start] == ","
        ):
            clause_start += 1
        while clause_end > clause_start and (
            goal[clause_end - 1].isspace() or goal[clause_end - 1] in ",;."
        ):
            clause_end -= 1
        if clause_start < clause_end:
            clauses.append(
                _GoalClause(
                    text=goal[clause_start:clause_end],
                    start=clause_start,
                    end=clause_end,
                )
            )
        if boundary is not None:
            start = boundary.end()
    expanded: list[_GoalClause] = []
    for clause in clauses:
        # A bounded return followed by another imperative is two observable checkpoints even
        # when written as one sentence. Do not split generic introductory commas ("From X,
        # open Y"): only a clause explicitly beginning with return/navigation owns this shape.
        split = (
            _RETURN_SEQUENCE.search(clause.text)
            if re.match(r"^(?:return|navigate|go\s+back)\b", clause.text, flags=re.IGNORECASE)
            else None
        )
        if split is None:
            expanded.append(clause)
            continue
        left_end = clause.start + split.start()
        right_start = clause.start + split.end()
        expanded.extend(
            [
                _GoalClause(clause.text[: split.start()].rstrip(), clause.start, left_end),
                _GoalClause(clause.text[split.end() :].lstrip(), right_start, clause.end),
            ]
        )
    return expanded


def _branch_condition(text: str) -> Literal["present", "missing"] | None:
    if _PRESENT_BRANCH.search(text):
        return "present"
    if _MISSING_BRANCH.search(text):
        return "missing"
    return None


def _alternative_branches(
    clause: _GoalClause, following: _GoalClause | None
) -> tuple[list[GoalBranch], _GoalClause] | None:
    """Compile explicit alternatives without making every branch mandatory."""
    first_condition = _branch_condition(clause.text)
    second_condition = _branch_condition(following.text) if following is not None else None
    if first_condition == "present" and second_condition == "missing" and following is not None:
        combined = _GoalClause(
            text=f"{clause.text}; {following.text}",
            start=clause.start,
            end=following.end,
        )
        return (
            [
                GoalBranch(
                    id="branch_present",
                    condition="present",
                    objective=clause.text,
                ),
                GoalBranch(
                    id="branch_missing",
                    condition="missing",
                    objective=following.text,
                ),
            ],
            combined,
        )

    if re.search(r"\botherwise\b", clause.text, flags=re.IGNORECASE):
        first, second = re.split(r"\botherwise\b", clause.text, maxsplit=1, flags=re.IGNORECASE)
        if first.strip(" ,;") and second.strip(" ,;"):
            return (
                [
                    GoalBranch(
                        id="branch_condition",
                        condition="condition_met",
                        objective=first.strip(" ,;"),
                    ),
                    GoalBranch(
                        id="branch_otherwise",
                        condition="otherwise",
                        objective=second.strip(" ,;"),
                    ),
                ],
                clause,
            )
    return None


def _only_words(text: str, allowed: frozenset[str]) -> bool:
    words = {
        word.casefold()
        for word in re.findall(r"[^\W_]+", text)
        if word.casefold() != "s" and not word.isdigit()
    }
    return bool(words) and words <= allowed


_OFFLINE_METHOD_WORDS = frozenset(
    {
        "a",
        "and",
        "an",
        "aua",
        "by",
        "command",
        "confirm",
        "confirmed",
        "device",
        "emulator",
        "ensure",
        "establish",
        "its",
        "make",
        "method",
        "mode",
        "network",
        "operation",
        "phone",
        "provably",
        "put",
        "requested",
        "reversible",
        "safe",
        "safely",
        "set",
        "state",
        "tablet",
        "take",
        "the",
        "through",
        "tool",
        "use",
        "using",
        "verified",
        "verifiably",
        "via",
        "with",
    }
)
_NETWORK_STATUS_METHOD_WORDS = frozenset(
    {
        "a",
        "active",
        "and",
        "an",
        "capture",
        "connection",
        "connections",
        "connectivity",
        "current",
        "default",
        "device",
        "emulator",
        "initial",
        "internet",
        "its",
        "network",
        "observe",
        "read",
        "record",
        "report",
        "state",
        "status",
        "the",
        "transport",
        "verified",
    }
)
_CLEANUP_FINALIZER_WORDS = frozenset(
    {
        "after",
        "already",
        "and",
        "as",
        "aua",
        "before",
        "call",
        "check",
        "checked",
        "clean",
        "cleanup",
        "command",
        "complete",
        "completed",
        "completion",
        "connectivity",
        "end",
        "ending",
        "ends",
        "emulator",
        "enable",
        "enabled",
        "fi",
        "finalise",
        "finalize",
        "finish",
        "finished",
        "finishing",
        "goal",
        "internet",
        "its",
        "network",
        "networking",
        "of",
        "on",
        "once",
        "original",
        "part",
        "re",
        "reconnect",
        "reconnected",
        "return",
        "restore",
        "restored",
        "restoring",
        "run",
        "running",
        "session",
        "starting",
        "state",
        "task",
        "the",
        "then",
        "to",
        "up",
        "use",
        "verify",
        "verified",
        "when",
        "while",
        "wi",
    }
)


def _is_offline_method_modifier(text: str) -> bool:
    return not text or _only_words(text, _OFFLINE_METHOD_WORDS)


def _is_network_status_modifier(text: str) -> bool:
    return not text or _only_words(text, _NETWORK_STATUS_METHOD_WORDS)


def _is_cleanup_finalizer(text: str) -> bool:
    return not text or _only_words(text, _CLEANUP_FINALIZER_WORDS)


_ABSENCE_PREFIX = re.compile(
    r"\b(?:without|no)\s+(?P<subject>[^.;,]+?)"
    r"(?=\s+\b(?:and|but|while|then|plus)\b|$)",
    re.IGNORECASE,
)
_ABSENCE_SUFFIX = re.compile(
    r"(?:^|[;,]|\b(?:and|but|while|then|plus)\b)\s*"
    r"(?P<subject>[A-Za-z0-9](?:(?!\b(?:and|but|while|then|plus)\b)[^.;,]){0,56}?)\s+"
    r"(?:(?:is|are|was|were|remains?)\s+)?"
    r"(?:absent|missing|not\s+(?:present|visible|shown))\b",
    re.IGNORECASE,
)
_PRESENCE_SUFFIX = re.compile(
    r"(?:^|[;,]|\b(?:and|but|while|then|plus)\b)\s*"
    r"(?P<subject>[A-Za-z0-9](?:(?!\b(?:and|but|while|then|plus|without)\b)[^.;,]){0,96}?)\s+"
    r"(?:(?:is|are|was|were|becomes?|remains?)\s+)?"
    r"(?:visible|shown|present|displayed|readable|open|opened|opens)\b",
    re.IGNORECASE,
)
_VERIFY_OBJECT = re.compile(
    r"\b(?:verify|confirm|check|inspect|observe)\s+(?P<subject>[^.;,]+?)"
    r"(?=\s+\b(?:and|but|plus|with|without|while|then|offline)\b|[.;,]|$)",
    re.IGNORECASE,
)
_NAMED_STATE = r"enabled|disabled|checked|unchecked|selected|unselected"
_STATE_PREFIX = re.compile(
    rf"\b(?P<state>{_NAMED_STATE})\s+(?P<subject>[^.;,]+?)"
    r"(?=\s+\b(?:and|but|while|then|plus|offline)\b|$)",
    re.IGNORECASE,
)
_STATE_SUFFIX = re.compile(
    r"(?:^|[;,]|\b(?:and|but|while|then|plus)\b)\s*"
    r"(?P<subject>[A-Za-z0-9](?:(?!\b(?:and|but|while|then|plus)\b)[^.;,]){0,56}?)\s+"
    rf"(?:(?:is|are|was|were|remains?)\s+)?(?P<state>{_NAMED_STATE})\b",
    re.IGNORECASE,
)
_ASSERTION_SUBJECT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "be",
        "being",
        "behind",
        "check",
        "confirm",
        "condition",
        "checked",
        "ensure",
        "disabled",
        "enabled",
        "inspect",
        "is",
        "longer",
        "open",
        "opened",
        "opens",
        "present",
        "readable",
        "remaining",
        "shown",
        "state",
        "still",
        "selected",
        "the",
        "unchecked",
        "unselected",
        "verify",
        "visible",
        "was",
        "were",
    }
)


def _phase_requirement(
    subject: str,
    expected: RequirementExpected,
) -> PhaseRequirement | None:
    terms = sorted(_proof_terms(subject) - _ASSERTION_SUBJECT_STOP_WORDS)
    if not terms:
        return None
    return PhaseRequirement(
        subject=" ".join(subject.strip().split())[:120],
        terms=terms,
        expected=expected,
    )


def _phase_requirements(objective: str) -> list[PhaseRequirement]:
    """Compile explicit observable assertions into one conjunctive proof contract."""
    requirements: list[PhaseRequirement] = []
    for pattern in (_ABSENCE_PREFIX, _ABSENCE_SUFFIX):
        for match in pattern.finditer(objective):
            requirement = _phase_requirement(match.group("subject"), "absent")
            if requirement is not None:
                requirements.append(requirement)
    for pattern in (_STATE_PREFIX, _STATE_SUFFIX):
        for match in pattern.finditer(objective):
            requirement = _phase_requirement(
                match.group("subject"),
                cast(RequirementExpected, match.group("state").casefold()),
            )
            if requirement is not None:
                requirements.append(requirement)
    # Preserve the long-standing ordering of explicit negative/state assertions in output.
    # Positive requirements are additive and primarily serve automatic observation/job proof.
    for pattern in (_PRESENCE_SUFFIX, _VERIFY_OBJECT):
        for match in pattern.finditer(objective):
            requirement = _phase_requirement(match.group("subject"), "present")
            if requirement is not None:
                requirements.append(requirement)
    unique: list[PhaseRequirement] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for requirement in requirements:
        key = (requirement.expected, tuple(requirement.terms))
        if key not in seen:
            unique.append(requirement)
            seen.add(key)
    return unique


def _phase(
    phases: list[GoalPhase],
    *,
    objective: str,
    kind: Literal["environment", "verify", "cleanup"],
    intent: PhaseIntent,
    source_span: tuple[int, int],
    satisfaction: PhaseSatisfaction,
    branches: list[GoalBranch] | None = None,
    terminal: bool = False,
) -> None:
    # A clause may legitimately carry two different intents (for example, enter a verified
    # offline state *and* inspect cached UI). Only an identical semantic draft is a duplicate;
    # source-span equality alone is not enough to discard real work.
    normalized = " ".join(objective.casefold().split())
    if any(
        phase.intent == intent
        and phase.satisfaction == satisfaction
        and phase.source_span == source_span
        and " ".join(phase.objective.casefold().split()) == normalized
        for phase in phases
    ):
        return
    phases.append(
        GoalPhase(
            id=f"phase_{len(phases) + 1}",
            objective=objective,
            kind=kind,
            intent=intent,
            source_span=source_span,
            branches=branches or [],
            satisfaction=satisfaction,
            terminal=terminal,
            requirements=(_phase_requirements(objective) if intent == "ui_verification" else []),
        )
    )


def goal_phases(goal: str) -> list[GoalPhase]:
    """Extract conservative ordered checkpoints without pretending to understand the app.

    Only explicit sequence markers split the user's prose. Ordinary ``and`` stays inside one
    checkpoint, so "open and verify" is not turned into two invented tasks. Offline setup and
    connectivity cleanup are deterministic environment phases; UI checkpoints remain explicit
    evidence contracts owned by the agent using the app.
    """
    cleaned = " ".join(goal.strip().split())
    clauses = _goal_clauses(cleaned)
    phases: list[GoalPhase] = []
    offline_added = False
    cleanup_span: tuple[int, int] | None = None
    index = 0
    clauses = clauses or [_GoalClause(cleaned, 0, len(cleaned))]
    while index < len(clauses):
        clause = clauses[index]
        following = clauses[index + 1] if index + 1 < len(clauses) else None
        alternative = _alternative_branches(clause, following)
        if alternative is not None:
            branches, combined = alternative
            _phase(
                phases,
                objective=combined.text,
                kind="verify",
                intent="alternative",
                source_span=(combined.start, combined.end),
                satisfaction="relevant_evidence",
                branches=branches,
            )
            index += 2 if following is not None and combined.end == following.end else 1
            continue

        segment = clause.text
        advance = 1
        if _CONSTRAINT_GOAL.search(segment):
            if phases:
                phases[-1].constraints.append(segment)
                if phases[-1].source_span is not None:
                    phases[-1].source_span = (phases[-1].source_span[0], clause.end)
            index += 1
            continue

        cleanup_here = bool(
            _RESTORE_GOAL.search(segment)
            or _FINISH_CLEANUP_GOAL.search(segment)
            or _CONNECTIVITY_RESTORED_GOAL.search(segment)
        )
        objective = segment
        if cleanup_here:
            cleanup_end = clause.end
            if following is not None and _is_cleanup_finalizer(following.text):
                cleanup_end = following.end
                advance = 2
            cleanup_span = cleanup_span or (clause.start, cleanup_end)
            objective = _RESTORE_GOAL.sub("", objective)
            objective = _FINISH_CLEANUP_GOAL.sub("", objective)
            objective = _CONNECTIVITY_RESTORED_GOAL.sub("", objective).strip(" ,;.")
            if _is_cleanup_finalizer(objective):
                objective = ""
        network_status_here = _NETWORK_STATUS_GOAL.search(objective)
        if network_status_here is not None:
            _phase(
                phases,
                objective="Record the verified current network status",
                kind="environment",
                intent="network_observation",
                source_span=(clause.start, clause.end),
                satisfaction="verified_network_status",
            )
            objective = _NETWORK_STATUS_GOAL.sub("", objective, count=1).strip(" ,;.")
            objective = re.sub(r"^(?:and|then)\s+", "", objective, flags=re.IGNORECASE)
            if _is_network_status_modifier(objective):
                objective = ""
        offline_here = bool(_OFFLINE_GOAL.search(objective))
        offline_phase: GoalPhase | None = None
        if offline_here and not offline_added:
            _phase(
                phases,
                objective="Establish and verify the requested offline network state",
                kind="environment",
                intent="offline_transition",
                source_span=(clause.start, clause.end),
                satisfaction="verified_offline",
            )
            offline_phase = phases[-1]
            offline_added = True
        if offline_here:
            transition = _OFFLINE_GOAL.search(objective)
            if transition is not None:
                original_objective = objective
                residual = _OFFLINE_GOAL.sub("", objective, count=1).strip(" ,;.")
                residual = re.sub(r"^(?:and|before)\s+", "", residual, flags=re.IGNORECASE)
                if _is_offline_method_modifier(residual):
                    objective = ""
                    if following is not None and _is_offline_method_modifier(following.text):
                        advance = 2
                        if offline_phase is not None:
                            offline_phase.source_span = (clause.start, following.end)
                elif transition.start() == 0:
                    objective = residual
                else:
                    # Here "offline" qualifies app behavior rather than leading a transition
                    # instruction ("Verify Example content opens offline"). Keep that context in
                    # the human proof contract while using the stripped residual only to decide
                    # whether independently observable UI work exists.
                    objective = original_objective
        if objective and objective.casefold() not in {"and", "before finishing"}:
            _phase(
                phases,
                objective=objective,
                kind="verify",
                intent="ui_verification",
                source_span=(clause.start, clause.end),
                satisfaction="relevant_evidence",
            )
        index += advance
    if cleanup_span is not None:
        _phase(
            phases,
            objective="Restore the session-owned network state",
            kind="cleanup",
            intent="cleanup_finalizer",
            source_span=cleanup_span,
            satisfaction="session_cleanup",
            terminal=True,
        )
    if not phases:
        _phase(
            phases,
            objective=cleaned,
            kind="verify",
            intent="ui_verification",
            source_span=(0, len(cleaned)),
            satisfaction="relevant_evidence",
        )
    phases[0].status = "active"
    return phases


def contract_phases(contract: SessionContract) -> list[GoalPhase]:
    """Compile an authored contract into ordered phases with non-manual proof policy."""

    phases = [
        GoalPhase(
            id=checkpoint.id,
            objective=checkpoint.description,
            kind="verify",
            intent="contract_checkpoint",
            satisfaction="fresh_assertions",
            assertions=[step.model_copy(deep=True) for step in checkpoint.assertions],
            proof_mode=checkpoint.proof_mode,
            manual_completion_allowed=checkpoint.manual_completion_allowed,
        )
        for checkpoint in contract.checkpoints
    ]
    if contract.cleanup is not None:
        used_ids = {phase.id for phase in phases}
        cleanup_id = "cleanup"
        suffix = 2
        while cleanup_id in used_ids:
            cleanup_id = f"cleanup_{suffix}"
            suffix += 1
        phases.append(
            GoalPhase(
                id=cleanup_id,
                objective=contract.cleanup.description,
                kind="cleanup",
                intent="contract_cleanup",
                satisfaction="fresh_assertions",
                terminal=True,
                assertions=[step.model_copy(deep=True) for step in contract.cleanup.assertions],
                proof_mode=contract.cleanup.proof_mode,
                manual_completion_allowed=contract.cleanup.manual_completion_allowed,
            )
        )
    phases[0].status = "active"
    return phases


def _blocking_phase(phase: GoalPhase) -> dict[str, Any]:
    satisfaction = _phase_satisfaction(phase)
    required = {
        "verified_network_status": "a verified network_status result",
        "verified_offline": "a verified network_offline result",
        "session_cleanup": "successful session cleanup",
        "relevant_evidence": "phase-specific observable evidence",
        "fresh_assertions": "all authored assertions passing on one fresh observation",
    }[satisfaction]
    if satisfaction == "relevant_evidence" and phase.requirements:
        assertions = ", ".join(
            f"{'/'.join(requirement.terms)}={requirement.expected}"
            for requirement in phase.requirements
        )
        required = f"phase-specific observable evidence including {assertions}"
    return {
        "id": phase.id,
        "objective": phase.objective,
        "required_evidence": required,
    }


def _session_phases(state: SessionState) -> list[GoalPhase]:
    if state.phases:
        return state.phases
    if state.contract is not None:
        return contract_phases(state.contract)
    return goal_phases(state.goal)


def phase_progress(state: SessionState, *, compact: bool = False) -> dict[str, Any]:
    """Return goal progress; ordinary results use the compact non-duplicating form."""
    phases = _session_phases(state)
    current = next((phase for phase in phases if phase.status != "completed"), None)
    completed = sum(phase.status == "completed" for phase in phases)
    terminated = state.finished_ms is not None
    current_payload = current.model_dump(mode="json") if current is not None else None
    if (compact or terminated) and isinstance(current_payload, dict):
        current_payload.pop("recommended_call", None)
    manual_checkpoint = (
        current is not None
        and not terminated
        and _phase_satisfaction(current) == "relevant_evidence"
        and current.manual_completion_allowed
    )
    payload: dict[str, Any] = {
        "session_id": state.session_id,
        "completed": completed,
        "total": len(phases),
        "done": current is None,
        "terminated": terminated,
        "status": (
            "completed" if current is None else "terminated_incomplete" if terminated else "active"
        ),
        "current": current_payload,
        # A terminated session is immutable and has no active next call. Preserve the incomplete
        # phase as evidence of what was not done rather than inviting work against a closed owner.
        "next_call": current.recommended_call if current is not None and not terminated else None,
        "checkpoint": (
            None
            if not manual_checkpoint or current is None
            else {
                "cli": (
                    f'--phase-done {current.id}="<observable facts or '
                    'observation_contract.evidence_id>"'
                ),
                "mcp": {
                    "phase_done": {
                        "id": current.id,
                        "evidence": "<observable facts or observation evidence_id>",
                    }
                },
                "proof_required": True,
                "minimum_relevant_terms": _required_proof_matches(current),
                "note": (
                    "Acknowledge this phase on the next AUA call with phase-specific observable "
                    "facts, or reuse the current result's observation_contract.evidence_id when "
                    "that exact frame satisfies the checkpoint. Unrelated evidence is rejected."
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
    if terminated:
        payload["blocking_phases"] = [
            _blocking_phase(phase) for phase in phases if phase.status != "completed"
        ]
    return payload


_PROOF_STOP_WORDS = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "android",
        "app",
        "as",
        "at",
        "before",
        "by",
        "check",
        "checked",
        "complete",
        "completed",
        "completes",
        "completing",
        "confirm",
        "confirmed",
        "destructive",
        "displayed",
        "done",
        "end",
        "ending",
        "ensure",
        "explore",
        "explored",
        "exploring",
        "finally",
        "flow",
        "for",
        "from",
        "if",
        "in",
        "inspect",
        "into",
        "item",
        "meaningful",
        "next",
        "non",
        "of",
        "on",
        "one",
        "only",
        "open",
        "opened",
        "opens",
        "or",
        "otherwise",
        "result",
        "screen",
        "shown",
        "success",
        "successful",
        "successfully",
        "test",
        "tested",
        "testing",
        "the",
        "then",
        "through",
        "to",
        "tool",
        "ui",
        "user",
        "verified",
        "verify",
        "visible",
        "was",
        "were",
        "with",
        "works",
        "working",
    }
)
_PROOF_CANONICAL = {
    "absent": "missing",
    "availability": "present",
    "available": "present",
    "cache": "cache",
    "cached": "cache",
    "create": "create",
    "created": "create",
    "creates": "create",
    "creating": "create",
    "creation": "create",
    "existing": "present",
    "exists": "present",
    "fixtures": "fixture",
    "missing": "missing",
    "navigate": "navigation",
    "navigated": "navigation",
    "navigation": "navigation",
    "recent": "recent",
    "recents": "recent",
    "return": "return",
    "returned": "return",
    "returning": "return",
    "reuse": "reuse",
    "reused": "reuse",
    "threads": "thread",
    "unavailable": "missing",
}

_UNFINISHED_EVIDENCE = re.compile(
    r"\b(?:still\s+(?:needs?|requires?)|remains?\s+(?:to\s+be\s+)?|not\s+yet|"
    r"pending|unverified|incomplete|failed\s+to|did\s+not|does\s+not|has\s+not|have\s+not)"
    r"\b.{0,48}\b(?:verify|verified|verification|open|opened|complete|completed|done|check|checked)\b",
    re.IGNORECASE,
)


def _proof_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[^\W_]+", text.casefold()):
        if len(raw) < 3 or raw in _PROOF_STOP_WORDS:
            continue
        term = _PROOF_CANONICAL.get(raw, raw)
        if term.endswith("s") and len(term) > 4 and not term.endswith("ss"):
            term = term[:-1]
        terms.add(term)
    return terms


_GENERIC_ASSERTION_TERMS = frozenset(
    {
        "affordance",
        "button",
        "checkbox",
        "control",
        "field",
        "interaction",
        "item",
        "option",
        "switch",
        "toggle",
    }
)

_REQUIREMENT_ASSERTIONS: dict[RequirementExpected, str] = {
    "present": r"(?:visible|shown|present|displayed|readable|open|opened|available)",
    "absent": r"(?:absent|missing|gone)",
    "enabled": r"(?:enabled|available|interactive|usable)",
    "disabled": r"(?:disabled|unavailable|inactive)",
    "checked": r"checked",
    "unchecked": r"unchecked",
    "selected": r"selected",
    "unselected": r"unselected",
}
_ASSERTION_NEGATION = r"(?:not|never)\s+(?:(?:actually|currently|presently|still|yet)\s+){0,2}"


def _evidence_explicitly_negates_requirement(
    requirement: PhaseRequirement,
    evidence: str,
) -> bool:
    """Reject a literal assertion when the evidence applies explicit opposite polarity.

    Lexical manual proof intentionally remains permissive: a caller does not need to repeat
    every positive assertion word-for-word. It must never, however, turn ``not visible`` or
    ``not enabled`` into proof merely because the positive token and subject both occur.
    """
    specific = [term for term in requirement.terms if term not in _GENERIC_ASSERTION_TERMS]
    anchors = specific or requirement.terms
    assertion = _REQUIREMENT_ASSERTIONS[requirement.expected]
    negated = rf"\b{_ASSERTION_NEGATION}{assertion}\b"
    matches = [
        re.search(
            rf"{negated}[^.;]{{0,48}}?{_assertion_term_pattern(term)}",
            evidence,
            re.IGNORECASE,
        )
        or re.search(
            rf"{_assertion_term_pattern(term)}[^.;]{{0,48}}?{negated}",
            evidence,
            re.IGNORECASE,
        )
        for term in anchors
    ]
    return bool(matches) and all(matches)


def _assertion_term_pattern(term: str) -> str:
    return rf"\b{re.escape(term)}(?:s|es)?\b"


def _evidence_matches_requirement(requirement: PhaseRequirement, evidence: str) -> bool:
    if _evidence_explicitly_negates_requirement(requirement, evidence):
        return False
    specific = [term for term in requirement.terms if term not in _GENERIC_ASSERTION_TERMS]
    anchors = specific or requirement.terms
    expected = requirement.expected
    if expected == "present":
        before = (
            r"\b(?:visible|shown|present|displayed|readable|open|opened|available)\b"
            r"[^.;]{0,48}?{target}"
        )
        after = (
            r"{target}[^.;]{0,48}?\b(?:visible|shown|present|displayed|readable|"
            r"open|opened|available)\b"
        )
    elif expected == "absent":
        before = r"\b(?:no|without)\b[^.;]{0,48}?{target}"
        after = (
            r"{target}[^.;]{0,48}?\b(?:absent|missing|gone|"
            r"not\s+(?:present|visible|shown))\b"
        )
    else:
        state = {
            "enabled": r"(?:enabled|available|interactive|usable)",
            "disabled": r"(?:disabled|unavailable|inactive)",
            "checked": "checked",
            "unchecked": "unchecked",
            "selected": "selected",
            "unselected": "unselected",
        }[expected]
        before = rf"\b{state}\b[^.;]{{0,48}}?{{target}}"
        after = rf"{{target}}[^.;]{{0,48}}?\b{state}\b"
    matches = [
        re.search(
            before.replace("{target}", _assertion_term_pattern(term)),
            evidence,
            re.IGNORECASE,
        )
        or re.search(
            after.replace("{target}", _assertion_term_pattern(term)),
            evidence,
            re.IGNORECASE,
        )
        for term in anchors
    ]
    return bool(matches) and all(matches)


def _phase_satisfaction(
    phase: GoalPhase,
) -> PhaseSatisfaction:
    if phase.satisfaction is not None:
        return phase.satisfaction
    if phase.kind == "environment":
        return "verified_offline"
    if phase.kind == "cleanup":
        return "session_cleanup"
    return "relevant_evidence"


def _required_proof_matches(phase: GoalPhase) -> int:
    objective_terms = _proof_terms(phase.objective)
    return 2 if phase.intent == "alternative" or len(objective_terms) >= 5 else 1


_PRESENT_CONDITION_EVIDENCE = re.compile(
    r"\b(?:present|available|existing|exists?|existed|found)\b",
    re.IGNORECASE,
)
_MISSING_CONDITION_EVIDENCE = re.compile(
    r"\b(?:missing|absent|unavailable)\b"
    r"|\b(?:not\s+found|(?:did|does)\s+not\s+exist|no\s+longer\s+exists?|none\s+existed)\b",
    re.IGNORECASE,
)


def _has_unnegated_condition_evidence(pattern: re.Pattern[str], evidence: str) -> bool:
    for match in pattern.finditer(evidence):
        # Some missing-state assertions (for example ``not found``) include their negation in
        # the match itself. For a bare marker, reject an immediately governing ``not``/``never``.
        prefix = evidence[max(0, match.start() - 32) : match.start()]
        if not re.search(rf"\b{_ASSERTION_NEGATION}$", prefix, re.IGNORECASE):
            return True
    return False


def _alternative_condition_is_substantiated(condition: str, evidence: str) -> bool:
    if condition == "present":
        return _has_unnegated_condition_evidence(_PRESENT_CONDITION_EVIDENCE, evidence)
    if condition == "missing":
        return _has_unnegated_condition_evidence(_MISSING_CONDITION_EVIDENCE, evidence)
    # Generic ``if … otherwise …`` branches do not have a portable state vocabulary. Their
    # unique branch-specific lexical score remains the substantiation contract.
    return True


def _manual_phase_proof(phase: GoalPhase, evidence: str) -> PhaseProof:
    if phase.intent != "alternative" and _UNFINISHED_EVIDENCE.search(evidence):
        raise ValueError(
            f"evidence explicitly says {phase.id!r} remains unfinished; do not advance it"
        )
    all_requirements = phase.requirements or _phase_requirements(phase.objective)
    requirements = (
        []
        if phase.intent == "alternative"
        else [requirement for requirement in all_requirements if requirement.expected != "present"]
    )
    missing = [
        requirement
        for requirement in requirements
        if not _evidence_matches_requirement(requirement, evidence)
    ]
    # Manual proof keeps its established lexical treatment of positive UI facts, but an
    # explicit denial of a required positive fact is never valid evidence for that fact.
    missing.extend(
        requirement
        for requirement in all_requirements
        if requirement.expected == "present"
        and _evidence_explicitly_negates_requirement(requirement, evidence)
    )
    if missing:
        expected = ", ".join(
            f"{'/'.join(requirement.terms)}={requirement.expected}" for requirement in missing
        )
        raise ValueError(
            f"evidence does not substantiate {phase.id!r}; explicitly confirm the named "
            f"state with matching polarity: {expected}"
        )
    objective_terms = _proof_terms(phase.objective)
    evidence_terms = _proof_terms(evidence)
    matched = sorted(objective_terms & evidence_terms)
    if not objective_terms and not phase.requirements:
        if len(evidence_terms) < 2:
            raise ValueError(
                f"evidence does not substantiate {phase.id!r}; include at least 2 distinct "
                "observable facts from the completed flow"
            )
        return PhaseProof(
            source="manual_evidence",
            matched_terms=sorted(evidence_terms)[:12],
        )
    required = _required_proof_matches(phase)
    if len(matched) < required:
        examples = ", ".join(sorted(objective_terms)[:5]) or "the observable checkpoint"
        raise ValueError(
            f"evidence does not substantiate {phase.id!r}; include at least {required} distinct "
            f"phase-specific observable fact(s) related to: {examples}"
        )

    branch_id: str | None = None
    if phase.branches:
        scored = [
            (
                len(_proof_terms(branch.objective) & evidence_terms),
                branch.id,
                _alternative_condition_is_substantiated(branch.condition, evidence),
            )
            for branch in phase.branches
        ]
        eligible = [(score, candidate) for score, candidate, proven in scored if proven]
        best_score = max((score for score, _candidate in eligible), default=0)
        winners = [candidate for score, candidate in eligible if score == best_score and score > 0]
        if len(winners) == 1:
            branch_id = winners[0]
        else:
            raise ValueError(
                f"evidence does not substantiate one exact alternative branch for {phase.id!r}; "
                "explicitly confirm which branch condition was observed"
            )
    return PhaseProof(
        source="manual_evidence",
        matched_terms=matched,
        branch_id=branch_id,
    )


def _requirement_key(requirement: PhaseRequirement) -> tuple[str, tuple[str, ...]]:
    return requirement.expected, tuple(requirement.terms)


def _requirement_anchors(requirement: PhaseRequirement) -> set[str]:
    specific = {term for term in requirement.terms if term not in _GENERIC_ASSERTION_TERMS}
    return specific or set(requirement.terms)


def _label_terms(value: str) -> set[str]:
    # Resource IDs commonly use camelCase/snake_case while the goal uses words. Normalize those
    # spellings before applying the same lexical canonicalization used by manual proof.
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).replace("_", " ")
    return _proof_terms(value)


def _element_terms(observation: AnalyzeResult) -> list[tuple[Any, set[str]]]:
    rows: list[tuple[Any, set[str]]] = []
    for element in observation.elements:
        values = [
            value for value in (element.text, element.content_desc, element.resource_id) if value
        ]
        rows.append((element, _label_terms(" ".join(values))))
    return rows


def _observation_satisfies_requirement(
    requirement: PhaseRequirement,
    elements: list[tuple[Any, set[str]]],
) -> bool:
    anchors = _requirement_anchors(requirement)
    if not anchors:
        return False
    if requirement.expected == "absent":
        # Absence is deliberately stricter than presence: any visible subject anchor keeps the
        # negative assertion unproven. This avoids declaring a multi-word loading indicator gone
        # merely because one descriptive word was omitted by accessibility.
        return not any(anchors & terms for _element, terms in elements)
    candidates = [element for element, terms in elements if anchors <= terms]
    if requirement.expected == "present":
        return bool(candidates)
    if requirement.expected == "enabled":
        return any(element.enabled is True for element in candidates)
    if requirement.expected == "disabled":
        return any(element.enabled is False for element in candidates)
    if requirement.expected == "checked":
        return any(element.checked is True for element in candidates)
    if requirement.expected == "unchecked":
        return any(element.checkable is True and element.checked is False for element in candidates)
    if requirement.expected == "selected":
        return any(element.selected is True for element in candidates)
    if requirement.expected == "unselected":
        return any(element.selected is False for element in candidates)
    return False


def _observation_provenance(
    state: SessionState,
    observation: AnalyzeResult,
) -> ObservationProvenance | None:
    fingerprint = (observation.meta.fingerprint or "").strip()
    serial = (observation.meta.device_serial or "").strip()
    package = (observation.screen.package or "").strip()
    source_value = getattr(observation.screen.source, "value", observation.screen.source)
    source = str(source_value)
    if (
        not fingerprint
        or observation.meta.stale_risk is not None
        or serial != state.serial
        or not package
        or source not in {"hierarchy", "vision", "mixed"}
    ):
        return None
    via_value = observation.meta.via or getattr(
        observation.meta.path, "value", observation.meta.path
    )
    return ObservationProvenance(
        fingerprint=fingerprint,
        source=cast(Literal["hierarchy", "vision", "mixed"], source),
        via=str(via_value) if via_value else None,
        device_serial=serial,
        package=package,
    )


def _observation_phase_proof(
    state: SessionState,
    phase: GoalPhase,
    observation: AnalyzeResult,
    *,
    source: Literal["observation", "job_result"],
) -> PhaseProof | None:
    """Return proof only when one exact frame satisfies the whole typed UI contract."""
    positive_anchors = {
        term
        for requirement in phase.requirements
        if requirement.expected == "present"
        for term in _requirement_anchors(requirement)
    }
    if (
        state.finished_ms is not None
        or _phase_satisfaction(phase) != "relevant_evidence"
        or phase.intent == "alternative"
        or not phase.requirements
        # A one-word coincidence is discovery evidence, not enough to close a goal checkpoint.
        or len(positive_anchors) < 2
    ):
        return None
    provenance = _observation_provenance(state, observation)
    if provenance is None:
        return None
    elements = _element_terms(observation)
    if not all(
        _observation_satisfies_requirement(requirement, elements)
        for requirement in phase.requirements
    ):
        return None
    matched = sorted(
        {term for requirement in phase.requirements for term in _requirement_anchors(requirement)}
    )
    return PhaseProof(
        source=source,
        matched_terms=matched,
        satisfied_requirements=[
            requirement.model_copy(deep=True) for requirement in phase.requirements
        ],
        command="analyze" if source == "observation" else "job_result",
        verified=True,
        observation=provenance,
    )


def _automatic_evidence(proof: PhaseProof) -> str:
    facts = ", ".join(
        f"{requirement.subject}={requirement.expected}"
        for requirement in proof.satisfied_requirements
    )
    frame = proof.observation.fingerprint[:12] if proof.observation is not None else "unknown"
    if proof.source == "job_result":
        return f"Correlated job {proof.job_id} verified frame {frame}: {facts}"
    return f"Current observation frame {frame} verified: {facts}"


def complete_current_ui_phase_from_observation(
    cache_dir: str | Path,
    state: SessionState,
    *,
    observation: AnalyzeResult,
) -> SessionState:
    """Advance the current UI phase only from its exact, fingerprinted frame contract.

    A plain goal-term overlap is intentionally insufficient. Automatic completion requires a
    compiled positive assertion plus every explicit negative/control-state assertion on the same
    non-stale observation from this session's device.
    """
    phases = _session_phases(state)
    current = next((phase for phase in phases if phase.status != "completed"), None)
    if current is None:
        return state
    proof = _observation_phase_proof(state, current, observation, source="observation")
    if proof is None:
        return state
    return mark_phase_complete(
        cache_dir,
        state,
        phase_id=current.id,
        evidence=_automatic_evidence(proof),
        _proof=proof,
    )


def _await_term_row(value: Any) -> tuple[bool, set[str]] | None:
    if not isinstance(value, dict) or value.get("satisfied") is not True:
        return None
    raw = str(value.get("term") or "").strip()
    negated = raw.startswith("!")
    candidate = raw[1:] if negated else raw
    field, separator, selector = candidate.partition(":")
    if not separator or field.casefold() not in {"text", "rid", "desc"}:
        return None
    present = value.get("present")
    if present is not (not negated):
        return None
    terms = _label_terms(selector.replace(r"\,", ",").replace(r"\!", "!"))
    return negated, terms


def _job_predicate_satisfies_requirements(
    requirements: Sequence[PhaseRequirement],
    result: dict[str, Any],
) -> list[str] | None:
    values = result.get("await_terms")
    if not isinstance(values, list) or not values:
        return None
    parsed: list[tuple[str, bool, set[str]]] = []
    for value in values:
        row = _await_term_row(value)
        if row is None:
            return None
        negated, terms = row
        parsed.append((str(value["term"]), negated, terms))
    for requirement in requirements:
        anchors = _requirement_anchors(requirement)
        expected_negated = requirement.expected == "absent"
        if not any(
            negated == expected_negated and anchors <= terms for _term, negated, terms in parsed
        ):
            return None
    return [term for term, _negated, _terms in parsed]


def complete_current_ui_phase_from_job(
    cache_dir: str | Path,
    state: SessionState,
    *,
    job: JobState,
) -> SessionState:
    """Advance from a successful wait only when job, session, predicate, and frame correlate."""
    if (
        state.finished_ms is not None
        or job.status != "succeeded"
        or job.operation != "await"
        or job.session_id != state.session_id
        or job.serial != state.serial
        or job.owner != state.owner
        or not isinstance(job.result, dict)
        or job.result.get("ok") is not True
        or job.result.get("await_outcome") not in {"satisfied", "absence-satisfied"}
        or not str(job.args.get("predicate") or "").strip()
    ):
        return state
    phases = _session_phases(state)
    current = next((phase for phase in phases if phase.status != "completed"), None)
    if current is None:
        return state
    raw_observation = job.result.get("observation")
    if not isinstance(raw_observation, dict):
        return state
    try:
        observation = AnalyzeResult.model_validate(raw_observation)
    except (TypeError, ValueError):
        return state
    proof = _observation_phase_proof(state, current, observation, source="job_result")
    if proof is None:
        return state
    predicate_terms = _job_predicate_satisfies_requirements(
        current.requirements,
        job.result,
    )
    if predicate_terms is None:
        return state
    proof.job_id = job.job_id
    proof.job_operation = job.operation
    proof.predicate_terms = predicate_terms
    return mark_phase_complete(
        cache_dir,
        state,
        phase_id=current.id,
        evidence=_automatic_evidence(proof),
        _proof=proof,
    )


def _completion_proof(
    phase: GoalPhase,
    evidence: str,
    structured: PhaseProof | None,
) -> PhaseProof:
    satisfaction = _phase_satisfaction(phase)
    if structured is None:
        if satisfaction != "relevant_evidence" or not phase.manual_completion_allowed:
            required = {
                "verified_network_status": "a verified network_status result",
                "verified_offline": "a verified network_offline result",
                "session_cleanup": "successful session cleanup",
                "fresh_assertions": "all authored assertions passing on one fresh observation",
                "relevant_evidence": "structured non-manual evidence",
            }[satisfaction]
            raise ValueError(
                f"{phase.id!r} requires {required}; manual evidence cannot complete it"
            )
        return _manual_phase_proof(phase, evidence)

    valid_network_status = (
        satisfaction == "verified_network_status"
        and structured.source == "verified_event"
        and structured.command == "network_status"
        and structured.verified is True
    )
    valid_offline = (
        satisfaction == "verified_offline"
        and structured.source == "verified_event"
        and structured.command == "network_offline"
        and structured.verified is True
    )
    valid_cleanup = (
        satisfaction == "session_cleanup"
        and structured.source == "session_cleanup"
        and structured.command == "session_finish"
        and structured.verified is True
    )
    valid_contract_assertions = (
        satisfaction == "fresh_assertions"
        and phase.proof_mode == "fresh_assertions"
        and phase.manual_completion_allowed is False
        and structured.source == "contract_assertions"
        and structured.command == "contract_assertions"
        and structured.verified is True
        and structured.observation is not None
        and bool((structured.observation.fingerprint or "").strip())
        and bool((structured.evidence_id or "").strip())
        and bool(phase.assertions)
        and structured.assertions_verified == len(phase.assertions)
    )
    required_keys = {_requirement_key(requirement) for requirement in phase.requirements}
    proven_keys = {
        _requirement_key(requirement) for requirement in structured.satisfied_requirements
    }
    positive_terms = {
        term
        for requirement in phase.requirements
        if requirement.expected == "present"
        for term in _requirement_anchors(requirement)
    }
    valid_ui_observation = (
        satisfaction == "relevant_evidence"
        and structured.source in {"observation", "job_result"}
        and structured.verified is True
        and structured.observation is not None
        and bool(required_keys)
        and len(positive_terms) >= 2
        and required_keys <= proven_keys
        and (
            structured.source != "job_result"
            or (
                bool(structured.job_id)
                and structured.job_operation == "await"
                and bool(structured.predicate_terms)
            )
        )
    )
    if not (
        valid_network_status
        or valid_offline
        or valid_cleanup
        or valid_ui_observation
        or valid_contract_assertions
    ):
        raise ValueError(f"structured proof does not satisfy {phase.id!r}")
    return structured


def mark_phase_complete(
    cache_dir: str | Path,
    state: SessionState,
    *,
    phase_id: str,
    evidence: str,
    _proof: PhaseProof | None = None,
) -> SessionState:
    """Complete only the current phase, preserving ordered goal semantics."""
    if state.finished_ms is not None:
        raise ValueError("cannot complete a goal phase after the session has finished")
    evidence = " ".join(evidence.strip().split())
    if not evidence:
        raise ValueError("phase evidence must not be empty")
    phases = [phase.model_copy(deep=True) for phase in _session_phases(state)]
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
    proof = _completion_proof(current, evidence, _proof)
    contract_fingerprint: str | None = None
    if proof.source == "contract_assertions":
        assert proof.observation is not None  # established by `_completion_proof`
        if proof.observation.device_serial != state.serial:
            raise ValueError("contract proof belongs to a different session device")
        evidence_prefix = f"session-{state.session_id}:observation:"
        if not str(proof.evidence_id).startswith(evidence_prefix):
            raise ValueError("contract proof evidence belongs to a different session")
        contract_fingerprint = proof.observation.fingerprint
        if contract_fingerprint == state.last_contract_fingerprint:
            raise ValueError(
                "an unchanged observation cannot complete two ordered contract checkpoints"
            )
    current.status = "completed"
    current.completed_ms = int(time.time() * 1000)
    current.evidence = evidence[:600]
    current.proof = proof
    if current_index + 1 < len(phases):
        phases[current_index + 1].status = "active"
    update: dict[str, Any] = {"phases": phases}
    if contract_fingerprint is not None:
        update["last_contract_fingerprint"] = contract_fingerprint
    updated = state.model_copy(update=update)
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
    if state.finished_ms is not None:
        return state
    phases = [phase.model_copy(deep=True) for phase in _session_phases(state)]
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
    phases = _session_phases(state)
    current = next((phase for phase in phases if phase.status != "completed"), None)
    if current is None:
        return state
    satisfaction = _phase_satisfaction(current)
    network_value = result.get("state")
    network_state: dict[str, Any] = network_value if isinstance(network_value, dict) else {}
    if (
        satisfaction == "verified_network_status"
        and command == "network_status"
        and result.get("ok")
        and result.get("verified") is True
        and network_state.get("active_network") is not None
    ):
        transports_value = network_state.get("active_transports")
        transports = transports_value if isinstance(transports_value, list) else []
        transport_text = ",".join(str(value) for value in transports) or "none"
        return mark_phase_complete(
            cache_dir,
            state,
            phase_id=current.id,
            evidence=(
                "AUA verified current network status: "
                f"active_network={network_state.get('active_network')}, "
                f"active_transports={transport_text}, "
                f"internet_validated={network_state.get('internet_validated')}, "
                f"offline={network_state.get('offline')}"
            ),
            _proof=PhaseProof(
                source="verified_event",
                matched_terms=["network", "status", "transport"],
                command="network_status",
                verified=True,
            ),
        )
    if (
        satisfaction == "verified_offline"
        and command == "network_offline"
        and result.get("ok")
        and result.get("verified")
        and network_state.get("offline") is True
    ):
        return mark_phase_complete(
            cache_dir,
            state,
            phase_id=current.id,
            evidence="AUA verified no active default network and offline=true",
            _proof=PhaseProof(
                source="verified_event",
                matched_terms=["offline"],
                command="network_offline",
                verified=True,
            ),
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


def update_session_state(
    cache_dir: str | Path,
    state: SessionState,
    **changes: Any,
) -> SessionState:
    """Revalidate and persist additive lifecycle metadata without changing session identity."""

    for field in ("session_id", "serial", "owner", "started_ms"):
        if field in changes and changes[field] != getattr(state, field):
            raise ValueError(f"session identity field {field!r} cannot be changed")
    payload = state.model_dump(mode="python")
    payload.update(changes)
    updated = SessionState.model_validate(payload)
    _write_state(cache_dir, updated)
    return updated


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
    animations_enabled: bool = False,
    animation_backup_path: str | None = None,
    contract: SessionContract | None = None,
    contract_yaml: str | None = None,
    artifact_dir: str | None = None,
    evidence: Literal["none", "failures", "all"] = "failures",
    junit: bool = False,
    capture_package: str | None = None,
    capture_context_id: str | None = None,
    capture_segment: int | None = None,
    capture_start_order: int | None = None,
) -> SessionState:
    if contract_yaml is not None:
        parsed_contract = parse_session_contract_yaml(contract_yaml)
        if contract is not None and parsed_contract != contract:
            raise ValueError("contract and contract_yaml describe different proof contracts")
        contract = parsed_contract
    canonical_contract_yaml = (
        render_session_contract_yaml(contract) if contract is not None else None
    )
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
        animations_enabled=animations_enabled,
        animation_backup_path=animation_backup_path,
        phases=contract_phases(contract) if contract is not None else goal_phases(goal),
        contract=contract.model_copy(deep=True) if contract is not None else None,
        contract_yaml=canonical_contract_yaml,
        artifact_dir=artifact_dir,
        evidence=evidence,
        junit=junit,
        capture_package=capture_package,
        capture_context_id=capture_context_id,
        capture_segment=capture_segment,
        capture_start_order=capture_start_order,
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
    phases = [phase.model_copy(deep=True) for phase in _session_phases(state)]
    now_ms = int(time.time() * 1000)
    for phase in phases:
        if (
            phase.kind == "cleanup"
            and phase.status != "completed"
            and _phase_satisfaction(phase) == "session_cleanup"
        ):
            phase.status = "completed"
            phase.completed_ms = now_ms
            phase.evidence = "session finish restored session-owned reversible state"
            phase.proof = PhaseProof(
                source="session_cleanup",
                matched_terms=["restore", "network"],
                command="session_finish",
                verified=True,
            )
        if phase.status != "completed":
            # A terminated session is immutable. Preserve the missing checkpoint and its proof
            # contract, but never persist an action invitation against a closed owner.
            phase.recommended_call = None
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


def _result_has_reusable_observation(value: Any) -> bool:
    """Whether a prior observation is safe to reuse instead of waiting for readiness."""
    result = _mapping(value)
    observation = result.get("observation")
    if not isinstance(observation, dict):
        return False
    meta = _mapping(observation.get("meta"))
    contract = _mapping(result.get("observation_contract"))
    return not (
        result.get("stale_risk")
        or meta.get("stale_risk")
        or result.get("observation_empty")
        or result.get("settled_unmet")
        or contract.get("reusable") is False
    )


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
        prior_observed = _result_has_reusable_observation(prior_result)
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
    expected_probes = [
        event
        for event in top_level
        if isinstance((event.get("extra") or {}).get("expected_error_code"), str)
    ]
    expected_matches = [
        event
        for event in expected_probes
        if (event.get("extra") or {}).get("expected_error_matched") is True
    ]
    failures = sum(
        1
        for event in top_level
        if (
            not event.get("ok")
            and (event.get("extra") or {}).get("expected_error_matched") is not True
            and _base_command(event.get("cmd")) != "session_review"
        )
        or (
            bool(event.get("ok"))
            and isinstance((event.get("extra") or {}).get("expected_error_code"), str)
        )
    )
    lifecycle_calls = sum(
        1 for event in top_level if _base_command(event.get("cmd")).startswith("session_")
    )
    accounting = {
        "journal_events": len(scoped),
        "top_level_calls": len(top_level),
        "folded_internal_events": max(0, len(scoped) - len(top_level)),
        "lifecycle_calls": lifecycle_calls,
        "task_calls": len(top_level) - lifecycle_calls,
        # Reviews returned by ``session review`` and ``session finish`` are computed before
        # the reporting invocation itself is appended to the journal. Make that boundary
        # explicit so evaluators do not mistake journal events for caller-visible calls or
        # silently under-count the call that carried this report.
        "reporting_call_included": False,
        "top_level_calls_including_reporting_call": len(top_level) + 1,
        "expected_error_probes": len(expected_probes),
        "expected_error_matches": len(expected_matches),
        "unexpected_failures": failures,
    }
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
        "accounting": accounting,
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


def _match_score(goal: str, *values: str | None, exactness: str | None = None) -> int:
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
    if exactness is not None:
        # A control whose VISIBLE label says more than the requested destination is a weaker
        # match for that destination. Without this, `Battery` and `Battery Saver` score
        # identically against the goal `Battery` — the goal phrase is a substring of both — and
        # the tie fell through to `-element.id`, so the winner was whichever control happened to
        # come first in the frame. Measured 2026-08-18 on containment-shaped candidate sets: 6 of
        # 24 rotations took the longer label, one per target, only in the ordering that put it
        # first.
        #
        # Only the visible label counts. Folding the resource id in here punished a control for
        # HAVING a descriptive id (`openGrammar` contributes `open`), which inverted the ranking
        # in favour of unaddressable duplicates. Capped below one matched term (5) so this can
        # only break a tie, never outrank real evidence.
        extra = set(_GOAL_WORD.findall(exactness.casefold())) - set(terms)
        score -= min(len(extra), 4)
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
    flow: Flow,
    *,
    package: str | None,
    destructive_labels: Sequence[str],
    mapped_screens: dict[str, Any] | None = None,
    context_id: str | None = None,
    resolved_evidence: dict[str, Any] | None = None,
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
    if resolved_evidence is not None:
        # Preserve `nested_execution` from the wrapper step: a goal match is evidence for
        # relevance, never authorization to execute another authored journey.  Add the
        # recursively resolved child effects alongside it so review can see what that child
        # would actually change rather than receiving only the generic delegation warning.
        for item in resolved_evidence.get("risks") or []:
            if isinstance(item, dict) and item not in risks:
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
    arrival_invalid = False
    if flow.arrival:
        try:
            # Shared grammar, imported lazily to keep this pure planner module free of an
            # Engine import cycle at module load time.
            from .engine import _parse_await_terms

            _parse_await_terms(flow.arrival, require_positive=True)
        except UsageError:
            arrival_invalid = True
    if arrival_invalid:
        risks.append(
            {
                "code": "arrival_invalid",
                "reason": "the declared flow arrival is invalid or has no positive evidence",
                "path": "arrival",
            }
        )
    elif not flow.arrival_screen and not flow.arrival:
        risks.append(
            {
                "code": "arrival_unverified",
                "reason": "the flow declares no mapped arrival screen or positive arrival evidence",
                "path": "arrival",
            }
        )
    if flow.arrival_screen:
        record = mapped_screens.get(flow.arrival_screen) if mapped_screens is not None else None
        if (
            record is None
            or getattr(record, "stale", True)
            or getattr(record, "context_id", None) not in {context_id, "legacy-default"}
        ):
            risks.append(
                {
                    "code": "arrival_screen_invalid",
                    "reason": "the claimed mapped arrival is missing, stale, or context-incompatible",
                    "path": "arrival_screen",
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
    target = resolve_goal(
        app,
        goal,
        start=current_screen,
        context_id=context_id,
        destructive_labels=destructive_labels,
    )
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
                destructive_labels=destructive_labels,
            )
    if target is None:
        return None
    quoted = shlex.quote(goal)
    if current_screen == target:
        proof = target_arrival_evidence(
            app,
            target,
            goal,
            observation.elements,
            screen_height=observation.screen.height,
        )
        if proof is None:
            # The current mapped screen may contain a clickable row named after the destination.
            # That makes the destination discoverable, but it is not proof that the current
            # frame has arrived there. Let another route/flow/manual option win.
            return None
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
            evidence={"known_screen": observation.meta.known_screen, "arrival_proof": proof},
            call=call,
        )
    path = _shortest_path(
        app,
        target,
        start=current_screen,
        context_id=context_id,
        destructive_labels=destructive_labels,
    )
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
    context_id: str,
    destructive_labels: Sequence[str],
    mapped_screens: dict[str, Any] | None = None,
    resolved_flow_evidence: dict[str, dict[str, Any]] | None = None,
) -> list[GoalCandidate]:
    out: list[GoalCandidate] = []
    for flow in flows:
        if flow.app not in (None, observation.screen.package):
            continue
        if flow.context_id not in (None, context_id):
            continue
        score = _match_score(
            goal,
            flow.name,
            flow.description,
            flow.arrival,
            flow.arrival_screen,
            *flow.aliases,
        )
        # One incidental shared word is not intent.  In a multi-part test goal it made a
        # destructive account-reset flow outrank the visible controls merely because both
        # descriptions said "online".  A single-term goal still scores 25 (token + all-terms),
        # while longer goals need several terms or an exact phrase.
        if score < 20:
            continue
        resolved_evidence = (resolved_flow_evidence or {}).get(flow.name)
        risks = _flow_risks(
            flow,
            package=observation.screen.package,
            destructive_labels=destructive_labels,
            mapped_screens=mapped_screens,
            context_id=context_id,
            resolved_evidence=resolved_evidence,
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
                else (
                    "Preview this matching journey before supplying parameters, authorizing "
                    "effects, or relying on an unverified arrival."
                )
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
                    "arrival_screen": flow.arrival_screen,
                    "arrival_status": flow.arrival_status or "unverified",
                    "context_id": flow.context_id,
                    "steps": [step_display(step) for step in flow.steps],
                    **(
                        {
                            "resolved_steps": resolved_evidence.get("steps", []),
                            "resolved_effects": resolved_evidence.get("effects", []),
                            "resolved_flow_graph": resolved_evidence.get("flow_graph", []),
                        }
                        if resolved_evidence is not None
                        else {}
                    ),
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
            mcp={"tool": "open_link_and_analyze", "arguments": {"uri": link.uri}},
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


# A screen's own words for "there is nothing here". Anchored on a word boundary so
# "notifications" and "notes" are not read as "no ...", which would cry wolf on half an app.
_EMPTY_STATE_RE = re.compile(
    r"\b(?:no\s+\w|nothing\s+(?:here|yet)|empty[\s_-]?state|emptystate|"
    r"is\s+empty|nothing\s+to\s+show)",
    re.IGNORECASE,
)


def empty_state_anchor(screen: ScreenRecord | None) -> str | None:
    """The anchor proving this screen was last seen showing an empty state, if any.

    Reads the screen's own recorded copy rather than inferring from a missing element: "no
    drafts" is the app stating the fact, and quoting it back is what lets an agent trust the
    claim without navigating there to check.
    """
    if screen is None:
        return None
    for anchor in screen.anchors:
        field, _, value = anchor.partition(":")
        if field not in {"tx", "cd", "id"} or not value:
            continue
        if _EMPTY_STATE_RE.search(value):
            return anchor
    return None



def inherited_device_state_warning(
    status_rows: Iterable[dict[str, Any]], serial: str | None, owner: str | None
) -> str | None:
    """Device changes this session is starting behind but did not make.

    These are *device* settings — a proxy, a moved clock, an enabled service. They outlive the
    process that set them and outlive the app, so force-stopping or reinstalling the app cannot
    clear one; only the registered undo can. Saying so at bootstrap turns an inherited proxy
    from a mystery into a named variable.

    A change this session recorded itself is skipped: ``session finish`` will undo it, and
    warning an agent about its own bookkeeping is noise it learns to ignore.
    """
    if not serial:
        return None
    kinds: list[str] = []
    for row in status_rows:
        if str(row.get("serial") or "") != serial:
            continue
        changes = row.get("changes")
        for change in changes if isinstance(changes, list) else []:
            if not isinstance(change, dict):
                continue
            change_owner = change.get("owner")
            if change_owner and owner and str(change_owner) == str(owner):
                continue
            kind = str(change.get("kind") or "").strip()
            if kind and kind not in kinds:
                kinds.append(kind)
    if not kinds:
        return None
    return (
        f"This device carries {len(kinds)} device change(s) another run left behind: "
        f"{', '.join(kinds)}. These are device settings, not app state — restarting or "
        "reinstalling the app does not clear them, so every observation is taken through "
        f"them until they are undone. Inspect with `aua teardown status`; clear with "
        f"`aua teardown run --serial {serial}` once you know no live run owns them."
    )


def _empty_state_target(
    goal: str, app: AppMap | None, candidates: Sequence[GoalCandidate]
) -> tuple[str, str] | None:
    """The screen this goal is about, and the anchor proving it was last seen empty.

    Prefers a ranked candidate's own target. Falls back to the mapped screen the goal names,
    because the incident this exists for had *no* verified route: a candidate-only check would
    have stayed silent for exactly the run that needed it. The fallback demands a shared word
    of four characters or more, so a one-letter overlap cannot conjure a warning.
    """
    if app is None:
        return None
    names = [candidate.target for candidate in candidates if candidate.target]
    if not names:
        terms = {term for term in _goal_terms(goal) if len(term) >= 4}
        scored = [
            (_match_score(goal, name, *(record.aliases or [])), name)
            for name, record in app.screens.items()
            if terms & set(_GOAL_WORD.findall(name.casefold()))
        ]
        if not scored:
            return None
        names = [max(scored)[1]]
    for name in names:
        anchor = empty_state_anchor(app.screens.get(name))
        if anchor is not None:
            return name, anchor
    return None


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
    resolved_flow_evidence: dict[str, dict[str, Any]] | None = None,
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
            context_id=context_id,
            destructive_labels=destructive_labels,
            mapped_screens=app.screens if app is not None else None,
            resolved_flow_evidence=resolved_flow_evidence,
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
    # The map may already know the destination is empty. Saying so here costs nothing and
    # saves an agent from reading "arrived, but nothing is here" as "navigated wrong" — the
    # mistake that turned one live run into twelve minutes of relaunches.
    empty_target = _empty_state_target(goal, app, candidates)
    if empty_target is not None:
        target_name, empty_anchor = empty_target
        warnings.append(
            f"The mapped screen {target_name!r} was last seen showing an empty state "
            f"({empty_anchor!r}). Navigating there will succeed and still show nothing. If "
            "the goal needs content, seed or create it first rather than re-navigating."
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
