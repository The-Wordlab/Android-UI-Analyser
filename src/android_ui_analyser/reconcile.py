"""Audit and externally-researched correction of persistent app maps.

AUA owns detection, validation, transactions, and rollback.  The caller owns the
research agent: ``reconcile plan`` emits questions and ``reconcile submit`` accepts the
agent's evidence and operations.  No model/provider is spawned from this module.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .atomic import atomic_write_text
from .memory import (
    LEGACY_CONTEXT_ID,
    AppMap,
    AppMemoryStore,
    KnowledgeEvidence,
    KnowledgeItem,
    KnowledgeScope,
    RouteEdge,
    ScreenRecord,
    _now_iso,
    _stable_id,
    anchor_similarity,
    slug,
)


class MapIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal[
        "poor_name",
        "stale_screen",
        "duplicate_screen",
        "route_conflict",
        "orphan_route",
        "legacy_context",
        "provisional_route",
        "unreplayable_route",
        "unverified_context",
    ]
    severity: Literal["info", "warning", "error"]
    message: str
    screen_ids: list[str] = Field(default_factory=list)
    route_ids: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class MapAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package: str
    app_version: str | None = None
    context_id: str | None = None
    generated_at: str
    issues: list[MapIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def summarize_audit(
    audit: MapAudit,
    *,
    research_tasks: list[ResearchTask] | None = None,
) -> dict[str, Any]:
    """Token-cheap map-health counts for agents; the canonical audit remains unchanged."""
    severity = Counter(issue.severity for issue in audit.issues)
    issue_types = Counter(issue.type for issue in audit.issues)
    tasks = research_tasks or []
    task_status = Counter(task.status for task in tasks)
    return {
        "package": audit.package,
        "app_version": audit.app_version,
        "context_id": audit.context_id,
        "generated_at": audit.generated_at,
        "ok": audit.ok,
        "issues": {
            "total": len(audit.issues),
            "by_severity": {
                "error": severity["error"],
                "warning": severity["warning"],
                "info": severity["info"],
            },
            "by_type": dict(sorted(issue_types.items(), key=lambda item: (-item[1], item[0]))),
        },
        "research_tasks": {
            "total": len(tasks),
            "open": task_status.get("open", 0),
            "by_status": dict(sorted(task_status.items())),
        },
    }


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    package: str
    app_version: str | None = None
    context_id: str | None = None
    flags: dict[str, str] = Field(default_factory=dict)
    issue_type: str
    affected_ids: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    created_at: str
    status: Literal["open", "submitted", "applied", "review", "rejected"] = "open"


class CorrectionOperation(BaseModel):
    """One agent-authored correction. Fields unused by the selected op are ignored."""

    model_config = ConfigDict(extra="forbid")
    op: Literal[
        "rename",
        "alias",
        "merge",
        "split",
        "set_variant",
        "set_state",
        "set_context",
        "route_guard",
        "route_replace",
        "route_delete",
        "route_verify",
        "route_reject",
        "knowledge_upsert",
        "mark_stale",
    ]
    screen_id: str | None = None
    screen_ids: list[str] = Field(default_factory=list)
    route_id: str | None = None
    target_screen_id: str | None = None
    value: str | bool | None = None
    from_screen: str | None = None
    to_screen: str | None = None
    action: str | None = None
    context_id: str | None = None
    guards: dict[str, str] = Field(default_factory=dict)
    knowledge: dict[str, object] = Field(default_factory=dict)


class ResearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    agent: str
    session: str | None = None
    verdict: Literal["apply", "review", "reject"]
    rationale: str
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)
    operations: list[CorrectionOperation] = Field(default_factory=list)
    knowledge: list[dict[str, object]] = Field(default_factory=list)


OutcomeCode = Literal[
    # applied
    "renamed",
    "confirmed",
    "settled_by_sibling",
    # skipped
    "unknown_task",
    "ambiguous_task",
    "duplicate_task",
    "not_open",
    "not_a_naming_question",
    "no_screen",
    "unknown_screen",
    "empty_value",
    "name_collision",
    "name_taken_in_batch",
    "name_freed_in_batch",
    "conflicting_answer",
    "operation_failed",
]


class TaskOutcome(BaseModel):
    """What one answered task in a correction event actually got, and why.

    A bulk correction cannot report itself with a single status: measured 2026-08-18, 496 open
    naming questions on one map named only 82 distinct screens, so 414 of them are siblings of a
    question another row already settled. One bad row must cost one row, so every input gets an
    outcome and every skip gets a code and a sentence the caller can act on.
    """

    model_config = ConfigDict(extra="forbid")
    task_id: str
    status: Literal["applied", "skipped"]
    code: OutcomeCode
    reason: str
    screen_id: str | None = None
    value: str | None = None
    name: str | None = None


class CorrectionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    package: str
    # A single-task event names its task here; a bulk event lists every task it closed in
    # `task_ids` and leaves this empty. `report` is likewise absent for a bulk event, which has
    # no one authoring report. Both keep defaults so the events already on disk still parse.
    task_id: str = ""
    task_ids: list[str] = Field(default_factory=list)
    applied_at: str
    report: ResearchReport | None = None
    agent: str = ""
    operations: list[CorrectionOperation]
    outcomes: list[TaskOutcome] = Field(default_factory=list)
    validation: list[str]
    before_snapshot: str
    after_snapshot: str
    rollback_id: str
    rolled_back_at: str | None = None


_POOR_NAME = re.compile(
    r"^(?:screen|home_\d+|\d[\d_]*|[a-z][a-z0-9_]*_\d+|just_once|yes_delete|"
    r"while_using_the_app|only_this_time)$|__[0-9a-f]{8}$"
)


def audit_map(app: AppMap, *, context_id: str | None = None) -> MapAudit:
    """Find structural ambiguity worth runtime/source research."""
    issues: list[MapIssue] = []
    screens = [
        rec
        for rec in app.screens.values()
        if context_id is None or rec.context_id in (context_id, LEGACY_CONTEXT_ID)
    ]
    routes = [
        route
        for route in app.routes
        if context_id is None or route.context_id in (context_id, LEGACY_CONTEXT_ID)
    ]
    for rec in screens:
        sid = rec.id or rec.name
        weak_source = rec.name_source in {"route", "activity", "legacy"}
        # KNOWN, not fixed: a screen whose `name_source` is already `explicit` still lands here
        # when its accepted name happens to match `_POOR_NAME`, and
        # `ask_about_current_screen` refuses to offer a question about an explicitly named screen —
        # so those rows can never be drained by any surface. Measured 2026-08-18: 20 such rows over
        # 8 screens on one real map, which the per-context merge then reduced to 3. Suppressing them
        # here is a one-line change, but `name_hint=` stamps `explicit`, and several tests build a
        # poorly-named screen exactly that way, so the fixture idiom has to be reinterpreted across
        # the suite first. Left visible rather than half-done.
        if _POOR_NAME.search(rec.name) or len(rec.name) < 2 or weak_source:
            issues.append(
                MapIssue(
                    id=_stable_id("issue", "poor_name", sid),
                    type="poor_name",
                    severity="warning",
                    message=f"Screen '{rec.name}' has a generated or action-derived name.",
                    screen_ids=[sid],
                    questions=[
                        "Which app destination or UI state does this screen represent?",
                        "Which source navigation constant, route, or stable resource id names it?",
                    ],
                )
            )
        if rec.stale:
            issues.append(
                MapIssue(
                    id=_stable_id("issue", "stale", sid),
                    type="stale_screen",
                    severity="warning",
                    message=f"Screen '{rec.name}' no longer matches its verified anchors.",
                    screen_ids=[sid],
                    questions=[
                        "Does this screen still exist in this app version and flag context?",
                        "Should it be refreshed, moved to another context, or removed?",
                    ],
                )
            )
    for index, left in enumerate(screens):
        for right in screens[index + 1 :]:
            if left.context_id != right.context_id:
                continue
            if (
                left.logical_name
                and left.logical_name == right.logical_name
                and left.state != right.state
            ):
                continue
            if left.surface and right.surface and left.surface != right.surface:
                continue
            if not left.anchors or not right.anchors:
                continue
            similarity = anchor_similarity(set(left.anchors), set(right.anchors))
            if similarity < 0.86:
                continue
            ids = [left.id or left.name, right.id or right.name]
            issues.append(
                MapIssue(
                    id=_stable_id("issue", "duplicate", *ids),
                    type="duplicate_screen",
                    severity="warning",
                    message=(
                        f"'{left.name}' and '{right.name}' share {similarity:.0%} of "
                        "weighted identity anchors."
                    ),
                    screen_ids=ids,
                    questions=[
                        "Are these the same destination, or distinct variants/states?",
                        "Which feature flags or source route distinguish them?",
                    ],
                )
            )
    grouped: dict[tuple[str, str, str], list[RouteEdge]] = defaultdict(list)
    for route in routes:
        if route.status == "provisional":
            issues.append(
                MapIssue(
                    id=_stable_id("issue", "provisional_route", route.id or route.action),
                    type="provisional_route",
                    severity="info",
                    message=(
                        f"Route '{route.action}' was observed once and is not used by goto yet."
                    ),
                    route_ids=[route.id or route.action],
                    questions=[
                        "Can this transition be replayed or observed a second time?",
                        "Does it always land on this target in the same flag context?",
                    ],
                )
            )
        elif route.status == "rejected":
            issues.append(
                MapIssue(
                    id=_stable_id("issue", "unreplayable_route", route.id or route.action),
                    type="unreplayable_route",
                    severity="warning",
                    message=(
                        f"Route '{route.action}' is excluded from navigation: "
                        f"{route.rejection_reason or 'unreplayable'}."
                    ),
                    route_ids=[route.id or route.action],
                    questions=[
                        "Which stable resource id, label, deeplink, or source route can replay it?"
                    ],
                )
            )
            continue
        grouped[(route.context_id, route.from_screen, route.action)].append(route)
        if route.from_screen not in app.screens or route.to_screen not in app.screens:
            issues.append(
                MapIssue(
                    id=_stable_id("issue", "orphan", route.id or route.action),
                    type="orphan_route",
                    severity="error",
                    message=(f"Route '{route.action}' references a missing source or destination."),
                    route_ids=[route.id or route.action],
                    questions=["Which valid source route should replace or remove this edge?"],
                )
            )
    for (route_context, source, action), edges in grouped.items():
        targets = {edge.to_screen for edge in edges}
        if len(targets) <= 1:
            continue
        issues.append(
            MapIssue(
                id=_stable_id("issue", "route_conflict", route_context, source, action),
                type="route_conflict",
                severity="warning",
                message=f"{source} --{action}--> has conflicting targets: {sorted(targets)}.",
                route_ids=[edge.id or edge.action for edge in edges],
                questions=[
                    "Which target is correct for this exact flag context?",
                    "Is the action conditional on state, flags, or app version?",
                ],
            )
        )
    if any(rec.context_id == LEGACY_CONTEXT_ID for rec in screens):
        issues.append(
            MapIssue(
                id=_stable_id("issue", "legacy", app.package),
                type="legacy_context",
                severity="info",
                message="Trusted legacy screens remain usable but are not flag-scoped.",
                screen_ids=[
                    rec.id or rec.name for rec in screens if rec.context_id == LEGACY_CONTEXT_ID
                ],
                questions=[
                    "Which legacy screens can be confirmed in an exact feature-flag context?"
                ],
            )
        )
    for context in app.contexts.values():
        if context_id is not None and context.id not in (context_id, LEGACY_CONTEXT_ID):
            continue
        if context.id == LEGACY_CONTEXT_ID or context.verified:
            continue
        issues.append(
            MapIssue(
                id=_stable_id("issue", "unverified_context", context.id),
                type="unverified_context",
                severity="warning",
                message=f"Feature context '{context.id}' has not been read back from the app.",
                questions=[
                    "Which runtime preference or source flag contract proves these values are active?"
                ],
            )
        )
    return MapAudit(
        package=app.package,
        app_version=app.app_version,
        context_id=context_id,
        generated_at=_now_iso(),
        issues=issues,
    )


def _match_tasks(app: AppMap, task_id: str) -> list[dict[str, Any]]:
    """Every task `task_id` could mean: the exact id if it exists, else open ids ending with it.

    Split out of `_resolve_task` so a bulk answer can decide what to do about none or many
    without catching an exception and re-parsing its message. The suffix branch only sees *open*
    tasks, which is why a batch has to resolve every row against the map as it was loaded: stamp
    statuses as you go and a tail that resolved on the first row resolves differently on the last.
    """
    wanted = task_id.strip()
    exact = [t for t in app.research_tasks if str(t.get("id")) == wanted]
    if exact:
        return exact[:1]
    return [
        t
        for t in app.research_tasks
        if t.get("status") == "open" and str(t.get("id")).endswith(wanted)
    ]


def _resolve_task(app: AppMap, task_id: str) -> dict[str, Any]:
    """Find an open task by its id, or by any unique suffix of it.

    The id is long and the caller is retyping it into an unrelated command; accepting the tail
    costs nothing and a wrong tail still fails loudly rather than answering the wrong question.
    """
    partial = _match_tasks(app, task_id)
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"unknown research task: {task_id}")
    raise ValueError(
        f"{task_id} matches {len(partial)} open tasks — pass more of the id: "
        + ", ".join(str(t.get("id")) for t in partial[:3])
    )


def validate_map(app: AppMap) -> list[str]:
    errors: list[str] = []
    record_names = [rec.name for rec in app.screens.values()]
    if len(record_names) != len(set(record_names)):
        errors.append("screen names must be unique")
    for key, rec in app.screens.items():
        if rec.name != key:
            errors.append(f"screen key/name mismatch: {key} != {rec.name}")
    screen_ids = [rec.id for rec in app.screens.values()]
    if any(not sid for sid in screen_ids):
        errors.append("every screen must have a stable id")
    if len(screen_ids) != len(set(screen_ids)):
        errors.append("screen ids must be unique")
    route_ids = [route.id for route in app.routes]
    if any(not rid for rid in route_ids):
        errors.append("every route must have a stable id")
    if len(route_ids) != len(set(route_ids)):
        errors.append("route ids must be unique")
    known_contexts = set(app.contexts)
    for rec in app.screens.values():
        if known_contexts and rec.context_id not in known_contexts:
            errors.append(f"screen {rec.id} has unknown context {rec.context_id}")
    for route in app.routes:
        if route.from_screen not in app.screens:
            errors.append(f"route {route.id} has missing source {route.from_screen}")
        if route.to_screen not in app.screens:
            errors.append(f"route {route.id} has missing target {route.to_screen}")
        if known_contexts and route.context_id not in known_contexts:
            errors.append(f"route {route.id} has unknown context {route.context_id}")
    return errors


def _question_identity(task: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """What makes two research tasks the same question: same kind, same targets.

    `questions` text is hard-coded per issue type, so two tasks that agree on both of these are
    asking a caller the identical thing in identical words.
    """
    return (
        str(task.get("issue_type")),
        tuple(sorted(str(i) for i in (task.get("affected_ids") or []))),
    )


# A settled question stays settled. Ranked so `min` prefers the most resolved row in a group:
# resurrecting an answered question as `open` is how already-named screens kept being asked about.
_STATUS_RANK = {"applied": 0, "rejected": 1, "review": 2, "submitted": 3, "open": 4}


def _dedupe_research_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per distinct question, however many contexts asked it.

    `plan(context_id=X)` mints `_stable_id(..., context_id or "all", issue.id)` and preserves the
    rows belonging to every *other* context, so one screen collects a task per flag context ever
    audited. A synthetic map with default and several flag contexts demonstrates how the backlog
    can mostly reflect audit frequency rather than distinct questions.

    The surviving row keeps the age the question really has, and becomes context-independent when
    it arrived from more than one context, so `ask_about_current_screen` can still offer it
    wherever the caller is standing.
    """
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    order: list[tuple[str, tuple[str, ...]]] = []
    passthrough: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict) or not task.get("id"):
            passthrough.append(task)
            continue
        key = _question_identity(task)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(task)

    merged: list[dict[str, Any]] = list(passthrough)
    for key in order:
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        winner = min(
            group,
            key=lambda task: (
                _STATUS_RANK.get(str(task.get("status")), len(_STATUS_RANK)),
                str(task.get("created_at") or ""),
                str(task.get("id")),
            ),
        )
        if len({task.get("context_id") for task in group}) > 1:
            winner = {**winner, "context_id": None}
        merged.append(winner)
    return merged


class ReconciliationStore:
    def __init__(self, store: AppMemoryStore) -> None:
        self.store = store

    def corrections_dir(self, package: str) -> Path:
        return self.store.app_dir(package) / "corrections"

    def plan(self, package: str, *, context_id: str | None = None) -> list[ResearchTask]:
        app = self.store.load(package) or AppMap(package=package)
        audit = audit_map(app, context_id=context_id)
        context = app.contexts.get(context_id or "")
        existing = {
            str(task.get("id")): task
            for task in app.research_tasks
            if isinstance(task, dict) and task.get("id")
        }
        tasks: list[ResearchTask] = []
        for issue in audit.issues:
            task_id = _stable_id("research", package, context_id or "all", issue.id)
            prior = existing.get(task_id, {})
            task = ResearchTask(
                id=task_id,
                package=package,
                app_version=app.app_version,
                context_id=context_id,
                flags=dict(context.flags) if context else {},
                issue_type=issue.type,
                affected_ids=[*issue.screen_ids, *issue.route_ids],
                observations=[issue.message],
                questions=issue.questions,
                created_at=str(prior.get("created_at") or _now_iso()),
                status=str(prior.get("status") or "open"),  # type: ignore[arg-type]
            )
            tasks.append(task)
        preserved = (
            [task for task in app.research_tasks if task.get("context_id") != context_id]
            if context_id is not None
            else []
        )
        merged = _dedupe_research_tasks(
            [*preserved, *(task.model_dump(mode="json") for task in tasks)]
        )
        app.research_tasks = merged
        self.store.save(app)
        # Return one row per audited issue, which after the merge may be an older surviving row
        # rather than the one just minted for this context.
        audited = {_question_identity(task.model_dump(mode="json")) for task in tasks}
        return [
            ResearchTask.model_validate(task)
            for task in merged
            if _question_identity(task) in audited
        ]

    def answer(
        self, package: str, task_id: str, value: str, *, agent: str = "cli"
    ) -> dict[str, object]:
        """Answer one open question inline, as a side effect of whatever the caller was doing.

        The agent standing on a screen is the one who knows what it is, and it is about to run
        another command anyway — so the answer rides along on that command instead of becoming
        a separate chore nobody ever does. Goes through `submit`, so an inline answer gets the
        same transaction, validation and rollback id as a researched one.
        """
        app = self.store.load(package)
        if app is None:
            raise ValueError(f"no map for {package}")
        task = _resolve_task(app, task_id)
        # Only a naming question can be settled by looking at the screen, and only a naming
        # question has exactly one screen to rename. Without this guard a `duplicate_screen`
        # task (two ids) silently renamed the first of the pair, and a route task raised an
        # error about a screen id that was really a route id.
        if task.get("issue_type") != "poor_name":
            raise ValueError(
                f"task {task.get('id')} is a {task.get('issue_type')} question, which "
                "`--answers` cannot settle; it takes research — see `aua reconcile plan`"
            )
        screen_ids = [str(i) for i in (task.get("affected_ids") or [])]
        if not screen_ids:
            raise ValueError(f"task {task.get('id')} names no screen to rename")
        report = ResearchReport(
            task_id=str(task.get("id")),
            agent=agent,
            verdict="apply",
            rationale=f"Answered inline by the agent on the screen: {value!r}.",
            operations=[CorrectionOperation(op="rename", screen_id=screen_ids[0], value=value)],
        )
        return self.submit(package, report)

    def answer_many(
        self,
        package: str,
        answers: Mapping[str, str] | Sequence[tuple[str, str]],
        *,
        agent: str = "cli",
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Answer many naming questions in one transaction, one snapshot pair, one rollback id.

        `answer` is correct but does not scale: every call writes a full before/after copy of the
        map. Measured 2026-08-18 on a 3.5 MB map, that is ~4.75 MB per answer and 54 MB for the
        twelve answers it had ever received; the 496 open naming questions would have cost ~2.4 GB
        to settle one at a time. They also name only 82 distinct screens, so 414 of those rows are
        siblings of a question another row already settles.

        So: task-keyed at the door, screen-keyed in the transaction. Every input gets an outcome,
        one bad row costs one row, and the whole batch is one undo. Caller order decides who wins
        a contested name — never whether a name is available.

        `dry_run` reports exactly what would happen and writes nothing. Draining a backlog wants
        partial success, but the inline `--answers` path promises the opposite — that a bad answer
        stops the command rather than quietly renaming something — so it previews first and only
        commits when every row is usable.
        """
        pairs: list[tuple[str, str]]
        if isinstance(answers, Mapping):
            pairs = [(str(k), str(v)) for k, v in answers.items()]
        else:
            pairs = [(str(k), str(v)) for k, v in answers]
        if not pairs:
            raise ValueError("answer_many needs at least one task_id=value pair")
        current = self.store.load(package)
        if current is None:
            raise ValueError(f"no map for {package}")

        outcomes: list[TaskOutcome] = []
        accepted: list[tuple[str, str, str, str]] = []
        seen: set[str] = set()

        def skip(
            task_id: str,
            code: OutcomeCode,
            reason: str,
            *,
            screen_id: str | None = None,
            value: str | None = None,
        ) -> None:
            outcomes.append(
                TaskOutcome(
                    task_id=task_id,
                    status="skipped",
                    code=code,
                    reason=reason,
                    screen_id=screen_id,
                    value=value,
                )
            )

        # Resolve every row against the map as loaded, before anything mutates it.
        for raw_id, raw_value in pairs:
            matches = _match_tasks(current, raw_id)
            if not matches:
                skip(raw_id, "unknown_task", f"unknown research task: {raw_id}", value=raw_value)
                continue
            if len(matches) > 1:
                ids = ", ".join(str(t.get("id")) for t in matches[:3])
                skip(
                    raw_id,
                    "ambiguous_task",
                    f"{raw_id} matches {len(matches)} open tasks — pass more of the id: {ids}",
                    value=raw_value,
                )
                continue
            task = matches[0]
            task_id = str(task.get("id"))
            if task_id in seen:
                skip(
                    task_id,
                    "duplicate_task",
                    "this task was already answered earlier in the batch",
                    value=raw_value,
                )
                continue
            seen.add(task_id)
            if task.get("status") != "open":
                skip(
                    task_id,
                    "not_open",
                    f"task is {task.get('status')}, not open",
                    value=raw_value,
                )
                continue
            if task.get("issue_type") != "poor_name":
                skip(
                    task_id,
                    "not_a_naming_question",
                    f"task {task_id} is a {task.get('issue_type')} question, which an answer "
                    "cannot settle; it takes research — see `aua reconcile plan`",
                    value=raw_value,
                )
                continue
            screen_ids = [str(i) for i in (task.get("affected_ids") or [])]
            if not screen_ids:
                skip(task_id, "no_screen", "task names no screen to rename", value=raw_value)
                continue
            proposed = slug(raw_value.strip())
            if not proposed:
                skip(
                    task_id,
                    "empty_value",
                    f"{raw_value!r} slugs to nothing",
                    screen_id=screen_ids[0],
                    value=raw_value,
                )
                continue
            accepted.append((task_id, screen_ids[0], raw_value, proposed))

        # Decide, screen-keyed. Still no mutation.
        by_id = {rec.id: name for name, rec in current.screens.items() if rec.id}
        taken: dict[str, str] = {}
        freed: dict[str, str] = {}
        renamed: dict[str, tuple[str, str]] = {}
        screen_of: dict[str, str] = {}
        queued: list[tuple[str, CorrectionOperation]] = []

        for task_id, screen_id, raw_value, proposed in accepted:
            name = by_id.get(screen_id) or (screen_id if screen_id in current.screens else None)
            if name is None:
                skip(
                    task_id,
                    "unknown_screen",
                    f"unknown screen: {screen_id} — it is no longer in the map",
                    screen_id=screen_id,
                    value=raw_value,
                )
                continue
            if screen_id in renamed:
                winner, applied = renamed[screen_id]
                if applied == proposed:
                    # The question is genuinely settled. Leaving it open would mint a zombie:
                    # the rename stamps `name_source = "explicit"`, which permanently stops
                    # `ask_about_current_screen` offering this screen again.
                    outcomes.append(
                        TaskOutcome(
                            task_id=task_id,
                            status="applied",
                            code="settled_by_sibling",
                            reason=f"same screen already named {applied!r} by {winner}",
                            screen_id=screen_id,
                            value=raw_value,
                            name=applied,
                        )
                    )
                else:
                    skip(
                        task_id,
                        "conflicting_answer",
                        f"{winner} already named this screen {applied!r}; this row says "
                        f"{proposed!r}. Left open — the disagreement needs deciding, not ordering",
                        screen_id=screen_id,
                        value=raw_value,
                    )
                continue
            if proposed == name:
                queued.append(
                    (
                        task_id,
                        CorrectionOperation(op="rename", screen_id=screen_id, value=raw_value),
                    )
                )
                renamed[screen_id] = (task_id, proposed)
                taken[proposed] = task_id
                screen_of[task_id] = screen_id
                outcomes.append(
                    TaskOutcome(
                        task_id=task_id,
                        status="applied",
                        code="confirmed",
                        reason=f"confirms the name the screen already has: {proposed!r}",
                        screen_id=screen_id,
                        value=raw_value,
                        name=proposed,
                    )
                )
                continue
            if proposed in taken:
                skip(
                    task_id,
                    "name_taken_in_batch",
                    f"{proposed!r} was claimed by {taken[proposed]} earlier in this batch; this "
                    "screen needs a distinguishing name",
                    screen_id=screen_id,
                    value=raw_value,
                )
                continue
            if proposed in freed:
                # Reserving vacated names makes availability independent of order. Without it a
                # later row could claim a name that now lives in an earlier screen's aliases,
                # where `_resolve_screen_name` silently shadows it.
                skip(
                    task_id,
                    "name_freed_in_batch",
                    f"{proposed!r} was vacated by {freed[proposed]} in this batch and is kept "
                    "reserved, because it is still that screen's alias",
                    screen_id=screen_id,
                    value=raw_value,
                )
                continue
            if proposed in current.screens:
                skip(
                    task_id,
                    "name_collision",
                    f"screen name already exists: {proposed}",
                    screen_id=screen_id,
                    value=raw_value,
                )
                continue
            queued.append(
                (task_id, CorrectionOperation(op="rename", screen_id=screen_id, value=raw_value))
            )
            renamed[screen_id] = (task_id, proposed)
            taken[proposed] = task_id
            freed[name] = task_id
            screen_of[task_id] = screen_id
            outcomes.append(
                TaskOutcome(
                    task_id=task_id,
                    status="applied",
                    code="renamed",
                    reason=f"renamed {name!r} to {proposed!r}",
                    screen_id=screen_id,
                    value=raw_value,
                    name=proposed,
                )
            )

        if not queued:
            # Writing a 4.8 MB snapshot pair for zero changes is the waste this method exists to
            # remove, and a rollback id that undoes nothing is a lie.
            return self._answer_many_summary(package, outcomes, event=None)

        if dry_run:
            return self._answer_many_summary(package, outcomes, event=None, dry_run=True)

        candidate = current.model_copy(deep=True)
        stub = ResearchReport(
            task_id="",
            agent=agent,
            verdict="apply",
            rationale=f"Bulk answer of {len(queued)} naming question(s) by {agent}.",
        )
        operations: list[CorrectionOperation] = []
        for task_id, operation in queued:
            try:
                self._apply_operation(candidate, operation, stub)
            except ValueError as err:
                # Unreachable if the pre-checks above model a rename completely; here so an
                # unmodelled interaction costs one screen rather than the whole batch.
                self._fail_row(outcomes, task_id, screen_of.get(task_id), str(err))
                continue
            operations.append(operation)

        applied_ids = [o.task_id for o in outcomes if o.status == "applied"]
        if not applied_ids:
            return self._answer_many_summary(package, outcomes, event=None)
        event = self._commit(
            package,
            current,
            candidate,
            event_key="bulk|" + "|".join(applied_ids),
            task_ids=applied_ids,
            operations=operations,
            outcomes=outcomes,
            agent=agent,
            report=None,
        )
        return self._answer_many_summary(package, outcomes, event=event)

    @staticmethod
    def _fail_row(
        outcomes: list[TaskOutcome], task_id: str, screen_id: str | None, reason: str
    ) -> None:
        """Flip a queued row to skipped, and any sibling that was settled by it."""
        for index, outcome in enumerate(outcomes):
            if outcome.status != "applied":
                continue
            same = outcome.task_id == task_id or (
                screen_id is not None
                and outcome.screen_id == screen_id
                and outcome.code == "settled_by_sibling"
            )
            if same:
                outcomes[index] = outcome.model_copy(
                    update={"status": "skipped", "code": "operation_failed", "reason": reason}
                )

    @staticmethod
    def _answer_many_summary(
        package: str,
        outcomes: list[TaskOutcome],
        *,
        event: CorrectionEvent | None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Counts first, detail on disk — a 496-row batch must not become an MCP response."""
        codes = Counter(o.code for o in outcomes)
        skipped = [o for o in outcomes if o.status == "skipped"]
        shown = skipped[:50]
        would = [o for o in outcomes if o.status == "applied"]
        return {
            "status": ("dry_run" if dry_run else "applied" if event is not None else "rejected"),
            "would_apply": len(would) if dry_run else None,
            "package": package,
            "rollback_id": event.rollback_id if event else None,
            "event_id": event.id if event else None,
            "event": event.before_snapshot.replace(".before.json", ".event.json")
            if event
            else None,
            "answered": len(outcomes),
            "renamed": codes["renamed"],
            "confirmed": codes["confirmed"],
            "settled": codes["settled_by_sibling"],
            "skipped": len(skipped),
            "skipped_answers": [
                {"task_id": o.task_id, "code": o.code, "reason": o.reason} for o in shown
            ],
            "skipped_truncated": len(skipped) - len(shown),
        }

    def submit(self, package: str, report: ResearchReport) -> dict[str, object]:
        app = self.store.load(package)
        if app is None:
            raise ValueError(f"no map for {package}")
        task = next(
            (item for item in app.research_tasks if item.get("id") == report.task_id),
            None,
        )
        if task is None:
            raise ValueError(f"unknown research task: {report.task_id}")
        if report.verdict == "apply":
            # Nothing is written until `apply` validates, because `apply` is where every
            # operation is checked and it used to run AFTER this method had already stamped
            # `status: submitted` and saved. A rejected correction therefore left the task no
            # longer `open`, so it was never offered again and one mistyped answer retired the
            # question permanently — `apply` only rolls back its own save. It marks the task
            # `applied` and drops the pending report itself, so this path writes nothing.
            event = self.apply(package, report)
            return {"status": "applied", "event": event.model_dump(mode="json")}
        task["status"] = report.verdict
        app.pending_reports = [
            item for item in app.pending_reports if item.get("task_id") != report.task_id
        ]
        app.pending_reports.append(report.model_dump(mode="json"))
        self.store.save(app)
        return {"status": report.verdict, "task_id": report.task_id}

    def apply(self, package: str, report: ResearchReport) -> CorrectionEvent:
        current = self.store.load(package)
        if current is None:
            raise ValueError(f"no map for {package}")
        candidate = current.model_copy(deep=True)
        for operation in report.operations:
            self._apply_operation(candidate, operation, report)
        for knowledge in report.knowledge:
            self._apply_knowledge(candidate, knowledge, report)
        return self._commit(
            package,
            current,
            candidate,
            event_key=report.task_id,
            task_ids=[report.task_id],
            operations=report.operations,
            outcomes=[
                TaskOutcome(
                    task_id=report.task_id,
                    status="applied",
                    code="renamed",
                    reason="single-task correction",
                )
            ],
            agent=report.agent,
            report=report,
        )

    def _commit(
        self,
        package: str,
        current: AppMap,
        candidate: AppMap,
        *,
        event_key: str,
        task_ids: list[str],
        operations: list[CorrectionOperation],
        outcomes: list[TaskOutcome],
        agent: str,
        report: ResearchReport | None = None,
    ) -> CorrectionEvent:
        """Write one correction transaction: validate, snapshot, stamp, save, record.

        Shared by the single-answer path and the bulk one so there is exactly one place that
        decides what a correction is allowed to do and exactly one snapshot pair per transaction.
        `event_key` is whatever makes this event's id stable — a task id for a single answer, the
        joined applied ids for a batch.
        """
        # A correction has to leave the map no worse than it found it — not find it perfect.
        # Validating the candidate absolutely meant one stale row vetoed every unrelated
        # correction, and since corrections are the only way to repair the map, the repair
        # mechanism was locked shut by the damage it exists to repair. Measured 2026-08-10: one
        # dangling route in 623 blocked all 210 open research tasks, and a rename of an
        # unrelated screen was refused with that route's id as the reason.
        errors = validate_map(candidate)
        inherited = set(validate_map(current))
        introduced = [e for e in errors if e not in inherited]
        if introduced:
            raise ValueError("correction rejected: " + "; ".join(introduced))
        carried = [e for e in errors if e in inherited]

        now = _now_iso()
        directory = self.corrections_dir(package)
        directory.mkdir(parents=True, exist_ok=True)
        # `_now_iso` is second-resolution, so an identical batch replayed inside the same second
        # would derive the same id and overwrite a live snapshot pair — silently destroying the
        # rollback point it was meant to create. A batch event is worth every rename in it, so
        # step aside rather than clobber.
        event_id = _stable_id("correction", package, event_key, now)
        suffix = 1
        while (directory / f"{event_id}.event.json").exists():
            suffix += 1
            event_id = f"{_stable_id('correction', package, event_key, now)}-{suffix}"
        before = directory / f"{event_id}.before.json"
        after = directory / f"{event_id}.after.json"
        event_path = directory / f"{event_id}.event.json"
        self._atomic_json(before, current.model_dump(mode="json"))
        closed = set(task_ids)
        for task in candidate.research_tasks:
            if task.get("id") in closed:
                task["status"] = "applied"
        candidate.pending_reports = [
            item for item in candidate.pending_reports if item.get("task_id") not in closed
        ]
        self._atomic_json(after, candidate.model_dump(mode="json"))
        try:
            self.store.save(candidate)
        except Exception:
            self.store.save(current)
            raise
        event = CorrectionEvent(
            id=event_id,
            package=package,
            task_id=report.task_id if report is not None else "",
            task_ids=list(task_ids),
            applied_at=now,
            report=report,
            agent=agent,
            operations=operations,
            outcomes=outcomes,
            # What was actually checked, not a fixed claim: this correction introduced no new
            # violation. Anything the map already had is carried here so it stays visible —
            # silently inheriting damage is how one dangling route survived 623 routes.
            validation=["introduced no new violation", *(f"pre-existing: {e}" for e in carried)],
            before_snapshot=str(before),
            after_snapshot=str(after),
            rollback_id=event_id,
        )
        self._atomic_json(event_path, event.model_dump(mode="json"))
        return event

    def rollback(self, package: str, rollback_id: str) -> CorrectionEvent:
        directory = self.corrections_dir(package)
        event_path = directory / f"{rollback_id}.event.json"
        before = directory / f"{rollback_id}.before.json"
        if not event_path.is_file() or not before.is_file():
            raise ValueError(f"unknown rollback id: {rollback_id}")
        event = CorrectionEvent.model_validate_json(event_path.read_text(encoding="utf-8"))
        restored = AppMap.model_validate_json(before.read_text(encoding="utf-8"))
        # A snapshot is by definition a state this map was already in, so a rollback never refuses
        # to return to it. Validating it absolutely made the undo unavailable on exactly the maps
        # that need one: measured 2026-08-18, all twelve corrections on a 1005-route map were
        # un-rollbackable because every `before` snapshot carried the same two violations the live
        # map had. Inherited damage is carried onto the event, the way `apply` carries it, so it
        # stays visible rather than becoming invisible.
        errors = validate_map(restored)
        self.store.save(restored)
        event.rolled_back_at = _now_iso()
        event.validation = [
            *event.validation,
            *(f"restored with pre-existing: {e}" for e in errors),
        ]
        self._atomic_json(event_path, event.model_dump(mode="json"))
        return event

    def status(self, package: str) -> dict[str, object]:
        app = self.store.load(package) or AppMap(package=package)
        directory = self.corrections_dir(package)
        events = (
            sorted(path.stem.removesuffix(".event") for path in directory.glob("*.event.json"))
            if directory.is_dir()
            else []
        )
        return {
            "package": package,
            "tasks": app.research_tasks,
            "reports": app.pending_reports,
            "events": events,
        }

    def _screen(self, app: AppMap, screen_id: str | None) -> tuple[str, ScreenRecord]:
        if not screen_id:
            raise ValueError("operation requires screen_id")
        for name, rec in app.screens.items():
            if rec.id == screen_id or name == screen_id:
                return name, rec
        raise ValueError(f"unknown screen: {screen_id}")

    def _route(self, app: AppMap, route_id: str | None) -> RouteEdge:
        if not route_id:
            raise ValueError("operation requires route_id")
        route = next((edge for edge in app.routes if edge.id == route_id), None)
        if route is None:
            raise ValueError(f"unknown route: {route_id}")
        return route

    def _apply_operation(
        self, app: AppMap, operation: CorrectionOperation, report: ResearchReport
    ) -> None:
        op = operation.op
        if op == "rename":
            old, rec = self._screen(app, operation.screen_id)
            new = slug(str(operation.value or ""))
            if not new:
                raise ValueError("rename requires a non-empty value")
            if new == old:
                # Confirming the name a screen already has is still an answer: stamp the source
                # so the question stops being asked, but do not pop and re-insert the record
                # (which aliased the screen to itself) or sweep every route for a rename that is
                # not happening.
                rec.canonical_name = new
                rec.name_source = "explicit"
                return
            if new in app.screens and new != old:
                raise ValueError(f"screen name already exists: {new}")
            app.screens.pop(old)
            if old not in rec.aliases:
                rec.aliases.append(old)
            rec.name = new
            rec.canonical_name = new
            rec.name_source = "explicit"
            app.screens[new] = rec
            for route in app.routes:
                if route.from_screen == old:
                    route.from_screen = new
                if route.to_screen == old:
                    route.to_screen = new
            return
        if op == "alias":
            _, rec = self._screen(app, operation.screen_id)
            alias = slug(str(operation.value or ""))
            if alias and alias not in rec.aliases:
                rec.aliases.append(alias)
            return
        if op == "merge":
            target_name, target = self._screen(app, operation.target_screen_id)
            for sid in operation.screen_ids:
                source_name, source = self._screen(app, sid)
                if source_name == target_name:
                    continue
                target.aliases = sorted(set(target.aliases + source.aliases + [source_name]))
                target.anchors = sorted(set(target.anchors) | set(source.anchors))
                target.key_elements.extend(
                    element for element in source.key_elements if element not in target.key_elements
                )
                for route in app.routes:
                    if route.from_screen == source_name:
                        route.from_screen = target_name
                    if route.to_screen == source_name:
                        route.to_screen = target_name
                app.screens.pop(source_name)
            app.routes = [route for route in app.routes if route.from_screen != route.to_screen]
            return
        if op == "split":
            _, source = self._screen(app, operation.screen_id)
            new_name = slug(str(operation.value or ""))
            if not new_name:
                raise ValueError("split requires the new screen name in value")
            if new_name in app.screens:
                raise ValueError(f"screen name already exists: {new_name}")
            clone = source.model_copy(deep=True)
            clone.name = new_name
            clone.canonical_name = new_name
            clone.logical_name = source.logical_name or source.canonical_name or source.name
            clone.context_id = operation.context_id or source.context_id
            clone.variant = operation.context_id or new_name
            clone.id = _stable_id("screen", app.package, clone.context_id, new_name, _now_iso())
            clone.aliases = []
            app.screens[new_name] = clone
            return
        if op in {"set_variant", "set_state", "set_context", "mark_stale"}:
            _, rec = self._screen(app, operation.screen_id)
            if op == "set_variant":
                rec.variant = str(operation.value) if operation.value is not None else None
            elif op == "set_state":
                rec.state = str(operation.value) if operation.value is not None else None
            elif op == "set_context":
                rec.context_id = operation.context_id or str(operation.value)
            else:
                rec.stale = bool(operation.value if operation.value is not None else True)
            return
        if op == "route_delete":
            route = self._route(app, operation.route_id)
            app.routes.remove(route)
            return
        if op in {"route_verify", "route_reject"}:
            route = self._route(app, operation.route_id)
            if op == "route_verify":
                route.status = "verified"
                route.verification_count += 1
                route.rejection_reason = None
            else:
                route.status = "rejected"
                route.rejection_reason = str(operation.value or "rejected by research")
            return
        if op in {"route_guard", "route_replace"}:
            route = self._route(app, operation.route_id)
            if op == "route_guard":
                route.guards = dict(operation.guards)
                return
            route.from_screen = operation.from_screen or route.from_screen
            route.to_screen = operation.to_screen or route.to_screen
            route.action = operation.action or route.action
            route.context_id = operation.context_id or route.context_id
            route.guards = operation.guards or route.guards
            return
        if op == "knowledge_upsert":
            self._apply_knowledge(app, operation.knowledge, report)
            return
        raise ValueError(f"unsupported correction operation: {op}")

    def _apply_knowledge(self, app: AppMap, raw: dict[str, object], report: ResearchReport) -> None:
        text = str(raw.get("text") or "").strip()
        if not text:
            raise ValueError("knowledge_upsert requires text")
        kind = str(raw.get("kind") or "claim")
        name = str(raw["name"]) if raw.get("name") is not None else None
        context_id = str(raw["context_id"]) if raw.get("context_id") is not None else None
        raw_flags = raw.get("flags")
        flags = raw_flags if isinstance(raw_flags, dict) else {}
        kid = _stable_id("knowledge", app.package, kind, name or "", text)
        existing = next((item for item in app.knowledge if item.id == kid), None)
        if existing is not None:
            existing.status = "accepted"
            existing.evidence = report.evidence
            existing.last_verified = _now_iso()
            return
        app.knowledge.append(
            KnowledgeItem(
                id=kid,
                kind=kind,  # type: ignore[arg-type]
                text=text,
                name=name,
                scope=KnowledgeScope(
                    package=app.package,
                    app_version=app.app_version,
                    context_id=context_id,
                    flags={str(key): str(value) for key, value in flags.items()},
                ),
                source="agent",
                agent=report.agent,
                session=report.session,
                evidence=report.evidence,
                created_at=_now_iso(),
                last_verified=_now_iso(),
            )
        )

    @staticmethod
    def _atomic_json(path: Path, payload: object) -> None:
        atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
