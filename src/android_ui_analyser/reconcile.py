"""Audit and externally-researched correction of persistent app maps.

AUA owns detection, validation, transactions, and rollback.  The caller owns the
research agent: ``reconcile plan`` emits questions and ``reconcile submit`` accepts the
agent's evidence and operations.  No model/provider is spawned from this module.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class CorrectionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    package: str
    task_id: str
    applied_at: str
    report: ResearchReport
    operations: list[CorrectionOperation]
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
            [
                task
                for task in app.research_tasks
                if task.get("context_id") != context_id
            ]
            if context_id is not None
            else []
        )
        app.research_tasks = [
            *preserved,
            *(task.model_dump(mode="json") for task in tasks),
        ]
        self.store.save(app)
        return tasks

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
        task["status"] = report.verdict if report.verdict != "apply" else "submitted"
        app.pending_reports = [
            item for item in app.pending_reports if item.get("task_id") != report.task_id
        ]
        app.pending_reports.append(report.model_dump(mode="json"))
        self.store.save(app)
        if report.verdict == "apply":
            event = self.apply(package, report)
            return {"status": "applied", "event": event.model_dump(mode="json")}
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
        errors = validate_map(candidate)
        if errors:
            raise ValueError("correction rejected: " + "; ".join(errors))

        now = _now_iso()
        event_id = _stable_id("correction", package, report.task_id, now)
        directory = self.corrections_dir(package)
        directory.mkdir(parents=True, exist_ok=True)
        before = directory / f"{event_id}.before.json"
        after = directory / f"{event_id}.after.json"
        event_path = directory / f"{event_id}.event.json"
        self._atomic_json(before, current.model_dump(mode="json"))
        for task in candidate.research_tasks:
            if task.get("id") == report.task_id:
                task["status"] = "applied"
        candidate.pending_reports = [
            item for item in candidate.pending_reports if item.get("task_id") != report.task_id
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
            task_id=report.task_id,
            applied_at=now,
            report=report,
            operations=report.operations,
            validation=["stable ids unique", "all route endpoints exist"],
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
        errors = validate_map(restored)
        if errors:
            raise ValueError("snapshot is invalid: " + "; ".join(errors))
        self.store.save(restored)
        event.rolled_back_at = _now_iso()
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
