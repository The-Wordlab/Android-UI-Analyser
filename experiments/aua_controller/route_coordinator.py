"""Host-approved flow dispatch plus advisory AUA route knowledge.

The caller owns the session, phase scope, approval and evidence. Discovery never
promotes a remembered route to an executable candidate. Native AUA still owns
flow preflight, actions, arrival checks and divergence. This is not a planner.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.aua_controller.hosted_projection import hosted_model_view

CallTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
MAX_ITEMS = 16
MAX_TEXT = 320
MAX_FLOW_BYTES = 1_048_576
MAX_VIEW_BYTES = 7_500
MAX_KNOWLEDGE_BYTES = 5_000


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return str(hosted_model_view(value))[:MAX_TEXT]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value[:MAX_ITEMS] if (text := _text(item)) is not None]


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _bounded_knowledge(value: dict[str, Any], budget: int) -> dict[str, Any]:
    """Trim advisory detail, keeping honest availability/counts and truncation."""
    if _size(value) <= budget:
        return value
    value["truncated"] = True
    value.setdefault("truncated_fields", [])
    while _size(value) > budget:
        choices = []
        for key, child in value.items():
            if isinstance(child, list) and child and key != "truncated_fields":
                choices.append((_size(child), key, value, key))
        for key, child in value.get("orient", {}).items():
            if isinstance(child, (list, str)) and child:
                choices.append((_size(child), "orient." + key, value["orient"], key))
        if not choices:
            break  # Scalar provenance is bounded independently of tool output.
        _, field, owner, key = max(choices, key=lambda row: row[0])
        if isinstance(owner[key], list):
            owner[key].pop()
        else:
            del owner[key]
        if field not in value["truncated_fields"]:
            value["truncated_fields"].append(field)
    return value


def _byte_text(value: str, limit: int) -> str:
    return (_text(value) or "").encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    out = {key: text for key in ("name", "app", "description", "arrival_screen", "arrival_status")
           if (text := _text(row.get(key))) is not None}
    if type(row.get("steps")) is int and row["steps"] >= 0:
        out["steps"] = row["steps"]
    if row.get("context_compatible") in (True, False, None):
        out["context_compatible"] = row.get("context_compatible")
    out["params"] = _strings(row.get("params"))
    out["execution_approved"] = False
    return out


def _read(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_FLOW_BYTES + 1)
    if not data or len(data) > MAX_FLOW_BYTES:
        raise ValueError("approved flow must be nonempty and at most 1 MiB")
    data.decode("utf-8")
    return data


@dataclass(frozen=True)
class _Approved:
    label: str
    source: Path
    resolved: Path
    sha256: str
    params: tuple[tuple[str, str], ...] = ()


class RouteCoordinator:
    """A finite catalogue whose opaque IDs confer no path/argument authority."""

    def __init__(self, app: str, approved: dict[str, _Approved], knowledge: dict[str, Any]):
        self.app = app
        self._approved = approved
        self._knowledge = knowledge

    def model_view(self) -> dict[str, Any]:
        # Rebuild a detached projection; a consumer cannot mutate the approval map.
        out = {
            "app": _byte_text(self.app, 256),
            "policy": (
                "Remembered knowledge is advisory, not permission to navigate. Only the "
                "host-approved candidate IDs below may execute, within the caller's current "
                "phase. No arbitrary route, file, parameters, reset or assistance is accepted."
            ),
            "candidates": [{"candidate_id": candidate_id, "label": _byte_text(item.label, 160),
                            "execution_approved": True}
                           for candidate_id, item in self._approved.items()],
            "knowledge": {},
        }
        knowledge_budget = min(MAX_KNOWLEDGE_BYTES, MAX_VIEW_BYTES - _size(out))
        out["knowledge"] = _bounded_knowledge(hosted_model_view(self._knowledge), knowledge_budget)
        return out

    async def execute(self, candidate_id: str, call_tool: CallTool) -> dict[str, Any]:
        """Validate a pinned file immediately before native, unassisted execution.

        The pin covers this approved file. AUA's own preflight resolves and freezes
        any nested flow dependencies; approval of those remains the caller's duty.
        No retry, resumption or replacement is performed here.
        """
        candidate = self._approved.get(candidate_id) if isinstance(candidate_id, str) else None
        if candidate is None:
            return {"ok": False, "executed": False, "code": "unknown_route_candidate"}
        try:
            unchanged = candidate.source.resolve(strict=True) == candidate.resolved
            unchanged = unchanged and hashlib.sha256(_read(candidate.resolved)).hexdigest() == candidate.sha256
        except (OSError, ValueError, UnicodeError):
            unchanged = False
        if not unchanged:
            return {"ok": False, "executed": False, "code": "approved_flow_changed",
                    "candidate_id": candidate_id}
        # flow_run's public default allows destructive actions: always override it.
        # assist is a supported public parameter; allow_unsafe is not.
        arguments: dict[str, Any] = {"file": str(candidate.resolved),
                                     "allow_destructive": False, "assist": False}
        if candidate.params:
            arguments["params"] = dict(candidate.params)
        result = await call_tool("flow_run", arguments)
        return {"candidate_id": candidate_id, "source_sha256": candidate.sha256,
                "executed": True, "ok": result.get("ok") is True, "result": result}

    async def draft(self, call_tool: CallTool, session_id: str) -> dict[str, Any]:
        """Preview only; return a LOCAL descriptor, never an executable candidate.

        Native session_candidate_flow requires completed authored checkpoints,
        correlated capture provenance and one app/context segment. replay=False /
        save=False performs no device actions or flow-library promotion, but AUA
        may write candidate-flow.yaml into the session's host artifact directory.
        A separately approved reset plus successful replay is required to save.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("preview requires an explicit session ID")
        result = await call_tool("session_candidate_flow", {
            "name": "candidate-" + secrets.token_hex(8), "session_id": session_id,
            "replay": False, "save": False,
        })
        preview_ok = result.get("ok") is True and result.get("replayed") is False and result.get("saved") is False
        return {"ok": preview_ok, "kind": "local_unapproved_candidate_preview",
                "execution_approved": False, "verified": False,
                "requires": "Independent scope review, explicit reset flow and successful replay before saving.",
                "native_preview": result}


async def discover(call_tool: CallTool, *, app: str,
                   approved_flows: dict[str, Path],
                   flow_parameters: dict[str, dict[str, str]] | None = None) -> RouteCoordinator:
    """Pin caller approvals, then read bounded native orient/flow_list knowledge.

    Keys are caller-authored display labels, not executable native flow names.
    Empty approvals are valid and discovery cannot populate them from memory.
    Tool exceptions remain terminal for the caller's lifecycle to handle.
    """
    if not isinstance(app, str) or not app.strip():
        raise ValueError("route discovery requires an explicit app")
    if len(approved_flows) > MAX_ITEMS:
        raise ValueError(f"at most {MAX_ITEMS} approved flows may be offered per phase")
    parameters = flow_parameters or {}
    if set(parameters) - set(approved_flows):
        raise ValueError("flow parameter bindings must name an approved flow")
    approved = {}
    for label, source in approved_flows.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("approved flow labels must be nonempty strings")
        source = Path(source).expanduser().absolute()
        resolved = source.resolve(strict=True)
        values = parameters.get(label, {})
        if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                  for k, v in values.items()):
            raise ValueError("flow parameter bindings must be string mappings")
        approved["route-" + secrets.token_hex(8)] = _Approved(
            label, source, resolved, hashlib.sha256(_read(resolved)).hexdigest(), tuple(sorted(values.items())))

    orient = await call_tool("orient", {})
    listed = await call_tool("flow_list", {"app": app})
    knowledge: dict[str, Any] = {"execution_approved": False}
    # orient describes the CURRENT foreground app, which may differ at setup.
    if orient.get("package") == app and orient.get("ok") is not False and not orient.get("error"):
        known = {"known": orient.get("known") is True}
        for key in ("screens", "routes"):
            if type(orient.get(key)) is int and orient[key] >= 0:
                known[key] = orient[key]
        for key in ("suggested_gotos", "research_tasks", "notes"):
            known[key] = _strings(orient.get(key))
        if (description := _text(orient.get("description"))) is not None:
            known["description"] = description
        recipes = orient.get("recipes")
        if isinstance(recipes, dict):
            known["recipes"] = [{"name": _text(name), "note": _text(note)}
                                for name, note in list(recipes.items())[:MAX_ITEMS]]
        # URI strings can contain query secrets and are not approved entry routes.
        if isinstance(orient.get("deeplinks"), list):
            known["remembered_deeplink_count"] = len(orient["deeplinks"])
        knowledge["orient"] = known
    else:
        knowledge["orient"] = {"available": False, "reason": "unavailable_or_other_foreground_app"}
    rows = listed.get("flows")
    if (listed.get("app") == app and isinstance(rows, list)
            and listed.get("ok") is not False and not listed.get("error")):
        knowledge["known_flows"] = [_summary(row) for row in rows[:MAX_ITEMS]
                                    if isinstance(row, dict)]
        knowledge["known_flows_total"] = len(rows)
        if len(rows) > MAX_ITEMS:
            knowledge["truncated"] = True
            knowledge["truncated_fields"] = ["known_flows"]
    else:
        knowledge["known_flows"] = []
        knowledge["flow_list_available"] = False
    return RouteCoordinator(app, approved, knowledge)
