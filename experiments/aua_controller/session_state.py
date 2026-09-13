"""Single-writer host session memory; claims are never authoritative QA verdicts.

Raw observations and native/model reasoning belong in their existing evidence
archives. This ledger keeps only bounded summaries, hashes and trace references.
The JSONL journal is durable history; the atomic JSON file is a reload snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT = "aua-session-state-v1"
RECENT_LIMIT = 12
MAX_ATTEMPTS = 3
CHECK_STATUSES = {"pending", "claimed_verified", "claimed_failed", "not_verified"}
WRAPPERS = ("observation", "result", "data", "state", "structuredContent")
READ_TOOLS = {
    "initial_observation", "analyze", "analyze_screen", "observe", "orient", "has", "inspect",
    "capture_evidence", "screenshot", "system_appearance", "capabilities", "policy_status",
    "session_progress", "session_review", "session_context", "map_find", "flow_list",
    "verify_account_tier", "source_tool_link_facts", "record_persona", "update_checks", "known_routes",
}
ELEMENT_FIELDS = (
    "id", "parent", "stable_key", "type", "text", "content_desc", "desc", "resource_id", "rid",
    "bounds", "clickable", "enabled", "focused", "checkable", "checked", "selected",
    "scrollable", "long_clickable", "password", "window",
)
SCREEN_FIELDS = ("width", "height", "package", "activity", "app_id", "surface_id")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def observation_frame(value: Any) -> dict | None:
    """Resolve one fresh full frame, never search unrelated history/arrays.

    A failure may include a useful current observation. Contradictory wrappers,
    stale flags, missing geometry/fingerprint, and multiple different frames are
    not current proof. Returned data is copied, not modified.
    """
    frames = []
    invalid = False

    def visit(item, depth=0):
        nonlocal invalid
        if not isinstance(item, dict) or depth > 6:
            return
        contract = item.get("observation_contract")
        if (item.get("stale_risk") is True or item.get("stale") is True or item.get("fresh") is False
                or item.get("observation_present") is False
                or isinstance(contract, dict) and (contract.get("reusable") is False
                                                  or contract.get("stale_risk") is True
                                                  or contract.get("stale") is True or contract.get("fresh") is False)):
            invalid = True
            return
        # Native flow results keep a legacy elements-only summary alongside the
        # full final observation. That summary is not a second frame.
        if "screen" in item or ("elements" in item and not isinstance(item.get("observation"), dict)):
            screen, elements, meta = item.get("screen"), item.get("elements"), item.get("meta")
            if (not isinstance(screen, dict) or not isinstance(elements, list)
                    or not all(type(screen.get(k)) is int and screen[k] > 0 for k in ("width", "height"))
                    or not all(isinstance(e, dict) and isinstance(e.get("id"), (str, int))
                               and not isinstance(e.get("id"), bool)
                               and ("type" not in e or isinstance(e["type"], str))
                               and ("bounds" not in e or isinstance(e["bounds"], list)
                                    and len(e["bounds"]) == 4 and all(type(n) is int for n in e["bounds"]))
                               for e in elements)
                    or not isinstance(meta, dict) or not isinstance(meta.get("fingerprint"), str)
                    or not meta["fingerprint"].strip() or meta.get("stale_risk") is True
                    or meta.get("stale") is True or meta.get("fresh") is False):
                invalid = True
                return
            frames.append(item)
        for key in WRAPPERS:
            if isinstance(item.get(key), dict):
                visit(item[key], depth + 1)
        error = item.get("error")
        if isinstance(error, dict) and error.get("observation_present") is True:
            visit(error, depth + 1)

    visit(value)
    if invalid or not frames:
        return None
    # Identical duplicated wrappers are harmless; disagreement is not resolved by
    # choosing whichever dictionary happened to come first.
    try:
        if len({_digest({"screen": f["screen"], "elements": f["elements"],
                         "fingerprint": f["meta"]["fingerprint"]}) for f in frames}) != 1:
            return None
    except (TypeError, ValueError):
        return None
    return copy.deepcopy(frames[0])


def _meaningful(tool):
    return tool not in READ_TOOLS and not tool.startswith(("wait", "await", "submit", "get_", "list_", "read_"))


# Compatibility aliases for callers that previously used a private frame helper.
_frame = observation_frame
current_frame = observation_frame


def _action_key(tool, arguments, bindings):
    normalized = copy.deepcopy(arguments)
    handle = normalized.get("id")
    if isinstance(handle, (str, int)) and str(handle) in bindings:
        # Only a unique element observed in this frame can bind a new ephemeral
        # handle to the same action. This never changes the actual call arguments.
        normalized["id"] = {"observed_element": bindings[str(handle)]}
    return _digest({"tool": tool, "arguments": normalized})


def _frame_summary(frame, ref, sequence):
    positions = {e["id"]: i for i, e in enumerate(frame["elements"])}
    elements = []
    for element in frame["elements"]:
        semantic_element = {k: element[k] for k in ELEMENT_FIELDS
                            if k not in {"id", "parent", "stable_key"} and k in element}
        if element.get("parent") is not None:
            semantic_element["parent_index"] = positions.get(element["parent"], "unresolved")
        elements.append(semantic_element)
    identities = [_digest(e) for e in elements]
    counts = Counter(identities)
    bindings = {str(e["id"]): identity for e, identity in zip(frame["elements"], identities, strict=True)
                if counts[identity] == 1}
    semantic = {"screen": {k: frame["screen"][k] for k in SCREEN_FIELDS if k in frame["screen"]},
                "elements": elements}
    labels = [e.get("text") or e.get("content_desc") for e in frame["elements"]]
    return {
        "signature": _digest(semantic), "source_fingerprint": frame["meta"]["fingerprint"][:256],
        "surface": {k: v[:160] if isinstance(v, str) else v for k, v in semantic["screen"].items()},
        "element_count": len(frame["elements"]),
        "visible_labels": [v[:96] for v in labels if isinstance(v, str) and v][:12],
        "evidence_ref": ref, "observed_sequence": sequence,
        "observed_at": datetime.now(UTC).isoformat(), "freshness": "fresh_returned_frame",
    }, bindings


class SessionState:
    def __init__(self, path: Path, session_id: str, contracts: list[str]):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be nonempty")
        if (not isinstance(contracts, list) or not contracts
                or not all(isinstance(c, str) and c.strip() for c in contracts)):
            raise ValueError("contracts must be a nonempty list of authored clauses")
        self.path = Path(path)
        self.history_path = self.path.with_name(self.path.stem + ".history.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checks = [{"id": f"C{i:03d}", "clause": c, "status": "pending", "evidence_refs": [], "note": ""}
                  for i, c in enumerate(contracts, 1)]
        self._state = {"format": FORMAT, "session_id": session_id, "sequence": 0, "phase_id": None, "checks": checks,
                       "current_observation": None, "selector_bindings": {},
                       "attempts": {}, "knowledge": {}, "recent": []}
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
            if (self._state.get("format") != FORMAT or self._state.get("session_id") != session_id
                    or [(c["id"], c["clause"]) for c in self._state.get("checks", [])]
                    != [(c["id"], c["clause"]) for c in checks]
                    or any(c.get("status") not in CHECK_STATUSES for c in self._state["checks"])):
                raise ValueError("session state identity or contracts mismatch")
        elif self.history_path.exists():
            raise ValueError("journal exists without its session identity snapshot")
        if self.history_path.exists():
            sequence = 0
            for line in self.history_path.read_text().splitlines():
                record = json.loads(line)
                sequence += 1
                if record.get("session_id") != session_id or record["event"]["sequence"] != sequence:
                    raise ValueError("session history sequence mismatch")
                if sequence > self._state["sequence"]:
                    self._apply(record)
            if sequence < self._state["sequence"]:
                raise ValueError("session history is incomplete")
        elif self._state["sequence"]:
            raise ValueError("session history is missing")
        self._save()

    def _save(self):
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_json(self._state) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def _apply(self, record):
        self._state.update(copy.deepcopy(record["patch"]))
        self._state["sequence"] = record["event"]["sequence"]
        self._state["recent"] = (self._state["recent"] + [record["event"]])[-RECENT_LIMIT:]

    def _commit(self, event, patch):
        event = {"sequence": self._state["sequence"] + 1,
                 "recorded_at": datetime.now(UTC).isoformat(), **event}
        record = {"session_id": self._state["session_id"], "event": event, "patch": patch}
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._apply(record)
        self._save()

    def observe(self, tool: str, arguments: dict, result: dict, evidence_ref: str | None = None):
        if not isinstance(tool, str) or not tool or not isinstance(arguments, dict) or not isinstance(result, dict):
            raise ValueError("observe requires a tool name, argument object and result object")
        if evidence_ref is not None and (not isinstance(evidence_ref, str) or not evidence_ref or len(evidence_ref) > 256):
            raise ValueError("evidence_ref must be a short nonempty reference")
        key = _action_key(tool, arguments, self._state.get("selector_bindings", {}))
        prior = self._state["current_observation"]
        current = copy.deepcopy(prior)
        bindings = copy.deepcopy(self._state.get("selector_bindings", {}))
        attempts = copy.deepcopy(self._state["attempts"])
        frame = observation_frame(result)
        if frame is not None:
            current, bindings = _frame_summary(frame, evidence_ref, self._state["sequence"] + 1)
        changed = frame is not None and (prior is None or current["signature"] != prior["signature"])
        if changed:
            attempts = {}
        error = result.get("error")
        refused = result.get("executed") is False or isinstance(error, dict) and error.get("executed") is False
        outcome = "rejected" if refused else "error" if result.get("ok") is False or error else "returned"
        if _meaningful(tool) and prior is not None and not changed:
            entry = attempts.get(key, {"tool": tool, "action_digest": key,
                                       "argument_keys": sorted(arguments), "count": 0})
            entry["count"] += 1
            attempts[key] = entry
            if frame is None and not refused:
                current["freshness"] = "unobserved_after_action"
        event = {"kind": "observation", "tool": tool, "argument_digest": key,
                 "evidence_ref": evidence_ref, "outcome": outcome, "screen_changed": changed,
                 "frame_returned": frame is not None,
                 "screen_signature": current["signature"] if current else None}
        if frame is not None:
            event["observation_summary"] = {"surface": current["surface"],
                                            "visible_labels": current["visible_labels"],
                                            "element_count": current["element_count"]}
        self._commit(event, {"current_observation": current, "selector_bindings": bindings, "attempts": attempts})
        return copy.deepcopy(self._state["recent"][-1])

    def rejection(self, tool, args):
        if not _meaningful(tool):
            return None
        current = self._state["current_observation"]
        if current is None or current["freshness"] != "fresh_returned_frame":
            return None
        attempt = self._state["attempts"].get(_action_key(tool, args, self._state.get("selector_bindings", {})), {})
        if attempt.get("count", 0) < MAX_ATTEMPTS:
            return None
        return {"ok": False, "error": {"code": "repeated_action_without_progress", "executed": False,
                "attempts": attempt["count"], "limit": MAX_ATTEMPTS,
                "message": "This exact action has made no screen progress three times. Choose a different action, wait for a real state change, or report the missing route."},
                "observation_ref": current["evidence_ref"]}

    def context(self):
        return copy.deepcopy({"format": FORMAT, "sequence": self._state["sequence"],
            "phase_id": self._state.get("phase_id"),
            "checks_are_untrusted_claims": True, "checks": self._state["checks"],
            "current_observation": self._state["current_observation"], "knowledge": self._state["knowledge"],
            "route_attempts": list(self._state["attempts"].values())[-RECENT_LIMIT:],
            "recent_history": self._state["recent"], "history_events": self._state["sequence"]})

    def begin_phase(self, phase_id: str):
        if not isinstance(phase_id, str) or not phase_id.strip() or len(phase_id) > 256:
            raise ValueError("phase_id must be a short nonempty host-owned identifier")
        changed = self._state.get("phase_id") != phase_id
        if changed:
            self._commit({"kind": "phase", "phase_id": phase_id}, {"phase_id": phase_id, "attempts": {}})
        return {"ok": True, "phase_id": phase_id, "changed": changed}

    def set_knowledge(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("knowledge must be a host-authored summary object")
        knowledge = {**self._state["knowledge"], **copy.deepcopy(value)}
        encoded = _json(knowledge)
        if len(encoded.encode()) > 8192:
            raise ValueError("knowledge summary exceeds 8192 bytes")
        forbidden = {"elements", "reasoning", "reasoning_content", "reasoning_details", "messages", "tool_results"}
        def contains_raw(item):
            return (isinstance(item, dict) and any(k in forbidden or contains_raw(v) for k, v in item.items())
                    or isinstance(item, list) and any(contains_raw(v) for v in item))
        if contains_raw(knowledge):
            raise ValueError("knowledge must summarize facts, not raw observations or model history")
        self._commit({"kind": "knowledge", "keys": sorted(value)}, {"knowledge": knowledge})
        return {"ok": True, "keys": sorted(knowledge)}

    def update_checks(self, updates: list[dict]):
        checks = copy.deepcopy(self._state["checks"])
        indexed = {c["id"]: c for c in checks}
        seen = set()
        if not isinstance(updates, list) or not updates:
            return {"ok": False, "error": {"code": "invalid_check_updates", "applied": False}}
        for update in updates:
            if (not isinstance(update, dict) or set(update) - {"id", "status", "evidence_refs", "note"}
                    or not isinstance(update.get("id"), str) or not isinstance(update.get("status"), str)
                    or update.get("id") not in indexed or update["id"] in seen
                    or update.get("status") not in CHECK_STATUSES
                    or not isinstance(update.get("note", ""), str) or len(update.get("note", "")) > 512
                    or not isinstance(update.get("evidence_refs", []), list)
                    or len(update.get("evidence_refs", [])) > 12
                    or not all(isinstance(ref, str) and 0 < len(ref) <= 256 for ref in update.get("evidence_refs", []))):
                return {"ok": False, "error": {"code": "invalid_check_updates", "applied": False}}
            seen.add(update["id"])
            indexed[update["id"]].update(copy.deepcopy(update))
        self._commit({"kind": "check_claims", "ids": list(seen)}, {"checks": checks})
        return {"ok": True, "updated": [u["id"] for u in updates], "claims_only": True}
