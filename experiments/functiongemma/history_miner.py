"""Mine historical AUA usage into privacy-safe V9 curriculum seeds.

The AUA journal is deliberately useful for operational accounting, but it is not a public
training dataset: short UI copy, goals, selectors, packages, and URIs can be present in clear
text, while successful observations usually retain only an element count.  This module joins
journal events to final session state and emits *structural* episodes.  It never emits source
copy, selector values, package names, serials, owners, timestamps, or raw errors.

The result is one layer before learning-material authoring.  It identifies high-confidence
positive, hard-negative, handoff, cleanup, and recovery seeds.  A later author must fictionalize
the semantics and render candidates through the exact production policy serializer before any
model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "aua-history-curriculum-seeds-v1"

SAFE_COMMANDS = frozenset(
    {
        "analyze",
        "ask_screen",
        "await_predicate",
        "back_until",
        "capture_explain",
        "capture_export",
        "capture_last",
        "capture_off",
        "capture_on",
        "clear",
        "cli_help",
        "cli_usage_error",
        "dashboard_start",
        "database_execute",
        "database_list",
        "database_query",
        "database_restore",
        "database_schema",
        "expect",
        "explore_plan",
        "flow_delete",
        "flow_run",
        "flow_save",
        "goto",
        "has",
        "helper.offloaded",
        "helper.partial",
        "helper.skipped",
        "hide_keyboard",
        "input",
        "install_app",
        "job_cancel",
        "job_list",
        "job_start",
        "job_status",
        "job_wait",
        "key",
        "long_press",
        "memory_update",
        "navigate",
        "network_offline",
        "network_restore",
        "network_status",
        "open_link",
        "orient",
        "scroll",
        "scroll_to",
        "session_candidate_flow",
        "session_finish",
        "session_progress",
        "session_review",
        "session_start",
        "swipe",
        "tap",
        "tap_point",
        "target_report",
        "wait",
        "wait_after_change",
        "wait_changed",
        "wait_stable",
    }
)
SAFE_SOURCES = frozenset({"cli", "daemon", "dashboard", "helper"})
SAFE_PHASE_KINDS = frozenset({"verify", "environment", "cleanup"})
SAFE_PHASE_STATUSES = frozenset({"active", "completed", "pending"})
SAFE_PROOF_SOURCES = frozenset(
    {
        "contract_assertions",
        "job",
        "manual_evidence",
        "observation",
        "session_cleanup",
        "verified_event",
    }
)
STRUCTURED_PROOF_SOURCES = SAFE_PROOF_SOURCES - {"manual_evidence"}
SAFE_POLICY_STATUSES = frozenset(
    {
        "deterministic",
        "invalid_selection",
        "no_candidate",
        "selected",
        "selector_unavailable",
        "skipped_deterministic",
        "skipped_unbound_observation",
        "unavailable",
        "unsupported_cardinality",
        "unsupported_mode",
    }
)
SAFE_POLICY_MODES = frozenset({"off", "shadow", "advisory"})
SAFE_REQUIREMENTS = frozenset(
    {"absent", "checked", "disabled", "enabled", "present", "selected", "unchecked"}
)
SENSITIVE_KEYS = frozenset(
    {
        "activity",
        "body",
        "content_desc",
        "description",
        "evidence",
        "goal",
        "label",
        "objective",
        "package",
        "resource_id",
        "text",
        "uri",
        "value",
    }
)

_TOOL_TO_COMMAND = {
    "analyze_screen": "analyze",
    "await_and_analyze": "await_predicate",
    "back_until_and_analyze": "back_until",
    "input_and_analyze": "input",
    "key_and_analyze": "key",
    "long_press_and_analyze": "long_press",
    "scroll_and_analyze": "scroll",
    "swipe_and_analyze": "swipe",
    "tap_and_analyze": "tap",
}
_SELECTOR_KEYS = frozenset({"id", "rid", "text", "desc"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"{SCHEMA}:{value}".encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _safe_enum(value: Any, allowed: frozenset[str]) -> str:
    text = str(value or "").strip()
    return text if text in allowed else "other"


def _safe_command(value: Any) -> str:
    return _safe_enum(value, SAFE_COMMANDS)


def _bounded_int(value: Any, *, low: int = 0, high: int = 1_000_000) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if low <= parsed <= high else None


def _event_session_id(event: Mapping[str, Any]) -> str | None:
    value = event.get("session_id")
    if not value and isinstance(event.get("extra"), Mapping):
        value = event["extra"].get("session_id")
    return str(value) if value else None


def _event_invocation_id(event: Mapping[str, Any]) -> str | None:
    value = event.get("invocation_id")
    if not value and isinstance(event.get("extra"), Mapping):
        value = event["extra"].get("invocation_id")
    return str(value) if value else None


def _goal_progress(event: Mapping[str, Any]) -> Mapping[str, Any]:
    result = event.get("result")
    if not isinstance(result, Mapping):
        return {}
    progress = result.get("goal_progress")
    return progress if isinstance(progress, Mapping) else {}


def _read_jsonl_snapshot(path: Path) -> tuple[bytes, list[dict[str, Any]], int]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    bad = 0
    for raw in payload.splitlines():
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            bad += 1
    return payload, rows, bad


def _load_journal(journal_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    content_hashes: list[str] = []
    bad = 0
    paths = sorted(
        (path for path in journal_dir.glob("*.jsonl*") if path.is_file()),
        key=lambda path: path.name,
    )
    for path in paths:
        payload, parsed, invalid = _read_jsonl_snapshot(path)
        content_hashes.append(hashlib.sha256(payload).hexdigest())
        rows.extend(parsed)
        bad += invalid
    rows.sort(key=lambda row: (int(row.get("ts_ms") or 0), str(row.get("cmd") or "")))
    digest = hashlib.sha256("".join(sorted(content_hashes)).encode()).hexdigest()
    return rows, {
        "files": len(paths),
        "rows": len(rows),
        "invalid_rows": bad,
        "content_digest": digest,
    }


def _load_sessions(session_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    content_hashes: list[str] = []
    invalid = 0
    files = 0
    for path in sorted(item for item in session_dir.rglob("*") if item.is_file()):
        files += 1
        payload = path.read_bytes()
        content_hashes.append(hashlib.sha256(payload).hexdigest())
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid += 1
            continue
        if not isinstance(value, dict) or not value.get("session_id"):
            invalid += 1
            continue
        sessions[str(value["session_id"])] = value
    digest = hashlib.sha256("".join(sorted(content_hashes)).encode()).hexdigest()
    return sessions, {
        "files": files,
        "sessions": len(sessions),
        "invalid_files": invalid,
        "content_digest": digest,
    }


def _is_emulator_session(session: Mapping[str, Any]) -> bool:
    return str(session.get("serial") or "").startswith("emulator-")


def _structured_phase_complete(phase: Mapping[str, Any]) -> bool:
    if phase.get("status") != "completed":
        return False
    proof = phase.get("proof")
    if not isinstance(proof, Mapping):
        return False
    source = str(proof.get("source") or "")
    return source in STRUCTURED_PROOF_SOURCES and proof.get("verified") is not False


def _phase_summary(phase: Mapping[str, Any]) -> dict[str, Any]:
    proof = phase.get("proof")
    proof_source = (
        _safe_enum(proof.get("source"), SAFE_PROOF_SOURCES)
        if isinstance(proof, Mapping)
        else "other"
    )
    requirements: Counter[str] = Counter()
    for requirement in phase.get("requirements") or []:
        if not isinstance(requirement, Mapping):
            continue
        requirements[_safe_enum(requirement.get("expected"), SAFE_REQUIREMENTS)] += 1
    return {
        "kind": _safe_enum(phase.get("kind"), SAFE_PHASE_KINDS),
        "status": _safe_enum(phase.get("status"), SAFE_PHASE_STATUSES),
        "proof_source": proof_source,
        "structured_proof": _structured_phase_complete(phase),
        "requirements": dict(sorted(requirements.items())),
        "assertion_count": len(phase.get("assertions") or []),
    }


def _episode_outcome(session: Mapping[str, Any], phases: Sequence[Mapping[str, Any]]) -> str:
    if not session.get("finished_ms"):
        return "active"
    if not phases:
        return "finished_unphased"
    if all(phase.get("status") == "completed" for phase in phases):
        return "completed"
    return "terminated_incomplete"


def _sanitized_event(event: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = event.get("result") if isinstance(event.get("result"), Mapping) else {}
    observation = result.get("observation") if isinstance(result, Mapping) else None
    progress = _goal_progress(event)
    return {
        "index": index,
        "command": _safe_command(event.get("cmd")),
        "source": _safe_enum(event.get("source"), SAFE_SOURCES),
        "ok": bool(event.get("ok")),
        "duration_ms": _bounded_int(round(float(event.get("duration_ms") or 0)), high=3_600_000),
        "observation_returned": isinstance(observation, Mapping),
        "observation_element_count": (
            _bounded_int(observation.get("elements_count"), high=100_000)
            if isinstance(observation, Mapping)
            else None
        ),
        "progress_completed": _bounded_int(progress.get("completed"), high=10_000),
        "progress_total": _bounded_int(progress.get("total"), high=10_000),
        "progress_done": bool(progress.get("done")),
        "progress_terminated": bool(progress.get("terminated")),
    }


def _next_top_level_event(
    events: Sequence[Mapping[str, Any]], index: int
) -> Mapping[str, Any] | None:
    current_invocation = _event_invocation_id(events[index])
    for candidate in events[index + 1 :]:
        invocation = _event_invocation_id(candidate)
        if current_invocation and invocation == current_invocation:
            continue
        if str(candidate.get("cmd") or "").startswith("helper."):
            continue
        return candidate
    return None


def _selector_value(arguments: Mapping[str, Any], key: str) -> Any:
    if key in arguments:
        return arguments.get(key)
    selector = arguments.get("selector")
    return selector.get(key) if isinstance(selector, Mapping) else None


def _suggestion_relation(
    suggestion: Mapping[str, Any] | None,
    next_event: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if not isinstance(suggestion, Mapping):
        return "none", "other"
    mcp = suggestion.get("mcp")
    if not isinstance(mcp, Mapping):
        return "none", "other"
    tool = str(mcp.get("tool") or "")
    expected = _TOOL_TO_COMMAND.get(tool, tool)
    safe_expected = _safe_command(expected)
    if next_event is None:
        return "no_next_action", safe_expected
    if str(next_event.get("cmd") or "") != expected:
        return "different_command", safe_expected
    suggested_args = mcp.get("arguments")
    actual_args = next_event.get("args")
    if not isinstance(suggested_args, Mapping) or not isinstance(actual_args, Mapping):
        return "same_command", safe_expected
    compared = False
    for key in _SELECTOR_KEYS:
        if key not in suggested_args:
            continue
        compared = True
        if _selector_value(actual_args, key) != suggested_args.get(key):
            return "same_command_different_selector", safe_expected
    return ("exact_action" if compared else "same_command"), safe_expected


def _progress_delta(current: Mapping[str, Any], next_event: Mapping[str, Any] | None) -> int:
    before = _bounded_int(_goal_progress(current).get("completed"), high=10_000) or 0
    after = _bounded_int(_goal_progress(next_event or {}).get("completed"), high=10_000) or before
    return max(0, after - before)


def _policy_classification(
    *,
    status: str,
    relation: str,
    next_ok: bool | None,
    progress_delta: int,
    phase_structured_complete: bool,
) -> tuple[str, str]:
    if status in {"no_candidate", "skipped_deterministic"}:
        return "handoff_or_no_candidate", "handoff_template"
    if status in {"unavailable", "selector_unavailable", "unsupported_mode"}:
        return "policy_unavailable", "manual_review"
    if status == "unsupported_cardinality":
        return "unsupported_cardinality", "coverage_template"
    if status != "selected":
        return "non_model_decision", "manual_review"
    followed = relation in {"exact_action", "same_command"}
    if followed and next_ok is False:
        return "selected_followed_action_failed", "recovery_template"
    if followed and progress_delta > 0 and phase_structured_complete:
        return "selected_followed_structured_progress", "positive_template"
    if not followed and next_ok and progress_delta > 0 and phase_structured_complete:
        return "selected_rejected_alternative_structured_progress", "hard_negative_template"
    if relation == "no_next_action":
        return "selected_without_followup", "manual_review"
    if followed:
        return "selected_followed_unproven", "manual_review"
    return "selected_rejected_or_ignored", "manual_review"


def _policy_decisions(
    episode_id: str,
    events: Sequence[Mapping[str, Any]],
    phases_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        progress = _goal_progress(event)
        policy = progress.get("policy")
        if not isinstance(policy, Mapping):
            continue
        suggestion = progress.get("policy_suggestion")
        next_event = _next_top_level_event(events, index)
        relation, suggested_command = _suggestion_relation(
            suggestion if isinstance(suggestion, Mapping) else None,
            next_event,
        )
        status = _safe_enum(policy.get("status"), SAFE_POLICY_STATUSES)
        phase_id = None
        current = progress.get("current")
        if isinstance(current, Mapping) and current.get("id"):
            phase_id = str(current["id"])
        phase = phases_by_id.get(phase_id or "", {})
        structured = _structured_phase_complete(phase)
        delta = _progress_delta(event, next_event)
        next_ok = bool(next_event.get("ok")) if next_event is not None else None
        classification, training_use = _policy_classification(
            status=status,
            relation=relation,
            next_ok=next_ok,
            progress_delta=delta,
            phase_structured_complete=structured,
        )
        invocation = _event_invocation_id(event) or str(index)
        decision_id = _stable_id("pd", f"{episode_id}:{invocation}:{index}")
        compiler_value = policy.get("compiler")
        compiler: dict[str, Any] = (
            dict(compiler_value) if isinstance(compiler_value, Mapping) else {}
        )
        stages_value = compiler.get("stages")
        stages: dict[str, Any] = dict(stages_value) if isinstance(stages_value, Mapping) else {}
        eligible = policy.get("eligible_candidate_ids")
        decisions.append(
            {
                "schema": SCHEMA,
                "decision_id": decision_id,
                "episode_id": episode_id,
                "event_index": index,
                "mode": _safe_enum(policy.get("mode"), SAFE_POLICY_MODES),
                "status": status,
                "model_used": bool(policy.get("model_used")),
                "candidate_count": _bounded_int(policy.get("candidate_count"), high=32),
                "eligible_candidate_count": len(eligible) if isinstance(eligible, list) else 0,
                "selected_candidate_id": _bounded_int(
                    policy.get("selected_candidate_id"), low=-1, high=32
                ),
                "target_term_count": _bounded_int(compiler.get("target_term_count"), high=128),
                "compiler_offered": _bounded_int(stages.get("offered"), high=128),
                "recommended_call_offered": bool(compiler.get("recommended_call_offered")),
                "suggested_command": suggested_command,
                "next_action_relation": relation,
                "next_action_ok": next_ok,
                "immediate_progress_delta": delta,
                "phase_structured_complete": structured,
                "classification": classification,
                "training_use": training_use,
            }
        )
    return decisions


def _episode_seeds(
    episode: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []

    def add(family: str, use: str, evidence: str, decision_id: str | None = None) -> None:
        ordinal = len(seeds)
        seeds.append(
            {
                "schema": SCHEMA,
                "seed_id": _stable_id(
                    "cs", f"{episode['episode_id']}:{decision_id or 'episode'}:{family}:{ordinal}"
                ),
                "episode_id": episode["episode_id"],
                "decision_id": decision_id,
                "family": family,
                "training_use": use,
                "evidence_level": evidence,
            }
        )

    if episode["outcome"] == "completed" and episode["all_completed_phases_structured"]:
        add("structured_sequence_success", "sequence_template", "structured_proof")
    if episode["outcome"] == "terminated_incomplete":
        add("terminated_incomplete", "recovery_template", "session_truth")
    if episode["failed_command_count"]:
        add("action_failure", "recovery_template", "journal_failure")
    if episode["finish_called"] and episode["outcome"] == "terminated_incomplete":
        add("premature_or_incomplete_finish", "cleanup_template", "session_truth")
    if episode["cleanup_incomplete"]:
        add("cleanup_incomplete", "cleanup_template", "session_truth")
    for decision in decisions:
        use = str(decision["training_use"])
        if use == "manual_review":
            continue
        add(
            str(decision["classification"]),
            use,
            "structured_policy_join",
            str(decision["decision_id"]),
        )
    return seeds


def _sensitive_values(value: Any, *, path: tuple[str, ...] = ()) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _sensitive_values(item, path=path + (str(key).casefold(),))
    elif isinstance(value, list):
        for item in value:
            yield from _sensitive_values(item, path=path + ("[]",))
    elif path and path[-1] in SENSITIVE_KEYS and isinstance(value, str):
        text = value.strip()
        if len(text) >= 4:
            yield text


def _privacy_violations(outputs: Sequence[str], source_values: Iterable[str]) -> list[str]:
    combined = "\n".join(outputs)
    violations: list[str] = []
    for value in sorted(set(source_values), key=lambda item: (-len(item), item)):
        # Match a complete serialized string value. A short private label such as ``Finish``
        # may legitimately be a substring of the controlled enum ``session_finish``; that is
        # not source-copy leakage. Exact JSON-string matching still rejects the raw value.
        if json.dumps(value, ensure_ascii=False) in combined:
            violations.append(hashlib.sha256(value.encode()).hexdigest()[:16])
    return violations


def _payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def build_history_corpus(
    journal_rows: Sequence[Mapping[str, Any]],
    sessions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events_by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    command_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    policy_status_counts: Counter[str] = Counter()
    physical_events = 0
    host_events = 0
    for event in journal_rows:
        command = _safe_command(event.get("cmd"))
        command_counts[command] += 1
        if not event.get("ok"):
            failure_counts[command] += 1
        serial = str(event.get("serial") or "")
        if not serial:
            host_events += 1
        elif not serial.startswith("emulator-"):
            physical_events += 1
        session_id = _event_session_id(event)
        if session_id:
            events_by_session[session_id].append(event)

    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    excluded_sessions: Counter[str] = Counter()
    for session_id in sorted(sessions):
        session = sessions[session_id]
        if not _is_emulator_session(session):
            excluded_sessions["non_emulator"] += 1
            continue
        raw_phases = [phase for phase in session.get("phases") or [] if isinstance(phase, Mapping)]
        phases_by_id = {
            str(phase.get("id")): phase for phase in raw_phases if phase.get("id") is not None
        }
        phase_rows = [_phase_summary(phase) for phase in raw_phases]
        events = sorted(
            events_by_session.get(session_id, []),
            key=lambda row: (int(row.get("ts_ms") or 0), str(row.get("cmd") or "")),
        )
        episode_id = _stable_id("ep", session_id)
        outcome = _episode_outcome(session, raw_phases)
        completed_phases = [phase for phase in raw_phases if phase.get("status") == "completed"]
        all_structured = bool(completed_phases) and all(
            _structured_phase_complete(phase) for phase in completed_phases
        )
        cleanup_incomplete = any(
            phase.get("kind") == "cleanup" and phase.get("status") != "completed"
            for phase in raw_phases
        )
        episode = {
            "schema": SCHEMA,
            "episode_id": episode_id,
            "device_kind": "emulator",
            "outcome": outcome,
            "phase_count": len(raw_phases),
            "completed_phase_count": len(completed_phases),
            "all_completed_phases_structured": all_structured,
            "cleanup_incomplete": cleanup_incomplete,
            "event_count": len(events),
            "failed_command_count": sum(not bool(event.get("ok")) for event in events),
            "finish_called": any(event.get("cmd") == "session_finish" for event in events),
            "phases": phase_rows,
            "events": [_sanitized_event(event, index) for index, event in enumerate(events)],
        }
        episode_decisions = _policy_decisions(episode_id, events, phases_by_id)
        for decision in episode_decisions:
            policy_status_counts[str(decision["status"])] += 1
        episode["policy_decision_ids"] = [row["decision_id"] for row in episode_decisions]
        episode_seeds = _episode_seeds(episode, episode_decisions)
        episodes.append(episode)
        decisions.extend(episode_decisions)
        seeds.extend(episode_seeds)

    episodes.sort(key=lambda row: str(row["episode_id"]))
    decisions.sort(key=lambda row: str(row["decision_id"]))
    seeds.sort(key=lambda row: str(row["seed_id"]))
    summary = {
        "journal_events": len(journal_rows),
        "correlated_journal_events": sum(len(rows) for rows in events_by_session.values()),
        "uncorrelated_journal_events": len(journal_rows)
        - sum(len(rows) for rows in events_by_session.values()),
        "physical_or_other_events_excluded": physical_events,
        "host_events": host_events,
        "sessions_seen": len(sessions),
        "episodes_emitted": len(episodes),
        "sessions_excluded": dict(sorted(excluded_sessions.items())),
        "policy_decisions": len(decisions),
        "curriculum_seeds": len(seeds),
        "native_training_rows": 0,
        "training_status": "requires_fictionalization",
        "command_counts": dict(sorted(command_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "episode_outcomes": dict(sorted(Counter(row["outcome"] for row in episodes).items())),
        "policy_statuses": dict(sorted(policy_status_counts.items())),
        "seed_families": dict(sorted(Counter(row["family"] for row in seeds).items())),
    }
    return episodes, decisions, seeds, summary


def write_history_corpus(
    *,
    journal_dir: Path,
    session_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    journal_rows, journal_meta = _load_journal(journal_dir)
    sessions, session_meta = _load_sessions(session_dir)
    episodes, decisions, seeds, summary = build_history_corpus(journal_rows, sessions)
    sensitive = [
        item for source in (*journal_rows, *sessions.values()) for item in _sensitive_values(source)
    ]
    summary["sensitive_source_values_scanned"] = len(set(sensitive))
    payloads = {
        "episodes.jsonl": _payload(episodes),
        "policy_decisions.jsonl": _payload(decisions),
        "curriculum_seeds.jsonl": _payload(seeds),
    }
    violations = _privacy_violations(list(payloads.values()), sensitive)
    if violations:
        raise ValueError(
            "privacy validation rejected sanitized history output: " + ",".join(violations[:8])
        )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "journal_source": journal_meta,
        "session_source": session_meta,
        "summary": summary,
        "privacy": {
            "raw_source_values_emitted": 0,
            "violations": 0,
            "physical_sessions_allowed": False,
            "source_copy_allowed": False,
            "selector_values_allowed": False,
            "packages_allowed": False,
        },
        "files": {
            name: {
                "rows": payload.count("\n"),
                "bytes": len(payload.encode()),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
            for name, payload in sorted(payloads.items())
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _atomic_write(output_dir / name, payload)
    _atomic_write(
        output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create privacy-safe V9 curriculum seeds from historical AUA usage."
    )
    cache = Path.home() / ".cache" / "android-ui-analyser"
    parser.add_argument("--journal-dir", type=Path, default=cache / "journal")
    parser.add_argument("--session-dir", type=Path, default=cache / "sessions")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/functiongemma/history-v9-prep"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = write_history_corpus(
        journal_dir=args.journal_dir.expanduser(),
        session_dir=args.session_dir.expanduser(),
        output_dir=args.output_dir.expanduser(),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
