"""App-agnostic scoring for AUA session artifact bundles.

The evaluator is deliberately offline: it reads evidence produced by other processes and never
launches an agent, AUA, an Android tool, or the public fixture app.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PASS_STATUSES = {"pass", "passed", "success", "succeeded", "complete", "completed"}


class CampaignError(ValueError):
    """Raised when campaign input or a required bundle artifact is invalid."""


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    scenario_id: str
    lane: str
    repeat: int
    reported_pass: bool
    completed: bool
    cleanup_verified: bool
    verifier_pass: bool | None
    false_pass: bool | None
    calls: int | None
    accounting_status: str
    accounting_issues: list[str]
    post_finish_calls: int | None
    unexpected_failures: int | None
    comparison_fingerprint: str | None
    analyze_calls: int
    redundant_analyze_calls: int | None
    recovery_calls: int
    duration_ms: float | None
    within_time_limit: bool | None
    evidence_expected: int
    evidence_resolved: int
    evidence_completeness: float | None
    candidate_flow_reuse_expected: bool
    candidate_flow_reused: bool
    candidate_flow_reuse_met: bool
    bundle: str


def _load_json(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise CampaignError(f"required JSON file does not exist: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read valid JSON from {path}: {exc}") from exc


def _status_pass(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in PASS_STATUSES


def _first(mapping: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        value: Any = mapping
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _checkpoint_entries(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _first(
        result,
        (("contract", "checkpoints"), ("checkpoints",), ("goal_progress", "phases")),
    )
    if isinstance(raw, Mapping):
        entries: list[Mapping[str, Any]] = []
        for checkpoint_id, value in raw.items():
            if isinstance(value, Mapping):
                entries.append({"id": str(checkpoint_id), **value})
            else:
                entries.append({"id": str(checkpoint_id), "status": value})
        return entries
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, Mapping)]
    return []


def _checkpoint_passes(result: Mapping[str, Any]) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for entry in _checkpoint_entries(result):
        checkpoint_id = entry.get("id") or entry.get("name")
        if checkpoint_id is None:
            continue
        status = entry.get("status", entry.get("passed", entry.get("complete")))
        output[str(checkpoint_id)] = _status_pass(status)
    return output


def _cleanup_pass(result: Mapping[str, Any]) -> bool:
    cleanup = result.get("cleanup")
    if isinstance(cleanup, list):
        return bool(cleanup) and all(
            isinstance(entry, Mapping) and entry.get("ok") is True for entry in cleanup
        ) and any(
            entry.get("action") == "lease_release"
            and isinstance(entry.get("result"), Mapping)
            and entry["result"].get("released") is True
            for entry in cleanup
        )
    value = _first(
        result,
        (
            ("cleanup_verified",),
            ("cleanup", "verified"),
            ("cleanup", "passed"),
            ("cleanup", "status"),
            ("contract", "cleanup", "verified"),
            ("contract", "cleanup", "status"),
        ),
    )
    if _status_pass(value):
        return True
    return any(
        str(entry.get("kind", "")).lower() == "cleanup"
        and _status_pass(entry.get("status"))
        for entry in _checkpoint_entries(result)
    )


def _manifest_duration_ms(manifest: Mapping[str, Any]) -> float | None:
    started = manifest.get("started_at")
    finished = manifest.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    from datetime import datetime

    try:
        return max(0.0, (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds() * 1000)
    except ValueError:
        return None


def _iter_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    records: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CampaignError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise CampaignError(f"JSONL record at {path}:{line_number} must be an object")
        records.append(value)
    return records


def _operation(record: Mapping[str, Any]) -> str:
    value = _first(
        record,
        (
            ("cmd",),
            ("operation",),
            ("command",),
            ("tool",),
            ("method",),
            ("name",),
            ("request", "operation"),
            ("request", "command"),
            ("request", "tool"),
            ("request", "method"),
        ),
    )
    if isinstance(value, list):
        return " ".join(str(part) for part in value).strip().lower()
    return str(value or "").strip().lower()


def _is_call(record: Mapping[str, Any]) -> bool:
    event = str(record.get("event", record.get("kind", record.get("type", "")))).lower()
    if event in {"response", "result", "call_result", "tool_result", "completed"}:
        return False
    return bool(_operation(record)) or event in {"call", "invocation", "request", "tool_call"}


def _call_metrics(records: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> tuple[int, int, int, int]:
    calls = [record for record in records if _is_call(record)]
    analyze = [record for record in calls if "analyze" in _operation(record).replace(".", " ").split()]
    redundant = 0
    for record in calls:
        if "analyze" not in _operation(record).replace(".", " ").split():
            continue
        redundant_value = _first(record, (("redundant",), ("metrics", "redundant")))
        # Legacy records need an explicit classification. Freshness alone cannot prove
        # a repeated read was wasted (the caller may request a different view).
        if redundant_value is True:
            redundant += 1
    recovery = sum(
        1
        for record in calls
        if record.get("recovery") is True
        or str(record.get("category", "")).lower() == "recovery"
        or "recovery" in str(record.get("reason", "")).lower()
    )

    # Early bundle implementations may store only aggregate counts in result.json.
    def count(paths: Iterable[Sequence[str]], fallback: int) -> int:
        value = _first(result, paths)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else fallback

    total = count((("metrics", "calls"), ("call_count",)), len(calls))
    analyze_total = count((("metrics", "analyze_calls"),), len(analyze))
    redundant_total = count((("metrics", "redundant_analyze_calls"),), redundant)
    recovery_total = count((("metrics", "recovery_calls"), ("recovery_calls",)), recovery)
    return total, analyze_total, redundant_total, recovery_total


def _native_accounting(
    bundle: Path, result: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Review complete journal evidence, never the sparse artifact call list."""
    saved_review = result.get("review")
    if not isinstance(saved_review, Mapping) and not (bundle / "journal.jsonl").exists():
        return None
    missing = [name for name in ("session.json", "journal.jsonl") if not (bundle / name).is_file()]
    if missing:
        return {"status": "unavailable", "issues": [f"missing native evidence: {', '.join(missing)}"],
                "calls": None, "post_finish_calls": None, "failures": None,
                "analyze": 0, "redundant": None}
    from android_ui_analyser.session import SessionState, review_session_events

    try:
        state = SessionState.model_validate(_load_json(bundle / "session.json"))
    except ValueError as exc:
        raise CampaignError(f"invalid native session state in {bundle}") from exc
    events = [dict(event) for event in _iter_jsonl(bundle / "journal.jsonl")
              if (event.get("session_id") or (event.get("extra") or {}).get("session_id")) == state.session_id]
    events.sort(key=lambda event: event.get("ts_ms", 0))
    review = review_session_events(state, events)
    saved_review = saved_review if isinstance(saved_review, Mapping) else {}
    saved = saved_review.get("accounting") or {}
    issues = []

    def terminal(payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        checkpoint = payload.get("review") or {}
        return (payload.get("session_id") == state.session_id
                and isinstance(payload.get("finished"), bool) and payload.get("terminated") is True
                and isinstance(checkpoint, Mapping) and bool(state.finished_ms)
                and checkpoint.get("session_id") == state.session_id
                and checkpoint.get("started_ms") == state.started_ms
                and checkpoint.get("finished_ms") == state.finished_ms
                and checkpoint.get("accounting") == saved)

    finishes = [i for i, event in enumerate(events)
                if event.get("cmd") == "session_finish" and terminal(event.get("result"))]
    if not terminal(result) or len(finishes) != 1:
        issues.append("no unique terminal finish matches saved state/accounting")
        prefix, post = events, []
    else:
        prefix, post = events[:finishes[0] + 1], events[finishes[0] + 1:]
    before = review_session_events(state, prefix)["accounting"]
    after = review_session_events(state, post)["accounting"]
    expected_calls = saved.get("top_level_calls") if saved.get("reporting_call_included") is True else saved.get("top_level_calls_including_reporting_call")
    expected_events = saved.get("journal_events")
    if isinstance(expected_events, int) and saved.get("reporting_call_included") is False:
        expected_events += 1
    if (expected_calls != before["top_level_calls"] or expected_events != len(prefix)
            or not any(event.get("cmd") == "session_start" for event in prefix)):
        issues.append("retained through-finish call/event counts do not match saved accounting")
    if saved.get("journal_events", 0) >= 2000:
        issues.append("saved review reached its 2000-event retention limit")
    ids = {event.get("invocation_id") or (event.get("extra") or {}).get("invocation_id") for event in events} - {None}
    # Artifacts omit some failed/non-dict results and can append after finish.
    if manifest.get("session_id") != state.session_id or not set(manifest.get("invocations") or []).issubset(ids):
        issues.append("sparse manifest IDs are not corroborated by retained journal")
    if review["accounting"]["top_level_calls"] != before["top_level_calls"] + after["top_level_calls"]:
        issues.append("caller folding crosses terminal finish boundary")
    if review["patterns"].get("ambiguous_invocation"):
        issues.append("retained invocations contain ambiguous caller outcomes")
    return {"status": "incomplete" if issues else "verified_through_finish", "issues": issues,
            "calls": review["accounting"]["top_level_calls"], "post_finish_calls": after["top_level_calls"],
            "failures": None if review["run_ok"] is None else review["accounting"]["unexpected_failures"],
            "analyze": review["commands"].get("analyze", 0),
            "redundant": None if any(event.get("_detail_hydrated") is not True for event in events) else
            sum(item.get("confirmed") is True for item in review["patterns"].get("redundant_analyze", []))}


def _collect_evidence_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"evidence_id", "evidenceId"} and isinstance(child, (str, int)):
                found.add(str(child))
            else:
                found.update(_collect_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_evidence_ids(child))
    return found


def _manifest_evidence_ids(manifest: Mapping[str, Any], bundle: Path) -> set[str]:
    found: set[str] = set()
    def exists(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        path = (bundle / value).resolve()
        return path.is_relative_to(bundle.resolve()) and path.is_file()

    evidence = manifest.get("evidence")
    if isinstance(evidence, Mapping):
        for key, value in evidence.items():
            path = value.get("path") if isinstance(value, Mapping) else value
            if exists(path):
                found.add(str(key))
    elif isinstance(evidence, list):
        for entry in evidence:
            if isinstance(entry, Mapping):
                identifier = entry.get("id") or entry.get("evidence_id")
                if identifier is not None and exists(entry.get("path")):
                    found.add(str(identifier))
    entries = manifest.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            identifier = entry.get("evidence_id")
            captured = entry.get("observation") or entry.get("screenshot")
            if identifier is not None and exists(captured):
                found.add(str(identifier))
    return found


def _checkpoint_evidence(
    result: Mapping[str, Any],
    required: Sequence[str],
    cleanup_required: bool,
) -> dict[str, str | None]:
    proof: dict[str, str | None] = {str(checkpoint): None for checkpoint in required}
    for entry in _checkpoint_entries(result):
        checkpoint_id = entry.get("id") or entry.get("name")
        if checkpoint_id is None or str(checkpoint_id) not in proof:
            continue
        evidence_ids = _collect_evidence_ids(entry)
        proof[str(checkpoint_id)] = sorted(evidence_ids)[0] if evidence_ids else None
    if cleanup_required:
        cleanup = _first(result, (("cleanup",), ("contract", "cleanup")))
        evidence_ids = _collect_evidence_ids(cleanup)
        if isinstance(cleanup, list) and _cleanup_pass(result):
            evidence_ids.add("result.json#cleanup")
        if not evidence_ids:
            evidence_ids = {
                evidence_id
                for entry in _checkpoint_entries(result)
                if str(entry.get("kind", "")).lower() == "cleanup"
                for evidence_id in _collect_evidence_ids(entry)
            }
        proof["cleanup"] = sorted(evidence_ids)[0] if evidence_ids else None
    return proof


def _candidate_reused(result: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> bool:
    explicit = _first(
        result,
        (
            ("candidate_flow_reused",),
            ("candidate_flow", "reused"),
            ("metrics", "candidate_flow_reused"),
        ),
    )
    if isinstance(explicit, bool):
        return explicit
    for record in calls:
        operation = _operation(record)
        source = str(_first(record, (("flow", "source"), ("metadata", "flow_source"))) or "").lower()
        if "flow" in operation and source in {"candidate", "promoted_candidate"}:
            return True
    return False


def _validate_campaign(campaign: Any) -> Mapping[str, Any]:
    if not isinstance(campaign, Mapping):
        raise CampaignError("campaign root must be an object")
    if campaign.get("schema_version") != 1:
        raise CampaignError("campaign schema_version must be 1")
    if not isinstance(campaign.get("campaign_id"), str) or not campaign["campaign_id"]:
        raise CampaignError("campaign_id must be a non-empty string")
    scenarios = campaign.get("scenarios")
    runs = campaign.get("runs")
    if not isinstance(scenarios, list) or not scenarios:
        raise CampaignError("scenarios must be a non-empty array")
    if not isinstance(runs, list) or not runs:
        raise CampaignError("runs must be a non-empty array")
    scenario_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or not isinstance(scenario.get("id"), str):
            raise CampaignError("every scenario must have a string id")
        if scenario["id"] in scenario_ids:
            raise CampaignError(f"duplicate scenario id: {scenario['id']}")
        scenario_ids.add(scenario["id"])
        for field in ("title", "goal"):
            if not isinstance(scenario.get(field), str) or not scenario[field]:
                raise CampaignError(f"scenario {scenario['id']} must have a non-empty {field}")
        if (
            isinstance(scenario.get("time_limit_s"), bool)
            or not isinstance(scenario.get("time_limit_s"), (int, float))
            or scenario["time_limit_s"] <= 0
        ):
            raise CampaignError(f"scenario {scenario['id']} must have a positive time_limit_s")
        checkpoints = scenario.get("required_checkpoints", [])
        if not isinstance(checkpoints, list) or any(
            not isinstance(checkpoint, str) or not checkpoint for checkpoint in checkpoints
        ):
            raise CampaignError(
                f"scenario {scenario['id']} required_checkpoints must contain non-empty strings"
            )
    run_ids: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise CampaignError("every run must be an object")
        required = ("run_id", "scenario_id", "lane", "bundle")
        if any(not isinstance(run.get(key), str) or not run[key] for key in required):
            raise CampaignError(f"every run requires non-empty string fields: {', '.join(required)}")
        if run["run_id"] in run_ids:
            raise CampaignError(f"duplicate run id: {run['run_id']}")
        if run["scenario_id"] not in scenario_ids:
            raise CampaignError(f"run {run['run_id']} references unknown scenario {run['scenario_id']}")
        repeat = run.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise CampaignError(f"run {run['run_id']} repeat must be a positive integer")
        if "verifier" in run and (
            not isinstance(run["verifier"], str) or not run["verifier"]
        ):
            raise CampaignError(f"run {run['run_id']} verifier must be a non-empty string")
        run_ids.add(run["run_id"])
    return campaign


def evaluate_campaign(campaign_path: Path) -> dict[str, Any]:
    campaign = _validate_campaign(_load_json(campaign_path))
    root = campaign_path.resolve().parent
    scenarios = {str(item["id"]): item for item in campaign["scenarios"]}
    run_metrics: list[RunMetrics] = []

    for run in campaign["runs"]:
        scenario = scenarios[str(run["scenario_id"])]
        bundle = (root / str(run["bundle"])).resolve()
        result = _load_json(bundle / "result.json", required=False)
        manifest = _load_json(bundle / "manifest.json", required=False)
        if not isinstance(result, Mapping) or not isinstance(manifest, Mapping):
            raise CampaignError(f"bundle result and manifest must be JSON objects: {bundle}")
        calls = _iter_jsonl(bundle / "calls.jsonl")
        execution = _load_json(bundle / "execution.json", required=False)
        fingerprint = execution.get("comparison_fingerprint") if isinstance(execution, Mapping) else None
        fingerprint = fingerprint if isinstance(fingerprint, str) and fingerprint else None

        status = _first(
            result,
            (("verdict",), ("status",), ("result", "status"), ("finished",)),
        )
        reported_pass = _status_pass(status)
        checkpoint_passes = _checkpoint_passes(result)
        required_checkpoints = [str(value) for value in scenario.get("required_checkpoints", [])]
        checkpoints_pass = all(
            checkpoint_passes.get(checkpoint, False)
            for checkpoint in required_checkpoints
            if checkpoint != "cleanup"
        )
        cleanup_required = bool(scenario.get("cleanup_required", True))
        cleanup_verified = _cleanup_pass(result)

        verifier: Mapping[str, Any] | None = None
        if run.get("verifier"):
            loaded_verifier = _load_json((root / str(run["verifier"])).resolve(), required=False)
            if not isinstance(loaded_verifier, Mapping):
                raise CampaignError(f"verifier must be a JSON object for run {run['run_id']}")
            verifier = loaded_verifier
        verifier_pass = None
        false_pass = None
        if verifier is not None:
            decision = _first(verifier, (("passed",), ("verdict",), ("status",)))
            cleanup_decision = _first(verifier, (("cleanup_verified",), ("cleanup", "verified"), ("cleanup", "status")))
            known = isinstance(decision, bool) or str(decision).lower() in PASS_STATUSES | {"fail", "failed"}
            cleanup_known = isinstance(cleanup_decision, bool) or str(cleanup_decision).lower() in PASS_STATUSES | {"fail", "failed"}
            if (known and not _status_pass(decision)) or (cleanup_required and cleanup_known and not _status_pass(cleanup_decision)):
                verifier_pass = False
            elif known and (not cleanup_required or cleanup_known):
                verifier_pass = _status_pass(decision) and (not cleanup_required or _status_pass(cleanup_decision))
            if verifier_pass is not None:
                false_pass = reported_pass and not verifier_pass

        total_calls: int | None
        redundant_calls: int | None
        total_calls, analyze_calls, redundant_calls, recovery_calls = _call_metrics(calls, result)
        native = _native_accounting(bundle, result, manifest)
        accounting_status, accounting_issues = "legacy", []
        post_finish_calls = unexpected_failures = None
        if native is not None:
            total_calls, analyze_calls, redundant_calls = native["calls"], native["analyze"], native["redundant"]
            post_finish_calls, unexpected_failures = native["post_finish_calls"], native["failures"]
            accounting_status, accounting_issues = native["status"], native["issues"]
        elif not result or not manifest:
            accounting_status = "unavailable"
            accounting_issues = ["missing result.json or manifest.json; attempted run remains in denominator"]
            total_calls = len([call for call in calls if _is_call(call)]) if calls else None
            redundant_calls = None
        duration_ms = _float_or_none(
            _first(result, (("duration_ms",), ("metrics", "duration_ms")))
        ) or _float_or_none(_first(manifest, (("duration_ms",), ("metrics", "duration_ms"))))
        duration_ms = duration_ms or _manifest_duration_ms(manifest)
        time_limit_ms = float(scenario["time_limit_s"]) * 1000
        within_time_limit = None if duration_ms is None else duration_ms <= time_limit_ms
        completed = (
            # Optional loading preserves partial attempts, but both bundle artifacts
            # remain required even when cleanup is the only requested evidence.
            bool(result) and bool(manifest)
            and reported_pass
            and checkpoints_pass
            and (cleanup_verified or not cleanup_required)
            and within_time_limit is True
        )

        evidence_proof = _checkpoint_evidence(result, required_checkpoints, cleanup_required)
        available_evidence = _manifest_evidence_ids(manifest, bundle)
        if isinstance(result.get("cleanup"), list) and _cleanup_pass(result) and (bundle / "result.json").is_file():
            available_evidence.add("result.json#cleanup")
        resolved = sum(1 for evidence_id in evidence_proof.values() if evidence_id in available_evidence)
        expected = len(evidence_proof)
        completeness = None if expected == 0 else resolved / expected
        # An affirmative checkpoint without its saved proof is not a verified completion.
        completed = completed and (not expected or resolved == expected) and (verifier is None or verifier_pass is True)
        if "terminated" in result:
            completed = completed and result["terminated"] is True
        if native is not None:
            completed = completed and result.get("finished") is True
        candidate_flow_reuse_expected = bool(scenario.get("candidate_reuse_expected", False))
        candidate_flow_reused = _candidate_reused(result, calls)

        run_metrics.append(RunMetrics(
            run_id=str(run["run_id"]),
            scenario_id=str(run["scenario_id"]),
            lane=str(run["lane"]),
            repeat=int(run.get("repeat", 1)),
            reported_pass=reported_pass,
            completed=completed,
            cleanup_verified=cleanup_verified,
            verifier_pass=verifier_pass,
            false_pass=false_pass,
            calls=total_calls,
            accounting_status=accounting_status,
            accounting_issues=accounting_issues,
            post_finish_calls=post_finish_calls,
            unexpected_failures=unexpected_failures,
            comparison_fingerprint=fingerprint,
            analyze_calls=analyze_calls,
            redundant_analyze_calls=redundant_calls,
            recovery_calls=recovery_calls,
            duration_ms=duration_ms,
            within_time_limit=within_time_limit,
            evidence_expected=expected,
            evidence_resolved=resolved,
            evidence_completeness=completeness,
            candidate_flow_reuse_expected=candidate_flow_reuse_expected,
            candidate_flow_reused=candidate_flow_reused,
            candidate_flow_reuse_met=(
                not candidate_flow_reuse_expected or candidate_flow_reused
            ),
            bundle=str(bundle),
        ))

    lanes = _aggregate_lanes(run_metrics)
    baseline_lane = str(campaign.get("baseline_lane", "baseline"))
    candidate_lane = str(campaign.get("candidate_lane", "candidate"))
    return {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "baseline_lane": baseline_lane,
        "candidate_lane": candidate_lane,
        "runs": [asdict(metric) for metric in run_metrics],
        "lanes": lanes,
        "comparisons": _compare_scenarios(run_metrics, baseline_lane, candidate_lane),
    }


def _rate(values: Sequence[bool]) -> float | None:
    return None if not values else sum(values) / len(values)


def _mean(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else statistics.fmean(present)


def _median(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else float(statistics.median(present))


def _aggregate_lanes(metrics: Sequence[RunMetrics]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.lane].append(metric)
    output: list[dict[str, Any]] = []
    for lane in sorted(grouped):
        values = grouped[lane]
        known_false_passes = [value.false_pass for value in values if value.false_pass is not None]
        output.append({
            "lane": lane,
            "runs": len(values),
            "completion_rate": _rate([value.completed for value in values]),
            "cleanup_rate": _rate([value.cleanup_verified for value in values]),
            "false_passes": sum(value is True for value in known_false_passes),
            "verified_runs": len(known_false_passes),
            "accounting_verified_runs": sum(value.accounting_status == "verified_through_finish" for value in values),
            "accounting_incomplete_runs": sum(value.accounting_status in {"incomplete", "unavailable"} for value in values),
            "observed_caller_invocations": sum(value.calls or 0 for value in values),
            "all_attempt_calls_per_completed_task": (
                sum(value.calls or 0 for value in values) / sum(value.completed for value in values)
                if all(value.accounting_status == "verified_through_finish" for value in values)
                and any(value.completed for value in values) else None
            ),
            "median_calls": _median(value.calls for value in values),
            "median_redundant_analyze_calls": _median(value.redundant_analyze_calls for value in values),
            "median_recovery_calls": _median(value.recovery_calls for value in values),
            "median_duration_ms": _median(value.duration_ms for value in values),
            "mean_evidence_completeness": _mean(value.evidence_completeness for value in values),
            "candidate_reuse_rate": _rate([
                value.candidate_flow_reused
                for value in values
                if value.candidate_flow_reuse_expected
            ]),
        })
    return output


def _compare_scenarios(
    metrics: Sequence[RunMetrics], baseline_lane: str, candidate_lane: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RunMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.scenario_id, metric.lane)].append(metric)
    comparisons: list[dict[str, Any]] = []
    scenarios = sorted({metric.scenario_id for metric in metrics})
    for scenario_id in scenarios:
        baseline = grouped.get((scenario_id, baseline_lane), [])
        candidate = grouped.get((scenario_id, candidate_lane), [])
        if not baseline or not candidate:
            continue
        baseline_calls = _median(value.calls for value in baseline)
        candidate_calls = _median(value.calls for value in candidate)
        all_runs = [*baseline, *candidate]
        fingerprints = {value.comparison_fingerprint for value in all_runs}
        legacy = fingerprints == {None} and all(value.accounting_status == "legacy" for value in all_runs)
        compatible = legacy or (len(fingerprints) == 1 and None not in fingerprints)
        call_accounting_available = all(
            value.calls is not None and value.accounting_status not in {"incomplete", "unavailable"}
            for value in [*baseline, *candidate]
        )
        baseline_duration = _median(value.duration_ms for value in baseline)
        candidate_duration = _median(value.duration_ms for value in candidate)
        comparisons.append({
            "scenario_id": scenario_id,
            "baseline_runs": len(baseline),
            "candidate_runs": len(candidate),
            "comparison_status": "legacy_unverified" if legacy else "matched" if compatible else "incompatible_or_missing",
            "baseline_completion_rate": _rate([value.completed for value in baseline]),
            "candidate_completion_rate": _rate([value.completed for value in candidate]),
            "median_call_delta": (
                None if not compatible or not call_accounting_available or baseline_calls is None or candidate_calls is None else candidate_calls - baseline_calls
            ),
            "call_improvement_rate": (
                None
                if not compatible or not call_accounting_available or not baseline_calls or candidate_calls is None
                else (baseline_calls - candidate_calls) / baseline_calls
            ),
            "median_duration_delta_ms": (
                None
                if not compatible or baseline_duration is None or candidate_duration is None
                else candidate_duration - baseline_duration
            ),
            "duration_regression_rate": (
                None
                if not compatible or not baseline_duration or candidate_duration is None
                else (candidate_duration - baseline_duration) / baseline_duration
            ),
        })
    return comparisons


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def render_markdown(evaluation: Mapping[str, Any]) -> str:
    lines = [
        f"# Agent-loop evaluation: {evaluation['campaign_id']}",
        "",
        "## Lane summary",
        "",
        "| Lane | Runs | Completion | Cleanup | False passes | Median calls | Median duration | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane in evaluation["lanes"]:
        lines.append(
            "| {lane} | {runs} | {completion} | {cleanup} | {false_passes}/{verified} | "
            "{calls} | {duration} | {evidence} |".format(
                lane=lane["lane"],
                runs=lane["runs"],
                completion=_percent(lane["completion_rate"]),
                cleanup=_percent(lane["cleanup_rate"]),
                false_passes=lane["false_passes"],
                verified=lane["verified_runs"],
                calls=_number(lane["median_calls"]),
                duration=_number(lane["median_duration_ms"], " ms"),
                evidence=_percent(lane["mean_evidence_completeness"]),
            )
        )
    lines.extend([
        "",
        "## Baseline vs candidate",
        "",
        "| Scenario | Comparison | Completion baseline → candidate | Call improvement | Duration regression |",
        "|---|---|---:|---:|---:|",
    ])
    if evaluation["comparisons"]:
        for comparison in evaluation["comparisons"]:
            lines.append(
                "| {scenario_id} | {status} | {baseline} → {candidate} | {calls} | {duration} |".format(
                    scenario_id=comparison["scenario_id"],
                    status=comparison["comparison_status"],
                    baseline=_percent(comparison["baseline_completion_rate"]),
                    candidate=_percent(comparison["candidate_completion_rate"]),
                    calls=_percent(comparison["call_improvement_rate"]),
                    duration=_percent(comparison["duration_regression_rate"]),
                )
            )
    else:
        lines.append("| _No paired lanes_ | n/a | n/a | n/a | n/a |")
    lines.extend([
        "",
        "## Runs",
        "",
        "| Run | Scenario | Lane | Closed loop | Verifier | Calls | Accounting | After finish | Redundant analyze | Recovery | Evidence |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ])
    for run in evaluation["runs"]:
        verifier = "n/a" if run["verifier_pass"] is None else ("pass" if run["verifier_pass"] else "fail")
        lines.append(
            "| {run_id} | {scenario_id} | {lane} | {completed} | {verifier} | {calls} | "
            "{accounting} | {post} | {redundant} | {recovery} | {evidence} |".format(
                run_id=run["run_id"],
                scenario_id=run["scenario_id"],
                lane=run["lane"],
                completed="yes" if run["completed"] else "no",
                verifier=verifier,
                calls="n/a" if run["calls"] is None else run["calls"],
                accounting=run["accounting_status"],
                post="n/a" if run["post_finish_calls"] is None else run["post_finish_calls"],
                redundant="n/a" if run["redundant_analyze_calls"] is None else run["redundant_analyze_calls"],
                recovery=run["recovery_calls"],
                evidence=_percent(run["evidence_completeness"]),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AUA agent-loop session bundles")
    parser.add_argument("campaign", type=Path, help="campaign JSON matching campaign.schema.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for JSON and Markdown reports")
    args = parser.parse_args(argv)
    try:
        evaluation = evaluate_campaign(args.campaign)
    except CampaignError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "evaluation.md").write_text(render_markdown(evaluation), encoding="utf-8")
    print(args.output_dir / "evaluation.json")
    print(args.output_dir / "evaluation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
