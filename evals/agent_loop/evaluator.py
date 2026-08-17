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
    calls: int
    analyze_calls: int
    redundant_analyze_calls: int
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
    for position, record in enumerate(calls):
        if "analyze" not in _operation(record).replace(".", " ").split():
            continue
        redundant_value = _first(record, (("redundant",), ("metrics", "redundant")))
        analyze_needed = _first(
            record,
            (("observation_contract", "analyze_needed"),),
        )
        reason = str(_first(record, (("reason",), ("metrics", "reason"))) or "").lower()
        previous_analyze_needed = None
        if position > 0:
            previous_analyze_needed = _first(
                calls[position - 1],
                (
                    ("observation_contract", "analyze_needed"),
                    ("result", "observation_contract", "analyze_needed"),
                    ("result", "meta", "observation_contract", "analyze_needed"),
                    ("result", "observation", "meta", "observation_contract", "analyze_needed"),
                ),
            )
        if (
            redundant_value is True
            or analyze_needed is False
            or previous_analyze_needed is False
            or "redundant" in reason
            or "fresh observation" in reason
        ):
            redundant += 1
    recovery = sum(
        1
        for record in calls
        if record.get("recovery") is True
        or str(record.get("category", "")).lower() == "recovery"
        or "recovery" in str(record.get("reason", "")).lower()
    )

    # Early bundle implementations may store only aggregate counts in result.json.
    total = int(_first(result, (("metrics", "calls"), ("call_count",))) or len(calls))
    analyze_total = int(_first(result, (("metrics", "analyze_calls"),)) or len(analyze))
    redundant_total = int(_first(result, (("metrics", "redundant_analyze_calls"),)) or redundant)
    recovery_total = int(_first(result, (("metrics", "recovery_calls"), ("recovery_calls",))) or recovery)
    return total, analyze_total, redundant_total, recovery_total


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


def _manifest_evidence_ids(manifest: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    evidence = manifest.get("evidence")
    if isinstance(evidence, Mapping):
        found.update(str(key) for key in evidence)
    elif isinstance(evidence, list):
        for entry in evidence:
            if isinstance(entry, Mapping):
                identifier = entry.get("id") or entry.get("evidence_id")
                if identifier is not None:
                    found.add(str(identifier))
    entries = manifest.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            identifier = entry.get("evidence_id")
            captured = entry.get("observation") or entry.get("screenshot")
            if identifier is not None and captured:
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
        result = _load_json(bundle / "result.json")
        manifest = _load_json(bundle / "manifest.json")
        if not isinstance(result, Mapping) or not isinstance(manifest, Mapping):
            raise CampaignError(f"bundle result and manifest must be JSON objects: {bundle}")
        calls = _iter_jsonl(bundle / "calls.jsonl")

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
            loaded_verifier = _load_json((root / str(run["verifier"])).resolve())
            if not isinstance(loaded_verifier, Mapping):
                raise CampaignError(f"verifier must be a JSON object for run {run['run_id']}")
            verifier = loaded_verifier
        verifier_pass = None
        false_pass = None
        if verifier is not None:
            verifier_pass = _status_pass(
                _first(verifier, (("passed",), ("verdict",), ("status",)))
            ) and (not cleanup_required or _status_pass(
                _first(verifier, (("cleanup_verified",), ("cleanup", "verified"), ("cleanup", "status")))
            ))
            false_pass = reported_pass and not verifier_pass

        total_calls, analyze_calls, redundant_calls, recovery_calls = _call_metrics(calls, result)
        duration_ms = _float_or_none(
            _first(result, (("duration_ms",), ("metrics", "duration_ms")))
        ) or _float_or_none(_first(manifest, (("duration_ms",), ("metrics", "duration_ms"))))
        duration_ms = duration_ms or _manifest_duration_ms(manifest)
        time_limit_ms = float(scenario["time_limit_s"]) * 1000
        within_time_limit = None if duration_ms is None else duration_ms <= time_limit_ms
        completed = (
            reported_pass
            and checkpoints_pass
            and (cleanup_verified or not cleanup_required)
            and within_time_limit is True
        )

        evidence_proof = _checkpoint_evidence(result, required_checkpoints, cleanup_required)
        available_evidence = _manifest_evidence_ids(manifest)
        resolved = sum(1 for evidence_id in evidence_proof.values() if evidence_id in available_evidence)
        expected = len(evidence_proof)
        completeness = None if expected == 0 else resolved / expected
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
        baseline_duration = _median(value.duration_ms for value in baseline)
        candidate_duration = _median(value.duration_ms for value in candidate)
        comparisons.append({
            "scenario_id": scenario_id,
            "baseline_runs": len(baseline),
            "candidate_runs": len(candidate),
            "baseline_completion_rate": _rate([value.completed for value in baseline]),
            "candidate_completion_rate": _rate([value.completed for value in candidate]),
            "median_call_delta": (
                None if baseline_calls is None or candidate_calls is None else candidate_calls - baseline_calls
            ),
            "call_improvement_rate": (
                None
                if not baseline_calls or candidate_calls is None
                else (baseline_calls - candidate_calls) / baseline_calls
            ),
            "median_duration_delta_ms": (
                None
                if baseline_duration is None or candidate_duration is None
                else candidate_duration - baseline_duration
            ),
            "duration_regression_rate": (
                None
                if not baseline_duration or candidate_duration is None
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
        "| Scenario | Completion baseline → candidate | Call improvement | Duration regression |",
        "|---|---:|---:|---:|",
    ])
    if evaluation["comparisons"]:
        for comparison in evaluation["comparisons"]:
            lines.append(
                "| {scenario_id} | {baseline} → {candidate} | {calls} | {duration} |".format(
                    scenario_id=comparison["scenario_id"],
                    baseline=_percent(comparison["baseline_completion_rate"]),
                    candidate=_percent(comparison["candidate_completion_rate"]),
                    calls=_percent(comparison["call_improvement_rate"]),
                    duration=_percent(comparison["duration_regression_rate"]),
                )
            )
    else:
        lines.append("| _No paired lanes_ | n/a | n/a | n/a |")
    lines.extend([
        "",
        "## Runs",
        "",
        "| Run | Scenario | Lane | Closed loop | Verifier | Calls | Redundant analyze | Recovery | Evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in evaluation["runs"]:
        verifier = "n/a" if run["verifier_pass"] is None else ("pass" if run["verifier_pass"] else "fail")
        lines.append(
            "| {run_id} | {scenario_id} | {lane} | {completed} | {verifier} | {calls} | "
            "{redundant} | {recovery} | {evidence} |".format(
                run_id=run["run_id"],
                scenario_id=run["scenario_id"],
                lane=run["lane"],
                completed="yes" if run["completed"] else "no",
                verifier=verifier,
                calls=run["calls"],
                redundant=run["redundant_analyze_calls"],
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
