"""Independent host-only benchmark for AUA's trusted action compiler.

This is deliberately outside the FunctionGemma curriculum generators.  It measures the trusted
side of the boundary before any model runs:

* did destination extraction isolate the requested target from visible alternatives?
* did the guarded model shortlist contain the exact oracle call when model choice was needed?
* did AUA's deterministic recommendation name the correct tap or recovery call?
* did controls that must be withheld produce a safe abstention?

All fixtures are fictional and value-free.  No Android device, hierarchy dump, journal, map, model,
or adapter is read.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import arrival_destination_terms
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    family: str
    goal: str
    target: str
    observation: AnalyzeResult
    oracle_call: dict[str, Any] | None
    expect_policy_candidate: bool = True


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    family: str
    target_extracted: bool
    oracle_offered: bool | None
    deterministic_correct: bool | None
    safe_abstain: bool | None
    offered_count: int
    failure: dict[str, Any] | None


_GROUPS: tuple[tuple[str, str, str, str], ...] = (
    ("Atlas archive", "Beacon board", "Cedar catalog", "Delta desk"),
    ("Saved articles", "Saved experiments", "Saved lessons", "Saved searches"),
    ("Amber center", "Birch center", "Coral center", "Dune center"),
    ("Advanced reports", "Report archive", "Report builder", "Report schedule"),
    ("Forest index", "Garden index", "Harbor index", "Island index"),
    ("Jasper library", "Kite library", "Lagoon library", "Meadow library"),
    ("North records", "Orchid records", "Prairie records", "Quartz records"),
    ("River studio", "Summit studio", "Timber studio", "Valley studio"),
)

_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "open_from_choices",
        "Open {target} from these Example destinations: {choices}.",
    ),
    (
        "tap_from_choices",
        "Tap {target} from these visible choices: {choices}.",
    ),
    (
        "choose_among",
        "Choose {target} among the available Example destinations: {choices}.",
    ),
    (
        "origin_then_select",
        "From the Example shelf, select {target} and verify its visible heading; choices are "
        "{choices}.",
    ),
)

_SUMMARIES = (
    "Review stored lessons",
    "Manage pinned topics",
    "Inspect connected examples",
    "Browse recent notes",
)


class _HostDevice:
    serial = "functiongemma-candidate-benchmark"


def _rid_tail(label: str) -> str:
    words = [word for word in label.replace("-", " ").split() if word]
    return words[0].casefold() + "".join(word.title() for word in words[1:])


def _control(
    element_id: int,
    label: str,
    summary: str,
    *,
    representation: str,
    enabled: bool = True,
) -> tuple[Element, dict[str, Any]]:
    rendered = f"{label} {summary}"
    top = 120 + element_id * 150
    common = {
        "id": element_id,
        "type": "android.widget.Button",
        "bounds": (20, top, 1040, top + 120),
        "center": (530, top + 60),
        "clickable": True,
        "enabled": enabled,
        "source": Source.hierarchy,
    }
    if representation == "text":
        return Element(text=rendered, **common), {
            "tool": "tap_and_analyze",
            "arguments": {"text": rendered},
        }
    if representation == "desc":
        return Element(content_desc=rendered, **common), {
            "tool": "tap_and_analyze",
            "arguments": {"desc": rendered},
        }
    rid = _rid_tail(label)
    return Element(
        text=rendered,
        resource_id=f"com.example.catalog:id/{rid}",
        **common,
    ), {
        "tool": "tap_and_analyze",
        "arguments": {"rid": rid},
    }


def _observation(
    case_id: str,
    elements: list[Element],
    *,
    stale_risk: str | None = None,
) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="com.example.catalog",
            source="hierarchy",
        ),
        elements=elements,
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            device_serial=_HostDevice.serial,
            fingerprint=f"compiler-{case_id}",
            stale_risk=stale_risk,
        ),
    )


def _passive(
    element_id: int,
    *,
    kind: str,
    text: str | None = None,
    scrollable: bool | None = None,
) -> Element:
    top = 100 + element_id * 140
    return Element(
        id=element_id,
        type=kind,
        text=text,
        bounds=(20, top, 1040, top + 110),
        center=(530, top + 55),
        enabled=True,
        scrollable=scrollable,
        window="app",
        source=Source.hierarchy,
    )


def benchmark_cases() -> list[BenchmarkCase]:
    """Return taps, deterministic recoveries, and fail-closed abstention cases."""

    cases: list[BenchmarkCase] = []
    representations = ("rid", "text", "desc", "rid")
    for group_index, group in enumerate(_GROUPS):
        for target_index, target in enumerate(group):
            for prompt_index, (family, prompt) in enumerate(_PROMPTS):
                case_id = f"g{group_index:02d}-t{target_index}-{family}"
                representation = representations[(group_index + prompt_index) % 4]
                rotation = (group_index + target_index + prompt_index) % len(group)
                ordered = [*group[rotation:], *group[:rotation]]
                elements: list[Element] = []
                calls: dict[str, dict[str, Any]] = {}
                for element_index, label in enumerate(ordered, start=1):
                    source_index = group.index(label)
                    element, call = _control(
                        element_index,
                        label,
                        _SUMMARIES[source_index],
                        representation=representation,
                    )
                    elements.append(element)
                    calls[label] = call
                cases.append(
                    BenchmarkCase(
                        case_id=case_id,
                        family=family,
                        goal=prompt.format(target=target, choices=", ".join(ordered)),
                        target=target,
                        observation=_observation(case_id, elements),
                        oracle_call=calls[target],
                    )
                )

    # Explicitly exercise the guard boundary. A disabled requested control must not become a
    # candidate or a deterministic action just because its label is an exact semantic match.
    for group_index, group in enumerate(_GROUPS[:4]):
        for target_index, target in enumerate(group):
            case_id = f"withheld-g{group_index:02d}-t{target_index}"
            element, _call = _control(
                1,
                target,
                _SUMMARIES[target_index],
                representation="rid",
                enabled=False,
            )
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    family="disabled_target_abstain",
                    goal=f"Open {target}",
                    target=target,
                    observation=_observation(case_id, [element]),
                    oracle_call=None,
                    expect_policy_candidate=False,
                )
            )

    recovery_groups = _GROUPS[:4]
    for group_index, group in enumerate(recovery_groups):
        for target_index, target in enumerate(group):
            summary = _SUMMARIES[target_index]

            stale_id = f"stale-g{group_index:02d}-t{target_index}"
            stale_target, _ = _control(1, target, summary, representation="rid")
            cases.append(
                BenchmarkCase(
                    case_id=stale_id,
                    family="stale_refresh",
                    goal=f"Open {target}",
                    target=target,
                    observation=_observation(
                        stale_id,
                        [stale_target],
                        stale_risk="the preceding action outcome is not established",
                    ),
                    oracle_call={
                        "tool": "analyze_screen",
                        "arguments": {"source": "hierarchy", "no_cache": True},
                    },
                    expect_policy_candidate=False,
                )
            )

            loading_id = f"loading-g{group_index:02d}-t{target_index}"
            loading_target, _ = _control(1, target, summary, representation="text")
            cases.append(
                BenchmarkCase(
                    case_id=loading_id,
                    family="named_loading_wait",
                    goal=f"Open {target}",
                    target=target,
                    observation=_observation(
                        loading_id,
                        [
                            loading_target,
                            _passive(
                                2,
                                kind="android.widget.TextView",
                                text="Loading Example records",
                            ),
                        ],
                    ),
                    oracle_call={
                        "tool": "await_and_analyze",
                        "arguments": {
                            "predicate": "!text:Loading",
                            "timeout_ms": 15000,
                            "poll_ms": 200,
                            "ignore_case": True,
                        },
                    },
                    expect_policy_candidate=False,
                )
            )

            progress_id = f"progress-g{group_index:02d}-t{target_index}"
            progress_target, _ = _control(1, target, summary, representation="desc")
            cases.append(
                BenchmarkCase(
                    case_id=progress_id,
                    family="unlabeled_progress_wait",
                    goal=f"Open {target}",
                    target=target,
                    observation=_observation(
                        progress_id,
                        [
                            progress_target,
                            _passive(2, kind="android.widget.ProgressBar"),
                        ],
                    ),
                    oracle_call={
                        "tool": "wait_changed_and_analyze",
                        "arguments": {"timeout_ms": 15000, "interval_ms": 150},
                    },
                    expect_policy_candidate=False,
                )
            )

            scroll_id = f"scroll-g{group_index:02d}-t{target_index}"
            cases.append(
                BenchmarkCase(
                    case_id=scroll_id,
                    family="scroll_to_reveal",
                    goal=f"Open {target}",
                    target=target,
                    observation=_observation(
                        scroll_id,
                        [
                            _passive(
                                1,
                                kind="androidx.recyclerview.widget.RecyclerView",
                                scrollable=True,
                            ),
                            _passive(
                                2,
                                kind="android.widget.TextView",
                                text="More Example destinations below",
                            ),
                        ],
                    ),
                    oracle_call={
                        "tool": "scroll_and_analyze",
                        "arguments": {"direction": "up", "percent": 70},
                    },
                    expect_policy_candidate=False,
                )
            )
    return cases


def _engine(cache_root: Path) -> Engine:
    config = Config(
        cache={"dir": str(cache_root / "cache")},
        memory={"enabled": False, "dir": str(cache_root / "memory")},
        policy={"enabled": True, "mode": "shadow", "max_candidates": 4},
    )
    return Engine(config, device=_HostDevice())  # type: ignore[arg-type]


def evaluate_cases(cases: Sequence[BenchmarkCase]) -> dict[str, Any]:
    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory(prefix="aua-candidate-benchmark-") as tmp:
        engine = _engine(Path(tmp))
        state = SimpleNamespace(session_id="benchmark-session", serial=_HostDevice.serial)
        for case in cases:
            phase = SimpleNamespace(
                id=f"phase-{case.case_id}",
                objective=case.goal,
                kind="verify",
                constraints=[],
                recommended_call=None,
            )
            candidates = engine._policy_tap_candidates(  # noqa: SLF001 - production seam under test
                state,
                phase,
                case.observation,
            )
            calls = [candidate.trusted_call() for candidate in candidates]
            recommended = engine._phase_recommended_call(  # noqa: SLF001
                state,
                phase,
                case.observation,
            )
            expected_terms = arrival_destination_terms(f"Open {case.target}")
            actual_terms = arrival_destination_terms(case.goal)
            target_extracted = actual_terms == expected_terms
            recommended_call = recommended.get("mcp") if isinstance(recommended, dict) else None
            if case.oracle_call is not None:
                oracle_offered = case.oracle_call in calls if case.expect_policy_candidate else None
                deterministic_correct: bool | None = recommended_call == case.oracle_call
                safe_abstain: bool | None = None
                failed = not (
                    target_extracted and deterministic_correct and (oracle_offered is not False)
                )
            else:
                oracle_offered = None
                deterministic_correct = None
                safe_abstain = not calls and (
                    not isinstance(recommended, dict)
                    or recommended.get("kind") == "manual_observation"
                )
                failed = not (target_extracted and safe_abstain)
            failure = None
            if failed:
                failure = {
                    "expected_terms": expected_terms,
                    "actual_terms": actual_terms,
                    "oracle_call": case.oracle_call,
                    "offered_calls": calls,
                    "recommended_call": recommended_call,
                    "recommended_kind": (
                        recommended.get("kind") if isinstance(recommended, dict) else None
                    ),
                }
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    family=case.family,
                    target_extracted=target_extracted,
                    oracle_offered=oracle_offered,
                    deterministic_correct=deterministic_correct,
                    safe_abstain=safe_abstain,
                    offered_count=len(calls),
                    failure=failure,
                )
            )

    policy_actionable = [result for result in results if result.oracle_offered is not None]
    actionable = [result for result in results if result.deterministic_correct is not None]
    recoveries = [
        result
        for result, case in zip(results, cases, strict=True)
        if case.oracle_call is not None and not case.expect_policy_candidate
    ]
    abstentions = [result for result in results if result.safe_abstain is not None]
    target_correct = sum(result.target_extracted for result in results)
    offered_correct = sum(bool(result.oracle_offered) for result in policy_actionable)
    deterministic_correct = sum(bool(result.deterministic_correct) for result in actionable)
    recovery_correct = sum(bool(result.deterministic_correct) for result in recoveries)
    abstain_correct = sum(bool(result.safe_abstain) for result in abstentions)
    failures = [
        {
            "case_id": result.case_id,
            "family": result.family,
            **(result.failure or {}),
        }
        for result in results
        if result.failure is not None
    ]
    passed = not failures
    return {
        "schema_version": 2,
        "benchmark": "aua-trusted-action-compiler-v2",
        "host_only": True,
        "cases": len(results),
        "actionable_cases": len(actionable),
        "policy_candidate_cases": len(policy_actionable),
        "deterministic_recovery_cases": len(recoveries),
        "abstention_cases": len(abstentions),
        "metrics": {
            "target_extraction_accuracy": target_correct / len(results),
            "oracle_action_offered_rate": offered_correct / len(policy_actionable),
            "deterministic_action_accuracy": deterministic_correct / len(actionable),
            "deterministic_recovery_accuracy": recovery_correct / len(recoveries),
            "safe_abstention_rate": abstain_correct / len(abstentions),
            "mean_offered_candidates": sum(result.offered_count for result in results)
            / len(results),
        },
        "failures": failures,
        "passed": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_cases(benchmark_cases())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
