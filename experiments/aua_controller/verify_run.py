"""Offline pilot oracle over AUA-owned evidence, independent of the model's report.

This reuses AUA's assertion grammar/evaluator; it is not an independent implementation of
Android perception. Evidence is trusted only when produced by the harness's AUA process.
Files are checked and assertions replayed, rather than accepting an evidence ID as proof.
Missing or unusable evidence gives ``verified=False`` and ``false_pass=None``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from android_ui_analyser.assertions import evaluate_assertion_step
from android_ui_analyser.errors import AuaError
from android_ui_analyser.schema import AnalyzeResult
from android_ui_analyser.session_artifacts import observation_evidence_id
from android_ui_analyser.session_contracts import load_session_contract

CONTRACT_DIR = Path(__file__).with_name("contracts")
ORACLE_VERSION = "aua_contract_replay_v2"
SCENARIO_GOALS = {
    "classic-sort": (
        "Open the Classic View grid. Verify all four products initially appear in name order "
        "with correct product-price pairing. Sort by ascending price and verify the complete "
        "order and every product-price pairing. Restore name order and verify it before "
        "returning to the fixture home. Finish only after all checks and cleanup pass."
    ),
    "compose-sort": (
        "Open the Compose grid. Verify all four products initially appear in name order "
        "with correct product-price pairing. Sort by ascending price and verify the complete "
        "order and every product-price pairing. Restore name order and verify it before "
        "returning to the fixture home. Finish only after all checks and cleanup pass."
    ),
    "async-recovery": (
        "Open Async recovery. Observe the initial temporary signal error and available retry "
        "control. Recover using the UI and verify the expedition is ready and the retry "
        "control is gone. Return to the fixture home. Finish only after all checks and cleanup pass."
    ),
}


def _object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _asset(root: Path, recorded: Any) -> Path:
    # AUA records absolute paths. Rebase only the fixed evidence/<filename> suffix so a
    # copied bundle stays portable and neither traversal nor symlinks read outside it.
    if not isinstance(recorded, str):
        raise ValueError("missing artifact path")
    parts = Path(recorded).parts
    if len(parts) < 2 or parts[-2] != "evidence" or parts[-1] in {".", ".."}:
        raise ValueError("artifact must be an evidence/<filename> asset")
    path = (root / "evidence" / parts[-1]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing, empty, or external artifact: {parts[-1]}")
    return path


def _observation(result: dict[str, Any]) -> dict[str, Any] | None:
    nested = result.get("observation")
    if isinstance(nested, dict):
        return nested
    if isinstance(result.get("screen"), dict) and isinstance(result.get("elements"), list):
        return result
    return None


def _progress(result: dict[str, Any]) -> dict[str, Any]:
    direct = result.get("goal_progress")
    if isinstance(direct, dict):
        return direct
    observation = _observation(result) or {}
    nested = (observation.get("meta") or {}).get("goal_progress")
    return nested if isinstance(nested, dict) else {}


def _assertions(steps: list[Any], observation: AnalyzeResult) -> list[dict[str, Any]]:
    rows = []
    for index, step in enumerate(steps):
        verdict = evaluate_assertion_step(step, observation.elements)
        rows.append({"index": index, "passed": verdict.ok, "detail": verdict.detail})
    return rows


def _frame(
    root: Path, entry: dict[str, Any], call: dict[str, Any], session_id: str
) -> AnalyzeResult:
    if entry.get("invocation_id") != call.get("invocation_id"):
        raise ValueError("manifest/call invocation mismatch")
    raw = _object(_asset(root, entry.get("observation")))
    # The bundle redactor also masks Element.password (a tri-state UI flag). None means
    # unknown, not false; none of these pilot assertions depend on password state.
    # coaching.py attaches goal_progress at the top level of direct analyze dicts after
    # perception. It is response decoration, not an AnalyzeResult field. Keep the raw
    # asset for identity/record comparisons and normalize only this documented key.
    # _frame is shared by checkpoint and post-cleanup replay, so both use the same rule.
    parsed = {
        **{key: value for key, value in raw.items() if key != "goal_progress"},
        "elements": [
            {**element, "password": None} if element.get("password") == "<redacted>" else element
            for element in raw.get("elements", [])
        ],
    }
    observed = AnalyzeResult.model_validate(parsed)
    if observed.screen.package != "dev.aua.fixture":
        raise ValueError("observation is not the public fixture")
    if not observed.meta.fingerprint or observed.meta.stale_risk:
        raise ValueError("observation lacks a fresh fingerprint")
    if observation_evidence_id(session_id, raw) != entry.get("evidence_id"):
        raise ValueError("observation/evidence identity mismatch")
    archived_call = _observation(call.get("result") or {})
    if (
        not archived_call
        or any(raw.get(key) != archived_call.get(key) for key in ("screen", "elements"))
        or (archived_call.get("meta") or {}).get("fingerprint") != observed.meta.fingerprint
    ):
        raise ValueError("observation asset does not match its recorded call")
    with Image.open(_asset(root, entry.get("screenshot"))) as screenshot:
        if screenshot.format != "PNG":
            raise ValueError("screenshot is not a PNG")
        screenshot.verify()
    return observed


def verify_run(bundle_dir: Path, scenario_id: str) -> dict[str, Any]:
    """Verify an actual AUA bundle directory; never invoke a model, AUA, or a device.

    ``passed`` requires completed AUA lifecycle, the exact pilot contract, ordered automatic
    proofs, usable observation and PNG files, and successful offline assertion replay.
    ``false_pass`` compares AUA's completion report to this oracle only when verified; the
    runner separately compares the model's finish request to this outcome.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "oracle": ORACLE_VERSION,
        "passed": False,
        "verified": False,
        "reported_pass": None,
        "false_pass": None,
        "cleanup_verified": False,
        "checkpoints": [],
        "reasons": [],
        "evidence_expected": None,
        "evidence_resolved": 0,
    }
    try:
        if scenario_id not in SCENARIO_GOALS:
            raise ValueError("unknown pilot scenario")
        root = Path(bundle_dir).resolve()
        contract = load_session_contract(file=CONTRACT_DIR / f"{scenario_id}.yaml")
        assert contract.cleanup is not None
        expected = [(p.id, p.assertions) for p in contract.checkpoints]
        expected.append(("cleanup", contract.cleanup.assertions))
        report["evidence_expected"] = len(expected)
        manifest = _object(root / "manifest.json")
        result = _object(root / "result.json")
        report["reported_pass"] = (
            result.get("finished") is True or result.get("verdict") == "passed"
        )
        stored_contract = load_session_contract(file=root / "contract.yaml")
        if stored_contract != contract:
            raise ValueError("bundle contract differs from the authored pilot contract")
        session_id = manifest.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id
            or result.get("session_id") != session_id
        ):
            raise ValueError("bundle session identity mismatch")
        if manifest.get("evidence") != "all":
            raise ValueError("pilot requires evidence=all")
        entries = manifest.get("entries")
        phases = (result.get("goal_progress") or {}).get("phases")
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise ValueError("manifest entries are missing or invalid")
        if not isinstance(phases, list) or [p.get("id") for p in phases] != [
            p[0] for p in expected
        ]:
            raise ValueError("result lacks the exact ordered pilot phases")
        calls = [
            json.loads(line)
            for line in (root / "calls.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not all(isinstance(c, dict) for c in calls):
            raise ValueError("invalid call record")
        call_by_sequence = {c.get("sequence"): c for c in calls}
        sequences = [e.get("sequence") for e in entries]
        if (
            any(type(s) is not int or s <= 0 for s in sequences)
            or sequences != sorted(set(sequences))
            or len(call_by_sequence) != len(calls)
            or set(call_by_sequence) != set(sequences)
        ):
            raise ValueError("call/manifest sequence mismatch")

        previous_sequence = 0
        previous_capture_order: int | None = None
        selected: dict[str, int] = {}
        for phase_number, ((phase_id, steps), phase) in enumerate(
            zip(expected, phases, strict=True), 1
        ):
            row: dict[str, Any] = {"id": phase_id, "verified": False, "passed": False}
            report["checkpoints"].append(row)
            try:
                proof = phase.get("proof") or {}
                if (
                    phase.get("status") != "completed"
                    or phase.get("proof_mode") != "fresh_assertions"
                    or phase.get("manual_completion_allowed") is not False
                    or proof.get("source") != "contract_assertions"
                    or proof.get("verified") is not True
                    or proof.get("assertions_verified") != len(steps)
                ):
                    raise ValueError("missing completed automatic assertion proof")
                capture_order = proof.get("capture_order")
                if capture_order is not None and (
                    type(capture_order) is not int or capture_order < 0
                    or (previous_capture_order is not None and capture_order <= previous_capture_order)
                ):
                    raise ValueError("automatic proof capture order is not strictly increasing")
                # Progress is optional response decoration, not the frame's proof identity.
                # Live AUA calls can archive goal_progress=null even when the final session
                # contains the successful automatic proof. Repeated screens also share an
                # evidence ID. Require a distinct later invocation with the matching actual
                # frame, and reject contradictory progress when it was recorded.
                candidates = [
                    e
                    for e in entries
                    if e["sequence"] > previous_sequence
                    and e.get("evidence_id") == proof.get("evidence_id")
                    and _progress(call_by_sequence[e["sequence"]].get("result") or {}).get(
                        "completed"
                    ) in (None, phase_number)
                ]
                if not candidates:
                    raise ValueError("proof has no correctly ordered recorded observation")
                entry = candidates[0]
                observed = _frame(root, entry, call_by_sequence[entry["sequence"]], session_id)
                provenance = proof.get("observation") or {}
                if (
                    provenance.get("fingerprint") != observed.meta.fingerprint
                    or provenance.get("package") != observed.screen.package
                    or provenance.get("device_serial") != observed.meta.device_serial
                ):
                    raise ValueError("proof/observation provenance mismatch")
                row["assertions"] = _assertions(steps, observed)
                row.update(
                    verified=True,
                    passed=all(v["passed"] for v in row["assertions"]),
                    sequence=entry["sequence"],
                    progress_recorded=_progress(
                        call_by_sequence[entry["sequence"]].get("result") or {}
                    ).get("completed") is not None,
                )
                previous_sequence = entry["sequence"]
                if capture_order is not None:
                    previous_capture_order = capture_order
                selected[phase_id] = previous_sequence
                report["evidence_resolved"] += 1
            except (AuaError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                row["reason"] = str(exc)

        # AUA checkpoints are historical. Do not allow a later frame to undo restored order
        # or leave home after earning its checkpoint and then reuse that old completion.
        tail_checks = []
        boundary = selected.get("name_order_restored", selected.get("cleanup"))
        if boundary is not None:
            restored = next(
                (steps for key, steps in expected if key == "name_order_restored"), None
            )
            for entry in entries:
                if entry["sequence"] <= boundary:
                    continue
                call = call_by_sequence[entry["sequence"]]
                if not entry.get("observation"):
                    if _observation(call.get("result") or {}) is not None or call.get(
                        "command"
                    ) not in {"session_finish", "session_progress"}:
                        raise ValueError("post-cleanup call has no observable outcome")
                    continue
                observed = _frame(root, entry, call, session_id)
                home = all(v["passed"] for v in _assertions(contract.cleanup.assertions, observed))
                before_home = entry["sequence"] < selected.get("cleanup", 0)
                still_restored = bool(restored) and all(
                    v["passed"] for v in _assertions(restored or [], observed)
                )
                tail_checks.append(
                    {
                        "sequence": entry["sequence"],
                        "passed": home or (before_home and still_restored),
                    }
                )
        report["post_cleanup_observations"] = tail_checks
        report["verified"] = all(row["verified"] for row in report["checkpoints"])
        terminal = (
            all(result.get(key) is True for key in ("ok", "finished", "terminated"))
            and manifest.get("verdict") == "passed"
        )
        proof_pass = all(row["passed"] for row in report["checkpoints"])
        tail_pass = all(row["passed"] for row in tail_checks)
        report["cleanup_verified"] = (
            report["verified"]
            and tail_pass
            and all(
                row["passed"]
                for row in report["checkpoints"]
                if row["id"] in {"cleanup", "name_order_restored"}
            )
        )
        report["passed"] = report["verified"] and terminal and proof_pass and tail_pass
        if report["verified"]:
            report["false_pass"] = report["reported_pass"] and not report["passed"]
        if not terminal:
            report["reasons"].append("AUA session did not finish successfully")
        if not proof_pass:
            report["reasons"].append("one or more checkpoint proofs failed or lack evidence")
        if not tail_pass:
            report["reasons"].append("a later observation undid cleanup")
    except (AuaError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        report["reasons"].append(str(exc))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("scenario_id", choices=SCENARIO_GOALS)
    args = parser.parse_args()
    report = verify_run(args.bundle_dir, args.scenario_id)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
