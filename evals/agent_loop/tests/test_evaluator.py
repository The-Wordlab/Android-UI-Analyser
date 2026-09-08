from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from evaluator import CampaignError, evaluate_campaign, render_markdown  # noqa: E402


class EvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_json(self, relative: str, value: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def write_bundle(
        self,
        name: str,
        *,
        calls: int,
        duration_ms: int,
        candidate_reused: bool = False,
        passed: bool = True,
    ) -> None:
        bundle = self.root / name
        bundle.mkdir(parents=True)
        self.write_json(
            f"{name}/result.json",
            {
                "verdict": "passed" if passed else "failed",
                "duration_ms": duration_ms,
                "metrics": {"calls": calls, "candidate_flow_reused": candidate_reused},
                "checkpoints": [
                    {"id": "sorted", "status": "passed", "evidence_id": "proof-sort"},
                ],
                "cleanup": {"status": "passed", "evidence_id": "proof-cleanup"},
            },
        )
        self.write_json(
            f"{name}/manifest.json",
            {
                "evidence": [
                    {"id": "proof-sort", "path": "evidence/sort.json"},
                    {"id": "proof-cleanup", "path": "evidence/cleanup.json"},
                ]
            },
        )
        self.write_json(f"{name}/evidence/sort.json", {"fixture": "sorted"})
        self.write_json(f"{name}/evidence/cleanup.json", {"fixture": "restored"})
        (bundle / "calls.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"event": "call", "operation": "tap"}),
                    json.dumps({
                        "event": "call",
                        "operation": "analyze",
                        "redundant": True,
                    }),
                    json.dumps({
                        "event": "call",
                        "operation": "wait",
                        "category": "recovery",
                    }),
                ]
            ) + "\n",
            encoding="utf-8",
        )

    def test_compares_lanes_and_scores_loop_evidence(self) -> None:
        self.write_bundle("baseline", calls=10, duration_ms=5000)
        self.write_bundle("candidate", calls=6, duration_ms=4500, candidate_reused=True)
        self.write_json("baseline-verifier.json", {"passed": True, "cleanup_verified": True})
        self.write_json("candidate-verifier.json", {"passed": True, "cleanup_verified": True})
        campaign = self.write_json(
            "campaign.json",
            {
                "schema_version": 1,
                "campaign_id": "test-campaign",
                "scenarios": [
                    {
                        "id": "sort",
                        "title": "Sort and restore",
                        "goal": "Sort fictional items and restore them.",
                        "time_limit_s": 10,
                        "required_checkpoints": ["sorted"],
                        "cleanup_required": True,
                    }
                ],
                "runs": [
                    {
                        "run_id": "baseline-1",
                        "scenario_id": "sort",
                        "lane": "baseline",
                        "bundle": "baseline",
                        "verifier": "baseline-verifier.json",
                    },
                    {
                        "run_id": "candidate-1",
                        "scenario_id": "sort",
                        "lane": "candidate",
                        "bundle": "candidate",
                        "verifier": "candidate-verifier.json",
                    },
                ],
            },
        )

        evaluation = evaluate_campaign(campaign)

        baseline, candidate = evaluation["runs"]
        self.assertTrue(baseline["completed"])
        self.assertEqual(baseline["evidence_completeness"], 1.0)
        self.assertEqual(baseline["redundant_analyze_calls"], 1)
        self.assertEqual(baseline["recovery_calls"], 1)
        self.assertTrue(candidate["candidate_flow_reused"])
        comparison = evaluation["comparisons"][0]
        self.assertEqual(comparison["median_call_delta"], -4.0)
        self.assertEqual(comparison["call_improvement_rate"], 0.4)
        self.assertIn("baseline-1", render_markdown(evaluation))

    def test_independent_verifier_detects_false_pass(self) -> None:
        self.write_bundle("bundle", calls=2, duration_ms=1000)
        self.write_json("verifier.json", {"passed": False, "cleanup_verified": False})
        campaign = self.write_json(
            "campaign.json",
            {
                "schema_version": 1,
                "campaign_id": "false-pass",
                "scenarios": [
                    {
                        "id": "scenario",
                        "title": "Scenario",
                        "goal": "Complete a fictional task.",
                        "time_limit_s": 5,
                        "required_checkpoints": ["sorted"],
                    }
                ],
                "runs": [
                    {
                        "run_id": "run",
                        "scenario_id": "scenario",
                        "lane": "candidate",
                        "bundle": "bundle",
                        "verifier": "verifier.json",
                    }
                ],
            },
        )

        evaluation = evaluate_campaign(campaign)

        self.assertTrue(evaluation["runs"][0]["reported_pass"])
        self.assertTrue(evaluation["runs"][0]["false_pass"])
        self.assertEqual(evaluation["lanes"][0]["false_passes"], 1)

    def test_pass_after_time_limit_does_not_count_as_closed_loop(self) -> None:
        self.write_bundle("bundle", calls=2, duration_ms=6000)
        campaign = self.write_json(
            "campaign.json",
            {
                "schema_version": 1,
                "campaign_id": "time-limit",
                "scenarios": [
                    {
                        "id": "scenario",
                        "title": "Scenario",
                        "goal": "Complete a fictional task.",
                        "time_limit_s": 5,
                        "required_checkpoints": ["sorted"],
                    }
                ],
                "runs": [
                    {
                        "run_id": "run",
                        "scenario_id": "scenario",
                        "lane": "candidate",
                        "bundle": "bundle",
                    }
                ],
            },
        )

        metrics = evaluate_campaign(campaign)["runs"][0]

        self.assertTrue(metrics["reported_pass"])
        self.assertFalse(metrics["within_time_limit"])
        self.assertFalse(metrics["completed"])

    def test_reads_native_session_phases_cleanup_and_manifest_duration(self) -> None:
        bundle = self.root / "bundle"
        bundle.mkdir()
        self.write_json(
            "bundle/result.json",
            {
                "finished": True,
                "goal_progress": {
                    "phases": [
                        {
                            "id": "sorted",
                            "kind": "verify",
                            "status": "completed",
                            "proof": {"evidence_id": "proof-sort"},
                        },
                        {
                            "id": "cleanup",
                            "kind": "cleanup",
                            "status": "completed",
                            "proof": {"evidence_id": "proof-cleanup"},
                        },
                    ]
                },
            },
        )
        self.write_json(
            "bundle/manifest.json",
            {
                "started_at": "2026-08-17T18:00:00+00:00",
                "finished_at": "2026-08-17T18:00:04.500000+00:00",
                "entries": [
                    {"evidence_id": "proof-sort", "observation": "evidence/sort.json"},
                    {
                        "evidence_id": "proof-cleanup",
                        "observation": "evidence/cleanup.json",
                    },
                ],
            },
        )
        self.write_json("bundle/evidence/sort.json", {"fixture": "sorted"})
        self.write_json("bundle/evidence/cleanup.json", {"fixture": "restored"})
        (bundle / "calls.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({
                        "command": "tap",
                        "result": {
                            "observation_contract": {"analyze_needed": False},
                        },
                    }),
                    json.dumps({"command": "analyze"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        campaign = self.write_json(
            "campaign.json",
            {
                "schema_version": 1,
                "campaign_id": "native-bundle",
                "scenarios": [
                    {
                        "id": "sort",
                        "title": "Sort and restore",
                        "goal": "Sort fictional items and restore them.",
                        "time_limit_s": 10,
                        "required_checkpoints": ["sorted"],
                        "cleanup_required": True,
                    }
                ],
                "runs": [
                    {
                        "run_id": "candidate-1",
                        "scenario_id": "sort",
                        "lane": "candidate",
                        "bundle": "bundle",
                    }
                ],
            },
        )

        metrics = evaluate_campaign(campaign)["runs"][0]

        self.assertTrue(metrics["completed"])
        self.assertTrue(metrics["cleanup_verified"])
        self.assertEqual(metrics["duration_ms"], 4500.0)
        self.assertEqual(metrics["evidence_completeness"], 1.0)
        self.assertEqual(metrics["redundant_analyze_calls"], 0)

    def test_missing_bundle_stays_in_attempt_denominator_with_unknown_calls(self) -> None:
        campaign = self.write_json(
            "campaign.json",
            {
                "schema_version": 1,
                "campaign_id": "missing",
                "scenarios": [
                    {
                        "id": "scenario",
                        "title": "Scenario",
                        "goal": "Complete a fictional task.",
                        "time_limit_s": 5,
                    }
                ],
                "runs": [
                    {
                        "run_id": "run",
                        "scenario_id": "scenario",
                        "lane": "candidate",
                        "bundle": "does-not-exist",
                    }
                ],
            },
        )

        evaluation = evaluate_campaign(campaign)
        self.assertFalse(evaluation["runs"][0]["completed"])
        self.assertIsNone(evaluation["runs"][0]["calls"])
        self.assertEqual(evaluation["runs"][0]["accounting_status"], "unavailable")
        self.assertEqual(evaluation["lanes"][0]["runs"], 1)

    def native_campaign(self, *, finished: bool = True, nonterminal_probe: bool = False) -> Path:
        from android_ui_analyser.session import SessionState, review_session_events

        self.write_bundle("native", calls=2, duration_ms=1000)
        state = SessionState(session_id="fixture-session", goal="Verify the fictional grid",
                             goal_hash="fixture-goal", serial="fixture-target", started_ms=1,
                             finished_ms=50, recommended_kind="manual", recommended_cli="aua analyze")
        events = [
            {"cmd": "session_start", "invocation_id": "start", "ok": True, "ts_ms": 2},
            {"cmd": "tap_and_analyze", "invocation_id": "tap", "ok": True, "ts_ms": 3},
            {"cmd": "await", "invocation_id": "wait", "ok": True, "ts_ms": 4,
             "args": {"adopt_action": True}},
            {"cmd": "tap_and_analyze", "invocation_id": "failed", "ok": False, "ts_ms": 5,
             "error": {"code": "element_not_found"}},
        ]
        if nonterminal_probe:
            events.append({"cmd": "session_finish", "invocation_id": "early-finish", "ok": False,
                           "ts_ms": 6, "result": {"finished": False, "terminated": False}})
        for event in events:
            event["session_id"] = state.session_id
            event["_detail_hydrated"] = True
        result = {"session_id": state.session_id, "finished": finished, "terminated": True,
                  "verdict": "passed" if finished else "incomplete", "duration_ms": 1000,
                  "review": review_session_events(state, events),
                  "checkpoints": [{"id": "sorted", "status": "passed" if finished else "pending",
                                   "evidence_id": "proof-sort"}],
                  "cleanup": [{"action": "lease_release", "ok": True, "result": {"released": True}}]}
        events += [
            {"cmd": "session_finish", "invocation_id": "finish", "ok": True, "ts_ms": 51, "result": result},
            {"cmd": "list_devices", "invocation_id": "devices", "ok": True, "ts_ms": 52, "result": []},
            {"cmd": "emulator_status", "invocation_id": "status", "ok": True, "ts_ms": 53, "result": {"ok": True}},
        ]
        for event in events:
            event["session_id"] = state.session_id
            event["_detail_hydrated"] = True
        self.write_json("native/session.json", state.model_dump(mode="json"))
        self.write_json("native/result.json", result)
        manifest = json.loads((self.root / "native/manifest.json").read_text())
        manifest.update(session_id=state.session_id, invocations=["start", "tap", "finish", "status"])
        self.write_json("native/manifest.json", manifest)
        (self.root / "native/journal.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        return self.write_json("native-campaign.json", {
            "schema_version": 1, "campaign_id": "native-fixture",
            "scenarios": [{"id": "fixture", "title": "Fixture", "goal": "Verify the grid",
                           "time_limit_s": 10, "required_checkpoints": ["sorted"]}],
            "runs": [{"run_id": "native", "scenario_id": "fixture", "lane": "candidate", "bundle": "native"}],
        })

    def test_native_journal_counts_errors_and_postfinish_but_folds_internal_wait(self) -> None:
        evaluation = evaluate_campaign(self.native_campaign())
        run = evaluation["runs"][0]
        self.assertEqual(run["accounting_status"], "verified_through_finish")
        self.assertEqual(run["calls"], 6)  # Four through finish, two after; artifacts are sparse.
        self.assertEqual(run["post_finish_calls"], 2)
        self.assertEqual(run["unexpected_failures"], 1)
        self.assertTrue(run["cleanup_verified"])
        self.assertTrue(run["completed"])
        self.assertEqual(evaluation["lanes"][0]["all_attempt_calls_per_completed_task"], 6)

    def test_dropped_internal_event_keeps_cost_but_marks_accounting_incomplete(self) -> None:
        campaign = self.native_campaign()
        path = self.root / "native/journal.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(event) + "\n" for event in events if event["cmd"] != "await"))
        evaluation = evaluate_campaign(campaign)
        run = evaluation["runs"][0]
        self.assertEqual(run["calls"], 6)
        self.assertEqual(run["accounting_status"], "incomplete")
        self.assertEqual(evaluation["lanes"][0]["runs"], 1)
        self.assertIsNone(evaluation["lanes"][0]["all_attempt_calls_per_completed_task"])

    def test_terminal_incomplete_is_a_failed_attempt_not_missing_data(self) -> None:
        evaluation = evaluate_campaign(self.native_campaign(finished=False))
        run = evaluation["runs"][0]
        self.assertEqual(run["accounting_status"], "verified_through_finish")
        self.assertFalse(run["completed"])
        self.assertEqual(run["calls"], 6)
        self.assertEqual(evaluation["lanes"][0]["completion_rate"], 0)

    def test_failed_nonterminal_finish_is_counted_without_selecting_it(self) -> None:
        run = evaluate_campaign(self.native_campaign(nonterminal_probe=True))["runs"][0]
        self.assertEqual(run["accounting_status"], "verified_through_finish")
        self.assertEqual(run["calls"], 7)
        self.assertEqual(run["unexpected_failures"], 2)

    def test_native_saved_review_without_journal_cannot_claim_zero_or_sparse_total(self) -> None:
        campaign = self.native_campaign()
        (self.root / "native/journal.jsonl").unlink()
        run = evaluate_campaign(campaign)["runs"][0]
        self.assertIsNone(run["calls"])
        self.assertEqual(run["accounting_status"], "unavailable")
        self.assertIsNone(run["redundant_analyze_calls"])

    def test_missing_evidence_file_prevents_completion(self) -> None:
        campaign = self.native_campaign()
        (self.root / "native/evidence/sort.json").unlink()
        run = evaluate_campaign(campaign)["runs"][0]
        self.assertTrue(run["reported_pass"])
        self.assertFalse(run["completed"])
        self.assertEqual(run["evidence_completeness"], 0.5)

    def test_pending_independent_review_is_unknown_and_cannot_complete(self) -> None:
        campaign = self.native_campaign()
        self.write_json("verifier.json", {"status": "pending_review", "passed": None, "cleanup_verified": None})
        value = json.loads(campaign.read_text())
        value["runs"][0]["verifier"] = "verifier.json"
        campaign.write_text(json.dumps(value))
        run = evaluate_campaign(campaign)["runs"][0]
        self.assertIsNone(run["verifier_pass"])
        self.assertIsNone(run["false_pass"])
        self.assertFalse(run["completed"])

    def test_missing_configured_verifier_is_unknown_and_preserves_attempt(self) -> None:
        campaign = self.native_campaign()
        value = json.loads(campaign.read_text())
        value["runs"][0]["verifier"] = "not-yet-written.json"
        campaign.write_text(json.dumps(value))
        evaluation = evaluate_campaign(campaign)
        self.assertIsNone(evaluation["runs"][0]["verifier_pass"])
        self.assertIsNone(evaluation["runs"][0]["false_pass"])
        self.assertFalse(evaluation["runs"][0]["completed"])
        self.assertEqual(evaluation["lanes"][0]["runs"], 1)

    def test_missing_hydration_proof_does_not_assert_zero_redundant_reads(self) -> None:
        campaign = self.native_campaign()
        self.assertEqual(evaluate_campaign(campaign)["runs"][0]["redundant_analyze_calls"], 0)
        path = self.root / "native/journal.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        for event in events:
            event.pop("_detail_hydrated", None)
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        run = evaluate_campaign(campaign)["runs"][0]
        self.assertEqual(run["accounting_status"], "verified_through_finish")
        self.assertIsNone(run["redundant_analyze_calls"])

    def test_native_comparison_requires_matching_execution_provenance(self) -> None:
        import shutil

        campaign = self.native_campaign()
        shutil.copytree(self.root / "native", self.root / "baseline")
        value = json.loads(campaign.read_text())
        value["runs"].append({"run_id": "baseline", "scenario_id": "fixture", "lane": "baseline", "bundle": "baseline"})
        campaign.write_text(json.dumps(value))
        comparison = evaluate_campaign(campaign)["comparisons"][0]
        self.assertEqual(comparison["comparison_status"], "incompatible_or_missing")
        self.assertIsNone(comparison["call_improvement_rate"])
        for bundle in ("native", "baseline"):
            self.write_json(f"{bundle}/execution.json", {"comparison_fingerprint": "a" * 64})
        comparison = evaluate_campaign(campaign)["comparisons"][0]
        self.assertEqual(comparison["comparison_status"], "matched")
        self.assertEqual(comparison["call_improvement_rate"], 0)
        self.write_json("native/execution.json", {"comparison_fingerprint": "b" * 64})
        comparison = evaluate_campaign(campaign)["comparisons"][0]
        self.assertIsNone(comparison["call_improvement_rate"])
        self.assertIsNone(comparison["duration_regression_rate"])

    def test_rejects_unknown_scenario_reference(self) -> None:
        campaign = self.write_json(
            "campaign.json",
            {
                "schema_version": 1,
                "campaign_id": "invalid",
                "scenarios": [
                    {"id": "one", "title": "One", "goal": "Do one.", "time_limit_s": 5}
                ],
                "runs": [
                    {
                        "run_id": "run",
                        "scenario_id": "two",
                        "lane": "candidate",
                        "bundle": "bundle",
                    }
                ],
            },
        )

        with self.assertRaisesRegex(CampaignError, "unknown scenario"):
            evaluate_campaign(campaign)


if __name__ == "__main__":
    unittest.main()
