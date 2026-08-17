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
        self.assertEqual(metrics["redundant_analyze_calls"], 1)

    def test_requires_existing_bundle_artifacts(self) -> None:
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

        with self.assertRaisesRegex(CampaignError, "result.json"):
            evaluate_campaign(campaign)

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
