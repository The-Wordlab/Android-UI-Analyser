"""Either AUA drove the run or the caller did, and the result must never leave that ambiguous.

Driving it with AUA's own model is what makes a prepared scenario cheap: the tap-by-tap loop stops
crossing back into the calling agent's context. But a caller that cannot tell a driven run from a
described one would happily report a verdict nobody produced, so every path here states which it
was, and why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.prepare_run import (
    bundled_controller,
    child_environment,
    controller_state,
    harness_command,
    manual_plan,
    run_scenario,
)
from conftest import make_config

SCENARIO: dict[str, Any] = {
    "name": "badge-shows-once",
    "package": "com.example.app",
    "goal": "the hub badge shows once on first open",
    "contract": "/tmp/badge.yaml",
    "setup": [{"step": "seed", "detail": "write the key"}],
    "answers": {"build": "/tmp/app-debug.apk", "seeding": "datastore", "scope": "ui"},
}


def _cfg(**controller: Any):
    return make_config(controller={"enabled": True, **controller})


def test_a_disabled_controller_hands_back_commands_and_says_so(tmp_path) -> None:
    result = run_scenario(make_config(), SCENARIO, output=tmp_path, environ={})

    assert result["driven_by"] == "caller"
    assert "controller.enabled is false" in result["why"]
    assert result["commands"][0].startswith("aua session start")
    assert "--contract /tmp/badge.yaml" in result["commands"][0]
    assert result["contract"] == "/tmp/badge.yaml"


def test_an_enabled_controller_with_no_key_is_not_half_available(tmp_path) -> None:
    state = controller_state(_cfg(), environ={})
    assert not state["available"]
    assert "OPEN_ROUTER_API_KEY is not set" in state["reason"]

    result = run_scenario(_cfg(), SCENARIO, output=tmp_path, environ={})
    assert result["driven_by"] == "caller"


def test_a_configured_controller_is_reported_ready() -> None:
    state = controller_state(_cfg(), environ={"OPEN_ROUTER_API_KEY": "sk-test"})
    assert state["available"] and state["reason"] == "ready"
    assert state["model"] == "or-deepseek-v4-flash-0731-low-open"


def test_the_controller_shipped_with_the_source_tree_is_found() -> None:
    script = bundled_controller()
    assert script is not None and script.name == "run_realapp.py"


def test_the_invocation_carries_the_contract_the_budget_and_the_judge_ladder(tmp_path) -> None:
    argv = harness_command(_cfg(), SCENARIO, output=tmp_path, command=["python", "run.py"])

    pairs = dict(zip(argv, argv[1:], strict=False))
    assert pairs["--contract"] == "/tmp/badge.yaml"
    assert pairs["--goal"] == SCENARIO["goal"]
    assert pairs["--package"] == "com.example.app"
    assert pairs["--apk"] == "/tmp/app-debug.apk"
    assert pairs["--judge-fallback"] == "or-gemma4-26b-thinking"
    assert float(pairs["--cost-limit-usd"]) > 0
    assert "--record" in argv
    # Only the strategy that means "start from nothing" is allowed to wipe the app.
    assert "--fresh" not in argv


def test_choosing_the_wipe_is_the_only_thing_that_wipes(tmp_path) -> None:
    scenario = {**SCENARIO, "answers": {**SCENARIO["answers"], "seeding": "reinstall"}}
    argv = harness_command(_cfg(), scenario, output=tmp_path, command=["python", "run.py"])
    assert "--fresh" in argv


def test_a_driven_run_returns_the_verdict_with_its_evidence(tmp_path) -> None:
    recorded: dict[str, Any] = {}

    def fake_execute(argv, **kwargs):
        recorded["argv"] = argv
        recorded["env"] = kwargs["env"]
        output = Path(kwargs["cwd"])
        (output / "result.json").write_text(
            json.dumps({"verdict": "passed", "steps": 8, "usd": 0.013})
        )
        (output / "verdict.md").write_text("# passed\n")
        (output / "journey.mp4").write_bytes(b"\x00" * 20)
        (output / "aua").mkdir()
        (output / "aua" / "01.png").write_bytes(b"\x89PNG")
        (output / "aua" / "calls.jsonl").write_text("{}\n")

        class Done:
            returncode = 0
            stderr = ""

        return Done()

    result = run_scenario(
        _cfg(),
        SCENARIO,
        output=tmp_path / "run",
        environ={"OPEN_ROUTER_API_KEY": "sk-test"},
        execute=fake_execute,
    )

    assert result["driven_by"] == "aua" and result["ok"]
    assert result["verdict"] == "passed"
    assert result["model"] == "or-deepseek-v4-flash-0731-low-open"
    assert result["evidence"]["counts"]["image"] == 1
    assert [entry["name"] for entry in result["evidence"]["videos"]] == ["journey.mp4"]
    assert result["evidence"]["report"].endswith("verdict.md")
    # The raw device calls came back, flagged rather than dropped.
    assert "aua/calls.jsonl" in result["evidence"]["review_before_publishing"]
    assert "AUA has not read them for you" in result["publishing"]


def test_a_controller_that_never_judged_is_blocked_not_failed(tmp_path) -> None:
    class Died:
        returncode = 3
        stderr = "boom"

    result = run_scenario(
        _cfg(),
        SCENARIO,
        output=tmp_path / "run",
        environ={"OPEN_ROUTER_API_KEY": "sk-test"},
        execute=lambda argv, **kwargs: Died(),
    )

    # Reporting a product verdict for an infrastructure failure is the one mistake a QA harness
    # must not make.
    assert result["verdict"] == "blocked"
    assert result["exit_code"] == 3
    assert "boom" in result["stderr_tail"]
    assert not result["ok"]


def test_the_child_imports_this_aua_rather_than_whichever_is_on_the_path() -> None:
    script = bundled_controller()
    assert script is not None
    env = child_environment(["python", str(script)], {"PYTHONPATH": "/somewhere"})
    assert env["PYTHONPATH"].endswith("/somewhere")
    assert env["PYTHONPATH"].split(":")[0].endswith("/src")


@pytest.mark.parametrize("seeding", ["datastore", "reinstall"])
def test_the_manual_plan_matches_the_seeding_that_was_agreed(tmp_path, seeding: str) -> None:
    scenario = {**SCENARIO, "answers": {**SCENARIO["answers"], "seeding": seeding}}
    plan = manual_plan(scenario, output=tmp_path, reason="no key")
    assert ("--fresh --yes" in plan["commands"][0]) == (seeding == "reinstall")
    assert plan["setup"] == SCENARIO["setup"]


def test_a_flag_gated_surface_and_its_setup_flow_reach_the_invocation(tmp_path) -> None:
    # A flag-gated surface is simply absent without its flag, and an absent surface looks exactly
    # like a broken one. Found on the first real app: the tab under test did not exist at all.
    scenario = {
        **SCENARIO,
        "answers": {
            **SCENARIO["answers"],
            "flags": "myFeatureExperiment=a, otherExperiment=b",
            "setup_flow": "flows/common/enter-as-guest.yaml",
        },
    }
    argv = harness_command(_cfg(), scenario, output=tmp_path, command=["python", "run.py"])

    assert argv.count("--flags") == 2
    assert "myFeatureExperiment=a" in argv and "otherExperiment=b" in argv
    assert argv[argv.index("--setup-flow") + 1] == "flows/common/enter-as-guest.yaml"


def test_saying_none_is_not_a_flag(tmp_path) -> None:
    scenario = {
        **SCENARIO,
        "answers": {**SCENARIO["answers"], "flags": "none", "setup_flow": "none"},
    }
    argv = harness_command(_cfg(), scenario, output=tmp_path, command=["python", "run.py"])
    assert "--flags" not in argv and "--setup-flow" not in argv
