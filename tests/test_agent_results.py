"""The caller sees errors and existing UI evidence consistently across transports."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.agent_results import from_cli, normalize_result
from android_ui_analyser.errors import SelectorNotFoundError
from android_ui_analyser.observation_contract import build_observation_contract
from android_ui_analyser.projection import Projection, trim_observation_payload
from android_ui_analyser.schema import ActionResult, AnalyzeResult


def _observation() -> dict[str, Any]:
    return AnalyzeResult.model_validate(
        {
            "screen": {"width": 400, "height": 800, "source": "hierarchy"},
            "elements": [
                {
                    "id": 1,
                    "stable_key": "rid:continue",
                    "type": "button",
                    "text": "Continue",
                    "bounds": [0, 0, 100, 40],
                    "center": [50, 20],
                    "clickable": True,
                }
            ],
            "meta": {"duration_ms": 3, "tier_used": "hierarchy", "path": "hierarchy"},
        }
    ).as_dict("json")


def test_real_analyze_and_action_producers_have_one_observation_location() -> None:
    observation = _observation()
    action = json.loads(
        ActionResult(
            ok=True,
            action="tap",
            detail="Action sent.",
            observation=AnalyzeResult.model_validate(observation),
        ).render("json")
    )

    analyzed = normalize_result(observation, command="analyze_screen")
    acted = normalize_result(action, command="tap_and_analyze")

    assert analyzed["schema_version"] == acted["schema_version"] == 1
    assert analyzed["ok"] is acted["ok"] is True
    assert analyzed["observation"] == acted["observation"] == observation
    assert analyzed["result"] == {}
    assert acted["result"]["action"] == "tap"
    assert acted["result"]["detail"] == "Action sent."
    assert "observation" not in acted["result"]
    assert analyzed["observation_contract"]["readiness"] == "not_checked"


def test_selector_error_keeps_already_returned_screen_and_exact_error() -> None:
    error = SelectorNotFoundError(
        "No target matched.",
        hint="Use the returned controls.",
        observation=_observation(),
    ).to_dict()

    out = normalize_result(error, command="tap_and_analyze")

    assert out["ok"] is False
    assert out["error"] == {
        "code": "selector_not_found",
        "message": "No target matched.",
        "hint": "Use the returned controls.",
        "observation_present": True,
    }
    assert out["observation"] == _observation()
    assert out["observation_contract"]["reusable"] is True
    assert out["result"] == {}


def test_uncertain_delivery_preserves_error_and_recovery_without_duplicate_ui() -> None:
    payload = {
        "error": {
            "code": "mic_delivery_uncertain",
            "message": "Delivery is uncertain.",
            "hint": "Inspect this frame before deciding what to do.",
            "followup_errors": [{"code": "release_failed"}],
            "result": {
                "ok": False,
                "action": "mic",
                "delivery": "unknown",
                "observation": _observation(),
            },
        }
    }

    out = normalize_result(payload, command="mic_and_analyze", exit_code=3)

    assert out["ok"] is False
    assert out["error"]["code"] == "mic_delivery_uncertain"
    assert out["error"]["followup_errors"] == [{"code": "release_failed"}]
    assert "result" not in out["error"]
    assert out["observation"] == _observation()
    assert out["observation_contract"]["action_succeeded"] is False
    assert out["result"] == {"ok": False, "action": "mic", "delivery": "unknown"}


def test_a_partial_success_in_an_error_is_not_overall_action_success() -> None:
    payload = {
        "error": {
            "code": "mic_delivery_uncertain",
            "result": {
                "ok": True,
                "action": "mic",
                "observation": _observation(),
            },
        }
    }

    out = normalize_result(payload, command="mic_and_analyze")

    assert out["ok"] is False
    assert out["result"]["ok"] is True
    assert "action_succeeded" not in out["observation_contract"]
    assert out["observation"] == _observation()


@pytest.mark.parametrize(
    "extra,readiness,analyze_needed",
    [
        ({"await_outcome": "timeout"}, "unmet", False),
        ({"settled_unmet": True}, "unmet", False),
        ({"arrival": {"state": "transitioning"}}, "unconfirmed", False),
        ({"stale_risk": "The frame predates the action."}, "unconfirmed", True),
    ],
)
def test_raw_readiness_caveats_override_positive_existing_contract(
    extra: dict[str, Any],
    readiness: str,
    analyze_needed: bool,
) -> None:
    payload = {"ok": True, "action": "tap", "observation": _observation()}
    payload["observation_contract"] = build_observation_contract(payload, command="tap")
    payload.update(extra)

    contract = normalize_result(payload, command="tap_and_analyze")["observation_contract"]

    assert contract["reusable"] is False
    assert contract["readiness"] == readiness
    assert contract["analyze_needed"] is analyze_needed


def test_producer_refusal_survives_a_projection_that_omits_the_original_caveat() -> None:
    observation = _observation()
    observation["meta"]["observation_contract"] = {
        "reusable": False,
        "evidence_fresh": False,
        "analyze_needed": True,
        "fingerprint": "frame-a",
        "evidence_id": "observation-a",
    }

    out = normalize_result(observation, command="analyze_screen")

    assert out["observation_contract"]["reusable"] is False
    assert out["observation_contract"]["evidence_fresh"] is False
    assert out["observation_contract"]["analyze_needed"] is True
    assert out["observation_contract"]["evidence_id"] == "observation-a"
    assert out["observation_contract"]["fingerprint"] == "frame-a"
    assert "observation_contract" not in out["observation"]["meta"]


def test_projection_is_not_expanded_or_replaced_by_empty_screen_guess() -> None:
    payload = {"ok": True, "action": "tap", "observation": _observation()}
    projected = trim_observation_payload(
        payload,
        Projection.parse(fields="id,text", no_meta=True, where_text=["absent"]),
    )

    out = normalize_result(projected, command="tap_and_analyze")

    assert out["ok"] is True
    assert out["observation"] == projected["observation"]
    assert out["observation"]["elements"] == []
    assert "meta" not in out["observation"]
    assert out["observation_contract"]["elements_available"] is False
    assert out["observation_contract"]["reusable"] is False


def test_empty_semantic_view_reuses_existing_image_without_loading_it(monkeypatch) -> None:
    def forbidden_read(*args, **kwargs):
        pytest.fail("Normalization must not read or capture an image.")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    observation = _observation()
    observation["elements"] = []
    observation["meta"]["raw_image"] = "/fictional/already-returned.png"

    out = normalize_result(observation, command="analyze_screen")

    assert out["observation"]["elements"] == []
    assert out["observation_contract"]["image_path"] == "/fictional/already-returned.png"
    assert out["observation_contract"]["evidence_fresh"] is True
    assert out["observation_contract"]["reusable"] is False
    assert out["observation_contract"]["analyze_needed"] is False


def test_a_summary_count_is_not_a_list_of_addressable_controls() -> None:
    out = normalize_result(
        {"ok": True, "observation": {"elements_count": 12}},
        command="analyze_screen",
    )

    assert out["observation"] == {"elements_count": 12}
    assert out["observation_contract"]["elements_available"] is False
    assert out["observation_contract"]["reusable"] is False


def test_capture_session_id_never_replaces_supplied_goal_context() -> None:
    context = {"session_id": "goal-session", "owner": "caller-a", "target_id": "target-a"}
    out = normalize_result(
        {"ok": True, "session_id": "capture-buffer", "path": "/fictional/sheet.png"},
        command="capture_sheet",
        context=context,
    )

    assert out["context"] == context
    assert out["result"]["session_id"] == "capture-buffer"
    assert out["observation"] is None
    assert out["observation_contract"]["previous_observation_validity"] == "unchanged"


@pytest.mark.parametrize("payload", [[], [{"serial": "target-a"}], {"capabilities": []}])
def test_non_ui_inventory_payloads_remain_valid(payload) -> None:
    out = normalize_result(payload, command="list_devices")

    assert out["ok"] is True
    assert out["result"] == payload
    assert out["observation"] is None


@pytest.mark.parametrize(
    "payload", [None, "", "not json", 0, True, {}, {"ok": "false"}, {"error": "failure"}]
)
def test_unknown_or_malformed_payloads_fail_explicitly(payload) -> None:
    out = normalize_result(payload, command="analyze_screen")

    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_response"


@pytest.mark.parametrize(
    "payload,kwargs,error_code",
    [
        ({"ok": False, "detail": "Condition absent."}, {}, None),
        ({"ok": True}, {"exit_code": 3}, "cli_exit_error"),
        ({"ok": True}, {"transport_error": True}, "transport_error"),
    ],
)
def test_success_is_not_inferred_from_a_transport_or_action_failure(
    payload, kwargs, error_code
) -> None:
    out = normalize_result(payload, command="wait_and_analyze", **kwargs)

    assert out["ok"] is False
    assert (out["error"] or {}).get("code") == error_code
    assert out["result"] == payload


def test_cli_stderr_error_wins_even_when_stdout_and_exit_claim_success() -> None:
    payload = {"ok": True, "action": "tap", "observation": _observation()}
    stderr = 'Progress: waiting\n{"error":{"code":"device_leased","message":"Target busy.","hint":"Keep caller context."}}\n'

    out = from_cli(json.dumps(payload), stderr, 0, command="tap-and-analyze")

    assert out["ok"] is False
    assert out["error"]["code"] == "device_leased"
    assert out["error"]["hint"] == "Keep caller context."
    assert out["observation"] == _observation()
    assert out["result"]["action"] == "tap"


def test_cli_typed_error_preserves_attached_observation_and_non_ui_result() -> None:
    error = {
        "error": {
            "code": "delivery_unknown",
            "result": {
                "ok": False,
                "action": "tap",
                "detail": "No replay.",
                "observation": _observation(),
            },
        }
    }

    out = from_cli("", json.dumps(error), 3, command="tap-and-analyze")

    assert out["ok"] is False
    assert out["observation"] == _observation()
    assert out["result"] == {"ok": False, "action": "tap", "detail": "No replay."}


@pytest.mark.parametrize("attachment", ["result", "observation"])
def test_cli_error_recovery_frame_wins_over_earlier_stdout_frame(attachment: str) -> None:
    previous = _observation()
    recovery = _observation()
    recovery["elements"][0]["text"] = "Recovered control"
    error: dict[str, Any] = {"code": "selector_not_found"}
    error[attachment] = (
        {"ok": False, "action": "tap", "observation": recovery}
        if attachment == "result"
        else recovery
    )

    out = from_cli(
        json.dumps({"ok": True, "observation": previous}),
        json.dumps({"error": error}),
        3,
        command="tap-and-analyze",
    )

    assert out["ok"] is False
    assert out["observation"] == recovery
    assert "observation" not in out["result"]
    assert "observation" not in out["error"]
    assert "result" not in out["error"]


def test_malformed_contract_metadata_is_an_error_not_a_normalizer_crash() -> None:
    payload = {"ok": True, "await_outcome": [], "observation": _observation()}

    out = normalize_result(payload, command="wait_and_analyze")

    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_response"
    assert out["observation_contract"]["reusable"] is False
    assert out["observation"] == _observation()


def test_non_ui_error_result_details_are_preserved() -> None:
    payload = {"error": {"code": "operation_failed", "result": ["partial result"]}}

    out = normalize_result(payload, command="inventory")

    assert out["ok"] is False
    assert out["error"] == {"code": "operation_failed"}
    assert out["result"] == ["partial result"]


@pytest.mark.parametrize(
    "stdout,exit_code", [("", 0), ("", 1), ("true", 0), ("<html>Error</html>", 1)]
)
def test_cli_empty_and_unstructured_output_is_never_a_successful_screen(stdout, exit_code) -> None:
    out = from_cli(stdout, "", exit_code, command="has")

    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_response"
    assert out["observation"] is None


def test_cli_plain_stderr_logs_do_not_hide_a_valid_result() -> None:
    out = from_cli(json.dumps(_observation()), "Warning: provider is slow.\n", 0, command="analyze")

    assert out["ok"] is True
    assert out["error"] is None
    assert out["observation"] == _observation()


def test_cli_syntax_error_retains_the_diagnostic_instead_of_suggesting_an_empty_screen() -> None:
    stderr = "Usage: aua tap-and-analyze [OPTIONS] [ID]\nError: No such option: --wrong-flag\n"

    out = from_cli("", stderr, 2, command="tap-and-analyze")

    assert out["ok"] is False
    assert out["error"] == {
        "code": "invalid_response",
        "message": "AUA did not return JSON output.",
        "exit_code": 2,
        "diagnostic": stderr,
    }
    assert out["observation"] is None
    assert "elements_available" not in out["observation_contract"]


def test_unstructured_cli_diagnostic_is_bounded() -> None:
    stderr = "Error: No such command 'unknown'.\n" + "diagnostic details\n" * 200

    out = from_cli("", stderr, 2, command="unknown")

    assert out["error"]["diagnostic"] == stderr[:2000]
    assert out["error"]["exit_code"] == 2


def test_structured_cli_errors_do_not_duplicate_stderr_diagnostics() -> None:
    stderr = 'Progress: waiting\n{"error":{"code":"device_leased","message":"Target busy."}}\n'

    out = from_cli("", stderr, 9, command="session_start")

    assert out["error"] == {"code": "device_leased", "message": "Target busy."}


def test_normalization_does_not_mutate_or_alias_payload_and_context() -> None:
    payload = {"ok": True, "action": "tap", "observation": _observation()}
    context = {"session_id": "goal-a", "nested": {"owner": "caller-a"}}
    original = deepcopy(payload)

    out = normalize_result(payload, command="tap_and_analyze", context=context)
    out["observation"]["elements"].clear()
    out["context"]["nested"]["owner"] = "caller-b"

    assert payload == original
    assert context["nested"]["owner"] == "caller-a"
