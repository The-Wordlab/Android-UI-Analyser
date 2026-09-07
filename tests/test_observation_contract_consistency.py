"""Reuse guidance follows the returned view, without a second device observation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import coaching, journal
from android_ui_analyser.observation_contract import (
    build_observation_contract,
    refresh_observation_contract,
    result_has_reusable_observation,
)
from android_ui_analyser.projection import Projection, render_action_tsv, trim_observation_payload
from android_ui_analyser.schema import ActionResult, AnalyzeResult
from android_ui_analyser.session_artifacts import SessionArtifactStore


def _observation() -> dict[str, Any]:
    return {
        "screen": {"width": 400, "height": 800, "source": "hierarchy"},
        "elements": [
            {
                "id": "el:example",
                "type": "button",
                "text": "Continue",
                "bounds": [0, 0, 100, 40],
                "center": [50, 20],
                "clickable": True,
            }
        ],
        "meta": {
            "fingerprint": "example-frame",
            "duration_ms": 3,
            "tier_used": "hierarchy",
            "path": "hierarchy",
        },
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"settled_unmet": True},
        {"await_outcome": "timeout"},
        {"await_outcome": "settled-unmet"},
        {"arrival": {"state": "transitioning"}},
        {"stale_risk": "the frame may predate the action"},
    ],
)
def test_a_successful_action_does_not_claim_an_unmet_destination_is_ready(extra) -> None:
    result = {"ok": True, "action": "tap", "observation": _observation(), **extra}

    contract = build_observation_contract(result, command="tap")

    assert contract["action_succeeded"] is True
    assert contract["reusable"] is False
    assert contract["analyze_needed"] is bool(extra.get("stale_risk"))
    assert contract["readiness"] in {"unmet", "unconfirmed"}
    assert "fresh settled" not in contract["reason"]
    assert result_has_reusable_observation(result) is False


def test_plain_analyze_does_not_prove_a_destination_predicate() -> None:
    contract = build_observation_contract(_observation(), command="analyze")

    assert contract["reusable"] is True
    assert contract["evidence_fresh"] is True
    assert contract["readiness"] == "not_checked"


def test_a_satisfied_positive_predicate_can_confirm_readiness() -> None:
    result = {
        "ok": True,
        "action": "tap",
        "await_outcome": "satisfied",
        "observation": _observation(),
    }

    assert build_observation_contract(result, command="tap")["readiness"] == "ready"


def test_opaque_visual_evidence_is_fresh_without_claiming_semantic_elements() -> None:
    observation = _observation()
    observation["elements"] = []
    observation["meta"]["raw_image"] = "/example/returned-frame.png"
    result = {"ok": True, "action": "tap", "observation": observation}

    contract = build_observation_contract(result, command="tap")

    assert contract["evidence_fresh"] is True
    assert contract["elements_available"] is False
    assert contract["reusable"] is False
    assert contract["analyze_needed"] is False
    assert contract["image_path"] == "/example/returned-frame.png"
    assert "existing image" in contract["reason"]


@pytest.mark.parametrize("command", ["capture_sheet", "capture-export", "capture_last"])
def test_artifact_export_does_not_invalidate_prior_observation(command) -> None:
    contract = build_observation_contract({"ok": True, "action": command}, command=command)

    assert contract["reusable"] is False  # No new semantic observation was returned.
    assert contract["analyze_needed"] is False
    assert contract["previous_observation_validity"] == "unchanged"


def test_projection_rechecks_the_elements_that_were_actually_emitted() -> None:
    result = {
        "ok": True,
        "action": "tap",
        "await_outcome": "satisfied",
        "observation": _observation(),
    }
    result["observation_contract"] = build_observation_contract(result, command="tap")
    assert result["observation_contract"]["readiness"] == "ready"

    projected = trim_observation_payload(result, Projection.parse(where_text=["missing label"]))

    assert projected["observation"]["elements"] == []
    assert projected["observation_contract"]["reusable"] is False
    assert projected["observation_contract"]["readiness"] == "unconfirmed"
    assert result_has_reusable_observation(projected) is False


def test_analyze_projection_refreshes_metadata_without_changing_the_original() -> None:
    original = _observation()
    original["meta"]["observation_contract"] = build_observation_contract(
        original, command="analyze"
    )

    projected = Projection.parse(where_text=["missing label"]).apply(original)

    assert projected["meta"]["observation_contract"]["reusable"] is False
    assert original["meta"]["observation_contract"]["reusable"] is True


def test_tsv_action_contract_describes_only_the_emitted_rows() -> None:
    result = {"ok": True, "action": "tap", "observation": _observation()}
    result["observation_contract"] = build_observation_contract(result, command="tap")

    text = render_action_tsv(result, Projection.parse(where_text=["Missing"]))

    assert "# observation_contract.reusable=false" in text
    assert result["observation_contract"]["reusable"] is True


def test_tsv_analyze_metadata_contract_describes_only_the_emitted_rows() -> None:
    observation = _observation()
    observation["meta"]["observation_contract"] = build_observation_contract(
        observation, command="analyze"
    )
    view = Projection.parse(where_text=["Missing"], meta="observation_contract")

    text = view.render_tsv(observation)

    assert "'reusable': False" in text
    assert observation["meta"]["observation_contract"]["reusable"] is True


def test_refresh_after_adopting_a_wait_uses_its_new_outcome() -> None:
    result = {"ok": True, "action": "tap", "observation": _observation()}
    result["observation_contract"] = build_observation_contract(result, command="tap")
    result["await_outcome"] = "timeout"

    refresh_observation_contract(result)

    assert result["observation_contract"]["readiness"] == "unmet"
    assert result["observation_contract"]["action_succeeded"] is True


def test_compact_journal_summary_uses_the_same_reuse_decision() -> None:
    assert result_has_reusable_observation({"observation": {"elements_count": 2}})
    assert not result_has_reusable_observation({"observation": {"elements_count": 0}})
    assert not result_has_reusable_observation(
        {"observation": {"elements_count": 2}, "settled_unmet": True}
    )


def _store(tmp_path: Path, *, evidence: str = "none") -> SessionArtifactStore:
    return SessionArtifactStore.create(
        tmp_path / "run",
        session_id="example-session",
        goal="Verify the example catalog",
        evidence=evidence,
        junit=False,
        contract_yaml=None,
    )


def test_live_model_and_stored_artifact_have_the_same_contract(tmp_path: Path) -> None:
    store = _store(tmp_path)
    engine = SimpleNamespace(
        _session_state=lambda: SimpleNamespace(session_id="example-session", artifact_dir=None)
    )
    result = ActionResult(
        ok=True,
        action="wait_after_change",
        observation=AnalyzeResult.model_validate(_observation()),
        settled_unmet=True,
    )
    coaching._record_session_artifact(
        engine, "wait_after_change", result, invocation_id="call-one", duration_ms=1, args={}
    )
    stored = store.record(
        command="wait_after_change", result=result, invocation_id="call-one", duration_ms=1
    )

    assert result.observation_contract is not None
    assert result.observation_contract.model_dump(exclude_none=True) == {
        key: value for key, value in stored.items() if value is not None
    }


def test_storing_existing_image_never_takes_an_additional_device_screenshot(tmp_path: Path) -> None:
    image = tmp_path / "returned.png"
    image.write_bytes(b"the exact returned frame")
    observation = _observation()
    observation["meta"]["raw_image"] = str(image)
    store = _store(tmp_path, evidence="all")

    def no_capture(_path):
        pytest.fail("the returned screenshot must be reused, not captured again")

    store.record(
        command="tap",
        result={"ok": True, "action": "tap", "observation": observation},
        invocation_id="call-one",
        duration_ms=1,
        screenshot=no_capture,
    )

    manifest = json.loads((store.root / "manifest.json").read_text())
    retained = Path(manifest["entries"][0]["screenshot"])
    assert retained != image
    assert retained.read_bytes() == image.read_bytes()


def test_artifact_store_and_live_export_agree_about_retained_validity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = {"ok": True, "action": "capture-sheet", "path": "/example/sheet.png"}
    engine = SimpleNamespace(_session_state=lambda: None)
    coaching._record_session_artifact(
        engine, "capture_sheet", result, invocation_id="call-one", duration_ms=1, args={}
    )
    store.record(command="capture_sheet", result=result, invocation_id="call-one", duration_ms=1)

    saved = json.loads((store.root / "calls.jsonl").read_text())["result"]
    assert saved["observation_contract"] == result["observation_contract"]
    assert saved["observation_contract"]["analyze_needed"] is False


@pytest.mark.parametrize("unready", [{"settled_unmet": True}, {"observation_empty": True}])
def test_recovery_analyze_gets_no_contradictory_reuse_coaching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unready: dict[str, Any]
) -> None:
    prior = {"ok": True, "action": "tap", "observation": _observation(), **unready}
    monkeypatch.setattr(
        journal, "read_since", lambda *_args, **_kwargs: [{"cmd": "tap", "result": prior}]
    )
    engine = SimpleNamespace(
        config=SimpleNamespace(cache=SimpleNamespace(dir=tmp_path)),
        platform=SimpleNamespace(name="example"),
        _session_state=lambda: None,
    )

    result = coaching.decorate_result(
        engine, "analyze", deepcopy(_observation()), current_recorded=False
    )

    assert not any(item["id"] == "reuse_observation" for item in result.get("advice", []))
