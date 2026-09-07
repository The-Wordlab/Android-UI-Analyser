"""A folded action names and retains the final emitted evidence, without another capture."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from android_ui_analyser import cli, coaching, journal
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import ActionResult, AnalyzeResult, OutputFormat
from android_ui_analyser.session_artifacts import SessionArtifactStore, observation_evidence_id


def _observation(fingerprint: str, image: Path) -> AnalyzeResult:
    return AnalyzeResult.model_validate({
        "screen": {"width": 400, "height": 800, "source": "hierarchy"},
        "elements": [{
            "id": "el:continue", "type": "button", "text": "Continue",
            "bounds": [0, 0, 100, 40], "center": [50, 20], "clickable": True,
        }],
        "meta": {
            "fingerprint": fingerprint, "raw_image": str(image), "duration_ms": 3,
            "tier_used": "hierarchy", "path": "hierarchy",
        },
    })


@pytest.mark.parametrize("daemon_dict", [False, True])
@pytest.mark.parametrize("surface", ["json", "tsv", "empty_projection", "no_meta"])
def test_until_finalizes_one_call_with_its_emitted_frame_and_retained_image(
    monkeypatch, tmp_path, capsys, daemon_dict, surface
):
    store = SessionArtifactStore.create(
        tmp_path / "bundle", session_id="example-session", goal="Reach the catalog",
        evidence="all", junit=False, contract_yaml=None,
    )
    state = SimpleNamespace(session_id="example-session", artifact_dir=str(store.root))

    class HostOnlyEngine:
        def _session_state(self):
            return state

        @property
        def platform(self):
            pytest.fail("finalizing evidence must not acquire another screenshot")

    engine = HostOnlyEngine()
    early_image = tmp_path / "early.png"
    early_image.write_bytes(b"the early frame")
    final_image = tmp_path / "final.png"
    final_image.write_bytes(b"the final frame")
    action = ActionResult(
        ok=True, action="tap", observation_present=True,
        observation=_observation("early-frame", early_image),
        arrival={"state": "transitioning", "evidence": ["confirmation_timeout"]},
        stale_risk="the early frame may predate the action", observation_empty=True,
        settled_unmet=True,
    )
    coaching._record_session_artifact(
        engine, "tap", action, invocation_id="example-call", duration_ms=12,
        args={"target": "text:Continue"},
    )
    early_id = action.observation_contract.evidence_id
    final = ActionResult(
        ok=True, action="await_predicate", observation_present=True,
        observation=_observation("final-frame", final_image), await_outcome="satisfied",
    )

    def route(_engine, method, **kwargs):
        assert method == "await_predicate"
        assert kwargs["adopt_action"] is True
        coaching._record_session_artifact(
            engine, method, final, invocation_id="example-call", duration_ms=8, args=kwargs,
        )
        return final.model_dump(mode="json") if daemon_dict else final

    monkeypatch.setattr(cli, "_route", route)
    monkeypatch.setattr(cli, "_UNTIL", ("text:Continue", 1000, 10))
    monkeypatch.setattr(cli, "_ENGINE", engine)
    monkeypatch.setattr(cli, "_caller_turn", lambda result: result)
    monkeypatch.setattr(cli, "_ANNOTATION_WARNINGS", [])
    monkeypatch.setattr(journal, "record_emitted_response", lambda **kwargs: None)
    monkeypatch.setattr(
        cli, "_OBSERVATION_VIEW",
        (Projection.parse(where_text=["Missing"]) if surface == "empty_projection"
         else Projection.parse(no_meta=True) if surface == "no_meta" else None),
    )
    context = cli._CliJournalContext(
        cache_dir=tmp_path, serial="example-device", platform="fictional",
        invocation_id="example-call", detail_id=None, cmd="tap",
        args={"target": "text:Continue"}, client={"until": "text:Continue"},
    )
    cli._emit(
        action.model_dump(mode="json") if daemon_dict else action,
        OutputFormat.tsv if surface == "tsv" else OutputFormat.json,
        _journal_context=context,
    )
    stdout = capsys.readouterr().out
    expected_id = observation_evidence_id(
        "example-session", final.observation.model_dump(mode="json")
    )
    manifest = json.loads((store.root / "manifest.json").read_text())
    calls = [json.loads(line) for line in (store.root / "calls.jsonl").read_text().splitlines()]
    assert len(manifest["entries"]) == len(calls) == 1
    entry, call = manifest["entries"][0], calls[0]
    assert call["command"] == "tap"
    assert call["args"] == {"target": "text:Continue"}
    assert call["duration_ms"] == 12
    assert call["result"]["action"] == "tap"
    contract = call["result"]["observation_contract"]
    assert expected_id != early_id
    assert contract["evidence_id"] == entry["evidence_id"] == expected_id
    assert contract["fingerprint"] == "final-frame"
    assert contract["produced_by"] == "tap"
    assert contract["action_succeeded"] is True
    assert contract["readiness"] == (
        "unconfirmed" if surface == "empty_projection" else "ready"
    )
    assert Path(entry["screenshot"]).read_bytes() == b"the final frame"
    assert json.loads(Path(entry["observation"]).read_text())["meta"]["fingerprint"] == "final-frame"
    assert not call["result"].get("arrival")
    assert not call["result"].get("stale_risk")
    assert not call["result"].get("settled_unmet")
    assert not call["result"].get("observation_empty")
    if surface == "tsv":
        assert f"# observation_contract.evidence_id={expected_id}" in stdout
        assert "# observation_contract.fingerprint=final-frame" in stdout
        assert "# observation_contract.readiness=ready" in stdout
    else:
        emitted = json.loads(stdout)
        assert {key: value for key, value in emitted["observation_contract"].items()
                if value is not None} == contract
        if surface != "no_meta":
            assert emitted["observation"]["meta"]["fingerprint"] == "final-frame"
        else:
            assert "meta" not in emitted["observation"]
    if surface not in {"empty_projection", "no_meta"}:
        assert store.observation_for_evidence(expected_id)["meta"]["fingerprint"] == "final-frame"


def test_until_retains_the_final_reads_own_stale_caveat(monkeypatch, tmp_path):
    image = tmp_path / "returned.png"
    action = ActionResult(
        ok=True, action="tap", observation_present=True,
        observation=_observation("early-frame", image),
        arrival={"state": "transitioning"},
    )
    final = ActionResult(
        ok=True, action="await_predicate", observation_present=True,
        observation=_observation("final-frame", image), await_outcome="satisfied",
        stale_risk="semantic readback may predate the final capture",
        arrival={"state": "unconfirmed", "evidence": ["mixed_readback"]},
    )
    monkeypatch.setattr(cli, "_route", lambda *args, **kwargs: final)
    monkeypatch.setattr(cli, "_UNTIL", ("text:Continue", 1000, 10))
    monkeypatch.setattr(cli, "_ENGINE", object())

    folded = cli._await_until(action)

    assert folded.ok is True
    assert folded.stale_risk == final.stale_risk
    assert folded.arrival == final.arrival
    assert folded.observation_contract.fingerprint == "final-frame"
    assert folded.observation_contract.readiness == "unconfirmed"
    assert folded.observation_contract.evidence_fresh is False


def test_finalization_never_creates_an_unrecorded_call_or_captures(tmp_path):
    store = SessionArtifactStore.create(
        tmp_path / "bundle", session_id="example-session", goal="Reach the catalog",
        evidence="all", junit=False, contract_yaml=None,
    )
    store.record(
        command="tap", result={"ok": True}, invocation_id="unknown", duration_ms=None,
        finalize_existing=True,
        screenshot=lambda path: pytest.fail("finalization must never capture"),
    )
    assert (store.root / "calls.jsonl").read_text() == ""
