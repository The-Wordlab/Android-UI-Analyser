"""Journey transports preserve session ownership, cleanup and portable evidence."""

from __future__ import annotations

import json

from android_ui_analyser import coaching, daemon
from android_ui_analyser.cli import _route
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _MCP_STARTED_TARGETS, _dispatch_tool
from android_ui_analyser.session_artifacts import SessionArtifactStore
from conftest import FakeDevice, make_config


def test_mcp_journey_tracks_exact_started_target_and_forgets_only_stopped_target(tmp_path, monkeypatch):
    engine = Engine(make_config(cache={"dir": str(tmp_path)}, lease={"enabled": False}), device=FakeDevice())
    captures = []
    monkeypatch.setattr(engine, "capture_service_start", lambda **kwargs: captures.append(kwargs))
    started = {"ok": True, "virtual_target_started": True, "target_id": "fictional-target",
               "instance_token": "fictional-boot-token"}
    monkeypatch.setattr(engine, "journey_start", lambda **kwargs: started)
    monkeypatch.setattr(engine, "journey_finish", lambda **kwargs: {
        "ok": True, "terminated": True, "cleanup": [{
            "action": "owned_virtual_target_stop", "ok": True,
            "virtual_target": {"target_id": "fictional-target"},
        }],
    })
    try:
        assert _dispatch_tool(engine, "journey_start", {"yaml": "authored journey"}) == started
        assert captures == [{"restart": True}]
        key = (id(engine.platform), "fictional-target", "fictional-boot-token")
        assert key in _MCP_STARTED_TARGETS
        assert _dispatch_tool(engine, "journey_finish", {"retain_started_target": False})["ok"]
        assert key not in _MCP_STARTED_TARGETS
    finally:
        for key in list(_MCP_STARTED_TARGETS):
            if key[0] == id(engine.platform):
                del _MCP_STARTED_TARGETS[key]
        engine.close()


def test_cli_journey_preserves_new_lease_origin_for_runtime_session_bootstrap(tmp_path, monkeypatch):
    config = make_config(cache={"dir": str(tmp_path)}, lease={"enabled": False}, daemon={"enabled": True})
    engine = Engine(config, device=FakeDevice())
    original_supports = engine.platform.supports
    monkeypatch.setattr(engine.platform, "supports", lambda name: name == "session.state" or original_supports(name))
    monkeypatch.setattr(engine, "_lease_device", lambda: "fictional-target")
    engine._lease_was_preexisting = False
    calls = []
    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def call(self, command, **kwargs):
            calls.append(kwargs)
            return {"ok": True, "result": {}, "response_decorated": True}

    monkeypatch.setattr(daemon, "is_running", lambda _: True)
    monkeypatch.setattr(daemon, "running_version", lambda _: daemon._aua_version())
    monkeypatch.setattr(daemon, "running_runtime_fingerprint", daemon.runtime_config_fingerprint)
    monkeypatch.setattr(daemon, "running_policy_fingerprint", daemon.policy_config_fingerprint)
    monkeypatch.setattr(daemon, "DaemonClient", Client)
    try:
        _route(engine, "journey_start", yaml="authored journey")
        assert calls[0]["_bootstrap_new_lease"] is True
    finally:
        engine.close()


def test_journey_finish_finalizes_artifacts_with_agent_assessment_verdict(tmp_path):
    from types import SimpleNamespace

    store = SessionArtifactStore.create(
        tmp_path / "artifacts", session_id="fixture-session", goal="Review fictional catalog",
        evidence="all", junit=False, contract_yaml=None,
    )
    state = SimpleNamespace(artifact_dir=str(store.root), session_id="fixture-session")
    engine = SimpleNamespace(_session_state=lambda: state)
    result = {
        "ok": True, "finished": True, "terminated": True, "verified": False,
        "verdict": "agent_assessed_pass", "artifacts_dir": str(store.root),
        "goal_progress": {"phases": [{"id": "visual", "status": "completed",
                                      "proof": {"verified": False, "source": "manual_evidence"}}]},
    }
    coaching._record_session_artifact(
        engine, "journey_finish", result, invocation_id="finish-call", duration_ms=1, args={},
    )
    manifest = json.loads((store.root / "manifest.json").read_text())
    assert manifest["verdict"] == "agent_assessed_pass"
    assert manifest["finished_at"]
    assert json.loads((store.root / "result.json").read_text())["verified"] is False
    assert "AGENT_ASSESSED_PASS" in (store.root / "report.md").read_text()


def test_journey_record_is_archived_without_taking_another_screenshot(tmp_path):
    from types import SimpleNamespace

    store = SessionArtifactStore.create(
        tmp_path / "artifacts", session_id="fixture-session", goal="Review fictional catalog",
        evidence="all", junit=False, contract_yaml=None,
    )
    engine = SimpleNamespace(_session_state=lambda: SimpleNamespace(artifact_dir=str(store.root)))
    result = {"ok": False, "verified": False, "assessment": "agent_assessment",
              "artifacts_dir": str(store.root), "fingerprint": "previously-seen-frame"}
    coaching._record_session_artifact(
        engine, "journey_record", result, invocation_id="record-call", duration_ms=1, args={},
    )
    call = json.loads((store.root / "calls.jsonl").read_text())
    assert call["result"]["fingerprint"] == "previously-seen-frame"
    assert not (store.root / "evidence").exists()


def test_record_and_finished_status_use_exact_saved_target_without_leasing(tmp_path, monkeypatch):
    from android_ui_analyser.journeys import journey_phases, load_journey, save_companion
    from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
    from android_ui_analyser.session import (
        create_session_state,
        finish_session_state,
        update_session_state,
    )

    config = make_config(cache={"dir": str(tmp_path)}, lease={"enabled": False}, memory={"enabled": False})
    engine = Engine(config)
    journey = load_journey(yaml_text="""name: Fictional review
steps:
- id: visual
  instruction: Inspect the title and button.
  review: Catalog title and Continue button have clear spacing.
""")
    state = create_session_state(
        tmp_path, goal=journey.name, serial="exact-fixture-target", owner=engine._ledger_identity()["owner"],
        recommended_kind="manual", recommended_cli="aua analyze",
        network_backup_preexisting=False, network_profile_preexisting=False, platform=engine.platform.name,
    )
    state = update_session_state(tmp_path, state, phases=journey_phases(journey))
    save_companion(tmp_path, state, {"session_id": state.session_id,
                                   "journey": journey.model_dump(mode="json"), "attempts": []})
    observed = AnalyzeResult(
        screen=Screen(width=400, height=800, source="hierarchy", package="com.example.catalog"),
        elements=[], meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy",
                               fingerprint="reviewed-frame", device_serial=state.serial),
    )
    engine._write_cache(observed)

    def forbidden(*args, **kwargs):
        raise AssertionError("saved journey metadata and review must not connect or lease")

    monkeypatch.setattr(engine.platform, "connect", forbidden)
    monkeypatch.setattr(engine, "_lease_device", forbidden)
    result = _route(
        engine, "journey_record", session_id=state.session_id, fingerprint="reviewed-frame",
        evidence="Catalog title and Continue button have clear spacing and readable labels.",
    )
    assert result["ok"] and result["verified"] is False
    finish_session_state(tmp_path, engine._session_state(state.session_id))
    status = _route(engine, "journey_status", session_id=state.session_id)
    assert status["journey"]["finished"] is True
    assert status["journey"]["next_call"] is None
    assert engine._device is None
