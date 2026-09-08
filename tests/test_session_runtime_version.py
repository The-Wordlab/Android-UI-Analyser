"""Usage audits must distinguish old, current and mixed-runtime sessions."""

import json

import pytest

from android_ui_analyser import journal, session
from android_ui_analyser.session_artifacts import SessionArtifactStore


def _start(tmp_path):
    return session.create_session_state(
        tmp_path,
        goal="Verify catalog",
        serial="version-target",
        owner="version-owner",
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )


def test_review_preserves_the_start_version_and_detects_runtime_turnover(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "__version__", "0.1.0")
    state = _start(tmp_path)
    for version in ("0.1.0", "0.2.0"):
        monkeypatch.setattr(journal, "__version__", version)
        journal.record(
            cache_dir=tmp_path, serial=state.serial, owner=state.owner,
            source="cli", cmd="has", result={"present": True},
        )
    monkeypatch.setattr(session, "__version__", "0.3.0")
    loaded = session.load_session_state(tmp_path, session_id=state.session_id)
    assert loaded is not None
    assert loaded.aua_version == "0.1.0"
    events = journal.read_since(tmp_path, state.serial)
    review = session.review_session_events(loaded, events)
    assert review["aua_version"] == "0.1.0"
    assert review["runtime_versions"] == ["0.1.0", "0.2.0"]
    assert review["unversioned_events"] == 0
    with pytest.raises(ValueError, match="identity field"):
        session.update_session_state(tmp_path, loaded, aua_version="0.3.0")


def test_legacy_sessions_and_events_stay_unversioned(tmp_path, monkeypatch):
    state = _start(tmp_path)
    path = next((tmp_path / "sessions").rglob(f"{state.session_id}.json"))
    payload = json.loads(path.read_text())
    payload.pop("aua_version")
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(session, "__version__", "0.9.0")
    loaded = session.load_session_state(tmp_path, session_id=state.session_id)
    assert loaded is not None
    review = session.review_session_events(
        loaded, [{"session_id": state.session_id, "cmd": "has", "ok": True}],
    )
    assert review["aua_version"] is None
    assert review["runtime_versions"] == []
    assert review["unversioned_events"] == 1


def test_portable_manifest_keeps_the_recorded_start_version(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "__version__", "0.1.0")
    state = _start(tmp_path / "cache")
    monkeypatch.setattr(session, "__version__", "0.2.0")
    store = SessionArtifactStore.create(
        tmp_path / "artifacts", session_id=state.session_id, goal=state.goal,
        evidence="none", junit=False, contract_yaml=None, aua_version=state.aua_version,
    )
    manifest = json.loads((store.root / "manifest.json").read_text())
    assert manifest["aua_version"] == "0.1.0"
