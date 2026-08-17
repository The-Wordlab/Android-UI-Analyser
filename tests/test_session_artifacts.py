from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser.session_artifacts import SessionArtifactStore


def _observation() -> dict[str, object]:
    return {
        "screen": {
            "width": 1080,
            "height": 2400,
            "package": "com.example.catalog",
            "activity": ".MainActivity",
            "source": "hierarchy",
        },
        "elements": [
            {
                "id": 1,
                "type": "android.widget.TextView",
                "text": "Example catalog",
                "bounds": [0, 0, 600, 100],
                "center": [300, 50],
            }
        ],
        "meta": {
            "duration_ms": 4,
            "tier_used": "hierarchy",
            "path": "hierarchy",
            "device_serial": "example-device",
            "fingerprint": "frame-example-123",
        },
    }


def test_session_bundle_deduplicates_invocations_and_links_evidence(tmp_path: Path) -> None:
    store = SessionArtifactStore.create(
        tmp_path / "run",
        session_id="session-example",
        goal="Verify the example catalog",
        evidence="all",
        junit=True,
        contract_yaml="version: 1\ncheckpoints: []\n",
    )
    captures: list[str] = []

    def screenshot(path: Path) -> str:
        path.write_bytes(b"png")
        captures.append(str(path))
        return str(path)

    result = {"ok": True, "action": "tap", "observation": _observation()}
    contract = store.record(
        command="tap",
        result=result,
        invocation_id="call-1",
        duration_ms=12,
        screenshot=screenshot,
    )
    duplicate = store.record(
        command="tap",
        result=result,
        invocation_id="call-1",
        duration_ms=12,
        screenshot=screenshot,
    )

    assert contract is not None
    assert contract["reusable"] is True
    assert contract["analyze_needed"] is False
    assert duplicate is None
    assert len(captures) == 1
    manifest = json.loads((store.root / "manifest.json").read_text())
    assert len(manifest["entries"]) == 1
    assert len((store.root / "calls.jsonl").read_text().splitlines()) == 1
    assert list((store.root / "evidence").glob("*-observation.json"))


def test_session_bundle_finalizes_checkpoint_junit_and_redacts_inputs(tmp_path: Path) -> None:
    store = SessionArtifactStore.create(
        tmp_path / "run",
        session_id="session-example",
        goal="Enter a secret and verify",
        evidence="failures",
        junit=True,
        contract_yaml=None,
    )
    store.record(
        command="input",
        result={
            "ok": True,
            "kind": "input",
            "text": "private value",
            "observation": _observation(),
        },
        invocation_id="call-secret",
        duration_ms=2,
    )
    result = {"ok": True, "duration_ms": 20}
    finalized = store.finalize(
        result,
        verdict="passed",
        checkpoints=[
            {
                "id": "verified",
                "objective": "Verify result",
                "status": "completed",
                "evidence_id": "session-example:observation:frame-example-123",
            }
        ],
        candidate_yaml="app: com.example.catalog\nsteps: []\n",
    )

    assert Path(finalized["artifacts"]["junit"]).is_file()
    assert Path(finalized["artifacts"]["candidate_flow"]).is_file()
    assert "private value" not in (store.root / "calls.jsonl").read_text()
    assert "<redacted>" in (store.root / "calls.jsonl").read_text()


def test_missing_observation_requires_analyze(tmp_path: Path) -> None:
    store = SessionArtifactStore.create(
        tmp_path / "run",
        session_id="session-example",
        goal="Verify",
        evidence="none",
        junit=False,
        contract_yaml=None,
    )

    contract = store.record(
        command="clipboard-set",
        result={"ok": True, "action": "clipboard-set"},
        invocation_id="call-host",
        duration_ms=1,
    )

    assert contract is None
    payload = json.loads((store.root / "calls.jsonl").read_text())
    assert payload["result"]["observation_contract"]["analyze_needed"] is True


def test_finalize_records_full_session_duration_when_result_has_none(tmp_path: Path) -> None:
    store = SessionArtifactStore.create(
        tmp_path / "run",
        session_id="session-example",
        goal="Verify",
        evidence="none",
        junit=False,
        contract_yaml=None,
    )

    finalized = store.finalize(
        {"ok": True},
        verdict="passed",
        checkpoints=[],
    )

    manifest = json.loads((store.root / "manifest.json").read_text())
    assert manifest["duration_ms"] >= 0
    assert finalized["duration_ms"] == manifest["duration_ms"]
