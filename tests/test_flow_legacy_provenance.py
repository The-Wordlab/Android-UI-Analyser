"""Legacy action journals never fabricate capture-time context evidence."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.memory import AppMemoryStore, RouteStep
from test_memory import HOME, P, _engine
from test_navigation import ScriptedDevice


def test_legacy_flow_capture_omits_current_context_and_arrival_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    device = ScriptedDevice([HOME], package=P, serial="emu-legacy-provenance")
    engine = _engine(tmp_path, device)
    current = engine.analyze(source="hierarchy")
    store = AppMemoryStore(engine.config.memory)
    session = store.load_session(device.serial)
    session.package = P
    session.active_context_id = "feature-new"
    session.recent = [
        RouteStep(kind="tap", resource_id="nav_apps", by="id", package=P)
    ]
    store.save_session(device.serial, session)
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: current)

    preview = engine.flow_save("legacy_capture")

    assert preview["scope"]["context_id"] is None
    assert "context_id:" not in preview["preview"]
    assert preview["arrival_proof"]["status"] == "unverified"
    assert any("no per-action origin/context provenance" in item for item in preview["warnings"])
