"""Persistent servers share capture startup without selecting a device in the background."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from android_ui_analyser import journal, session
from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.schema import Element
from conftest import FakeDevice, make_config, make_png


class CapturePlatform(PlatformAdapter):
    name = "capture-fixture"
    capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot"})

    def connect(self, target_id=None):
        raise AssertionError("background capture must not connect")

    def list_targets(self):
        raise AssertionError("background capture must not discover targets")

    def normalize_tree(self, raw_tree, screen_size, **kwargs):
        return NormalizedTree([
            Element(id=1, type="Button", text="Continue", resource_id="example:id/continue",
                    bounds=(0, 0, 40, 40), center=(20, 20), clickable=True, source="hierarchy"),
        ])

    def capture_screenshot(self, runtime):
        return runtime.screenshot()


def _engine(tmp_path, *, enabled=True, connected=True, screenshots=True):
    cfg = make_config(
        cache={"dir": str(tmp_path)}, capture={"enabled": enabled, "idle_fps": 20,
                                               "burst_fps": 20},
    )
    platform = CapturePlatform(cfg)
    if not screenshots:
        platform.capabilities = frozenset({"ui.tree", "ui.input"})
    device = FakeDevice(width=80, height=120, serial="capture-target") if connected else None
    return Engine(cfg, device=device, platform=platform)


def _initialized(engine):
    thread = engine._capture_initializer
    assert thread is not None, "persistent server never enabled capture"
    thread.join(timeout=2)
    assert not thread.is_alive(), "capture did not initialize"
    assert engine._capture is not None and engine._capture.running
    return engine._capture


def _payload(response):
    return json.loads(next(block.text for block in response.content if block.type == "text"))


def test_mcp_action_exposes_one_exportable_capture_window_without_android_tools(
    tmp_path, monkeypatch,
):
    def no_android(*args, **kwargs):
        raise AssertionError("neutral capture reached Android tooling")

    for method in ("connect", "capture_screenshot", "dump_tree"):
        monkeypatch.setattr(AndroidPlatform, method, no_android)
    engine = _engine(tmp_path)
    state = session.create_session_state(
        tmp_path, goal="Verify capture evidence", serial="capture-target", owner=None,
        recommended_kind="manual_observation", recommended_cli="reuse observation",
        network_backup_preexisting=False, network_profile_preexisting=False,
    )
    engine._session_id = state.session_id
    server = build_server(engine)
    buffer = _initialized(engine)
    initializer = engine._capture_initializer

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            await client.call_tool("analyze_screen", {"source": "hierarchy"})
            status = _payload(await client.call_tool("capture_status", {}))
            assert status["session_id"] == buffer.session_id != state.session_id
            action = _payload(await client.call_tool("tap_and_analyze", {"rid": "example:id/continue"}))
            assert action["ok"] is True
            reference = action["capture_evidence"]["ref"]
            # Quiesce background sampling before deterministic frame injection. Export itself
            # must use these recorded pixels and cannot acquire another frame from the adapter.
            buffer.pause("test", settle_s=2)
            device = engine._device
            assert device is not None
            device._png = make_png(80, 120, color=(80, 100, 120))
            buffer._tick()
            monkeypatch.setattr(engine.platform, "capture_screenshot", no_android)
            sheet = _payload(await client.call_tool("capture_sheet", {
                "path": str(tmp_path / "sheet.png"), "evidence_ref": reference,
            }))
            exported = _payload(await client.call_tool("capture_export", {
                "path": str(tmp_path / "clip.gif"), "evidence_ref": reference,
            }))
            assert sheet["capture_evidence"] == exported["capture_evidence"]
            assert sheet["session_id"] == exported["session_id"] == buffer.session_id
            assert sheet["source_frames"] == exported["frames"] >= 1
            assert Path(sheet["path"]).is_file()
            assert Path(exported["path"]).is_file()
            engine.capture_service_start()
            assert engine._capture_initializer is initializer
            assert engine._capture is buffer and buffer.paused

            events = journal.read_since(tmp_path, state.serial, platform=engine.platform.name)
            captures = [event for event in events if event["cmd"].startswith("capture_")]
            assert len(captures) == 3
            assert all(event["session_id"] == state.session_id for event in captures)
            review = session.review_session_events(state, events)
            assert review["calls"] == review["engine_events"] == 5
            assert all(review["commands"][name] == 1 for name in (
                "capture_status", "capture_sheet", "capture_export",
            ))

    try:
        anyio.run(run)
    finally:
        engine.close()
    assert not buffer.running
    assert engine._capture is None


@pytest.mark.parametrize("enabled,screenshots", [(False, True), (True, False)])
def test_mcp_capture_opt_out_or_missing_capability_never_starts_sampling(
    tmp_path, enabled, screenshots,
):
    engine = _engine(tmp_path, enabled=enabled, screenshots=screenshots)
    try:
        build_server(engine)
        assert engine._capture_initializer is None
        assert engine._capture is None
        assert engine._device.screenshot_calls == 0
    finally:
        engine.close()


def test_mcp_capture_waits_for_foreground_target_and_close_cancels_the_wait(tmp_path):
    engine = _engine(tmp_path, connected=False)
    build_server(engine)
    initializer = engine._capture_initializer
    assert initializer is not None
    assert engine._capture is None and engine._device is None
    engine.close()
    assert not initializer.is_alive()
    engine.config.device.serial = "target-selected-too-late"
    engine.capture_service_start()
    assert engine._capture is None


def test_mcp_capture_uses_the_target_selected_after_server_start(tmp_path):
    engine = _engine(tmp_path, connected=False)
    build_server(engine)
    engine.config.device.serial = "foreground-selected-target"
    try:
        buffer = _initialized(engine)
        assert buffer.serial == "foreground-selected-target"
        assert engine._device is None
    finally:
        engine.close()


def test_mcp_new_bootstrap_after_target_close_gets_a_new_scoped_buffer(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    server = build_server(engine)
    prior = _initialized(engine)
    engine.close()

    def bootstrap(goal, **kwargs):
        engine._device = FakeDevice(serial="replacement-target", width=80, height=120)
        engine.config.device.serial = "replacement-target"
        return {"ok": True, "goal": goal, "serial": "replacement-target"}

    monkeypatch.setattr(engine, "session_start", bootstrap)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            result = _payload(await client.call_tool("session_start", {"goal": "verify replacement"}))
            assert result["ok"] is True
            current = _initialized(engine)
            assert current is not prior
            assert current.serial == "replacement-target"
            assert not prior.running

    try:
        anyio.run(run)
    finally:
        engine.close()


def test_close_waits_for_inflight_capture_initialization_before_stopping_device(
    tmp_path, monkeypatch,
):
    engine = _engine(tmp_path)
    started, release, closing = threading.Event(), threading.Event(), threading.Event()
    original_start = engine.capture_start
    device = engine._device
    assert device is not None
    events = []

    def delayed_start(*, connect_if_needed):
        assert connect_if_needed is False
        started.set()
        assert release.wait(timeout=2)
        result = original_start(connect_if_needed=False)
        events.append("capture-started")
        return result

    monkeypatch.setattr(engine, "capture_start", delayed_start)
    monkeypatch.setattr(device, "close", lambda: events.append("device-closed"))
    build_server(engine)
    assert started.wait(timeout=2)

    def close():
        closing.set()
        engine.close()

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert closing.wait(timeout=2)
        assert "device-closed" not in events
    finally:
        release.set()
        closer.join(timeout=2)
    assert not closer.is_alive()
    assert events == ["capture-started", "device-closed"]
    assert engine._capture is None
