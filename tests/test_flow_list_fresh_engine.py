"""Context-aware flow listing from a fresh CLI/MCP-style engine."""

from __future__ import annotations

from pathlib import Path

import pytest

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import FlowStore, parse_flow_yaml
from android_ui_analyser.memory import AppMemoryStore, SessionState
from android_ui_analyser.schema import DeviceInfo
from conftest import FakeDevice, make_config

PACKAGE = "com.example.catalog"
CONTEXT = "flags-catalog-v2"


def _config_with_contextual_flow(tmp_path: Path):
    cfg = make_config(memory={"dir": str(tmp_path / "memory")})
    FlowStore(cfg.memory).save(
        parse_flow_yaml(
            f"""
name: variant_catalog
app: {PACKAGE}
context_id: {CONTEXT}
steps:
  - key: back
"""
        )
    )
    return cfg


def test_fresh_engine_flow_list_discovers_attached_foreground_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config_with_contextual_flow(tmp_path)
    serial = "fake-flow-list-attached"
    device = FakeDevice(serial=serial, package=PACKAGE)
    AppMemoryStore(cfg.memory).save_session(
        serial,
        SessionState(package=PACKAGE, active_context_id=CONTEXT),
    )
    monkeypatch.setattr(
        engine_mod.Engine,
        "_list_targets",
        lambda _engine: [DeviceInfo(serial=serial, state="device")],
    )
    connected: list[str | None] = []
    monkeypatch.setattr(
        engine_mod.Engine,
        "_connect_target",
        lambda _engine, requested=None: connected.append(requested) or device,
    )
    engine = Engine(cfg)
    assert engine._device is None

    listed = engine.flow_list()

    assert connected == [serial]
    assert listed["active_package"] == PACKAGE
    assert listed["active_context_id"] == CONTEXT
    assert listed["flows"][0]["context_compatible"] is True
    assert device.calls == []


@pytest.mark.parametrize(
    "devices",
    [
        [],
        [DeviceInfo(serial="fake-flow-list-offline", state="offline")],
    ],
    ids=["absent", "offline"],
)
def test_fresh_engine_flow_list_stays_offline_and_compatibility_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    devices: list[DeviceInfo],
) -> None:
    cfg = _config_with_contextual_flow(tmp_path)
    monkeypatch.setattr(engine_mod.Engine, "_list_targets", lambda _engine: devices)
    monkeypatch.setattr(
        engine_mod.Engine,
        "_connect_target",
        lambda _engine, _serial=None: pytest.fail("offline flow listing must not connect"),
    )
    engine = Engine(cfg)

    listed = engine.flow_list()

    assert engine._device is None
    assert listed["active_package"] is None
    assert listed["active_context_id"] is None
    assert listed["flows"][0]["context_compatible"] is None
