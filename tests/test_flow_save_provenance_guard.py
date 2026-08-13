"""Ordinary flow capture never promotes one step's provenance over its neighbors."""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import FlowStore
from android_ui_analyser.memory import AppMemoryStore, RouteStep, SessionState
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from conftest import FakeDevice, make_config

PACKAGE = "com.example.catalog"
OTHER_PACKAGE = "com.example.reader"
SERIAL = "flow-save-provenance"


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            device_serial=SERIAL,
        ),
    )


def _step(resource_id: str, *, origin: str = PACKAGE, context: str = "default") -> RouteStep:
    return RouteStep(
        kind="tap",
        resource_id=resource_id,
        by="id",
        package=PACKAGE,
        origin_package=origin,
        context_id=context,
        capture_segment=4,
    )


def _engine_with_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: list[RouteStep],
) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "memory")}, daemon={"enabled": False})
    device = FakeDevice(serial=SERIAL, package=PACKAGE)
    AppMemoryStore(cfg.memory).save_session(
        SERIAL,
        SessionState(
            package=PACKAGE,
            active_context_id="default",
            capture_segment=4,
            recent=steps,
        ),
    )
    engine = Engine(cfg, device=device)
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: _observation())
    return engine


def test_flow_save_refuses_same_segment_mixed_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine_with_steps(
        tmp_path,
        monkeypatch,
        [_step("foreignAction", origin=OTHER_PACKAGE), _step("ownedAction")],
    )

    with pytest.raises(UsageError, match="mixed origin/context provenance"):
        engine.flow_save("mixed_origin", save=True)

    assert not FlowStore(engine.config.memory).path("mixed_origin").exists()


def test_flow_save_refuses_same_segment_mixed_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine_with_steps(
        tmp_path,
        monkeypatch,
        [_step("oldVariant", context="flags-old"), _step("currentVariant")],
    )

    with pytest.raises(UsageError, match="mixed origin/context provenance"):
        engine.flow_save("mixed_context", save=True)

    assert not FlowStore(engine.config.memory).path("mixed_context").exists()


def test_flow_save_accepts_homogeneous_modern_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine_with_steps(
        tmp_path,
        monkeypatch,
        [_step("firstAction"), _step("secondAction")],
    )

    preview = engine.flow_save("homogeneous")

    assert preview["ok"] is True
    assert preview["scope"] == {
        "requested_last": 12,
        "selected": 2,
        "origin_package": PACKAGE,
        "context_id": "default",
        "capture_segment": 4,
        "boundary_omitted": 0,
    }
    assert "id: firstAction" in preview["preview"]
    assert "id: secondAction" in preview["preview"]
