"""Dashboard reads stay behind the selected platform's neutral capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import dashboard
from android_ui_analyser.config import Config
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.diagnostics import DiagnosticWindow
from android_ui_analyser.platforms.geometry import DisplayGeometry
from android_ui_analyser.platforms.supervision import TargetSupervisionStatus
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.schema import AppContext, TargetInfo


class _NeutralRuntime:
    serial = "simulator-1"

    def current_app(self) -> AppContext:
        return AppContext(app_id="example.notes", surface_id="main")

    def display_geometry(self) -> DisplayGeometry:
        return DisplayGeometry.identity(390, 844)

    def window_size(self) -> tuple[int, int]:
        return (390, 844)

    def dump_hierarchy(self, compressed: bool = False) -> str:
        del compressed
        return "<neutral-tree/>"


class _SupervisionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def target_supervision_status(
        self, target_id: str, *, cache_dir: str | Path
    ) -> TargetSupervisionStatus:
        self.calls.append((target_id, Path(cache_dir)))
        return TargetSupervisionStatus(
            target_id=target_id,
            managed=True,
            owner="worker-a",
            instance_id="owned-instance",
            started_at=900.0,
            last_activity=980.0,
            idle_timeout_s=120.0,
            monitor_running=True,
        )


class _NeutralDashboardPlatform(PlatformAdapter):
    name = "apple-test"
    capabilities = frozenset(
        {"ui.tree", "ui.screenshot", "device.logs", "target_supervision"}
    )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.runtime = _NeutralRuntime()
        self.supervision = _SupervisionService()
        self.calls: list[tuple[str, Any]] = []

    def connect(self, target_id: str | None = None) -> _NeutralRuntime:  # type: ignore[override]
        self.calls.append(("connect", target_id))
        return self.runtime

    def list_targets(self) -> list[TargetInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        del raw_tree, screen_size, geometry, ignored_app_ids
        return NormalizedTree([])

    def runtime_capability(self, capability: str, runtime: Any) -> Any:
        self.calls.append(("runtime_gate", capability))
        return super().runtime_capability(capability, runtime)

    def adapter_capability(self, capability: str) -> PlatformAdapter:
        self.calls.append(("adapter_gate", capability))
        return super().adapter_capability(capability)

    def capture_screenshot(self, runtime: Any) -> ScreenImage:
        assert runtime is self.runtime
        self.calls.append(("screenshot", runtime))
        return ScreenImage(b"\x89PNG\r\n\x1a\nneutral")

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        self.calls.append(("logs", (target_id, limit, app_id)))
        return ["neutral diagnostic"]

    def diagnostic_logs(
        self,
        runtime: Any,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        del runtime, lines, since_ms, app_id
        return ""

    def diagnostic_window(
        self,
        runtime: Any,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        del runtime, lines, since, app_id
        raise AssertionError("dashboard should use the compact recent-logs contract")

    def mark_diagnostics(
        self,
        runtime: Any,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, Any]:
        del runtime, name, clear, refresh_clock
        return {}

    def clear_diagnostics(self, runtime: Any) -> None:
        del runtime

    def load_capability(self, capability: str) -> object | None:
        if capability == "target_supervision":
            return self.supervision
        return None


def _state(tmp_path: Path) -> tuple[dashboard._DashboardState, _NeutralDashboardPlatform]:
    config = Config()
    config.cache.dir = str(tmp_path)
    config.memory.dir = str(tmp_path)
    state = dashboard._DashboardState(
        serials=["simulator-1"],
        focus="simulator-1",
        mode="detail",
        cache_dir=tmp_path,
        ensures={},
        poll_ms=500,
        config=config,
    )
    platform = _NeutralDashboardPlatform(config)
    state.platform = platform
    state.platform_name = platform.name
    return state, platform


def test_non_android_dashboard_reads_use_declared_capabilities(tmp_path: Path) -> None:
    state, platform = _state(tmp_path)
    state.note_capture_live("simulator-1", False)

    assert state.foreground_package() == "example.notes"
    assert state.frame_bytes() == (b"\x89PNG\r\n\x1a\nneutral", "image/png")
    assert state.log_lines("simulator-1", 12, app_id="example.notes") == [
        "neutral diagnostic"
    ]

    assert ("screenshot", platform.runtime) in platform.calls
    assert ("logs", ("simulator-1", 12, "example.notes")) in platform.calls
    assert ("runtime_gate", "ui.tree") in platform.calls
    assert ("adapter_gate", "ui.screenshot") in platform.calls
    assert ("adapter_gate", "device.logs") in platform.calls


def test_non_android_dashboard_gets_owner_and_retirement_from_supervision_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "time", lambda: 1_000.0)
    state, platform = _state(tmp_path)

    status = state.device_runtime("simulator-1")

    assert platform.supervision.calls == [("simulator-1", tmp_path)]
    assert status["owner"] == "worker-a"
    assert status["watchdog"] == {
        "managed": True,
        "enabled": True,
        "running": True,
        "idle_s": 20.0,
        "timeout_s": 120.0,
        "remaining_s": 100.0,
        "instance": "owned-instance",
        "explicit": False,
    }
