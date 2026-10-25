from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from conftest import FakeDevice, make_config


class NoScreenshotPlatform(PlatformAdapter):
    name = "no-screenshot"
    capabilities = frozenset({"ui.tree"})

    def connect(self, target_id: str | None = None):  # type: ignore[no-untyped-def]
        raise AssertionError("not needed")

    def list_targets(self):  # type: ignore[no-untyped-def]
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids=(),  # type: ignore[no-untyped-def]
    ) -> NormalizedTree:
        return NormalizedTree(elements=[])


def test_optional_screenshot_capability_fails_without_native_fallback(tmp_path) -> None:
    platform = NoScreenshotPlatform(make_config(cache={"dir": str(tmp_path)}))
    runtime = FakeDevice(serial="example-no-screenshot")

    with pytest.raises(DeviceError) as raised:
        platform.capture_screenshot(runtime)

    assert raised.value.code == "unsupported_capability"
    assert runtime.screenshot_calls == 0


def test_android_screenshot_delegates_to_runtime(tmp_path) -> None:
    platform = AndroidPlatform(make_config(cache={"dir": str(tmp_path)}))
    runtime = FakeDevice(serial="example-android-screenshot")

    image = platform.capture_screenshot(runtime)

    assert image.width == 1080
    assert image.height == 2400
    assert runtime.screenshot_calls == 1


class ScreenshotOnlyPlatform(NoScreenshotPlatform):
    name = "screenshot-fixture"
    capabilities = frozenset({"ui.screenshot"})

    def capture_screenshot(self, runtime):
        return runtime.screenshot()


def test_screenshot_only_session_has_visual_evidence_without_claiming_ui_inspection(
    tmp_path, monkeypatch
):
    cfg = make_config(
        cache={"dir": str(tmp_path)},
        memory={"enabled": False},
        lease={"enabled": False},
        teardown={"enabled": False},
        capture={"enabled": False},
        perf={"auto_daemon": False},
    )
    runtime = FakeDevice(serial="capture-fixture")
    platform = ScreenshotOnlyPlatform(cfg)
    engine = Engine(cfg, device=runtime, platform=platform)
    monkeypatch.setattr(
        engine, "analyze", lambda **kw: pytest.fail("screenshot-only session requested a UI tree")
    )
    try:
        started = engine.session_start("Capture current screen")
        observation = started["observation"]
        assert observation["elements"] == []
        assert observation["screen"]["package"] is None
        assert observation["meta"]["via"] == "screenshot-only"
        assert observation["meta"]["observation_contract"]["elements_available"] is False
        assert observation["meta"]["observation_contract"]["readiness"] == "not_checked"
        assert started["recommended_call"]["kind"] == "screenshot"
        assert runtime.screenshot_calls == 1
        incomplete = engine.session_finish(started["session_id"])
        assert incomplete["code"] == "session_incomplete"
        assert incomplete["finished"] is False
        engine.session_mark_phase("phase_1", "Current screen captured and visually inspected")
        finished = engine.session_finish(started["session_id"])
        assert finished["finished"] is True
        assert finished["goal_progress"]["done"] is True
    finally:
        engine.close()


def test_mcp_screenshot_uses_the_same_neutral_adapter(tmp_path):
    from android_ui_analyser.mcp_server import _dispatch

    cfg = make_config(
        cache={"dir": str(tmp_path)},
        lease={"enabled": False},
        teardown={"enabled": False},
        capture={"enabled": False},
    )
    runtime = FakeDevice(serial="mcp-capture-fixture")
    engine = Engine(cfg, device=runtime, platform=ScreenshotOnlyPlatform(cfg))
    try:
        output = tmp_path / "mcp.png"
        result = _dispatch(engine, "screenshot", {"path": str(output)})
        assert result["ok"] is True and output.is_file()
        assert runtime.screenshot_calls == 1
    finally:
        engine.close()
