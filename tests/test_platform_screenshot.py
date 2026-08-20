from __future__ import annotations

import pytest

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
