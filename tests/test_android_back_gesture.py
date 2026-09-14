"""Android owns edge-back geometry; shared callers ask only for the semantic gesture."""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError, UnsupportedPlatformCapabilityError
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from conftest import FakeDevice, make_config


class _NoBackGesturePlatform(PlatformAdapter):
    name = "no-back-gesture"
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


def test_back_gesture_is_capability_gated_for_non_android_platforms(tmp_path) -> None:  # type: ignore[no-untyped-def]
    platform = _NoBackGesturePlatform(make_config(cache={"dir": str(tmp_path)}))
    runtime = FakeDevice(serial="example-no-back-gesture")

    with pytest.raises(UnsupportedPlatformCapabilityError) as caught:
        platform.runtime_capability("ui.back_gesture", runtime)

    assert caught.value.code == "platform_capability_unsupported"
    assert not platform.supports("ui.back_gesture")
    assert runtime.calls == []


def test_android_back_gesture_derives_coordinates_inside_the_adapter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    device = FakeDevice(width=1080, height=2400)
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)

    result = engine.back_gesture(observe=False)

    assert result.ok is True and result.action == "back-gesture"
    assert device.calls == [("swipe", (10, 1200, 432, 1200, 300))]
    assert AndroidPlatform(make_config()).supports("ui.back_gesture")
    AndroidPlatform(make_config()).validate_runtime(device)


def test_android_back_gesture_refuses_invalid_screen_geometry() -> None:
    device = FakeDevice(width=2, height=2)

    with pytest.raises(DeviceError) as caught:
        device.back_gesture()

    assert caught.value.code == "invalid_screen_geometry"
    assert not device.calls
