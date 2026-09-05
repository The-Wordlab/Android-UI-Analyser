from __future__ import annotations

from android_ui_analyser.schema import (
    AppContext,
    DeviceInfo,
    Screen,
    ScreenSource,
    TargetInfo,
    TargetStatus,
)


def test_app_context_reads_neutral_and_legacy_runtime_shapes() -> None:
    neutral = AppContext.coerce({"app_id": "example.app", "surface_id": "settings"})
    legacy = AppContext.coerce({"package": "example.app", "activity": "settings"})

    assert neutral == legacy
    assert neutral.package == "example.app"
    assert neutral.activity == "settings"
    assert neutral.get("package") == "example.app"
    assert neutral.compatibility_dict() == {
        "package": "example.app",
        "activity": "settings",
    }


def test_screen_exposes_app_context_without_changing_legacy_wire_shape() -> None:
    screen = Screen(
        width=100,
        height=200,
        package="example.app",
        activity="home",
        source=ScreenSource.hierarchy,
    )

    assert screen.app_id == "example.app"
    assert screen.surface_id == "home"
    assert screen.app_context == AppContext(app_id="example.app", surface_id="home")
    assert screen.model_dump(mode="json") == {
        "width": 100,
        "height": 200,
        "package": "example.app",
        "activity": "home",
        "source": "hierarchy",
    }


def test_android_device_info_preserves_wire_shape_and_projects_neutral_status() -> None:
    device = DeviceInfo(
        serial="emulator-5554",
        model="Example Phone",
        android_version="14",
        state="device",
    )

    assert device.target_id == "emulator-5554"
    assert device.platform == "android"
    assert device.status is TargetStatus.online
    assert device.os_version == "14"
    assert device.model_dump(mode="json") == {
        "serial": "emulator-5554",
        "model": "Example Phone",
        "android_version": "14",
        "locale": None,
        "state": "device",
    }


def test_external_target_info_has_legacy_selection_properties() -> None:
    target = TargetInfo(
        target_id="shared-id",
        platform="example-os",
        status=TargetStatus.online,
        os_name="ExampleOS",
        os_version="2.0",
    )

    assert target.serial == "shared-id"
    assert target.state == "device"
    assert target.android_version is None
    assert target.model_dump(mode="json")["platform"] == "example-os"
