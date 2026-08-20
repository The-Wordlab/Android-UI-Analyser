"""The ``app.install`` capability is reached through the adapter, never through raw Android tooling.

AGENTS.md requires two proofs for every new device operation: that a platform without the
capability fails explicitly instead of quietly falling back to Android, and that the Android
adapter really does implement it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import AppBundle, InstalledApp
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from conftest import FakeDevice, make_config


class NoInstallPlatform(PlatformAdapter):
    name = "no-install"
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


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("inspect_app_bundle", (Path("example-debug.apk"),)),
        ("installed_app", (None, "com.example.app")),
        ("install_app_bundle", (None, Path("example-debug.apk"))),
        ("uninstall_app", (None, "com.example.app")),
    ],
)
def test_install_is_capability_gated_and_never_falls_back_to_android(
    tmp_path, method: str, args: tuple[object, ...]
) -> None:
    platform = NoInstallPlatform(make_config(cache={"dir": str(tmp_path)}))
    runtime = FakeDevice(serial="example-no-install")
    call_args = tuple(runtime if value is None else value for value in args)

    with pytest.raises(DeviceError) as raised:
        getattr(platform, method)(*call_args)

    assert raised.value.code == "unsupported_capability"
    assert not platform.supports("app.install")
    # The point of the assertion: an unsupported capability must not have reached the runtime.
    assert runtime.calls == []


def test_a_platform_that_says_nothing_about_persistence_claims_nothing(tmp_path) -> None:
    platform = NoInstallPlatform(make_config(cache={"dir": str(tmp_path)}))

    assert platform.install_persistence_warning(FakeDevice()) is None


def test_android_advertises_the_capability_and_delegates_each_operation(tmp_path, monkeypatch):
    platform = AndroidPlatform(make_config(cache={"dir": str(tmp_path)}))
    runtime = FakeDevice(serial="emulator-5554")
    seen: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def record(name: str, result: object = None):
        def call(*args: object, **kwargs: object) -> object:
            seen.append((name, args, kwargs))
            return result

        return call

    from android_ui_analyser.platforms import android_apk

    monkeypatch.setattr(AndroidPlatform, "prepare_host", lambda self: None)
    monkeypatch.setattr(
        android_apk,
        "inspect_bundle",
        record("inspect", android_apk.BundleInfo("com.example.app", "2.1.0", "7")),
    )
    monkeypatch.setattr(
        android_apk,
        "installed_app",
        record("installed", android_apk.InstalledApp("com.example.app", True, "2.0.0", "6")),
    )
    monkeypatch.setattr(android_apk, "install_bundle", record("install"))
    monkeypatch.setattr(android_apk, "uninstall", record("uninstall"))

    assert platform.supports("app.install")
    assert platform.inspect_app_bundle(Path("example-debug.apk")) == AppBundle(
        app_id="com.example.app", version_name="2.1.0", version_code="7"
    )
    assert platform.installed_app(runtime, "com.example.app") == InstalledApp(
        app_id="com.example.app", installed=True, version_name="2.0.0", version_code="6"
    )
    platform.install_app_bundle(runtime, Path("example-debug.apk"), grant_permissions=True)
    platform.uninstall_app(runtime, "com.example.app")

    assert [name for name, _, _ in seen] == ["inspect", "installed", "install", "uninstall"]
    # The serial comes off the connected runtime, not off config: the engine may be driving a
    # leased device that differs from the configured one.
    assert seen[1][1][0] == "emulator-5554"
    assert seen[2][1][0] == "emulator-5554"
    assert seen[2][2]["grant_permissions"] is True
    assert runtime.calls == []


def test_android_warns_only_when_aua_recorded_a_read_only_boot(tmp_path) -> None:
    import json

    platform = AndroidPlatform(make_config(cache={"dir": str(tmp_path)}))
    records = tmp_path / "emulator"
    records.mkdir(parents=True, exist_ok=True)
    (records / "example.p5566.json").write_text(
        json.dumps({"avd": "example", "serial": "emulator-5566", "read_only": True}),
        encoding="utf-8",
    )

    warned = platform.install_persistence_warning(FakeDevice(serial="emulator-5566"))
    assert warned is not None
    assert "read-only" in warned

    # A device AUA did not boot is not knowably disposable; guessing would block the ordinary case.
    assert platform.install_persistence_warning(FakeDevice(serial="emulator-5554")) is None
