"""Platform factory/strategy seam; Android remains the only built-in implementation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from android_ui_analyser.cli import GlobalOpts, hoist_global_options
from android_ui_analyser.config import Config, load_config
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ConfigError
from android_ui_analyser.platforms import (
    NormalizedTree,
    PlatformAdapter,
    PlatformFactory,
    register_platform,
    registered_platforms,
)
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.schema import DeviceInfo, Element
from conftest import FakeDevice


@register_platform("test-native")
class _RegisteredPlatform(PlatformAdapter):
    capabilities = frozenset({"ui.tree"})

    def connect(self, target_id: str | None = None) -> Device:
        raise AssertionError("factory selection must not connect eagerly")

    def list_targets(self) -> list[DeviceInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree([])


class _InjectedPlatform(_RegisteredPlatform):
    name = "injected"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[tuple[str, object]] = []

    def dump_tree(self, runtime: Device, *, compact: bool = False) -> str:
        self.calls.append(("dump", compact))
        return "native-tree"

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        self.calls.append(("normalize", (raw_tree, screen_size, tuple(ignored_app_ids))))
        element = Element(
            id=0,
            type="Button",
            text="From plugin",
            bounds=(10, 20, 110, 70),
            center=(60, 45),
            clickable=True,
        )
        return NormalizedTree([element], app_id="example.native")


def test_android_is_the_only_builtin_platform() -> None:
    builtins = registered_platforms()
    assert builtins["android"] is AndroidPlatform
    assert AndroidPlatform(Config()).supports("ui.tree")


def test_factory_selects_and_memoizes_registered_strategy() -> None:
    cfg = Config.model_validate({"device": {"platform": "TEST-NATIVE"}})
    factory = PlatformFactory(cfg)

    first = factory.create()

    assert isinstance(first, _RegisteredPlatform)
    assert factory.create() is first


def test_factory_reports_installed_platforms_for_unknown_name() -> None:
    factory = PlatformFactory(Config.model_validate({"device": {"platform": "missing"}}))

    with pytest.raises(ConfigError, match="unknown platform 'missing'") as exc:
        factory.create()

    assert "android" in (exc.value.hint or "")


def test_engine_uses_injected_strategy_for_tree_capture_and_normalization() -> None:
    cfg = Config.model_validate(
        {
            "memory": {"enabled": False},
            "perf": {"prefetch": False},
            "lease": {"enabled": False},
        }
    )
    runtime = FakeDevice(package="example.native")
    platform = _InjectedPlatform(cfg)
    engine = Engine(cfg, device=runtime, platform=platform)

    result = engine.analyze(source="hierarchy", record=False)

    assert [element.text for element in result.elements] == ["From plugin"]
    assert platform.calls == [
        ("dump", True),
        (
            "normalize",
            ("native-tree", (1080, 2400), tuple(cfg.memory.ignore_packages)),
        ),
    ]


def test_flow_artifacts_do_not_fall_back_to_android_logs_on_another_platform(tmp_path) -> None:
    cfg = Config.model_validate(
        {
            "memory": {"enabled": False},
            "cache": {"dir": str(tmp_path / "cache")},
            "perf": {"prefetch": False},
            "lease": {"enabled": False},
        }
    )
    runtime = FakeDevice(package="example.native")
    platform = _InjectedPlatform(cfg)

    result = Engine(cfg, device=runtime, platform=platform).flow_run(
        yaml="steps:\n  - assert: {text: Missing, exists: true}\n",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert result["ok"] is False
    assert not any(call[0] == "logcat" for call in runtime.calls)
    assert not (tmp_path / "artifacts" / "failure-diagnostics.txt").exists()


def test_android_platform_provides_bounded_failure_diagnostics() -> None:
    runtime = FakeDevice()
    runtime.log_now(tag="First", msg="older")
    runtime.log_now(tag="Second", msg="newer")

    logs = AndroidPlatform(Config()).diagnostic_logs(runtime, lines=1)

    assert "newer" in logs
    assert "older" not in logs
    assert ("logcat", (None, True)) in runtime.calls


def test_android_strategy_normalizes_xml_and_ignores_system_overlay_package() -> None:
    raw = """<hierarchy>
      <node class="android.widget.TextView" text="App" package="com.example.app"
            bounds="[0,0][100,50]" />
      <node class="android.widget.TextView" text="Clock" package="com.android.systemui"
            bounds="[0,50][100,100]" />
      <node class="android.widget.TextView" text="Shade" package="com.android.systemui"
            bounds="[0,100][100,150]" />
    </hierarchy>"""

    normalized = AndroidPlatform(Config()).normalize_tree(
        raw,
        (100, 150),
        ignored_app_ids=("com.android.systemui",),
    )

    assert normalized.app_id == "com.example.app"
    assert [element.text for element in normalized.elements] == ["App", "Clock", "Shade"]


def test_platform_selection_flows_through_env_and_combined_cli_device_overrides(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cfg = load_config(cwd=tmp_path, env={"AUA_PLATFORM": "TEST-NATIVE"})

    overrides = GlobalOpts(platform="test-native", serial="target-1").cli_overrides()

    assert cfg.device.platform == "test-native"
    assert overrides["device"] == {"platform": "test-native", "serial": "target-1"}
    assert hoist_global_options(["devices", "--platform", "test-native"]) == [
        "--platform",
        "test-native",
        "devices",
    ]
