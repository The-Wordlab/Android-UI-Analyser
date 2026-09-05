from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.errors import (
    ConfigError,
    IncompatiblePlatformPluginError,
    PlatformPluginLoadError,
)
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter, registry
from android_ui_analyser.platforms.runtime import TargetRuntime
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.schema import DeviceInfo


class _PluginPlatform(PlatformAdapter):
    capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot"})

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        raise AssertionError("registry tests do not connect")

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

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        raise AssertionError("registry tests do not capture")


@dataclass
class _EntryPoint:
    name: str
    value: str
    result: object = _PluginPlatform
    error: Exception | None = None
    loads: int = 0

    def load(self) -> object:
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.result


def _entries(monkeypatch: pytest.MonkeyPatch, **entries: _EntryPoint) -> None:
    monkeypatch.setattr(registry, "_ENTRY_POINTS", {name: [entry] for name, entry in entries.items()})


def test_factory_imports_only_the_selected_platform_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _EntryPoint("registry-lazy-good", "test:Good")
    broken = _EntryPoint(
        "registry-unrelated-broken",
        "missing:Broken",
        error=ImportError("optional native dependency is absent"),
    )
    _entries(
        monkeypatch,
        **{
            selected.name: selected,
            broken.name: broken,
        },
    )
    config = Config.model_validate({"device": {"platform": selected.name}})

    platform = registry.PlatformFactory(config).create()

    assert isinstance(platform, _PluginPlatform)
    assert selected.loads == 1
    assert broken.loads == 0


def test_selected_plugin_import_failure_is_precise(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _EntryPoint(
        "registry-load-failure",
        "missing:Plugin",
        error=ImportError("native bridge not installed"),
    )
    _entries(monkeypatch, **{entry.name: entry})
    factory = registry.PlatformFactory(
        Config.model_validate({"device": {"platform": entry.name}})
    )

    with pytest.raises(PlatformPluginLoadError) as exc:
        factory.create()

    assert exc.value.code == "platform_plugin_load_failed"
    assert "native bridge not installed" in exc.value.message


def test_plugin_api_version_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FuturePlugin(_PluginPlatform):
        platform_api_version = registry.PLATFORM_API_VERSION + 1

    entry = _EntryPoint("registry-future-api", "test:FuturePlugin", result=FuturePlugin)
    _entries(monkeypatch, **{entry.name: entry})
    factory = registry.PlatformFactory(
        Config.model_validate({"device": {"platform": entry.name}})
    )

    with pytest.raises(IncompatiblePlatformPluginError) as exc:
        factory.create()

    assert exc.value.code == "platform_api_incompatible"
    assert str(registry.PLATFORM_API_VERSION + 1) in exc.value.message


def test_duplicate_installed_plugin_names_fail_without_importing_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _EntryPoint("registry-duplicate", "one:Plugin")
    second = _EntryPoint("registry-duplicate", "two:Plugin")
    monkeypatch.setattr(registry, "_ENTRY_POINTS", {first.name: [first, second]})
    factory = registry.PlatformFactory(
        Config.model_validate({"device": {"platform": first.name}})
    )

    with pytest.raises(ConfigError) as exc:
        factory.create()

    assert exc.value.code == "platform_plugin_ambiguous"
    assert first.loads == second.loads == 0


def test_available_platform_names_do_not_import_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _EntryPoint(
        "registry-name-only",
        "missing:Plugin",
        error=ImportError("must not be imported"),
    )
    _entries(monkeypatch, **{entry.name: entry})

    assert "registry-name-only" in registry.available_platforms()
    assert entry.loads == 0


def test_plugin_initialization_failure_has_a_stable_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenInit(_PluginPlatform):
        def __init__(self, config: Config) -> None:
            raise RuntimeError("bridge handshake exploded")

    entry = _EntryPoint("registry-init-failure", "test:BrokenInit", result=BrokenInit)
    _entries(monkeypatch, **{entry.name: entry})

    with pytest.raises(PlatformPluginLoadError) as exc:
        registry.PlatformFactory(
            Config.model_validate({"device": {"platform": entry.name}})
        ).create()

    assert exc.value.code == "platform_plugin_load_failed"
    assert "adapter initialization failed" in exc.value.message


def test_selected_plugin_owns_and_normalizes_its_namespaced_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OptionsPlugin(_PluginPlatform):
        seen: dict[str, object] | None = None

        def validate_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
            self.seen = dict(options)
            return {"endpoint": str(options["endpoint"]).rstrip("/")}

    entry = _EntryPoint("registry-options", "test:OptionsPlugin", result=OptionsPlugin)
    _entries(monkeypatch, **{entry.name: entry})
    config = Config.model_validate(
        {
            "device": {"platform": entry.name},
            "platforms": {
                entry.name: {"endpoint": "https://example.invalid/"},
                "unselected": {"broken": True},
            },
        }
    )

    platform = registry.PlatformFactory(config).create()

    assert isinstance(platform, OptionsPlugin)
    assert platform.seen == {"endpoint": "https://example.invalid/"}
    assert dict(platform.options) == {"endpoint": "https://example.invalid"}
