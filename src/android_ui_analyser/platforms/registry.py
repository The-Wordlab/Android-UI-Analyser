"""Platform strategy registration and selection."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from importlib import metadata
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar, cast

from ..errors import (
    ConfigError,
    IncompatiblePlatformPluginError,
    PlatformPluginLoadError,
)
from .api import PLATFORM_API_VERSION
from .base import PlatformAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

# Third-party distributions can publish a PlatformAdapter class under this group.
ENTRY_POINT_GROUP = "aua.platforms"
_AdapterT = TypeVar("_AdapterT", bound=type[PlatformAdapter])
_REGISTRY: dict[str, type[PlatformAdapter]] = {}
_BUILTINS_LOADED = False
_BUILTIN_MODULES: dict[str, str] = {
    "android": "android_ui_analyser.platforms.android",
}
_ENTRY_POINTS: dict[str, list[metadata.EntryPoint]] | None = None


def _normalise_name(name: str) -> str:
    value = name.strip().lower()
    if not value:
        raise ConfigError("platform name cannot be empty")
    return value


def register_platform(name: str) -> Callable[[_AdapterT], _AdapterT]:
    """Register a platform adapter class under a stable configuration name."""

    key = _normalise_name(name)

    def decorate(adapter: _AdapterT) -> _AdapterT:
        if not issubclass(adapter, PlatformAdapter):
            raise TypeError(f"{adapter!r} is not a PlatformAdapter")
        existing = _REGISTRY.get(key)
        if (
            existing is not None
            and existing is not adapter
            and getattr(existing, "_aua_registration_source", None) is not adapter
        ):
            raise ConfigError(f"platform '{key}' is already registered")
        if existing is not None:
            return cast(_AdapterT, existing)
        # A distribution may expose the same implementation through multiple entry points.
        # Never mutate that shared class: doing so changes the identity of live adapters and
        # can redirect their leases, worker sockets, and undo records to the other platform.
        bound = cast(_AdapterT, type(adapter.__name__, (adapter,), {
            "name": key,
            "__module__": adapter.__module__,
            "_aua_registration_source": adapter,
        }))
        _REGISTRY[key] = bound
        return bound

    return decorate


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    for module in _BUILTIN_MODULES.values():
        importlib.import_module(module)


def _load_builtin(name: str) -> None:
    """Load one selected built-in without importing unrelated native platforms."""

    global _BUILTINS_LOADED
    module = _BUILTIN_MODULES.get(name)
    if module is None or name in _REGISTRY:
        return
    importlib.import_module(module)
    if set(_BUILTIN_MODULES).issubset(_REGISTRY):
        _BUILTINS_LOADED = True


def _discover_entry_points() -> dict[str, list[metadata.EntryPoint]]:
    """Index installed plugins without importing any of them."""

    global _ENTRY_POINTS
    if _ENTRY_POINTS is not None:
        return _ENTRY_POINTS
    found: dict[str, list[metadata.EntryPoint]] = {}
    try:
        entries = metadata.entry_points().select(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - broken Python packaging installation
        raise PlatformPluginLoadError("<discovery>", f"{type(exc).__name__}: {exc}") from exc
    for entry in entries:
        try:
            key = _normalise_name(entry.name)
        except ConfigError:
            # An empty entry-point name cannot ever be selected, so it must not prevent Android
            # or another well-formed plugin from loading.
            continue
        found.setdefault(key, []).append(entry)
    _ENTRY_POINTS = found
    return found


def _validate_adapter_type(name: str, value: object) -> type[PlatformAdapter]:
    if not isinstance(value, type) or not issubclass(value, PlatformAdapter):
        raise PlatformPluginLoadError(
            name,
            "entry point must resolve to a PlatformAdapter subclass",
        )
    actual_version = getattr(value, "platform_api_version", None)
    if actual_version != PLATFORM_API_VERSION:
        raise IncompatiblePlatformPluginError(
            name,
            expected=PLATFORM_API_VERSION,
            actual=actual_version,
        )
    return value


def _load_selected_entry_point(name: str) -> type[PlatformAdapter] | None:
    """Import exactly the selected plugin; unrelated broken plugins stay isolated."""

    matches = _discover_entry_points().get(name, [])
    if not matches:
        return None
    if len(matches) > 1:
        providers = ", ".join(
            sorted(
                str(getattr(entry, "dist", None) or getattr(entry, "value", "unknown"))
                for entry in matches
            )
        )
        raise ConfigError(
            f"multiple platform plugins are installed as {name!r}: {providers}",
            hint=(
                f"Keep exactly one {ENTRY_POINT_GROUP!r} entry point named {name!r}, then "
                "retry."
            ),
            code="platform_plugin_ambiguous",
        )
    entry = matches[0]
    try:
        loaded = entry.load()
    except Exception as exc:
        raise PlatformPluginLoadError(name, f"plugin import failed ({type(exc).__name__})") from None
    adapter = _validate_adapter_type(name, loaded)
    return register_platform(name)(adapter)


def available_platforms() -> tuple[str, ...]:
    """Names that may be selected, discovered without importing plugin modules."""

    return tuple(
        sorted(set(_BUILTIN_MODULES) | set(_REGISTRY) | set(_discover_entry_points()))
    )


def registered_platforms() -> dict[str, type[PlatformAdapter]]:
    """Return adapter classes already loaded in this process.

    Installed plugins are intentionally absent until selected. Use :func:`available_platforms`
    when only their names are needed.
    """

    _load_builtins()
    return dict(_REGISTRY)


class PlatformFactory:
    """Create and memoize the platform strategy selected by configuration."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._instances: dict[str, PlatformAdapter] = {}

    def create(self, name: str | None = None) -> PlatformAdapter:
        key = _normalise_name(name or self.config.device.platform)
        if key in self._instances:
            return self._instances[key]
        adapter_type = _REGISTRY.get(key)
        if adapter_type is None and key in _BUILTIN_MODULES:
            _load_builtin(key)
            adapter_type = _REGISTRY.get(key)
        if adapter_type is None:
            adapter_type = _load_selected_entry_point(key)
        if adapter_type is None:
            known = ", ".join(available_platforms()) or "none"
            raise ConfigError(
                f"unknown platform '{key}'",
                hint=(
                    f"Installed platforms: {known}. Install a plugin exposing the "
                    f"'{ENTRY_POINT_GROUP}' entry-point group, or choose one of these names."
                ),
            )
        _validate_adapter_type(key, adapter_type)
        try:
            adapter = adapter_type(self.config)
        except Exception as exc:
            raise PlatformPluginLoadError(
                key,
                f"adapter initialization failed ({type(exc).__name__})",
            ) from None
        try:
            validated_options = adapter.validate_options(self.config.platform_options(key))
        except Exception as exc:
            raise ConfigError(
                f"platform {key!r} rejected its configuration ({type(exc).__name__})",
                hint="Fix the selected platform's options and retry.",
                code="platform_options_invalid",
            ) from None
        if not isinstance(validated_options, Mapping):
            raise ConfigError(
                f"platform {key!r} returned invalid normalized options",
                hint="The adapter validate_options hook must return a mapping.",
                code="platform_options_invalid",
            )
        adapter.options = MappingProxyType(dict(validated_options))
        adapter.validate_declared_capabilities()
        self._instances[key] = adapter
        return adapter
