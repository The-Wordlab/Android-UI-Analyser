"""Platform strategy registration and selection."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from importlib import metadata
from typing import TYPE_CHECKING, TypeVar

from ..errors import ConfigError
from .base import PlatformAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

logger = logging.getLogger("android_ui_analyser.platforms")

# Third-party distributions can publish a PlatformAdapter class under this group.
ENTRY_POINT_GROUP = "aua.platforms"

_AdapterT = TypeVar("_AdapterT", bound=type[PlatformAdapter])
_REGISTRY: dict[str, type[PlatformAdapter]] = {}
_BUILTINS_LOADED = False
_ENTRY_POINTS_LOADED = False


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
        if existing is not None and existing is not adapter:
            raise ConfigError(f"platform '{key}' is already registered")
        adapter.name = key
        _REGISTRY[key] = adapter
        return adapter

    return decorate


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    importlib.import_module("android_ui_analyser.platforms.android")


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    entries = metadata.entry_points().select(group=ENTRY_POINT_GROUP)
    for entry in entries:
        try:
            adapter = entry.load()
            register_platform(entry.name)(adapter)
        except Exception as exc:  # a broken optional plugin must not break built-in Android
            logger.warning("ignoring platform plugin %s: %s", entry.name, exc)


def registered_platforms() -> dict[str, type[PlatformAdapter]]:
    """Return all built-in and installed platform strategies by name."""

    _load_builtins()
    _load_entry_points()
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
        available = registered_platforms()
        adapter_type = available.get(key)
        if adapter_type is None:
            known = ", ".join(sorted(available)) or "none"
            raise ConfigError(
                f"unknown platform '{key}'",
                hint=(
                    f"Installed platforms: {known}. Install a plugin exposing the "
                    f"'{ENTRY_POINT_GROUP}' entry-point group, or choose one of these names."
                ),
            )
        adapter = adapter_type(self.config)
        self._instances[key] = adapter
        return adapter
