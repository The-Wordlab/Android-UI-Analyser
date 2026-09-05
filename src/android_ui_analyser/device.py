"""Compatibility facade for AUA's historical Android ``Device`` module.

Reusable platform code now types against :class:`platforms.runtime.TargetRuntime`.  These
re-exports keep existing integrations and tests source-compatible while Android implementation
details live in platform-owned modules.
"""

from __future__ import annotations

from typing import Any

from .errors import DeviceError, UsageError
from .platforms import android_device as _android
from .platforms.android_device import (
    AndroidRuntimeBase as Device,
)
from .platforms.android_device import (
    Uiautomator2Device,
    connect,
    finalized_mp4,
    list_devices,
    parse_locale,
    resolve_serial,
)
from .platforms.runtime import TargetRuntime


def __getattr__(name: str) -> Any:
    """Forward legacy helpers, including private test seams, to the Android backend."""

    try:
        return getattr(_android, name)
    except AttributeError as exc:  # pragma: no cover - normal module attribute semantics
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_android)})


__all__ = [
    "Device",
    "DeviceError",
    "TargetRuntime",
    "Uiautomator2Device",
    "UsageError",
    "connect",
    "finalized_mp4",
    "list_devices",
    "parse_locale",
    "resolve_serial",
]
