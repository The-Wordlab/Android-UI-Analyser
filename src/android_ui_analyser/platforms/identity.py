"""Stable, platform-scoped identity for one automation target.

Historically AUA used an Android ``serial`` as the global identity of a device.  That is
safe only while Android is the sole platform: an iOS simulator and an Android emulator may
legitimately expose the same target identifier, while leases, undo records and detached
processes must never be shared between them.

``TargetRef`` is deliberately small and dependency-free so coordination modules can use it
without importing a platform runtime.  Public APIs continue to accept a bare serial, which
is interpreted as Android for backwards compatibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

LEGACY_PLATFORM = "android"


def _required(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def safe_component(value: object) -> str:
    """A readable filesystem component with a collision-resistant suffix when escaped."""

    raw = _required(value, field="identity component")
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in raw)
    if safe == raw:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}"


@dataclass(frozen=True, slots=True)
class TargetRef:
    """The globally unique identity of one platform automation target."""

    platform: str
    target_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required(self.platform, field="platform").lower())
        object.__setattr__(self, "target_id", _required(self.target_id, field="target_id"))

    @property
    def serial(self) -> str:
        """Compatibility spelling used by the existing Android public surfaces."""

        return self.target_id

    @property
    def storage_key(self) -> str:
        """Filesystem key, preserving every historical Android path exactly."""

        legacy = "".join(
            char if char.isalnum() or char in "-_." else "_" for char in self.target_id
        )
        if self.platform == LEGACY_PLATFORM:
            return legacy
        return f"{safe_component(self.platform)}--{safe_component(self.target_id)}"

    def to_json(self) -> dict[str, str]:
        """Compatibility metadata: neutral fields plus the historical ``serial`` alias."""

        return {
            "platform": self.platform,
            "target_id": self.target_id,
            "serial": self.target_id,
        }


TargetLike = str | TargetRef


def target_ref(value: TargetLike, *, platform: str = LEGACY_PLATFORM) -> TargetRef:
    """Normalize a target argument; a bare serial remains an Android target."""

    if isinstance(value, TargetRef):
        if platform != LEGACY_PLATFORM and value.platform != platform.lower():
            raise ValueError(
                f"target platform {value.platform!r} does not match requested {platform!r}"
            )
        return value
    return TargetRef(platform=platform, target_id=value)


@dataclass(frozen=True, slots=True)
class AppRef:
    """The platform-scoped identity of an installed application."""

    platform: str
    app_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required(self.platform, field="platform").lower())
        object.__setattr__(self, "app_id", _required(self.app_id, field="app_id"))

    @property
    def package(self) -> str:
        """Compatibility spelling for Android-facing call sites."""

        return self.app_id

    @property
    def storage_key(self) -> str:
        legacy = "".join(
            char if char.isalnum() or char in "-_." else "_" for char in self.app_id
        )
        if self.platform == LEGACY_PLATFORM:
            return legacy
        return f"{safe_component(self.platform)}--{safe_component(self.app_id)}"

    def to_json(self) -> dict[str, str]:
        return {
            "platform": self.platform,
            "app_id": self.app_id,
            "package": self.app_id,
        }


AppLike = str | AppRef


def app_ref(value: AppLike, *, platform: str = LEGACY_PLATFORM) -> AppRef:
    """Normalize an application identity; a bare package remains Android."""

    if isinstance(value, AppRef):
        if platform != LEGACY_PLATFORM and value.platform != platform.lower():
            raise ValueError(
                f"app platform {value.platform!r} does not match requested {platform!r}"
            )
        return value
    return AppRef(platform=platform, app_id=value)


def target_from_metadata(
    raw: dict[str, Any],
    *,
    fallback_id: str | None = None,
    fallback_platform: str = LEGACY_PLATFORM,
) -> TargetRef | None:
    """Read current or legacy state metadata without guessing a non-Android platform.

    Untagged records predate platform plugins and therefore came from Android.  Corrupt or
    incomplete metadata returns ``None`` so callers can preserve their existing fail-closed
    behaviour.
    """

    platform = raw.get("platform")
    if not isinstance(platform, str) or not platform.strip():
        platform = fallback_platform
    target_id = raw.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        target_id = raw.get("serial")
    if not isinstance(target_id, str) or not target_id.strip():
        target_id = fallback_id
    if not target_id:
        return None
    try:
        return TargetRef(platform=platform, target_id=target_id)
    except ValueError:
        return None


__all__ = [
    "AppLike",
    "AppRef",
    "LEGACY_PLATFORM",
    "TargetLike",
    "TargetRef",
    "app_ref",
    "safe_component",
    "target_from_metadata",
    "target_ref",
]
