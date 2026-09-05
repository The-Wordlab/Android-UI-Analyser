"""Structural contracts for platform and per-target capabilities.

Capability discovery is part of AUA's public plugin boundary. A string in an adapter's
``capabilities`` set is therefore a promise, not documentation: this module describes which
object owns that promise and which members must exist before shared code may invoke it.

Host-wide optional services keep their larger contracts in :mod:`platforms.services`; importing
that catalogue here would create an unnecessary cycle, so :class:`PlatformAdapter` combines the
two catalogues when resolving a service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from inspect import getattr_static
from typing import Any


class CapabilityScope(StrEnum):
    """The object on which a capability's structural members live."""

    ADAPTER = "adapter"
    RUNTIME = "runtime"
    SERVICE = "service"


@dataclass(frozen=True)
class CapabilitySpec:
    """One stable platform capability and its complete callable/data surface."""

    name: str
    scope: CapabilityScope
    members: frozenset[str]
    inherited_defaults: frozenset[str] = frozenset()


def _spec(
    name: str,
    scope: CapabilityScope,
    *members: str,
    inherited_defaults: tuple[str, ...] = (),
) -> CapabilitySpec:
    return CapabilitySpec(
        name=name,
        scope=scope,
        members=frozenset(members),
        inherited_defaults=frozenset(inherited_defaults),
    )


# Per-target operations. The runtime is deliberately split into focused surfaces so a platform
# can ship hierarchy/input first and explicitly refuse features it cannot support.
RUNTIME_CAPABILITIES: dict[str, CapabilitySpec] = {
    "ui.tree": _spec(
        "ui.tree",
        CapabilityScope.RUNTIME,
        "current_app",
        "display_geometry",
        "window_size",
        "dump_hierarchy",
        inherited_defaults=("display_geometry",),
    ),
    "ui.input": _spec(
        "ui.input",
        CapabilityScope.RUNTIME,
        "click",
        "long_click",
        "send_text",
        "clear_text",
        "send_ime_action",
        "swipe",
        "press",
        "find_text",
    ),
    "app.lifecycle": _spec(
        "app.lifecycle",
        CapabilityScope.RUNTIME,
        "launch_app",
        "stop_app",
        "clear_app",
        "grant_permissions",
        "granted_permissions",
        "restore_permissions",
    ),
    "app.files": _spec(
        "app.files",
        CapabilityScope.RUNTIME,
        "read_app_file",
        "write_app_file",
        "remove_app_files",
    ),
    "app.links": _spec(
        "app.links", CapabilityScope.RUNTIME, "open_link", "query_uri_handlers"
    ),
    "device.keyboard": _spec(
        "device.keyboard",
        CapabilityScope.RUNTIME,
        "hide_keyboard",
        "keyboard_visible",
        "erase_chars",
    ),
    "device.clipboard": _spec(
        "device.clipboard",
        CapabilityScope.RUNTIME,
        "set_clipboard",
        "get_clipboard",
        "paste",
    ),
    "device.location": _spec("device.location", CapabilityScope.RUNTIME, "set_location"),
    "device.orientation": _spec(
        "device.orientation", CapabilityScope.RUNTIME, "set_orientation", "get_orientation"
    ),
    "device.airplane": _spec(
        "device.airplane", CapabilityScope.RUNTIME, "set_airplane_mode", "get_airplane_mode"
    ),
    "device.media": _spec(
        "device.media",
        CapabilityScope.RUNTIME,
        "media_directory",
        "add_media",
        "remove_added_media",
    ),
    "device.recording": _spec(
        "device.recording",
        CapabilityScope.RUNTIME,
        "active_recording",
        "recording_destination",
        "start_recording",
        "stop_recording",
        "discard_recording",
    ),
    "device.clock": _spec(
        "device.clock",
        CapabilityScope.RUNTIME,
        "set_clock",
        "get_clock_ms",
        "utc_offset_minutes",
    ),
    "device.accessibility": _spec(
        "device.accessibility", CapabilityScope.RUNTIME, "a11y_action"
    ),
    "device.touch": _spec(
        "device.touch", CapabilityScope.RUNTIME, "click_once", "touch_down", "touch_up"
    ),
    "device.proxy": _spec(
        "device.proxy",
        CapabilityScope.RUNTIME,
        "set_http_proxy",
        "reverse_port",
        "remove_reverse_port",
    ),
    "device.shell": _spec(
        "device.shell", CapabilityScope.RUNTIME, "run_read_only_shell"
    ),
}


# Operations implemented by the adapter itself rather than its connected runtime.
ADAPTER_CAPABILITIES: dict[str, CapabilitySpec] = {
    "ui.screenshot": _spec(
        "ui.screenshot", CapabilityScope.ADAPTER, "capture_screenshot"
    ),
    "app.status": _spec("app.status", CapabilityScope.ADAPTER, "installed_app"),
    "app.install": _spec(
        "app.install",
        CapabilityScope.ADAPTER,
        "inspect_app_bundle",
        "installed_app",
        "install_app_bundle",
        "uninstall_app",
    ),
    "device.logs": _spec(
        "device.logs",
        CapabilityScope.ADAPTER,
        "clear_diagnostics",
        "diagnostic_logs",
        "diagnostic_window",
        "mark_diagnostics",
        "recent_logs",
    ),
}


DIRECT_CAPABILITIES: dict[str, CapabilitySpec] = {
    **RUNTIME_CAPABILITIES,
    **ADAPTER_CAPABILITIES,
}


def normalize_capability(name: str) -> str:
    """Canonical spelling shared by declaration, lookup, error, and cache keys."""

    normalized = str(name).strip().lower().replace("-", "_")
    # These names predate the platform contract and described Android's AVD implementation.
    # Keep accepting them without freezing Android vocabulary into API v1.
    if normalized in {"emulator", "virtual_devices"}:
        return "virtual_targets"
    return normalized


def missing_structural_members(
    target: Any,
    members: frozenset[str],
    *,
    default_owner: type[Any] | None = None,
    inherited_defaults: frozenset[str] = frozenset(),
) -> list[str]:
    """Return members that are absent or not callable on *target*.

    The current contracts contain operations only. Keeping this check strict catches the common
    plugin failure where a capability is copied into the declaration but its implementation was
    omitted or set to ``None``.
    """

    missing: list[str] = []
    for name in members:
        if not callable(getattr(target, name, None)):
            missing.append(name)
            continue
        if default_owner is None:
            continue
        try:
            implementation = getattr_static(type(target), name)
            default = getattr_static(default_owner, name)
        except AttributeError:
            continue
        if implementation is default and name not in inherited_defaults:
            missing.append(name)
    return sorted(missing)
