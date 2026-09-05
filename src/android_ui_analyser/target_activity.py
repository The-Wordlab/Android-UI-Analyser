"""Platform-neutral heartbeat for target activity observed by shared transports.

The journal used to update Android emulator metadata directly.  That made a generic evidence
writer import an Android service and meant another platform's virtual-target supervisor could
not observe the same activity.  A heartbeat is coordination state, so it is keyed by
``TargetRef`` and contains no native-platform details.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text
from .platforms.identity import LEGACY_PLATFORM, TargetLike, target_ref


def activity_path(
    cache_dir: str | Path,
    target: TargetLike,
    *,
    platform: str = LEGACY_PLATFORM,
) -> Path:
    """Return the namespaced heartbeat path for one automation target."""

    ref = target_ref(target, platform=platform)
    return Path(cache_dir).expanduser() / "target-activity" / f"{ref.storage_key}.json"


def touch(
    cache_dir: str | Path,
    target: TargetLike | None,
    *,
    platform: str = LEGACY_PLATFORM,
    at: float | None = None,
) -> None:
    """Record target activity without importing or connecting its platform adapter."""

    if target is None:
        return
    ref = target_ref(target, platform=platform)
    payload = {
        **ref.to_json(),
        "last_activity": time.time() if at is None else float(at),
    }
    atomic_write_text(
        activity_path(cache_dir, ref),
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    )


def read(
    cache_dir: str | Path,
    target: TargetLike,
    *,
    platform: str = LEGACY_PLATFORM,
) -> dict[str, Any] | None:
    """Read one valid heartbeat, treating corrupt or missing state as absent."""

    ref = target_ref(target, platform=platform)
    try:
        raw = json.loads(activity_path(cache_dir, ref).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("platform", LEGACY_PLATFORM) != ref.platform:
        return None
    seen_id = raw.get("target_id") or raw.get("serial")
    if seen_id != ref.target_id:
        return None
    try:
        stamp = float(raw["last_activity"])
    except (KeyError, TypeError, ValueError):
        return None
    return {**raw, "last_activity": stamp}


__all__ = ["activity_path", "read", "touch"]
