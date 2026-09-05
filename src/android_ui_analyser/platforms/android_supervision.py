"""Android implementation of optional target lifecycle supervision metadata."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from .supervision import TargetSupervisionStatus


def _pid_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_record(cache_dir: str | Path, target_id: str) -> dict[str, Any] | None:
    """Return Android's newest AUA emulator record for *target_id*."""

    root = Path(cache_dir).expanduser() / "emulator"
    if not root.is_dir():
        return None
    matches: list[tuple[float, dict[str, Any]]] = []
    for path in root.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or raw.get("serial") != target_id:
            continue
        stamp = _number(raw.get("started_at")) or 0.0
        with contextlib.suppress(OSError):
            stamp = max(stamp, path.stat().st_mtime)
        matches.append((stamp, dict(raw)))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def target_supervision_status(
    target_id: str, *, cache_dir: str | Path
) -> TargetSupervisionStatus | None:
    """Translate Android emulator bookkeeping into the neutral supervision contract."""

    meta = _latest_record(cache_dir, target_id)
    if meta is None:
        return None
    return TargetSupervisionStatus(
        target_id=target_id,
        managed=meta.get("started_by_aua") is True,
        owner=str(meta["owner"]) if meta.get("owner") else None,
        instance_id=(
            str(meta.get("instance") or meta.get("avd"))
            if meta.get("instance") or meta.get("avd")
            else None
        ),
        started_at=_number(meta.get("started_at")),
        last_activity=_number(meta.get("last_activity")),
        idle_timeout_s=_number(meta.get("idle_timeout_s")),
        monitor_running=_pid_is_alive(meta.get("watchdog_pid")),
        idle_stop_explicit=bool(meta.get("idle_stop_explicit")),
    )


__all__ = ["target_supervision_status"]
