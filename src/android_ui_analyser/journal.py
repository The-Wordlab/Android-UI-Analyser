"""Append-only JSONL journal of aua commands — feeds the dashboard + usage analysis.

Every daemon dispatch and every in-process CLI ``_route`` writes one line under
``cache.dir/journal/<serial_or_host>.jsonl``. Dashboard tails it for live agent I/O.
Secrets in args are redacted (password/token/key/secret/text bodies truncated).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_REDACT_KEYS = frozenset(
    {
        "password",
        "passwd",
        "token",
        "api_key",
        "apikey",
        "secret",
        "authorization",
        "cookie",
    }
)
_MAX_STR = 400
_MAX_RESULT = 1200
_MAX_FILE_BYTES = 8 * 1024 * 1024  # rotate soft-cap per serial file


def journal_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir).expanduser() / "journal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def journal_path(cache_dir: str | Path, serial: str | None) -> Path:
    safe = "host"
    if serial:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in serial)
    return journal_dir(cache_dir) / f"{safe}.jsonl"


def _truncate(val: Any, limit: int = _MAX_STR) -> Any:
    if isinstance(val, str) and len(val) > limit:
        return val[:limit] + f"…(+{len(val) - limit})"
    if isinstance(val, dict):
        return {k: _truncate(v, limit) for k, v in list(val.items())[:40]}
    if isinstance(val, list):
        return [_truncate(v, limit) for v in val[:30]]
    return val


def redact_args(args: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        lk = str(k).lower()
        if lk in _REDACT_KEYS or any(p in lk for p in ("password", "secret", "token")):
            out[k] = "<redacted>"
        elif k in ("text", "body", "value") and isinstance(v, str) and len(v) > 80:
            out[k] = v[:80] + "…"
        else:
            out[k] = _truncate(v)
    return out


def summarize_result(result: Any) -> Any:
    """Keep journal lines small but useful for the dashboard."""
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        with contextlib.suppress(Exception):
            result = result.model_dump(mode="json")
    if isinstance(result, dict):
        slim: dict[str, Any] = {}
        for key in (
            "ok",
            "action",
            "detail",
            "code",
            "hint",
            "serial",
            "found",
            "matched",
            "path",
            "package",
            "activity",
            "known_screen",
            "tier_used",
            "duration_ms",
            "via",
            "error",
            "message",
        ):
            if key in result:
                slim[key] = _truncate(result[key], 200)
        meta = result.get("meta")
        if isinstance(meta, dict):
            slim["meta"] = {
                k: meta.get(k)
                for k in (
                    "known_screen",
                    "tier_used",
                    "duration_ms",
                    "via",
                    "package",
                    "capture_hint",
                    "path",
                )
                if k in meta
            }
        screen = result.get("screen")
        if isinstance(screen, dict):
            slim["screen"] = {
                k: screen.get(k) for k in ("package", "activity", "width", "height") if k in screen
            }
        if "elements" in result and isinstance(result["elements"], list):
            slim["elements_count"] = len(result["elements"])
        if "observation" in result and isinstance(result["observation"], dict):
            obs = result["observation"]
            slim["observation"] = {
                "elements_count": len(obs.get("elements") or []),
                "meta": (obs.get("meta") or {}).get("known_screen")
                if isinstance(obs.get("meta"), dict)
                else None,
            }
        raw = json.dumps(slim, ensure_ascii=False, default=str)
        if len(raw) > _MAX_RESULT:
            return _truncate(slim, 120)
        return slim
    return _truncate(result, 200)


def record(
    *,
    cache_dir: str | Path,
    serial: str | None,
    source: str,
    cmd: str,
    args: dict[str, Any] | None = None,
    ok: bool = True,
    duration_ms: float | None = None,
    result: Any = None,
    error: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one journal event (best-effort; never raises into the caller)."""
    try:
        path = journal_path(cache_dir, serial)
        event: dict[str, Any] = {
            "ts": time.time(),
            "ts_ms": int(time.time() * 1000),
            "source": source,
            "cmd": cmd,
            "args": redact_args(args),
            "ok": ok,
            "serial": serial,
            "pid": os.getpid(),
        }
        if duration_ms is not None:
            event["duration_ms"] = round(float(duration_ms), 1)
        if error:
            event["error"] = _truncate(error, 300)
        if result is not None:
            event["result"] = summarize_result(result)
        if extra:
            event["extra"] = _truncate(extra, 200)
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with _lock:
            if path.is_file() and path.stat().st_size > _MAX_FILE_BYTES:
                rotated = path.with_suffix(".jsonl.1")
                with contextlib.suppress(OSError):
                    rotated.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    path.rename(rotated)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        # Keep headless idle-watchdog from auto-stopping mid-session.
        with contextlib.suppress(Exception):
            from .emulator import touch_activity

            touch_activity(cache_dir, serial)
    except Exception as exc:  # noqa: BLE001
        logger.debug("journal write failed: %s", exc)


def read_since(
    cache_dir: str | Path,
    serial: str | None,
    *,
    since_ms: int | None = None,
    limit: int = 200,
    include_dashboard: bool = False,
) -> list[dict[str, Any]]:
    path = journal_path(cache_dir, serial)
    # Also merge host journal if serial-specific is empty of recent events.
    paths = [path]
    host = journal_path(cache_dir, None)
    if host != path:
        paths.append(host)
    events: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[-(limit * 2) :]:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if not include_dashboard and row.get("source") == "dashboard":
                    continue
                if since_ms is not None and int(row.get("ts_ms") or 0) < since_ms:
                    continue
                # keep host events without a serial; drop other devices
                if serial and row.get("serial") not in (None, serial, ""):
                    continue
                events.append(row)
    events.sort(key=lambda e: int(e.get("ts_ms") or 0))
    return events[-limit:]


def failure_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Cheap usage rollup for the dashboard (failing cmds, slow calls)."""
    by_cmd: dict[str, dict[str, int]] = {}
    slow: list[dict[str, Any]] = []
    fails: list[dict[str, Any]] = []
    for e in events:
        cmd = str(e.get("cmd") or "?")
        slot = by_cmd.setdefault(cmd, {"ok": 0, "fail": 0})
        if e.get("ok"):
            slot["ok"] += 1
        else:
            slot["fail"] += 1
            fails.append(
                {
                    "ts_ms": e.get("ts_ms"),
                    "cmd": cmd,
                    "error": (e.get("error") or {}).get("message")
                    if isinstance(e.get("error"), dict)
                    else e.get("error"),
                }
            )
        dur = e.get("duration_ms")
        if isinstance(dur, (int, float)) and dur >= 1500:
            slow.append({"ts_ms": e.get("ts_ms"), "cmd": cmd, "duration_ms": dur})
    return {
        "by_cmd": by_cmd,
        "failures": fails[-30:],
        "slow": slow[-30:],
        "total": len(events),
        "fail_count": sum(1 for e in events if not e.get("ok")),
    }
