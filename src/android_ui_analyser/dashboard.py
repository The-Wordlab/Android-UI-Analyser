"""Local sneak-peek dashboard for headless (or headed) agent runs.

A separate process from the agent: ``aua dashboard`` enables capture if needed
(daemon or sidecar), then serves a localhost HTML page that live-polls frames,
the agent I/O journal, app map, logcat, and capture marks. Bind 127.0.0.1 only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import mimetypes
import os
import secrets
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .errors import AuaError, UsageError

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8765

# A capture file older than this is only trusted while capture is known to be alive:
# the ring dedupes unchanged screens, so an old frame is normal there and a lie once
# whatever was writing it has stopped.
_FRAME_STALE_S = 3.0

# Header names whose VALUE is a credential or a tracking identity. The proxy captures whole
# exchanges, so the panel would otherwise hand a bearer token to anything that can reach
# localhost — and to anyone the page is screenshotted for.
_SECRET_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-access-token",
        "x-session-token",
        "x-csrf-token",
        "x-device-key",
        "x-ad-id",
        "x-advertising-id",
        "x-correlation-id",
        "x-organization-id",
        "x-goog-api-key",
        "api-key",
        "authentication",
    }
)

# Body keys whose value is a credential. Matched case-insensitively as a substring, so
# `streamToken`, `access_token` and `refreshToken` are all caught by "token".
_SECRET_BODY_KEYS = ("token", "password", "secret", "api_key", "apikey", "authorization",
                     "credential", "session_id", "cookie")

_REDACTED = "<redacted>"


def _redact_headers(headers: Any) -> Any:
    if not isinstance(headers, dict):
        return headers
    return {
        k: (_REDACTED if str(k).lower() in _SECRET_HEADERS else v) for k, v in headers.items()
    }


def _redact_body(value: Any, depth: int = 0) -> Any:
    """Blank out credential-shaped values anywhere in a decoded JSON body."""
    if depth > 6:
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            low = str(k).lower()
            if any(marker in low for marker in _SECRET_BODY_KEYS):
                out[k] = _REDACTED
            else:
                out[k] = _redact_body(v, depth + 1)
        return out
    if isinstance(value, list):
        return [_redact_body(v, depth + 1) for v in value[:200]]
    return value


def redact_flow(entry: dict[str, Any]) -> dict[str, Any]:
    """A captured exchange with its credentials removed.

    The panel is read over plain HTTP on localhost and is routinely screenshotted into
    bug reports, so a captured `authorization` bearer must never leave this function.
    Header and field NAMES are kept — knowing an endpoint sends a bearer is exactly the
    kind of thing the panel is for; knowing its value is not.
    """
    out = dict(entry)
    for field in ("request_headers", "response_headers"):
        if field in out:
            out[field] = _redact_headers(out[field])
    for field in ("request_body", "response_body"):
        raw = out.get(field)
        if not isinstance(raw, str) or not raw.strip().startswith(("{", "[")):
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        out[field] = json.dumps(_redact_body(parsed), ensure_ascii=False)
    if isinstance(out.get("query"), str) and out["query"]:
        out["query"] = _redact_query(out["query"])
    return out


def _redact_query(query: str) -> str:
    parts = []
    for chunk in query.split("&"):
        name, sep, _value = chunk.partition("=")
        low = name.lower()
        if sep and any(marker in low for marker in _SECRET_BODY_KEYS):
            parts.append(f"{name}={_REDACTED}")
        else:
            parts.append(chunk)
    return "&".join(parts)

# url -> (server, thread) for dashboards started with block=False, so callers that
# do not own the serve loop can still stop one.
_SERVERS: dict[str, tuple[ThreadingHTTPServer, threading.Thread]] = {}

# 1x1 black PNG - served when a device has neither a capture file nor a screencap.
_PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _script_json(value: Any) -> str:
    """Serialize trusted bootstrap data without allowing an inline-script end tag."""

    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _safe_serial(serial: str) -> str:
    return str(serial).replace(":", "_").replace("/", "_")


def list_online_serials(config: Any | None = None) -> list[str]:
    from .config import load_config
    from .platforms import PlatformFactory

    cfg = config or load_config()
    platform = PlatformFactory(cfg).create()
    return [d.serial for d in platform.list_targets() if d.state == "device"]


def discover_online_serials(config: Any | None = None) -> tuple[list[str], str | None]:
    """Online serials plus the reason discovery failed, if it did.

    The dashboard only watches; it never drives a device. A missing or unreadable
    device list is therefore something to display, not a reason to refuse to open.
    """
    try:
        online = list_online_serials() if config is None else list_online_serials(config)
    except Exception as exc:  # noqa: BLE001 - the page still has to come up
        logger.warning("dashboard device discovery failed: %s", exc)
        return [], str(exc)
    return online, None


def resolve_dashboard_targets(
    serial: str | None = None, *, grid: bool = True, config: Any | None = None
) -> dict[str, Any]:
    """Pick grid vs detail mode.

    * Explicit ``--serial`` → detail for that device.
    * Unpinned/default ``--grid`` → grid that can discover later devices.
    * Explicit ``--detail`` with no serial → detail for the first online device.

    Nothing attached is not an error. With no serial to pin there is nothing to detail
    yet, so the empty grid opens and picks devices up as they boot - which is the
    useful order when you start the dashboard before the emulator.
    """
    online, discovery_error = discover_online_serials(config)
    if serial:
        if serial not in online and online:
            # Still allow watching a serial that briefly dropped offline.
            logger.warning("serial %s not currently online; dashboard will retry frames", serial)
        return {
            "mode": "detail",
            "serials": [serial],
            "focus": serial,
            "discovery_error": discovery_error,
        }
    if not online:
        if not grid:
            logger.info("no device to focus; opening the discovering grid instead")
        return {
            "mode": "grid",
            "serials": [],
            "focus": None,
            "discovery_error": discovery_error,
        }
    if grid:
        return {
            "mode": "grid",
            "serials": online,
            "focus": None,
            "discovery_error": discovery_error,
        }
    return {
        "mode": "detail",
        "serials": online,
        "focus": online[0],
        "discovery_error": discovery_error,
    }


def _emulator_meta_for_serial(
    cache_dir: str | Path, serial: str
) -> dict[str, Any] | None:
    """Return the newest AUA emulator record for *serial*, ignoring sidecar metadata."""

    root = Path(cache_dir).expanduser() / "emulator"
    if not root.is_dir():
        return None
    matches: list[tuple[float, dict[str, Any]]] = []
    for path in root.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or meta.get("serial") != serial:
            continue
        stamp = float(meta.get("started_at") or 0)
        with contextlib.suppress(OSError):
            stamp = max(stamp, path.stat().st_mtime)
        matches.append((stamp, dict(meta)))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def owner_for_serial(cache_dir: str | Path, serial: str) -> str | None:
    """Look up the owner tag that started the AUA-managed emulator, if any."""

    meta = _emulator_meta_for_serial(cache_dir, serial)
    owner = meta.get("owner") if meta else None
    return str(owner) if owner else None


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


def device_runtime_status(
    cache_dir: str | Path,
    serial: str,
    *,
    lease_registry_dir: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Describe the live lease and AUA emulator idle-stop watchdog for the dashboard."""

    from . import journal as journal_mod
    from . import leases

    current_time = time.time() if now is None else now
    lease = leases.read_lease(lease_registry_dir or cache_dir, serial)
    lease_status: dict[str, Any] = {"held": lease is not None}
    if lease is not None:
        last_lease_activity = float(lease.get("last_activity") or 0)
        ttl_s = float(lease.get("ttl_s") or 0)
        lease_status.update(
            {
                "owner": str(lease.get("owner") or "unknown"),
                "owner_pid": lease.get("owner_pid"),
                "app": lease.get("app"),
                "idle_s": max(0.0, current_time - last_lease_activity),
                "ttl_s": ttl_s,
                "expires_in_s": max(0.0, ttl_s - (current_time - last_lease_activity)),
            }
        )

    meta = _emulator_meta_for_serial(cache_dir, serial)
    watchdog: dict[str, Any] = {"managed": bool(meta and meta.get("started_by_aua"))}
    if watchdog["managed"] and meta is not None:
        timeout_s = max(0.0, float(meta.get("idle_timeout_s") or 0))
        last_activity = float(meta.get("last_activity") or meta.get("started_at") or 0)
        journal_path = journal_mod.journal_path(cache_dir, serial)
        if journal_path.is_file():
            with contextlib.suppress(OSError):
                last_activity = max(last_activity, journal_path.stat().st_mtime)
        idle_s = max(0.0, current_time - last_activity)
        enabled = timeout_s > 0
        watchdog.update(
            {
                "enabled": enabled,
                "running": enabled and _pid_is_alive(meta.get("watchdog_pid")),
                "idle_s": idle_s,
                "timeout_s": timeout_s,
                "remaining_s": max(0.0, timeout_s - idle_s) if enabled else None,
                "instance": meta.get("instance") or meta.get("avd"),
                "explicit": bool(meta.get("idle_stop_explicit")),
            }
        )
    return {"lease": lease_status, "watchdog": watchdog}


def captures_root(cache_dir: str | Path, serial: str) -> Path:
    return Path(cache_dir).expanduser() / "captures" / _safe_serial(serial)


def latest_frame(cache_dir: str | Path, serial: str) -> Path | None:
    root = captures_root(cache_dir, serial)
    if not root.is_dir():
        return None
    frames = list(root.glob("*/frames/*.jpg"))
    if not frames:
        return None
    return max(frames, key=lambda p: p.stat().st_mtime_ns)


def recent_marks(cache_dir: str | Path, serial: str, *, limit: int = 40) -> list[dict[str, Any]]:
    """Parse the newest session index.jsonl for action-stamped frames."""
    root = captures_root(cache_dir, serial)
    if not root.is_dir():
        return []
    sessions = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    marks: list[dict[str, Any]] = []
    for sess in sessions[:3]:
        idx = sess / "index.jsonl"
        if not idx.is_file():
            continue
        try:
            lines = idx.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[-200:]:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if row.get("action"):
                    marks.append(
                        {
                            "t_ms": row.get("t_ms"),
                            "action": row.get("action"),
                            "path": row.get("path"),
                            "session": sess.name,
                        }
                    )
        if marks:
            break
    return marks[-limit:]


def ensure_capture(*, serial: str, config: Any, allow_sidecar: bool = True) -> dict[str, Any]:
    """Turn capture on for *serial* without disturbing the agent's workflow.

    Prefer the warm daemon's buffer; otherwise start the host capture sidecar so a
    headless agent run becomes watchable even when no daemon was started.

    *allow_sidecar*: when False (multi-device grid), skip the single-serial sidecar —
    the dashboard falls back to adb screenshots per tile so agents are not disrupted.
    """
    cache = Path(config.cache.dir).expanduser()
    out: dict[str, Any] = {"ok": True, "serial": serial, "via": None}

    # 1) Warm daemon (agent may already be using it).
    try:
        from . import daemon as daemon_mod

        candidates = [daemon_mod.socket_path(config, serial)]
        base = os.path.expanduser(config.daemon.socket)
        if base not in candidates:
            candidates.append(base)
        for sock in candidates:
            if not Path(sock).exists():
                continue
            with daemon_mod.DaemonClient(sock, timeout=3.0) as client:
                if client.ping():
                    resp = client.call("capture_on", journal=False)
                    result = resp.get("result") if isinstance(resp, dict) else None
                    out["via"] = "daemon"
                    out["capture"] = result if isinstance(result, dict) else resp
                    out["socket"] = sock
                    return out
    except Exception as exc:  # noqa: BLE001 — fall through to sidecar
        logger.debug("daemon capture_on skipped: %s", exc)
        out["daemon_error"] = str(exc)

    # 2) Capture sidecar (independent process — ideal for sneak-peek).
    if not allow_sidecar:
        out["via"] = "screencap"
        out["hint"] = "no daemon; grid uses adb screencap per tile"
        return out
    if not getattr(config.capture, "sidecar", True):
        raise UsageError(
            "capture sidecar disabled and no warm daemon",
            hint="Set capture.sidecar: true, or `aua daemon start` then re-run dashboard.",
        )
    from . import capture_sidecar as cs

    started = cs.start(
        serial=serial,
        cache_dir=cache,
        cfg=config.capture,
        platform=str(config.device.platform),
    )
    out["via"] = "sidecar"
    out["capture"] = started
    out["socket"] = started.get("socket")
    if not started.get("ok", True) and started.get("status") == "timeout":
        out["ok"] = False
        out["hint"] = started.get("hint")
    return out


def _pick_free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise UsageError(f"no free port near {preferred}", hint="Pass --port explicitly.")


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>aua dashboard</title>
<style>
  :root {
    --bg: #0c0e12;
    --panel: #151820;
    --panel2: #1a1e28;
    --text: #e6e8ec;
    --muted: #8b929e;
    --accent: #3ddc84;
    --border: #2a303c;
    --danger: #ef6b5a;
    --warn: #e0a84a;
    --wide: 1900px;
    --tok-key: #7fd3ff;
    --tok-str: #b7e08a;
    --tok-num: #f0b26a;
    --tok-bool: #d59bf6;
    --tok-dim: #6d7686;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
  }
  header {
    display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
    padding: 0.65rem 1.1rem; border-bottom: 1px solid var(--border);
    background: var(--panel); position: sticky; top: 0; z-index: 2;
  }
  header h1 { font-size: 0.95rem; font-weight: 600; margin: 0; letter-spacing: 0.02em; }
  header a.back {
    color: var(--accent); text-decoration: none; font-size: 0.78rem; margin-right: 0.25rem;
  }
  .pill {
    font-size: 0.72rem; padding: 0.18rem 0.5rem; border-radius: 999px;
    border: 1px solid var(--border); color: var(--muted); white-space: nowrap;
  }
  .pill.ok { color: var(--accent); border-color: #2a6b4f; }
  .pill.bad { color: var(--danger); border-color: #7a3a35; }
  .layout {
    display: grid;
    /* `auto` sizes the stage column to the frame itself. A fractional column handed the
       widest part of the page to a portrait emulator - which is what a device is almost
       all of the time - and left the journal, where the evidence actually is, in the
       offcut. Rotate the device and the column widens on its own. */
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.85rem; padding: 0.85rem; max-width: var(--wide); margin: 0 auto;
    align-items: start;
  }
  @media (max-width: 980px) { .layout { grid-template-columns: minmax(0, 1fr); } }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.7rem 0.85rem; min-height: 0;
  }
  .panel h2 {
    font-size: 0.72rem; margin: 0 0 0.55rem; color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
  }
  .stage img {
    display: block; height: min(74vh, 880px); width: auto;
    min-width: 210px; max-width: min(38vw, 470px);
    object-fit: contain; background: #000; border-radius: 6px;
  }
  @media (max-width: 980px) {
    .stage img { width: 100%; height: auto; max-width: 100%; max-height: 62vh; }
  }
  .meta { font-size: 0.75rem; color: var(--muted); display: flex; gap: 0.9rem; flex-wrap: wrap; margin-top: 0.4rem; }
  /* `contain` keeps a wheel gesture inside the panel it started in: without it, hitting
     the end of the journal scrolled the whole page out from under the reader. */
  .scroll { overflow: auto; max-height: 68vh; overscroll-behavior: contain; }
  .scroll.sm { max-height: 14rem; }
  .scroll.md { max-height: 18rem; }
  .panel.journal { display: flex; flex-direction: column; min-width: 0; }
  .journal-tools, .logcat-tools {
    display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap;
    margin-bottom: 0.45rem;
  }
  .journal-tools h2, .logcat-tools h2 { margin: 0 0.35rem 0 0; }
  .journal-tools .grow, .logcat-tools .grow { flex: 1 1 8rem; min-width: 0; }
  .journal-tools .db-button, .logcat-tools .db-button {
    padding: 0.2rem 0.45rem; font-size: 0.7rem;
  }
  .journal-tools .db-input, .logcat-tools .db-input, .logcat-tools .db-select {
    padding: 0.24rem 0.45rem; font-size: 0.74rem;
  }
  .journal-tools .pill, .logcat-tools .pill {
    display: inline-flex; align-items: center; gap: 0.3rem; cursor: pointer;
  }
  #journal-wrap {
    /* Deterministic anchoring lives in preserveJournalScroll(). The browser's own scroll
       anchoring would apply a second shift on top of it, so the row being read jumped
       twice for every event that arrived. `position: relative` makes each row's
       offsetTop measurable against this viewport, which is what the anchor compares. */
    overflow-anchor: none; position: relative;
    height: min(74vh, 880px); max-height: none; overflow: auto;
    overscroll-behavior: contain; flex: 1 1 auto;
  }
  #journal-jump {
    color: #06130c; background: var(--accent); border: 1px solid var(--accent);
    border-radius: 999px; padding: 0.16rem 0.6rem; font: 600 0.7rem inherit;
    cursor: pointer; white-space: nowrap;
  }
  #journal li.filtered { display: none; }
  #journal, #marks, #fail-list, #slow { list-style: none; margin: 0; padding: 0; font-size: 0.78rem; }
  #journal li, #marks li, #fail-list li, #slow li {
    padding: 0.4rem 0.35rem; border-bottom: 1px solid var(--border);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #journal li.fail { background: rgba(239,107,90,0.08); border-left: 2px solid var(--danger); }
  #journal .t, #marks .t, #fail-list .t, #slow .t {
    color: var(--muted); margin-right: 0.45rem;
  }
  #journal .badge {
    display: inline-block; min-width: 2.4rem; font-size: 0.65rem;
    padding: 0.05rem 0.3rem; border-radius: 4px; margin-right: 0.35rem;
  }
  #journal .badge.ok { color: var(--accent); background: rgba(61,220,132,0.12); }
  #journal .badge.fail { color: var(--danger); background: rgba(239,107,90,0.15); }
  #journal .cmd { color: var(--text); font-weight: 600; }
  #journal .args { color: var(--muted); }
  #journal .dur { color: var(--warn); margin-left: 0.35rem; }
  #journal .err { color: var(--danger); display: block; margin-top: 0.15rem; font-size: 0.72rem; }
  #journal details > summary { cursor: pointer; line-height: 1.35; }
  #journal details > summary::marker { color: var(--accent); }
  #journal details[open] > summary { margin-bottom: 0.55rem; }
  #journal .exchange {
    display: grid; gap: 0.6rem; padding: 0.55rem; border: 1px solid var(--border);
    border-radius: 6px; background: #0e1118;
  }
  #journal .exchange-section h3 {
    margin: 0 0 0.3rem; color: var(--accent); font: 600 0.68rem ui-sans-serif, system-ui;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  #journal .exchange-head { display: flex; align-items: baseline; gap: 0.5rem; }
  #journal .exchange pre {
    /* One scroll surface per panel. A capped, independently scrollable pre nested inside
       the scrollable journal meant a wheel gesture went to whichever of the two the
       pointer happened to be over, and reaching the end of a payload stalled dead. */
    margin: 0; max-height: none; overflow: visible; padding: 0.55rem;
    border-radius: 5px; background: #090b10; color: #cdd2db; font: inherit;
    font-size: 0.7rem; line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere;
    user-select: text;
  }
  .copy-button {
    color: var(--muted); background: transparent; border: 1px solid var(--border);
    border-radius: 5px; padding: 0.04rem 0.36rem; font: inherit; font-size: 0.62rem;
    cursor: pointer;
  }
  .copy-button:hover { color: var(--accent); border-color: var(--accent); }
  /* --- syntax colour, shared by the payload panes and logcat --- */
  .tok-key { color: var(--tok-key); }
  .tok-str { color: var(--tok-str); }
  .tok-num { color: var(--tok-num); }
  .tok-bool { color: var(--tok-bool); }
  .tok-null { color: var(--muted); font-style: italic; }
  .tok-punc { color: var(--tok-dim); }
  #journal .detail-note { color: var(--muted); font: 0.68rem ui-sans-serif, system-ui; }
  .lower {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem; padding: 0 0.85rem 0.85rem; max-width: var(--wide); margin: 0 auto;
  }
  .lower.wide { grid-template-columns: minmax(0, 1fr); }
  @media (max-width: 1100px) { .lower { grid-template-columns: minmax(0, 1fr); } }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
  @media (max-width: 600px) { .stats-grid { grid-template-columns: 1fr; } }
  table.cmdstats {
    width: 100%; border-collapse: collapse; font-size: 0.75rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  table.cmdstats th, table.cmdstats td {
    text-align: left; padding: 0.25rem 0.35rem; border-bottom: 1px solid var(--border);
  }
  table.cmdstats th { color: var(--muted); font-weight: 500; }
  table.cmdstats .failc { color: var(--danger); }
  #map-screens, #map-routes { list-style: none; margin: 0; padding: 0; font-size: 0.78rem; }
  #map-screens li, #map-routes li {
    padding: 0.3rem 0; border-bottom: 1px solid var(--border);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #map-pkg { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.5rem; }
  .logcat-scroll {
    /* Logcat is evidence, not a footnote: a whole row of its own, and real height. One
       third of a three-column strip with `word-break: break-all` chopped identifiers
       mid-token, so nothing in it could be read or searched. */
    height: min(46vh, 560px); max-height: none; overflow: auto;
    overscroll-behavior: contain; overflow-anchor: none;
    border: 1px solid var(--border); border-radius: 6px; background: #090b10;
    padding: 0.4rem 0.5rem;
  }
  #logcat {
    margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.7rem; line-height: 1.5; color: #b8bec8; white-space: pre;
  }
  #logcat.wrap { white-space: pre-wrap; overflow-wrap: anywhere; }
  .lc-line { display: block; }
  .lc-line.err { background: rgba(239,107,90,0.09); }
  .lc-line.warn { background: rgba(224,168,74,0.07); }
  .lc-date { color: #5f6b7d; }
  .lc-time { color: var(--tok-key); }
  .lc-pid, .lc-tid { color: var(--muted); font-variant-numeric: tabular-nums; }
  .lc-tag { color: var(--tok-num); font-weight: 600; }
  .lc-pkg { color: var(--tok-str); }
  .lc-sep { color: var(--tok-dim); }
  .lc-msg { color: #c8cedb; }
  .lc-raw { color: var(--tok-dim); }
  .lc-lvl { font-weight: 700; }
  .lc-lvl-v, .lc-lvl-d { color: #7f8794; }
  .lc-lvl-i { color: var(--accent); }
  .lc-lvl-w { color: var(--warn); }
  .lc-lvl-e, .lc-lvl-f, .lc-lvl-a { color: var(--danger); }
  footer { padding: 0.4rem 1.1rem 1rem; color: var(--muted); font-size: 0.72rem; }
  .empty { color: var(--muted); font-size: 0.78rem; padding: 0.4rem 0; }
  /* --- grid mode --- */
  .grid-empty {
    padding: 1.6rem 1.1rem; color: var(--muted); font-size: 0.82rem;
    max-width: 1800px; margin: 0 auto; line-height: 1.5;
  }
  .grid-empty.bad { color: var(--danger); }
  .device-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.85rem; padding: 0.85rem; max-width: 1800px; margin: 0 auto;
  }
  .tile {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.55rem 0.65rem 0.7rem; cursor: pointer; transition: border-color 0.15s;
    text-decoration: none; color: inherit; display: block;
  }
  .tile:hover { border-color: var(--accent); }
  .tile-head {
    display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
    margin-bottom: 0.45rem; font-size: 0.72rem;
  }
  .tile-head .ser {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600; color: var(--text);
  }
  .tile img {
    width: 100%; aspect-ratio: 9/16; object-fit: contain; background: #000;
    border-radius: 6px; display: block;
  }
  .tile .tile-meta {
    margin-top: 0.4rem; font-size: 0.7rem; color: var(--muted);
    display: flex; gap: 0.6rem; flex-wrap: wrap;
  }
  .tile .tile-runtime {
    margin-top: 0.4rem; display: grid; gap: 0.18rem;
    color: var(--muted); font-size: 0.68rem; overflow-wrap: anywhere;
  }
  .tile .tile-runtime .held { color: var(--accent); }
  .tile .tile-runtime .down { color: var(--danger); }
  /* --- proxy workspace --- */
  .proxy-workspace { max-width: 1600px; margin: 0 auto 0.85rem; padding: 0 0.85rem; }
  .proxy-head { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
  .proxy-head h2 { margin-right: auto; }
  .proxy-grid {
    display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 0.85rem;
  }
  @media (max-width: 1100px) { .proxy-grid { grid-template-columns: 1fr; } }
  .flow-table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  .flow-table th {
    text-align: left; color: var(--muted); font-weight: 500; position: sticky; top: 0;
    background: var(--panel2); padding: 0.25rem 0.4rem;
  }
  .flow-table td { padding: 0.22rem 0.4rem; border-top: 1px solid var(--border); }
  .flow-table tbody tr { cursor: pointer; }
  .flow-table tbody tr:hover { background: var(--panel2); }
  .flow-table tr.touched td { color: var(--accent); }
  .flow-table .num { color: var(--muted); font-variant-numeric: tabular-nums; }
  .flow-table .meth { font-weight: 600; }
  .flow-table .upath {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere;
  }
  .rule-row { border-top: 1px solid var(--border); font-size: 0.72rem; }
  .rule-row > summary {
    display: flex; gap: 0.45rem; align-items: baseline; flex-wrap: wrap;
    padding: 0.3rem 0; cursor: pointer;
  }
  .rule-row > summary::marker { color: var(--accent); }
  .rule-row .db-button { padding: 0.16rem 0.42rem; font-size: 0.68rem; }
  .rule-row .rid {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted);
  }
  .rule-row .spec { overflow-wrap: anywhere; flex: 1 1 8rem; min-width: 0; }
  #px-rulelist { max-height: min(52vh, 580px); }
  .rule-body { padding: 0 0 0.55rem; display: grid; gap: 0.4rem; }
  .rule-body pre {
    margin: 0; padding: 0.5rem; border-radius: 5px; background: #090b10; color: #cdd2db;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.68rem;
    line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .proxy-form { display: grid; gap: 0.45rem; margin-top: 0.5rem; }
  .proxy-form .row { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: end; }
  .proxy-note { color: var(--muted); font-size: 0.72rem; line-height: 1.45; margin: 0 0 0.5rem; }
  .proxy-warn { color: var(--danger); }
  .database-workspace { max-width: 1600px; margin: 0 auto 0.85rem; padding: 0 0.85rem; }
  .db-toolbar, .db-actions {
    display: flex; gap: 0.55rem; align-items: end; flex-wrap: wrap;
  }
  .db-field { display: grid; gap: 0.2rem; color: var(--muted); font-size: 0.68rem; }
  .db-field.grow { flex: 1 1 260px; }
  .db-input, .db-select, .db-sql {
    color: var(--text); background: #0e1118; border: 1px solid var(--border);
    border-radius: 6px; padding: 0.42rem 0.52rem; font: inherit; min-width: 0;
  }
  .db-input:focus, .db-select:focus, .db-sql:focus {
    outline: 1px solid var(--accent); border-color: var(--accent);
  }
  .db-select { min-width: 190px; }
  .db-sql {
    width: 100%; min-height: 8.5rem; resize: vertical;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.4;
  }
  .db-button {
    color: var(--text); background: var(--panel2); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.43rem 0.65rem; cursor: pointer; font: inherit;
  }
  .db-button:hover:not(:disabled) { border-color: var(--accent); }
  .db-button.primary { color: #06130c; background: var(--accent); border-color: var(--accent); font-weight: 650; }
  .db-button.danger { color: #fff; background: #873a34; border-color: var(--danger); }
  .db-button:disabled { opacity: 0.45; cursor: not-allowed; }
  .db-note, .db-status { color: var(--muted); font-size: 0.74rem; line-height: 1.45; }
  .db-status { min-height: 1.2rem; margin: 0.55rem 0; }
  .db-status.ok { color: var(--accent); }
  .db-status.bad { color: var(--danger); }
  .db-grid { display: grid; grid-template-columns: 0.85fr 1.35fr; gap: 0.85rem; }
  @media (max-width: 980px) { .db-grid { grid-template-columns: 1fr; } }
  .db-subpanel { background: var(--panel2); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem; min-width: 0; }
  .db-subpanel h3 { margin: 0 0 0.5rem; font-size: 0.78rem; }
  .db-schema-object, .db-backup {
    border-bottom: 1px solid var(--border); padding: 0.45rem 0; font-size: 0.75rem;
  }
  .db-schema-object:last-child, .db-backup:last-child { border-bottom: 0; }
  .db-object-name { color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; cursor: pointer; }
  .db-columns { color: var(--muted); margin-top: 0.22rem; word-break: break-word; }
  .db-table-wrap { overflow: auto; max-height: 25rem; border: 1px solid var(--border); border-radius: 6px; }
  table.db-results { width: 100%; border-collapse: collapse; font-size: 0.72rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  table.db-results th, table.db-results td { text-align: left; padding: 0.34rem 0.45rem; border-bottom: 1px solid var(--border); white-space: pre-wrap; vertical-align: top; max-width: 32rem; overflow-wrap: anywhere; }
  table.db-results th { color: var(--accent); position: sticky; top: 0; background: #10141b; }
  .db-options { display: flex; gap: 0.7rem; align-items: center; flex-wrap: wrap; margin: 0.5rem 0; }
  .db-options label { color: var(--muted); font-size: 0.7rem; }
  dialog.db-dialog { color: var(--text); background: var(--panel); border: 1px solid var(--danger); border-radius: 10px; width: min(560px, calc(100% - 2rem)); padding: 1rem; }
  dialog.db-dialog::backdrop { background: rgba(0,0,0,0.72); }
  .db-dialog h3 { margin-top: 0; }
  .db-dialog code { color: var(--warn); }
  .hidden { display: none !important; }
</style>
</head>
<body>
<header>
  <a id="back" class="back hidden" href="/">← grid</a>
  <h1>aua dashboard</h1>
  <span id="modepill" class="pill">—</span>
  <span id="serial" class="pill">—</span>
  <span id="capture" class="pill">capture …</span>
  <span id="via" class="pill">via …</span>
  <span id="lease" class="pill">lease …</span>
  <span id="watchdog" class="pill">auto-stop …</span>
  <span id="age" class="pill">frame …</span>
  <span id="failpill" class="pill hidden">fails 0</span>
  <span id="pkg" class="pill hidden">pkg …</span>
  <span id="count" class="pill hidden">0 devices</span>
</header>

<div id="grid-view" class="hidden">
  <p id="grid-empty" class="grid-empty hidden"></p>
  <div class="device-grid" id="tiles"></div>
  <footer>
    Multi-agent grid — one tile per online emulator. Click a tile for journal / map / logcat.
    Capture uses each agent's daemon when present; otherwise adb screencap per tile.
  </footer>
</div>

<div id="detail-view" class="hidden">
<div class="layout">
  <section class="panel stage">
    <h2>Live frame</h2>
    <img id="frame" alt="device frame" src=""/>
    <div class="meta">
      <span id="session">session —</span>
      <span id="fps">poll —</span>
    </div>
  </section>
  <aside class="panel journal">
    <div class="journal-tools">
      <h2>Agent I/O journal</h2>
      <input id="journal-filter" class="db-input grow" placeholder="filter cmd, args, error…"
             autocomplete="off" spellcheck="false"/>
      <label class="pill"><input id="journal-fails-only" type="checkbox"/>fails only</label>
      <button id="journal-expand" class="db-button" type="button">expand</button>
      <button id="journal-collapse" class="db-button" type="button">collapse</button>
      <span id="journal-shown" class="db-status">—</span>
      <button id="journal-jump" class="hidden" type="button">newest ↑</button>
    </div>
    <div class="scroll" id="journal-wrap">
      <ul id="journal"><li class="empty">waiting for events…</li></ul>
    </div>
  </aside>
</div>
<div class="lower">
  <section class="panel">
    <h2>Usage stats</h2>
    <div class="stats-grid">
      <div>
        <h2>by command</h2>
        <div class="scroll sm" id="bycmd-wrap"><table class="cmdstats" id="bycmd"><thead><tr><th>cmd</th><th>ok</th><th>fail</th></tr></thead><tbody></tbody></table></div>
      </div>
      <div>
        <h2>recent failures</h2>
        <ul id="fail-list" class="scroll sm"><li class="empty">none</li></ul>
        <h2 style="margin-top:0.6rem">slow calls (≥1.5s)</h2>
        <ul id="slow" class="scroll sm"><li class="empty">none</li></ul>
      </div>
    </div>
  </section>
  <section class="panel">
    <h2>App map</h2>
    <div id="map-pkg">package —</div>
    <h2>screens</h2>
    <ul id="map-screens" class="scroll sm"><li class="empty">—</li></ul>
    <h2 style="margin-top:0.55rem">routes</h2>
    <ul id="map-routes" class="scroll sm"><li class="empty">—</li></ul>
  </section>
  <section class="panel">
    <h2>Capture marks</h2>
    <ul id="marks" class="scroll sm"><li class="empty">—</li></ul>
  </section>
</div>
<div class="lower wide">
  <section class="panel">
    <div class="logcat-tools">
      <h2>Logcat</h2>
      <input id="logcat-filter" class="db-input grow" placeholder="filter tag or message…"
             autocomplete="off" spellcheck="false"/>
      <label class="pill">min level
        <select id="logcat-level" class="db-select" style="min-width:4.5rem">
          <option value="V">V</option>
          <option value="D">D</option>
          <option value="I">I</option>
          <option value="W">W</option>
          <option value="E">E</option>
        </select>
      </label>
      <label class="pill"><input id="logcat-wrap" type="checkbox"/>wrap</label>
      <label class="pill"><input id="logcat-follow" type="checkbox" checked/>follow</label>
      <span id="logcat-shown" class="db-status">—</span>
    </div>
    <div class="logcat-scroll" id="logcat-view"><pre id="logcat">…</pre></div>
  </section>
</div>
<div class="proxy-workspace">
  <section class="panel">
    <div class="proxy-head">
      <h2>Proxy</h2>
      <span id="px-state" class="pill">proxy …</span>
      <span id="px-port" class="pill hidden">port —</span>
      <span id="px-rules" class="pill hidden">0 rules</span>
      <span id="px-touched" class="pill hidden">0 manipulated</span>
    </div>
    <p class="proxy-note" id="px-note">
      Live HTTP exchanges through the AUA proxy, which rules are armed, and which of them
      actually fired. Click a request to inspect it and arm a rule from what you just saw:
      <strong>stub</strong> answers from the rule and the server never sees it,
      <strong>rewrite</strong> lets it through and patches the real response. Rules apply to
      the <em>next</em> matching request — re-trigger it in the app to see the effect.
    </p>
    <div class="proxy-grid">
      <section class="db-subpanel">
        <div class="db-actions">
          <h3 style="margin-right:auto">Live traffic</h3>
          <span id="px-count" class="db-status">—</span>
        </div>
        <div class="scroll md">
          <table class="flow-table" id="px-flows">
            <thead><tr><th>#</th><th>method</th><th>path</th><th>status</th><th>rule</th></tr></thead>
            <tbody><tr><td colspan="5" class="empty">No traffic seen yet.</td></tr></tbody>
          </table>
        </div>
        <div id="px-detail" class="scroll sm" style="margin-top:0.5rem">
          <div class="empty">Select a request to see headers and bodies.</div>
        </div>
      </section>
      <section class="db-subpanel">
        <div class="db-actions">
          <h3 style="margin-right:auto">Armed rules</h3>
          <button id="px-clear" class="db-button">Clear all</button>
        </div>
        <div id="px-rulelist" class="scroll md"><div class="empty">No rules armed.</div></div>
        <div class="proxy-form">
          <div class="row">
            <label class="db-field" style="width:6.5rem">Action
              <select id="px-action" class="db-select">
                <option value="rewrite">rewrite</option>
                <option value="stub">stub</option>
              </select>
            </label>
            <label class="db-field" style="width:6rem">Method
              <input id="px-method" class="db-input" placeholder="GET" autocomplete="off"/>
            </label>
            <label class="db-field grow">Path
              <input id="px-path" class="db-input" placeholder="/v1/feed" autocomplete="off"/>
            </label>
          </div>
          <div class="row">
            <label class="db-field grow">Host (optional, but scopes the rule)
              <input id="px-host" class="db-input" placeholder="api.example.com" autocomplete="off"/>
            </label>
            <label class="db-field" style="width:6rem">Status
              <input id="px-status" class="db-input" placeholder="429" autocomplete="off"/>
            </label>
            <label class="db-field" style="width:6rem">Times
              <input id="px-times" class="db-input" placeholder="0" autocomplete="off"/>
            </label>
          </div>
          <label class="db-field">Body (whole replacement response; JSON or raw)
            <textarea id="px-body" class="db-sql" style="min-height:4rem" spellcheck="false"></textarea>
          </label>
          <label class="db-field">
            <span>Set JSON fields (rewrite only) — one <code>path=value</code> per line</span>
            <textarea id="px-set" class="db-sql" style="min-height:3rem" spellcheck="false" placeholder="items[0].title=&quot;patched&quot;"></textarea>
          </label>
          <div class="row">
            <button id="px-arm" class="db-button">Arm rule</button>
            <span id="px-status-line" class="db-status">Rules apply immediately, no confirmation.</span>
          </div>
        </div>
      </section>
    </div>
  </section>
</div>

<div class="database-workspace">
  <section class="panel">
    <h2>App database workspace</h2>
    <p class="db-note">
      Inspect SQLite data through the app's debuggable <code>run-as</code> sandbox.
      Read-only queries preserve the current app screen by default. Schema, backups, mutations,
      restores, and optional coherent queries briefly stop the app; keep “restart app” selected
      to relaunch it afterward.
    </p>
    <div class="db-toolbar">
      <label class="db-field grow">Package
        <input id="db-package" class="db-input" placeholder="com.example.debug" autocomplete="off"/>
      </label>
      <button id="db-refresh" class="db-button">Find databases</button>
      <label class="db-field">Database
        <select id="db-database" class="db-select"><option value="">— select —</option></select>
      </label>
      <button id="db-schema-button" class="db-button" disabled>Load schema</button>
      <button id="db-backup-button" class="db-button" disabled>Create backup</button>
    </div>
    <div class="db-options">
      <label><input id="db-restart" type="checkbox" checked/> restart app after operation</label>
      <span id="db-status" class="db-status">Waiting for a foreground debuggable package…</span>
    </div>
    <div class="db-grid">
      <div>
        <section class="db-subpanel">
          <h3>Schema</h3>
          <div id="db-schema" class="scroll md"><div class="empty">Load a database schema to browse tables and views.</div></div>
        </section>
        <section class="db-subpanel" style="margin-top:0.85rem">
          <div class="db-actions">
            <h3 style="margin-right:auto">Restore points</h3>
            <button id="db-backups-refresh" class="db-button" disabled>Refresh</button>
          </div>
          <div id="db-backups" class="scroll sm"><div class="empty">Select a database.</div></div>
        </section>
      </div>
      <section class="db-subpanel">
        <h3>SQL</h3>
        <textarea id="db-sql" class="db-sql" spellcheck="false">SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name;</textarea>
        <div class="db-options">
          <label class="db-field grow">Parameters (JSON object or array; optional)
            <input id="db-params" class="db-input" placeholder='{"id": 42}' autocomplete="off"/>
          </label>
          <label class="db-field">Row limit
            <input id="db-limit" class="db-input" type="number" min="1" max="1000" value="100" style="width:6.5rem"/>
          </label>
          <label><input id="db-coherent-query" type="checkbox"/> coherent query (stops app)</label>
        </div>
        <div class="db-actions">
          <button id="db-query-button" class="db-button primary" disabled>Run read-only query</button>
          <button id="db-execute-button" class="db-button danger" disabled>Execute mutation…</button>
        </div>
        <div id="db-result-meta" class="db-status">No query run yet.</div>
        <div id="db-results" class="db-table-wrap"><div class="empty" style="padding:0.6rem">Results appear here.</div></div>
      </section>
    </div>
  </section>
</div>
<dialog id="db-confirm-dialog" class="db-dialog">
  <h3 id="db-confirm-title">Confirm database operation</h3>
  <p id="db-confirm-message" class="db-note"></p>
  <p class="db-note">Type <code id="db-confirm-phrase"></code> to continue.</p>
  <input id="db-confirm-input" class="db-input" style="width:100%" autocomplete="off"/>
  <div class="db-actions" style="justify-content:flex-end;margin-top:0.8rem">
    <button id="db-confirm-cancel" class="db-button">Cancel</button>
    <button id="db-confirm-submit" class="db-button danger" disabled>Confirm</button>
  </div>
</dialog>
<footer>
  Live sneak-peek for headless agent runs. Frames from the capture ring buffer
  (daemon or sidecar); journal from cache/journal. Close this tab anytime — the agent keeps running.
</footer>
</div>

<script nonce="__DATABASE_TOKEN__">
/* Pure tokenisers, deliberately kept free of the DOM and in a script block of their own
   so `tests/test_dashboard_layout_and_syntax.py` can run them under node and assert on
   the tokens themselves. Colouring is worth nothing if the split is wrong: a colon
   inside a string value must not promote it to a key, and no token may lose a byte. */
function jsonTokens(text) {
  const src = text == null ? '' : String(text);
  const out = [];
  const push = (kind, value) => {
    if (!value) return;
    const last = out[out.length - 1];
    if (last && last[0] === kind) last[1] += value;
    else out.push([kind, value]);
  };
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === '"') {
      let j = i + 1;
      while (j < src.length) {
        if (src[j] === '\\\\') { j += 2; continue; }
        if (src[j] === '"') { j += 1; break; }
        j += 1;
      }
      // A key is a string whose next non-blank character is a colon. Scanning forward
      // from the closing quote is what keeps `"a:b"` a value and not a key.
      let k = j;
      while (k < src.length && (src[k] === ' ' || src[k] === '\\t')) k += 1;
      push(src[k] === ':' ? 'key' : 'str', src.slice(i, j));
      i = j;
      continue;
    }
    const digit = ch >= '0' && ch <= '9';
    const signed = ch === '-' && src[i + 1] >= '0' && src[i + 1] <= '9';
    if (digit || signed) {
      let j = i + 1;
      while (j < src.length && '0123456789.eE+-'.indexOf(src[j]) >= 0) j += 1;
      push('num', src.slice(i, j));
      i = j;
      continue;
    }
    if (src.startsWith('true', i) || src.startsWith('false', i)) {
      const word = src.startsWith('true', i) ? 'true' : 'false';
      push('bool', word);
      i += word.length;
      continue;
    }
    if (src.startsWith('null', i)) { push('nul', 'null'); i += 4; continue; }
    if ('{}[],:'.indexOf(ch) >= 0) { push('punc', ch); i += 1; continue; }
    push('plain', ch);
    i += 1;
  }
  return out;
}

const LOGCAT_THREADTIME = new RegExp(
  '^(\\\\d\\\\d-\\\\d\\\\d\\\\s+)?' +          // optional date, absent from some buffers
  '(\\\\d\\\\d:\\\\d\\\\d:\\\\d\\\\d\\\\.\\\\d+)' +   // time
  '(\\\\s+)(\\\\d+)(\\\\s+)(\\\\d+)(\\\\s+)' +  // pid, tid
  '([VDIWEFAS])(\\\\s+)' +               // level
  '([^:]*?)(\\\\s*:\\\\s?)' +            // tag, then the separator
  '([\\\\s\\\\S]*)$'                     // message
);
const LOGCAT_PACKAGE = /[a-z][a-z0-9_]*(?:\\.[a-z0-9_]+){2,}/g;

function logcatTokens(line) {
  const src = line == null ? '' : String(line);
  const m = LOGCAT_THREADTIME.exec(src);
  if (!m) return [['raw', src]];
  const out = [];
  if (m[1]) out.push(['date', m[1]]);
  out.push(['time', m[2]], ['sp', m[3]], ['pid', m[4]], ['sp', m[5]]);
  out.push(['tid', m[6]], ['sp', m[7]], ['lvl', m[8]], ['sp', m[9]]);
  out.push(['tag', m[10]], ['sep', m[11]]);
  // Package names are the thing you actually scan a log for, so they get their own
  // colour instead of disappearing into the message.
  const message = m[12];
  let at = 0;
  LOGCAT_PACKAGE.lastIndex = 0;
  let hit = LOGCAT_PACKAGE.exec(message);
  while (hit) {
    if (hit.index > at) out.push(['msg', message.slice(at, hit.index)]);
    out.push(['pkg', hit[0]]);
    at = hit.index + hit[0].length;
    hit = LOGCAT_PACKAGE.exec(message);
  }
  if (at < message.length) out.push(['msg', message.slice(at)]);
  return out;
}
</script>
<script nonce="__DATABASE_TOKEN__">
const POLL_MS = __POLL_MS__;
const MAP_MS = Math.max(POLL_MS * 4, 2000);
const BOOT_MODE = __MODE_JSON__;
const BOOT_SERIAL = __SERIAL_JSON__;
const DATABASE_TOKEN = '__DATABASE_TOKEN__';
const params = new URLSearchParams(location.search);
const focusSerial = params.get('serial') || (BOOT_MODE === 'detail' ? BOOT_SERIAL : '');
const isGrid = !focusSerial && BOOT_MODE === 'grid';

const gridView = document.getElementById('grid-view');
const detailView = document.getElementById('detail-view');
const back = document.getElementById('back');
document.getElementById('modepill').textContent = isGrid ? 'grid' : 'detail';
if (isGrid) {
  gridView.classList.remove('hidden');
  document.getElementById('count').classList.remove('hidden');
  document.getElementById('failpill').classList.add('hidden');
  document.getElementById('pkg').classList.add('hidden');
  document.getElementById('serial').classList.add('hidden');
  document.getElementById('capture').classList.add('hidden');
  document.getElementById('via').classList.add('hidden');
  document.getElementById('lease').classList.add('hidden');
  document.getElementById('watchdog').classList.add('hidden');
  document.getElementById('age').classList.add('hidden');
} else {
  detailView.classList.remove('hidden');
  document.getElementById('failpill').classList.remove('hidden');
  document.getElementById('pkg').classList.remove('hidden');
  if (BOOT_MODE === 'grid' || params.get('from') === 'grid') {
    back.classList.remove('hidden');
  }
}

const frame = document.getElementById('frame');
const journalEl = document.getElementById('journal');
const journalWrap = document.getElementById('journal-wrap');
const journalJump = document.getElementById('journal-jump');
const journalFilter = document.getElementById('journal-filter');
const journalFailsOnly = document.getElementById('journal-fails-only');
const journalShown = document.getElementById('journal-shown');
// Pinned to the newest row, or parked somewhere in the backlog? Everything the journal
// does about scrolling hangs off this one answer.
let journalFollow = true;
let journalPending = 0;
let lastSrc = '';
let sinceMs = 0;
const seenKeys = new Set();
let detailRevision = '';
const tileSrc = {};

function qSerial(extra) {
  const p = new URLSearchParams(extra || {});
  if (focusSerial) p.set('serial', focusSerial);
  const s = p.toString();
  return s ? ('?' + s) : '';
}
function fmtAge(ms) {
  if (ms == null) return 'no frame yet';
  if (ms < 1000) return Math.round(ms) + ' ms ago';
  return (ms / 1000).toFixed(1) + 's ago';
}
function fmtDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return '—';
  const value = Math.max(0, Number(seconds));
  if (value < 60) return Math.ceil(value) + 's';
  if (value < 3600) return Math.ceil(value / 60) + 'm';
  const totalMinutes = Math.ceil(value / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours + 'h' + (minutes ? ' ' + minutes + 'm' : '');
}
function leaseText(lease) {
  if (!lease || !lease.held) return 'unleased';
  return 'lease ' + (lease.owner || 'unknown');
}
function watchdogText(watchdog, lease) {
  if (!watchdog || !watchdog.managed) return 'auto-stop n/a';
  if (!watchdog.enabled) return 'auto-stop off';
  if (!watchdog.running) return 'watchdog down';
  if (watchdog.remaining_s <= 0) {
    return lease && lease.held ? 'idle limit reached · lease blocks stop' : 'auto-stop due';
  }
  const countdown = 'auto-stop in ' + fmtDuration(watchdog.remaining_s);
  return lease && lease.held ? countdown + ' · lease blocks' : countdown;
}
function fmtTime(ms) {
  if (!ms) return '';
  return new Date(ms).toLocaleTimeString();
}
function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text).then(() => true, () => false);
  }
  return Promise.resolve(false);
}
function highlightJson(pre, text) {
  const source = text === undefined ? pre.textContent : (text == null ? '' : String(text));
  pre.textContent = '';
  const frag = document.createDocumentFragment();
  jsonTokens(source).forEach(pair => {
    const kind = pair[0];
    if (kind === 'plain') {
      frag.appendChild(document.createTextNode(pair[1]));
      return;
    }
    const span = document.createElement('span');
    span.className = 'tok-' + (kind === 'nul' ? 'null' : kind);
    span.textContent = pair[1];
    frag.appendChild(span);
  });
  pre.appendChild(frag);
}
// The row the reader is actually looking at. Prepending above it, or an expanded payload
// growing above it, both move it by a measurable amount - and that amount is exactly the
// correction the viewport needs.
function journalAnchor() {
  const top = journalWrap.scrollTop;
  const rows = journalEl.children;
  for (let i = 0; i < rows.length; i += 1) {
    if (rows[i].offsetTop + rows[i].offsetHeight > top + 1) return rows[i];
  }
  return null;
}
function preserveJournalScroll(mutate) {
  const anchor = journalAnchor();
  const anchorTop = anchor ? anchor.offsetTop : 0;
  const top = journalWrap.scrollTop;
  const result = mutate();
  if (journalFollow) {
    journalWrap.scrollTop = 0;
    return result;
  }
  if (!anchor || !anchor.isConnected) return result;
  const shift = anchor.offsetTop - anchorTop;
  if (shift) journalWrap.scrollTop = top + shift;
  return result;
}
function updateJournalFollow() {
  if (journalFollow || !journalPending) {
    journalJump.classList.add('hidden');
  } else {
    journalJump.classList.remove('hidden');
    journalJump.textContent = journalPending + ' new \u2191';
  }
}
function applyJournalFilter() {
  const needle = journalFilter.value.trim().toLowerCase();
  const failsOnly = journalFailsOnly.checked;
  const rows = journalEl.children;
  let shown = 0;
  let total = 0;
  for (let i = 0; i < rows.length; i += 1) {
    const li = rows[i];
    if (li.classList.contains('empty')) continue;
    total += 1;
    const hidden = (failsOnly && !li.classList.contains('fail')) ||
      (needle !== '' && (li.dataset.search || '').indexOf(needle) < 0);
    li.classList.toggle('filtered', hidden);
    if (!hidden) shown += 1;
  }
  journalShown.textContent = total === 0
    ? '—'
    : (shown === total ? (total + ' events') : (shown + ' / ' + total + ' events'));
}
function argsSummary(args) {
  if (!args || typeof args !== 'object') return '';
  const parts = [];
  for (const [k, v] of Object.entries(args)) {
    if (v == null || v === '') continue;
    let s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    if (s.length > 48) s = s.slice(0, 48) + '…';
    parts.push(k + '=' + s);
    if (parts.length >= 4) break;
  }
  return parts.join(' ');
}
function errText(err) {
  if (!err) return '';
  if (typeof err === 'string') return err;
  return err.message || err.detail || JSON.stringify(err);
}
function addEventText(parent, className, value) {
  const span = document.createElement('span');
  span.className = className;
  span.textContent = value;
  parent.appendChild(span);
  return span;
}
function storedExchange(e) {
  const response = {ok: e.ok !== false};
  if (Object.prototype.hasOwnProperty.call(e, 'result')) response.result = e.result;
  if (Object.prototype.hasOwnProperty.call(e, 'error')) response.error = e.error;
  return {
    request: {cmd: e.cmd || '?', args: e.args || {}},
    response: response,
  };
}
function prettyJson(value) {
  try { return JSON.stringify(value, null, 2); }
  catch (error) { return String(value); }
}
function payloadHead(title, text) {
  const head = document.createElement('div');
  head.className = 'exchange-head';
  const heading = document.createElement('h3');
  heading.textContent = title;
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'copy-button';
  copy.textContent = 'copy';
  copy.addEventListener('click', event => {
    event.preventDefault();
    event.stopPropagation();
    copyText(text).then(ok => { copy.textContent = ok ? 'copied' : 'copy failed'; });
  });
  head.append(heading, copy);
  return head;
}
function renderExchange(panel, exchange, note) {
  const requestText = prettyJson(exchange.request);
  const responseText = prettyJson(exchange.response);
  // Every poll re-renders open payloads. Re-rendering byte-identical content still
  // resized the row and dragged the viewport, so an unchanged exchange is left alone.
  const signature = JSON.stringify([requestText, responseText, note || '']);
  if (panel.dataset.signature === signature) return;
  preserveJournalScroll(() => {
    panel.dataset.signature = signature;
    panel.textContent = '';
    const requestSection = document.createElement('section');
    requestSection.className = 'exchange-section';
    const requestPayload = document.createElement('pre');
    requestPayload.className = 'request-payload';
    requestPayload.textContent = requestText;
    highlightJson(requestPayload);
    requestSection.append(payloadHead('Agent request', requestText), requestPayload);
    const responseSection = document.createElement('section');
    responseSection.className = 'exchange-section';
    const responsePayload = document.createElement('pre');
    responsePayload.className = 'response-payload';
    responsePayload.textContent = responseText;
    highlightJson(responsePayload);
    responseSection.append(payloadHead('AUA response', responseText), responsePayload);
    panel.append(requestSection, responseSection);
    if (note) {
      const noteElement = document.createElement('div');
      noteElement.className = 'detail-note';
      noteElement.textContent = note;
      panel.appendChild(noteElement);
    }
  });
}
async function loadEventExchange(e, panel, showLoading = true) {
  if (!e.detail_id) {
    renderExchange(
      panel,
      storedExchange(e),
      'This older event predates full detail capture; showing the compact journaled payload.'
    );
    return;
  }
  if (showLoading) {
    panel.dataset.signature = '';
    panel.textContent = 'Loading full request and response…';
  }
  try {
    const response = await fetch('/api/event' + qSerial({detail_id: e.detail_id}), {
      cache: 'no-store',
      headers: {'X-AUA-Dashboard-Token': DATABASE_TOKEN},
    });
    const payload = await response.json();
    if (!response.ok || payload.ok === false || !payload.detail) {
      throw new Error(payload.error || 'journal detail unavailable');
    }
    renderExchange(
      panel,
      payload.detail,
      'Credentials, typed input, SQL, parameters, microphone speech, and audio paths stay redacted.'
    );
  } catch (error) {
    renderExchange(
      panel,
      storedExchange(e),
      'Full detail could not be loaded; showing the compact journaled payload.'
    );
  }
}
function loadEventDetails(details, showLoading = true) {
  const exchange = details.querySelector('.exchange');
  if (!exchange || !details.journalEvent) return Promise.resolve();
  if (details.dataset.loading === 'true') {
    details.dataset.refreshPending = 'true';
    return Promise.resolve();
  }
  details.dataset.loading = 'true';
  return loadEventExchange(details.journalEvent, exchange, showLoading).finally(() => {
    details.dataset.loading = 'false';
    const refreshPending = details.dataset.refreshPending === 'true';
    details.dataset.refreshPending = 'false';
    if (refreshPending && details.open) return loadEventDetails(details, false);
  });
}
function eventKey(e) {
  return e.detail_id ||
    ((e.ts_ms || 0) + ':' + (e.cmd || '') + ':' + (e.source || '') + ':' + (e.pid || ''));
}
function buildEventRow(e) {
  const li = document.createElement('li');
  const ok = e.ok !== false;
  li.className = ok ? '' : 'fail';
  const details = document.createElement('details');
  if (e.detail_id) details.dataset.detailId = e.detail_id;
  const summary = document.createElement('summary');
  addEventText(summary, 't', fmtTime(e.ts_ms));
  addEventText(summary, 'badge ' + (ok ? 'ok' : 'fail'), ok ? 'ok' : 'fail');
  addEventText(summary, 'cmd', e.cmd || '?');
  summary.appendChild(document.createTextNode(' '));
  const args = argsSummary(e.args);
  addEventText(summary, 'args', args);
  if (e.duration_ms != null) addEventText(summary, 'dur', e.duration_ms + 'ms');
  const failure = ok ? '' : errText(e.error);
  if (!ok && e.error) addEventText(summary, 'err', failure);
  const exchange = document.createElement('div');
  exchange.className = 'exchange';
  exchange.textContent = 'Expand to load the full request and response.';
  details.journalEvent = e;
  // The `toggle` event fires after the row has already grown or shrunk, so anchoring from
  // there is a frame too late. Owning the click means the measurement brackets the actual
  // layout change - which matters most on collapse, where the row above the reader
  // disappears and nothing else would pull the viewport back.
  summary.addEventListener('click', event => {
    if (event.target.closest('button')) return;
    event.preventDefault();
    preserveJournalScroll(() => { details.open = !details.open; });
  });
  details.addEventListener('toggle', () => {
    if (details.open) preserveJournalScroll(() => loadEventDetails(details));
  });
  details.append(summary, exchange);
  li.appendChild(details);
  li.dataset.search =
    [e.cmd || '', args, failure, ok ? 'ok' : 'fail'].join(' ').toLowerCase();
  return li;
}
// Batched on purpose. Inserting one row at a time meant one reflow and one scroll
// correction per event, and a burst of ten events walked the viewport ten times.
function prependEvents(events) {
  const fresh = [];
  (events || []).forEach(e => {
    const key = eventKey(e);
    if (seenKeys.has(key)) return;
    seenKeys.add(key);
    fresh.push(e);
  });
  if (!fresh.length) return;
  const empty = journalEl.querySelector('.empty');
  if (empty) empty.remove();
  preserveJournalScroll(() => {
    fresh.forEach(e => journalEl.insertBefore(buildEventRow(e), journalEl.firstChild));
    while (journalEl.children.length > 300) journalEl.removeChild(journalEl.lastChild);
    applyJournalFilter();
  });
  if (!journalFollow) journalPending += fresh.length;
  updateJournalFollow();
}

journalWrap.addEventListener('scroll', () => {
  const atTop = journalWrap.scrollTop <= 2;
  if (atTop === journalFollow) return;
  journalFollow = atTop;
  if (atTop) journalPending = 0;
  updateJournalFollow();
});
journalJump.addEventListener('click', () => {
  journalFollow = true;
  journalPending = 0;
  journalWrap.scrollTop = 0;
  updateJournalFollow();
});
journalFilter.addEventListener('input', () => {
  applyJournalFilter();
  journalWrap.scrollTop = 0;
});
journalFailsOnly.addEventListener('change', () => {
  applyJournalFilter();
  journalWrap.scrollTop = 0;
});
document.getElementById('journal-expand').addEventListener('click', () => {
  // Capped: every expansion is a fetch, and 300 rows would be 300 of them.
  const rows = journalEl.querySelectorAll('li:not(.filtered) > details:not([open])');
  for (let i = 0; i < rows.length && i < 20; i += 1) rows[i].open = true;
});
document.getElementById('journal-collapse').addEventListener('click', () => {
  journalEl.querySelectorAll('details[open]').forEach(d => { d.open = false; });
  journalFollow = true;
  journalPending = 0;
  journalWrap.scrollTop = 0;
  updateJournalFollow();
});

async function refreshOpenEventExchanges() {
  const loads = [];
  journalEl.querySelectorAll('details[open][data-detail-id]').forEach(details => {
    loads.push(loadEventDetails(details, false));
  });
  await Promise.all(loads);
}

function ensureTile(d) {
  const tiles = document.getElementById('tiles');
  let a = document.getElementById('tile-' + d.serial);
  if (!a) {
    a = document.createElement('a');
    a.className = 'tile';
    a.id = 'tile-' + d.serial;
    a.href = '/?serial=' + encodeURIComponent(d.serial) + '&from=grid';
    a.innerHTML =
      '<div class="tile-head">' +
        '<span class="ser"></span>' +
        '<span class="pill owner"></span>' +
        '<span class="pill cap"></span>' +
      '</div>' +
      '<img alt="frame" src=""/>' +
      '<div class="tile-meta">' +
        '<span class="age"></span>' +
        '<span class="pkg"></span>' +
      '</div>' +
      '<div class="tile-runtime">' +
        '<span class="lease"></span>' +
        '<span class="watchdog"></span>' +
      '</div>';
    tiles.appendChild(a);
  }
  a.querySelector('.ser').textContent = d.serial;
  const own = a.querySelector('.owner');
  if (d.owner) { own.textContent = 'started ' + d.owner; own.classList.remove('hidden'); }
  else { own.textContent = ''; own.classList.add('hidden'); }
  const cap = a.querySelector('.cap');
  cap.textContent = d.capture_running ? 'live' : (d.has_frame ? 'frame' : 'idle');
  cap.className = 'pill cap ' + (d.capture_running ? 'ok' : '');
  a.querySelector('.age').textContent = fmtAge(d.frame_age_ms);
  a.querySelector('.pkg').textContent = d.package ? ('pkg ' + d.package) : '';
  const lease = d.lease || {};
  const leaseEl = a.querySelector('.tile-runtime .lease');
  leaseEl.textContent = leaseText(lease);
  leaseEl.className = 'lease' + (lease.held ? ' held' : '');
  const watchdog = d.watchdog || {};
  const watchdogEl = a.querySelector('.tile-runtime .watchdog');
  watchdogEl.textContent = watchdogText(watchdog, lease);
  watchdogEl.className = 'watchdog' + (
    watchdog.managed && watchdog.enabled && !watchdog.running ? ' down' : ''
  );
  const img = a.querySelector('img');
  // The token changes exactly when the served bytes do, so this both refreshes the
  // tile and skips the transfer while the screen is unchanged. Do not gate it on
  // has_frame: with no capture running the bytes come from a live screencap.
  const token = d.frame_token || '';
  if (tileSrc[d.serial] !== token) {
    img.src = '/api/frame.jpg?serial=' + encodeURIComponent(d.serial) + '&t=' + encodeURIComponent(token);
    tileSrc[d.serial] = token;
  }
  return a;
}

async function tickGrid() {
  try {
    const r = await fetch('/api/devices', {cache: 'no-store'});
    const d = await r.json();
    const list = d.devices || [];
    document.getElementById('count').textContent = list.length + ' device' + (list.length === 1 ? '' : 's');
    const emptyEl = document.getElementById('grid-empty');
    emptyEl.className = 'grid-empty' + (list.length ? ' hidden' : (d.discovery_error ? ' bad' : ''));
    emptyEl.textContent = list.length
      ? ''
      : (d.discovery_error
          ? ('Cannot list devices: ' + d.discovery_error)
          : 'No device attached yet. Start one with `aua emulator start` \u2014 it appears here on its own.');
    const seen = new Set();
    list.forEach(dev => {
      seen.add(dev.serial);
      ensureTile(dev);
    });
    document.querySelectorAll('.tile').forEach(el => {
      const ser = el.id.replace(/^tile-/, '');
      if (!seen.has(ser)) el.remove();
    });
  } catch (e) {
    document.getElementById('count').textContent = 'error';
    document.getElementById('count').className = 'pill bad';
  }
}

let pxSelected = null;

function pxRuleSpec(rule) {
  const m = rule.match || rule.request || {};
  const where = [m.method || '*', m.path || '*'].join(' ') + (m.host ? (' @' + m.host) : '');
  if (rule.action === 'stub') {
    const r = rule.response || {};
    return where + '  \u2192 ' + (r.status != null ? r.status : 200);
  }
  const w = rule.rewrite || {};
  const bits = [];
  if (w.status != null) bits.push('status ' + w.status);
  if (w.headers) bits.push('headers ' + Object.keys(w.headers).join(','));
  if (w.set_json) bits.push('set ' + Object.keys(w.set_json).join(','));
  if (w.delete_json) bits.push('delete ' + w.delete_json.join(','));
  if (w.replace) bits.push(w.replace.length + ' replacement(s)');
  if (w.body !== undefined) bits.push('body');
  return where + '  \u2192 ' + (bits.join(' \u00b7 ') || 'unchanged');
}

function pxLoadRuleIntoForm(rule) {
  const match = rule.match || rule.request || {};
  const spec = (rule.action === 'stub' ? rule.response : rule.rewrite) || {};
  document.getElementById('px-action').value = rule.action === 'stub' ? 'stub' : 'rewrite';
  document.getElementById('px-method').value = match.method || '';
  document.getElementById('px-path').value = match.path || '';
  document.getElementById('px-host').value = match.host || '';
  document.getElementById('px-status').value =
    spec.status != null ? String(spec.status) : '';
  document.getElementById('px-times').value = rule.times ? String(rule.times) : '';
  const body = spec.body;
  document.getElementById('px-body').value =
    body == null ? '' : (typeof body === 'string' ? body : prettyJson(body));
  const setJson = spec.set_json || {};
  document.getElementById('px-set').value = Object.keys(setJson)
    .map(key => key + '=' + JSON.stringify(setJson[key])).join('\\n');
  document.getElementById('px-status-line').textContent =
    'Loaded ' + (rule.id || 'rule') + ' into the form. Arming adds a new rule.';
  document.getElementById('px-path').focus();
}

function pxRenderRules(rules) {
  const host = document.getElementById('px-rulelist');
  host.textContent = '';
  if (!rules.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No rules armed.';
    host.appendChild(empty);
    return;
  }
  rules.forEach(rule => {
    // A rule's address is not its behaviour. The summary line says which request it
    // catches; only the full spec says what the app will actually receive, so the row
    // opens onto it instead of leaving the body unknowable from the page.
    const row = document.createElement('details');
    row.className = 'rule-row';
    const summary = document.createElement('summary');
    const id = document.createElement('span');
    id.className = 'rid';
    id.textContent = rule.id || '?';
    const act = document.createElement('span');
    act.className = 'pill';
    act.textContent = rule.action || 'rule';
    const spec = document.createElement('span');
    spec.className = 'spec';
    spec.textContent = pxRuleSpec(rule);
    const fired = document.createElement('span');
    fired.className = 'pill' + (rule.fired ? '' : ' ok');
    fired.textContent = rule.fired ? ('fired ' + rule.fired + '\u00d7') : 'armed';
    const load = document.createElement('button');
    load.className = 'db-button px-rule-load';
    load.type = 'button';
    load.textContent = 'edit';
    load.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      pxLoadRuleIntoForm(rule);
    });
    const rm = document.createElement('button');
    rm.className = 'db-button';
    rm.type = 'button';
    rm.textContent = 'remove';
    rm.addEventListener('click', event => {
      // Both buttons live in the summary, where a bare click would also toggle the row.
      event.preventDefault();
      event.stopPropagation();
      pxPost('rm', {id: rule.id});
    });
    summary.append(id, act, spec, fired, load, rm);
    const body = document.createElement('div');
    body.className = 'rule-body';
    const note = document.createElement('div');
    note.className = 'detail-note';
    note.textContent = rule.action === 'stub'
      ? 'Answered from this rule; the server never sees the request.'
      : 'The real response is fetched, then patched with the changes below.';
    const payload = document.createElement('pre');
    payload.textContent = prettyJson(rule);
    highlightJson(payload);
    body.append(note, payload);
    row.append(summary, body);
    host.appendChild(row);
  });
}

function pxRenderFlows(flows) {
  const body = document.querySelector('#px-flows tbody');
  body.innerHTML = '';
  if (!flows.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No traffic seen yet.</td></tr>';
    return;
  }
  flows.slice().reverse().forEach(f => {
    const tr = document.createElement('tr');
    if (f.action) tr.className = 'touched';
    const cells = [
      ['num', f.n],
      ['meth', f.method || ''],
      ['upath', (f.host ? f.host : '') + (f.path || '')],
      ['num', f.status || ''],
      ['', f.action ? (f.action + ' ' + (f.rule || '')) : ''],
    ];
    cells.forEach(([cls, text]) => {
      const td = document.createElement('td');
      td.className = cls;
      td.textContent = String(text);
      tr.appendChild(td);
    });
    tr.addEventListener('click', () => pxSelectFlow(f));
    body.appendChild(tr);
  });
}

async function pxSelectFlow(f) {
  pxSelected = f;
  document.getElementById('px-method').value = f.method || '';
  document.getElementById('px-path').value = f.path || '';
  document.getElementById('px-host').value = f.host || '';
  const box = document.getElementById('px-detail');
  box.innerHTML = '<div class="empty">Loading\u2026</div>';
  try {
    const r = await fetch('/api/proxy/flow' + qSerial({n: f.n, ts: f.ts || ''}), {cache: 'no-store'});
    const d = await r.json();
    if (!d.ok) {
      box.innerHTML = '';
      const p = document.createElement('div');
      p.className = 'empty';
      p.textContent = (d.error && d.error.message) || 'no detail';
      box.appendChild(p);
      return;
    }
    const pre = document.createElement('pre');
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.overflowWrap = 'anywhere';
    pre.style.fontSize = '0.7rem';
    pre.textContent = JSON.stringify(d.flow, null, 2);
    box.innerHTML = '';
    box.appendChild(pre);
  } catch (e) {
    box.innerHTML = '<div class="empty">detail request failed</div>';
  }
}

function pxParseSet(text) {
  const out = {};
  (text || '').split('\\n').forEach(line => {
    const t = line.trim();
    if (!t) return;
    const i = t.indexOf('=');
    if (i < 1) return;
    const key = t.slice(0, i).trim();
    const raw = t.slice(i + 1).trim();
    try { out[key] = JSON.parse(raw); } catch (e) { out[key] = raw; }
  });
  return Object.keys(out).length ? out : null;
}

async function pxPost(action, payload) {
  const line = document.getElementById('px-status-line');
  line.className = 'db-status';
  line.textContent = 'Working\u2026';
  try {
    const r = await fetch('/api/proxy/' + action, {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-AUA-Dashboard-Token': DATABASE_TOKEN},
      body: JSON.stringify(Object.assign({serial: focusSerial}, payload || {})),
    });
    const d = await r.json();
    if (!d.ok) {
      line.className = 'db-status proxy-warn';
      line.textContent = (d.error && d.error.message) || 'request failed';
    } else {
      line.textContent = d.warning ? ('Done \u2014 ' + d.warning) : 'Done.';
      if (d.warning) line.className = 'db-status proxy-warn';
    }
    tickProxy();
  } catch (e) {
    line.className = 'db-status proxy-warn';
    line.textContent = 'request failed';
  }
}

function pxArm() {
  const action = document.getElementById('px-action').value;
  const statusRaw = document.getElementById('px-status').value.trim();
  const payload = {
    method: document.getElementById('px-method').value.trim() || '*',
    path: document.getElementById('px-path').value.trim(),
    host: document.getElementById('px-host').value.trim() || null,
    body: document.getElementById('px-body').value.trim() || null,
    times: parseInt(document.getElementById('px-times').value, 10) || 0,
  };
  if (statusRaw) payload.status = parseInt(statusRaw, 10);
  if (action === 'rewrite') payload.set_json = pxParseSet(document.getElementById('px-set').value);
  if (!payload.path) {
    const line = document.getElementById('px-status-line');
    line.className = 'db-status proxy-warn';
    line.textContent = 'A path is required.';
    return;
  }
  pxPost(action, payload);
}

async function tickProxy() {
  if (isGrid) return;
  try {
    const r = await fetch('/api/proxy' + qSerial({limit: 300}), {cache: 'no-store'});
    const d = await r.json();
    const state = document.getElementById('px-state');
    if (!d.supported) {
      state.textContent = 'proxy unsupported';
      state.className = 'pill';
      return;
    }
    state.textContent = d.intercepting ? 'intercepting' : (d.on ? 'on' : (d.state || 'off'));
    state.className = 'pill' + (d.intercepting ? ' ok' : (d.on ? '' : ' bad'));
    const port = document.getElementById('px-port');
    if (d.port) { port.textContent = 'port ' + d.port; port.classList.remove('hidden'); }
    else port.classList.add('hidden');
    const rules = d.rules || [];
    const rp = document.getElementById('px-rules');
    rp.textContent = rules.length + ' rule' + (rules.length === 1 ? '' : 's');
    rp.classList.remove('hidden');
    const tp = document.getElementById('px-touched');
    tp.textContent = (d.manipulated || 0) + ' manipulated';
    tp.className = 'pill' + (d.manipulated ? ' ok' : '');
    tp.classList.remove('hidden');
    const live = (d.flows || []).filter(f => f.live).length;
    document.getElementById('px-count').textContent =
      (d.flow_count || 0) + ' exchange(s) logged' + (live ? (', ' + live + ' while watching') : '');
    pxRenderRules(rules);
    pxRenderFlows(d.flows || []);
  } catch (e) {
    const state = document.getElementById('px-state');
    state.textContent = 'proxy error';
    state.className = 'pill bad';
  }
}

async function tickStatus() {
  try {
    const r = await fetch('/api/status' + qSerial(), {cache: 'no-store'});
    const s = await r.json();
    document.getElementById('serial').textContent = s.serial || '—';
    const cap = document.getElementById('capture');
    cap.textContent = s.capture_running ? 'capture on' : 'capture off';
    cap.className = 'pill ' + (s.capture_running ? 'ok' : 'bad');
    document.getElementById('via').textContent = 'via ' + (s.via || '—');
    const lease = s.lease || {};
    const leasePill = document.getElementById('lease');
    leasePill.textContent = leaseText(lease);
    leasePill.className = 'pill' + (lease.held ? ' ok' : '');
    const watchdog = s.watchdog || {};
    const watchdogPill = document.getElementById('watchdog');
    watchdogPill.textContent = watchdogText(watchdog, lease);
    watchdogPill.className = 'pill' + (
      watchdog.managed && watchdog.enabled && !watchdog.running ? ' bad' : ''
    );
    document.getElementById('age').textContent = fmtAge(s.frame_age_ms);
    const fc = (s.stats && s.stats.fail_count) || 0;
    const fp = document.getElementById('failpill');
    fp.textContent = 'fails ' + fc;
    fp.className = 'pill ' + (fc ? 'bad' : 'ok');
    document.getElementById('pkg').textContent = 'pkg ' + (s.package || '—');
    if (s.package && !dbPackage.value && !databaseBootstrapped) {
      dbPackage.value = s.package;
      databaseBootstrapped = true;
      loadDatabases();
    }
    document.getElementById('session').textContent = 'session ' + (s.session_id || '—');
    document.getElementById('fps').textContent = 'poll ' + (POLL_MS / 1000) + 's';

    const marks = document.getElementById('marks');
    marks.innerHTML = '';
    const ml = (s.marks || []).slice().reverse();
    if (!ml.length) {
      marks.innerHTML = '<li class="empty">no marks yet</li>';
    } else {
      ml.forEach(m => {
        const li = document.createElement('li');
        li.innerHTML = '<span class="t">' + fmtTime(m.t_ms) + '</span>' + (m.action || '');
        marks.appendChild(li);
      });
    }

    const st = s.stats || {};
    const tbody = document.querySelector('#bycmd tbody');
    tbody.innerHTML = '';
    const by = st.by_cmd || {};
    const keys = Object.keys(by).sort();
    if (!keys.length) {
      tbody.innerHTML = '<tr><td colspan="3" class="empty">no data</td></tr>';
    } else {
      keys.forEach(cmd => {
        const row = by[cmd] || {};
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + cmd + '</td><td>' + (row.ok || 0) + '</td><td class="failc">' + (row.fail || 0) + '</td>';
        tbody.appendChild(tr);
      });
    }
    const failsUl = document.getElementById('fail-list');
    failsUl.innerHTML = '';
    const fl = (st.failures || []).slice().reverse();
    if (!fl.length) failsUl.innerHTML = '<li class="empty">none</li>';
    else fl.forEach(f => {
      const li = document.createElement('li');
      li.innerHTML = '<span class="t">' + fmtTime(f.ts_ms) + '</span>' + (f.cmd || '') + ' — ' + (f.error || '');
      failsUl.appendChild(li);
    });
    const slowUl = document.getElementById('slow');
    slowUl.innerHTML = '';
    const sl = (st.slow || []).slice().reverse();
    if (!sl.length) slowUl.innerHTML = '<li class="empty">none</li>';
    else sl.forEach(f => {
      const li = document.createElement('li');
      li.innerHTML = '<span class="t">' + fmtTime(f.ts_ms) + '</span>' + (f.cmd || '') + ' ' + (f.duration_ms || '') + 'ms';
      slowUl.appendChild(li);
    });

    const frameToken = s.frame_token || '';
    if (frameToken !== lastSrc) {
      frame.src = '/api/frame.jpg' + qSerial({t: frameToken});
      lastSrc = frameToken;
    }
  } catch (e) {
    document.getElementById('capture').textContent = 'error';
    document.getElementById('capture').className = 'pill bad';
  }
}

async function tickEvents() {
  try {
    const base = sinceMs ? ('since_ms=' + sinceMs + '&limit=150') : 'limit=150';
    const r = await fetch('/api/events' + qSerial() + (qSerial() ? '&' : '?') + base, {cache: 'no-store'});
    const d = await r.json();
    const nextDetailRevision = String(d.detail_revision || '');
    const detailChanged = Boolean(
      detailRevision && nextDetailRevision && nextDetailRevision !== detailRevision
    );
    detailRevision = nextDetailRevision;
    const evs = d.events || [];
    evs.forEach(e => {
      if (e.ts_ms && e.ts_ms >= sinceMs) sinceMs = e.ts_ms + 1;
    });
    prependEvents(evs);
    if (detailChanged) await refreshOpenEventExchanges();
  } catch (e) {}
}

async function tickMap() {
  try {
    const r = await fetch('/api/map' + qSerial(), {cache: 'no-store'});
    const d = await r.json();
    document.getElementById('map-pkg').textContent =
      'package ' + (d.package || '—') + (d.known ? ' (known)' : ' (no map)');
    const sc = document.getElementById('map-screens');
    sc.innerHTML = '';
    const screens = d.screens || [];
    if (!screens.length) sc.innerHTML = '<li class="empty">no screens</li>';
    else screens.forEach(s => {
      const li = document.createElement('li');
      const name = typeof s === 'string' ? s : (s.name || '?');
      const extra = typeof s === 'object' ? (' visits=' + (s.visit_count || 0) + (s.stale ? ' stale' : '')) : '';
      li.textContent = name + extra;
      sc.appendChild(li);
    });
    const rt = document.getElementById('map-routes');
    rt.innerHTML = '';
    const routes = d.routes || [];
    if (!routes.length) rt.innerHTML = '<li class="empty">no routes</li>';
    else routes.forEach(e => {
      const li = document.createElement('li');
      if (typeof e === 'string') li.textContent = e;
      else li.textContent = (e.from || e.from_screen || '?') + ' → ' + (e.to || e.to_screen || '?') +
        (e.action ? '  [' + e.action + ']' : '');
      rt.appendChild(li);
    });
  } catch (e) {}
}

const LOGCAT_ORDER = {V: 0, D: 1, I: 2, W: 3, E: 4, F: 5, A: 5, S: 5};
const logcatEl = document.getElementById('logcat');
const logcatView = document.getElementById('logcat-view');
const logcatFilter = document.getElementById('logcat-filter');
const logcatLevel = document.getElementById('logcat-level');
const logcatFollow = document.getElementById('logcat-follow');
let logcatLines = [];

function renderLogcat(lines) {
  const needle = logcatFilter.value.trim().toLowerCase();
  const floor = LOGCAT_ORDER[logcatLevel.value] || 0;
  const kept = [];
  lines.forEach(line => {
    const tokens = logcatTokens(line);
    let level = '';
    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i][0] === 'lvl') { level = tokens[i][1]; break; }
    }
    if (level && (LOGCAT_ORDER[level] || 0) < floor) return;
    if (needle !== '' && line.toLowerCase().indexOf(needle) < 0) return;
    kept.push([tokens, level]);
  });
  document.getElementById('logcat-shown').textContent =
    kept.length + ' / ' + lines.length + ' lines';
  const signature = JSON.stringify(lines) + '\\u0000' + needle + '\\u0000' + floor;
  if (logcatEl.dataset.signature === signature) return;
  // Bottom-anchored, like a terminal: hold position unless the reader is at the end or
  // has asked to follow. Re-rendering the whole pre used to slam it back to the top.
  const atBottom =
    logcatView.scrollHeight - logcatView.scrollTop - logcatView.clientHeight < 12;
  logcatEl.dataset.signature = signature;
  logcatEl.textContent = '';
  if (!kept.length) {
    logcatEl.textContent = lines.length ? '(nothing matches the filter)' : '(empty)';
    return;
  }
  const frag = document.createDocumentFragment();
  kept.forEach(entry => {
    const tokens = entry[0];
    const level = entry[1];
    const row = document.createElement('span');
    row.className = 'lc-line' +
      (level === 'E' || level === 'F' || level === 'A' ? ' err' : '') +
      (level === 'W' ? ' warn' : '');
    tokens.forEach(pair => {
      if (pair[0] === 'sp') {
        row.appendChild(document.createTextNode(pair[1]));
        return;
      }
      const span = document.createElement('span');
      span.className = pair[0] === 'lvl'
        ? ('lc-lvl lc-lvl-' + String(pair[1]).toLowerCase())
        : ('lc-' + pair[0]);
      span.textContent = pair[1];
      row.appendChild(span);
    });
    frag.appendChild(row);
  });
  logcatEl.appendChild(frag);
  if (logcatFollow.checked || atBottom) logcatView.scrollTop = logcatView.scrollHeight;
}

logcatFilter.addEventListener('input', () => renderLogcat(logcatLines));
logcatLevel.addEventListener('change', () => renderLogcat(logcatLines));
logcatFollow.addEventListener('change', () => {
  if (logcatFollow.checked) logcatView.scrollTop = logcatView.scrollHeight;
});
document.getElementById('logcat-wrap').addEventListener('change', event => {
  logcatEl.classList.toggle('wrap', event.target.checked);
});

async function tickLogcat() {
  try {
    const r = await fetch('/api/logcat' + qSerial({lines: 400}), {cache: 'no-store'});
    const d = await r.json();
    logcatLines = d.lines || [];
    renderLogcat(logcatLines);
  } catch (e) {}
}

const dbPackage = document.getElementById('db-package');
const dbDatabase = document.getElementById('db-database');
const dbStatus = document.getElementById('db-status');
const dbSchema = document.getElementById('db-schema');
const dbBackups = document.getElementById('db-backups');
const dbResults = document.getElementById('db-results');
const dbResultMeta = document.getElementById('db-result-meta');
const dbSql = document.getElementById('db-sql');
const dbParams = document.getElementById('db-params');
const dbLimit = document.getElementById('db-limit');
const dbRestart = document.getElementById('db-restart');
const dbCoherentQuery = document.getElementById('db-coherent-query');
const dbConfirmDialog = document.getElementById('db-confirm-dialog');
const dbConfirmInput = document.getElementById('db-confirm-input');
const dbConfirmSubmit = document.getElementById('db-confirm-submit');
let databaseBootstrapped = false;
let databaseBusy = false;
let pendingDatabaseConfirmation = null;

function databaseSelection() {
  return {
    serial: focusSerial,
    package: dbPackage.value.trim(),
    database: dbDatabase.value,
    restart: dbRestart.checked,
  };
}

function databaseError(data, fallback) {
  const err = data && data.error;
  if (typeof err === 'string') return err;
  if (err && err.message) return err.message + (err.hint ? ' — ' + err.hint : '');
  return fallback || 'database operation failed';
}

function setDatabaseStatus(message, kind) {
  dbStatus.textContent = message;
  dbStatus.className = 'db-status' + (kind ? ' ' + kind : '');
}

function updateDatabaseControls() {
  const selected = Boolean(dbPackage.value.trim() && dbDatabase.value);
  document.getElementById('db-refresh').disabled = databaseBusy;
  document.getElementById('db-schema-button').disabled = databaseBusy || !selected;
  document.getElementById('db-backup-button').disabled = databaseBusy || !selected;
  document.getElementById('db-backups-refresh').disabled = databaseBusy || !selected;
  document.getElementById('db-query-button').disabled = databaseBusy || !selected;
  document.getElementById('db-execute-button').disabled = databaseBusy || !selected;
}

async function databaseRequest(action, payload) {
  databaseBusy = true;
  updateDatabaseControls();
  try {
    const response = await fetch('/api/database/' + action, {
      method: 'POST',
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
        'X-AUA-Dashboard-Token': DATABASE_TOKEN,
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(databaseError(data));
    return data;
  } finally {
    databaseBusy = false;
    updateDatabaseControls();
  }
}

function parseDatabaseParameters() {
  const raw = dbParams.value.trim();
  if (!raw) return null;
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) && (parsed === null || typeof parsed !== 'object')) {
    throw new Error('Parameters must be a JSON object or array.');
  }
  return parsed;
}

function renderDatabaseTable(columns, rows) {
  dbResults.innerHTML = '';
  if (!columns || !columns.length) {
    dbResults.innerHTML = '<div class="empty" style="padding:0.6rem">No tabular result.</div>';
    return;
  }
  const table = document.createElement('table');
  table.className = 'db-results';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  columns.forEach(column => {
    const th = document.createElement('th');
    th.textContent = column;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  (rows || []).forEach(row => {
    const tr = document.createElement('tr');
    row.forEach(value => {
      const td = document.createElement('td');
      td.textContent = value === null ? 'NULL' :
        (typeof value === 'object' ? JSON.stringify(value) : String(value));
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  dbResults.appendChild(table);
}

async function loadDatabases() {
  const selection = databaseSelection();
  if (!selection.package) {
    setDatabaseStatus('Enter an Android package name.', 'bad');
    return;
  }
  setDatabaseStatus('Discovering databases…');
  try {
    const data = await databaseRequest('list', selection);
    const previous = dbDatabase.value;
    dbDatabase.innerHTML = '<option value="">— select —</option>';
    (data.databases || []).forEach(database => {
      const option = document.createElement('option');
      option.value = database.name;
      const sidecars = database.wal_size_bytes != null ? ' + WAL' : '';
      option.textContent = database.name + ' (' + database.size_bytes + ' bytes' + sidecars + ')';
      dbDatabase.appendChild(option);
    });
    if ((data.databases || []).some(database => database.name === previous)) {
      dbDatabase.value = previous;
    } else if ((data.databases || []).length) {
      dbDatabase.value = data.databases[0].name;
    }
    updateDatabaseControls();
    setDatabaseStatus(
      data.count ? 'Found ' + data.count + ' database' + (data.count === 1 ? '.' : 's.') :
        'No databases found. The package must be installed and debuggable.',
      data.count ? 'ok' : ''
    );
    if (dbDatabase.value) loadBackups();
  } catch (error) {
    setDatabaseStatus(error.message, 'bad');
  }
}

function renderSchema(objects) {
  dbSchema.innerHTML = '';
  if (!objects.length) {
    dbSchema.innerHTML = '<div class="empty">No tables or views.</div>';
    return;
  }
  objects.forEach(object => {
    const item = document.createElement('div');
    item.className = 'db-schema-object';
    const name = document.createElement('div');
    name.className = 'db-object-name';
    name.textContent = object.type + ' · ' + object.name;
    name.title = 'Prepare a SELECT query for this object';
    name.addEventListener('click', () => {
      const quoted = '"' + object.name.replaceAll('"', '""') + '"';
      dbSql.value = 'SELECT * FROM ' + quoted + ' LIMIT 100;';
    });
    const columns = document.createElement('div');
    columns.className = 'db-columns';
    columns.textContent = (object.columns || []).map(column =>
      column.name + (column.type ? ' ' + column.type : '') + (column.primary_key ? ' PK' : '')
    ).join(' · ') || '(no columns)';
    item.appendChild(name);
    item.appendChild(columns);
    dbSchema.appendChild(item);
  });
}

async function loadSchema() {
  setDatabaseStatus('Capturing a coherent database snapshot and reading schema…');
  try {
    const data = await databaseRequest('schema', databaseSelection());
    renderSchema(data.objects || []);
    setDatabaseStatus('Loaded ' + data.count + ' schema object' + (data.count === 1 ? '.' : 's.'), 'ok');
  } catch (error) {
    setDatabaseStatus(error.message, 'bad');
  }
}

async function runDatabaseQuery() {
  let parameters;
  try {
    parameters = parseDatabaseParameters();
  } catch (error) {
    setDatabaseStatus(error.message, 'bad');
    return;
  }
  setDatabaseStatus(
    dbCoherentQuery.checked ?
      'Stopping the app for a coherent read-only snapshot…' :
      'Running read-only query without stopping the app…'
  );
  try {
    const data = await databaseRequest('query', {
      ...databaseSelection(),
      sql: dbSql.value,
      parameters: parameters,
      limit: Number(dbLimit.value || 100),
      live: !dbCoherentQuery.checked,
    });
    renderDatabaseTable(data.columns, data.rows);
    dbResultMeta.textContent = data.row_count + ' row' + (data.row_count === 1 ? '' : 's') +
      (data.truncated ? ' (truncated)' : '') + ' · ' + data.duration_ms + ' ms';
    setDatabaseStatus('Read-only query completed.', 'ok');
  } catch (error) {
    setDatabaseStatus(error.message, 'bad');
  }
}

function renderBackups(backups) {
  dbBackups.innerHTML = '';
  if (!backups.length) {
    dbBackups.innerHTML = '<div class="empty">No restore points yet.</div>';
    return;
  }
  backups.forEach(backup => {
    const item = document.createElement('div');
    item.className = 'db-backup';
    const actions = document.createElement('div');
    actions.className = 'db-actions';
    const label = document.createElement('span');
    label.style.marginRight = 'auto';
    label.textContent = backup.id + ' · ' + (backup.reason || 'backup');
    label.title = backup.created_at || '';
    const restore = document.createElement('button');
    restore.className = 'db-button danger';
    restore.textContent = 'Restore…';
    restore.addEventListener('click', () => openDatabaseConfirmation('restore', backup.id));
    actions.appendChild(label);
    actions.appendChild(restore);
    item.appendChild(actions);
    dbBackups.appendChild(item);
  });
}

async function loadBackups() {
  if (!dbDatabase.value) return;
  try {
    const data = await databaseRequest('backups', databaseSelection());
    renderBackups(data.backups || []);
  } catch (error) {
    dbBackups.textContent = error.message;
  }
}

async function createDatabaseBackup() {
  setDatabaseStatus('Creating restore point…');
  try {
    const data = await databaseRequest('backup', databaseSelection());
    setDatabaseStatus('Created backup ' + data.backup.id + '.', 'ok');
    loadBackups();
  } catch (error) {
    setDatabaseStatus(error.message, 'bad');
  }
}

function openDatabaseConfirmation(action, backupId) {
  let parameters = null;
  if (action === 'execute') {
    try {
      parameters = parseDatabaseParameters();
    } catch (error) {
      setDatabaseStatus(error.message, 'bad');
      return;
    }
  }
  const phrase = action === 'execute' ? 'MUTATE ' + dbDatabase.value : 'RESTORE ' + backupId;
  pendingDatabaseConfirmation = {action: action, backupId: backupId, parameters: parameters, phrase: phrase};
  document.getElementById('db-confirm-title').textContent =
    action === 'execute' ? 'Execute database mutation?' : 'Restore database backup?';
  document.getElementById('db-confirm-message').textContent = action === 'execute' ?
    'AUA will create an automatic restore point, run the SQL in one transaction, verify schema, foreign keys, and integrity, then replace the app database.' :
    'AUA will preserve the current database as a new safety backup before installing this restore point.';
  document.getElementById('db-confirm-phrase').textContent = phrase;
  dbConfirmInput.value = '';
  dbConfirmSubmit.disabled = true;
  dbConfirmDialog.showModal();
  dbConfirmInput.focus();
}

async function submitDatabaseConfirmation() {
  const pending = pendingDatabaseConfirmation;
  if (!pending || dbConfirmInput.value !== pending.phrase) return;
  dbConfirmDialog.close();
  setDatabaseStatus(pending.action === 'execute' ? 'Executing guarded mutation…' : 'Restoring backup…');
  try {
    if (pending.action === 'execute') {
      const data = await databaseRequest('execute', {
        ...databaseSelection(),
        sql: dbSql.value,
        parameters: pending.parameters,
        confirmation: pending.phrase,
      });
      const columns = ['statement', 'kind', 'changes', 'rowcount', 'lastrowid'];
      const rows = (data.statements || []).map(item => columns.map(column => item[column]));
      renderDatabaseTable(columns, rows);
      dbResultMeta.textContent = data.changes + ' change' + (data.changes === 1 ? '' : 's') +
        ' · backup ' + data.backup.id + ' · ' + data.duration_ms + ' ms';
      setDatabaseStatus('Mutation completed and verified.', 'ok');
    } else {
      const data = await databaseRequest('restore', {
        ...databaseSelection(),
        backup_id: pending.backupId,
        confirmation: pending.phrase,
      });
      setDatabaseStatus(
        'Restored ' + data.restored_backup.id + '; safety backup ' + data.safety_backup.id + '.',
        'ok'
      );
    }
    loadBackups();
  } catch (error) {
    setDatabaseStatus(error.message, 'bad');
  } finally {
    pendingDatabaseConfirmation = null;
  }
}

document.getElementById('db-refresh').addEventListener('click', loadDatabases);
document.getElementById('db-schema-button').addEventListener('click', loadSchema);
document.getElementById('db-query-button').addEventListener('click', runDatabaseQuery);
document.getElementById('db-execute-button').addEventListener('click', () => openDatabaseConfirmation('execute', null));
document.getElementById('db-backup-button').addEventListener('click', createDatabaseBackup);
document.getElementById('db-backups-refresh').addEventListener('click', loadBackups);
dbDatabase.addEventListener('change', () => {
  updateDatabaseControls();
  dbSchema.innerHTML = '<div class="empty">Load the selected database schema.</div>';
  loadBackups();
});
dbPackage.addEventListener('input', () => {
  databaseBootstrapped = true;
  updateDatabaseControls();
});
dbConfirmInput.addEventListener('input', () => {
  dbConfirmSubmit.disabled = !pendingDatabaseConfirmation ||
    dbConfirmInput.value !== pendingDatabaseConfirmation.phrase;
});
document.getElementById('db-confirm-cancel').addEventListener('click', () => {
  pendingDatabaseConfirmation = null;
  dbConfirmDialog.close();
});
dbConfirmSubmit.addEventListener('click', submitDatabaseConfirmation);
updateDatabaseControls();

if (isGrid) {
  tickGrid();
  setInterval(tickGrid, Math.max(POLL_MS, 800));
} else {
  if (focusSerial && frame) {
    frame.src = '/api/frame.jpg?serial=' + encodeURIComponent(focusSerial);
  }
  tickStatus(); tickEvents(); tickMap(); tickLogcat();
  setInterval(() => { tickStatus(); tickEvents(); }, POLL_MS);
  setInterval(() => { tickMap(); tickLogcat(); }, MAP_MS);
  if (!isGrid) {
    document.getElementById('px-arm').addEventListener('click', pxArm);
    document.getElementById('px-clear').addEventListener('click', () => pxPost('clear', {}));
    tickProxy();
    setInterval(tickProxy, MAP_MS);
  }
}
</script>
</body>
</html>
"""


class _DashboardState:
    def __init__(
        self,
        *,
        serials: list[str],
        focus: str | None,
        mode: str,
        cache_dir: Path,
        ensures: dict[str, dict[str, Any]],
        poll_ms: int,
        config: Any,
    ) -> None:
        self.serials = list(serials)
        self.focus = focus or (serials[0] if serials else None)
        self.mode = mode  # "grid" | "detail"
        self.cache_dir = cache_dir
        self.ensures = ensures
        self.poll_ms = poll_ms
        self.config = config
        from .platforms import PlatformFactory

        self.platform = PlatformFactory(config).create()
        # serial -> (bytes, taken_at, mime). The mime travels with the bytes: a cached
        # PNG served as image/jpeg is a broken image in the tile.
        self._fallback: dict[str, tuple[bytes, float, str]] = {}
        self._fallback_lock = threading.Lock()
        # serial -> True/False when we have an authoritative capture state, absent
        # when we do not know yet and should keep trusting the capture file.
        self._capture_live: dict[str, bool] = {}
        self.discovery_error: str | None = None
        self._pkg_cache: dict[str, tuple[str | None, float]] = {}
        self._map_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self._runtime_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self.database_token = secrets.token_urlsafe(32)
        self._database_lock = threading.Lock()
        self._proxy_lock = threading.Lock()
        self._engine: Any = None
        # When this dashboard opened. Not a filter — the whole point of the panel is to
        # show what an agent already did, so a run that finished before you opened the
        # page must still be here. It only labels which exchanges you have watched live.
        self._proxy_opened_at = time.time()

    @property
    def serial(self) -> str | None:
        return self.focus

    def _ensure_via(self, serial: str) -> str | None:
        ens = self.ensures.get(serial) or {}
        via = ens.get("via")
        return str(via) if via else None

    def _leased_elsewhere(self, serial: str) -> bool:
        """Is another agent holding this device right now?"""
        from . import leases

        try:
            lease = leases.read_lease(self.cache_dir, serial)
        except Exception:  # noqa: BLE001 — a watcher never fails on a bookkeeping read
            return False
        if not lease:
            return False
        try:
            return str(lease.get("owner") or "") != str(leases.resolve_owner(None))
        except Exception:  # noqa: BLE001
            return True

    def note_capture_live(self, serial: str, live: bool | None) -> None:
        """Record whether something is writing capture frames for *serial* right now.

        ``_served_frame`` needs this and cannot infer it from frame age: the capture
        ring dedupes unchanged screens, so an old file means "nothing moved" while
        capture is alive and "this is a corpse" once it is not. ``None`` means we have
        no authoritative answer, in which case the file is still trusted.
        """
        if live is None:
            self._capture_live.pop(serial, None)
        else:
            self._capture_live[serial] = bool(live)

    def _served_frame(self, serial: str) -> Path | None:
        """The capture file ``frame_bytes`` will serve, or None when it must screencap."""
        frame = latest_frame(self.cache_dir, serial)
        if frame is None or not frame.is_file():
            return None
        if self._capture_live.get(serial) is not False:
            return frame
        try:
            fresh = (time.time() - frame.stat().st_mtime) <= _FRAME_STALE_S
        except OSError:
            return None
        return frame if fresh else None

    def frame_token(self, serial: str) -> str:
        """Cache key for the bytes ``/api/frame.jpg`` will return for *serial*.

        While capture is alive this is the frame's mtime, so an unchanged screen costs
        no transfer. Otherwise the bytes come from a live screencap and the token has
        to advance with the clock — a key that never changed is what pinned every grid
        tile to the first frame it ever drew.
        """
        frame = self._served_frame(serial)
        if frame is not None:
            try:
                return f"f{int(frame.stat().st_mtime * 1000)}"
            except OSError:
                pass
        return f"s{int(time.time() * 1000)}"

    def _scoped_serial(self, serial: str | None) -> str:
        selected = serial or self.focus
        if not selected:
            raise UsageError("dashboard request needs a device serial")
        if selected not in self.serials:
            raise UsageError(
                f"device {selected!r} is not part of this dashboard session",
                code="dashboard_device_scope",
            )
        return selected

    def _daemon_call(self, serial: str, cmd: str, timeout: float = 1.5) -> dict[str, Any] | None:
        try:
            from . import daemon as daemon_mod

            sock = daemon_mod.socket_path(self.config, serial)
            if not Path(sock).exists():
                base = os.path.expanduser(self.config.daemon.socket)
                if Path(base).exists():
                    sock = base
                else:
                    return None
            with daemon_mod.DaemonClient(sock, timeout=timeout) as client:
                if not client.ping():
                    return None
                resp = client.call(cmd, journal=False)
                if isinstance(resp, dict):
                    result = resp.get("result")
                    return result if isinstance(result, dict) else resp
        except Exception as exc:  # noqa: BLE001
            logger.debug("daemon %s skipped: %s", cmd, exc)
        return None

    def device_runtime(self, serial: str) -> dict[str, Any]:
        now = time.time()
        cached = self._runtime_cache.get(serial)
        if cached and (now - cached[1]) < 1.0:
            return cached[0]
        status = device_runtime_status(
            self.cache_dir,
            serial,
            lease_registry_dir=self.config.lease.registry_dir,
            now=now,
        )
        self._runtime_cache[serial] = (status, now)
        return status

    def foreground_package(self, serial: str | None = None) -> str | None:
        ser = serial or self.focus
        if not ser:
            return None
        now = time.time()
        cached = self._pkg_cache.get(ser)
        if cached and (now - cached[1]) < 5.0:
            return cached[0]
        pkg: str | None = None
        try:
            info = self.platform.connect(ser).current_app() or {}
            pkg = info.get("package") or None
        except Exception as exc:  # noqa: BLE001
            logger.debug("current_app failed: %s", exc)
        self._pkg_cache[ser] = (pkg or None, now)
        return pkg or None

    @staticmethod
    def _database_text(payload: dict[str, Any], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise UsageError(f"dashboard database request needs {name!r}")
        return value.strip()

    @staticmethod
    def _database_bool(payload: dict[str, Any], name: str, default: bool) -> bool:
        value = payload.get(name, default)
        if not isinstance(value, bool):
            raise UsageError(f"dashboard database field {name!r} must be a boolean")
        return value

    @staticmethod
    def _database_int(payload: dict[str, Any], name: str, default: int, *, maximum: int) -> int:
        value = payload.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise UsageError(f"dashboard database field {name!r} must be a positive integer")
        if value > maximum:
            raise UsageError(f"dashboard database field {name!r} must be at most {maximum}")
        return value

    def database_operation(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one dashboard database request through the guarded database service."""
        app_database = self.platform.capability("app_database")

        serial = self._database_text(payload, "serial") if payload.get("serial") else self.focus
        if not serial:
            raise UsageError("dashboard database request needs a device serial")
        if serial not in self.serials:
            raise UsageError(
                f"device {serial!r} is not part of this dashboard session",
                code="dashboard_device_scope",
            )
        package = self._database_text(payload, "package")
        restart = self._database_bool(payload, "restart", True)
        device = self.platform.connect(serial)

        with self._database_lock:
            if action == "list":
                return app_database.list_databases(device, package)

            database = self._database_text(payload, "database")
            if action == "schema":
                table = payload.get("table")
                if table is not None and not isinstance(table, str):
                    raise UsageError("dashboard database field 'table' must be a string")
                return app_database.database_schema(
                    device,
                    package,
                    database,
                    table=table.strip() if table and table.strip() else None,
                    restart=restart,
                )
            if action == "query":
                live = self._database_bool(payload, "live", True)
                return app_database.query_database(
                    device,
                    package,
                    database,
                    self._database_text(payload, "sql"),
                    parameters=payload.get("parameters"),
                    limit=self._database_int(payload, "limit", 100, maximum=1000),
                    timeout_ms=self._database_int(payload, "timeout_ms", 5000, maximum=60_000),
                    restart=restart,
                    live=live,
                )
            if action == "backup":
                return app_database.backup_database(
                    device,
                    self.cache_dir,
                    package,
                    database,
                    restart=restart,
                )
            if action == "backups":
                return app_database.list_backups(
                    device,
                    self.cache_dir,
                    package,
                    database,
                )
            if action == "execute":
                expected = f"MUTATE {database}"
                if payload.get("confirmation") != expected:
                    raise UsageError(
                        f"type {expected!r} to confirm this database mutation",
                        code="database_confirmation_required",
                    )
                return app_database.execute_database(
                    device,
                    self.cache_dir,
                    package,
                    database,
                    self._database_text(payload, "sql"),
                    parameters=payload.get("parameters"),
                    timeout_ms=self._database_int(payload, "timeout_ms", 5000, maximum=60_000),
                    restart=restart,
                    confirmed=True,
                )
            if action == "restore":
                backup_id = self._database_text(payload, "backup_id")
                expected = f"RESTORE {backup_id}"
                if payload.get("confirmation") != expected:
                    raise UsageError(
                        f"type {expected!r} to confirm this database restore",
                        code="database_confirmation_required",
                    )
                return app_database.restore_database(
                    device,
                    self.cache_dir,
                    package,
                    database,
                    backup_id,
                    restart=restart,
                    confirmed=True,
                )
        raise UsageError(f"unknown dashboard database operation: {action!r}")

    def journal_bundle(
        self, serial: str | None = None, *, since_ms: int | None = None, limit: int = 150
    ) -> dict[str, Any]:
        from . import journal as journal_mod

        ser = self._scoped_serial(serial)
        events = journal_mod.read_since(self.cache_dir, ser, since_ms=since_ms, limit=limit)
        window = journal_mod.read_since(self.cache_dir, ser, since_ms=None, limit=400)
        stats = journal_mod.failure_stats(window)
        return {
            "events": events,
            "stats": stats,
            "detail_revision": journal_mod.detail_revision(self.cache_dir, ser),
        }

    def journal_detail(
        self, detail_id: str, serial: str | None = None
    ) -> dict[str, Any] | None:
        from . import journal as journal_mod

        if not detail_id or len(detail_id) > 128 or not all(
            char.isalnum() or char in "-_." for char in detail_id
        ):
            raise UsageError("invalid dashboard journal detail id")
        return journal_mod.read_detail(
            self.cache_dir, self._scoped_serial(serial), detail_id
        )

    def map_payload(self, serial: str | None = None) -> dict[str, Any]:
        ser = serial or self.focus
        now = time.time()
        if ser:
            hit = self._map_cache.get(ser)
            if hit is not None and (now - hit[1]) < 5.0:
                return hit[0]
        out: dict[str, Any] = {
            "ok": True,
            "package": None,
            "known": False,
            "screens": [],
            "routes": [],
            "serial": ser,
        }
        try:
            pkg = self.foreground_package(ser)
            out["package"] = pkg
            if not pkg:
                if ser:
                    self._map_cache[ser] = (out, now)
                return out
            from .memory import AppMemoryStore

            store = AppMemoryStore(self.config.memory)
            app = store.load(pkg)
            if app is None:
                if ser:
                    self._map_cache[ser] = (out, now)
                return out
            out["known"] = True
            out["label"] = app.label
            out["description"] = app.description
            screens = sorted(
                app.screens.values(),
                key=lambda s: (-int(s.visit_count or 0), s.name),
            )
            out["screens"] = [
                {
                    "name": s.name,
                    "activity": s.activity,
                    "visit_count": s.visit_count,
                    "stale": s.stale,
                    "context_id": s.context_id,
                }
                for s in screens[:80]
            ]
            out["routes"] = [
                {
                    "from": e.from_screen,
                    "to": e.to_screen,
                    "action": e.action,
                    "count": e.count,
                    "status": e.status,
                }
                for e in app.routes[:120]
            ]
            out["screen_count"] = len(app.screens)
            out["route_count"] = len(app.routes)
        except Exception as exc:  # noqa: BLE001 — never crash the dashboard
            logger.debug("map payload failed: %s", exc)
            out["ok"] = False
            out["error"] = str(exc)
        if ser:
            self._map_cache[ser] = (out, now)
        return out

    def device_tile(self, serial: str) -> dict[str, Any]:
        frame = latest_frame(self.cache_dir, serial)
        age_ms = None
        if frame is not None:
            age_ms = int((time.time() - frame.stat().st_mtime) * 1000)
        via = self._ensure_via(serial)
        capture_running = frame is not None and (age_ms is not None and age_ms < 15_000)
        # Only a daemon (or the absence of any writer) is authoritative; frame age is
        # not, because an idle screen is deduped rather than re-written.
        live: bool | None = False if via in (None, "screencap") else None
        if via == "daemon":
            detail = self._daemon_call(serial, "capture_status")
            if isinstance(detail, dict):
                capture_running = bool(detail.get("running")) and not detail.get("paused")
                live = capture_running
        self.note_capture_live(serial, live)
        pkg = None
        with contextlib.suppress(Exception):
            pkg = self.foreground_package(serial)
        runtime = self.device_runtime(serial)
        return {
            "serial": serial,
            "owner": owner_for_serial(self.cache_dir, serial),
            **runtime,
            "package": pkg,
            "via": via,
            "capture_running": capture_running,
            "has_frame": frame is not None or serial in self._fallback,
            "frame_token": self.frame_token(serial),
            "frame_age_ms": age_ms,
        }

    def devices_payload(self) -> dict[str, Any]:
        known = list(self.serials)
        if self.mode == "grid":
            # Grid dashboards intentionally discover devices as they appear. A
            # detail dashboard stays scoped to the serial it was started for.
            online, self.discovery_error = discover_online_serials(self.config)
            known = list(dict.fromkeys([*known, *online]))
            self.serials = known
            for ser in online:
                if ser not in self.ensures:
                    with contextlib.suppress(Exception):
                        self.ensures[ser] = ensure_capture(
                            serial=ser,
                            config=self.config,
                            allow_sidecar=False,
                        )
        return {
            "ok": True,
            "mode": self.mode,
            "devices": [self.device_tile(s) for s in known],
            "discovery_error": self.discovery_error,
        }

    def status(self, serial: str | None = None) -> dict[str, Any]:
        ser = serial or self.focus
        if not ser:
            return {"ok": False, "error": "no serial"}
        frame = latest_frame(self.cache_dir, ser)
        age_ms = None
        session_id = None
        if frame is not None:
            age_ms = int((time.time() - frame.stat().st_mtime) * 1000)
            session_id = frame.parent.parent.name
        marks = recent_marks(self.cache_dir, ser)
        via = self._ensure_via(ser)
        capture_running = frame is not None and (age_ms is not None and age_ms < 15_000)
        # Frame age cannot decide this: the capture ring dedupes unchanged screens.
        live: bool | None = False if via in (None, "screencap") else None
        detail: dict[str, Any] | None = None
        try:
            if via == "sidecar":
                from . import capture_sidecar as cs

                sock = cs.socket_path(self.cache_dir)
                if Path(sock).exists():
                    detail = cs.call(sock, "status")
                    capture_running = bool(detail.get("running")) and not detail.get("paused")
                    session_id = detail.get("session_id") or session_id
                    live = capture_running
            elif via == "daemon":
                detail = self._daemon_call(ser, "capture_status")
                if isinstance(detail, dict):
                    capture_running = bool(detail.get("running")) and not detail.get("paused")
                    session_id = detail.get("session_id") or session_id
                    live = capture_running
        except Exception as exc:  # noqa: BLE001
            detail = {"error": str(exc)}
        self.note_capture_live(ser, live)

        pkg = None
        with contextlib.suppress(Exception):
            pkg = self.foreground_package(ser)

        stats: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            stats = self.journal_bundle(ser, limit=1).get("stats") or {}

        ens = self.ensures.get(ser) or {}
        runtime = self.device_runtime(ser)
        return {
            "ok": True,
            "serial": ser,
            "owner": owner_for_serial(self.cache_dir, ser),
            **runtime,
            "via": via,
            "package": pkg,
            "capture_running": capture_running,
            "has_frame": frame is not None or ser in self._fallback,
            "frame_token": self.frame_token(ser),
            "frame_age_ms": age_ms,
            "frame_path": str(frame) if frame else None,
            "session_id": session_id,
            "marks": marks,
            "stats": {
                "fail_count": stats.get("fail_count", 0),
                "total": stats.get("total", 0),
                "by_cmd": stats.get("by_cmd", {}),
                "failures": stats.get("failures", []),
                "slow": stats.get("slow", []),
            },
            "capture_detail": detail,
            "ensure": {k: ens.get(k) for k in ("via", "ok", "hint", "daemon_error") if k in ens},
            "poll_ms": self.poll_ms,
            "mode": self.mode,
        }

    def _proxy_service(self) -> Any:
        """The platform's proxy capability, or a typed refusal on a platform without one."""
        return self.platform.capability("proxy")

    def _proxy_engine(self) -> Any:
        """A lazily built engine for the *write* half of the panel.

        Reads go straight to the proxy capability, but arming or removing a rule changes
        state the device keeps until something clears it, and only the engine knows how to
        journal that undo first. None of the mock methods it is used for connect to the
        device, so this never competes with a running agent for the UiAutomation slot.
        """
        if self._engine is None:
            from .engine import Engine

            self._engine = Engine(self.config)
        return self._engine

    def proxy_payload(self, serial: str | None = None, *, limit: int = 200) -> dict[str, Any]:
        """Proxy health, armed rules and the live traffic feed for one device."""
        ser = serial or self.focus
        try:
            pm = self._proxy_service()
        except AuaError as exc:
            return {"ok": False, "supported": False, **exc.to_dict()}

        out: dict[str, Any] = {"ok": True, "supported": True, "serial": ser}

        health: dict[str, Any] | None = None
        if ser:
            with contextlib.suppress(Exception):
                health = pm.proxy_health(ser, self.cache_dir, self_heal=False)
        out["health"] = health
        state_name = str((health or {}).get("state") or "unknown")
        # `ok` is not "a proxy is on": proxy_health reports ok for a clean *unproxied*
        # device too, because its network path is sane. The state is the real answer.
        out["on"] = bool(health) and state_name != "unproxied"
        out["intercepting"] = bool(health and health.get("intercepting"))
        out["port"] = (health or {}).get("port")
        out["state"] = (health or {}).get("state")

        rules: list[dict[str, Any]] = []
        mode = "off"
        owner = None
        with contextlib.suppress(Exception):
            doc = pm.load_doc(pm.rules_path(self.cache_dir, ser))
            rules, _changed = pm.backfill_rule_ids(doc["rules"])
            mode = str(doc.get("mode") or "off")
            owner = doc.get("owner")
        out["mode"] = mode
        out["rules_owner"] = owner

        flows: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            flows = pm.read_flows_since(self.cache_dir, 0, ser)
        # Which rules have actually fired, so an armed rule reads differently from a spent
        # one. The addon spends a rule's `times` budget in its own process and deliberately
        # never writes it back, so the flow log is the only place this is knowable.
        fired: dict[str, int] = {}
        for entry in flows:
            rid = entry.get("rule")
            if rid:
                fired[str(rid)] = fired.get(str(rid), 0) + 1
        out["rules"] = [dict(rule, fired=fired.get(str(rule.get("id")), 0)) for rule in rules]
        out["flows"] = [
            dict(f, live=float(f.get("ts") or 0) > self._proxy_opened_at)
            for f in flows[-max(1, min(int(limit), 500)) :]
        ]
        out["flow_count"] = len(flows)
        out["manipulated"] = sum(1 for f in flows if f.get("action"))
        return out

    def proxy_flow_detail(
        self, n: int, ts: float | None = None, serial: str | None = None
    ) -> dict[str, Any]:
        """Full headers and bodies for one logged exchange, when body capture was on.

        *ts* disambiguates: the addon's sequence number restarts at 1 with every mitmdump
        process while the log is append-only, so after one `proxy stop`/`start` two rows
        share an ``n`` and matching on it alone hands back the wrong exchange.
        """
        pm = self._proxy_service()
        bodies: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            bodies = pm.read_flow_bodies(self.cache_dir, serial or self.focus)
        candidates = [e for e in bodies if int(e.get("n") or 0) == int(n)]
        if ts is not None and len(candidates) > 1:
            candidates.sort(key=lambda e: abs(float(e.get("ts") or 0) - float(ts)))
        if candidates:
            chosen = candidates[0] if ts is not None else candidates[-1]
            return {"ok": True, "flow": redact_flow(chosen)}
        return {
            "ok": False,
            "error": {
                "code": "proxy_flow_body_missing",
                "message": f"no captured body for flow {n}",
                "hint": "Bodies are only kept while the proxy runs with body capture on.",
            },
        }

    def proxy_operation(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Arm or remove a mock rule from the browser, through the engine that owns undo."""
        engine = self._proxy_engine()
        serial = self._database_text(payload, "serial") if payload.get("serial") else self.focus
        if serial and serial not in self.serials:
            raise UsageError(
                f"device {serial!r} is not part of this dashboard session",
                code="dashboard_device_scope",
            )
        with self._proxy_lock:
            if action == "list":
                return engine.mock_list()
            if action == "clear":
                return engine.mock_clear()
            if action == "rm":
                return engine.mock_rm(self._database_text(payload, "id"))
            if action == "stub":
                return engine.mock_map(
                    self._database_text(payload, "method"),
                    self._database_text(payload, "path"),
                    status=self._database_int(payload, "status", 200, maximum=599),
                    body=self._proxy_optional_text(payload, "body"),
                    host=self._proxy_optional_text(payload, "host"),
                    times=self._proxy_count(payload, "times", maximum=10_000),
                    serial=serial,
                )
            if action == "rewrite":
                status_raw = payload.get("status")
                return engine.mock_rewrite(
                    self._database_text(payload, "method"),
                    self._database_text(payload, "path"),
                    host=self._proxy_optional_text(payload, "host"),
                    status=(
                        self._database_int(payload, "status", 200, maximum=599)
                        if status_raw not in (None, "")
                        else None
                    ),
                    headers=self._proxy_mapping(payload, "headers"),
                    body=self._proxy_optional_text(payload, "body"),
                    set_json=self._proxy_mapping(payload, "set_json"),
                    delete_json=self._proxy_string_list(payload, "delete_json"),
                    times=self._proxy_count(payload, "times", maximum=10_000),
                    serial=serial,
                )
        raise UsageError(f"unknown dashboard proxy action {action!r}")

    @staticmethod
    def _proxy_count(payload: dict[str, Any], field: str, *, maximum: int) -> int:
        """A non-negative bound. Unlike the database helper, 0 is meaningful here: it is
        how a rule says "fire every time", which is the default a caller sends."""
        value = payload.get(field, 0)
        if value in (None, ""):
            return 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UsageError(f"dashboard proxy field {field!r} must be a non-negative integer")
        return min(value, maximum)

    @staticmethod
    def _proxy_optional_text(payload: dict[str, Any], field: str) -> str | None:
        value = payload.get(field)
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise UsageError(f"dashboard proxy field {field!r} must be a string")
        return value

    @staticmethod
    def _proxy_mapping(payload: dict[str, Any], field: str) -> dict[str, Any] | None:
        value = payload.get(field)
        if value in (None, {}, ""):
            return None
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise UsageError(f"dashboard proxy field {field!r} must be an object of strings")
        return dict(value)

    @staticmethod
    def _proxy_string_list(payload: dict[str, Any], field: str) -> list[str] | None:
        value = payload.get(field)
        if value in (None, [], ""):
            return None
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise UsageError(f"dashboard proxy field {field!r} must be a list of strings")
        return [v for v in value if v.strip()]

    def frame_bytes(self, serial: str | None = None) -> tuple[bytes, str]:
        ser = serial or self.focus
        if not ser:
            return _PLACEHOLDER_PNG, "image/png"
        path = self._served_frame(ser)
        if path is not None:
            with contextlib.suppress(OSError):
                return path.read_bytes(), "image/jpeg"
        with self._fallback_lock:
            hit = self._fallback.get(ser)
            if hit and (time.time() - hit[1]) < 0.8:
                return hit[0], hit[2]
        if self._leased_elsewhere(ser):
            # Screenshotting here attaches uiautomator2 and takes the UiAutomation slot
            # from the agent that holds this device — for a preview thumbnail. Whatever
            # capture already wrote is the honest picture; a watcher never interrupts.
            stale = latest_frame(self.cache_dir, ser)
            if stale is not None and stale.is_file():
                with contextlib.suppress(OSError):
                    return stale.read_bytes(), "image/jpeg"
            return _PLACEHOLDER_PNG, "image/png"
        try:
            img = self.platform.connect(ser).screenshot()
            raw = getattr(img, "png_bytes", None)
            if raw is None and hasattr(img, "pil"):
                import io

                buf = io.BytesIO()
                img.pil().convert("RGB").save(buf, format="JPEG", quality=70)
                raw = buf.getvalue()
            if raw:
                data = bytes(raw)
                mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
                with self._fallback_lock:
                    self._fallback[ser] = (data, time.time(), mime)
                return data, mime
        except Exception as exc:  # noqa: BLE001
            logger.debug("fallback screencap failed: %s", exc)
        # Screencap is gone too - a stale frame still says more than a blank tile.
        stale = latest_frame(self.cache_dir, ser)
        if stale is not None and stale.is_file():
            with contextlib.suppress(OSError):
                return stale.read_bytes(), "image/jpeg"
        return _PLACEHOLDER_PNG, "image/png"

    def log_lines(self, serial: str, lines: int = 80) -> list[str]:
        n = max(1, min(int(lines), 500))
        try:
            return self.platform.recent_logs(serial, limit=n)
        except Exception as exc:  # noqa: BLE001 — dashboard remains available
            return [f"<device logs failed: {exc}>"]


def _make_handler(state: _DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            logger.debug("dashboard: " + fmt, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline'; "
                f"script-src 'nonce-{state.database_token}'; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _qs_serial(self, qs: dict[str, list[str]]) -> str | None:
            raw = (qs.get("serial") or [""])[0].strip()
            return raw or state.focus

        def _scoped_qs_serial(self, qs: dict[str, list[str]]) -> str | None:
            try:
                return state._scoped_serial(self._qs_serial(qs))
            except UsageError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
                return None

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if path in ("/", "/index.html"):
                focus = (qs.get("serial") or [""])[0].strip()
                if focus and focus not in state.serials:
                    self._send(404, b"device not part of this dashboard session", "text/plain")
                    return
                mode = "detail" if focus else state.mode
                serial_boot = state.focus or ""
                html = (
                    _DASHBOARD_HTML.replace("__POLL_MS__", str(state.poll_ms))
                    .replace("__MODE_JSON__", _script_json(mode if focus else state.mode))
                    .replace("__SERIAL_JSON__", _script_json(serial_boot))
                    .replace("__DATABASE_TOKEN__", state.database_token)
                )
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/devices":
                self._json(state.devices_payload())
                return
            if path == "/api/proxy":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                raw_limit = (qs.get("limit") or [""])[0]
                limit = int(raw_limit) if raw_limit.isdigit() else 200
                self._json(state.proxy_payload(ser, limit=limit))
                return
            if path == "/api/proxy/flow":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                raw = (qs.get("n") or [""])[0]
                if not raw.isdigit():
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "dashboard_request",
                                "message": "flow number must be a positive integer",
                            },
                        },
                        400,
                    )
                    return
                ts_raw = (qs.get("ts") or [""])[0]
                try:
                    ts = float(ts_raw) if ts_raw else None
                except ValueError:
                    ts = None
                try:
                    self._json(state.proxy_flow_detail(int(raw), ts, ser))
                except AuaError as exc:
                    self._json({"ok": False, **exc.to_dict()}, 400)
                return
            if path == "/api/status":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                self._json(state.status(ser))
                return
            if path == "/api/frame.jpg":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                data, mime = state.frame_bytes(ser)
                self._send(200, data, mime)
                return
            if path == "/api/events":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                since_raw = (qs.get("since_ms") or [""])[0]
                limit_raw = (qs.get("limit") or ["150"])[0]
                since_ms = int(since_raw) if since_raw.isdigit() else None
                try:
                    limit = max(1, min(int(limit_raw), 500))
                except ValueError:
                    limit = 150
                try:
                    bundle = state.journal_bundle(ser, since_ms=since_ms, limit=limit)
                    self._json({"ok": True, **bundle})
                except UsageError as exc:
                    self._json({"ok": False, "events": [], "stats": {}, "error": str(exc)}, 400)
                except Exception as exc:  # noqa: BLE001
                    self._json({"ok": False, "events": [], "stats": {}, "error": str(exc)})
                return
            if path == "/api/event":
                supplied = self.headers.get("X-AUA-Dashboard-Token", "")
                if not secrets.compare_digest(supplied, state.database_token):
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "dashboard_token",
                                "message": "invalid dashboard request token",
                            },
                        },
                        403,
                    )
                    return
                detail_id = (qs.get("detail_id") or [""])[0].strip()
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                try:
                    detail = state.journal_detail(detail_id, ser)
                except UsageError as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                    return
                if detail is None:
                    self._json({"ok": False, "error": "journal detail not found"}, 404)
                    return
                self._json({"ok": True, "detail": detail})
                return
            if path == "/api/map":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                self._json(state.map_payload(ser))
                return
            if path == "/api/logcat":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                lines_raw = (qs.get("lines") or ["80"])[0]
                try:
                    n = max(1, min(int(lines_raw), 500))
                except ValueError:
                    n = 80
                self._json({"ok": True, "lines": state.log_lines(ser, n) if ser else []})
                return
            if path.startswith("/api/file"):
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                rel = (qs.get("path") or [""])[0]
                root = captures_root(state.cache_dir, ser).resolve()
                target = Path(rel).expanduser().resolve()
                if not str(target).startswith(str(root)) or not target.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self._send(200, target.read_bytes(), mime)
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            prefix = ""
            for candidate in ("/api/database/", "/api/proxy/"):
                if parsed.path.startswith(candidate):
                    prefix = candidate
                    break
            if not prefix:
                self._send(404, b"not found", "text/plain")
                return
            supplied = self.headers.get("X-AUA-Dashboard-Token", "")
            if not secrets.compare_digest(supplied, state.database_token):
                self._json(
                    {
                        "ok": False,
                        "error": {
                            "code": "dashboard_token",
                            "message": "invalid dashboard request token",
                        },
                    },
                    403,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 1_000_000:
                self._json(
                    {
                        "ok": False,
                        "error": {
                            "code": "dashboard_request",
                            "message": "database request body must be between 1 byte and 1 MB",
                        },
                    },
                    400,
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise UsageError("dashboard request body must be a JSON object")
                action = parsed.path[len(prefix) :]
                if prefix == "/api/proxy/":
                    result = state.proxy_operation(action, payload)
                else:
                    result = state.database_operation(action, payload)
                self._json(result)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._json(
                    {
                        "ok": False,
                        "error": {
                            "code": "dashboard_json",
                            "message": f"invalid JSON request: {exc}",
                        },
                    },
                    400,
                )
            except AuaError as exc:
                self._json({"ok": False, **exc.to_dict()}, 400)
            except Exception as exc:  # noqa: BLE001 — dashboard must stay available
                logger.exception("dashboard database request failed")
                self._json(
                    {
                        "ok": False,
                        "error": {
                            "code": "internal_error",
                            "message": str(exc),
                        },
                    },
                    500,
                )

    return Handler


def run(
    *,
    serial: str | None = None,
    port: int = _DEFAULT_PORT,
    cache_dir: str | Path | None = None,
    config: Any | None = None,
    open_browser: bool = True,
    poll_ms: int = 500,
    block: bool = True,
    grid: bool = False,
) -> dict[str, Any]:
    """Ensure capture, serve the dashboard, optionally open a browser.

    By default, serves a tile grid that discovers online devices as they appear.
    Pass ``serial`` (or open ``/?serial=…``) for the full detail view.
    """
    from .config import load_config

    cfg = config or load_config()
    targets = resolve_dashboard_targets(
        serial or getattr(cfg.device, "serial", None), grid=grid, config=cfg
    )
    mode = str(targets["mode"])
    serials: list[str] = list(targets["serials"])
    focus: str | None = targets.get("focus")
    discovery_error: str | None = targets.get("discovery_error")
    cache = Path(cache_dir or cfg.cache.dir).expanduser()

    # Multi-device: prefer per-serial daemons; skip the single-process sidecar so we
    # do not steal capture from another agent's serial.
    allow_sidecar = mode == "detail" and len(serials) == 1
    ensures: dict[str, dict[str, Any]] = {}
    for ser in serials:
        with contextlib.suppress(Exception):
            ensures[ser] = ensure_capture(serial=ser, config=cfg, allow_sidecar=allow_sidecar)

    listen = _pick_free_port(port)
    state = _DashboardState(
        serials=serials,
        focus=focus,
        mode=mode,
        cache_dir=cache,
        ensures=ensures,
        poll_ms=max(200, int(poll_ms)),
        config=cfg,
    )
    state.discovery_error = discovery_error
    handler = _make_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", listen), handler)
    # port=0 lets the OS choose, so report what we actually bound rather than the ask.
    listen = int(httpd.server_address[1])
    url = f"http://127.0.0.1:{listen}/"
    primary_via = (ensures.get(focus or (serials[0] if serials else "")) or {}).get("via")
    info = {
        "ok": True,
        "action": "dashboard",
        "url": url,
        "mode": mode,
        "serial": focus,
        "serials": serials,
        "via": primary_via,
        "port": listen,
        "discovery_error": discovery_error,
        "hint": (
            (
                "No device attached yet — the grid picks one up as soon as it boots. "
                if mode == "grid" and not serials
                else f"Grid of {len(serials)} device(s). Click a tile for detail. "
                if mode == "grid"
                else f"Watching {focus} via {primary_via}. "
            )
            + "Leave this running; stop with Ctrl-C. Agent work is unaffected."
        ),
        "ensures": {
            k: {kk: vv for kk, vv in v.items() if kk in ("via", "ok", "hint")}
            for k, v in ensures.items()
        },
    }

    def _serve() -> None:
        with httpd:
            httpd.serve_forever(poll_interval=0.5)

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    if block:
        logger.info("dashboard on %s (mode=%s serials=%s)", url, mode, ",".join(serials))
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        try:
            _serve()
        except KeyboardInterrupt:
            info["stopped"] = True
        finally:
            with contextlib.suppress(Exception):
                httpd.shutdown()
        return info

    t = threading.Thread(target=_serve, name="aua-dashboard", daemon=True)
    t.start()
    info["thread"] = True
    _SERVERS[url] = (httpd, t)
    return info


def shutdown(info: dict[str, Any]) -> None:
    """Stop a dashboard started with ``block=False``."""

    entry = _SERVERS.pop(str(info.get("url") or ""), None)
    if entry is None:
        return
    httpd, thread = entry
    with contextlib.suppress(Exception):
        httpd.shutdown()
    with contextlib.suppress(Exception):
        httpd.server_close()
    thread.join(timeout=2)
