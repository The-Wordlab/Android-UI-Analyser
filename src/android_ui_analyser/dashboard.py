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

from .errors import AuaError, DeviceError, UsageError

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8765


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


def resolve_serial(serial: str | None, *, config: Any | None = None) -> str:
    """Resolve a single device serial (detail view / legacy callers)."""
    if serial:
        return serial
    online = list_online_serials() if config is None else list_online_serials(config)
    if not online:
        raise DeviceError(
            "no device found for dashboard",
            hint="Start a headless AVD (`aua emulator start --headless`) or pass --serial.",
        )
    if len(online) > 1:
        listing = ", ".join(online)
        raise DeviceError(
            f"multiple devices attached ({listing})",
            hint="Pass --serial <id>, or `aua dashboard --grid` to watch all.",
        )
    return online[0]


def resolve_dashboard_targets(
    serial: str | None = None, *, grid: bool = False, config: Any | None = None
) -> dict[str, Any]:
    """Pick grid vs detail mode.

    * Explicit ``--serial`` → detail for that device.
    * ``--grid`` or multiple online devices with no serial → grid of all.
    * Exactly one device → detail.
    """
    online = list_online_serials() if config is None else list_online_serials(config)
    if serial:
        if serial not in online and online:
            # Still allow watching a serial that briefly dropped offline.
            logger.warning("serial %s not currently online; dashboard will retry frames", serial)
        return {"mode": "detail", "serials": [serial], "focus": serial}
    if not online:
        raise DeviceError(
            "no device found for dashboard",
            hint="Start headless AVDs (`aua emulator start --headless --parallel`) "
            "or pass --serial.",
        )
    if grid or len(online) > 1:
        return {"mode": "grid", "serials": online, "focus": None}
    return {"mode": "detail", "serials": online, "focus": online[0]}


def owner_for_serial(cache_dir: str | Path, serial: str) -> str | None:
    """Look up parallel-agent owner tag from aua emulator meta, if any."""
    root = Path(cache_dir).expanduser() / "emulator"
    if not root.is_dir():
        return None
    for path in root.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict) and meta.get("serial") == serial:
            owner = meta.get("owner")
            return str(owner) if owner else None
    return None


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
    grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.9fr);
    gap: 0.85rem; padding: 0.85rem; max-width: 1600px; margin: 0 auto;
  }
  @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.7rem 0.85rem; min-height: 0;
  }
  .panel h2 {
    font-size: 0.72rem; margin: 0 0 0.55rem; color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
  }
  .stage img {
    width: 100%; max-height: 68vh; object-fit: contain; background: #000;
    border-radius: 6px;
  }
  .meta { font-size: 0.75rem; color: var(--muted); display: flex; gap: 0.9rem; flex-wrap: wrap; margin-top: 0.4rem; }
  .scroll { overflow: auto; max-height: 68vh; }
  .scroll.sm { max-height: 14rem; }
  .scroll.md { max-height: 18rem; }
  #journal, #marks, #fail-list, #slow { list-style: none; margin: 0; padding: 0; font-size: 0.78rem; }
  #journal li, #marks li, #fail-list li, #slow li {
    padding: 0.4rem 0.35rem; border-bottom: 1px solid var(--border);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #journal li.fail { background: rgba(239,107,90,0.08); border-left: 2px solid var(--danger); }
  #journal .t, #marks .t { color: var(--muted); margin-right: 0.45rem; }
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
  #journal .exchange pre {
    margin: 0; max-height: 28rem; overflow: auto; padding: 0.55rem;
    border-radius: 5px; background: #090b10; color: #cdd2db; font: inherit;
    font-size: 0.7rem; line-height: 1.4; white-space: pre-wrap; overflow-wrap: anywhere;
    user-select: text;
  }
  #journal .detail-note { color: var(--muted); font: 0.68rem ui-sans-serif, system-ui; }
  .lower {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 0.85rem; padding: 0 0.85rem 0.85rem; max-width: 1600px; margin: 0 auto;
  }
  @media (max-width: 1100px) { .lower { grid-template-columns: 1fr; } }
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
  #logcat {
    margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.68rem; line-height: 1.35; color: #b8bec8; white-space: pre-wrap;
    word-break: break-all;
  }
  footer { padding: 0.4rem 1.1rem 1rem; color: var(--muted); font-size: 0.72rem; }
  .empty { color: var(--muted); font-size: 0.78rem; padding: 0.4rem 0; }
  /* --- grid mode --- */
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
  <span id="age" class="pill">frame …</span>
  <span id="failpill" class="pill hidden">fails 0</span>
  <span id="pkg" class="pill hidden">pkg …</span>
  <span id="count" class="pill hidden">0 devices</span>
</header>

<div id="grid-view" class="hidden">
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
  <aside class="panel">
    <h2>Agent I/O journal</h2>
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
    <h2>Logcat</h2>
    <pre id="logcat" class="scroll md">…</pre>
  </section>
</div>
<div class="lower" style="grid-template-columns:1fr">
  <section class="panel">
    <h2>Capture marks</h2>
    <ul id="marks" class="scroll sm"><li class="empty">—</li></ul>
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
let lastSrc = '';
let sinceMs = 0;
const seenKeys = new Set();
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
function fmtTime(ms) {
  if (!ms) return '';
  return new Date(ms).toLocaleTimeString();
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
function renderExchange(panel, exchange, note) {
  panel.textContent = '';
  const requestSection = document.createElement('section');
  requestSection.className = 'exchange-section';
  const requestHeading = document.createElement('h3');
  requestHeading.textContent = 'Agent request';
  const requestPayload = document.createElement('pre');
  requestPayload.className = 'request-payload';
  requestPayload.textContent = prettyJson(exchange.request);
  requestSection.append(requestHeading, requestPayload);
  const responseSection = document.createElement('section');
  responseSection.className = 'exchange-section';
  const responseHeading = document.createElement('h3');
  responseHeading.textContent = 'AUA response';
  const responsePayload = document.createElement('pre');
  responsePayload.className = 'response-payload';
  responsePayload.textContent = prettyJson(exchange.response);
  responseSection.append(responseHeading, responsePayload);
  panel.append(requestSection, responseSection);
  if (note) {
    const noteElement = document.createElement('div');
    noteElement.className = 'detail-note';
    noteElement.textContent = note;
    panel.appendChild(noteElement);
  }
}
async function loadEventExchange(e, panel) {
  if (!e.detail_id) {
    renderExchange(
      panel,
      storedExchange(e),
      'This older event predates full detail capture; showing the compact journaled payload.'
    );
    return;
  }
  panel.textContent = 'Loading full request and response…';
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
function prependEvent(e) {
  const key = e.detail_id || ((e.ts_ms || 0) + ':' + (e.cmd || '') + ':' + (e.source || '') + ':' + (e.pid || ''));
  if (seenKeys.has(key)) return;
  seenKeys.add(key);
  const empty = journalEl.querySelector('.empty');
  if (empty) empty.remove();
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
  addEventText(summary, 'args', argsSummary(e.args));
  if (e.duration_ms != null) addEventText(summary, 'dur', e.duration_ms + 'ms');
  if (!ok && e.error) addEventText(summary, 'err', errText(e.error));
  const exchange = document.createElement('div');
  exchange.className = 'exchange';
  exchange.textContent = 'Expand to load the full request and response.';
  let loaded = false;
  details.addEventListener('toggle', () => {
    if (!details.open || loaded) return;
    loaded = true;
    loadEventExchange(e, exchange);
  });
  details.append(summary, exchange);
  li.appendChild(details);
  journalEl.insertBefore(li, journalEl.firstChild);
  while (journalEl.children.length > 200) journalEl.removeChild(journalEl.lastChild);
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
      '</div>';
    tiles.appendChild(a);
  }
  a.querySelector('.ser').textContent = d.serial;
  const own = a.querySelector('.owner');
  if (d.owner) { own.textContent = 'owner ' + d.owner; own.classList.remove('hidden'); }
  else { own.textContent = ''; }
  const cap = a.querySelector('.cap');
  cap.textContent = d.capture_running ? 'live' : (d.has_frame ? 'frame' : 'idle');
  cap.className = 'pill cap ' + (d.capture_running ? 'ok' : '');
  a.querySelector('.age').textContent = fmtAge(d.frame_age_ms);
  a.querySelector('.pkg').textContent = d.package ? ('pkg ' + d.package) : '';
  const img = a.querySelector('img');
  if (d.has_frame) {
    const src = '/api/frame.jpg?serial=' + encodeURIComponent(d.serial) + '&t=' + Date.now();
    if (tileSrc[d.serial] !== src.slice(0, src.indexOf('&t='))) {
      img.src = src;
      tileSrc[d.serial] = src.slice(0, src.indexOf('&t='));
    }
  }
  return a;
}

async function tickGrid() {
  try {
    const r = await fetch('/api/devices', {cache: 'no-store'});
    const d = await r.json();
    const list = d.devices || [];
    document.getElementById('count').textContent = list.length + ' device' + (list.length === 1 ? '' : 's');
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

async function tickStatus() {
  try {
    const r = await fetch('/api/status' + qSerial(), {cache: 'no-store'});
    const s = await r.json();
    document.getElementById('serial').textContent = s.serial || '—';
    const cap = document.getElementById('capture');
    cap.textContent = s.capture_running ? 'capture on' : 'capture off';
    cap.className = 'pill ' + (s.capture_running ? 'ok' : 'bad');
    document.getElementById('via').textContent = 'via ' + (s.via || '—');
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

    if (s.has_frame) {
      const src = '/api/frame.jpg' + qSerial({t: Date.now()});
      if (src !== lastSrc) { frame.src = src; lastSrc = src; }
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
    const evs = d.events || [];
    evs.forEach(e => {
      if (e.ts_ms && e.ts_ms >= sinceMs) sinceMs = e.ts_ms + 1;
      prependEvent(e);
    });
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

async function tickLogcat() {
  try {
    const r = await fetch('/api/logcat' + qSerial({lines: 80}), {cache: 'no-store'});
    const d = await r.json();
    document.getElementById('logcat').textContent = (d.lines || []).join('\\n') || '(empty)';
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
        self._fallback: dict[str, tuple[bytes, float]] = {}
        self._fallback_lock = threading.Lock()
        self._pkg_cache: dict[str, tuple[str | None, float]] = {}
        self._map_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self.database_token = secrets.token_urlsafe(32)
        self._database_lock = threading.Lock()

    @property
    def serial(self) -> str | None:
        return self.focus

    def _ensure_via(self, serial: str) -> str | None:
        ens = self.ensures.get(serial) or {}
        via = ens.get("via")
        return str(via) if via else None

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
        return {"events": events, "stats": stats}

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
        if via == "daemon":
            detail = self._daemon_call(serial, "capture_status")
            if isinstance(detail, dict):
                capture_running = bool(detail.get("running")) and not detail.get("paused")
        pkg = None
        with contextlib.suppress(Exception):
            pkg = self.foreground_package(serial)
        return {
            "serial": serial,
            "owner": owner_for_serial(self.cache_dir, serial),
            "package": pkg,
            "via": via,
            "capture_running": capture_running,
            "has_frame": frame is not None or serial in self._fallback,
            "frame_age_ms": age_ms,
        }

    def devices_payload(self) -> dict[str, Any]:
        known = list(self.serials)
        if self.mode == "grid":
            # Grid dashboards intentionally discover devices as they appear. A
            # detail dashboard stays scoped to the serial it was started for.
            online = list_online_serials(self.config)
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
        detail: dict[str, Any] | None = None
        try:
            if via == "sidecar":
                from . import capture_sidecar as cs

                sock = cs.socket_path(self.cache_dir)
                if Path(sock).exists():
                    detail = cs.call(sock, "status")
                    capture_running = bool(detail.get("running")) and not detail.get("paused")
                    session_id = detail.get("session_id") or session_id
            elif via == "daemon":
                detail = self._daemon_call(ser, "capture_status")
                if isinstance(detail, dict):
                    capture_running = bool(detail.get("running")) and not detail.get("paused")
                    session_id = detail.get("session_id") or session_id
        except Exception as exc:  # noqa: BLE001
            detail = {"error": str(exc)}

        pkg = None
        with contextlib.suppress(Exception):
            pkg = self.foreground_package(ser)

        stats: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            stats = self.journal_bundle(ser, limit=1).get("stats") or {}

        ens = self.ensures.get(ser) or {}
        return {
            "ok": True,
            "serial": ser,
            "owner": owner_for_serial(self.cache_dir, ser),
            "via": via,
            "package": pkg,
            "capture_running": capture_running,
            "has_frame": frame is not None or ser in self._fallback,
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

    def frame_bytes(self, serial: str | None = None) -> tuple[bytes, str]:
        ser = serial or self.focus
        if not ser:
            placeholder = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
                b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return placeholder, "image/png"
        path = latest_frame(self.cache_dir, ser)
        if path is not None and path.is_file():
            return path.read_bytes(), "image/jpeg"
        with self._fallback_lock:
            hit = self._fallback.get(ser)
            if hit and (time.time() - hit[1]) < 0.8:
                return hit[0], "image/jpeg"
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
                with self._fallback_lock:
                    self._fallback[ser] = (data, time.time())
                mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
                return data, mime
        except Exception as exc:  # noqa: BLE001
            logger.debug("fallback screencap failed: %s", exc)
        placeholder = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return placeholder, "image/png"

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
            prefix = "/api/database/"
            if not parsed.path.startswith(prefix):
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
                    raise UsageError("dashboard database request body must be a JSON object")
                action = parsed.path[len(prefix) :]
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

    With multiple online devices (or ``grid=True``), serves a tile grid of live
    frames. Pass ``serial`` (or open ``/?serial=…``) for the full detail view.
    """
    from .config import load_config

    cfg = config or load_config()
    targets = resolve_dashboard_targets(
        serial or getattr(cfg.device, "serial", None), grid=grid, config=cfg
    )
    mode = str(targets["mode"])
    serials: list[str] = list(targets["serials"])
    focus: str | None = targets.get("focus")
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
    handler = _make_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", listen), handler)
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
        "hint": (
            (
                f"Grid of {len(serials)} device(s). Click a tile for detail. "
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
    return info
