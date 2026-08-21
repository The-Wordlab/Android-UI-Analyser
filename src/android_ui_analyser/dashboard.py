"""Local sneak-peek dashboard for headless (or headed) agent runs.

A separate process from the agent: ``aua dashboard`` enables capture if needed
(daemon or sidecar), then serves a localhost HTML page that live-polls frames,
the agent I/O journal, app map, logcat, and capture marks. Bind 127.0.0.1 only.
"""

from __future__ import annotations

import contextlib
import http.cookies
import io
import json
import logging
import mimetypes
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

from .errors import AuaError, DaemonOutcomeUnknownError, UsageError

logger = logging.getLogger(__name__)

# One deliberately fixed port for the detached dashboard. Service mode never walks to a
# neighbouring port: a bookmark on a phone must remain valid, and an occupied port must not be
# mistaken for a healthy AUA dashboard. Foreground ``run()`` retains its legacy nearby-port
# behaviour unless ``exact_port=True`` is requested.
DEFAULT_DASHBOARD_PORT = 48765
_DEFAULT_PORT = DEFAULT_DASHBOARD_PORT
_SERVICE_ID = "aua-dashboard-v1"
_SERVICE_STATE_NAME = "dashboard-service.json"
_SERVICE_LOG_NAME = "dashboard.log"
_SERVICE_START_TIMEOUT_S = 5.0
_ACCESS_COOKIE = "AUA_DASHBOARD_ACCESS"

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
_SECRET_BODY_KEYS = (
    "token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "session_id",
    "cookie",
)

_REDACTED = "<redacted>"


def _redact_headers(headers: Any) -> Any:
    if not isinstance(headers, dict):
        return headers
    return {k: (_REDACTED if str(k).lower() in _SECRET_HEADERS else v) for k, v in headers.items()}


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


def _dashboard_step_payload(step: Any) -> dict[str, Any]:
    """Human-inspectable route/flow step without screenshot-prone private values."""

    raw = step.model_dump(exclude_none=True) if hasattr(step, "model_dump") else dict(step)
    for field in ("origin_package", "capture_segment", "capture_order", "arrival_proof"):
        raw.pop(field, None)
    if "text" in raw:
        raw["text"] = _REDACTED
    if isinstance(raw.get("data"), dict):
        raw["data"] = _redact_body(raw["data"])
    if isinstance(raw.get("substeps"), list):
        raw["substeps"] = [_dashboard_step_payload(item) for item in raw["substeps"]]
    return raw


# url -> (server, thread) for dashboards started with block=False, so callers that
# do not own the serve loop can still stop one.
_SERVERS: dict[str, tuple[ThreadingHTTPServer, threading.Thread]] = {}

# 1x1 black PNG - served when a device has neither a capture file nor a screencap.
_PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

_DASHBOARD_LOGO = Path(__file__).parent / "data" / "aua-dashboard-logo.png"


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


def _emulator_meta_for_serial(cache_dir: str | Path, serial: str) -> dict[str, Any] | None:
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


def _service_state_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / _SERVICE_STATE_NAME


def _service_log_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / _SERVICE_LOG_NAME


def _read_service_state(cache_dir: str | Path) -> dict[str, Any]:
    path = _service_state_path(cache_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_service_state(cache_dir: str | Path, value: dict[str, Any]) -> Path:
    """Write dashboard ownership and its LAN credential as a user-private file."""
    path = _service_state_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


def _dashboard_health(port: int) -> dict[str, Any] | None:
    try:
        with urlopen(f"http://127.0.0.1:{int(port)}/api/health", timeout=0.35) as response:
            payload = json.loads(response.read())
    except (HTTPError, URLError, OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("service") != _SERVICE_ID:
        return None
    return payload


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def _lan_addresses() -> list[str]:
    """Best-effort private/LAN IPv4 addresses suitable for a phone URL."""
    found: set[str] = set()
    with contextlib.suppress(OSError):
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = str(item[4][0])
            if address and not address.startswith("127.") and address != "0.0.0.0":
                found.add(address)
    # Hostnames on macOS do not always resolve to the active Wi-Fi interface. Connecting a UDP
    # socket chooses a route but sends no packet, and gives us the source address for that route.
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("192.0.2.1", 9))
        address = str(sock.getsockname()[0])
        if address and not address.startswith("127.") and address != "0.0.0.0":
            found.add(address)
    return sorted(found)


def _service_urls(*, port: int, lan: bool, access_token: str | None) -> dict[str, Any]:
    local_url = f"http://127.0.0.1:{int(port)}/"
    hosts = _lan_addresses() if lan else []
    lan_urls = [f"http://{host}:{int(port)}/" for host in hosts]
    token_suffix = f"?token={access_token}" if access_token else ""
    return {
        "url": local_url,
        "access_url": local_url + token_suffix,
        "lan_urls": lan_urls,
        "lan_access_urls": [url + token_suffix for url in lan_urls],
    }


def _qr_svg(value: str) -> bytes:
    try:
        import qrcode
        import qrcode.image.svg
    except ModuleNotFoundError as exc:
        raise UsageError(
            "dashboard QR support is not installed",
            hint="Refresh AUA with `uv tool install --force --editable .` from the repository.",
        ) from exc

    code = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
        image_factory=qrcode.image.svg.SvgPathFillImage,
    )
    code.add_data(value)
    code.make(fit=True)
    image = code.make_image()
    return image.to_string(encoding="utf-8")


def _qr_png(value: str) -> bytes:
    try:
        import qrcode
    except ModuleNotFoundError as exc:
        raise UsageError(
            "dashboard QR support is not installed",
            hint="Refresh AUA with `uv tool install --force --editable .` from the repository.",
        ) from exc

    code = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    code.add_data(value)
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def create_access_qr(
    config: Any,
    *,
    port: int = DEFAULT_DASHBOARD_PORT,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Write a user-private QR image for the running dashboard's authenticated LAN URL."""
    status = service_status(config, port=port)
    if not status.get("running"):
        raise UsageError("dashboard is not running", hint="Run `aua dashboard start --lan`.")
    urls = status.get("lan_access_urls") or []
    if not urls:
        raise UsageError(
            "dashboard is local-only, so it has no phone URL",
            hint="Run `aua dashboard stop`, then `aua dashboard start --lan`.",
        )
    access_url = str(urls[0])
    path = (
        Path(output).expanduser()
        if output
        else Path(config.cache.dir).expanduser() / "dashboard-access.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _qr_png(access_url)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return {
        "ok": True,
        "action": "dashboard-qr",
        "port": int(port),
        "url": access_url,
        "path": str(path.resolve()),
    }


def service_status(
    config: Any, *, port: int = DEFAULT_DASHBOARD_PORT
) -> dict[str, Any]:
    """Return detached-dashboard status without adopting an arbitrary port owner."""
    cache = Path(config.cache.dir).expanduser()
    state = _read_service_state(cache)
    health = _dashboard_health(port)
    if health is not None:
        lan = bool(health.get("lan"))
        token = str(state.get("access_token") or "") if lan else ""
        return {
            "ok": True,
            "action": "dashboard-status",
            "running": True,
            "status": "running",
            "pid": health.get("pid"),
            "port": int(port),
            "bind": health.get("bind"),
            "lan": lan,
            "authenticated": bool(health.get("authenticated")),
            **_service_urls(port=port, lan=lan, access_token=token or None),
        }
    occupied = _port_is_open(port)
    return {
        "ok": not occupied,
        "action": "dashboard-status",
        "running": False,
        "status": "port_occupied" if occupied else "not_running",
        "port": int(port),
        "hint": (
            f"Port {port} is occupied by a process that is not the AUA dashboard."
            if occupied
            else "Start it with `aua dashboard start`."
        ),
    }


def start_service(
    config: Any,
    *,
    serial: str | None = None,
    port: int = DEFAULT_DASHBOARD_PORT,
    lan: bool = False,
    poll_ms: int = 500,
    grid: bool = True,
    explicit_config: str | None = None,
    profile: str | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """Start one idempotent detached dashboard on an exact localhost/LAN port."""
    current = service_status(config, port=port)
    if current.get("running"):
        if bool(current.get("lan")) != bool(lan):
            raise UsageError(
                "dashboard is already running with a different network scope",
                hint="Run `aua dashboard stop`, then start it again with or without `--lan`.",
            )
        return {**current, "action": "dashboard-start", "status": "already_running"}
    if current.get("status") == "port_occupied":
        raise UsageError(
            f"dashboard port {port} is already owned by another process",
            hint=f"Stop that process or choose one dedicated port with `--port`; AUA will not move away from {port} automatically.",
        )

    cache = Path(config.cache.dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32) if lan else ""
    initial_state = {
        "service": _SERVICE_ID,
        "pid": None,
        "port": int(port),
        "bind": "0.0.0.0" if lan else "127.0.0.1",
        "lan": bool(lan),
        "access_token": token,
        "started_at_ms": int(time.time() * 1000),
    }
    state_path = _write_service_state(cache, initial_state)
    log_path = _service_log_path(cache)
    cmd = [
        sys.executable,
        "-m",
        "android_ui_analyser.dashboard",
        "--serve-service",
        "--state-file",
        str(state_path),
        "--port",
        str(int(port)),
        "--bind",
        "0.0.0.0" if lan else "127.0.0.1",
        "--poll-ms",
        str(max(200, int(poll_ms))),
        "--cache-dir",
        str(cache),
    ]
    cmd.append("--grid" if grid else "--detail")
    if serial:
        cmd += ["--serial", serial]
    if explicit_config:
        cmd += ["--config", str(Path(explicit_config).expanduser().resolve())]
    if profile:
        cmd += ["--profile", profile]
    if platform:
        cmd += ["--platform", platform]

    with open(log_path, "a", encoding="utf-8") as log_fh:  # noqa: SIM115
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
        )
    initial_state["pid"] = proc.pid
    _write_service_state(cache, initial_state)

    deadline = time.monotonic() + _SERVICE_START_TIMEOUT_S
    while time.monotonic() < deadline:
        health = _dashboard_health(port)
        if health is not None:
            return {
                "ok": True,
                "action": "dashboard-start",
                "running": True,
                "status": "started",
                "pid": proc.pid,
                "port": int(port),
                "bind": initial_state["bind"],
                "lan": bool(lan),
                "authenticated": bool(lan),
                "log": str(log_path),
                **_service_urls(port=port, lan=lan, access_token=token or None),
            }
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(proc.pid, signal.SIGTERM)
    raise UsageError(
        "dashboard service did not become ready",
        hint=f"See {log_path}.",
    )


def stop_service(config: Any, *, port: int = DEFAULT_DASHBOARD_PORT) -> dict[str, Any]:
    """Stop only the dashboard process proven by both health and its private state file."""
    cache = Path(config.cache.dir).expanduser()
    health = _dashboard_health(port)
    if health is None:
        occupied = _port_is_open(port)
        return {
            "ok": not occupied,
            "action": "dashboard-stop",
            "running": False,
            "status": "port_occupied" if occupied else "not_running",
            "port": int(port),
        }
    state = _read_service_state(cache)
    pid = health.get("pid")
    if not isinstance(pid, int) or state.get("pid") != pid or state.get("port") != int(port):
        raise UsageError(
            "refusing to stop a dashboard without matching ownership state",
            hint=f"Expected private state at {_service_state_path(cache)}.",
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError) as exc:
        raise UsageError(f"could not stop dashboard process {pid}: {exc}") from exc
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _dashboard_health(port) is not None:
        time.sleep(0.1)
    running = _dashboard_health(port) is not None
    if not running:
        with contextlib.suppress(OSError):
            _service_state_path(cache).unlink()
    return {
        "ok": not running,
        "action": "dashboard-stop",
        "running": running,
        "status": "stopping" if running else "stopped",
        "pid": pid,
        "port": int(port),
    }


def open_service(config: Any, *, port: int = DEFAULT_DASHBOARD_PORT) -> dict[str, Any]:
    status = service_status(config, port=port)
    if not status.get("running"):
        raise UsageError(
            "dashboard is not running",
            hint="Start it with `aua dashboard start`.",
        )
    candidates = status.get("lan_access_urls") or [status["access_url"]]
    target = str(candidates[0])
    opened = bool(webbrowser.open(target))
    return {**status, "action": "dashboard-open", "opened": opened, "opened_url": target}


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AuA Dashboard</title>
<link rel="icon" type="image/png" href="/assets/aua-dashboard-logo.png"/>
<style>
  :root {
    --bg: #080a12;
    --bg-raised: #0d1020;
    --panel: rgba(18, 22, 39, 0.88);
    --panel2: rgba(24, 29, 50, 0.9);
    --text: #f2f4ff;
    --muted: #8d96b2;
    --faint: #626b87;
    --accent: #63e6be;
    --accent-2: #8c7bff;
    --border: rgba(142, 157, 211, 0.18);
    --border-strong: rgba(142, 157, 211, 0.32);
    --danger: #ff7b8e;
    --warn: #f5c76b;
    --wide: 1900px;
    --shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    --tok-key: #7fd3ff;
    --tok-str: #b7e08a;
    --tok-num: #f0b26a;
    --tok-bool: #d59bf6;
    --tok-dim: #6d7686;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    background:
      radial-gradient(circle at 8% -10%, rgba(140, 123, 255, 0.18), transparent 32rem),
      radial-gradient(circle at 94% 6%, rgba(99, 230, 190, 0.09), transparent 26rem),
      linear-gradient(145deg, #080a12 0%, #0b0e1a 52%, #090b14 100%);
    color: var(--text); min-height: 100vh; letter-spacing: 0.005em;
  }
  header {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 0.85rem clamp(1rem, 3vw, 2.4rem); border-bottom: 1px solid var(--border);
    background: rgba(10, 13, 25, 0.78); backdrop-filter: blur(22px);
    position: sticky; top: 0; z-index: 2; box-shadow: 0 10px 36px rgba(0, 0, 0, 0.16);
  }
  .header-brand, .header-actions { display: flex; align-items: center; gap: 0.7rem; min-width: 0; }
  header h1 { font-size: 1rem; font-weight: 750; margin: 0; letter-spacing: -0.02em; }
  header h1 span { display: block; color: var(--muted); font-size: 0.62rem; font-weight: 550; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.12rem; }
  .brand-mark { width: 2.35rem; height: 2.35rem; display: block; object-fit: contain; filter: drop-shadow(0 0 12px rgba(99, 230, 190, 0.28)); }
  .header-title { margin-right: auto; }
  .header-live { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--accent); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
  .header-live::before { content: ''; width: 0.42rem; height: 0.42rem; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 0.24rem rgba(99, 230, 190, 0.12), 0 0 0.8rem var(--accent); }
  .phone-qr-button {
    display: inline-flex; align-items: center; gap: 0.38rem; min-height: 2.15rem;
    padding: 0.4rem 0.62rem; color: var(--text); background: rgba(140,123,255,0.1);
    border: 1px solid rgba(140,123,255,0.34); border-radius: 9px; cursor: pointer;
    font: 700 0.65rem ui-sans-serif, system-ui;
  }
  .phone-qr-button:hover { color: var(--accent); border-color: rgba(99,230,190,0.5); }
  .phone-qr-dialog {
    width: min(390px, calc(100vw - 2rem)); padding: 0; overflow: hidden;
    color: var(--text); background: #111626; border: 1px solid var(--border-strong);
    border-radius: 18px; box-shadow: 0 28px 90px rgba(0,0,0,0.62);
  }
  .phone-qr-dialog::backdrop { background: rgba(3,5,11,0.78); backdrop-filter: blur(7px); }
  .phone-qr-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; padding: 1rem 1.1rem 0; }
  .phone-qr-head h2 { margin: 0; font-size: 0.95rem; }
  .phone-qr-head p { margin: 0.28rem 0 0; color: var(--muted); font-size: 0.68rem; line-height: 1.45; }
  .phone-qr-close { color: var(--muted); background: transparent; border: 0; cursor: pointer; font-size: 1.25rem; }
  .phone-qr-image { display: block; width: min(78vw, 290px); aspect-ratio: 1; object-fit: contain; margin: 1rem auto 0.75rem; padding: 0.55rem; background: #fff; border-radius: 13px; }
  .phone-qr-url { display: block; margin: 0 1rem; padding: 0.62rem; overflow-wrap: anywhere; color: var(--accent); background: rgba(3,6,14,0.72); border: 1px solid var(--border); border-radius: 9px; font: 0.58rem/1.45 ui-monospace, monospace; }
  .phone-qr-actions { display: flex; justify-content: flex-end; gap: 0.5rem; padding: 0.8rem 1rem 1rem; }
  header a.back {
    display: inline-flex; align-items: center; gap: 0.4rem; color: var(--muted);
    text-decoration: none; font-size: 0.72rem; font-weight: 650; white-space: nowrap;
    padding: 0.42rem 0.62rem; margin-right: 0.2rem; border: 1px solid var(--border);
    border-radius: 9px; background: rgba(255,255,255,0.025); transition: 0.15s ease;
  }
  header a.back:hover { color: var(--accent); border-color: rgba(99,230,190,0.4); background: rgba(99,230,190,0.07); }
  .pill {
    font-size: 0.68rem; padding: 0.28rem 0.58rem; border-radius: 999px;
    border: 1px solid var(--border); color: var(--muted); white-space: nowrap;
    background: rgba(255,255,255,0.025);
  }
  .pill.ok { color: var(--accent); border-color: #2a6b4f; }
  .pill.bad { color: var(--danger); border-color: #7a3a35; }
  .detail-overview {
    display: grid; grid-template-columns: minmax(230px, 0.75fr) minmax(0, 2fr); gap: 1rem;
    align-items: stretch; padding: 1.15rem 0.85rem 0; max-width: var(--wide); margin: 0 auto;
  }
  .detail-device, .detail-health {
    background: linear-gradient(150deg, rgba(25,30,52,0.82), rgba(13,17,31,0.86));
    border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 14px 40px rgba(0,0,0,0.2);
  }
  .detail-device { display: flex; flex-direction: column; justify-content: center; padding: 1rem 1.1rem; min-width: 0; position: relative; overflow: hidden; }
  .detail-device::before { content: ''; position: absolute; inset: 0 auto 0 0; width: 2px; background: linear-gradient(var(--accent), var(--accent-2)); }
  .detail-eyebrow { color: var(--accent); font-size: 0.6rem; font-weight: 750; letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.35rem; }
  .detail-serial-row { display: flex; align-items: center; gap: 0.5rem; min-width: 0; }
  .detail-device-dot { width: 0.48rem; height: 0.48rem; flex: 0 0 auto; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0.75rem var(--accent); }
  #serial { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); font: 750 1rem ui-monospace, SFMono-Regular, Menlo, monospace; }
  #pkg { margin-top: 0.42rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font: 0.68rem ui-monospace, SFMono-Regular, Menlo, monospace; }
  .detail-health { display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); overflow: hidden; }
  .detail-status { min-width: 0; padding: 0.85rem 0.9rem; border-left: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center; gap: 0.3rem; }
  .detail-status:first-child { border-left: 0; }
  .detail-status-label { color: var(--faint); font-size: 0.56rem; font-weight: 750; letter-spacing: 0.1em; text-transform: uppercase; }
  .detail-status-value { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); font-size: 0.7rem; font-weight: 650; }
  .detail-status-value.ok { color: var(--accent); }
  .detail-status-value.bad { color: var(--danger); }
  @media (max-width: 1250px) {
    .detail-overview { grid-template-columns: 1fr; }
    .detail-health { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
    .detail-status:nth-child(4) { border-left: 0; }
    .detail-status:nth-child(n+4) { border-top: 1px solid var(--border); }
  }
  @media (max-width: 650px) {
    .detail-health { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .detail-status:nth-child(odd) { border-left: 0; }
    .detail-status:nth-child(n+3) { border-top: 1px solid var(--border); }
  }
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
    background: linear-gradient(150deg, rgba(25, 30, 52, 0.92), rgba(14, 18, 32, 0.9));
    border: 1px solid var(--border); border-radius: 18px;
    padding: 1rem 1.1rem; min-height: 0; box-shadow: var(--shadow);
  }
  .panel h2 {
    font-size: 0.69rem; margin: 0 0 0.7rem; color: var(--muted); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
  }
  .stage-heading { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.7rem; }
  .stage-heading h2 { margin: 0; }
  .analyze-button {
    display: inline-flex; align-items: center; gap: 0.42rem; border-color: rgba(106, 232, 194, 0.42);
    color: #d9fff4; background: rgba(42, 126, 108, 0.2); font-weight: 760;
  }
  .analyze-button::before { content: '⌗'; color: var(--accent); font-size: 0.86rem; }
  .analyze-button:hover { border-color: var(--accent); background: rgba(42, 126, 108, 0.32); }
  .analyze-button:disabled { opacity: 0.62; cursor: wait; }
  .frame-shell { position: relative; display: block; width: fit-content; max-width: 100%; }
  .stage img {
    display: block; height: min(74vh, 880px); width: auto;
    min-width: 210px; max-width: min(38vw, 470px);
    object-fit: contain; background: #030408; border-radius: 13px; border: 1px solid var(--border-strong);
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.42);
  }
  .element-overlay { position: absolute; inset: 0; overflow: hidden; border-radius: 13px; pointer-events: none; }
  .element-overlay.busy .element-box { pointer-events: none; opacity: 0.5; }
  .element-box {
    position: absolute; display: block; min-width: 0; min-height: 0; padding: 0;
    border: 1px solid rgba(116, 176, 255, 0.72); border-radius: 3px;
    background: rgba(65, 126, 220, 0.055); color: #fff; pointer-events: auto; cursor: pointer;
    box-shadow: inset 0 0 0 1px rgba(3, 7, 18, 0.28); transition: 0.12s ease;
  }
  .element-box.clickable { border-color: rgba(106, 232, 194, 0.94); background: rgba(49, 205, 157, 0.09); }
  .element-box:hover, .element-box:focus-visible { z-index: 1000 !important; outline: none; border-width: 2px; background: rgba(116, 176, 255, 0.22); box-shadow: 0 0 0 2px rgba(4, 8, 18, 0.72), 0 0 14px rgba(90, 163, 255, 0.55); }
  .element-box.clickable:hover, .element-box.clickable:focus-visible { background: rgba(49, 205, 157, 0.2); box-shadow: 0 0 0 2px rgba(4, 8, 18, 0.72), 0 0 14px rgba(106, 232, 194, 0.55); }
  .element-label {
    position: absolute; top: -1px; left: -1px; display: inline-flex; align-items: stretch;
    max-width: 8rem; padding: 0; border: 0; border-radius: 3px 0 5px 0; overflow: hidden;
    color: inherit; background: transparent; pointer-events: auto; cursor: zoom-in;
    box-shadow: 0 1px 4px rgba(1, 5, 12, 0.42);
  }
  .element-label:hover, .element-label:focus-visible { outline: 1px solid #fff; outline-offset: 1px; }
  .element-id {
    flex: 0 0 auto; min-width: 1.25rem; padding: 0.11rem 0.28rem;
    color: #06120e; background: var(--accent);
    font: 800 0.56rem ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1;
  }
  .element-box:not(.clickable) .element-id { color: #07101d; background: #74b0ff; }
  .element-key {
    min-width: 0; max-width: 6.35rem; padding: 0.12rem 0.24rem; overflow: hidden;
    color: rgba(218, 231, 250, 0.82); background: rgba(6, 12, 24, 0.9);
    font: 650 0.42rem ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .element-overlay.clickable-only .element-box:not(.clickable) { display: none; }
  .inspection-status { margin-top: 0.55rem; min-height: 1rem; color: var(--muted); font-size: 0.66rem; line-height: 1.4; }
  .inspection-status.bad { color: var(--danger); }
  .inspection-output { margin-top: 0.65rem; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: rgba(3, 6, 14, 0.52); }
  .inspection-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 0.7rem; padding: 0.48rem 0.58rem; border-bottom: 1px solid var(--border); }
  .inspection-count { color: var(--accent); font: 720 0.62rem ui-monospace, SFMono-Regular, Menlo, monospace; }
  .inspection-filter { display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.61rem; cursor: pointer; }
  .inspection-raw summary { padding: 0.55rem 0.62rem; color: var(--muted); cursor: pointer; font-size: 0.64rem; font-weight: 720; }
  .inspection-json-tools { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; align-items: center; gap: 0.35rem; padding: 0.48rem 0.58rem; border-top: 1px solid var(--border); background: rgba(8, 13, 25, 0.72); }
  .inspection-json-search { position: relative; min-width: 0; }
  .inspection-json-search .db-input { width: 100%; padding-right: 4rem; background: rgba(3, 7, 15, 0.82); }
  .inspection-json-search span { position: absolute; right: 0.5rem; top: 50%; transform: translateY(-50%); color: var(--faint); font: 0.56rem ui-monospace, SFMono-Regular, Menlo, monospace; pointer-events: none; }
  .inspection-raw pre { max-width: min(38vw, 470px); max-height: 23rem; margin: 0; padding: 0.7rem; overflow: auto; border-top: 1px solid var(--border); color: #c9d6ef; background: rgba(2, 4, 10, 0.72); font: 0.61rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
  .inspection-json-line { display: block; min-height: 1.5em; margin: 0 -0.35rem; padding: 0 0.35rem; border-left: 2px solid transparent; }
  .inspection-json-line.element-object { background: rgba(116, 176, 255, 0.025); }
  .inspection-json-line.search-match { background: rgba(255, 205, 92, 0.1); }
  .inspection-json-line.search-current { background: rgba(255, 205, 92, 0.25); border-left-color: #ffcd5c; }
  .inspection-json-line.element-current { background: rgba(106, 232, 194, 0.16); border-left-color: var(--accent); }
  @media (max-width: 980px) {
    .stage img { width: 100%; height: auto; max-width: 100%; max-height: 62vh; }
    .inspection-raw pre { max-width: 100%; }
  }
  .meta { font-size: 0.75rem; color: var(--muted); display: flex; gap: 0.9rem; flex-wrap: wrap; margin-top: 0.4rem; }
  /* `contain` keeps a wheel gesture inside the panel it started in: without it, hitting
     the end of the journal scrolled the whole page out from under the reader. */
  .scroll { overflow: auto; max-height: 68vh; overscroll-behavior: contain; }
  .scroll.sm { max-height: 14rem; }
  .scroll.md { max-height: 18rem; }
  .panel.journal { display: flex; flex-direction: column; min-width: 0; padding: 0; overflow: hidden; }
  .journal-tools {
    display: grid; grid-template-columns: minmax(170px, auto) minmax(240px, 1fr) auto;
    align-items: center; gap: 0.75rem; padding: 0.8rem 0.9rem;
    border-bottom: 1px solid var(--border); background: rgba(10,14,26,0.72);
  }
  .journal-title { display: flex; align-items: center; gap: 0.6rem; min-width: 0; }
  .journal-title h2 { margin: 0; white-space: nowrap; }
  .journal-title #journal-shown { margin: 0; min-height: 0; white-space: nowrap; font-size: 0.65rem; }
  .journal-search { position: relative; min-width: 0; }
  .journal-search::before { content: '⌕'; position: absolute; left: 0.62rem; top: 50%; transform: translateY(-52%); color: var(--faint); font-size: 0.9rem; pointer-events: none; }
  .journal-search .db-input { width: 100%; padding: 0.45rem 2.3rem 0.45rem 1.75rem; background: rgba(4,7,15,0.76); }
  .journal-search kbd { position: absolute; right: 0.55rem; top: 50%; transform: translateY(-50%); color: var(--faint); border: 1px solid var(--border); border-radius: 5px; padding: 0.05rem 0.3rem; font: 0.58rem ui-monospace, monospace; pointer-events: none; }
  .journal-actions { display: flex; align-items: center; gap: 0.4rem; }
  .journal-toggle { display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); font-size: 0.66rem; white-space: nowrap; cursor: pointer; }
  .journal-toggle input {
    appearance: none; width: 1.8rem; height: 1rem; margin: 0; padding: 0.1rem;
    border: 1px solid var(--border-strong); border-radius: 999px; background: rgba(4,7,15,0.76);
    cursor: pointer; transition: 0.15s ease;
  }
  .journal-toggle input::after {
    content: ''; display: block; width: 0.68rem; height: 0.68rem; border-radius: 50%;
    background: var(--muted); transition: transform 0.15s ease, background 0.15s ease;
  }
  .journal-toggle input:checked { background: rgba(255,123,142,0.16); border-color: rgba(255,123,142,0.5); }
  .journal-toggle input:checked::after { transform: translateX(0.78rem); background: var(--danger); }
  .journal-button-group { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .journal-button-group .db-button { border: 0; border-radius: 0; border-left: 1px solid var(--border); }
  .journal-button-group .db-button:first-child { border-left: 0; }
  .logcat-tools {
    display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap;
    margin-bottom: 0.45rem;
  }
  .logcat-tools h2 { margin: 0 0.35rem 0 0; }
  .logcat-tools .grow { flex: 1 1 8rem; min-width: 0; }
  .journal-tools .db-button, .logcat-tools .db-button {
    padding: 0.32rem 0.5rem; font-size: 0.64rem;
  }
  .logcat-tools .db-input, .logcat-tools .db-select {
    padding: 0.24rem 0.45rem; font-size: 0.74rem;
  }
  .logcat-tools .logcat-app-filter {
    flex: 0 1 20rem; min-width: 13rem;
    color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .logcat-tools .pill {
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
    border-radius: 999px; padding: 0.3rem 0.62rem; font: 700 0.64rem inherit;
    cursor: pointer; white-space: nowrap;
  }
  #journal li.filtered { display: none; }
  #journal, #marks, #fail-list, #slow { list-style: none; margin: 0; padding: 0; font-size: 0.78rem; }
  #journal { display: grid; gap: 0.38rem; padding: 0.55rem; }
  #journal li {
    padding: 0; border: 1px solid rgba(142,157,211,0.13); border-radius: 11px;
    background: rgba(8,11,21,0.34); overflow: clip; transition: border-color 0.15s, background 0.15s;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #marks li, #fail-list li, #slow li {
    padding: 0.4rem 0.35rem; border-bottom: 1px solid var(--border);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #journal li:hover { border-color: rgba(99,230,190,0.28); background: rgba(99,230,190,0.025); }
  #journal li.fail { background: rgba(255,123,142,0.065); border-color: rgba(255,123,142,0.28); }
  #journal .t, #marks .t, #fail-list .t, #slow .t {
    color: var(--muted);
  }
  #journal .badge {
    display: inline-flex; justify-content: center; min-width: 2.7rem; font-size: 0.56rem;
    font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.18rem 0.35rem; border-radius: 6px;
  }
  #journal .badge.ok { color: var(--accent); background: rgba(61,220,132,0.12); }
  #journal .badge.fail { color: var(--danger); background: rgba(239,107,90,0.15); }
  #journal .event-main { min-width: 0; display: grid; gap: 0.18rem; }
  #journal .cmd { color: var(--text); font-weight: 700; font-size: 0.73rem; }
  #journal .args { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.65rem; }
  #journal .dur { color: var(--warn); padding: 0.22rem 0.4rem; border-radius: 6px; background: rgba(245,199,107,0.08); font-size: 0.62rem; font-variant-numeric: tabular-nums; }
  #journal .dur.slow { color: var(--danger); background: rgba(255,123,142,0.1); }
  #journal .err { color: var(--danger); display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.64rem; }
  #journal details > summary { display: grid; grid-template-columns: auto 5.8rem auto minmax(0,1fr) auto; align-items: center; gap: 0.55rem; padding: 0.62rem 0.7rem; cursor: pointer; line-height: 1.3; list-style: none; }
  #journal details > summary::-webkit-details-marker { display: none; }
  #journal .event-chevron { color: var(--faint); font: 1rem ui-sans-serif, system-ui; transition: transform 0.15s, color 0.15s; }
  #journal details[open] .event-chevron { color: var(--accent); transform: rotate(90deg); }
  #journal details[open] > summary {
    position: sticky; top: 0; z-index: 4;
    background: linear-gradient(100deg, #1a2035, #14192b);
    border-bottom: 1px solid var(--border-strong);
    box-shadow: 0 8px 20px rgba(2,4,10,0.34);
  }
  #journal .exchange {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.65rem;
    padding: 0.7rem; border: 0; border-radius: 0; background: rgba(5,8,16,0.62);
  }
  #journal .exchange-section { min-width: 0; border: 1px solid var(--border); border-radius: 9px; overflow: hidden; background: rgba(8,11,21,0.52); }
  #journal .exchange-section h3 {
    margin: 0; color: var(--accent); font: 700 0.62rem ui-sans-serif, system-ui;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  #journal .exchange-head { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); }
  #journal .exchange pre {
    /* One scroll surface per panel. A capped, independently scrollable pre nested inside
       the scrollable journal meant a wheel gesture went to whichever of the two the
       pointer happened to be over, and reaching the end of a payload stalled dead. */
    margin: 0; max-height: none; overflow: visible; padding: 0.7rem;
    border-radius: 0; background: rgba(3,5,11,0.56); color: #cdd2db; font: inherit;
    font-size: 0.7rem; line-height: 1.45; white-space: pre-wrap; overflow-wrap: anywhere;
    user-select: text;
  }
  #journal .exchange > .detail-note { grid-column: 1 / -1; padding: 0.35rem 0.2rem 0; }
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
  @media (max-width: 1200px) {
    .journal-tools { grid-template-columns: 1fr; }
    .journal-actions { flex-wrap: wrap; }
    #journal .exchange { grid-template-columns: minmax(0, 1fr); }
  }
  @media (max-width: 720px) {
    #journal details > summary { grid-template-columns: auto auto minmax(0,1fr) auto; }
    #journal .t { display: none; }
  }
  .lower {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem; padding: 0 0.85rem 0.85rem; max-width: var(--wide); margin: 0 auto;
  }
  .lower.wide { grid-template-columns: minmax(0, 1fr); }
  .lower.summary-row { grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr); }
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
  .knowledge-workspace { max-width: var(--wide); margin: 0 auto; padding: 0 0.85rem 0.85rem; }
  .knowledge-panel { padding: 0; overflow: hidden; }
  .knowledge-head {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 1rem 1.1rem; border-bottom: 1px solid var(--border); background: rgba(10,14,26,0.58);
  }
  .knowledge-head h2 { margin: 0 0 0.25rem; color: var(--text); font-size: 0.78rem; }
  .knowledge-head p { margin: 0; color: var(--muted); font-size: 0.7rem; }
  .knowledge-head code { color: var(--accent); }
  .knowledge-head-actions, .knowledge-column-controls, .knowledge-actions {
    display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap;
  }
  .knowledge-head-actions { justify-content: flex-end; }
  .knowledge-column-controls { justify-content: flex-end; min-width: 0; }
  .knowledge-head .db-button, .knowledge-column-head .db-button, .knowledge-actions .db-button {
    padding: 0.3rem 0.5rem; font-size: 0.61rem;
  }
  .knowledge-action-status {
    display: none; margin: 0; padding: 0.55rem 1rem; border-bottom: 1px solid var(--border);
    color: var(--muted); background: rgba(4, 7, 15, 0.56); font-size: 0.66rem;
  }
  .knowledge-action-status.visible { display: block; }
  .knowledge-action-status.ok { color: var(--accent); }
  .knowledge-action-status.bad { color: var(--danger); }
  .knowledge-action-result { margin: 0; border-bottom: 1px solid var(--border); background: rgba(3,5,11,0.72); }
  .knowledge-action-result summary { padding: 0.45rem 1rem; color: var(--muted); cursor: pointer; font-size: 0.62rem; }
  .knowledge-action-result pre { margin: 0; max-height: 20rem; padding: 0.65rem 1rem; overflow: auto; color: #cdd2db; font: 0.6rem/1.45 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .knowledge-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .knowledge-column { min-width: 0; padding: 1rem; }
  .knowledge-column + .knowledge-column { border-left: 1px solid var(--border); }
  .knowledge-column-head { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin-bottom: 0.9rem; }
  .knowledge-column-head h3 { margin: 0.18rem 0 0; font-size: 1rem; }
  .knowledge-eyebrow { color: var(--accent); font-size: 0.58rem; font-weight: 750; letter-spacing: 0.11em; text-transform: uppercase; }
  .knowledge-package { max-width: 58%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font: 0.65rem ui-monospace, SFMono-Regular, Menlo, monospace; }
  .knowledge-section-head { display: flex; align-items: center; justify-content: space-between; margin: 0.7rem 0 0.4rem; }
  .knowledge-section-head h4 { margin: 0; color: var(--text); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em; }
  .knowledge-section-head span { color: var(--faint); font-size: 0.62rem; }
  .knowledge-section-head.routes-head { margin-top: 1rem; }
  .knowledge-list { display: grid; gap: 0.4rem; max-height: 24rem; overflow: auto; overscroll-behavior: contain; scrollbar-color: rgba(140,123,255,0.45) transparent; scrollbar-width: thin; }
  .flow-list { max-height: 52rem; }
  .knowledge-item { border: 1px solid rgba(142,157,211,0.15); border-radius: 10px; background: rgba(7,10,20,0.42); overflow: hidden; }
  .knowledge-item > summary { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; padding: 0.65rem 0.7rem; cursor: pointer; list-style: none; }
  .knowledge-item > summary::-webkit-details-marker { display: none; }
  .knowledge-item > summary:hover { background: rgba(99,230,190,0.035); }
  .knowledge-item[open] > summary { border-bottom: 1px solid var(--border); background: rgba(140,123,255,0.05); }
  .knowledge-summary-main { min-width: 0; display: grid; gap: 0.18rem; }
  .knowledge-summary-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); font: 700 0.7rem ui-monospace, SFMono-Regular, Menlo, monospace; }
  .knowledge-summary-subtitle { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font-size: 0.62rem; }
  .knowledge-badges { display: flex; align-items: center; justify-content: flex-end; gap: 0.3rem; flex: 0 1 auto; flex-wrap: wrap; }
  .knowledge-badge { padding: 0.18rem 0.36rem; border-radius: 6px; color: var(--muted); background: rgba(255,255,255,0.04); border: 1px solid var(--border); font-size: 0.56rem; white-space: nowrap; }
  .knowledge-badge.ok { color: var(--accent); border-color: rgba(99,230,190,0.26); }
  .knowledge-badge.bad { color: var(--danger); border-color: rgba(255,123,142,0.26); }
  .knowledge-detail { display: grid; gap: 0.65rem; padding: 0.7rem; }
  .knowledge-command { display: flex; align-items: center; gap: 0.55rem; padding: 0.5rem 0.6rem; border-radius: 8px; background: rgba(4,7,15,0.74); border: 1px solid var(--border); }
  .knowledge-command span { color: var(--faint); font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
  .knowledge-command code { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--accent); font-size: 0.66rem; }
  .knowledge-json { margin: 0; padding: 0.65rem; border-radius: 8px; background: rgba(3,5,11,0.72); color: #cdd2db; font: 0.62rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .knowledge-steps { display: grid; gap: 0.35rem; margin: 0; padding: 0; list-style: none; }
  .knowledge-step { display: grid; grid-template-columns: 1.6rem minmax(0,1fr); gap: 0.45rem; padding: 0.5rem; border-radius: 8px; border: 1px solid rgba(142,157,211,0.12); background: rgba(255,255,255,0.018); }
  .knowledge-step-index { color: var(--faint); font: 0.6rem ui-monospace, monospace; }
  .knowledge-step-main { min-width: 0; display: grid; gap: 0.2rem; }
  .knowledge-step-kind { color: var(--accent); font: 700 0.65rem ui-monospace, monospace; }
  .knowledge-step-data { color: var(--muted); font: 0.6rem/1.45 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .flow-group { display: grid; gap: 0.4rem; }
  .flow-group + .flow-group { margin-top: 0.9rem; }
  .flow-group-title { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; padding: 0 0.15rem; }
  .flow-group-title h4 { margin: 0; max-width: 80%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font: 700 0.63rem ui-monospace, monospace; }
  .flow-group-title span { color: var(--faint); font-size: 0.58rem; }
  @media (max-width: 1100px) {
    .knowledge-grid { grid-template-columns: minmax(0, 1fr); }
    .knowledge-column + .knowledge-column { border-left: 0; border-top: 1px solid var(--border); }
    .flow-list { max-height: 36rem; }
  }
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
    width: min(520px, calc(100% - 2rem)); margin: clamp(3rem, 12vh, 8rem) auto;
    padding: clamp(1.8rem, 5vw, 2.7rem); color: var(--muted); text-align: center;
    background: linear-gradient(145deg, rgba(25,30,52,0.9), rgba(12,16,29,0.94));
    border: 1px solid var(--border-strong); border-radius: 22px; box-shadow: var(--shadow);
    position: relative; overflow: hidden;
  }
  .grid-empty::before {
    content: ''; position: absolute; inset: 0 0 auto; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), transparent);
  }
  .grid-empty.bad { border-color: rgba(255,123,142,0.38); }
  .grid-empty.bad::before { background: linear-gradient(90deg, transparent, var(--danger), transparent); }
  .grid-empty-logo {
    width: 4.2rem; height: 4.2rem; object-fit: contain; margin-bottom: 0.9rem;
    filter: drop-shadow(0 0 18px rgba(99,230,190,0.25));
  }
  .grid-empty-kicker {
    margin: 0 0 0.45rem; color: var(--accent); font-size: 0.66rem; font-weight: 750;
    letter-spacing: 0.14em; text-transform: uppercase;
  }
  .grid-empty h2 { margin: 0; color: var(--text); font-size: clamp(1.25rem, 4vw, 1.65rem); letter-spacing: -0.025em; }
  .grid-empty-copy { margin: 0.75rem auto 1.15rem; max-width: 38rem; font-size: 0.86rem; line-height: 1.65; }
  .grid-empty-command {
    display: inline-flex; align-items: center; padding: 0.58rem 0.85rem; color: var(--accent);
    background: rgba(4,8,16,0.7); border: 1px solid rgba(99,230,190,0.24);
    border-radius: 10px; font: 600 0.78rem ui-monospace, SFMono-Regular, Menlo, monospace;
    box-shadow: inset 0 1px rgba(255,255,255,0.04);
  }
  .grid-empty-foot { margin: 1rem 0 0; color: var(--faint); font-size: 0.7rem; }
  .grid-empty.bad .grid-empty-kicker, .grid-empty.bad h2 { color: var(--danger); }
  .device-notice {
    width: min(760px, calc(100% - 2rem)); margin: 1rem auto 0; padding: 0.72rem 0.9rem;
    color: #ffd9df; text-align: center; background: rgba(132, 42, 58, 0.24);
    border: 1px solid rgba(255,123,142,0.38); border-radius: 11px; font-size: 0.72rem;
  }
  .device-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 340px));
    justify-content: center; align-items: start;
    gap: 1.25rem; padding: clamp(1.4rem, 4vw, 3rem); max-width: 1600px; margin: 0 auto;
  }
  .tile {
    background: #02040a; border: 1px solid var(--border-strong); border-radius: 20px;
    cursor: pointer; text-decoration: none; color: inherit; display: block; overflow: hidden;
    box-shadow: 0 24px 70px rgba(0,0,0,0.42);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .tile:hover {
    border-color: rgba(99,230,190,0.64); transform: translateY(-5px);
    box-shadow: 0 30px 80px rgba(0,0,0,0.52), 0 0 0 1px rgba(99,230,190,0.12);
  }
  .tile-screen { position: relative; aspect-ratio: 9 / 20; overflow: hidden; background: #02040a; }
  .tile-screen::before, .tile-screen::after {
    content: ''; position: absolute; z-index: 1; inset-inline: 0; pointer-events: none;
  }
  .tile-screen::before { top: 0; height: 24%; background: linear-gradient(to bottom, rgba(3,6,14,0.88), transparent); }
  .tile-screen::after { bottom: 0; height: 52%; background: linear-gradient(to top, rgba(3,6,14,0.98) 14%, rgba(3,6,14,0.76) 58%, transparent); }
  .tile img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.35s ease; }
  .tile:hover img { transform: scale(1.012); }
  .tile-overlay { position: absolute; z-index: 2; left: 0; right: 0; padding: 1rem; }
  .tile-overlay-top { top: 0; display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
  .tile-identity { min-width: 0; display: flex; align-items: center; gap: 0.5rem; }
  .tile-device-dot { width: 0.48rem; height: 0.48rem; flex: 0 0 auto; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0.75rem var(--accent); }
  .tile .ser {
    min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font: 700 0.78rem ui-monospace, SFMono-Regular, Menlo, monospace; color: #fff;
    text-shadow: 0 1px 8px rgba(0,0,0,0.8);
  }
  .tile .cap {
    flex: 0 0 auto; padding: 0.28rem 0.5rem; border-radius: 999px;
    color: var(--muted); background: rgba(8,11,21,0.72); border: 1px solid rgba(255,255,255,0.16);
    font-size: 0.62rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase;
    backdrop-filter: blur(12px);
  }
  .tile .cap.ok { color: var(--accent); border-color: rgba(99,230,190,0.36); }
  .tile-overlay-bottom { bottom: 0; }
  .tile-app-label { display: block; color: rgba(255,255,255,0.52); font-size: 0.6rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; margin-bottom: 0.25rem; }
  .tile .pkg {
    display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: #fff; font: 600 0.74rem ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .tile-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin-top: 0.8rem; }
  .tile-stat { min-width: 0; padding: 0.55rem 0.6rem; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; background: rgba(12,16,29,0.64); backdrop-filter: blur(12px); }
  .tile-stat-label { display: block; color: rgba(255,255,255,0.46); font-size: 0.56rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.2rem; }
  .tile-stat-value { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: rgba(255,255,255,0.86); font-size: 0.66rem; }
  .tile-stat-value.held { color: var(--accent); }
  .tile-stat-value.down { color: var(--danger); }
  .tile-card-footer { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-top: 0.8rem; padding-top: 0.7rem; border-top: 1px solid rgba(255,255,255,0.1); }
  .tile .owner { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: rgba(255,255,255,0.52); font-size: 0.62rem; }
  .tile-inspect { flex: 0 0 auto; color: var(--accent); font-size: 0.66rem; font-weight: 750; }
  @media (max-width: 520px) {
    .device-grid { grid-template-columns: minmax(0, 360px); padding: 1rem; }
  }
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
    color: var(--text); background: rgba(5, 8, 17, 0.72); border: 1px solid var(--border);
    border-radius: 9px; padding: 0.48rem 0.58rem; font: inherit; min-width: 0;
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
    color: var(--text); background: rgba(255,255,255,0.045); border: 1px solid var(--border);
    border-radius: 9px; padding: 0.48rem 0.72rem; cursor: pointer; font: inherit;
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  }
  .db-button:hover:not(:disabled) { border-color: var(--accent); background: rgba(99,230,190,0.09); transform: translateY(-1px); }
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
  .model-workspace { max-width: var(--wide); margin: 0 auto 1rem; padding: 0 0.85rem; }
  .model-panel { padding: 0; overflow: hidden; }
  .model-head {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 1rem 1.1rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(100deg, rgba(21,27,48,0.96), rgba(13,18,34,0.92));
  }
  .model-head h2 { margin: 0 0 0.25rem; color: var(--text); font-size: 0.82rem; }
  .model-head p { margin: 0; color: var(--muted); font-size: 0.68rem; }
  .model-master { display: flex; align-items: center; gap: 0.75rem; }
  .model-master-copy { text-align: right; }
  .model-master-copy strong { display: block; font-size: 0.7rem; }
  .model-master-copy span { color: var(--muted); font-size: 0.6rem; }
  .model-switch {
    appearance: none; width: 3.2rem; height: 1.65rem; margin: 0; padding: 0.18rem;
    border: 1px solid var(--border-strong); border-radius: 999px; background: #080b14;
    cursor: pointer; transition: 0.18s ease;
  }
  .model-switch::after {
    content: ''; display: block; width: 1.18rem; height: 1.18rem; border-radius: 50%;
    background: var(--muted); transition: 0.18s ease;
  }
  .model-switch:checked { background: rgba(99,230,190,0.18); border-color: var(--accent); }
  .model-switch:checked::after { transform: translateX(1.5rem); background: var(--accent); box-shadow: 0 0 15px rgba(99,230,190,0.48); }
  .model-cards { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 0.75rem; padding: 0.9rem; }
  .model-card {
    min-width: 0; padding: 0.85rem; border: 1px solid var(--border); border-radius: 13px;
    background: radial-gradient(circle at top right, rgba(140,123,255,0.1), transparent 44%), rgba(7,10,20,0.54);
  }
  .model-card.busy { border-color: rgba(245,199,107,0.55); box-shadow: inset 0 0 24px rgba(245,199,107,0.05); }
  .model-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.7rem; }
  .model-card h3 { margin: 0; font-size: 0.9rem; }
  .model-card-sub { margin-top: 0.2rem; color: var(--muted); font-size: 0.62rem; }
  .model-state { padding: 0.24rem 0.45rem; border-radius: 7px; border: 1px solid var(--border); color: var(--muted); font: 700 0.58rem ui-monospace,monospace; text-transform: uppercase; }
  .model-state.ready { color: var(--accent); border-color: rgba(99,230,190,0.35); }
  .model-state.busy { color: var(--warn); border-color: rgba(245,199,107,0.4); }
  .model-state.bad { color: var(--danger); border-color: rgba(255,123,142,0.4); }
  .model-metrics { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 0.45rem; margin: 0.75rem 0; }
  .model-metric { padding: 0.5rem; border: 1px solid rgba(142,157,211,0.12); border-radius: 9px; background: rgba(3,6,13,0.46); }
  .model-metric span { display: block; color: var(--faint); font-size: 0.54rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .model-metric strong { display: block; margin-top: 0.18rem; overflow: hidden; text-overflow: ellipsis; color: var(--text); font: 0.67rem ui-monospace,monospace; }
  .model-card-actions { display: flex; align-items: center; gap: 0.45rem; }
  .model-card-actions .journal-toggle { margin-left: auto; }
  .model-body { display: grid; grid-template-columns: minmax(320px,0.82fr) minmax(0,1.18fr); border-top: 1px solid var(--border); }
  .model-lab, .model-monitor { min-width: 0; padding: 0.9rem; }
  .model-monitor { border-left: 1px solid var(--border); }
  .model-section-head { display: flex; align-items: center; justify-content: space-between; gap: 0.7rem; margin-bottom: 0.7rem; }
  .model-section-head h3 { margin: 0; font-size: 0.75rem; }
  .model-chat { height: 20rem; overflow: auto; display: grid; align-content: start; gap: 0.5rem; padding: 0.55rem; border: 1px solid var(--border); border-radius: 10px; background: rgba(3,5,11,0.58); }
  .model-message { max-width: 92%; padding: 0.55rem 0.65rem; border-radius: 10px; white-space: pre-wrap; overflow-wrap: anywhere; font: 0.66rem/1.45 ui-monospace,monospace; }
  .model-message.user { justify-self: end; color: var(--text); background: rgba(140,123,255,0.15); border: 1px solid rgba(140,123,255,0.24); }
  .model-message.assistant { justify-self: start; color: #d7fdeb; background: rgba(99,230,190,0.08); border: 1px solid rgba(99,230,190,0.2); }
  .model-message.meta { max-width: 100%; color: var(--muted); background: transparent; padding: 0.2rem; }
  .model-compose { display: grid; gap: 0.5rem; margin-top: 0.6rem; }
  .model-compose textarea { min-height: 5rem; resize: vertical; }
  .model-compose-row { display: flex; align-items: end; gap: 0.5rem; }
  .model-sample-row { display: grid; grid-template-columns: minmax(150px,0.7fr) minmax(220px,1.3fr); gap: 0.5rem; }
  .model-prompt-guide { margin: -0.12rem 0 0; color: var(--muted); font-size: 0.61rem; line-height: 1.45; }
  .model-prompt-guide code { color: var(--accent); font-family: ui-monospace,monospace; }
  .model-traces { height: 29rem; overflow: auto; display: grid; align-content: start; gap: 0.45rem; }
  .model-trace { border: 1px solid var(--border); border-radius: 10px; overflow: clip; background: rgba(5,8,16,0.55); }
  .model-trace > summary { display: grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 0.55rem; align-items: center; padding: 0.58rem 0.65rem; cursor: pointer; list-style: none; }
  .model-trace > summary::-webkit-details-marker { display: none; }
  .model-trace-title { min-width: 0; display: grid; gap: 0.12rem; }
  .model-trace-title strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font: 0.68rem ui-monospace,monospace; }
  .model-trace-title span { color: var(--muted); font-size: 0.58rem; }
  .model-trace-metrics { color: var(--warn); font: 0.58rem ui-monospace,monospace; white-space: nowrap; }
  .model-trace-body { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 0.55rem; padding: 0.6rem; border-top: 1px solid var(--border); }
  .model-trace-pane { min-width: 0; }
  .model-trace-pane h4 { margin: 0 0 0.3rem; color: var(--accent); font-size: 0.56rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .model-trace-pane pre { margin: 0; max-height: 20rem; overflow: auto; padding: 0.55rem; border-radius: 8px; background: #05070d; color: #cdd2db; font: 0.6rem/1.45 ui-monospace,monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  @media (max-width: 1050px) {
    .model-body, .model-cards { grid-template-columns: minmax(0,1fr); }
    .model-monitor { border-left: 0; border-top: 1px solid var(--border); }
  }
  .hidden { display: none !important; }

  /* --- visual system: quiet depth, clear hierarchy, and generous reading surfaces --- */
  .layout, .lower { gap: 1rem; }
  .layout { padding-top: 1.35rem; }
  .lower { padding-bottom: 1rem; }
  .stage, .journal, .lower > .panel, .proxy-workspace > .panel, .database-workspace > .panel, .model-workspace > .panel {
    position: relative; overflow: hidden;
  }
  .stage::before, .journal::before, .proxy-workspace > .panel::before, .database-workspace > .panel::before, .model-workspace > .panel::before {
    content: ''; position: absolute; inset: 0 0 auto; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,230,190,0.72), rgba(140,123,255,0.62), transparent);
    opacity: 0.8;
  }
  .stage h2::before, .journal-tools h2::before, .logcat-tools h2::before {
    content: ''; display: inline-block; width: 0.42rem; height: 0.42rem; margin: 0 0.42rem 0.08rem 0;
    border-radius: 50%; background: var(--accent); box-shadow: 0 0 0.7rem rgba(99,230,190,0.8);
  }
  .meta span { padding: 0.3rem 0.48rem; border: 1px solid var(--border); border-radius: 7px; background: rgba(255,255,255,0.025); }
  #marks li, #fail-list li, #slow li { padding: 0.55rem 0.5rem; border-bottom-color: rgba(142,157,211,0.11); }
  #marks li:hover, #fail-list li:hover, #slow li:hover { background: rgba(140,123,255,0.06); }
  #journal .exchange pre, #logcat, .rule-body pre { background: rgba(3,5,12,0.7); }
  .scroll, .logcat-scroll, .db-table-wrap { scrollbar-color: rgba(140,123,255,0.45) transparent; scrollbar-width: thin; }
  .scroll::-webkit-scrollbar, .logcat-scroll::-webkit-scrollbar, .db-table-wrap::-webkit-scrollbar { width: 7px; height: 7px; }
  .scroll::-webkit-scrollbar-thumb, .logcat-scroll::-webkit-scrollbar-thumb, .db-table-wrap::-webkit-scrollbar-thumb { background: rgba(140,123,255,0.42); border-radius: 999px; }
  .proxy-workspace, .database-workspace, .model-workspace { margin-bottom: 1rem; }
  .proxy-grid, .db-grid { gap: 1rem; }
  .db-subpanel { border-radius: 12px; background: rgba(7,10,20,0.36); }
  .flow-table th { background: rgba(30,36,61,0.9); }
  footer { padding: 0.7rem clamp(1rem, 3vw, 2.4rem) 1.5rem; text-align: center; color: var(--faint); }
  @media (max-width: 700px) {
    html, body { max-width: 100%; overflow-x: clip; }
    body { padding-bottom: env(safe-area-inset-bottom); }
    header {
      padding: calc(0.58rem + env(safe-area-inset-top)) 0.7rem 0.58rem;
      gap: 0.45rem;
    }
    .brand-mark { width: 2rem; height: 2rem; }
    header h1 { font-size: 0.88rem; }
    header h1 span { display: none; }
    .header-live { font-size: 0.58rem; }
    .phone-qr-button { min-width: 2.5rem; justify-content: center; padding: 0.48rem; }
    .phone-qr-button span { display: none; }
    header a.back { padding: 0.48rem; margin: 0; min-height: 2.5rem; }
    header a.back span { display: none; }
    header .pill { order: 3; }
    .detail-overview { gap: 0.5rem; padding: 0.55rem 0.5rem 0; }
    .detail-device { padding: 0.65rem 0.8rem; border-radius: 12px; }
    .detail-eyebrow, #pkg { display: none; }
    #serial { font-size: 0.78rem; }
    .detail-health { grid-template-columns: repeat(2, minmax(0, 1fr)); border-radius: 12px; }
    .detail-status { display: none; padding: 0.58rem 0.7rem; border-top: 0 !important; }
    .detail-status:nth-child(1), .detail-status:nth-child(3) { display: flex; }
    .detail-status:nth-child(3) { border-left: 1px solid var(--border); }
    .layout { gap: 0.6rem; padding: 0.55rem 0.5rem 0.7rem; }
    .lower, .knowledge-workspace, .proxy-workspace, .database-workspace, .model-workspace {
      padding-left: 0.5rem; padding-right: 0.5rem;
    }
    .lower.summary-row { grid-template-columns: minmax(0, 1fr); }
    .panel { border-radius: 13px; padding: 0.78rem; }
    .stage { padding: 0.62rem; }
    .stage-heading { margin-bottom: 0.52rem; }
    .stage-heading h2 { font-size: 0.62rem; }
    .frame-shell { margin-inline: auto; }
    .stage img {
      width: auto; height: auto; min-width: 0; max-width: 100%;
      max-height: calc(100svh - 12.5rem); border-radius: 10px;
    }
    .element-overlay { border-radius: 10px; }
    .element-label { max-width: 6.5rem; }
    .element-id { min-width: 1.4rem; padding: 0.16rem 0.32rem; font-size: 0.62rem; }
    .element-key { display: none; }
    button, a, summary, input, select, textarea { touch-action: manipulation; }
    .db-button, .analyze-button { min-height: 2.6rem; padding: 0.55rem 0.72rem; }
    .db-input, input, select, textarea { font-size: 16px; }
    .inspection-toolbar { align-items: flex-start; flex-direction: column; }
    .inspection-toolbar > div { width: 100%; justify-content: space-between; }
    .inspection-json-tools { grid-template-columns: minmax(0, 1fr) 2.6rem 2.6rem; }
    .inspection-raw pre { max-height: 52svh; font-size: 0.66rem; }
    .journal-tools { padding: 0.7rem; }
    .journal-actions, .journal-button-group { width: 100%; }
    .journal-actions { justify-content: space-between; }
    .journal-button-group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    #journal details > summary { gap: 0.38rem; padding: 0.72rem 0.58rem; }
    #journal .exchange { padding: 0.5rem; }
    .scroll { max-height: 72svh; }
    .knowledge-head, .knowledge-column-head { align-items: flex-start; flex-direction: column; }
    .knowledge-head-actions, .knowledge-column-controls { width: 100%; justify-content: flex-start; }
    .knowledge-package { max-width: 100%; }
    .knowledge-item > summary { align-items: flex-start; }
    .knowledge-badges { max-width: 48%; }
    .logcat-tools .logcat-app-filter { flex-basis: 100%; min-width: 0; }
    .flow-table { table-layout: fixed; }
    .flow-table th:nth-child(1), .flow-table td:nth-child(1),
    .flow-table th:nth-child(5), .flow-table td:nth-child(5) { display: none; }
    .flow-table th:nth-child(2), .flow-table td:nth-child(2) { width: 4.4rem; }
    .flow-table th:nth-child(4), .flow-table td:nth-child(4) { width: 3.8rem; }
    .flow-table .upath { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .proxy-form .db-field.grow { flex-basis: 100%; width: 100%; }
    .db-toolbar { display: grid; grid-template-columns: minmax(0, 1fr); align-items: stretch; }
    .db-toolbar > *, .db-toolbar .db-select { width: 100%; min-width: 0; }
    .db-actions .db-button { flex: 1 1 10rem; }
    .db-table-wrap { max-width: 100%; overscroll-behavior-inline: contain; }
    table.db-results { min-width: 34rem; }
    .model-head { align-items: stretch; flex-direction: column; }
    .model-master { width: 100%; justify-content: space-between; }
    .model-master-copy { text-align: left; }
    .model-section-head { align-items: flex-start; flex-direction: column; }
    .model-sample-row, .model-trace-body { grid-template-columns: minmax(0, 1fr); }
    .model-compose-row {
      display: grid; grid-template-columns: minmax(0, 1fr) minmax(6.5rem, 0.42fr);
      align-items: end;
    }
    .model-compose-row .db-field,
    .model-compose-row .db-select,
    .model-compose-row .db-input { width: 100%; min-width: 0; }
    .model-compose-row .db-button { grid-column: 1 / -1; width: 100%; }
    .model-trace > summary { grid-template-columns: auto minmax(0, 1fr); }
    .model-trace-metrics {
      grid-column: 2; justify-self: start; max-width: 100%; white-space: normal;
    }
    .model-chat { height: 52svh; }
    .logcat-scroll { height: 62svh; }
  }
  @media (max-height: 520px) and (orientation: landscape) {
    .detail-overview { display: none; }
    .layout { grid-template-columns: auto minmax(0, 1fr); align-items: start; }
    .stage img { height: calc(100svh - 5.4rem); width: auto; max-width: 42vw; }
    .scroll { max-height: calc(100svh - 5.4rem); }
  }
</style>
</head>
<body>
<script nonce="__DATABASE_TOKEN__">
// Some mobile browsers do not persist a cookie set on an HTTP redirect from a QR URL.
// The token-bearing request is therefore served directly, then scrubbed from browser history.
if (new URLSearchParams(window.location.search).has('token')) {
  const cleanAccessUrl = new URL(window.location.href);
  cleanAccessUrl.searchParams.delete('token');
  window.history.replaceState(null, '', cleanAccessUrl.pathname + cleanAccessUrl.search + cleanAccessUrl.hash);
}
</script>
<header>
  <div class="header-brand">
    <a id="back" class="back hidden" href="/">← <span>All devices</span></a>
    <img class="brand-mark" src="/assets/aua-dashboard-logo.png" alt="" aria-hidden="true"/>
    <div class="header-title"><h1>AuA Dashboard <span>runtime observability</span></h1></div>
  </div>
  <div class="header-actions">
    <button id="phone-qr-button" class="phone-qr-button hidden" type="button" title="Connect a phone">▦ <span>Phone QR</span></button>
    <span class="header-live">live</span>
    <span id="count" class="pill hidden">0 devices</span>
  </div>
</header>

<dialog id="phone-qr-dialog" class="phone-qr-dialog">
  <div class="phone-qr-head">
    <div><h2>Open on your phone</h2><p>Scan while your phone is on the same trusted network.</p></div>
    <button id="phone-qr-close" class="phone-qr-close" type="button" aria-label="Close">×</button>
  </div>
  <img id="phone-qr-image" class="phone-qr-image" alt="Authenticated dashboard QR code"/>
  <code id="phone-qr-url" class="phone-qr-url"></code>
  <div class="phone-qr-actions"><button id="phone-qr-copy" class="db-button" type="button">Copy link</button></div>
</dialog>

<div id="grid-view" class="hidden">
  <div id="device-notice" class="device-notice hidden" role="status"></div>
  <section id="grid-empty" class="grid-empty hidden" aria-live="polite">
    <img class="grid-empty-logo" src="/assets/aua-dashboard-logo.png" alt="" aria-hidden="true"/>
    <p class="grid-empty-kicker">Device monitor</p>
    <h2 id="grid-empty-title">No devices online</h2>
    <p id="grid-empty-copy" class="grid-empty-copy">Start an emulator and it will appear here automatically.</p>
    <code id="grid-empty-command" class="grid-empty-command">aua emulator start</code>
    <p id="grid-empty-foot" class="grid-empty-foot">This dashboard keeps watching in the background.</p>
  </section>
  <div class="device-grid" id="tiles"></div>
</div>

<div id="detail-view" class="hidden">
<section class="detail-overview">
  <div class="detail-device">
    <span class="detail-eyebrow">Device details</span>
    <div class="detail-serial-row"><span class="detail-device-dot"></span><span id="serial">—</span></div>
    <span id="pkg">—</span>
  </div>
  <div class="detail-health">
    <div class="detail-status"><span class="detail-status-label">Capture</span><span id="capture" class="detail-status-value">—</span></div>
    <div class="detail-status"><span class="detail-status-label">Source</span><span id="via" class="detail-status-value">—</span></div>
    <div class="detail-status"><span class="detail-status-label">Lease</span><span id="lease" class="detail-status-value">—</span></div>
    <div class="detail-status"><span class="detail-status-label">Auto-stop</span><span id="watchdog" class="detail-status-value">—</span></div>
    <div class="detail-status"><span class="detail-status-label">Frame age</span><span id="age" class="detail-status-value">—</span></div>
    <div class="detail-status"><span class="detail-status-label">Failures</span><span id="failpill" class="detail-status-value">0</span></div>
  </div>
</section>
<div class="layout">
  <section class="panel stage">
    <div class="stage-heading">
      <h2>Live frame</h2>
      <button id="screen-analyze" class="db-button analyze-button" type="button">Analyze</button>
    </div>
    <div id="frame-shell" class="frame-shell">
      <img id="frame" alt="device frame" src=""/>
      <div id="element-overlay" class="element-overlay clickable-only" aria-label="AUA element bounds"></div>
    </div>
    <div id="inspection-status" class="inspection-status">Analyze to inspect AUA's raw response and element IDs.</div>
    <div id="inspection-output" class="inspection-output hidden">
      <div class="inspection-toolbar">
        <span id="inspection-count" class="inspection-count">0 elements</span>
        <div style="display:flex;align-items:center;gap:.5rem">
          <label class="inspection-filter"><input id="inspection-clickable-only" type="checkbox" checked/>Interactive only</label>
          <button id="inspection-live" class="db-button" type="button">Live</button>
        </div>
      </div>
      <details id="inspection-raw-details" class="inspection-raw" open>
        <summary>Raw AUA response</summary>
        <div class="inspection-json-tools">
          <div class="inspection-json-search">
            <input id="inspection-json-search" class="db-input" type="search"
                   placeholder="Search id, text, stable key…" aria-label="Search raw AUA response"
                   autocomplete="off" spellcheck="false"/>
            <span id="inspection-json-search-count">0</span>
          </div>
          <button id="inspection-json-prev" class="db-button" type="button" title="Previous match">↑</button>
          <button id="inspection-json-next" class="db-button" type="button" title="Next match">↓</button>
        </div>
        <pre id="inspection-raw" tabindex="0">{}</pre>
      </details>
    </div>
    <div class="meta">
      <span id="session">session —</span>
      <span id="fps">poll —</span>
    </div>
  </section>
  <aside class="panel journal">
    <div class="journal-tools">
      <div class="journal-title">
        <h2>Agent I/O journal</h2>
        <span id="journal-shown" class="db-status">—</span>
      </div>
      <div class="journal-search">
        <input id="journal-filter" class="db-input" type="search"
               placeholder="Search commands, arguments, or errors…"
               aria-label="Search journal" autocomplete="off" spellcheck="false"/>
        <kbd>/</kbd>
      </div>
    <div class="journal-actions">
      <label class="journal-toggle"><input id="journal-fails-only" type="checkbox"/>Failures only</label>
      <div class="journal-button-group">
        <button id="journal-expand" class="db-button" type="button">Expand visible</button>
        <button id="journal-collapse" class="db-button" type="button">Collapse all</button>
      </div>
      <button id="journal-clear" class="db-button danger" type="button">Clear logs</button>
      <button id="journal-jump" class="hidden" type="button">Newest ↑</button>
      </div>
    </div>
    <div class="scroll" id="journal-wrap">
      <ul id="journal"><li class="empty">waiting for events…</li></ul>
    </div>
  </aside>
</div>
<div class="lower summary-row">
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
    <h2>Capture marks</h2>
    <ul id="marks" class="scroll sm"><li class="empty">—</li></ul>
  </section>
</div>
<div class="knowledge-workspace">
  <section class="panel knowledge-panel">
    <div class="knowledge-head">
      <div>
        <h2>Navigation library</h2>
        <p>What AuA knows, what <code>goto</code> can target, and every saved flow available to agents.</p>
      </div>
      <div class="knowledge-head-actions">
        <span id="knowledge-total" class="pill">loading…</span>
        <button id="knowledge-clear-all" class="db-button danger" type="button">Clear all</button>
      </div>
    </div>
    <div id="knowledge-action-status" class="knowledge-action-status"></div>
    <details id="knowledge-action-result" class="knowledge-action-result hidden">
      <summary>Last navigation result</summary><pre>{}</pre>
    </details>
    <div class="knowledge-grid">
      <section class="knowledge-column">
        <div class="knowledge-column-head">
          <div><span class="knowledge-eyebrow">Current app</span><h3>App map</h3></div>
          <div class="knowledge-column-controls">
            <span id="map-pkg" class="knowledge-package">package —</span>
            <button id="map-clear" class="db-button danger" type="button" disabled>Clear map</button>
          </div>
        </div>
        <div class="knowledge-section-head"><h4>Screens</h4><span>Goto targets</span></div>
        <div id="map-screens" class="knowledge-list"><div class="empty">—</div></div>
        <div class="knowledge-section-head routes-head"><h4>Routes</h4><span>Recorded edges</span></div>
        <div id="map-routes" class="knowledge-list"><div class="empty">—</div></div>
      </section>
      <section class="knowledge-column">
        <div class="knowledge-column-head">
          <div><span class="knowledge-eyebrow">All apps</span><h3>Saved flows</h3></div>
          <div class="knowledge-column-controls">
            <span id="flow-count" class="knowledge-package">0 flows</span>
            <button id="flows-clear" class="db-button danger" type="button" disabled>Clear flows</button>
          </div>
        </div>
        <div id="flow-groups" class="knowledge-list flow-list"><div class="empty">—</div></div>
      </section>
    </div>
  </section>
</div>
<div class="lower wide">
  <section class="panel">
    <div class="logcat-tools">
      <h2>Logcat</h2>
      <input id="logcat-filter" class="db-input grow" placeholder="filter tag or message…"
             autocomplete="off" spellcheck="false"/>
      <input id="logcat-app-filter" class="db-input logcat-app-filter" type="search"
             placeholder="App ID / package…" aria-label="Filter Logcat by App ID"
             autocomplete="off" autocapitalize="none" spellcheck="false"/>
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
      <label class="pill"><input id="logcat-follow" type="checkbox" checked/>follow newest</label>
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

<div class="model-workspace" id="models">
  <section class="panel model-panel">
    <div class="model-head">
      <div>
        <h2>Local Model Control</h2>
        <p>Load, test, and watch the exact privacy-screened policy context seen by local models.</p>
      </div>
      <div class="model-master">
        <div class="model-master-copy">
          <strong id="model-intercept-label">Agent interception off</strong>
          <span>OFF discards in-flight results and restores deterministic AUA</span>
        </div>
        <input id="model-intercept" class="model-switch" type="checkbox"
               aria-label="Intercept policy-eligible agent decisions with local models"/>
      </div>
    </div>
    <div class="model-cards">
      <article id="model-card-functiongemma" class="model-card" data-provider="functiongemma">
        <div class="model-card-top">
          <div><h3>FunctionGemma v10</h3><div class="model-card-sub">Fast guarded candidate selector</div></div>
          <span class="model-state">checking</span>
        </div>
        <div class="model-metrics">
          <div class="model-metric"><span>Context</span><strong data-field="context">—</strong></div>
          <div class="model-metric"><span>Runtime</span><strong data-field="runtime">—</strong></div>
          <div class="model-metric"><span>Last call</span><strong data-field="latency">—</strong></div>
        </div>
        <div class="model-card-actions">
          <button class="db-button model-load" data-action="load">Load</button>
          <button class="db-button model-unload" data-action="unload">Unload</button>
          <label class="journal-toggle">enabled <input class="model-provider-toggle" type="checkbox" checked/></label>
        </div>
      </article>
      <article id="model-card-gemma4" class="model-card" data-provider="gemma4">
        <div class="model-card-top">
          <div><h3>Gemma 4</h3><div class="model-card-sub">Deep semantic policy reviewer</div></div>
          <span class="model-state">checking</span>
        </div>
        <div class="model-metrics">
          <div class="model-metric"><span>Context</span><strong data-field="context">—</strong></div>
          <div class="model-metric"><span>Runtime</span><strong data-field="runtime">—</strong></div>
          <div class="model-metric"><span>Last call</span><strong data-field="latency">—</strong></div>
        </div>
        <div class="model-card-actions">
          <button class="db-button model-load" data-action="load">Load</button>
          <button class="db-button model-unload" data-action="unload">Unload</button>
          <label class="journal-toggle">enabled <input class="model-provider-toggle" type="checkbox" checked/></label>
        </div>
      </article>
    </div>
    <div class="model-body">
      <section class="model-lab">
        <div class="model-section-head">
          <h3>Model playground</h3>
          <span id="model-chat-status" class="db-status">Resident daemon session</span>
        </div>
        <div id="model-chat" class="model-chat"><div class="model-message meta">Choose a model and send a message. FunctionGemma is selector-tuned, so its raw replies may be tool-shaped.</div></div>
        <div class="model-compose">
          <div class="model-sample-row">
            <label class="db-field">Request shape
              <select id="model-request-kind" class="db-select">
                <option value="direct">Direct message</option>
                <option value="agent">Agent request</option>
              </select>
            </label>
            <label class="db-field">Sample
              <select id="model-request-sample" class="db-select">
                <option value="">Choose a sample…</option>
                <option value="settings">Choose the matching control</option>
                <option value="next_step">Choose the next waypoint</option>
                <option value="handoff">No candidate matches → handoff</option>
              </select>
            </label>
          </div>
          <textarea id="model-prompt" class="db-sql" placeholder="Message the local model…" spellcheck="false"></textarea>
          <p id="model-prompt-guide" class="model-prompt-guide">Plain text sends a normal chat message and returns raw text.</p>
          <div class="model-compose-row">
            <label class="db-field grow"><span id="model-provider-label">Model</span>
              <select id="model-chat-provider" class="db-select">
                <option value="functiongemma">FunctionGemma v10</option>
                <option value="gemma4">Gemma 4</option>
                <option value="agent_chain" hidden>Configured agent chain</option>
              </select>
            </label>
            <label class="db-field">Max tokens
              <input id="model-max-tokens" class="db-input" type="number" min="1" max="1024" value="128" style="width:6.5rem"/>
            </label>
            <button id="model-send" class="db-button primary">Send</button>
          </div>
        </div>
      </section>
      <section class="model-monitor">
        <div class="model-section-head">
          <div><h3>Live model exchanges</h3><span id="model-trace-count" class="db-status">—</span></div>
          <button id="model-clear" class="db-button">Clear</button>
        </div>
        <div id="model-traces" class="model-traces"><div class="empty">No model activity yet.</div></div>
      </section>
    </div>
  </section>
</div>
<dialog id="knowledge-confirm-dialog" class="db-dialog">
  <h3 id="knowledge-confirm-title">Confirm action</h3>
  <p id="knowledge-confirm-message" class="db-note"></p>
  <div class="db-actions" style="justify-content:flex-end;margin-top:0.8rem">
    <button id="knowledge-confirm-cancel" class="db-button">Cancel</button>
    <button id="knowledge-confirm-submit" class="db-button danger">Confirm</button>
  </div>
</dialog>
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
const PHONE_ACCESS_URL = __PHONE_ACCESS_URL_JSON__;
const DATABASE_TOKEN = '__DATABASE_TOKEN__';
const params = new URLSearchParams(location.search);
const focusSerial = params.get('serial') || (BOOT_MODE === 'detail' ? BOOT_SERIAL : '');
const detachedSerial = params.get('detached') || '';
const isGrid = !focusSerial && BOOT_MODE === 'grid';

const gridView = document.getElementById('grid-view');
const detailView = document.getElementById('detail-view');
const back = document.getElementById('back');
if (isGrid) {
  gridView.classList.remove('hidden');
  document.getElementById('count').classList.remove('hidden');
} else {
  detailView.classList.remove('hidden');
  if (BOOT_MODE === 'grid' || params.get('from') === 'grid') {
    back.classList.remove('hidden');
  }
}

const frame = document.getElementById('frame');
const screenAnalyze = document.getElementById('screen-analyze');
const elementOverlay = document.getElementById('element-overlay');
const inspectionStatus = document.getElementById('inspection-status');
const inspectionOutput = document.getElementById('inspection-output');
const inspectionCount = document.getElementById('inspection-count');
const inspectionRaw = document.getElementById('inspection-raw');
const inspectionRawDetails = document.getElementById('inspection-raw-details');
const inspectionJsonSearch = document.getElementById('inspection-json-search');
const inspectionJsonSearchCount = document.getElementById('inspection-json-search-count');
const inspectionClickableOnly = document.getElementById('inspection-clickable-only');
const knowledgeActionStatus = document.getElementById('knowledge-action-status');
const knowledgeActionResult = document.getElementById('knowledge-action-result');
const knowledgeConfirmDialog = document.getElementById('knowledge-confirm-dialog');
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
let currentInspectionId = '';
let inspectionFrameActive = false;
let inspectionBusy = false;
let inspectionJsonMatches = [];
let inspectionJsonMatchIndex = -1;
let knowledgeBusy = false;
let pendingKnowledgeConfirmation = null;

function qSerial(extra) {
  const p = new URLSearchParams(extra || {});
  if (focusSerial) p.set('serial', focusSerial);
  const s = p.toString();
  return s ? ('?' + s) : '';
}

async function inspectionPost(action, payload) {
  const body = Object.assign({}, payload || {});
  if (focusSerial) body.serial = focusSerial;
  const response = await fetch('/api/inspect/' + action, {
    method: 'POST',
    cache: 'no-store',
    headers: {'Content-Type': 'application/json', 'X-AUA-Dashboard-Token': DATABASE_TOKEN},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    const error = data.error || {};
    const message = typeof error === 'string' ? error : (error.message || 'AUA inspection failed');
    throw Object.assign(new Error(message), {payload: data});
  }
  return data;
}

function inspectionElementLabel(element) {
  return element.text || element.content_desc || element.desc || element.resource_id ||
    element.rid || element.type || ('element ' + element.id);
}

function inspectionObjectRange(lines, element) {
  const idText = '"id": ' + Number(element.id);
  const stableText = element.stable_key ? JSON.stringify(String(element.stable_key)) : '';
  for (let pivot = 0; pivot < lines.length; pivot += 1) {
    if (lines[pivot].trim().replace(/,$/, '') !== idText) continue;
    let start = pivot;
    while (start >= 0 && lines[start].trim() !== '{') start -= 1;
    if (start < 0) continue;
    const indent = lines[start].length - lines[start].trimStart().length;
    let end = start;
    for (let index = start + 1; index < lines.length; index += 1) {
      const trimmed = lines[index].trim();
      const lineIndent = lines[index].length - lines[index].trimStart().length;
      if (lineIndent === indent && /^},?$/.test(trimmed)) {
        end = index;
        break;
      }
    }
    if (end <= start) continue;
    if (stableText && !lines.slice(start, end + 1).some(line => line.includes(stableText))) {
      continue;
    }
    return {start: start, end: end};
  }
  return null;
}

function scrollInspectionJsonLine(line) {
  if (!line) return;
  inspectionRaw.scrollTop = Math.max(
    0,
    line.offsetTop - inspectionRaw.clientHeight / 2 + line.offsetHeight / 2
  );
}

function updateInspectionJsonSearch(step = 0) {
  inspectionRaw.querySelectorAll('.search-match, .search-current').forEach(line => {
    line.classList.remove('search-match', 'search-current');
  });
  const query = inspectionJsonSearch.value.trim().toLocaleLowerCase();
  if (!query) {
    inspectionJsonMatches = [];
    inspectionJsonMatchIndex = -1;
    inspectionJsonSearchCount.textContent = '0';
    return;
  }
  const nextMatches = Array.from(inspectionRaw.querySelectorAll('.inspection-json-line')).filter(
    line => line.textContent.toLocaleLowerCase().includes(query)
  );
  const sameMatches = nextMatches.length === inspectionJsonMatches.length &&
    nextMatches.every((line, index) => line === inspectionJsonMatches[index]);
  inspectionJsonMatches = nextMatches;
  if (!inspectionJsonMatches.length) {
    inspectionJsonMatchIndex = -1;
    inspectionJsonSearchCount.textContent = '0';
    return;
  }
  if (!sameMatches || inspectionJsonMatchIndex < 0 || step === 0) {
    inspectionJsonMatchIndex = 0;
  } else {
    inspectionJsonMatchIndex =
      (inspectionJsonMatchIndex + step + inspectionJsonMatches.length) % inspectionJsonMatches.length;
  }
  inspectionJsonMatches.forEach(line => line.classList.add('search-match'));
  const current = inspectionJsonMatches[inspectionJsonMatchIndex];
  current.classList.add('search-current');
  inspectionJsonSearchCount.textContent =
    (inspectionJsonMatchIndex + 1) + '/' + inspectionJsonMatches.length;
  scrollInspectionJsonLine(current);
}

function renderInspectionRaw(result, elements) {
  const lines = JSON.stringify(result || {}, null, 2).split('\\n');
  const owners = new Map();
  elements.forEach(element => {
    const range = inspectionObjectRange(lines, element);
    if (!range) return;
    for (let line = range.start; line <= range.end; line += 1) {
      owners.set(line, {id: element.id, start: line === range.start});
    }
  });
  inspectionRaw.textContent = '';
  lines.forEach((text, index) => {
    const line = document.createElement('span');
    line.className = 'inspection-json-line';
    const owner = owners.get(index);
    if (owner) {
      line.classList.add('element-object');
      line.dataset.elementId = String(owner.id);
      if (owner.start) line.dataset.elementStart = 'true';
    }
    highlightJson(line, text || ' ');
    inspectionRaw.appendChild(line);
  });
  inspectionJsonMatches = [];
  inspectionJsonMatchIndex = -1;
  updateInspectionJsonSearch();
}

function focusInspectionJson(elementId) {
  inspectionRawDetails.open = true;
  const id = String(elementId);
  inspectionRaw.querySelectorAll('.element-current').forEach(line => {
    line.classList.remove('element-current');
  });
  inspectionJsonSearch.value = '"id": ' + id;
  updateInspectionJsonSearch();
  const objectLines = Array.from(
    inspectionRaw.querySelectorAll('.inspection-json-line[data-element-id="' + id + '"]')
  );
  objectLines.forEach(line => line.classList.add('element-current'));
  const start = objectLines.find(line => line.dataset.elementStart === 'true') || objectLines[0];
  if (start) {
    scrollInspectionJsonLine(start);
    inspectionRaw.focus({preventScroll: true});
    inspectionStatus.className = 'inspection-status';
    inspectionStatus.textContent =
      'Located #' + id + ' in the raw AUA response · click the outlined control body to tap it.';
  }
}

function renderInspection(data, tappedId) {
  const view = data.view || {};
  const screen = view.screen || {};
  const width = Number(screen.width || 0);
  const height = Number(screen.height || 0);
  const elements = Array.isArray(view.elements) ? view.elements.slice() : [];
  currentInspectionId = data.inspection_id || '';
  inspectionFrameActive = true;
  lastSrc = 'inspection:' + currentInspectionId;
  frame.src = data.frame_url + '&t=' + encodeURIComponent(Date.now());
  renderInspectionRaw(data.result || {}, elements);
  inspectionOutput.classList.remove('hidden');
  elementOverlay.innerHTML = '';

  const bounded = elements.filter(element => {
    const b = element && element.bounds;
    return Array.isArray(b) && b.length === 4 && width > 0 && height > 0 &&
      b.every(value => Number.isFinite(Number(value)));
  });
  // Containers go down first; smaller, more specific controls remain clickable above them.
  bounded.sort((a, b) => {
    const aa = Math.max(0, Number(a.bounds[2]) - Number(a.bounds[0])) *
      Math.max(0, Number(a.bounds[3]) - Number(a.bounds[1]));
    const ba = Math.max(0, Number(b.bounds[2]) - Number(b.bounds[0])) *
      Math.max(0, Number(b.bounds[3]) - Number(b.bounds[1]));
    return ba - aa;
  });
  bounded.forEach((element, index) => {
    const bounds = element.bounds.map(Number);
    const box = document.createElement('div');
    box.className = 'element-box' + (element.clickable ? ' clickable' : '');
    box.tabIndex = 0;
    box.setAttribute('role', 'button');
    box.dataset.elementId = String(element.id);
    box.style.left = (100 * bounds[0] / width) + '%';
    box.style.top = (100 * bounds[1] / height) + '%';
    box.style.width = (100 * Math.max(1, bounds[2] - bounds[0]) / width) + '%';
    box.style.height = (100 * Math.max(1, bounds[3] - bounds[1]) / height) + '%';
    box.style.zIndex = String(index + 1);
    const label = inspectionElementLabel(element);
    const stableKey = String(element.stable_key || '');
    box.title = '#' + element.id + (stableKey ? ' · ' + stableKey : '') + ' · ' + label +
      (element.clickable ? ' · clickable' : ' · AUA will resolve the acting control');
    box.setAttribute('aria-label', 'Tap and analyze element ' + element.id + ': ' + label);
    const badge = document.createElement('button');
    badge.type = 'button';
    badge.className = 'element-label';
    badge.title = 'Locate #' + element.id + ' in the raw AUA response' +
      (stableKey ? ' · ' + stableKey : '');
    badge.setAttribute('aria-label', 'Locate element ' + element.id + ' in raw AUA response');
    const idBadge = document.createElement('span');
    idBadge.className = 'element-id';
    idBadge.textContent = String(element.id);
    badge.appendChild(idBadge);
    if (stableKey) {
      const keyBadge = document.createElement('span');
      keyBadge.className = 'element-key';
      keyBadge.textContent = stableKey;
      badge.appendChild(keyBadge);
    }
    box.appendChild(badge);
    badge.addEventListener('click', event => {
      event.stopPropagation();
      focusInspectionJson(element.id);
    });
    box.addEventListener('click', () => tapInspectionElement(element.id, label));
    box.addEventListener('keydown', event => {
      if (event.target !== box || (event.key !== 'Enter' && event.key !== ' ')) return;
      event.preventDefault();
      tapInspectionElement(element.id, label);
    });
    elementOverlay.appendChild(box);
  });
  const clickable = elements.filter(element => element && element.clickable).length;
  inspectionCount.textContent = elements.length + ' elements · ' + clickable + ' interactive';
  inspectionStatus.className = 'inspection-status';
  inspectionStatus.textContent = tappedId == null
    ? 'Analysis ready · click any outlined item to run tap-and-analyze with its AUA id.'
    : 'Tapped #' + tappedId + ' · overlays and raw JSON now describe the fresh result screen.';
  screenAnalyze.textContent = 'Analyze again';
}

async function analyzeScreen() {
  if (inspectionBusy) return;
  inspectionBusy = true;
  screenAnalyze.disabled = true;
  elementOverlay.classList.add('busy');
  inspectionStatus.className = 'inspection-status';
  inspectionStatus.textContent = 'Analyzing the current device frame…';
  try {
    renderInspection(await inspectionPost('analyze', {}), null);
  } catch (error) {
    inspectionStatus.className = 'inspection-status bad';
    inspectionStatus.textContent = error.message;
    if (error.payload) {
      inspectionOutput.classList.remove('hidden');
      renderInspectionRaw(error.payload, []);
    }
  } finally {
    inspectionBusy = false;
    screenAnalyze.disabled = false;
    elementOverlay.classList.remove('busy');
  }
}

async function tapInspectionElement(elementId, label) {
  if (inspectionBusy || !currentInspectionId) return;
  inspectionBusy = true;
  screenAnalyze.disabled = true;
  elementOverlay.classList.add('busy');
  inspectionStatus.className = 'inspection-status';
  inspectionStatus.textContent = 'Tapping #' + elementId + ' · ' + label + '…';
  try {
    const data = await inspectionPost('tap', {
      inspection_id: currentInspectionId,
      element_id: Number(elementId),
    });
    renderInspection(data, elementId);
  } catch (error) {
    inspectionStatus.className = 'inspection-status bad';
    inspectionStatus.textContent = error.message + ' · Analyze again before choosing another id.';
    currentInspectionId = '';
    elementOverlay.innerHTML = '';
    if (error.payload) renderInspectionRaw(error.payload, []);
  } finally {
    inspectionBusy = false;
    screenAnalyze.disabled = false;
    elementOverlay.classList.remove('busy');
  }
}

function resumeLiveFrame() {
  currentInspectionId = '';
  inspectionFrameActive = false;
  elementOverlay.innerHTML = '';
  inspectionOutput.classList.add('hidden');
  inspectionJsonSearch.value = '';
  inspectionStatus.className = 'inspection-status';
  inspectionStatus.textContent = 'Live frame resumed. Analyze to inspect AUA element IDs.';
  screenAnalyze.textContent = 'Analyze';
  lastSrc = '';
  frame.src = '/api/frame.jpg' + qSerial({t: Date.now()});
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
  addEventText(summary, 'event-chevron', '›');
  addEventText(summary, 't', fmtTime(e.ts_ms));
  addEventText(summary, 'badge ' + (ok ? 'ok' : 'fail'), ok ? 'ok' : 'fail');
  const main = document.createElement('span');
  main.className = 'event-main';
  addEventText(main, 'cmd', e.cmd || '?');
  const args = argsSummary(e.args);
  if (args) addEventText(main, 'args', args);
  const failure = ok ? '' : errText(e.error);
  if (!ok && e.error) addEventText(main, 'err', failure);
  summary.appendChild(main);
  if (e.duration_ms != null) {
    addEventText(summary, 'dur' + (Number(e.duration_ms) >= 1500 ? ' slow' : ''), e.duration_ms + 'ms');
  } else {
    addEventText(summary, 'dur', '—');
  }
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
document.addEventListener('keydown', event => {
  const target = event.target;
  const isTyping = target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
  if (event.key === '/' && !isTyping) {
    event.preventDefault();
    journalFilter.focus();
    journalFilter.select();
  } else if (event.key === 'Escape' && document.activeElement === journalFilter) {
    journalFilter.value = '';
    applyJournalFilter();
    journalFilter.blur();
  }
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
document.getElementById('journal-clear').addEventListener('click', async () => {
  const confirmed = await confirmKnowledgeAction(
    'Clear Agent I/O journal?',
    'Clear compact and full-detail journal logs visible for ' + focusSerial +
      '. This cannot be undone.',
    'Clear logs'
  );
  if (!confirmed) return;
  const button = document.getElementById('journal-clear');
  button.disabled = true;
  try {
    await dashboardControlPost('journal', 'clear', {
      confirmation: 'CLEAR JOURNAL ' + focusSerial,
    });
    sinceMs = 0;
    detailRevision = '';
    seenKeys.clear();
    journalPending = 0;
    journalFollow = true;
    journalEl.innerHTML = '<li class="empty">journal cleared</li>';
    journalShown.textContent = '0 events';
    journalWrap.scrollTop = 0;
    updateJournalFollow();
    tickStatus();
  } catch (error) {
    journalShown.textContent = error.message;
  } finally {
    button.disabled = false;
  }
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
      '<div class="tile-screen">' +
        '<img alt="device frame" src=""/>' +
        '<div class="tile-overlay tile-overlay-top">' +
          '<span class="tile-identity"><span class="tile-device-dot"></span><span class="ser"></span></span>' +
          '<span class="cap"></span>' +
        '</div>' +
        '<div class="tile-overlay tile-overlay-bottom">' +
          '<div class="tile-app"><span class="tile-app-label">Foreground app</span><span class="pkg"></span></div>' +
          '<div class="tile-stats">' +
            '<div class="tile-stat"><span class="tile-stat-label">Frame</span><span class="tile-stat-value age"></span></div>' +
            '<div class="tile-stat"><span class="tile-stat-label">Lease</span><span class="tile-stat-value lease"></span></div>' +
            '<div class="tile-stat"><span class="tile-stat-label">Auto-stop</span><span class="tile-stat-value watchdog"></span></div>' +
            '<div class="tile-stat"><span class="tile-stat-label">Capture</span><span class="tile-stat-value capture-detail"></span></div>' +
          '</div>' +
          '<div class="tile-card-footer"><span class="owner"></span><span class="tile-inspect">Inspect →</span></div>' +
        '</div>' +
      '</div>';
    tiles.appendChild(a);
  }
  a.querySelector('.ser').textContent = d.serial;
  const own = a.querySelector('.owner');
  if (d.owner) { own.textContent = 'Owner · ' + d.owner; own.classList.remove('hidden'); }
  else { own.textContent = 'Available'; own.classList.remove('hidden'); }
  const cap = a.querySelector('.cap');
  cap.textContent = d.capture_running ? 'live' : (d.has_frame ? 'frame' : 'idle');
  cap.className = 'cap ' + (d.capture_running ? 'ok' : '');
  a.querySelector('.capture-detail').textContent =
    d.capture_running ? 'Streaming' : (d.has_frame ? 'Snapshot' : 'Waiting');
  a.querySelector('.age').textContent = fmtAge(d.frame_age_ms);
  a.querySelector('.pkg').textContent = d.package || 'No foreground package';
  const lease = d.lease || {};
  const leaseEl = a.querySelector('.lease');
  leaseEl.textContent = lease.held ? 'Held' : 'Available';
  leaseEl.className = 'tile-stat-value lease' + (lease.held ? ' held' : '');
  const watchdog = d.watchdog || {};
  const watchdogEl = a.querySelector('.watchdog');
  watchdogEl.textContent = !watchdog.managed
    ? 'External device'
    : (!watchdog.enabled
        ? 'Disabled'
        : (lease.held
            ? 'Paused by lease'
            : (watchdog.running ? ('In ' + fmtDuration(watchdog.remaining_s)) : 'Watchdog offline')));
  watchdogEl.className = 'tile-stat-value watchdog' + (
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
    if (!list.length && d.discovery_error) {
      document.getElementById('grid-empty-title').textContent = 'Device discovery failed';
      document.getElementById('grid-empty-copy').textContent = d.discovery_error;
      document.getElementById('grid-empty-command').classList.add('hidden');
      document.getElementById('grid-empty-foot').textContent = 'AuA will retry automatically.';
    } else if (!list.length) {
      document.getElementById('grid-empty-title').textContent = 'No devices online';
      document.getElementById('grid-empty-copy').textContent =
        'Start an emulator and it will appear here automatically.';
      document.getElementById('grid-empty-command').classList.remove('hidden');
      document.getElementById('grid-empty-foot').textContent =
        'This dashboard keeps watching in the background.';
    }
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
    const emptyEl = document.getElementById('grid-empty');
    emptyEl.className = 'grid-empty bad';
    document.getElementById('grid-empty-title').textContent = 'Dashboard connection lost';
    document.getElementById('grid-empty-copy').textContent =
      'The dashboard could not refresh its device list.';
    document.getElementById('grid-empty-command').classList.add('hidden');
    document.getElementById('grid-empty-foot').textContent = 'AuA will retry automatically.';
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
    if (s.detached) {
      window.location.replace('/?detached=' + encodeURIComponent(s.serial || focusSerial));
      return;
    }
    if (!r.ok || s.ok === false) throw new Error(s.error || 'Device status failed');
    document.getElementById('serial').textContent = s.serial || '—';
    const cap = document.getElementById('capture');
    cap.textContent = s.capture_running ? 'Active' : 'Inactive';
    cap.className = 'detail-status-value ' + (s.capture_running ? 'ok' : 'bad');
    document.getElementById('via').textContent = s.via || '—';
    const lease = s.lease || {};
    const leasePill = document.getElementById('lease');
    leasePill.textContent = lease.held ? (lease.owner || 'Held') : 'Available';
    leasePill.className = 'detail-status-value' + (lease.held ? ' ok' : '');
    const watchdog = s.watchdog || {};
    const watchdogPill = document.getElementById('watchdog');
    watchdogPill.textContent = !watchdog.managed
      ? 'External device'
      : (!watchdog.enabled
          ? 'Disabled'
          : (lease.held
              ? 'Paused by lease'
              : (watchdog.running ? ('In ' + fmtDuration(watchdog.remaining_s)) : 'Offline')));
    watchdogPill.className = 'detail-status-value' + (
      watchdog.managed && watchdog.enabled && !watchdog.running ? ' bad' : ''
    );
    document.getElementById('age').textContent = fmtAge(s.frame_age_ms);
    const fc = (s.stats && s.stats.fail_count) || 0;
    const fp = document.getElementById('failpill');
    fp.textContent = String(fc);
    fp.className = 'detail-status-value ' + (fc ? 'bad' : 'ok');
    document.getElementById('pkg').textContent = s.package || 'No foreground package';
    if (s.package && !logcatAppTouched && logcatAppFilter.value !== s.package) {
      logcatAppFilter.value = s.package;
      tickLogcat();
    }
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
    if (!inspectionFrameActive && frameToken !== lastSrc) {
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

let knowledgeSignature = '';

async function dashboardControlPost(scope, action, payload) {
  const body = Object.assign({}, payload || {});
  if (focusSerial) body.serial = focusSerial;
  const response = await fetch('/api/' + scope + '/' + action, {
    method: 'POST',
    cache: 'no-store',
    headers: {'Content-Type': 'application/json', 'X-AUA-Dashboard-Token': DATABASE_TOKEN},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    const error = data.error || {};
    throw new Error(typeof error === 'string' ? error : (error.message || 'Dashboard action failed'));
  }
  return data;
}

function confirmKnowledgeAction(title, message, confirmLabel) {
  if (pendingKnowledgeConfirmation) pendingKnowledgeConfirmation(false);
  document.getElementById('knowledge-confirm-title').textContent = title;
  document.getElementById('knowledge-confirm-message').textContent = message;
  document.getElementById('knowledge-confirm-submit').textContent = confirmLabel || 'Confirm';
  knowledgeConfirmDialog.showModal();
  return new Promise(resolve => { pendingKnowledgeConfirmation = resolve; });
}

function closeKnowledgeConfirmation(confirmed) {
  const resolve = pendingKnowledgeConfirmation;
  pendingKnowledgeConfirmation = null;
  knowledgeConfirmDialog.close();
  if (resolve) resolve(Boolean(confirmed));
}

function setKnowledgeActionStatus(message, kind, result) {
  knowledgeActionStatus.textContent = message;
  knowledgeActionStatus.className = 'knowledge-action-status visible' + (kind ? (' ' + kind) : '');
  if (result === undefined) return;
  knowledgeActionResult.classList.remove('hidden');
  const pre = knowledgeActionResult.querySelector('pre');
  pre.textContent = prettyJson(result);
  highlightJson(pre);
}

function navigationResultMessage(action, data) {
  const result = data.result || {};
  if (action === 'goto') {
    if (result.arrived || result.already_there) return 'Goto reached ' + (result.target || data.target) + '.';
    return 'Goto stopped safely: ' + (result.code || result.status || 'destination not verified') + '.';
  }
  if (action === 'flow-run') {
    return result.ok === false
      ? 'Flow stopped: ' + (result.code || result.status || 'journey not completed') + '.'
      : 'Flow ' + data.flow + ' finished.';
  }
  if (action === 'flow-delete') return 'Flow ' + data.flow + ' cleared.';
  if (action === 'route-delete') return data.deleted ? 'Route cleared.' : 'Route was already absent.';
  if (action === 'map-clear') return data.deleted ? 'App map cleared.' : 'App map was already absent.';
  if (action === 'flows-clear') return data.deleted + ' flow' + (data.deleted === 1 ? '' : 's') + ' cleared.';
  if (action === 'clear-all') return data.maps_deleted + ' maps and ' + data.flows_deleted + ' flows cleared.';
  return 'Action completed.';
}

async function runNavigationAction(action, payload, confirmation) {
  if (knowledgeBusy) return;
  if (confirmation) {
    const confirmed = await confirmKnowledgeAction(
      confirmation.title,
      confirmation.message,
      confirmation.label
    );
    if (!confirmed) return;
    payload = Object.assign({}, payload, {confirmation: confirmation.phrase});
  }
  knowledgeBusy = true;
  document.querySelectorAll('.knowledge-workspace .db-button').forEach(button => {
    button.disabled = true;
  });
  setKnowledgeActionStatus(action === 'goto' ? 'Running goto…' : 'Running navigation action…');
  try {
    const data = await dashboardControlPost('navigation', action, payload);
    setKnowledgeActionStatus(navigationResultMessage(action, data), 'ok', data.result || data);
    knowledgeSignature = '';
    await tickMap();
    if (action === 'goto' || action === 'flow-run') {
      resumeLiveFrame();
      tickStatus();
      tickEvents();
    }
  } catch (error) {
    setKnowledgeActionStatus(error.message, 'bad');
  } finally {
    knowledgeBusy = false;
    document.querySelectorAll('.knowledge-workspace .knowledge-actions .db-button').forEach(button => {
      button.disabled = button.dataset.initiallyDisabled === 'true';
    });
    const mapData = window.__auaKnowledgeData || {};
    document.getElementById('map-clear').disabled = !mapData.known;
    document.getElementById('flows-clear').disabled = !(mapData.flows || []).length;
    document.getElementById('knowledge-clear-all').disabled =
      !Number(mapData.map_count || 0) && !(mapData.flows || []).length;
  }
}

function knowledgeActions(actions) {
  const row = document.createElement('div');
  row.className = 'knowledge-actions';
  actions.forEach(action => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'db-button' + (action.danger ? ' danger' : '') +
      (action.primary ? ' primary' : '');
    button.textContent = action.label;
    button.disabled = Boolean(action.disabled);
    button.dataset.initiallyDisabled = action.disabled ? 'true' : 'false';
    button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      runNavigationAction(action.action, Object.assign({}, action.payload), action.confirmation);
    });
    row.appendChild(button);
  });
  return row;
}

function knowledgeBadge(text, kind = '') {
  const badge = document.createElement('span');
  badge.className = 'knowledge-badge' + (kind ? (' ' + kind) : '');
  badge.textContent = text;
  return badge;
}

function knowledgeCommand(label, command) {
  const row = document.createElement('div');
  row.className = 'knowledge-command';
  const title = document.createElement('span');
  title.textContent = label;
  const code = document.createElement('code');
  code.textContent = command;
  row.append(title, code);
  return row;
}

function knowledgeJson(value) {
  const pre = document.createElement('pre');
  pre.className = 'knowledge-json';
  pre.textContent = prettyJson(value);
  highlightJson(pre);
  return pre;
}

function knowledgeSteps(steps, emptyMessage) {
  if (!steps || !steps.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = emptyMessage;
    return empty;
  }
  const list = document.createElement('ol');
  list.className = 'knowledge-steps';
  steps.forEach((step, index) => {
    const item = document.createElement('li');
    item.className = 'knowledge-step';
    const number = document.createElement('span');
    number.className = 'knowledge-step-index';
    number.textContent = String(index + 1).padStart(2, '0');
    const main = document.createElement('div');
    main.className = 'knowledge-step-main';
    const kind = document.createElement('span');
    kind.className = 'knowledge-step-kind';
    kind.textContent = step.kind || 'step';
    const data = document.createElement('span');
    data.className = 'knowledge-step-data';
    const detail = Object.assign({}, step);
    delete detail.kind;
    data.textContent = Object.keys(detail).length ? prettyJson(detail) : 'No additional arguments';
    main.append(kind, data);
    item.append(number, main);
    list.appendChild(item);
  });
  return list;
}

function knowledgeItem(titleText, subtitleText, badges, bodyNodes, actions) {
  const details = document.createElement('details');
  details.className = 'knowledge-item';
  const summary = document.createElement('summary');
  const main = document.createElement('span');
  main.className = 'knowledge-summary-main';
  const title = document.createElement('span');
  title.className = 'knowledge-summary-title';
  title.textContent = titleText;
  const subtitle = document.createElement('span');
  subtitle.className = 'knowledge-summary-subtitle';
  subtitle.textContent = subtitleText || 'Expand for full details';
  main.append(title, subtitle);
  const badgeHost = document.createElement('span');
  badgeHost.className = 'knowledge-badges';
  badges.forEach(badge => badgeHost.appendChild(badge));
  if (actions) badgeHost.appendChild(actions);
  summary.append(main, badgeHost);
  const body = document.createElement('div');
  body.className = 'knowledge-detail';
  bodyNodes.forEach(node => body.appendChild(node));
  details.append(summary, body);
  return details;
}

function renderKnowledge(d) {
  const screens = d.screens || [];
  const routes = d.routes || [];
  const flows = d.flows || [];
  window.__auaKnowledgeData = d;
  document.getElementById('map-pkg').textContent =
    (d.package || 'No foreground package') + (d.known ? '' : ' · no map yet');
  document.getElementById('knowledge-total').textContent =
    screens.length + ' screens · ' + routes.length + ' routes · ' + flows.length + ' flows';
  document.getElementById('map-clear').disabled = !d.known || !d.package;
  document.getElementById('flows-clear').disabled = !flows.length;
  document.getElementById('knowledge-clear-all').disabled =
    !Number(d.map_count || 0) && !flows.length;

  const screenHost = document.getElementById('map-screens');
  screenHost.textContent = '';
  if (!screens.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No learned screens for the foreground app.';
    screenHost.appendChild(empty);
  }
  screens.forEach(screen => {
    const name = screen.name || '?';
    const bodyData = Object.assign({}, screen);
    delete bodyData.name;
    screenHost.appendChild(knowledgeItem(
      name,
      (screen.activity || 'No activity recorded') + (screen.stale ? ' · stale' : ''),
      [knowledgeBadge('goto target', 'ok'), knowledgeBadge((screen.visit_count || 0) + ' visits')],
      [
        knowledgeCommand('CLI', 'aua goto "' + name + '"'),
        knowledgeJson(bodyData),
      ],
      knowledgeActions([{
        label: 'Run goto', action: 'goto', payload: {target: name}, primary: true,
      }])
    ));
  });

  const routeHost = document.getElementById('map-routes');
  routeHost.textContent = '';
  if (!routes.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No recorded routes for the foreground app.';
    routeHost.appendChild(empty);
  }
  routes.forEach(route => {
    const from = route.from || route.from_screen || '?';
    const to = route.to || route.to_screen || '?';
    const meta = Object.assign({}, route);
    delete meta.steps;
    routeHost.appendChild(knowledgeItem(
      from + ' → ' + to,
      route.action || 'No action label',
      [
        knowledgeBadge(route.status || 'unknown', route.status === 'verified' ? 'ok' : ''),
        knowledgeBadge((route.steps || []).length + ' steps'),
      ],
      [
        knowledgeCommand('Goto target', 'aua goto "' + to + '"'),
        knowledgeSteps(route.steps || [], 'Legacy route: no structured steps were recorded.'),
        knowledgeJson(meta),
      ],
      knowledgeActions([
        {label: 'Run goto', action: 'goto', payload: {target: to}, primary: true},
        {
          label: 'Clear', action: 'route-delete', danger: true, disabled: !route.id,
          payload: {package: d.package, route_id: route.id},
          confirmation: {
            title: 'Clear recorded route?',
            message: from + ' → ' + to + ' will be removed from ' + d.package + '.',
            label: 'Clear route', phrase: 'DELETE ROUTE ' + route.id,
          },
        },
      ])
    ));
  });

  const flowHost = document.getElementById('flow-groups');
  flowHost.textContent = '';
  document.getElementById('flow-count').textContent =
    flows.length + ' flow' + (flows.length === 1 ? '' : 's');
  const groups = new Map();
  flows.forEach(flow => {
    const app = flow.app || 'App-agnostic';
    if (!groups.has(app)) groups.set(app, []);
    groups.get(app).push(flow);
  });
  const packages = Array.from(groups.keys()).sort((a, b) => {
    if (a === d.package) return -1;
    if (b === d.package) return 1;
    if (a === 'App-agnostic') return -1;
    if (b === 'App-agnostic') return 1;
    return a.localeCompare(b);
  });
  if (!packages.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No saved flows found.';
    flowHost.appendChild(empty);
  }
  packages.forEach(app => {
    const section = document.createElement('section');
    section.className = 'flow-group';
    const head = document.createElement('div');
    head.className = 'flow-group-title';
    const heading = document.createElement('h4');
    heading.textContent = app;
    const count = document.createElement('span');
    count.textContent = groups.get(app).length + ' flow' + (groups.get(app).length === 1 ? '' : 's');
    head.append(heading, count);
    section.appendChild(head);
    groups.get(app).forEach(flow => {
      const ref = flow.ref || '';
      const displayRef = ref || (flow.storage_name || flow.name || 'unaddressable flow');
      const meta = Object.assign({}, flow);
      delete meta.steps_detail;
      section.appendChild(knowledgeItem(
        flow.name || flow.storage_name || '?',
        flow.description || displayRef,
        [
          knowledgeBadge((flow.steps_detail || []).length + ' steps'),
          knowledgeBadge(flow.arrival_status || 'unverified', flow.arrival_status === 'mapped' ? 'ok' : ''),
        ],
        [
          knowledgeCommand(
            ref ? 'CLI' : 'Unavailable',
            ref ? ('aua flow run "' + ref + '"') : 'This flow needs a unique storage reference.'
          ),
          knowledgeSteps(flow.steps_detail || [], flow.error || 'No steps recorded.'),
          knowledgeJson(meta),
        ],
        knowledgeActions([
          {
            label: 'Run flow', action: 'flow-run', primary: true, disabled: !ref,
            payload: {ref: ref},
            confirmation: {
              title: 'Run authored flow?',
              message: 'Run ' + ref + ' on ' + focusSerial +
                '? Authored flows may change app or device state.',
              label: 'Run flow', phrase: 'RUN FLOW ' + ref,
            },
          },
          {
            label: 'Clear', action: 'flow-delete', danger: true, disabled: !ref,
            payload: {ref: ref},
            confirmation: {
              title: 'Clear saved flow?',
              message: ref + ' will be permanently removed from the flow library.',
              label: 'Clear flow', phrase: 'DELETE FLOW ' + ref,
            },
          },
        ])
      ));
    });
    flowHost.appendChild(section);
  });
}

async function tickMap() {
  try {
    const r = await fetch('/api/map' + qSerial(), {cache: 'no-store'});
    const d = await r.json();
    const signature = JSON.stringify([
      d.package, d.known, d.map_count, d.screens, d.routes, d.flows,
    ]);
    if (signature === knowledgeSignature) return;
    knowledgeSignature = signature;
    renderKnowledge(d);
  } catch (e) {
    document.getElementById('knowledge-total').textContent = 'library unavailable';
  }
}

const LOGCAT_ORDER = {V: 0, D: 1, I: 2, W: 3, E: 4, F: 5, A: 5, S: 5};
const logcatEl = document.getElementById('logcat');
const logcatView = document.getElementById('logcat-view');
const logcatFilter = document.getElementById('logcat-filter');
const logcatAppFilter = document.getElementById('logcat-app-filter');
const logcatLevel = document.getElementById('logcat-level');
const logcatFollow = document.getElementById('logcat-follow');
let logcatLines = [];
let logcatAppTouched = false;
let logcatAppTimer = null;

function logcatAnchor() {
  const rows = logcatEl.children;
  const top = logcatView.scrollTop;
  for (let i = 0; i < rows.length; i += 1) {
    if (rows[i].offsetTop + rows[i].offsetHeight > top + 1) {
      return {raw: rows[i].dataset.raw || rows[i].textContent, top: rows[i].offsetTop};
    }
  }
  return null;
}

function renderLogcat(lines) {
  const needle = logcatFilter.value.trim().toLowerCase();
  const appId = logcatAppFilter.value.trim();
  const floor = LOGCAT_ORDER[logcatLevel.value] || 0;
  const kept = [];
  // The API returns Android's chronological buffer; the dashboard is newest-first so
  // fresh evidence appears beside the filters instead of below hundreds of older lines.
  lines.slice().reverse().forEach(line => {
    const tokens = logcatTokens(line);
    let level = '';
    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i][0] === 'lvl') { level = tokens[i][1]; break; }
    }
    if (level && (LOGCAT_ORDER[level] || 0) < floor) return;
    if (needle !== '' && line.toLowerCase().indexOf(needle) < 0) return;
    kept.push([tokens, level, line]);
  });
  document.getElementById('logcat-shown').textContent =
    kept.length + ' / ' + lines.length + ' lines' + (appId ? (' · ' + appId) : ' · all apps');
  const signature = JSON.stringify(lines) + '\\u0000' + needle + '\\u0000' + floor + '\\u0000' + appId;
  if (logcatEl.dataset.signature === signature) return;
  // Newest-first: follow hugs the top. When the reader scrolls into history, retain the
  // first visible raw line even as newer rows are inserted above it.
  const atTop = logcatView.scrollTop < 12;
  const anchor = !logcatFollow.checked && !atTop ? logcatAnchor() : null;
  logcatEl.dataset.signature = signature;
  logcatEl.textContent = '';
  if (!kept.length) {
    logcatEl.textContent = lines.length ? '(nothing matches the filter)' : '(empty)';
    logcatView.scrollTop = 0;
    return;
  }
  const frag = document.createDocumentFragment();
  kept.forEach(entry => {
    const tokens = entry[0];
    const level = entry[1];
    const raw = entry[2];
    const row = document.createElement('span');
    row.className = 'lc-line' +
      (level === 'E' || level === 'F' || level === 'A' ? ' err' : '') +
      (level === 'W' ? ' warn' : '');
    row.dataset.raw = raw;
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
  if (logcatFollow.checked || atTop) {
    logcatView.scrollTop = 0;
  } else if (anchor) {
    const rows = logcatEl.children;
    for (let i = 0; i < rows.length; i += 1) {
      if ((rows[i].dataset.raw || rows[i].textContent) === anchor.raw) {
        logcatView.scrollTop += rows[i].offsetTop - anchor.top;
        break;
      }
    }
  }
}

logcatFilter.addEventListener('input', () => renderLogcat(logcatLines));
logcatAppFilter.addEventListener('input', () => {
  logcatAppTouched = true;
  if (logcatAppTimer) clearTimeout(logcatAppTimer);
  logcatAppTimer = setTimeout(tickLogcat, 250);
});
logcatLevel.addEventListener('change', () => renderLogcat(logcatLines));
logcatFollow.addEventListener('change', () => {
  if (logcatFollow.checked) logcatView.scrollTop = 0;
});
document.getElementById('logcat-wrap').addEventListener('change', event => {
  logcatEl.classList.toggle('wrap', event.target.checked);
});

async function tickLogcat() {
  try {
    const params = {lines: 400};
    const appId = logcatAppFilter.value.trim();
    if (appId) params.app_id = appId;
    const r = await fetch('/api/logcat' + qSerial(params), {cache: 'no-store'});
    const d = await r.json();
    if (!r.ok || d.ok === false) {
      logcatLines = ['Logcat filter error: ' + (d.error || ('HTTP ' + r.status))];
      renderLogcat(logcatLines);
      return;
    }
    logcatLines = d.lines || [];
    renderLogcat(logcatLines);
  } catch (e) {
    logcatLines = ['Logcat unavailable: ' + (e && e.message ? e.message : String(e))];
    renderLogcat(logcatLines);
  }
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

const modelIntercept = document.getElementById('model-intercept');
const modelInterceptLabel = document.getElementById('model-intercept-label');
const modelTraces = document.getElementById('model-traces');
const modelChat = document.getElementById('model-chat');
const modelPrompt = document.getElementById('model-prompt');
const modelChatProvider = document.getElementById('model-chat-provider');
const modelMaxTokens = document.getElementById('model-max-tokens');
const modelChatStatus = document.getElementById('model-chat-status');
const modelRequestKind = document.getElementById('model-request-kind');
const modelRequestSample = document.getElementById('model-request-sample');
const modelPromptGuide = document.getElementById('model-prompt-guide');
const modelProviderLabel = document.getElementById('model-provider-label');
const modelHistories = {functiongemma: [], gemma4: [], agent_chain: []};
let modelDirectProvider = 'functiongemma';
const MODEL_AGENT_SAMPLES = {
  settings: {
    goal: 'Open Settings',
    phase: 'Choose the next current-screen control',
    observation: {fresh: true, known_screen: 'home', source: 'hierarchy'},
    constraints: ['Choose only a supplied current-frame control.'],
    candidates: [
      {id: 0, label: 'Search', purpose: 'Open search', proof: 'Visible Search control'},
      {id: 1, label: 'Settings', purpose: 'Open Settings', proof: 'Visible Settings control directly matches the goal'},
      {id: 2, label: 'Profile', purpose: 'Open profile', proof: 'Visible Profile control'},
    ],
    allow_handoff: true,
  },
  next_step: {
    goal: 'Open Account, then Notifications',
    phase: 'Reach Account before looking for Notifications',
    observation: {fresh: true, known_screen: 'home', source: 'hierarchy'},
    constraints: ['Choose the earliest incomplete waypoint.'],
    candidates: [
      {id: 0, label: 'Home', purpose: 'Stay on Home', proof: 'Visible Home control'},
      {id: 1, label: 'Account', purpose: 'Open Account', proof: 'Visible Account control reaches the next waypoint'},
      {id: 2, label: 'Help', purpose: 'Open Help', proof: 'Visible Help control'},
    ],
    allow_handoff: true,
  },
  handoff: {
    goal: 'Open Notifications',
    phase: 'Choose the next current-screen control',
    observation: {fresh: true, known_screen: 'home', source: 'hierarchy'},
    constraints: ['Return handoff instead of guessing.'],
    candidates: [
      {id: 0, label: 'Home', purpose: 'Open Home', proof: 'Visible Home control'},
      {id: 1, label: 'Search', purpose: 'Open Search', proof: 'Visible Search control'},
      {id: 2, label: 'Profile', purpose: 'Open Profile', proof: 'Visible Profile control'},
    ],
    allow_handoff: true,
  },
};
let modelPolling = false;

function updateModelRequestShape(populate) {
  const agent = modelRequestKind.value === 'agent';
  modelMaxTokens.disabled = agent;
  modelRequestSample.disabled = !agent;
  if (agent) {
    if (modelChatProvider.value !== 'agent_chain') modelDirectProvider = modelChatProvider.value;
    modelChatProvider.querySelector('[value="agent_chain"]').hidden = false;
    modelChatProvider.value = 'agent_chain';
    modelChatProvider.disabled = true;
    modelProviderLabel.textContent = 'Execution path';
    modelPrompt.placeholder = '{"goal":"Open Settings","candidates":[…]}';
    modelPromptGuide.innerHTML = 'Uses the exact configured agent policy chain, guards, consensus, ' +
      'semantic review and fallback. Candidate IDs must be dense <code>0…N</code>; the trusted ' +
      'result is shown but never executed.';
    if (populate && !modelRequestSample.value) modelRequestSample.value = 'settings';
    if (populate && modelRequestSample.value) {
      modelPrompt.value = JSON.stringify(MODEL_AGENT_SAMPLES[modelRequestSample.value], null, 2);
    }
  } else {
    modelChatProvider.disabled = false;
    modelChatProvider.value = modelDirectProvider;
    modelChatProvider.querySelector('[value="agent_chain"]').hidden = true;
    modelProviderLabel.textContent = 'Model';
    modelPrompt.placeholder = 'Message the local model…';
    modelPromptGuide.textContent = 'Plain text sends a normal chat message and returns raw text.';
  }
  renderModelChat(modelChatProvider.value);
}

function modelError(data, fallback) {
  const error = data && data.error;
  if (typeof error === 'string') return error;
  if (error && error.message) return error.message;
  return fallback || 'local model operation failed';
}

async function modelPost(action, payload) {
  const response = await fetch('/api/models/' + action, {
    method: 'POST',
    cache: 'no-store',
    headers: {'Content-Type': 'application/json', 'X-AUA-Dashboard-Token': DATABASE_TOKEN},
    body: JSON.stringify(Object.assign({serial: focusSerial}, payload || {})),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(modelError(data));
  return data;
}

function modelFormatTokens(value) {
  if (value == null) return '—';
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'm';
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k';
  return String(n);
}

function modelGroupedEvents(events) {
  const grouped = new Map();
  (events || []).forEach(event => {
    const id = event.id || ('event-' + event.timestamp_ms);
    const previous = grouped.get(id) || {};
    grouped.set(id, Object.assign(previous, event, {
      id: id,
      started_ms: previous.started_ms || previous.timestamp_ms || event.timestamp_ms,
    }));
  });
  return Array.from(grouped.values()).sort((a, b) =>
    Number(b.timestamp_ms || 0) - Number(a.timestamp_ms || 0));
}

function modelRenderTraces(events) {
  const grouped = modelGroupedEvents(events);
  document.getElementById('model-trace-count').textContent =
    grouped.length + ' exchange' + (grouped.length === 1 ? '' : 's');
  modelTraces.textContent = '';
  if (!grouped.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No model activity yet.';
    modelTraces.appendChild(empty);
    return;
  }
  grouped.forEach(event => {
    const details = document.createElement('details');
    details.className = 'model-trace';
    const summary = document.createElement('summary');
    const state = document.createElement('span');
    state.className = 'model-state ' +
      (event.phase === 'error' ? 'bad' : (event.phase === 'running' || event.phase === 'loading' ? 'busy' : 'ready'));
    state.textContent = event.phase || 'event';
    const title = document.createElement('span');
    title.className = 'model-trace-title';
    const strong = document.createElement('strong');
    strong.textContent = (event.provider || 'control') + ' · ' +
      (event.operation || (event.source === 'playground' ? 'message' : 'policy decision'));
    const sub = document.createElement('span');
    sub.textContent = (event.source || 'runtime') + ' · ' + fmtTime(event.timestamp_ms);
    title.append(strong, sub);
    const metrics = document.createElement('span');
    metrics.className = 'model-trace-metrics';
    const tokenBits = event.input_tokens != null || event.output_tokens != null
      ? (modelFormatTokens(event.input_tokens) + ' → ' + modelFormatTokens(event.output_tokens) + ' tok')
      : '';
    metrics.textContent = [event.duration_ms != null ? event.duration_ms + ' ms' : '', tokenBits]
      .filter(Boolean).join(' · ') || '—';
    summary.append(state, title, metrics);
    const body = document.createElement('div');
    body.className = 'model-trace-body';
    const inputPane = document.createElement('section');
    inputPane.className = 'model-trace-pane';
    const inputTitle = document.createElement('h4');
    inputTitle.textContent = 'Model input';
    const input = document.createElement('pre');
    const fullInput = event.tools ? {messages: event.input, tools: event.tools} : event.input;
    input.textContent = fullInput == null ? 'No prompt for this operation.' : prettyJson(fullInput);
    highlightJson(input);
    inputPane.append(inputTitle, input);
    const outputPane = document.createElement('section');
    outputPane.className = 'model-trace-pane';
    const outputTitle = document.createElement('h4');
    outputTitle.textContent = event.error ? 'Error' : 'Model output';
    const output = document.createElement('pre');
    output.textContent = event.error || (event.output == null
      ? (event.selected_id != null ? ('selected candidate ' + event.selected_id) : 'Waiting for output…')
      : (typeof event.output === 'string' ? event.output : prettyJson(event.output)));
    outputPane.append(outputTitle, output);
    body.append(inputPane, outputPane);
    details.append(summary, body);
    modelTraces.appendChild(details);
  });
}

function modelLatestFor(events, provider) {
  return modelGroupedEvents(events).find(event => event.provider === provider) || null;
}

function renderModels(data) {
  const control = data.control || {};
  modelIntercept.checked = Boolean(control.intercept_enabled);
  modelInterceptLabel.textContent = modelIntercept.checked
    ? 'Intercepting eligible agent decisions'
    : 'Agent interception off';
  const events = data.events || [];
  const byName = {};
  (data.providers || []).forEach(provider => { byName[provider.provider] = provider; });
  ['functiongemma', 'gemma4'].forEach(name => {
    const provider = byName[name] || {provider: name};
    const card = document.getElementById('model-card-' + name);
    const latest = modelLatestFor(events, name);
    const busy = Boolean(latest && (latest.phase === 'running' || latest.phase === 'loading'));
    card.classList.toggle('busy', busy);
    const state = card.querySelector('.model-state');
    state.textContent = busy ? latest.phase : (provider.loaded ? 'resident' : (provider.available ? 'ready' : 'unavailable'));
    state.className = 'model-state ' + (busy ? 'busy' : (provider.available ? 'ready' : 'bad'));
    card.querySelector('[data-field="context"]').textContent = modelFormatTokens(provider.context_window);
    card.querySelector('[data-field="runtime"]').textContent = provider.loaded ? 'loaded' : (provider.available ? 'cold' : 'missing');
    card.querySelector('[data-field="latency"]').textContent = latest && latest.duration_ms != null
      ? latest.duration_ms + ' ms' : '—';
    const toggle = card.querySelector('.model-provider-toggle');
    toggle.checked = provider.enabled !== false;
    card.querySelector('.model-load').disabled = busy || provider.loaded || !provider.available;
    card.querySelector('.model-unload').disabled = busy || !provider.loaded;
    card.title = provider.reason || '';
  });
  modelRenderTraces(events);
  modelChatStatus.textContent = data.daemon_connected === false
    ? 'Warm daemon unavailable' : 'Resident daemon connected';
}

async function tickModels() {
  if (isGrid || modelPolling) return;
  modelPolling = true;
  try {
    const response = await fetch('/api/models' + qSerial(), {cache: 'no-store'});
    const data = await response.json();
    if (response.ok && data.ok !== false) renderModels(data);
  } catch (error) {
    modelChatStatus.textContent = 'Model control unavailable';
  } finally {
    modelPolling = false;
  }
}

function renderModelChat(provider) {
  modelChat.textContent = '';
  const history = modelHistories[provider] || [];
  if (!history.length) {
    const note = document.createElement('div');
    note.className = 'model-message meta';
    note.textContent = provider === 'agent_chain'
      ? 'Agent requests run through the same configured evaluator as a real agent turn.'
      : (provider === 'functiongemma'
          ? 'FunctionGemma v10 is selector-tuned; raw replies may be terse or tool-shaped.'
          : 'Gemma 4 is the deeper semantic reviewer and may include reasoning before its answer.');
    modelChat.appendChild(note);
  }
  history.forEach(message => {
    const bubble = document.createElement('div');
    bubble.className = 'model-message ' + message.role;
    bubble.textContent = message.content;
    modelChat.appendChild(bubble);
  });
  modelChat.scrollTop = modelChat.scrollHeight;
}

async function sendModelMessage() {
  const provider = modelChatProvider.value;
  const content = modelPrompt.value.trim();
  if (!content) return;
  const history = modelHistories[provider];
  const agentRequest = modelRequestKind.value === 'agent';
  let parsedRequest = null;
  if (agentRequest) {
    try {
      parsedRequest = JSON.parse(content);
    } catch (error) {
      modelChatStatus.textContent = 'Agent request must be valid JSON: ' + error.message;
      return;
    }
  }
  history.push({role: 'user', content: content});
  if (!agentRequest) modelPrompt.value = '';
  renderModelChat(provider);
  modelChatStatus.textContent = agentRequest
    ? 'Sending through the real agent selector…'
    : 'Generating locally…';
  document.getElementById('model-send').disabled = true;
  try {
    if (agentRequest) {
      const data = await modelPost('agent-test', {
        provider: 'agent_chain',
        request: parsedRequest,
      });
      const exchanges = Array.isArray(data.exchanges) ? data.exchanges : [];
      const exchange = data.exchange || {};
      const raw = exchanges.length
        ? exchanges.map((item, index) => {
            const output = typeof item.output === 'string' ? item.output : prettyJson(item.output);
            return '[' + (index + 1) + '] ' + (item.provider || 'model') + '\\n' + output;
          }).join('\\n\\n')
        : (data.provider_error || 'No model output');
      const verdict = data.status === 'handoff'
        ? 'HANDOFF'
        : (data.selected_candidate
            ? ('SELECTED #' + data.selected_id + ' · ' +
              (data.selected_candidate.purpose || 'candidate'))
            : (String(data.status || 'NO VALID SELECTION').toUpperCase() +
              (data.provider_error ? ' · ' + data.provider_error : '')));
      const trace = data.decision && data.decision.selection_trace
        ? '\\n\\nEVALUATOR TRACE\\n' + prettyJson(data.decision.selection_trace)
        : '';
      history.push({role: 'assistant', content: raw + '\\n\\n' + verdict + trace});
      modelChatStatus.textContent = (exchange.duration_ms == null ? '' : exchange.duration_ms + ' ms · ') +
        (data.providers || []).join(' → ') + ' · ' + verdict;
    } else {
      const data = await modelPost('chat', {
        provider: provider,
        messages: history.filter(message => ['user', 'assistant'].includes(message.role)).slice(-30),
        max_tokens: Number(modelMaxTokens.value || 128),
      });
      history.push({role: 'assistant', content: String(data.output || '')});
      modelChatStatus.textContent = data.duration_ms + ' ms · ' +
        modelFormatTokens(data.input_tokens) + ' → ' + modelFormatTokens(data.output_tokens) + ' tokens';
    }
  } catch (error) {
    history.push({role: 'meta', content: 'Error: ' + error.message});
    modelChatStatus.textContent = error.message;
  } finally {
    document.getElementById('model-send').disabled = false;
    renderModelChat(provider);
    tickModels();
  }
}

modelIntercept.addEventListener('change', async () => {
  const enabled = modelIntercept.checked;
  modelInterceptLabel.textContent = enabled ? 'Enabling interception…' : 'Disabling immediately…';
  try { renderModels(await modelPost('set-intercept', {enabled: enabled})); }
  catch (error) { modelChatStatus.textContent = error.message; tickModels(); }
});
document.querySelectorAll('.model-provider-toggle').forEach(toggle => {
  toggle.addEventListener('change', async event => {
    const provider = event.target.closest('.model-card').dataset.provider;
    try { renderModels(await modelPost('set-provider', {provider: provider, enabled: event.target.checked})); }
    catch (error) { modelChatStatus.textContent = error.message; tickModels(); }
  });
});
document.querySelectorAll('.model-load, .model-unload').forEach(button => {
  button.addEventListener('click', async event => {
    const card = event.target.closest('.model-card');
    const provider = card.dataset.provider;
    const action = event.target.dataset.action;
    modelChatStatus.textContent = (action === 'load' ? 'Loading ' : 'Unloading ') + provider + '…';
    card.classList.add('busy');
    try {
      await modelPost(action, {provider: provider});
      modelChatStatus.textContent = provider + (action === 'load' ? ' is resident.' : ' unloaded.');
    } catch (error) { modelChatStatus.textContent = error.message; }
    tickModels();
  });
});
document.getElementById('model-send').addEventListener('click', sendModelMessage);
modelPrompt.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') sendModelMessage();
});
modelChatProvider.addEventListener('change', () => renderModelChat(modelChatProvider.value));
modelRequestKind.addEventListener('change', () => updateModelRequestShape(true));
modelRequestSample.addEventListener('change', () => {
  if (!modelRequestSample.value) return;
  modelRequestKind.value = 'agent';
  updateModelRequestShape(true);
});
document.getElementById('model-clear').addEventListener('click', async () => {
  try { renderModels(await modelPost('clear', {})); }
  catch (error) { modelChatStatus.textContent = error.message; }
});
renderModelChat(modelChatProvider.value);
updateModelRequestShape(false);

document.getElementById('knowledge-confirm-cancel').addEventListener(
  'click', () => closeKnowledgeConfirmation(false)
);
document.getElementById('knowledge-confirm-submit').addEventListener(
  'click', () => closeKnowledgeConfirmation(true)
);
knowledgeConfirmDialog.addEventListener('cancel', event => {
  event.preventDefault();
  closeKnowledgeConfirmation(false);
});
document.getElementById('map-clear').addEventListener('click', () => {
  const data = window.__auaKnowledgeData || {};
  if (!data.package || !data.known) return;
  runNavigationAction(
    'map-clear',
    {package: data.package},
    {
      title: 'Clear this app map?',
      message: 'All learned screens and routes for ' + data.package + ' will be removed.',
      label: 'Clear map',
      phrase: 'CLEAR MAP ' + data.package,
    }
  );
});
document.getElementById('flows-clear').addEventListener('click', () => {
  runNavigationAction(
    'flows-clear',
    {},
    {
      title: 'Clear every saved flow?',
      message: 'Every indexed flow for every app will be permanently removed.',
      label: 'Clear flows',
      phrase: 'CLEAR ALL FLOWS',
    }
  );
});
document.getElementById('knowledge-clear-all').addEventListener('click', () => {
  runNavigationAction(
    'clear-all',
    {},
    {
      title: 'Clear the navigation library?',
      message: 'All app maps, recorded routes, and saved flows will be permanently removed.',
      label: 'Clear everything',
      phrase: 'CLEAR ALL NAVIGATION',
    }
  );
});

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

screenAnalyze.addEventListener('click', analyzeScreen);
document.getElementById('inspection-live').addEventListener('click', resumeLiveFrame);
inspectionJsonSearch.addEventListener('input', () => updateInspectionJsonSearch());
inspectionJsonSearch.addEventListener('keydown', event => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  updateInspectionJsonSearch(event.shiftKey ? -1 : 1);
});
document.getElementById('inspection-json-prev').addEventListener(
  'click', () => updateInspectionJsonSearch(-1)
);
document.getElementById('inspection-json-next').addEventListener(
  'click', () => updateInspectionJsonSearch(1)
);
inspectionClickableOnly.addEventListener('change', () => {
  elementOverlay.classList.toggle('clickable-only', inspectionClickableOnly.checked);
});

const phoneQrButton = document.getElementById('phone-qr-button');
const phoneQrDialog = document.getElementById('phone-qr-dialog');
const phoneQrCopy = document.getElementById('phone-qr-copy');
const phoneQrImage = document.getElementById('phone-qr-image');
document.getElementById('phone-qr-close').addEventListener('click', () => phoneQrDialog.close());
phoneQrDialog.addEventListener('click', event => {
  if (event.target === phoneQrDialog) phoneQrDialog.close();
});
if (PHONE_ACCESS_URL) {
  document.getElementById('phone-qr-url').textContent = PHONE_ACCESS_URL;
  phoneQrButton.classList.remove('hidden');
  phoneQrButton.addEventListener('click', () => {
    // Load only after the token-entry navigation has finished committing its cookie.
    if (!phoneQrImage.getAttribute('src')) phoneQrImage.src = '/api/dashboard-access-qr.svg';
    phoneQrDialog.showModal();
  });
  phoneQrCopy.addEventListener('click', async () => {
    const copied = await copyText(PHONE_ACCESS_URL);
    phoneQrCopy.textContent = copied ? 'Copied' : 'Copy failed';
    setTimeout(() => { phoneQrCopy.textContent = 'Copy link'; }, 1600);
  });
}

if (isGrid) {
  if (detachedSerial) {
    const notice = document.getElementById('device-notice');
    notice.textContent = detachedSerial + ' disconnected and was removed from the dashboard.';
    notice.classList.remove('hidden');
    const clean = new URL(window.location.href);
    clean.searchParams.delete('detached');
    history.replaceState({}, '', clean.pathname + clean.search);
    setTimeout(() => notice.classList.add('hidden'), 7000);
  }
  tickGrid();
  setInterval(tickGrid, Math.max(POLL_MS, 800));
} else {
  if (focusSerial && frame) {
    frame.src = '/api/frame.jpg?serial=' + encodeURIComponent(focusSerial);
  }
  tickStatus(); tickEvents(); tickMap(); tickLogcat(); tickModels();
  setInterval(() => { tickStatus(); tickEvents(); }, POLL_MS);
  setInterval(() => { tickMap(); tickLogcat(); }, MAP_MS);
  setInterval(tickModels, 1000);
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
        bind_host: str = "127.0.0.1",
        require_auth: bool = False,
        access_token: str | None = None,
    ) -> None:
        self.serials = list(serials)
        self._online_serials = set(serials)
        self.focus = focus or (serials[0] if serials else None)
        self.mode = mode  # "grid" | "detail"
        self.cache_dir = cache_dir
        self.ensures = ensures
        self.poll_ms = poll_ms
        self.config = config
        self.bind_host = bind_host
        self.require_auth = bool(require_auth)
        self.access_token = access_token or ""
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
        self._model_status_cache: dict[str, Any] | None = None
        self._inspection_lock = threading.Lock()
        self._inspections: dict[str, dict[str, Any]] = {}
        from . import leases

        self._inspection_owner = leases.resolve_owner(f"aua-dashboard-{os.getpid()}")
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

    def _forget_detached_runtime(self, serials: set[str]) -> None:
        """Drop only live/runtime state for targets discovery proved are offline."""
        if not serials:
            return
        from . import leases

        for serial in serials:
            self.ensures.pop(serial, None)
            self._fallback.pop(serial, None)
            self._capture_live.pop(serial, None)
            self._pkg_cache.pop(serial, None)
            self._map_cache.pop(serial, None)
            self._runtime_cache.pop(serial, None)
            with self._inspection_lock:
                self._inspections.pop(serial, None)
            # Analyze/tap operations use a process-bound dashboard owner. If the emulator dies,
            # keeping that live owner's lease would block the same serial after it boots again.
            with contextlib.suppress(Exception):
                leases.release(self.cache_dir, serial, owner=self._inspection_owner)

    def _require_online(self, serial: str) -> None:
        """Refuse a device action when authoritative discovery says it detached."""
        online, error = discover_online_serials(self.config)
        self.discovery_error = error
        if error is not None:
            return
        detached = self._online_serials.difference(online)
        self._online_serials = set(online)
        self._forget_detached_runtime(detached)
        self.serials = list(dict.fromkeys([*self.serials, *online]))
        if serial in online:
            return
        raise UsageError(
            f"device {serial!r} disconnected and was removed from the dashboard",
            code="dashboard_device_detached",
            hint="Return to All devices; it will reappear automatically if it reconnects.",
        )

    def _daemon_call(
        self,
        serial: str,
        cmd: str,
        timeout: float = 1.5,
        *,
        journal: bool = False,
        owner: str | None = None,
        uncertain_is_error: bool = False,
        **args: Any,
    ) -> dict[str, Any] | None:
        try:
            from . import daemon as daemon_mod

            sock = daemon_mod.socket_path(self.config, serial)
            if not Path(sock).exists():
                base = os.path.expanduser(self.config.daemon.socket)
                if Path(base).exists():
                    sock = base
                else:
                    return None
            # Health checks are lease-free and anonymous. Reusing an owner-bearing client for
            # ping made the daemon adopt that owner without claiming the device; the following
            # Analyze then saw the same owner, skipped adoption, and failed validation against a
            # lease that had never existed.
            with daemon_mod.DaemonClient(sock, timeout=timeout) as health:
                if not health.ping():
                    return None
            with daemon_mod.DaemonClient(sock, timeout=timeout, owner=owner) as client:
                resp = client.call(cmd, journal=journal, **args)
                if isinstance(resp, dict):
                    result = resp.get("result")
                    return result if isinstance(result, dict) else resp
        except DaemonOutcomeUnknownError:
            # A device mutation may already have happened. Never turn transport uncertainty
            # into a second tap by treating it like a daemon that was never reached.
            if uncertain_is_error:
                raise
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("daemon %s skipped: %s", cmd, exc)
        return None

    def _model_daemon_call(
        self,
        serial: str,
        cmd: str,
        *,
        timeout: float = 300.0,
        **args: Any,
    ) -> dict[str, Any] | None:
        """Reach the resident model host, starting it when the operator got here first."""

        result = self._daemon_call(serial, cmd, timeout=timeout, **args)
        if result is not None:
            return result
        try:
            from . import daemon as daemon_mod

            started = daemon_mod.start(self.config, serial=serial)
        except Exception as exc:  # noqa: BLE001
            logger.debug("dashboard could not start model host: %s", exc)
            return None
        if not bool(started.get("running")):
            return None
        return self._daemon_call(serial, cmd, timeout=timeout, **args)

    def _inspection_daemon_call(
        self,
        serial: str,
        cmd: str,
        *,
        timeout: float = 90.0,
        **args: Any,
    ) -> dict[str, Any] | None:
        """Run a dashboard-authored device command through the shared warm Engine."""

        call_args = {
            "timeout": timeout,
            "journal": True,
            "owner": self._inspection_owner,
            "uncertain_is_error": cmd in {"tap", "goto", "flow_run"},
            **args,
        }
        result = self._daemon_call(serial, cmd, **call_args)
        if result is not None:
            return result
        try:
            from . import daemon as daemon_mod

            started = daemon_mod.start(self.config, serial=serial)
        except Exception as exc:  # noqa: BLE001
            logger.debug("dashboard could not start inspection host: %s", exc)
            return None
        if not bool(started.get("running")):
            return None
        return self._daemon_call(serial, cmd, **call_args)

    def _inspection_path(self, serial: str, inspection_id: str) -> Path:
        safe_serial = "".join(char if char.isalnum() or char in "-_." else "_" for char in serial)
        root = self.cache_dir / "dashboard-inspection" / safe_serial
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{inspection_id}.png"

    @staticmethod
    def _inspection_error(result: dict[str, Any] | None) -> None:
        if result is None:
            raise UsageError("the dashboard could not start the AUA inspection host")
        error = result.get("error")
        if result.get("ok") is False and isinstance(error, dict):
            raise UsageError(
                str(error.get("message") or "AUA inspection failed"),
                code=str(error.get("code") or "dashboard_inspection"),
                hint=str(error.get("hint")) if error.get("hint") else None,
            )

    def _store_inspection(
        self,
        serial: str,
        inspection_id: str,
        frame_path: Path,
        raw_result: dict[str, Any],
        view: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record = {
            "inspection_id": inspection_id,
            "frame_path": frame_path,
            "result": raw_result,
            "view": view or {},
            "busy": False,
        }
        with self._inspection_lock:
            previous = self._inspections.get(serial)
            self._inspections[serial] = record
        old_path = previous.get("frame_path") if previous else None
        if isinstance(old_path, Path) and old_path != frame_path:
            with contextlib.suppress(OSError):
                old_path.unlink()
        return {
            "ok": True,
            "inspection_id": inspection_id,
            "frame_url": f"/api/inspection-frame?serial={serial}&inspection_id={inspection_id}",
            "result": raw_result,
            "view": view,
        }

    def inspection_operation(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Analyze one exact frame, or consume its id once for a guarded tap-and-analyze."""

        requested = payload.get("serial") or self.focus
        if not isinstance(requested, str) or not requested:
            raise UsageError("dashboard inspection needs a device serial")
        self._require_online(requested)
        ser = self._scoped_serial(requested)
        inspection_id = secrets.token_urlsafe(12)
        frame_path = self._inspection_path(ser, inspection_id)
        if action == "analyze":
            result = self._inspection_daemon_call(
                ser,
                "analyze",
                source="auto",
                no_cache=True,
                with_image=str(frame_path),
            )
            self._inspection_error(result)
            assert result is not None
            return self._store_inspection(ser, inspection_id, frame_path, result, result)
        if action != "tap":
            raise UsageError(f"unknown dashboard inspection action {action!r}")

        source_id = payload.get("inspection_id")
        element_id = payload.get("element_id")
        if not isinstance(source_id, str) or not source_id:
            raise UsageError("dashboard tap needs the analysis frame id")
        if isinstance(element_id, bool) or not isinstance(element_id, int) or element_id < 0:
            raise UsageError("dashboard tap needs a non-negative AUA element id")
        with self._inspection_lock:
            current = self._inspections.get(ser)
            if current is None or current.get("inspection_id") != source_id:
                raise UsageError(
                    "that analysis frame is no longer current",
                    code="stale_dashboard_analysis",
                    hint="Click Analyze again and choose an id from the fresh overlay.",
                )
            if current.get("busy"):
                raise UsageError("that analysis frame is already being acted on")
            view = current.get("view") or {}
            valid_ids = {
                item.get("id") for item in (view.get("elements") or []) if isinstance(item, dict)
            }
            if element_id not in valid_ids:
                raise UsageError(f"element id {element_id} is not in that analysis frame")
            # Consume the frame before sending the mutation. A double-click can never replay it.
            current["busy"] = True
        try:
            result = self._inspection_daemon_call(
                ser,
                "tap",
                element_id=element_id,
                observe=True,
                with_image=str(frame_path),
            )
            self._inspection_error(result)
        except Exception:
            with self._inspection_lock:
                if self._inspections.get(ser) is current:
                    current["busy"] = False
            raise
        assert result is not None
        observation = result.get("observation")
        view = observation if isinstance(observation, dict) else None
        return self._store_inspection(ser, inspection_id, frame_path, result, view)

    def inspection_frame(self, serial: str, inspection_id: str) -> tuple[bytes, str] | None:
        with self._inspection_lock:
            current = self._inspections.get(serial)
            if current is None or current.get("inspection_id") != inspection_id:
                return None
            path = current.get("frame_path")
        if not isinstance(path, Path) or not path.is_file():
            return None
        try:
            return path.read_bytes(), "image/png"
        except OSError:
            return None

    def model_payload(self, serial: str | None = None) -> dict[str, Any]:
        """Live daemon model state plus out-of-band controls and inference telemetry."""

        from .model_control import ModelControlStore

        ser = self._scoped_serial(serial)
        store = ModelControlStore(self.config)
        live = self._daemon_call(ser, "model_status", timeout=0.8, limit=120)
        if live and live.get("ok"):
            self._model_status_cache = live
        out = dict(self._model_status_cache or {})
        if not out:
            # Readiness remains useful before a daemon exists; this Engine is host-only until a
            # load/chat action and does not connect to Android merely to report provider status.
            out = self._dashboard_engine().model_control_status(limit=120)
            out["daemon_connected"] = False
        else:
            out["daemon_connected"] = True
        # These two files remain observable even while the daemon is busy generating, which is
        # why the dashboard can show RUNNING and apply OFF without waiting behind inference.
        out["control"] = store.read_state()
        out["events"] = store.events(limit=120)
        out["serial"] = ser
        return out

    def model_operation(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        from .model_control import MODEL_NAMES, ModelControlStore

        ser = self._scoped_serial(payload.get("serial"))
        provider = payload.get("provider")
        enabled = payload.get("enabled")
        store = ModelControlStore(self.config)
        if action == "set-intercept":
            if not isinstance(enabled, bool):
                raise UsageError("model intercept action requires enabled=true or false")
            store.update(intercept_enabled=enabled)
            store.record(
                {
                    "source": "dashboard",
                    "phase": "complete",
                    "operation": "intercept_on" if enabled else "intercept_off",
                }
            )
            return self.model_payload(ser)
        if action == "set-provider":
            if provider not in MODEL_NAMES or not isinstance(enabled, bool):
                raise UsageError("model provider action needs a known provider and boolean enabled")
            store.update(provider=provider, provider_enabled=enabled)
            store.record(
                {
                    "provider": provider,
                    "source": "dashboard",
                    "phase": "complete",
                    "operation": "enable" if enabled else "disable",
                }
            )
            return self.model_payload(ser)
        if action == "clear":
            store.clear_events()
            return self.model_payload(ser)
        if action in {"load", "unload"}:
            if provider not in MODEL_NAMES:
                raise UsageError("model load action needs a known provider")
            result = self._model_daemon_call(
                ser,
                "model_action",
                timeout=300.0,
                action=action,
                provider=provider,
            )
            if result is None:
                raise UsageError("the dashboard could not start the local model host")
            self._model_status_cache = None
            return result
        if action == "chat":
            if provider not in MODEL_NAMES:
                raise UsageError("model chat needs a known provider")
            messages = payload.get("messages")
            max_tokens = payload.get("max_tokens")
            result = self._model_daemon_call(
                ser,
                "model_chat",
                timeout=300.0,
                provider=provider,
                messages=messages,
                max_tokens=max_tokens,
            )
            if result is None:
                raise UsageError("the dashboard could not start the local model host")
            self._model_status_cache = None
            return result
        if action == "agent-test":
            if provider != "agent_chain":
                raise UsageError("agent-shaped model test must use the configured agent chain")
            request = payload.get("request")
            result = self._model_daemon_call(
                ser,
                "model_agent_test",
                timeout=300.0,
                provider=provider,
                request=request,
            )
            if result is None:
                raise UsageError("the dashboard could not start the local model host")
            self._model_status_cache = None
            return result
        raise UsageError(f"unknown dashboard model action {action!r}")

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

    def journal_detail(self, detail_id: str, serial: str | None = None) -> dict[str, Any] | None:
        from . import journal as journal_mod

        if (
            not detail_id
            or len(detail_id) > 128
            or not all(char.isalnum() or char in "-_." for char in detail_id)
        ):
            raise UsageError("invalid dashboard journal detail id")
        return journal_mod.read_detail(self.cache_dir, self._scoped_serial(serial), detail_id)

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
            "flows": [],
            "map_count": 0,
            "serial": ser,
        }
        try:
            from .flows import FlowStore

            flow_store = FlowStore(self.config.memory)
            flow_entries = flow_store.list()
            for entry in flow_entries:
                detail = dict(entry)
                path = detail.get("path")
                if path and not detail.get("error"):
                    try:
                        flow = flow_store.load_file(Path(path))
                        detail["steps_detail"] = [
                            _dashboard_step_payload(step) for step in flow.steps
                        ]
                    except Exception as exc:  # noqa: BLE001 — expose a broken saved flow
                        detail["error"] = str(exc)
                # A local absolute memory path is implementation detail, not useful UI.
                detail.pop("path", None)
                out["flows"].append(detail)
            out["flow_count"] = len(flow_entries)

            from .memory import AppMemoryStore

            store = AppMemoryStore(self.config.memory)
            out["map_count"] = len(store.list_apps())
            pkg = self.foreground_package(ser)
            out["package"] = pkg
            if not pkg:
                if ser:
                    self._map_cache[ser] = (out, now)
                return out
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
                    "id": s.id,
                    "canonical_name": s.canonical_name,
                    "logical_name": s.logical_name,
                    "aliases": s.aliases,
                    "activity": s.activity,
                    "visit_count": s.visit_count,
                    "stale": s.stale,
                    "context_id": s.context_id,
                    "surface": s.surface,
                    "tier": s.tier,
                    "anchors": s.anchors,
                    "notes": s.notes,
                    "last_verified": s.last_verified,
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
                    "id": e.id,
                    "context_id": e.context_id,
                    "guards": e.guards,
                    "verification_count": e.verification_count,
                    "last_seen": e.last_seen,
                    "steps": [_dashboard_step_payload(step) for step in e.steps],
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

    @staticmethod
    def _navigation_text(payload: dict[str, Any], field: str, *, maximum: int = 300) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise UsageError(f"navigation action needs a non-empty {field}")
        clean = value.strip()
        if len(clean) > maximum:
            raise UsageError(f"navigation action {field} exceeds {maximum} characters")
        return clean

    @staticmethod
    def _require_navigation_confirmation(payload: dict[str, Any], phrase: str) -> None:
        if payload.get("confirmation") != phrase:
            raise UsageError(
                f"confirm this navigation-library action with {phrase!r}",
                code="navigation_confirmation_required",
            )

    def navigation_operation(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run shared navigation commands or explicitly prune their host-side library."""

        ser = self._scoped_serial(payload.get("serial"))
        if action == "goto":
            target = self._navigation_text(payload, "target")
            result = self._inspection_daemon_call(ser, "goto", timeout=300.0, goal=target)
            self._inspection_error(result)
            return {"ok": True, "action": "goto", "target": target, "result": result}
        if action == "flow-run":
            ref = self._navigation_text(payload, "ref")
            self._require_navigation_confirmation(payload, f"RUN FLOW {ref}")
            result = self._inspection_daemon_call(ser, "flow_run", timeout=300.0, name=ref)
            self._inspection_error(result)
            return {"ok": True, "action": "flow-run", "flow": ref, "result": result}
        if action == "flow-delete":
            ref = self._navigation_text(payload, "ref")
            self._require_navigation_confirmation(payload, f"DELETE FLOW {ref}")
            result = self._inspection_daemon_call(ser, "flow_delete", timeout=30.0, name=ref)
            self._inspection_error(result)
            self._map_cache.clear()
            return {"ok": True, "action": "flow-delete", "flow": ref, "result": result}

        from .flows import FlowStore
        from .memory import AppMemoryStore

        memory = AppMemoryStore(self.config.memory)
        flows = FlowStore(self.config.memory)
        if action == "route-delete":
            package = self._navigation_text(payload, "package", maximum=255)
            route_id = self._navigation_text(payload, "route_id", maximum=255)
            current = self.map_payload(ser).get("package")
            if current != package:
                raise UsageError(
                    "the foreground app changed; refresh before deleting this route",
                    code="dashboard_stale_map",
                )
            self._require_navigation_confirmation(payload, f"DELETE ROUTE {route_id}")
            forgotten = memory.forget_route(package, route_id).get("forgot")
            self._map_cache.clear()
            return {
                "ok": True,
                "action": "route-delete",
                "package": package,
                "route_id": route_id,
                "deleted": forgotten is not None,
            }
        if action == "map-clear":
            package = self._navigation_text(payload, "package", maximum=255)
            current = self.map_payload(ser).get("package")
            if current != package:
                raise UsageError(
                    "the foreground app changed; refresh before clearing its map",
                    code="dashboard_stale_map",
                )
            self._require_navigation_confirmation(payload, f"CLEAR MAP {package}")
            map_deleted = memory.forget(package).get("forgot") is not None
            self._map_cache.clear()
            return {
                "ok": True,
                "action": "map-clear",
                "package": package,
                "deleted": map_deleted,
            }
        if action == "flows-clear":
            self._require_navigation_confirmation(payload, "CLEAR ALL FLOWS")
            deleted_flows = flows.clear()
            self._map_cache.clear()
            return {"ok": True, "action": "flows-clear", "deleted": len(deleted_flows)}
        if action == "clear-all":
            self._require_navigation_confirmation(payload, "CLEAR ALL NAVIGATION")
            packages = memory.list_apps()
            maps_deleted = sum(
                memory.forget(package).get("forgot") is not None for package in packages
            )
            flows_deleted = len(flows.clear())
            self._map_cache.clear()
            return {
                "ok": True,
                "action": "clear-all",
                "maps_deleted": maps_deleted,
                "flows_deleted": flows_deleted,
            }
        raise UsageError(f"unknown dashboard navigation action {action!r}")

    def journal_operation(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Clear the compact and detailed journal files visible in this device view."""

        if action != "clear":
            raise UsageError(f"unknown dashboard journal action {action!r}")
        ser = self._scoped_serial(payload.get("serial"))
        self._require_navigation_confirmation(payload, f"CLEAR JOURNAL {ser}")
        from . import journal as journal_mod

        deleted = journal_mod.clear(self.cache_dir, ser, include_host=True)
        return {"ok": True, "action": "journal-clear", "serial": ser, "deleted": len(deleted)}

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
        known = (
            [serial for serial in self.serials if serial in self._online_serials]
            if self.mode == "grid"
            else list(self.serials)
        )
        detached: list[str] = []
        if self.mode == "grid":
            # Grid dashboards intentionally discover devices as they appear. A
            # successful discovery is authoritative in both directions: keeping the
            # old union here left dead emulators visible forever. On discovery failure,
            # preserve the last good list instead of mistaking an ADB outage for every
            # device disconnecting at once.
            online, self.discovery_error = discover_online_serials(self.config)
            if self.discovery_error is None:
                detached = [serial for serial in known if serial not in online]
                self._forget_detached_runtime(set(detached))
                known = online
                self._online_serials = set(online)
                # Keep the ever-seen set for request scoping and historical journal reads;
                # only ``known`` is rendered. Reconnected serials are ensured again because
                # detached runtime state was cleared above.
                self.serials = list(dict.fromkeys([*self.serials, *online]))
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
            "detached_serials": detached,
            "discovery_error": self.discovery_error,
        }

    def status(self, serial: str | None = None) -> dict[str, Any]:
        ser = serial or self.focus
        if not ser:
            return {"ok": False, "error": "no serial"}
        online, discovery_error = discover_online_serials(self.config)
        self.discovery_error = discovery_error
        if discovery_error is None and ser not in online:
            self._online_serials = set(online)
            self._forget_detached_runtime({ser})
            return {
                "ok": False,
                "detached": True,
                "serial": ser,
                "online_serials": online,
                "error": f"device {ser!r} is no longer attached",
            }
        if discovery_error is None:
            self._online_serials = set(online)
            self.serials = list(dict.fromkeys([*self.serials, *online]))
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

    def _dashboard_engine(self) -> Any:
        """A lazily built host-only engine for dashboard control operations.

        Reads go straight to the proxy capability, but arming or removing a rule changes
        state the device keeps until something clears it, and only the engine knows how to
        journal that undo first. None of the mock methods it is used for connect to the
        device. Model readiness is likewise host-only, so neither use competes with a running
        agent for the UiAutomation slot.
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
        engine = self._dashboard_engine()
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

    def log_lines(self, serial: str, lines: int = 80, *, app_id: str | None = None) -> list[str]:
        n = max(1, min(int(lines), 500))
        try:
            return self.platform.recent_logs(serial, limit=n, app_id=app_id)
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
            if getattr(self, "_issue_access_cookie", False):
                self.send_header(
                    "Set-Cookie",
                    f"{_ACCESS_COOKIE}={state.access_token}; Path=/; HttpOnly; "
                    "SameSite=Strict; Max-Age=2592000",
                )
                self._issue_access_cookie = False
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
            # Browsers cancel superseded frame polls aggressively. That is a normal client
            # disconnect, not a dashboard failure worth an 18-line server traceback.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def _json(self, payload: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _redirect(self, location: str) -> None:
            self.send_response(303)
            self.send_header("Location", location)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

        def _cookie_access_token(self) -> str:
            raw = self.headers.get("Cookie", "")
            try:
                cookie = http.cookies.SimpleCookie(raw)
            except http.cookies.CookieError:
                return ""
            morsel = cookie.get(_ACCESS_COOKIE)
            return morsel.value if morsel is not None else ""

        def _has_dashboard_access(self) -> bool:
            if not state.require_auth:
                return True
            supplied = self.headers.get("X-AUA-Dashboard-Access", "")
            if not supplied:
                supplied = self._cookie_access_token()
            return bool(supplied) and secrets.compare_digest(supplied, state.access_token)

        def _authorize_get(self, parsed: Any, qs: dict[str, list[str]]) -> bool:
            if self._has_dashboard_access():
                return True
            supplied = (qs.get("token") or [""])[0]
            if supplied and secrets.compare_digest(supplied, state.access_token):
                # Serve the token-bearing request itself and let the page remove the token
                # with history.replaceState. Android Chrome can discard Set-Cookie when the
                # QR entry response is a 303, leaving the redirected page unauthorized.
                self._issue_access_cookie = True
                return True
            if parsed.path.startswith("/api/"):
                self._json(
                    {
                        "ok": False,
                        "error": {
                            "code": "dashboard_auth",
                            "message": "dashboard access token required",
                        },
                    },
                    401,
                )
            else:
                body = (
                    b"<!doctype html><meta name=viewport content='width=device-width'>"
                    b"<title>AuA Dashboard</title><style>body{margin:0;min-height:100vh;display:grid;"
                    b"place-items:center;background:#090b14;color:#e8edf7;font:16px system-ui}"
                    b"main{max-width:28rem;padding:2rem;text-align:center}p{color:#9aa5b7;line-height:1.5}"
                    b"</style><main><h1>AuA Dashboard</h1><p>This network dashboard needs its "
                    b"private access link. Run <code>aua dashboard status</code> on the laptop to "
                    b"print it.</p></main>"
                )
                self._send(401, body, "text/html; charset=utf-8")
            return False

        def _qs_serial(self, qs: dict[str, list[str]]) -> str | None:
            raw = (qs.get("serial") or [""])[0].strip()
            return raw or state.focus

        def _phone_access_url(self) -> str:
            if state.bind_host != "0.0.0.0":
                return ""
            urls = _service_urls(
                port=int(getattr(self.server, "server_port", DEFAULT_DASHBOARD_PORT)),
                lan=True,
                access_token=state.access_token,
            )
            return str((urls.get("lan_access_urls") or [""])[0])

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
            if path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "service": _SERVICE_ID,
                        "pid": os.getpid(),
                        "bind": state.bind_host,
                        "lan": state.bind_host == "0.0.0.0",
                        "authenticated": state.require_auth,
                    }
                )
                return
            if not self._authorize_get(parsed, qs):
                return
            if path in ("/", "/index.html"):
                focus = (qs.get("serial") or [""])[0].strip()
                if focus and focus not in state.serials:
                    safe_focus = len(focus) <= 255 and all(
                        char.isalnum() or char in "._:-" for char in focus
                    )
                    online, error = discover_online_serials(state.config)
                    if safe_focus and error is None and focus not in online:
                        self._redirect("/?" + urlencode({"detached": focus}))
                        return
                    if focus in online:
                        state.serials.append(focus)
                        state._online_serials.add(focus)
                    else:
                        self._send(404, b"device not part of this dashboard session", "text/plain")
                        return
                mode = "detail" if focus else state.mode
                serial_boot = state.focus or ""
                html = (
                    _DASHBOARD_HTML.replace("__POLL_MS__", str(state.poll_ms))
                    .replace("__MODE_JSON__", _script_json(mode if focus else state.mode))
                    .replace("__SERIAL_JSON__", _script_json(serial_boot))
                    .replace("__PHONE_ACCESS_URL_JSON__", _script_json(self._phone_access_url()))
                    .replace("__DATABASE_TOKEN__", state.database_token)
                )
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/assets/aua-dashboard-logo.png":
                self._send(200, _DASHBOARD_LOGO.read_bytes(), "image/png")
                return
            if path == "/api/dashboard-access-qr.svg":
                phone_url = self._phone_access_url()
                if not phone_url:
                    self._json(
                        {
                            "ok": False,
                            "error": {
                                "code": "dashboard_local_only",
                                "message": "phone QR is available after starting with --lan",
                            },
                        },
                        404,
                    )
                    return
                self._send(200, _qr_svg(phone_url), "image/svg+xml; charset=utf-8")
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
                ser = self._qs_serial(qs)
                if not ser:
                    self._json({"ok": False, "error": "no serial"}, 400)
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
            if path == "/api/inspection-frame":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                inspection_id = (qs.get("inspection_id") or [""])[0].strip()
                frame = state.inspection_frame(ser, inspection_id)
                if frame is None:
                    self._send(404, b"inspection frame not found", "text/plain")
                else:
                    self._send(200, frame[0], frame[1])
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
            if path == "/api/models":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                self._json(state.model_payload(ser))
                return
            if path == "/api/logcat":
                ser = self._scoped_qs_serial(qs)
                if ser is None:
                    return
                lines_raw = (qs.get("lines") or ["80"])[0]
                app_id = (qs.get("app_id") or [""])[0].strip() or None
                if app_id and (
                    len(app_id) > 255 or not all(char.isalnum() or char in "._-" for char in app_id)
                ):
                    self._json({"ok": False, "error": "invalid app id", "lines": []}, 400)
                    return
                try:
                    n = max(1, min(int(lines_raw), 500))
                except ValueError:
                    n = 80
                self._json(
                    {
                        "ok": True,
                        "app_id": app_id,
                        "lines": state.log_lines(ser, n, app_id=app_id) if ser else [],
                    }
                )
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
            if not self._has_dashboard_access():
                self._json(
                    {
                        "ok": False,
                        "error": {
                            "code": "dashboard_auth",
                            "message": "dashboard access token required",
                        },
                    },
                    401,
                )
                return
            prefix = ""
            for candidate in (
                "/api/database/",
                "/api/proxy/",
                "/api/models/",
                "/api/inspect/",
                "/api/navigation/",
                "/api/journal/",
            ):
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
                            "message": "dashboard request body must be between 1 byte and 1 MB",
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
                elif prefix == "/api/models/":
                    result = state.model_operation(action, payload)
                elif prefix == "/api/inspect/":
                    result = state.inspection_operation(action, payload)
                elif prefix == "/api/navigation/":
                    result = state.navigation_operation(action, payload)
                elif prefix == "/api/journal/":
                    result = state.journal_operation(action, payload)
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
                logger.exception("dashboard request failed")
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
    bind_host: str = "127.0.0.1",
    exact_port: bool = False,
    require_auth: bool = False,
    access_token: str | None = None,
    announce: bool = True,
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

    listen = int(port) if exact_port else _pick_free_port(port)
    if bind_host not in {"127.0.0.1", "0.0.0.0"}:
        raise UsageError("dashboard bind host must be 127.0.0.1 or 0.0.0.0")
    if require_auth and not access_token:
        raise UsageError("network dashboard requires an access token")
    state = _DashboardState(
        serials=serials,
        focus=focus,
        mode=mode,
        cache_dir=cache,
        ensures=ensures,
        poll_ms=max(200, int(poll_ms)),
        config=cfg,
        bind_host=bind_host,
        require_auth=require_auth,
        access_token=access_token,
    )
    state.discovery_error = discovery_error
    handler = _make_handler(state)
    try:
        httpd = ThreadingHTTPServer((bind_host, listen), handler)
    except OSError as exc:
        raise UsageError(
            f"dashboard could not bind {bind_host}:{listen}: {exc}",
            hint="The detached dashboard uses one exact port and never silently moves.",
        ) from exc
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
        "bind": bind_host,
        "lan": bind_host == "0.0.0.0",
        "authenticated": require_auth,
        "discovery_error": discovery_error,
        "hint": (
            (
                "No device attached yet — the grid picks one up as soon as it boots. "
                if mode == "grid" and not serials
                else f"Grid of {len(serials)} device(s). Click a tile for detail. "
                if mode == "grid"
                else f"Watching {focus} via {primary_via}. "
            )
            + (
                "Background service is ready; stop it with `aua dashboard stop`."
                if not announce
                else "Leave this running; stop with Ctrl-C. Agent work is unaffected."
            )
        ),
        "ensures": {
            k: {kk: vv for kk, vv in v.items() if kk in ("via", "ok", "hint")}
            for k, v in ensures.items()
        },
    }

    def _serve() -> None:
        with httpd:
            httpd.serve_forever(poll_interval=0.5)

    browser_url = url + (f"?token={access_token}" if require_auth and access_token else "")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(browser_url)).start()

    if block:
        logger.info("dashboard on %s (mode=%s serials=%s)", url, mode, ",".join(serials))
        if announce:
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


def _service_main(argv: list[str] | None = None) -> int:
    """Private detached-process entrypoint used by :func:`start_service`."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--serve-service", action="store_true")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--bind", choices=("127.0.0.1", "0.0.0.0"), required=True)
    parser.add_argument("--poll-ms", type=int, default=500)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--serial")
    parser.add_argument("--config")
    parser.add_argument("--profile")
    parser.add_argument("--platform")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--grid", action="store_true")
    mode.add_argument("--detail", action="store_true")
    args = parser.parse_args(argv)
    if not args.serve_service:
        return 2

    state_path = Path(args.state_file).expanduser()
    try:
        launch_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.error("cannot read dashboard service state: %s", exc)
        return 2
    access_token = str(launch_state.get("access_token") or "")

    from .config import load_config

    overrides: dict[str, Any] = {"cache": {"dir": args.cache_dir}}
    if args.platform:
        overrides["device"] = {"platform": args.platform}
    cfg = load_config(
        explicit_path=args.config,
        profile=args.profile,
        cli_overrides=overrides,
    )

    # Raise into ``run``'s normal finally path so the listening socket closes immediately.
    def _terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    try:
        run(
            serial=args.serial,
            port=args.port,
            cache_dir=args.cache_dir,
            config=cfg,
            open_browser=False,
            poll_ms=args.poll_ms,
            block=True,
            grid=not args.detail,
            bind_host=args.bind,
            exact_port=True,
            require_auth=args.bind == "0.0.0.0",
            access_token=access_token or None,
            announce=False,
        )
    except AuaError as exc:
        logger.error("dashboard service failed: %s", exc)
        return 1
    finally:
        current = _read_service_state(args.cache_dir)
        if current.get("pid") == os.getpid():
            with contextlib.suppress(OSError):
                state_path.unlink()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through detached CLI integration
    raise SystemExit(_service_main())
