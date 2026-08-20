"""Headless HTTP mock / record / replay via mitmproxy (optional extra).

No GUI. Cassettes are YAML under ``memory.dir/cassettes/``. Live rules live in a JSON
sidecar the mitmproxy addon reloads. Device wiring (``http_proxy`` + ``adb reverse`` +
system CA install on emulators) is owned by the Engine; this module owns cassette/rule
logic, mitmdump process helpers, and the Android system-CA install script.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .errors import DeviceError, UsageError

logger = logging.getLogger(__name__)

# Avoid well-known service ports (8080 etc.). Ephemeral-ish high band.
_PORT_LO = 40_000
_PORT_HI = 60_000

# Host mitmproxy CA dir (created on first mitmdump run).
_MITM_CONFDIR = Path.home() / ".mitmproxy"
_MITM_CA_PEM = _MITM_CONFDIR / "mitmproxy-ca-cert.pem"
_MITM_CA_FULL = _MITM_CONFDIR / "mitmproxy-ca.pem"

ADDON_SCRIPT = '''\
"""aua mitmproxy addon.

Everything the agent can change is hot-reloaded from the rules file on each exchange, so
flipping record on or arming a rewrite never requires bouncing mitmdump — a restart would
leave the device pointed at a dead port for as long as it took to come back up.

Two ways to touch an exchange:
  stub    — answer from the rule, the request never reaches the server.
  rewrite — let the real request through, then patch the real response.
Everything unmatched is relayed untouched, which is the whole point: one request is under
test, the rest of the app has to keep working.
"""
from __future__ import annotations

import fnmatch
import json
import os
import time
from pathlib import Path

from mitmproxy import http

_RULES_PATH = Path(os.environ.get("AUA_MOCK_RULES", ""))
_RECORD_PATH = Path(os.environ.get("AUA_MOCK_RECORD", ""))
_ENV_MODE = os.environ.get("AUA_MOCK_MODE", "map")
# Always-on, append-only exchange log for `await net:` — deliberately separate from the
# cassette record: waiting on a backend response must not require `record` mode, and the
# cassette format must not change under replay. One JSONL line per completed response,
# headers and body omitted so a chat stream does not write megabytes per turn.
_FLOW_LOG_PATH = Path(os.environ.get("AUA_FLOW_LOG", ""))
# The bodies behind those lines, for `aua proxy flow <n>`: capped per entry and rotated as
# a whole, because this is a debugging buffer and must never be able to fill a disk.
_BODY_LOG_PATH = Path(os.environ.get("AUA_FLOW_BODIES", ""))

_MAX_BODY = 64 * 1024
_MAX_BODY_LOG = 16 * 1024 * 1024
# Allow-list rather than a binary block-list: anything not known to be text decodes into
# mojibake that is useless to read and expensive to carry. protobuf/gRPC in particular are
# very common on Android and would otherwise sail through a block-list.
_TEXT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-www-form-urlencoded",
    "application/graphql",
    "+json",
    "+xml",
)

# Rule id -> applications so far. Kept in the process rather than written back to the rules
# file: the file is re-read on every exchange and the CLI appends to it, so a counter
# living there would be lost the next time a rule was added.
_APPLIED = {}
_SEQ = [0]


def _doc():
    """The rules file, tolerating the legacy bare-list form."""
    empty = {"mode": _ENV_MODE, "capture_bodies": True, "rules": []}
    if not _RULES_PATH.is_file():
        return empty
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return empty
    if isinstance(data, list):
        return {"mode": _ENV_MODE, "capture_bodies": True, "rules": data}
    if not isinstance(data, dict):
        return empty
    return {
        "mode": data.get("mode") or _ENV_MODE,
        "capture_bodies": bool(data.get("capture_bodies", True)),
        "rules": data.get("rules") or data.get("entries") or [],
    }


def _match_path(pattern, url_path):
    """Exact, path-segment prefix, or glob — never a bare substring.

    Substring matching is how a rule for `/` silently stubs the entire internet, and how a
    rule meant for one endpoint swallows every path that happens to contain it.
    """
    if not pattern or pattern == "*":
        return True
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(url_path, pattern)
    if url_path == pattern:
        return True
    return url_path.startswith(pattern.rstrip("/") + "/")


def _matches(flow, rule):
    spec = rule.get("match") or rule.get("request") or {}
    method = str(spec.get("method") or "*").upper()
    if method != "*" and method != flow.request.method.upper():
        return False
    host = spec.get("host")
    if host and not fnmatch.fnmatch((flow.request.host or "").lower(), str(host).lower()):
        return False
    if not _match_path(spec.get("path"), flow.request.path.split("?", 1)[0]):
        return False
    query = spec.get("query")
    if query and str(query) not in flow.request.path:
        return False
    needle = spec.get("body")
    if needle:
        try:
            if str(needle) not in (flow.request.get_text(strict=False) or ""):
                return False
        except Exception:
            return False
    return True


def _pick(flow, action):
    """First matching rule of *action* whose --times budget is not yet spent."""
    for rule in _doc()["rules"]:
        if (rule.get("action") or "stub") != action:
            continue
        if not _matches(flow, rule):
            continue
        times = int(rule.get("times") or 0)
        if times > 0 and _APPLIED.get(str(rule.get("id") or ""), 0) >= times:
            continue
        return rule
    return None


def _consume(rule):
    key = str(rule.get("id") or "")
    _APPLIED[key] = _APPLIED.get(key, 0) + 1


def _path_parts(path):
    text = str(path)
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        text = text[1:]
    return [p for p in text.split(".") if p != ""]


def _descend(node, parts):
    for part in parts:
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            return None
    return node


def _assign(doc, parts, value):
    if not parts:
        return False
    node = _descend(doc, parts[:-1]) if len(parts) > 1 else doc
    leaf = parts[-1]
    if isinstance(node, list):
        try:
            node[int(leaf)] = value
        except (ValueError, IndexError):
            return False
        return True
    if isinstance(node, dict):
        node[leaf] = value
        return True
    return False


def _unset(doc, parts):
    if not parts:
        return False
    node = _descend(doc, parts[:-1]) if len(parts) > 1 else doc
    leaf = parts[-1]
    if isinstance(node, list):
        try:
            del node[int(leaf)]
        except (ValueError, IndexError):
            return False
        return True
    if isinstance(node, dict):
        return node.pop(leaf, _SENTINEL) is not _SENTINEL
    return False


_SENTINEL = object()


def _stub(flow, rule):
    resp = rule.get("response") or {}
    status = int(resp.get("status") or 200)
    body = resp.get("body")
    if body is None:
        body = ""
    if not isinstance(body, (str, bytes)):
        body = json.dumps(body)
    if isinstance(body, str):
        body = body.encode("utf-8")
    headers = resp.get("headers") or {"Content-Type": "application/json"}
    flow.response = http.Response.make(status, body, headers)


def _rewrite(flow, rule):
    """Patch the response the server actually sent."""
    spec = rule.get("rewrite") or {}
    resp = flow.response
    if resp is None:
        return
    status = spec.get("status")
    if status is not None:
        resp.status_code = int(status)
    for key, value in (spec.get("headers") or {}).items():
        resp.headers[str(key)] = str(value)
    body = spec.get("body")
    if body is not None:
        resp.text = body if isinstance(body, str) else json.dumps(body)

    sets = spec.get("set_json") or {}
    unsets = spec.get("delete_json") or []
    if sets or unsets:
        try:
            parsed = json.loads(resp.get_text(strict=False) or "")
        except Exception:
            parsed = None
        if parsed is not None:
            for path, value in sets.items():
                _assign(parsed, _path_parts(path), value)
            for path in unsets:
                _unset(parsed, _path_parts(path))
            resp.text = json.dumps(parsed)

    for pair in spec.get("replace") or []:
        try:
            old, new = pair[0], pair[1]
        except (IndexError, TypeError, KeyError):
            continue
        current = resp.get_text(strict=False)
        if current is not None:
            resp.text = current.replace(str(old), str(new))


def _append(path, payload):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\\n")
    except Exception:
        pass  # observability must never break the proxied request


def _snippet(message):
    if message is None:
        return None
    ctype = str(message.headers.get("content-type", "")).lower()
    body = message.raw_content or b""
    if ctype and not any(t in ctype for t in _TEXT_TYPES):
        return "<%s omitted, %d bytes>" % (ctype.split(";")[0].strip() or "binary", len(body))
    try:
        text = message.get_text(strict=False)
    except Exception:
        return None
    if text is None:
        return None
    return text[:_MAX_BODY]


class AuaMock:
    def request(self, flow: http.HTTPFlow) -> None:
        rule = _pick(flow, "stub")
        if rule is not None:
            _consume(rule)
            flow.metadata["aua_rule"] = rule.get("id")
            flow.metadata["aua_action"] = "stub"
            _stub(flow, rule)

    def response(self, flow: http.HTTPFlow) -> None:
        # mitmproxy calls this when the response is *complete*, so for a streamed chat
        # surface it fires at stream end — which is the moment `await net:` waits for.
        doc = _doc()
        if flow.metadata.get("aua_action") != "stub":
            rule = _pick(flow, "rewrite")
            if rule is not None:
                _consume(rule)
                flow.metadata["aua_rule"] = rule.get("id")
                flow.metadata["aua_action"] = "rewrite"
                try:
                    _rewrite(flow, rule)
                except Exception:
                    pass  # a bad rule must not take the app's network down with it

        _SEQ[0] += 1
        seq = _SEQ[0]
        status = flow.response.status_code if flow.response else 0
        summary = {
            "n": seq,
            "ts": time.time(),
            "method": flow.request.method.upper(),
            "path": flow.request.path.split("?", 1)[0],
            "host": flow.request.host,
            "status": status,
            "action": flow.metadata.get("aua_action"),
            # Which rule fired. The per-rule `times` budget is spent in this process and is
            # deliberately not written back to the rules file (the addon re-reads that file
            # on every exchange and the CLI appends to it, so a counter there would be lost
            # the next time a rule was added). Logging it here is what lets `mock list` tell
            # an armed rule from one that has already been used up.
            "rule": flow.metadata.get("aua_rule"),
        }
        if str(_FLOW_LOG_PATH):
            _append(_FLOW_LOG_PATH, summary)

        if doc["capture_bodies"] and str(_BODY_LOG_PATH):
            try:
                if _BODY_LOG_PATH.is_file() and _BODY_LOG_PATH.stat().st_size > _MAX_BODY_LOG:
                    _BODY_LOG_PATH.unlink()
            except Exception:
                pass
            _append(
                _BODY_LOG_PATH,
                dict(
                    summary,
                    query=flow.request.path.split("?", 1)[1] if "?" in flow.request.path else "",
                    request_headers=dict(flow.request.headers),
                    request_body=_snippet(flow.request),
                    response_headers=dict(flow.response.headers) if flow.response else {},
                    response_body=_snippet(flow.response),
                ),
            )

        # Cassette capture is orthogonal to rules now: recording a flow you are also
        # rewriting is a normal thing to want, and the old early-return made it impossible.
        if doc["mode"] == "record" and str(_RECORD_PATH):
            _append(
                _RECORD_PATH,
                {
                    "request": {
                        "method": flow.request.method.upper(),
                        "path": flow.request.path.split("?", 1)[0],
                        "host": flow.request.host,
                    },
                    "response": {
                        "status": status,
                        "body": (_snippet(flow.response) or "")[:200_000],
                    },
                },
            )


addons = [AuaMock()]
'''


def cassette_dir(memory_dir: str | Path) -> Path:
    return Path(memory_dir).expanduser() / "cassettes"


def rules_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "mock_rules.json"


def record_path(cache_dir: str | Path) -> Path:
    """In-progress cassette capture — append-only.

    Was a JSON array the addon re-read and rewrote on every single response, which is
    O(n²) writes and turns a chat surface into a stall.
    """
    return Path(cache_dir).expanduser() / "mock_record.jsonl"


def load_record(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Entries captured so far, skipping any half-written trailing line."""
    return _read_jsonl(record_path(cache_dir))


def reset_record(cache_dir: str | Path) -> None:
    """Start a clean JSONL capture file for a new ``mock record start``.

    Must not write anything but an empty file: the addon opens this path in ``"a"`` mode and
    appends ``json.dumps(entry) + "\\n"`` per completed flow (see ``AuaMock.response()``). A
    non-empty, non-newline-terminated seed (the old code wrote the literal text ``"[]"``) glues
    onto the very first appended line with no separator, corrupting the file from that point on
    for any reader that expects either a single JSON document or clean JSONL.
    """
    path = record_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def record_window_path(cache_dir: str | Path) -> Path:
    """Where the *scope* of the current recording is remembered across the two CLI calls.

    ``mock record start`` and ``mock record stop`` are separate process invocations, so the
    only way ``stop`` can tell a stale log line (a previous, unrelated run) from evidence about
    *this* recording is to have ``start`` write down where "this recording" begins: a wall-clock
    timestamp (to filter ``flow_log.jsonl``, which already carries real timestamps) and a byte
    offset into ``mitmdump.log`` (which has none — see ``diagnose_empty_recording``).
    """
    return Path(cache_dir).expanduser() / "mock_record_window.json"


def save_record_window(cache_dir: str | Path, *, since_ts: float, log_offset: int) -> None:
    path = record_window_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"since_ts": float(since_ts), "log_offset": int(log_offset)}),
        encoding="utf-8",
    )


def load_record_window(cache_dir: str | Path) -> dict[str, Any] | None:
    path = record_window_path(cache_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "since_ts": float(data.get("since_ts") or 0.0),
        "log_offset": int(data.get("log_offset") or 0),
    }


def clear_record_window(cache_dir: str | Path) -> None:
    with contextlib.suppress(OSError):
        record_window_path(cache_dir).unlink()


def flow_bodies_path(cache_dir: str | Path) -> Path:
    """Full request/response detail for ``aua proxy flow <n>`` (bounded, rotated)."""
    return Path(cache_dir).expanduser() / "flow_bodies.jsonl"


def read_flow_bodies(cache_dir: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(flow_bodies_path(cache_dir))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # the writer is mid-append
                if isinstance(entry, dict):
                    out.append(entry)
    except OSError:
        return []
    return out


def flow_log_path(cache_dir: str | Path) -> Path:
    """Append-only JSONL of completed HTTP exchanges, for ``await net:``.

    Separate from :func:`record_path` on purpose. That one is the cassette record: it is
    written only in ``record`` mode, keeps whole bodies, and re-reads + rewrites the entire
    file on every response — fine for capturing a fixture, useless to poll while a chat
    surface streams. This one is always on, one small line per exchange, and append-only so
    a reader can tail it without racing the writer.
    """
    return Path(cache_dir).expanduser() / "flow_log.jsonl"


def read_flows_since(cache_dir: str | Path, since_ts: float) -> list[dict[str, Any]]:
    """Completed exchanges logged after *since_ts* (epoch seconds).

    Tolerates partial trailing lines: the proxy appends while we read.
    """
    path = flow_log_path(cache_dir)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue  # a half-written final line
                if isinstance(entry, dict) and float(entry.get("ts") or 0) > since_ts:
                    out.append(entry)
    except OSError:
        return []
    return out


def flow_matches(entry: dict[str, Any], spec: str) -> bool:
    """Does *entry* satisfy ``[METHOD ]PATH[=STATUS]``?

    ``POST /v1/chat=200`` · ``/v1/chat`` · ``GET /feed``. Path match is a substring so a
    caller does not have to reproduce a full templated route.
    """
    text = spec.strip()
    status: int | None = None
    if "=" in text:
        text, _, raw = text.rpartition("=")
        with contextlib.suppress(ValueError):
            status = int(raw.strip())
    text = text.strip()
    method: str | None = None
    parts = text.split(None, 1)
    if len(parts) == 2 and parts[0].isalpha() and parts[0].upper() in {
        "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
    }:
        method, text = parts[0].upper(), parts[1].strip()
    if method and str(entry.get("method", "")).upper() != method:
        return False
    if status is not None and int(entry.get("status") or 0) != status:
        return False
    return text in str(entry.get("path", ""))


# --------------------------------------------------------------------------- ownership
#
# The device's `http_proxy` is a *persistent* global setting pointing at a *non-persistent*
# host process, reached through an `adb reverse` tunnel that dies with the adb transport.
# When either half goes away the device is left proxied to a dead loopback port: every app
# reports "Offline" and NetworkMonitor flags the network unvalidated (the Wi-Fi "!"). So a
# device-global setting needs device-global bookkeeping — who owns the proxy, on which port,
# under which boot — recorded where *any* process can find it.


def proxy_state_dir() -> Path:
    """Serial-keyed proxy ownership records, deliberately NOT under ``AUA_CACHE__DIR``.

    Parallel agents are told to keep separate caches so their mock rules cannot leak into
    one another, which also means the port agent A wrote into its own cache is invisible to
    agent B — and B is the one that inherits the emulator. Mirrors the emulator's
    port-reservation directory: process-wide facts live at a fixed path.
    """
    d = Path.home() / ".cache/android-ui-analyser/proxy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(serial: str) -> Path:
    return proxy_state_dir() / f"{str(serial).replace(':', '_')}.json"


def write_state(serial: str, state: dict[str, Any]) -> None:
    from .atomic import atomic_write_text

    path = state_path(serial)
    payload = {"serial": serial, "written": time.time(), **state}
    # Per-writer scratch name: a shared `.tmp` is what crashed a live agent run when two
    # processes published the same key (see atomic.py).
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def read_state(serial: str) -> dict[str, Any] | None:
    path = state_path(serial)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def clear_state(serial: str) -> None:
    with contextlib.suppress(OSError):
        state_path(serial).unlink()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, just not ours to signal
    except OSError:
        return False
    return True


def port_listening(port: int, *, timeout: float = 0.25) -> bool:
    """Is something accepting connections on ``127.0.0.1:port`` right now?"""
    if port <= 0 or port > 65535:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except (OSError, OverflowError):
        return False
    finally:
        with contextlib.suppress(OSError):
            sock.close()


def orphan_reason(state: dict[str, Any] | None, *, boot_id: str | None) -> str | None:
    """Why the proxy recorded for a device is dead, or ``None`` when it is still healthy.

    Deliberately conservative: a proxy another agent is actively using must survive this
    check, so every reason here is positive evidence of death rather than of foreignness.
    """
    if state is None:
        return "no aua owns it (no ownership record for this device)"
    recorded_boot = state.get("boot_id")
    if boot_id and recorded_boot and str(recorded_boot) != str(boot_id):
        return "the device rebooted since the proxy was set"
    pid = int(state.get("pid") or 0)
    if pid and not pid_alive(pid):
        return f"its mitmdump (pid {pid}) is gone"
    port = int(state.get("port") or 0)
    if port and not port_listening(port):
        return f"nothing is listening on host port {port}"
    if not pid and not port:
        return "its ownership record names neither a process nor a port"
    return None


def reverse_tunnel_active(serial: str, port: int, *, local_port: int | None = None) -> bool:
    """Whether ``adb reverse`` still forwards the device's ``tcp:<port>`` to this host port.

    A dropped tunnel is invisible to both of the checks ``orphan_reason`` already made: the
    mitmdump process keeps running and listening on the host side regardless, and the device's
    ``http_proxy`` setting is a value Android stores independently of whether anything can
    still reach it. ``adb reverse --list`` is the only source of truth for the tunnel itself.
    """
    if port <= 0:
        return False
    try:
        proc = _adb(serial, "reverse", "--list", check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    remote = f"tcp:{int(port)}"
    local = f"tcp:{int(local_port)}" if local_port is not None else None
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        # An unowned device endpoint is reachable through any local mapping. Owned AUA state
        # passes ``local_port`` and requires its exact symmetric mapping.
        if len(parts) >= 3 and parts[-2] == remote and (local is None or parts[-1] == local):
            return True
    return False


def read_device_http_proxy(serial: str) -> str | None:
    """The device's current ``global http_proxy`` setting, or ``None`` when unset/unreadable.

    ``settings get`` prints the literal string ``"null"`` for an unset key, and a proxy that
    was never armed reads back as ``":0"`` on some Android versions — both mean "no proxy",
    never a value to compare a port against.
    """
    try:
        proc = _adb(
            serial, "shell", "settings", "get", "global", "http_proxy", check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    if not value or value.lower() == "null" or value == ":0":
        return None
    return value


def ensure_reverse_tunnel(serial: str, port: int) -> bool:
    """(Re)establish ``adb reverse tcp:<port> tcp:<port>``; return whether it is active after.

    Idempotent by construction — ``adb reverse`` replaces a stale mapping for the same port
    rather than erroring on one that already exists, so this is safe to call whether or not the
    tunnel was already there. Callers decide whether calling it is *safe* (the process behind
    the port must actually be alive); this function only does the one thing it is asked.
    """
    if port <= 0:
        return False
    with contextlib.suppress(OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        _adb(serial, "reverse", f"tcp:{int(port)}", f"tcp:{int(port)}", check=False, timeout=10)
    return reverse_tunnel_active(serial, port)


# Hosts that mean "this machine" from the device's point of view, and how the device gets
# there. The distinction decides which checks are even *askable*, which is why it is a
# classification rather than a boolean: asserting a missing `adb reverse` tunnel for a
# `10.0.2.2` target would be a fabricated failure (that route needs no tunnel at all), and
# asserting anything about an off-host address from here would be a guess.
_LOOPBACK_HOSTS = frozenset({"localhost", "::1", "0.0.0.0"})  # noqa: S104 - matched, not bound
# The emulator's stable alias for the host loopback, reachable without any `adb reverse`.
_EMULATOR_HOST_ALIASES = frozenset({"10.0.2.2"})

TARGET_LOOPBACK = "loopback"
TARGET_EMULATOR_HOST = "emulator_host"
TARGET_EXTERNAL = "external"
TARGET_INVALID = "invalid"


def classify_proxy_host(host: str) -> str:
    """Which of the ownership-free reachability checks apply to *host*.

    ``loopback`` — the device can only reach it through an ``adb reverse`` tunnel, so both the
    tunnel and the host listener are checkable. ``emulator_host`` — the emulator routes
    ``10.0.2.2`` to the host itself, so the listener is checkable but a tunnel is not required
    and its absence is not a fault. ``external`` — an address off this machine: nothing here
    can say whether it is reachable, and saying so is better than guessing.
    """
    h = (host or "").strip().strip(".").lower()
    if not h:
        return TARGET_EXTERNAL
    if h in _LOOPBACK_HOSTS or h.startswith("127."):
        return TARGET_LOOPBACK
    if h in _EMULATOR_HOST_ALIASES:
        return TARGET_EMULATOR_HOST
    return TARGET_EXTERNAL


def parse_proxy_target(raw: str | None) -> dict[str, Any] | None:
    """``settings get global http_proxy``'s value as ``{raw, host, port, kind}``, or ``None``.

    The device's setting is the only ownership-free statement of *what this device is pointed
    at*, so everything a black-hole diagnosis can know starts here. Android stores it as
    ``host:port``, but not only that: it prints the literal ``"null"`` when unset, ``":0"`` on
    some versions when it was never armed, and a value written by hand can carry a scheme.
    ``None`` means genuinely unset. A non-empty malformed setting is returned as ``invalid``;
    treating it as unset falsely reports a clean device while Android is configured with a
    value callers cannot route through.
    """
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.lower() == "null" or value == ":0":
        return None
    def invalid(reason: str) -> dict[str, Any]:
        return {"raw": value, "host": None, "port": None, "kind": TARGET_INVALID, "error": reason}

    body = value.split("://", 1)[1] if "://" in value else value
    body = body.split("/", 1)[0]
    if not body:
        return invalid("missing proxy endpoint")
    host = ""
    port_text = ""
    if body.startswith("["):  # bracketed IPv6, e.g. "[::1]:3128"
        close = body.find("]")
        if close == -1:
            return invalid("unclosed IPv6 address")
        host = body[1:close]
        rest = body[close + 1 :]
        port_text = rest[1:] if rest.startswith(":") else ""
    elif body.count(":") == 1:
        host, _, port_text = body.partition(":")
    else:
        # No port at all, or a bare (unbracketed) IPv6 literal — neither names a reachable
        # proxy endpoint, and inventing a port for it would invent the diagnosis too.
        return invalid("proxy endpoint must include one explicit port")
    try:
        port = int(port_text)
    except ValueError:
        return invalid("proxy port is not an integer")
    if not host or port <= 0 or port > 65535:
        return invalid("proxy host is empty or port is outside 1..65535")
    return {"raw": value, "host": host, "port": port, "kind": classify_proxy_host(host)}


def connect_failures_in_logcat(
    serial: str, host: str, port: int, *, lines: int = 400
) -> int:
    """How many recent app requests failed to reach ``host:port``, from logcat.

    Passive corroboration for a black-hole diagnosis, and deliberately the *only* device-side
    reachability evidence collected: sending an HTTP request through an unowned proxy to
    identify it would write that request into another agent's flow log and cassette (the mitm
    addon's ``response`` hook appends every completed flow), which is exactly the artifact
    corruption ``_claim_or_reap_proxy`` exists to prevent. Reading a log corrupts nothing.

    Absence of hits is never evidence of health — an app that has not tried yet produces none.
    """
    if port <= 0 or not host:
        return 0
    try:
        proc = _adb(
            serial, "logcat", "-d", "-t", str(int(lines)), check=False, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if proc.returncode != 0:
        return 0
    needle = f"Failed to connect to /{host}:{int(port)}"
    return sum(1 for line in (proc.stdout or "").splitlines() if needle in line)


def _self_proof(cache_dir: str | Path, port: int) -> int | None:
    """The pid of *this session's own* mitm when it is the thing on ``port``, else ``None``.

    Two independent sidecars, both written by ``start_mitm``: ``mitmproxy.port`` says which
    port this cache dir's mitm owns, ``mitmproxy.pid`` names the process. When the port matches
    the one the device is pointed at *and* that pid is still alive, the proxy is ours and a
    missing ownership record is the bug — ``proxy_start`` wraps its own ``write_state`` in
    ``contextlib.suppress(Exception)``, so a perfectly healthy aua proxy can exist with no
    record at all.

    ``pid_alive`` is not optional here. This host carries ``mitmproxy.pid`` files naming
    long-dead processes; the file merely existing proves nothing.
    """
    if port <= 0:
        return None
    if load_listen_port(cache_dir) != int(port):
        return None
    try:
        pid = int(pid_path(cache_dir).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 and pid_alive(pid) else None


# The closed vocabulary of `proxy_health`'s `state`. This is a contract, not a free-text
# field: callers branch on it, so a new value is a breaking change.
#
#   unproxied   http_proxy unset and no aua owns a proxy here — a clean device
#   healthy     an aua owns it and every check it can make is green
#   degraded    an aua owns it but at least one check is red — that owner can fix it
#   foreign     the device is proxied, no aua owns it, and it is reachable — traffic flows,
#               but this session's mock rules are inert against it
#   blackholed  the device is proxied, no aua owns it, and nothing can reach it — every app
#               request fails, and no `aua teardown` will ever clean it up
PROXY_STATES = ("unproxied", "healthy", "degraded", "foreign", "blackholed", "unknown")


def _owner_remedy(serial_hint: str = "<serial>") -> str:
    return (
        f" Fix: `aua --serial {serial_hint} proxy status --heal`, or "
        f"`aua --serial {serial_hint} proxy stop` then "
        f"`aua --serial {serial_hint} proxy start` to rebuild it."
    )


def _unowned_health(
    serial: str,
    cache_dir: str | Path,
    target: dict[str, Any],
    *,
    stale_record_port: int | None = None,
) -> dict[str, Any]:
    """Diagnose a device pointed at a proxy no ownership record claims.

    Everything here is ownership-free by construction — the device's own setting, ``adb reverse
    --list``, a TCP connect to the host port, and a passive logcat read. That is the whole
    point: the state this function exists for is precisely the one where there is no record to
    read, and answering it with "nothing is armed" (which is what happened) is the exact
    opposite of the truth and indistinguishable from a genuinely clean device.

    Where it stops, deliberately: it cannot say *who* owns the proxy (no record, no answer),
    nor whether the listener is a proxy at all (``port_listening`` is a ``connect_ex``; a dev
    server passes it). So the verdict word for a reachable one is ``foreign``, never *healthy*.
    """
    kind = str(target["kind"])
    if kind == TARGET_INVALID:
        return {
            "ok": False,
            "state": "unknown",
            "owned": False,
            "intercepting": False,
            "target": dict(target),
            "port": None,
            "pid": None,
            "checks": {
                "device_setting": {
                    "ok": False,
                    "detail": str(target.get("error") or "malformed http_proxy setting"),
                }
            },
            "detail": f"device http_proxy is malformed: {target['raw']}",
            "hint": "Clear the malformed setting with `aua --serial <serial> proxy stop`.",
        }
    host = str(target["host"])
    port = int(target["port"])
    where = f"{host}:{port}"
    checks: dict[str, Any] = {}

    if kind in (TARGET_LOOPBACK, TARGET_EMULATOR_HOST):
        listening = port_listening(port)
        checks["listener"] = {
            "ok": listening,
            "detail": (
                f"127.0.0.1:{port} accepts connections"
                if listening
                else f"nothing is listening on host port {port}"
            ),
        }
    if kind == TARGET_LOOPBACK:
        # Only a loopback target needs the tunnel; `10.0.2.2` reaches the host without one, and
        # a red tunnel check there would be an invented failure.
        tunnel = reverse_tunnel_active(serial, port)
        checks["tunnel"] = {
            "ok": tunnel,
            "detail": (
                f"adb reverse tcp:{port} tcp:{port} is active"
                if tunnel
                else f"no `adb reverse` tunnel forwards tcp:{port} — the device cannot reach "
                "this host port at all"
            ),
        }

    adoptable_pid = _self_proof(cache_dir, port) if kind == TARGET_LOOPBACK else None
    know_reachability = bool(checks)
    reachable = know_reachability and all(bool(c["ok"]) for c in checks.values())
    state = "foreign" if reachable else ("blackholed" if know_reachability else "unknown")
    out: dict[str, Any] = {
        "ok": reachable,
        "state": state,
        "owned": False,
        "intercepting": False,
        "target": dict(target),
        "port": port,
        "pid": None,
        "checks": checks,
    }

    stale_note = ""
    if stale_record_port:
        stale_note = (
            f" An ownership record for this device names port {stale_record_port}, but the "
            f"device points at {port} — the record is stale; `aua proxy stop` clears both."
        )

    if adoptable_pid is not None:
        # Not a stranger's proxy at all: the sidecars this cache dir wrote say it is ours.
        out["adoptable"] = True
        out["adoptable_pid"] = adoptable_pid
        out["detail"] = (
            f"this device is proxied through {where}, which is this session's own mitmdump "
            f"(pid {adoptable_pid}, per mitmproxy.port/mitmproxy.pid) — but no ownership "
            "record names it, so no other process can see who owns it"
        )
        out["hint" if state == "blackholed" else "warning"] = (
            "run `aua proxy status` (healing enabled, the default) to rebuild the missing "
            "ownership record from this session's own mitm sidecars." + stale_note
        )
        return out

    if kind == TARGET_EXTERNAL:
        out["detail"] = (
            f"this device is proxied through {target['raw']} — an address off this host, so "
            "aua cannot tell from here whether it is reachable"
        )
        out["warning"] = (
            f"this device is proxied through {target['raw']}, which no aua owns and which is "
            "not on this machine. Your `aua mock` rules are NOT applied to it and `aua mock "
            "record` will capture nothing." + stale_note
        )
        return out

    if state == "foreign":
        out["detail"] = (
            f"this device is proxied through {where} and that proxy is reachable, but no aua "
            "owns it"
        )
        out["warning"] = (
            f"this device is proxied through {where}, which no aua owns. Traffic flows, so the "
            "app works — but it is intercepted by a process this session did not start: your "
            "`aua mock` rules are NOT applied to it and `aua mock record` will capture "
            "nothing. Another agent may be using it — do NOT take it over. If you know that "
            "holder is dead, `aua --serial <serial> proxy stop` un-points the device."
            + stale_note
        )
        return out

    broken = "; ".join(c["detail"] for c in checks.values() if not c["ok"])
    hint = (
        f"BLACK HOLE: this device's http_proxy points at {where} and nothing can reach it — "
        f"{broken}. EVERY app network request on this device fails with "
        f"`java.net.ConnectException: Failed to connect to /{where}`, which appears only in "
        "logcat: buttons do nothing, screens stay empty, logins silently never complete. No "
        "aua owns this proxy, so nothing in `aua teardown` will ever clean it up. Fix: "
        "`aua --serial <serial> proxy stop` un-points the device (safe — no aua owns this "
        "proxy), or `aua --serial <serial> proxy start` to put a working proxy there."
        + stale_note
    )
    failures = connect_failures_in_logcat(serial, host, port)
    if failures:
        hint += (
            f" Confirmed in logcat: {failures} recent app requests failed with "
            f"ConnectException to {where}."
        )
    out["detail"] = f"this device is proxied through {where} and nothing can reach it"
    out["hint"] = hint
    return out


def proxy_health(
    serial: str, cache_dir: str | Path, *, self_heal: bool = True
) -> dict[str, Any]:
    """Whether this device's traffic is reaching a proxy, and whether that proxy is ours.

    Three independent pieces of state all have to hold together for the device's traffic to
    reach mitmdump: the process has to be alive and listening, the device's ``http_proxy``
    setting has to point at that port, and the ``adb reverse`` tunnel that lets a loopback-only
    device reach a host port at all has to still exist. Checking any one or two of these and
    calling it "healthy" is exactly how a device was found pointed at a dead tunnel while
    ``settings get global http_proxy`` and the mitmdump process each individually looked fine —
    every app's network calls failed with ``ConnectException``, visible only in logcat, with
    nothing in any `aua` surface saying so (measured 2026-08-19).

    The harder half, measured 2026-08-20: **none of that requires an ownership record.** This
    used to return ``{"ok": false, "armed": false, "checks": {}}`` the instant ``read_state``
    came back ``None``, which reads as *nothing is configured* — the exact opposite of a device
    whose every request is failing, and indistinguishable from a genuinely clean one. So the
    device's own setting is now read first and always, and the tunnel and host-listener checks
    run against whatever port it names, owned or not.

    ``ok`` and ``intercepting`` are separate answers to separate questions, because a single
    boolean cannot serve both: ``ok`` is *this device's network path is sane* (true for a
    stranger's working proxy), ``intercepting`` is *traffic is reaching a proxy this aua owns*
    (false for it). ``state`` is one of ``PROXY_STATES`` and is the field to branch on.

    A check is present in ``checks`` **iff it was actually performed**: ``process`` needs a pid
    we can vouch for, so it appears only when owned; ``listener`` and ``tunnel`` appear only
    when the target host makes them askable (see ``classify_proxy_host``). Never a fabricated
    red for something unknowable.

    ``self_heal`` is retained for capability compatibility but this service is diagnostic.
    Device mutation belongs to ``Engine.proxy_status``, which records the undo before asking
    the adapter capability to restore a tunnel. Keeping the decision there also means CLI and
    MCP use the same ledger-aware path.
    """
    state_rec = read_state(serial)
    target = parse_proxy_target(read_device_http_proxy(serial))

    if target is not None and target.get("kind") == TARGET_INVALID:
        stale = None
        if isinstance(state_rec, dict):
            with contextlib.suppress(TypeError, ValueError):
                stale = int(state_rec.get("port") or 0) or None
        return _unowned_health(serial, cache_dir, target, stale_record_port=stale)

    if not isinstance(state_rec, dict):
        if target is None:
            return {
                "ok": True,
                "state": "unproxied",
                "owned": False,
                "intercepting": False,
                "target": None,
                "port": None,
                "pid": None,
                "checks": {},
                "detail": (
                    "no proxy on this device — http_proxy is unset and no aua owns a proxy "
                    "for it"
                ),
            }
        return _unowned_health(serial, cache_dir, target)

    port = int(state_rec.get("port") or 0) or (load_listen_port(cache_dir) or 0)
    pid = int(state_rec.get("pid") or 0)

    if target is not None and port and target["port"] != port:
        # The device is ground truth for what it is pointed at. Our record names a different
        # port, so whatever the device actually uses is not ours — diagnose *that*, and say the
        # record is stale rather than reporting `owned: true` about a port nobody uses.
        return _unowned_health(serial, cache_dir, target, stale_record_port=port)

    pid_ok = bool(pid) and pid_alive(pid)
    listening_ok = bool(port) and port_listening(port)
    if pid_ok and listening_ok:
        process_detail = f"mitmdump pid {pid} is alive and listening on 127.0.0.1:{port}"
    elif not pid:
        process_detail = "no owning process recorded for this proxy"
    elif pid_ok:
        process_detail = f"mitmdump pid {pid} is alive"
    else:
        process_detail = f"its mitmdump (pid {pid}) is gone"
    checks: dict[str, Any] = {"process": {"ok": pid_ok, "detail": process_detail}}
    checks["listener"] = {
        "ok": listening_ok,
        "detail": (
            f"127.0.0.1:{port} accepts connections"
            if listening_ok
            else f"nothing is listening on host port {port or '(none recorded)'}"
        ),
    }

    tunnel_ok = bool(port) and reverse_tunnel_active(serial, port, local_port=port)
    tunnel_check: dict[str, Any] = {
        "ok": tunnel_ok,
        "detail": (
            f"adb reverse tcp:{port} tcp:{port} is active"
            if tunnel_ok
            else f"no `adb reverse` tunnel forwards tcp:{port} — the device cannot reach the "
            "host proxy even though the setting points at it"
        ),
    }
    checks["tunnel"] = tunnel_check

    expected = f"127.0.0.1:{port}" if port else None
    device_proxy = target["raw"] if target else None
    setting_ok = bool(
        port
        and target is not None
        and target.get("kind") == TARGET_LOOPBACK
        and target["port"] == port
    )
    setting_check: dict[str, Any] = {
        "ok": setting_ok,
        "detail": (
            f"device http_proxy is {device_proxy}"
            if device_proxy
            else "device http_proxy is unset"
        ),
    }
    if port and not setting_ok:
        setting_check["expected"] = expected
    checks["device_setting"] = setting_check

    ok = all(c["ok"] for c in checks.values())
    out: dict[str, Any] = {
        "ok": ok,
        "state": "healthy" if ok else "degraded",
        "owned": True,
        "intercepting": ok,
        "target": dict(target) if target else None,
        "port": port or None,
        "pid": pid or None,
        "checks": checks,
    }
    if ok:
        out["detail"] = f"interception is working end to end on 127.0.0.1:{port}"
    else:
        broken = [name for name, c in checks.items() if not c["ok"]]
        out["hint"] = (
            "interception is NOT actually working end to end — broken: "
            + ", ".join(broken)
            + " — "
            + "; ".join(c["detail"] for name, c in checks.items() if not c["ok"])
            + "."
            + _owner_remedy(serial)
        )
    return out


def pid_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "mitmproxy.pid"


def port_path(cache_dir: str | Path) -> Path:
    """Sidecar that remembers which listen port the running mitm owns."""
    return Path(cache_dir).expanduser() / "mitmproxy.port"


def addon_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "aua_mitm_addon.py"


def _port_free(port: int) -> bool:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        with contextlib.suppress(OSError):
            srv.close()


def pick_listen_port(*, preferred: int | None = None) -> int:
    """Return a free localhost TCP port — never the well-known 8080 default.

    ``preferred`` (when >0) is tried first; otherwise a random high port in
    ``[_PORT_LO, _PORT_HI]``. Falls back to OS-assigned ``bind(..., 0)`` if the
    random band is exhausted.
    """
    candidates: list[int] = []
    if preferred and preferred > 0:
        candidates.append(int(preferred))
    # A few random probes beat walking the whole range.
    candidates.extend(random.sample(range(_PORT_LO, _PORT_HI + 1), k=32))
    for port in candidates:
        if _port_free(port):
            return port
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.bind(("127.0.0.1", 0))
        return int(srv.getsockname()[1])
    finally:
        with contextlib.suppress(OSError):
            srv.close()


def save_listen_port(cache_dir: str | Path, port: int) -> None:
    path = port_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(port)), encoding="utf-8")


def load_listen_port(cache_dir: str | Path) -> int | None:
    path = port_path(cache_dir)
    if not path.is_file():
        return None
    try:
        port = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return port if port > 0 else None


def clear_listen_port(cache_dir: str | Path) -> None:
    with contextlib.suppress(OSError):
        port_path(cache_dir).unlink()


# Android validates a network by fetching these; if that fetch goes through a proxy the
# app under test is also breaking, the whole network is marked unvalidated and every app
# shows an offline state for a reason that has nothing to do with the app. Sending the
# probes direct keeps the "!" off the Wi-Fi icon while the proxy is live.
CONNECTIVITY_CHECK_HOSTS = (
    "connectivitycheck.gstatic.com",
    "connectivitycheck.android.com",
    "clients3.google.com",
    "clients1.google.com",
    "www.google.com",
    "play.googleapis.com",
    "android.clients.google.com",
)


def load_doc(path: Path) -> dict[str, Any]:
    """The rules file as ``{mode, capture_bodies, rules, owner}``, whatever shape it is on disk.

    ``mode`` lives in the file rather than in mitmdump's environment so record can be
    flipped without restarting the proxy — a restart drops the device's only route to the
    network for as long as it takes to rebind.

    ``owner`` is a best-effort attribution tag (see ``leases.resolve_owner``): whichever
    session's ``mock map``/``mock replay`` last (re)armed this file from empty. It lets a later
    ``mock map`` in a *different* session recognise it is appending onto rules it did not
    create — measured: 14 stale, untagged rules from unrelated earlier sessions accumulated
    silently in one shared cache dir. Unknown/missing on legacy files, which is itself the
    signal a caller needs (see ``Engine.mock_map``).
    """
    doc: dict[str, Any] = {"mode": "map", "capture_bodies": True, "rules": [], "owner": None}
    if not path.is_file():
        return doc
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return doc
    if isinstance(data, list):
        doc["rules"] = data
        return doc
    if isinstance(data, dict):
        doc["mode"] = str(data.get("mode") or "map")
        doc["capture_bodies"] = bool(data.get("capture_bodies", True))
        doc["rules"] = list(data.get("rules") or data.get("entries") or [])
        owner = data.get("owner")
        doc["owner"] = str(owner) if isinstance(owner, str) and owner.strip() else None
    return doc


def save_doc(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # the addon re-reads this file constantly; never show it partial


def write_rules(path: Path, entries: list[dict[str, Any]], *, owner: str | None = None) -> None:
    """Replace the rule list, preserving mode/capture settings already in the file.

    ``owner`` (when given) stamps who now owns this rule set — used by a full replace
    (``mock replay``) where it is correct to reattribute the file to whoever just armed it.
    Omitted, the previous owner tag (if any) is left untouched.
    """
    doc = load_doc(path)
    doc["rules"] = entries
    if owner is not None:
        doc["owner"] = owner
    save_doc(path, doc)


def clear_rules(cache_dir: str | Path) -> int:
    """Disarm mock mode and drop every rule and owner tag; return how many rules were removed.

    Also the undo for the ``mock_rules`` ledger mutation (see ``device_ledger``): arming a rule
    or record mode is host-side state that outlives the command, and this is the one call that
    fully resets it — used by both ``aua mock clear`` and a stranger's reaper.
    """
    path = rules_path(cache_dir)
    removed = len(load_rules(path))
    save_doc(path, {"mode": "map", "capture_bodies": True, "rules": []})
    return removed


def backfill_rule_ids(rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Assign a stable id to any rule missing one; return the (possibly new) list and whether
    anything changed, so a caller only has to persist when it did.

    Several real on-disk rules were found with ``id: null`` — written before ids were always
    assigned, or by a hand-edited cassette — which made them impossible to target with
    ``mock rm``. Backfilling here, rather than in ``load_doc``, keeps the common read path free
    of a surprise write: only the CLI-facing callers that actually need a stable id
    (``mock list``, ``mock rm``) opt into persisting one.
    """
    changed = False
    stamp = int(time.time() * 1000) % 10_000_000
    seq = 0
    out: list[dict[str, Any]] = []
    for rule in rules:
        if not rule.get("id"):
            seq += 1
            rule = dict(rule)
            rule["id"] = f"r{stamp}-{seq:03d}"
            changed = True
        out.append(rule)
    return out, changed


def set_mode(path: Path, mode: str) -> None:
    doc = load_doc(path)
    doc["mode"] = mode
    save_doc(path, doc)


def rewrite_rule(
    *,
    host: str | None = None,
    method: str = "*",
    path: str = "*",
    query: str | None = None,
    request_body: str | None = None,
    status: int | None = None,
    headers: dict[str, str] | None = None,
    body: Any = None,
    set_json: dict[str, Any] | None = None,
    delete_json: list[str] | None = None,
    replace: list[tuple[str, str]] | None = None,
    times: int = 0,
) -> dict[str, Any]:
    """A rule that lets the request reach the server, then patches the real response."""
    match: dict[str, Any] = {"method": method.upper(), "path": path}
    if host:
        match["host"] = host
    if query:
        match["query"] = query
    if request_body:
        match["body"] = request_body
    spec: dict[str, Any] = {}
    if status is not None:
        spec["status"] = int(status)
    if headers:
        spec["headers"] = headers
    if body is not None:
        spec["body"] = body
    if set_json:
        spec["set_json"] = set_json
    if delete_json:
        spec["delete_json"] = delete_json
    if replace:
        spec["replace"] = [list(pair) for pair in replace]
    if not spec:
        raise UsageError(
            "a rewrite rule must change something",
            hint="Pass at least one of --status / --set / --delete / --replace / "
            "--header / --body.",
        )
    return {
        "id": f"r{int(time.time() * 1000) % 10_000_000}-{random.randrange(1000):03d}",
        "action": "rewrite",
        "match": match,
        "rewrite": spec,
        "times": int(times),
    }


def guard_rule_scope(rule: dict[str, Any]) -> None:
    """Refuse a rule broad enough to take the whole device offline.

    An unhosted `/` (or `*`) matches every request on every host, including Android's own
    connectivity probes, so arming one looks identical to the device losing its network.
    """
    match = rule.get("match") or rule.get("request") or {}
    path = str(match.get("path") or "*").strip()
    if match.get("host"):
        return
    if path in ("", "*", "/", "/*", "**"):
        raise UsageError(
            f"rule path {path!r} with no --host matches every request on every host",
            hint="Scope it: add `--host api.example.com`, or give a real path like "
            "`/v1/chat`.",
        )


def reset_session_files(cache_dir: str | Path) -> list[str]:
    """Drop every leftover rule/record artefact; return the names actually removed.

    A cache dir outlives the run that filled it, so without this the next ``proxy start``
    silently re-arms the previous scenario's stubs — which is indistinguishable, from the
    agent's side, from the app misbehaving.
    """
    cache = Path(cache_dir).expanduser()
    removed: list[str] = []
    for name in (
        "mock_rules.json",
        "mock_record.jsonl",
        "mock_record.json",  # pre-JSONL capture file
        "mock_mode.txt",
        "mock_record_name.txt",
        "mock_record_window.json",
        "flow_log.jsonl",
        "flow_bodies.jsonl",
    ):
        path = cache / name
        if path.exists():
            with contextlib.suppress(OSError):
                path.unlink()
                removed.append(name)
    return removed


def load_rules(path: Path) -> list[dict[str, Any]]:
    return load_doc(path)["rules"]


def load_cassette(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise UsageError(f"cassette not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise UsageError(f"cassette YAML does not parse: {exc}") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.get("entries") or [])
    raise UsageError("cassette must be a mapping with `entries:` or a list")


def save_cassette(path: Path, name: str, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"name": name, "entries": entries}, sort_keys=False),
        encoding="utf-8",
    )


def map_rule(
    method: str,
    path: str,
    *,
    status: int = 200,
    body: Any = None,
    headers: dict[str, str] | None = None,
    host: str | None = None,
    times: int = 0,
) -> dict[str, Any]:
    """A rule that answers from the rule itself — the request never reaches the server."""
    resp: dict[str, Any] = {"status": status}
    if body is not None:
        if isinstance(body, str):
            try:
                resp["body"] = json.loads(body)
            except json.JSONDecodeError:
                resp["body"] = body
        else:
            resp["body"] = body
    if headers:
        resp["headers"] = headers
    request: dict[str, Any] = {"method": method.upper(), "path": path}
    if host:
        request["host"] = host
    rule: dict[str, Any] = {
        "id": f"r{int(time.time() * 1000) % 10_000_000}-{random.randrange(1000):03d}",
        "action": "stub",
        # Kept under `request` (not `match`) so cassettes written by any earlier version
        # still load, and so a hand-edited cassette reads the way its author expects.
        "request": request,
        "response": resp,
    }
    if times:
        rule["times"] = int(times)
    return rule


def ensure_addon(cache: Path) -> Path:
    path = addon_path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ADDON_SCRIPT, encoding="utf-8")
    return path


def mitmdump_bin() -> str:
    """Resolve ``mitmdump`` next to this Python / ``aua``, then PATH.

    ``sys.executable.resolve()`` can jump out of a uv/venv into a shared CPython
    tree that has no sibling scripts — check the unresolved bin dir and ``sys.argv[0]``
    first so ``.venv/bin/mitmdump`` still wins.
    """
    name = "mitmdump.exe" if os.name == "nt" else "mitmdump"
    candidates: list[Path] = []
    exe = Path(sys.executable)
    candidates.append(exe.parent / name)
    candidates.append(exe.resolve().parent / name)
    if sys.argv and sys.argv[0]:
        argv0 = Path(sys.argv[0])
        candidates.append(argv0.parent / name)
        with contextlib.suppress(OSError):
            candidates.append(argv0.resolve().parent / name)
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file() and os.access(cand, os.X_OK):
            return key
    found = shutil.which("mitmdump")
    if found:
        return found
    raise UsageError(
        "mitmdump not found",
        hint='Install the optional extra: `pip install "android-ui-analyser[proxy]"` '
        "(and use the same venv's `aua` / `python -m`).",
    )


def ensure_mitm_ca() -> Path:
    """Return ``~/.mitmproxy/mitmproxy-ca-cert.pem``, generating the CA if needed."""
    if _MITM_CA_PEM.is_file():
        return _MITM_CA_PEM
    # First mitmdump run creates the CA; bind a throwaway port and quit.
    listen = pick_listen_port()
    bin_ = mitmdump_bin()
    proc = subprocess.Popen(  # noqa: S603
        [bin_, "-p", str(listen), "--set", "block_global=false", "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if _MITM_CA_PEM.is_file() or _MITM_CA_FULL.is_file():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.1)
    finally:
        with contextlib.suppress(Exception):
            proc.send_signal(signal.SIGTERM)
        with contextlib.suppress(Exception):
            proc.wait(timeout=3)
    # Older mitmproxy only writes mitmproxy-ca.pem; extract the cert half.
    if not _MITM_CA_PEM.is_file() and _MITM_CA_FULL.is_file():
        text = _MITM_CA_FULL.read_text(encoding="utf-8")
        idx = text.find("-----BEGIN CERTIFICATE-----")
        if idx >= 0:
            _MITM_CONFDIR.mkdir(parents=True, exist_ok=True)
            _MITM_CA_PEM.write_text(text[idx:], encoding="utf-8")
    if not _MITM_CA_PEM.is_file():
        raise UsageError(
            "could not generate the mitmproxy CA",
            hint=f"Run `{mitmdump_bin()} -p 0` once, then retry. Expected {_MITM_CA_PEM}.",
        )
    return _MITM_CA_PEM


def android_cert_hash(pem: Path) -> str:
    """OpenSSL ``subject_hash_old`` — Android's ``<hash>.0`` filename."""
    try:
        out = subprocess.check_output(  # noqa: S603
            [
                "openssl",
                "x509",
                "-inform",
                "PEM",
                "-subject_hash_old",
                "-noout",
                "-in",
                str(pem),
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise UsageError(
            "openssl is required to hash the mitm CA for Android",
            hint="Install openssl (macOS: `brew install openssl`) and retry.",
        ) from exc
    digest = (out or "").strip().splitlines()[0].strip()
    if not digest:
        raise UsageError(f"openssl returned an empty subject hash for {pem}")
    return digest


def _adb(serial: str, *args: str, check: bool = True, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["adb", "-s", serial, *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def install_system_ca(serial: str, *, pem: Path | None = None) -> dict[str, Any]:
    """Install the mitm CA into the *system* trust store (emulator / rooted device).

    Android 7+ apps ignore user CAs unless their NSC opts in. An NSC that lists only
    ``src="system"`` makes user-store installs useless — we must overlay the system
    store. On API 34+ that means a tmpfs + zygote ``nsenter`` bind (HTTP Toolkit recipe).
    The overlay is runtime-only and must be re-applied after reboot.
    """
    cert = ensure_mitm_ca() if pem is None else pem
    digest = android_cert_hash(cert)
    named = cert.with_name(f"{digest}.0")
    if named.resolve() != cert.resolve():
        named.write_bytes(cert.read_bytes())

    # Root the adbd; Google Play / user-build images often refuse this.
    root = _adb(serial, "root", check=False, timeout=30)
    root_out = ((root.stdout or "") + (root.stderr or "")).lower()
    if "cannot" in root_out or "production" in root_out or "unauthorized" in root_out:
        raise DeviceError(
            f"adb root refused on {serial}",
            hint=(
                "System CA install needs a *Google APIs* (rootable) AVD, not Google Play. "
                "Create a small one: `aua emulator ensure-proxy` → "
                "`aua emulator start --avd aua_proxy --headless` → `aua proxy start` "
                "(or `aua emulator list` / `recommend-proxy` for the suggested package)."
            ),
        )
    with contextlib.suppress(subprocess.TimeoutExpired, subprocess.CalledProcessError):
        _adb(serial, "wait-for-device", check=False, timeout=60)
    time.sleep(1.0)

    # Whoami after root — must be root for mount/nsenter.
    who = _adb(serial, "shell", "id", check=False, timeout=15)
    who_out = (who.stdout or "") + (who.stderr or "")
    if "uid=0" not in who_out:
        raise DeviceError(
            f"device {serial} is not root after `adb root` (got: {who_out.strip()!r})",
            hint=(
                "Use a Google APIs emulator image (rootable). "
                "`aua emulator ensure-proxy` then boot `--avd aua_proxy`."
            ),
        )

    remote_cert = f"/data/local/tmp/{digest}.0"
    remote_script = "/data/local/tmp/aua_install_ca.sh"
    script = f"""#!/system/bin/sh
set -e
CERT="{remote_cert}"
HASH="{digest}"
mkdir -p -m 700 /data/local/tmp/aua-ca-copy
rm -f /data/local/tmp/aua-ca-copy/*
if [ -d /apex/com.android.conscrypt/cacerts ]; then
  cp /apex/com.android.conscrypt/cacerts/* /data/local/tmp/aua-ca-copy/ 2>/dev/null || true
fi
if [ ! "$(ls -A /data/local/tmp/aua-ca-copy 2>/dev/null)" ]; then
  if [ -d /system/etc/security/cacerts ]; then
    cp /system/etc/security/cacerts/* /data/local/tmp/aua-ca-copy/ 2>/dev/null || true
  fi
fi
cp "$CERT" /data/local/tmp/aua-ca-copy/"$HASH".0
mount -t tmpfs tmpfs /system/etc/security/cacerts
cp /data/local/tmp/aua-ca-copy/* /system/etc/security/cacerts/
chown root:root /system/etc/security/cacerts/*
chmod 644 /system/etc/security/cacerts/*
chcon u:object_r:system_file:s0 /system/etc/security/cacerts/* 2>/dev/null || true
ZYGOTE_PID=$(pidof zygote 2>/dev/null || true)
ZYGOTE64_PID=$(pidof zygote64 2>/dev/null || true)
for Z_PID in $ZYGOTE_PID $ZYGOTE64_PID; do
  [ -n "$Z_PID" ] || continue
  if [ -d /apex/com.android.conscrypt/cacerts ]; then
    nsenter --mount=/proc/$Z_PID/ns/mnt -- \\
      /bin/mount --bind /system/etc/security/cacerts /apex/com.android.conscrypt/cacerts \\
      2>/dev/null || true
  fi
done
APP_PIDS=""
for Z in $ZYGOTE_PID $ZYGOTE64_PID; do
  [ -n "$Z" ] || continue
  APP_PIDS="$APP_PIDS $(ps -o PID= -P $Z 2>/dev/null || true)"
done
for PID in $APP_PIDS; do
  [ -n "$PID" ] || continue
  if [ -d /apex/com.android.conscrypt/cacerts ]; then
    nsenter --mount=/proc/$PID/ns/mnt -- \\
      /bin/mount --bind /system/etc/security/cacerts /apex/com.android.conscrypt/cacerts \\
      2>/dev/null || true
  fi
done
test -f /system/etc/security/cacerts/"$HASH".0
echo AUA_CA_OK:$HASH
"""
    # Push cert + script as files — avoids adb shell -c quoting breakage.
    try:
        _adb(serial, "push", str(named), remote_cert, timeout=30)
        local_script = named.parent / "aua_install_ca.sh"
        local_script.write_text(script, encoding="utf-8")
        _adb(serial, "push", str(local_script), remote_script, timeout=30)
        _adb(serial, "shell", "chmod", "755", remote_script, check=False, timeout=15)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise DeviceError(
            f"adb push of mitm CA / install script failed: {exc}",
            hint="Is the device online?",
        ) from exc

    result = _adb(serial, "shell", remote_script, check=False, timeout=90)
    out = (result.stdout or "") + (result.stderr or "")
    if "AUA_CA_OK:" not in out:
        raise DeviceError(
            "could not install mitm CA as a system trust anchor",
            hint=(
                "Need a rootable Google APIs AVD (`aua emulator ensure-proxy`). "
                "Script output:\n" + out[-800:]
            ),
        )
    return {
        "ok": True,
        "hash": digest,
        "pem": str(cert),
        "detail": "system CA overlay installed (re-apply after emulator reboot)",
        "hint": "Force-stop + relaunch the app under test so it inherits the Zygote mounts.",
    }


def tls_failures_in_log(cache_dir: Path, *, limit: int = 5) -> list[str]:
    """Return recent mitmdump lines that indicate the client rejected the forged cert."""
    log = Path(cache_dir).expanduser() / "mitmdump.log"
    if not log.is_file():
        return []
    with contextlib.suppress(OSError):
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = [
            ln.strip()
            for ln in lines
            if "does not trust the proxy" in ln
            or "Client TLS handshake failed" in ln
            or "certificate verify failed" in ln.lower()
        ]
        return hits[-limit:]
    return []


# Hostnames that belong to the Android platform / Google Play services rather than to any
# particular app under test. A handful of Google system components pin their own certificates
# and legitimately never trust a user/system CA overlay, even on a correctly provisioned
# rootable device — so a TLS failure against one of these hosts is not evidence that the *app
# under test* mistrusts the proxy, and lumping it in with app traffic is what previously
# produced a false "the CA is not trusted" diagnosis while the app's own traffic decrypted fine.
_SYSTEM_HOST_SUFFIXES = (
    "googleapis.com",
    "google.com",
    "gstatic.com",
    "googleusercontent.com",
    "gvt1.com",
    "gvt2.com",
    "android.com",
    "googlesource.com",
)

_TLS_FAIL_MARKERS = (
    "does not trust the proxy",
    "Client TLS handshake failed",
    "certificate verify failed",
)
_TLS_FAIL_HOST_RE = re.compile(r"certificate for ([^\s(]+)")
_CONNECT_RE = re.compile(r"^\S+:\s+CONNECT\s+([^\s:]+):\d+\s*$")
_SERVER_CONNECT_RE = re.compile(r"\]\s+server connect\s+([^\s:]+):\d+")


def _is_system_host(host: str) -> bool:
    host = (host or "").lower().strip(".")
    if not host:
        return False
    return any(host == suf or host.endswith("." + suf) for suf in _SYSTEM_HOST_SUFFIXES)


def diagnose_empty_recording(
    cache_dir: str | Path, *, since_ts: float, log_offset: int = 0
) -> dict[str, Any]:
    """Why ``mock record stop`` captured no flows, scoped to *this* recording only.

    Two independent, windowed sources of evidence — mixing in anything from outside the window
    is exactly how a stale TLS failure from an unrelated earlier run (or from a Google system
    service that never trusts the overlay) previously got blamed on the app under test:

    * ``mitmdump.log`` bytes written after *log_offset* — CONNECTs and client TLS failures.
      The log carries no date, only a per-line ``HH:MM:SS``, so a byte offset captured at
      ``mock record start`` is the only reliable window boundary across a log that spans
      multiple sessions (and, via the mode-flip restart, keeps appending across it).
    * ``flow_log.jsonl`` entries timestamped after *since_ts* — flows that were fully
      decrypted and relayed, independent of what the cassette recorder captured. This is the
      addon's always-on exchange log (see ``AuaMock.response()``), so it is evidence even when
      the cassette recorder itself is broken.

    Returns a dict with a ``diagnosis`` of:

    * ``"decrypted_not_recorded"`` — the app under test's traffic decrypted fine in this
      window, but the cassette still came out empty. Not a CA problem; likely an aua bug.
    * ``"tls_failed"`` — the app under test's own traffic failed the TLS handshake in this
      window. The one case where "does not trust the mitm CA" is actually justified.
    * ``"no_traffic"`` — no CONNECT, TLS failure, or decrypted flow of any kind in this window.
    * ``"system_traffic_only"`` — only OS/Google-services traffic was seen; the app under test
      made no HTTPS calls in this window.
    """
    log = Path(cache_dir).expanduser() / "mitmdump.log"
    window_text = ""
    if log.is_file():
        with contextlib.suppress(OSError), log.open("rb") as fh:
            fh.seek(max(0, int(log_offset)))
            window_text = fh.read().decode("utf-8", errors="replace")

    connects = 0
    tls_system: list[str] = []
    tls_other: list[str] = []
    for raw_line in window_text.splitlines():
        line = raw_line.strip()
        if _CONNECT_RE.match(line) or _SERVER_CONNECT_RE.search(line):
            connects += 1
        if any(marker in line for marker in _TLS_FAIL_MARKERS):
            host_match = _TLS_FAIL_HOST_RE.search(line)
            host = host_match.group(1) if host_match else ""
            (tls_system if _is_system_host(host) else tls_other).append(line)

    flows = read_flows_since(cache_dir, since_ts)
    flows_system = [f for f in flows if _is_system_host(str(f.get("host") or ""))]
    flows_other = [f for f in flows if not _is_system_host(str(f.get("host") or ""))]

    if flows_other:
        diagnosis = "decrypted_not_recorded"
    elif tls_other:
        diagnosis = "tls_failed"
    elif connects or tls_system or flows_system:
        diagnosis = "system_traffic_only"
    else:
        diagnosis = "no_traffic"

    return {
        "diagnosis": diagnosis,
        "connects_seen": connects,
        "decrypted_flows_app": len(flows_other),
        "decrypted_flows_system": len(flows_system),
        "tls_failures_app": tls_other[-5:],
        "tls_failures_system": tls_system[-5:],
    }


def start_mitm(
    *,
    cache_dir: Path,
    port: int | None = None,
    mode: str = "map",
) -> tuple[int, int]:
    """Spawn mitmdump; return ``(pid, listen_port)``.

    When *port* is omitted / ``<=0``, picks a free random high port (not 8080).
    Persists the chosen port under ``mitmproxy.port`` so stop/record/replay share it.
    """
    try:
        import mitmproxy  # noqa: F401
    except ImportError as exc:
        raise UsageError(
            "mitmproxy is not installed",
            hint='Install the optional extra: `pip install "android-ui-analyser[proxy]"` '
            "or `uv pip install mitmproxy`.",
        ) from exc
    bin_ = mitmdump_bin()
    ensure_mitm_ca()  # CA must exist before clients CONNECT
    listen = pick_listen_port(preferred=port if port and port > 0 else None)
    addon = ensure_addon(cache_dir)
    rules = rules_path(cache_dir)
    if not rules.is_file():
        write_rules(rules, [])
    set_mode(rules, mode)
    log = Path(cache_dir).expanduser() / "mitmdump.log"
    env = os.environ.copy()
    env["AUA_MOCK_RULES"] = str(rules)
    env["AUA_MOCK_MODE"] = mode
    env["AUA_MOCK_RECORD"] = str(record_path(cache_dir))
    env["AUA_FLOW_LOG"] = str(flow_log_path(cache_dir))
    env["AUA_FLOW_BODIES"] = str(flow_bodies_path(cache_dir))
    log_fh = open(log, "ab")  # noqa: SIM115 — kept open for child stderr
    try:
        proc = subprocess.Popen(  # noqa: S603
            [
                bin_,
                "-p",
                str(listen),
                "-s",
                str(addon),
                "--set",
                "block_global=false",
            ],
            env=env,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    finally:
        log_fh.close()
    pid_path(cache_dir).write_text(str(proc.pid), encoding="utf-8")
    save_listen_port(cache_dir, listen)
    time.sleep(0.4)
    if proc.poll() is not None:
        clear_listen_port(cache_dir)
        with contextlib.suppress(OSError):
            pid_path(cache_dir).unlink()
        tail = ""
        with contextlib.suppress(OSError):
            tail = log.read_text(encoding="utf-8", errors="replace")[-800:]
        raise UsageError(
            f"mitmdump exited immediately (port {listen})",
            hint=(
                f"Last log lines ({log}):\n{tail}"
                if tail
                else f"Inspect {log}, or pass an explicit free `--port`."
            ),
        )
    return proc.pid, listen


def stop_mitm(cache_dir: Path) -> bool:
    path = pid_path(cache_dir)
    if not path.is_file():
        clear_listen_port(cache_dir)
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        path.unlink(missing_ok=True)
        clear_listen_port(cache_dir)
        return False
    with contextlib.suppress(Exception):
        os.kill(pid, signal.SIGTERM)
    path.unlink(missing_ok=True)
    clear_listen_port(cache_dir)
    return True


__all__ = [
    "CONNECTIVITY_CHECK_HOSTS",
    "android_cert_hash",
    "backfill_rule_ids",
    "cassette_dir",
    "clear_listen_port",
    "clear_record_window",
    "clear_rules",
    "clear_state",
    "diagnose_empty_recording",
    "ensure_addon",
    "ensure_mitm_ca",
    "ensure_reverse_tunnel",
    "flow_bodies_path",
    "guard_rule_scope",
    "install_system_ca",
    "load_cassette",
    "load_doc",
    "load_listen_port",
    "load_record",
    "load_record_window",
    "load_rules",
    "map_rule",
    "mitmdump_bin",
    "orphan_reason",
    "pick_listen_port",
    "pid_alive",
    "pid_path",
    "port_listening",
    "port_path",
    "proxy_health",
    "proxy_state_dir",
    "read_device_http_proxy",
    "read_flow_bodies",
    "read_state",
    "record_path",
    "record_window_path",
    "reset_record",
    "reset_session_files",
    "reverse_tunnel_active",
    "rewrite_rule",
    "rules_path",
    "save_cassette",
    "save_doc",
    "save_listen_port",
    "save_record_window",
    "set_mode",
    "start_mitm",
    "state_path",
    "stop_mitm",
    "tls_failures_in_log",
    "write_rules",
    "write_state",
]
