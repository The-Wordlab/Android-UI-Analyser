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
"""aua mitmproxy addon — reload rules from RULES_PATH env."""
from __future__ import annotations

import json
import os
from pathlib import Path

from mitmproxy import http

_RULES_PATH = Path(os.environ.get("AUA_MOCK_RULES", ""))
_RECORD_PATH = Path(os.environ.get("AUA_MOCK_RECORD", ""))
_MODE = os.environ.get("AUA_MOCK_MODE", "map")  # map | record | replay


def _load_rules():
    if not _RULES_PATH.is_file():
        return []
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else data.get("entries") or []


def _match(flow: http.HTTPFlow, rule: dict) -> bool:
    req = rule.get("request") or {}
    method = (req.get("method") or "*").upper()
    path = req.get("path") or "*"
    if method not in ("*", flow.request.method.upper()):
        return False
    url_path = flow.request.path.split("?", 1)[0]
    if path != "*" and path not in url_path and url_path != path:
        # prefix or exact
        if not (path.endswith("*") and url_path.startswith(path[:-1])):
            if url_path != path and not url_path.startswith(path.rstrip("/") + "/"):
                return False
    return True


def _apply(flow: http.HTTPFlow, rule: dict) -> None:
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


class AuaMock:
    def request(self, flow: http.HTTPFlow) -> None:
        mode = os.environ.get("AUA_MOCK_MODE", _MODE)
        if mode == "record":
            return
        for rule in _load_rules():
            if _match(flow, rule):
                _apply(flow, rule)
                return

    def response(self, flow: http.HTTPFlow) -> None:
        mode = os.environ.get("AUA_MOCK_MODE", _MODE)
        if mode != "record" or not _RECORD_PATH:
            return
        entry = {
            "request": {
                "method": flow.request.method.upper(),
                "path": flow.request.path.split("?", 1)[0],
            },
            "response": {
                "status": flow.response.status_code if flow.response else 0,
                "body": (flow.response.text if flow.response else "")[:200_000],
            },
        }
        entries = []
        if _RECORD_PATH.is_file():
            try:
                entries = json.loads(_RECORD_PATH.read_text(encoding="utf-8"))
            except Exception:
                entries = []
        if not isinstance(entries, list):
            entries = []
        entries.append(entry)
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECORD_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


addons = [AuaMock()]
'''


def cassette_dir(memory_dir: str | Path) -> Path:
    return Path(memory_dir).expanduser() / "cassettes"


def rules_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "mock_rules.json"


def record_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "mock_record.json"


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


def write_rules(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def load_rules(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return data
    return list(data.get("entries") or [])


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
) -> dict[str, Any]:
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
    return {"request": {"method": method.upper(), "path": path}, "response": resp}


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
    log = Path(cache_dir).expanduser() / "mitmdump.log"
    env = os.environ.copy()
    env["AUA_MOCK_RULES"] = str(rules)
    env["AUA_MOCK_MODE"] = mode
    env["AUA_MOCK_RECORD"] = str(record_path(cache_dir))
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
    "android_cert_hash",
    "cassette_dir",
    "clear_listen_port",
    "ensure_addon",
    "ensure_mitm_ca",
    "install_system_ca",
    "load_cassette",
    "load_listen_port",
    "load_rules",
    "map_rule",
    "mitmdump_bin",
    "pick_listen_port",
    "pid_path",
    "port_path",
    "record_path",
    "rules_path",
    "save_cassette",
    "save_listen_port",
    "start_mitm",
    "stop_mitm",
    "tls_failures_in_log",
    "write_rules",
]
