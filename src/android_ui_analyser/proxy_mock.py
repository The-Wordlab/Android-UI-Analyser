"""Headless HTTP mock / record / replay via mitmproxy (optional extra).

No GUI. Cassettes are YAML under ``memory.dir/cassettes/``. Live rules live in a JSON
sidecar the mitmproxy addon reloads. Device wiring (``http_proxy`` + ``adb reverse``) is
owned by the Engine; this module is pure cassette/rule logic + process helpers.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from .errors import UsageError

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


def addon_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "aua_mitm_addon.py"


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


def start_mitm(
    *,
    cache_dir: Path,
    port: int = 8080,
    mode: str = "map",
) -> int:
    """Spawn mitmdump; return pid. Raises UsageError if mitmproxy missing."""
    try:
        import mitmproxy  # noqa: F401
    except ImportError as exc:
        raise UsageError(
            "mitmproxy is not installed",
            hint='Install the optional extra: `pip install "android-ui-analyser[proxy]"` '
            "or `uv pip install mitmproxy`.",
        ) from exc
    addon = ensure_addon(cache_dir)
    rules = rules_path(cache_dir)
    if not rules.is_file():
        write_rules(rules, [])
    env = os.environ.copy()
    env["AUA_MOCK_RULES"] = str(rules)
    env["AUA_MOCK_MODE"] = mode
    env["AUA_MOCK_RECORD"] = str(record_path(cache_dir))
    proc = subprocess.Popen(  # noqa: S603
        [
            "mitmdump",
            "-p",
            str(port),
            "-s",
            str(addon),
            "--set",
            "block_global=false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_path(cache_dir).write_text(str(proc.pid), encoding="utf-8")
    time.sleep(0.4)
    if proc.poll() is not None:
        raise UsageError(
            "mitmdump exited immediately",
            hint="Check port conflicts or run `mitmdump -p 8080` manually.",
        )
    return proc.pid


def stop_mitm(cache_dir: Path) -> bool:
    path = pid_path(cache_dir)
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        path.unlink(missing_ok=True)
        return False
    with contextlib.suppress(Exception):
        os.kill(pid, signal.SIGTERM)
    path.unlink(missing_ok=True)
    return True


__all__ = [
    "cassette_dir",
    "ensure_addon",
    "load_cassette",
    "load_rules",
    "map_rule",
    "pid_path",
    "record_path",
    "rules_path",
    "save_cassette",
    "start_mitm",
    "stop_mitm",
    "write_rules",
]
