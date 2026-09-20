"""Attach AUA to one user-approved Chrome tab through the bundled extension.

The extension owns the Chrome debugger attachment.  This module owns only a short-lived,
user-private Unix socket used by Chrome's native-messaging host.  Losing either endpoint
causes the extension to detach, so AUA never leaves a personal tab under automation after a
daemon crash or a killed command.
"""

from __future__ import annotations

import base64
import contextlib
import hmac
import io
import json
import os
import secrets
import socket
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from PIL import Image

from ..errors import ConfigError, DeviceError, UnsupportedPlatformCapabilityError
from .web_tools import _DOM_SNAPSHOT_SCRIPT

BRIDGE_PROTOCOL = 1
BRIDGE_HOST_NAME = "com.aua.chrome_bridge"
ATTACHED_TARGET_ID = "existing-chrome"


def bridge_config_path() -> Path:
    override = os.environ.get("AUA_CHROME_BRIDGE_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "android-ui-analyser" / "chrome-bridge.json"


def bridge_socket_path() -> Path:
    override = os.environ.get("AUA_CHROME_BRIDGE_SOCKET")
    if override:
        return Path(override).expanduser().resolve()
    identity = str(os.getuid()) if hasattr(os, "getuid") else secrets.token_hex(4)
    # AF_UNIX paths are limited to roughly 104 bytes on macOS.  A stable path in the system
    # temporary directory stays comfortably below that limit even with a long home directory.
    return Path(tempfile.gettempdir()) / f"aua-chrome-{identity}.sock"


@dataclass(frozen=True)
class ChromeAttachOptions:
    attach_timeout_ms: int = 30_000
    action_timeout_ms: int = 5_000
    bridge_socket: str | None = None
    bridge_config: str | None = None


class ChromeBridge(Protocol):
    def request(
        self, method: str, params: Mapping[str, Any] | None = None, *, timeout_ms: int
    ) -> Any: ...

    def event_snapshot(self) -> list[dict[str, Any]]: ...

    def clear_events(self) -> int: ...

    def close(self) -> None: ...


@dataclass
class _PendingReply:
    ready: threading.Event
    value: dict[str, Any] | None = None


class ChromeBridgeServer:
    """Authenticated request/reply bridge between one AUA runtime and the extension."""

    def __init__(
        self,
        *,
        socket_path: Path,
        config_path: Path,
        attach_timeout_ms: int,
    ) -> None:
        self.socket_path = socket_path
        self.config_path = config_path
        self.attach_timeout_ms = attach_timeout_ms
        self._listener: socket.socket | None = None
        self._peer: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, _PendingReply] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._hello = threading.Event()
        self._disconnected = threading.Event()
        self._next_id = 1
        self._started = False

    @staticmethod
    def _read_line(stream: Any) -> dict[str, Any]:
        raw = stream.readline()
        if not raw:
            raise EOFError("Chrome native-messaging host disconnected")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeviceError(
                f"Chrome bridge sent invalid JSON: {exc}", code="chrome_bridge_protocol"
            ) from None
        if not isinstance(value, dict):
            raise DeviceError(
                "Chrome bridge message must be a JSON object", code="chrome_bridge_protocol"
            )
        return value

    def _remove_stale_socket(self) -> None:
        if not self.socket_path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.15)
            probe.connect(str(self.socket_path))
        except OSError:
            self.socket_path.unlink(missing_ok=True)
        else:
            raise DeviceError(
                "another AUA runtime is already waiting for the Chrome extension",
                code="chrome_bridge_busy",
                hint="Use the warm AUA daemon that owns the browser, or stop it before reconnecting.",
            )
        finally:
            probe.close()

    def _publish_config(self, token: str) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "protocol": BRIDGE_PROTOCOL,
            "socket": str(self.socket_path),
            "token": token,
        }
        temporary = self.config_path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"))
                stream.write("\n")
            os.replace(temporary, self.config_path)
            os.chmod(self.config_path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    def start(self) -> None:
        if self._started:
            return
        if not hasattr(socket, "AF_UNIX"):
            raise DeviceError(
                "existing-Chrome attachment currently requires macOS or Linux",
                code="chrome_bridge_unsupported",
            )
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(max(0.1, self.attach_timeout_ms / 1000))
            token = secrets.token_urlsafe(32)
            self._publish_config(token)
            self._listener = listener
            try:
                peer, _address = listener.accept()
            except TimeoutError:
                raise DeviceError(
                    "timed out waiting for a tab approved in the AUA Chrome extension",
                    code="chrome_extension_not_attached",
                    hint=(
                        "Open the AUA extension on the Chrome tab you want to share and choose "
                        "Attach this tab, then retry."
                    ),
                ) from None
            peer.settimeout(max(1.0, self.attach_timeout_ms / 1000))
            stream = peer.makefile("rb")
            hello = self._read_line(stream)
            if (
                hello.get("type") != "host_hello"
                or int(hello.get("protocol", 0)) != BRIDGE_PROTOCOL
                or not hmac.compare_digest(str(hello.get("token") or ""), token)
            ):
                peer.close()
                raise DeviceError(
                    "Chrome native-messaging host failed bridge authentication",
                    code="chrome_bridge_auth_failed",
                )
            peer.settimeout(None)
            self._peer = peer
            self._reader = threading.Thread(
                target=self._read_messages,
                args=(stream,),
                name="aua-chrome-bridge-reader",
                daemon=True,
            )
            self._reader.start()
            if not self._hello.wait(max(0.1, self.attach_timeout_ms / 1000)):
                raise DeviceError(
                    "the Chrome extension connected but no tab was approved",
                    code="chrome_extension_no_tab",
                    hint="Open the AUA extension on the target tab and choose Attach this tab.",
                )
        except BaseException:
            self.close()
            raise
        finally:
            if self._listener is listener:
                listener.close()
                self._listener = None
        self._started = True

    def _read_messages(self, stream: Any) -> None:
        try:
            while True:
                message = self._read_line(stream)
                reply_to = message.get("reply_to")
                if reply_to is not None:
                    with self._pending_lock:
                        pending = self._pending.get(str(reply_to))
                    if pending is not None:
                        pending.value = message
                        pending.ready.set()
                    continue
                kind = message.get("type")
                if kind == "hello":
                    if int(message.get("protocol", 0)) == BRIDGE_PROTOCOL and message.get(
                        "attached"
                    ):
                        self._hello.set()
                    continue
                if kind == "event" and isinstance(message.get("event"), dict):
                    self._events.append(dict(message["event"]))
        except (EOFError, OSError, DeviceError):
            pass
        finally:
            self._disconnected.set()
            with self._pending_lock:
                for pending in self._pending.values():
                    pending.ready.set()

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout_ms: int,
    ) -> Any:
        if not self._started or self._peer is None or self._disconnected.is_set():
            raise DeviceError(
                "the approved Chrome tab is no longer attached",
                code="chrome_extension_disconnected",
                hint="Open the AUA extension on the target tab and attach it again.",
            )
        with self._pending_lock:
            request_id = str(self._next_id)
            self._next_id += 1
            pending = _PendingReply(threading.Event())
            self._pending[request_id] = pending
        payload = {
            "id": request_id,
            "method": str(method),
            "params": dict(params or {}),
        }
        try:
            with self._write_lock:
                self._peer.sendall(
                    (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
                )
            if not pending.ready.wait(max(0.001, timeout_ms / 1000)):
                raise DeviceError(
                    f"Chrome extension operation {method!r} timed out",
                    code="chrome_extension_timeout",
                )
            message = pending.value
            if message is None:
                raise DeviceError(
                    "the approved Chrome tab disconnected during an operation",
                    code="chrome_extension_disconnected",
                )
            if not message.get("ok"):
                error = message.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise DeviceError(
                    f"Chrome extension operation failed: {detail or method}",
                    code=(
                        str(error.get("code"))
                        if isinstance(error, dict) and error.get("code")
                        else "chrome_extension_operation_failed"
                    ),
                )
            return message.get("result")
        except OSError:
            raise DeviceError(
                "the approved Chrome tab disconnected during an operation",
                code="chrome_extension_disconnected",
            ) from None
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def event_snapshot(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def clear_events(self) -> int:
        count = len(self._events)
        self._events.clear()
        return count

    def close(self) -> None:
        peer = self._peer
        self._peer = None
        if peer is not None:
            with contextlib.suppress(OSError):
                peer.shutdown(socket.SHUT_RDWR)
            peer.close()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        self.socket_path.unlink(missing_ok=True)
        self.config_path.unlink(missing_ok=True)
        self._disconnected.set()


class ChromeExtensionConnection:
    """A WebConnection backed by one explicitly approved existing Chrome tab."""

    def __init__(
        self,
        options: ChromeAttachOptions,
        *,
        bridge: ChromeBridge | None = None,
    ) -> None:
        self._options = options
        self._bridge = bridge
        self._closed = False
        self._marks: dict[str, int] = {}
        if self._bridge is None:
            server = ChromeBridgeServer(
                socket_path=(
                    Path(options.bridge_socket).expanduser().resolve()
                    if options.bridge_socket
                    else bridge_socket_path()
                ),
                config_path=(
                    Path(options.bridge_config).expanduser().resolve()
                    if options.bridge_config
                    else bridge_config_path()
                ),
                attach_timeout_ms=options.attach_timeout_ms,
            )
            server.start()
            self._bridge = server

    def _request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout_ms: int | None = None,
    ) -> Any:
        if self._closed:
            raise DeviceError("the attached Chrome tab is closed", code="web_target_closed")
        assert self._bridge is not None
        return self._bridge.request(
            method,
            params,
            timeout_ms=timeout_ms or self._options.action_timeout_ms,
        )

    @staticmethod
    def _unsupported(capability: str) -> Any:
        raise UnsupportedPlatformCapabilityError(
            "web existing-chrome", capability
        )

    def _state(self) -> dict[str, Any]:
        value = self._request("page_state")
        if not isinstance(value, dict):
            raise DeviceError("Chrome returned invalid page state", code="chrome_bridge_protocol")
        return value

    @property
    def url(self) -> str:
        return str(self._state().get("url") or "")

    def viewport_size(self) -> tuple[int, int]:
        viewport = self._state().get("viewport") or {}
        return int(viewport.get("width") or 0), int(viewport.get("height") or 0)

    def snapshot(self) -> str:
        value = self._request(
            "evaluate",
            {"expression": f"({_DOM_SNAPSHOT_SCRIPT})()", "await_promise": True},
        )
        if not isinstance(value, dict):
            raise DeviceError("Chrome returned invalid DOM data", code="chrome_bridge_protocol")
        return json.dumps(value, ensure_ascii=False)

    def screenshot_png(self) -> bytes:
        value = self._request("capture_screenshot")
        encoded = value.get("data") if isinstance(value, dict) else None
        if not isinstance(encoded, str):
            raise DeviceError("Chrome returned no PNG screenshot", code="screencap_failed")
        try:
            png = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise DeviceError("Chrome returned an invalid PNG screenshot", code="screencap_failed") from None
        viewport = value.get("viewport") if isinstance(value, dict) else None
        if isinstance(viewport, dict):
            expected = (int(viewport.get("width") or 0), int(viewport.get("height") or 0))
            if expected[0] > 0 and expected[1] > 0:
                try:
                    with Image.open(io.BytesIO(png)) as image:
                        if image.size != expected:
                            output = io.BytesIO()
                            image.resize(expected, Image.Resampling.LANCZOS).save(
                                output, format="PNG"
                            )
                            png = output.getvalue()
                except OSError:
                    raise DeviceError(
                        "Chrome returned an invalid PNG screenshot", code="screencap_failed"
                    ) from None
        return png

    def click(self, x: int, y: int) -> None:
        self._request("click", {"x": x, "y": y})

    def long_click(self, x: int, y: int, duration_ms: int) -> None:
        self._request("long_click", {"x": x, "y": y, "duration_ms": duration_ms})

    def type_text(self, text: str) -> None:
        self._request("type_text", {"text": text})

    def clear_text(self) -> None:
        self._request("clear_text")

    def press(self, key: str) -> None:
        self._request("press", {"key": key})

    def go_back(self) -> None:
        self._request("go_back")

    def reload(self) -> None:
        self._request("reload")

    def scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> None:
        self._request(
            "scroll", {"x": x, "y": y, "delta_x": delta_x, "delta_y": delta_y}
        )

    def goto(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigError("attached Chrome navigation requires an absolute http(s) URL")
        self._request("goto", {"url": url}, timeout_ms=self._options.attach_timeout_ms)

    def wait_idle(self, timeout_ms: int) -> None:
        self._request("wait_idle", {"timeout_ms": timeout_ms}, timeout_ms=timeout_ms + 500)

    def storage(self, *, include_values: bool = False) -> dict[str, Any]:
        del include_values
        return self._unsupported("browser.storage")

    def storage_export(self, path: str) -> dict[str, Any]:
        del path
        return self._unsupported("browser.storage")

    def storage_import(self, path: str) -> dict[str, Any]:
        del path
        return self._unsupported("browser.storage")

    def storage_clear(self, kinds: Sequence[str]) -> dict[str, Any]:
        del kinds
        return self._unsupported("browser.storage")

    def cache_clear(self) -> dict[str, Any]:
        return self._unsupported("browser.storage")

    def reset(self) -> dict[str, Any]:
        return self._unsupported("browser.storage")

    def network_status(self) -> dict[str, Any]:
        return self._unsupported("browser.network")

    def set_offline(self, offline: bool) -> dict[str, Any]:
        del offline
        return self._unsupported("browser.network")

    def set_throttle(
        self, *, latency_ms: int, download_kbps: int, upload_kbps: int
    ) -> dict[str, Any]:
        del latency_ms, download_kbps, upload_kbps
        return self._unsupported("browser.network")

    def set_cors(
        self,
        *,
        origin: str,
        hosts: Sequence[str],
        methods: Sequence[str],
        headers: Sequence[str],
        credentials: bool,
    ) -> dict[str, Any]:
        del origin, hosts, methods, headers, credentials
        return self._unsupported("browser.network")

    def clear_cors(self) -> dict[str, Any]:
        return self._unsupported("browser.network")

    def set_proxy(
        self,
        server: str,
        *,
        bypass: str | None,
        username: str | None,
        password: str | None,
    ) -> dict[str, Any]:
        del server, bypass, username, password
        return self._unsupported("browser.network")

    def clear_proxy(self) -> dict[str, Any]:
        return self._unsupported("browser.network")

    def har_start(self, path: str) -> dict[str, Any]:
        del path
        return self._unsupported("browser.network")

    def har_stop(self) -> dict[str, Any]:
        return self._unsupported("browser.network")

    def har_replay(
        self, path: str, *, url: str | None, not_found: str
    ) -> dict[str, Any]:
        del path, url, not_found
        return self._unsupported("browser.network")

    def har_clear(self) -> dict[str, Any]:
        return self._unsupported("browser.network")

    def mock_add(
        self,
        url: str,
        *,
        status: int,
        body: str,
        headers: Mapping[str, str] | None,
        abort: bool,
    ) -> dict[str, Any]:
        del url, status, body, headers, abort
        return self._unsupported("browser.network")

    def mock_clear(self, rule_id: str | None = None) -> dict[str, Any]:
        del rule_id
        return self._unsupported("browser.network")

    def diagnostics(
        self, *, limit: int, kinds: Sequence[str], since_ms: int | None
    ) -> dict[str, Any]:
        assert self._bridge is not None
        selected = {str(kind).strip().casefold() for kind in kinds if str(kind).strip()}
        events = [
            event
            for event in self._bridge.event_snapshot()
            if (since_ms is None or int(event.get("timestamp_ms") or 0) >= since_ms)
            and (not selected or str(event.get("kind") or "").casefold() in selected)
        ]
        bounded = events[-max(1, int(limit)) :]
        return {
            "ok": True,
            "action": "browser-diagnostics",
            "count": len(bounded),
            "total_count": len(events),
            "truncated": len(bounded) < len(events),
            "events": bounded,
        }

    def diagnostics_clear(self) -> dict[str, Any]:
        assert self._bridge is not None
        count = self._bridge.clear_events()
        self._marks.clear()
        return {"ok": True, "action": "browser-diagnostics-clear", "cleared": count}

    def mark_diagnostics(self, name: str, *, clear: bool = False) -> dict[str, Any]:
        if clear:
            self.diagnostics_clear()
        timestamp = int(time.time() * 1000)
        self._marks[str(name)] = timestamp
        return {
            "ok": True,
            "action": "browser-diagnostics-mark",
            "name": str(name),
            "timestamp_ms": timestamp,
        }

    def pages(self) -> dict[str, Any]:
        state = self._state()
        return {
            "ok": True,
            "action": "browser-pages",
            "pages": [
                {
                    "id": str(state.get("page_id") or "attached-tab"),
                    "index": 0,
                    "active": True,
                    "attached": True,
                    "url": state.get("url"),
                    "title": state.get("title"),
                    "frames": [],
                }
            ],
        }

    def page_select(self, page_id: str) -> dict[str, Any]:
        current = self.pages()["pages"][0]
        if page_id != current["id"] and page_id != "0":
            raise ConfigError(
                f"tab {page_id!r} was not approved",
                hint="Open the AUA extension on that tab and choose Attach this tab.",
            )
        self._request("focus")
        return {
            "ok": True,
            "action": "browser-page-select",
            "page_id": current["id"],
            "url": current["url"],
        }

    def page_close(self, page_id: str) -> dict[str, Any]:
        del page_id
        return self._unsupported("browser.pages.close")

    def trace_start(self) -> dict[str, Any]:
        return self._unsupported("browser.trace")

    def trace_stop(self, path: str) -> dict[str, Any]:
        del path
        return self._unsupported("browser.trace")

    def session_begin(self, session_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "action": "browser-session-begin",
            "session_id": session_id,
            "attached": True,
            "captured": [],
        }

    def session_finish(self, session_id: str) -> dict[str, Any]:
        self._request("detach", timeout_ms=1000)
        return {
            "ok": True,
            "action": "browser-session-finish",
            "session_id": session_id,
            "attached": True,
            "restored": [],
            "detached": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            with contextlib.suppress(DeviceError):
                self._request("detach", timeout_ms=1000)
        finally:
            self._closed = True
            assert self._bridge is not None
            self._bridge.close()


class ChromeExtensionLauncher:
    def launch(self, target: str, options: ChromeAttachOptions) -> ChromeExtensionConnection:
        del target
        return ChromeExtensionConnection(options)


__all__ = [
    "ATTACHED_TARGET_ID",
    "BRIDGE_HOST_NAME",
    "BRIDGE_PROTOCOL",
    "ChromeAttachOptions",
    "ChromeBridgeServer",
    "ChromeExtensionConnection",
    "ChromeExtensionLauncher",
    "bridge_config_path",
    "bridge_socket_path",
]
