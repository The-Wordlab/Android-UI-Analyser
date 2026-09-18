"""Semantic runtime for one booted iOS simulator.

Coordinates arrive from the engine as screenshot pixels and leave for AXe as logical points;
:class:`DisplayGeometry` holds the scale. Nothing here knows about Android.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import read_budget
from ..errors import DeviceError, UsageError
from ..providers.base import Bounds, ScreenImage
from ..schema import AppContext, MatchMode
from . import ios_tree
from .geometry import DisplayGeometry
from .ios_tools import IOSTools, png_size
from .runtime import TargetRuntime

# HID usage ids for the keys AUA names semantically (USB HID keyboard page).
HID_KEYS: dict[str, int] = {
    "enter": 40,
    "return": 40,
    "search": 40,
    "done": 40,
    "go": 40,
    "escape": 41,
    "esc": 41,
    "delete": 42,
    "del": 42,
    "backspace": 42,
    "tab": 43,
    "space": 44,
    "right": 79,
    "left": 80,
    "down": 81,
    "up": 82,
}
HARDWARE_BUTTONS: dict[str, str] = {
    "home": "home",
    "lock": "lock",
    "power": "lock",
    "siri": "siri",
    "side_button": "side-button",
    "side-button": "side-button",
    "apple_pay": "apple-pay",
    "apple-pay": "apple-pay",
}
# iOS has no back key: the platform gesture is a swipe in from the left edge.
GESTURE_KEYS: dict[str, str] = {"back": "swipe-from-left-edge"}
KEY_NAMES = frozenset(HID_KEYS) | frozenset(HARDWARE_BUTTONS) | frozenset(GESTURE_KEYS)
_HID_LEFT_COMMAND = 227
_HID_A = 4
_HID_V = 25
_IME_ACTIONS = frozenset({"search", "done", "go", "send", "next", "enter", "return"})

# simctl only grants what TCC records under these services; everything else it can neither
# grant nor restore, so those rows are not reported as restorable state.
_TCC_TO_SIMCTL: dict[str, str] = {
    "kTCCServiceCalendar": "calendar",
    "kTCCServiceAddressBook": "contacts",
    "kTCCServicePhotos": "photos",
    "kTCCServicePhotosAdd": "photos-add",
    "kTCCServiceMediaLibrary": "media-library",
    "kTCCServiceMicrophone": "microphone",
    "kTCCServiceMotion": "motion",
    "kTCCServiceReminders": "reminders",
    "kTCCServiceSiri": "siri",
}
_TCC_ALLOWED = 2


def keycode_for(name: str) -> int | None:
    """HID usage id for a semantic key name, ``hid:<n>`` spelling, or plain number."""

    key = name.strip().casefold()
    if key in HID_KEYS:
        return HID_KEYS[key]
    if key.startswith("hid:") and key[4:].isdigit():
        return int(key[4:])
    if key.isdigit():
        return int(key)
    return None


def is_known_key(name: str) -> bool:
    return name.strip().casefold() in KEY_NAMES or keycode_for(name) is not None


def _fmt(value: float) -> str:
    return (
        f"{value:.2f}".rstrip("0").rstrip(".") if not float(value).is_integer() else str(int(value))
    )


class IOSSimulatorRuntime(TargetRuntime):
    """One connected simulator, driven through AXe and simctl."""

    def __init__(
        self,
        tools: IOSTools,
        udid: str,
        *,
        geometry: DisplayGeometry,
        boot_token: str | None = None,
        data_path: str | None = None,
    ) -> None:
        self.target_id = udid
        self._tools = tools
        self._geometry = geometry
        self._boot_token = boot_token
        self._data_path = data_path
        # pid -> bundle id. Pids are not recycled within a session, so a hit is always right and
        # only an unknown pid (a fresh launch) costs another `launchctl list`.
        self._apps: dict[int, str] = {}

    def _running_apps(self, *pids: int) -> dict[int, str]:
        if any(pid not in self._apps for pid in pids):
            self._apps.update(self._tools.running_apps(self.target_id))
        return self._apps

    # -- reads ------------------------------------------------------------------------------

    def read_deadline(
        self, budget: read_budget.ReadBudget
    ) -> contextlib.AbstractContextManager[None]:
        return read_budget.activate(budget)

    def display_geometry(self) -> DisplayGeometry:
        return self._geometry

    def window_size(self) -> tuple[int, int]:
        return self._geometry.canonical_size

    def _describe_ui(self, *extra: str) -> list[dict[str, Any]]:
        result = self._tools.axe("describe-ui", "--udid", self.target_id, *extra, timeout_s=30.0)
        payload = json.loads(result.text or "[]")
        if isinstance(payload, dict):
            return [payload]
        return (
            [node for node in payload if isinstance(node, dict)]
            if isinstance(payload, list)
            else []
        )

    def dump_hierarchy(self, compressed: bool = False) -> str:
        del compressed  # AXe has one level of detail
        roots = self._describe_ui()
        pids = [int(root["pid"]) for root in roots if str(root.get("pid", "")).isdigit()]
        return ios_tree.envelope(roots, self._running_apps(*pids))

    def screenshot(self) -> ScreenImage:
        with tempfile.TemporaryDirectory(prefix="aua-ios-") as tmp:
            data = self._tools.screenshot_png(self.target_id, Path(tmp) / "frame.png")
        size = png_size(data)
        if size is None:
            raise DeviceError("simctl returned no PNG screenshot", code="screencap_failed")
        return ScreenImage(data, width=size[0], height=size[1])

    def current_app(self) -> AppContext:
        # The element under the screen centre names the frontmost process in ~0.1 s; the whole
        # tree is not needed to know who owns the screen.
        native_x, native_y = self._geometry.native_size
        roots = self._describe_ui("--point", f"{_fmt(native_x / 4)},{_fmt(native_y / 3)}")
        pid = next((int(root["pid"]) for root in roots if str(root.get("pid", "")).isdigit()), None)
        if pid is None:
            return AppContext()
        return AppContext(app_id=self._running_apps(pid).get(pid))

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        return ios_tree.find_bounds(
            json.dumps(self._describe_ui()),
            geometry=self._geometry,
            query=text,
            match=match,
            ignore_case=ignore_case,
            by=by,
        )

    # -- input --------------------------------------------------------------------------------

    def _native(self, x: int, y: int) -> tuple[str, str]:
        native_x, native_y = self._geometry.to_native((x, y))
        return _fmt(native_x), _fmt(native_y)

    def click(self, x: int, y: int) -> None:
        native_x, native_y = self._native(x, y)
        self._tools.axe("tap", "-x", native_x, "-y", native_y, "--udid", self.target_id)

    def click_once(self, x: int, y: int) -> None:
        native_x, native_y = self._native(x, y)
        self._tools.axe(
            "touch", "-x", native_x, "-y", native_y, "--down", "--up", "--udid", self.target_id
        )

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        native_x, native_y = self._native(x, y)
        self._tools.axe(
            "touch",
            "-x",
            native_x,
            "-y",
            native_y,
            "--down",
            "--up",
            "--delay",
            _fmt(max(duration_ms, 1) / 1000.0),
            "--udid",
            self.target_id,
        )

    def touch_down(self, x: int, y: int) -> None:
        native_x, native_y = self._native(x, y)
        self._tools.axe("touch", "-x", native_x, "-y", native_y, "--down", "--udid", self.target_id)

    def touch_up(self, x: int, y: int) -> None:
        native_x, native_y = self._native(x, y)
        self._tools.axe("touch", "-x", native_x, "-y", native_y, "--up", "--udid", self.target_id)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        start_x, start_y = self._native(x1, y1)
        end_x, end_y = self._native(x2, y2)
        self._tools.axe(
            "swipe",
            "--start-x",
            start_x,
            "--start-y",
            start_y,
            "--end-x",
            end_x,
            "--end-y",
            end_y,
            "--duration",
            _fmt(max(duration_ms, 1) / 1000.0),
            "--udid",
            self.target_id,
        )

    def send_text(self, text: str, *, clear: bool = True) -> None:
        if clear:
            self.clear_text()
        if not text:
            return
        if text.isascii() and text.isprintable():
            self._tools.axe(
                "type", "--stdin", "--udid", self.target_id, input_bytes=text.encode("utf-8")
            )
            return
        # AXe types through a US HID keyboard, so anything beyond ASCII goes in via the pasteboard.
        self.set_clipboard(text)
        self.paste()

    def clear_text(self) -> None:
        self._key_combo(_HID_LEFT_COMMAND, _HID_A)
        self._key(HID_KEYS["delete"])

    def send_ime_action(self, action: str = "search") -> None:
        if action.strip().casefold() not in _IME_ACTIONS:
            raise UsageError(
                f"unknown input action '{action}'",
                hint="one of: " + ", ".join(sorted(_IME_ACTIONS)),
            )
        self._key(HID_KEYS["enter"])

    def press(self, key: str) -> None:
        name = key.strip().casefold()
        if name in HARDWARE_BUTTONS:
            self._tools.axe("button", HARDWARE_BUTTONS[name], "--udid", self.target_id)
            return
        if name in GESTURE_KEYS:
            width, height = self._geometry.native_size
            self._tools.axe(
                "gesture",
                GESTURE_KEYS[name],
                "--screen-width",
                _fmt(width),
                "--screen-height",
                _fmt(height),
                "--udid",
                self.target_id,
            )
            return
        code = keycode_for(name)
        if code is None:
            raise UsageError(
                f"unknown key '{key}'",
                hint="Valid: " + ", ".join(sorted(KEY_NAMES)) + ", or hid:<code>.",
            )
        self._key(code)

    def _key(self, code: int) -> None:
        self._tools.axe("key", str(code), "--udid", self.target_id)

    def _key_combo(self, modifier: int, code: int) -> None:
        self._tools.axe(
            "key-combo", "--modifiers", str(modifier), "--key", str(code), "--udid", self.target_id
        )

    # -- metadata -----------------------------------------------------------------------------

    def app_version(self, app_id: str) -> str | None:
        info = self._tools.app_info(self.target_id, app_id)
        version = info.get("CFBundleShortVersionString") or info.get("CFBundleVersion")
        return str(version) if version else None

    def device_locale(self) -> str | None:
        result = self._tools.simctl(
            "spawn",
            self.target_id,
            "defaults",
            "read",
            "-g",
            "AppleLocale",
            timeout_s=10.0,
            check=False,
        )
        if not result.ok:
            return None
        raw = result.text.strip().split("@", 1)[0].replace("_", "-")
        return raw or None

    def instance_token(self) -> str | None:
        return self._boot_token

    # -- app lifecycle ------------------------------------------------------------------------

    def launch_app(self, app_id: str, *, activity: str | None = None) -> None:
        del activity  # iOS apps have one entry point
        result = self._tools.simctl("launch", self.target_id, app_id, timeout_s=60.0, check=False)
        if not result.ok:
            raise DeviceError(
                f"could not launch {app_id}: {result.error_text}",
                code="app_launch_failed",
                hint="Check the bundle id with `aua app exists <bundle>`; install the .app first.",
            )

    def stop_app(self, app_id: str) -> None:
        result = self._tools.simctl(
            "terminate", self.target_id, app_id, timeout_s=30.0, check=False
        )
        # simctl puts the idempotent "already stopped" explanation below its generic
        # NSPOSIXErrorDomain header. error_text intentionally contains only the first line.
        detail = (result.stderr.decode("utf-8", "replace") or result.text).casefold()
        if not result.ok and "found nothing to terminate" not in detail:
            raise self._tools.error(result, "simctl terminate")

    def clear_app(self, app_id: str) -> str | None:
        self.stop_app(app_id)
        result = self._tools.simctl(
            "get_app_container", self.target_id, app_id, "data", timeout_s=15.0, check=False
        )
        container = result.text.strip()
        if not result.ok or not container or container == "(null)":
            raise DeviceError(
                f"{app_id} has no data container on {self.target_id}",
                code="app_not_installed",
                hint="Install the app first; system apps cannot be cleared.",
            )
        path = Path(container)
        if "/Containers/Data/Application/" not in path.as_posix() or not path.is_dir():
            raise DeviceError(
                f"unexpected data container path {container!r}", code="ios_tool_failed"
            )
        for child in path.iterdir():
            if child.name == ".com.apple.mobile_container_manager.metadata.plist":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        self._tools.simctl(
            "privacy", self.target_id, "reset", "all", app_id, timeout_s=30.0, check=False
        )
        return f"cleared {app_id} data container and reset its permissions"

    def grant_permissions(self, app_id: str) -> None:
        self._tools.simctl("privacy", self.target_id, "grant", "all", app_id, timeout_s=30.0)

    def granted_permissions(self, app_id: str) -> list[str]:
        if not self._data_path:
            return []
        db = Path(self._data_path) / "Library" / "TCC" / "TCC.db"
        if not db.is_file():
            return []
        try:
            with contextlib.closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
                rows = conn.execute(
                    "SELECT service, auth_value FROM access WHERE client = ?", (app_id,)
                ).fetchall()
        except sqlite3.Error:
            return []
        granted = {
            _TCC_TO_SIMCTL[str(service)]
            for service, auth in rows
            if str(service) in _TCC_TO_SIMCTL and int(auth) == _TCC_ALLOWED
        }
        return sorted(granted)

    def restore_permissions(self, app_id: str, granted: Sequence[str]) -> None:
        self._tools.simctl(
            "privacy", self.target_id, "reset", "all", app_id, timeout_s=30.0, check=False
        )
        for service in granted:
            if service in _TCC_TO_SIMCTL.values():
                self._tools.simctl(
                    "privacy", self.target_id, "grant", service, app_id, timeout_s=30.0, check=False
                )

    # -- links / clipboard / location -----------------------------------------------------------

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        del package  # iOS routes a URL to its registered handler; there is no chooser
        result = self._tools.simctl("openurl", self.target_id, uri, timeout_s=30.0, check=False)
        if not result.ok:
            raise DeviceError(
                f"no app on the simulator handles {uri!r}: {result.error_text}",
                code="link_unhandled",
                hint="Install the app that registers this scheme or universal link.",
            )

    def query_uri_handlers(self, uri: str) -> list[str]:
        return []

    def set_clipboard(self, text: str) -> None:
        self._tools.simctl(
            "pbcopy", self.target_id, timeout_s=15.0, input_bytes=text.encode("utf-8")
        )

    def get_clipboard(self) -> str:
        return self._tools.simctl("pbpaste", self.target_id, timeout_s=15.0).text

    def paste(self) -> None:
        self._key_combo(_HID_LEFT_COMMAND, _HID_V)

    def set_location(self, lat: float, lon: float) -> None:
        self._tools.simctl("location", self.target_id, "set", f"{lat},{lon}", timeout_s=15.0)

    def close(self) -> None:
        return None


__all__ = [
    "GESTURE_KEYS",
    "HARDWARE_BUTTONS",
    "HID_KEYS",
    "KEY_NAMES",
    "IOSSimulatorRuntime",
    "is_known_key",
    "keycode_for",
]
