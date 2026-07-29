"""Device plumbing: a thin, mockable wrapper over ``uiautomator2`` (PRD §6, §14).

``Device`` is an ABC defining the exact surface the engine/daemon/CLI use. The real
``Uiautomator2Device`` lazy-imports ``uiautomator2`` (so the core CLI works with the
library absent), keeps a warm connection, and reconnects once on a transient error
before failing. Tests supply a fake conforming to the same ABC — no device required.
"""

from __future__ import annotations

import contextlib
import logging
import re
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import DeviceError
from .providers.base import Bounds, ScreenImage
from .schema import DeviceInfo, MatchMode

logger = logging.getLogger("android_ui_analyser.device")


def _bounds_from_info(info: dict[str, Any]) -> Bounds | None:
    b = info.get("bounds") if isinstance(info, dict) else None
    if not isinstance(b, dict):
        return None
    try:
        return (int(b["left"]), int(b["top"]), int(b["right"]), int(b["bottom"]))
    except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
        return None


class Device(ABC):
    """The device surface the rest of the tool depends on."""

    serial: str

    # -- capture -----------------------------------------------------------
    @abstractmethod
    def window_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def dump_hierarchy(self, compressed: bool = False) -> str: ...

    @abstractmethod
    def screenshot(self) -> ScreenImage: ...

    @abstractmethod
    def current_app(self) -> dict[str, str]: ...

    # -- raw input primitives ---------------------------------------------
    @abstractmethod
    def click(self, x: int, y: int) -> None: ...

    @abstractmethod
    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None: ...

    @abstractmethod
    def send_text(self, text: str, *, clear: bool = True) -> None: ...

    @abstractmethod
    def clear_text(self) -> None: ...

    @abstractmethod
    def send_ime_action(self, action: str = "search") -> None: ...

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None: ...

    @abstractmethod
    def press(self, key: str) -> None: ...

    # -- hierarchy selectors (T0/T1) --------------------------------------
    @abstractmethod
    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        """Cheap selector locate — return the box of the first match, or None.

        ``by``: ``"text"`` searches text + content-desc (default); ``"id"`` matches the
        resource-id (a bare tail like ``containerChatDetail`` matches the id's suffix) —
        this can find containers that the parsed element list prunes; ``"desc"`` is
        content-desc only.
        """

    # -- optional metadata (best-effort; default unknown) -----------------
    def app_version(self, package: str) -> str | None:
        """Best-effort app versionName for memory freshness; ``None`` if unknown."""
        return None

    # -- composed helpers (built on the primitives; usually not overridden)-
    def input_text(
        self, x: int, y: int, text: str, *, clear: bool = True, submit: bool = False
    ) -> None:
        self.click(x, y)
        self.send_text(text, clear=clear)
        if submit:
            self.send_ime_action("search")

    def wait_for(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        timeout_ms: int = 5000,
        by: str = "text",
    ) -> Bounds | None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            found = self.find_text(text, match=match, ignore_case=ignore_case, by=by)
            if found is not None:
                return found
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)

    def wait_idle(self, timeout_ms: int = 5000) -> None:  # overridden by real device
        return None

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        raise DeviceError("app launch requires a real device")  # overridden by real device

    def stop_app(self, package: str) -> None:  # overridden by real device
        raise DeviceError("app stop requires a real device")

    def clear_app(self, package: str) -> None:  # overridden by real device
        """Wipe app data/cache (``pm clear``) — Maestro ``clearState``."""
        raise DeviceError("app clear requires a real device")

    def grant_permissions(self, package: str) -> None:  # overridden by real device
        """Auto-grant runtime permissions for *package* (best-effort)."""
        raise DeviceError("app grant requires a real device")

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        raise DeviceError("open-link requires a real device")  # overridden by real device

    def query_uri_handlers(self, uri: str) -> list[str]:
        """Packages that claim *uri* (best-effort; empty when unknown)."""
        return []

    def double_click(self, x: int, y: int) -> None:
        """Double-tap at pixel coordinates (default: two quick clicks)."""
        self.click(x, y)
        time.sleep(0.05)
        self.click(x, y)

    def hide_keyboard(self) -> None:
        """Dismiss the soft keyboard without navigating away when possible.

        Prefers ``KEYCODE_ESCAPE`` (does not finish the activity); falls back to
        ``back``, which on Android usually only hides the IME when it is showing.
        """
        try:
            self.press("KEYCODE_ESCAPE")
        except Exception:  # pragma: no cover - device-specific
            self.press("back")

    # -- Maestro-style device controls (best-effort; emulators fare better) ---------

    def set_clipboard(self, text: str) -> None:
        raise DeviceError("clipboard requires a real device")

    def get_clipboard(self) -> str:
        raise DeviceError("clipboard requires a real device")

    def paste(self) -> None:
        """Paste clipboard into the focused field (``KEYCODE_PASTE``)."""
        self.press("KEYCODE_PASTE")

    def set_location(self, lat: float, lon: float) -> None:
        raise DeviceError("location requires a real device")

    def set_orientation(self, mode: str) -> None:
        raise DeviceError("orientation requires a real device")

    def get_orientation(self) -> str:
        return "unknown"

    def set_airplane_mode(self, enabled: bool) -> None:
        raise DeviceError("airplane mode requires a real device")

    def get_airplane_mode(self) -> bool | None:
        return None

    def add_media(self, local_path: str, *, remote_dir: str = "/sdcard/DCIM/Camera") -> str:
        raise DeviceError("add media requires a real device")

    def start_recording(self, remote_path: str = "/sdcard/aua_recording.mp4") -> str:
        raise DeviceError("screen recording requires a real device")

    def stop_recording(self, local_path: str) -> str:
        raise DeviceError("screen recording requires a real device")

    def set_clock(self, timestamp_ms: int) -> None:
        raise DeviceError("clock travel requires a real device")

    def erase_chars(self, count: int) -> None:
        """Delete *count* characters before the caret in the focused field."""
        for _ in range(max(0, count)):
            self.press("KEYCODE_DEL")

    def close(self) -> None:  # overridden by real device
        """Release the device connection / on-device agent (no-op by default)."""
        return None

    def scroll_to(
        self,
        query: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        max_swipes: int = 8,
        by: str = "text",
    ) -> Bounds | None:
        found = self.find_text(query, match=match, ignore_case=ignore_case, by=by)
        if found is not None:
            return found
        w, h = self.window_size()
        for _ in range(max_swipes):
            self.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.3), 300)
            found = self.find_text(query, match=match, ignore_case=ignore_case, by=by)
            if found is not None:
                return found
        return None


# --------------------------------------------------------------------------- real impl


_PRESS_ALIASES = {
    "back": "back",
    "home": "home",
    "enter": "enter",
    "recents": "recent",
    "recent": "recent",
    "menu": "menu",
    "search": "search",
    "power": "power",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "del": "del",
    "delete": "del",
    "backspace": "del",
    "paste": "paste",
}

_ORIENTATION_ALIASES = {
    "portrait": "n",
    "natural": "n",
    "n": "n",
    "landscape": "l",
    "left": "l",
    "l": "l",
    "right": "r",
    "r": "r",
    "upsidedown": "u",
    "u": "u",
}


class Uiautomator2Device(Device):
    """Warm ``uiautomator2`` connection with single auto-reconnect."""

    def __init__(self, serial: str, settle_wait: float = 0.0) -> None:
        self.serial = serial
        self._settle = settle_wait
        self._d: Any = None
        self._winsize: tuple[int, int] | None = None
        self._recording_remote: str | None = None
        self._recording_proc: subprocess.Popen[bytes] | None = None
        self._connect()

    # -- connection --------------------------------------------------------

    def _connect(self) -> None:
        try:
            import uiautomator2 as u2
        except ImportError as exc:  # pragma: no cover - exercised only without dep
            raise DeviceError(
                "uiautomator2 is not installed",
                hint="pip install 'android-ui-analyser' (uiautomator2 is a base dependency).",
            ) from exc
        try:
            self._d = u2.connect(self.serial)
            # Don't block on idle for our reads; we manage waits explicitly.
            with contextlib.suppress(Exception):  # pragma: no cover - older u2
                self._d.settings["wait_timeout"] = 5.0
        except Exception as exc:
            raise DeviceError(
                f"could not connect to device '{self.serial}': {exc}",
                hint="Run `aua devices` and check the emulator/phone is reachable via adb.",
            ) from exc

    def close(self) -> None:
        """Stop the on-device uiautomator2 server, releasing the UiAutomation slot.

        Without this, the server (an ``app_process``) survives ``aua daemon stop`` and
        blocks other tools (adb ``uiautomator dump``, Maestro) that need UiAutomation.
        """
        d = self._d
        self._d = None
        if d is not None:
            with contextlib.suppress(Exception):
                d.stop_uiautomator()

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a uiautomator2 attribute (method OR property) with one auto-reconnect.

        Name-based dispatch (so the reconnect rebind is correct) plus a callable check
        (so it tolerates u2 versions exposing ``window_size``/``app_current`` as either a
        method or a property — the source of the ``'dict' object is not callable`` error).
        """

        def invoke() -> Any:
            attr = getattr(self._d, name)
            return attr(*args, **kwargs) if callable(attr) else attr

        try:
            return invoke()
        except Exception as exc:
            logger.warning("device op '%s' failed (%s); reconnecting once", name, exc)
            try:
                self._connect()
                return invoke()
            except Exception as exc2:
                raise DeviceError(
                    f"device operation '{name}' failed after reconnect: {exc2}",
                    hint="Check the device is still attached (`aua devices`).",
                ) from exc2

    # -- capture -----------------------------------------------------------

    def window_size(self) -> tuple[int, int]:
        # Screen size is effectively static within a session; memoize to save an RPC
        # on the warm hierarchy hot path (PRD G1 < 150 ms).
        if self._winsize is None:
            ws = self._call("window_size")
            self._winsize = (int(ws[0]), int(ws[1]))
        return self._winsize

    def dump_hierarchy(self, compressed: bool = False) -> str:
        return str(self._call("dump_hierarchy", compressed=compressed))

    def screenshot(self) -> ScreenImage:
        img = self._call("screenshot")  # PIL.Image by default
        return ScreenImage.from_pil(img)

    def current_app(self) -> dict[str, str]:
        info = self._call("app_current") or {}
        return {
            "package": info.get("package", ""),
            "activity": info.get("activity", "") or "",
        }

    def app_version(self, package: str) -> str | None:
        try:
            info = self._d.app_info(package)
        except Exception:  # pragma: no cover - best effort / app not installed
            return None
        if isinstance(info, dict) and info.get("versionName"):
            return str(info["versionName"])
        return None

    # -- input -------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        self._call("click", x, y)

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        self._call("long_click", x, y, duration_ms / 1000.0)

    def send_text(self, text: str, *, clear: bool = True) -> None:
        # Prefer accessibility ACTION_SET_TEXT on the focused field: it replaces the
        # content in one shot with no input injection, so it works on Android 14+ where
        # u2's injectKeyEvent-based clear hits NoSuchMethodException
        # (InputManager.getInstance removed). Fall back to the IME send_keys path.
        if clear:
            try:
                self._d(focused=True).set_text(text)
                return
            except Exception as exc:
                logger.debug("set_text on focused field failed (%s); using send_keys", exc)
        self._call("send_keys", text, clear=False)

    def clear_text(self) -> None:
        try:
            self._d(focused=True).set_text("")
            return
        except Exception as exc:
            logger.debug("set_text('') clear failed (%s); using clear_text", exc)
        self._call("clear_text")

    def send_ime_action(self, action: str = "search") -> None:
        try:
            self._call("send_action", action)
        except Exception:  # pragma: no cover - fall back to ENTER
            self._call("press", "enter")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._call("swipe", x1, y1, x2, y2, duration_ms / 1000.0)

    def press(self, key: str) -> None:
        k = key.strip()
        if k.upper().startswith("KEYCODE_"):
            self._call("press", k.upper())
            return
        mapped = _PRESS_ALIASES.get(k.lower())
        self._call("press", mapped if mapped is not None else k)

    # -- selectors ---------------------------------------------------------

    def _selector_kwargs(self, text: str, match: MatchMode, ignore_case: bool, field: str) -> dict:
        if match is MatchMode.regex:
            pattern = f"(?i){text}" if ignore_case else text
            return {f"{field}Matches": pattern}
        if ignore_case:
            esc = re.escape(text)
            pattern = f"(?i){esc}" if match is MatchMode.exact else f"(?i).*{esc}.*"
            return {f"{field}Matches": pattern}
        if match is MatchMode.exact:
            return {field: text}
        return {f"{field}Contains": text}

    def _resource_id_kwargs(self, text: str, match: MatchMode) -> dict:
        """Selector for a resource-id. A full ``pkg:id/name`` matches exactly; a bare tail
        (``containerChatDetail``) matches any id ending in ``:id/<tail>``."""
        if ":id/" in text:
            return {"resourceId": text} if match is not MatchMode.contains else {
                "resourceIdMatches": f".*{re.escape(text)}.*"
            }
        if match is MatchMode.contains:
            return {"resourceIdMatches": f".*{re.escape(text)}.*"}
        return {"resourceIdMatches": f".*:id/{re.escape(text)}$"}

    def _fields_for(self, by: str) -> list[str]:
        return {"id": ["resourceId"], "desc": ["description"]}.get(by, ["text", "description"])

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        match = MatchMode(match)
        for field in self._fields_for(by):
            kwargs = (
                self._resource_id_kwargs(text, match)
                if field == "resourceId"
                else self._selector_kwargs(text, match, ignore_case, field)
            )
            try:
                el = self._d(**kwargs)
                exists = el.exists
                if exists() if callable(exists) else exists:
                    info = el.info
                    bounds = _bounds_from_info(info() if callable(info) else info)
                    if bounds is not None:
                        return bounds
            except Exception as exc:  # pragma: no cover - bad regex etc.
                logger.debug("selector %s failed: %s", kwargs, exc)
        return None

    def wait_idle(self, timeout_ms: int = 5000) -> None:
        try:
            self._d.jsonrpc.waitForIdle(timeout_ms)
        except Exception:  # pragma: no cover - best effort
            time.sleep(0.1)

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        if activity is not None:
            self._call("app_start", package, activity=activity)
        else:
            self._call("app_start", package)

    def stop_app(self, package: str) -> None:
        self._call("app_stop", package)

    def clear_app(self, package: str) -> None:
        # u2 wraps `pm clear` — resets to a fresh-install state (Maestro clearState).
        self._call("app_clear", package)

    def grant_permissions(self, package: str) -> None:
        # Best-effort grant of all declared dangerous permissions (camera/mic/…).
        self._call("app_auto_grant_permissions", package)

    def double_click(self, x: int, y: int) -> None:
        self._call("double_click", x, y)

    def hide_keyboard(self) -> None:
        # KEYCODE_ESCAPE (111) dismisses the IME without finishing the Activity; fall
        # back to back if the shell path is unavailable on this OEM/build.
        try:
            self._d.shell("input keyevent 111")
            return
        except Exception as exc:
            logger.debug("hide_keyboard escape failed (%s); using back", exc)
        self.press("back")

    def set_clipboard(self, text: str) -> None:
        self._call("set_clipboard", text)

    def get_clipboard(self) -> str:
        clip = self._d.clipboard
        return str(clip) if clip is not None else ""

    def paste(self) -> None:
        self._d.shell("input keyevent 279")  # KEYCODE_PASTE

    def set_location(self, lat: float, lon: float) -> None:
        # Emulator console accepts longitude first; physical devices may need mock-location.
        try:
            subprocess.run(
                ["adb", "-s", self.serial, "emu", "geo", "fix", str(lon), str(lat)],
                check=True,
                capture_output=True,
                timeout=15,
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        try:
            self._d.shell(f"cmd location set-location {lat} {lon}")
        except Exception as exc:
            raise DeviceError(
                f"could not set location to {lat},{lon}",
                hint="On emulators use an AVD; on physical devices enable mock locations.",
            ) from exc

    def set_orientation(self, mode: str) -> None:
        key = _ORIENTATION_ALIASES.get(mode.lower())
        if key is None:
            raise DeviceError(
                f"unknown orientation {mode!r}",
                hint="Use portrait|landscape|left|right|natural (or n|l|r|u).",
            )
        self._call("set_orientation", key)

    def get_orientation(self) -> str:
        try:
            return str(self._d.orientation or "unknown")
        except Exception:  # pragma: no cover - best effort
            return "unknown"

    def set_airplane_mode(self, enabled: bool) -> None:
        val = "1" if enabled else "0"
        self._d.shell(f"settings put global airplane_mode_on {val}")
        state = "true" if enabled else "false"
        self._d.shell(
            f"am broadcast -a android.intent.action.AIRPLANE_MODE --ez state {state}"
        )

    def get_airplane_mode(self) -> bool | None:
        try:
            raw = str(self._d.shell("settings get global airplane_mode_on")).strip()
            if raw in ("0", "1"):
                return raw == "1"
        except Exception:  # pragma: no cover
            return None
        return None

    def add_media(self, local_path: str, *, remote_dir: str = "/sdcard/DCIM/Camera") -> str:
        src = Path(local_path).expanduser().resolve()
        if not src.is_file():
            raise DeviceError(f"media file not found: {src}")
        remote = f"{remote_dir.rstrip('/')}/{src.name}"
        self._call("push", str(src), remote)
        self._d.shell(
            f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{remote}"
        )
        return remote

    def start_recording(self, remote_path: str = "/sdcard/aua_recording.mp4") -> str:
        if self._recording_remote is not None:
            raise DeviceError(
                "a screen recording is already in progress",
                hint="Run `aua record stop` before starting another.",
            )
        self._recording_remote = remote_path
        self._recording_proc = subprocess.Popen(  # noqa: S603
            ["adb", "-s", self.serial, "shell", "screenrecord", remote_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return remote_path

    def stop_recording(self, local_path: str) -> str:
        if self._recording_remote is None:
            raise DeviceError(
                "no screen recording in progress", hint="Start one with `aua record start`."
            )
        remote = self._recording_remote
        proc = self._recording_proc
        self._recording_remote = None
        self._recording_proc = None
        try:
            subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "shell", "pkill", "-l", "2", "screenrecord"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        finally:
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=5)
        dest = Path(local_path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._call("pull", remote, str(dest))
        with contextlib.suppress(Exception):
            self._d.shell(f"rm {remote}")
        return str(dest)

    def set_clock(self, timestamp_ms: int) -> None:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
        stamp = dt.strftime("%m%d%H%M%Y.%S")
        try:
            self._d.shell(f"date {stamp}")
        except Exception as exc:
            raise DeviceError(
                "could not set device clock",
                hint="Usually works on emulators/rooted devices only (Maestro travel).",
            ) from exc

    def erase_chars(self, count: int) -> None:
        for _ in range(max(0, count)):
            self._d.shell("input keyevent 67")  # KEYCODE_DEL

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        # Prefer package-scoped VIEW intent to skip the system "Open with…" chooser.
        if package:
            from shlex import quote

            self._d.shell(
                f"am start -a android.intent.action.VIEW -d {quote(uri)} -p {quote(package)}"
            )
            return
        # u2's open_url shells out to `am start -a VIEW -d <uri>`; jumps straight to a
        # deeplinked screen (or triggers app actions like setting feature flags).
        self._call("open_url", uri)

    def query_uri_handlers(self, uri: str) -> list[str]:
        from shlex import quote

        try:
            out = self._d.shell(f"cmd package resolve-activity --brief -a android.intent.action.VIEW -d {quote(uri)}")
            text = out if isinstance(out, str) else getattr(out, "output", str(out))
        except Exception:
            return []
        pkgs: list[str] = []
        for line in str(text).splitlines():
            line = line.strip()
            if "/" in line and not line.startswith("priority"):
                pkgs.append(line.split("/", 1)[0])
        return pkgs


# --------------------------------------------------------------------------- factory


def connect(serial: str | None = None) -> Device:
    """Connect to ``serial`` (or the only/first device). Raises DeviceError clearly."""
    if serial is None:
        devices = list_devices()
        online = [d for d in devices if d.state == "device"]
        if not online:
            raise DeviceError(
                "no device found",
                hint="Start an emulator or attach a device; run `aua devices` to list them.",
            )
        if len(online) > 1:
            listing = ", ".join(d.serial for d in online)
            raise DeviceError(
                f"multiple devices attached ({listing})",
                hint="Pass --serial <id> to choose one.",
            )
        serial = online[0].serial
    return Uiautomator2Device(serial)


def list_devices() -> list[DeviceInfo]:
    """List attached devices via adbutils (a uiautomator2 dependency)."""
    try:
        import adbutils
    except ImportError as exc:  # pragma: no cover
        raise DeviceError(
            "adbutils not available",
            hint="Install with the uiautomator2 dependency, or ensure adb is on PATH.",
        ) from exc
    out: list[DeviceInfo] = []
    try:
        for dev in adbutils.adb.device_list():
            state = "device"
            model: str | None = None
            version: str | None = None
            try:
                model = dev.prop.model
                version = dev.getprop("ro.build.version.release") or None
            except Exception:  # pragma: no cover - offline device
                state = "offline"
            out.append(
                DeviceInfo(serial=dev.serial, model=model, android_version=version, state=state)
            )
    except Exception as exc:
        raise DeviceError(
            f"could not list devices: {exc}",
            hint="Is the adb server running? Try `adb devices`.",
        ) from exc
    return out
