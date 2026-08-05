"""Device plumbing: a thin, mockable wrapper over ``uiautomator2`` (PRD §6, §14).

``Device`` is an ABC defining the exact surface the engine/daemon/CLI use. The real
``Uiautomator2Device`` lazy-imports ``uiautomator2`` (so the core CLI works with the
library absent), keeps a warm connection, and reconnects once on a transient error
before failing. Tests supply a fake conforming to the same ABC — no device required.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import DeviceError, UsageError
from .providers.base import Bounds, ScreenImage
from .schema import DeviceInfo, MatchMode

logger = logging.getLogger("android_ui_analyser.device")

# `screenrecord` rejects a --time-limit above this and exits at once, so asking for
# more records nothing. The platform also stops at this limit on its own.
_SCREENRECORD_MAX_S = 180

_RECONNECT_WARN_WINDOW_S = 30.0
_last_reconnect_warn: dict[str, float] = {}


def _warn_reconnect(name: str, exc: Exception) -> None:
    """One line per op per window: a detached device retries forever and floods the log."""
    now = time.monotonic()
    if now - _last_reconnect_warn.get(name, 0.0) < _RECONNECT_WARN_WINDOW_S:
        return
    _last_reconnect_warn[name] = now
    logger.warning("device op '%s' failed (%s); reconnecting once", name, exc)


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

    def screencap_png(self) -> ScreenImage:
        """Raw PNG via ``adb exec-out screencap -p`` when available; else :meth:`screenshot`.

        Capture loops prefer this path — it skips the u2 Java instrumentation round-trip.
        """
        return self.screenshot()

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
    # `rid` is the spelling of the resource-id everywhere else in the vocabulary — the
    # `--rid` flag, the selector dict key, `_SELECTOR_FIELDS` — so `--by rid` is what a
    # runner reaches for. It used to fall through to the text/desc default, meaning
    # `wait --for containerLogin --by rid` searched the *label* for "containerLogin",
    # found nothing and timed out on a screen where that container was plainly present.
    _BY_FIELDS = {
        "id": ["resourceId"],
        "rid": ["resourceId"],
        "desc": ["description"],
        "text": ["text", "description"],
    }

    def _fields_for(self, by: str) -> list[str]:
        """Fields to search for a ``by`` token — refusing one it does not know.

        This used to `.get(..., ["text", "description"])`, so an unrecognised token quietly
        became a label search. That is how `--by rid` (before it was a synonym) spent a full
        15s timeout looking for the literal string "containerLogin" in the *text* of a screen
        where that container was plainly present, and then reported the screen wrong.

        A degrade is worse than a refusal here because of what the caller concludes. A refusal
        says "you held it wrong" and costs one turn; a silent text search says "the element is
        not on this screen", which is a claim about the *product* — and it arrives with a
        screenshot that contradicts it, so it also costs the reader's trust in the tool. The
        `tap` family already refused an unknown `--by` loudly (`cli.py:_selector`), so the two
        surfaces disagreed about the same token, which is the divergence being closed.
        """
        fields = self._BY_FIELDS.get((by or "text").lower())
        if fields is None:
            raise UsageError(
                f"unknown selector field 'by={by}'",
                hint="Choose one of: " + ", ".join(sorted(self._BY_FIELDS)) + ".",
            )
        return fields

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

        ``by``: ``"text"`` searches text + content-desc (default); ``"id"`` (or ``"rid"``)
        matches the resource-id (a bare tail like ``containerDetail`` matches the id's suffix)
        — this can find containers that the parsed element list prunes; ``"desc"`` is
        content-desc only. Resolve it through :meth:`_fields_for`, which is where the one
        vocabulary lives: an implementation that maps ``by`` itself will drift from the real
        device, and a test double that drifts makes a fix look green when it is not.
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

    def focused_text(self) -> str | None:
        """Current value of the focused text field, or None when it cannot be read.

        None means "unknown", never "empty" — the caller must not treat an unreadable
        field as an unchanged one.
        """
        return None

    def instance_token(self) -> str | None:
        """Identity of *this boot* of this device, or None when it cannot be read.

        A serial is not an identity: console ports are recycled, so `emulator-5554` may be
        three different devices in an hour. Anything cached against a serial needs to know
        when the thing behind it was replaced.
        """
        return None

    def set_clock(self, timestamp_ms: int) -> None:
        raise DeviceError("clock travel requires a real device")

    def get_clock_ms(self) -> int | None:
        """Current device wall-clock as unix ms, or None if unreadable."""
        return None

    def utc_offset_minutes(self) -> int | None:
        """The device's UTC offset in minutes, or None if unreadable.

        ``-v threadtime`` timestamps are device-local with no zone, so this is what turns
        one into an epoch. Only the degraded post-filter path needs it.
        """
        return None

    def erase_chars(self, count: int) -> None:
        """Delete *count* characters before the caret in the focused field."""
        for _ in range(max(0, count)):
            self.press("KEYCODE_DEL")

    def shell(self, command: str) -> str:
        """Run a shell command on the device; return combined stdout text."""
        raise DeviceError("shell requires a real device")

    def read_app_file(self, package: str, path: str) -> bytes:
        """Read one private app file as bytes through Android ``run-as``."""
        raise DeviceError("private app file reads require a real device")

    def write_app_file(self, package: str, path: str, data: bytes) -> None:
        """Atomically replace one private app file through Android ``run-as``."""
        raise DeviceError("private app file writes require a real device")

    def remove_app_files(self, package: str, paths: list[str]) -> None:
        """Remove private app files through Android ``run-as``."""
        raise DeviceError("private app file removal requires a real device")

    def a11y_action(self, x: int, y: int, action: str) -> None:
        """Perform an accessibility action on the node at *(x, y)*."""
        raise DeviceError("a11y action requires a real device")

    def set_http_proxy(self, host_port: str | None) -> None:
        """Set or clear the global HTTP proxy (``host:port`` or ``None`` to clear)."""
        raise DeviceError("http proxy requires a real device")

    def adb_reverse(self, device_port: int, host_port: int) -> None:
        """``adb reverse tcp:device_port tcp:host_port``."""
        raise DeviceError("adb reverse requires a real device")

    def adb_reverse_remove(self, device_port: int) -> None:
        """Remove a reverse port mapping (no-op if absent)."""
        return None

    def logcat(self, *, since_ms: int | None = None, dump: bool = True) -> str:
        """Dump (or clear) logcat. ``dump=False`` clears the buffer and returns ``""``.

        *since_ms* is a **device**-clock epoch and an implementation MUST apply it — the
        caller does no time filtering of its own. Raise :class:`DeviceError` if
        unavailable.
        """
        raise DeviceError("logcat requires a real device")

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

# Same alias set as keycode names, for the `input keyevent` path. Measured on a headless
# emulator over one uiautomator2 connection: `shell("input keyevent KEYCODE_BACK")` takes
# ~103 ms, `press("back")` ~1125 ms for the identical keystroke. A `key back` is the second
# half of most navigation steps, so that second was pure loop tax.
_KEYCODE_NAMES = {
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "enter": "KEYCODE_ENTER",
    "recents": "KEYCODE_APP_SWITCH",
    "recent": "KEYCODE_APP_SWITCH",
    "menu": "KEYCODE_MENU",
    "search": "KEYCODE_SEARCH",
    "power": "KEYCODE_POWER",
    "volume_up": "KEYCODE_VOLUME_UP",
    "volume_down": "KEYCODE_VOLUME_DOWN",
    "del": "KEYCODE_DEL",
    "delete": "KEYCODE_DEL",
    "backspace": "KEYCODE_DEL",
    "paste": "KEYCODE_PASTE",
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
            _warn_reconnect(name, exc)
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

    def screencap_png(self) -> ScreenImage:
        """Prefer ``adb exec-out screencap -p``; fall back to u2 on any failure."""
        try:
            proc = subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "exec-out", "screencap", "-p"],
                check=True,
                capture_output=True,
                timeout=8,
            )
            raw = proc.stdout
            if raw and raw[:8] == b"\x89PNG\r\n\x1a\n":
                return ScreenImage(raw)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("adb screencap failed (%s); using u2 screenshot", exc)
        return self.screenshot()

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

    def focused_text(self) -> str | None:
        """Read back the focused field's value, so `input` can check its own effect."""
        try:
            value = self._d(focused=True).get_text()
        except Exception as exc:
            logger.debug("could not read the focused field (%s)", exc)
            return None
        return value if isinstance(value, str) else None

    def send_text(self, text: str, *, clear: bool = True) -> None:
        """Type *text* into the focused field, preferring fast one-shot paths.

        Order:
        1. Accessibility ``ACTION_SET_TEXT`` via u2 ``set_text`` (replace) — fastest when it works.
        2. Clipboard set + ``KEYCODE_PASTE`` (previous clipboard restored) — fast for long strings
           and when ``set_text`` is unavailable; used for append when ``clear=False``.
        3. IME ``send_keys`` — last resort (slow / char-by-char on some devices).
        """
        if clear:
            try:
                self._d(focused=True).set_text(text)
                return
            except Exception as exc:
                logger.debug("set_text on focused field failed (%s); trying clipboard paste", exc)
            if self._paste_via_clipboard(text, clear=True):
                return
            try:
                self.clear_text()
            except Exception as exc:
                logger.debug("clear before send_keys failed (%s)", exc)
            self._call("send_keys", text, clear=False)
            return

        # Append: set_text would replace, so skip straight to paste / keys.
        if self._paste_via_clipboard(text, clear=False):
            return
        self._call("send_keys", text, clear=False)

    def _paste_via_clipboard(self, text: str, *, clear: bool) -> bool:
        """Set clipboard → optional clear → paste → restore clipboard. Returns False on failure."""
        previous: str | None
        try:
            previous = self.get_clipboard()
        except Exception:
            previous = None
        try:
            if clear:
                try:
                    self.clear_text()
                except Exception as exc:
                    logger.debug("clear_text before paste failed (%s)", exc)
            self.set_clipboard(text)
            self.paste()
            return True
        except Exception as exc:
            logger.debug("clipboard paste path failed (%s)", exc)
            return False
        finally:
            if previous is not None:
                with contextlib.suppress(Exception):
                    self.set_clipboard(previous)

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
        """Inject a keystroke, preferring `input keyevent` — see :data:`_KEYCODE_NAMES`.

        Falls back to uiautomator2's own press for keys with no keycode name, so an exotic
        key still works (slowly) rather than failing.
        """
        k = key.strip()
        keycode = k.upper() if k.upper().startswith("KEYCODE_") else _KEYCODE_NAMES.get(k.lower())
        if keycode is not None:
            try:
                self.shell(f"input keyevent {keycode}")
                return
            except DeviceError:
                pass  # a device that refuses `input` still deserves the keystroke
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
        (``containerDetail``) matches any id ending in ``:id/<tail>``."""
        if ":id/" in text:
            return {"resourceId": text} if match is not MatchMode.contains else {
                "resourceIdMatches": f".*{re.escape(text)}.*"
            }
        if match is MatchMode.contains:
            return {"resourceIdMatches": f".*{re.escape(text)}.*"}
        return {"resourceIdMatches": f".*:id/{re.escape(text)}$"}

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
        if activity is None:
            self._call("app_start", package)
            return
        # `am start` prints its refusal (commonly a non-exported Activity) and exits 0, and
        # uiautomator2's app_start discards that output — so the caller learned nothing and
        # went on to drive a screen that was never opened. Run it here and read the answer.
        component = f"{package}/{activity}"
        out = self.shell(f"am start -n {component}")
        lowered = out.lower()
        if "permission denial" in lowered or "securityexception" in lowered or "error:" in lowered:
            first = next(
                (line.strip() for line in out.splitlines() if line.strip().startswith(("Error", "java."))),
                out.strip().splitlines()[-1] if out.strip() else "am start failed",
            )
            raise DeviceError(
                f"am start refused {component}: {first}",
                hint="Pick an exported launcher Activity, or drop --activity to let the "
                "platform resolve one (`aua app launch <pkg>`).",
            )

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

    def _recording_state_path(self) -> Path:
        """Where an in-flight recording is noted on disk, keyed by serial.

        Recording state cannot live only in process memory. The normal usage is
        ``aua record start`` … many separate ``aua`` calls … ``aua record stop``, so an
        in-memory handle makes that sequence fail with "no screen recording in progress"
        and silently loses the evidence — worse than never recording. Honour
        ``AUA_CACHE__DIR`` so per-worker caches stay isolated, exactly as the proxy does.
        """
        base = os.environ.get("AUA_CACHE__DIR") or "~/.cache/android-ui-analyser"
        d = Path(base).expanduser() / "recordings"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.serial.replace(':', '_')}.json"

    def instance_token(self) -> str | None:
        """The kernel's boot id — a fresh UUID for every boot, so it names this instance.

        Measured at ~10ms on the emulator pool, which is why the caller can afford to ask
        once per invocation rather than inventing a heuristic about elapsed time.
        """
        with contextlib.suppress(Exception):
            token = self.shell("cat /proc/sys/kernel/random/boot_id").strip()
            # Guard against a shell that answers with an error page rather than failing.
            if 30 <= len(token) <= 40 and token.count("-") == 4:
                return token
        return None

    def _live_recording(self) -> tuple[bool, str | None]:
        """Ask the *device* whether a ``screenrecord`` is running: ``(observed, remote)``.

        The on-disk handle is not authoritative and cannot be, because it is addressed by
        two things that both move underneath it:

        - **the cache directory.** The handle lives under ``AUA_CACHE__DIR``, so a worker
          that exports a per-worker cache dir *between* its ``record start`` and its
          ``record stop`` looks somewhere else at stop time and finds nothing — reporting
          "no screen recording in progress" for a recording that plainly ran, and losing
          the video rather than truncating it.
        - **the serial.** Console ports are recycled within minutes, so a handle orphaned
          by a killed worker makes the *next* worker's first ``record start`` fail with
          "already in progress" before it has started anything.

        ``ps`` describes this boot of this device, so it answers both. The first element
        distinguishes "nothing is recording" from "``ps`` could not be read" — without
        that, an unreadable ``ps`` would look like an idle device and we would clear a live
        handle and start a second recorder over the top of the first.
        """
        try:
            text = self.shell("ps -A -o ARGS")
        except DeviceError:
            return False, None
        for line in text.splitlines():
            if "screenrecord" not in line:
                continue
            # `screenrecord [options] <remote.mp4>` — the destination is the trailing
            # argument. Match on shape rather than position so an added option (or a
            # recorder started by something other than aua) is still recognised.
            tail = line.split()[-1] if line.split() else ""
            return True, tail if tail.startswith("/") else None
        return True, None

    def start_recording(
        self, remote_path: str = "/sdcard/aua_recording.mp4", *, time_limit_s: int = 1800
    ) -> str:
        state = self._recording_state_path()
        observed, live_remote = self._live_recording()
        if self._recording_remote is not None or live_remote is not None:
            raise DeviceError(
                "a screen recording is already in progress",
                hint="Run `aua record stop <path>` before starting another.",
            )
        if state.exists():
            if not observed:
                # `ps` unreadable: fall back to trusting the handle, as before. Refusing on
                # an unverifiable handle risks a duplicate recorder; that is the safer error.
                raise DeviceError(
                    "a screen recording is already in progress",
                    hint="Run `aua record stop <path>` before starting another.",
                )
            # The device says nothing is recording, so this handle is an orphan from a
            # previous instance on this serial. Drop it rather than refusing the caller.
            logger.info("clearing an orphaned recording handle at %s", state)
            with contextlib.suppress(Exception):
                state.unlink()
        self._recording_remote = remote_path
        # `screenrecord` refuses anything above 180s - "Time limit 1800s outside acceptable
        # range [1,180]" - and exits immediately. An earlier attempt to defeat the platform's
        # silent 180s truncation by asking for 1800 therefore recorded *nothing at all*, while
        # `record start` still reported ok. Two lanes of a sweep lost their video to it. Clamp
        # to what the platform accepts, and say what the effective limit was.
        limit = max(1, min(int(time_limit_s), _SCREENRECORD_MAX_S))
        self._recording_proc = subprocess.Popen(  # noqa: S603
            [
                "adb", "-s", self.serial, "shell", "screenrecord",
                "--time-limit", str(limit), remote_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Confirm it is actually recording before claiming success. `screenrecord` writes its
        # container header straight away, so the file appearing is the signal; without this
        # check any future rejection is silent again, which is the whole defect.
        if not self._wait_for_remote_file(remote_path, timeout_s=4.0):
            self._recording_remote = None
            with contextlib.suppress(Exception):
                self._recording_proc.kill()
            self._recording_proc = None
            raise DeviceError(
                f"screen recording did not start (no {remote_path} on device)",
                hint=(
                    "Check `adb -s <serial> shell screenrecord --time-limit "
                    f"{limit} {remote_path}` by hand; some emulator images and headless GPU "
                    "modes refuse to encode. Screenshots remain a valid evidence fallback."
                ),
            )
        with contextlib.suppress(Exception):
            state.write_text(json.dumps({"remote": remote_path, "time_limit_s": limit}))
        return remote_path

    def _wait_for_remote_file(self, remote_path: str, *, timeout_s: float) -> bool:
        """True once *remote_path* exists on the device, within *timeout_s*."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                out = self._d.shell(f"ls -l {remote_path} 2>/dev/null")
                text = out if isinstance(out, str) else getattr(out, "output", "")
                if remote_path.split("/")[-1] in str(text):
                    return True
            time.sleep(0.25)
        return False

    def stop_recording(self, local_path: str) -> str:
        state_file = self._recording_state_path()
        if self._recording_remote is None and state_file.exists():
            # Started by an earlier `aua` invocation — recover the handle from disk.
            with contextlib.suppress(Exception):
                self._recording_remote = json.loads(state_file.read_text()).get("remote")
        if self._recording_remote is None:
            # No handle here — but the recording may still be running, started under a
            # different cache directory. The device knows where it is writing.
            _, live_remote = self._live_recording()
            if live_remote is not None:
                logger.info("recovered a running recording from the device: %s", live_remote)
                self._recording_remote = live_remote
        if self._recording_remote is None:
            raise DeviceError(
                "no screen recording in progress", hint="Start one with `aua record start`."
            )
        remote = self._recording_remote
        proc = self._recording_proc
        self._recording_remote = None
        self._recording_proc = None
        with contextlib.suppress(Exception):
            state_file.unlink()
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
        # `adb pull` of a remote path that does not exist creates a *directory* at the
        # destination and returns success, so a failed recording used to land as an empty
        # `evidence.mp4/` folder while `record stop` reported ok. Check before pulling, and
        # check what arrived, because an empty directory named .mp4 is worse than an error:
        # the agent files it as evidence and nobody looks inside.
        if not self._wait_for_remote_file(remote, timeout_s=2.0):
            raise DeviceError(
                f"no recording found on the device at {remote}",
                hint=(
                    "The recording never started, or the platform stopped it. Nothing was "
                    "written locally. Screenshots remain a valid evidence fallback."
                ),
            )
        self._call("pull", remote, str(dest))
        if dest.is_dir() or not dest.is_file() or dest.stat().st_size == 0:
            with contextlib.suppress(Exception):
                if dest.is_dir():
                    dest.rmdir()
                elif dest.is_file():
                    dest.unlink()
            raise DeviceError(
                f"pulling the recording produced nothing usable at {dest}",
                hint="Try `adb -s <serial> pull` by hand; the device file may be zero-length.",
            )
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

    def get_clock_ms(self) -> int | None:
        try:
            out = self._d.shell("date +%s%3N")
            text = out if isinstance(out, str) else getattr(out, "output", str(out))
            digits = "".join(c for c in str(text).strip() if c.isdigit())
            if len(digits) >= 10:
                # %s%3N → seconds + millis; fall back to seconds*1000
                if len(digits) >= 13:
                    return int(digits[:13])
                return int(digits[:10]) * 1000
        except Exception:
            return None
        return None

    def utc_offset_minutes(self) -> int | None:
        try:
            out = self._d.shell("date +%z")
            text = out if isinstance(out, str) else getattr(out, "output", str(out))
            m = re.search(r"([+-])(\d{2})(\d{2})", str(text).strip())
            if not m:
                return None
            sign = -1 if m.group(1) == "-" else 1
            return sign * (int(m.group(2)) * 60 + int(m.group(3)))
        except Exception:
            return None

    def erase_chars(self, count: int) -> None:
        for _ in range(max(0, count)):
            self._d.shell("input keyevent 67")  # KEYCODE_DEL

    def shell(self, command: str) -> str:
        try:
            out = self._d.shell(command)
        except Exception as exc:
            raise DeviceError(
                f"shell failed: {exc}",
                hint="Check the device is online (`adb devices`) and the command is valid.",
            ) from exc
        return out if isinstance(out, str) else str(getattr(out, "output", out) or "")

    def read_app_file(self, package: str, path: str) -> bytes:
        try:
            result = subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "exec-out", "run-as", package, "cat", path],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(f"could not read {package}/{path}: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise DeviceError(f"could not read {package}/{path}: {detail or 'adb failed'}")
        return result.stdout

    def write_app_file(self, package: str, path: str, data: bytes) -> None:
        from shlex import quote

        target = Path(path)
        temporary = str(target.with_name(f".aua-{target.name}-{os.getpid()}-{time.time_ns()}"))
        script = f"umask 077; cat > {quote(temporary)}"
        try:
            result = subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "exec-in", "run-as", package, "sh", "-c", script],
                input=data,
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(f"could not write {package}/{path}: {exc}") from exc
        if result.returncode != 0:
            with contextlib.suppress(Exception):
                self.remove_app_files(package, [temporary])
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise DeviceError(f"could not write {package}/{path}: {detail or 'adb failed'}")
        try:
            moved = subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "shell", "run-as", package, "mv", temporary, path],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            with contextlib.suppress(Exception):
                self.remove_app_files(package, [temporary])
            raise DeviceError(f"could not replace {package}/{path}: {exc}") from exc
        if moved.returncode != 0:
            with contextlib.suppress(Exception):
                self.remove_app_files(package, [temporary])
            detail = (moved.stderr or moved.stdout).strip()
            raise DeviceError(f"could not replace {package}/{path}: {detail or 'adb failed'}")

    def remove_app_files(self, package: str, paths: list[str]) -> None:
        if not paths:
            return
        try:
            result = subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "shell", "run-as", package, "rm", "-f", *paths],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(f"could not remove private files for {package}: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise DeviceError(
                f"could not remove private files for {package}: {detail or 'adb failed'}"
            )

    def a11y_action(self, x: int, y: int, action: str) -> None:
        """Resolve the smallest hierarchy node containing *(x, y)* and perform *action*."""
        action_u = (action or "").strip().upper().replace("-", "_")
        node = self._node_at(x, y)
        if node is None:
            raise DeviceError(
                f"no accessibility node at ({x}, {y})",
                hint="Re-analyze and pass a visible element id / selector.",
            )
        obj = self._u2_object_for(node)
        try:
            if action_u in ("CLICK", "ACTION_CLICK"):
                if obj is not None:
                    obj.click()
                else:
                    self.click(x, y)
                return
            if action_u in ("LONG_CLICK", "ACTION_LONG_CLICK", "LONG_PRESS"):
                if obj is not None:
                    obj.long_click()
                else:
                    self.long_click(x, y)
                return
            if action_u in ("SCROLL_FORWARD", "ACTION_SCROLL_FORWARD", "FORWARD"):
                if obj is None:
                    raise DeviceError("SCROLL_FORWARD needs a selectable scrollable node")
                obj.scroll.forward()
                return
            if action_u in ("SCROLL_BACKWARD", "ACTION_SCROLL_BACKWARD", "BACKWARD"):
                if obj is None:
                    raise DeviceError("SCROLL_BACKWARD needs a selectable scrollable node")
                obj.scroll.backward()
                return
            if action_u in ("EXPAND", "ACTION_EXPAND"):
                if obj is None:
                    raise DeviceError(f"{action_u} needs a selectable node")
                with contextlib.suppress(Exception):
                    obj.expand()
                    return
                raise DeviceError(f"action {action_u} unsupported on this node")
            if action_u in ("COLLAPSE", "ACTION_COLLAPSE"):
                if obj is None:
                    raise DeviceError(f"{action_u} needs a selectable node")
                with contextlib.suppress(Exception):
                    obj.collapse()
                    return
                raise DeviceError(f"action {action_u} unsupported on this node")
            if action_u in ("DISMISS", "ACTION_DISMISS"):
                if obj is None:
                    raise DeviceError(f"{action_u} needs a selectable node")
                with contextlib.suppress(Exception):
                    obj.dismiss()
                    return
                raise DeviceError(f"action {action_u} unsupported on this node")
            if action_u in ("SET_TEXT", "ACTION_SET_TEXT"):
                raise DeviceError(
                    "SET_TEXT via a11y action needs a value — use `aua input` instead",
                )
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(
                f"a11y action {action_u} failed: {exc}",
                hint="The node may not support that accessibility action.",
            ) from exc
        raise DeviceError(
            f"unsupported a11y action {action!r}",
            hint="Supported: CLICK, LONG_CLICK, SCROLL_FORWARD, SCROLL_BACKWARD, "
            "EXPAND, COLLAPSE, DISMISS.",
        )

    def _node_at(self, x: int, y: int) -> dict[str, str] | None:
        """Smallest on-screen node whose bounds contain *(x, y)*."""
        xml = self.dump_hierarchy()
        best: dict[str, str] | None = None
        best_area: int | None = None
        for m in re.finditer(r"<node\b([^>]*)/?>", xml):
            attrs = m.group(1)
            bm = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', attrs)
            if not bm:
                continue
            x1, y1, x2, y2 = (int(bm.group(i)) for i in range(1, 5))
            if not (x1 <= x < x2 and y1 <= y < y2):
                continue
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if best_area is not None and area >= best_area:
                continue
            info: dict[str, str] = {
                "bounds": f"[{x1},{y1}][{x2},{y2}]",
            }
            for key in ("resource-id", "text", "content-desc", "class", "package"):
                am = re.search(rf'{key}="([^"]*)"', attrs)
                if am and am.group(1):
                    info[key] = am.group(1)
            best, best_area = info, area
        return best

    def _u2_object_for(self, node: dict[str, str]) -> Any | None:
        rid = node.get("resource-id")
        if rid:
            obj = self._d(resourceId=rid)
            if obj.exists:
                return obj
        text = node.get("text")
        if text:
            obj = self._d(text=text)
            if obj.exists:
                return obj
        desc = node.get("content-desc")
        if desc:
            obj = self._d(description=desc)
            if obj.exists:
                return obj
        return None

    def set_http_proxy(self, host_port: str | None) -> None:
        if host_port:
            self.shell(f"settings put global http_proxy {host_port}")
        else:
            # `:0` clears the proxy on modern Android; delete is a fallback.
            self.shell("settings put global http_proxy :0")
            with contextlib.suppress(Exception):
                self.shell("settings delete global http_proxy")

    def adb_reverse(self, device_port: int, host_port: int) -> None:
        try:
            subprocess.run(  # noqa: S603
                [
                    "adb",
                    "-s",
                    self.serial,
                    "reverse",
                    f"tcp:{device_port}",
                    f"tcp:{host_port}",
                ],
                check=True,
                capture_output=True,
                timeout=15,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(
                f"adb reverse tcp:{device_port} tcp:{host_port} failed",
                hint="Check `adb` is on PATH and the device is reachable.",
            ) from exc

    def adb_reverse_remove(self, device_port: int) -> None:
        with contextlib.suppress(Exception):
            subprocess.run(  # noqa: S603
                ["adb", "-s", self.serial, "reverse", "--remove", f"tcp:{device_port}"],
                check=False,
                capture_output=True,
                timeout=15,
            )

    def _logcat_dump(self, args: list[str]) -> str:
        proc = subprocess.run(  # noqa: S603
            ["adb", "-s", self.serial, "logcat", "-d", "-v", "threadtime", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout or ""

    def logcat(self, *, since_ms: int | None = None, dump: bool = True) -> str:
        if not dump:
            try:
                subprocess.run(  # noqa: S603
                    ["adb", "-s", self.serial, "logcat", "-c"],
                    check=True,
                    capture_output=True,
                    timeout=15,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                raise DeviceError(
                    "could not clear logcat buffer",
                    hint="Check `adb` is on PATH and the device is reachable.",
                ) from exc
            return ""
        # logcat's own `-T <sec.nsec>` compares against the same clock that stamped the
        # lines, which no amount of host-side parsing can match.
        native = [] if since_ms is None else ["-T", f"{since_ms // 1000}.{since_ms % 1000:03d}000000"]
        try:
            return self._logcat_dump(native)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            if not native:
                raise DeviceError(
                    "could not dump logcat",
                    hint="Check `adb` is on PATH and the device is reachable.",
                ) from exc
        try:
            raw = self._logcat_dump([])
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(
                "could not dump logcat",
                hint="Check `adb` is on PATH and the device is reachable.",
            ) from exc
        from .logcat import filter_logcat

        return "\n".join(
            filter_logcat(
                raw, since_ms=since_ms, tz_offset_minutes=self.utc_offset_minutes() or 0
            )
        )

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
    else:
        _require_attached(serial)
    return Uiautomator2Device(serial)


_MISSING_SERIAL_GRACE_S = 1.0


def _require_attached(serial: str) -> None:
    """Fail immediately when *serial* is not attached at all.

    Without this, a pinned serial goes straight to uiautomator2, which retries its way to
    the same conclusion in about 29 seconds - to report something adb answers in 20 ms. It
    matters because losing a device mid-run is routine, not exotic: an emulator hits its
    idle-stop, a CI job runs `pkill -f qemu-system`, a parallel worker tears down the wrong
    instance. An agent then pays half a minute per call, several calls in a row, before it
    can even see that the device is what broke.

    Only a *completely absent* serial fails fast. One that is attached but not yet ready
    (``offline``, ``unauthorized``, still booting) is handed to uiautomator2 as before, so
    the ordinary start-up race keeps working - a booting emulator appears in adb long before
    it is usable. The single short retry covers the narrow window where a just-launched
    emulator has not yet bound its console port.
    """
    for attempt in (0, 1):
        try:
            attached = {d.serial: d.state for d in list_devices()}
        except DeviceError:
            return  # adb itself is the problem; let the normal path report it
        if serial in attached:
            return
        if attempt == 0:
            time.sleep(_MISSING_SERIAL_GRACE_S)

    listing = ", ".join(f"{s} ({st})" for s, st in sorted(attached.items())) or "none"
    raise DeviceError(
        f"device {serial!r} is not attached (attached: {listing})",
        hint=(
            "The device went away or was never started. If a worker owns it, restart it with "
            "`aua emulator start --avd <name>`; `aua devices` lists what adb can see. A "
            "headless emulator also auto-stops after its --idle-stop window."
        ),
    )


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
