"""Device plumbing: a thin, mockable wrapper over ``uiautomator2`` (PRD §6, §14).

``Device`` is an ABC defining the exact surface the engine/daemon/CLI use. The real
``Uiautomator2Device`` lazy-imports ``uiautomator2`` (so the core CLI works with the
library absent), keeps a warm connection, and reconnects once on a transient error
before failing. Tests supply a fake conforming to the same ABC — no device required.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
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
        self, text: str, *, match: MatchMode | str = MatchMode.contains, ignore_case: bool = False
    ) -> Bounds | None:
        """Cheap selector locate — return the box of the first match, or None."""

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
    ) -> Bounds | None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            found = self.find_text(text, match=match, ignore_case=ignore_case)
            if found is not None:
                return found
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)

    def wait_idle(self, timeout_ms: int = 5000) -> None:  # overridden by real device
        return None

    def scroll_to(
        self,
        query: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        max_swipes: int = 8,
    ) -> Bounds | None:
        found = self.find_text(query, match=match, ignore_case=ignore_case)
        if found is not None:
            return found
        w, h = self.window_size()
        for _ in range(max_swipes):
            self.swipe(w // 2, int(h * 0.7), w // 2, int(h * 0.3), 300)
            found = self.find_text(query, match=match, ignore_case=ignore_case)
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
}


class Uiautomator2Device(Device):
    """Warm ``uiautomator2`` connection with single auto-reconnect."""

    def __init__(self, serial: str, settle_wait: float = 0.0) -> None:
        self.serial = serial
        self._settle = settle_wait
        self._d: Any = None
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
            try:
                self._d.settings["wait_timeout"] = 5.0
            except Exception:  # pragma: no cover - older u2
                pass
        except Exception as exc:
            raise DeviceError(
                f"could not connect to device '{self.serial}': {exc}",
                hint="Run `aua devices` and check the emulator/phone is reachable via adb.",
            ) from exc

    def _guard(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a uiautomator2 op; reconnect once on a transient failure (PRD §14)."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("device op failed (%s); reconnecting once", exc)
            try:
                self._connect()
                # rebind the bound method to the fresh connection
                if hasattr(fn, "__name__") and fn.__self__ is not None:  # type: ignore[attr-defined]
                    fn = getattr(self._d, fn.__name__)
                return fn(*args, **kwargs)
            except Exception as exc2:
                raise DeviceError(
                    f"device operation failed after reconnect: {exc2}",
                    hint="Check the device is still attached (`aua devices`).",
                ) from exc2

    # -- capture -----------------------------------------------------------

    def window_size(self) -> tuple[int, int]:
        w, h = self._guard(self._d.window_size)
        return int(w), int(h)

    def dump_hierarchy(self, compressed: bool = False) -> str:
        return str(self._guard(self._d.dump_hierarchy, compressed=compressed))

    def screenshot(self) -> ScreenImage:
        img = self._guard(self._d.screenshot)  # PIL.Image by default
        return ScreenImage.from_pil(img)

    def current_app(self) -> dict[str, str]:
        info = self._guard(self._d.app_current) or {}
        return {
            "package": info.get("package", ""),
            "activity": info.get("activity", "") or "",
        }

    # -- input -------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        self._guard(self._d.click, x, y)

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        self._guard(self._d.long_click, x, y, duration_ms / 1000.0)

    def send_text(self, text: str, *, clear: bool = True) -> None:
        self._guard(self._d.send_keys, text, clear=clear)

    def clear_text(self) -> None:
        self._guard(self._d.clear_text)

    def send_ime_action(self, action: str = "search") -> None:
        try:
            self._guard(self._d.send_action, action)
        except Exception:  # pragma: no cover - fall back to ENTER
            self._guard(self._d.press, "enter")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._guard(self._d.swipe, x1, y1, x2, y2, duration_ms / 1000.0)

    def press(self, key: str) -> None:
        k = key.strip()
        if k.upper().startswith("KEYCODE_"):
            self._guard(self._d.press, k.upper())
            return
        mapped = _PRESS_ALIASES.get(k.lower())
        if mapped is None:
            # allow raw keycode names/numbers
            self._guard(self._d.press, k)
        else:
            self._guard(self._d.press, mapped)

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

    def find_text(
        self, text: str, *, match: MatchMode | str = MatchMode.contains, ignore_case: bool = False
    ) -> Bounds | None:
        match = MatchMode(match)
        for field in ("text", "description"):
            kwargs = self._selector_kwargs(text, match, ignore_case, field)
            try:
                el = self._d(**kwargs)
                if self._guard(el.exists):
                    info = self._guard(el.info)
                    bounds = _bounds_from_info(info)
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
