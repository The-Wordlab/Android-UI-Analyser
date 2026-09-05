"""Platform-neutral runtime contract for one connected automation target.

Adapters return a ``TargetRuntime`` from :meth:`PlatformAdapter.connect`.  The contract uses
semantic UI operations and AUA's canonical screen-coordinate space; native transports, selector
grammars, key codes, paths, and diagnostic formats belong to the platform implementation.

``android_ui_analyser.device.Device`` remains as a compatibility name for the historical Android
runtime surface.  New adapters should inherit this class directly.
"""

from __future__ import annotations

import time
from abc import ABC
from collections.abc import Sequence

from ..errors import DeviceError
from ..providers.base import Bounds, ScreenImage
from ..schema import AppContext, MatchMode, ShellResult
from .geometry import DisplayGeometry


class TargetRuntime(ABC):
    """Semantic operations on one connected target.

    Every operation is capability-gated. Conservative defaults make a runtime instantiable with
    only the surfaces its adapter declares, while structural validation rejects a declaration
    whose methods were left on these stubs. This keeps a discovery-only or hierarchy-only plugin
    from implementing unrelated input, lifecycle, or Android behavior merely to satisfy an ABC.

    ``target_id`` is the one required identity. ``serial`` is a read-only compatibility
    projection for historical Android-facing callers; a new runtime implements only
    ``target_id`` and does not need to model Android serials.
    """

    target_id: str
    """Stable adapter-local identifier required at the adapter connection boundary."""

    @property
    def serial(self) -> str:
        """Legacy Android spelling of :attr:`target_id`.

        Shared compatibility surfaces still emit ``serial`` in some payloads. Keeping the alias
        here lets an API-v1 plugin expose only the neutral identity while those callers migrate;
        platform implementations must not treat it as a second identifier.
        """

        return self.target_id

    def display_geometry(self) -> DisplayGeometry:
        """Coordinate transform for this frame, identity for existing runtimes.

        Bounds returned to shared code and coordinates accepted by every input operation are
        canonical screenshot pixels. A runtime backed by native logical points overrides this
        method and performs the inverse transform inside its input methods.
        """

        return DisplayGeometry.identity(*self.window_size())

    # -- capability-gated capture -----------------------------------------

    def window_size(self) -> tuple[int, int]:
        """Size of AUA's canonical screen-coordinate space."""

        raise DeviceError("screen geometry is unsupported by this target runtime")

    def dump_hierarchy(self, compressed: bool = False) -> str:
        """Return the platform-native accessibility hierarchy."""

        raise DeviceError("UI hierarchy capture is unsupported by this target runtime")

    def screenshot(self) -> ScreenImage:
        """Capture a frame whose pixels use the canonical coordinate space."""

        raise DeviceError("screenshots are unsupported by this target runtime")

    def screencap_png(self) -> ScreenImage:
        """Capture a lossless frame when available, otherwise use :meth:`screenshot`."""

        return self.screenshot()

    def current_app(self) -> AppContext:
        """Return the platform-neutral foreground application context."""

        raise DeviceError("foreground app context is unsupported by this target runtime")

    # -- capability-gated input ------------------------------------------

    def click(self, x: int, y: int) -> None:
        """Tap canonical screenshot-pixel coordinates."""

        raise DeviceError("tap input is unsupported by this target runtime")

    def click_once(self, x: int, y: int) -> None:
        """Send one non-retrying tap, or fail if delivery cannot be bounded to one attempt."""

        raise DeviceError(
            "this target runtime cannot guarantee a single-attempt tap",
            code="single_tap_unsupported",
            hint="Use the default hold gesture, or add click_once support to the adapter.",
        )

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        raise DeviceError("long-press input is unsupported by this target runtime")

    def touch_down(self, x: int, y: int) -> None:
        """Begin a touch that remains held until :meth:`touch_up`."""

        raise DeviceError("held touch gestures are unsupported by this target runtime")

    def touch_up(self, x: int, y: int) -> None:
        """Release a touch begun with :meth:`touch_down`."""

        raise DeviceError("held touch gestures are unsupported by this target runtime")

    def send_text(self, text: str, *, clear: bool = True) -> None:
        raise DeviceError("text input is unsupported by this target runtime")

    def clear_text(self) -> None:
        raise DeviceError("text clearing is unsupported by this target runtime")

    def send_ime_action(self, action: str = "search") -> None:
        raise DeviceError("input submission is unsupported by this target runtime")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        raise DeviceError("swipe input is unsupported by this target runtime")

    def press(self, key: str) -> None:
        """Send a platform-neutral key name such as ``back``, ``home``, or ``enter``."""

        raise DeviceError("key input is unsupported by this target runtime")

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        """Return the first matching canonical bounds, or ``None``."""

        raise DeviceError("semantic text search is unsupported by this target runtime")

    # -- optional metadata ------------------------------------------------

    def app_version(self, app_id: str) -> str | None:
        """Best-effort application version used for memory freshness."""

        return None

    def device_locale(self) -> str | None:
        """Best-effort target UI locale as a BCP-47 tag."""

        return None

    def instance_token(self) -> str | None:
        """Identity of this target boot/runtime instance, if available."""

        return None

    # -- composed baseline helpers ---------------------------------------

    def input_text(
        self,
        x: int,
        y: int,
        text: str,
        *,
        clear: bool = True,
        submit: bool = False,
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

    def wait_idle(self, timeout_ms: int = 5000) -> None:
        """Wait for target UI work to settle when the platform can observe it."""

        return None

    def double_click(self, x: int, y: int) -> None:
        self.click(x, y)
        time.sleep(0.05)
        self.click(x, y)

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
        width, height = self.window_size()
        for _ in range(max_swipes):
            self.swipe(width // 2, int(height * 0.7), width // 2, int(height * 0.3), 300)
            found = self.find_text(query, match=match, ignore_case=ignore_case, by=by)
            if found is not None:
                return found
        return None

    # -- optional application operations ---------------------------------

    def launch_app(self, app_id: str, *, activity: str | None = None) -> None:
        raise DeviceError("application launch is unsupported by this target runtime")

    def launcher_activities(self, app_id: str) -> list[str]:
        """Known launch surfaces for an application; empty means unknown or unsupported."""

        return []

    def stop_app(self, app_id: str) -> None:
        raise DeviceError("application stop is unsupported by this target runtime")

    def clear_app(self, app_id: str) -> str | None:
        raise DeviceError("application data clearing is unsupported by this target runtime")

    def grant_permissions(self, app_id: str) -> None:
        raise DeviceError("permission granting is unsupported by this target runtime")

    def granted_permissions(self, app_id: str) -> list[str]:
        """Runtime permissions currently granted to *app_id*.

        This read is part of the lifecycle capability because a permission grant is durable
        device state.  Shared code snapshots it before granting so teardown can restore the
        exact prior set after an interrupted run.
        """

        raise DeviceError("permission inspection is unsupported by this target runtime")

    def restore_permissions(self, app_id: str, granted: Sequence[str]) -> None:
        """Restore *app_id* to an earlier set of granted runtime permissions."""

        raise DeviceError("permission restoration is unsupported by this target runtime")

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        raise DeviceError("opening links is unsupported by this target runtime")

    def query_uri_handlers(self, uri: str) -> list[str]:
        return []

    # -- optional UI/environment operations -------------------------------

    def hide_keyboard(self) -> None:
        raise DeviceError("keyboard dismissal is unsupported by this target runtime")

    def keyboard_visible(self) -> bool | None:
        return None

    def set_clipboard(self, text: str) -> None:
        raise DeviceError("clipboard access is unsupported by this target runtime")

    def get_clipboard(self) -> str:
        raise DeviceError("clipboard access is unsupported by this target runtime")

    def paste(self) -> None:
        raise DeviceError("clipboard paste is unsupported by this target runtime")

    def set_location(self, lat: float, lon: float) -> None:
        raise DeviceError("location changes are unsupported by this target runtime")

    def set_orientation(self, mode: str) -> None:
        raise DeviceError("orientation changes are unsupported by this target runtime")

    def get_orientation(self) -> str:
        return "unknown"

    def set_airplane_mode(self, enabled: bool) -> None:
        raise DeviceError("airplane mode is unsupported by this target runtime")

    def get_airplane_mode(self) -> bool | None:
        return None

    def media_directory(self, requested: str | None = None) -> str:
        """Resolve a caller override or this platform's default media destination."""

        raise DeviceError("media injection is unsupported by this target runtime")

    def add_media(self, local_path: str, *, remote_dir: str) -> str:
        raise DeviceError("media injection is unsupported by this target runtime")

    def remove_added_media(self, local_path: str, *, remote_dir: str) -> None:
        """Remove the artifact that :meth:`add_media` would create for these arguments."""

        raise DeviceError("media removal is unsupported by this target runtime")

    def recording_destination(self, requested: str | None = None) -> str:
        """Resolve a caller override or this platform's default recording destination."""

        raise DeviceError("screen recording is unsupported by this target runtime")

    def start_recording(self, remote_path: str) -> str:
        raise DeviceError("screen recording is unsupported by this target runtime")

    def active_recording(self) -> str | None:
        """Target path of a running recording, or ``None`` when the target is proven idle."""

        raise DeviceError("screen recording status is unsupported by this target runtime")

    def stop_recording(self, local_path: str) -> str:
        raise DeviceError("screen recording is unsupported by this target runtime")

    def discard_recording(self, remote_path: str) -> None:
        """Stop and delete an unfinished recording without installing it as evidence."""

        raise DeviceError("screen recording cleanup is unsupported by this target runtime")

    def focused_text(self) -> str | None:
        return None

    def set_clock(self, timestamp_ms: int) -> None:
        raise DeviceError("clock changes are unsupported by this target runtime")

    def get_clock_ms(self) -> int | None:
        return None

    def utc_offset_minutes(self) -> int | None:
        return None

    def erase_chars(self, count: int) -> None:
        for _ in range(max(0, count)):
            self.press("delete")

    def run_read_only_shell(self, argv: list[str], *, timeout_s: float = 30.0) -> ShellResult:
        raise DeviceError(
            "structured diagnostics are unsupported by this target runtime",
            code="unsupported_capability",
        )

    def read_app_file(self, app_id: str, path: str) -> bytes:
        raise DeviceError("private application files are unsupported by this target runtime")

    def write_app_file(self, app_id: str, path: str, data: bytes) -> None:
        raise DeviceError("private application files are unsupported by this target runtime")

    def remove_app_files(self, app_id: str, paths: list[str]) -> None:
        raise DeviceError("private application files are unsupported by this target runtime")

    def a11y_action(self, x: int, y: int, action: str) -> None:
        raise DeviceError("accessibility actions are unsupported by this target runtime")

    def set_http_proxy(self, host_port: str | None) -> None:
        raise DeviceError("HTTP proxy changes are unsupported by this target runtime")

    def reverse_port(self, target_port: int, host_port: int) -> None:
        raise DeviceError("reverse port forwarding is unsupported by this target runtime")

    def remove_reverse_port(self, target_port: int) -> None:
        raise DeviceError("reverse port forwarding is unsupported by this target runtime")

    def close(self) -> None:
        """Release the runtime and any target-side automation agent."""

        return None


__all__ = ["TargetRuntime"]
