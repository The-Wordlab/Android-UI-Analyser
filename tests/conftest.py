"""Shared, device-less test scaffolding (PRD §0 environment note, §13.1).

Provides:
- ``FakeDevice`` — a :class:`Device` returning fixture XML / synthetic screenshots and
  recording every action call (so tests can assert taps/inputs happened).
- ``make_config`` / ``make_engine`` — build a :class:`Config` / :class:`Engine` wired to
  a fake device, no phone required.
- stub-provider + chain builders for the fallback-chain runner tests (AC4).
- image helpers for merge/annotate tests (AC10).
- a ``dummy`` provider of each kind, registered purely in this file, to prove a provider
  is selectable by config alone (STEP 2 / open-closed requirement).
"""

from __future__ import annotations

import io
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import (
    Availability,
    Bounds,
    ChainSpec,
    DetBox,
    DetectionProvider,
    GroundingProvider,
    OcrProvider,
    PlannerDecision,
    PlannerProvider,
    Point,
    Provider,
    ScreenImage,
    TextBox,
)
from android_ui_analyser.providers.registry import (
    ProviderFactory,
    register_detection,
    register_grounding,
    register_ocr,
)
from android_ui_analyser.schema import MatchMode

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- images


def make_png(
    width: int = 200,
    height: int = 400,
    color: tuple[int, int, int] = (240, 240, 240),
    boxes: list[tuple[Bounds, tuple[int, int, int]]] | None = None,
) -> bytes:
    """A solid-colour PNG with optional filled rectangles, as raw bytes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color)
    if boxes:
        draw = ImageDraw.Draw(img)
        for bounds, fill in boxes:
            draw.rectangle(bounds, fill=fill)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_screen_image(width: int = 200, height: int = 400, **kw: Any) -> ScreenImage:
    return ScreenImage(make_png(width, height, **kw), width=width, height=height)


# --------------------------------------------------------------------------- device


class FakeDevice(Device):
    """In-memory device for tests; records actions, returns canned perception."""

    def __init__(
        self,
        *,
        hierarchy_xml: str = '<hierarchy rotation="0"></hierarchy>',
        width: int = 1080,
        height: int = 2400,
        package: str = "com.test.app",
        activity: str = ".MainActivity",
        text_index: dict[str, Bounds] | None = None,
        resource_index: dict[str, Bounds] | None = None,
        screenshot_bytes: bytes | None = None,
        screenshots: list[bytes] | None = None,
        app_version: str | None = None,
        serial: str = "fake-emulator-5554",
        clock_skew_ms: int = 0,
        utc_offset: int = 0,
        prefs: dict[str, dict[str, str]] | None = None,
        app_files: dict[str, bytes] | None = None,
        run_as_error: str | None = None,
        network_preference: str = "wifi",
    ) -> None:
        self.serial = serial
        # shared_prefs XML the app "owns": filename → {key: value}, served over `run-as`
        # exactly as Android writes it (so the real parser is under test, not a stub).
        self.prefs = {k: dict(v) for k, v in (prefs or {}).items()}
        self.app_files = dict(app_files or {})
        self.run_as_error = run_as_error
        self._xml = hierarchy_xml
        self._w = width
        self._h = height
        self._pkg = package
        self._act = activity
        # MAIN/LAUNCHER classes the fake package "declares"; more than one models a dev build
        # with a Dev Tools icon beside the product entry. Empty = the platform resolves it.
        self._text_index = text_index or {}
        # Resource-id → bounds (for by="id" lookups; keys may be a tail or full id).
        self._resource_index = resource_index or {}
        self._png = screenshot_bytes or make_png(width, height)
        # An optional stream of screenshots (for wait --for-stable tests); the last frame
        # repeats once exhausted.
        self._stream = list(screenshots) if screenshots else None
        self._stream_i = 0
        self._app_version = app_version
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.hierarchy_calls = 0
        self.screenshot_calls = 0
        self._clipboard = ""
        self._orientation = "n"
        self._airplane = False
        self._wifi_enabled = True
        self._mobile_data_enabled = True
        self._network_preference = network_preference
        self._location: tuple[float, float] | None = None
        self._recording: str | None = None
        self._logcat_lines: list[str] = []
        self._logcat_cleared = False
        # host - device, so a positive skew means the host runs ahead (the emulator case).
        self.clock_skew_ms = clock_skew_ms
        self.utc_offset = utc_offset
        self._clock_ms: int | None = None
        # MAIN+LAUNCHER Activities returned by ``launcher_activities`` (empty = unknown).
        self._launcher_activities: list[str] = []

    # capture
    def window_size(self) -> tuple[int, int]:
        return self._w, self._h

    def dump_hierarchy(self, compressed: bool = False) -> str:
        self.hierarchy_calls += 1
        return self._xml

    def screenshot(self) -> ScreenImage:
        self.screenshot_calls += 1
        if self._stream:
            png = self._stream[min(self._stream_i, len(self._stream) - 1)]
            self._stream_i += 1
            return ScreenImage(png, width=self._w, height=self._h)
        return ScreenImage(self._png, width=self._w, height=self._h)

    def current_app(self) -> dict[str, str]:
        return {"package": self._pkg, "activity": self._act}

    def app_version(self, package: str) -> str | None:
        return self._app_version

    # input primitives (recorded)
    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", (x, y)))

    def click_once(self, x: int, y: int) -> None:
        self.calls.append(("click_once", (x, y)))

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        self.calls.append(("long_click", (x, y, duration_ms)))

    def touch_down(self, x: int, y: int) -> None:
        self.calls.append(("touch_down", (x, y)))

    def touch_up(self, x: int, y: int) -> None:
        self.calls.append(("touch_up", (x, y)))

    def send_text(self, text: str, *, clear: bool = True) -> None:
        self.calls.append(("send_text", (text, clear)))

    def clear_text(self) -> None:
        self.calls.append(("clear_text", ()))

    def send_ime_action(self, action: str = "search") -> None:
        self.calls.append(("send_ime_action", (action,)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.calls.append(("swipe", (x1, y1, x2, y2, duration_ms)))

    def press(self, key: str) -> None:
        self.calls.append(("press", (key,)))

    def launcher_activities(self, package: str) -> list[str]:
        self.calls.append(("launcher_activities", (package,)))
        return list(self._launcher_activities)

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        self.calls.append(("launch_app", (package,) if activity is None else (package, activity)))
        # A launch fronts the app. Model it, or `current_app` keeps naming the old package and
        # callers that verify arrival see a launch that never happened.
        self._pkg = package
        if activity is not None:
            self._act = activity

    def stop_app(self, package: str) -> None:
        self.calls.append(("stop_app", (package,)))

    def clear_app(self, package: str) -> None:
        self.calls.append(("clear_app", (package,)))

    def grant_permissions(self, package: str) -> None:
        self.calls.append(("grant_permissions", (package,)))

    def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", (x, y)))

    def hide_keyboard(self) -> None:
        self.calls.append(("hide_keyboard", ()))

    def set_clipboard(self, text: str) -> None:
        self._clipboard = text
        self.calls.append(("set_clipboard", (text,)))

    def get_clipboard(self) -> str:
        self.calls.append(("get_clipboard", ()))
        return self._clipboard

    def paste(self) -> None:
        self.calls.append(("paste", ()))

    def set_location(self, lat: float, lon: float) -> None:
        self._location = (lat, lon)
        self.calls.append(("set_location", (lat, lon)))

    def set_orientation(self, mode: str) -> None:
        self._orientation = mode
        self.calls.append(("set_orientation", (mode,)))

    def get_orientation(self) -> str:
        self.calls.append(("get_orientation", ()))
        return self._orientation

    def set_airplane_mode(self, enabled: bool) -> None:
        self._airplane = enabled
        self.calls.append(("set_airplane_mode", (enabled,)))

    def get_airplane_mode(self) -> bool | None:
        self.calls.append(("get_airplane_mode", ()))
        return self._airplane

    def add_media(self, local_path: str, *, remote_dir: str = "/sdcard/DCIM/Camera") -> str:
        from pathlib import Path

        name = Path(local_path).name
        remote = f"{remote_dir.rstrip('/')}/{name}"
        self.calls.append(("add_media", (local_path, remote_dir)))
        return remote

    def start_recording(self, remote_path: str = "/sdcard/aua_recording.mp4") -> str:
        self._recording = remote_path
        self.calls.append(("start_recording", (remote_path,)))
        return remote_path

    def stop_recording(self, local_path: str) -> str:
        self.calls.append(("stop_recording", (local_path,)))
        self._recording = None
        return local_path

    def set_clock(self, timestamp_ms: int) -> None:
        self.calls.append(("set_clock", (timestamp_ms,)))
        self._clock_ms = timestamp_ms

    def get_clock_ms(self) -> int | None:
        """The device wall clock — host time shifted by ``clock_skew_ms`` unless pinned.

        Real emulators run seconds away from their host, which is precisely what logcat
        windows have to be computed against, so the fake models the offset rather than a
        shared clock.
        """
        if self._clock_ms is not None:
            return self._clock_ms
        return int(time.time() * 1000) - self.clock_skew_ms

    def utc_offset_minutes(self) -> int | None:
        return self.utc_offset

    def advance_clock(self, ms: int) -> None:
        """Move the DEVICE clock forward without moving the host's.

        Lets a fake interaction consume time the way an adb round-trip does, so "when the
        app logged" and "when the interaction returned" are distinguishable instants.
        """
        self.clock_skew_ms -= ms

    def log_now(self, tag: str = "Test", msg: str = "hello", *, offset_ms: int = 0) -> str:
        """Append a line stamped off the DEVICE clock in device-local time, as apps do."""
        device_ms = (self.get_clock_ms() or 0) + offset_ms
        local = datetime.fromtimestamp(device_ms / 1000.0 + self.utc_offset * 60, tz=UTC)
        line = (
            f"{local.month:02d}-{local.day:02d} {local.hour:02d}:{local.minute:02d}:"
            f"{local.second:02d}.{local.microsecond // 1000:03d}  1234  5678 I {tag}: {msg}"
        )
        self._logcat_lines.append(line)
        return line

    def logcat(self, *, since_ms: int | None = None, dump: bool = True) -> str:
        self.calls.append(("logcat", (since_ms, dump)))
        if not dump:
            self._logcat_cleared = True
            self._logcat_lines = []
            return ""
        from android_ui_analyser.logcat import filter_logcat

        raw = "\n".join(self._logcat_lines)
        if since_ms is None:
            return raw
        # Models `logcat -T <device epoch>`: the device compares against its own clock.
        return "\n".join(filter_logcat(raw, since_ms=since_ms, tz_offset_minutes=self.utc_offset))

    def erase_chars(self, count: int) -> None:
        self.calls.append(("erase_chars", (count,)))

    def prefs_xml(self, name: str) -> str:
        """One shared_prefs file, in Android's own layout."""
        entries = []
        for key, value in self.prefs.get(name, {}).items():
            if value in ("true", "false"):
                entries.append(f'    <boolean name="{key}" value="{value}" />')
            else:
                entries.append(f'    <string name="{key}">{value}</string>')
        body = "\n".join(entries)
        return f"<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n<map>\n{body}\n</map>"

    def _run_as(self, command: str) -> str:
        """Serve `run-as <pkg> ls|grep|cat` against :attr:`prefs` (or refuse like Android)."""
        import re as _re
        import shlex

        if self.run_as_error:
            return self.run_as_error
        argv = shlex.split(command)[2:]
        verb, args = argv[0], argv[1:]
        if verb == "ls":
            if args and args[-1].rstrip("/").endswith("databases"):
                rows = ["total 0"]
                for path, data in sorted(self.app_files.items()):
                    if path.startswith("databases/") and "/" not in path[len("databases/") :]:
                        name = path.rsplit("/", 1)[-1]
                        rows.append(f"-rw------- 1 u0_a1 u0_a1 {len(data)} 2026-01-01 00:00 {name}")
                return "\n".join(rows)
            return "\n".join(sorted(self.prefs))
        names = [a.rsplit("/", 1)[-1] for a in args if a.endswith(".xml")]
        if verb == "grep":
            pattern = args[1]
            directory = args[2].rsplit("/", 1)[0]
            return "\n".join(
                f"{directory}/{n}"
                for n in names
                if n in self.prefs and _re.search(pattern, self.prefs_xml(n))
            )
        if verb == "cat":
            return "\n".join(self.prefs_xml(n) for n in names if n in self.prefs)
        return ""

    def shell(self, command: str) -> str:
        self.calls.append(("shell", (command,)))
        if command.startswith("run-as "):
            return self._run_as(command)
        if not hasattr(self, "_settings"):
            self._settings = {
                ("global", "window_animation_scale"): "1",
                ("global", "transition_animation_scale"): "1",
                ("global", "animator_duration_scale"): "1",
                ("global", "hide_error_dialogs"): "1",
                ("secure", "anr_show_background"): "0",
                ("global", "always_finish_activities"): "0",
                ("global", "http_proxy"): ":0",
                ("global", "wifi_on"): "1" if self._wifi_enabled else "0",
                ("global", "mobile_data"): "1" if self._mobile_data_enabled else "0",
            }
        parts = command.split()
        if command == "pm has-feature android.hardware.wifi":
            return "true"
        if command == "pm has-feature android.hardware.telephony":
            return "true"
        if command == "svc wifi enable":
            self._wifi_enabled = True
            self._settings[("global", "wifi_on")] = "1"
            return ""
        if command == "svc wifi disable":
            self._wifi_enabled = False
            self._settings[("global", "wifi_on")] = "0"
            return ""
        if command == "svc data enable":
            self._mobile_data_enabled = True
            self._settings[("global", "mobile_data")] = "1"
            return ""
        if command == "svc data disable":
            self._mobile_data_enabled = False
            self._settings[("global", "mobile_data")] = "0"
            return ""
        if command == "dumpsys connectivity":
            if self._wifi_enabled and (
                self._network_preference == "wifi" or not self._mobile_data_enabled
            ):
                transport = "WIFI"
            elif self._mobile_data_enabled and not self._airplane:
                transport = "CELLULAR"
            elif self._wifi_enabled:
                transport = "WIFI"
            else:
                return "Active default network: none\n"
            return (
                "Active default network: 100\n"
                "Current Networks:\n"
                f"  NetworkAgentInfo{{network{{100}} nc{{[ Transports: {transport} "
                "Capabilities: INTERNET&VALIDATED&TRUSTED ]}}\n"
            )
        if len(parts) >= 4 and parts[0] == "settings" and parts[1] == "get":
            return str(self._settings.get((parts[2], parts[3]), "null"))
        if len(parts) >= 5 and parts[0] == "settings" and parts[1] == "put":
            self._settings[(parts[2], parts[3])] = parts[4]
            return ""
        if len(parts) >= 4 and parts[0] == "settings" and parts[1] == "delete":
            self._settings.pop((parts[2], parts[3]), None)
            return ""
        return ""

    @staticmethod
    def _prefs_file_of(path: str) -> str | None:
        """The prefs filename *path* addresses, or None when it is some other app file."""
        if path.startswith("shared_prefs/") and path.endswith(".xml"):
            return path[len("shared_prefs/") :]
        return None

    def read_app_file(self, package: str, path: str) -> bytes:
        self.calls.append(("read_app_file", (package, path)))
        if path in self.app_files:
            return self.app_files[path]
        # A prefs file seeded through `prefs=` exists on the "device" without anyone having
        # written bytes for it, exactly as one the app itself wrote does.
        name = self._prefs_file_of(path)
        if name is not None and name in self.prefs:
            return self.prefs_xml(name).encode("utf-8")
        raise OSError(f"missing app file: {path}")

    def write_app_file(self, package: str, path: str, data: bytes) -> None:
        self.calls.append(("write_app_file", (package, path, len(data))))
        self.app_files[path] = data
        name = self._prefs_file_of(path)
        if name is not None:
            from android_ui_analyser.flags import parse_all_prefs

            # A written prefs file must also become visible to the `run-as ls|grep|cat` reads,
            # or a test could "write" a preference the read path never sees. Parsed with the
            # production parser, not a stub: a writer emitting XML the read-back cannot
            # understand has to fail the test rather than pass it.
            self.prefs[name] = {
                key: values[-1]
                for key, values in parse_all_prefs(data.decode("utf-8", "replace")).items()
                if values
            }

    def remove_app_files(self, package: str, paths: list[str]) -> None:
        self.calls.append(("remove_app_files", (package, tuple(paths))))
        for path in paths:
            self.app_files.pop(path, None)
            name = self._prefs_file_of(path)
            if name is not None:
                self.prefs.pop(name, None)

    def a11y_action(self, x: int, y: int, action: str) -> None:
        self.calls.append(("a11y_action", (x, y, action)))

    def set_http_proxy(self, host_port: str | None, *, exclusion_list=None) -> None:
        self.calls.append(("set_http_proxy", (host_port,)))
        if not hasattr(self, "_settings"):
            self.shell("settings get global http_proxy")  # ensure dict
        if host_port:
            self._settings[("global", "http_proxy")] = host_port
        else:
            self._settings[("global", "http_proxy")] = ":0"
        key = ("global", "global_http_proxy_exclusion_list")
        if exclusion_list:
            self._settings[key] = ",".join(exclusion_list)
        else:
            self._settings.pop(key, None)

    def get_http_proxy(self) -> str | None:
        raw = str(self.shell("settings get global http_proxy")).strip()
        return None if raw.lower() in ("", "null", ":0", "0") else raw

    def get_proxy_exclusion_list(self) -> list[str]:
        raw = str(self.shell("settings get global global_http_proxy_exclusion_list")).strip()
        if raw.lower() in ("", "null"):
            return []
        return [h.strip() for h in raw.split(",") if h.strip()]

    def set_wifi_enabled(self, enabled: bool) -> None:
        self.calls.append(("set_wifi_enabled", (enabled,)))

    def set_mobile_data_enabled(self, enabled: bool) -> None:
        self.calls.append(("set_mobile_data_enabled", (enabled,)))

    def adb_reverse(self, device_port: int, host_port: int) -> None:
        self.calls.append(("adb_reverse", (device_port, host_port)))
        self._reverses = sorted({*getattr(self, "_reverses", set()), device_port})

    def adb_reverse_remove(self, device_port: int) -> None:
        self.calls.append(("adb_reverse_remove", (device_port,)))
        self._reverses = [p for p in getattr(self, "_reverses", []) if p != device_port]

    def adb_reverse_list(self) -> list[int]:
        return list(getattr(self, "_reverses", []))

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        self.calls.append(("open_link", (uri,) if package is None else (uri, package)))

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        self.calls.append(("find_text", (text, str(match), ignore_case, by)))
        mode = MatchMode(match)
        # Resolve `by` through the shared vocabulary rather than comparing it here. This double
        # used to test `by == "id"` itself, so it kept answering a *text* search for `by="rid"`
        # long after the real device had learned the synonym — a double that disagrees with
        # production is how a fix looks green without being one.
        if self._fields_for(by) == ["resourceId"]:
            # tail-match against the resource index (keys may be a tail or full id)
            for key, bounds in self._resource_index.items():
                tail = key.split("/")[-1]
                if key == text or tail == text or (mode is MatchMode.contains and text in key):
                    return bounds
            return None
        needle = text.lower() if ignore_case else text
        for key, bounds in self._text_index.items():
            hay = key.lower() if ignore_case else key
            if mode is MatchMode.exact and hay == needle:
                return bounds
            if mode is MatchMode.contains and needle in hay:
                return bounds
            if mode is MatchMode.regex:
                import re

                flags = re.IGNORECASE if ignore_case else 0
                if re.search(text, key, flags):
                    return bounds
        return None


# --------------------------------------------------------------------------- config / engine


def make_config(**overrides: Any) -> Config:
    """Config from defaults with shallow section overrides (dicts deep-merge).

    When the autouse ``_aua_isolate_state`` fixture has set ``_AUA_TEST_STATE_DIR``, the
    memory/cache dirs default there so the suite never writes to the real $HOME. Explicit
    ``overrides`` still win (AC13 passes its own memory dir).
    """
    import os

    from android_ui_analyser.config import _deep_merge

    base = Config().model_dump(mode="python")
    state = os.environ.get("_AUA_TEST_STATE_DIR")
    if state:
        base["memory"]["dir"] = str(Path(state) / "memory_home")
        base["cache"]["dir"] = str(Path(state) / "cache")
        base["lease"]["registry_dir"] = str(Path(state) / "lease_registry")
        # Capture is always-on with a real daemon; keep unit tests quiet unless opted in.
        base["capture"]["enabled"] = False
        # Deterministic memory assertions: record on the calling thread.
        base["perf"]["async_memory"] = False
        base["perf"]["prefetch"] = False
        base["perf"]["auto_daemon"] = False
        # Most tests exercise hierarchy semantics, not host OCR. Production defaults to
        # augmentation; focused OCR tests opt in explicitly.
        base["ocr"]["augment_hierarchy"] = False
    # Legacy unit tests commonly use ``cache={"dir": tmp_path}`` as their complete isolated
    # coordination root. Preserve that test-helper shorthand; production Config keeps its
    # host-wide lease registry, while new multi-run tests pass ``lease.registry_dir`` explicitly.
    if "cache" in overrides and "lease" not in overrides:
        cache_override = overrides.get("cache")
        if isinstance(cache_override, dict) and cache_override.get("dir"):
            base["lease"]["registry_dir"] = str(cache_override["dir"])
    merged = _deep_merge(base, overrides) if overrides else base
    return Config.model_validate(merged)


@pytest.fixture(autouse=True)
def _aua_isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ALL persistent state (memory + cache) to a per-test tmp dir.

    Belt-and-suspenders for both config paths: ``make_config`` reads ``_AUA_TEST_STATE_DIR``
    directly, and ``load_config`` (the CLI path) reads the ``AUA_*`` env overrides.
    """
    state = tmp_path / "_aua_state"
    monkeypatch.setenv("_AUA_TEST_STATE_DIR", str(state))
    monkeypatch.setenv("AUA_MEMORY__DIR", str(state / "memory_home"))
    monkeypatch.setenv("AUA_CACHE__DIR", str(state / "cache"))
    monkeypatch.setenv("AUA_LEASE__REGISTRY_DIR", str(state / "lease_registry"))
    # CLI commands run in-process (never reach a stray dev-machine daemon socket).
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setenv("AUA_PERF__ASYNC_MEMORY", "false")
    monkeypatch.setenv("AUA_PERF__PREFETCH", "false")
    monkeypatch.setenv("AUA_PERF__AUTO_DAEMON", "false")
    # Unstyled `--help`: several tests assert an option name appears in the output, and rich
    # splits those names with styling escapes ("--max-back" arriving as bold segments) whenever
    # it believes it is writing to a capable terminal. Whether it believes that depends on the
    # host, which is how three of these passed locally and failed in CI. `TERM=dumb` is the one
    # switch that disables styling outright — `NO_COLOR` drops colour but keeps bold, and a wide
    # `COLUMNS` does not help at all (both verified against `FORCE_COLOR=1`).
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("NO_COLOR", "1")
    # Console-port reservations are the one piece of persistent state no env var here can
    # redirect: `emulator._reservation_dir` is a fixed global path on purpose, because parallel
    # agents must coordinate port allocation process-wide. Shared between tests it is
    # cross-talk — two tests allocating a port at the same moment hand each other 5556 instead
    # of 5554, which is invisible in a single-process run and intermittent under `-n auto`.
    from android_ui_analyser import emulator as emulator_mod

    def _isolated_reservation_dir() -> Path:
        # Created on call, never up front: the real `_reservation_dir` does the same, and a test
        # that never allocates a port must still see an untouched tmp_path — one of them asserts
        # exactly that (`test_a_failed_write_leaves_no_litter`).
        reservations = state / "portlocks"
        reservations.mkdir(parents=True, exist_ok=True)
        return reservations

    monkeypatch.setattr(emulator_mod, "_reservation_dir", _isolated_reservation_dir)

    # Same reasoning, same shape: the undo ledger and the proxy ownership records live at fixed
    # cross-process paths *on purpose* — a parallel agent with its own cache still has to find
    # them — so no `AUA_*` override can redirect them and the suite would write into the
    # developer's real ~/.cache. Verified the hard way: a full run left ledger files and pending
    # undos for serials like `fake-ethernet` in the real home directory.
    from android_ui_analyser import device_ledger as ledger_mod
    from android_ui_analyser import proxy_mock as proxy_mod

    def _isolated_ledger_dir() -> Path:
        ledger = state / "device-state"
        ledger.mkdir(parents=True, exist_ok=True)
        return ledger

    def _isolated_proxy_state_dir() -> Path:
        proxy = state / "proxy-state"
        proxy.mkdir(parents=True, exist_ok=True)
        return proxy

    monkeypatch.setattr(ledger_mod, "ledger_dir", _isolated_ledger_dir)
    monkeypatch.setattr(proxy_mod, "proxy_state_dir", _isolated_proxy_state_dir)
    return state


@pytest.fixture(autouse=True)
def _aua_forget_policy_provider_health() -> Any:
    """Give every test a provider with no history.

    `policy_health` deliberately keeps its rolling per-provider validity window in memory for
    the life of the process, because that is the lifetime of the daemon whose model it
    describes. In a suite that means one test feeding a fixture selector deliberately malformed
    output would condemn the *next* test's identically-named selector.
    """

    from android_ui_analyser import policy_health

    policy_health.registry().reset()
    yield
    policy_health.registry().reset()


@pytest.fixture(autouse=True)
def _aua_never_spawn_a_real_teardown_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the detached per-device watchdog out of the suite.

    ``ensure_watchdog`` launches a real ``start_new_session=True`` process whose whole job is to
    outlive its parent — so a test that merely records a device change left a watchdog running
    after pytest exited, once per test, polling forever for a serial that never existed. A test
    that means to exercise the watchdog calls ``run_watchdog`` directly instead.
    """

    from android_ui_analyser import teardown as teardown_mod

    monkeypatch.setattr(teardown_mod, "ensure_watchdog", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _aua_never_ask_a_real_emulator_who_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `avd_name_of_serial` away from whatever is plugged into the dev machine.

    `_wait_for_serial` cross-checks the AVD name so a port collision cannot make one agent
    drive another's device. That check shells out to `adb -s <serial> emu avd name`, and no
    amount of patching `running_emulators` stops it: a test whose fake device is
    "emulator-5554" asked the *real* emulator-5554 what it was, got a genuine AVD name back,
    and failed the comparison against its own fixture — so the suite passed or failed
    depending on whether a developer happened to have an emulator open.

    None is the honest stand-in: it is exactly what the real call returns for a console that
    will not answer, and the production code already treats that as "keep waiting" rather
    than as a mismatch. A test that wants a specific name overrides this.
    """

    from android_ui_analyser import emulator as emulator_mod

    monkeypatch.setattr(emulator_mod, "avd_name_of_serial", lambda serial: None)


@pytest.fixture(autouse=True)
def _aua_never_kill_a_real_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `adb emu kill` unreachable from the suite.

    `emulator.stop` terminates emulators for real, and a test exercising it only needs the
    device list faked to look plausible — the kill still went out. `test_emulator_stop_guard`
    faked `emulator-5554`/`emulator-5556`, the serials a developer actually runs, and stubbed
    the kill in one test out of four (against a name that did not exist, so it stubbed
    nothing). Every full-suite run killed the live emulators mid-session.

    A test that wants to observe kills patches this same seam itself, which wins over this
    fixture; nothing has to remember to add protection.
    """
    from android_ui_analyser import emulator as emulator_mod

    def _refuse(serial: str) -> None:
        raise AssertionError(
            f"a unit test tried to kill emulator {serial!r} for real; "
            "patch emulator._adb_emu_kill in the test if you meant to observe the call"
        )

    monkeypatch.setattr(emulator_mod, "_adb_emu_kill", _refuse)


def make_engine(
    *,
    config: Config | None = None,
    device: Device | None = None,
    factory: ProviderFactory | None = None,
    **config_overrides: Any,
) -> Engine:
    cfg = config or make_config(**config_overrides)
    dev = device if device is not None else FakeDevice()
    return Engine(cfg, device=dev, factory=factory or ProviderFactory(cfg))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def fake_device() -> FakeDevice:
    return FakeDevice()


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# --------------------------------------------------------------------------- stub providers


class StubOcr(OcrProvider):
    name = "stub_ocr"

    def __init__(
        self,
        *,
        available: bool = True,
        reason: str = "ok",
        result: list[TextBox] | None = None,
        raises: Exception | None = None,
    ) -> None:
        super().__init__()
        self._available = available
        self._reason = reason
        self._result = result if result is not None else []
        self._raises = raises
        self.calls = 0

    def is_available(self) -> Availability:
        return Availability(self._available, self._reason)

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        self.calls += 1
        if self._raises:
            raise self._raises
        return list(self._result)


class StubDetection(DetectionProvider):
    name = "stub_detection"

    def __init__(
        self,
        *,
        available: bool = True,
        reason: str = "ok",
        result: list[DetBox] | None = None,
        raises: Exception | None = None,
    ) -> None:
        super().__init__()
        self._available = available
        self._reason = reason
        self._result = result if result is not None else []
        self._raises = raises
        self.calls = 0

    def is_available(self) -> Availability:
        return Availability(self._available, self._reason)

    def detect(self, image: ScreenImage) -> list[DetBox]:
        self.calls += 1
        if self._raises:
            raise self._raises
        return list(self._result)


class StubGrounding(GroundingProvider):
    name = "stub_grounding"

    def __init__(
        self,
        *,
        available: bool = True,
        reason: str = "ok",
        result: Point | DetBox | None = None,
        raises: Exception | None = None,
    ) -> None:
        super().__init__()
        self._available = available
        self._reason = reason
        self._result = result
        self._raises = raises
        self.calls = 0

    def is_available(self) -> Availability:
        return Availability(self._available, self._reason)

    def locate(self, image: ScreenImage, instruction: str) -> Point | DetBox | None:
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._result


class StubPlanner(PlannerProvider):
    """A scriptable planner for tests.

    Pass ``decisions=[...]`` for a fixed sequence (one per ``decide`` call; the last
    repeats), or ``decide_fn(objective, elements)`` to compute a decision from the live
    element list (so a test can target an element by its label without knowing its id).
    A convenience ``tap_label`` helper builds such a function. Records objectives in
    ``.seen`` and counts ``.calls``; ``images`` records whether a screenshot was attached.
    """

    name = "stub_planner"

    def __init__(
        self,
        *,
        available: bool = True,
        reason: str = "ok",
        decisions: list[PlannerDecision] | None = None,
        decide_fn: Any = None,
        raises: Exception | None = None,
    ) -> None:
        super().__init__()
        self._available = available
        self._reason = reason
        self._decisions = list(decisions or [])
        self._decide_fn = decide_fn
        self._raises = raises
        self.calls = 0
        self.seen: list[str] = []
        self.images: list[bool] = []

    def is_available(self) -> Availability:
        return Availability(self._available, self._reason)

    def decide(
        self, objective: str, elements: list[dict[str, Any]], image: ScreenImage | None = None
    ) -> PlannerDecision | None:
        self.calls += 1
        self.seen.append(objective)
        self.images.append(image is not None)
        if self._raises:
            raise self._raises
        if self._decide_fn is not None:
            return self._decide_fn(objective, elements)
        if not self._decisions:
            return PlannerDecision(action="give-up", reason="no scripted decisions")
        return self._decisions[min(self.calls - 1, len(self._decisions) - 1)]


def make_chain(kind: str, providers: list[Provider]) -> ChainSpec:
    return ChainSpec(kind=kind, providers=providers)


# --------------------------------------------------------------- dummy registered providers
# Registered here only — proves a provider is selectable by config alone (no engine edits).


@register_ocr("dummy")
class _DummyOcr(OcrProvider):
    def is_available(self) -> Availability:
        return Availability(True, "dummy always available")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        return [TextBox(text="dummy", bounds=(0, 0, 10, 10), confidence=1.0)]


@register_detection("dummy")
class _DummyDetection(DetectionProvider):
    def is_available(self) -> Availability:
        return Availability(True, "dummy always available")

    def detect(self, image: ScreenImage) -> list[DetBox]:
        return [DetBox(bounds=(0, 0, 10, 10), label="dummy", interactable=True, confidence=1.0)]


@register_grounding("dummy")
class _DummyGrounding(GroundingProvider):
    def is_available(self) -> Availability:
        return Availability(True, "dummy always available")

    def locate(self, image: ScreenImage, instruction: str) -> Point | DetBox | None:
        return Point(x=5, y=5, confidence=1.0)


@pytest.fixture(autouse=True)
def _never_record_into_a_real_training_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite out of the operator's policy trace.

    The trace is switched on by an environment variable, and a developer collecting real decisions
    exports it for every shell — including the one running pytest. The suite then appended its own
    fixture decisions to the corpus: 441 of the first 494 records came from `com.example.fixture`
    and friends, which is precisely the synthetic material this trace exists to stop us training
    on. Tests that exercise tracing set the variable themselves, and still can: this only clears
    an inherited value first.
    """

    monkeypatch.delenv("AUA_POLICY_TRACE_DIR", raising=False)
