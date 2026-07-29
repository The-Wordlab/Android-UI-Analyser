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
    ) -> None:
        self.serial = serial
        self._xml = hierarchy_xml
        self._w = width
        self._h = height
        self._pkg = package
        self._act = activity
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
        self._location: tuple[float, float] | None = None
        self._recording: str | None = None
        self._logcat_lines: list[str] = []
        self._logcat_cleared = False
        # host - device, so a positive skew means the host runs ahead (the emulator case).
        self.clock_skew_ms = clock_skew_ms
        self.utc_offset = utc_offset
        self._clock_ms: int | None = None

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

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        self.calls.append(("long_click", (x, y, duration_ms)))

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

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        self.calls.append(("launch_app", (package,) if activity is None else (package, activity)))

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
        local = datetime.fromtimestamp(
            device_ms / 1000.0 + self.utc_offset * 60, tz=UTC
        )
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
        return "\n".join(
            filter_logcat(raw, since_ms=since_ms, tz_offset_minutes=self.utc_offset)
        )

    def erase_chars(self, count: int) -> None:
        self.calls.append(("erase_chars", (count,)))

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
        if by == "id":
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
    # CLI commands run in-process (never reach a stray dev-machine daemon socket).
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    return state


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
