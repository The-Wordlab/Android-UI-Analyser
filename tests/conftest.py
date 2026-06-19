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


def make_png(width: int = 200, height: int = 400, color: tuple[int, int, int] = (240, 240, 240),
             boxes: list[tuple[Bounds, tuple[int, int, int]]] | None = None) -> bytes:
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
        hierarchy_xml: str = "<hierarchy rotation=\"0\"></hierarchy>",
        width: int = 1080,
        height: int = 2400,
        package: str = "com.test.app",
        activity: str = ".MainActivity",
        text_index: dict[str, Bounds] | None = None,
        screenshot_bytes: bytes | None = None,
        serial: str = "fake-emulator-5554",
    ) -> None:
        self.serial = serial
        self._xml = hierarchy_xml
        self._w = width
        self._h = height
        self._pkg = package
        self._act = activity
        self._text_index = text_index or {}
        self._png = screenshot_bytes or make_png(width, height)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    # capture
    def window_size(self) -> tuple[int, int]:
        return self._w, self._h

    def dump_hierarchy(self, compressed: bool = False) -> str:
        return self._xml

    def screenshot(self) -> ScreenImage:
        return ScreenImage(self._png, width=self._w, height=self._h)

    def current_app(self) -> dict[str, str]:
        return {"package": self._pkg, "activity": self._act}

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

    def find_text(
        self, text: str, *, match: MatchMode | str = MatchMode.contains, ignore_case: bool = False
    ) -> Bounds | None:
        self.calls.append(("find_text", (text, str(match), ignore_case)))
        mode = MatchMode(match)
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
    """Config from defaults with shallow section overrides (dicts deep-merge)."""
    from android_ui_analyser.config import _deep_merge

    base = Config().model_dump(mode="python")
    merged = _deep_merge(base, overrides) if overrides else base
    return Config.model_validate(merged)


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

    def __init__(self, *, available: bool = True, reason: str = "ok",
                 result: list[TextBox] | None = None, raises: Exception | None = None) -> None:
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

    def __init__(self, *, available: bool = True, reason: str = "ok",
                 result: list[DetBox] | None = None, raises: Exception | None = None) -> None:
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

    def __init__(self, *, available: bool = True, reason: str = "ok",
                 result: Point | DetBox | None = None, raises: Exception | None = None) -> None:
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
