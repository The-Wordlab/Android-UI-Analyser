"""Strict, transport-free platform plugin loaded by the public conformance tests.

This file deliberately lives outside the AUA package.  Tests load it through a real
``importlib.metadata.EntryPoint`` so importing the selected plugin is exercised rather than
short-circuited with a registered in-process class.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image

from android_ui_analyser.platforms import (
    PLATFORM_API_VERSION,
    AppContext,
    Bounds,
    DiscoveredTarget,
    DisplayGeometry,
    Element,
    MatchMode,
    NormalizedTree,
    PlatformAdapter,
    ScreenImage,
    TargetInfo,
    TargetRuntime,
)

TARGET_ID = "shared-target"
APP_ID = "org.example.conformance"
NATIVE_BOUNDS = (10.0, 20.0, 30.0, 50.0)
CANONICAL_BOUNDS: Bounds = (300, 20, 360, 60)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), (245, 245, 245)).save(buffer, format="PNG")
    return buffer.getvalue()


class StrictExternalRuntime(TargetRuntime):
    """The minimum semantic runtime: no adb/logcat/shell/run-as convenience surface."""

    target_id = TARGET_ID

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.closed = False
        self._scroll_offset = 0.0
        self._image = _png()
        self._geometry = DisplayGeometry(
            native_size=(100.0, 200.0),
            canonical_size=(400, 200),
            # Native portrait points rotated clockwise into a 2x landscape screenshot.
            native_to_canonical=(0.0, 2.0, -2.0, 0.0, 400.0, 0.0),
        )

    def window_size(self) -> tuple[int, int]:
        return self._geometry.canonical_size

    def display_geometry(self) -> DisplayGeometry:
        return self._geometry

    def dump_hierarchy(self, compressed: bool = False) -> str:
        self.events.append(("dump_hierarchy", compressed))
        return json.dumps(
            {
                "app_id": APP_ID,
                "nodes": [
                    {
                        "type": "Button",
                        "text": "Continue",
                        "bounds": list(NATIVE_BOUNDS),
                        "clickable": True,
                    },
                    {
                        "type": "ScrollView",
                        "text": None,
                        "bounds": [5.0, 10.0, 95.0, 190.0],
                        "clickable": False,
                        "scrollable": True,
                    },
                    {
                        "type": "StaticText",
                        "text": "Card Alpha",
                        "bounds": [50.0 - self._scroll_offset, 30.0, 65.0 - self._scroll_offset, 80.0],
                        "clickable": False,
                    },
                    {
                        "type": "StaticText",
                        "text": "Card Beta",
                        "bounds": [75.0 - self._scroll_offset, 30.0, 90.0 - self._scroll_offset, 80.0],
                        "clickable": False,
                    },
                ],
            }
        )

    def screenshot(self) -> ScreenImage:
        self.events.append(("screenshot", None))
        return ScreenImage(self._image, width=400, height=200)

    def current_app(self) -> AppContext:
        return AppContext(app_id=APP_ID, surface_id="main")

    def click(self, x: int, y: int) -> None:
        # Native transports receive the inverse transform; shared Engine code always sends
        # canonical screenshot pixels.
        self.events.append(("click", self._geometry.to_native((x, y))))

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        self.events.append(
            ("long_click", (self._geometry.to_native((x, y)), duration_ms))
        )

    def send_text(self, text: str, *, clear: bool = True) -> None:
        self.events.append(("send_text", (text, clear)))

    def clear_text(self) -> None:
        self.events.append(("clear_text", None))

    def send_ime_action(self, action: str = "search") -> None:
        self.events.append(("send_ime_action", action))

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        self.events.append(
            (
                "swipe",
                (
                    self._geometry.to_native((x1, y1)),
                    self._geometry.to_native((x2, y2)),
                    duration_ms,
                ),
            )
        )
        self._scroll_offset += 10.0

    def press(self, key: str) -> None:
        self.events.append(("press", key))

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        if by != "text":
            return None
        candidate = "Continue"
        wanted = text
        if ignore_case:
            candidate, wanted = candidate.casefold(), wanted.casefold()
        mode = MatchMode(match)
        matched = candidate == wanted if mode is MatchMode.exact else wanted in candidate
        return CANONICAL_BOUNDS if matched else None

    def close(self) -> None:
        self.closed = True


class StrictExternalPlatform(PlatformAdapter):
    platform_api_version = PLATFORM_API_VERSION
    capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot"})
    last_runtime: StrictExternalRuntime | None = None

    def validate_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = sorted(set(options) - {"endpoint"})
        if unknown:
            raise ValueError(f"unknown strict fixture options: {', '.join(unknown)}")
        return {"endpoint": str(options.get("endpoint", "memory://fixture")).rstrip("/")}

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        if target_id not in (None, TARGET_ID):
            raise ValueError(f"unknown fixture target {target_id!r}")
        runtime = StrictExternalRuntime()
        type(self).last_runtime = runtime
        return runtime

    def list_targets(self) -> list[DiscoveredTarget]:
        return [
            TargetInfo(
                target_id=TARGET_ID,
                platform=self.name,
                model="Strict external fixture",
                os_name="fixture-os",
                os_version="1",
            )
        ]

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        if geometry is None:
            raise ValueError("the strict external fixture requires explicit display geometry")
        if screen_size != geometry.canonical_size:
            raise ValueError("screen size and display geometry disagree")
        payload = json.loads(raw_tree)
        elements: list[Element] = []
        for index, node in enumerate(payload["nodes"]):
            bounds = geometry.bounds_to_canonical(tuple(node["bounds"]))
            elements.append(
                Element(
                    id=index,
                    type=node["type"],
                    text=node["text"],
                    resource_id=f"fixture:{index}",
                    bounds=bounds,
                    center=((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2),
                    clickable=bool(node.get("clickable")),
                    scrollable=bool(node.get("scrollable")),
                    window="app",
                )
            )
        app_id = str(payload["app_id"])
        return NormalizedTree(
            elements=elements,
            app_id=None if app_id in ignored_app_ids else app_id,
        )

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        return runtime.screenshot()


class FutureStrictExternalPlatform(StrictExternalPlatform):
    platform_api_version = PLATFORM_API_VERSION + 1
