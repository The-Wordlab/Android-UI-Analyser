"""Semantic runtime for one isolated browser page."""

from __future__ import annotations

from urllib.parse import urlsplit
from uuid import uuid4

from ..errors import UsageError
from ..providers.base import Bounds, ScreenImage
from ..schema import AppContext, MatchMode
from . import web_tree
from .runtime import TargetRuntime
from .web_tools import WebConnection

_KEYS = {
    "enter": "Enter",
    "return": "Enter",
    "search": "Enter",
    "done": "Enter",
    "go": "Enter",
    "send": "Enter",
    "next": "Tab",
    "escape": "Escape",
    "esc": "Escape",
    "delete": "Backspace",
    "del": "Backspace",
    "backspace": "Backspace",
    "tab": "Tab",
    "space": "Space",
    "right": "ArrowRight",
    "left": "ArrowLeft",
    "down": "ArrowDown",
    "up": "ArrowUp",
    "page_down": "PageDown",
    "page-down": "PageDown",
    "page_up": "PageUp",
    "page-up": "PageUp",
}
KEY_NAMES = frozenset(_KEYS) | {"back", "home", "refresh", "reload"}


class WebRuntime(TargetRuntime):
    def __init__(self, connection: WebConnection, target_url: str) -> None:
        self.target_id = target_url
        self._connection = connection
        self._home_url = target_url
        self._instance_token = f"web:{uuid4().hex}"

    def window_size(self) -> tuple[int, int]:
        return self._connection.viewport_size()

    def dump_hierarchy(self, compressed: bool = False) -> str:
        del compressed
        return self._connection.snapshot()

    def screenshot(self) -> ScreenImage:
        width, height = self.window_size()
        return ScreenImage(self._connection.screenshot_png(), width=width, height=height)

    def current_app(self) -> AppContext:
        parsed = urlsplit(self._connection.url)
        surface = parsed.path or "/"
        if parsed.query:
            surface += f"?{parsed.query}"
        return AppContext(app_id=parsed.hostname or parsed.scheme or None, surface_id=surface)

    def find_text(
        self,
        text: str,
        *,
        match: MatchMode | str = MatchMode.contains,
        ignore_case: bool = False,
        by: str = "text",
    ) -> Bounds | None:
        return web_tree.find_bounds(
            self.dump_hierarchy(),
            screen_size=self.window_size(),
            query=text,
            match=match,
            ignore_case=ignore_case,
            by=by,
        )

    def click(self, x: int, y: int) -> None:
        self._connection.click(x, y)

    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None:
        self._connection.long_click(x, y, duration_ms)

    def send_text(self, text: str, *, clear: bool = True) -> None:
        if clear:
            self.clear_text()
        if text:
            self._connection.type_text(text)

    def clear_text(self) -> None:
        self._connection.clear_text()

    def send_ime_action(self, action: str = "search") -> None:
        candidate = action.strip().casefold()
        if candidate not in {"search", "done", "go", "send", "next", "enter", "return"}:
            raise UsageError(
                f"unknown web submit action {action!r}",
                hint="Choose search, done, go, send, next, enter, or return.",
            )
        self._connection.press(_KEYS[candidate])

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        del duration_ms
        self._connection.scroll(x1, y1, x1 - x2, y1 - y2)

    def press(self, key: str) -> None:
        candidate = key.strip().casefold()
        if candidate == "back":
            self._connection.go_back()
            return
        if candidate == "home":
            self._connection.goto(self._home_url)
            return
        if candidate in {"refresh", "reload"}:
            self._connection.reload()
            return
        translated = _KEYS.get(candidate)
        if translated is None:
            raise UsageError(
                f"unknown web key {key!r}",
                hint="Valid: " + ", ".join(sorted(KEY_NAMES)) + ".",
            )
        self._connection.press(translated)

    def open_link(self, uri: str, *, package: str | None = None) -> None:
        del package
        self._connection.goto(uri)

    def query_uri_handlers(self, uri: str) -> list[str]:
        parsed = urlsplit(uri)
        return [parsed.scheme] if parsed.scheme in {"http", "https"} else []

    def wait_idle(self, timeout_ms: int = 5000) -> None:
        self._connection.wait_idle(timeout_ms)

    def instance_token(self) -> str | None:
        return self._instance_token

    def close(self) -> None:
        self._connection.close()


__all__ = ["KEY_NAMES", "WebRuntime"]
