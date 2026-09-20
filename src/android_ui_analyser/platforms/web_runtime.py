"""Semantic runtime for one isolated browser page."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from urllib.parse import urlsplit
from uuid import uuid4

from .. import read_budget
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
    def __init__(
        self,
        connection: WebConnection,
        target_url: str,
        *,
        home_url: str | None = None,
    ) -> None:
        self.target_id = target_url
        self._connection = connection
        self._home_url = home_url or target_url
        self._instance_token = f"web:{uuid4().hex}"

    def read_deadline(self, budget: read_budget.ReadBudget) -> AbstractContextManager[None]:
        return read_budget.activate(budget)

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

    def browser_storage(self, *, include_values: bool = False) -> dict[str, object]:
        return self._connection.storage(include_values=include_values)

    def browser_storage_export(self, path: str) -> dict[str, object]:
        return self._connection.storage_export(path)

    def browser_storage_import(self, path: str) -> dict[str, object]:
        return self._connection.storage_import(path)

    def browser_storage_clear(self, kinds: Sequence[str]) -> dict[str, object]:
        return self._connection.storage_clear(kinds)

    def browser_cache_clear(self) -> dict[str, object]:
        return self._connection.cache_clear()

    def browser_reset(self) -> dict[str, object]:
        return self._connection.reset()

    def browser_network_status(self) -> dict[str, object]:
        return self._connection.network_status()

    def browser_set_offline(self, offline: bool) -> dict[str, object]:
        return self._connection.set_offline(offline)

    def browser_set_throttle(
        self,
        *,
        latency_ms: int = 0,
        download_kbps: int = 0,
        upload_kbps: int = 0,
    ) -> dict[str, object]:
        return self._connection.set_throttle(
            latency_ms=latency_ms,
            download_kbps=download_kbps,
            upload_kbps=upload_kbps,
        )

    def browser_set_cors(
        self,
        *,
        origin: str,
        hosts: Sequence[str],
        methods: Sequence[str],
        headers: Sequence[str],
        credentials: bool = False,
    ) -> dict[str, object]:
        return self._connection.set_cors(
            origin=origin,
            hosts=hosts,
            methods=methods,
            headers=headers,
            credentials=credentials,
        )

    def browser_clear_cors(self) -> dict[str, object]:
        return self._connection.clear_cors()

    def browser_set_proxy(
        self,
        server: str,
        *,
        bypass: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> dict[str, object]:
        return self._connection.set_proxy(
            server,
            bypass=bypass,
            username=username,
            password=password,
        )

    def browser_clear_proxy(self) -> dict[str, object]:
        return self._connection.clear_proxy()

    def browser_har_start(self, path: str) -> dict[str, object]:
        return self._connection.har_start(path)

    def browser_har_stop(self) -> dict[str, object]:
        return self._connection.har_stop()

    def browser_har_replay(
        self, path: str, *, url: str | None = None, not_found: str = "abort"
    ) -> dict[str, object]:
        return self._connection.har_replay(path, url=url, not_found=not_found)

    def browser_har_clear(self) -> dict[str, object]:
        return self._connection.har_clear()

    def browser_mock_add(
        self,
        url: str,
        *,
        status: int = 200,
        body: str = "",
        headers: Mapping[str, str] | None = None,
        abort: bool = False,
    ) -> dict[str, object]:
        return self._connection.mock_add(
            url,
            status=status,
            body=body,
            headers=headers,
            abort=abort,
        )

    def browser_mock_clear(self, rule_id: str | None = None) -> dict[str, object]:
        return self._connection.mock_clear(rule_id)

    def browser_diagnostics(
        self,
        *,
        limit: int = 100,
        kinds: Sequence[str] = (),
        since_ms: int | None = None,
    ) -> dict[str, object]:
        return self._connection.diagnostics(limit=limit, kinds=kinds, since_ms=since_ms)

    def browser_diagnostics_clear(self) -> dict[str, object]:
        return self._connection.diagnostics_clear()

    def browser_diagnostics_mark(
        self, name: str, *, clear: bool = False
    ) -> dict[str, object]:
        return self._connection.mark_diagnostics(name, clear=clear)

    def browser_pages(self) -> dict[str, object]:
        return self._connection.pages()

    def browser_page_select(self, page_id: str) -> dict[str, object]:
        return self._connection.page_select(page_id)

    def browser_page_close(self, page_id: str) -> dict[str, object]:
        return self._connection.page_close(page_id)

    def browser_trace_start(self) -> dict[str, object]:
        return self._connection.trace_start()

    def browser_trace_stop(self, path: str) -> dict[str, object]:
        return self._connection.trace_stop(path)

    def session_state_begin(self, session_id: str) -> dict[str, object]:
        return self._connection.session_begin(session_id)

    def session_state_finish(self, session_id: str) -> dict[str, object]:
        return self._connection.session_finish(session_id)

    def instance_token(self) -> str | None:
        return self._instance_token

    def close(self) -> None:
        self._connection.close()


__all__ = ["KEY_NAMES", "WebRuntime"]
