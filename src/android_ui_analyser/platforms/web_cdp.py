"""Attach to an existing local Chromium/Electron page without owning its lifetime."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ..errors import ConfigError, DeviceError, UnsupportedPlatformCapabilityError
from .web_bounded_reads import read as _read
from .web_tools import PlaywrightConnection, WebLaunchOptions


def cdp_endpoint(value: object) -> str:
    candidate = str(value or "").strip()
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and port is not None
            and port > 0
            and not parsed.username
            and not parsed.password
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError(
            "cdp_endpoint must be a loopback HTTP endpoint with an explicit port",
            hint="Example: http://127.0.0.1:9222. Enable remote debugging in the app first.",
        )
    # Normalize the transport address; the target identity also covers IPv6 aliases.
    return f"http://127.0.0.1:{port}" if parsed.hostname != "::1" else f"http://[::1]:{port}"


def cdp_page_url(value: object) -> str:
    candidate = str(value or "").strip()
    try:
        parsed = urlsplit(candidate)
        valid = (
            (parsed.scheme in {"http", "https"} and bool(parsed.hostname))
            or (parsed.scheme == "file" and not parsed.netloc and parsed.path.startswith("/"))
        ) and not (parsed.username or parsed.password)
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError(
            "page_url must be an absolute http(s) or local file URL without credentials"
        )
    return candidate


def cdp_target_id(endpoint: str) -> str:
    # IPv4 and IPv6 may reach the same app. Conservatively reserve the whole local port.
    identity = f"loopback:{urlsplit(endpoint).port}"
    return "cdp:" + hashlib.sha256(identity.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CdpAttachOptions:
    endpoint: str
    page_url: str | None = None
    attach_timeout_ms: int = 30_000
    action_timeout_ms: int = 5_000


class CdpConnection(PlaywrightConnection):
    def initialize_attached(self, page_url: str | None) -> None:
        def attach() -> None:
            pages = [
                page
                for context in self._browser.contexts
                for page in context.pages
                if not page.is_closed()
                and urlsplit(page.url).scheme in {"http", "https", "file"}
                and (page_url is None or page.url == page_url)
            ]
            if len(pages) != 1:
                raise DeviceError(
                    f"CDP attachment matched {len(pages)} pages; exactly one is required",
                    code="web_cdp_target_ambiguous" if pages else "web_cdp_target_missing",
                    hint="Set platforms.web.page_url to the exact URL of the intended window.",
                )
            self._page = pages[0]
            self._context = self._page.context
            self._home_url = str(self._page.url)
            # Client-side timeouts only; no viewport, routing, storage or network changes.
            self._page.set_default_timeout(self._options.action_timeout_ms)
            self._page.set_default_navigation_timeout(self._options.navigation_timeout_ms)
            self._on_page(self._page)

        self._call(attach)

    def _ensure_active_page(self) -> None:
        if self._page is not None and self._page.is_closed():
            raise DeviceError("the attached page is closed", code="web_target_closed")

    def _on_page_closed(self, page: Any, opener: Any) -> None:
        # Keep the closed target so the next call fails rather than selecting another window.
        pass

    def _apply_throttle_to_page(self, page: Any) -> None:
        # The shared observer installs diagnostics; attached pages retain their network state.
        pass

    def _snapshot_viewport(self) -> dict[str, int]:
        size = _read(self._page, "evaluate", "() => ({width: innerWidth, height: innerHeight})")
        return {"width": int(size["width"]), "height": int(size["height"])}

    def screenshot_png(self) -> bytes:
        def capture() -> bytes:
            data = _read(self._page, "screenshot", type="png", scale="css", animations="allow")
            if not isinstance(data, bytes):
                raise DeviceError("browser returned no PNG screenshot", code="screencap_failed")
            return data

        return self._call(capture)

    def pages(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            page = self._page
            return {
                "ok": True,
                "action": "browser-pages",
                "pages": [
                    {
                        "id": self._page_id(page),
                        "index": 0,
                        "active": True,
                        "attached": True,
                        "url": self._safe_url(str(page.url)),
                        "title": _read(page, "title"),
                        "frames": [
                            {
                                "name": str(frame.name or ""),
                                "url": self._safe_url(str(frame.url)),
                                "main": frame is page.main_frame,
                            }
                            for frame in page.frames
                        ],
                    }
                ],
            }

        return self._call(read)

    def _lookup_page(self, page_id: str) -> Any:
        if page_id not in {"0", self._page_id(self._page)}:
            raise ConfigError("page was not selected for this CDP session")
        return self._page

    def page_close(self, page_id: str) -> dict[str, Any]:
        raise UnsupportedPlatformCapabilityError("web", "browser.pages.close")

    def _storage_unsupported(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise UnsupportedPlatformCapabilityError("web", "browser.storage")

    storage = storage_export = storage_import = storage_clear = _storage_unsupported
    cache_clear = reset = _storage_unsupported

    def _network_unsupported(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise UnsupportedPlatformCapabilityError("web", "browser.network")

    network_status = set_offline = set_throttle = set_cors = clear_cors = _network_unsupported
    set_proxy = clear_proxy = har_start = har_stop = har_replay = har_clear = _network_unsupported
    mock_add = mock_clear = _network_unsupported

    def _trace_unsupported(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise UnsupportedPlatformCapabilityError("web", "browser.trace")

    trace_start = trace_stop = _trace_unsupported

    def session_begin(self, session_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "action": "browser-session-begin",
            "session_id": session_id,
            "attached": True,
            "captured": [],
        }

    def session_finish(self, session_id: str) -> dict[str, Any]:
        self.close()
        return {
            "ok": True,
            "action": "browser-session-finish",
            "session_id": session_id,
            "attached": True,
            "restored": [],
            "detached": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            # Stop only our driver. Browser/context.close would close the user's app windows.
            # Bypass _call: cleanup must also work after the selected page has closed itself.
            self._executor.submit(self._playwright.stop).result()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


class CdpLauncher:
    def launch(self, options: CdpAttachOptions) -> CdpConnection:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise DeviceError(
                "CDP attachment needs the optional Playwright dependency",
                code="web_driver_missing",
                hint="Install android-ui-analyser[web]; no browser download is needed for attachment.",
            ) from None

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aua-web-cdp")

        def start() -> tuple[Any, Any]:
            playwright = sync_playwright().start()
            try:
                if not callable(getattr(playwright._impl_obj._connection, "_abort", None)):
                    raise DeviceError(
                        "CDP attachment requires Playwright >=1.63", code="web_driver_outdated"
                    )
                browser = playwright.chromium.connect_over_cdp(
                    options.endpoint,
                    timeout=options.attach_timeout_ms,
                    no_defaults=True,
                )
                return playwright, browser
            except BaseException:
                playwright.stop()
                raise

        connection = None
        try:
            playwright, browser = executor.submit(start).result()
            connection = CdpConnection(
                executor,
                playwright,
                browser,
                playwright.chromium,
                WebLaunchOptions(action_timeout_ms=options.action_timeout_ms),
                "",
            )
            connection.initialize_attached(options.page_url)
            return connection
        except BaseException as exc:
            if connection is not None:
                connection.close()
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            if isinstance(exc, (ConfigError, DeviceError)) or not isinstance(exc, Exception):
                raise
            raise DeviceError(
                "could not attach to the configured CDP endpoint",
                code="web_cdp_unavailable",
                hint="Start the app with --remote-debugging-port=<port>, then retry session start.",
            ) from None
