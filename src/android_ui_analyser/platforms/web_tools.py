"""Playwright-owned browser transport for the built-in web platform."""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

from ..errors import ConfigError, DeviceError


@dataclass(frozen=True)
class WebLaunchOptions:
    browser: str = "chromium"
    headless: bool = True
    channel: str | None = None
    executable_path: str | None = None
    viewport_width: int = 1280
    viewport_height: int = 800
    navigation_timeout_ms: int = 30_000
    action_timeout_ms: int = 5_000
    storage_state: str | None = None
    ignore_https_errors: bool = False
    bypass_csp: bool = False
    service_workers: str = "allow"
    proxy_server: str | None = None
    proxy_bypass: str | None = None
    proxy_username: str | None = None
    proxy_password: str | None = None


class WebConnection(Protocol):
    @property
    def url(self) -> str: ...

    def viewport_size(self) -> tuple[int, int]: ...

    def snapshot(self) -> str: ...

    def screenshot_png(self) -> bytes: ...

    def click(self, x: int, y: int) -> None: ...

    def long_click(self, x: int, y: int, duration_ms: int) -> None: ...

    def type_text(self, text: str) -> None: ...

    def clear_text(self) -> None: ...

    def press(self, key: str) -> None: ...

    def go_back(self) -> None: ...

    def reload(self) -> None: ...

    def scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> None: ...

    def goto(self, url: str) -> None: ...

    def wait_idle(self, timeout_ms: int) -> None: ...

    def storage(self, *, include_values: bool = False) -> dict[str, Any]: ...

    def storage_export(self, path: str) -> dict[str, Any]: ...

    def storage_import(self, path: str) -> dict[str, Any]: ...

    def storage_clear(self, kinds: Sequence[str]) -> dict[str, Any]: ...

    def cache_clear(self) -> dict[str, Any]: ...

    def reset(self) -> dict[str, Any]: ...

    def network_status(self) -> dict[str, Any]: ...

    def set_offline(self, offline: bool) -> dict[str, Any]: ...

    def set_throttle(
        self, *, latency_ms: int, download_kbps: int, upload_kbps: int
    ) -> dict[str, Any]: ...

    def set_cors(
        self,
        *,
        origin: str,
        hosts: Sequence[str],
        methods: Sequence[str],
        headers: Sequence[str],
        credentials: bool,
    ) -> dict[str, Any]: ...

    def clear_cors(self) -> dict[str, Any]: ...

    def set_proxy(
        self,
        server: str,
        *,
        bypass: str | None,
        username: str | None,
        password: str | None,
    ) -> dict[str, Any]: ...

    def clear_proxy(self) -> dict[str, Any]: ...

    def har_start(self, path: str) -> dict[str, Any]: ...

    def har_stop(self) -> dict[str, Any]: ...

    def har_replay(self, path: str, *, url: str | None, not_found: str) -> dict[str, Any]: ...

    def har_clear(self) -> dict[str, Any]: ...

    def mock_add(
        self,
        url: str,
        *,
        status: int,
        body: str,
        headers: Mapping[str, str] | None,
        abort: bool,
    ) -> dict[str, Any]: ...

    def mock_clear(self, rule_id: str | None = None) -> dict[str, Any]: ...

    def diagnostics(
        self, *, limit: int, kinds: Sequence[str], since_ms: int | None
    ) -> dict[str, Any]: ...

    def diagnostics_clear(self) -> dict[str, Any]: ...

    def mark_diagnostics(self, name: str, *, clear: bool = False) -> dict[str, Any]: ...

    def pages(self) -> dict[str, Any]: ...

    def page_select(self, page_id: str) -> dict[str, Any]: ...

    def page_close(self, page_id: str) -> dict[str, Any]: ...

    def trace_start(self) -> dict[str, Any]: ...

    def trace_stop(self, path: str) -> dict[str, Any]: ...

    def session_begin(self, session_id: str) -> dict[str, Any]: ...

    def session_finish(self, session_id: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


class WebLauncher(Protocol):
    def launch(self, url: str, options: WebLaunchOptions) -> WebConnection: ...


_DOM_SNAPSHOT_SCRIPT = r"""
() => {
  const skipped = new Set(['HEAD', 'META', 'LINK', 'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE']);
  const interactiveTags = new Set(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'SUMMARY']);
  const interactiveRoles = new Set([
    'button', 'checkbox', 'combobox', 'link', 'menuitem', 'option', 'radio',
    'searchbox', 'slider', 'spinbutton', 'switch', 'tab', 'textbox'
  ]);
  const compact = value => String(value || '').replace(/\s+/g, ' ').trim().slice(0, 500);
  const parentOf = element => element.parentElement || (element.getRootNode()?.host ?? null);
  const referencedText = (element, attribute) => {
    const root = element.getRootNode();
    return compact((element.getAttribute(attribute) || '').split(/\s+/).filter(Boolean)
      .map(id => root.getElementById?.(id) || document.getElementById(id))
      .filter(Boolean).map(node => node.innerText || node.textContent).join(' '));
  };
  const all = [];
  const walk = root => {
    for (const child of root.children || []) {
      all.push(child);
      if (child.shadowRoot) walk(child.shadowRoot);
      walk(child);
    }
  };
  walk(document.documentElement);

  const candidates = [];
  for (const element of all) {
    if (skipped.has(element.tagName) || element.getAttribute('aria-hidden') === 'true') continue;
    const style = getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) continue;
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0 || rect.right <= 0 || rect.bottom <= 0 ||
        rect.left >= innerWidth || rect.top >= innerHeight) continue;

    const tag = element.tagName.toLowerCase();
    const role = compact(element.getAttribute('role')).toLowerCase();
    const inputType = tag === 'input' ? compact(element.getAttribute('type') || 'text').toLowerCase() : '';
    const labels = element.labels ? Array.from(element.labels).map(label => compact(label.innerText)).filter(Boolean) : [];
    const accessibleName = compact(element.getAttribute('aria-label') ||
      referencedText(element, 'aria-labelledby') || labels.join(' ') ||
      element.getAttribute('alt') || element.getAttribute('title') || '');
    const describedBy = referencedText(element, 'aria-describedby');
    const ownText = compact(Array.from(element.childNodes)
      .filter(node => node.nodeType === Node.TEXT_NODE)
      .map(node => node.textContent).join(' '));
    let text = ownText;
    if (tag === 'input') {
      text = inputType === 'password' ? compact(element.getAttribute('placeholder')) :
        compact(element.value || element.getAttribute('placeholder'));
    } else if (tag === 'textarea') {
      text = compact(element.value || element.getAttribute('placeholder'));
    } else if (tag === 'select') {
      text = compact(element.selectedOptions?.[0]?.textContent || '');
    } else if (interactiveTags.has(element.tagName) || interactiveRoles.has(role)) {
      text = compact(element.innerText || ownText);
    }
    const resourceId = compact(element.getAttribute('data-testid') ||
      element.getAttribute('data-test-id') || element.getAttribute('data-test') || element.id || '');
    const overflowY = style.overflowY;
    const overflowX = style.overflowX;
    const scrollable = ((overflowY === 'auto' || overflowY === 'scroll') && element.scrollHeight > element.clientHeight) ||
      ((overflowX === 'auto' || overflowX === 'scroll') && element.scrollWidth > element.clientWidth);
    const clickable = interactiveTags.has(element.tagName) || interactiveRoles.has(role) ||
      typeof element.onclick === 'function' || element.tabIndex >= 0 || style.cursor === 'pointer';
    if (!clickable && !scrollable && !text && !accessibleName && !describedBy && !resourceId) continue;
    const checkable = inputType === 'checkbox' || inputType === 'radio' || role === 'checkbox' ||
      role === 'radio' || role === 'switch';
    const ariaChecked = element.getAttribute('aria-checked');
    const checked = checkable ? (typeof element.checked === 'boolean' ? element.checked : ariaChecked === 'true') : null;
    const ariaSelected = element.getAttribute('aria-selected');
    const selected = tag === 'option' || ariaSelected !== null ?
      (typeof element.selected === 'boolean' ? element.selected : ariaSelected === 'true') : null;
    candidates.push({
      element, tag, role, input_type: inputType || null, text: text || null,
      description: compact([
        accessibleName !== text ? accessibleName : '', describedBy,
      ].filter(Boolean).join(' — ')) || null,
      resource_id: resourceId || null,
      bounds: [rect.left, rect.top, rect.right, rect.bottom], clickable,
      enabled: !element.disabled && element.getAttribute('aria-disabled') !== 'true',
      focused: document.activeElement === element, checkable: checkable ? true : null,
      checked, selected, scrollable: scrollable ? true : null,
      password: tag === 'input' ? inputType === 'password' : null,
    });
  }
  const indexes = new Map(candidates.map((candidate, index) => [candidate.element, index]));
  const nodes = candidates.map(candidate => {
    let parent = parentOf(candidate.element);
    while (parent && !indexes.has(parent)) parent = parentOf(parent);
    const {element, ...node} = candidate;
    node.parent = parent ? indexes.get(parent) : null;
    return node;
  });
  return {
    format: 'aua-web-dom/1', url: location.href, title: document.title,
    viewport: {width: innerWidth, height: innerHeight}, nodes,
  };
}
"""

_STORAGE_DETAILS_SCRIPT = r"""
async () => {
  const local = Object.fromEntries(Array.from({length: localStorage.length}, (_, index) => {
    const key = localStorage.key(index); return [key, localStorage.getItem(key)];
  }));
  const session = Object.fromEntries(Array.from({length: sessionStorage.length}, (_, index) => {
    const key = sessionStorage.key(index); return [key, sessionStorage.getItem(key)];
  }));
  const databases = indexedDB.databases ? (await indexedDB.databases()).map(db => db.name).filter(Boolean) : [];
  const cacheNames = self.caches ? await caches.keys() : [];
  const registrations = navigator.serviceWorker ? await navigator.serviceWorker.getRegistrations() : [];
  return {
    origin: location.origin,
    local_storage: local,
    session_storage: session,
    indexed_db: databases,
    cache_storage: cacheNames,
    service_workers: registrations.map(item => item.scope),
  };
}
"""

_CLEAR_STORAGE_SCRIPT = r"""
async kinds => {
  const done = [];
  if (kinds.includes('local')) { localStorage.clear(); done.push('local'); }
  if (kinds.includes('session')) { sessionStorage.clear(); done.push('session'); }
  if (kinds.includes('indexeddb') && indexedDB.databases) {
    for (const db of await indexedDB.databases()) {
      if (db.name) await new Promise(resolve => {
        const request = indexedDB.deleteDatabase(db.name);
        request.onsuccess = request.onerror = request.onblocked = () => resolve();
      });
    }
    done.push('indexeddb');
  }
  if (kinds.includes('cache') && self.caches) {
    for (const name of await caches.keys()) await caches.delete(name);
    done.push('cache');
  }
  if (kinds.includes('service-workers') && navigator.serviceWorker) {
    for (const registration of await navigator.serviceWorker.getRegistrations()) {
      await registration.unregister();
    }
    done.push('service-workers');
  }
  return done;
}
"""


_T = TypeVar("_T")


class PlaywrightConnection:
    """One Playwright context plus browser-lab state, owned by a single worker thread."""

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        playwright: Any,
        browser: Any,
        browser_type: Any,
        options: WebLaunchOptions,
        home_url: str,
    ) -> None:
        self._executor = executor
        self._playwright = playwright
        self._browser = browser
        self._browser_type = browser_type
        self._options = options
        self._home_url = home_url
        self._context: Any = None
        self._page: Any = None
        self._closed = False
        self._page_ids: dict[int, str] = {}
        self._next_page_id = 1
        self._events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._marks: dict[str, int] = {}
        self._cors_rules: list[dict[str, Any]] = []
        self._mock_rules: list[dict[str, Any]] = []
        self._offline = False
        self._throttle = {"latency_ms": 0, "download_kbps": 0, "upload_kbps": 0}
        self._proxy: dict[str, str] | None = self._configured_proxy(options)
        self._har_record_path: str | None = None
        self._har_replay: dict[str, str | None] | None = None
        self._trace_active = False
        self._session_baselines: dict[str, dict[str, Any]] = {}
        self._route_handler = lambda route, request: self._dispatch_route(route, request)

    @staticmethod
    def _configured_proxy(options: WebLaunchOptions) -> dict[str, str] | None:
        if not options.proxy_server:
            return None
        out = {"server": options.proxy_server}
        for key, value in (
            ("bypass", options.proxy_bypass),
            ("username", options.proxy_username),
            ("password", options.proxy_password),
        ):
            if value:
                out[key] = value
        return out

    def initialize(self) -> None:
        self._call(lambda: self._replace_context(storage_state=self._initial_storage_state()))

    def _call(self, operation: Callable[[], _T]) -> _T:
        if self._closed:
            raise DeviceError("the web target is closed", code="web_target_closed")
        try:
            return self._executor.submit(operation).result()
        except (ConfigError, DeviceError):
            raise
        except Exception as exc:
            raise DeviceError(
                f"browser operation failed: {exc}",
                code="web_operation_failed",
            ) from None

    def _initial_storage_state(self) -> str | None:
        if not self._options.storage_state:
            return None
        state = Path(self._options.storage_state).expanduser()
        if not state.is_file():
            raise ConfigError(f"web storage_state does not exist: {state}")
        return str(state)

    def _context_kwargs(self, storage_state: Any = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "viewport": {
                "width": self._options.viewport_width,
                "height": self._options.viewport_height,
            },
            "device_scale_factor": 1,
            "ignore_https_errors": self._options.ignore_https_errors,
            "bypass_csp": self._options.bypass_csp,
            "service_workers": self._options.service_workers,
        }
        if storage_state is not None:
            kwargs["storage_state"] = storage_state
        if self._proxy:
            kwargs["proxy"] = dict(self._proxy)
        if self._har_record_path:
            kwargs.update(
                record_har_path=self._har_record_path,
                record_har_mode="full",
                record_har_content="embed",
            )
        return kwargs

    def _storage_state(self) -> dict[str, Any]:
        try:
            return dict(
                self._context.storage_state(
                    indexed_db=True,
                    opfs=True,
                    credentials=False,
                )
            )
        except TypeError:
            try:
                return dict(self._context.storage_state(indexed_db=True))
            except TypeError:
                return dict(self._context.storage_state())

    def _session_storage(self) -> dict[str, str]:
        try:
            values = self._page.evaluate(
                "() => Object.fromEntries(Array.from({length: sessionStorage.length}, "
                "(_, i) => { const key = sessionStorage.key(i); "
                "return [key, sessionStorage.getItem(key)]; }))"
            )
            return {str(key): str(value) for key, value in dict(values or {}).items()}
        except Exception:
            return {}

    def _replace_context(
        self,
        *,
        storage_state: Any = None,
        session_storage: Mapping[str, str] | None = None,
        target_url: str | None = None,
    ) -> None:
        if self._trace_active and self._context is not None:
            self._context.tracing.stop()
            self._trace_active = False
            self._add_event("trace", "warning", "trace discarded because the context reset")
        old = self._context
        if old is not None:
            old.close()
        self._page_ids.clear()
        self._context = self._browser.new_context(**self._context_kwargs(storage_state))
        self._context.set_default_timeout(self._options.action_timeout_ms)
        self._context.set_default_navigation_timeout(self._options.navigation_timeout_ms)
        self._context.on("page", self._on_page)
        if self._har_replay:
            self._context.route_from_har(
                str(self._har_replay["path"]),
                url=self._har_replay.get("url"),
                not_found=str(self._har_replay.get("not_found") or "abort"),
            )
        self._context.route("**/*", self._route_handler)
        self._context.set_offline(self._offline)
        page = self._context.new_page()
        self._page = page
        destination = target_url or self._home_url
        if destination:
            page.goto(destination, wait_until="domcontentloaded")
        if session_storage:
            page.evaluate(
                "values => { sessionStorage.clear(); for (const [key, value] of "
                "Object.entries(values)) sessionStorage.setItem(key, value); }",
                dict(session_storage),
            )

    @staticmethod
    def _safe_url(value: str | None) -> str | None:
        if not value:
            return None
        try:
            parsed = urlsplit(value)
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except Exception:
            return None

    def _add_event(
        self,
        kind: str,
        level: str,
        message: str,
        *,
        url: str | None = None,
        **fields: Any,
    ) -> None:
        self._events.append(
            {
                "timestamp_ms": int(time.time() * 1000),
                "kind": kind,
                "level": level,
                "message": str(message)[:2000],
                "url": self._safe_url(url),
                **{key: value for key, value in fields.items() if value is not None},
            }
        )

    def _page_id(self, page: Any) -> str:
        key = id(page)
        value = self._page_ids.get(key)
        if value is None:
            value = f"page-{self._next_page_id}"
            self._next_page_id += 1
            self._page_ids[key] = value
        return value

    def _on_page(self, page: Any) -> None:
        self._page_id(page)
        self._page = page
        page.on(
            "console",
            lambda message: self._add_event(
                "console",
                str(getattr(message, "type", "info")),
                str(getattr(message, "text", message)),
                url=getattr(page, "url", None),
            ),
        )
        page.on(
            "pageerror",
            lambda error: self._add_event(
                "page_error", "error", str(error), url=getattr(page, "url", None)
            ),
        )
        page.on(
            "request",
            lambda request: self._add_event(
                "request",
                "debug",
                f"{request.method} {self._safe_url(request.url)}",
                url=request.url,
                method=request.method,
                resource_type=request.resource_type,
            ),
        )
        page.on(
            "response",
            lambda response: self._add_event(
                "response",
                "error" if response.status >= 500 else "warning" if response.status >= 400 else "debug",
                f"{response.status} {self._safe_url(response.url)}",
                url=response.url,
                status=response.status,
            ),
        )
        page.on(
            "requestfailed",
            lambda request: self._add_event(
                "request_failed",
                "error",
                str(request.failure or "request failed"),
                url=request.url,
                method=request.method,
                resource_type=request.resource_type,
            ),
        )
        page.on(
            "websocket",
            lambda socket: self._add_event(
                "websocket", "info", "websocket opened", url=socket.url
            ),
        )
        self._apply_throttle_to_page(page)

    def _apply_throttle_to_page(self, page: Any) -> None:
        if self._options.browser != "chromium":
            return
        throttle = self._throttle
        session = self._context.new_cdp_session(page)
        session.send("Network.enable")
        session.send(
            "Network.emulateNetworkConditions",
            {
                "offline": self._offline,
                "latency": int(throttle["latency_ms"]),
                "downloadThroughput": (
                    int(throttle["download_kbps"]) * 1024 // 8
                    if throttle["download_kbps"]
                    else -1
                ),
                "uploadThroughput": (
                    int(throttle["upload_kbps"]) * 1024 // 8
                    if throttle["upload_kbps"]
                    else -1
                ),
            },
        )

    def _matching_cors_rule(self, request: Any) -> dict[str, Any] | None:
        origin = request.headers.get("origin")
        host = urlsplit(request.url).hostname or ""
        for rule in reversed(self._cors_rules):
            if not any(fnmatch(host, pattern) for pattern in rule["hosts"]):
                continue
            if origin and rule["origin"] not in {"*", origin}:
                continue
            return rule
        return None

    def _dispatch_route(self, route: Any, request: Any) -> None:
        for rule in reversed(self._mock_rules):
            if fnmatch(request.url, rule["url"]):
                if rule["abort"]:
                    route.abort()
                else:
                    route.fulfill(
                        status=rule["status"],
                        body=rule["body"],
                        headers=rule["headers"],
                    )
                return
        cors = self._matching_cors_rule(request)
        if self._options.browser != "chromium" and self._throttle["latency_ms"]:
            time.sleep(self._throttle["latency_ms"] / 1000.0)
        if cors is None:
            route.fallback()
            return
        origin = request.headers.get("origin") or cors["origin"]
        allow_origin = origin if cors["credentials"] and origin != "*" else cors["origin"]
        allow_headers = ", ".join(cors["headers"]) or request.headers.get(
            "access-control-request-headers", "*"
        )
        response_headers = {
            "access-control-allow-origin": allow_origin,
            "access-control-allow-methods": ", ".join(cors["methods"]),
            "access-control-allow-headers": allow_headers,
            "vary": "Origin",
        }
        if cors["credentials"]:
            response_headers["access-control-allow-credentials"] = "true"
        if request.method.upper() == "OPTIONS":
            route.fulfill(status=204, body="", headers=response_headers)
            return
        response = route.fetch()
        headers = dict(response.headers)
        headers.update(response_headers)
        route.fulfill(response=response, headers=headers)

    @property
    def url(self) -> str:
        return self._call(lambda: str(self._page.url))

    def viewport_size(self) -> tuple[int, int]:
        def read() -> tuple[int, int]:
            value = self._page.viewport_size
            if not isinstance(value, dict):
                value = self._page.evaluate("() => ({width: innerWidth, height: innerHeight})")
            return int(value["width"]), int(value["height"])

        return self._call(read)

    def snapshot(self) -> str:
        def capture() -> str:
            merged: list[dict[str, Any]] = []
            top = self._page
            # Sync Playwright dispatches route/response callbacks while an API call is active.
            # Give a just-completed intercepted response one event-loop turn before reading DOM
            # text, otherwise the hierarchy can capture the pre-promise "loading" value.
            top.wait_for_timeout(0)
            for frame in list(top.frames):
                try:
                    payload = dict(frame.evaluate(_DOM_SNAPSHOT_SCRIPT))
                    offset_x = 0.0
                    offset_y = 0.0
                    if frame is not top.main_frame:
                        handle = frame.frame_element()
                        try:
                            box = handle.bounding_box()
                        finally:
                            handle.dispose()
                        if not box:
                            continue
                        offset_x = float(box["x"])
                        offset_y = float(box["y"])
                    base = len(merged)
                    for node in payload.get("nodes") or []:
                        item = dict(node)
                        bounds = item.get("bounds")
                        if isinstance(bounds, list) and len(bounds) == 4:
                            item["bounds"] = [
                                float(bounds[0]) + offset_x,
                                float(bounds[1]) + offset_y,
                                float(bounds[2]) + offset_x,
                                float(bounds[3]) + offset_y,
                            ]
                        parent = item.get("parent")
                        item["parent"] = base + int(parent) if isinstance(parent, int) else None
                        item["frame_url"] = self._safe_url(str(frame.url))
                        merged.append(item)
                except Exception as exc:
                    self._add_event(
                        "frame_error",
                        "warning",
                        f"could not inspect frame: {exc}",
                        url=getattr(frame, "url", None),
                    )
            payload = {
                "format": "aua-web-dom/1",
                "url": str(top.url),
                "title": top.title(),
                "viewport": {
                    "width": self._options.viewport_width,
                    "height": self._options.viewport_height,
                },
                "nodes": merged,
            }
            return json.dumps(payload, ensure_ascii=False)

        return self._call(capture)

    def screenshot_png(self) -> bytes:
        def capture() -> bytes:
            data = self._page.screenshot(type="png", animations="disabled")
            if not isinstance(data, bytes):
                raise DeviceError("browser returned no PNG screenshot", code="screencap_failed")
            return data

        return self._call(capture)

    def click(self, x: int, y: int) -> None:
        self._call(lambda: self._page.mouse.click(x, y))

    def long_click(self, x: int, y: int, duration_ms: int) -> None:
        def hold() -> None:
            self._page.mouse.move(x, y)
            self._page.mouse.down()
            self._page.wait_for_timeout(max(1, duration_ms))
            self._page.mouse.up()

        self._call(hold)

    def type_text(self, text: str) -> None:
        self._call(lambda: self._page.keyboard.insert_text(text))

    def clear_text(self) -> None:
        def clear() -> None:
            modifier = "Meta" if sys.platform == "darwin" else "Control"
            self._page.keyboard.press(f"{modifier}+A")
            self._page.keyboard.press("Backspace")

        self._call(clear)

    def press(self, key: str) -> None:
        self._call(lambda: self._page.keyboard.press(key))

    def go_back(self) -> None:
        self._call(lambda: self._page.go_back(wait_until="domcontentloaded"))

    def reload(self) -> None:
        self._call(lambda: self._page.reload(wait_until="domcontentloaded"))

    def scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> None:
        def wheel() -> None:
            self._page.mouse.move(x, y)
            self._page.mouse.wheel(delta_x, delta_y)

        self._call(wheel)

    def goto(self, url: str) -> None:
        self._call(lambda: self._page.goto(url, wait_until="domcontentloaded"))

    def wait_idle(self, timeout_ms: int) -> None:
        def wait() -> None:
            try:
                self._page.wait_for_load_state("domcontentloaded", timeout=max(1, timeout_ms))
            except Exception as exc:
                if "Timeout" not in type(exc).__name__:
                    raise

        self._call(wait)

    def _storage_details(self) -> dict[str, Any]:
        try:
            return dict(self._page.evaluate(_STORAGE_DETAILS_SCRIPT) or {})
        except Exception:
            return {
                "origin": None,
                "local_storage": {},
                "session_storage": {},
                "indexed_db": [],
                "cache_storage": [],
                "service_workers": [],
            }

    def storage(self, *, include_values: bool = False) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            state = self._storage_state()
            details = self._storage_details()
            cookies = list(state.get("cookies") or [])
            origins = list(state.get("origins") or [])
            if not include_values:
                cookies = [
                    {
                        key: cookie.get(key)
                        for key in (
                            "name",
                            "domain",
                            "path",
                            "expires",
                            "httpOnly",
                            "secure",
                            "sameSite",
                        )
                        if key in cookie
                    }
                    for cookie in cookies
                ]
                origins = [
                    {
                        "origin": origin.get("origin"),
                        "localStorage": [
                            item.get("name") for item in origin.get("localStorage") or []
                        ],
                        "indexedDB": [
                            item.get("name") for item in origin.get("indexedDB") or []
                        ],
                    }
                    for origin in origins
                ]
                details["local_storage"] = sorted(details.get("local_storage") or {})
                details["session_storage"] = sorted(details.get("session_storage") or {})
            return {
                "ok": True,
                "action": "browser-storage",
                "include_values": include_values,
                "cookies": cookies,
                "origins": origins,
                "active_origin": details,
            }

        return self._call(read)

    def storage_export(self, path: str) -> dict[str, Any]:
        def export() -> dict[str, Any]:
            destination = Path(path).expanduser().resolve()
            if destination.exists():
                raise ConfigError(f"browser storage export already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "format": "aua-web-storage/1",
                "url": str(self._page.url),
                "storage_state": self._storage_state(),
                "session_storage": self._session_storage(),
            }
            destination.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return {
                "ok": True,
                "action": "browser-storage-export",
                "path": str(destination),
            }

        return self._call(export)

    def storage_import(self, path: str) -> dict[str, Any]:
        def restore() -> dict[str, Any]:
            source = Path(path).expanduser().resolve()
            if not source.is_file():
                raise ConfigError(f"browser storage import does not exist: {source}")
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ConfigError("browser storage import must contain a JSON object")
            state = payload.get("storage_state", payload)
            session = payload.get("session_storage") or {}
            target = str(payload.get("url") or self._page.url)
            parsed_target = urlsplit(target)
            if parsed_target.scheme not in {"http", "https"} or not parsed_target.hostname:
                raise ConfigError(
                    "browser storage import URL must be an absolute http(s) URL"
                )
            self._replace_context(
                storage_state=state,
                session_storage=session if isinstance(session, dict) else None,
                target_url=target,
            )
            return {
                "ok": True,
                "action": "browser-storage-import",
                "path": str(source),
                "url": self._safe_url(str(self._page.url)),
            }

        return self._call(restore)

    def storage_clear(self, kinds: Sequence[str]) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            requested = {str(kind).strip().casefold() for kind in kinds if str(kind).strip()}
            if not requested or "all" in requested:
                requested = {
                    "cookies",
                    "local",
                    "session",
                    "indexeddb",
                    "cache",
                    "service-workers",
                }
            known = {
                "cookies",
                "local",
                "session",
                "indexeddb",
                "cache",
                "service-workers",
            }
            unknown = sorted(requested - known)
            if unknown:
                raise ConfigError(f"unknown browser storage kinds: {', '.join(unknown)}")
            if requested == known:
                target = str(self._page.url)
                self._replace_context(storage_state=None, target_url=target)
            else:
                if "cookies" in requested:
                    self._context.clear_cookies()
                script_kinds = sorted(requested - {"cookies"})
                if script_kinds:
                    self._page.evaluate(_CLEAR_STORAGE_SCRIPT, script_kinds)
            return {
                "ok": True,
                "action": "browser-storage-clear",
                "cleared": sorted(requested),
            }

        return self._call(clear)

    def cache_clear(self) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            self._replace_context(
                storage_state=state,
                session_storage=session,
                target_url=target,
            )
            return {"ok": True, "action": "browser-cache-clear", "url": self._safe_url(target)}

        return self._call(clear)

    def reset(self) -> dict[str, Any]:
        def reset_context() -> dict[str, Any]:
            self._offline = False
            self._throttle = {"latency_ms": 0, "download_kbps": 0, "upload_kbps": 0}
            self._cors_rules.clear()
            self._mock_rules.clear()
            self._proxy = self._configured_proxy(self._options)
            self._har_record_path = None
            self._har_replay = None
            self._replace_context(storage_state=self._initial_storage_state(), target_url=self._home_url)
            return {"ok": True, "action": "browser-reset", "url": self._safe_url(self._home_url)}

        return self._call(reset_context)

    def network_status(self) -> dict[str, Any]:
        def status() -> dict[str, Any]:
            proxy = None
            if self._proxy:
                proxy = {
                    "server": self._proxy.get("server"),
                    "bypass": self._proxy.get("bypass"),
                    "username_configured": bool(self._proxy.get("username")),
                    "password_configured": bool(self._proxy.get("password")),
                }
            return {
                "ok": True,
                "action": "browser-network-status",
                "offline": self._offline,
                "throttle": dict(self._throttle),
                "cors_rules": [
                    {key: value for key, value in rule.items() if key != "id"} | {"id": rule["id"]}
                    for rule in self._cors_rules
                ],
                "proxy": proxy,
                "har_recording": self._har_record_path,
                "har_replay": dict(self._har_replay) if self._har_replay else None,
                "mock_rules": [
                    {
                        "id": rule["id"],
                        "url": rule["url"],
                        "status": rule["status"],
                        "abort": rule["abort"],
                        "body_chars": len(rule["body"]),
                        "header_names": sorted(rule["headers"]),
                    }
                    for rule in self._mock_rules
                ],
            }

        return self._call(status)

    def set_offline(self, offline: bool) -> dict[str, Any]:
        def apply() -> dict[str, Any]:
            self._context.set_offline(bool(offline))
            self._offline = bool(offline)
            if self._options.browser == "chromium":
                for page in self._context.pages:
                    self._apply_throttle_to_page(page)
            return {"ok": True, "action": "browser-offline", "offline": self._offline}

        return self._call(apply)

    def set_throttle(
        self, *, latency_ms: int, download_kbps: int, upload_kbps: int
    ) -> dict[str, Any]:
        def apply() -> dict[str, Any]:
            values = (latency_ms, download_kbps, upload_kbps)
            if any(isinstance(value, bool) or int(value) < 0 for value in values):
                raise ConfigError("browser throttle values must be non-negative integers")
            if self._options.browser != "chromium" and (download_kbps or upload_kbps):
                raise DeviceError(
                    "bandwidth throttling needs a Chromium browser",
                    code="web_bandwidth_throttle_unsupported",
                    hint="Firefox and WebKit support offline and latency controls; choose chromium for bandwidth limits.",
                )
            self._throttle = {
                "latency_ms": int(latency_ms),
                "download_kbps": int(download_kbps),
                "upload_kbps": int(upload_kbps),
            }
            if self._options.browser == "chromium":
                for page in self._context.pages:
                    self._apply_throttle_to_page(page)
            return {"ok": True, "action": "browser-throttle", **self._throttle}

        return self._call(apply)

    def set_cors(
        self,
        *,
        origin: str,
        hosts: Sequence[str],
        methods: Sequence[str],
        headers: Sequence[str],
        credentials: bool,
    ) -> dict[str, Any]:
        def apply() -> dict[str, Any]:
            allowed_origin = str(origin).strip()
            parsed = urlsplit(allowed_origin) if allowed_origin != "*" else None
            if allowed_origin != "*" and (parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname):
                raise ConfigError("CORS origin must be '*' or an absolute http(s) origin")
            if allowed_origin == "*" and credentials:
                raise ConfigError("credentialed CORS cannot use wildcard origin '*'")
            patterns = [str(item).strip().casefold() for item in hosts if str(item).strip()]
            if not patterns:
                raise ConfigError("CORS control needs at least one target host pattern")
            rule = {
                "id": f"cors-{uuid.uuid4().hex[:10]}",
                "origin": allowed_origin,
                "hosts": patterns,
                "methods": [str(item).strip().upper() for item in methods if str(item).strip()]
                or ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                "headers": [str(item).strip() for item in headers if str(item).strip()],
                "credentials": bool(credentials),
            }
            self._cors_rules.append(rule)
            return {"ok": True, "action": "browser-cors-add", "rule": dict(rule)}

        return self._call(apply)

    def clear_cors(self) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            count = len(self._cors_rules)
            self._cors_rules.clear()
            return {"ok": True, "action": "browser-cors-clear", "cleared": count}

        return self._call(clear)

    @staticmethod
    def _validate_proxy(server: str) -> str:
        candidate = str(server).strip()
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https", "socks4", "socks5"} or not parsed.hostname:
            raise ConfigError("browser proxy server must be an absolute http(s), socks4, or socks5 URL")
        if parsed.username or parsed.password:
            raise ConfigError("browser proxy credentials must not be embedded in the server URL")
        return candidate

    def set_proxy(
        self,
        server: str,
        *,
        bypass: str | None,
        username: str | None,
        password: str | None,
    ) -> dict[str, Any]:
        def apply() -> dict[str, Any]:
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            self._proxy = {"server": self._validate_proxy(server)}
            for key, value in (("bypass", bypass), ("username", username), ("password", password)):
                if value:
                    self._proxy[key] = str(value)
            self._replace_context(storage_state=state, session_storage=session, target_url=target)
            return {
                "ok": True,
                "action": "browser-proxy-set",
                "server": self._proxy["server"],
                "bypass": self._proxy.get("bypass"),
                "username_configured": bool(self._proxy.get("username")),
                "password_configured": bool(self._proxy.get("password")),
            }

        return self._call(apply)

    def clear_proxy(self) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            self._proxy = None
            self._replace_context(storage_state=state, session_storage=session, target_url=target)
            return {"ok": True, "action": "browser-proxy-clear"}

        return self._call(clear)

    def har_start(self, path: str) -> dict[str, Any]:
        def start() -> dict[str, Any]:
            if self._har_record_path:
                raise ConfigError(f"HAR recording is already active: {self._har_record_path}")
            destination = Path(path).expanduser().resolve()
            if destination.exists():
                raise ConfigError(f"HAR destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            self._har_record_path = str(destination)
            self._replace_context(storage_state=state, session_storage=session, target_url=target)
            return {"ok": True, "action": "browser-har-start", "path": str(destination)}

        return self._call(start)

    def har_stop(self) -> dict[str, Any]:
        def stop() -> dict[str, Any]:
            if not self._har_record_path:
                raise ConfigError("HAR recording is not active")
            path = self._har_record_path
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            self._har_record_path = None
            self._replace_context(storage_state=state, session_storage=session, target_url=target)
            return {
                "ok": Path(path).is_file(),
                "action": "browser-har-stop",
                "path": path,
            }

        return self._call(stop)

    def har_replay(self, path: str, *, url: str | None, not_found: str) -> dict[str, Any]:
        def replay() -> dict[str, Any]:
            source = Path(path).expanduser().resolve()
            if not source.is_file():
                raise ConfigError(f"HAR replay file does not exist: {source}")
            if not_found not in {"abort", "fallback"}:
                raise ConfigError("HAR not_found must be 'abort' or 'fallback'")
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            self._har_replay = {"path": str(source), "url": url, "not_found": not_found}
            self._replace_context(storage_state=state, session_storage=session, target_url=target)
            return {"ok": True, "action": "browser-har-replay", **self._har_replay}

        return self._call(replay)

    def har_clear(self) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            state = self._storage_state()
            session = self._session_storage()
            target = str(self._page.url)
            active = self._har_replay
            self._har_replay = None
            self._replace_context(storage_state=state, session_storage=session, target_url=target)
            return {"ok": True, "action": "browser-har-clear", "cleared": bool(active)}

        return self._call(clear)

    def mock_add(
        self,
        url: str,
        *,
        status: int,
        body: str,
        headers: Mapping[str, str] | None,
        abort: bool,
    ) -> dict[str, Any]:
        def add() -> dict[str, Any]:
            if not str(url).strip():
                raise ConfigError("browser mock needs a non-empty URL glob")
            if not abort and not 100 <= int(status) <= 599:
                raise ConfigError("browser mock status must be between 100 and 599")
            rule = {
                "id": f"mock-{uuid.uuid4().hex[:10]}",
                "url": str(url),
                "status": int(status),
                "body": str(body),
                "headers": {str(key): str(value) for key, value in (headers or {}).items()},
                "abort": bool(abort),
            }
            self._mock_rules.append(rule)
            return {"ok": True, "action": "browser-mock-add", "rule": dict(rule)}

        return self._call(add)

    def mock_clear(self, rule_id: str | None = None) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            before = len(self._mock_rules)
            if rule_id:
                self._mock_rules = [rule for rule in self._mock_rules if rule["id"] != rule_id]
            else:
                self._mock_rules.clear()
            removed = before - len(self._mock_rules)
            return {
                "ok": bool(removed) or rule_id is None,
                "action": "browser-mock-clear",
                "removed": removed,
                "rule_id": rule_id,
            }

        return self._call(clear)

    def diagnostics(
        self, *, limit: int, kinds: Sequence[str], since_ms: int | None
    ) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            selected = {str(kind).strip().casefold() for kind in kinds if str(kind).strip()}
            events = [
                dict(event)
                for event in self._events
                if (since_ms is None or int(event["timestamp_ms"]) >= since_ms)
                and (not selected or str(event["kind"]).casefold() in selected)
            ]
            bounded = events[-max(1, int(limit)) :]
            return {
                "ok": True,
                "action": "browser-diagnostics",
                "count": len(bounded),
                "total_count": len(events),
                "truncated": len(bounded) < len(events),
                "events": bounded,
            }

        return self._call(read)

    def diagnostics_clear(self) -> dict[str, Any]:
        def clear() -> dict[str, Any]:
            count = len(self._events)
            self._events.clear()
            self._marks.clear()
            return {"ok": True, "action": "browser-diagnostics-clear", "cleared": count}

        return self._call(clear)

    def mark_diagnostics(self, name: str, *, clear: bool = False) -> dict[str, Any]:
        def mark() -> dict[str, Any]:
            if clear:
                self._events.clear()
            timestamp = int(time.time() * 1000)
            self._marks[str(name)] = timestamp
            return {
                "ok": True,
                "action": "browser-diagnostics-mark",
                "name": str(name),
                "timestamp_ms": timestamp,
            }

        return self._call(mark)

    def pages(self) -> dict[str, Any]:
        def list_pages() -> dict[str, Any]:
            rows = []
            for index, page in enumerate(self._context.pages):
                if page.is_closed():
                    continue
                frames = [
                    {
                        "name": str(frame.name or ""),
                        "url": self._safe_url(str(frame.url)),
                        "main": frame is page.main_frame,
                    }
                    for frame in page.frames
                ]
                rows.append(
                    {
                        "id": self._page_id(page),
                        "index": index,
                        "active": page is self._page,
                        "url": self._safe_url(str(page.url)),
                        "title": page.title(),
                        "frames": frames,
                    }
                )
            return {"ok": True, "action": "browser-pages", "pages": rows}

        return self._call(list_pages)

    def _lookup_page(self, page_id: str) -> Any:
        pages = [page for page in self._context.pages if not page.is_closed()]
        if str(page_id).isdigit():
            index = int(page_id)
            if 0 <= index < len(pages):
                return pages[index]
        for page in pages:
            if self._page_id(page) == page_id:
                return page
        raise ConfigError(f"unknown browser page {page_id!r}")

    def page_select(self, page_id: str) -> dict[str, Any]:
        def select() -> dict[str, Any]:
            page = self._lookup_page(page_id)
            page.bring_to_front()
            self._page = page
            return {
                "ok": True,
                "action": "browser-page-select",
                "page_id": self._page_id(page),
                "url": self._safe_url(str(page.url)),
            }

        return self._call(select)

    def page_close(self, page_id: str) -> dict[str, Any]:
        def close_page() -> dict[str, Any]:
            page = self._lookup_page(page_id)
            closed_id = self._page_id(page)
            page.close()
            pages = [candidate for candidate in self._context.pages if not candidate.is_closed()]
            if pages:
                self._page = pages[-1]
            else:
                self._page = self._context.new_page()
                self._page.goto(self._home_url, wait_until="domcontentloaded")
            self._page.bring_to_front()
            return {
                "ok": True,
                "action": "browser-page-close",
                "closed": closed_id,
                "active": self._page_id(self._page),
            }

        return self._call(close_page)

    def trace_start(self) -> dict[str, Any]:
        def start() -> dict[str, Any]:
            if self._trace_active:
                raise ConfigError("browser tracing is already active")
            self._context.tracing.start(screenshots=True, snapshots=True, sources=True)
            self._trace_active = True
            return {"ok": True, "action": "browser-trace-start"}

        return self._call(start)

    def trace_stop(self, path: str) -> dict[str, Any]:
        def stop() -> dict[str, Any]:
            if not self._trace_active:
                raise ConfigError("browser tracing is not active")
            destination = Path(path).expanduser().resolve()
            if destination.exists():
                raise ConfigError(f"browser trace destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._context.tracing.stop(path=str(destination))
            self._trace_active = False
            return {"ok": destination.is_file(), "action": "browser-trace-stop", "path": str(destination)}

        return self._call(stop)

    def _control_state(self) -> dict[str, Any]:
        return {
            "offline": self._offline,
            "throttle": dict(self._throttle),
            "cors": [dict(rule) for rule in self._cors_rules],
            "mocks": [dict(rule) for rule in self._mock_rules],
            "proxy": dict(self._proxy) if self._proxy else None,
            "har_record": self._har_record_path,
            "har_replay": dict(self._har_replay) if self._har_replay else None,
        }

    def session_begin(self, session_id: str) -> dict[str, Any]:
        def begin() -> dict[str, Any]:
            if session_id not in self._session_baselines:
                self._session_baselines[session_id] = {
                    "storage_state": self._storage_state(),
                    "session_storage": self._session_storage(),
                    "url": str(self._page.url),
                    "controls": self._control_state(),
                }
            return {"ok": True, "action": "browser-session-begin", "session_id": session_id}

        return self._call(begin)

    def session_finish(self, session_id: str) -> dict[str, Any]:
        def finish() -> dict[str, Any]:
            baseline = self._session_baselines.pop(session_id, None)
            if baseline is None:
                return {
                    "ok": True,
                    "action": "browser-session-restore",
                    "session_id": session_id,
                    "restored": False,
                }
            controls = baseline["controls"]
            self._offline = bool(controls["offline"])
            self._throttle = dict(controls["throttle"])
            self._cors_rules = [dict(rule) for rule in controls["cors"]]
            self._mock_rules = [dict(rule) for rule in controls["mocks"]]
            self._proxy = dict(controls["proxy"]) if controls["proxy"] else None
            self._har_record_path = controls["har_record"]
            self._har_replay = (
                dict(controls["har_replay"]) if controls["har_replay"] else None
            )
            self._replace_context(
                storage_state=baseline["storage_state"],
                session_storage=baseline["session_storage"],
                target_url=baseline["url"],
            )
            return {
                "ok": True,
                "action": "browser-session-restore",
                "session_id": session_id,
                "restored": True,
            }

        return self._call(finish)

    def close(self) -> None:
        if self._closed:
            return

        def shutdown() -> None:
            try:
                if self._trace_active and self._context is not None:
                    self._context.tracing.stop()
                if self._context is not None:
                    self._context.close()
            finally:
                try:
                    self._browser.close()
                finally:
                    self._playwright.stop()

        try:
            self._call(shutdown)
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


class PlaywrightLauncher:
    def launch(self, url: str, options: WebLaunchOptions) -> WebConnection:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise DeviceError(
                "web support needs the optional Playwright dependency",
                code="web_driver_missing",
                hint="Install `android-ui-analyser[web]`, then run `playwright install chromium`.",
            ) from None

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aua-web-playwright")

        def start() -> tuple[Any, Any, Any]:
            playwright = sync_playwright().start()
            browser = None
            try:
                browser_type = getattr(playwright, options.browser, None)
                if browser_type is None:
                    raise ConfigError(
                        f"unknown web browser {options.browser!r}",
                        hint="Choose chromium, firefox, or webkit.",
                    )
                launch_kwargs: dict[str, Any] = {"headless": options.headless}
                if options.channel:
                    launch_kwargs["channel"] = options.channel
                if options.executable_path:
                    launch_kwargs["executable_path"] = options.executable_path
                try:
                    browser = browser_type.launch(**launch_kwargs)
                except Exception:
                    if options.browser != "chromium" or options.channel or options.executable_path:
                        raise
                    browser = browser_type.launch(headless=options.headless, channel="chrome")
                return playwright, browser_type, browser
            except BaseException:
                if browser is not None:
                    browser.close()
                playwright.stop()
                raise

        connection: PlaywrightConnection | None = None
        try:
            playwright, browser_type, browser = executor.submit(start).result()
            connection = PlaywrightConnection(
                executor,
                playwright,
                browser,
                browser_type,
                options,
                url,
            )
            connection.initialize()
            return connection
        except (ConfigError, DeviceError):
            if connection is not None:
                connection.close()
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        except Exception as exc:
            if connection is not None:
                connection.close()
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            raise DeviceError(
                f"could not start the web target: {exc}",
                code="web_target_unavailable",
                hint=(
                    "Run `playwright install chromium` (or configure channel: chrome), "
                    "check the URL, and retry."
                ),
            ) from None


__all__ = [
    "PlaywrightLauncher",
    "WebConnection",
    "WebLaunchOptions",
    "WebLauncher",
]
