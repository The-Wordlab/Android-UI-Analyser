"""Playwright-owned browser transport for the built-in web platform."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

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


_T = TypeVar("_T")


class PlaywrightConnection:
    """Serialize every sync-Playwright call onto the thread that created the browser.

    MCP dispatch is async. Playwright's synchronous API refuses to start inside an asyncio loop,
    and its objects are thread-affine, so merely moving launch to a worker is insufficient. One
    single-thread executor owns creation and every later operation for the connection.
    """

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        playwright: Any,
        browser: Any,
        context: Any,
        page: Any,
    ) -> None:
        self._executor = executor
        self._playwright = playwright
        self._browser = browser
        self._context = context
        self._page = page
        self._closed = False

    def _call(self, operation: Callable[[], _T]) -> _T:
        if self._closed:
            raise DeviceError("the web target is closed", code="web_target_closed")
        return self._executor.submit(operation).result()

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
        return self._call(
            lambda: json.dumps(self._page.evaluate(_DOM_SNAPSHOT_SCRIPT), ensure_ascii=False)
        )

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

    def close(self) -> None:
        if self._closed:
            return

        def shutdown() -> None:
            try:
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

        def start() -> tuple[Any, Any, Any, Any]:
            playwright = sync_playwright().start()
            browser = None
            context = None
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
                    # A source/plugin install may have Playwright but not its downloaded browser.
                    # Installed Google Chrome is a safe compatible fallback and avoids turning
                    # first use into an implicit 150 MB download.
                    browser = browser_type.launch(headless=options.headless, channel="chrome")
                context_kwargs: dict[str, Any] = {
                    "viewport": {
                        "width": options.viewport_width,
                        "height": options.viewport_height,
                    },
                    "device_scale_factor": 1,
                    "ignore_https_errors": options.ignore_https_errors,
                }
                if options.storage_state:
                    state = Path(options.storage_state).expanduser()
                    if not state.is_file():
                        raise ConfigError(f"web storage_state does not exist: {state}")
                    context_kwargs["storage_state"] = str(state)
                context = browser.new_context(**context_kwargs)
                context.set_default_timeout(options.action_timeout_ms)
                context.set_default_navigation_timeout(options.navigation_timeout_ms)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded")
                return playwright, browser, context, page
            except BaseException:
                if context is not None:
                    context.close()
                if browser is not None:
                    browser.close()
                playwright.stop()
                raise

        try:
            playwright, browser, context, page = executor.submit(start).result()
            return PlaywrightConnection(executor, playwright, browser, context, page)
        except (ConfigError, DeviceError):
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        except Exception as exc:
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
