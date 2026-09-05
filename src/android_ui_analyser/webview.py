"""WebView enrichment — try DOM/a11y children before escalating to OCR/vision.

Hollow WebViews (a single unlabeled ``WebView`` node) normally trip the soft gate into
vision. Before that paid/slow path, we:

1. Re-walk the raw hierarchy for **descendants inside WebView** that the interesting-
   filter skipped (unlabeled containers, nested text).
2. Optionally evaluate a tiny JS snippet via ``adb`` + Chrome DevTools Protocol when the
   app has set ``WebView.setWebContentsDebuggingEnabled(true)`` (best-effort).

Elements produced here get ``source=webview`` so agents can tell them apart from native
hierarchy nodes.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .hierarchy import _attr, _is_true, _parse_bounds, _short_type
from .identity import attach_stable_keys
from .schema import Element, Source, center_of

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .platforms.runtime import TargetRuntime

_WEBVIEW_CLASS = re.compile(r"(?:^|\.)WebView$", re.I)
_ATTR_RE = re.compile(r"""([\w:-]+)\s*=\s*["']([^"']*)["']""")
_TAG_RE = re.compile(
    r"<(?P<tag>a|button|input|textarea|select|label|h[1-6]|p|span|div|li)\b(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)</(?P=tag)>|"
    r"<(?P<void>input|img|br)\b(?P<void_attrs>[^>]*)/?>",
    re.I | re.S,
)


def webview_nodes(xml: str) -> list[ET.Element]:
    """Top-level WebView nodes in a hierarchy dump."""
    if not xml or not xml.strip():
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    return [n for n in root.iter("node") if _WEBVIEW_CLASS.search(n.get("class") or "")]


def enrich_from_hierarchy(xml: str, screen_size: tuple[int, int] | None = None) -> list[Element]:
    """Pull labeled/actionable descendants out of WebView subtrees the main filter missed."""
    collected: list[Element] = []
    for wv in webview_nodes(xml):
        for node in wv.iter("node"):
            if node is wv:
                continue
            bounds = _parse_bounds(node.get("bounds"))
            if bounds is None:
                continue
            if screen_size is not None:
                w, h = screen_size
                x1, y1, x2, y2 = bounds
                if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                    continue
            text = _attr(node, "text")
            desc = _attr(node, "content-desc")
            clickable = _is_true(node, "clickable") or _is_true(node, "long-clickable")
            if not (text or desc or clickable):
                continue
            collected.append(
                Element(
                    id=-1,
                    type=_short_type(node.get("class")),
                    text=text,
                    resource_id=_attr(node, "resource-id"),
                    content_desc=desc,
                    bounds=bounds,
                    center=center_of(bounds),
                    clickable=clickable,
                    enabled=_is_true(node, "enabled"),
                    focused=_is_true(node, "focused"),
                    source=Source.webview,
                    confidence=None,
                )
            )
    collected.sort(key=lambda el: (el.bounds[1], el.bounds[0]))
    return attach_stable_keys(
        [el.model_copy(update={"id": i}) for i, el in enumerate(collected)]
    )


def parse_dom_html(
    html: str,
    *,
    frame: tuple[int, int, int, int],
    start_id: int = 0,
) -> list[Element]:
    """Best-effort HTML → elements laid out in a grid inside *frame* bounds."""
    if not html or not html.strip():
        return []
    x1, y1, x2, y2 = frame
    height = max(1, y2 - y1)
    items: list[tuple[str, str | None, bool]] = []
    for m in _TAG_RE.finditer(html):
        tag = (m.group("tag") or m.group("void") or "").lower()
        attrs_s = m.group("attrs") or m.group("void_attrs") or ""
        body = re.sub(r"<[^>]+>", " ", m.group("body") or "")
        body = re.sub(r"\s+", " ", body).strip()
        attrs = dict(_ATTR_RE.findall(attrs_s))
        label = (
            body
            or attrs.get("aria-label")
            or attrs.get("alt")
            or attrs.get("placeholder")
            or attrs.get("name")
            or attrs.get("value")
            or attrs.get("href")
        )
        if not label:
            continue
        clickable = tag in {"a", "button", "input", "textarea", "select"} or "onclick" in attrs
        items.append((tag.capitalize(), label[:120], clickable))
        if len(items) >= 40:
            break
    if not items:
        return []
    n = len(items)
    row_h = max(24, height // max(1, min(n, 12)))
    elements: list[Element] = []
    for i, (typ, label, clickable) in enumerate(items):
        top = y1 + (i % max(1, height // row_h)) * row_h
        box = (x1 + 8, top, x2 - 8, min(y2, top + row_h - 4))
        if box[3] <= box[1]:
            continue
        elements.append(
            Element(
                id=start_id + len(elements),
                type=typ,
                text=label,
                resource_id=None,
                content_desc=None,
                bounds=box,
                center=center_of(box),
                clickable=clickable,
                enabled=True,
                focused=False,
                source=Source.webview,
                confidence=0.5,
            )
        )
    return attach_stable_keys(elements)


def try_cdp_dom(
    shell: Callable[[str], str],
    *,
    frame: tuple[int, int, int, int],
    timeout_s: float = 1.5,
) -> list[Element]:
    """Fetch ``document.body.innerText``-ish HTML via CDP if port 9222 is open.

    Requires the app to enable WebView debugging. Failures return ``[]`` silently.
    """
    del timeout_s  # reserved for future httpx timeout wiring
    try:
        # Discover a page target; empty → debugging off.
        raw = shell(
            "curl -s --max-time 1 http://127.0.0.1:9222/json/list 2>/dev/null || true"
        )
        if not raw.strip().startswith("["):
            # Device-side: often need adb forward; try host-side via empty.
            return []
        pages = json.loads(raw)
        if not isinstance(pages, list) or not pages:
            return []
        # Prefer a page whose URL looks like content (not chrome-error).
        page = next(
            (p for p in pages if isinstance(p, dict) and "webSocketDebuggerUrl" in p),
            None,
        )
        if page is None:
            return []
        # Without a websocket client we can't run Runtime.evaluate; fall back to
        # title/url as a single labeled element so agents at least see *something*.
        title = (page.get("title") or "").strip() or None
        url = (page.get("url") or "").strip() or None
        label = title or (urlparse(url).path if url else None) or url
        if not label:
            return []
        return parse_dom_html(
            f"<a href='{url or '#'}'>{label}</a>",
            frame=frame,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("webview CDP probe failed: %s", exc)
        return []


def enrich(
    xml: str,
    *,
    runtime: TargetRuntime | None = None,
    screen_size: tuple[int, int] | None = None,
    shell: Callable[[str], str] | None = None,
    cdp: bool = False,
) -> list[Element]:
    """Best-effort Android WebView elements; empty when nothing useful is found.

    ``runtime`` is the neutral service-contract input used by the engine. Resolving Android's
    native shell surface happens here, inside the Android-only service loaded by
    :class:`AndroidPlatform`. ``shell`` remains as a compatibility/testing seam for direct
    callers and is never supplied by generic engine code.
    """
    elements = enrich_from_hierarchy(xml, screen_size=screen_size)
    if elements:
        return elements
    if shell is None and runtime is not None:
        candidate: Any = getattr(runtime, "shell", None)
        if callable(candidate):
            def native_shell(command: str) -> str:
                return str(candidate(command))

            shell = native_shell
    if not cdp or shell is None:
        return []
    for wv in webview_nodes(xml):
        bounds = _parse_bounds(wv.get("bounds"))
        if bounds is None:
            continue
        found = try_cdp_dom(shell, frame=bounds)
        if found:
            return found
    return []


def should_try_webview(elements: list[Element], xml: str) -> bool:
    """True when the screen looks like a hollow/weak WebView worth enriching."""
    if webview_nodes(xml):
        labeled = sum(1 for e in elements if e.text or e.content_desc)
        if len(elements) < 8 or (elements and labeled / max(1, len(elements)) < 0.3):
            return True
    return False


__all__ = [
    "enrich",
    "enrich_from_hierarchy",
    "parse_dom_html",
    "should_try_webview",
    "try_cdp_dom",
    "webview_nodes",
]
