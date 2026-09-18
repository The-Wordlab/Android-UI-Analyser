"""Browser DOM snapshot -> AUA's canonical element list.

The browser transport collects only rendered, viewport-intersecting nodes. Keeping the parser
pure makes web hierarchy behavior testable without Playwright or a running browser.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from ..errors import UsageError
from ..identity import attach_stable_keys
from ..schema import Bounds, Element, MatchMode, Source, center_of
from .base import NormalizedTree

TREE_FORMAT = "aua-web-dom/1"
_BY_TOKENS = frozenset({"text", "id", "rid", "desc"})

_TYPE_BY_ROLE = {
    "button": "Button",
    "checkbox": "Checkbox",
    "combobox": "ComboBox",
    "dialog": "Dialog",
    "link": "Link",
    "list": "List",
    "listbox": "List",
    "menuitem": "MenuItem",
    "option": "Option",
    "radio": "RadioButton",
    "searchbox": "SearchField",
    "slider": "Slider",
    "spinbutton": "Stepper",
    "switch": "Switch",
    "tab": "Tab",
    "textbox": "TextField",
}
_TYPE_BY_TAG = {
    "a": "Link",
    "button": "Button",
    "input": "TextField",
    "select": "ComboBox",
    "textarea": "TextView",
}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    compact = re.sub(r"\s+", " ", str(value)).strip()
    return compact[:500] or None


def parse_snapshot(raw_tree: str) -> dict[str, Any]:
    if not raw_tree or not raw_tree.strip():
        return {"format": TREE_FORMAT, "nodes": []}
    payload = json.loads(raw_tree)
    if not isinstance(payload, dict):
        return {"format": TREE_FORMAT, "nodes": []}
    nodes = payload.get("nodes")
    payload["nodes"] = (
        [node for node in nodes if isinstance(node, dict)] if isinstance(nodes, list) else []
    )
    return payload


def _bounds(node: dict[str, Any], screen_size: tuple[int, int]) -> Bounds | None:
    raw = node.get("bounds")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in raw)
    except (TypeError, ValueError):
        return None
    width, height = screen_size
    clipped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _type_name(node: dict[str, Any]) -> str:
    role = (_text(node.get("role")) or "").casefold()
    tag = (_text(node.get("tag")) or "").casefold()
    input_type = (_text(node.get("input_type")) or "").casefold()
    if tag == "input":
        return {
            "checkbox": "Checkbox",
            "radio": "RadioButton",
            "range": "Slider",
            "search": "SearchField",
        }.get(input_type, "TextField")
    return (
        _TYPE_BY_ROLE.get(role)
        or _TYPE_BY_TAG.get(tag)
        or (role.title() if role else tag.title() or "Element")
    )


def app_id(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    return parsed.hostname or parsed.scheme or None


def normalize(
    raw_tree: str,
    screen_size: tuple[int, int],
    *,
    ignored_app_ids: Sequence[str] = (),
) -> NormalizedTree:
    payload = parse_snapshot(raw_tree)
    ignored = {value.casefold() for value in ignored_app_ids}
    collected: list[tuple[int, Element, int | None]] = []
    source_to_slot: dict[int, int] = {}
    for source_index, node in enumerate(payload["nodes"]):
        bounds = _bounds(node, screen_size)
        if bounds is None:
            continue
        slot = len(collected)
        source_to_slot[source_index] = slot
        collected.append(
            (
                source_index,
                Element(
                    id=slot,
                    type=_type_name(node),
                    text=_text(node.get("text")),
                    resource_id=_text(node.get("resource_id")),
                    content_desc=_text(node.get("description")),
                    bounds=bounds,
                    center=center_of(bounds),
                    clickable=bool(node.get("clickable")),
                    enabled=bool(node.get("enabled", True)),
                    focused=bool(node.get("focused")),
                    checkable=node.get("checkable")
                    if isinstance(node.get("checkable"), bool)
                    else None,
                    checked=node.get("checked") if isinstance(node.get("checked"), bool) else None,
                    selected=node.get("selected")
                    if isinstance(node.get("selected"), bool)
                    else None,
                    scrollable=node.get("scrollable")
                    if isinstance(node.get("scrollable"), bool)
                    else None,
                    long_clickable=bool(node.get("clickable")),
                    password=node.get("password")
                    if isinstance(node.get("password"), bool)
                    else None,
                    source=Source.hierarchy,
                ),
                int(node["parent"]) if isinstance(node.get("parent"), int) else None,
            )
        )

    elements: list[Element] = []
    for new_id, (_source_index, element, parent_source) in enumerate(collected):
        parent_slot = source_to_slot.get(parent_source) if parent_source is not None else None
        elements.append(element.model_copy(update={"id": new_id, "parent": parent_slot}))
    elements = attach_stable_keys(elements)
    current = app_id(_text(payload.get("url")))
    if current and current.casefold() in ignored:
        current = None
    return NormalizedTree(elements=elements, app_id=current)


def find_bounds(
    raw_tree: str,
    *,
    screen_size: tuple[int, int],
    query: str,
    match: MatchMode | str = MatchMode.contains,
    ignore_case: bool = False,
    by: str = "text",
) -> Bounds | None:
    field = (by or "text").lower()
    if field not in _BY_TOKENS:
        raise UsageError(
            f"unknown selector field 'by={field}'",
            hint="Choose one of: " + ", ".join(sorted(_BY_TOKENS)) + ".",
        )
    mode = MatchMode(match)
    pattern = (
        re.compile(query, re.IGNORECASE if ignore_case else 0) if mode is MatchMode.regex else None
    )
    wanted = query.casefold() if ignore_case else query

    def matches(candidate: str | None) -> bool:
        if candidate is None:
            return False
        if pattern is not None:
            return pattern.search(candidate) is not None
        subject = candidate.casefold() if ignore_case else candidate
        return subject == wanted if mode is MatchMode.exact else wanted in subject

    payload = parse_snapshot(raw_tree)
    for node in payload["nodes"]:
        candidates: tuple[str | None, ...]
        if field in {"id", "rid"}:
            candidates = (_text(node.get("resource_id")),)
        elif field == "desc":
            candidates = (_text(node.get("description")),)
        else:
            candidates = (_text(node.get("text")), _text(node.get("description")))
        if any(matches(candidate) for candidate in candidates):
            bounds = _bounds(node, screen_size)
            if bounds is not None:
                return bounds
    return None


__all__ = ["TREE_FORMAT", "app_id", "find_bounds", "normalize", "parse_snapshot"]
