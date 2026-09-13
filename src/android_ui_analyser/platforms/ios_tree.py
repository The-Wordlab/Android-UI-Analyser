"""AXe accessibility JSON -> AUA's canonical element list.

Pure: JSON in, elements out. The runtime wraps the raw ``axe describe-ui`` output in a small
envelope that also names the bundle id of every root process, because the tree itself only
knows pids. The same interesting/absorb/roll-up rules as the Android parser apply, so an agent
gets one labelled, tappable id per row on either platform:

* keep a node when it is actionable, scrollable, carries its own label, has an accessibility
  identifier, or is a drawn leaf with non-zero area;
* label an unlabelled actionable node from its subtree;
* absorb non-actionable descendants of an actionable node (decorative images inside a button
  are absorbed even when they carry an identifier - SF Symbol names are not controls).

Frames are logical points; every published bound goes through :class:`DisplayGeometry` so the
engine sees screenshot pixels.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

from ..errors import UsageError
from ..identity import attach_stable_keys
from ..schema import Bounds, Element, MatchMode, Source, center_of
from .base import NormalizedTree
from .geometry import DisplayGeometry
from .ios_tools import SYSTEM_APP_IDS

TREE_FORMAT = "aua-ios-ax/1"

_INTERACTIVE_TYPES = frozenset(
    {
        "Button",
        "Cell",
        "Checkbox",
        "ColorWell",
        "ComboBox",
        "DatePicker",
        "DisclosureTriangle",
        "Incrementor",
        "Key",
        "Link",
        "MenuButton",
        "MenuItem",
        "Picker",
        "PickerWheel",
        "PopUpButton",
        "RadioButton",
        "SearchField",
        "SecureTextField",
        "SegmentedControl",
        "Slider",
        "Stepper",
        "Switch",
        "Tab",
        "TextField",
        "TextView",
        "Toggle",
        "ToolbarButton",
    }
)
_INTERACTIVE_ROLES = frozenset(
    {
        "AXButton",
        "AXCell",
        "AXCheckBox",
        "AXComboBox",
        "AXDisclosureTriangle",
        "AXIncrementor",
        "AXLink",
        "AXMenuItem",
        "AXPopUpButton",
        "AXRadioButton",
        "AXSearchField",
        "AXSecureTextField",
        "AXSlider",
        "AXSwitch",
        "AXTextArea",
        "AXTextField",
    }
)
_INTERACTIVE_TRAITS = frozenset({"Button", "Link", "Adjustable", "KeyboardKey"})
_SCROLLABLE_TYPES = frozenset(
    {"CollectionView", "List", "Outline", "ScrollArea", "ScrollView", "Table", "WebView"}
)
_SCROLLABLE_ROLES = frozenset({"AXList", "AXOutline", "AXScrollArea", "AXTable", "AXWebArea"})
_TEXT_ENTRY_TYPES = frozenset(
    {"ComboBox", "SearchField", "SecureTextField", "TextField", "TextView"}
)
_TOGGLE_TYPES = frozenset({"Checkbox", "RadioButton", "Switch", "Toggle"})
_KEYBOARD_TYPES = frozenset({"Keyboard"})
# Layout containers: never drawn content on their own, so an empty one is not a leaf worth an id,
# and the application root's label is the app's name rather than anything on screen.
_CONTAINER_TYPES = frozenset({"Application", "Group", "Other", "Window", "GenericElement"})
_TRUE_VALUES = frozenset({"1", "on", "true", "yes", "checked", "selected"})
_FALSE_VALUES = frozenset({"0", "off", "false", "no", "unchecked", "unselected"})
_MAX_LABEL = 120


def envelope(roots: list[dict[str, Any]], apps: dict[int, str]) -> str:
    """Serialize raw AXe roots plus the pid -> bundle map the runtime resolved."""

    return json.dumps(
        {
            "format": TREE_FORMAT,
            "roots": roots,
            "apps": {str(pid): bundle for pid, bundle in apps.items()},
        }
    )


def parse_envelope(raw_tree: str) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Accept the envelope or bare ``axe describe-ui`` output (a list or one node)."""

    if not raw_tree or not raw_tree.strip():
        return [], {}
    payload = json.loads(raw_tree)
    if isinstance(payload, list):
        return [node for node in payload if isinstance(node, dict)], {}
    if not isinstance(payload, dict):
        return [], {}
    if payload.get("format") == TREE_FORMAT or "roots" in payload:
        roots = [node for node in (payload.get("roots") or []) if isinstance(node, dict)]
        apps: dict[int, str] = {}
        for pid, bundle in (payload.get("apps") or {}).items():
            if str(pid).isdigit() and bundle:
                apps[int(pid)] = str(bundle)
        return roots, apps
    return [payload], {}


def root_app_ids(roots: Sequence[dict[str, Any]], apps: dict[int, str]) -> list[str | None]:
    return [
        apps.get(int(root["pid"])) if str(root.get("pid", "")).isdigit() else None for root in roots
    ]


def foreground_app_id(
    roots: Sequence[dict[str, Any]],
    apps: dict[int, str],
    ignored_app_ids: Sequence[str] = (),
) -> str | None:
    """The first root process that is not an ignored app, else the first root at all."""

    ignored = {item.casefold() for item in ignored_app_ids}
    candidates = [app_id for app_id in root_app_ids(roots, apps) if app_id]
    for app_id in candidates:
        if app_id.casefold() not in ignored:
            return app_id
    return candidates[0] if candidates else None


def _frame(node: dict[str, Any]) -> tuple[float, float, float, float] | None:
    frame = node.get("frame")
    if not isinstance(frame, dict):
        return None
    try:
        x = float(frame.get("x", 0.0))
        y = float(frame.get("y", 0.0))
        width = float(frame.get("width", 0.0))
        height = float(frame.get("height", 0.0))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (x, y, x + width, y + height)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _type_name(node: dict[str, Any]) -> str:
    type_name = _text(node.get("type"))
    if type_name:
        return type_name
    role = _text(node.get("role")) or ""
    return role[2:] if role.startswith("AX") and len(role) > 2 else (role or "Other")


def _is_actionable(node: dict[str, Any], type_name: str) -> bool:
    if type_name in _INTERACTIVE_TYPES or _text(node.get("role")) in _INTERACTIVE_ROLES:
        return True
    return any(trait in _INTERACTIVE_TRAITS for trait in _traits(node))


def _is_scrollable(node: dict[str, Any], type_name: str) -> bool:
    return type_name in _SCROLLABLE_TYPES or _text(node.get("role")) in _SCROLLABLE_ROLES


def _checked(value: str | None) -> bool | None:
    if value is None:
        return None
    folded = value.casefold()
    if folded in _TRUE_VALUES:
        return True
    if folded in _FALSE_VALUES:
        return False
    return None


def _on_screen(bounds: Bounds, screen_size: tuple[int, int]) -> bool:
    width, height = screen_size
    x1, y1, x2, y2 = bounds
    return not (x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height)


def _traits(node: dict[str, Any]) -> list[str]:
    traits = node.get("traits")
    return [str(trait) for trait in traits] if isinstance(traits, list) else []


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("children")
    return (
        [child for child in children if isinstance(child, dict)]
        if isinstance(children, list)
        else []
    )


def _iter_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in _children(node):
        yield from _iter_nodes(child)


def _own_labels(node: dict[str, Any], type_name: str) -> tuple[str | None, str | None]:
    """``(text, content_desc)`` in AUA's vocabulary.

    Text entries show their contents (or placeholder) as ``AXValue`` and their caption as
    ``AXLabel``. Anything else with a distinct ``AXValue`` (a slider's percentage, a segmented
    control's selection) follows the same rule, value as text and caption as description, so
    stable keys and recorded selectors follow the caption rather than the changing state.
    """

    label = _text(node.get("AXLabel")) or _text(node.get("title"))
    value = _text(node.get("AXValue"))
    if type_name in _TEXT_ENTRY_TYPES:
        return (value or label, label if value else None)
    if type_name in _TOGGLE_TYPES:
        return (label, None)
    if value and value != label:
        return (value, label)
    return (label, None)


def _gather_text(node: dict[str, Any]) -> str | None:
    parts: list[str] = []
    seen: set[str] = set()
    for descendant in _iter_nodes(node):
        type_name = _type_name(descendant)
        text, desc = _own_labels(descendant, type_name)
        for candidate in (text, desc):
            if candidate and candidate.casefold() not in seen:
                seen.add(candidate.casefold())
                parts.append(candidate)
    label = " ".join(parts).strip()
    return label[:_MAX_LABEL] if label else None


def normalize(
    raw_tree: str,
    screen_size: tuple[int, int],
    *,
    geometry: DisplayGeometry,
    ignored_app_ids: Sequence[str] = (),
) -> NormalizedTree:
    roots, apps = parse_envelope(raw_tree)
    collected: list[tuple[Bounds, Element, int | None]] = []

    def visit(
        node: dict[str, Any],
        *,
        actionable_ancestor: bool,
        parent_slot: int | None,
        window: str,
    ) -> None:
        type_name = _type_name(node)
        if type_name in _KEYBOARD_TYPES:
            window = "ime"
        native = _frame(node)
        bounds = geometry.bounds_to_canonical(native) if native is not None else None
        valid = (
            bounds is not None
            and bounds[2] > bounds[0]
            and bounds[3] > bounds[1]
            and _on_screen(bounds, screen_size)
        )
        text, content_desc = _own_labels(node, type_name)
        identifier = _text(node.get("AXUniqueId"))
        actionable = _is_actionable(node, type_name)
        scrollable = _is_scrollable(node, type_name)
        children = _children(node)
        is_leaf = not children
        container = type_name in _CONTAINER_TYPES
        has_own_label = (bool(text) or bool(content_desc)) and type_name != "Application"
        drawn_leaf = is_leaf and not container
        interesting = actionable or scrollable or has_own_label or drawn_leaf or bool(identifier)
        decorative = type_name == "Image"
        absorbed = (
            actionable_ancestor
            and not actionable
            and not scrollable
            and (not identifier or decorative)
        )

        own_slot = parent_slot
        if valid and interesting and not absorbed:
            assert bounds is not None
            own_slot = len(collected)
            label = text
            if actionable and not has_own_label:
                label = _gather_text(node)
            checkable = type_name in _TOGGLE_TYPES
            traits = _traits(node)
            collected.append(
                (
                    bounds,
                    Element(
                        id=-1,
                        type=type_name,
                        text=label,
                        resource_id=identifier,
                        content_desc=content_desc,
                        bounds=bounds,
                        center=center_of(bounds),
                        clickable=actionable,
                        enabled=bool(node.get("enabled", True)),
                        focused="Focused" in traits,
                        source=Source.hierarchy,
                        window=window,
                        checkable=checkable or None,
                        checked=_checked(_text(node.get("AXValue"))) if checkable else None,
                        selected=True if "Selected" in traits else None,
                        scrollable=True if scrollable else None,
                        password=True if type_name == "SecureTextField" else None,
                    ),
                    parent_slot,
                )
            )
        for child in children:
            visit(
                child,
                actionable_ancestor=actionable_ancestor or actionable,
                parent_slot=own_slot,
                window=window,
            )

    for root, app_id in zip(roots, root_app_ids(roots, apps), strict=True):
        window = "system" if (app_id or "").casefold() in SYSTEM_APP_IDS else "app"
        visit(root, actionable_ancestor=False, parent_slot=None, window=window)

    order = sorted(range(len(collected)), key=lambda i: (collected[i][0][1], collected[i][0][0]))
    slot_to_id = {slot: new_id for new_id, slot in enumerate(order)}

    def renumbered(new_id: int, slot: int) -> Element:
        parent_slot = collected[slot][2]
        parent = slot_to_id.get(parent_slot) if parent_slot is not None else None
        return collected[slot][1].model_copy(update={"id": new_id, "parent": parent})

    elements = attach_stable_keys([renumbered(new_id, slot) for new_id, slot in enumerate(order)])
    return NormalizedTree(elements=elements, app_id=foreground_app_id(roots, apps, ignored_app_ids))


# The engine spells a resource id `rid` on the wait path (`id:`/`rid:` both map to it), so both
# tokens name the identifier; anything else is refused rather than degraded to a text search.
_BY_TOKENS = frozenset({"text", "id", "rid", "desc"})


def find_bounds(
    raw_tree: str,
    *,
    geometry: DisplayGeometry,
    query: str,
    match: MatchMode | str = MatchMode.contains,
    ignore_case: bool = False,
    by: str = "text",
) -> Bounds | None:
    """Canonical bounds of the first node whose text/identifier/description matches."""

    by = (by or "text").lower()
    if by not in _BY_TOKENS:
        raise UsageError(
            f"unknown selector field 'by={by}'",
            hint="Choose one of: " + ", ".join(sorted(_BY_TOKENS)) + ".",
        )
    roots, _apps = parse_envelope(raw_tree)
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

    for root in roots:
        for node in _iter_nodes(root):
            type_name = _type_name(node)
            text, desc = _own_labels(node, type_name)
            if by in ("id", "rid"):
                candidates: tuple[str | None, ...] = (_text(node.get("AXUniqueId")),)
            elif by == "desc":
                candidates = (_text(node.get("AXLabel")), desc)
            else:
                candidates = (text, _text(node.get("AXLabel")), _text(node.get("AXValue")))
            if not any(matches(candidate) for candidate in candidates):
                continue
            native = _frame(node)
            if native is None:
                continue
            bounds = geometry.bounds_to_canonical(native)
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                return bounds
    return None


def root_frame_size(roots: Sequence[dict[str, Any]]) -> tuple[float, float] | None:
    """The largest root frame in points: the simulator's logical screen size."""

    best: tuple[float, float] | None = None
    for root in roots:
        native = _frame(root)
        if native is None:
            continue
        size = (native[2] - native[0], native[3] - native[1])
        if best is None or size[0] * size[1] > best[0] * best[1]:
            best = size
    return best


__all__ = [
    "TREE_FORMAT",
    "envelope",
    "find_bounds",
    "foreground_app_id",
    "normalize",
    "parse_envelope",
    "root_app_ids",
    "root_frame_size",
]
