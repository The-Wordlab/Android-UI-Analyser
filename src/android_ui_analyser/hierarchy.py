"""Android UiAutomator hierarchy XML → element list (PRD §6 step 2).

This is tier T2 of the escalation ladder: a full parse of the accessibility/view
hierarchy dumped by ``uiautomator2`` into the canonical :class:`Element` list the rest
of the engine acts on. It is pure (XML in, elements out) and device-free so it can be
golden-tested against committed fixtures (AC2).

What we keep (the "interesting" filter)
---------------------------------------
A UiAutomator dump is mostly nested layout containers (``FrameLayout``,
``LinearLayout``, ``RecyclerView`` …) that an agent can never usefully act on. We emit
an :class:`Element` for a node only when **all** of these hold:

* it has a **non-zero area** (``x2 > x1`` and ``y2 > y1``); and
* (when ``screen_size`` is given) it is **not fully off-screen** — at least part of its
  box intersects ``[0, 0, w, h]``; and
* it is **interesting**, meaning at least one of:
    - it carries non-empty ``text`` **or** ``content-desc`` (it says something), or
    - it is actionable: ``clickable`` / ``long-clickable`` / ``checkable`` /
      ``scrollable`` is ``true``, or
    - it is a **leaf** node (no element children) with non-zero area — leaves are the
      concrete drawn things (an icon, an image, a custom view) even when unlabeled.

A node that is only a non-leaf, non-actionable, text-less container is dropped: its
interesting descendants are kept in its place. This is exactly the rule the golden
``*.json`` fixtures encode — eyeball those to see it applied.

ID assignment
-------------
After filtering, the kept elements are sorted **stable top-to-bottom then
left-to-right** (key ``(y1, x1)``) and assigned ``id`` ``0..n-1`` in that order, so IDs
are deterministic and reading-order for a caller.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .schema import Bounds, Element, Source, center_of

# bounds look like "[x1,y1][x2,y2]"
_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _parse_bounds(raw: str | None) -> Bounds | None:
    """Parse a UiAutomator ``bounds="[x1,y1][x2,y2]"`` string to a 4-tuple of ints."""
    if not raw:
        return None
    m = _BOUNDS_RE.search(raw)
    if not m:
        return None
    x1, y1, x2, y2 = (int(g) for g in m.groups())
    return (x1, y1, x2, y2)


def _attr(node: ET.Element, name: str) -> str | None:
    """Return a string attribute, or ``None`` if missing/empty after stripping."""
    val = node.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _is_true(node: ET.Element, name: str) -> bool:
    """UiAutomator booleans are the literal strings ``"true"``/``"false"``."""
    return node.get(name) == "true"


def _short_type(class_name: str | None) -> str:
    """Short class name: the segment after the last ``.`` (``android.widget.Button`` → ``Button``)."""
    if not class_name:
        return ""
    return class_name.rsplit(".", 1)[-1]


def _on_screen(bounds: Bounds, screen_size: tuple[int, int] | None) -> bool:
    """True unless the box lies fully outside ``[0, 0, w, h]`` (only checked if size given)."""
    if screen_size is None:
        return True
    w, h = screen_size
    x1, y1, x2, y2 = bounds
    return not (x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h)


def _iter_nodes(root: ET.Element) -> list[ET.Element]:
    """All ``<node>`` elements anywhere under ``root`` (the ``<hierarchy>`` wrapper)."""
    return root.findall(".//node")


def parse_hierarchy(xml: str, screen_size: tuple[int, int] | None = None) -> list[Element]:
    """Parse UiAutomator hierarchy ``xml`` into a list of :class:`Element`.

    See the module docstring for the filtering and ID-assignment rules. ``screen_size``
    is ``(width, height)``; when provided, fully off-screen nodes are dropped.
    Returns an empty list if the XML is empty/blank.
    """
    if not xml or not xml.strip():
        return []
    root = ET.fromstring(xml)

    kept: list[tuple[Bounds, Element]] = []
    for node in _iter_nodes(root):
        bounds = _parse_bounds(node.get("bounds"))
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        # zero-area
        if x2 <= x1 or y2 <= y1:
            continue
        # fully off-screen
        if not _on_screen(bounds, screen_size):
            continue

        text = _attr(node, "text")
        content_desc = _attr(node, "content-desc")
        clickable = _is_true(node, "clickable")
        long_clickable = _is_true(node, "long-clickable")
        checkable = _is_true(node, "checkable")
        scrollable = _is_true(node, "scrollable")

        is_leaf = len(node.findall("node")) == 0
        actionable = clickable or long_clickable or checkable or scrollable
        interesting = bool(text) or bool(content_desc) or actionable or is_leaf
        if not interesting:
            continue

        element = Element(
            id=-1,  # assigned after sorting
            type=_short_type(node.get("class")),
            text=text,
            resource_id=_attr(node, "resource-id"),
            content_desc=content_desc,
            bounds=bounds,
            center=center_of(bounds),
            clickable=clickable,
            enabled=_is_true(node, "enabled"),
            focused=_is_true(node, "focused"),
            source=Source.hierarchy,
            confidence=None,
        )
        kept.append((bounds, element))

    # stable top-to-bottom, then left-to-right
    kept.sort(key=lambda pair: (pair[0][1], pair[0][0]))

    elements: list[Element] = []
    for new_id, (_bounds, element) in enumerate(kept):
        elements.append(element.model_copy(update={"id": new_id}))
    return elements
