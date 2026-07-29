"""Scroll-container geometry helpers over a hierarchy dump.

Pure functions used by the engine's swipe/scroll paths — kept separate so the
engine module stays about orchestration.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger("android_ui_analyser.scroll_geom")

_XML_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")
Box = tuple[int, int, int, int]
Sample = tuple[tuple[str, int, int], ...]


def _node_box(node: Any) -> Box | None:
    m = _XML_BOUNDS_RE.search(node.get("bounds") or "")
    if not m:
        return None
    x1, y1, x2, y2 = (int(g) for g in m.groups())
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _iter_nodes(xml: str) -> Any:
    """Every ``<node>`` of a hierarchy dump, in document order (empty on unparseable XML)."""
    import xml.etree.ElementTree as ET

    if not xml or not xml.strip():
        return []
    try:
        return list(ET.fromstring(xml).iter("node"))
    except ET.ParseError as exc:  # pragma: no cover - malformed dump
        logger.warning("could not parse hierarchy dump: %s", exc)
        return []


def scrollable_boxes(xml: str, screen: tuple[int, int] | None = None) -> list[Box]:
    """Bounds of every ``scrollable="true"`` container, largest area first.

    A view reports ``scrollable`` only while its content overflows its viewport, so an
    empty list means *this screen has nothing to scroll anywhere* — the signal that tells
    "already at the end" apart from "the swipe missed the list".
    """
    boxes: list[Box] = []
    for node in _iter_nodes(xml):
        if node.get("scrollable") != "true":
            continue
        box = _node_box(node)
        if box is None:
            continue
        if screen is not None:
            w, h = screen
            x1, y1, x2, y2 = box
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
        boxes.append(box)
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def _contains(box: Box, point: tuple[int, int]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def region_probe(
    xml: str, box: Box, *, ignore_packages: Sequence[str] = ("com.android.systemui",)
) -> Sample:
    """One scroll-position sample: ordered ``(label, left, top)`` inside *box*.

    Three deliberate restrictions, each from a false reading:

    * **only inside the container** — a whole-screen sample flips when the status-bar clock
      ticks, reporting a scroll that never happened;
    * **no system chrome** — same reason, for screens whose container spans the display;
    * **labelled nodes only** (text / content-desc, falling back to class names when the
      container carries no labels at all, e.g. an image grid) — layout containers keep their
      position while their contents scroll, so including them dilutes the signal.
    """
    labelled: list[tuple[str, int, int]] = []
    fallback: list[tuple[str, int, int]] = []
    for node in _iter_nodes(xml):
        if node.get("package") in ignore_packages:
            continue
        nbox = _node_box(node)
        if nbox is None or not _contains(box, ((nbox[0] + nbox[2]) // 2, (nbox[1] + nbox[3]) // 2)):
            continue
        label = (node.get("text") or node.get("content-desc") or "").strip()
        if label:
            labelled.append((label, nbox[0], nbox[1]))
        else:
            fallback.append(((node.get("class") or "?"), nbox[0], nbox[1]))
    return tuple(labelled or fallback)


def _median(values: list[int]) -> int:
    return sorted(values)[len(values) // 2]


def travel(before: Sample, after: Sample) -> tuple[int, int, bool]:
    """``(dx, dy, moved)`` between two :func:`region_probe` samples.

    ``moved`` is any change to the sample. The shift is measured only from labels that occur
    exactly once in both samples — a repeated label cannot be paired up, and pairing it by
    position order would invent a distance.
    """
    moved = before != after
    b_count = Counter(label for label, _x, _y in before)
    a_count = Counter(label for label, _x, _y in after)
    b_pos = {label: (x, y) for label, x, y in before if b_count[label] == 1}
    a_pos = {label: (x, y) for label, x, y in after if a_count[label] == 1}
    shared = b_pos.keys() & a_pos.keys()
    dx = _median([b_pos[k][0] - a_pos[k][0] for k in shared]) if shared else 0
    dy = _median([b_pos[k][1] - a_pos[k][1] for k in shared]) if shared else 0
    return dx, dy, moved
