"""Scroll-container geometry helpers over normalized UI elements.

Pure functions used by the engine's swipe/scroll paths — kept separate so the
engine module stays about orchestration. Native hierarchy grammars belong to the
selected platform adapter; this module consumes only AUA's canonical schema.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .schema import Element

logger = logging.getLogger("android_ui_analyser.scroll_geom")

Box = tuple[int, int, int, int]
Sample = tuple[tuple[str, int, int], ...]
_SCROLL_JITTER_PX = 8
_ContainerNode = tuple[str, str | None, str | None, str | None, Box]


@dataclass(frozen=True)
class ScrollProbe:
    sample: Sample
    app_id: str | None
    container: tuple[_ContainerNode, ...] | None
    roots: tuple[_ContainerNode, ...]
    controls: Sample = ()

    def same_container(self, other: ScrollProbe) -> bool:
        return (
            self.container is not None
            and self.app_id == other.app_id
            and self.container == other.container
            and self.roots == other.roots
        )


def scroll_probe(elements: Sequence[Element], box: Box, app_id: str | None) -> ScrollProbe:
    """Retain the selected container and its context from the same position sample.

    Frame-local element IDs only link ancestors within a sample. They are never compared
    across frames, and row labels are excluded from container identity so virtualization
    can replace every visible row without replacing the container.
    """
    visible = [element for element in elements if element.window not in {"system", "ime"}]
    by_id = {element.id: element for element in visible}

    def identity(element: Element) -> _ContainerNode:
        x1, y1, x2, y2 = element.bounds
        return (
            element.type,
            element.resource_id,
            element.content_desc,
            element.window,
            (x1, y1, x2, y2),
        )

    roots = tuple(
        sorted((identity(element) for element in visible if element.parent is None), key=repr)
    )
    candidates = [
        element
        for element in visible
        if element.scrollable is True and tuple(element.bounds) == box
    ]
    if len(candidates) != 1:
        return ScrollProbe(region_probe(elements, box), app_id, None, roots)
    selected = candidates[0]
    chain: list[_ContainerNode] = []
    visited: set[int | str] = set()
    node = selected
    while node.id not in visited:
        visited.add(node.id)
        chain.append(identity(node))
        if node.parent is None:
            break
        parent = by_id.get(node.parent)
        if parent is None:
            return ScrollProbe(region_probe(elements, box), app_id, None, roots)
        node = parent
    else:
        return ScrollProbe(region_probe(elements, box), app_id, None, roots)

    descendants = {selected.id}
    pending = [element for element in visible if element.id != selected.id]
    while pending:
        children = [element for element in pending if element.parent in descendants]
        if not children:
            break
        descendants.update(element.id for element in children)
        pending = [element for element in pending if element.id not in descendants]
    scoped = [element for element in visible if element.id in descendants]
    if len(scoped) == 1:
        # A flat tree cannot prove which labels belong to the list rather than an overlay.
        return ScrollProbe(region_probe(elements, box), app_id, None, roots)
    return ScrollProbe(
        region_probe(scoped, box), app_id, tuple(chain), roots, _control_probe(scoped, box)
    )


def _control_probe(elements: Sequence[Element], box: Box) -> Sample:
    """Sample named leaf controls independently of clipped/sticky text labels.

    Resource multiplicity is retained: two identically named row controls are not a
    correspondence. Size is part of the key so a clipped or resized control cannot
    supply translation evidence. These samples never assign element identities.
    """
    parents = {element.parent for element in elements}
    counts = Counter(element.resource_id for element in elements if element.resource_id)
    x1, y1, x2, y2 = box
    return tuple(
        (
            json.dumps([
                element.window, element.resource_id, element.type,
                element.text, element.content_desc,
                element.bounds[2] - element.bounds[0], element.bounds[3] - element.bounds[1],
            ]),
            element.bounds[0], element.bounds[1],
        )
        for element in elements
        if element.resource_id and counts[element.resource_id] == 1
        and element.id not in parents
        and (element.clickable or element.long_clickable or element.checkable)
        and not element.scrollable
        and not element.password
        and element.type.casefold() not in {
            "edittext", "textfield", "textbox", "input", "textarea", "searchfield",
        }
        and x1 <= element.bounds[0] < element.bounds[2] <= x2
        and y1 <= element.bounds[1] < element.bounds[3] <= y2
    )


def control_movement(before: Sample, after: Sample, direction: str) -> int:
    """Two distinct, unchanged-size controls must translate together on the swipe axis."""
    old = {key: (x, y) for key, x, y in before}
    new = {key: (x, y) for key, x, y in after}
    horizontal = direction in {"left", "right"}
    axis, cross = (0, 1) if horizontal else (1, 0)
    shifts = [
        old[key][axis] - new[key][axis]
        for key in old.keys() & new.keys()
        if abs(old[key][cross] - new[key][cross]) <= _SCROLL_JITTER_PX
        and abs(old[key][axis] - new[key][axis]) > _SCROLL_JITTER_PX
    ]
    if len(shifts) < 2:
        return 0
    distance = _median(shifts)
    return distance if all(abs(shift - distance) <= _SCROLL_JITTER_PX for shift in shifts) else 0


def _element_box(element: Element) -> Box | None:
    x1, y1, x2, y2 = element.bounds
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def scrollable_boxes(
    elements: Sequence[Element], screen: tuple[int, int] | None = None
) -> list[Box]:
    """Bounds of every canonical scrollable container, largest area first.

    A view reports ``scrollable`` only while its content overflows its viewport, so an
    empty list means *this screen has nothing to scroll anywhere* — the signal that tells
    "already at the end" apart from "the swipe missed the list".
    """
    boxes: list[Box] = []
    for element in elements:
        if element.scrollable is not True or element.window in {"system", "ime"}:
            continue
        box = _element_box(element)
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
    elements: Sequence[Element],
    box: Box,
    *,
    excluded_windows: Sequence[str] = ("system", "ime"),
) -> Sample:
    """One scroll-position sample: ordered ``(label, left, top)`` inside *box*.

    Three deliberate restrictions, each from a false reading:

    * **only inside the container** — a whole-screen sample flips when the status-bar clock
      ticks, reporting a scroll that never happened;
    * **no system/IME chrome** — same reason, for screens whose container spans the display;
    * **labelled nodes only** (text / content-desc, falling back to class names when the
      container carries no labels at all, e.g. an image grid) — layout containers keep their
      position while their contents scroll, so including them dilutes the signal.
    """
    labelled: list[tuple[str, int, int]] = []
    fallback: list[tuple[str, int, int]] = []
    excluded = set(excluded_windows)
    for element in elements:
        if element.window in excluded:
            continue
        nbox = _element_box(element)
        if nbox is None or not _contains(box, ((nbox[0] + nbox[2]) // 2, (nbox[1] + nbox[3]) // 2)):
            continue
        label = (element.text or element.content_desc or "").strip()
        if label:
            labelled.append((label, nbox[0], nbox[1]))
        else:
            fallback.append((element.type or "?", nbox[0], nbox[1]))
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


def _repeated_label_shift(before: Sample, after: Sample, direction: str) -> int:
    """Return a coherent axis shift for labels that cannot be paired uniquely.

    Unlabelled image rows and icon grids fall back to a shared class-name label. Pairing
    those occurrences in axis order is useful only when at least two pairs move by more
    than normal hierarchy jitter and the median movement is coherent.
    """
    horizontal = direction in ("left", "right")

    def axis_position(item: tuple[str, int, int]) -> int:
        return item[1] if horizontal else item[2]

    def cross_position(item: tuple[str, int, int]) -> int:
        return item[2] if horizontal else item[1]

    before_by_label: dict[str, list[tuple[str, int, int]]] = {}
    after_by_label: dict[str, list[tuple[str, int, int]]] = {}
    for item in before:
        before_by_label.setdefault(item[0], []).append(item)
    for item in after:
        after_by_label.setdefault(item[0], []).append(item)

    shifts: list[int] = []
    for label in before_by_label.keys() & after_by_label.keys():
        old = before_by_label[label]
        new = after_by_label[label]
        if len(old) < 2 or len(old) != len(new):
            continue
        old = sorted(old, key=lambda item: (axis_position(item), cross_position(item)))
        new = sorted(new, key=lambda item: (axis_position(item), cross_position(item)))
        shifts.extend(
            axis_position(old_item) - axis_position(new_item)
            for old_item, new_item in zip(old, new, strict=True)
        )

    material = [shift for shift in shifts if abs(shift) > _SCROLL_JITTER_PX]
    if len(material) < 2:
        return 0
    median = _median(material)
    same_direction = sum(1 for shift in material if shift * median > 0)
    return median if median and same_direction * 2 > len(material) else 0


def scroll_movement(
    before: Sample,
    after: Sample,
    direction: str,
    *,
    allow_content_turnover: bool = False,
) -> tuple[int, bool, str | None]:
    """Classify a post-swipe sample as ``(axis_distance, moved, evidence)``.

    A measurable shift of shared labels is the strongest evidence. Some virtualized grids keep
    sticky tabs/headings at fixed coordinates while replacing every visible card, though, so
    their shared-label median is zero even after a real scroll. Inside a hierarchy-declared
    scrollable container, removing and adding at least two labels is independent evidence of
    content turnover. An unlabelled row of icons or thumbnails falls back to its class name in
    :func:`region_probe`, so repeated occurrences are paired in axis order and accepted only
    when at least two move coherently beyond normal hierarchy jitter. Requiring multiple
    changed items keeps a one-off toast, loading badge, ripple, or layout wobble from turning
    a no-op into a successful scroll.
    """
    dx, dy, changed = travel(before, after)
    distance = dx if direction in ("left", "right") else dy
    if distance:
        return distance, True, "axis-shift"
    if not changed or not allow_content_turnover:
        return 0, False, None

    repeated_shift = _repeated_label_shift(before, after, direction)
    if repeated_shift:
        return repeated_shift, True, "axis-shift"

    before_labels = Counter(label for label, _x, _y in before)
    after_labels = Counter(label for label, _x, _y in after)
    removed = sum((before_labels - after_labels).values())
    added = sum((after_labels - before_labels).values())
    if removed >= 2 and added >= 2:
        return 0, True, "content-turnover"
    return 0, False, None
