"""Android UiAutomator hierarchy XML → element list (PRD §6 step 2).

This is tier T2 of the escalation ladder: a full parse of the accessibility/view
hierarchy dumped by ``uiautomator2`` into the canonical :class:`Element` list the rest
of the engine acts on. It is pure (XML in, elements out) and device-free so it can be
golden-tested against committed fixtures (AC2).

What we keep (the "interesting" filter) + text roll-up
------------------------------------------------------
A UiAutomator dump is mostly nested layout containers (``FrameLayout``,
``LinearLayout``, ``RecyclerView`` …) that an agent can never usefully act on. We emit
an :class:`Element` for a node only when **all** of these hold:

* it has a **non-zero area** (``x2 > x1`` and ``y2 > y1``); and
* (when ``screen_size`` is given) it is **not fully off-screen** — at least part of its
  box intersects ``[0, 0, w, h]``; and
* it is **interesting**, meaning at least one of:
    - it carries non-empty ``text`` **or** ``content-desc`` (it says something), or
    - it is **actionable**: ``clickable`` / ``long-clickable`` / ``checkable``, or
    - it is a **leaf** node (no element children) with non-zero area — leaves are the
      concrete drawn things (an icon, an image, a custom view) even when unlabeled.

**Roll-up + absorption (Set-of-Marks cleanliness).** Android list rows put the label on
inner ``TextView``s while the *clickable* element is the parent container. So when an
actionable node has no own ``text``/``content-desc``, we label it by joining the text of
its whole subtree. Non-actionable descendants of an actionable node are then **absorbed**
(not emitted separately) — the agent gets exactly one labelled, tappable id per row
instead of an unlabelled clickable container plus loose text. Nested actionable elements
are still kept. This is the rule the golden ``*.json`` fixtures encode — eyeball them.

ID assignment
-------------
After filtering, the kept elements are sorted **stable top-to-bottom then
left-to-right** (key ``(y1, x1)``) and assigned ``id`` ``0..n-1`` in that order, so IDs
are deterministic and reading-order for a caller.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from .schema import Bounds, Element, Source, center_of

# bounds look like "[x1,y1][x2,y2]"
_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

_LXML: Any | None
try:  # optional C-speed parser (perf.lxml / pip install lxml)
    from lxml import etree as _LXML  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - optional dep
    _LXML = None


def _parse_xml_root(xml: str) -> Any:
    """Parse hierarchy XML; prefer lxml when available."""
    if _LXML is not None:
        try:
            return _LXML.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
        except Exception:  # noqa: BLE001 — fall back to stdlib
            pass
    return ET.fromstring(xml)


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


def _tri(node: ET.Element, name: str) -> bool | None:
    """Tri-state boolean: ``None`` when the node never reported the attribute.

    A caller reading an interaction state (is this switch on?) must be able to tell
    *off* from *unknown*, so an absent attribute must not collapse to ``False``.
    """
    raw = node.get(name)
    if raw is None:
        return None
    return raw == "true"


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


_MAX_LABEL = 120  # cap on a rolled-up label so a big clickable card can't dump its whole subtree


def _gather_descendant_text(node: ET.Element) -> str | None:
    """Join ``text`` + ``content-desc`` from a node's whole subtree, in document order.

    Used to *label a clickable container from its children* — Android list rows put the
    label on inner ``TextView``s while the clickable element is the parent. De-dupes
    case-insensitively and caps length.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for desc in node.iter("node"):
        for attr in ("text", "content-desc"):
            raw = desc.get(attr)
            if not raw:
                continue
            val = raw.strip()
            if val and val.lower() not in seen:
                seen.add(val.lower())
                parts.append(val)
    label = " ".join(parts).strip()
    return label[:_MAX_LABEL] if label else None


def classify_window(package: str | None) -> str | None:
    """Map an a11y node ``package`` to a window layer for agents (``--no-ime`` etc.)."""
    if not package:
        return None
    p = package.lower()
    if "inputmethod" in p or p.endswith(".ime") or "keyboard" in p:
        return "ime"
    if "systemui" in p or p in {
        "com.android.launcher3",
        "com.android.launcher",
        "com.google.android.apps.nexuslauncher",
    }:
        return "system"
    if (
        p in {"android", "com.android.intentresolver", "com.android.internal.app"}
        or "permissioncontroller" in p
    ):
        return "overlay"
    return "app"


def parse_hierarchy(xml: str, screen_size: tuple[int, int] | None = None) -> list[Element]:
    """Parse UiAutomator hierarchy ``xml`` into a list of :class:`Element`.

    See the module docstring for the filtering, roll-up, and ID-assignment rules.
    ``screen_size`` is ``(width, height)``; when provided, fully off-screen nodes are
    dropped. Returns an empty list if the XML is empty/blank.

    Prefers ``lxml`` when installed (optional extra) for faster parsing of large dumps;
    falls back to stdlib :mod:`xml.etree.ElementTree`.
    """
    if not xml or not xml.strip():
        return []
    root = _parse_xml_root(xml)

    # (bounds, element, slot of the nearest collected ancestor). The slot is the index in this
    # list, i.e. pre-sort order; it is remapped to the final ids below. Parentage has to be
    # recorded here because it is not recoverable afterwards: the acting control of a
    # design-system tile is a sibling of the label that names it, outside its bounds.
    collected: list[tuple[Bounds, Element, int | None]] = []

    def visit(node: Any, actionable_ancestor: bool, parent_slot: int | None = None) -> None:
        bounds = _parse_bounds(node.get("bounds"))
        valid = (
            bounds is not None
            and bounds[2] > bounds[0]
            and bounds[3] > bounds[1]
            and _on_screen(bounds, screen_size)
        )
        text = _attr(node, "text")
        content_desc = _attr(node, "content-desc")
        clickable = _is_true(node, "clickable")
        long_clickable = _is_true(node, "long-clickable")
        checkable = _is_true(node, "checkable")
        scrollable = _is_true(node, "scrollable")
        state = {
            "checkable": _tri(node, "checkable"),
            "checked": _tri(node, "checked"),
            "selected": _tri(node, "selected"),
            "scrollable": _tri(node, "scrollable"),
            "long_clickable": _tri(node, "long-clickable"),
            "password": _tri(node, "password"),
        }
        # `actionable` drives roll-up/absorption; a scrollable container stays a
        # separate element (it must NOT swallow the rows inside it).
        actionable = clickable or long_clickable or checkable
        children = node.findall("node")
        is_leaf = not children
        has_own_label = bool(text) or bool(content_desc)
        resource_id = _attr(node, "resource-id")
        # App resource-ids (com.app:id/containerDetail) must stay in the analyze list —
        # Compose hubs are often unlabeled containers the agent addresses with --rid / has --by id.
        # Skip framework android:id/* noise so we don't bury the screen under layout wrappers.
        has_app_id = bool(
            resource_id and ":id/" in resource_id and not resource_id.startswith("android:id/")
        )
        interesting = actionable or scrollable or has_own_label or is_leaf or has_app_id
        # Fold unlabeled non-actionable kids into clickable ancestors — unless they carry an
        # app resource-id the agent needs to address.
        absorbed = actionable_ancestor and not actionable and not has_app_id

        own_slot = parent_slot
        if valid and interesting and not absorbed:
            assert bounds is not None
            own_slot = len(collected)
            label = text
            if actionable and not has_own_label:
                label = _gather_descendant_text(node)
            collected.append(
                (
                    bounds,
                    Element(
                        id=-1,  # assigned after sorting
                        type=_short_type(node.get("class")),
                        text=label,
                        resource_id=resource_id,
                        content_desc=content_desc,
                        bounds=bounds,
                        center=center_of(bounds),
                        clickable=clickable,
                        enabled=_is_true(node, "enabled"),
                        focused=_is_true(node, "focused"),
                        source=Source.hierarchy,
                        confidence=None,
                        window=classify_window(node.get("package")),
                        **state,
                    ),
                    parent_slot,
                )
            )

        child_ancestor = actionable_ancestor or actionable
        for child in children:
            visit(child, child_ancestor, own_slot)

    for top in root.findall("node"):
        visit(top, False)

    # stable top-to-bottom, then left-to-right
    order = sorted(range(len(collected)), key=lambda i: (collected[i][0][1], collected[i][0][0]))
    slot_to_id = {slot: new_id for new_id, slot in enumerate(order)}
    from .identity import attach_stable_keys

    return attach_stable_keys(
        [
            collected[slot][1].model_copy(
                update={
                    "id": new_id,
                    "parent": slot_to_id.get(collected[slot][2])
                    if collected[slot][2] is not None
                    else None,
                }
            )
            for new_id, slot in enumerate(order)
        ]
    )
