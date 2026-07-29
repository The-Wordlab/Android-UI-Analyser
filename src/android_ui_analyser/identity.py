"""Cross-frame element identity — stable keys that survive re-analyze ID churn.

Frame-local integer ``id`` values are rewritten every analyze (reading order). A
``stable_key`` fingerprints an element so ``aua resolve`` can remap an old id (or the
key itself) onto the current frame.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .schema import Element

_ID_TAIL = re.compile(r"(?:.*/)?([^/]+)$")


def id_tail(resource_id: str | None) -> str | None:
    """Last segment of a resource-id (``com.app:id/foo`` → ``foo``)."""
    if not resource_id:
        return None
    m = _ID_TAIL.match(resource_id.strip())
    return m.group(1) if m else resource_id.strip() or None


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _quadrant(bounds: tuple[int, int, int, int], *, cols: int = 4, rows: int = 6) -> str:
    """Coarse position bucket so identical labels in different regions stay distinct."""
    x1, y1, x2, y2 = bounds
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    col = min(cols - 1, max(0, int(cx // 270)))
    row = min(rows - 1, max(0, int(cy // 200)))
    return f"q{row}{col}"


def stable_key(el: Element | dict) -> str:
    """Deterministic fingerprint for an element (resource-id preferred).

    Format:
    - ``rid:<tail>`` when a resource-id exists (most stable)
    - ``cd:<hash>`` / ``tx:<hash>`` for labeled nodes without rid
    - ``geo:<type>:<quadrant>:<hash>`` last resort (type + coarse position)
    """
    if isinstance(el, dict):
        rid = el.get("resource_id")
        typ = el.get("type") or "?"
        text = el.get("text")
        desc = el.get("content_desc")
        bounds = tuple(el.get("bounds") or (0, 0, 0, 0))
    else:
        rid = el.resource_id
        typ = el.type or "?"
        text = el.text
        desc = el.content_desc
        bounds = tuple(el.bounds)

    tail = id_tail(rid)
    if tail:
        return f"rid:{tail}"

    label = _norm(desc) or _norm(text)
    q = _quadrant(bounds)  # type: ignore[arg-type]
    if label:
        digest = hashlib.sha1(f"{typ}|{label}|{q}".encode()).hexdigest()[:10]
        kind = "cd" if _norm(desc) else "tx"
        return f"{kind}:{digest}"

    digest = hashlib.sha1(f"{typ}|{q}|{bounds}".encode()).hexdigest()[:10]
    return f"geo:{typ}:{q}:{digest}"


def attach_stable_keys(elements: Sequence[Element]) -> list[Element]:
    """Return copies of *elements* with ``stable_key`` filled in."""
    out: list[Element] = []
    for el in elements:
        key = el.stable_key or stable_key(el)
        out.append(el if el.stable_key == key else el.model_copy(update={"stable_key": key}))
    return out


def remap_ids(
    previous: Sequence[Element],
    current: Sequence[Element],
) -> dict[int, int]:
    """Map previous-frame ids → current-frame ids via ``stable_key`` (exact match)."""
    by_key: dict[str, list[Element]] = {}
    for el in current:
        key = el.stable_key or stable_key(el)
        by_key.setdefault(key, []).append(el)

    mapping: dict[int, int] = {}
    for prev in previous:
        key = prev.stable_key or stable_key(prev)
        cands = by_key.get(key) or []
        if len(cands) == 1:
            mapping[prev.id] = cands[0].id
        elif len(cands) > 1:
            # Disambiguate by IoU with previous bounds.
            best, best_iou = None, -1.0
            for cand in cands:
                iou = _iou(prev.bounds, cand.bounds)
                if iou > best_iou:
                    best, best_iou = cand, iou
            if best is not None:
                mapping[prev.id] = best.id
    return mapping


def find_by_stable_key(elements: Sequence[Element], key: str) -> list[Element]:
    """Elements whose stable_key equals *key* (exact)."""
    needle = key.strip()
    hits: list[Element] = []
    for el in elements:
        sk = el.stable_key or stable_key(el)
        if sk == needle:
            hits.append(el)
    return hits


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


__all__ = [
    "attach_stable_keys",
    "find_by_stable_key",
    "id_tail",
    "remap_ids",
    "stable_key",
]
