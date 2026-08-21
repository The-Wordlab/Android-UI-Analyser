"""Cross-frame element identity — stable keys that survive re-analyze ID churn.

Frame-local integer ``id`` values are rewritten every analyze (reading order). A
``stable_key`` fingerprints an element so ``aua resolve`` can remap an old id (or the
key itself) onto the current frame.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .schema import Element

if TYPE_CHECKING:
    from .providers.base import ScreenImage

_ID_TAIL = re.compile(r"(?:.*/)?([^/]+)$")
_PIXEL_KEY = re.compile(r"^px:([^:]+):([0-9a-f]{16})$")

# A compact 8x8 difference hash gives 64 visual bits. Six changed bits tolerates a measured
# one-pixel bounds/render shift after edge normalization while still refusing distinct shapes.
PIXEL_HASH_SIDE = 8
PIXEL_MAX_DISTANCE = 6
PIXEL_MIN_INFORMATION_BITS = 3


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
    - ``geo:<type>:<quadrant>:<hash>`` metadata-only last resort

    ``attach_visual_stable_keys`` upgrades an actionable ``geo:`` element to ``px:`` when
    the engine has a screenshot. Keeping geometry generation here preserves offline hierarchy
    parsing and a backward-compatible alias for old caches.
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
    q = _quadrant(bounds)
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


def needs_visual_stable_key(el: Element | dict) -> bool:
    """Whether an element has no semantic identity and can actually receive an action."""
    if isinstance(el, dict):
        rid = el.get("resource_id")
        text = el.get("text")
        desc = el.get("content_desc")
        bounds = tuple(el.get("bounds") or (0, 0, 0, 0))
        actionable = any(
            bool(el.get(name))
            for name in ("clickable", "long_clickable", "checkable", "scrollable")
        )
    else:
        rid = el.resource_id
        text = el.text
        desc = el.content_desc
        bounds = tuple(el.bounds)
        actionable = any((el.clickable, el.long_clickable, el.checkable, el.scrollable))
    x1, y1, x2, y2 = bounds
    return bool(
        actionable
        and not id_tail(rid)
        and not _norm(desc)
        and not _norm(text)
        and x2 - x1 >= 2
        and y2 - y1 >= 2
    )


def _pixel_type(value: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "unknown").strip())
    return (cleaned or "unknown")[:48]


def pixel_stable_key(
    el: Element | dict,
    image: ScreenImage,
    *,
    screen_size: tuple[int, int] | None = None,
) -> str | None:
    """Fingerprint one unlabeled actionable element from its normalized screenshot crop.

    The crop is contrast-normalized and reduced to edges before hashing. That deliberately
    values the control's shape over its colour/background, which is important for transparent
    toolbar icons rendered over changing artwork. The result is a candidate identity, not an
    exact-pixel assertion; resolution compares hashes by Hamming distance.
    """
    if not needs_visual_stable_key(el):
        return None
    typ = el.get("type") if isinstance(el, dict) else el.type
    bounds = tuple(el.get("bounds") or (0, 0, 0, 0)) if isinstance(el, dict) else el.bounds
    source_w, source_h = screen_size or (image.width, image.height)
    if source_w <= 0 or source_h <= 0:
        return None
    scale_x = image.width / source_w
    scale_y = image.height / source_h
    x1, y1, x2, y2 = bounds
    left = max(0, min(image.width, round(x1 * scale_x)))
    top = max(0, min(image.height, round(y1 * scale_y)))
    right = max(0, min(image.width, round(x2 * scale_x)))
    bottom = max(0, min(image.height, round(y2 * scale_y)))
    if right - left < 2 or bottom - top < 2:
        return None

    from PIL import ImageFilter, ImageOps

    from .imaging import dhash_pil

    crop = image.pil().crop((left, top, right, bottom)).convert("L")
    normalized = ImageOps.autocontrast(crop).filter(ImageFilter.FIND_EDGES)
    # FIND_EDGES gives the crop border artificial contrast. Remove it when the crop is large
    # enough so a one-pixel bounds change does not become the element's strongest feature.
    if normalized.width > 4 and normalized.height > 4:
        normalized = normalized.crop((1, 1, normalized.width - 1, normalized.height - 1))
    fingerprint = dhash_pil(normalized, side=PIXEL_HASH_SIDE)
    information = fingerprint.bit_count()
    if information < PIXEL_MIN_INFORMATION_BITS or information > 64 - PIXEL_MIN_INFORMATION_BITS:
        # A nearly uniform crop cannot identify itself; geometry is more honest than a hash
        # shared by every blank/solid control on the screen.
        return None
    return f"px:{_pixel_type(typ)}:{fingerprint:016x}"


def attach_visual_stable_keys(
    elements: Sequence[Element],
    image: ScreenImage,
    *,
    screen_size: tuple[int, int] | None = None,
) -> list[Element]:
    """Replace geometry-only keys with visual fingerprints from one shared screenshot."""
    out: list[Element] = []
    for el in elements:
        key = pixel_stable_key(el, image, screen_size=screen_size)
        if key is None:
            key = el.stable_key or stable_key(el)
        out.append(el if el.stable_key == key else el.model_copy(update={"stable_key": key}))
    return out


def _pixel_parts(key: str | None) -> tuple[str, int] | None:
    match = _PIXEL_KEY.fullmatch((key or "").strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2), 16)


def _identity_distance(previous: Element, current: Element) -> int | None:
    """Zero for exact/legacy identity, positive for a close visual fingerprint."""
    previous_key = previous.stable_key or stable_key(previous)
    current_key = current.stable_key or stable_key(current)
    if previous_key == current_key:
        return 0

    # Migration in either direction: a cached geo key remains resolvable after px keys begin
    # shipping, and a screenshot failure can still bind a prior px element by unchanged bounds.
    previous_geometry = stable_key(previous)
    current_geometry = stable_key(current)
    if previous_key == current_geometry or current_key == previous_geometry:
        return 0

    previous_pixel = _pixel_parts(previous_key)
    current_pixel = _pixel_parts(current_key)
    if previous_pixel is None or current_pixel is None or previous_pixel[0] != current_pixel[0]:
        return None
    distance = (previous_pixel[1] ^ current_pixel[1]).bit_count()
    return distance if distance <= PIXEL_MAX_DISTANCE else None


def remap_ids(
    previous: Sequence[Element],
    current: Sequence[Element],
) -> dict[int, int]:
    """Map previous-frame ids → current-frame ids via exact or perceptually-close identity."""
    mapping: dict[int, int] = {}
    for prev in previous:
        scored = [
            (distance, candidate)
            for candidate in current
            if (distance := _identity_distance(prev, candidate)) is not None
        ]
        if not scored:
            continue
        best_distance = min(distance for distance, _candidate in scored)
        cands = [candidate for distance, candidate in scored if distance == best_distance]
        if len(cands) == 1:
            mapping[prev.id] = cands[0].id
        elif len(cands) > 1:
            # Disambiguate by IoU with previous bounds — but only when there is actually some
            # overlap to reason from. This used to seed best_iou at -1.0 and accept the winner
            # unconditionally, so a candidate overlapping the original by *nothing* still won, and
            # which one won came down to list order. On 2026-08-07 that re-pointed a live element id
            # onto the system nav bar's Home button twice, silently backgrounding the app under a
            # scenario that then had to explain a "crash" that never happened.
            #
            # rid: carries no position at all, cd:/tx: carry only a coarse quadrant, and two
            # visually identical px: controls may sit anywhere on screen. Same-key candidates
            # therefore still need spatial evidence; geo: alone hashes exact bounds.
            #
            # With several same-key candidates and no overlap anywhere, there is simply no evidence
            # for choosing one. Decline, and let the caller raise ElementNotFoundError: recovery is
            # one `analyze` away, whereas a silent mis-tap costs the whole journey.
            best, best_iou = None, 0.0
            for cand in cands:
                iou = _iou(prev.bounds, cand.bounds)
                if iou > best_iou:
                    best, best_iou = cand, iou
            if best is not None:
                mapping[prev.id] = best.id
    return mapping


def find_by_stable_key(elements: Sequence[Element], key: str) -> list[Element]:
    """Elements matching *key*, including close ``px:`` and legacy ``geo:`` identities."""
    needle = key.strip()
    exact: list[Element] = []
    for el in elements:
        sk = el.stable_key or stable_key(el)
        if sk == needle:
            exact.append(el)
    if exact:
        return exact

    if needle.startswith("geo:"):
        return [el for el in elements if stable_key(el) == needle]

    wanted = _pixel_parts(needle)
    if wanted is None:
        return []
    scored: list[tuple[int, Element]] = []
    for el in elements:
        candidate = _pixel_parts(el.stable_key or stable_key(el))
        if candidate is None or candidate[0] != wanted[0]:
            continue
        distance = (candidate[1] ^ wanted[1]).bit_count()
        if distance <= PIXEL_MAX_DISTANCE:
            scored.append((distance, el))
    if not scored:
        return []
    best = min(distance for distance, _el in scored)
    return [el for distance, el in scored if distance == best]


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
    "attach_visual_stable_keys",
    "find_by_stable_key",
    "id_tail",
    "needs_visual_stable_key",
    "pixel_stable_key",
    "remap_ids",
    "stable_key",
]
