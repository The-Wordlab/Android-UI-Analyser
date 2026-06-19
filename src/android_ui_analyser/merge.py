"""Vision merge: detections + OCR → element list (PRD §6 step 4, AC10).

This is the back half of the T3 vision path. Detection providers give us *interactable
boxes* (no text); OCR providers give us *text boxes* (no notion of "control"). This
module fuses them into the same canonical :class:`Element` list the hierarchy path
produces, so the rest of the engine is source-agnostic:

1. **Dedupe detections** — detection chains (and overlapping models) can emit the same
   control twice. Two boxes overlapping with ``IoU > iou_threshold`` are collapsed to
   one; the survivor is the one with the higher ``confidence`` (ties broken by larger
   area).
2. **Associate OCR → detection** — a text box that sits *mostly inside* a detection box
   (``containment ≥ CONTAINMENT_MIN`` of the text's area) becomes that control's label
   (best-containing box wins). This is how a detected button gets its caption.
3. **Build elements** — each surviving detection → an :class:`Element`
   (``source=detection``); each *unassociated* text box → a standalone text element
   (``source=ocr``, never clickable).
4. **Assign IDs** — stable top-to-bottom then left-to-right (key ``(y1, x1)``), starting
   at ``start_id`` so vision elements can be appended after a hierarchy pool without ID
   collisions.
"""

from __future__ import annotations

from .providers.base import Bounds, DetBox, TextBox
from .schema import Element, Source, center_of

# A text box counts as "inside" a detection box when at least this fraction of the
# text box's own area lies within the detection box.
CONTAINMENT_MIN = 0.6


def _area(box: Bounds) -> int:
    """Area of a box; 0 if degenerate (non-positive width/height)."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0
    return w * h


def _intersection_area(a: Bounds, b: Bounds) -> int:
    """Area of the overlap of two boxes; 0 if they don't overlap."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0
    return iw * ih


def iou(a: Bounds, b: Bounds) -> float:
    """Intersection-over-union of two ``[x1,y1,x2,y2]`` boxes.

    Returns ``0.0`` when there is no overlap (or either box is degenerate), ``1.0`` for
    identical boxes.
    """
    inter = _intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _containment(inner: Bounds, outer: Bounds) -> float:
    """Fraction of ``inner``'s area that lies within ``outer`` (0.0 if no overlap)."""
    inner_area = _area(inner)
    if inner_area == 0:
        return 0.0
    return _intersection_area(inner, outer) / inner_area


def _dedupe_detections(detections: list[DetBox], iou_threshold: float) -> list[DetBox]:
    """Drop near-duplicate detection boxes (IoU > threshold).

    Greedy: process boxes best-first (higher confidence, then larger area) and keep a
    box only if it doesn't overlap an already-kept one beyond the threshold. Processing
    best-first means the survivor of any duplicate pair is the better box.
    """

    def sort_key(d: DetBox) -> tuple[float, int]:
        conf = d.confidence if d.confidence is not None else -1.0
        return (conf, _area(d.bounds))

    ordered = sorted(detections, key=sort_key, reverse=True)
    kept: list[DetBox] = []
    for det in ordered:
        if any(iou(det.bounds, k.bounds) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def merge_vision(
    detections: list[DetBox],
    texts: list[TextBox],
    *,
    iou_threshold: float = 0.5,
    start_id: int = 0,
) -> list[Element]:
    """Fuse detection + OCR boxes into a sorted, ID-assigned element list.

    See the module docstring for the full algorithm. ``iou_threshold`` controls
    detection dedup; ``start_id`` offsets the assigned IDs (so vision elements can be
    appended after a hierarchy pool).
    """
    kept_dets = _dedupe_detections(detections, iou_threshold)

    # Associate each text box with its best-containing detection (if any).
    # text_for_det[i] collects the labels attached to kept_dets[i].
    text_for_det: dict[int, list[TextBox]] = {}
    associated: set[int] = set()
    for ti, tb in enumerate(texts):
        best_det = -1
        best_cover = 0.0
        best_area = 0
        for di, det in enumerate(kept_dets):
            cover = _containment(tb.bounds, det.bounds)
            area = _area(det.bounds)
            # take the most-containing box; on a containment tie prefer the tighter
            # (smaller-area) box, so a label attaches to the chip, not its container.
            better = cover > best_cover or (cover == best_cover and 0 < area < best_area)
            if cover > 0 and better:
                best_cover = cover
                best_area = area
                best_det = di
        if best_det >= 0 and best_cover >= CONTAINMENT_MIN:
            text_for_det.setdefault(best_det, []).append(tb)
            associated.add(ti)

    built: list[tuple[Bounds, Element]] = []

    # detection → Element (source=detection)
    for di, det in enumerate(kept_dets):
        labels = text_for_det.get(di, [])
        ocr_text = " ".join(tb.text for tb in labels if tb.text).strip() or None
        text = ocr_text if ocr_text is not None else det.label
        built.append(
            (
                det.bounds,
                Element(
                    id=-1,
                    type=det.label or "Element",
                    text=text,
                    resource_id=None,
                    content_desc=None,
                    bounds=det.bounds,
                    center=center_of(det.bounds),
                    clickable=det.interactable,
                    enabled=True,
                    focused=False,
                    source=Source.detection,
                    confidence=det.confidence,
                ),
            )
        )

    # standalone OCR → Element (source=ocr)
    for ti, tb in enumerate(texts):
        if ti in associated:
            continue
        built.append(
            (
                tb.bounds,
                Element(
                    id=-1,
                    type="Text",
                    text=tb.text or None,
                    resource_id=None,
                    content_desc=None,
                    bounds=tb.bounds,
                    center=center_of(tb.bounds),
                    clickable=False,
                    enabled=True,
                    focused=False,
                    source=Source.ocr,
                    confidence=tb.confidence,
                ),
            )
        )

    # stable top-to-bottom, then left-to-right; assign sequential ids from start_id
    built.sort(key=lambda pair: (pair[0][1], pair[0][0]))
    elements: list[Element] = []
    for offset, (_bounds, element) in enumerate(built):
        elements.append(element.model_copy(update={"id": start_id + offset}))
    return elements
