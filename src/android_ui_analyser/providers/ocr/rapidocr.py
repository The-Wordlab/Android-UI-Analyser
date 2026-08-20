"""RapidOCR provider (cross-platform, via rapidocr-onnxruntime + onnxruntime).

The engine is cached on the instance after the first call.

RapidOCR returns results as::

    ([[box_4pts, text, score], ...], [det_time, cls_time, rec_time])

where ``box_4pts`` is ``[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]`` (all four
corners of the detected text region).  We convert to an axis-aligned bounding
box ``(min_x, min_y, max_x, max_y)``.

Tunable via ``models.rapidocr`` config block:
  lang: "en"  (passed as ``lang`` to RapidOCR constructor — unused by default
               engine but forwarded for custom model paths)
"""

from __future__ import annotations

from ..base import Availability, OcrProvider, ScreenImage, TextBox
from ..registry import register_ocr


@register_ocr("rapidocr")
class RapidOcrProvider(OcrProvider):
    """Cross-platform OCR using rapidocr-onnxruntime."""

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self._engine = None  # lazy-initialised on first recognize() call

    def is_available(self) -> Availability:
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        except ImportError as exc:
            return Availability(
                False,
                f"rapidocr not installed: {exc} (pip install android-ui-analyser[rapidocr])",
            )
        return Availability(True, "rapidocr available")

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        avail = self.is_available()
        if not avail.ok:
            return []

        engine = self._get_engine()
        arr = image.numpy()
        result, _ = engine(arr)

        if not result:
            return []

        boxes: list[TextBox] = []
        for item in result:
            if item is None:
                continue
            # item is [box_4pts, text, score]
            # box_4pts: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            box_4pts, text, score = item
            if not text:
                continue

            xs = [pt[0] for pt in box_4pts]
            ys = [pt[1] for pt in box_4pts]
            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs))
            y2 = int(max(ys))

            boxes.append(
                TextBox(
                    text=str(text),
                    bounds=(x1, y1, x2, y2),
                    confidence=float(score) if score is not None else None,
                )
            )

        return boxes
