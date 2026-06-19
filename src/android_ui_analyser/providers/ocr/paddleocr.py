"""PaddleOCR provider (PP-OCRv5, highest accuracy).

Not installed in the default environment.  ``is_available()`` returns False with
a helpful hint whenever the ``paddleocr`` package is missing.

When the dep *is* present, ``recognize()`` uses ``PaddleOCR`` with
``use_angle_cls=True`` and the language from ``settings["lang"]`` (default
``"en"``).  PaddleOCR returns::

    [line, ...]

where each ``line`` is::

    [box_4pts, [text, confidence]]

and ``box_4pts`` is ``[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]``.

Tunable via ``models.paddleocr`` config block:
  lang: "en"
"""

from __future__ import annotations

from ..base import Availability, OcrProvider, ScreenImage, TextBox
from ..registry import register_ocr


@register_ocr("paddleocr")
class PaddleOcrProvider(OcrProvider):
    """PaddleOCR provider (PP-OCRv5)."""

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self._engine = None

    def is_available(self) -> Availability:
        try:
            import paddleocr  # noqa: F401
        except ImportError as exc:
            return Availability(
                False,
                f"paddleocr not installed: {exc} (pip install android-ui-analyser[paddle])",
            )
        return Availability(True, "paddleocr available")

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR

            lang = self.settings.get("lang", "en")
            self._engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        return self._engine

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        avail = self.is_available()
        if not avail.ok:
            return []

        engine = self._get_engine()
        arr = image.numpy()
        raw = engine.ocr(arr, cls=True)

        if not raw:
            return []

        # PaddleOCR returns a list of pages; we always pass a single image.
        lines = raw[0] if isinstance(raw[0], list) and raw else raw
        if not lines:
            return []

        boxes: list[TextBox] = []
        for line in lines:
            if line is None:
                continue
            box_4pts, (text, score) = line
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
