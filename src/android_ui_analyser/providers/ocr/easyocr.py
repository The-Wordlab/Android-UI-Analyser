"""EasyOCR provider (optional).

Not installed in the default environment.  ``is_available()`` returns False with
a helpful hint whenever the ``easyocr`` package is missing.

When the dep *is* present, ``recognize()`` uses an ``easyocr.Reader`` cached on
the instance.  EasyOCR returns::

    [(box_4pts, text, confidence), ...]

where ``box_4pts`` is ``[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]``.

Tunable via ``models.easyocr`` config block:
  lang: ["en"]   (list of language codes)
"""

from __future__ import annotations

from ..base import Availability, OcrProvider, ScreenImage, TextBox
from ..registry import register_ocr


@register_ocr("easyocr")
class EasyOcrProvider(OcrProvider):
    """EasyOCR provider."""

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self._reader = None

    def is_available(self) -> Availability:
        try:
            import easyocr  # noqa: F401
        except ImportError as exc:
            return Availability(
                False,
                f"easyocr not installed: {exc} (pip install android-ui-analyser[easyocr])",
            )
        return Availability(True, "easyocr available")

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            lang_cfg = self.settings.get("lang", ["en"])
            if isinstance(lang_cfg, str):
                lang_cfg = [lang_cfg]
            self._reader = easyocr.Reader(lang_cfg, gpu=False, verbose=False)
        return self._reader

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        avail = self.is_available()
        if not avail.ok:
            return []

        reader = self._get_reader()
        arr = image.numpy()
        raw = reader.readtext(arr)

        if not raw:
            return []

        boxes: list[TextBox] = []
        for item in raw:
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
