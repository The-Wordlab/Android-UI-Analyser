"""Tesseract OCR provider (via pytesseract).

Not installed in the default environment.  ``is_available()`` returns False with
a helpful hint whenever ``pytesseract`` is missing **or** the system ``tesseract``
binary cannot be found.

When both the Python package *and* the binary are present, ``recognize()`` uses
``pytesseract.image_to_data`` (TSV output with per-word bounding boxes) and
filters out low-confidence words.

Tunable via ``models.tesseract`` config block:
  lang: "eng"
"""

from __future__ import annotations

from ..base import Availability, OcrProvider, ScreenImage, TextBox
from ..registry import register_ocr


@register_ocr("tesseract")
class TesseractOcrProvider(OcrProvider):
    """Tesseract OCR provider (via pytesseract + system tesseract binary)."""

    def is_available(self) -> Availability:
        try:
            import pytesseract  # noqa: F401
        except ImportError as exc:
            return Availability(
                False,
                f"pytesseract not installed: {exc} (pip install android-ui-analyser[tesseract])",
            )
        # Package is present; check the system binary.
        try:
            import pytesseract as _pt

            _pt.get_tesseract_version()
        except Exception as exc:
            return Availability(
                False,
                f"tesseract system binary not found or not executable: {exc} "
                "(install tesseract-ocr via your OS package manager, e.g. "
                "'brew install tesseract' or 'apt-get install tesseract-ocr')",
            )
        return Availability(True, "tesseract available")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        avail = self.is_available()
        if not avail.ok:
            return []

        import pytesseract

        pil_img = image.pil()
        lang = self.settings.get("lang", "eng")

        data = pytesseract.image_to_data(
            pil_img,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

        boxes: list[TextBox] = []
        n = len(data["text"])
        for i in range(n):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue  # tesseract uses -1 for non-text rows

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            boxes.append(
                TextBox(
                    text=text,
                    bounds=(x, y, x + w, y + h),
                    confidence=conf / 100.0,
                )
            )

        return boxes
