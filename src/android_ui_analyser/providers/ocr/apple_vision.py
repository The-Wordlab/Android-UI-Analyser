"""Apple Vision OCR provider (macOS only, via pyobjc).

Uses VNRecognizeTextRequest to recognise text.  Vision returns normalised
bounding boxes with a *bottom-left* origin; we convert to pixel coords with a
*top-left* origin before returning TextBox objects.

Tunable via ``models.apple_vision`` config block:
  recognition_level: "accurate" (default) | "fast"
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..base import Availability, OcrProvider, ScreenImage, TextBox
from ..registry import register_ocr

if TYPE_CHECKING:
    pass


@register_ocr("apple_vision")
class AppleVisionOcrProvider(OcrProvider):
    """macOS Vision framework OCR provider."""

    def is_available(self) -> Availability:
        if sys.platform != "darwin":
            return Availability(False, "apple_vision requires macOS (sys.platform != 'darwin')")
        try:
            import Quartz  # noqa: F401 – pyobjc-framework-Quartz
            import Vision  # noqa: F401 – pyobjc-framework-Vision
        except ImportError as exc:
            return Availability(
                False,
                f"apple_vision requires pyobjc Vision/Quartz frameworks: {exc} "
                "(pip install android-ui-analyser[apple])",
            )
        return Availability(True, "apple_vision available")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        avail = self.is_available()
        if not avail.ok:
            return []

        import Quartz
        import Vision

        png_bytes = image.png_bytes
        ns_data = Quartz.CFDataCreate(None, png_bytes, len(png_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, {})

        # Determine recognition level from settings
        level_str = self.settings.get("recognition_level", "accurate")
        if level_str == "fast":
            level = Vision.VNRequestTextRecognitionLevelFast
        else:
            level = Vision.VNRequestTextRecognitionLevelAccurate

        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(level)

        ok, _err = handler.performRequests_error_([req], None)
        if not ok:
            return []

        observations = req.results()
        if not observations:
            return []

        w = image.width
        h = image.height
        boxes: list[TextBox] = []
        for obs in observations:
            candidates = obs.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = candidate.string()
            if not text:
                continue

            bbox = obs.boundingBox()
            origin = bbox.origin
            size = bbox.size

            # Vision uses normalised coords, bottom-left origin → convert to
            # pixel coords with top-left origin.
            x1 = int(origin.x * w)
            y2 = int((1.0 - origin.y) * h)
            x2 = int((origin.x + size.width) * w)
            y1 = int((1.0 - origin.y - size.height) * h)

            boxes.append(
                TextBox(
                    text=text,
                    bounds=(x1, y1, x2, y2),
                    confidence=float(obs.confidence()),
                )
            )

        return boxes
