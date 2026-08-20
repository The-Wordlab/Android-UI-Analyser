"""Apple Vision OCR provider (macOS only, via pyobjc).

Uses VNRecognizeTextRequest to recognise text.  Vision returns normalised
bounding boxes with a *bottom-left* origin; we convert to pixel coords with a
*top-left* origin before returning TextBox objects.

Tunable via ``models.apple_vision`` config block:
  recognition_level: "accurate" (default) | "fast" (Neural Engine hot path, truncates)
  max_width: downscale wider screenshots before OCR, then map boxes to original pixels
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

        original_w = image.width
        original_h = image.height
        working = image
        max_width = int(self.settings.get("max_width", 720) or 0)
        if max_width > 0 and image.width > max_width:
            from PIL import Image

            height = max(1, round(image.height * max_width / image.width))
            resized = image.pil().resize((max_width, height), Image.Resampling.LANCZOS)
            working = ScreenImage.from_pil(resized)

        png_bytes = working.png_bytes
        ns_data = Quartz.CFDataCreate(None, png_bytes, len(png_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, {})

        # Determine recognition level from settings
        level_str = self.settings.get("recognition_level", "accurate")
        if level_str == "accurate":
            level = Vision.VNRequestTextRecognitionLevelAccurate
        else:
            level = Vision.VNRequestTextRecognitionLevelFast

        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(level)

        ok, _err = handler.performRequests_error_([req], None)
        if not ok:
            return []

        observations = req.results()
        if not observations:
            return []

        w = working.width
        h = working.height
        scale_x = original_w / w
        scale_y = original_h / h
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
            x1 = round(origin.x * w * scale_x)
            y2 = round((1.0 - origin.y) * h * scale_y)
            x2 = round((origin.x + size.width) * w * scale_x)
            y1 = round((1.0 - origin.y - size.height) * h * scale_y)

            boxes.append(
                TextBox(
                    text=text,
                    bounds=(x1, y1, x2, y2),
                    confidence=float(obs.confidence()),
                )
            )

        return boxes
