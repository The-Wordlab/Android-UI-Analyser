"""Set-of-Marks overlay annotator (PRD §5, AC10).

Draws numbered bounding-box labels on a copy of a screenshot so an AI agent can
identify elements by their integer id.  Uses only Pillow — no system fonts required.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers.base import ScreenImage
    from .schema import Element


# Palette: evenly-spaced hues, high saturation/value so labels pop on any background.
_PALETTE_SIZE = 20


def _palette_color(index: int) -> tuple[int, int, int]:
    """Return a distinct RGB color for the given index (cycles after _PALETTE_SIZE)."""
    hue = (index % _PALETTE_SIZE) / _PALETTE_SIZE
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def annotate(
    image: ScreenImage,
    elements: list[Element],
    out_path: str,
    *,
    font_size: int | None = None,  # accepted but ignored — default font only
) -> str:
    """Draw Set-of-Marks boxes and labels onto *a copy* of ``image``, save as PNG.

    For each element: draw a 2-px-wide rectangle around ``element.bounds``, then
    place a filled rectangle with the element's ``id`` as white text near the
    top-left corner of the box.  Uses ``ImageFont.load_default()`` — no system
    fonts required.

    Args:
        image: Source screenshot.
        elements: Elements to annotate; order determines label color cycling.
        out_path: Destination PNG path (parent dirs are created).
        font_size: Accepted for API compatibility; ignored (default font only).

    Returns:
        ``out_path`` (the saved file path).
    """
    from PIL import ImageDraw, ImageFont

    # Work on a copy so the caller's ScreenImage is not mutated.
    src = image.pil()
    canvas = src.copy()
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()

    for idx, el in enumerate(elements):
        color = _palette_color(idx)
        x1, y1, x2, y2 = el.bounds

        # --- bounding box ---
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # --- label background + text ---
        label = str(el.id)

        # Measure text with getbbox for newer Pillow; fall back to textlength.
        try:
            bbox = font.getbbox(label)  # (left, top, right, bottom)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw = int(font.getlength(label))
            th = 12  # safe default for load_default()

        pad = 2
        lx1 = x1
        ly1 = max(0, y1 - th - pad * 2)
        # If box is at the very top, push label inside.
        if ly1 < 0:
            ly1 = y1
        lx2 = lx1 + tw + pad * 2
        ly2 = ly1 + th + pad * 2

        draw.rectangle([lx1, ly1, lx2, ly2], fill=color)
        draw.text((lx1 + pad, ly1 + pad), label, fill=(255, 255, 255), font=font)

    # Ensure parent directory exists.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path
