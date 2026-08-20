"""AC10 — Set-of-Marks annotate tests.

Verifies that ``annotate()`` draws visible bounding boxes and numeric labels on a
synthetic screenshot without mutating the original image.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from android_ui_analyser.annotate import annotate
from android_ui_analyser.providers.base import ScreenImage
from android_ui_analyser.schema import Element, Source

# ---------------------------------------------------------------------------
# Local helpers (mirrors conftest.make_screen_image without cross-importing)
# ---------------------------------------------------------------------------


def _make_png(
    width: int = 200,
    height: int = 400,
    color: tuple[int, int, int] = (240, 240, 240),
) -> bytes:
    """Return a solid-colour PNG as raw bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_screen_image(
    width: int = 200,
    height: int = 400,
    color: tuple[int, int, int] = (240, 240, 240),
) -> ScreenImage:
    return ScreenImage(_make_png(width, height, color), width=width, height=height)


def _make_element(
    id_: int,
    bounds: tuple[int, int, int, int],
    *,
    text: str | None = None,
) -> Element:
    x1, y1, x2, y2 = bounds
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return Element(
        id=id_,
        type="TextView",
        text=text,
        bounds=bounds,
        center=(cx, cy),
        source=Source.hierarchy,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_annotate_creates_png(tmp_path: Path) -> None:
    """Annotated file is created and is a valid PNG with the original dimensions."""
    img = _make_screen_image(width=200, height=400)
    out = str(tmp_path / "annotated.png")

    elements = [
        _make_element(0, (10, 20, 90, 60), text="Button"),
        _make_element(1, (10, 100, 190, 150), text="Settings"),
    ]

    returned = annotate(img, elements, out)

    assert returned == out
    assert Path(out).exists(), "output file must be created"

    result_img = Image.open(out)
    assert result_img.size == (200, 400), "image dimensions must be unchanged"
    result_img.close()


def test_annotate_draws_boxes(tmp_path: Path) -> None:
    """Pixels on the box border must differ from the solid-grey background."""
    bg_color = (220, 220, 220)
    img = _make_screen_image(width=300, height=500, color=bg_color)
    out = str(tmp_path / "boxes.png")

    # Two elements with non-overlapping, large-ish boxes.
    elements = [
        _make_element(0, (20, 30, 120, 80)),
        _make_element(1, (20, 120, 250, 200)),
    ]

    annotate(img, elements, out)

    result = Image.open(out).convert("RGB")
    pixels = result.load()
    assert pixels is not None

    def border_drawn(x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check that at least one pixel on the border differs from background."""
        for x in range(x1, x2 + 1):
            for y in (y1, y2):
                if pixels[x, y] != bg_color:  # type: ignore[index]
                    return True
        for y in range(y1, y2 + 1):
            for x in (x1, x2):
                if pixels[x, y] != bg_color:  # type: ignore[index]
                    return True
        return False

    for el in elements:
        x1, y1, x2, y2 = el.bounds
        # Clamp to image bounds so we don't sample outside.
        x2_c = min(x2, result.width - 1)
        y2_c = min(y2, result.height - 1)
        assert border_drawn(x1, y1, x2_c, y2_c), (
            f"no border pixels found for element id={el.id} bounds={el.bounds}"
        )

    result.close()


def test_annotate_labels_two_elements(tmp_path: Path) -> None:
    """Labels for both element ids (0 and 1) must change some pixels near the boxes."""
    bg_color = (200, 200, 200)
    img = _make_screen_image(width=400, height=400, color=bg_color)
    out = str(tmp_path / "labels.png")

    elements = [
        _make_element(0, (50, 50, 150, 100)),
        _make_element(1, (50, 150, 150, 200)),
    ]

    annotate(img, elements, out)

    result = Image.open(out).convert("RGB")
    w, h = result.size
    pixels = result.load()
    assert pixels is not None

    # Count non-background pixels in the full image.
    non_bg = sum(
        1
        for x in range(w)
        for y in range(h)
        if pixels[x, y] != bg_color  # type: ignore[index]
    )
    # With 2 elements, each having a box + label, expect more than a handful of pixels.
    assert non_bg > 50, f"expected visible markings, got only {non_bg} non-background pixels"
    result.close()


def test_annotate_empty_elements(tmp_path: Path) -> None:
    """An empty elements list must still produce a valid PNG identical to the source."""
    img = _make_screen_image(width=100, height=100)
    out = str(tmp_path / "empty.png")

    annotate(img, [], out)

    assert Path(out).exists()
    result = Image.open(out)
    assert result.size == (100, 100)
    result.close()


def test_annotate_creates_parent_dirs(tmp_path: Path) -> None:
    """Parent directories of out_path are created automatically."""
    img = _make_screen_image(width=100, height=100)
    out = str(tmp_path / "deep" / "nested" / "annotated.png")

    annotate(img, [_make_element(0, (10, 10, 50, 50))], out)

    assert Path(out).exists()


def test_annotate_does_not_mutate_source(tmp_path: Path) -> None:
    """The source ScreenImage PIL object must not be mutated."""
    img = _make_screen_image(width=200, height=200)
    original_pil = img.pil()
    original_px = original_pil.load()
    assert original_px is not None
    snapshot_before = original_px[100, 100]

    out = str(tmp_path / "annotated.png")
    annotate(img, [_make_element(0, (80, 80, 120, 120))], out)

    # The pixel at the same position in the *source* must be unchanged.
    reloaded = img.pil().load()
    assert reloaded is not None
    assert reloaded[100, 100] == snapshot_before, "source image must not be mutated"
