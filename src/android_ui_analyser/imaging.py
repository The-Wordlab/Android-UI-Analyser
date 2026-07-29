"""Cheap perceptual hashing for ``wait --for-stable``, plus crop/downscale for capture.

The hashing half is deliberately tiny and dependency-light: it uses only Pillow (already a
base dependency) to reduce a screenshot to a small grayscale difference-hash. Comparing two
hashes by Hamming distance answers "did the screen change?" without OCR or a hierarchy
parse — the whole point of ``--for-stable`` (it works on opaque / Compose / video
screens an accessibility tree can't see, and is cheap enough to poll in a tight loop).

The crop/downscale half serves ``aua screenshot --region/--scale/--max-width``: an agent
that only needs the header pays for a full 1080x2400 PNG in image tokens otherwise, so
narrowing the capture before it is written is an order-of-magnitude saving.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import UsageError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .providers.base import ScreenImage

Bounds = tuple[int, int, int, int]

# dHash side length. The hash compares each pixel with its right neighbour over an
# (N x N+1) grayscale thumbnail → N*N bits. 16 → 256 bits: sensitive enough to catch a
# spinner frame, coarse enough to ignore sub-pixel noise / JPEG-ish wobble.
HASH_SIDE = 16
HASH_BITS = HASH_SIDE * HASH_SIDE

# Default Hamming distance under which two frames are considered "the same screen".
# ~3% of the bits — tolerates tiny rendering jitter, trips on a real content change.
DEFAULT_STABLE_DISTANCE = 8


def dhash(image: ScreenImage, *, side: int = HASH_SIDE) -> int:
    """Return a difference-hash of *image* as an integer of ``side*side`` bits."""
    pil = image.pil().convert("L").resize((side + 1, side), _RESAMPLE)
    # Row-major grayscale samples (avoids PixelAccess typing); compare each to its
    # right neighbour → 1 bit per pixel.
    px = list(pil.getdata())
    width = side + 1
    bits = 0
    pos = 0
    for y in range(side):
        row = y * width
        for x in range(side):
            bits |= (1 if px[row + x] < px[row + x + 1] else 0) << pos
            pos += 1
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes (Python 3.11: int.bit_count)."""
    return (a ^ b).bit_count()


def is_stable(a: int, b: int, *, distance: int = DEFAULT_STABLE_DISTANCE) -> bool:
    """True if two frame hashes are within *distance* bits (i.e. visually unchanged)."""
    return hamming(a, b) <= distance


def _resample():  # pragma: no cover - trivial import shim
    from PIL import Image

    # Pillow ≥ 9.1 moved resampling filters under Image.Resampling.
    return getattr(Image, "Resampling", Image).LANCZOS


_RESAMPLE = _resample()


# --------------------------------------------------------------------------- crop / scale


def capture_path(cache_dir: str | Path, serial: str, *, suffix: str = "screenshot") -> str:
    """A fresh path under the run dir for one capture.

    Timestamped: a caller cropping a before/after pair in the same second must not have the
    second write clobber the first.
    """
    run_dir = Path(cache_dir).expanduser() / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    return str(run_dir / f"{serial.replace(':', '_')}_{suffix}_{stamp}.png")


def parse_region(raw: str) -> Bounds:
    """Parse a ``x1,y1,x2,y2`` CLI region into a normalised box."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise UsageError(
            f"invalid --region '{raw}'",
            hint="Give four integers: --region x1,y1,x2,y2 (e.g. 0,0,1080,300).",
        )
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError as exc:
        raise UsageError(
            f"invalid --region '{raw}'",
            hint="All four values must be integers, e.g. --region 0,0,1080,300.",
        ) from exc
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def crop_and_scale(
    image: ScreenImage,
    *,
    region: Bounds | None = None,
    scale: float | None = None,
    max_width: int | None = None,
) -> ScreenImage:
    """Return *image* cropped to *region* then downscaled, or itself when asked for nothing.

    ``region`` is clamped to the screen (an off-screen box is a usage error, not a crash).
    ``max_width`` only ever shrinks — upscaling a screenshot adds bytes and no detail.
    """
    if region is None and scale is None and max_width is None:
        return image
    from .providers.base import ScreenImage as _ScreenImage

    pil = image.pil()
    if region is not None:
        pil = pil.crop(_clamp(region, pil.width, pil.height))
    target = _target_width(pil.width, scale=scale, max_width=max_width)
    if target is not None and target != pil.width:
        height = max(1, round(pil.height * target / pil.width))
        pil = pil.resize((target, height), _RESAMPLE)
    return _ScreenImage.from_pil(pil)


def _clamp(region: Bounds, width: int, height: int) -> Bounds:
    x1, y1, x2, y2 = region
    box = (max(0, min(x1, width)), max(0, min(y1, height)), min(x2, width), min(y2, height))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise UsageError(
            f"--region {','.join(str(v) for v in region)} does not overlap the {width}x{height} screen",
            hint="Regions are screen pixels: --region 0,0,1080,300 is the top strip.",
        )
    return box


def _target_width(width: int, *, scale: float | None, max_width: int | None) -> int | None:
    if scale is not None:
        if scale <= 0:
            raise UsageError("invalid --scale", hint="--scale takes a positive factor, e.g. 0.5.")
        width = max(1, round(width * scale))
    if max_width is not None:
        if max_width <= 0:
            raise UsageError("invalid --max-width", hint="--max-width takes a positive pixel count.")
        width = min(width, max_width)
    return width
