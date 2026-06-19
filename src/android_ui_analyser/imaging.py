"""Cheap perceptual hashing for the ``wait --for-stable`` settle check (PRD §5, AC14).

This is deliberately tiny and dependency-light: it uses only Pillow (already a base
dependency) to reduce a screenshot to a small grayscale difference-hash. Comparing two
hashes by Hamming distance answers "did the screen change?" without OCR or a hierarchy
parse — the whole point of ``--for-stable`` (it works on opaque / Compose / video
screens an accessibility tree can't see, and is cheap enough to poll in a tight loop).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .providers.base import ScreenImage

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
