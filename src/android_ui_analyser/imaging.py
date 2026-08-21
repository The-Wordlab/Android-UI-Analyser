"""Cheap perceptual hashing for ``wait --for-stable``, plus crop/downscale for capture.

The hashing half is deliberately tiny and dependency-light: it uses only Pillow (already a
base dependency) to reduce a screenshot to a small grayscale difference-hash. Comparing two
hashes by Hamming distance answers "did the screen change?" without OCR or a hierarchy
parse — the whole point of ``--for-stable`` (it works on opaque / Compose / video
screens an accessibility tree can't see, and is cheap enough to poll in a tight loop).

GridSettle extends this with a cell grid so looping animations (spinners, Lottie, video)
can be masked while the rest of the screen is treated as settled — critical for
post-action observe that must not return mid-transition OR hang forever on a spinner.

The crop/downscale half serves ``aua screenshot --region/--scale/--max-width``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .errors import UsageError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

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


def dhash_pil(image: PILImage, *, side: int = HASH_SIDE) -> int:
    """Return a difference-hash for a Pillow image without re-encoding it.

    Element identity crops already live inside one decoded screenshot.  Keeping this small
    helper beside :func:`dhash` lets every crop reuse those pixels instead of taking a PNG
    encode/decode round trip merely to satisfy ``ScreenImage``'s transport wrapper.
    """
    pil = image.convert("L").resize((side + 1, side), _RESAMPLE)
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


def dhash(image: ScreenImage, *, side: int = HASH_SIDE) -> int:
    """Return a difference-hash of *image* as an integer of ``side*side`` bits."""
    return dhash_pil(image.pil(), side=side)


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


# --------------------------------------------------------------------------- grid-based settle


GRID_COLS = 4
GRID_ROWS = 6
# A cell that changed on this many consecutive samples is flagged as live animation.
ANIMATION_STREAK = 2
# Mean absolute difference (0–255) above which a cell counts as changed.
# Solid-colour flips (spinner palette) must trip this — dHash alone does not.
CELL_MAD_THRESHOLD = 6.0


def grid_signatures(
    image: ScreenImage, *, cols: int = GRID_COLS, rows: int = GRID_ROWS, side: int = 8
) -> list[tuple[float, ...]]:
    """Per-cell content fingerprints (downscaled grayscale samples).

    Unlike dHash, a uniform red cell and a uniform blue cell differ — required for
    detecting solid-colour spinner / splash changes.
    """
    pil = image.pil().convert("L")
    w, h = pil.size
    cell_w = max(1, w // cols)
    cell_h = max(1, h // rows)
    out: list[tuple[float, ...]] = []
    for gy in range(rows):
        for gx in range(cols):
            box = (gx * cell_w, gy * cell_h, (gx + 1) * cell_w, (gy + 1) * cell_h)
            cell = pil.crop(box).resize((side, side), _RESAMPLE)
            # Quantise a bit to ignore sub-pixel noise. Pillow's stubs type `getdata()` as a
            # bare `ImagingCore`, which declares no `__iter__` even though the object it
            # returns is a sequence — the cast states what Pillow documents.
            out.append(tuple(float(v) for v in cast("Iterable[int]", cell.getdata())))
    return out


def cell_mad(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Mean absolute difference between two cell signatures."""
    if not a or not b or len(a) != len(b):
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def frame_signature(image: ScreenImage, *, side: int = 16) -> tuple[float, ...]:
    """Whole-frame coarse signature for 'did anything change?' checks."""
    pil = image.pil().convert("L").resize((side, side), _RESAMPLE)
    return tuple(float(v) for v in cast("Iterable[int]", pil.getdata()))


def frames_differ(
    a: tuple[float, ...], b: tuple[float, ...], *, threshold: float = 4.0
) -> bool:
    return cell_mad(a, b) >= threshold


# Back-compat alias used by earlier drafts / tests.
def grid_hashes(
    image: ScreenImage, *, cols: int = GRID_COLS, rows: int = GRID_ROWS
) -> list[int]:
    """Legacy int hashes derived from cell mean luminance (for simple equality checks)."""
    sigs = grid_signatures(image, cols=cols, rows=rows)
    return [int(sum(s) / max(1, len(s))) for s in sigs]


class GridSettle:
    """Stateful grid-based stability detector that tolerates live animations.

    A cell that flips on every sample (spinner, video, Lottie) is masked out after
    ``streak`` consecutive changes. The screen is "settled" when all non-masked cells
    are unchanged vs the previous sample.
    """

    def __init__(
        self,
        *,
        cols: int = GRID_COLS,
        rows: int = GRID_ROWS,
        streak: int = ANIMATION_STREAK,
        mad_threshold: float = CELL_MAD_THRESHOLD,
    ) -> None:
        self.cols = cols
        self.rows = rows
        self.streak = streak
        self.mad_threshold = mad_threshold
        n = cols * rows
        self._prev: list[tuple[float, ...]] | None = None
        self._change_streak: list[int] = [0] * n
        self._masked: list[bool] = [False] * n
        self.samples = 0

    @property
    def masked_cells(self) -> list[int]:
        """Indices of cells currently flagged as live animation."""
        return [i for i, m in enumerate(self._masked) if m]

    def feed(self, image: ScreenImage) -> bool:
        """Feed a new frame. Returns True if the non-animated portion is stable."""
        sigs = grid_signatures(image, cols=self.cols, rows=self.rows)
        self.samples += 1
        if self._prev is None:
            self._prev = sigs
            return False
        n = len(sigs)
        all_stable = True
        for i in range(n):
            if self._masked[i]:
                continue
            changed = cell_mad(sigs[i], self._prev[i]) >= self.mad_threshold
            if changed:
                self._change_streak[i] += 1
                if self._change_streak[i] >= self.streak:
                    self._masked[i] = True
                all_stable = False
            else:
                self._change_streak[i] = 0
        self._prev = sigs
        return all_stable


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
