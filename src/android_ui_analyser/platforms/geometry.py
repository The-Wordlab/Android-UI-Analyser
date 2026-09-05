"""Coordinate-space contract shared by platform adapters and AUA providers.

AUA publishes element, OCR, detection, grounding, annotation, and input coordinates in one
space: physical pixels in the screenshot returned for that frame. Native automation APIs may
instead use logical points, rotated coordinates, or a cropped viewport. ``DisplayGeometry``
stores the adapter-supplied affine transform between those native coordinates and AUA's
canonical screenshot-pixel space.

The transform is intentionally explicit rather than inferred from a scale factor. This covers
Retina scale, rotation, mirroring, and viewport offsets without teaching the shared engine any
platform-specific orientation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..providers.base import Bounds

Point = tuple[float, float]
AffineTransform = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class DisplayGeometry:
    """Map native automation coordinates to canonical screenshot pixels.

    ``native_to_canonical`` uses the conventional six-value affine representation
    ``(a, b, c, d, tx, ty)``::

        canonical_x = a * native_x + c * native_y + tx
        canonical_y = b * native_x + d * native_y + ty

    The matrix must be invertible. Dimensions describe the complete native and canonical
    surfaces and are metadata for validation; the affine transform remains the source of truth.
    """

    canonical_size: tuple[int, int]
    native_size: tuple[float, float]
    native_to_canonical: AffineTransform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        canonical_width, canonical_height = self.canonical_size
        native_width, native_height = self.native_size
        if canonical_width <= 0 or canonical_height <= 0:
            raise ValueError("canonical display dimensions must be positive")
        if native_width <= 0 or native_height <= 0:
            raise ValueError("native display dimensions must be positive")
        if not all(isfinite(value) for value in self.native_to_canonical):
            raise ValueError("display transform values must be finite")
        a, b, c, d, _tx, _ty = self.native_to_canonical
        if abs(a * d - b * c) < 1e-12:
            raise ValueError("display transform must be invertible")

    @classmethod
    def identity(cls, width: int, height: int) -> DisplayGeometry:
        """A runtime whose automation and screenshot coordinates already match."""

        return cls(canonical_size=(width, height), native_size=(width, height))

    @classmethod
    def scaled(
        cls,
        *,
        native_size: tuple[float, float],
        canonical_size: tuple[int, int],
        offset: Point = (0.0, 0.0),
    ) -> DisplayGeometry:
        """Create the common axis-aligned scale-and-offset mapping."""

        native_width, native_height = native_size
        canonical_width, canonical_height = canonical_size
        scale_x = canonical_width / native_width
        scale_y = canonical_height / native_height
        return cls(
            canonical_size=canonical_size,
            native_size=native_size,
            native_to_canonical=(scale_x, 0.0, 0.0, scale_y, offset[0], offset[1]),
        )

    def to_canonical(self, point: Point) -> Point:
        """Transform one native point into screenshot-pixel coordinates."""

        x, y = point
        a, b, c, d, tx, ty = self.native_to_canonical
        return (a * x + c * y + tx, b * x + d * y + ty)

    def to_native(self, point: Point) -> Point:
        """Transform one screenshot-pixel point into native automation coordinates."""

        x, y = point
        a, b, c, d, tx, ty = self.native_to_canonical
        determinant = a * d - b * c
        translated_x = x - tx
        translated_y = y - ty
        return (
            (d * translated_x - c * translated_y) / determinant,
            (-b * translated_x + a * translated_y) / determinant,
        )

    def bounds_to_canonical(self, bounds: tuple[float, float, float, float]) -> Bounds:
        """Transform a native rectangle and enclose every transformed corner."""

        x1, y1, x2, y2 = bounds
        corners = (
            self.to_canonical((x1, y1)),
            self.to_canonical((x1, y2)),
            self.to_canonical((x2, y1)),
            self.to_canonical((x2, y2)),
        )
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        return (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))

    def canonical_bounds_to_native(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Inverse-transform a canonical rectangle and enclose its native corners."""

        x1, y1, x2, y2 = bounds
        corners = (
            self.to_native((x1, y1)),
            self.to_native((x1, y2)),
            self.to_native((x2, y1)),
            self.to_native((x2, y2)),
        )
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        return (min(xs), min(ys), max(xs), max(ys))


__all__ = ["AffineTransform", "DisplayGeometry", "Point"]
