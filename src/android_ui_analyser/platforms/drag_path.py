"""The interpolated path every touch-based runtime follows for a drag."""

from __future__ import annotations

_STEP_MS = 16
_MAX_STEPS = 40


def drag_path(
    x1: int, y1: int, x2: int, y2: int, duration_ms: int
) -> tuple[list[tuple[int, int]], float]:
    """Evenly spaced points after the start, ending exactly at ``(x2, y2)``.

    Returns the points and the pause in seconds between them, so the whole path takes about
    *duration_ms*.
    """

    steps = max(2, min(_MAX_STEPS, duration_ms // _STEP_MS))
    points = [
        (round(x1 + (x2 - x1) * i / steps), round(y1 + (y2 - y1) * i / steps))
        for i in range(1, steps + 1)
    ]
    return points, max(duration_ms, 0) / 1000.0 / steps
