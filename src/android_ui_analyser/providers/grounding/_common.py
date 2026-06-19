"""Shared helpers for grounding (instruction -> coordinate) VLM providers.

Two concerns live here so every provider stays tiny and consistent:

1. **Request building** — encode the screenshot as a base64 PNG and build a strict
   system/user prompt that pins the model to PIXEL coordinates for *this* image and to
   JSON-only output (one of ``{"point":[x,y]}``, ``{"box":[x1,y1,x2,y2]}``,
   ``{"found":false}``).
2. **Defensive parsing** — pull the text out of a (provider-specific) response, strip
   markdown fences / surrounding prose, find the first balanced ``{...}`` object,
   ``json.loads`` it, and map it to a :class:`Point` / :class:`DetBox` / ``None``.
   Coordinates are clamped to the image; normalized (``<= 1.0``) or 0-1000 spaces are
   rescaled to pixels. The parser **never raises** — a bad response simply yields
   ``None`` so the fallback chain can advance.

Network errors are intentionally *not* swallowed here; providers let httpx errors
propagate so the chain runner logs them and moves on.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

from ..base import DetBox, Point

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..base import ScreenImage

# --------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = (
    "You are a precise UI grounding model for Android screenshots. "
    "Given a screenshot and an instruction, you return the on-screen location that "
    "best satisfies the instruction. You respond with JSON ONLY — no prose, no "
    "markdown, no code fences. Use EXACTLY one of these shapes, with integer PIXEL "
    'coordinates measured from the top-left of THIS image: {"point":[x,y]} for a click '
    'target, {"box":[x1,y1,x2,y2]} for a bounding box, or {"found":false} if the '
    "instruction does not match anything on screen."
)


def build_user_prompt(image: ScreenImage, instruction: str) -> str:
    """A strict per-image instruction that restates the size and the JSON contract."""
    return (
        f"The screenshot is {image.width}x{image.height} pixels "
        f"(width x height). Locate: {instruction}\n"
        "Return PIXEL coordinates for this exact image as JSON only, one of: "
        '{"point":[x,y]}, {"box":[x1,y1,x2,y2]}, or {"found":false}.'
    )


def image_data_url(image: ScreenImage) -> str:
    """The screenshot as a ``data:image/png;base64,...`` URL (OpenAI-style content)."""
    return f"data:image/png;base64,{image_b64(image)}"


def image_b64(image: ScreenImage) -> str:
    """Raw base64 PNG (no data-URL prefix; for Anthropic/Gemini inline blocks)."""
    return base64.b64encode(image.png_bytes).decode("ascii")


# --------------------------------------------------------------------------- parsing


def _strip_fences(text: str) -> str:
    """Remove a single surrounding markdown code fence if present.

    Handles ```json\n...\n``` and ```\n...\n```. If no fence is found the text is
    returned unchanged (the balanced-brace scan below still copes with prose).
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # Drop the opening fence line (``` or ```json / ```JSON etc).
    newline = s.find("\n")
    if newline == -1:
        return s
    inner = s[newline + 1 :]
    end = inner.rfind("```")
    if end != -1:
        inner = inner[:end]
    return inner.strip()


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, respecting strings/escapes."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _as_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        out.append(float(v))
    return out


def _clamp(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def _scale_factor(coords: list[float], settings: dict[str, Any] | None) -> tuple[bool, float]:
    """Decide how to map model coords into pixels.

    Returns ``(normalized, divisor)``:
    - explicit ``coordinate_space`` setting wins: ``"0-1000"`` -> divide by 1000 then
      multiply by size; ``"normalized"`` / ``"0-1"`` -> multiply by size; ``"pixels"``
      / ``"absolute"`` -> identity.
    - otherwise: if every value is ``<= 1.0`` treat as normalized (multiply by size).
    """
    space = ""
    if settings:
        space = str(settings.get("coordinate_space", "")).strip().lower()
    if space in {"0-1000", "1000", "0_1000"}:
        return True, 1000.0
    if space in {"normalized", "0-1", "0_1", "norm"}:
        return True, 1.0
    if space in {"pixels", "pixel", "absolute", "abs"}:
        return False, 1.0
    if coords and all(c <= 1.0 for c in coords):
        return True, 1.0
    return False, 1.0


def parse_grounding_json(
    text: str | None,
    image: ScreenImage,
    *,
    settings: dict[str, Any] | None = None,
) -> Point | DetBox | None:
    """Best-effort parse of a VLM grounding reply into a :class:`Point`/:class:`DetBox`.

    Returns ``None`` for "not found" or any malformed/unrecognized response. Never
    raises.
    """
    if not text:
        return None
    try:
        candidate = _first_json_object(_strip_fences(text))
        if candidate is None:
            return None
        data = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    # Explicit not-found.
    if data.get("found") is False:
        return None

    w, h = image.width, image.height

    point = _as_float_list(data.get("point"))
    if point is not None and len(point) >= 2:
        normalized, divisor = _scale_factor(point[:2], settings)
        x, y = point[0], point[1]
        if normalized:
            x = x / divisor * w
            y = y / divisor * h
        return Point(x=_clamp(x, 0, w), y=_clamp(y, 0, h))

    box = _as_float_list(data.get("box") or data.get("bbox") or data.get("bounds"))
    if box is not None and len(box) >= 4:
        normalized, divisor = _scale_factor(box[:4], settings)
        vals = list(box[:4])
        if normalized:
            vals = [
                vals[0] / divisor * w,
                vals[1] / divisor * h,
                vals[2] / divisor * w,
                vals[3] / divisor * h,
            ]
        x1 = _clamp(vals[0], 0, w)
        y1 = _clamp(vals[1], 0, h)
        x2 = _clamp(vals[2], 0, w)
        y2 = _clamp(vals[3], 0, h)
        # Normalize ordering so x1<=x2, y1<=y2.
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return DetBox(bounds=(x1, y1, x2, y2))

    return None
