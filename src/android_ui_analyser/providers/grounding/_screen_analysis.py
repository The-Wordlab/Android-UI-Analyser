"""Shared contract and image preparation for provider-neutral screen questions."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping
from typing import Any

from ..base import ScreenImage
from ._common import _first_json_object, _strip_fences

DEFAULT_SCREEN_MAX_TOKENS = 1200
# The original 360px / quality-35 preview saved bytes but damaged small text without a
# measurable latency win. Keep a meaningful reduction from a 1080px device screenshot
# while retaining enough detail for icons and visual-only labels.
DEFAULT_SCREEN_PREVIEW_MAX_WIDTH = 720
DEFAULT_SCREEN_PREVIEW_JPEG_QUALITY = 55
DEFAULT_SCREEN_GRAPH_MAX_ELEMENTS = 160

SCREEN_SYSTEM_PROMPT = (
    "You analyze Android screens for a UI automation agent. Use BOTH the screenshot and "
    "the supplied UI element graph: pixels are the visual truth; graph ids, text, roles, "
    "states, and bounds provide exact structured evidence. Answer the user's question "
    "directly. For a screen description, walk top-to-bottom and identify regions such as "
    "header, body, overlays, and bottom navigation. For locations, include pixel bounds and "
    "the graph id when available. The image may be a downscaled preview; always report bounds "
    "in the original graph.screen coordinate space. Never invent a graph id. Keep the answer "
    "under 120 words, list at most 8 regions and 20 relevant elements, and do not duplicate "
    "elements inside regions."
)

_BOUNDS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 4,
    "maxItems": 4,
}

SCREEN_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "screen_summary": {"type": "string"},
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "bounds": _BOUNDS_SCHEMA,
                    "description": {"type": "string"},
                },
                "required": ["name", "bounds", "description"],
                "additionalProperties": False,
            },
        },
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": ["integer", "null"]},
                    "role": {"type": "string"},
                    "text": {"type": ["string", "null"]},
                    "bounds": _BOUNDS_SCHEMA,
                    "position": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "enum": ["graph", "image", "both"],
                    },
                },
                "required": [
                    "graph_id",
                    "role",
                    "text",
                    "bounds",
                    "position",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "screen_summary", "regions", "elements", "uncertainties"],
    "additionalProperties": False,
}


def prepare_screen_preview(
    image: ScreenImage, settings: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    """Create the shared, moderately compressed JPEG used by remote screen analyzers."""
    max_width = max(
        1,
        int(settings.get("screen_preview_max_width", DEFAULT_SCREEN_PREVIEW_MAX_WIDTH)),
    )
    quality = max(
        10,
        min(
            95,
            int(settings.get("screen_preview_jpeg_quality", DEFAULT_SCREEN_PREVIEW_JPEG_QUALITY)),
        ),
    )
    preview = image.pil()
    if preview.width > max_width:
        height = max(1, round(preview.height * max_width / preview.width))
        from PIL import Image

        preview = preview.resize((max_width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    return data, {
        "original_width": image.width,
        "original_height": image.height,
        "width": preview.width,
        "height": preview.height,
        "bytes": len(data),
        "format": "jpeg",
        "quality": quality,
    }


def preview_data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def preview_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_screen_prompt(
    image: ScreenImage,
    question: str,
    elements: list[dict[str, Any]],
    preview: Mapping[str, Any],
    *,
    graph_limit: int = DEFAULT_SCREEN_GRAPH_MAX_ELEMENTS,
) -> str:
    """Build one provider-independent question + compact graph prompt."""
    graph = {
        "screen": {
            "width": image.width,
            "height": image.height,
            "preview_width": preview["width"],
            "preview_height": preview["height"],
        },
        "elements": elements[: max(0, graph_limit)],
    }
    return f"QUESTION: {question}\n\nUI_ELEMENT_GRAPH:\n" + json.dumps(
        graph, ensure_ascii=False, separators=(",", ":")
    )


def parse_screen_analysis(text: str | None) -> dict[str, Any] | None:
    """Parse a provider response while allowing a useful plain-text fallback."""
    if not text:
        return None
    candidate = _first_json_object(_strip_fences(text))
    if candidate is None:
        return {"answer": text.strip()}
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
