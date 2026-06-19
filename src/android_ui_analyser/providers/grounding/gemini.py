"""``gemini`` grounding provider — Gemini vision via ``generateContent``.

POSTs to ``{base_url}/models/{model}:generateContent``. The key is passed in the
``x-goog-api-key`` header (NOT the URL/query, to avoid leaking it in request logs) and
is read at runtime from the env var named by ``settings["api_key_env"]`` (default
``GEMINI_API_KEY``). The screenshot rides as an ``inline_data`` part with a strict
JSON-only prompt.
"""

from __future__ import annotations

from typing import Any

import httpx

from ...config import read_env_secret
from ..base import Availability, DetBox, GroundingProvider, Point, ScreenImage
from ..registry import register_grounding
from ._common import (
    SYSTEM_PROMPT,
    build_user_prompt,
    image_b64,
    parse_grounding_json,
)
from .openai import _commercial_availability

DEFAULT_TIMEOUT_S = 30.0


@register_grounding("gemini")
class GeminiGrounding(GroundingProvider):
    """Gemini ``generateContent`` grounding (instruction -> point/box)."""

    def is_available(self) -> Availability:
        return _commercial_availability(self.settings)

    def _timeout_s(self) -> float:
        return float(self.settings.get("timeout_s", DEFAULT_TIMEOUT_S))

    def _payload(self, image: ScreenImage, instruction: str) -> dict[str, Any]:
        # Gemini has no separate system role here; fold the system prompt into the text.
        text = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(image, instruction)}"
        return {
            "contents": [
                {
                    "parts": [
                        {"text": text},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64(image),
                            }
                        },
                    ]
                }
            ]
        }

    def locate(self, image: ScreenImage, instruction: str) -> Point | DetBox | None:
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(
            self.settings.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        ).rstrip("/")
        model = self.settings.get("model")
        resp = httpx.post(
            f"{base_url}/models/{model}:generateContent",
            json=self._payload(image, instruction),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key or "",
            },
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        return parse_grounding_json(_extract_text(resp.json()), image, settings=self.settings)


def _extract_text(data: Any) -> str | None:
    """Concatenate text parts from the first candidate of a Gemini response."""
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(parts, list):
        return None
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "".join(texts) or None
