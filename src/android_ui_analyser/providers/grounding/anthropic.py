"""``anthropic`` grounding provider — Claude vision via the Messages API.

POSTs to ``{base_url}/messages`` with an ``x-api-key`` header and the screenshot as a
base64 ``image`` content block, plus a strict JSON-only prompt. The key is read at
runtime from the env var named by ``settings["api_key_env"]`` (default
``ANTHROPIC_API_KEY``); it is never stored in config or logged.
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
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


@register_grounding("anthropic")
class AnthropicGrounding(GroundingProvider):
    """Claude Messages-API grounding (instruction -> point/box)."""

    def is_available(self) -> Availability:
        return _commercial_availability(self.settings)

    def _timeout_s(self) -> float:
        return float(self.settings.get("timeout_s", DEFAULT_TIMEOUT_S))

    def _payload(self, image: ScreenImage, instruction: str) -> dict[str, Any]:
        return {
            "model": self.settings.get("model"),
            "max_tokens": int(self.settings.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(image, instruction)},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64(image),
                            },
                        },
                    ],
                }
            ],
        }

    def locate(self, image: ScreenImage, instruction: str) -> Point | DetBox | None:
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(self.settings.get("base_url", "https://api.anthropic.com/v1")).rstrip("/")
        resp = httpx.post(
            f"{base_url}/messages",
            json=self._payload(image, instruction),
            headers={
                "Content-Type": "application/json",
                "x-api-key": key or "",
                "anthropic-version": ANTHROPIC_VERSION,
            },
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        return parse_grounding_json(_extract_text(resp.json()), image, settings=self.settings)


def _extract_text(data: Any) -> str | None:
    """Concatenate text blocks from an Anthropic Messages response."""
    content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(content, list):
        return None
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts) or None
