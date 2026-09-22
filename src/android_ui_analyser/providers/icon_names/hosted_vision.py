"""Name one cropped icon through an OpenAI-compatible vision endpoint.

The default target is OpenRouter with a small DeepSeek vision model: measured on a hamburger
icon it answered "Hamburger menu button, opens the side drawer" three times out of three in
0.6-2.6 s for $0.00004-0.00008. The crop is upscaled before it is sent, because a 96 px icon
is a blur to a model that resizes its input.

Settings (``models.hosted_vision``): ``model``, ``base_url``, ``api_key_env``, ``timeout_s``
and an optional ``reasoning`` object passed through as-is (OpenRouter turns a model's chain
of thought off with ``{"enabled": false}``; a 30 s think about a 96 px icon is what it costs
otherwise).
"""

from __future__ import annotations

from typing import Any

import httpx

from ...config import read_env_secret
from ..base import Availability, IconNamerProvider, ScreenImage
from ..grounding._common import commercial_availability, image_data_url
from ..registry import register_icon_names

PROMPT = (
    "This is one control cropped from a mobile app screen. In at most 8 words, say what "
    "control it is and what it does, for example 'hamburger menu button, opens the side "
    "drawer'. If nothing is drawn, answer 'nothing'."
)
MIN_SEND_PX = 192


@register_icon_names("hosted_vision")
class HostedVisionNamer(IconNamerProvider):
    def is_available(self) -> Availability:
        if not self.settings.get("model"):
            return Availability(False, "model not configured")
        return commercial_availability(self.settings)

    def _payload(self, image: ScreenImage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.get("model"),
            "max_tokens": 40,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(_upscaled(image))}},
                ],
            }],
        }
        reasoning = self.settings.get("reasoning")
        if isinstance(reasoning, dict):
            payload["reasoning"] = reasoning
        return payload

    def name_icon(self, image: ScreenImage) -> str | None:
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(self.settings.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/")
        response = httpx.post(
            f"{base_url}/chat/completions",
            json=self._payload(image),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            timeout=float(self.settings.get("timeout_s", 8.0)),
        )
        response.raise_for_status()
        text = _content(response.json())
        if not text:
            return None
        if text.strip().lower().rstrip(".") == "nothing":
            return ""
        return text


def _upscaled(image: ScreenImage) -> ScreenImage:
    if min(image.width, image.height) >= MIN_SEND_PX:
        return image
    from PIL import Image

    scale = max(2, -(-MIN_SEND_PX // max(1, min(image.width, image.height))))
    pil = image.pil()
    return ScreenImage.from_pil(pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.LANCZOS))


def _content(body: Any) -> str | None:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, list):  # some endpoints return content parts
        content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content) if content else None
