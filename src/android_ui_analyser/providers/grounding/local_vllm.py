"""``local_vllm`` grounding provider — any OpenAI-compatible vision endpoint.

Targets a local server (vLLM / Ollama / LM Studio / HF TGI) exposing
``/chat/completions``. Configure ``base_url`` + ``model`` under ``models.local_vllm``
(e.g. ``Hcompany/Holo1.5-7B``). No API key is required; if ``api_key_env`` is set and
present it is sent as a bearer token (some local gateways want one).
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
    image_data_url,
    parse_grounding_json,
)

DEFAULT_TIMEOUT_S = 30.0


@register_grounding("local_vllm")
class LocalVllmGrounding(GroundingProvider):
    """OpenAI-compatible chat/completions grounding against a local endpoint."""

    def is_available(self) -> Availability:
        base_url = self.settings.get("base_url")
        if not base_url:
            return Availability(False, "local_vllm base_url not set")
        return Availability(True, "ok")

    def _timeout_s(self) -> float:
        return float(self.settings.get("timeout_s", DEFAULT_TIMEOUT_S))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = read_env_secret(self.settings.get("api_key_env"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, image: ScreenImage, instruction: str) -> dict[str, Any]:
        return {
            "model": self.settings.get("model"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(image, instruction)},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(image)},
                        },
                    ],
                },
            ],
        }

    def locate(self, image: ScreenImage, instruction: str) -> Point | DetBox | None:
        base_url = str(self.settings["base_url"]).rstrip("/")
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=self._payload(image, instruction),
            headers=self._headers(),
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        return parse_grounding_json(_extract_text(resp.json()), image, settings=self.settings)


def _extract_text(data: Any) -> str | None:
    """Pull assistant text out of an OpenAI chat/completions response shape."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some servers return content parts
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts) or None
    return None
