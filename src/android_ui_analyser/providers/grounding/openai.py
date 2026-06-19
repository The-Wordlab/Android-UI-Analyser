"""``openai`` grounding provider — GPT-class vision via the OpenAI REST API.

POSTs to ``{base_url}/chat/completions`` with the screenshot as an ``image_url`` content
part and a strict JSON-only prompt. The key is read at runtime from the env var named by
``settings["api_key_env"]`` (default ``OPENAI_API_KEY``) and sent as a bearer token; it
is never stored in config or logged.
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
from .local_vllm import _extract_text

DEFAULT_TIMEOUT_S = 30.0


@register_grounding("openai")
class OpenAiGrounding(GroundingProvider):
    """OpenAI-compatible chat/completions grounding against the OpenAI API."""

    def is_available(self) -> Availability:
        return _commercial_availability(self.settings)

    def _timeout_s(self) -> float:
        return float(self.settings.get("timeout_s", DEFAULT_TIMEOUT_S))

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
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(self.settings.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=self._payload(image, instruction),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        return parse_grounding_json(_extract_text(resp.json()), image, settings=self.settings)


def _commercial_availability(settings: dict[str, Any]) -> Availability:
    """Shared key-presence check for commercial providers (no network).

    OK only when ``api_key_env`` is set AND that env var is present. The reason names the
    env var only — never a secret value.
    """
    env_name = settings.get("api_key_env")
    if not env_name:
        return Availability(False, "api_key_env not configured")
    if read_env_secret(env_name) is None:
        return Availability(False, f"{env_name} not set")
    return Availability(True, "ok")
