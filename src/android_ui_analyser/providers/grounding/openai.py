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
from ..base import (
    Availability,
    DetBox,
    GroundingProvider,
    Point,
    ScreenAnalysisResult,
    ScreenImage,
)
from ..registry import register_grounding
from ._common import (
    SYSTEM_PROMPT,
    build_user_prompt,
    commercial_availability,
    image_data_url,
    parse_grounding_json,
)
from ._screen_analysis import (
    DEFAULT_SCREEN_GRAPH_MAX_ELEMENTS,
    DEFAULT_SCREEN_MAX_TOKENS,
    SCREEN_ANALYSIS_SCHEMA,
    SCREEN_SYSTEM_PROMPT,
    build_screen_prompt,
    parse_screen_analysis,
    prepare_screen_preview,
    preview_data_url,
)
from .local_vllm import _extract_text

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_SCREEN_IMAGE_DETAIL = "high"

SCREEN_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "screen_analysis",
        "strict": True,
        "schema": SCREEN_ANALYSIS_SCHEMA,
    },
}


@register_grounding("openai")
class OpenAiGrounding(GroundingProvider):
    """OpenAI-compatible chat/completions grounding against the OpenAI API."""

    def is_available(self) -> Availability:
        return commercial_availability(self.settings)

    def _timeout_s(self) -> float:
        return float(self.settings.get("timeout_s", DEFAULT_TIMEOUT_S))

    def _payload(self, image: ScreenImage, instruction: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        reasoning_effort = self.settings.get("reasoning_effort")
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        return payload

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

    def ask(
        self,
        image: ScreenImage,
        question: str,
        elements: list[dict[str, Any]],
    ) -> ScreenAnalysisResult | None:
        """Answer a screen question from the screenshot fused with AUA's element graph."""
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(self.settings.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        graph_limit = int(
            self.settings.get("screen_graph_max_elements", DEFAULT_SCREEN_GRAPH_MAX_ELEMENTS)
        )
        image_url, preview = self._screen_preview(image)
        payload: dict[str, Any] = {
            "model": self.settings.get("model"),
            "messages": [
                {"role": "developer", "content": SCREEN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_screen_prompt(
                                image,
                                question,
                                elements,
                                preview,
                                graph_limit=graph_limit,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": self.settings.get(
                                    "screen_image_detail", DEFAULT_SCREEN_IMAGE_DETAIL
                                ),
                            },
                        },
                    ],
                },
            ],
            "response_format": SCREEN_RESPONSE_FORMAT,
            "max_completion_tokens": int(
                self.settings.get("screen_max_completion_tokens", DEFAULT_SCREEN_MAX_TOKENS)
            ),
        }
        reasoning_effort = self.settings.get("reasoning_effort")
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        data = resp.json()
        text = _extract_text(data)
        analysis = parse_screen_analysis(text)
        if analysis is None:
            return None
        usage = data.get("usage")
        return ScreenAnalysisResult(
            model=data.get("model") or self.settings.get("model"),
            analysis=analysis,
            usage=usage if isinstance(usage, dict) else {},
            input_image=preview,
        )

    def _screen_preview(self, image: ScreenImage) -> tuple[str, dict[str, Any]]:
        """Return a JPEG preview; graph bounds remain in original screen pixels."""
        data, metadata = prepare_screen_preview(image, self.settings)
        metadata["detail"] = self.settings.get("screen_image_detail", DEFAULT_SCREEN_IMAGE_DETAIL)
        return preview_data_url(data), metadata
