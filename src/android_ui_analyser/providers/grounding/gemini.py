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
    image_b64,
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
    preview_b64,
)

DEFAULT_TIMEOUT_S = 30.0


@register_grounding("gemini")
class GeminiGrounding(GroundingProvider):
    """Gemini ``generateContent`` grounding (instruction -> point/box)."""

    def is_available(self) -> Availability:
        return commercial_availability(self.settings)

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

    def ask(
        self,
        image: ScreenImage,
        question: str,
        elements: list[dict[str, Any]],
    ) -> ScreenAnalysisResult | None:
        """Answer a screen question through the same provider-neutral contract as OpenAI."""
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(
            self.settings.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        ).rstrip("/")
        model = self.settings.get("model")
        graph_limit = int(
            self.settings.get("screen_graph_max_elements", DEFAULT_SCREEN_GRAPH_MAX_ELEMENTS)
        )
        image_data, preview = prepare_screen_preview(image, self.settings)
        prompt = build_screen_prompt(
            image,
            question,
            elements,
            preview,
            graph_limit=graph_limit,
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SCREEN_SYSTEM_PROMPT}\n\n{prompt}"},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": preview_b64(image_data),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": SCREEN_ANALYSIS_SCHEMA,
                "maxOutputTokens": int(
                    self.settings.get("screen_max_completion_tokens", DEFAULT_SCREEN_MAX_TOKENS)
                ),
            },
        }
        resp = httpx.post(
            f"{base_url}/models/{model}:generateContent",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key or "",
            },
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        data = resp.json()
        analysis = parse_screen_analysis(_extract_text(data))
        if analysis is None:
            return None
        return ScreenAnalysisResult(
            model=data.get("modelVersion") or model,
            analysis=analysis,
            usage=_usage(data),
            input_image=preview,
        )


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


def _usage(data: Any) -> dict[str, Any]:
    """Normalize Gemini token counters to the names returned by OpenAI."""
    raw = data.get("usageMetadata") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    aliases = {
        "prompt_tokens": ("promptTokenCount", "prompt_token_count"),
        "completion_tokens": ("candidatesTokenCount", "candidates_token_count"),
        "total_tokens": ("totalTokenCount", "total_token_count"),
    }
    usage: dict[str, Any] = {}
    for output_name, input_names in aliases.items():
        for input_name in input_names:
            if input_name in raw:
                usage[output_name] = raw[input_name]
                break
    return usage
