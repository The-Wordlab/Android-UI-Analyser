"""``gemini_flash`` planner provider — Gemini Flash Lite via ``generateContent``.

Given a goal + the on-screen element list, returns the next action (a
:class:`PlannerDecision`). Text-only by default (cheap); the screenshot is attached as
an ``inline_data`` part only when the caller passes one (weakly-labelled screens). The
key rides in the ``x-goog-api-key`` header, read at runtime from ``api_key_env``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ...config import read_env_secret
from ..base import PlannerDecision, PlannerProvider, ScreenImage
from ..grounding._common import commercial_availability, image_b64
from ..registry import register_planner
from ._common import SYSTEM_PROMPT, build_user_prompt, parse_planner_json

DEFAULT_TIMEOUT_S = 15.0


@register_planner("gemini_flash")
class GeminiFlashPlanner(PlannerProvider):
    """Gemini Flash Lite planner (goal + elements -> next action)."""

    def is_available(self) -> Any:
        return commercial_availability(self.settings)

    def _timeout_s(self) -> float:
        return float(self.settings.get("timeout_s", DEFAULT_TIMEOUT_S))

    def _payload(
        self, objective: str, elements: list[dict[str, Any]], image: ScreenImage | None
    ) -> dict[str, Any]:
        text = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(objective, elements)}"
        parts: list[dict[str, Any]] = [{"text": text}]
        if image is not None:  # only on weakly-labelled screens (cost control)
            parts.append({"inline_data": {"mime_type": "image/png", "data": image_b64(image)}})
        return {"contents": [{"parts": parts}]}

    def decide(
        self,
        objective: str,
        elements: list[dict[str, Any]],
        image: ScreenImage | None = None,
    ) -> PlannerDecision | None:
        key = read_env_secret(self.settings.get("api_key_env"))
        base_url = str(
            self.settings.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        ).rstrip("/")
        model = self.settings.get("model")
        resp = httpx.post(
            f"{base_url}/models/{model}:generateContent",
            json=self._payload(objective, elements, image),
            headers={"Content-Type": "application/json", "x-goog-api-key": key or ""},
            timeout=self._timeout_s(),
        )
        resp.raise_for_status()
        return parse_planner_json(_extract_text(resp.json()))


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
