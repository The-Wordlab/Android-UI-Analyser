"""Planner-provider tests (PRD §7.3) — mirror the grounding-provider suite.

Covers the Gemini Flash Lite planner: HTTP wiring with mocked responses (respx), the
JSON→PlannerDecision parse (plain + code-fenced + malformed), availability gating, that
the request is text-only by default and attaches the image only when passed, and that
the dummy key never leaks. Pure unit tests on `_common` too. No real API call.
"""

from __future__ import annotations

import json

import httpx
import respx

from android_ui_analyser.providers.base import PlannerDecision
from android_ui_analyser.providers.planner._common import (
    build_user_prompt,
    parse_planner_json,
    render_elements,
)
from android_ui_analyser.providers.planner.gemini_flash import GeminiFlashPlanner
from android_ui_analyser.providers.registry import get_provider_class
from conftest import make_screen_image

DUMMY_KEY = "gk-test-DUMMY-do-not-log-98765"
BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-2.5-flash-lite"
ELEMENTS = [
    {"id": 3, "label": "Settings", "clickable": True},
    {"id": 7, "label": "Not now", "clickable": True},
]


def _body(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _provider() -> GeminiFlashPlanner:
    return GeminiFlashPlanner({"model": MODEL, "api_key_env": "GEMINI_API_KEY", "base_url": BASE})


# --------------------------------------------------------------------------- HTTP wiring


@respx.mock
def test_decide_plain_json(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    respx.post(f"{BASE}/models/{MODEL}:generateContent").mock(
        return_value=httpx.Response(200, json=_body('{"action":"tap","id":3,"reason":"go"}'))
    )
    d = _provider().decide("open settings", ELEMENTS)
    assert d == PlannerDecision(action="tap", target_id=3, reason="go")


@respx.mock
def test_decide_code_fenced_json(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    fenced = "Sure:\n```json\n{\"action\": \"done\", \"reason\": \"already here\"}\n```"
    respx.post(f"{BASE}/models/{MODEL}:generateContent").mock(
        return_value=httpx.Response(200, json=_body(fenced))
    )
    d = _provider().decide("open settings", ELEMENTS)
    assert d is not None and d.action == "done"


@respx.mock
def test_decide_is_text_only_by_default_and_key_in_header(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    route = respx.post(f"{BASE}/models/{MODEL}:generateContent").mock(
        return_value=httpx.Response(200, json=_body('{"action":"tap","id":3}'))
    )
    _provider().decide("open settings", ELEMENTS)  # no image passed
    req = route.calls.last.request
    assert req.headers["x-goog-api-key"] == DUMMY_KEY
    assert DUMMY_KEY not in str(req.url)
    parts = json.loads(req.content)["contents"][0]["parts"]
    assert not any("inline_data" in p for p in parts)  # text-only → cheap
    # the element list rode in the prompt
    assert "Settings" in parts[0]["text"] and "[3]" in parts[0]["text"]


@respx.mock
def test_decide_attaches_image_when_provided(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    route = respx.post(f"{BASE}/models/{MODEL}:generateContent").mock(
        return_value=httpx.Response(200, json=_body('{"action":"give-up"}'))
    )
    _provider().decide("open settings", ELEMENTS, image=make_screen_image(100, 200))
    parts = json.loads(route.calls.last.request.content)["contents"][0]["parts"]
    assert any("inline_data" in p for p in parts)  # vision only when explicitly needed


# --------------------------------------------------------------------------- availability


def test_unavailable_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    avail = _provider().is_available()
    assert avail.ok is False and "GEMINI_API_KEY" in avail.reason
    assert DUMMY_KEY not in avail.reason


def test_registered_by_name() -> None:
    assert get_provider_class("planner", "gemini_flash") is GeminiFlashPlanner


# --------------------------------------------------------------------------- parse units


def test_parse_variants() -> None:
    assert parse_planner_json('{"action":"tap","id":"5"}').target_id == 5  # string id coerced
    assert parse_planner_json('{"action":"key","arg":"back"}').arg == "back"
    assert parse_planner_json('prose {"action":"done"} trailing').action == "done"
    assert parse_planner_json('{"action":"frobnicate","id":1}') is None  # unknown action
    assert parse_planner_json("not json at all") is None
    assert parse_planner_json("") is None
    assert parse_planner_json('{"action":"tap","id":true}').target_id is None  # bool != id


def test_render_elements_compact() -> None:
    out = render_elements(ELEMENTS)
    assert '[3] "Settings" (clickable)' in out
    assert '[7] "Not now" (clickable)' in out
    assert build_user_prompt("goal x", ELEMENTS).startswith("GOAL: goal x")
