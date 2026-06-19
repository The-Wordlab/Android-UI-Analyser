"""Grounding-provider tests (PRD §7.2, §13.1 AC6, §14).

Covers:
- AC6 commercial wiring with mocked HTTP (respx): a plain JSON answer AND a code-fenced
  ```json {...}``` answer wrapped in prose both yield the right Point/DetBox, for
  gemini AND openai (plus anthropic + local_vllm for good measure).
- Availability: with the api_key_env var unset, ``is_available().ok`` is False and the
  reason names the env var; the dummy key value never leaks into reason/output.
- Defensive parsing unit tests on ``_common`` (fences, prose+JSON, found:false,
  malformed, normalized-coords scaling, 0-1000 space).
- Request shape: respx asserts the auth header carries the (dummy) key and the body
  includes the base64 image. No secret is ever printed.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from android_ui_analyser.providers.base import DetBox, Point
from android_ui_analyser.providers.grounding._common import (
    build_user_prompt,
    image_b64,
    parse_grounding_json,
)
from android_ui_analyser.providers.grounding.anthropic import AnthropicGrounding
from android_ui_analyser.providers.grounding.gemini import GeminiGrounding
from android_ui_analyser.providers.grounding.local_vllm import LocalVllmGrounding
from android_ui_analyser.providers.grounding.openai import OpenAiGrounding
from android_ui_analyser.providers.registry import get_provider_class
from conftest import make_screen_image

DUMMY_KEY = "sk-test-DUMMY-do-not-log-12345"
IMG_W, IMG_H = 200, 400


# --------------------------------------------------------------------------- helpers


def _img():
    return make_screen_image(IMG_W, IMG_H)


def _openai_body(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _anthropic_body(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _gemini_body(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


FENCED = (
    "Sure, here is the location you asked for:\n"
    "```json\n"
    '{"point": [120, 240]}\n'
    "```\n"
    "Let me know if you need anything else."
)


# --------------------------------------------------------------------------- registration


def test_providers_registered():
    for name, cls in [
        ("local_vllm", LocalVllmGrounding),
        ("openai", OpenAiGrounding),
        ("anthropic", AnthropicGrounding),
        ("gemini", GeminiGrounding),
    ]:
        assert get_provider_class("grounding", name) is cls
        assert cls.kind == "grounding"
        assert cls.name == name


# --------------------------------------------------------------------------- AC6: openai


@respx.mock
def test_openai_locate_plain_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_body('{"point": [50, 100]}'))
    )
    provider = OpenAiGrounding(
        {"model": "gpt-5", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"}
    )
    result = provider.locate(_img(), "the search box")
    assert result == Point(x=50, y=100)
    assert route.called


@respx.mock
def test_openai_locate_code_fenced_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_body(FENCED))
    )
    provider = OpenAiGrounding({"model": "gpt-5", "api_key_env": "OPENAI_API_KEY"})
    result = provider.locate(_img(), "the search box")
    assert result == Point(x=120, y=240)


@respx.mock
def test_openai_request_shape_carries_key_and_image(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_body('{"found": false}'))
    )
    provider = OpenAiGrounding({"model": "gpt-5", "api_key_env": "OPENAI_API_KEY"})
    provider.locate(_img(), "anything")

    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {DUMMY_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "gpt-5"
    # image present as a data URL inside the user content parts
    user = body["messages"][-1]
    image_parts = [p for p in user["content"] if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_b64(_img()) in image_parts[0]["image_url"]["url"]


# --------------------------------------------------------------------------- AC6: gemini


@respx.mock
def test_gemini_locate_plain_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    base = "https://generativelanguage.googleapis.com/v1beta"
    respx.post(f"{base}/models/gemini-2.5-flash:generateContent").mock(
        return_value=httpx.Response(200, json=_gemini_body('{"box": [10, 20, 60, 80]}'))
    )
    provider = GeminiGrounding(
        {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY", "base_url": base}
    )
    result = provider.locate(_img(), "the banner")
    assert result == DetBox(bounds=(10, 20, 60, 80))


@respx.mock
def test_gemini_locate_code_fenced_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    base = "https://generativelanguage.googleapis.com/v1beta"
    respx.post(f"{base}/models/gemini-2.5-flash:generateContent").mock(
        return_value=httpx.Response(200, json=_gemini_body(FENCED))
    )
    provider = GeminiGrounding(
        {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY", "base_url": base}
    )
    result = provider.locate(_img(), "the search box")
    assert result == Point(x=120, y=240)


@respx.mock
def test_gemini_key_in_header_not_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    base = "https://generativelanguage.googleapis.com/v1beta"
    route = respx.post(f"{base}/models/gemini-2.5-flash:generateContent").mock(
        return_value=httpx.Response(200, json=_gemini_body('{"found": false}'))
    )
    provider = GeminiGrounding(
        {"model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY", "base_url": base}
    )
    provider.locate(_img(), "anything")

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == DUMMY_KEY
    # The key must NOT leak into the URL/query string.
    assert DUMMY_KEY not in str(request.url)
    body = json.loads(request.content)
    parts = body["contents"][0]["parts"]
    inline = [p for p in parts if "inline_data" in p]
    assert inline and inline[0]["inline_data"]["data"] == image_b64(_img())


# --------------------------------------------------------------------------- anthropic


@respx.mock
def test_anthropic_locate_and_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    base = "https://api.anthropic.com/v1"
    route = respx.post(f"{base}/messages").mock(
        return_value=httpx.Response(200, json=_anthropic_body('{"point": [5, 5]}'))
    )
    provider = AnthropicGrounding(
        {"model": "claude-opus-4-8", "api_key_env": "ANTHROPIC_API_KEY", "base_url": base}
    )
    result = provider.locate(_img(), "the menu")
    assert result == Point(x=5, y=5)

    request = route.calls.last.request
    assert request.headers["x-api-key"] == DUMMY_KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(request.content)
    blocks = body["messages"][0]["content"]
    img_blocks = [b for b in blocks if b.get("type") == "image"]
    assert img_blocks and img_blocks[0]["source"]["data"] == image_b64(_img())
    assert img_blocks[0]["source"]["media_type"] == "image/png"


# --------------------------------------------------------------------------- local_vllm


@respx.mock
def test_local_vllm_locate_no_key_needed():
    base = "http://localhost:8000/v1"
    route = respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_body('{"point": [10, 10]}'))
    )
    provider = LocalVllmGrounding({"base_url": base, "model": "Hcompany/Holo1.5-7B"})
    result = provider.locate(_img(), "the icon")
    assert result == Point(x=10, y=10)
    # No Authorization header when no key configured.
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_local_vllm_sends_key_when_present(monkeypatch):
    monkeypatch.setenv("VLLM_KEY", DUMMY_KEY)
    base = "http://localhost:8000/v1"
    route = respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_body('{"found": false}'))
    )
    provider = LocalVllmGrounding({"base_url": base, "model": "m", "api_key_env": "VLLM_KEY"})
    provider.locate(_img(), "x")
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {DUMMY_KEY}"


# --------------------------------------------------------------------------- availability


def test_commercial_unavailable_when_key_unset_names_env_var(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cases = [
        (OpenAiGrounding({"api_key_env": "OPENAI_API_KEY"}), "OPENAI_API_KEY"),
        (GeminiGrounding({"api_key_env": "GEMINI_API_KEY"}), "GEMINI_API_KEY"),
        (AnthropicGrounding({"api_key_env": "ANTHROPIC_API_KEY"}), "ANTHROPIC_API_KEY"),
    ]
    for provider, env_name in cases:
        avail = provider.is_available()
        assert avail.ok is False
        assert env_name in avail.reason
        # The dummy secret value must never appear in the reason.
        assert DUMMY_KEY not in avail.reason
    out = capsys.readouterr()
    assert DUMMY_KEY not in out.out + out.err


def test_commercial_available_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    avail = OpenAiGrounding({"api_key_env": "OPENAI_API_KEY"}).is_available()
    assert avail.ok is True
    assert DUMMY_KEY not in avail.reason


def test_commercial_unavailable_when_api_key_env_missing(monkeypatch):
    # No api_key_env configured at all.
    avail = OpenAiGrounding({"model": "gpt-5"}).is_available()
    assert avail.ok is False
    assert "api_key_env" in avail.reason


def test_local_vllm_availability(monkeypatch):
    assert LocalVllmGrounding({"base_url": "http://x:8000/v1"}).is_available().ok is True
    bad = LocalVllmGrounding({"model": "m"}).is_available()
    assert bad.ok is False
    assert "base_url" in bad.reason


# --------------------------------------------------------------------------- error handling


@respx.mock
def test_http_error_propagates_for_chain(monkeypatch):
    # Non-2xx should raise (so run_chain logs + advances), not silently return None.
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    provider = OpenAiGrounding({"model": "gpt-5", "api_key_env": "OPENAI_API_KEY"})
    with pytest.raises(httpx.HTTPStatusError):
        provider.locate(_img(), "x")


@respx.mock
def test_unparseable_response_returns_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_body("I could not find that, sorry."))
    )
    provider = OpenAiGrounding({"model": "gpt-5", "api_key_env": "OPENAI_API_KEY"})
    assert provider.locate(_img(), "x") is None


# --------------------------------------------------------------------------- _common parse


def test_parse_plain_point():
    assert parse_grounding_json('{"point": [10, 20]}', _img()) == Point(x=10, y=20)


def test_parse_plain_box():
    assert parse_grounding_json('{"box": [1, 2, 3, 4]}', _img()) == DetBox(bounds=(1, 2, 3, 4))


def test_parse_fenced():
    assert parse_grounding_json(FENCED, _img()) == Point(x=120, y=240)


def test_parse_bare_fence_no_lang():
    text = '```\n{"point": [7, 8]}\n```'
    assert parse_grounding_json(text, _img()) == Point(x=7, y=8)


def test_parse_prose_then_json():
    text = 'The button is here {"point": [33, 44]} hope that helps'
    assert parse_grounding_json(text, _img()) == Point(x=33, y=44)


def test_parse_found_false_is_none():
    assert parse_grounding_json('{"found": false}', _img()) is None


def test_parse_malformed_is_none():
    assert parse_grounding_json("not json at all {{{", _img()) is None
    assert parse_grounding_json("", _img()) is None
    assert parse_grounding_json(None, _img()) is None
    assert parse_grounding_json('{"point": ["a", "b"]}', _img()) is None


def test_parse_normalized_point_scaled():
    # All values <= 1.0 -> treat as normalized fractions of W/H.
    result = parse_grounding_json('{"point": [0.5, 0.25]}', _img())
    assert result == Point(x=100, y=100)  # 0.5*200, 0.25*400


def test_parse_normalized_box_scaled():
    result = parse_grounding_json('{"box": [0.0, 0.0, 1.0, 0.5]}', _img())
    assert result == DetBox(bounds=(0, 0, 200, 200))


def test_parse_0_1000_space_setting():
    result = parse_grounding_json(
        '{"point": [500, 250]}', _img(), settings={"coordinate_space": "0-1000"}
    )
    assert result == Point(x=100, y=100)  # 500/1000*200, 250/1000*400


def test_parse_pixels_setting_forces_absolute():
    # Even though values are <= 1.0, an explicit pixel space keeps them absolute.
    result = parse_grounding_json(
        '{"point": [1, 1]}', _img(), settings={"coordinate_space": "pixels"}
    )
    assert result == Point(x=1, y=1)


def test_parse_clamps_to_bounds():
    result = parse_grounding_json('{"point": [9999, -50]}', _img())
    assert result == Point(x=IMG_W, y=0)


def test_parse_box_reorders_corners():
    result = parse_grounding_json('{"box": [60, 80, 10, 20]}', _img())
    assert result == DetBox(bounds=(10, 20, 60, 80))


def test_build_user_prompt_mentions_size_and_json():
    prompt = build_user_prompt(_img(), "the login button")
    assert "200x400" in prompt
    assert "the login button" in prompt
    assert '{"found":false}' in prompt
