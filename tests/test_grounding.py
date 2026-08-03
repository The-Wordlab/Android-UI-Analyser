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

from android_ui_analyser.providers.base import DetBox, Point, ScreenAnalysisResult
from android_ui_analyser.providers.grounding._common import (
    build_user_prompt,
    image_b64,
    parse_grounding_json,
)
from android_ui_analyser.providers.grounding._screen_analysis import prepare_screen_preview
from android_ui_analyser.providers.grounding.anthropic import AnthropicGrounding
from android_ui_analyser.providers.grounding.gemini import GeminiGrounding
from android_ui_analyser.providers.grounding.local_vllm import LocalVllmGrounding
from android_ui_analyser.providers.grounding.openai import OpenAiGrounding
from android_ui_analyser.providers.registry import ProviderFactory, get_provider_class, run_chain
from conftest import make_config, make_screen_image

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
    provider = OpenAiGrounding(
        {
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "none",
        }
    )
    provider.locate(_img(), "anything")

    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {DUMMY_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "gpt-5.6-luna"
    assert body["reasoning_effort"] == "none"
    # image present as a data URL inside the user content parts
    user = body["messages"][-1]
    image_parts = [p for p in user["content"] if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_b64(_img()) in image_parts[0]["image_url"]["url"]


def test_openai_omits_reasoning_effort_when_unconfigured():
    provider = OpenAiGrounding({"model": "gpt-5", "api_key_env": "OPENAI_API_KEY"})
    assert "reasoning_effort" not in provider._payload(_img(), "anything")


@respx.mock
def test_openai_ask_fuses_image_and_element_graph(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    analysis = {
        "answer": "A header with a back button and centered title.",
        "screen_summary": "App detail screen",
        "regions": [{"name": "header", "bounds": [0, 0, 200, 80]}],
        "elements": [
            {
                "graph_id": 4,
                "role": "back button",
                "text": None,
                "bounds": [0, 0, 40, 40],
                "position": "top-left",
                "evidence": "both",
            }
        ],
        "uncertainties": [],
    }
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                **_openai_body(json.dumps(analysis)),
                "model": "gpt-5.6-luna",
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            },
        )
    )
    provider = OpenAiGrounding(
        {
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "none",
            "screen_image_detail": "low",
            "screen_preview_max_width": 100,
            "screen_preview_jpeg_quality": 30,
        }
    )
    result = provider.ask(
        _img(),
        "Describe the header",
        [{"id": 4, "type": "ImageButton", "bounds": [0, 0, 40, 40], "clickable": True}],
    )

    assert result is not None
    assert isinstance(result, ScreenAnalysisResult)
    assert result.model == "gpt-5.6-luna"
    assert result.analysis == analysis
    assert result.usage["total_tokens"] == 140
    body = json.loads(route.calls.last.request.content)
    assert body["reasoning_effort"] == "none"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["max_completion_tokens"] == 1200
    user_parts = body["messages"][-1]["content"]
    assert "Describe the header" in user_parts[0]["text"]
    assert '"id":4' in user_parts[0]["text"]
    assert user_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user_parts[1]["image_url"]["detail"] == "low"
    assert result.input_image == {
        "original_width": 200,
        "original_height": 400,
        "width": 100,
        "height": 200,
        "bytes": result.input_image["bytes"],
        "format": "jpeg",
        "quality": 30,
        "detail": "low",
    }
    assert result.input_image["bytes"] < len(_img().png_bytes)


def test_screen_preview_defaults_relax_quality_without_using_original_size():
    image = make_screen_image(1080, 2400)
    _data, metadata = prepare_screen_preview(image, {})
    assert metadata == {
        "original_width": 1080,
        "original_height": 2400,
        "width": 720,
        "height": 1600,
        "bytes": metadata["bytes"],
        "format": "jpeg",
        "quality": 55,
    }


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


@respx.mock
def test_gemini_ask_uses_shared_screen_analysis_contract(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    base = "https://generativelanguage.googleapis.com/v1beta"
    analysis = {
        "answer": "The settings control is in the top-right.",
        "screen_summary": "Chats screen",
        "regions": [{"name": "header", "bounds": [0, 0, 200, 80], "description": "Top bar"}],
        "elements": [],
        "uncertainties": [],
    }
    route = respx.post(f"{base}/models/gemini-2.5-flash:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                **_gemini_body(json.dumps(analysis)),
                "modelVersion": "gemini-2.5-flash-001",
                "usageMetadata": {
                    "promptTokenCount": 90,
                    "candidatesTokenCount": 30,
                    "totalTokenCount": 120,
                },
            },
        )
    )
    provider = GeminiGrounding(
        {
            "model": "gemini-2.5-flash",
            "api_key_env": "GEMINI_API_KEY",
            "base_url": base,
            "screen_preview_max_width": 100,
            "screen_preview_jpeg_quality": 50,
        }
    )
    result = provider.ask(
        _img(),
        "Where is Settings?",
        [{"id": 25, "type": "ImageButton", "bounds": [170, 0, 200, 40]}],
    )

    assert result is not None
    assert result.model == "gemini-2.5-flash-001"
    assert result.analysis == analysis
    assert result.usage == {
        "prompt_tokens": 90,
        "completion_tokens": 30,
        "total_tokens": 120,
    }
    body = json.loads(route.calls.last.request.content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseJsonSchema"]["required"] == [
        "answer",
        "screen_summary",
        "regions",
        "elements",
        "uncertainties",
    ]
    parts = body["contents"][0]["parts"]
    assert "Where is Settings?" in parts[0]["text"]
    assert '"id":25' in parts[0]["text"]
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert result.input_image["quality"] == 50


@respx.mock
def test_grounding_factory_falls_through_to_openai_when_only_openai_key_exists(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_openai_body(
                json.dumps(
                    {
                        "answer": "OpenAI answered",
                        "screen_summary": "screen",
                        "regions": [],
                        "elements": [],
                        "uncertainties": [],
                    }
                )
            ),
        )
    )
    cfg = make_config(
        grounding={"enabled": True, "chain": ["gemini", "openai"]},
    )
    result, provider = run_chain(
        ProviderFactory(cfg).build_chain("grounding"),
        lambda item: item.ask(_img(), "Describe it", []),  # type: ignore[attr-defined]
    )

    assert provider == "openai"
    assert isinstance(result, ScreenAnalysisResult)
    assert result.analysis["answer"] == "OpenAI answered"
    assert route.call_count == 1


@respx.mock
def test_grounding_factory_uses_configured_order_when_both_keys_exist(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    gemini_route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json=_gemini_body(
                json.dumps(
                    {
                        "answer": "Gemini answered",
                        "screen_summary": "screen",
                        "regions": [],
                        "elements": [],
                        "uncertainties": [],
                    }
                )
            ),
        )
    )
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "must not be called"})
    )
    cfg = make_config(grounding={"enabled": True, "chain": ["gemini", "openai"]})
    result, provider = run_chain(
        ProviderFactory(cfg).build_chain("grounding"),
        lambda item: item.ask(_img(), "Describe it", []),  # type: ignore[attr-defined]
    )

    assert provider == "gemini"
    assert isinstance(result, ScreenAnalysisResult)
    assert result.analysis["answer"] == "Gemini answered"
    assert gemini_route.call_count == 1
    assert openai_route.call_count == 0


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
