"""A model call goes straight to OpenAI when an OpenAI key is set, and through OpenRouter otherwise.

Every AUA model request is written once, in OpenRouter's shape, and names its model the way
OpenRouter does (`openai/gpt-6-luna`). `llm_route` decides where it goes: a company OpenAI key is
cheaper than OpenRouter's resale, so an OpenAI model is sent to api.openai.com whenever that key
exists. OpenAI does not report a price, so the route computes one — without it every spend stop
and QA total would read $0.
"""

from __future__ import annotations

import pytest

from android_ui_analyser import llm_route

LUNA = "openai/gpt-6-luna"
OPENROUTER_ONLY = {"OPEN_ROUTER_API_KEY": "sk-or-test"}
BOTH = {"OPEN_ROUTER_API_KEY": "sk-or-test", "OPENAI_API_KEY": "sk-openai-test"}


def _payload(**extra):
    return {
        "model": LUNA,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "ok", "reasoning": "r",
             "reasoning_details": [{"type": "reasoning.text"}]},
        ],
        "max_tokens": 900,
        "temperature": 0,
        "tools": [{"type": "function", "function": {"name": "tap", "parameters": {}}}],
        "tool_choice": "auto",
        "provider": {"allow_fallbacks": True, "max_price": {"prompt": 0.11}},
        "plugins": [{"id": "context-compression", "enabled": False}],
        "reasoning": {"effort": "low", "exclude": False, "max_tokens": 4096},
        **extra,
    }


def test_without_an_openai_key_the_request_goes_to_openrouter_unchanged() -> None:
    call = llm_route.prepare(_payload(), OPENROUTER_ONLY)
    assert call.url == "https://openrouter.ai/api/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer sk-or-test"
    assert call.body == _payload()
    assert call.route == "openrouter"


def test_with_an_openai_key_an_openai_model_goes_direct_in_openais_shape() -> None:
    call = llm_route.prepare(_payload(reasoning={"enabled": False}), BOTH)
    assert call.url == "https://api.openai.com/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer sk-openai-test"
    assert call.route == "openai"
    body = call.body
    assert body["model"] == "gpt-6-luna"
    for field in ("provider", "plugins", "reasoning", "max_tokens", "temperature"):
        assert field not in body, field
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 900
    assert body["tools"] and body["tool_choice"] == "auto"
    assert body["messages"][1] == {"role": "assistant", "content": "ok"}


def test_reasoning_switched_off_becomes_the_lowest_openai_effort() -> None:
    body = llm_route.prepare(_payload(reasoning={"enabled": False}), BOTH).body
    assert body["reasoning_effort"] == "none"


def test_a_non_openai_model_stays_on_openrouter_even_with_an_openai_key() -> None:
    call = llm_route.prepare(_payload(model="google/gemma-4-26b"), BOTH)
    assert call.route == "openrouter"


def test_an_openai_model_without_a_known_price_stays_on_openrouter() -> None:
    """Going direct without a price would make every budget read $0."""
    call = llm_route.prepare(_payload(model="openai/gpt-unpriced"), BOTH)
    assert call.route == "openrouter"


def test_a_direct_answer_carries_the_price_openrouter_would_have_charged() -> None:
    """Measured 2026-09-23: this formula matched OpenRouter's billed cost on 13 of 13 requests."""
    call = llm_route.prepare(_payload(reasoning={"enabled": False}), BOTH)
    answer = call.finish({
        "choices": [{"message": {"role": "assistant", "content": "x"}}],
        "usage": {"prompt_tokens": 4174, "completion_tokens": 40,
                  "prompt_tokens_details": {"cached_tokens": 1876, "cache_write_tokens": 2295}},
    })
    assert answer["usage"]["cost"] == pytest.approx(0.00032594, rel=1e-3)
    assert answer["provider"] == "OpenAI"


def test_an_openrouter_answer_is_passed_through() -> None:
    call = llm_route.prepare(_payload(), OPENROUTER_ONLY)
    answer = {"usage": {"cost": 0.5}, "provider": "Azure"}
    assert call.finish(answer) == answer


def test_no_key_at_all_is_a_clear_error() -> None:
    with pytest.raises(llm_route.RouteError, match="OPEN_ROUTER_API_KEY"):
        llm_route.prepare(_payload(), {})


def test_an_openai_key_alone_still_reaches_an_openai_model() -> None:
    call = llm_route.prepare(_payload(reasoning={"enabled": False}), {"OPENAI_API_KEY": "sk"})
    assert call.route == "openai"


def test_tools_with_reasoning_go_direct_through_the_responses_api() -> None:
    """Live 2026-09-23: /v1/chat/completions answers 400 "Function tools with reasoning_effort are
    not supported for gpt-6-luna ... use /v1/responses". Kept on OpenRouter, that made every QA
    judge depend on OpenRouter credit even with a company key, and an exhausted balance stopped
    every judgement from 2026-10-05. The Responses API serves the same forced tool call."""
    judge = _payload(
        messages=[
            {"role": "system", "content": "Judge the run."},
            {"role": "user", "content": [
                {"type": "text", "text": "Frame E0001"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}}]},
            {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "verdict", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "missing field: criteria"},
        ],
        tools=[{"type": "function", "function": {
            "name": "verdict", "description": "Submit it.",
            "parameters": {"type": "object", "properties": {"pass": {"type": "boolean"}}}}}],
        tool_choice={"type": "function", "function": {"name": "verdict"}},
        parallel_tool_calls=False,
    )
    call = llm_route.prepare(judge, BOTH)
    assert call.route == "openai"
    assert call.url == "https://api.openai.com/v1/responses"
    assert call.headers["Authorization"] == "Bearer sk-openai-test"
    body = call.body
    for field in ("messages", "provider", "plugins", "max_tokens", "temperature", "reasoning_effort"):
        assert field not in body, field
    assert body["model"] == "gpt-6-luna"
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 900
    assert body["tools"] == [{"type": "function", "name": "verdict", "description": "Submit it.",
                              "parameters": {"type": "object", "properties": {"pass": {"type": "boolean"}}},
                              "strict": False}]
    assert body["tool_choice"] == {"type": "function", "name": "verdict"}
    assert body["parallel_tool_calls"] is False
    assert body["input"] == [
        {"role": "system", "content": "Judge the run."},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Frame E0001"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"}]},
        {"type": "function_call", "call_id": "call_1", "name": "verdict", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "missing field: criteria"},
    ]
    # Reasoning switched off still takes chat completions, and OpenRouter-only keys never change.
    assert llm_route.prepare(_payload(reasoning={"enabled": False}), BOTH).url.endswith("/chat/completions")
    assert llm_route.prepare(judge, OPENROUTER_ONLY).body == judge


def test_a_reasoning_budget_without_an_effort_reasons_at_low_effort() -> None:
    """The judge swaps its effort for a token budget to keep an answer reserve; OpenAI has none."""
    body = llm_route.prepare(_payload(reasoning={"max_tokens": 2048, "exclude": False}), BOTH).body
    assert body["reasoning"] == {"effort": "low"}


def test_a_responses_answer_comes_back_in_chat_shape_with_its_price() -> None:
    call = llm_route.prepare(_payload(), BOTH)
    answer = call.finish({
        "id": "resp_1", "model": "gpt-6-luna-2026", "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "call_9", "name": "tap", "arguments": '{"id": "e1"}'},
        ],
        "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200},
                  "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 60}, "total_tokens": 1100},
    })
    message = answer["choices"][0]["message"]
    assert message["tool_calls"] == [{"id": "call_9", "type": "function",
                                      "function": {"name": "tap", "arguments": '{"id": "e1"}'}}]
    assert message["content"] is None
    assert answer["choices"][0]["finish_reason"] == "tool_calls"
    assert answer["provider"] == "OpenAI"
    usage = answer["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (1000, 100)
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 60
    assert usage["cost"] == pytest.approx((800 * 0.10 + 200 * 0.01 + 100 * 0.50) / 1e6)
    text = call.finish({"status": "completed", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": '{"pass": true}'}]}]})
    assert text["choices"][0]["message"]["content"] == '{"pass": true}'
    assert text["choices"][0]["finish_reason"] == "stop"
    cut = call.finish({"status": "incomplete", "output": []})
    assert cut["choices"][0]["finish_reason"] == "length"


def test_reasoning_without_tools_goes_direct() -> None:
    call = llm_route.prepare(_payload(tools=None, tool_choice=None), BOTH)
    assert call.route == "openai" and call.body["reasoning_effort"] == "low"


def test_a_model_is_reachable_with_either_key_it_can_use() -> None:
    assert llm_route.reachable(LUNA, {"OPENAI_API_KEY": "sk"})
    assert llm_route.reachable(LUNA, OPENROUTER_ONLY)
    assert not llm_route.reachable("google/gemma-4-26b", {"OPENAI_API_KEY": "sk"})
    assert not llm_route.reachable(LUNA, {})
    assert llm_route.reachable("google/gemma-4-26b", {}, openrouter_key="sk-or")


def test_openai_shaped_fields_survive_both_routes() -> None:
    """Grounding writes reasoning_effort and max_completion_tokens itself; both endpoints take them."""
    payload = {"model": LUNA, "messages": [], "reasoning_effort": "none", "max_completion_tokens": 50}
    assert llm_route.prepare(payload, BOTH).body["reasoning_effort"] == "none"
    assert llm_route.prepare(payload, OPENROUTER_ONLY).body == payload


def test_a_bare_model_id_means_openai_itself() -> None:
    """`gpt-5` names no OpenRouter vendor: it is an OpenAI-only configuration, as grounding has."""
    call = llm_route.prepare({"model": "gpt-5", "messages": []}, {"OPENAI_API_KEY": "sk"})
    assert call.route == "openai" and call.body["model"] == "gpt-5"
    with pytest.raises(llm_route.RouteError, match="OPENAI_API_KEY"):
        llm_route.prepare({"model": "gpt-5", "messages": []}, OPENROUTER_ONLY)
    assert not llm_route.reachable("gpt-5", OPENROUTER_ONLY)


def test_a_tool_schema_with_top_level_alternatives_is_flattened_for_openai() -> None:
    """Live 2026-09-23: api.openai.com answered 400 "Invalid schema for function 'tap_and_analyze':
    schema must have type 'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the
    top level", and the run silently fell back to another model. AUA validates the arguments of
    every call it executes, so the direct request only loses the alternative-requirement hint."""
    tap = {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
           "oneOf": [{"required": ["id"]}, {"required": ["text"]}]}
    union = {"anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                       {"type": "object", "properties": {"b": {"type": "integer"}}}]}
    payload = _payload(reasoning={"enabled": False}, tools=[
        {"type": "function", "function": {"name": "tap_and_analyze", "parameters": tap}},
        {"type": "function", "function": {"name": "either", "parameters": union}},
    ])
    tools = llm_route.prepare(payload, BOTH).body["tools"]
    assert tools[0]["function"]["parameters"] == {
        "type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}}}
    assert tools[1]["function"]["parameters"] == {
        "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    assert payload["tools"][0]["function"]["parameters"]["oneOf"], "the caller's payload is untouched"
    assert llm_route.prepare(payload, OPENROUTER_ONLY).body["tools"][0]["function"]["parameters"] == tap
