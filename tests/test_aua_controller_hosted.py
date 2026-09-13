from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.hosted import (
    CostGuard,
    HostedError,
    assistant_message,
    configure_payload,
    validate_endpoint,
    validate_request_config,
)


def settings():
    return {"provider": {"only": ["fictional/bf16"], "order": ["fictional/bf16"],
                         "allow_fallbacks": False, "require_parameters": True,
                         "max_price": {"prompt": 0.3, "completion": 1.2}},
            "reasoning": {"enabled": True, "exclude": False}}


@pytest.mark.parametrize("url", [
    "http://openrouter.ai/api/v1", "https://evil.invalid/api/v1",
    "https://openrouter.ai.evil.invalid/api/v1", "https://user@openrouter.ai/api/v1",
    "https://openrouter.ai/api/v1?key=fictional", "https://openrouter.ai/api/v1#fragment",
    "https://openrouter.ai/api/v1/chat/completions", "https://openrouter.ai:444/api/v1",
])
def test_hosted_endpoint_rejects_noncanonical_origins(url):
    with pytest.raises(HostedError):
        validate_endpoint(url, "fictional-test-key")


@pytest.mark.parametrize("key", [None, "", "   "])
def test_hosted_requires_nonempty_key(key):
    with pytest.raises(HostedError, match="missing or empty"):
        validate_endpoint("https://openrouter.ai/api/v1", key)


def test_request_adapter_does_not_override_benchmark_and_omits_unrequested_temperature():
    payload = {"model": "fictional/model", "messages": [], "tools": [],
               "temperature": 0, "parallel_tool_calls": False, "max_tokens": 4096,
               "chat_template_kwargs": {"enable_thinking": True}}
    original = copy.deepcopy(payload)
    out = configure_payload(payload, settings())
    assert payload == original
    assert set(out) == {"model", "messages", "tools", "max_tokens", "provider", "reasoning", "plugins"}
    assert out["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert configure_payload(payload, {**settings(), "temperature": 0.5})["temperature"] == 0.5
    for field in ("tools", "messages", "model", "max_tokens", "stream", "chat_template_kwargs"):
        with pytest.raises(HostedError):
            configure_payload(payload, {**settings(), field: []})


@pytest.mark.parametrize("change", [
    {"only": ["a", "b"]}, {"allow_fallbacks": True}, {"require_parameters": "yes"},
    {"order": ["other"]}, {"max_price": {}}, {"max_price": {"prompt": -1, "completion": 1}},
])
def test_routing_must_be_pinned_and_price_capped(change):
    config = settings()
    config["provider"].update(change)
    with pytest.raises(HostedError):
        validate_request_config(config)


@pytest.mark.parametrize("require_parameters", [False, None])
def test_require_parameters_is_optional(require_parameters):
    config = settings()
    if require_parameters is None:
        del config["provider"]["require_parameters"]
    else:
        config["provider"]["require_parameters"] = require_parameters
    assert validate_request_config(config)["provider"]["only"] == ["fictional/bf16"]


def test_reasoning_signatures_and_native_tool_fields_are_copied_without_modification():
    message = {"role": "assistant", "content": None,
               "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque==", "index": 0}],
               "tool_calls": [{"id": "native", "extra_content": {"google": {"thought_signature": "sig=="}}}],
               "refusal": None, "annotations": []}
    copied = assistant_message(message)
    assert copied == {key: value for key, value in message.items() if key not in {"refusal", "annotations"}}
    assert copied["reasoning_details"] is not message["reasoning_details"]


@pytest.mark.parametrize("cost", [None, -1, True, "NaN", float("inf"), {}, "not-a-number"])
def test_missing_or_invalid_cost_blocks_future_requests(cost):
    guard = CostGuard(0.1)
    with pytest.raises(HostedError, match="usage.cost"):
        guard.consume({"usage": {"cost": cost}})
    with pytest.raises(HostedError):
        guard.before_request()
    assert guard.report()["missing_or_invalid_cost"] is True


def test_reported_spend_boundary_tracks_zero_and_stops_after_exact_limit():
    guard = CostGuard(0.1)
    for cost in (0, 0.03, 0.07):
        guard.before_request()
        guard.consume({"usage": {"cost": cost}})
    assert guard.report()["reported_usd"] == 0.1
    assert guard.report()["responses_with_cost"] == 3
    with pytest.raises(HostedError, match="limit reached"):
        guard.before_request()


def test_candidate_manifest_configurations_are_supported():
    path = Path(__file__).resolve().parents[1] / "experiments/aua_controller/openrouter-comparison.json"
    if not path.exists():
        pytest.skip("separate candidate manifest not present")
    for model in json.loads(path.read_text())["models"]:
        validate_request_config(model["request_config"])
        CostGuard(model["cost_limit_usd"])
