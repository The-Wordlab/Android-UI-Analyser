from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.protocol_smoke import run_probe


def _response(
    name: str = "read_probe_value",
    arguments: Any = '{"key":"probe"}',
    *,
    finish_reason: str = "tool_calls",
) -> dict[str, Any]:
    return {
        "model": "fictional-served-checkpoint",
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ],
    }


def test_native_tool_result_is_used_on_second_turn_and_reasoning_is_preserved() -> None:
    requests: list[dict[str, Any]] = []
    first = _response()
    first_message = first["choices"][0]["message"]
    first_message.update(
        {
            "reasoning_content": "I must read the value before recording it.",
            "reasoning": "The temporary value is not known yet.",
            "refusal": None,
            "annotations": [],
        }
    )

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(payload))
        if len(requests) == 1:
            assert [message["role"] for message in payload["messages"]] == ["system", "user"]
            return first
        assert len(requests) == 2
        history = payload["messages"]
        assert [message["role"] for message in history] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
        assistant, result = history[-2:]
        assert assistant["tool_calls"] == first_message["tool_calls"]
        assert assistant["reasoning_content"] == first_message["reasoning_content"]
        assert assistant["reasoning"] == first_message["reasoning"]
        assert "refusal" not in assistant and "annotations" not in assistant
        assert result["name"] == "read_probe_value"
        assert result["tool_call_id"] == first_message["tool_calls"][0]["id"]
        value = json.loads(result["content"])["value"]
        assert isinstance(value, str) and len(value) == 24
        assert value not in json.dumps(requests[0])
        return _response("record_probe_value", json.dumps({"value": value}))

    report = run_probe(
        send,
        model="fictional-requested-model",
        max_tokens=777,
        temperature=0.2,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert report["passed"] is True
    assert report["aua_task_success"] is None
    assert report["checkpoint_verified"] is False
    assert report["device_accessed"] is False
    assert len(report["turns"]) == len(requests) == 2
    for request, turn in zip(requests, report["turns"], strict=True):
        assert request["model"] == "fictional-requested-model"
        assert request["max_tokens"] == 777
        assert request["temperature"] == 0.2
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert request["parallel_tool_calls"] is False
        assert turn["model_returned"] == "fictional-served-checkpoint"
        assert turn["usage"] == {"prompt_tokens": 40, "completion_tokens": 12}
        assert turn["request_ms"] >= 0


@pytest.mark.parametrize(
    ("response", "error"),
    [
        pytest.param(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": 'read_probe_value({"key":"probe"})',
                        },
                    }
                ]
            },
            "native tool call",
            id="plain-text-function-is-not-native",
        ),
        pytest.param(_response(finish_reason="length"), "truncated", id="truncated-call"),
        pytest.param(_response(arguments="{broken"), "serialized JSON", id="malformed-json"),
        pytest.param(
            _response(arguments={"key": "probe"}), "serialized JSON", id="unserialized-args"
        ),
        pytest.param(_response(arguments='["probe"]'), "must be an object", id="non-object-args"),
        pytest.param(_response(name="invented_tool"), "differ", id="wrong-tool"),
        pytest.param(
            _response(arguments='{"key":"probe","extra":true}'), "differ", id="extra-args"
        ),
    ],
)
def test_invalid_first_turn_fails_without_a_followup(
    response: dict[str, Any],
    error: str,
) -> None:
    requests = []

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(payload)
        return response

    report = run_probe(send, model="fictional-model")

    assert report["passed"] is False
    assert error in report["error"]
    assert len(requests) == len(report["turns"]) == 1


def test_parallel_calls_are_rejected_even_when_each_is_well_formed() -> None:
    response = _response()
    calls = response["choices"][0]["message"]["tool_calls"]
    second = copy.deepcopy(calls[0])
    second["id"] = "another_call"
    calls.append(second)

    report = run_probe(lambda payload: response, model="fictional-model")

    assert report["passed"] is False
    assert "one native tool call" in report["error"]
    assert len(report["turns"]) == 1


def test_second_turn_must_copy_the_actual_tool_result() -> None:
    requests = []

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(payload))
        if len(requests) == 1:
            return _response()
        actual = json.loads(payload["messages"][-1]["content"])["value"]
        return _response("record_probe_value", json.dumps({"value": actual + "wrong"}))

    report = run_probe(send, model="fictional-model")

    assert report["passed"] is False
    assert "differ" in report["error"]
    assert len(requests) == len(report["turns"]) == 2
    assert report["turns"][1]["expected_tool"] == "record_probe_value"


def test_copying_json_wrapper_instead_of_its_value_still_fails() -> None:
    def send(payload: dict[str, Any]) -> dict[str, Any]:
        if len(payload["messages"]) == 2:
            return _response()
        return _response(
            "record_probe_value", json.dumps({"value": payload["messages"][-1]["content"]})
        )

    report = run_probe(send, model="fictional-model")
    assert report["format"] == "aua-controller-protocol-smoke-v2"
    assert report["passed"] is False
    assert "differ" in report["error"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finish_reason", None),
        ("finish_reason", "content_filter"),
        ("finish_reason", "unknown"),
        ("missing_finish_reason", None),
        ("id", None),
        ("id", 123),
        ("id", {"invalid": True}),
        ("id", "  "),
    ],
)
def test_malformed_final_response_cannot_pass(field: str, value: Any) -> None:
    def send(payload: dict[str, Any]) -> dict[str, Any]:
        if len(payload["messages"]) == 2:
            return _response()
        actual = json.loads(payload["messages"][-1]["content"])["value"]
        response = _response("record_probe_value", json.dumps({"value": actual}))
        choice = response["choices"][0]
        if field == "id":
            choice["message"]["tool_calls"][0]["id"] = value
        elif field == "missing_finish_reason":
            del choice["finish_reason"]
        else:
            choice[field] = value
        return response

    report = run_probe(send, model="fictional-model")
    assert report["passed"] is False
    assert len(report["turns"]) == 2
    assert "finish reason" in report["error"] or "call id" in report["error"]


@pytest.mark.parametrize("error_type", [httpx.HTTPStatusError, httpx.ConnectError])
def test_http_failure_report_does_not_expose_endpoint_credentials(error_type: type) -> None:
    secret = "fictional-test-api-secret"
    url = f"https://fixture-user:{secret}@inference.invalid/v1?key={secret}"
    request = httpx.Request("POST", url, headers={"Authorization": f"Bearer {secret}"})
    kwargs: dict[str, Any] = {"request": request}
    if error_type is httpx.HTTPStatusError:
        kwargs["response"] = httpx.Response(401, request=request, text=secret)
    error = error_type(f"Failure calling {url}: Authorization Bearer {secret}", **kwargs)

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        raise error

    report = run_probe(send, model="fictional-model")

    assert report["passed"] is False
    assert report["error"] == error_type.__name__
    serialized = json.dumps(report)
    assert secret not in serialized
    assert "fixture-user" not in serialized
    assert "inference.invalid" not in serialized


def _hosted_settings() -> dict[str, Any]:
    return {"provider": {"only": ["fictional"], "allow_fallbacks": False, "require_parameters": True,
                         "max_price": {"prompt": 0.3, "completion": 1.2}},
            "reasoning": {"effort": "low", "exclude": False}, "temperature": 0.5}


def test_hosted_probe_preserves_exact_native_blocks_records_actual_payload_and_cost() -> None:
    requests = []
    details = [{"type": "reasoning.encrypted", "data": "exact-opaque-native==", "index": 0}]

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(payload))
        assert "parallel_tool_calls" not in payload and "chat_template_kwargs" not in payload
        assert payload["temperature"] == 0.5
        if len(requests) == 1:
            result = _response()
        else:
            assert payload["messages"][-2]["reasoning_details"] == details
            value = json.loads(payload["messages"][-1]["content"])["value"]
            result = _response("record_probe_value", json.dumps({"value": value}))
        result["choices"][0]["message"]["reasoning_details"] = copy.deepcopy(details)
        result["usage"]["cost"] = 0.003
        return result

    report = run_probe(send, model="fictional", backend="openrouter", request_config=_hosted_settings())
    assert report["passed"] is True
    assert report["cost_accounting"]["reported_usd"] == 0.006
    assert [turn["request"] for turn in report["turns"]] == requests
    assert all(turn["response"]["usage"]["cost"] == 0.003 for turn in report["turns"])


@pytest.mark.parametrize("cost", [None, 0.1])
def test_hosted_probe_does_not_continue_without_cost_or_after_limit(cost: Any) -> None:
    sent = []

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        result = _response()
        if cost is not None:
            result["usage"]["cost"] = cost
        return result

    report = run_probe(send, model="fictional", backend="openrouter", request_config=_hosted_settings())
    assert report["passed"] is False and len(sent) == 1
    assert "cost" in report["error"]
