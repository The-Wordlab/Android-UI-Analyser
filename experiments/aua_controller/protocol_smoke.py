"""Check native two-turn tool calling on an explicitly supplied inference endpoint.

Uses fictional probe tools only. No model is loaded, trained or downloaded here;
no device is accessed. Passing establishes protocol support, not AUA ability.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from experiments.aua_controller.hosted import (
    BACKENDS,
    CostGuard,
    HostedError,
    assistant_message,
    configure_payload,
    validate_endpoint,
    validate_request_config,
)


class ProtocolError(ValueError):
    """The endpoint did not return the required native tool-call structure."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_probe_value",
            "description": (
                "Read the protocol probe. Returns JSON text containing a string field named value."
            ),
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "enum": ["probe"]}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_probe_value",
            "description": "Record only the value field parsed from read_probe_value's JSON text.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    },
]


def _call(response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ProtocolError("expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") == "length":
        raise ProtocolError("missing choice or truncated completion")
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise ProtocolError("unsupported or missing completion finish reason")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ProtocolError("missing assistant message")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ProtocolError("expected one native tool call; content-text parsing is not used")
    call = calls[0]
    if (
        not isinstance(call, dict)
        or call.get("type") != "function"
        or not isinstance(call.get("id"), str)
        or not call["id"].strip()
    ):
        raise ProtocolError("invalid native function call or missing call id")
    function = call.get("function")
    if not isinstance(function, dict):
        raise ProtocolError("missing native function")
    try:
        arguments = json.loads(function["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("function arguments must be a serialized JSON object") from exc
    if not isinstance(arguments, dict):
        raise ProtocolError("function arguments must be an object")
    return message, {"id": call["id"], "name": function.get("name"), "arguments": arguments}


def run_probe(
    send: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    chat_template_kwargs: dict[str, Any] | None = None,
    backend: str = "openai-compatible",
    request_config: dict[str, Any] | None = None,
    cost_limit_usd: float = 0.10,
) -> dict[str, Any]:
    """Keep model-specific reasoning fields across a genuine tool-result turn."""
    if backend not in BACKENDS:
        raise ProtocolError("unknown model backend")
    hosted = backend == "openrouter"
    settings = validate_request_config(request_config or {}) if hosted else {}
    cost_guard = CostGuard(cost_limit_usd) if hosted else None
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are performing a fictional tool-protocol check. Use the supplied tools.",
        },
        {
            "role": "user",
            "content": (
                "Read the probe with key 'probe'. Its tool result contains JSON text. Parse that "
                "JSON and pass only its string field named value to record_probe_value, not the "
                "surrounding JSON text or a response wrapper. Call one tool at a time."
            ),
        },
    ]
    # Generated outside the model and revealed only after its first tool call.
    value = secrets.token_hex(12)
    expected = [
        ("read_probe_value", {"key": "probe"}),
        ("record_probe_value", {"value": value}),
    ]
    turns: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "format": "aua-controller-protocol-smoke-v2",
        "model_requested": model,
        "checkpoint_verified": False,
        "fictional": True,
        "device_accessed": False,
        "aua_task_success": None,
        "max_tokens_per_turn": max_tokens,
        "temperature": settings.get("temperature") if hosted else temperature,
        "chat_template_kwargs": {} if hosted else chat_template_kwargs or {},
        "backend": backend, "request_config": settings,
        "passed": False,
        "turns": turns,
    }
    for name, arguments in expected:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if hosted:
            payload = configure_payload(payload, settings)
        started = time.perf_counter()
        turn: dict[str, Any] = {"expected_tool": name}
        if hosted:
            turn["request"] = copy.deepcopy(payload)
        turns.append(turn)
        try:
            if cost_guard is not None:
                cost_guard.before_request()
            response = send(payload)
            if hosted:
                turn["response"] = copy.deepcopy(response)
            turn["request_ms"] = (time.perf_counter() - started) * 1000
            turn["model_returned"] = response.get("model")
            turn["usage"] = response.get("usage")
            if cost_guard is not None:
                cost_guard.consume(response)
            message, call = _call(response)
            turn["call"] = call
            if call["name"] != name or call["arguments"] != arguments:
                raise ProtocolError(
                    "tool name or arguments differ from the required probe operation"
                )
        except (ProtocolError, httpx.HTTPError, ValueError) as exc:
            turn.setdefault("request_ms", (time.perf_counter() - started) * 1000)
            # HTTP errors can contain endpoint credentials or server-echoed input.
            report["error"] = str(exc) if isinstance(exc, (ProtocolError, HostedError)) else type(exc).__name__
            if cost_guard is not None:
                report["cost_accounting"] = cost_guard.report()
            return report
        # Preserve thinking during tool turns. Drop response-only metadata such as
        # refusal/annotations that some chat-completions servers do not accept.
        messages.append(assistant_message(message))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps({"value": value}),
            }
        )
    report["passed"] = True
    if cost_guard is not None:
        report["cost_accounting"] = cost_guard.report()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Explicit endpoint API root, ending /v1")
    parser.add_argument("--model", required=True, help="Exact name served by that endpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key-env", help="Defaults to OPEN_ROUTER_API_KEY for OpenRouter")
    parser.add_argument("--backend", choices=BACKENDS, default="openai-compatible")
    parser.add_argument("--request-config", default="{}", help="Hosted reasoning/provider/plugins JSON")
    parser.add_argument("--cost-limit-usd", type=float, default=0.10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--chat-template-kwargs", default="{}", help="Model-specific JSON object")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0 or args.temperature < 0:
        parser.error("token/timeout limits must be positive and temperature nonnegative")
    try:
        kwargs = json.loads(args.chat_template_kwargs)
    except json.JSONDecodeError:
        parser.error("chat-template-kwargs must be JSON")
    if not isinstance(kwargs, dict):
        parser.error("chat-template-kwargs must be an object")
    api_key_env = args.api_key_env or ("OPEN_ROUTER_API_KEY" if args.backend == "openrouter" else "AUA_BENCHMARK_API_KEY")
    try:
        settings = json.loads(args.request_config)
        if args.backend == "openrouter":
            validate_endpoint(args.base_url, os.environ.get(api_key_env))
            settings = validate_request_config(settings)
            CostGuard(args.cost_limit_usd)
    except (HostedError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    headers = {}
    if key := os.environ.get(api_key_env):
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(timeout=args.timeout, headers=headers, follow_redirects=False) as client:

        def send(payload: dict[str, Any]) -> dict[str, Any]:
            response = client.post(args.base_url.rstrip("/") + "/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ProtocolError("endpoint returned a non-object JSON response")
            return data

        report = run_probe(
            send,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            chat_template_kwargs=kwargs, backend=args.backend, request_config=settings,
            cost_limit_usd=args.cost_limit_usd,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
