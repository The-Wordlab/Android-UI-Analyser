"""Where one model request goes: straight to OpenAI when there is a key for it, else OpenRouter.

Every AUA model request is written once, in OpenRouter's chat-completions shape, and names its
model the way OpenRouter does (``openai/gpt-6-luna``). :func:`prepare` picks the endpoint from the
keys that exist and returns the exact URL, headers and body for it:

* an ``openai/*`` model whose price is known goes to api.openai.com when ``OPENAI_API_KEY`` is
  set -- a company key is cheaper than OpenRouter's resale of the same model. A request that
  combines function tools with reasoning, which OpenAI's chat completions refuse, goes to its
  Responses API instead, and the answer comes back in chat-completions shape;
* everything else, and every OpenAI model without that key, goes through OpenRouter.

OpenAI does not report what a request cost, so :meth:`Call.finish` writes the price into
``usage.cost`` the way OpenRouter does. Spend stops, run totals and QA reports then read one field
whichever endpoint answered. An OpenAI model missing from :data:`OPENAI_PRICES` never goes direct:
a request with no price would make every budget read $0.

The caller keeps its own HTTP client, retries and budget; this module does no I/O.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_URL = "https://api.openai.com/v1"
OPENROUTER_KEYS = ("OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY")
OPENAI_KEY = "OPENAI_API_KEY"

# USD per million tokens: (input, cached input, cache write, output). OpenAI's list prices, which
# OpenRouter also charges for these models; on 2026-09-23 this reproduced OpenRouter's billed cost
# on 13 of 13 GPT-6 Luna requests.
OPENAI_PRICES: dict[str, tuple[float, float, float, float]] = {
    "gpt-6-luna": (0.10, 0.01, 0.125, 0.50),
}

# Request fields only OpenRouter understands; OpenAI answers 400 to an unknown field.
_OPENROUTER_ONLY = ("provider", "plugins", "transforms", "models", "route", "usage", "reasoning")
# Assistant-message fields OpenRouter returns for reasoning continuity; OpenAI rejects them.
_OPENROUTER_MESSAGE_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")


class RouteError(ValueError):
    """No key can reach the requested model."""


@dataclass(frozen=True)
class Call:
    route: str  # "openai" or "openrouter"
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    finish: Callable[[dict[str, Any]], dict[str, Any]]


def openai_model(model: str) -> str | None:
    """The OpenAI-direct id of *model*, or None when it cannot go direct.

    A bare id (``gpt-5``) names OpenAI itself. A vendor id goes direct only for ``openai/*`` with a
    known price; any other stays on OpenRouter, which reports what it charged.
    """
    if "/" not in model:
        return model or None
    vendor, _, name = model.partition("/")
    return name if vendor == "openai" and name in OPENAI_PRICES else None


def openai_cost(model: str, usage: Mapping[str, Any]) -> float:
    """What OpenAI charges for one answer's usage, in USD."""
    if model not in OPENAI_PRICES:
        raise RouteError(f"no price is known for {model}; add it to llm_route.OPENAI_PRICES")
    fresh, cached_rate, write_rate, out = (rate / 1e6 for rate in OPENAI_PRICES[model])
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return ((prompt - cached - written) * fresh + cached * cached_rate + written * write_rate
            + int(usage.get("completion_tokens") or 0) * out)


_TOP_LEVEL_REFUSED = ("oneOf", "anyOf", "allOf", "enum", "const", "not")


def _openai_parameters(schema: Any) -> Any:
    """A function schema OpenAI accepts: an object, with no alternatives at the top level.

    Live 2026-09-23: api.openai.com refuses `oneOf`/`anyOf`/`allOf`/`enum`/`const`/`not` at a
    tool schema's top level, which OpenRouter accepts. The branches' properties are folded into
    the object; AUA validates the arguments of every call it executes, so only the "one of these
    is required" hint is lost.
    """
    if not isinstance(schema, dict) or not any(key in schema for key in _TOP_LEVEL_REFUSED):
        return schema
    flat = {key: value for key, value in schema.items() if key not in _TOP_LEVEL_REFUSED}
    flat["type"] = "object"
    properties = dict(flat.get("properties") or {})
    for key in ("oneOf", "anyOf", "allOf"):
        for branch in schema.get(key) or []:
            if isinstance(branch, dict):
                for name, spec in (branch.get("properties") or {}).items():
                    properties.setdefault(name, spec)
    flat["properties"] = properties
    return flat


def _openai_body(payload: dict[str, Any], model: str) -> dict[str, Any]:
    body = {key: copy.deepcopy(value) for key, value in payload.items()
            if key not in _OPENROUTER_ONLY and value is not None}
    body["model"] = model
    reasoning = payload.get("reasoning") or {}
    if reasoning.get("enabled") is False:
        body["reasoning_effort"] = "none"
    elif reasoning.get("effort"):
        body["reasoning_effort"] = reasoning["effort"]
    # OpenAI's reasoning models take max_completion_tokens and only their default temperature.
    if "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
    body.pop("temperature", None)
    for tool in body.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and "parameters" in function:
            function["parameters"] = _openai_parameters(function["parameters"])
    body["messages"] = [
        {key: value for key, value in message.items() if key not in _OPENROUTER_MESSAGE_FIELDS}
        if message.get("role") == "assistant" else message
        for message in body.get("messages", [])
    ]
    return body


def _reasons_with_tools(payload: Mapping[str, Any]) -> bool:
    """OpenAI's chat completions refuse function tools with reasoning on for GPT-6 models.

    Live 2026-09-23: "Function tools with reasoning_effort are not supported for gpt-6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses". Such a request used to stay
    on OpenRouter, which made every judge (a forced tool call with reasoning on) depend on
    OpenRouter credit even with a company key: from 2026-10-05 an exhausted OpenRouter balance
    stopped every QA judgement for a week. It now goes to the Responses API.
    """
    reasoning = payload.get("reasoning") or {}
    return bool(payload.get("tools")) and reasoning.get("enabled") is not False and bool(reasoning)


def _responses_part(part: Any, role: str) -> Any:
    """One chat content part in the Responses API's shape."""
    if not isinstance(part, dict):
        return part
    if part.get("type") == "text":
        kind = "output_text" if role == "assistant" else "input_text"
        return {"type": kind, "text": part.get("text") or ""}
    if part.get("type") == "image_url":
        image = part.get("image_url")
        item = {"type": "input_image", "image_url": image.get("url") if isinstance(image, dict) else image}
        if isinstance(image, dict) and image.get("detail"):
            item["detail"] = image["detail"]
        return item
    return part


def _as_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat messages as Responses input items: tool calls and their results are items of their own."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message.get("tool_call_id"),
                          "output": _as_text(message.get("content"))})
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = [_responses_part(part, role) for part in content]
        if content not in (None, "", []):
            items.append({"role": role, "content": content})
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            items.append({"type": "function_call", "call_id": call.get("id"), "name": function.get("name"),
                          "arguments": function.get("arguments") or "{}"})
    return items


def _responses_body(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """*payload*, in OpenRouter's chat shape, as an OpenAI Responses API request."""
    chat = _openai_body(payload, model)
    body: dict[str, Any] = {"model": model, "input": _responses_input(chat.get("messages") or []),
                            # Nothing a QA judge sends needs to be kept on OpenAI's side.
                            "store": False}
    reasoning = payload.get("reasoning") or {}
    # A token budget (the judge's answer reserve) has no OpenAI equivalent; QA's judges run at
    # low effort, which is also the lowest effort that still reasons.
    body["reasoning"] = {"effort": chat.get("reasoning_effort") or reasoning.get("effort") or "low"}
    if "max_completion_tokens" in chat:
        body["max_output_tokens"] = chat["max_completion_tokens"]
    tools = []
    for tool in chat.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        converted = {"type": "function", "name": function.get("name"),
                     "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                     # Chat completions default to non-strict schemas; Responses default to strict.
                     "strict": bool(function.get("strict", False))}
        if function.get("description"):
            converted["description"] = function["description"]
        tools.append(converted)
    if tools:
        body["tools"] = tools
    choice = chat.get("tool_choice")
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        body["tool_choice"] = {"type": "function", "name": choice["function"].get("name")}
    elif isinstance(choice, str):
        body["tool_choice"] = choice
    if "parallel_tool_calls" in chat:
        body["parallel_tool_calls"] = chat["parallel_tool_calls"]
    response_format = chat.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema") or {}
        body["text"] = {"format": {"type": "json_schema", "name": schema.get("name") or "answer",
                                   "schema": schema.get("schema") or {}, "strict": bool(schema.get("strict", False))}}
    elif isinstance(response_format, dict) and response_format.get("type") == "json_object":
        body["text"] = {"format": {"type": "json_object"}}
    return body


def _chat_answer(response: dict[str, Any]) -> dict[str, Any]:
    """A Responses API answer in chat-completions shape, which every AUA caller reads."""
    text: list[str] = []
    calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            text += [str(part.get("text") or "") for part in item.get("content") or []
                     if isinstance(part, dict) and part.get("type") == "output_text"]
        elif item.get("type") == "function_call":
            calls.append({"id": item.get("call_id"), "type": "function",
                          "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}})
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    finish = "tool_calls" if calls else ("length" if response.get("status") == "incomplete" else "stop")
    usage = response.get("usage") or {}
    prompt, completion = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return {
        "id": response.get("id"), "object": "chat.completion", "model": response.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens") or prompt + completion),
            "prompt_tokens_details": {
                "cached_tokens": int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)},
            "completion_tokens_details": {
                "reasoning_tokens": int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)},
        },
    }


def prepare(
    payload: dict[str, Any],
    environ: Mapping[str, str] | None = None,
    *,
    openrouter_url: str = OPENROUTER_URL,
    openrouter_key: str | None = None,
) -> Call:
    """The endpoint, headers and body for *payload*, chosen by which keys exist.

    *openrouter_key* overrides the OpenRouter key names, for a caller told to read another one.
    """
    environ = os.environ if environ is None else environ
    requested = str(payload.get("model", ""))
    direct = openai_model(requested)
    openai_key = environ.get(OPENAI_KEY)
    only_openai = bool(direct) and "/" not in requested
    if only_openai and not openai_key:
        raise RouteError(f"{requested} is an OpenAI model id: set {OPENAI_KEY}")
    if direct and openai_key:

        def priced(answer: dict[str, Any]) -> dict[str, Any]:
            usage = answer.get("usage")
            if isinstance(usage, dict) and direct in OPENAI_PRICES:
                usage["cost"] = openai_cost(direct, usage)
            answer.setdefault("provider", "OpenAI")
            return answer

        headers = {"Authorization": f"Bearer {openai_key}"}
        if _reasons_with_tools(payload):
            return Call("openai", f"{OPENAI_URL}/responses", headers, _responses_body(payload, direct),
                        lambda answer: priced(_chat_answer(answer)))
        return Call("openai", f"{OPENAI_URL}/chat/completions", headers, _openai_body(payload, direct), priced)
    key = openrouter_key or next((environ[name] for name in OPENROUTER_KEYS if environ.get(name)), None)
    if not key:
        wanted = f"{OPENROUTER_KEYS[0]}" + (f" or {OPENAI_KEY}" if direct else "")
        raise RouteError(f"no key reaches {payload.get('model')}: set {wanted}")
    return Call("openrouter", f"{openrouter_url.rstrip('/')}/chat/completions",
                {"Authorization": f"Bearer {key}"}, payload, lambda answer: answer)


def reachable(
    model: str, environ: Mapping[str, str] | None = None, *, openrouter_key: str | None = None
) -> bool:
    """Whether some key can reach *model*: OpenRouter's for any model, OpenAI's for its own."""
    environ = os.environ if environ is None else environ
    if "/" not in model:
        return bool(model and environ.get(OPENAI_KEY))
    if openrouter_key or any(environ.get(name) for name in OPENROUTER_KEYS):
        return True
    return bool(openai_model(model) and environ.get(OPENAI_KEY))
