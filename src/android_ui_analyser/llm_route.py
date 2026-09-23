"""Where one model request goes: straight to OpenAI when there is a key for it, else OpenRouter.

Every AUA model request is written once, in OpenRouter's chat-completions shape, and names its
model the way OpenRouter does (``openai/gpt-6-luna``). :func:`prepare` picks the endpoint from the
keys that exist and returns the exact URL, headers and body for it:

* an ``openai/*`` model whose price is known goes to api.openai.com when ``OPENAI_API_KEY`` is
  set -- a company key is cheaper than OpenRouter's resale of the same model -- unless the request
  combines function tools with reasoning, which OpenAI's chat completions refuse;
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
    body["messages"] = [
        {key: value for key, value in message.items() if key not in _OPENROUTER_MESSAGE_FIELDS}
        if message.get("role") == "assistant" else message
        for message in body.get("messages", [])
    ]
    return body


def _reasons_with_tools(payload: Mapping[str, Any]) -> bool:
    """OpenAI's chat completions refuse function tools with reasoning on for GPT-6 models.

    Live 2026-09-23: "Function tools with reasoning_effort are not supported for gpt-6-luna in
    /v1/chat/completions ... set reasoning_effort to 'none'". OpenRouter serves the same request,
    so it stays there rather than being answered differently depending on which key exists.
    """
    reasoning = payload.get("reasoning") or {}
    return bool(payload.get("tools")) and reasoning.get("enabled") is not False and bool(reasoning)


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
    if direct and openai_key and (only_openai or not _reasons_with_tools(payload)):

        def priced(answer: dict[str, Any]) -> dict[str, Any]:
            usage = answer.get("usage")
            if isinstance(usage, dict) and direct in OPENAI_PRICES:
                usage["cost"] = openai_cost(direct, usage)
            answer.setdefault("provider", "OpenAI")
            return answer

        return Call("openai", f"{OPENAI_URL}/chat/completions",
                    {"Authorization": f"Bearer {openai_key}"}, _openai_body(payload, direct), priced)
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
