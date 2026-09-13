"""Explicit OpenRouter request configuration and reported-spend boundary.

No network, credential discovery, inference, or device operations live here.
The spend stop is checked on reported usage; one in-flight request can exceed it.
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit


class HostedError(ValueError):
    """Hosted configuration or cost accounting cannot be trusted."""


BACKENDS = ("openai-compatible", "openrouter")
ASSISTANT_FIELDS = {"role", "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details"}


def assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy native signatures/blocks intact, omitting response-only annotations."""
    return {key: copy.deepcopy(value) for key, value in message.items() if key in ASSISTANT_FIELDS}


def validate_endpoint(base_url: str, key: str | None) -> None:
    url = urlsplit(base_url)
    if (url.scheme != "https" or url.netloc not in {"openrouter.ai", "openrouter.ai:443"}
            or url.path.rstrip("/") != "/api/v1" or url.query or url.fragment):
        raise HostedError("OpenRouter requires https://openrouter.ai/api/v1 without credentials or query")
    if not isinstance(key, str) or not key.strip():
        raise HostedError("OpenRouter API key environment variable is missing or empty")


def _nonnegative(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise HostedError(f"{label} must be a finite nonnegative number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise HostedError(f"{label} must be a finite nonnegative number") from exc
    if not number.is_finite() or number < 0:
        raise HostedError(f"{label} must be a finite nonnegative number")
    return number


SORT_ORDERS = ("throughput", "latency", "price")


def _validate_pinned(provider: dict[str, Any]) -> None:
    """One named provider serves every request, so a result is attributable to it."""
    only = provider.get("only")
    if (not isinstance(only, list) or len(only) != 1 or not isinstance(only[0], str)
            or not only[0].strip() or provider.get("order", only) != only):
        raise HostedError("pin one provider with only/order and allow_fallbacks=false")
    if "sort" in provider:
        raise HostedError("a pinned route has nothing to sort; drop sort or allow fallbacks")


def _validate_open(provider: dict[str, Any]) -> None:
    """Any provider within the price cap may serve, so a burst on one does not end the run."""
    if provider.get("only") or provider.get("order"):
        raise HostedError("an open route must not also name only/order")
    sort = provider.get("sort")
    if sort is not None and sort not in SORT_ORDERS:
        raise HostedError(f"provider.sort must be one of {', '.join(SORT_ORDERS)}")


def validate_request_config(config: dict[str, Any]) -> dict[str, Any]:
    """Allow routing/reasoning options, never replacement messages/tools/models."""
    if not isinstance(config, dict) or set(config) - {"reasoning", "provider", "plugins", "temperature"}:
        raise HostedError("unsupported OpenRouter request_config field")
    out = copy.deepcopy(config)
    provider = out.get("provider")
    allowed = {"only", "order", "allow_fallbacks", "require_parameters", "data_collection", "zdr",
               "quantizations", "max_price", "enforce_distillable_text", "sort"}
    if not isinstance(provider, dict) or set(provider) - allowed:
        raise HostedError("OpenRouter requires explicit supported provider settings")
    # Two shapes are valid, and which one is in force must be stated rather than defaulted.
    # A benchmark comparing models needs the pinned shape: a number is only attributable to
    # a provider that actually served it. A QA run needs the open shape: 28 providers serve
    # deepseek-v4-flash-0731, and pinning one of them turned that provider's 429 into a
    # BLOCKED verdict for a product that was working.
    fallbacks = provider.get("allow_fallbacks")
    if fallbacks is False:
        _validate_pinned(provider)
    elif fallbacks is True:
        _validate_open(provider)
    else:
        raise HostedError("provider.allow_fallbacks must be explicitly true (open) or false (pinned)")
    # require_parameters is optional. OpenRouter's parameter filter has excluded pinned
    # providers that do serve the request (a 404 "Filter by Parameters" observed the day
    # after the pilot), so the harness no longer mandates it. On an open route it is worth
    # setting: it drops the endpoints that cannot serve native tool calls at all.
    if provider.get("require_parameters", False) not in (True, False):
        raise HostedError("provider.require_parameters must be a boolean when present")
    # Required on both shapes. On an open route it is the only thing standing between a
    # cheap model and the most expensive endpoint that happens to be fastest right now.
    caps = provider.get("max_price")
    if not isinstance(caps, dict) or set(caps) != {"prompt", "completion"}:
        raise HostedError("provider.max_price requires prompt and completion price caps per million tokens")
    for name, value in caps.items():
        _nonnegative(value, f"provider.max_price.{name}")
    reasoning = out.get("reasoning", {})
    if not isinstance(reasoning, dict) or set(reasoning) - {"enabled", "effort", "max_tokens", "exclude"}:
        raise HostedError("unsupported reasoning configuration")
    if reasoning.get("exclude") is True:
        raise HostedError("reasoning blocks must remain available for native tool continuity")
    if "temperature" in out:
        value = out["temperature"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise HostedError("temperature must be finite and nonnegative")
    disabled = [{"id": "context-compression", "enabled": False}]
    if out.get("plugins", disabled) != disabled:
        raise HostedError("hosted benchmark requires context compression disabled and no other plugins")
    out["plugins"] = disabled
    return out


def configure_payload(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.pop("chat_template_kwargs", None)
    # Some selected providers do not advertise this optional parameter. The host
    # still rejects every response containing more than one native tool call.
    out.pop("parallel_tool_calls", None)
    out.pop("temperature", None)  # Omitted means provider default, not an invented zero.
    out.update(validate_request_config(config))
    return out


class CostGuard:
    """Stop on absent cost and after the configured reported USD limit is reached."""

    def __init__(self, limit_usd: float = 0.10) -> None:
        self.limit = _nonnegative(limit_usd, "cost limit")
        if self.limit <= 0:
            raise HostedError("cost limit must be positive")
        self.total = Decimal(0)
        self.responses = 0
        self.missing = False

    def before_request(self) -> None:
        if self.missing:
            raise HostedError("reported usage.cost missing or invalid; further model requests stopped")
        if self.total >= self.limit:
            raise HostedError("reported OpenRouter cost limit reached; further model requests stopped")

    def consume(self, response: dict[str, Any]) -> None:
        usage = response.get("usage")
        try:
            cost = _nonnegative(usage.get("cost") if isinstance(usage, dict) else None, "usage.cost")
        except HostedError:
            self.missing = True
            raise HostedError("reported usage.cost missing or invalid; further model requests stopped") from None
        self.total += cost
        self.responses += 1

    def report(self) -> dict[str, Any]:
        return {"reported_usd": float(self.total), "limit_usd": float(self.limit),
                "responses_with_cost": self.responses, "missing_or_invalid_cost": self.missing,
                "limit_reached": self.total >= self.limit,
                "boundary": "reported spend; one in-flight request may exceed the limit"}
