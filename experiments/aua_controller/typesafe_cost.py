"""Published Jev pricing for estimates, and where a Jev request is sent."""

from __future__ import annotations

import os
from collections.abc import Mapping

# https://typesafe.ai/blog/introducing-system-one-models-and-jev (September 2026).
# Input: $42 per billion tokens. Output: free. OpenRouter lists typesafe/jev-* at the same price.
USD_PER_INPUT_TOKEN = 42 / 1e9

# OpenRouter serves Jev's System One API under /api/v1/systemone, which the TypeSafe SDK reaches
# with this base URL and an OpenRouter key. See openrouter.ai/docs/guides/community/typesafe-sdk.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


def client_options(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Constructor arguments for a TypeSafe client.

    A TypeSafe key wins, so its own environment (TYPESAFE_API_KEY, TYPESAFE_BASE_URL) is left to
    the SDK. Without one, the OpenRouter key every AUA controller run already holds reaches Jev
    through OpenRouter, so enabling the navigator or judge needs no second credential.
    """
    environ = os.environ if environ is None else environ
    if environ.get("TYPESAFE_API_KEY"):
        return {}
    key = environ.get("OPEN_ROUTER_API_KEY") or environ.get("OPENROUTER_API_KEY")
    if not key:
        return {}
    return {"api_key": key, "base_url": environ.get("TYPESAFE_BASE_URL") or OPENROUTER_BASE_URL}
