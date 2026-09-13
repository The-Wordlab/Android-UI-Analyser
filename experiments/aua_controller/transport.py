"""Retry policy for hosted model requests.

No network, credential discovery, inference, or device operations live here. The
policy is pure so a test can drive every branch without a socket or a real clock.

A provider that answers 429 or 5xx has not refused the work; it has asked for it
later. Before this module a single such answer ended a run that had already spent
three minutes driving a device, which is the expensive half of the work.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

# 408/409 are included because OpenRouter surfaces upstream queue conflicts as both.
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BASE_DELAY_S = 1.0
MAX_DELAY_S = 20.0


class TransportError(RuntimeError):
    """A hosted request failed and the policy has stopped retrying it."""


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """The provider's own Retry-After in seconds, when it sent a usable one.

    Only the delta-seconds form is honoured. The HTTP-date form would need the
    local clock to agree with the provider's, and a skewed clock turns a two
    second wait into a twenty minute one.
    """
    if not headers:
        return None
    raw = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds < 0 or seconds > MAX_DELAY_S * 3:
        return None
    return seconds


def retry_delay(
    attempt: int,
    *,
    status: int | None,
    headers: Mapping[str, str] | None = None,
    attempts: int = MAX_ATTEMPTS,
    jitter: Callable[[], float] = random.random,
) -> float | None:
    """Seconds to wait before retry *attempt*+1, or None to give up now.

    *status* is None for a connection or read failure, which is retried like a
    503: nothing was served, so nothing can have been charged.
    """
    if attempt + 1 >= attempts:
        return None
    if status is not None and status not in RETRY_STATUS:
        return None
    stated = retry_after_seconds(headers)
    if stated is not None:
        return stated
    backoff = min(BASE_DELAY_S * (2 ** attempt), MAX_DELAY_S)
    # Full jitter. Several workers share one provider, so a fixed backoff would
    # line their retries up on the same second and reproduce the burst.
    return backoff * jitter()


async def resilient_request(
    attempt_once: Callable[[], Any],
    *,
    classify: Callable[[BaseException], tuple[int | None, Mapping[str, str] | None] | None],
    sleep: Callable[[float], Any],
    attempts: int = MAX_ATTEMPTS,
    on_retry: Callable[[int, float, int | None], None] | None = None,
    jitter: Callable[[], float] = random.random,
) -> Any:
    """Await *attempt_once* until it succeeds or the policy stops retrying.

    *classify* turns an exception into (status, headers), or returns None when the
    failure is not a transport failure at all and must propagate untouched.
    """
    if attempts < 1:
        raise TransportError("retry policy needs at least one attempt")
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await attempt_once()
        except BaseException as exc:  # noqa: BLE001 - re-raised below unless retryable
            classified = classify(exc)
            if classified is None:
                raise
            status, headers = classified
            delay = retry_delay(attempt, status=status, headers=headers,
                                attempts=attempts, jitter=jitter)
            if delay is None:
                raise
            last = exc
            if on_retry is not None:
                on_retry(attempt + 1, delay, status)
            await sleep(delay)
    raise TransportError(f"request failed after {attempts} attempts") from last
