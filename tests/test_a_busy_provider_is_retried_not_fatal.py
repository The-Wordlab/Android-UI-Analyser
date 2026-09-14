"""A 429 from one provider used to end a run that had already driven the device for minutes.

2026-09-14, four controller-harness runs of the same scenario. Run 4 reached step 9, then::

    HTTPStatusError (HTTP 429): Provider returned error

``send()`` was a POST followed by ``raise_for_status()``. There was no second attempt, so the
device work already done was thrown away over a provider asking for the request later.

The policy here is deliberately narrow: retry only answers that mean "not now" (429, 5xx,
408/409/425) and transport failures where nothing was served, and honour the provider's own
Retry-After when it sends a usable one. A 400 or a 402 is a real refusal and still propagates
on the first try.
"""

from __future__ import annotations

import asyncio

import pytest
from experiments.aua_controller.transport import (
    MAX_ATTEMPTS,
    MAX_DELAY_S,
    RETRY_STATUS,
    TransportError,
    resilient_request,
    retry_after_seconds,
    retry_delay,
    retryable_http_status,
    tool_choice_route_missing,
)


def test_openrouter_tool_choice_route_miss_is_the_only_retryable_404():
    assert retryable_http_status(
        404, "No endpoints found that support the provided 'tool_choice' value."
    ) == 503
    assert retryable_http_status(404, "Not found") == 404
    assert tool_choice_route_missing(
        RuntimeError("HTTP 404: No endpoints found that support the provided 'tool_choice' value")
    )
    assert not tool_choice_route_missing(RuntimeError("HTTP 404: Not found"))


def test_tool_choice_route_miss_reads_the_http_response_body():
    class Response:
        text = "No endpoints found that support the provided 'tool_choice' value."

    class HttpError(RuntimeError):
        response = Response()

    assert tool_choice_route_missing(HttpError("404 Not Found for /chat/completions"))

NO_JITTER = lambda: 1.0  # noqa: E731 - a fixed jitter makes the backoff assertions exact


class Boom(Exception):
    """Stands in for httpx.HTTPStatusError without importing httpx into a unit test."""

    def __init__(self, status: int | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.headers = headers


def classify(exc: BaseException):
    return (exc.status, exc.headers) if isinstance(exc, Boom) else None


def test_the_statuses_that_mean_not_now_are_retried():
    for status in (408, 409, 425, 429, 500, 502, 503, 504):
        assert status in RETRY_STATUS
        assert retry_delay(0, status=status, jitter=NO_JITTER) is not None


def test_a_real_refusal_is_not_retried():
    # Bad request, unauthorised, payment required, not found, filtered-by-parameters.
    for status in (400, 401, 402, 403, 404, 422):
        assert status not in RETRY_STATUS
        assert retry_delay(0, status=status, jitter=NO_JITTER) is None


def test_a_connection_failure_served_nothing_so_it_is_retried():
    assert retry_delay(0, status=None, jitter=NO_JITTER) is not None


def test_the_backoff_doubles_and_then_stops_climbing():
    delays = [retry_delay(n, status=429, attempts=99, jitter=NO_JITTER) for n in range(8)]
    assert delays[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(delays) <= MAX_DELAY_S


def test_the_last_attempt_gives_up_rather_than_sleeping():
    assert retry_delay(MAX_ATTEMPTS - 1, status=429, jitter=NO_JITTER) is None


def test_jitter_spreads_workers_that_share_one_provider():
    spread = {retry_delay(2, status=429, attempts=99, jitter=(lambda v: lambda: v)(value))
              for value in (0.0, 0.25, 0.5, 1.0)}
    assert len(spread) == 4, "a fixed backoff would line every worker up on the same second"


def test_the_provider_s_own_retry_after_wins_over_the_backoff():
    assert retry_delay(0, status=429, headers={"Retry-After": "7"}, jitter=NO_JITTER) == 7.0
    assert retry_after_seconds({"retry-after": "2.5"}) == 2.5


def test_an_unusable_retry_after_falls_back_to_the_backoff():
    # The HTTP-date form needs the local clock to agree with the provider's; a skewed
    # clock turns a two second wait into a twenty minute one. An absurd value is ignored too.
    for headers in ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, {"Retry-After": "9000"},
                    {"Retry-After": "-1"}, {"Retry-After": ""}, None):
        assert retry_after_seconds(headers) is None


def test_a_run_survives_a_burst_and_returns_the_real_answer():
    calls = {"n": 0}
    slept: list[float] = []

    async def attempt():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom(429, {"Retry-After": "0"})
        return {"choices": [{"message": {"content": "ok"}}]}

    body = asyncio.run(resilient_request(
        attempt, classify=classify, sleep=lambda s: slept.append(s) or asyncio.sleep(0),
        jitter=NO_JITTER,
    ))
    assert body["choices"][0]["message"]["content"] == "ok"
    assert calls["n"] == 3 and slept == [0.0, 0.0]


def test_a_provider_that_stays_busy_still_surfaces_its_own_error():
    async def attempt():
        raise Boom(429)

    with pytest.raises(Boom):
        asyncio.run(resilient_request(
            attempt, classify=classify, sleep=lambda _: asyncio.sleep(0), jitter=NO_JITTER,
        ))


def test_a_non_transport_failure_propagates_untouched_on_the_first_try():
    calls = {"n": 0}

    async def attempt():
        calls["n"] += 1
        raise ValueError("endpoint returned non-object JSON")

    with pytest.raises(ValueError):
        asyncio.run(resilient_request(
            attempt, classify=classify, sleep=lambda _: asyncio.sleep(0), jitter=NO_JITTER,
        ))
    assert calls["n"] == 1, "a schema failure is not a busy provider; retrying it wastes money"


def test_every_retry_is_observable():
    seen: list[tuple[int, float, int | None]] = []
    calls = {"n": 0}

    async def attempt():
        calls["n"] += 1
        if calls["n"] < 2:
            raise Boom(503)
        return {}

    asyncio.run(resilient_request(
        attempt, classify=classify, sleep=lambda _: asyncio.sleep(0),
        on_retry=lambda n, delay, status: seen.append((n, delay, status)), jitter=NO_JITTER,
    ))
    assert seen == [(1, 1.0, 503)], "a silent retry hides a degrading provider"


def test_a_policy_with_no_attempts_is_rejected_rather_than_silently_doing_nothing():
    with pytest.raises(TransportError):
        asyncio.run(resilient_request(
            lambda: None, classify=classify, sleep=lambda _: asyncio.sleep(0), attempts=0,
        ))
