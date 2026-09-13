"""A QA run wants the answer; a benchmark wants to know who gave it. Both shapes are legal now.

The harness began as a model benchmark, where pinning one provider is the whole point: a
latency or a cost number is meaningless unless you know which endpoint produced it. So
``validate_request_config`` refused to start without ``only`` of length one and
``allow_fallbacks: false``.

Driving product QA through the same code inherited that pin. On 2026-09-14 DeepInfra answered
429 and the run came back BLOCKED - for a product that was working - while 27 other providers
served the same model. So an open shape is legal too, and which shape is in force must be
stated rather than defaulted, because silently unpinning a benchmark would quietly invalidate
its numbers.

``max_price`` stays required on both. It is the only ceiling left on an open route.
"""

from __future__ import annotations

import pytest
from experiments.aua_controller.hosted import SORT_ORDERS, HostedError, validate_request_config

CAPS = {"prompt": 0.15, "completion": 0.4}


def pinned(**provider):
    return {"provider": {"only": ["deepinfra"], "order": ["deepinfra"],
                         "allow_fallbacks": False, "max_price": CAPS, **provider}}


def open_route(**provider):
    return {"provider": {"allow_fallbacks": True, "max_price": CAPS, **provider}}


def test_the_benchmark_pin_still_works_exactly_as_before():
    out = validate_request_config(pinned())
    assert out["provider"]["only"] == ["deepinfra"]
    assert out["provider"]["allow_fallbacks"] is False


def test_an_open_route_is_accepted_and_may_choose_the_fastest_endpoint():
    out = validate_request_config(open_route(sort="throughput"))
    assert out["provider"]["allow_fallbacks"] is True
    assert out["provider"]["sort"] == "throughput"
    assert "only" not in out["provider"]


def test_every_documented_sort_order_is_accepted():
    for order in SORT_ORDERS:
        assert validate_request_config(open_route(sort=order))["provider"]["sort"] == order


def test_an_invented_sort_order_is_refused_rather_than_silently_ignored():
    with pytest.raises(HostedError, match="provider.sort"):
        validate_request_config(open_route(sort="cheapest"))


def test_the_shape_must_be_stated_so_a_benchmark_cannot_drift_open():
    # Omitting allow_fallbacks used to be a hard failure and must stay one: OpenRouter's own
    # default is to fall back, so a silently-accepted config would stop being attributable.
    with pytest.raises(HostedError, match="allow_fallbacks"):
        validate_request_config({"provider": {"max_price": CAPS}})
    with pytest.raises(HostedError, match="allow_fallbacks"):
        validate_request_config({"provider": {"allow_fallbacks": "yes", "max_price": CAPS}})


def test_a_half_open_route_is_refused_because_its_intent_is_ambiguous():
    with pytest.raises(HostedError, match="only/order"):
        validate_request_config(open_route(only=["deepinfra"], order=["deepinfra"]))


def test_sorting_a_pinned_route_is_refused_because_there_is_nothing_to_sort():
    with pytest.raises(HostedError, match="sort"):
        validate_request_config(pinned(sort="throughput"))


def test_a_pin_of_more_than_one_provider_is_still_refused():
    with pytest.raises(HostedError, match="pin one provider"):
        validate_request_config(pinned(only=["deepinfra", "novita"], order=["deepinfra", "novita"]))


def test_an_open_route_still_needs_a_price_ceiling():
    provider = {"allow_fallbacks": True, "sort": "throughput"}
    with pytest.raises(HostedError, match="max_price"):
        validate_request_config({"provider": provider})


def test_an_open_route_may_require_native_tool_support():
    out = validate_request_config(open_route(require_parameters=True))
    assert out["provider"]["require_parameters"] is True


def test_the_data_policy_is_unchanged_by_opening_the_route():
    out = validate_request_config(open_route(data_collection="deny"))
    assert out["provider"]["data_collection"] == "deny"
