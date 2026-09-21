"""`network_calls` is a list of URL paths, and the path scrubber ate them.

`_PATH` is there to keep the operator's own filesystem out of a hosted request. A URL path
looks exactly like one, so `GET /v1/profile -> 200` reached the hosted model as
`GET [private-path] -> 200` -- the evidence with the evidence taken out. These are host-observed
proxy records of the app talking to its own backend, never a local path, so the scrubber skips
them. Everything else it does to them stays: a value that is a known identity is still replaced.
"""

import json

from experiments.aua_controller.hosted_projection import hosted_model_view


def test_a_network_call_reaches_the_model_with_its_route_intact():
    result = hosted_model_view(
        {"observation": {"meta": {"network_calls": ["GET /v1/items -> 200",
                                                    "POST /v1/session -> no answer yet"]}}}
    )
    assert result["observation"]["meta"]["network_calls"] == [
        "GET /v1/items -> 200", "POST /v1/session -> no answer yet"]


def test_an_identity_inside_a_network_call_is_still_taken_out():
    result = hosted_model_view(
        {"session_id": "session-fictional-123",
         "observation": {"meta": {"network_calls": ["GET /v1/u/session-fictional-123 -> 200"]}}}
    )
    assert "session-fictional-123" not in json.dumps(result)
    assert "/v1/u/" in result["observation"]["meta"]["network_calls"][0]


def test_a_real_host_path_somewhere_else_is_still_scrubbed():
    result = hosted_model_view({"warnings": ["wrote /Users/fictional/private/report.json"]})
    assert "/Users/fictional" not in json.dumps(result)


def test_the_exemption_does_not_leak_to_a_neighbouring_field():
    result = hosted_model_view(
        {"observation": {"meta": {"network_calls": ["GET /v1/items -> 200"],
                                  "note": "saved to /Users/fictional/private/out.json"}}}
    )
    assert "/Users/fictional" not in json.dumps(result)
    assert result["observation"]["meta"]["network_calls"] == ["GET /v1/items -> 200"]
