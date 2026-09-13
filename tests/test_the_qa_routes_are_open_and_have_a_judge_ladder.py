"""The manifest entries the QA harness drives, checked as configuration rather than prose.

Two routes were added on 2026-09-14 so product QA stops inheriting the benchmark's provider
pin. They are deliberately separate entries rather than edits to the pinned ones: the pinned
entries are what the model comparison in openrouter-comparison.json was measured on, and
rewriting them would retroactively change what those numbers mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.aua_controller.hosted import validate_request_config
from experiments.aua_controller.run_realapp import __file__ as RUNNER

MANIFEST = json.loads(
    (Path(RUNNER).with_name("openrouter-comparison.json")).read_text(encoding="utf-8")
)
BY_ID = {model["id"]: model for model in MANIFEST["models"]}
OPEN_IDS = ("or-deepseek-v4-flash-0731-low-open", "or-deepseek-v4p1-flash-low-open")


def test_every_manifest_entry_still_validates():
    for model in MANIFEST["models"]:
        validate_request_config(model["request_config"])


def test_no_two_entries_share_an_id():
    ids = [model["id"] for model in MANIFEST["models"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("entry_id", OPEN_IDS)
def test_the_qa_route_is_open_sorted_and_capped(entry_id):
    provider = BY_ID[entry_id]["request_config"]["provider"]
    assert provider["allow_fallbacks"] is True
    assert provider["sort"] == "throughput"
    assert "only" not in provider and "order" not in provider
    assert provider["max_price"]["prompt"] > 0 and provider["max_price"]["completion"] > 0


@pytest.mark.parametrize("entry_id", OPEN_IDS)
def test_the_qa_route_still_refuses_data_collection(entry_id):
    assert BY_ID[entry_id]["request_config"]["provider"]["data_collection"] == "deny"


@pytest.mark.parametrize("entry_id", OPEN_IDS)
def test_the_qa_route_does_not_use_openrouter_s_parameter_filter(entry_id):
    """require_parameters looks right and is not: it has now emptied a healthy pool twice.

    Run A5 on 2026-09-14 answered 404 "No endpoints found that support the provided
    'tool_choice' value" on a route whose price cap admits twelve endpoints that every one
    advertise tools and tool_choice - and three runs of the identical config had just
    succeeded. The pinned DeepInfra entry carries the same note from the day after the pilot.

    allow_fallbacks does the job instead: OpenRouter moves on when a provider cannot serve.
    """
    assert "require_parameters" not in BY_ID[entry_id]["request_config"]["provider"]


def test_the_pinned_benchmark_entries_were_left_alone():
    for entry_id in ("or-deepseek-v4-flash-0731-low-deepinfra",
                     "or-deepseek-v4p1-flash-vision-deepinfra",
                     "or-deepseek-v4p1-flash-low-novita"):
        provider = BY_ID[entry_id]["request_config"]["provider"]
        assert provider["allow_fallbacks"] is False
        assert len(provider["only"]) == 1


def test_the_two_models_under_comparison_have_matching_open_routes():
    # A speed or cost comparison is only meaningful if both sides route the same way.
    first, second = (BY_ID[entry_id]["request_config"]["provider"] for entry_id in OPEN_IDS)
    assert first["sort"] == second["sort"]
    assert first["allow_fallbacks"] == second["allow_fallbacks"]
    assert first.get("require_parameters") == second.get("require_parameters")


def test_an_open_judge_route_can_take_images():
    # The QA harness runs with --vision, which needs an image-capable judge and ladder.
    assert BY_ID["or-deepseek-v4p1-flash-low-open"]["vision"] is True


def test_an_open_route_does_not_claim_a_quantization_it_cannot_know():
    for entry_id in OPEN_IDS:
        assert "not attributable" in BY_ID[entry_id]["provider_quantization"] or \
               "provider default" in BY_ID[entry_id]["provider_quantization"]
