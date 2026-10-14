"""Decisions may propose only offered moves and must account for paid refusals."""
from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest
from experiments.aua_controller.bounded_navigator import TAP_TOOL, Choice
from experiments.aua_controller.decisions_navigator import (
    DecisionsNavigator,
    decode_answers,
    translate_questions,
)
from PIL import Image


@pytest.fixture
def screen(tmp_path):
    path = tmp_path / "screen.png"
    Image.new("RGB", (900, 1600), "white").save(path)
    return {"ok": True, "observation": {"screen": {"package": "com.example.demo"},
            "meta": {"fingerprint": "fresh-1", "raw_image": str(path)}, "elements": [
                {"id": "el:a", "text": "Settings", "clickable": True, "bounds": [0, 0, 100, 100]},
                {"id": "el:b", "text": "Help", "clickable": True, "bounds": [0, 100, 100, 200]}]}}


class Client:
    def __init__(self, *, confidence=.99, mutate=None, status=200):
        self.calls = []
        self.confidence, self.mutate, self.status = confidence, mutate, status

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        body = {"answers": [{"name": q["name"], "type": "choice",
                             "choice": "press" if q["name"] == "operation" else "1",
                             "confidence": self.confidence} for q in kwargs["json"]["questions"]],
                "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 100}}}
        if self.mutate:
            self.mutate(body)
        return httpx.Response(self.status, json=body, request=httpx.Request("POST", url))


def nav(client, **kwargs):
    return DecisionsNavigator("Open settings", api_key="test-key", http_client=client,
                              tools=[TAP_TOOL], **kwargs)


def test_decisions_sends_image_and_offered_choices_then_proposes_semantic_id(screen):
    client = Client()
    navigator = nav(client)
    proposal = asyncio.run(navigator(screen))
    assert proposal["tool"] == TAP_TOOL and proposal["arguments"]["id"] == "el:a"
    url, call = client.calls[0]
    assert url == "https://api.openai.com/v1/decisions"
    payload = call["json"]
    assert payload["model"] == "gpt-6-luna"
    assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert all(len(q["choices"]) >= 2 for q in payload["questions"])
    assert navigator.report()["usd"] == pytest.approx(.00009)
    assert navigator.report()["accepted"] == navigator.report()["images_sent"] == 1
    assert navigator.last_decision["request"]["image"]["source"] == "observation_raw_image"


@pytest.mark.parametrize("confidence,shadow", [(.1, False), (.99, True)])
def test_uncertainty_and_shadow_hand_back_to_regular_controller(screen, confidence, shadow):
    navigator = nav(Client(confidence=confidence), shadow=shadow)
    assert asyncio.run(navigator(screen)) is None
    assert navigator.report()["accepted"] == 0 and navigator.report()["requests"] == 1


def test_missing_image_and_unoffered_tool_never_make_a_paid_request(screen):
    client = Client()
    no_image = copy.deepcopy(screen)
    no_image["observation"]["meta"].pop("raw_image")
    navigator = nav(client)
    assert asyncio.run(navigator(no_image)) is None
    assert navigator.last_declined == "screenshot_unavailable", "the step says why it was never asked"
    navigator = DecisionsNavigator("Open settings", api_key="test", http_client=client, tools=["session_finish"])
    assert asyncio.run(navigator(screen)) is None
    assert not client.calls


def test_repeat_on_unchanged_screen_escalates_without_a_second_action(screen):
    navigator = nav(Client())
    assert asyncio.run(navigator(screen)) is not None
    navigator.observed(TAP_TOOL, {"id": "el:a"})
    assert asyncio.run(navigator(screen)) is None
    assert navigator.report()["declined"]["repeat_on_unchanged_screen"] == 1


@pytest.mark.parametrize("mutation", [
    lambda body: body["answers"].clear(),
    lambda body: body["answers"][0].update(type="refusal"),
    lambda body: body["answers"][0].update(choice="invented"),
    lambda body: body["answers"].append(body["answers"][0]),
    lambda body: body["answers"][0].update(confidence=True),
    lambda body: body["answers"][0].update(confidence=1.1),
])
def test_paid_invalid_answers_fall_back_and_keep_the_charge(screen, mutation):
    navigator = nav(Client(mutate=mutation))
    assert asyncio.run(navigator(screen)) is None
    report = navigator.report()
    assert report["requests"] == 1 and report["failed_requests"] == 1
    assert report["usd"] > 0 and report["cost_complete"] is True


@pytest.mark.parametrize("client", [Client(status=503), Client(mutate=lambda body: body.pop("usage"))])
def test_unknown_transport_or_usage_charge_marks_cost_incomplete(screen, client):
    navigator = nav(client)
    assert asyncio.run(navigator(screen)) is None
    assert navigator.report()["cost_complete"] is False
    assert navigator.report()["attempted_requests"] == 1


def test_single_authored_text_is_deterministic_without_a_fabricated_choice():
    questions = {"operation": Choice("Choose", {"type": "Type", "other": "Hand off"}),
                 "text": Choice("Quoted text", {"1": "Hello there"})}
    api, fixed = translate_questions(questions)
    assert [q["name"] for q in api] == ["operation"]
    answers = decode_answers({"answers": [{"name": "operation", "type": "choice",
                                           "choice": "type", "confidence": .99}]}, questions, fixed)
    assert answers["text"].choice == "1"


def test_failed_transcript_never_contains_credentials(screen, tmp_path):
    transcript = tmp_path / "navigator.jsonl"
    navigator = nav(Client(status=401), transcript_path=transcript)
    asyncio.run(navigator(screen))
    assert "test-key" not in transcript.read_text()
    assert json.loads(transcript.read_text())["error_type"] == "HTTPStatusError"
