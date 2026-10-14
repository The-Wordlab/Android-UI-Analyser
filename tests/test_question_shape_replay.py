"""The question-shape replay asks both shapes about the same screen and scores them alike."""
from __future__ import annotations

import asyncio

import httpx
from experiments.aua_controller.bounded_navigator import INPUT_TOOL, TAP_TOOL
from experiments.aua_controller.replay_question_shapes import (
    BoundsReplay,
    FlatReplay,
    label_of,
    matches,
)
from PIL import Image


class Client:
    def __init__(self, choice):
        self.choice, self.questions = choice, []

    async def post(self, url, **kwargs):
        self.questions.append(kwargs["json"]["questions"])
        body = {"answers": [{"name": q["name"], "type": "choice", "choice": self.choice, "confidence": .99}
                            for q in kwargs["json"]["questions"]],
                "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0}}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))


def screen(tmp_path):
    path = tmp_path / "screen.png"
    Image.new("RGB", (90, 160), "white").save(path)
    return {"observation": {"screen": {"package": "com.example", "width": 90, "height": 160}, "meta": {"fingerprint": "f", "raw_image": str(path)},
                            "elements": [{"id": "el:send", "text": "Send", "clickable": True, "bounds": [0, 0, 10, 10]},
                                         {"id": "el:box", "text": "Ask", "editable": True, "clickable": True,
                                          "bounds": [0, 20, 90, 40]}]}}


def test_the_flat_list_is_one_question_and_types_the_quoted_text(tmp_path):
    frame = screen(tmp_path)
    client = Client("type:2:1")
    navigator = FlatReplay("Type exactly `Cat` into Ask.", api_key="k", http_client=client,
                           tools=[TAP_TOOL, INPUT_TOOL], action_space="full")
    navigator.ask_now = True
    action = asyncio.run(navigator(frame))
    [questions] = client.questions
    assert [q["name"] for q in questions] == ["move"]
    values = {c["value"] for c in questions[0]["choices"]}
    assert {"1", "2", "type:2:1", "finished", "other"} <= values and "press" not in values
    assert action == {"tool": INPUT_TOOL, "arguments": {"id": "el:box", "text": "Cat", "submit": False},
                      "reason": action["reason"]}
    proposal = navigator.last_decision["verdict"]
    label = label_of({"tool": INPUT_TOOL, "arguments": {"id": "el:box", "text": "Cat"}}, frame["observation"])
    assert matches(proposal, label, frame["observation"])
    assert not matches(proposal, {**label, "text": "Dog"}, frame["observation"])


def test_the_bounds_variant_gives_each_element_its_place_and_the_screen_its_size(tmp_path):
    navigator = BoundsReplay("Press Send.", api_key="k", tools=[TAP_TOOL], action_space="full")
    shown = navigator._state(screen(tmp_path))["this_is_the_new_screen"]
    assert shown["screen_size"] == [90, 160]
    assert shown["elements"] == [{"text": "Send", "bounds": [0, 0, 10, 10]},
                                 {"text": "Ask", "editable": True, "bounds": [0, 20, 90, 40]}]
