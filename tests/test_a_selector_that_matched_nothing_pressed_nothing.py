"""A text selector that matches nothing is refused before any press, like a stale target.

AUA resolves a `text`/`rid` selector before it acts, and a miss raises `selector_not_found` with
the screen it read. The journal counted it as a failed action: a guest row tapped `Ask me later`
on a prompt that had not appeared, went on, and passed both judges, and then the flow export
called the journal unproved and turned the pass into QA_ERROR (2026-09-28).
"""

from __future__ import annotations

from experiments.aua_controller.action_evidence import definitive_selector_miss

OBSERVATION = {"schema_version": 1, "screen": {"width": 720, "height": 1280}, "elements": [],
               "meta": {"fingerprint": "f1"}}


def miss(code: str, hint: str, **extra) -> dict:
    return {"error": {"code": code, "message": "no element matches text:Ask me later", "hint": hint,
                      "observation_present": True, "observation": dict(OBSERVATION), **extra}}


def test_a_selector_miss_with_the_screen_it_read_is_a_no_action_miss() -> None:
    assert definitive_selector_miss(miss("selector_not_found", "nearest: id=6 text='Chat'"))
    assert definitive_selector_miss(miss("element_not_found", "No action was sent: the target moved"))


def test_anything_that_may_have_pressed_or_lacks_the_screen_is_not() -> None:
    assert not definitive_selector_miss(miss("selector_not_found", "nearest: x", action_sent=True))
    assert not definitive_selector_miss(miss("element_not_found", "stale"))
    assert not definitive_selector_miss(miss("device_error", "adb went away"))
    blind = miss("selector_not_found", "nearest: x")
    blind["error"]["observation"] = None
    assert not definitive_selector_miss(blind)
