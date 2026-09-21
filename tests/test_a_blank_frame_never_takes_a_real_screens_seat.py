"""A frame that shows nothing must not take a judge seat from a frame that shows something.

``judged_frame_sample`` reserves one seat per screen family so a route can be judged from its
beginning, not just its tail. A zero-element capture -- a transition, a spinner, a screen not
yet drawn -- has no title, so its family was the empty string, which is a family the sampler
had never seen: it took a reserved seat. Seen live on a ten-frame run judged at eight: seat [1]
went to a frame with 0 elements while the 6-element frame before it was dropped, and a
criterion that frame proved came back "no frame captures that screen".
"""

from __future__ import annotations

from experiments.aua_controller.judgement import _frame_traits, judged_frame_sample


def _screen(title: str, ref: str, *, state: str = "") -> dict:
    elements = [{"text": title}, {"text": f"{state or title} option", "clickable": True}]
    return {"observation": {"screen": {"package": "example.app", "activity": ".Main"},
                            "meta": {"fingerprint": ref}, "elements": elements},
            "_judge_evidence": {"ref": ref}}


def _blank(ref: str) -> dict:
    return {"observation": {"screen": {"package": "example.app", "activity": ".Main"},
                            "meta": {"fingerprint": ref}, "elements": []},
            "_judge_evidence": {"ref": ref}}


# The live shape: the first family twice, a blank capture, six more families, then the final
# frame the judge is handed separately. Eight seats for nine candidates.
JOURNEY = [
    _screen("Welcome", "E0"),
    _screen("Welcome", "E1", state="Signed out"),
    _blank("E2"),
    *(_screen(title, f"E{index}") for index, title in
      enumerate(("Home", "Settings", "Language", "Themes", "About", "Help"), start=3)),
    _screen("Home", "E9"),
]


def _refs(frames: list[dict]) -> list[str]:
    return [frame["_judge_evidence"]["ref"] for frame in frames]


def test_a_blank_frame_does_not_push_a_real_screen_out() -> None:
    refs = _refs(judged_frame_sample(JOURNEY, 8))
    assert "E1" in refs, "a frame with six elements lost its seat to one with none"
    assert "E2" not in refs
    assert len(refs) == 8


def test_a_blank_frame_is_still_shown_when_there_is_room() -> None:
    # A loading capture is evidence of a transient state; it is only ever the lowest priority.
    assert "E2" in _refs(judged_frame_sample(JOURNEY, 9))


def test_a_blank_frame_claims_no_screen_family_even_with_an_activity_name() -> None:
    assert _frame_traits(_blank("E2"))[0] == ""
    assert _frame_traits(_screen("Welcome", "E0"))[0] == "Welcome"
