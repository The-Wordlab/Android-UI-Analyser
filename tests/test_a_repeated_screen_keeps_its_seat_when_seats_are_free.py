"""A screen the judge has already seen still gets a seat when there are seats to spare.

``judged_frame_sample`` ranks observations so that new screens come first and a repeat of a
state it has already shown comes last. A repeat was not ranked at all: it could never be kept,
even when the journey was shorter than the seat budget. Seen live on a six-observation run judged
at eight seats: the contract said "Cancel keeps it", the observation after Cancel was the very
same menu as before the dialog -- which is exactly the proof -- and it was dropped for being a
repeat. The judge, shown the dialog and then the list after the later Delete, could only answer
"not verified".

A repeat that proves nothing costs the judge one short line. A repeat that proves something is
the whole verdict. When seats are free it stays; when they are not, it still yields to a screen
the judge has not seen.
"""

from __future__ import annotations

from experiments.aua_controller.judgement import judged_frame_sample


def _screen(title: str, ref: str, *, fingerprint: str | None = None) -> dict:
    elements = [{"text": title}, {"text": f"{title} option", "clickable": True}]
    return {"observation": {"screen": {"package": "example.app", "activity": ".Main"},
                            "meta": {"fingerprint": fingerprint or ref}, "elements": elements},
            "_judge_evidence": {"ref": ref}}


def _refs(frames: list[dict]) -> list[str]:
    return [frame["_judge_evidence"]["ref"] for frame in frames]


# The live shape: list, long-press menu, confirm dialog, the menu again after Cancel, the dialog
# again, the list after Delete, then the final frame the judge is handed separately.
JOURNEY = [
    _screen("Threads", "E0"),
    _screen("Menu", "E1"),
    _screen("Confirm", "E2"),
    _screen("Menu", "E3", fingerprint="E1"),
    _screen("Confirm", "E4", fingerprint="E2"),
    _screen("Threads", "E5"),
    _screen("Threads", "E6"),
]


def test_every_observation_is_shown_when_they_all_fit() -> None:
    assert _refs(judged_frame_sample(JOURNEY, limit=8)) == ["E0", "E1", "E2", "E3", "E4", "E5"]


def test_a_repeat_still_yields_to_a_screen_the_judge_has_not_seen() -> None:
    crowded = JOURNEY[:5] + [_screen("Settings", "E5"), _screen("About", "E6"), _screen("Threads", "E7")]
    kept = _refs(judged_frame_sample(crowded, limit=5))
    assert kept == ["E0", "E1", "E2", "E5", "E6"], kept
