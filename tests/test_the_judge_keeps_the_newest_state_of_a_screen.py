"""When seats run out, the newest state of a screen must outlive the oldest repeat of one.

The judge is asked whether the goal happened. A goal that changes something -- a language, a
theme, a toggle -- proves itself in the *later* captures of screens the journey already visited:
Settings again, now in the new language. The sampler ranked those as "same family, new state",
correctly, and then broke ties oldest-first, so a second capture of the login screen outranked
the Spanish Settings screen. Seen live 2026-09-21: eleven frames, eight seats, the two frames that
proved the switch were the two dropped, and a run that achieved its goal was judged unverified
with the note "the frame produced by that action is missing".
"""

from __future__ import annotations

from experiments.aua_controller.judgement import judged_frame_sample


def _screen(title: str, ref: str, *rows: str, known: str | None = None) -> dict:
    """A frame as AUA reports it: the map's ``known_screen`` names the screen in every language."""
    elements = [{"text": title}, *({"text": row, "clickable": True} for row in rows)]
    meta = {"fingerprint": ref, **({"known_screen": known} if known else {})}
    return {"observation": {"screen": {"package": "example.app"}, "meta": meta, "elements": elements},
            "_judge_evidence": {"ref": ref}}


# The live shape: a language switch. Same screens twice, second time in the new language.
JOURNEY = [
    _screen("Welcome", "E00", "Sign in", "Browse as a guest"),
    _screen("Welcome", "E01", "Sign in"),                       # transitional second capture
    _screen("Welcome", "E02", "Sign in", "Browse as a guest"),  # and a third
    _screen("Home", "E03", "Chat", "Settings", known="home"),
    _screen("Settings", "E04", "Theme", "App language en", known="settings"),
    _screen("App language", "E05", "English", "Spanish", known="language_option"),
    _screen("Idioma", "E06", "Inglés", "Español", known="language_option"),  # now in Spanish
    _screen("Ajustes", "E07", "Tema", "Idioma es", known="settings"),         # now in Spanish
    _screen("Inicio", "E08", "Chat", "Ajustes", known="home"),                # now in Spanish
    _screen("Inicio", "E09", "Chat", "Ajustes", known="home"),                # final, judged apart
]


def _refs(frames: list[dict]) -> list[str]:
    return [frame["_judge_evidence"]["ref"] for frame in frames]


def test_the_frames_that_prove_the_change_survive_the_seat_limit() -> None:
    refs = _refs(judged_frame_sample(JOURNEY, 8))
    assert {"E06", "E07", "E08"} <= set(refs), f"the post-change screens were dropped: {refs}"


def test_the_oldest_repeat_is_what_gives_way() -> None:
    refs = _refs(judged_frame_sample(JOURNEY, 8))
    assert "E01" not in refs or "E02" not in refs
    assert len(refs) == 8


def test_every_screen_family_still_gets_its_first_seat() -> None:
    refs = _refs(judged_frame_sample(JOURNEY, 8))
    assert {"E00", "E03", "E04", "E05"} <= set(refs)


def test_order_shown_to_the_judge_stays_chronological() -> None:
    refs = _refs(judged_frame_sample(JOURNEY, 8))
    assert refs == sorted(refs)


def _with_progress(frame: dict) -> dict:
    return {**frame, "goal_progress": {"completed": 0, "current": {"id": "phase_1"}}}


# The live shape once more: every capture reports goal progress except one transitional repeat
# of the login screen, whose result carried no progress block at all.
JOURNEY_WITH_A_SILENT_FRAME = [_with_progress(f) if f["_judge_evidence"]["ref"] != "E02" else f
                               for f in JOURNEY]


def test_a_frame_that_says_nothing_about_progress_is_not_a_progress_change() -> None:
    # Absence of the field used to read as a checkpoint boundary and outrank every state change.
    refs = _refs(judged_frame_sample(JOURNEY_WITH_A_SILENT_FRAME, 8))
    assert {"E06", "E07", "E08"} <= set(refs), f"a silent login repeat took a seat: {refs}"
    assert "E01" not in refs, "the oldest repeat is the one that should give way"
