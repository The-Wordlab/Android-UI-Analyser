"""The judge must be shown the journey it is asked about, with the images it was promised.

Both bugs here were found by running one real scenario (guest cold start) and reading why it
came back BLOCKED. Neither is visible from the outside: the run reports a confident verdict and
a plausible reason, and the reason is *true* for the evidence the judge was handed.

- **Only the tail was judged.** ``frames[-4:-1]`` answered "what happened at the end", but a
  contract bullet is usually about the route -- "the first interactive screen is the
  authentication landing", "taking that option ends on home". Judged from the last three
  observations those are not false, they are unobservable, and the judge correctly recorded
  them as unevidenced. A run that passed was recorded BLOCKED, with notes claiming a screen
  "appears in no frame" that AUA had in fact screenshotted.
- **Vision attached nothing, always.** AUA's ``evidence_id`` ends with a 24-character *prefix*
  of the 40-character observation fingerprint, so the exact dict lookup could never hit. The
  result still said ``frames: 4`` and only ``images_attached: 0`` gave it away, so a vision
  judge had been reasoning from text alone for every run.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.aua_controller.judgement import (
    frame_fingerprint,
    image_frame_sample,
    judged_frame_sample,
    screenshot_for,
    screenshot_index,
)

# The real values from the run that exposed this: a 40-char frame fingerprint whose first 24
# characters are what the evidence id carries.
FULL = "2d5470aeb08700b5052a31eb00a480c74a2bec7b"
PREFIX = FULL[:24]


def _frame(name: str, fingerprint: str | None = None) -> dict:
    frame = {"observation": {"meta": {"screen": name}}}
    if fingerprint:
        frame["observation"]["meta"]["fingerprint"] = fingerprint
    return frame


def _name(frame: dict) -> str:
    return frame["observation"]["meta"]["screen"]


# --------------------------------------------------------------------------- frame selection


def test_a_short_journey_is_judged_whole() -> None:
    frames = [_frame(n) for n in ("launcher", "landing", "profile", "home", "final")]

    picked = judged_frame_sample(frames, limit=8)

    # The last frame is judged separately as `final_frame`, so it is not repeated here.
    assert [_name(f) for f in picked] == ["launcher", "landing", "profile", "home"]


def test_a_long_journey_still_shows_its_beginning() -> None:
    """The bug in one line: the old slice could not answer a question about step two."""
    frames = [_frame(f"s{i:02d}") for i in range(24)]

    picked = judged_frame_sample(frames, limit=8)

    names = [_name(f) for f in picked]
    assert len(names) == 8
    assert names[0] == "s00", "the start of the journey was dropped"
    assert names[-1] == "s22", "the frame before the final observation was dropped"
    assert names == sorted(names), "frames must stay in journey order"
    # Spread, not clustered: the old frames[-4:-1] would have been s20, s21, s22.
    assert len(set(names)) == 8


def test_the_sample_is_bounded_and_degenerate_limits_are_safe() -> None:
    frames = [_frame(f"s{i}") for i in range(10)]

    assert judged_frame_sample(frames, limit=1) == [frames[0]]
    assert judged_frame_sample(frames, limit=0) == frames[:-1]
    assert judged_frame_sample([], limit=8) == []
    assert judged_frame_sample([_frame("only")], limit=8) == []


def test_rendered_image_sample_spans_the_text_journey_without_dropping_its_tail() -> None:
    frames = [_frame(str(index)) for index in range(8)]

    picked = image_frame_sample(frames, limit=3)

    assert picked == [frames[0], frames[4], frames[7]]


# --------------------------------------------------------------------------- image pairing


def test_a_truncated_evidence_id_still_finds_its_screenshot(tmp_path: Path) -> None:
    shot = tmp_path / "003.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "command": "tap_and_analyze",
                        "evidence_id": f"session-abc:observation:{PREFIX}",
                        "screenshot": str(shot),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index = screenshot_index(manifest)
    assert list(index) == [PREFIX], "the index is keyed by the truncated evidence id"

    # This is the lookup that silently returned nothing on every run.
    assert index.get(FULL) is None
    assert screenshot_for(index, FULL) == str(shot)


def test_an_exact_fingerprint_is_preferred_over_a_prefix(tmp_path: Path) -> None:
    exact, prefixed = tmp_path / "exact.png", tmp_path / "prefixed.png"
    for path in (exact, prefixed):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {"evidence_id": f"s:observation:{PREFIX}", "screenshot": str(prefixed)},
                    {"evidence_id": f"s:observation:{FULL}", "screenshot": str(exact)},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert screenshot_for(screenshot_index(manifest), FULL) == str(exact)


def test_an_unknown_or_absent_fingerprint_is_not_guessed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": []}), encoding="utf-8")
    index = screenshot_index(manifest)

    assert screenshot_for(index, FULL) is None
    assert screenshot_for(index, None) is None
    assert screenshot_for({PREFIX: "/tmp/x.png"}, "ffffffffffffffffffffffff") is None


def test_frame_fingerprint_reads_the_full_value_the_frame_carries() -> None:
    assert frame_fingerprint(_frame("landing", FULL)) == FULL
    assert frame_fingerprint(_frame("landing")) is None
