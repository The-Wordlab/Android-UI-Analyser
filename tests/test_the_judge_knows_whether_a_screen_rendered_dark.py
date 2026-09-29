"""Every judged frame says whether its screenshot renders dark or light.

A theme run judged eleven appearance bullets from four images; the other frames reached the judge
as element text, which cannot say whether a screen is dark, so both votes left the dark-mode,
immediate-change and navigation bullets unverified (settings-app-theme, 2026-09-28).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.aua_controller.judgement import (
    annotate_judge_frames,
    judge_story,
    rendered_appearance,
)
from PIL import Image


def _png(tmp_path: Path, name: str, grey: int) -> Path:
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (720, 1280), (grey, grey, grey)).save(path)
    return path


def test_a_dark_and_a_light_screenshot_read_as_what_they_are(tmp_path: Path) -> None:
    dark = rendered_appearance(_png(tmp_path, "dark", 18))
    light = rendered_appearance(_png(tmp_path, "light", 245))
    assert dark == {"luminance": 0.07, "reads_as": "dark"}
    assert light == {"luminance": 0.96, "reads_as": "light"}
    assert rendered_appearance(_png(tmp_path, "grey", 128))["reads_as"] == "mixed"


def test_an_unreadable_screenshot_says_nothing(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    assert rendered_appearance(broken) is None
    assert rendered_appearance(tmp_path / "missing.png") is None


def test_the_story_carries_the_measure_only_where_it_was_taken() -> None:
    raw = {"ok": True, "observation": {"screen": {"package": "example.app"}, "meta": {"fingerprint": "f1"},
                                       "elements": [{"id": "el:t", "text": "Theme"}]}}
    frames = annotate_judge_frames([
        {"tool": "tap_and_analyze", "ref": "E0", "step": 0, "raw": copy.deepcopy(raw)},
        {"tool": "tap_and_analyze", "ref": "E1", "step": 1, "raw": copy.deepcopy(raw)},
    ])
    frames[0]["_judge_evidence"]["rendered"] = {"luminance": 0.08, "reads_as": "dark"}
    story = judge_story(frames, [{"step": 0, "tool": "tap_and_analyze", "arguments": {"text": "Dark Mode"}},
                                 {"step": 1, "tool": "tap_and_analyze", "arguments": {"text": "Theme"}}])
    assert story[0]["rendered"] == {"luminance": 0.08, "reads_as": "dark"}
    assert "rendered" not in story[1]


def test_a_frame_is_paired_with_its_own_screenshot_not_its_twins(tmp_path: Path) -> None:
    from experiments.aua_controller.judgement import frame_screenshot

    light, dark = _png(tmp_path, "light-home", 245), _png(tmp_path, "dark-home", 18)
    index = {"samefingerprint": str(dark)}  # the index keeps one image per element tree
    frame = {"observation": {"meta": {"fingerprint": "samefingerprint", "raw_image": str(light)}}}
    assert frame_screenshot(frame, index) == str(light)
    assert rendered_appearance(frame_screenshot(frame, index))["reads_as"] == "light"
    gone = {"observation": {"meta": {"fingerprint": "samefingerprint", "raw_image": str(tmp_path / "pruned.png")}}}
    assert frame_screenshot(gone, index) == str(dark), "a pruned capture falls back to the index"


def test_a_pruned_capture_is_paired_with_the_evidence_copy_of_itself(tmp_path: Path) -> None:
    """Live, the run cache had rotated the raw captures away before the judge ran, so every frame of
    the Chats home fell back to the first image of that element tree -- a dark one. The light home
    after Light Mode reached the judge as dark, and both judges failed a theme that had switched."""
    from experiments.aua_controller.judgement import frame_screenshot, screenshot_index

    dark, light = _png(tmp_path, "007-samefingerprint", 18), _png(tmp_path, "012-samefingerprint", 245)
    entries = [{"evidence_id": "s:observation:samefingerprint", "screenshot": str(shot),
                "observation_contract": {"image_path": str(tmp_path / raw)}}
               for shot, raw in ((dark, "cache-1.png"), (light, "cache-2.png"))]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    index = screenshot_index(manifest)
    light_home = {"observation": {"meta": {"fingerprint": "samefingerprint",
                                           "raw_image": str(tmp_path / "cache-2.png")}}}
    assert frame_screenshot(light_home, index) == str(light)
    assert rendered_appearance(frame_screenshot(light_home, index))["reads_as"] == "light"
