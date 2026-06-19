"""Vision-merge tests (PRD §13.1 AC10): IoU, dedupe, OCR association, ID ordering."""

from __future__ import annotations

import pytest

from android_ui_analyser.merge import iou, merge_vision
from android_ui_analyser.providers.base import DetBox, TextBox
from android_ui_analyser.schema import Source

# --------------------------------------------------------------------------- iou


def test_iou_identical_is_one() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap_is_zero() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # edge-touching counts as no overlap (zero-area intersection)
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap_known_value() -> None:
    # two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150 -> 1/3
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_contained_box_known_value() -> None:
    # inner 5x5 fully inside outer 10x10: inter=25, union=100 -> 0.25
    assert iou((0, 0, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.25)


def test_iou_symmetric() -> None:
    a, b = (0, 0, 8, 8), (4, 4, 12, 12)
    assert iou(a, b) == iou(b, a)


def test_iou_degenerate_box_is_zero() -> None:
    assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


# --------------------------------------------------------------------------- dedupe


def test_dedupe_overlapping_keeps_higher_confidence() -> None:
    dets = [
        DetBox(bounds=(0, 0, 100, 100), label="btn", confidence=0.6),
        DetBox(bounds=(2, 2, 102, 102), label="btn", confidence=0.9),  # ~same box, better conf
    ]
    els = merge_vision(dets, [], iou_threshold=0.5)
    assert len(els) == 1
    assert els[0].confidence == pytest.approx(0.9)
    assert els[0].bounds == (2, 2, 102, 102)


def test_dedupe_ties_break_on_larger_area() -> None:
    dets = [
        DetBox(bounds=(0, 0, 50, 50), confidence=None),
        DetBox(bounds=(0, 0, 60, 60), confidence=None),  # overlaps, larger
    ]
    els = merge_vision(dets, [], iou_threshold=0.3)
    assert len(els) == 1
    assert els[0].bounds == (0, 0, 60, 60)


def test_no_dedupe_when_below_threshold() -> None:
    dets = [
        DetBox(bounds=(0, 0, 10, 10)),
        DetBox(bounds=(5, 0, 15, 10)),  # IoU 1/3, below 0.5
    ]
    els = merge_vision(dets, [], iou_threshold=0.5)
    assert len(els) == 2


# --------------------------------------------------------------------------- OCR association


def test_ocr_text_inside_box_becomes_label() -> None:
    dets = [DetBox(bounds=(0, 0, 200, 80), label="button", interactable=True, confidence=0.8)]
    texts = [TextBox(text="Submit", bounds=(20, 20, 120, 60), confidence=0.95)]
    els = merge_vision(dets, texts)
    assert len(els) == 1  # text absorbed into the detection
    el = els[0]
    assert el.text == "Submit"
    assert el.source is Source.detection
    assert el.clickable is True


def test_ocr_text_outside_box_is_standalone() -> None:
    dets = [DetBox(bounds=(0, 0, 80, 80), label="icon")]
    texts = [TextBox(text="Caption", bounds=(300, 300, 460, 340), confidence=0.9)]
    els = merge_vision(dets, texts)
    assert len(els) == 2
    standalone = next(e for e in els if e.source is Source.ocr)
    assert standalone.type == "Text"
    assert standalone.text == "Caption"
    assert standalone.clickable is False
    assert standalone.confidence == pytest.approx(0.9)


def test_ocr_partially_inside_below_threshold_stays_standalone() -> None:
    # box covers only ~25% of the text's area -> not associated
    dets = [DetBox(bounds=(0, 0, 50, 100))]
    texts = [TextBox(text="wide", bounds=(0, 0, 200, 100))]
    els = merge_vision(dets, texts)
    assert {e.source for e in els} == {Source.detection, Source.ocr}


def test_ocr_picks_best_containing_box() -> None:
    # text is fully inside the small box and partially in the big one -> attach to small
    dets = [
        DetBox(bounds=(0, 0, 400, 400), label="container", confidence=0.5),
        DetBox(bounds=(10, 10, 110, 60), label="chip", confidence=0.5),
    ]
    texts = [TextBox(text="Tag", bounds=(20, 20, 100, 50))]
    els = merge_vision(dets, texts, iou_threshold=0.9)  # keep both boxes
    chip = next(e for e in els if e.type == "chip")
    container = next(e for e in els if e.type == "container")
    assert chip.text == "Tag"
    assert container.text == "container"  # falls back to its detection label


def test_detection_without_text_uses_label_then_default_type() -> None:
    els = merge_vision([DetBox(bounds=(0, 0, 10, 10), label=None)], [])
    assert els[0].type == "Element"
    assert els[0].text is None
    assert els[0].source is Source.detection


# --------------------------------------------------------------------------- id ordering


def test_ids_ordered_top_to_bottom_left_to_right() -> None:
    dets = [
        DetBox(bounds=(500, 500, 560, 560)),  # bottom
        DetBox(bounds=(0, 0, 60, 60)),  # top-left
        DetBox(bounds=(200, 0, 260, 60)),  # top-right (same row)
    ]
    els = merge_vision(dets, [])
    assert [e.id for e in els] == [0, 1, 2]
    assert els[0].bounds == (0, 0, 60, 60)
    assert els[1].bounds == (200, 0, 260, 60)
    assert els[2].bounds == (500, 500, 560, 560)


def test_start_id_offset_applied() -> None:
    dets = [DetBox(bounds=(0, 0, 60, 60)), DetBox(bounds=(0, 200, 60, 260))]
    texts = [TextBox(text="x", bounds=(500, 500, 560, 540))]
    els = merge_vision(dets, texts, start_id=10)
    assert [e.id for e in els] == [10, 11, 12]


def test_mixed_sources_and_ordering() -> None:
    dets = [DetBox(bounds=(0, 300, 100, 360), label="b")]
    texts = [TextBox(text="header", bounds=(0, 0, 200, 40))]
    els = merge_vision(dets, texts)
    # header (y=0) before detection (y=300)
    assert els[0].source is Source.ocr
    assert els[1].source is Source.detection
    assert [e.id for e in els] == [0, 1]


def test_empty_inputs_return_empty() -> None:
    assert merge_vision([], []) == []


def test_detection_interactable_false_not_clickable() -> None:
    els = merge_vision([DetBox(bounds=(0, 0, 10, 10), interactable=False)], [])
    assert els[0].clickable is False
