from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.compaction import FrameCompactor, compact_frame


def element(index, **fields):
    base = {"id": f"el:{index:02x}", "type": "node", "bounds": [0, index * 10, 100, index * 10 + 9],
            "text": "", "desc": None, "clickable": False, "editable": False, "checked": None,
            "enabled": True, "focusable": False, "long_clickable": False, "scrollable": False,
            "selected": False, "password": False, "center": [50, index * 10 + 4], "depth": 3,
            "resource_id": None, "class": "android.view.View", "index": index}
    base.update(fields)
    return base


def frame(fingerprint="fp-1", elements=None, **extra):
    return {"ok": True, "observation": {
        "screen": {"package": "com.example.fictional", "activity": None, "width": 720, "height": 1280,
                   "density": 2.0, "orientation": 0},
        "elements": elements if elements is not None else [
            element(1, text="Settings", clickable=True),
            element(2, resource_id="com.example.fictional:id/row_theme", text="Theme", clickable=True),
            element(3, editable=True, text=""),
            element(4),  # empty container: nothing to show
            element(5, resource_id="com.android.systemui:id/status_bar"),
        ],
        "meta": {"fingerprint": fingerprint, "source": "hierarchy", "elapsed_ms": 41, "cache": "miss",
                 "stale_risk": None, "known_screen": "settings_x", "history_len": 4,
                 "device": {"serial": "emulator-0000"}},
    }, **extra}


def test_compact_frame_keeps_actionable_fields_and_drops_noise():
    raw = frame()
    before = copy.deepcopy(raw)
    out = compact_frame(raw)
    assert raw == before, "raw evidence must never be modified"
    observation = out["observation"]
    assert observation["screen"] == {"package": "com.example.fictional", "activity": None, "width": 720, "height": 1280}
    assert observation["meta"] == {"fingerprint": "fp-1", "stale_risk": None, "known_screen": "settings_x"}
    ids = [item["id"] for item in observation["elements"]]
    assert ids == ["el:01", "el:02", "el:03"], "empty containers and status-bar chrome are dropped"
    assert observation["elements"][0] == {"id": "el:01", "text": "Settings", "clickable": True,
                                          "bounds": [0, 10, 100, 19]}
    assert observation["elements"][1] == {"id": "el:02", "text": "Theme",
                                          "resource_id": "com.example.fictional:id/row_theme", "clickable": True,
                                          "bounds": [0, 20, 100, 29]}
    assert observation["elements"][2] == {"id": "el:03", "editable": True, "bounds": [0, 30, 100, 39]}
    text = json.dumps(out)
    for noise in ("center", "depth", "long_clickable", "elapsed_ms", "history_len", "serial"):
        assert noise not in text


def test_bounds_distinguish_small_unlabelled_child_from_full_width_header():
    raw = frame(elements=[element(1, text="Profile", clickable=True, bounds=[0, 0, 720, 180]),
                          element(2, clickable=True, bounds=[12, 80, 80, 148])])
    compactor = FrameCompactor()
    for _ in range(2):
        elements = compactor(raw)["observation"]["elements"]
        assert [(e["id"], e["bounds"]) for e in elements] == [
            ("el:01", [0, 0, 720, 180]), ("el:02", [12, 80, 80, 148])]


def test_compaction_preserves_input_semantics_without_inventing_editable_buttons():
    raw = frame(elements=[
        {"id": "composer", "type": "EditText", "text": "Message", "resource_id": "message_input",
         "window": "app", "clickable": True},
        {"id": "attach", "type": "Button", "text": "+", "window": "app", "clickable": True},
        {"id": "keyboard", "type": "Text", "text": "Q", "window": "ime"},
        {"id": "send", "type": "Button", "resource_id": "send_message", "clickable": True},
        {"id": "readonly", "type": "EditText", "editable": False, "text": "Read only"},
    ], submitted=False, verified=True)
    before = copy.deepcopy(raw)
    out = compact_frame(raw)
    by_id = {e["id"]: e for e in out["observation"]["elements"]}
    assert by_id["composer"]["editable"] is True
    assert by_id["composer"]["resource_id"] == "message_input"
    assert by_id["composer"]["window"] == "app"
    assert by_id["keyboard"]["window"] == "ime"
    for name in ("attach", "keyboard", "send", "readonly"):
        assert not by_id[name].get("editable")
    assert out["submitted"] is False and out["verified"] is True
    assert raw == before
    repeated = compact_frame(raw, previous_fingerprint="fp-1")
    assert repeated["submitted"] is False
    assert next(e for e in repeated["observation"]["elements"] if e["id"] == "composer")["editable"]


def test_unlabelled_editable_type_survives_compaction_without_clickable_flag():
    out = compact_frame(frame(elements=[{"id": "input", "type": "android.widget.EditText"}]))
    assert out["observation"]["elements"] == [{"id": "input", "editable": True}]


def test_compact_frame_caps_elements_preferring_interactive_ones():
    elements = [element(i, text=f"Label {i}") for i in range(80)]
    elements[70]["clickable"] = True
    out = compact_frame(frame(elements=elements), max_elements=10)
    kept = out["observation"]["elements"]
    assert len(kept) == 10
    assert out["observation"]["elided_elements"] == 70
    assert any(item["id"] == "el:46" and item.get("clickable") for item in kept), "interactive element survives the cap"


def test_compact_frame_truncates_long_text():
    out = compact_frame(frame(elements=[element(1, text="x" * 500)]), max_text=50)
    text = out["observation"]["elements"][0]["text"]
    assert len(text) == 50 and text.endswith("…")


def test_unchanged_fingerprint_collapses_to_interactive_handles():
    out = compact_frame(frame(), previous_fingerprint="fp-1")
    observation = out["observation"]
    assert observation["unchanged"] is True
    assert [item["id"] for item in observation["elements"]] == ["el:01", "el:02", "el:03"]
    assert "Screen unchanged" in observation["note"]
    # An error result on the same fingerprint is not collapsed: the error is new information.
    failed = frame(ok=False, error={"code": "tap_failed", "message": "target vanished"})
    failed_out = compact_frame(failed, previous_fingerprint="fp-1")
    assert failed_out["error"]["code"] == "tap_failed"
    assert "unchanged" not in failed_out["observation"]


def test_goal_progress_and_contract_are_trimmed_and_results_without_frames_pass_through():
    raw = frame(goal_progress={"completed": 1, "total": 3, "done": False, "status": "active",
                               "current": {"id": "p2", "objective": "open theme", "kind": "verify",
                                           "status": "active", "checkpoints": ["a", "b"], "evidence": []},
                               "upcoming": [{"id": "p3", "objective": "finish", "internal": 1}],
                               "ledger": {"long": "x" * 1000}},
                observation_contract={"reusable": True, "analyze_needed": False, "stale_risk": False,
                                      "reason": "long explanation", "evidence_fresh": True})
    out = compact_frame(raw)
    assert out["goal_progress"] == {"completed": 1, "total": 3, "done": False, "status": "active",
                                    "current": {"id": "p2", "objective": "open theme", "kind": "verify", "status": "active"},
                                    "upcoming": [{"id": "p3", "objective": "finish"}]}
    assert out["observation_contract"] == {"reusable": True, "analyze_needed": False, "stale_risk": False, "evidence_fresh": True}
    receipt = {"ok": False, "error": {"code": "session_incomplete", "message": "no"}, "finished": False, "terminated": False}
    assert compact_frame(receipt) == receipt
    assert compact_frame("text") == "text"


def test_frame_compactor_tracks_fingerprints_across_a_conversation():
    compactor = FrameCompactor(max_elements=60)
    first = compactor(frame("fp-a"))
    second = compactor(frame("fp-a"))
    third = compactor(frame("fp-b"))
    assert "unchanged" not in first["observation"]
    assert second["observation"]["unchanged"] is True
    assert "unchanged" not in third["observation"]
    assert (compactor.frames_seen, compactor.unchanged_hits, compactor.last_fingerprint) == (3, 1, "fp-b")


def test_judge_view_can_drop_ids():
    out = compact_frame(frame(), keep_ids=False)
    assert all("id" not in item for item in out["observation"]["elements"])
