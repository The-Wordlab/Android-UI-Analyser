from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.action_evidence import judge_action_history, resolved_action_target


def frame(label):
    return {"screen": {"width": 100, "height": 200},
            "meta": {"fingerprint": "screen-" + label},
            "elements": [{"id": "el:reused", "text": label, "type": "Button",
                          "resource_id": "menu_action", "bounds": [0, 0, 80, 40]}]}


def test_reused_handle_resolves_from_each_preceding_frame_in_action_order():
    initial = frame("Pin")
    records = [{"step": step, "tool": "tap_and_analyze", "arguments": {"id": "el:reused"},
                "executed": True, "evidence_ref": f"E{step + 1:04d}",
                "resolved_target": {"text": "untrusted replacement"}, "result": frame(next_label)}
               for step, next_label in enumerate(("Unpin", "Save", "Done"))]
    original = copy.deepcopy(records)
    actions = judge_action_history(records, initial)
    assert [action["step"] for action in actions] == [0, 1, 2]
    assert [action["resolved_target"]["text"] for action in actions] == ["Pin", "Unpin", "Save"]
    assert [action["resolved_target"]["source_evidence_ref"] for action in actions] == ["E0000", "E0001", "E0002"]
    assert all(action["arguments"] == {"id": "el:reused"} for action in actions)
    assert records == original


@pytest.mark.parametrize("case", ["stale", "missing", "duplicate", "receipt", "wrong_tool"])
def test_unverified_target_is_never_guessed(case):
    previous = frame("Pin")
    tool = "tap_and_analyze"
    if case == "stale":
        previous["stale"] = True
    elif case == "missing":
        previous["elements"] = []
    elif case == "duplicate":
        previous["elements"].append(copy.deepcopy(previous["elements"][0]))
    elif case == "receipt":
        previous = {"ok": True}
    else:
        tool = "session_finish"
    assert resolved_action_target(tool, {"id": "el:reused", "text": "model claim"}, previous) is None


def test_receipt_breaks_binding_and_unexecuted_actions_are_not_reported():
    records = [
        {"step": 0, "tool": "read", "executed": False, "result": {"ok": False}},
        {"step": 1, "tool": "tap", "executed": True, "arguments": {"id": "el:reused"},
         "result": frame("Save")},
    ]
    actions = judge_action_history(records, frame("Pin"))
    assert [action["step"] for action in actions] == [1]
    assert "resolved_target" not in actions[0]


def test_password_text_is_not_copied_and_semantics_are_bounded():
    previous = frame("secret")
    previous["elements"][0].update({"password": True, "content_desc": "x" * 500})
    target = resolved_action_target("input", {"id": "el:reused"}, previous)
    assert "text" not in target
    assert len(target["content_desc"]) == 200
