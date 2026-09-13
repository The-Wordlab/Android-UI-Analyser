"""Privacy boundaries with fictional metadata; no devices, models or requests."""

import copy
import json

from experiments.aua_controller.hosted_projection import hosted_model_view


def test_nested_metadata_and_diagnostic_echoes_are_removed_without_mutating_evidence():
    value = {
        "ok": False, "session_id": "session-fictional-123", "serial": "device-fictional-456",
        "owner": {"id": "owner-fictional-789"}, "caller": {"pid": 1234},
        "target": {"id": "target-fictional-012"},
        "artifacts_dir": "/Users/fictional/private/runs/example",
        "app_logs": [{"message": "private application log"}],
        "capture_evidence": {"ref": "private capture reference", "frames": ["private frame"]},
        "observation": {"meta": {"deviceSerial": "device-fictional-456", "fingerprint": "fresh-fp"}},
        "warnings": ["session-fictional-123 on device-fictional-456 owned by owner-fictional-789",
                     "target-fictional-012 failed; see /Users/fictional/private/report.json"],
    }
    original = copy.deepcopy(value)
    result = hosted_model_view(value)
    encoded = json.dumps(result)
    assert value == original
    for private in ["session-fictional-123", "device-fictional-456", "owner-fictional-789",
                    "target-fictional-012", "/Users/fictional", "private application log",
                    "private capture reference", "private frame"]:
        assert private not in encoded
    assert result["ok"] is False
    assert result["observation"]["meta"] == {"fingerprint": "fresh-fp"}
    assert "failed" in result["warnings"][1]


def test_authoritative_handles_ui_semantics_and_progress_remain_exact():
    element = {
        "id": "el:opaque-current-19", "parent": "el:opaque-current-2",
        "resource_id": "dev.aua.fixture:id/compose_name_1", "stable_key": "rid:dev.aua.fixture:id/compose_name_1",
        "text": "Alpha Lantern", "content_desc": "Price ascending", "type": "Button",
        "bounds": [1, 2, 3, 4], "enabled": True, "clickable": True, "editable": False,
        "selected": False, "checked": False, "password": False,
    }
    expected = {
        "observation": {"elements": [element],
                        "screen": {"package": "dev.aua.fixture", "activity": "dev.aua.fixture/.MainActivity"},
                        "meta": {"fingerprint": "fp:current", "stale_risk": False, "unchanged": False}},
        "acting": {"id": element["id"], "relation": "exact", "detail": "clicked"},
        "selector": "rid:dev.aua.fixture:id/compose_name_1",
        "selectors": ["text:Alpha Lantern", "id:el:opaque-current-19"],
        "goal_progress": {"completed": 1, "total": 4, "done": False,
                          "current": {"id": "price_sorted", "objective": "Verify ascending prices", "status": "active"}},
        "observation_contract": {"evidence_fresh": True, "reusable": True, "analyze_needed": False},
        "finished": False, "errors": ["A fresh observation is required"],
    }
    assert hosted_model_view(expected) == expected


def test_serialized_nested_json_cannot_hide_private_metadata():
    nested = {"sessionId": "session-fictional", "observation": {"elements": [{"id": "el:fresh"}]} }
    result = hosted_model_view({"detail": json.dumps(nested),
                                "note": "session-fictional must obtain a fresh observation"})
    assert json.loads(result["detail"]) == {"observation": {"elements": [{"id": "el:fresh"}]}}
    assert "session-fictional" not in result["note"]
    assert "fresh observation" in result["note"]


def test_standalone_diagnostic_labels_and_paths_are_redacted():
    messages = [
        'session_id="fictional-session" serial=fictional-device: stale observation',
        "See file:///Users/fictional/image.png and /private/tmp/fixture/log.txt",
        "Logs: runs/fixture/debug.json or artifacts/fixture/result.json",
        r"Debug file C:\Users\Fictional\private\log.txt",
    ]
    result = hosted_model_view({"errors": messages})
    text = json.dumps(result)
    for private in ["fictional-session", "fictional-device", "Fictional", "fixture/log", "debug.json"]:
        assert private not in text
    assert "stale observation" in text
    assert text.count("private-path") == 5


def test_action_handles_are_not_rewritten_even_when_matching_removed_metadata():
    # Selection correctness must not depend on whether metadata happens to share
    # an opaque value. The projection is for public fixture observations only.
    value = {"session_id": "opaque-handle", "observation": {"elements": [{
        "id": "opaque-handle", "parent": "opaque-handle", "stable_key": "opaque-handle"}]}}
    assert hosted_model_view(value) == {"observation": value["observation"]}
