from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.session_state import (
    SessionState,
    judgement_observation_frame,
    observation_frame,
)


def frame(text="Ready", **meta):
    # AUA's compact representation legitimately omits type/bounds/default flags.
    return {"screen": {"width": 360, "height": 640, "package": "example.fictional"},
            "elements": [{"id": "el:1", "text": text}],
            "meta": {"fingerprint": "source-frame", **meta}}


def state(tmp_path):
    return SessionState(tmp_path / "session.json", "fictional-session", ["Open the catalogue.", "Return home."])


def prime(s):
    s.observe("initial_observation", {}, frame(), "E0000")


def loading_capture():
    return {"observation_present": True,
            "observation_contract": {"reusable": False, "evidence_fresh": False,
                                     "fingerprint": "source-frame"},
            "observation": frame("Working...", arrival_state="loading", stale_risk=True)}


def test_loading_is_assertion_evidence_not_selector_or_checkpoint_authority(tmp_path):
    raw = loading_capture()
    before = copy.deepcopy(raw)
    assert judgement_observation_frame(raw) == raw["observation"]
    assert observation_frame(raw) is None
    assert raw == before
    s = state(tmp_path)
    s.observe("tap_and_analyze", {"id": "old"}, raw, "E0001")
    assert s.context()["current_observation"] is None


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(observation_present=False),
    lambda r: r.update(stale=True),
    lambda r: r.update(fresh=False),
    lambda r: r["observation"]["meta"].update(stale=True),
    lambda r: r["observation"]["meta"].update(fresh=False),
    lambda r: r["observation_contract"].update(stale=True),
    lambda r: r["observation_contract"].update(fingerprint="different"),
    lambda r: r["observation"]["meta"].update(arrival_state="ready"),
    lambda r: r["observation"]["screen"].update(width=0),
    lambda r: r.update(result=frame("Contradiction", fingerprint="other")),
])
def test_judgement_loading_exception_rejects_stale_absent_or_ambiguous_captures(mutation):
    raw = loading_capture()
    mutation(raw)
    assert judgement_observation_frame(raw) is None


def no_effect_capture():
    """A press that changed nothing on screen: AUA marks it not reusable for acting."""
    return {"observation_present": True,
            "observation_contract": {"reusable": False, "analyze_needed": True,
                                     "fingerprint": "source-frame"},
            "observation": frame("Rename item", arrival_state="unconfirmed",
                                 stale_risk="no semantic destination beyond layout movement")}


def test_a_press_that_changed_nothing_is_still_evidence_of_what_was_on_screen(tmp_path):
    """Live, pressing Save on an empty name left the dialog as it was, the one proof that Save
    is disabled while empty. Its capture was dropped from the judge's evidence because it was
    not reusable for acting, and the run was judged unverified."""
    raw = no_effect_capture()
    assert judgement_observation_frame(raw) == raw["observation"]
    assert observation_frame(raw) is None, "still no authority to act on its ids"
    s = state(tmp_path)
    s.observe("tap_and_analyze", {"id": "old"}, raw, "E0001")
    assert s.context()["current_observation"] is None


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(stale=True),
    lambda r: r["observation"]["meta"].update(stale_risk=True),
    lambda r: r["observation_contract"].update(fingerprint="different"),
    lambda r: r["observation"]["meta"].update(arrival_state="ready"),
])
def test_a_no_effect_capture_is_still_refused_when_stale_or_contradictory(mutation):
    raw = no_effect_capture()
    mutation(raw)
    assert judgement_observation_frame(raw) is None


def test_flow_summary_uses_full_nested_observation_without_another_capture():
    current = frame()
    flow = {"ok": True, "elements": [{"id": "el:1", "label": "Ready"}], "observation": current}
    assert observation_frame(flow) == current
    flow["observation"]["meta"]["stale_risk"] = True
    assert observation_frame(flow) is None


def test_projected_resource_ids_distinguish_unlabelled_targets(tmp_path):
    s = state(tmp_path)
    current = frame()
    current["elements"] = [{"id": "el:1", "rid": "first", "clickable": True},
                           {"id": "el:2", "rid": "second", "clickable": True}]
    s.observe("initial_observation", {}, current, "E0000")
    for _ in range(3):
        s.observe("tap_and_analyze", {"id": "el:1"}, current, "E0001")
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is not None
    assert s.rejection("tap_and_analyze", {"id": "el:2"}) is None


def stall(s, result=None):
    s.observe("tap_and_analyze", {"id": "el:1"}, result or {"ok": True, "observation": frame()}, "E0001")


def test_reload_keeps_claims_knowledge_phase_and_history(tmp_path):
    s = state(tmp_path)
    s.begin_phase("first")
    prime(s)
    s.set_knowledge({"routes": ["Use an observed catalogue entry"]})
    s.set_knowledge({"scope": "fictional build"})
    assert s.update_checks([{"id": "C001", "status": "claimed_verified", "evidence_refs": ["E0000"]}])["ok"]
    stall(s)
    original = s.context()
    restarted = state(tmp_path)
    assert restarted.context() == original
    assert original["phase_id"] == "first"
    assert original["checks"][1]["status"] == "pending"
    assert original["knowledge"] == {"routes": ["Use an observed catalogue entry"], "scope": "fictional build"}
    assert "verdict" not in original and original["checks_are_untrusted_claims"] is True


@pytest.mark.parametrize("updates", [
    [{"id": "C001", "status": "verified"}],
    [{"id": "C001", "status": "claimed_verified"}, {"id": "C999", "status": "not_verified"}],
    [{"id": "C001", "status": "claimed_verified"}, {"id": "C001", "status": "pending"}],
    [{"id": ["C001"], "status": "pending"}],
    [{"id": "C001", "status": ["pending"]}],
    [{"id": "C001", "status": "pending", "verdict": "PASS"}],
    [{"id": "C001", "status": "pending", "evidence_refs": "E0001"}],
    [{"id": "C001", "status": "pending", "note": "x" * 513}],
    [], None,
])
def test_invalid_update_batch_is_all_or_none(tmp_path, updates):
    s = state(tmp_path)
    before = s.path.read_bytes()
    result = s.update_checks(updates)
    assert result == {"ok": False, "error": {"code": "invalid_check_updates", "applied": False}}
    assert s.path.read_bytes() == before and not s.history_path.exists()


def test_three_success_stalls_block_fourth_and_meta_noise_is_not_progress(tmp_path):
    s = state(tmp_path)
    prime(s)
    for index in range(3):
        assert s.rejection("tap_and_analyze", {"id": "el:1"}) is None
        stall(s, {"ok": True, "observation": frame(fingerprint=f"fresh-{index}", duration_ms=index)})
    rejection = s.rejection("tap_and_analyze", {"id": "el:1"})
    assert rejection["error"]["executed"] is False and rejection["error"]["attempts"] == 3
    assert s.rejection("tap_and_analyze", {"id": "el:2"}) is None
    assert state(tmp_path).rejection("tap_and_analyze", {"id": "el:1"}) == rejection


def test_explicit_refusals_without_frames_also_count(tmp_path):
    s = state(tmp_path)
    prime(s)
    for _ in range(3):
        stall(s, {"ok": False, "error": {"code": "scope_rejected", "executed": False}})
    assert s.rejection("tap_and_analyze", {"id": "el:1"})["error"]["attempts"] == 3


def test_changed_screen_clears_guard_even_after_an_error(tmp_path):
    s = state(tmp_path)
    prime(s)
    for _ in range(3):
        stall(s)
    s.observe("analyze_screen", {}, {"ok": False, "observation": frame("Catalogue")}, "E0005")
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is None
    assert s.context()["route_attempts"] == []
    stall(s, {"ok": True, "observation": frame("Catalogue")})
    assert s.context()["route_attempts"][0]["count"] == 1


def test_handle_churn_does_not_reset_same_meaningful_action(tmp_path):
    s = state(tmp_path)
    prime(s)
    previous = "el:1"
    for i in range(3):
        next_frame = frame(fingerprint=f"capture-{i}")
        next_frame["elements"][0]["id"] = f"el:fresh-{i}"
        assert s.rejection("tap_and_analyze", {"id": previous}) is None
        s.observe("tap_and_analyze", {"id": previous}, {"ok": True, "observation": next_frame}, f"E{i:04d}")
        previous = next_frame["elements"][0]["id"]
    assert s.rejection("tap_and_analyze", {"id": previous})["error"]["attempts"] == 3


def test_identical_ambiguous_elements_are_not_conflated(tmp_path):
    s = state(tmp_path)
    f = frame()
    f["elements"].append({"id": "el:2", "text": "Ready"})
    s.observe("initial_observation", {}, f)
    for _ in range(3):
        s.observe("tap_and_analyze", {"id": "el:1"}, {"ok": True, "observation": f})
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is not None
    assert s.rejection("tap_and_analyze", {"id": "el:2"}) is None


def test_unknown_action_outcome_does_not_claim_unchanged_screen(tmp_path):
    s = state(tmp_path)
    prime(s)
    for _ in range(3):
        stall(s, {"ok": False, "error": "transport timeout", "action_outcome": "unknown"})
    assert s.context()["current_observation"]["freshness"] == "unobserved_after_action"
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is None
    # Only a fresh confirming observation can make the no-progress guard usable.
    s.observe("analyze_screen", {}, frame(), "E0005")
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is not None


@pytest.mark.parametrize("tool", ["analyze_screen", "has", "capture_evidence", "known_routes",
                                  "wait_and_analyze", "await_and_analyze", "submit_result", "update_checks"])
def test_reads_waits_and_submissions_never_consume_or_trigger_guard(tmp_path, tool):
    s = state(tmp_path)
    prime(s)
    for _ in range(4):
        s.observe(tool, {}, {"ok": False}, "E0001")
    assert s.context()["route_attempts"] == [] and s.rejection(tool, {}) is None
    assert s.context()["current_observation"]["freshness"] == "fresh_returned_frame"


def test_phase_change_resets_attempts_only_once_and_preserves_ledger(tmp_path):
    s = state(tmp_path)
    s.begin_phase("first")
    prime(s)
    s.update_checks([{"id": "C001", "status": "claimed_failed", "note": "No observed transition"}])
    for _ in range(3):
        stall(s)
    assert s.begin_phase("first")["changed"] is False
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is not None
    assert s.begin_phase("second")["changed"] is True
    assert s.rejection("tap_and_analyze", {"id": "el:1"}) is None
    assert s.context()["checks"][0]["status"] == "claimed_failed"
    assert state(tmp_path).context()["phase_id"] == "second"


def test_context_is_bounded_and_archive_contains_only_lightweight_records(tmp_path):
    s = state(tmp_path)
    prime(s)
    for i in range(30):
        s.observe("analyze_screen", {}, {**frame(f"State {i}"), "reasoning": "DO_NOT_ARCHIVE",
                                         "raw_tool_payload": "DO_NOT_ARCHIVE"}, f"E{i:04d}")
    context = s.context()
    assert len(context["recent_history"]) == 12 and context["history_events"] == 31
    events = [json.loads(line) for line in s.history_path.read_text().splitlines()]
    assert len(events) == 31 and [e["event"]["sequence"] for e in events] == list(range(1, 32))
    assert "DO_NOT_ARCHIVE" not in s.history_path.read_text() + s.path.read_text()
    assert "elements" not in context["current_observation"]
    context["checks"][0]["status"] = "tampered"
    assert s.context()["checks"][0]["status"] == "pending"


def test_journal_recovers_a_snapshot_left_behind_after_commit(tmp_path):
    s = state(tmp_path)
    old = s.path.read_bytes()
    prime(s)
    current = s.context()
    s.path.write_bytes(old)
    assert state(tmp_path).context() == current


@pytest.mark.parametrize("session,clauses", [("different", ["Open the catalogue.", "Return home."]),
                                           ("fictional-session", ["Different contract."])])
def test_reload_refuses_different_session_or_contract(tmp_path, session, clauses):
    s = state(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        SessionState(s.path, session, clauses)


@pytest.mark.parametrize("value", [{"reasoning": "not knowledge"}, {"nested": {"elements": []}},
                                   {"messages": []}, {"huge": "x" * 8193}])
def test_knowledge_rejects_raw_or_unbounded_input_without_mutation(tmp_path, value):
    s = state(tmp_path)
    before = s.context()
    with pytest.raises(ValueError):
        s.set_knowledge(value)
    assert s.context() == before


def test_compact_fresh_frame_wrappers_and_input_immutability():
    wrapped = {"ok": False, "observation": frame()}
    old = copy.deepcopy(wrapped)
    assert observation_frame({"data": wrapped}) == frame()
    extracted = observation_frame(wrapped)
    extracted["elements"].clear()
    assert wrapped == old
    assert observation_frame({"observation": frame(), "result": frame()}) == frame()
    assert observation_frame({"ok": False, "error": {"observation_present": True, "observation": frame()}}) == frame()
    assert observation_frame({"error": {"observation": frame()}}) is None


@pytest.mark.parametrize("value", [
    {"observation": frame(), "result": frame("Other")},
    {"observation_present": False, "observation": frame()},
    {"observation": frame(stale_risk=True)},
    {"observation": frame(stale=True)}, {"observation": frame(fresh=False)},
    {"stale": True, "observation": frame()}, {"fresh": False, "observation": frame()},
    {"observation_contract": {"reusable": False}, "observation": frame()},
    {"history": [frame()]}, {"screen": "receipt", "elements": []},
    {**frame(), "meta": {}}, {**frame(), "elements": ["receipt"]},
    {**frame(), "screen": {"width": 0, "height": 640}},
    {**frame(), "elements": [{"id": "el:1", "bounds": [1, 2]}]},
])
def test_incomplete_stale_ambiguous_or_unrelated_frames_are_not_current(value):
    assert observation_frame(value) is None
