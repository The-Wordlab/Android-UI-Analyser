"""Authored journeys retain one session and distinguish assertions from agent judgement."""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser import engine_journeys as journeys
from android_ui_analyser.config import Config
from android_ui_analyser.errors import UsageError
from android_ui_analyser.journeys import load_journey
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import (
    create_session_state,
    finish_session_state,
    load_session_state,
    mark_phase_complete,
)

JOURNEY = """
version: 1
name: Catalog inspection
package: com.example.catalog
constraints: [Do not buy anything]
steps:
  - id: catalog
    instruction: Open the Catalog screen.
    assertions:
      - assert: {text: Catalog, exists: true}
  - id: layout
    instruction: Inspect the layout visually.
    review: Catalog title and Continue button have clear spacing.
"""


def frame(text="Catalog", fingerprint="frame-1", serial="fixture-1"):
    return AnalyzeResult(
        screen=Screen(width=400, height=800, package="com.example.catalog", source="hierarchy"),
        elements=[Element(id=1, text=text, type="TextView", bounds=(0, 0, 200, 40),
                          center=(100, 20), source=Source.hierarchy)],
        meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy",
                  fingerprint=fingerprint, device_serial=serial),
    )


class SessionEngine:
    """Keep real SessionState/proof persistence; replace only native session bootstrap/read."""

    def __init__(self, tmp_path):
        self.config = Config()
        self.config.cache.dir = str(tmp_path)
        self.observation = frame()
        self.session_id = None
        self.reads = 0
        self.cleanup = []
        self.owner = "fixture-agent"

    def _ledger_identity(self):
        return {"owner": self.owner}

    def session_start(self, goal, **kwargs):
        state = create_session_state(
            self.config.cache.dir, goal=goal, serial="fixture-1", owner="fixture-agent",
            recommended_kind="manual", recommended_cli="aua analyze",
            network_backup_preexisting=False, network_profile_preexisting=False,
            platform="fixture",
        )
        self.session_id = state.session_id
        return {"ok": True, "observation": self.observation.model_dump(mode="json")}

    def _session_state(self, session_id=None):
        return load_session_state(self.config.cache.dir, session_id=session_id or self.session_id, platform="fixture")

    def analyze(self, **kwargs):
        assert kwargs["no_cache"] is True
        self.reads += 1
        return self.observation

    def _read_cache(self):
        return self.observation

    def _cache_path(self, serial):
        assert serial == "fixture-1"
        path = Path(self.config.cache.dir) / "fixture-observation.json"
        path.write_text(self.observation.model_dump_json())
        return path

    def session_mark_phase(self, phase_id, evidence, *, session_id=None):
        return mark_phase_complete(self.config.cache.dir, self._session_state(session_id), phase_id=phase_id, evidence=evidence)

    def session_finish(self, session_id, **kwargs):
        self.cleanup.append(kwargs)
        finish_session_state(self.config.cache.dir, self._session_state(session_id))
        return {"ok": True, "finished": True, "verdict": "passed", "cleanup": []}


@pytest.fixture
def engine(tmp_path):
    return SessionEngine(tmp_path)


def test_validation_is_device_free_and_actions_are_not_assertions():
    result = journeys.journey_validate(None, yaml=JOURNEY)
    assert result["assertion_steps"] == 1
    assert result["review_steps"] == 1
    assert result["execution"] == "calling_agent"
    with pytest.raises(UsageError):
        load_journey(yaml_text=JOURNEY.replace("assert: {text: Catalog, exists: true}", "tap: Catalog"))


def test_authored_step_ids_and_order_replace_goal_guessing(engine):
    result = journeys.journey_start(engine, yaml=JOURNEY)
    assert [phase.id for phase in engine._session_state().phases] == ["catalog", "layout"]
    assert result["journey"]["current_step"]["id"] == "catalog"
    assert result["journey"]["current_step"]["assertions"][0]["label"] == "Catalog"
    assert result["journey"]["verified"] is False
    assert engine.reads == 0


def test_failed_assertions_do_not_advance_and_finish_does_not_cleanup(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    engine.observation = frame(text="Wrong screen")
    result = journeys.journey_check(engine)
    assert not result["ok"]
    assert result["journey"]["verdict"] == "failed"
    assert result["journey"]["current_step"]["id"] == "catalog"
    assert journeys.journey_finish(engine)["code"] == "journey_incomplete"
    assert not engine.cleanup


def test_manual_claim_cannot_complete_an_assertion_step(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    with pytest.raises(UsageError, match="manual claims"):
        journeys.journey_record(engine, "Catalog title is visible", "frame-1")


def test_abandonment_finishes_cleanup_without_claiming_completion(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    engine.observation = frame(text="Wrong screen")
    journeys.journey_check(engine)
    result = journeys.journey_finish(engine, allow_incomplete=True)
    assert result["finished"] and result["journey"]["finished"]
    assert result["verdict"] == "incomplete"
    assert not result["verified"]
    assert not result["journey"]["complete"]
    assert result["journey"]["next_call"] is None
    assert result["journey"]["verdict"] == "incomplete"
    assert engine.cleanup
    again = journeys.journey_finish(engine)
    assert again["finished"] and again["verdict"] == "incomplete"
    assert len(engine.cleanup) == 1


def test_reusing_one_frame_cannot_complete_two_assertion_steps(engine):
    authored = JOURNEY.split("  - id: layout")[0]
    authored += "  - id: second\n    instruction: Open Catalog again.\n    assertions:\n      - assert: {text: Catalog, exists: true}\n"
    journeys.journey_start(engine, yaml=authored)
    assert journeys.journey_check(engine)["ok"]
    with pytest.raises(UsageError, match="unchanged observation"):
        journeys.journey_check(engine)
    assert journeys.journey_status(engine)["journey"]["current_step"]["id"] == "second"


@pytest.mark.parametrize("failure", ["stale", "missing_fingerprint", "foreign_target"])
def test_unusable_assertion_observation_never_advances(engine, failure):
    journeys.journey_start(engine, yaml=JOURNEY)
    if failure == "stale":
        engine.observation.meta.stale_risk = "screen is transitioning"
    elif failure == "missing_fingerprint":
        engine.observation.meta.fingerprint = None
    else:
        engine.observation.meta.device_serial = "foreign-target"
    with pytest.raises(UsageError):
        journeys.journey_check(engine)
    assert journeys.journey_status(engine)["journey"]["current_step"]["id"] == "catalog"


def test_foreign_worker_cannot_check_record_or_finish_another_session(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    engine.owner = "foreign-agent"
    for action in (
        lambda: journeys.journey_check(engine),
        lambda: journeys.journey_record(engine, "Catalog title has clear spacing", "frame-1"),
        lambda: journeys.journey_finish(engine, allow_incomplete=True),
    ):
        with pytest.raises(UsageError, match="another worker"):
            action()
    assert engine.reads == 0
    assert not engine.cleanup


def test_mixed_journey_preserves_verified_and_agent_assessed_proofs(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    checked = journeys.journey_check(engine)
    assert checked["ok"]
    assert checked["journey"]["steps"][0]["verified"] is True
    assert checked["journey"]["current_step"]["id"] == "layout"
    engine.observation = frame(fingerprint="review-frame")
    recorded = journeys.journey_record(
        engine, "Catalog title and Continue button have clear spacing and do not overlap.", "review-frame",
    )
    assert recorded["ok"] and not recorded["verified"]
    assert engine.reads == 1  # recording consumes exactly the frame the caller saw
    assert recorded["journey"]["verdict"] == "agent_assessed_pass"
    proof = recorded["journey"]["steps"][1]["proof"]
    assert proof["source"] == "manual_evidence" and proof["verified"] is False
    assert proof["observation"]["fingerprint"] == "review-frame"
    result = journeys.journey_finish(engine, retain_started_target=False)
    assert result["verdict"] == "agent_assessed_pass"
    assert not result["verified"]
    assert engine.cleanup == [{"allow_incomplete": False, "retain_started_target": False}]


def test_assessment_of_old_frame_and_other_device_is_refused(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    journeys.journey_check(engine)
    with pytest.raises(UsageError, match="fingerprint"):
        journeys.journey_record(engine, "Catalog title and Continue button have clear spacing", "old")
    engine.observation = frame(serial="other-device")
    with pytest.raises(UsageError, match="device"):
        journeys.journey_record(engine, "Catalog title and Continue button have clear spacing", "frame-1")


def test_failed_visual_assessment_retains_pending_step(engine):
    journeys.journey_start(engine, yaml=JOURNEY)
    journeys.journey_check(engine)
    result = journeys.journey_record(engine, "Catalog title overlaps Continue button", "frame-1", passed=False)
    assert not result["ok"]
    assert result["journey"]["verdict"] == "failed"
    assert result["journey"]["current_step"]["id"] == "layout"


def test_all_assertion_journey_is_verified_and_cannot_be_changed_after_finish(engine):
    authored = JOURNEY.split("  - id: layout")[0]
    journeys.journey_start(engine, yaml=authored)
    checked = journeys.journey_check(engine)
    assert checked["journey"]["next_call"]["tool"] == "journey_finish"
    result = journeys.journey_finish(engine)
    assert result["verified"]
    assert result["verdict"] == "asserted_pass"
    with pytest.raises(UsageError, match="finished"):
        journeys.journey_check(engine)


def test_action_coaching_preserves_authored_review_and_does_not_infer_another_goal(engine):
    from android_ui_analyser.coaching import decorate_result

    journeys.journey_start(engine, yaml=JOURNEY)
    journeys.journey_check(engine)
    engine.journey_status = lambda session_id: journeys.journey_status(engine, session_id)

    def unexpected_inference(*args, **kwargs):
        pytest.fail("authored journey must not infer a replacement instruction")

    engine.session_progress = unexpected_inference
    result = {"ok": True, "action": "tap", "observation": frame().model_dump(mode="json")}
    decorated = decorate_result(engine, "tap", result)
    progress = decorated["goal_progress"]
    assert progress["next_call"]["tool"] == "journey_record"
    assert progress["journey_step"]["review"] == "Catalog title and Continue button have clear spacing."
    assert progress["checkpoint"] is None


@pytest.mark.parametrize("change", [
    lambda text: text.replace("id: layout", "id: catalog"),
    lambda text: text.replace("version: 1", "version: 2"),
    lambda text: text.replace("    review:", "    unknown:"),
])
def test_invalid_journey_cannot_bootstrap(engine, change):
    with pytest.raises(UsageError):
        journeys.journey_start(engine, yaml=change(JOURNEY))
    assert engine.session_id is None
