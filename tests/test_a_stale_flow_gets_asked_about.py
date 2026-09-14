"""A setup flow that no longer matches the app is a question, not a silent workaround.

The controller finishing a precondition a stale flow could not is the right outcome for one
run and the wrong one for every run after it: the next one pays for the same divergence, and
the flow keeps drifting. Measured on 2026-09-14 - a guest-entry flow waited on an arrival
marker the build no longer publishes, and three consecutive runs each spent ~25s of retries
working around it before anyone looked.

AUA cannot answer the question itself. "The step did not land" is what a stale flow and a
broken feature both look like, and only the agent that wrote the change knows which. So it
asks, with the evidence attached, and writes nothing: a flow is replayed by every later run,
and the lessons already record two sessions that concluded a marker was gone when it was not.
"""

from __future__ import annotations

from android_ui_analyser.prepare import flow_repair

DIVERGENCE = {
    "setup_flow": 0,
    "flow": "enter_app_as_guest",
    "code": "wait_timeout",
    "step_index": 4,
    "step": "wait-for 'containerDetail'",
    "reached_screen": "chat__52b9b373",
    "remaining_steps": ["wait-for 'containerDetail'", "wait-stable"],
    "markers_on_the_screen_reached": ["rid:buttonBack", "rid:composerField", "rid:detailTitle"],
}


class TestTheQuestion:
    def test_it_asks_whether_the_change_was_intended(self) -> None:
        repair = flow_repair(DIVERGENCE)

        assert "on purpose" in repair["ask"]
        assert "containerDetail" in repair["ask"]
        assert "chat__52b9b373" in repair["ask"]

    def test_both_answers_are_spelled_out(self) -> None:
        """An agent told only "this diverged" fixes the flow to make the red go away."""
        repair = flow_repair(DIVERGENCE)

        assert "update the flow" in repair["if_it_changed_on_purpose"]
        assert "leave the flow alone" in repair["if_it_did_not"]
        assert "defect" in repair["if_it_did_not"]

    def test_it_hands_over_what_the_new_screen_does_publish(self) -> None:
        """Without these the agent has to spend a device run rediscovering the replacement."""
        repair = flow_repair(DIVERGENCE)

        assert repair["markers_on_the_screen_reached"] == [
            "rid:buttonBack", "rid:composerField", "rid:detailTitle"
        ]
        assert repair["stopped_at_step"] == 4

    def test_it_names_the_file_to_edit_when_one_is_known(self) -> None:
        assert flow_repair(DIVERGENCE, flow_path="flows/common/enter.yaml")["flow"] == (
            "flows/common/enter.yaml"
        )
        assert flow_repair(DIVERGENCE)["flow"] == "enter_app_as_guest"

    def test_it_proposes_and_never_writes(self) -> None:
        """A shared flow rewritten on a guess is worse than one that diverges loudly."""
        repair = flow_repair(DIVERGENCE)

        assert "Confirm the replacement on a device" in repair["if_it_changed_on_purpose"]
        assert set(repair) == {
            "flow", "stopped_at_step", "step", "code", "reached_instead",
            "markers_on_the_screen_reached", "ask", "if_it_changed_on_purpose", "if_it_did_not",
        }

    def test_a_divergence_that_named_nothing_still_asks(self) -> None:
        repair = flow_repair({"code": "element_not_found"})

        assert "did not land" in repair["ask"]
        assert repair["markers_on_the_screen_reached"] == []
