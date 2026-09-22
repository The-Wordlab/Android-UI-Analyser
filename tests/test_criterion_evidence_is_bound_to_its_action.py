"""A criterion about an action must be judged on the frame that action produced.

2026-09-17: a run where back from a deeplink-opened screen went to Home was passed 8 of 8, because
the controller recovered by tapping the Tools tab and that tap's frame showed the grid the criterion
described. Both judges cited it. The controller's recovery manufactured the evidence that hid the
defect (`docs/aua-deferred-fixes.md` item 92, measured at one run in three).

The rules are read from the question the judge is actually asked, not from the source text of the
function that builds it: a string literal wrapped onto a new line is the same question.
"""

import inspect

from experiments.aua_controller import judgement

QUESTION = judgement.outcome_question("- Back from the tools screen returns to Home.")


class TestCausalAttribution:
    def test_the_judge_is_told_to_use_the_entry_the_action_produced(self) -> None:
        assert "verify it ONLY from the journey entry whose `step` is that action's" in QUESTION

    def test_it_names_the_recovery_trap_explicitly(self) -> None:
        """The rule only helps if it says why a later matching frame is not proof."""
        assert "recovers from a failure by navigating to the expected place" in QUESTION
        assert "does not verify it if a different action produced that entry" in QUESTION

    def test_an_eventual_outcome_binds_to_the_action_that_completes_it(self) -> None:
        """"X starts work that is finished when you come back" is proved by the return, not by X.

        The first version of this rule failed a criterion reading "the finished result is there on
        return" because the frame from the submit showed work still in progress -- which is what
        that contract says should happen.
        """
        assert "action that COMPLETES it" in QUESTION
        assert "'on return'" in QUESTION
        assert "shows work still in progress" in QUESTION

    def test_it_still_refuses_an_unrelated_later_action(self) -> None:
        """The relaxation must not reopen the hole it was written to close."""
        assert "not to require an outcome before the contract says it arrives" in QUESTION
        assert "later UNRELATED action supplying the proof" in QUESTION

    def test_an_unattributable_action_is_not_verified_rather_than_assumed(self) -> None:
        assert "mark the criterion not_verified rather than assuming" in QUESTION

    def test_a_goal_without_a_contract_still_gets_the_attribution_rule(self) -> None:
        assert "recovers from a failure" in judgement.outcome_question(None)

    def test_the_binding_data_the_rule_relies_on_is_actually_sent(self) -> None:
        """The instruction is only actionable because every entry carries its step."""
        source = inspect.getsource(judgement)
        assert '"after_step"' in source and '"after_tool"' in source
        assert '"step"' in inspect.getsource(judgement.judge_story)
