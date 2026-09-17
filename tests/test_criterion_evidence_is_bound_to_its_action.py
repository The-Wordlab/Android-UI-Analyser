"""A criterion about an action must be judged on the frame that action produced.

2026-09-17: a run where back from a deeplink-opened screen went to Home was passed 8 of 8, because
the controller recovered by tapping the Tools tab and that tap's frame showed the grid the criterion
described. Both judges cited it. The controller's recovery manufactured the evidence that hid the
defect (`docs/aua-deferred-fixes.md` item 92, measured at one run in three).
"""

import inspect

from experiments.aua_controller import judgement


class TestCausalAttribution:
    def test_the_judge_is_told_to_use_the_frame_the_action_produced(self) -> None:
        source = inspect.getsource(judgement.judge_outcome)
        assert "evidence_position.after_step is that action's step" in source

    def test_it_names_the_recovery_trap_explicitly(self) -> None:
        """The rule only helps if it says why a later matching frame is not proof."""
        source = inspect.getsource(judgement.judge_outcome)
        assert "recovers from a failure by navigating to the expected place" in source
        assert "does not verify it if a different action produced that frame" in source

    def test_an_unattributable_action_is_not_verified_rather_than_assumed(self) -> None:
        source = inspect.getsource(judgement.judge_outcome)
        assert "mark it not_verified rather than assuming" in source

    def test_the_binding_data_the_rule_relies_on_is_actually_sent(self) -> None:
        """The instruction is only actionable because every frame carries its step."""
        source = inspect.getsource(judgement)
        assert '"after_step"' in source and '"after_tool"' in source
        assert "evidence_position" in source
