"""A prepared contract has to be able to decide the run, not just inform the judge.

`aua prepare` exists to turn an agent's answers into checkpoints AUA can check. Measured on a
real run: the contract reached the harness, the harness gave it to the judge as reading
material, and the session fell back to one phase derived from the goal sentence carrying no
assertions. `session_finish` could then never be accepted, so a passing run came back
`model_judgement_v1` / `verified: false` - a model's reading of frames, with the provable
answer sitting unused in the same file.

Three things had to be true, and none of them were:
- the contract reaches `session_start`, not only the judge;
- the model driving has a way to prove an assertion, and is told that proof is what completes
  a checkpoint;
- something asks AUA whether the checkpoints were met, since the model's `session_finish` is
  intercepted as a claim and never reaches the session.
"""

from __future__ import annotations

from experiments.aua_controller.run_realapp import (
    CONTRACT_SYSTEM,
    CONTRACT_TOOL,
    REALAPP_SYSTEM,
    contract_satisfied,
    goal_progress_of,
    realapp_tools,
)

from test_aua_controller_realapp import MCP_SCHEMAS as SCHEMAS


def _names(tools):
    return [tool["function"]["name"] for tool in tools]


class TestTheToolToProveIt:
    def test_a_contract_run_can_assert(self) -> None:
        assert CONTRACT_TOOL in _names(realapp_tools(SCHEMAS, contract=True))

    def test_a_run_with_nothing_to_prove_does_not_carry_the_tool(self) -> None:
        """One more tool is one more way to spend a step on a run that has no checkpoints."""
        assert CONTRACT_TOOL not in _names(realapp_tools(SCHEMAS))

    def test_the_instructions_differ_because_the_authority_differs(self) -> None:
        assert "you decide when the goal is met" in REALAPP_SYSTEM
        assert "you are not the one who decides it is met" in CONTRACT_SYSTEM
        assert CONTRACT_TOOL in CONTRACT_SYSTEM
        assert "session_progress" in CONTRACT_SYSTEM


class TestAskingAuaWhetherItWasMet:
    def test_every_checkpoint_complete_is_the_strong_answer(self) -> None:
        assert contract_satisfied(
            {"ok": True, "goal_progress": {"completed": 2, "total": 2, "done": True}}
        )

    def test_a_half_proven_contract_is_not(self) -> None:
        assert not contract_satisfied(
            {"ok": True, "goal_progress": {"completed": 1, "total": 2, "done": False}}
        )

    def test_counts_at_the_top_level_are_read_too(self) -> None:
        assert contract_satisfied({"ok": True, "completed": 1, "total": 1, "done": True})

    def test_a_failed_call_proves_nothing(self) -> None:
        assert not contract_satisfied({"ok": False, "error": "no session"})

    def test_a_payload_we_do_not_understand_proves_nothing(self) -> None:
        """Reading "probably fine" out of an unknown shape is how a harness invents a verdict."""
        assert not contract_satisfied({"ok": True, "goal_progress": {"state": "looks good"}})
        assert not contract_satisfied(None)
        assert not contract_satisfied({"ok": True, "goal_progress": {"completed": 0, "total": 0,
                                                                    "done": True}})

    def test_the_counts_are_found_wherever_they_arrive(self) -> None:
        assert goal_progress_of({"ok": True, "goal_progress": {"total": 3}}) == {"total": 3}
        assert goal_progress_of({"ok": True, "total": 3})["total"] == 3
        assert goal_progress_of({"ok": True, "detail": "nothing useful"}) is None
