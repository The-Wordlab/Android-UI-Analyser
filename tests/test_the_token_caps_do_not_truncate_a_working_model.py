"""The cap should stop a runaway; the budget should stop the spending. They had swapped jobs.

Measured over twelve controller-harness runs on 2026-09-14, all on the same scenario:

  controller, V4 Flash 0731 : n=42  median  388  p90   939  max 2617   never truncated
  controller, V4.1 Flash    : n=56  median  406  p90  1563  max 4096   truncated once
  judge (both)              : n=29                p90  2967  max 3000   pinned to its ceiling

The 4096 controller cap cost run B4 its whole product verdict - nine steps of real device work
thrown away for "model completion truncated" - and the judge had been answering at exactly its
ceiling on nearly every request, which is what a silently clipped answer looks like from outside.

Neither number was chosen; both were inherited defaults. A cap costs nothing when it is not
reached, because tokens are billed as generated, so the only thing a tight cap buys is
truncation. CostGuard is checked before every request and is the real money stop, so it carries
matching headroom over the worst run actually observed.
"""

from __future__ import annotations

import inspect

from experiments.aua_controller import agent_loop, run_realapp

# (metric, observed worst case across the 2026-09-14 runs)
OBSERVED_CONTROLLER_MAX = 4096      # V4.1 Flash, and it truncated there
OBSERVED_JUDGE_MAX = 3000           # the cap it was given, hit exactly
OBSERVED_RUN_COST_USD = 0.02235     # most expensive complete run (B5)
OBSERVED_JUDGE_COST_USD = 0.01295   # most expensive judge (A5)


def default(func, name):
    return inspect.signature(func).parameters[name].default


def test_the_controller_cap_clears_the_output_that_truncated():
    cap = default(run_realapp.run_realapp, "max_tokens")
    assert cap >= OBSERVED_CONTROLLER_MAX * 4, "a cap barely above the observed max truncates again"
    assert default(agent_loop.run_agent, "max_tokens") == cap, "the loop and its runner must agree"


def test_the_judge_cap_clears_the_ceiling_it_was_pinned_to():
    assert default(run_realapp.run_realapp, "judge_max_tokens") >= OBSERVED_JUDGE_MAX * 2


def test_the_budget_carries_headroom_over_the_costliest_real_run():
    assert default(run_realapp.run_realapp, "cost_limit_usd") >= OBSERVED_RUN_COST_USD * 5
    assert default(run_realapp.run_realapp, "judge_cost_limit_usd") >= OBSERVED_JUDGE_COST_USD * 5


def test_a_budget_still_exists_because_a_loose_cap_needs_one():
    # Raising the cap without raising the guard would simply move the failure; removing the
    # guard would remove the only thing that stops a runaway generation costing real money.
    for name in ("cost_limit_usd", "judge_cost_limit_usd"):
        limit = default(run_realapp.run_realapp, name)
        assert 0 < limit < 5, f"{name} must stay a real ceiling, not a formality"


def test_the_cli_defaults_match_the_function_defaults():
    source = inspect.getsource(run_realapp.main)
    for flag, param in (("--max-tokens", "max_tokens"), ("--judge-max-tokens", "judge_max_tokens")):
        expected = f'"{flag}", type=int, default={default(run_realapp.run_realapp, param)}'
        assert expected in source, f"{flag} drifted from its function default"


def test_the_screen_namer_is_not_left_on_the_old_tight_cap():
    # It runs once per frame with reasoning enabled; 512 tokens was a truncation waiting to happen.
    assert "max_tokens=512" not in inspect.getsource(run_realapp.run_realapp)
