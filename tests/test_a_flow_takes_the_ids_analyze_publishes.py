"""Two papercuts found while authoring the first real flow from a scenario spec.

Neither is a wrong answer — both are the tool being right in a way that teaches nothing, which on a
suite of 200 scenarios is paid 200 times.

**`id:` means two different things.** ``aua analyze`` publishes a node's selector as
``"id": "rid:navBarPrimary"``. A flow's ``id:`` expects the resource id *bare*. Pasting across
produced ``element_not_found`` with no mention of the prefix, and the fix is invisible: the flow
looks correct, the control is on screen, and the run fails.

**The recorder refuses a scroll and does not say what to do instead.** ``recorded_step_blockers``
correctly declines to advertise a lossy capture — its own docstring notes that *authored* YAML may
use those steps, because an author supplies the arguments the journal lost. But the message stops at
"scroll capture omits container/pages/end-condition/percentage", so a reader learns the recording
failed and not that writing the step by hand is the supported path. Since almost every real journey
scrolls, that is the first wall anyone meets.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml, recorded_step_blockers
from android_ui_analyser.memory import RouteStep


def _one_step(step_yaml: str) -> RouteStep:
    flow = parse_flow_yaml(
        f"name: t\napp: com.example.app\nsteps:\n{step_yaml}", name="t"
    )
    return flow.steps[0]


# --------------------------------------------------------------------- the published id round-trips


@pytest.mark.parametrize(
    "step_yaml",
    [
        "  - tap:\n      id: rid:navBarPrimary",
        "  - long_press:\n      id: rid:navBarPrimary",
        "  - input:\n      id: rid:navBarPrimary\n      text: hello",
        "  - assert:\n      id: rid:navBarPrimary\n      count: 1",
        "  - assert_visible:\n      id: rid:navBarPrimary",
    ],
)
def test_the_prefixed_id_analyze_publishes_is_accepted(step_yaml: str) -> None:
    """What `analyze` hands you must work when pasted into a flow.

    The alternative — failing with `element_not_found` — is the worst shape of error available: the
    flow reads correctly and the control really is on screen.
    """

    step = _one_step(step_yaml)
    # `assert_visible`/`wait_for`/`scroll_to` keep their target in `arg` with `by` naming the field;
    # the element kinds keep it in `resource_id`. Either way the prefix must be gone.
    assert (step.resource_id or step.arg) == "navBarPrimary"


def test_the_bare_resource_id_still_works() -> None:
    """Every flow authored before this accepted the bare form and must keep doing so."""

    assert _one_step("  - tap:\n      id: navBarPrimary").resource_id == "navBarPrimary"


def test_a_resource_id_that_merely_contains_a_colon_is_left_alone() -> None:
    """Android's fully-qualified form carries a colon of its own and is not a published prefix."""

    step = _one_step("  - tap:\n      id: com.example.app:id/navBarPrimary")
    assert step.resource_id == "com.example.app:id/navBarPrimary"


@pytest.mark.parametrize("prefix", ["tx", "cd", "geo", "px"])
def test_the_content_hashed_ids_are_refused_with_the_field_to_use_instead(prefix: str) -> None:
    """`tx:9db2c18ecb` is a hash of the visible text, not a selector anything can resolve later.

    It is published so an agent can act on *this* frame, and it is meaningless in a saved flow. Since
    it appears in `analyze` output right beside `rid:`, refusing it by name — and saying which field
    to use — is the difference between a two-minute fix and a puzzle.
    """

    with pytest.raises(UsageError) as caught:
        _one_step(f"  - tap:\n      id: {prefix}:9db2c18ecb")
    message = f"{caught.value}\n{caught.value.hint or ''}"
    assert prefix in message
    assert "text:" in message or "desc:" in message


# --------------------------------------------------------------------- the recorder says what to do


def test_a_lossy_capture_names_the_step_to_author_by_hand() -> None:
    """The recorder is right to refuse and wrong to stop there.

    Almost every real journey scrolls, so this is the first wall anyone meets when saving a recorded
    flow — and the supported path, authoring the step with its arguments, is already documented in
    `recorded_step_blockers`' own docstring and nowhere the caller can see.
    """

    blockers = recorded_step_blockers(
        [RouteStep(kind="scroll", arg="up"), RouteStep(kind="swipe", arg="up")]
    )
    joined = " ".join(blockers).lower()
    assert blockers, "a lossy capture must still be refused"
    assert "author" in joined or "by hand" in joined, (
        f"no route forward offered: {blockers}"
    )
