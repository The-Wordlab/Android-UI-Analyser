"""``assert_not_visible: {id: …}`` must parse, like its positive twin.

Regression guard: the resource-id branch listed scroll-to/wait-for/assert-visible but not
assert-not-visible, so `assert_not_visible: {id: someId}` was rejected — with an error whose
own text offered `id:` as the fix. Asserting an id is ABSENT is not a niche case: it is how
you check a tab that drops its resource-id once selected, or an entry point that must not be
offered on a particular screen.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import AuaError
from android_ui_analyser.flows import parse_flow_yaml

BY_ID = {"assert_visible", "assert_not_visible", "wait_for", "scroll_to"}


@pytest.mark.parametrize("kind", sorted(BY_ID))
def test_id_selector_parses_for_every_kind_that_advertises_it(kind: str) -> None:
    flow = parse_flow_yaml(f'name: t\nsteps:\n  - {kind}: {{id: containerDetail}}\n')
    (step,) = flow.steps
    assert step.arg == "containerDetail"
    assert step.by == "id", f"{kind} took the id but did not mark by=id, so it matches as text"


@pytest.mark.parametrize("kind", sorted(BY_ID))
def test_neither_selector_is_a_usage_error(kind: str) -> None:
    with pytest.raises(AuaError) as err:
        parse_flow_yaml(f'name: t\nsteps:\n  - {kind}: {{}}\n')
    # The message must not offer `id:` from a branch that cannot read it.
    assert "id:" in str(err.value)
