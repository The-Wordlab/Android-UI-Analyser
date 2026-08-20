"""The action-observation contract belongs to actions, and only to them.

`observation_present`/`stable_elements` exist so a caller can branch on one payload without
probing for key existence. That promise is worth keeping *and* worth bounding: `ActionResult`
is also the return type of commands that perform no action at all (`screenshot`,
`record_start`, `record_stop`). Emitting `observation_present: false` there states that the
effect of an action was not observed, when no action was requested - the same confusion
between "not checked" and "checked and absent" that the `verified` tri-state exists to avoid.

`render()` strips only `None`, so a `False`/`[]` default is not a neutral default: it is a
published field. These two tests pin both edges, because the contract has no other coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeDevice
from test_memory import APPS, P, _engine


def test_an_action_publishes_the_contract_fields(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-contract")
    eng = _engine(tmp_path, dev)
    first = eng.analyze(source="hierarchy")
    target = next(e.id for e in first.elements if (e.text or "").startswith("Reports"))

    observed = eng.tap(target, observe=True)
    assert observed.observation_present is True
    assert observed.stable_elements  # ids the caller can carry across ID churn
    payload = json.loads(observed.render())
    assert payload["observation_present"] is True

    # Opting out still answers the question rather than going silent: the caller asked for no
    # read-back, so the honest report is "no observation", not a missing key.
    opted_out = eng.tap(target, observe=False)
    assert opted_out.observation is None
    assert opted_out.observation_present is False


def test_a_non_action_command_publishes_neither(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-contract2")
    eng = _engine(tmp_path, dev)
    out = tmp_path / "shot.png"

    shot = eng.screenshot(str(out))
    assert shot.observation_present is None, "screenshot performs no action to observe"
    assert shot.stable_elements is None
    payload = json.loads(shot.render())
    assert "observation_present" not in payload
    assert "stable_elements" not in payload
