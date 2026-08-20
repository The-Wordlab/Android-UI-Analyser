"""A flow must not silently destroy the effect of its own deeplink step.

Reproduced 2026-08-19: a reset flow ran ``open_link "scheme://set-flags?a=1&b=2"`` and then
``launch_app`` / ``stop_app`` on the app that received it. ``flow run`` answered ``ok=True``
with the open-link step "passing" in 1887ms, yet the shared-prefs file the handler writes did
not exist afterwards — the same URI applied standalone writes both values. Delivering an
intent returns when it is *delivered*, never when its work is durable, so the lifecycle step
tore the receiving process down before the handler's asynchronous commit flushed. Both steps
reported success, which is why nothing pointed at the deeplink.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.errors import UsageError
from test_memory import HOME, P, _engine
from test_navigation import ScriptedDevice

# The hazard: nothing between the hand-off and the restart of the process that received it.
TORN_DOWN = f"""
name: reset_state
app: {P}
steps:
  - open_link: "fiction://set-flags?some_experiment=a&other_experiment=a"
  - launch_app: {P}
  - stop_app: {P}
  - launch_app: {P}
  - wait_for: {{text: "Home", timeout_ms: 5000}}
"""

OBSERVED = f"""
name: reset_state
app: {P}
steps:
  - open_link: "fiction://set-flags?some_experiment=a&other_experiment=a"
  - wait_stable: {{timeout_ms: 5000}}
  - stop_app: {P}
  - launch_app: {P}
"""


def _run(tmp_path: Path, text: str, **kw: object) -> tuple[dict, ScriptedDevice]:
    device = ScriptedDevice([HOME, HOME], package=P, serial="emu-handoff")
    engine = _engine(tmp_path, device)
    path = tmp_path / "reset_state.yaml"
    path.write_text(text, encoding="utf-8")
    return engine.flow_run(file=str(path), **kw), device  # type: ignore[arg-type]


def test_a_deeplink_torn_down_before_anything_observed_it_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as caught:
        _run(tmp_path, TORN_DOWN)

    message = str(caught.value)
    assert "open_link" in message and "launch_app" in message
    assert "flow step 1" in message  # the hand-off, 1-based like every other step error


def test_the_refusal_happens_before_the_device_is_touched(tmp_path: Path) -> None:
    device = ScriptedDevice([HOME, HOME], package=P, serial="emu-handoff-clean")
    engine = _engine(tmp_path, device)
    path = tmp_path / "reset_state.yaml"
    path.write_text(TORN_DOWN, encoding="utf-8")

    with pytest.raises(UsageError):
        engine.flow_run(file=str(path))

    # The whole point of catching this in the step order is that no half-applied state is
    # left behind to debug.
    assert not [call for call in device.calls if call[0] in ("open_link", "app_stop")]


def test_dry_run_reports_the_hazard_instead_of_previewing_success(tmp_path: Path) -> None:
    with pytest.raises(UsageError):
        _run(tmp_path, TORN_DOWN, dry_run=True)


def test_a_deeplink_the_flow_waits_on_may_be_followed_by_a_restart(tmp_path: Path) -> None:
    out, device = _run(tmp_path, OBSERVED)

    assert out["ok"] is True, out
    assert ("open_link", ("fiction://set-flags?some_experiment=a&other_experiment=a", P)) in (
        device.calls
    )
    assert ("stop_app", (P,)) in device.calls


def test_a_blind_gesture_does_not_count_as_observing_the_deeplink(tmp_path: Path) -> None:
    # `key` fires and returns; it proves nothing about what the receiving process did.
    text = f"""
name: reset_state
app: {P}
steps:
  - open_link: "fiction://set-flags?some_experiment=a"
  - key: back
  - stop_app: {P}
"""
    with pytest.raises(UsageError) as caught:
        _run(tmp_path, text)
    assert "stop_app" in str(caught.value)


def test_the_hazard_is_caught_inside_a_repeat_block(tmp_path: Path) -> None:
    text = f"""
name: reset_state
app: {P}
steps:
  - repeat:
      times: 2
      steps:
        - open_link: "fiction://set-flags?some_experiment=a"
        - stop_app: {P}
"""
    with pytest.raises(UsageError):
        _run(tmp_path, text)


def test_a_nested_flow_carrying_the_hazard_is_refused_by_its_parent(tmp_path: Path) -> None:
    root = tmp_path / "flows"
    root.mkdir()
    (root / "child.yaml").write_text(
        f"name: child\napp: {P}\nsteps:\n"
        f'  - open_link: "fiction://set-flags?some_experiment=a"\n'
        f"  - stop_app: {P}\n",
        encoding="utf-8",
    )
    parent = root / "parent.yaml"
    parent.write_text(
        f"name: parent\napp: {P}\nsteps:\n  - flow: child.yaml\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME, HOME], package=P, serial="emu-handoff-nested")
    engine = _engine(tmp_path, device)

    with pytest.raises(UsageError):
        engine.flow_run(file=str(parent))
