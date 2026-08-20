"""A relative path given to `flow run` must mean what the caller meant by it.

Observed: `aua flow run --file <relative-path>` reported the file missing even though the
invoking `cwd` plainly contained it, while an absolute path worked every time — including
for a `flags_apply:` reference *inside* the flow, which had to be rewritten to an absolute
path to work at all.

The mechanism is `_route`: when the warm daemon is live the call is executed *there*, and
the daemon's working directory is wherever it was started. So the relative path was
resolved against a directory the caller had never seen. That is also why an absolute path
always worked — the shape of the bug names its own cause.

Two different resolutions, because two different things are being named:

- a path the **caller typed** belongs to the caller's working directory, so it is made
  absolute in the invoking process, before dispatch can move it;
- a path **inside a flow** belongs to the flow, so it resolves against the flow file's own
  directory — which also makes a checked-in flow directory portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli as cli_mod
from android_ui_analyser.cli import app
from android_ui_analyser.flows import anchor_paths, parse_flow_yaml, resolve_params
from android_ui_analyser.memory import RouteStep

runner = CliRunner()

_FLOW_WITH_FLAGS = """
name: guest_setup
app: com.example.app
steps:
  - flags_apply: flags/guest.yaml
  - tap: "Apps"
"""


# ------------------------------------------------------- in-flow paths, flow-relative


def test_a_flows_own_relative_path_resolves_next_to_the_flow(tmp_path: Path) -> None:
    """`flags_apply: flags/guest.yaml` means "next to me", not "next to the caller"."""
    flow_dir = tmp_path / "flows"
    flow_dir.mkdir()
    (flow_dir / "guest_setup.yaml").write_text(_FLOW_WITH_FLAGS, encoding="utf-8")

    flow = parse_flow_yaml(_FLOW_WITH_FLAGS)
    steps = anchor_paths(resolve_params(flow, {}), flow_dir)

    assert steps[0].arg == str(flow_dir / "flags" / "guest.yaml")
    assert Path(steps[0].arg).is_absolute()
    assert steps[1].label == "Apps"  # a label is not a path; left alone


def test_an_absolute_in_flow_path_is_left_exactly_as_written(tmp_path: Path) -> None:
    steps = anchor_paths(
        [RouteStep(kind="flags-apply", arg="/etc/aua/flags.yaml")], tmp_path / "flows"
    )
    assert steps[0].arg == "/etc/aua/flags.yaml"


def test_name_carrying_steps_are_never_treated_as_paths(tmp_path: Path) -> None:
    """`mock_replay`, `flow` and `dev_profile` take names — anchoring them would break them."""
    steps = anchor_paths(
        [
            RouteStep(kind="mock-replay", arg="empty_inbox"),
            RouteStep(kind="flow", arg="other_flow"),
            RouteStep(kind="dev-profile", arg="ac"),
            RouteStep(kind="goto", arg="home"),
        ],
        tmp_path / "flows",
    )
    assert [s.arg for s in steps] == ["empty_inbox", "other_flow", "ac", "home"]


def test_a_path_nested_in_a_repeat_block_is_anchored_too(tmp_path: Path) -> None:
    """Composite blocks carry substeps; a path inside one is still a path."""
    steps = anchor_paths(
        [
            RouteStep(
                kind="repeat",
                repeat=2,
                substeps=[
                    RouteStep(kind="flags-apply", arg="flags/a.yaml"),
                    RouteStep(kind="tap", label="Go"),
                ],
            )
        ],
        tmp_path / "flows",
    )
    assert steps[0].substeps[0].arg == str(tmp_path / "flows" / "flags" / "a.yaml")
    assert steps[0].substeps[1].label == "Go"


def test_anchoring_happens_after_param_substitution(tmp_path: Path) -> None:
    """`${DIR}/flags.yaml` must anchor the *value*, not the placeholder text."""
    flow = parse_flow_yaml(
        """
name: p
params:
  DIR: ""
steps:
  - flags_apply: "${DIR}/guest.yaml"
"""
    )
    steps = anchor_paths(resolve_params(flow, {"DIR": "profiles"}), tmp_path)
    assert steps[0].arg == str(tmp_path / "profiles" / "guest.yaml")


# --------------------------------------------------- caller paths, cwd-relative at the CLI


def test_the_cli_makes_a_relative_file_absolute_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix that matters in production: the daemon must never see a relative path.

    `_route` may execute this in the daemon, whose cwd is not the caller's, so resolving
    inside the engine would be too late.
    """
    (tmp_path / "journey.yaml").write_text(_FLOW_WITH_FLAGS, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    seen: dict[str, Any] = {}

    def _capture(engine: Any, method: str, **kwargs: Any) -> Any:
        seen.update(kwargs, method=method)
        return {"ok": True, "flow": "guest_setup"}

    monkeypatch.setattr(cli_mod, "_route", _capture)

    result = runner.invoke(app, ["flow", "run", "--file", "journey.yaml", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert seen["method"] == "flow_run"
    assert seen["file"] == str((tmp_path / "journey.yaml").resolve())


def test_a_missing_file_names_the_absolute_place_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Echoing the relative path back is what hid this bug for a whole sweep.

    "no flow file at flows/x.yaml" reads as a typo. The absolute form says plainly that the
    lookup happened in a directory the caller never chose.
    """
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.errors import UsageError
    from conftest import FakeDevice, make_config

    monkeypatch.chdir(tmp_path)
    eng = Engine(make_config(memory={"dir": str(tmp_path / "mem")}), device=FakeDevice())

    with pytest.raises(UsageError) as caught:
        eng.flow_run(file="nope.yaml")

    assert str(tmp_path / "nope.yaml") in str(caught.value)
