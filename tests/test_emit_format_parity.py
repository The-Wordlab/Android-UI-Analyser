"""`--format` must mean the same thing whether or not a daemon is serving.

The daemon answers with a plain dict, and `_emit` dumped a dict raw — so the result model's
`render` never ran and the requested format was silently ignored. `--format compact` returned
the full verbose payload whenever a daemon happened to be up: measured 20,532 bytes instead of
9,710 for the same 48-element screen, i.e. ~2x the tokens on the call an agent makes most, with
nothing in the output saying the flag had been dropped.
"""

from __future__ import annotations

import json

from android_ui_analyser import cli
from android_ui_analyser.schema import (
    ActionResult,
    AnalyzeResult,
    Element,
    Meta,
    OutputFormat,
    Screen,
)


def _analyze_result() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package="com.example", source="hierarchy"),
        elements=[
            Element(
                id=0,
                type="TextView",
                text="Hello",
                bounds=[0, 0, 100, 50],
                center=[50, 25],
                stable_key="tx:abc",
                window="app",
            ),
            Element(
                id=1,
                type="View",
                resource_id="btn",
                clickable=True,
                bounds=[0, 60, 100, 110],
                center=[50, 85],
                stable_key="rid:btn",
                window="app",
            ),
        ],
        meta=Meta(duration_ms=12, tier_used="hierarchy", path="hierarchy"),
    )


def _as_daemon_dict(model: AnalyzeResult | ActionResult) -> dict:
    """What the daemon puts on the wire: JSON-roundtripped, no model attached."""
    return json.loads(model.model_dump_json())


def test_analyze_compact_is_identical_through_the_daemon(capsys) -> None:
    model = _analyze_result()
    cli._emit(model, OutputFormat.compact)
    in_process = capsys.readouterr().out
    cli._emit(_as_daemon_dict(model), OutputFormat.compact)
    via_daemon = capsys.readouterr().out
    assert via_daemon == in_process


def test_compact_really_is_smaller_than_json_through_the_daemon(capsys) -> None:
    """Guards the symptom directly: a no-op `compact` would make these equal."""
    payload = _as_daemon_dict(_analyze_result())
    cli._emit(payload, OutputFormat.compact)
    compact = capsys.readouterr().out
    cli._emit(payload, OutputFormat.json)
    verbose = capsys.readouterr().out
    assert len(compact) < len(verbose)


def test_action_result_observation_is_trimmed_through_the_daemon(capsys) -> None:
    """Actions carry an inline observation — the expensive half of an act+observe call."""
    action = ActionResult(ok=True, action="tap", id=1, observation=_analyze_result())
    cli._emit(action, OutputFormat.compact)
    in_process = capsys.readouterr().out
    cli._emit(_as_daemon_dict(action), OutputFormat.compact)
    via_daemon = capsys.readouterr().out
    assert via_daemon == in_process
    assert "clickable" not in json.loads(via_daemon)["observation"]["elements"][0]


def test_an_unrecognised_dict_still_reaches_stdout(capsys) -> None:
    """Rehydration is opportunistic: payloads it cannot type must pass through, not vanish."""
    cli._emit({"ok": True, "custom": [1, 2, 3]}, OutputFormat.compact)
    assert json.loads(capsys.readouterr().out) == {"ok": True, "custom": [1, 2, 3]}
