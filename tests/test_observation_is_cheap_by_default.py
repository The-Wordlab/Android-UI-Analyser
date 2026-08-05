"""The folded post-action observation must be the *cheap* path, not just the default one.

Measured on a 5-scenario agent run against a real app: 37 taps produced 73 separate
``analyze`` calls and 37 ``wait`` calls. The agent was not ignoring the observation — it was
routing around it, because ``tap`` carried no ``--fields``/``--where-*`` and the only way to
get a small read of the new screen was ``--no-observe`` plus a filtered ``analyze``. One
unfilterable call cost more than two targeted ones, so the default lost on economics.

These tests pin the two properties that remove the incentive: the observation is projected to
a compact view, and ``stable_elements`` (derived from the same tree) is trimmed to match
rather than silently re-adding every system-bar node the view just dropped.
"""

from __future__ import annotations

import json

from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import (
    ActionResult,
    AnalyzeResult,
    Element,
    Meta,
    OutputFormat,
    Screen,
)


def _observation() -> AnalyzeResult:
    """A screen carrying both app content and the system status bar."""
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package="com.example.app", source="hierarchy"),
        elements=[
            Element(
                id=0,
                type="FrameLayout",
                bounds=[0, 0, 1080, 74],
                center=[540, 37],
                resource_id="com.android.systemui:id/status_bar",
                window="system",
            ),
            Element(
                id=1,
                type="TextView",
                bounds=[47, 8, 162, 66],
                center=[104, 37],
                text="3:40",
                resource_id="com.android.systemui:id/clock",
                window="system",
            ),
            Element(
                id=2,
                type="Button",
                bounds=[0, 500, 1080, 620],
                center=[540, 560],
                text="Continue",
                resource_id="com.example.app:id/buttonContinue",
                clickable=True,
                window="app",
            ),
        ],
        meta=Meta(duration_ms=12, tier_used="hierarchy", path="hierarchy"),
    )


def _action_result() -> ActionResult:
    obs = _observation()
    return ActionResult(
        ok=True,
        action="tap",
        id=2,
        observation=obs,
        observation_present=True,
        stable_elements=[
            {"id": 0, "stable_key": "rid:status_bar"},
            {"id": 1, "stable_key": "rid:clock"},
            {"id": 2, "stable_key": "rid:buttonContinue"},
        ],
    )


def _emitted(monkeypatch, view: Projection | None) -> dict:
    """Run a result through the CLI emit path with *view* installed, capture the JSON."""
    from android_ui_analyser import cli

    printed: list[str] = []
    monkeypatch.setattr(cli.typer, "echo", lambda s, *a, **k: printed.append(str(s)))
    monkeypatch.setattr(cli, "_OBSERVATION_VIEW", view)
    monkeypatch.setattr(cli, "_UNTIL", None)
    cli._emit(_action_result(), OutputFormat.json)
    assert printed, "emit produced no output"
    return json.loads(printed[-1])


def test_default_observation_drops_system_chrome_and_extra_columns(monkeypatch) -> None:
    view = Projection.for_observation("id,text,rid,clickable")
    data = _emitted(monkeypatch, view)

    ids = [e["id"] for e in data["observation"]["elements"]]
    assert ids == [2], "the status bar and clock are not what the caller acted on"

    columns = set(data["observation"]["elements"][0])
    assert columns <= {"id", "text", "rid", "clickable"}
    assert "bounds" not in columns and "center" not in columns


def test_stable_elements_follow_the_same_view(monkeypatch) -> None:
    """Trimming only ``elements`` would re-add the dropped nodes through the back door."""
    view = Projection.for_observation("id,text,rid,clickable")
    data = _emitted(monkeypatch, view)
    assert [s["id"] for s in data["stable_elements"]] == [2]


def test_observe_fields_all_is_an_exact_opt_out(monkeypatch) -> None:
    """``all`` must be byte-identical to the unprojected render, not merely similar."""
    assert Projection.for_observation("all") is None
    assert Projection.for_observation("  ALL  ") is None
    assert Projection.for_observation("") is None
    assert Projection.for_observation(None) is None

    data = _emitted(monkeypatch, None)
    assert [e["id"] for e in data["observation"]["elements"]] == [0, 1, 2]
    assert [s["id"] for s in data["stable_elements"]] == [0, 1, 2]


def test_compact_view_is_materially_smaller(monkeypatch) -> None:
    """The whole point is token cost, so assert the direction with a real margin."""
    full = json.dumps(_emitted(monkeypatch, None))
    compact = json.dumps(_emitted(monkeypatch, Projection.for_observation("id,text,rid,clickable")))
    assert len(compact) < len(full) / 2
