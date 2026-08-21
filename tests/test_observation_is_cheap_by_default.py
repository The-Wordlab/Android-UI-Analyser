"""The folded post-action observation must be the *cheap* path, not just the default one.

The regression models an agent routing around folded observations because ``tap`` carried no
``--fields``/``--where-*`` and the only way to
get a small read of the new screen was ``--no-observe`` plus a filtered ``analyze``. One
unfilterable call cost more than two targeted ones, so the default lost on economics.

These tests pin the two properties that remove the incentive: the observation is projected to
a compact view, and ``next_actions`` (derived from the same tree) is trimmed to match
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
            Element(
                id=3,
                type="Button",
                bounds=[0, 2200, 1080, 2400],
                center=[540, 2300],
                text="Search",
                resource_id="com.google.android.inputmethod.latin:id/key_search",
                clickable=True,
                window="ime",
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
        next_actions=[
            {"id": 0, "label": "Status bar"},
            {"id": "rid:buttonContinue", "label": "Continue"},
            {"id": 3, "label": "Keyboard search"},
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


def _key(ordinal: int) -> str:
    """The stable id the payload publishes for the fixture element with this ordinal."""
    from android_ui_analyser.identity import stable_key

    return stable_key(next(e for e in _observation().elements if e.id == ordinal))


def test_default_observation_drops_system_chrome_and_extra_columns(monkeypatch) -> None:
    view = Projection.for_observation("id,text,rid,clickable")
    data = _emitted(monkeypatch, view)

    ids = [e["id"] for e in data["observation"]["elements"]]
    assert ids == [_key(2)], "the status bar and clock are not what the caller acted on"

    columns = set(data["observation"]["elements"][0])
    assert columns <= {"id", "text", "rid", "clickable"}
    assert "bounds" not in columns and "center" not in columns



def test_next_actions_follow_the_same_view(monkeypatch) -> None:
    """Guidance cannot reference system/IME ids absent from the projected observation."""
    view = Projection.for_observation("id,text,rid,clickable")
    data = _emitted(monkeypatch, view)
    assert data["next_actions"] == [{"id": _key(2), "label": "Continue"}]


def test_observe_fields_all_is_an_exact_opt_out(monkeypatch) -> None:
    """``all`` must be byte-identical to the unprojected render, not merely similar."""
    assert Projection.for_observation("all") is None
    assert Projection.for_observation("  ALL  ") is None
    assert Projection.for_observation("") is None
    assert Projection.for_observation(None) is None

    data = _emitted(monkeypatch, None)
    assert [e["id"] for e in data["observation"]["elements"]] == [
        _key(i) for i in (0, 1, 2, 3)
    ]


def test_compact_view_is_materially_smaller(monkeypatch) -> None:
    """The whole point is token cost, so assert the direction with a real margin."""
    full = json.dumps(_emitted(monkeypatch, None))
    compact = json.dumps(_emitted(monkeypatch, Projection.for_observation("id,text,rid,clickable")))
    assert len(compact) < len(full) / 2
