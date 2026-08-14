"""`goto` looked at the screen, then planned from where it last wrote down instead.

The first line of `goto` is an `analyze`, so the real screen is in hand before a single hop
runs. It was thrown away: the route was planned from `session.current_screen`, a cursor written
by whichever command last succeeded in writing one.

Measured on a live agent run, 2026-08-10. A shared scratch file made that write fail
(`test_two_writers_do_not_share_one_scratch_file`), so the cursor still said `character_card`
while the device sat on the Android home screen. `goto` replayed the route it found there:

    hops: [{"action": "key 'back' + key 'back'", "expected": "character_card", "ok": false}]

Two blind `back` presses on the launcher. With the observed screen as the start, the same call
reports `route_unknown` and the agent keeps its bearings.

Mid-transit is the one exception: the foreground is a sign-in app, the map belongs to the app
that sent us there, and the cursor is the only thing that knows the journey.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMemoryStore, RouteStep, SessionState
from conftest import FakeDevice, make_config
from test_memory import APPS, HOME, P, _elements, _hier, _node

STRANGER = _hier(
    _node("android.widget.TextView", text="Nothing here is in the map", b="[40,120][1040,210]")
)


def _engine(tmp_path: object, *, serial: str, showing: str) -> tuple[Engine, AppMemoryStore]:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(P, "home", "apps", steps=[RouteStep(kind="tap", label="Apps")])
    device = FakeDevice(hierarchy_xml=showing, package=P, serial=serial)
    return Engine(cfg, device=device), store


def test_a_stale_cursor_does_not_send_us_down_a_route(tmp_path: object) -> None:
    """Cursor says `home`, device shows `apps`: the hop from `home` must not fire."""
    engine, store = _engine(tmp_path, serial="stale-cursor", showing=APPS)
    store.save_session("stale-cursor", SessionState(package=P, current_screen="home"))

    result = engine.goto("apps")

    assert result["ok"] is True
    assert result.get("already_there") is True, "we are on the target; the cursor was wrong"
    assert result["hops"] == []


def test_an_unfamiliar_screen_is_admitted_not_guessed_through(tmp_path: object) -> None:
    """This is the live failure in miniature: off-map, with a cursor that still claims a route."""
    engine, store = _engine(tmp_path, serial="unknown-screen", showing=STRANGER)
    store.save_session("unknown-screen", SessionState(package=P, current_screen="home"))

    result = engine.goto("apps")

    assert result["code"] == "route_unknown"
    assert result["current_screen"] != "home", "it reports where it is, not where it was"
    assert "hops" not in result, "nothing was replayed blind"


def test_the_route_still_runs_when_cursor_and_screen_agree(tmp_path: object) -> None:
    engine, store = _engine(tmp_path, serial="agreeing", showing=HOME)
    store.save_session("agreeing", SessionState(package=P, current_screen="home"))

    result = engine.goto("apps")

    assert result["hops"][0]["action"] == "tap 'Apps'"
    assert result["hops"][0]["expected"] == "apps"


def test_a_clickable_destination_row_cannot_prove_already_there(
    tmp_path: object, monkeypatch: Any
) -> None:
    launcher = _hier(
        _node(
            "android.widget.TextView",
            text="Display preferences",
            clk=True,
            b="[40,120][1040,230]",
        )
    )
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    store.record_screen(package=P, elements=_elements(launcher), name_hint="catalog_home")
    device = FakeDevice(hierarchy_xml=launcher, package=P, serial="clickable-is-not-arrival")
    engine = Engine(cfg, device=device)
    store.save_session(
        "clickable-is-not-arrival",
        SessionState(package=P, current_screen="catalog_home"),
    )
    # Pin the defensive contract independently of fuzzy ranking: even if a future resolver
    # again mistakes the launcher row for current-screen identity, goto may not claim success.
    monkeypatch.setattr(engine_mod, "resolve_goal", lambda *_args, **_kwargs: "catalog_home")

    result = engine.goto("Display preferences")

    assert result["ok"] is False
    assert result["code"] == "arrival_unproven"
    assert result["arrived"] is False
    assert result["elements"], "return the current frame so the caller can tap the row"
