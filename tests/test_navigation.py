"""Navigation affordances + autopilot (the "memory agents actually use" work).

Covers: usage-ranked suggestions (`_rank_score`), `navigation_hints` (inline
known_routes/suggested_gotos/map_hint), `resolve_goal`, `_shortest_path(start=)`,
`aua goto` drive+verify/handoff/plan, `engine.orient`, the new `Meta` fields, the
`MemoryCfg` knobs, the device/daemon cleanup hooks, and the daemon `goto`/`orient`
dispatch. Reuses the realistic fixtures from test_memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.daemon import DaemonClient, dispatch, stop
from android_ui_analyser.device import Uiautomator2Device
from android_ui_analyser.memory import (
    AppMemoryStore,
    ScreenRecord,
    SessionState,
    _rank_score,
    _shortest_path,
    resolve_goal,
)
from android_ui_analyser.schema import (
    AnalyzeResult,
    Meta,
    OutputFormat,
    PathKind,
    Screen,
    ScreenSource,
    Tier,
)
from conftest import FakeDevice, make_config
from test_memory import APPS, HOME, P, _elements, _engine, _hier, _node, _store

runner = CliRunner()

# A third screen reached from "apps" via the "Images" button (so we have a 2-hop route).
IMAGES = _hier(
    _node(
        "android.widget.TextView", text="Create image", rid="x:id/header", b="[40,120][1040,210]"
    ),
    _node(
        "android.widget.EditText",
        rid="x:id/prompt",
        desc="Prompt",
        clk=True,
        b="[40,400][1040,560]",
    ),
    _node("android.widget.Button", text="Generate", rid="x:id/go", clk=True, b="[40,640][400,740]"),
)
# An unrecorded "wrong turn" screen for the handoff test.
OTHER = _hier(
    _node("android.widget.TextView", text="Settings", rid="x:id/title", b="[40,120][1040,210]"),
    _node("android.widget.Button", text="Done", rid="x:id/done", clk=True, b="[40,400][400,500]"),
)


class ScriptedDevice(FakeDevice):
    """A FakeDevice that advances through an ordered list of screens on each tap."""

    def __init__(self, screens: list[str], **kw: object) -> None:
        super().__init__(hierarchy_xml=screens[0], **kw)  # type: ignore[arg-type]
        self._screens = screens
        self._idx = 0

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._idx = min(self._idx + 1, len(self._screens) - 1)
        self._xml = self._screens[self._idx]


def _build_three(tmp_path: Path) -> AppMemoryStore:
    """A home → apps → images map, recorded directly into the store."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), activity=".Home", name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), activity=".Apps", name_hint="apps")
    store.record_screen(package=P, elements=_elements(IMAGES), activity=".Img", name_hint="images")
    store.record_route(P, "home", "apps", "tap 'Apps'")
    store.record_route(P, "apps", "images", "tap 'Images'")
    return store


# --------------------------------------------------------------- ranking (_rank_score)


def _rec(name: str, *, visits: int, age_days: float, now: datetime) -> ScreenRecord:
    ts = (now - timedelta(days=age_days)).isoformat(timespec="seconds")
    return ScreenRecord(
        name=name, signature="s", first_seen=ts, last_seen=ts, last_verified=ts, visit_count=visits
    )


def test_rank_score_recency_beats_raw_frequency() -> None:
    now = datetime(2026, 6, 19).astimezone()
    recent = _rec("recent", visits=2, age_days=0.5, now=now)
    stale = _rec("stale", visits=5, age_days=30, now=now)
    assert _rank_score(recent, now=now, half_life_days=3.0) > _rank_score(
        stale, now=now, half_life_days=3.0
    )


def test_rank_score_last_goal_boost_floats_to_top() -> None:
    now = datetime(2026, 6, 19).astimezone()
    big = _rec("big", visits=50, age_days=0.1, now=now)
    target = _rec("target", visits=1, age_days=20, now=now)
    assert _rank_score(target, now=now, half_life_days=3.0, last_goal="target") > _rank_score(
        big, now=now, half_life_days=3.0
    )


# --------------------------------------------------------------- path + resolve


def test_shortest_path_from_start_node(tmp_path: Path) -> None:
    app_map = _build_three(tmp_path).load(P)
    assert app_map is not None
    assert len(_shortest_path(app_map, "images")) == 2  # from the root (home)
    assert len(_shortest_path(app_map, "images", start="apps")) == 1  # from apps
    assert _shortest_path(app_map, "apps", start="apps") == []  # already there


def test_resolve_goal_fuzzy_and_miss(tmp_path: Path) -> None:
    app_map = _build_three(tmp_path).load(P)
    assert app_map is not None
    assert resolve_goal(app_map, "images") == "images"  # exact name
    assert resolve_goal(app_map, "image") == "images"  # fuzzy → the images screen
    assert resolve_goal(app_map, "nonexistent-zzz") is None


# --------------------------------------------------------------- navigation_hints


def test_navigation_hints_routes_and_ranked_gotos(tmp_path: Path) -> None:
    store = _build_three(tmp_path)
    sess = store.load_session("emu-nav")
    sess.current_screen = "apps"
    store.save_session("emu-nav", sess)
    hints = store.navigation_hints("emu-nav", P)
    assert any("tap 'Images'" in r and "images" in r for r in hints.known_routes)
    assert "goto images" in hints.suggested_gotos


def test_navigation_hints_empty_for_unmapped(tmp_path: Path) -> None:
    store = _store(tmp_path)  # nothing recorded
    hints = store.navigation_hints("emu-x", P)
    assert hints.known_routes == [] and hints.suggested_gotos == [] and hints.map_hint is None


def test_session_back_compat_without_last_goal() -> None:
    old = SessionState.model_validate_json('{"package":"x","current_screen":"home","pending":[]}')
    assert old.last_goal is None  # new field defaults cleanly on old data


# --------------------------------------------------------------- engine: inline affordances


def test_analyze_pushes_known_routes_and_gotos(tmp_path: Path) -> None:
    _build_three(tmp_path)  # pre-seed the map
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-aff")
    eng = _engine(tmp_path, dev)
    meta = eng.analyze(source="hierarchy").meta
    assert meta.known_screen == "home"
    assert any("Apps" in r for r in meta.known_routes)
    assert meta.suggested_gotos  # ranked goto suggestions are pushed inline


def test_analyze_no_suggestions_when_disabled(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-off")
    eng = _engine(tmp_path, dev, suggest=False)
    meta = eng.analyze(source="hierarchy").meta
    assert meta.known_routes == [] and meta.suggested_gotos == []


# --------------------------------------------------------------- engine: goto autopilot


def test_goto_drives_and_verifies_each_hop(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-goto")
    eng = _engine(tmp_path, dev)
    out = eng.goto("images")
    assert out["ok"] and out["arrived"] and out["target"] == "images"
    assert [h["ok"] for h in out["hops"]] == [True, True]
    assert sum(1 for c in dev.calls if c[0] == "click") == 2  # exactly two taps driven


def test_goto_plan_does_not_act(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-plan")
    eng = _engine(tmp_path, dev)
    out = eng.goto("images", plan=True)
    assert out["ok"] and out.get("plan") and len(out["route"]) == 2
    assert not any(c[0] == "click" for c in dev.calls)  # nothing tapped


def test_goto_unknown_goal_is_route_unknown(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME], package=P, serial="emu-unk")
    eng = _engine(tmp_path, dev)
    out = eng.goto("there-is-no-such-screen")
    assert out["ok"] is False and out["code"] == "route_unknown"


def test_goto_hands_off_on_wrong_screen(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME, OTHER], package=P, serial="emu-div")  # tap → unexpected screen
    eng = _engine(tmp_path, dev)
    out = eng.goto("images")
    assert out["ok"] is False and out["code"] == "wrong_screen"
    assert out["remaining_route"]  # the un-walked tail is handed back
    assert out["elements"]  # current screen given so the caller can continue


def test_goto_records_last_goal(tmp_path: Path) -> None:
    store = _build_three(tmp_path)
    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-lg")
    eng = _engine(tmp_path, dev)
    eng.goto("images")
    assert store.load_session("emu-lg").last_goal == "images"


# --------------------------------------------------------------- engine: orient + close


def test_orient_reports_known_app(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-or")
    eng = _engine(tmp_path, dev)
    out = eng.orient()
    assert out["known"] and out["package"] == P and out["screens"] == 3
    assert out["suggested_gotos"]


def test_engine_close_releases_device(tmp_path: Path) -> None:
    dev = FakeDevice(package=P)
    eng = _engine(tmp_path, dev)
    _ = eng.device  # force-connect
    eng.close()
    assert eng._device is None
    eng.close()  # idempotent — no error on a second call


# --------------------------------------------------------------- device cleanup fix


def test_u2_device_close_stops_uiautomator() -> None:
    dev = Uiautomator2Device.__new__(Uiautomator2Device)  # skip real connect
    mock = MagicMock()
    dev._d = mock
    dev.close()
    mock.stop_uiautomator.assert_called_once()
    assert dev._d is None


# --------------------------------------------------------------- daemon dispatch


def test_daemon_dispatch_goto_and_orient() -> None:
    class FakeEng:
        def goto(self, **kw: object) -> dict[str, object]:
            return {"ok": True, "goal": kw.get("goal")}

        def orient(self) -> dict[str, object]:
            return {"known": False}

    r = dispatch(FakeEng(), {"cmd": "goto", "args": {"goal": "x", "plan": False}})
    assert r["ok"] and r["result"]["goal"] == "x"
    r2 = dispatch(FakeEng(), {"cmd": "orient"})
    assert r2["ok"] and r2["result"] == {"known": False}


def test_daemon_ping_treats_nonresponse_as_down(monkeypatch) -> None:
    client = DaemonClient("/no/such/daemon.sock")
    assert client.ping() is False  # connect refused (OSError)

    def empty(cmd: str, **k: object) -> dict[str, object]:
        raise json.JSONDecodeError("empty", "", 0)  # daemon mid-shutdown sends nothing

    monkeypatch.setattr(client, "call", empty)
    assert client.ping() is False  # the fix: don't crash the stop() poll loop


def test_daemon_stop_when_not_running_is_clean(tmp_path: Path) -> None:
    cfg = make_config(daemon={"socket": str(tmp_path / "absent.sock")})
    out = stop(cfg)
    assert out["running"] is False and out["status"] == "not_running"


# --------------------------------------------------------------- schema + config contracts


def _meta(**kw: object) -> Meta:
    base = {"duration_ms": 1, "tier_used": Tier.hierarchy, "path": PathKind.hierarchy}
    return Meta(**{**base, **kw})  # type: ignore[arg-type]


def _result(meta: Meta) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1, height=1, source=ScreenSource.hierarchy), elements=[], meta=meta
    )


def test_meta_new_fields_round_trip() -> None:
    m = _meta(known_routes=["tap 'X' → y"], suggested_gotos=["goto y"], map_hint="hi")
    again = Meta.model_validate(m.model_dump())
    assert again.known_routes == ["tap 'X' → y"]
    assert again.suggested_gotos == ["goto y"] and again.map_hint == "hi"


def test_compact_drops_empty_keeps_set_affordances() -> None:
    empty = json.loads(_result(_meta()).render(OutputFormat.compact))["meta"]
    assert (
        "known_routes" not in empty and "suggested_gotos" not in empty and "map_hint" not in empty
    )
    setm = json.loads(
        _result(_meta(known_routes=["tap 'X' → y"], map_hint="hi")).render(OutputFormat.compact)
    )["meta"]
    assert setm["known_routes"] == ["tap 'X' → y"] and setm["map_hint"] == "hi"


def test_memory_cfg_suggestion_knobs() -> None:
    cfg = make_config()
    assert cfg.memory.suggest is True
    assert cfg.memory.suggest_max == 4 and cfg.memory.rank_half_life_days == 3.0
    cfg2 = make_config(memory={"suggest": False, "suggest_max": 2, "rank_half_life_days": 1.0})
    assert cfg2.memory.suggest is False and cfg2.memory.suggest_max == 2


# --------------------------------------------------------------- CLI: aua goto


def test_cli_goto_drives_and_unknown(tmp_path: Path, monkeypatch) -> None:
    # The CLI reads memory from AUA_MEMORY__DIR (the autouse isolation dir), so build there.
    store = AppMemoryStore(make_config().memory)
    store.record_screen(package=P, elements=_elements(HOME), activity=".H", name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), activity=".A", name_hint="apps")
    store.record_screen(package=P, elements=_elements(IMAGES), activity=".I", name_hint="images")
    store.record_route(P, "home", "apps", "tap 'Apps'")
    store.record_route(P, "apps", "images", "tap 'Images'")

    dev = ScriptedDevice([HOME, APPS, IMAGES], package=P, serial="emu-cli-goto")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    ok = runner.invoke(app, ["--format", "compact", "goto", "images"])
    assert ok.exit_code == 0, ok.stderr
    assert json.loads(ok.stdout)["arrived"] is True

    miss = runner.invoke(app, ["--format", "compact", "goto", "no-such-goal"])
    assert miss.exit_code == 1  # not arrived → non-zero
    assert json.loads(miss.stdout)["code"] == "route_unknown"
