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
    RouteStep,
    ScreenRecord,
    SessionState,
    _rank_score,
    _shortest_path,
    resolve_goal,
)
from android_ui_analyser.schema import (
    ActionResult,
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

# A third screen reached from "apps" via the "Reports" button (so we have a 2-hop route).
REPORTS = _hier(
    _node(
        "android.widget.TextView", text="Build report", rid="x:id/header", b="[40,120][1040,210]"
    ),
    _node(
        "android.widget.EditText",
        rid="x:id/prompt",
        desc="Prompt",
        clk=True,
        b="[40,400][1040,560]",
    ),
    _node("android.widget.Button", text="Run", rid="x:id/go", clk=True, b="[40,640][400,740]"),
)
# An unrecorded "wrong turn" screen for the handoff test.
OTHER = _hier(
    _node("android.widget.TextView", text="Settings", rid="x:id/title", b="[40,120][1040,210]"),
    _node("android.widget.Button", text="Done", rid="x:id/done", clk=True, b="[40,400][400,500]"),
)


class ScriptedDevice(FakeDevice):
    """A FakeDevice that advances through an ordered list of screens on each tap/key."""

    def __init__(self, screens: list[str], **kw: object) -> None:
        super().__init__(hierarchy_xml=screens[0], **kw)  # type: ignore[arg-type]
        self._screens = screens
        self._idx = 0

    def _advance(self) -> None:
        self._idx = min(self._idx + 1, len(self._screens) - 1)
        self._xml = self._screens[self._idx]

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._advance()

    def press(self, key: str) -> None:
        super().press(key)
        self._advance()


def _build_three(tmp_path: Path) -> AppMemoryStore:
    """A home → apps → reports map, recorded directly into the store."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), activity=".Home", name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), activity=".Apps", name_hint="apps")
    store.record_screen(
        package=P,
        elements=_elements(REPORTS),
        activity=".Reports",
        name_hint="reports",
    )
    store.record_route(
        P,
        "home",
        "apps",
        steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")],
    )
    store.record_route(
        P,
        "apps",
        "reports",
        steps=[RouteStep(kind="tap", label="Reports", resource_id="tool_reports")],
    )
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
    assert len(_shortest_path(app_map, "reports")) == 2  # from the root (home)
    assert len(_shortest_path(app_map, "reports", start="apps")) == 1  # from apps
    assert _shortest_path(app_map, "apps", start="apps") == []  # already there


def test_resolve_goal_fuzzy_and_miss(tmp_path: Path) -> None:
    app_map = _build_three(tmp_path).load(P)
    assert app_map is not None
    assert resolve_goal(app_map, "reports") == "reports"  # exact name
    assert resolve_goal(app_map, "report") == "reports"  # fuzzy → the reports screen
    assert resolve_goal(app_map, "nonexistent-zzz") is None


def test_resolve_goal_prefers_a_current_screen_over_a_stale_better_rank(tmp_path: Path) -> None:
    app_map = _build_three(tmp_path).load(P)
    assert app_map is not None
    app_map.screens["reports"].stale = True
    current = app_map.screens["reports"].model_copy(
        update={
            "name": "image_gallery",
            "id": "screen_image_gallery",
            "stale": False,
            "visit_count": 1,
        }
    )
    app_map.screens["image_gallery"] = current

    assert resolve_goal(app_map, "image") == "image_gallery"


# --------------------------------------------------------------- navigation_hints


def test_navigation_hints_routes_and_ranked_gotos(tmp_path: Path) -> None:
    store = _build_three(tmp_path)
    sess = store.load_session("emu-nav")
    sess.current_screen = "apps"
    store.save_session("emu-nav", sess)
    hints = store.navigation_hints("emu-nav", P)
    assert any("tap 'Reports'" in r and "reports" in r for r in hints.known_routes)
    assert "goto reports" in hints.suggested_gotos


def test_navigation_hints_empty_for_unmapped(tmp_path: Path) -> None:
    store = _store(tmp_path)  # nothing recorded
    hints = store.navigation_hints("emu-x", P)
    assert hints.known_routes == [] and hints.suggested_gotos == [] and hints.map_hint is None


def test_navigation_hints_never_advertise_stale_or_unreachable_gotos(tmp_path: Path) -> None:
    store = _build_three(tmp_path)
    app_map = store.load(P)
    assert app_map is not None
    app_map.screens["reports"].stale = True
    # A healthy screen with no verified path is known, but not a ready-to-run goto.
    orphan = app_map.screens["reports"].model_copy(
        update={"name": "orphan", "id": "screen_orphan", "stale": False}
    )
    app_map.screens["orphan"] = orphan
    store.save(app_map)
    session = store.load_session("safe-hints")
    session.current_screen = "apps"
    store.save_session("safe-hints", session)

    hints = store.navigation_hints("safe-hints", P)

    assert "goto reports" not in hints.suggested_gotos
    assert "goto orphan" not in hints.suggested_gotos
    assert not any("reports" in route for route in hints.known_routes)


def test_navigation_hints_do_not_advertise_unsafe_only_goto_as_ready(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_route(
        P,
        "home",
        "catalog",
        steps=[RouteStep(kind="open-link", arg="fiction://catalog")],
    )
    session = store.load_session("unsafe-inline-hint")
    session.current_screen = "home"
    store.save_session("unsafe-inline-hint", session)

    hints = store.navigation_hints("unsafe-inline-hint", P)

    assert "goto catalog" not in hints.suggested_gotos
    assert not any("catalog" in route for route in hints.known_routes)


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
    dev = ScriptedDevice([HOME, APPS, REPORTS], package=P, serial="emu-goto")
    eng = _engine(tmp_path, dev)
    out = eng.goto("reports")
    assert out["ok"] and out["arrived"] and out["target"] == "reports"
    assert [h["ok"] for h in out["hops"]] == [True, True]
    assert sum(1 for c in dev.calls if c[0] == "click") == 2  # exactly two taps driven


def test_goto_plan_does_not_act(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME, APPS, REPORTS], package=P, serial="emu-plan")
    eng = _engine(tmp_path, dev)
    out = eng.goto("reports", plan=True)
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
    out = eng.goto("reports")
    assert out["ok"] is False and out["code"] == "wrong_screen"
    assert out["remaining_route"]  # the un-walked tail is handed back
    assert out["elements"]  # current screen given so the caller can continue


def test_goto_waits_for_a_loading_shell_before_calling_the_route_wrong(
    tmp_path: Path, monkeypatch
) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME], package=P, serial="emu-loading-route")
    eng = _engine(tmp_path, dev)
    loading = AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=P, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=10,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="loading",
            device_serial=dev.serial,
        ),
    )
    arrived = AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=P, source="hierarchy"),
        elements=_elements(APPS),
        meta=Meta(
            duration_ms=10,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="apps",
            device_serial=dev.serial,
        ),
    )

    monkeypatch.setattr(eng, "_run_steps", lambda *_args, **_kwargs: (None, loading))
    monkeypatch.setattr(eng, "_observation_is_loading", lambda _result: True)
    monkeypatch.setattr(
        eng,
        "_await_known_screen",
        lambda *_args, **_kwargs: ActionResult(
            ok=True,
            action="await",
            observation=arrived,
            observation_present=True,
            await_outcome="satisfied",
        ),
    )

    out = eng.goto("apps")

    assert out["ok"] is True
    assert out["final_screen"] == "apps"


def test_goto_records_last_goal(tmp_path: Path) -> None:
    store = _build_three(tmp_path)
    dev = ScriptedDevice([HOME, APPS, REPORTS], package=P, serial="emu-lg")
    eng = _engine(tmp_path, dev)
    eng.goto("reports")
    assert store.load_session("emu-lg").last_goal == "reports"


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


# --------------------------------------------------------------- actions return observation


def test_action_observe_attaches_fresh_screen(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=APPS, package=P, serial="emu-obs")
    eng = _engine(tmp_path, dev)
    r = eng.analyze(source="hierarchy")
    report_id = next(e.id for e in r.elements if (e.text or "").startswith("Reports"))
    observed = eng.tap(report_id, observe=True)
    assert observed.ok and observed.observation is not None and observed.observation.elements
    assert eng.tap(report_id, observe=False).observation is None  # opt out → no observation


def test_observe_repopulates_cache_for_type_then_send(tmp_path: Path) -> None:
    # The flow the user flagged: type → tap send WITHOUT a separate analyze in between.
    create = _hier(
        _node("android.widget.TextView", text="Create", rid="x:id/h", b="[40,120][1040,210]"),
        _node(
            "android.widget.EditText",
            rid="x:id/prompt",
            desc="Prompt",
            clk=True,
            b="[40,400][1040,560]",
        ),
        _node(
            "android.widget.Button", text="Send", rid="x:id/send", clk=True, b="[40,640][400,740]"
        ),
    )
    dev = FakeDevice(hierarchy_xml=create, package=P, serial="emu-obs2")
    eng = _engine(tmp_path, dev)
    r = eng.analyze(source="hierarchy")
    pid = next(e.id for e in r.elements if e.content_desc == "Prompt")
    typed = eng.input_text(pid, "hello world", observe=True)
    assert typed.observation is not None
    send_id = next(e.id for e in typed.observation.elements if e.text == "Send")
    # input() invalidated the cache, but observe's folded analyze re-populated it → tap resolves
    # with NO manual analyze call between type and send.
    assert eng.tap(send_id).ok
    assert any(c[0] == "click" for c in dev.calls)


def test_actionresult_render_embeds_then_drops_observation(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-r")
    eng = _engine(tmp_path, dev)
    tid = next(e.id for e in eng.analyze(source="hierarchy").elements if e.text == "Apps")
    with_obs = json.loads(eng.tap(tid, observe=True).render("compact"))["observation"]
    assert "elements" in with_obs
    assert "observation" not in json.loads(eng.tap(tid, observe=False).render("compact"))


def test_goto_returns_destination_elements(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = ScriptedDevice([HOME, APPS, REPORTS], package=P, serial="emu-gel")
    eng = _engine(tmp_path, dev)
    out = eng.goto("reports")
    assert out["arrived"] and out.get("elements")  # destination marks returned inline


def test_committed_project_skill_is_current() -> None:
    # Guard against forgetting to regenerate: the committed SKILL.md must equal the guide.
    from android_ui_analyser import guide

    skill = Path(__file__).resolve().parent.parent / ".claude/skills/android-ui-analyser/SKILL.md"
    assert skill.read_text() == guide.render_skill(), (
        "Project SKILL.md is stale — regenerate with `aua guide --emit-skill`."
    )


def test_observe_snapshot_does_not_pollute_memory(tmp_path: Path) -> None:
    # A tap that transitions to an unmapped screen: the observe snapshot must NOT record it
    # (it can be mid-render). Otherwise we'd get spurious screens like the live `apps_2`.
    store = _build_three(tmp_path)
    dev = ScriptedDevice([APPS, OTHER], package=P, serial="emu-norec")
    eng = _engine(tmp_path, dev)
    r = eng.analyze(source="hierarchy")  # recognises 'apps' (normal recording)
    before = set(store.load(P).screens)
    report_id = next(e.id for e in r.elements if (e.text or "").startswith("Reports"))
    res = eng.tap(
        report_id, observe=True
    )  # click → OTHER; observe snapshots it but must not record
    assert res.observation is not None and res.observation.elements
    assert set(store.load(P).screens) == before  # no new screen written by the snapshot


def test_cli_tap_observe(monkeypatch) -> None:
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-cli-obs")

    def connect_selected(serial=None):
        dev.serial = serial or dev.serial
        return dev

    monkeypatch.setattr(engine_mod, "connect", connect_selected)
    a = runner.invoke(app, ["--format", "compact", "analyze", "--source", "hierarchy"])
    tid = next(e["id"] for e in json.loads(a.stdout)["elements"] if e.get("text") == "Apps")
    r = runner.invoke(app, ["--format", "compact", "tap-and-analyze", str(tid)])
    assert r.exit_code == 0, r.stderr
    assert "observation" in json.loads(r.stdout)


# --------------------------------------------------------------- schema + config contracts


def _meta(**kw: object) -> Meta:
    base = {"duration_ms": 1, "tier_used": Tier.hierarchy, "path": PathKind.hierarchy}
    return Meta(**{**base, **kw})  # type: ignore[arg-type]


def _result(meta: Meta) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1, height=1, source=ScreenSource.hierarchy), elements=[], meta=meta
    )


def test_meta_new_fields_round_trip() -> None:
    m = _meta(
        known_routes=["tap 'X' → y"],
        suggested_gotos=["goto y"],
        research_tasks=["research task_1: inspect source"],
        map_hint="hi",
    )
    again = Meta.model_validate(m.model_dump())
    assert again.known_routes == ["tap 'X' → y"]
    assert again.suggested_gotos == ["goto y"] and again.map_hint == "hi"
    assert again.research_tasks == ["research task_1: inspect source"]


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
    assert cfg.memory.auto_research is True and cfg.memory.research_suggest_max == 3
    cfg2 = make_config(
        memory={
            "suggest": False,
            "suggest_max": 2,
            "rank_half_life_days": 1.0,
            "auto_research": False,
        }
    )
    assert cfg2.memory.suggest is False and cfg2.memory.suggest_max == 2
    assert cfg2.memory.auto_research is False


# --------------------------------------------------------------- CLI: aua goto


def test_cli_goto_drives_and_unknown(tmp_path: Path, monkeypatch) -> None:
    # The CLI reads memory from AUA_MEMORY__DIR (the autouse isolation dir), so build there.
    store = AppMemoryStore(make_config().memory)
    store.record_screen(package=P, elements=_elements(HOME), activity=".H", name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), activity=".A", name_hint="apps")
    store.record_screen(package=P, elements=_elements(REPORTS), activity=".R", name_hint="reports")
    store.record_route(P, "home", "apps", "tap 'Apps'")
    store.record_route(P, "apps", "reports", "tap 'Reports'")

    dev = ScriptedDevice([HOME, APPS, REPORTS], package=P, serial="emu-cli-goto")

    def connect_selected(serial=None):
        dev.serial = serial or dev.serial
        return dev

    monkeypatch.setattr(engine_mod, "connect", connect_selected)

    ok = runner.invoke(app, ["--format", "compact", "goto", "reports"])
    assert ok.exit_code == 0, ok.stderr
    assert json.loads(ok.stdout)["arrived"] is True

    miss = runner.invoke(app, ["--format", "compact", "goto", "no-such-goal"])
    assert miss.exit_code == 1  # not arrived → non-zero
    assert json.loads(miss.stdout)["code"] == "route_unknown"


# --------------------------------------------------------------- step-based replay (v2)

GEAR_HOME = _hier(
    _node("android.widget.TextView", text="Home", rid="x:id/header", b="[40,120][1040,210]"),
    _node("android.view.View", rid="x:id/buttonSettings", clk=True, b="[900,300][1040,400]"),
    _node(
        "android.widget.Button", text="Chat", rid="x:id/nav_chat", clk=True, b="[40,440][1040,540]"
    ),
)
PREFS = _hier(
    _node("android.widget.TextView", text="Preferences", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Account &amp; Data",  # raw & is invalid in XML attributes
        rid="x:id/account",
        clk=True,
        b="[40,300][1040,400]",
    ),
)
DANGER_HOME = _hier(
    _node("android.widget.TextView", text="Confirm", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Delete my account",
        rid="x:id/del",
        clk=True,
        b="[40,300][1040,400]",
    ),
)


def test_goto_replays_id_only_edge(tmp_path: Path) -> None:
    """An unlabeled settings gear is replayable via its resource-id."""
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(GEAR_HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(PREFS), name_hint="prefs")
    store.record_route(
        P, "home", "prefs", steps=[RouteStep(kind="tap", resource_id="buttonSettings")]
    )
    dev = ScriptedDevice([GEAR_HOME, PREFS], package=P, serial="emu-id")
    eng = _engine(tmp_path, dev)
    out = eng.goto("prefs")
    assert out["ok"] and out["arrived"]
    assert sum(1 for c in dev.calls if c[0] == "click") == 1


def test_goto_replays_key_step_edge(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(PREFS), name_hint="prefs")
    store.record_screen(package=P, elements=_elements(GEAR_HOME), name_hint="home")
    store.record_route(P, "prefs", "home", steps=[RouteStep(kind="key", arg="back")])
    dev = ScriptedDevice([PREFS, GEAR_HOME], package=P, serial="emu-key")
    eng = _engine(tmp_path, dev)
    out = eng.goto("home")
    assert out["ok"] and out["arrived"]
    assert any(c[0] == "press" for c in dev.calls)


def test_goto_compound_legacy_action_is_clean_unsupported(tmp_path: Path) -> None:
    """Regression: `tap 'A' + tap 'B'` used to parse into a garbage label."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(P, "home", "apps", "tap 'A' + tap 'B'")
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-compound")
    eng = _engine(tmp_path, dev)
    out = eng.goto("apps")
    assert out["ok"] is False and out["code"] == "unsupported_action"
    assert out["hops"] == [] and "re-record" in out["hint"]
    assert not any(c[0] == "click" for c in dev.calls)  # nothing blindly tapped


def test_goto_refuses_destructive_step_without_flag(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(DANGER_HOME), name_hint="confirm")
    store.record_screen(package=P, elements=_elements(PREFS), name_hint="prefs")
    store.record_route(
        P,
        "confirm",
        "prefs",
        steps=[RouteStep(kind="tap", label="Delete my account", resource_id="del")],
    )
    dev = ScriptedDevice([DANGER_HOME, PREFS], package=P, serial="emu-guard")
    eng = _engine(tmp_path, dev)
    out = eng.goto("prefs")
    assert out["ok"] is False and out["code"] == "destructive_step"
    assert out["step"]["display"] == "tap 'Delete my account'"
    assert not any(c[0] == "click" for c in dev.calls)

    out2 = eng.goto("prefs", allow_destructive=True)
    assert out2["ok"] and out2["arrived"]
    assert sum(1 for c in dev.calls if c[0] == "click") == 1


def test_goto_plan_flags_destructive_and_legacy(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(DANGER_HOME), name_hint="confirm")
    store.record_screen(package=P, elements=_elements(PREFS), name_hint="prefs")
    store.record_route(P, "home", "confirm", "tap 'Apps'")  # legacy but replayable
    store.record_route(
        P, "confirm", "prefs", steps=[RouteStep(kind="tap", label="Delete my account")]
    )
    dev = ScriptedDevice([HOME], package=P, serial="emu-plan2")
    eng = _engine(tmp_path, dev)
    out = eng.goto("prefs", plan=True)
    assert out["ok"] and out["plan"]
    legacy_edge, steps_edge = out["route"]
    assert legacy_edge["legacy"] is True and legacy_edge["replayable"] is True
    assert steps_edge["legacy"] is False
    assert steps_edge["destructive"] == ["Delete my account"]
    assert not any(c[0] == "click" for c in dev.calls)


def test_goto_handoff_includes_remaining_steps(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(REPORTS), name_hint="reports")
    store.record_route(
        P,
        "home",
        "reports",
        steps=[
            RouteStep(kind="tap", label="Apps", resource_id="nav_apps"),
            RouteStep(kind="tap", label="No Such Button"),
        ],
    )
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-remain")
    eng = _engine(tmp_path, dev)
    out = eng.goto("reports")
    assert out["ok"] is False and out["code"] == "element_not_found"
    assert out["step"]["display"] == "tap 'No Such Button'"
    assert out["remaining_steps"] == ["tap 'No Such Button'"]


def test_shortest_path_prefers_steps_edge_over_legacy(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(P, "home", "apps", "tap [View]")  # legacy, unreplayable
    store.record_route(
        P, "home", "apps", steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")]
    )
    am = store.load(P)
    assert am is not None and len(am.routes) == 2  # genuinely parallel edges
    path = _shortest_path(am, "apps", start="home")
    assert path[0].steps, "the replayable edge must win the tie-break"


def test_cli_goto_accepts_allow_destructive(tmp_path: Path, monkeypatch) -> None:
    from android_ui_analyser.memory import RouteStep

    store = AppMemoryStore(make_config().memory)
    store.record_screen(package=P, elements=_elements(DANGER_HOME), activity=".C", name_hint="c")
    store.record_screen(package=P, elements=_elements(PREFS), activity=".P", name_hint="prefs")
    store.record_route(P, "c", "prefs", steps=[RouteStep(kind="tap", label="Delete my account")])
    dev = ScriptedDevice([DANGER_HOME, PREFS], package=P, serial="emu-cli-guard")

    def connect_selected(serial=None):
        dev.serial = serial or dev.serial
        return dev

    monkeypatch.setattr(engine_mod, "connect", connect_selected)

    refused = runner.invoke(app, ["--format", "compact", "goto", "prefs"])
    assert refused.exit_code == 1
    assert json.loads(refused.stdout)["code"] == "destructive_step"

    ok = runner.invoke(app, ["--format", "compact", "goto", "prefs", "--allow-destructive"])
    assert ok.exit_code == 0, ok.stderr
    assert json.loads(ok.stdout)["arrived"] is True


# --------------------------------------------------------------- observation recording

FORM = _hier(
    _node("android.widget.TextView", text="Form", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.EditText",
        rid="x:id/prompt",
        desc="Prompt",
        clk=True,
        b="[40,300][1040,400]",
    ),
    _node("android.widget.Button", text="Send", rid="x:id/send", clk=True, b="[40,440][400,540]"),
)


def test_observation_records_single_step_edge_on_known_screen(tmp_path: Path) -> None:
    """Acting on observation ids must yield replayable single-action edges."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-obs")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    apps_id = next(e.id for e in res.elements if e.text == "Apps")
    out = eng.tap(apps_id, observe=True)  # observation recognises "apps" and records
    assert out.observation is not None
    assert out.observation.meta.known_screen == "apps"

    am = store.load(P)
    edge = next(e for e in am.routes if e.to_screen == "apps")
    assert edge.action == "tap 'Apps'" and len(edge.steps) == 1
    sess = store.load_session("emu-obs")
    assert sess.current_screen == "apps" and sess.pending == []  # cursor advanced


def test_observation_on_unknown_screen_defers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    dev = ScriptedDevice([HOME, OTHER], package=P, serial="emu-obs-unk")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=True)  # lands on an UNRECORDED screen
    am = store.load(P)
    assert am.routes == []  # nothing recorded from a possibly mid-transition frame
    sess = store.load_session("emu-obs-unk")
    assert sess.current_screen == "home" and len(sess.pending) == 1  # deferred

    eng.analyze(source="hierarchy")  # the next plain analyze records screen + edge
    am = store.load(P)
    assert len(am.routes) == 1 and am.routes[0].steps


def test_observation_same_screen_keeps_pending_for_compound_edge(tmp_path: Path) -> None:
    """input (same screen) + tap Send must record as ONE honest two-step edge."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(FORM), name_hint="form")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    dev = ScriptedDevice([FORM, FORM, APPS], package=P, serial="emu-obs-same")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    field = next(e.id for e in res.elements if e.resource_id == "x:id/prompt")
    out = eng.input_text(field, "hello", observe=True)  # stays on the form
    assert out.observation is not None and out.observation.meta.known_screen == "form"
    sess = store.load_session("emu-obs-same")
    assert len(sess.pending) == 1  # kept, not clobbered

    send = next(e.id for e in out.observation.elements if e.text == "Send")
    eng.tap(send, observe=True)  # now lands on apps → compound edge
    am = store.load(P)
    edge = next(e for e in am.routes if e.to_screen == "apps")
    assert [s.kind for s in edge.steps] == ["input", "tap"]
    assert "input '<filled>'" in edge.action and "tap 'Send'" in edge.action


# --------------------------------------------------------------- inline deeplink hints


def test_analyze_suggests_deeplinks_inline(tmp_path: Path) -> None:
    """A mapped app offers its deeplink shortcuts inline on analyze — no `about` needed."""
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.remember_deeplink(P, "myapp://orders", note="mined", probed=True)  # proven → first
    store.remember_deeplink(P, "myapp://pet", note="mined")  # concrete, unprobed
    store.remember_deeplink(P, "myapp://items/{itemId}", note="mined")  # templated → excluded
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-dl-hint")
    eng = _engine(tmp_path, dev)
    meta = eng.analyze(source="hierarchy").meta
    assert "open myapp://orders" in meta.suggested_deeplinks
    assert "open myapp://pet" in meta.suggested_deeplinks
    assert meta.suggested_deeplinks[0] == "open myapp://orders"  # probed ranked first
    assert not any("itemId" in s for s in meta.suggested_deeplinks)  # templated excluded


def test_suggested_deeplinks_empty_without_playbook(tmp_path: Path) -> None:
    _build_three(tmp_path)  # screens/routes but no deeplinks
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-no-dl")
    eng = _engine(tmp_path, dev)
    meta = eng.analyze(source="hierarchy").meta
    assert meta.suggested_deeplinks == []


def test_suggested_deeplinks_compact_dropped_when_empty(tmp_path: Path) -> None:
    _build_three(tmp_path)
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-cpt")
    eng = _engine(tmp_path, dev)
    compact = eng.analyze(source="hierarchy").as_dict("compact")
    assert "suggested_deeplinks" not in compact["meta"]  # empty list trimmed
