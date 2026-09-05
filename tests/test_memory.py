"""AC13 — persistent app memory (PRD §6b): record screens + routes, recognise revisits
(`meta.known_screen`), flag drift/version as stale, redact values, stay under memory.dir.

Covered at three levels: the pure store (deterministic recognition/drift/redaction), the
engine auto-record pipeline (analyze + actions build the map), and the CLI (`aua map`).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import android_ui_analyser.engine as engine_mod
from android_ui_analyser import hierarchy
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import (
    AppMemoryStore,
    RouteStep,
    find_result,
    matches_any,
    render_map,
    signature,
)
from android_ui_analyser.platforms.android import package_from_tree
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

runner = CliRunner()
P = "com.example.app"


# --------------------------------------------------------------------------- fixtures


def _node(cls: str, *, text="", rid=None, desc=None, clk=False, b="[0,0][400,80]", pkg=P) -> str:
    attrs = [f'class="{cls}"', f'package="{pkg}"']
    if text:
        attrs.append(f'text="{text}"')
    if rid:
        attrs.append(f'resource-id="{rid}"')
    if desc:
        attrs.append(f'content-desc="{desc}"')
    attrs += [f'clickable="{str(clk).lower()}"', 'enabled="true"', f'bounds="{b}"']
    return "<node " + " ".join(attrs) + "/>"


def _hier(*nodes: str) -> str:
    return '<hierarchy rotation="0">' + "".join(nodes) + "</hierarchy>"


# Realistic full-screen layouts (1080x2400): header below the status-bar band, body
# buttons mid-screen, nothing in the bottom-nav band — exercises the chrome heuristics
# without clipping the fixture content.
HOME = _hier(
    _node("android.widget.TextView", text="Home", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button", text="Apps", rid="x:id/nav_apps", clk=True, b="[40,300][1040,400]"
    ),
    _node(
        "android.widget.Button",
        text="Orders",
        rid="x:id/nav_orders",
        clk=True,
        b="[40,440][1040,540]",
    ),
    _node(
        "android.widget.Button",
        text="Account",
        rid="x:id/nav_account",
        clk=True,
        b="[40,580][1040,680]",
    ),
)
APPS = _hier(
    _node("android.widget.TextView", text="Apps", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Reports",
        rid="x:id/tool_reports",
        clk=True,
        b="[40,300][1040,400]",
    ),
    _node(
        "android.widget.Button",
        text="Calendar",
        rid="x:id/tool_calendar",
        clk=True,
        b="[40,440][1040,540]",
    ),
    _node(
        "android.widget.Button",
        text="Notes",
        rid="x:id/tool_notes",
        clk=True,
        b="[40,580][1040,680]",
    ),
)


def _elements(xml: str):
    return hierarchy.parse_hierarchy(xml, (400, 800))


def _store(tmp_path: Path, **memov) -> AppMemoryStore:
    cfg = make_config(memory={"dir": str(tmp_path / "home"), **memov})
    return AppMemoryStore(cfg.memory)


def _engine(tmp_path: Path, device: FakeDevice, output: dict | None = None, **memov) -> Engine:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home"), **memov},
        daemon={"enabled": False},
        **({"output": output} if output else {}),
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _blob(tmp_path: Path) -> str:
    d = tmp_path / "home" / "memory" / P
    return (d / "index.json").read_text() + (d / "MAP.md").read_text()


# --------------------------------------------------------------- store: recognition/drift


def test_record_creates_then_recognises_revisit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    els = _elements(HOME)
    o1 = store.record_screen(package=P, elements=els, activity=".Main", app_version="1.0")
    assert o1.created and not o1.was_known and not o1.stale
    o2 = store.record_screen(package=P, elements=els, activity=".Main", app_version="1.0")
    assert o2.was_known and o2.name == o1.name and not o2.stale  # same, fresh
    am = store.load(P)
    assert am is not None
    assert am.screens[o1.name].signature  # a signature is recorded
    assert am.screens[o1.name].signature == signature(".Main", set(am.screens[o1.name].anchors))


def test_version_bump_marks_stale(tmp_path: Path) -> None:
    store = _store(tmp_path)
    els = _elements(HOME)
    o1 = store.record_screen(package=P, elements=els, activity=".Main", app_version="1.0")
    o2 = store.record_screen(package=P, elements=els, activity=".Main", app_version="2.0")
    assert o2.was_known and o2.stale
    assert store.load(P).screens[o1.name].stale is True


def test_signature_divergence_marks_stale_but_still_recognised(tmp_path: Path) -> None:
    store = _store(tmp_path, drift_threshold=0.3)
    o1 = store.record_screen(package=P, elements=_elements(HOME), activity=".Main")
    # Same screen + two added nav items: ~0.64 Jaccard → recognised, but divergence > 0.3.
    changed = _hier(
        _node("android.widget.TextView", text="Home", b="[0,0][400,80]"),
        _node(
            "android.widget.Button",
            text="Apps",
            rid="x:id/nav_apps",
            clk=True,
            b="[0,100][200,160]",
        ),
        _node(
            "android.widget.Button",
            text="Orders",
            rid="x:id/nav_orders",
            clk=True,
            b="[0,200][200,260]",
        ),
        _node(
            "android.widget.Button",
            text="Account",
            rid="x:id/nav_account",
            clk=True,
            b="[0,300][200,360]",
        ),
        _node(
            "android.widget.Button",
            text="World",
            rid="x:id/nav_world",
            clk=True,
            b="[0,400][200,460]",
        ),
        _node(
            "android.widget.Button",
            text="Profile",
            rid="x:id/nav_prof",
            clk=True,
            b="[0,500][200,560]",
        ),
    )
    o2 = store.record_screen(package=P, elements=_elements(changed), activity=".Main")
    assert o2.was_known and o2.name == o1.name  # recognised as the same screen
    assert o2.stale  # but flagged stale for re-verification


def test_a_screen_that_really_changed_re_anchors_once_the_new_shape_repeats(
    tmp_path: Path,
) -> None:
    """Drift must be recoverable, or a screen that legitimately changed is stale forever.

    Anchors used to be refreshed only on a *non-stale* visit, which is unreachable once a
    screen has drifted past the threshold: every later visit compares against the same old
    anchors, diverges again, and re-flags stale without ever adopting the new shape. This fixture
    models a catalog screen that has stayed pinned to old anchors through many repeat visits.

    Recovery is deliberately not "believe the first thing you see": a screen caught mid-video
    or with the keyboard open would otherwise adopt that transient state as its identity,
    which is what produced the bad anchors in the first place. The new shape has to show up
    twice, the same way a route has to replay before it is trusted.
    """

    store = _store(tmp_path, drift_threshold=0.3)
    o1 = store.record_screen(package=P, elements=_elements(HOME), activity=".Main")

    # Recognisable as the same screen (header + one surviving control) but well past the
    # 30% drift line, which is the shape a substantial navigation reshuffle takes.
    shipped = _hier(
        _node("android.widget.TextView", text="Home", rid="x:id/header", b="[40,120][1040,210]"),
        _node(
            "android.widget.Button",
            text="Apps",
            rid="x:id/nav_apps",
            clk=True,
            b="[40,300][1040,400]",
        ),
        _node(
            "android.widget.Button",
            text="Orders",
            rid="x:id/nav_orders",
            clk=True,
            b="[40,440][1040,540]",
        ),
        _node(
            "android.widget.Button",
            text="Offers",
            rid="x:id/nav_offers",
            clk=True,
            b="[40,580][1040,680]",
        ),
        _node(
            "android.widget.Button",
            text="Wishlist",
            rid="x:id/nav_wishlist",
            clk=True,
            b="[40,720][1040,820]",
        ),
    )

    o2 = store.record_screen(package=P, elements=_elements(shipped), activity=".Main")
    assert o2.name == o1.name, "still the same screen, just rebuilt"
    assert o2.stale, "first sighting of a new shape is not yet trusted"

    o3 = store.record_screen(package=P, elements=_elements(shipped), activity=".Main")
    assert o3.name == o1.name
    assert not o3.stale, "the new shape corroborated itself; the screen must re-anchor"

    rec = store.load(P).screens[o1.name]
    assert not rec.stale
    assert "id:nav_offers" in rec.anchors, f"anchors were never adopted: {rec.anchors}"
    assert "id:nav_account" not in rec.anchors, "the old shape must be dropped, not merged"


def test_a_one_off_odd_frame_does_not_become_the_screens_identity(tmp_path: Path) -> None:
    """The reason recovery needs corroboration rather than a single sighting.

    A screen caught with the keyboard up, a video playing, or a banner open diverges hard for
    exactly one visit. Adopting that would pin the screen's identity to furniture that is
    normally absent — for example, a form can be anchored on keyboard-only controls and then
    never match itself again.
    """

    store = _store(tmp_path, drift_threshold=0.3)
    o1 = store.record_screen(package=P, elements=_elements(HOME), activity=".Main")

    with_keyboard = _hier(
        _node("android.widget.TextView", text="Home", rid="x:id/header", b="[40,120][1040,210]"),
        _node(
            "android.widget.Button",
            text="Apps",
            rid="x:id/nav_apps",
            clk=True,
            b="[40,300][1040,400]",
        ),
        _node(
            "android.widget.Button",
            text="Delete",
            rid="x:id/key_del",
            clk=True,
            b="[0,1600][200,1700]",
        ),
        _node(
            "android.widget.Button",
            text="Enter",
            rid="x:id/key_enter",
            clk=True,
            b="[200,1600][400,1700]",
        ),
        _node(
            "android.widget.Button",
            text="Voice",
            rid="x:id/key_voice",
            clk=True,
            b="[400,1600][600,1700]",
        ),
    )

    assert store.record_screen(package=P, elements=_elements(with_keyboard), activity=".Main").stale

    back = store.record_screen(package=P, elements=_elements(HOME), activity=".Main")
    assert not back.stale, "the original shape returned, so the screen is healthy again"

    rec = store.load(P).screens[o1.name]
    assert "id:key_voice" not in rec.anchors, f"transient keyboard became identity: {rec.anchors}"
    assert "id:nav_orders" in rec.anchors, "the durable anchors must survive"


# --------------------------------------------------------------- store: redaction (privacy)


def test_edittext_value_never_stored_verbatim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    secret_notes = "confidential quarterly figures draft"
    xml = _hier(
        _node("android.widget.TextView", text="Create", b="[0,0][400,80]"),
        _node(
            "android.widget.EditText",
            text=secret_notes,
            rid="x:id/notes",
            desc="Report notes",
            clk=True,
            b="[0,100][400,200]",
        ),
        _node(
            "android.widget.Button", text="Save", rid="x:id/save", clk=True, b="[0,220][200,280]"
        ),
    )
    store.record_screen(package=P, elements=_elements(xml), activity=".Create")
    blob = _blob(tmp_path)
    assert secret_notes not in blob  # the typed value is NOT persisted
    assert "<filled>" in blob  # stored only as a shape
    assert "Report notes" in blob  # the durable hint label IS kept


def test_secret_and_pii_fields_are_redacted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    xml = _hier(
        _node(
            "android.widget.TextView",
            text="john.doe@example.com",
            rid="x:id/email_label",
            clk=True,
            b="[0,0][400,40]",
        ),
        _node(
            "android.widget.EditText",
            text="hunter2",
            rid="x:id/password",
            desc="Password",
            clk=True,
            b="[0,100][400,160]",
        ),
    )
    store.record_screen(package=P, elements=_elements(xml), activity=".Login")
    blob = _blob(tmp_path)
    assert "john.doe@example.com" not in blob  # PII redacted
    assert "hunter2" not in blob  # secret value never stored
    assert "<redacted>" in blob


# --------------------------------------------------------------- store: writes stay home


def test_all_writes_stay_under_memory_dir(tmp_path: Path) -> None:
    memdir = tmp_path / "home"
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), activity=".Main")
    store.save_session("emu-x", store.load_session("emu-x"))
    files = [p for p in memdir.rglob("*") if p.is_file()]
    assert files, "expected memory writes"
    for p in files:
        assert str(p.resolve()).startswith(str(memdir.resolve()))
    assert (memdir / "memory" / P / "MAP.md").is_file()
    assert (memdir / "memory" / P / "index.json").is_file()


# --------------------------------------------------------------- engine: auto-record path


def test_engine_builds_map_and_sets_known_screen(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-1", app_version="3.2")
    eng = _engine(tmp_path, dev)

    r1 = eng.analyze(source="hierarchy")
    assert r1.meta.known_screen is None  # first visit → newly recorded, not yet "known"
    apps_id = next(e.id for e in r1.elements if e.text == "Apps")
    eng.tap(apps_id)

    dev._xml = APPS
    r2 = eng.analyze(source="hierarchy")
    assert r2.meta.known_screen is None  # apps first visit

    dev._xml = HOME
    r3 = eng.analyze(source="hierarchy")
    assert r3.meta.known_screen == "home"  # revisit recognised

    store = AppMemoryStore(eng.config.memory)
    am = store.load(P)
    assert set(am.screens) == {"home", "apps"}
    assert all(s.signature for s in am.screens.values())
    edges = [(e.from_screen, e.to_screen) for e in am.routes]
    assert ("home", "apps") in edges
    assert any("Apps" in e.action for e in am.routes)

    text = render_map(am, detail="default")
    assert "home" in text and "apps" in text and "Apps" in text
    fr = find_result(am, "report")
    assert fr["results"] and fr["results"][0]["route"] == []


def test_find_does_not_present_a_provisional_route_as_runnable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    home = store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    apps = store.record_screen(package=P, elements=_elements(APPS), name_hint="catalog")
    store.record_route(
        P,
        home.name,
        apps.name,
        steps=[RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog")],
        verified=False,
    )

    result = find_result(store.load(P), "catalog")

    assert result["results"][0]["route"] == []
    rendered = render_map(store.load(P), find="catalog")
    assert "route: (no verified route)" in rendered

    store.record_route(
        P,
        home.name,
        apps.name,
        steps=[RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog")],
        verified=True,
    )
    assert find_result(store.load(P), "catalog")["results"][0]["route"]


def test_engine_input_action_does_not_store_typed_value(tmp_path: Path) -> None:
    # Synthetic form flow: type private notes on a create screen → submit → result screen.
    create = _hier(
        _node(
            "android.widget.TextView", text="New report", rid="x:id/header", b="[40,120][1040,210]"
        ),
        _node(
            "android.widget.EditText",
            rid="x:id/notes",
            desc="Report notes",
            clk=True,
            b="[40,400][1040,560]",
        ),
        _node(
            "android.widget.Button", text="Save", rid="x:id/save", clk=True, b="[40,640][400,740]"
        ),
    )
    result = _hier(
        _node(
            "android.widget.TextView", text="Report ready", rid="x:id/title", b="[40,120][1040,210]"
        ),
        _node(
            "android.widget.Button", text="Share", rid="x:id/share", clk=True, b="[40,400][400,500]"
        ),
    )
    dev = FakeDevice(hierarchy_xml=create, package=P, serial="emu-2")
    eng = _engine(tmp_path, dev)
    r1 = eng.analyze(source="hierarchy")
    notes_id = next(e.id for e in r1.elements if e.content_desc == "Report notes")
    eng.input_text(notes_id, "top secret report notes", submit=True)
    dev._xml = result  # submit navigated to the result screen
    eng.analyze(source="hierarchy")

    blob = _blob(tmp_path)
    assert "top secret report notes" not in blob  # the typed value is never persisted
    assert "<filled>" in blob  # the route action records only the shape
    am = AppMemoryStore(eng.config.memory).load(P)
    assert any("filled" in e.action for e in am.routes)  # submitted input edge


# --------------------------------------------------------------- CLI: aua map


def test_cli_map_lists_screens_routes_and_find(tmp_path, monkeypatch) -> None:
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-cli")
    monkeypatch.setattr(engine_mod.Engine, "_connect_target", lambda _engine, serial=None: dev)

    a1 = runner.invoke(app, ["--format", "compact", "analyze", "--source", "hierarchy"])
    assert a1.exit_code == 0, a1.stderr
    apps_id = next(e["id"] for e in json.loads(a1.stdout)["elements"] if e.get("text") == "Apps")
    assert runner.invoke(app, ["tap-and-analyze", str(apps_id)]).exit_code == 0
    dev._xml = APPS
    assert runner.invoke(app, ["analyze", "--source", "hierarchy"]).exit_code == 0
    dev._xml = HOME
    r3 = runner.invoke(app, ["--format", "compact", "analyze", "--source", "hierarchy"])
    assert json.loads(r3.stdout)["meta"].get("known_screen") == "home"

    m = runner.invoke(app, ["map", "--app", P, "--json"])
    assert m.exit_code == 0, m.stderr
    data = json.loads(m.stdout)
    assert {"home", "apps"} <= set(data["screens"])
    assert all(s["signature"] for s in data["screens"].values())
    assert any(e["from_screen"] == "home" and e["to_screen"] == "apps" for e in data["routes"])

    f = runner.invoke(app, ["map", "--app", P, "--find", "report", "--json"])
    assert f.exit_code == 0, f.stderr
    fr = json.loads(f.stdout)
    assert fr["results"] and fr["results"][0]["route"] == []

    # text tree (default) also names screens + routes
    t = runner.invoke(app, ["map", "--app", P])
    assert t.exit_code == 0 and "home" in t.stdout and "apps" in t.stdout

    # Agents can ask for counts without receiving every question/task. The existing JSON
    # contract remains the full evidence unless --summary is explicit.
    summary = runner.invoke(app, ["map", "--app", P, "--audit", "--summary", "--json"])
    assert summary.exit_code == 0, summary.stderr
    health = json.loads(summary.stdout)
    assert set(health["issues"]["by_severity"]) == {"error", "warning", "info"}
    assert "by_type" in health["issues"]
    assert {"total", "open", "by_status"} <= set(health["research_tasks"])

    full_audit = runner.invoke(app, ["map", "--app", P, "--audit", "--json"])
    assert full_audit.exit_code == 0, full_audit.stderr
    evidence = json.loads(full_audit.stdout)
    assert isinstance(evidence["issues"], list), "full JSON must stay backward-compatible"
    assert isinstance(evidence["research_tasks"], list)


# --------------------------------------------------------------- package hygiene


def test_package_vote_ignores_ime() -> None:
    """An open keyboard dominating the dump must not win the foreground vote."""
    ime = "com.google.android.inputmethod.latin"
    xml = _hier(
        _node("android.widget.TextView", text="Hi", b="[40,120][1040,210]"),
        _node("android.view.View", pkg=ime, b="[0,1200][1080,1500]"),
        _node("android.view.View", pkg=ime, b="[0,1500][1080,1800]"),
        _node("android.view.View", pkg=ime, b="[0,1800][1080,2100]"),
    )
    assert package_from_tree(xml, ["com.android.systemui", "*inputmethod*"]) == P


def test_package_vote_all_ignored_falls_back_to_majority() -> None:
    xml = _hier(
        _node("android.view.View", pkg="com.android.systemui"),
        _node("android.view.View", pkg="com.android.systemui"),
    )
    assert package_from_tree(xml, ["com.android.systemui"]) == "com.android.systemui"


def test_matches_any_globs() -> None:
    globs = ["com.android.systemui", "*inputmethod*"]
    assert matches_any("com.android.systemui", globs)
    assert matches_any("com.google.android.inputmethod.latin", globs)
    assert matches_any("COM.GOOGLE.ANDROID.INPUTMETHOD.LATIN", globs)
    assert not matches_any("com.example.app", globs)
    assert not matches_any(None, globs)


def test_ignored_package_records_no_map(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ime = "com.google.android.inputmethod.latin"
    known = store.observe_screen(
        "serial-x", package=ime, elements=_elements(HOME), screen_height=800
    )
    assert known is None
    assert not (tmp_path / "home" / "memory" / ime).exists()
    assert store.list_apps() == []


# --------------------------------------------------------------- structured steps (v2)


SETTINGS_XML = _hier(
    _node("android.widget.TextView", text="Preferences", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Account",
        rid="x:id/account",
        clk=True,
        b="[40,300][1040,400]",
    ),
)
HOME_WITH_GEAR = _hier(
    _node("android.widget.TextView", text="Home", rid="x:id/header", b="[40,120][1040,210]"),
    _node("android.view.View", rid="x:id/buttonSettings", clk=True, b="[900,300][1040,400]"),
    _node(
        "android.widget.Button",
        text="Orders",
        rid="x:id/nav_orders",
        clk=True,
        b="[40,440][1040,540]",
    ),
)


def test_edge_records_structured_steps(tmp_path: Path) -> None:
    dev = FakeDevice(hierarchy_xml=HOME)
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    apps_id = next(e.id for e in res.elements if e.text == "Apps")
    eng.tap(apps_id, observe=False)
    dev._xml = APPS
    eng.analyze(source="hierarchy")

    am = AppMemoryStore(eng.config.memory).load(P)
    assert am is not None and am.schema_version == 4
    edge = next(e for e in am.routes if e.to_screen == "apps")
    assert edge.action == "tap 'Apps'"  # display string unchanged from v1
    assert len(edge.steps) == 1
    s = edge.steps[0]
    assert (s.kind, s.label, s.resource_id) == ("tap", "Apps", "nav_apps")
    assert s.package is None  # origin-package steps are normalized to None
    assert s.text is None


def test_unlabeled_id_tap_is_displayable_and_findable(tmp_path: Path) -> None:
    """An unlabeled settings gear must not be recorded as `tap [View]`."""
    from android_ui_analyser.memory import _find_targets

    dev = FakeDevice(hierarchy_xml=HOME_WITH_GEAR)
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    gear_id = next(e.id for e in res.elements if e.resource_id == "x:id/buttonSettings")
    eng.tap(gear_id, observe=False)
    dev._xml = SETTINGS_XML
    eng.analyze(source="hierarchy")

    am = AppMemoryStore(eng.config.memory).load(P)
    edge = next(e for e in am.routes if e.to_screen == "preferences")
    assert edge.action == "tap [#buttonSettings]"
    assert edge.steps[0].resource_id == "buttonSettings"
    assert "preferences" in _find_targets(am, "settings")  # findable via the id tail


def test_record_route_upgrades_legacy_edge_in_place(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(P, "home", "apps", "tap 'Apps'")  # legacy positional string
    am = store.load(P)
    assert am.routes[0].steps == []
    store.record_route(
        P, "home", "apps", steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")]
    )
    am = store.load(P)
    assert len(am.routes) == 1  # same derived action -> same edge
    assert am.routes[0].count == 2
    assert am.routes[0].steps and am.routes[0].steps[0].resource_id == "nav_apps"


def test_pending_overflow_drops_edge(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    serial = "s1"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    for i in range(13):
        store.observe_action(serial, RouteStep(kind="tap", label=f"B{i}"))
    sess = store.load_session(serial)
    assert len(sess.pending) == 12 and sess.pending_overflow is True
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)
    am = store.load(P)
    assert am.routes == []  # truncated sequences never become edges
    sess = store.load_session(serial)
    assert sess.pending == [] and sess.pending_overflow is False


def test_pending_ttl_drops_stale_edge(tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    serial = "s1"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="tap", label="Apps"))
    sess = store.load_session(serial)
    stale = (datetime.now().astimezone() - timedelta(seconds=700)).isoformat(timespec="seconds")
    sess.pending_since = stale
    store.save_session(serial, sess)
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)
    assert store.load(P).routes == []  # abandoned journeys don't smear edges


def test_recent_journal_survives_analyze(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    serial = "s1"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="tap", label="Apps"))
    store.observe_action(serial, RouteStep(kind="key", arg="back"))
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)
    sess = store.load_session(serial)
    assert sess.pending == []  # consumed by the analyze
    assert [s.kind for s in sess.recent] == ["tap", "key"]  # journal untouched


def test_legacy_string_pending_is_dropped_on_load(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.session_path("s1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"package": "p", "current_screen": "home", "pending": ["tap \'x\'"], "last_goal": null}',
        encoding="utf-8",
    )
    sess = store.load_session("s1")
    assert sess.pending == [] and sess.current_screen == "home"


def test_v1_map_loads_and_saves_as_v4(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    idx = store.index_path(P)
    data = json.loads(idx.read_text())
    data["schema_version"] = 1
    idx.write_text(json.dumps(data))
    am = store.load(P)
    assert am is not None  # loads version-agnostically
    store.record_screen(package=P, elements=_elements(HOME))
    assert json.loads(idx.read_text())["schema_version"] == 4


def test_generic_inbound_label_defers_to_title(tmp_path: Path) -> None:
    """A screen reached via tap 'Delete' is named from its title, not 'delete'."""
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    serial = "s1"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="tap", label="Delete"))
    store.observe_screen(serial, package=P, elements=_elements(SETTINGS_XML), screen_height=800)
    am = store.load(P)
    assert "preferences" in am.screens
    assert "delete" not in am.screens


def test_observe_action_strips_typed_text(tmp_path: Path) -> None:
    from android_ui_analyser.memory import RouteStep

    store = _store(tmp_path)
    store.observe_action("s1", RouteStep(kind="input", label="Report notes", text="private draft"))
    sess = store.load_session("s1")
    assert sess.pending[0].text is None and sess.recent[0].text is None


def test_is_destructive_step_word_boundaries() -> None:
    from android_ui_analyser.memory import RouteStep, is_destructive_step

    lex = ["delete", "sign out"]
    assert is_destructive_step(RouteStep(kind="tap", label="Delete my account"), lex)
    assert is_destructive_step(RouteStep(kind="long-press", label="Sign out"), lex)
    assert not is_destructive_step(RouteStep(kind="tap", label="Deleted files"), lex)
    assert not is_destructive_step(RouteStep(kind="scroll-to", arg="Delete"), lex)
    assert not is_destructive_step(RouteStep(kind="tap", label="<redacted>"), lex)
