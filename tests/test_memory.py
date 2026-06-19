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
from android_ui_analyser.memory import AppMemoryStore, find_result, render_map, signature
from android_ui_analyser.providers.registry import ProviderFactory
from conftest import FakeDevice, make_config

runner = CliRunner()
P = "co.thewordlab.luzia"


# --------------------------------------------------------------------------- fixtures


def _node(cls: str, *, text="", rid=None, desc=None, clk=False, b="[0,0][400,80]") -> str:
    attrs = [f'class="{cls}"', f'package="{P}"']
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
        "android.widget.Button", text="Chat", rid="x:id/nav_chat", clk=True, b="[40,440][1040,540]"
    ),
    _node(
        "android.widget.Button",
        text="Ideas",
        rid="x:id/nav_ideas",
        clk=True,
        b="[40,580][1040,680]",
    ),
)
APPS = _hier(
    _node("android.widget.TextView", text="Apps", rid="x:id/header", b="[40,120][1040,210]"),
    _node(
        "android.widget.Button",
        text="Images",
        rid="x:id/tool_images",
        clk=True,
        b="[40,300][1040,400]",
    ),
    _node(
        "android.widget.Button",
        text="Games",
        rid="x:id/tool_games",
        clk=True,
        b="[40,440][1040,540]",
    ),
    _node(
        "android.widget.Button",
        text="Summarize",
        rid="x:id/tool_sum",
        clk=True,
        b="[40,580][1040,680]",
    ),
)


def _elements(xml: str):
    return hierarchy.parse_hierarchy(xml, (400, 800))


def _store(tmp_path: Path, **memov) -> AppMemoryStore:
    cfg = make_config(memory={"dir": str(tmp_path / "home"), **memov})
    return AppMemoryStore(cfg.memory)


def _engine(tmp_path: Path, device: FakeDevice, **memov) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home"), **memov}, daemon={"enabled": False})
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
            text="Chat",
            rid="x:id/nav_chat",
            clk=True,
            b="[0,200][200,260]",
        ),
        _node(
            "android.widget.Button",
            text="Ideas",
            rid="x:id/nav_ideas",
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


# --------------------------------------------------------------- store: redaction (privacy)


def test_edittext_value_never_stored_verbatim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    secret_prompt = "a serene mountain lake at golden hour ultra detailed"
    xml = _hier(
        _node("android.widget.TextView", text="Create", b="[0,0][400,80]"),
        _node(
            "android.widget.EditText",
            text=secret_prompt,
            rid="x:id/prompt",
            desc="Prompt",
            clk=True,
            b="[0,100][400,200]",
        ),
        _node(
            "android.widget.Button", text="Send", rid="x:id/send", clk=True, b="[0,220][200,280]"
        ),
    )
    store.record_screen(package=P, elements=_elements(xml), activity=".Create")
    blob = _blob(tmp_path)
    assert secret_prompt not in blob  # the typed value is NOT persisted
    assert "<filled>" in blob  # stored only as a shape
    assert "Prompt" in blob  # the durable hint label IS kept


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
    fr = find_result(am, "image")
    assert fr["results"] and fr["results"][0]["route"]  # route to the image tool


def test_engine_input_action_does_not_store_typed_value(tmp_path: Path) -> None:
    # Mirrors the live flow: type a prompt on the create screen → send → result screen.
    create = _hier(
        _node("android.widget.TextView", text="Create", rid="x:id/header", b="[40,120][1040,210]"),
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
    result = _hier(
        _node(
            "android.widget.TextView", text="Your image", rid="x:id/title", b="[40,120][1040,210]"
        ),
        _node(
            "android.widget.Button", text="Share", rid="x:id/share", clk=True, b="[40,400][400,500]"
        ),
    )
    dev = FakeDevice(hierarchy_xml=create, package=P, serial="emu-2")
    eng = _engine(tmp_path, dev)
    r1 = eng.analyze(source="hierarchy")
    prompt_id = next(e.id for e in r1.elements if e.content_desc == "Prompt")
    eng.input_text(prompt_id, "top secret user prompt text", submit=True)
    dev._xml = result  # send navigated to the result screen
    eng.analyze(source="hierarchy")

    blob = _blob(tmp_path)
    assert "top secret user prompt text" not in blob  # the typed value is never persisted
    assert "<filled>" in blob  # the route action records only the shape
    am = AppMemoryStore(eng.config.memory).load(P)
    assert any("filled" in e.action for e in am.routes)  # 'input <filled> + send' edge


# --------------------------------------------------------------- CLI: aua map


def test_cli_map_lists_screens_routes_and_find(tmp_path, monkeypatch) -> None:
    dev = FakeDevice(hierarchy_xml=HOME, package=P, serial="emu-cli")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    a1 = runner.invoke(app, ["--format", "compact", "analyze", "--source", "hierarchy"])
    assert a1.exit_code == 0, a1.stderr
    apps_id = next(e["id"] for e in json.loads(a1.stdout)["elements"] if e.get("text") == "Apps")
    assert runner.invoke(app, ["tap", str(apps_id)]).exit_code == 0
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

    f = runner.invoke(app, ["map", "--app", P, "--find", "image", "--json"])
    assert f.exit_code == 0, f.stderr
    fr = json.loads(f.stdout)
    assert fr["results"] and fr["results"][0]["route"]

    # text tree (default) also names screens + routes
    t = runner.invoke(app, ["map", "--app", P])
    assert t.exit_code == 0 and "home" in t.stdout and "apps" in t.stdout
