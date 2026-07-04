"""Named flows (PRD §6b): YAML parse/render, params, one-call replay, save, CLI, daemon.

A flow is the agent-authored (or `flow save`-materialized) Maestro-style journey; the
executor is shared with `goto`, so these tests focus on the flow-specific surface:
parsing, ${PARAM} substitution, divergence + `--from-step` resume, privacy of saved
files, and the plumbing (CLI group, daemon dispatch).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import (
    Flow,
    FlowStore,
    parse_flow_yaml,
    render_flow_yaml,
    resolve_params,
    steps_from_recent,
)
from android_ui_analyser.memory import AppMemoryStore, RouteStep
from conftest import make_config
from test_memory import APPS, HOME, P, _elements, _engine, _hier, _node, _store
from test_navigation import ScriptedDevice

runner = CliRunner()

FLOW_YAML = """
name: open_images
app: co.thewordlab.luzia
params:
  TOOL: "Images"
steps:
  - launch_app: co.thewordlab.luzia
  - tap: "Apps"
  - tap: {text: "${TOOL}"}
"""


# --------------------------------------------------------------------------- parsing


def test_parse_flow_yaml_shorthand_and_mapping() -> None:
    flow = parse_flow_yaml(FLOW_YAML)
    assert flow.name == "open_images" and flow.app == P
    assert [s.kind for s in flow.steps] == ["launch-app", "tap", "tap"]
    assert flow.steps[0].arg == P
    assert flow.steps[1].label == "Apps"
    assert flow.steps[2].label == "${TOOL}"


def test_parse_flow_yaml_full_vocabulary() -> None:
    text = """
steps:
  - tap: {id: buttonSettings}
  - long_press: "Row"
  - input: {id: prompt, text: "${Q}", submit: true}
  - clear: {id: prompt}
  - key: back
  - swipe: up
  - scroll_to: "Translate"
  - wait_for: {text: "Done", timeout_ms: 5000}
  - wait_stable
  - assert_visible: "Done"
  - goto: images
  - tap: {text: "Continue", package: com.android.chrome}
"""
    flow = parse_flow_yaml(text, name="vocab")
    kinds = [s.kind for s in flow.steps]
    assert kinds == [
        "tap",
        "long-press",
        "input",
        "clear",
        "key",
        "swipe",
        "scroll-to",
        "wait-for",
        "wait-stable",
        "assert-visible",
        "goto",
        "tap",
    ]
    assert flow.steps[0].resource_id == "buttonSettings"
    assert flow.steps[2].submit is True and flow.steps[2].text == "${Q}"
    assert flow.steps[7].timeout_ms == 5000
    assert flow.steps[11].package == "com.android.chrome"


def test_parse_flow_yaml_errors_name_the_step() -> None:
    for text, needle in [
        ("steps:\n  - frobnicate: x\n", "unknown step kind"),
        ("steps:\n  - input: {id: f}\n", "needs `text:`"),
        ("steps:\n  - tap: {}\n", "selector"),
        ("steps: []\n", "non-empty"),
        ("- just\n- a list\n", "mapping"),
        ("steps:\n  - tap: {text: x, bogus: 1}\n", "unknown keys"),
    ]:
        try:
            parse_flow_yaml(text)
            raise AssertionError(f"expected UsageError for {text!r}")
        except UsageError as exc:
            assert needle in str(exc)


def test_render_round_trips() -> None:
    flow = parse_flow_yaml(FLOW_YAML)
    again = parse_flow_yaml(render_flow_yaml(flow))
    assert [s.model_dump() for s in again.steps] == [s.model_dump() for s in flow.steps]
    assert again.params == flow.params


# --------------------------------------------------------------------------- params


def test_resolve_params_defaults_and_overrides() -> None:
    flow = parse_flow_yaml(FLOW_YAML)
    steps = resolve_params(flow, {})
    assert steps[2].label == "Images"  # declared default applies
    steps = resolve_params(flow, {"TOOL": "Games"})
    assert steps[2].label == "Games"  # override wins


def test_resolve_params_missing_required_raises() -> None:
    flow = parse_flow_yaml("params: {ACCOUNT: \"\"}\nsteps:\n  - tap: \"${ACCOUNT}\"\n")
    try:
        resolve_params(flow, {})
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "ACCOUNT" in str(exc)
    steps = resolve_params(flow, {"ACCOUNT": "Engineering Team"})
    assert steps[0].label == "Engineering Team"


def test_resolve_params_undeclared_placeholder_raises() -> None:
    flow = parse_flow_yaml("steps:\n  - tap: \"${NOPE}\"\n")
    try:
        resolve_params(flow, {})
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "NOPE" in str(exc)


# --------------------------------------------------------------------------- flow_run


def _images_flow_text() -> str:
    return """
name: to_images
app: co.thewordlab.luzia
steps:
  - tap: "Apps"
  - tap: "Images"
  - assert_visible: "Generate"
"""


IMAGES = _hier(
    _node(
        "android.widget.TextView", text="Create image", rid="x:id/header", b="[40,120][1040,210]"
    ),
    _node("android.widget.Button", text="Generate", rid="x:id/go", clk=True, b="[40,640][400,740]"),
)


def test_flow_run_from_file_drives_whole_journey(tmp_path: Path) -> None:
    flow_file = tmp_path / "to_images.yaml"
    flow_file.write_text(_images_flow_text(), encoding="utf-8")
    dev = ScriptedDevice(
        [HOME, APPS, IMAGES],
        package=P,
        serial="emu-flow",
        text_index={"Generate": (40, 640, 400, 740)},
    )
    eng = _engine(tmp_path, dev)
    out = eng.flow_run(file=str(flow_file))
    assert out["ok"] is True, out
    assert out["flow"] == "to_images"
    assert [s["index"] for s in out["steps_run"]] == [0, 1, 2]
    assert sum(1 for c in dev.calls if c[0] == "click") == 2
    assert out["elements"]  # destination ids returned for immediate use


def test_flow_run_dry_run_touches_nothing(tmp_path: Path) -> None:
    flow_file = tmp_path / "f.yaml"
    flow_file.write_text(_images_flow_text(), encoding="utf-8")
    dev = ScriptedDevice([HOME], package=P, serial="emu-dry")
    eng = _engine(tmp_path, dev)
    out = eng.flow_run(file=str(flow_file), dry_run=True)
    assert out["ok"] and out["dry_run"] and len(out["steps"]) == 3
    assert dev.calls == []  # zero device interaction


def test_flow_run_divergence_hands_off_and_resumes(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory)
    flow = parse_flow_yaml(
        """
name: bumpy
app: co.thewordlab.luzia
steps:
  - tap: "Apps"
  - tap: "No Such Button"
  - tap: "Games"
""",
        name="bumpy",
    )
    store.save(flow)
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-bumpy")
    eng = _engine(tmp_path, dev)
    out = eng.flow_run("bumpy")
    assert out["ok"] is False and out["code"] == "element_not_found"
    assert out["step_index"] == 1
    assert out["failed_step"]["display"] == "tap 'No Such Button'"
    assert out["remaining_steps"][0] == "tap 'No Such Button'"
    assert "--from-step 1" in out["hint"]
    assert out["elements"]

    resumed = eng.flow_run("bumpy", from_step=2)  # skip the broken step
    assert resumed["ok"] is True
    assert [s["index"] for s in resumed["steps_run"]] == [2]


def test_flow_run_allows_destructive_by_default_unlike_goto(tmp_path: Path) -> None:
    danger = _hier(
        _node("android.widget.TextView", text="Confirm", rid="x:id/h", b="[40,120][1040,210]"),
        _node(
            "android.widget.Button",
            text="Delete my account",
            rid="x:id/del",
            clk=True,
            b="[40,300][1040,400]",
        ),
    )
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(danger), name_hint="confirm")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(
        P, "confirm", "apps", steps=[RouteStep(kind="tap", label="Delete my account")]
    )

    flow_file = tmp_path / "reset.yaml"
    flow_file.write_text(
        "name: reset\napp: co.thewordlab.luzia\nsteps:\n  - tap: \"Delete my account\"\n",
        encoding="utf-8",
    )

    dev = ScriptedDevice([danger, APPS], package=P, serial="emu-danger")
    eng = _engine(tmp_path, dev)
    refused = eng.goto("apps")  # auto-learned replay refuses
    assert refused["ok"] is False and refused["code"] == "destructive_step"

    ran = eng.flow_run(file=str(flow_file))  # authored intent proceeds
    assert ran["ok"] is True
    assert sum(1 for c in dev.calls if c[0] == "click") == 1

    dev2 = ScriptedDevice([danger, APPS], package=P, serial="emu-danger2")
    eng2 = _engine(tmp_path, dev2)
    blocked = eng2.flow_run(file=str(flow_file), allow_destructive=False)
    assert blocked["ok"] is False and blocked["code"] == "destructive_step"


def test_flow_run_goto_step_composes_map_navigation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    store.record_route(
        P, "home", "apps", steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")]
    )
    flow_file = tmp_path / "via_goto.yaml"
    flow_file.write_text(
        "name: via_goto\napp: co.thewordlab.luzia\nsteps:\n  - goto: apps\n", encoding="utf-8"
    )
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-goto-step")
    eng = _engine(tmp_path, dev)
    out = eng.flow_run(file=str(flow_file))
    assert out["ok"] is True
    assert sum(1 for c in dev.calls if c[0] == "click") == 1


def test_flow_run_launch_app_step(tmp_path: Path) -> None:
    flow_file = tmp_path / "l.yaml"
    flow_file.write_text(
        f"name: l\napp: {P}\nsteps:\n  - launch_app: {P}\n  - assert_visible: \"Home\"\n",
        encoding="utf-8",
    )
    dev = ScriptedDevice([HOME], package=P, serial="emu-launch", text_index={"Home": (0, 0, 9, 9)})
    eng = _engine(tmp_path, dev)
    out = eng.flow_run(file=str(flow_file))
    assert out["ok"] is True
    assert ("launch_app", (P,)) in dev.calls


# --------------------------------------------------------------------------- flow_save


def test_flow_save_materializes_recent_with_placeholders(tmp_path: Path) -> None:
    prompt_screen = _hier(
        _node("android.widget.TextView", text="Create", rid="x:id/h", b="[40,120][1040,210]"),
        _node(
            "android.widget.EditText",
            rid="x:id/prompt",
            desc="Prompt",
            clk=True,
            b="[40,300][1040,400]",
        ),
        _node(
            "android.widget.Button", text="Send", rid="x:id/send", clk=True, b="[40,440][400,540]"
        ),
    )
    dev = ScriptedDevice([prompt_screen], package=P, serial="emu-save")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    field = next(e.id for e in res.elements if e.resource_id == "x:id/prompt")
    eng.input_text(field, "a very secret prompt", observe=False)
    res = eng.analyze(source="hierarchy")  # ids are invalidated by every action
    send = next(e.id for e in res.elements if e.text == "Send")
    eng.tap(send, observe=False)

    out = eng.flow_save("my_flow")
    assert out["ok"] and out["steps"] == 2 and out["params_needed"] == ["PARAM_1"]
    saved = Path(out["path"]).read_text(encoding="utf-8")
    assert "a very secret prompt" not in saved  # privacy: typed values never persisted
    assert "${PARAM_1}" in saved
    assert "Send" in saved


def test_flow_save_requires_force_to_overwrite(tmp_path: Path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-save2")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)
    assert eng.flow_save("dup")["ok"]
    try:
        eng.flow_save("dup")
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "--force" in (exc.hint or "")
    assert eng.flow_save("dup", force=True)["ok"]


def test_steps_from_recent_parameterizes_redacted_labels() -> None:
    recent = [
        RouteStep(kind="tap", label="Apps"),
        RouteStep(kind="tap", label="<redacted>"),
        RouteStep(kind="input", label="Prompt", resource_id="prompt"),
    ]
    steps, params = steps_from_recent(recent)
    assert steps[0].label == "Apps"
    assert steps[1].label == "${PARAM_1}"
    assert steps[2].text == "${PARAM_2}"
    assert sorted(params) == ["PARAM_1", "PARAM_2"]


# --------------------------------------------------------------------------- store + CLI


def test_flow_store_save_load_list_delete(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}).memory
    store = FlowStore(cfg)
    flow = Flow(name="alpha", app=P, steps=[RouteStep(kind="tap", label="Apps")])
    path = store.save(flow)
    assert path.is_file()
    assert store.load("alpha").steps[0].label == "Apps"
    listed = store.list()
    assert listed and listed[0]["name"] == "alpha" and listed[0]["steps"] == 1
    assert store.delete("alpha") is True
    assert store.delete("alpha") is False


def test_cli_flow_list_show_delete(tmp_path: Path) -> None:
    # The CLI reads memory.dir from the autouse-isolated env config.
    store = FlowStore(make_config().memory)
    store.save(Flow(name="cli_flow", app=P, steps=[RouteStep(kind="key", arg="back")]))

    listed = runner.invoke(app, ["flow", "list"])
    assert listed.exit_code == 0, listed.stderr
    assert json.loads(listed.stdout)["flows"][0]["name"] == "cli_flow"

    shown = runner.invoke(app, ["flow", "show", "cli_flow"])
    assert shown.exit_code == 0 and "key: back" in shown.stdout

    gone = runner.invoke(app, ["flow", "delete", "cli_flow"])
    assert gone.exit_code == 0 and json.loads(gone.stdout)["ok"] is True
    assert runner.invoke(app, ["flow", "delete", "cli_flow"]).exit_code == 1


def test_cli_flow_run_dry_run(tmp_path: Path, monkeypatch) -> None:
    import android_ui_analyser.engine as engine_mod

    store = FlowStore(make_config().memory)
    store.save(Flow(name="dry", app=P, steps=[RouteStep(kind="tap", label="Apps")]))
    dev = ScriptedDevice([HOME], package=P, serial="emu-cli-dry")
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    out = runner.invoke(app, ["--format", "compact", "flow", "run", "dry", "--dry-run"])
    assert out.exit_code == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["dry_run"] and data["steps"][0]["step"] == "tap 'Apps'"
    assert dev.calls == []


def test_cli_flow_run_param_parsing_error() -> None:
    out = runner.invoke(app, ["flow", "run", "x", "--param", "not-a-pair"])
    assert out.exit_code != 0


# --------------------------------------------------------------------------- daemon


def test_daemon_dispatch_flow_run_and_save() -> None:
    class FakeEng:
        def flow_run(self, **kw: object) -> dict[str, object]:
            return {"ok": True, "flow": kw.get("name")}

        def flow_save(self, **kw: object) -> dict[str, object]:
            return {"ok": True, "flow": kw.get("name"), "action": "flow-save"}

    r = dispatch(FakeEng(), {"cmd": "flow_run", "args": {"name": "f", "dry_run": True}})
    assert r["ok"] and r["result"]["flow"] == "f"
    r2 = dispatch(FakeEng(), {"cmd": "flow_save", "args": {"name": "f"}})
    assert r2["ok"] and r2["result"]["action"] == "flow-save"


def test_engine_flow_save_uses_session_store(tmp_path: Path) -> None:
    """flow_save reads the same session journal observe_action writes."""
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-journal")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    apps_id = next(e.id for e in res.elements if e.text == "Apps")
    eng.tap(apps_id, observe=False)
    eng.analyze(source="hierarchy")

    out = eng.flow_save("journeyed", last=5)
    assert out["ok"]
    flow = FlowStore(eng.config.memory).load("journeyed")
    assert flow.app == P
    assert any(s.kind == "tap" and s.label == "Apps" for s in flow.steps)
    # the session store agrees
    sess = AppMemoryStore(eng.config.memory).load_session("emu-journal")
    assert sess.recent


# --------------------------------------------------------------- deeplinks + playbook


def test_open_link_action_records_deeplink(tmp_path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-dl")
    eng = _engine(tmp_path, dev)
    eng.analyze(source="hierarchy")  # seed a cached package
    out = eng.open_link("luzia-test://set-flags?foo=a")
    assert out.ok and out.action == "open-link"
    assert ("open_link", ("luzia-test://set-flags?foo=a",)) in dev.calls
    app = AppMemoryStore(eng.config.memory).load(P)
    assert app is not None
    assert [d.uri for d in app.deeplinks] == ["luzia-test://set-flags?foo=a"]


def test_remember_deeplink_dedups_and_counts(tmp_path) -> None:
    store = _store(tmp_path)
    store.remember_deeplink(P, "luzia-test://x", note="do x")
    store.remember_deeplink(P, "luzia-test://x")
    app = store.load(P)
    assert len(app.deeplinks) == 1
    assert app.deeplinks[0].count == 2 and app.deeplinks[0].note == "do x"


def test_playbook_notes_recipes_description(tmp_path) -> None:
    store = _store(tmp_path)
    store.set_description(P, "Luzia AI assistant (dev build)")
    store.remember_note(P, "gamification pill needs a feature flag")
    store.remember_recipe(P, "login_full", "tap 'Login with test user'")
    store.remember_recipe(P, "login_full", "tap testUserLogin")  # updates in place
    app = store.load(P)
    assert app.description == "Luzia AI assistant (dev build)"
    assert app.notes == ["gamification pill needs a feature flag"]
    assert len(app.recipes) == 1 and app.recipes[0].note == "tap testUserLogin"


def test_flow_open_link_and_bare_stop_app(tmp_path) -> None:
    # The real set-feature-flags recipe: open a deeplink, restart the app.
    text = """
name: set_flags
app: co.thewordlab.luzia
steps:
  - open_link: "luzia-test://set-flags?chat_v5=treatment"
  - stop_app
  - launch_app
  - wait_stable
"""
    flow = parse_flow_yaml(text, name="set_flags")
    kinds = [s.kind for s in flow.steps]
    assert kinds == ["open-link", "stop-app", "launch-app", "wait-stable"]
    assert flow.steps[0].arg == "luzia-test://set-flags?chat_v5=treatment"
    assert flow.steps[1].arg is None  # bare stop_app → defaults to flow.app at run

    dev = ScriptedDevice([HOME, HOME], package=P, serial="emu-flow-dl")
    eng = _engine(tmp_path, dev)
    flow_file = tmp_path / "set_flags.yaml"
    flow_file.write_text(text, encoding="utf-8")
    out = eng.flow_run(file=str(flow_file))
    assert out["ok"] is True, out
    assert ("open_link", ("luzia-test://set-flags?chat_v5=treatment",)) in dev.calls
    assert ("stop_app", ("co.thewordlab.luzia",)) in dev.calls  # bare → flow.app
    assert ("launch_app", ("co.thewordlab.luzia",)) in dev.calls


def test_v2_map_without_playbook_loads(tmp_path) -> None:
    # Older maps have no deeplinks/recipes/notes/description — they must load fine.
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    idx = store.index_path(P)
    data = json.loads(idx.read_text())
    for k in ("deeplinks", "recipes", "notes", "description"):
        data.pop(k, None)
    idx.write_text(json.dumps(data))
    app = store.load(P)
    assert app is not None and app.deeplinks == [] and app.description is None


# --------------------------------------------------------------- playbook surfacing


def test_render_map_shows_playbook(tmp_path) -> None:
    from android_ui_analyser.memory import render_map

    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.set_description(P, "Luzia AI assistant")
    store.remember_recipe(P, "login_full", "tap 'Login with test user'")
    store.remember_deeplink(P, "luzia-test://set-flags?x=a", note="set feature flags then restart")
    store.remember_note(P, "the Apps tab is bottomBarTools (Tools=Apps)")
    text = render_map(store.load(P))
    assert "## Playbook" in text
    assert "Luzia AI assistant" in text
    assert "recipe `login_full`" in text
    assert "luzia-test://set-flags" in text
    assert "Tools=Apps" in text


def test_orient_surfaces_playbook(tmp_path) -> None:
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    store.set_description(P, "Luzia")
    store.remember_recipe(P, "login_full", "tap testUserLogin")
    store.remember_deeplink(P, "luzia-test://set-flags?x=a", note="flags")
    dev = FakeDevice(hierarchy_xml=HOME, package=P)
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    out = eng.orient()
    assert out["known"] is True
    assert out["description"] == "Luzia"
    assert out["recipes"]["login_full"] == "tap testUserLogin"
    assert out["deeplinks"][0]["uri"] == "luzia-test://set-flags?x=a"


def test_cli_remember_and_about(tmp_path, monkeypatch) -> None:
    import android_ui_analyser.engine as engine_mod
    from conftest import FakeDevice

    dev = FakeDevice(hierarchy_xml=HOME, package=P)
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    r = runner.invoke(
        app, ["remember", "--app", P, "--recipe", "login_full", "--note", "tap testUserLogin"]
    )
    assert r.exit_code == 0, r.stderr
    assert json.loads(r.stdout)["saved"] == ["recipe:login_full"]

    r2 = runner.invoke(app, ["remember", "--app", P, "--deeplink", "luzia-test://x", "--note", "z"])
    assert r2.exit_code == 0

    about = runner.invoke(app, ["--format", "compact", "about", "--app", P])
    assert about.exit_code == 0
    data = json.loads(about.stdout)
    assert data["recipes"]["login_full"] == "tap testUserLogin"
    assert data["deeplinks"][0]["uri"] == "luzia-test://x"


def test_cli_remember_recipe_requires_note(tmp_path, monkeypatch) -> None:
    import android_ui_analyser.engine as engine_mod
    from conftest import FakeDevice

    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: FakeDevice(package=P))
    r = runner.invoke(app, ["remember", "--app", P, "--recipe", "x"])
    assert r.exit_code != 0


# --------------------------------------------------------------- flow composition (runFlow)


def test_flow_composition_runs_sub_flow(tmp_path) -> None:
    # A `setup` flow (tap Apps) reused by a parent flow via `flow:`.
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = FlowStore(cfg.memory)
    store.save(Flow(name="go_apps", app=P, steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")]))

    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "name: parent\napp: co.thewordlab.luzia\nsteps:\n  - flow: go_apps\n  - assert_visible: \"Images\"\n",
        encoding="utf-8",
    )
    flow = parse_flow_yaml(parent.read_text(), name="parent")
    assert flow.steps[0].kind == "flow" and flow.steps[0].arg == "go_apps"

    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory

    dev = ScriptedDevice(
        [HOME, APPS], package=P, serial="emu-compose", text_index={"Images": (0, 0, 9, 9)}
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    out = eng.flow_run(file=str(parent))
    assert out["ok"] is True, out
    assert sum(1 for c in dev.calls if c[0] == "click") == 1  # the sub-flow's tap ran


def test_flow_composition_run_flow_alias_and_render(tmp_path) -> None:
    flow = parse_flow_yaml("steps:\n  - run_flow: login\n", name="x")
    assert flow.steps[0].kind == "flow" and flow.steps[0].arg == "login"
    # canonical render key is `flow`
    assert "flow: login" in render_flow_yaml(flow)


def test_flow_composition_missing_sub_flow_diverges(tmp_path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    f = tmp_path / "p.yaml"
    f.write_text("name: p\napp: co.thewordlab.luzia\nsteps:\n  - flow: nonexistent\n", encoding="utf-8")
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory

    dev = ScriptedDevice([HOME], package=P, serial="emu-nosub")
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    out = eng.flow_run(file=str(f))
    assert out["ok"] is False and out["code"] == "route_unknown"
    assert out["failed_step"]["display"] == "flow nonexistent"


# --------------------------------------------------------------- resource-id matching (by=id)


def test_has_by_id_finds_pruned_container(tmp_path) -> None:
    # A container that the parsed element list prunes is still verifiable via by="id"
    # (u2 queries the raw tree; the FakeDevice mimics this with a resource_index).
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice

    cfg = make_config(daemon={"enabled": False})
    dev = FakeDevice(
        hierarchy_xml=HOME,
        package=P,
        resource_index={"co.thewordlab.luzia:id/containerChatDetail": (0, 0, 100, 100)},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    # by text: absent
    assert eng.has("containerChatDetail").found is False
    # by id (bare tail): found
    r = eng.has("containerChatDetail", by="id")
    assert r.found is True and r.source == "hierarchy"


def test_flow_assert_visible_by_id(tmp_path) -> None:
    text = (
        "name: t\napp: co.thewordlab.luzia\nsteps:\n"
        "  - assert_visible: {id: containerChatDetail}\n"
        "  - wait_for: {id: inputBar, timeout_ms: 2000}\n"
    )
    flow = parse_flow_yaml(text, name="t")
    assert flow.steps[0].kind == "assert-visible" and flow.steps[0].by == "id"
    assert flow.steps[0].arg == "containerChatDetail"
    assert flow.steps[1].kind == "wait-for" and flow.steps[1].by == "id"

    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    dev = FakeDevice(
        hierarchy_xml=HOME,
        package=P,
        resource_index={"x:id/containerChatDetail": (0, 0, 9, 9), "x:id/inputBar": (0, 0, 9, 9)},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    f = tmp_path / "t.yaml"
    f.write_text(text, encoding="utf-8")
    out = eng.flow_run(file=str(f))
    assert out["ok"] is True, out


def test_cli_has_by_id(tmp_path, monkeypatch) -> None:
    import android_ui_analyser.engine as engine_mod
    from conftest import FakeDevice

    dev = FakeDevice(hierarchy_xml=HOME, package=P, resource_index={"x:id/containerHome": (0, 0, 9, 9)})
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    ok = runner.invoke(app, ["has", "containerHome", "--by", "id"])
    assert ok.exit_code == 0, ok.stderr
    miss = runner.invoke(app, ["has", "containerHome"])  # by text → absent
    assert miss.exit_code == 1
