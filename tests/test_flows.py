"""Named flows (PRD §6b): YAML parse/render, params, one-call replay, save, CLI, daemon.

A flow is the agent-authored (or `flow save`-materialized) Maestro-style journey; the
executor is shared with `goto`, so these tests focus on the flow-specific surface:
parsing, ${PARAM} substitution, divergence + `--from-step` resume, privacy of saved
files, and the plumbing (CLI group, daemon dispatch).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.daemon import dispatch
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import (
    Flow,
    FlowStore,
    parse_flow_yaml,
    render_flow_yaml,
    resolve_params,
    steps_from_recent,
)
from android_ui_analyser.memory import (
    AppMemoryStore,
    RouteStep,
    SessionState,
    context_id_for_flags,
)
from android_ui_analyser.schema import ActionResult
from android_ui_analyser.selectors import match_step
from conftest import make_config
from test_memory import APPS, HOME, P, _elements, _engine, _hier, _node, _store
from test_navigation import ScriptedDevice

runner = CliRunner()

FLOW_YAML = """
name: open_images
app: com.example.app
params:
  TOOL: "Images"
steps:
  - launch_app: com.example.app
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


def test_desc_selector_round_trips_distinctly_and_matches_only_description() -> None:
    flow = parse_flow_yaml('steps:\n  - tap: {desc: "Open details", by: desc}\n')
    rendered = render_flow_yaml(flow)
    assert "desc: Open details" in rendered
    assert "by: desc" in rendered
    assert "text:" not in rendered
    step = parse_flow_yaml(rendered).steps[0]
    assert step.by == "desc" and step.content_desc == "Open details" and step.label is None
    elements = _elements(
        _hier(
            _node("android.widget.Button", text="Open details", b="[0,0][100,50]"),
            _node("android.widget.Button", desc="Open details", b="[0,50][100,100]"),
        )
    )
    assert match_step(elements, step) is elements[1]


def test_legacy_id_and_text_keeps_text_fallback_but_strict_capture_does_not() -> None:
    elements = _elements(_hier(_node("android.widget.Button", text="Continue", b="[0,0][100,50]")))
    legacy = parse_flow_yaml("steps:\n  - tap: {id: oldButton, text: Continue}\n").steps[0]
    assert legacy.by is None
    assert match_step(elements, legacy) is elements[0]

    strict = parse_flow_yaml("steps:\n  - tap: {id: oldButton, text: Continue, by: id}\n").steps[0]
    assert strict.by == "id"
    assert "by: id" in render_flow_yaml(Flow(name="strict", steps=[strict]))
    assert match_step(elements, strict) is None


def test_legacy_desc_keeps_text_or_description_fallback() -> None:
    elements = _elements(_hier(_node("android.widget.Button", text="Continue", b="[0,0][100,50]")))
    legacy = parse_flow_yaml("steps:\n  - tap: {desc: Continue}\n").steps[0]
    assert legacy.by is None and legacy.content_desc == "Continue"
    assert match_step(elements, legacy) is elements[0]

    strict = parse_flow_yaml("steps:\n  - tap: {desc: Continue, by: desc}\n").steps[0]
    assert match_step(elements, strict) is None


# --------------------------------------------------------------------------- params


def test_resolve_params_defaults_and_overrides() -> None:
    flow = parse_flow_yaml(FLOW_YAML)
    steps = resolve_params(flow, {})
    assert steps[2].label == "Images"  # declared default applies
    steps = resolve_params(flow, {"TOOL": "Games"})
    assert steps[2].label == "Games"  # override wins


def test_resolve_params_missing_required_raises() -> None:
    flow = parse_flow_yaml('params: {ACCOUNT: ""}\nsteps:\n  - tap: "${ACCOUNT}"\n')
    try:
        resolve_params(flow, {})
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "ACCOUNT" in str(exc)
    steps = resolve_params(flow, {"ACCOUNT": "Engineering Team"})
    assert steps[0].label == "Engineering Team"


def test_resolve_params_undeclared_placeholder_raises() -> None:
    flow = parse_flow_yaml('steps:\n  - tap: "${NOPE}"\n')
    try:
        resolve_params(flow, {})
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "NOPE" in str(exc)


# --------------------------------------------------------------------------- flow_run


def _images_flow_text() -> str:
    return """
name: to_images
app: com.example.app
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
app: com.example.app
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


def test_named_flow_failure_resumes_by_storage_key_not_declared_name(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")})
    store = FlowStore(cfg.memory)
    store.flows_dir().mkdir(parents=True)
    (store.flows_dir() / "open_cached.yaml").write_text(
        f"name: Friendly cached title\napp: {P}\nsteps:\n  - tap: {{text: Definitely missing}}\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-storage-resume")

    result = Engine(cfg, device=device).flow_run("open_cached")

    assert result["ok"] is False
    assert result["flow"] == "open_cached"
    assert result["declared_name"] == "Friendly cached title"
    assert result["resume_call"] == "aua flow run open_cached --from-step 0"
    assert result["resume_call"] in result["hint"]


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
        'name: reset\napp: com.example.app\nsteps:\n  - tap: "Delete my account"\n',
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
        "name: via_goto\napp: com.example.app\nsteps:\n  - goto: apps\n", encoding="utf-8"
    )
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-goto-step")
    eng = _engine(tmp_path, dev)
    out = eng.flow_run(file=str(flow_file))
    assert out["ok"] is True
    assert sum(1 for c in dev.calls if c[0] == "click") == 1


def test_no_allow_destructive_scans_the_whole_flow_before_step_zero(tmp_path: Path) -> None:
    flow_file = tmp_path / "late-danger.yaml"
    flow_file.write_text(
        f"name: late_danger\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - repeat:\n      times: 1\n      steps:\n"
        "        - tap: {id: deleteAccount}\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-late-danger")

    result = _engine(tmp_path, device).flow_run(file=str(flow_file), allow_destructive=False)

    assert result["ok"] is False and result["code"] == "destructive_step"
    assert result["steps_run"] == []
    assert not any(call[0] in {"press", "click"} for call in device.calls)


def test_composite_failure_reports_the_parent_index_for_safe_resume(tmp_path: Path) -> None:
    flow_file = tmp_path / "composite-failure.yaml"
    flow_file.write_text(
        f"name: composite_failure\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - repeat:\n      times: 2\n      steps:\n"
        "        - tap: {id: definitelyMissing}\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME, HOME], package=P, serial="emu-composite-failure")

    result = _engine(tmp_path, device).flow_run(file=str(flow_file))

    assert result["ok"] is False and result["step_index"] == 1
    assert result["remaining_steps"][0].startswith("repeat")
    assert sum(1 for call in device.calls if call[0] == "press") == 1


def test_flow_run_launch_app_step(tmp_path: Path) -> None:
    flow_file = tmp_path / "l.yaml"
    flow_file.write_text(
        f'name: l\napp: {P}\nsteps:\n  - launch_app: {P}\n  - assert_visible: "Home"\n',
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

    preview = eng.flow_save("my_flow")
    assert preview["ok"] and preview["saved"] is False
    assert preview["exists"] is False
    assert preview["collision"] is False
    assert preview["status"] == "preview_new"
    assert preview["required_save_mode"] == "create"
    assert not Path(preview["path"]).exists()
    out = eng.flow_save("my_flow", save=True)
    assert out["ok"] and out["steps"] == 2 and out["params_needed"] == ["PARAM_1"]
    saved = Path(out["path"]).read_text(encoding="utf-8")
    assert "a very secret prompt" not in saved  # privacy: typed values never persisted
    assert "${PARAM_1}" in saved
    assert "id: send" in saved


def test_flow_save_requires_force_to_overwrite(tmp_path: Path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-save2")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)
    assert eng.flow_save("dup", save=True)["ok"]
    preview = eng.flow_save("dup")
    assert preview["exists"] is True
    assert preview["collision"] is True
    assert preview["status"] == "preview_existing"
    assert preview["required_save_mode"] == "force"
    assert preview["save_call"].endswith("dup --last 12 --save --force")
    assert preview["invalid_mode_probe"] == {
        "case": "force_without_save",
        "error_code": "usage",
        "cli": "aua --expect-error usage flow save dup --last 12 --force",
        "mcp": {
            "tool": "flow_save",
            "arguments": {
                "name": "dup",
                "last": 12,
                "force": True,
                "expect_error": "usage",
            },
        },
    }
    try:
        eng.flow_save("dup", save=True)
        raise AssertionError("expected UsageError")
    except UsageError as exc:
        assert "--force" in (exc.hint or "")
    with pytest.raises(UsageError, match="--force only applies"):
        eng.flow_save("dup", force=True)
    assert eng.flow_save("dup", save=True, force=True)["ok"]


def test_flow_delete_is_idempotent_and_reports_the_authoritative_path(tmp_path: Path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-delete")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)
    saved = eng.flow_save("disposable", save=True)

    deleted = eng.flow_delete("disposable")
    absent = eng.flow_delete("disposable")

    assert deleted == {
        "ok": True,
        "action": "flow-delete",
        "flow": "disposable",
        "path": saved["path"],
        "deleted": True,
        "status": "deleted",
    }
    assert absent["ok"] is True
    assert absent["deleted"] is False
    assert absent["status"] == "already_absent"


def test_flow_save_dry_run_previews_without_writing(tmp_path: Path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-save-preview")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)

    out = eng.flow_save("preview_only", dry_run=True)

    assert out["dry_run"] is True
    assert "tap:" in out["preview"]
    assert out["saved"] is False
    assert out["save_call"].endswith("preview_only --last 12 --save")
    assert not Path(out["path"]).exists()


def test_recording_skips_a_duplicate_resource_id_for_unique_description(tmp_path: Path) -> None:
    screen = _hier(
        _node(
            "android.widget.Button",
            text="First",
            rid="android:id/title",
            desc="Open first item",
            clk=True,
            b="[0,0][500,100]",
        ),
        _node(
            "android.widget.Button",
            text="Second",
            rid="android:id/title",
            desc="Open second item",
            clk=True,
            b="[0,100][500,200]",
        ),
    )
    eng = _engine(tmp_path, ScriptedDevice([screen], package=P, serial="emu-duplicate-rid"))
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)

    recorded = AppMemoryStore(eng.config.memory).load_session("emu-duplicate-rid").recent[-1]
    assert recorded.resource_id is None
    assert recorded.by == "desc" and recorded.content_desc == "Open second item"
    preview = eng.flow_save("second_item")
    assert "desc: Open second item" in preview["preview"]
    assert "id: title" not in preview["preview"]


def test_flow_save_refuses_when_no_unique_privacy_safe_selector_exists(tmp_path: Path) -> None:
    screen = _hier(
        _node(
            "android.widget.Button",
            text="john@example.test",
            rid="android:id/title",
            desc="john@example.test",
            clk=True,
            b="[0,0][500,100]",
        ),
        _node(
            "android.widget.Button",
            text="jane@example.test",
            rid="android:id/title",
            desc="jane@example.test",
            clk=True,
            b="[0,100][500,200]",
        ),
    )
    eng = _engine(tmp_path, ScriptedDevice([screen], package=P, serial="emu-unsafe-selector"))
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[1].id, observe=False)

    preview = eng.flow_save("unsafe_selector")
    assert preview["ok"] is False and preview["saved"] is False
    assert "no unique stable id" in preview["selector_warnings"][0]
    assert not Path(preview["path"]).exists()


def test_unique_resource_id_saves_even_when_visible_label_is_pii(tmp_path: Path) -> None:
    screen = _hier(
        _node(
            "android.widget.Button",
            text="person@example.test",
            rid="com.example.app:id/accountButton",
            clk=True,
            b="[0,0][500,100]",
        )
    )
    eng = _engine(tmp_path, ScriptedDevice([screen], package=P, serial="emu-pii-with-rid"))
    res = eng.analyze(source="hierarchy")
    eng.tap(res.elements[0].id, observe=False)

    preview = eng.flow_save("account")
    assert preview["ok"] is True
    assert "id: accountButton" in preview["preview"]
    assert "person@example.test" not in preview["preview"]


def test_flow_save_uses_only_newest_provenance_segment(tmp_path: Path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-segmented")
    eng = _engine(tmp_path, dev)
    eng.analyze(source="hierarchy")
    store = AppMemoryStore(eng.config.memory)
    sess = store.load_session(dev.serial)
    sess.capture_segment = 2
    sess.recent = [
        RouteStep(
            kind="tap",
            resource_id="oldButton",
            by="id",
            origin_package="com.example.other",
            context_id="default",
            capture_segment=1,
        ),
        RouteStep(
            kind="tap",
            resource_id="apps",
            by="id",
            origin_package=P,
            context_id="default",
            capture_segment=2,
        ),
    ]
    store.save_session(dev.serial, sess)

    preview = eng.flow_save("newest", last=2)
    assert preview["scope"]["selected"] == 1
    assert preview["scope"]["boundary_omitted"] == 1
    assert f"app: {P}" in preview["preview"]
    assert "oldButton" not in preview["preview"]


def test_action_waits_for_async_screen_provenance_before_journaling(tmp_path: Path) -> None:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        daemon={"enabled": False},
        perf={"async_memory": True, "skip_unchanged_memory": False},
    )
    dev = ScriptedDevice([HOME], package=P, serial="emu-async-provenance")
    eng = Engine(cfg, device=dev)
    mem = eng._memory
    assert mem is not None
    mem.save_session(
        dev.serial,
        SessionState(
            package="com.example.previous",
            active_context_id="flags-old",
            capture_segment=4,
        ),
    )
    entered = threading.Event()
    release = threading.Event()
    original = mem.observe_screen

    def blocked_observe_screen(*args: object, **kwargs: object) -> str | None:
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    mem.observe_screen = blocked_observe_screen  # type: ignore[method-assign]
    res = eng.analyze(source="hierarchy")
    assert entered.wait(timeout=2)
    done = threading.Event()

    def act() -> None:
        eng.tap(next(e.id for e in res.elements if e.text == "Apps"), observe=False)
        done.set()

    actor = threading.Thread(target=act)
    actor.start()
    assert not done.wait(timeout=0.1), "action journal raced the blocked screen writer"
    release.set()
    actor.join(timeout=5)
    assert done.is_set()

    sess = mem.load_session(dev.serial)
    assert sess.package == P and sess.active_context_id == "default"
    assert sess.capture_segment == 5
    assert len(sess.recent) == 1
    recorded = sess.recent[0]
    assert recorded.origin_package == P
    assert recorded.context_id == "default"
    assert recorded.capture_segment == 5


def test_app_lifecycle_waits_for_async_screen_provenance_before_boundary(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        daemon={"enabled": False},
        perf={"async_memory": True, "skip_unchanged_memory": False},
    )
    dev = ScriptedDevice([HOME], package=P, serial="emu-async-lifecycle")
    eng = Engine(cfg, device=dev)
    mem = eng._memory
    assert mem is not None
    entered = threading.Event()
    release = threading.Event()
    original = mem.observe_screen

    def blocked_observe_screen(*args: object, **kwargs: object) -> str | None:
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    mem.observe_screen = blocked_observe_screen  # type: ignore[method-assign]
    eng.analyze(source="hierarchy")
    assert entered.wait(timeout=2)
    done = threading.Event()

    def stop() -> None:
        eng.app("stop", package=P)
        done.set()

    worker = threading.Thread(target=stop)
    worker.start()
    assert not done.wait(timeout=0.1), "lifecycle boundary raced the screen writer"
    release.set()
    worker.join(timeout=5)
    assert done.is_set()
    session = mem.load_session(dev.serial)
    assert session.package == P
    assert session.capture_segment == 1
    assert session.capture_boundary_reason == f"app process stopped for {P}"


def test_paste_is_journaled_as_lossy_so_flow_save_refuses_it(tmp_path: Path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-paste-capture")
    eng = _engine(tmp_path, dev)
    eng.analyze(source="hierarchy")

    eng.paste(observe=False)
    preview = eng.flow_save("clipboard_dependent")

    assert preview["ok"] is False and preview["saveable"] is False
    assert any("clipboard value" in warning for warning in preview["capture_warnings"])


def test_flow_save_infers_arrival_only_from_fresh_mapped_destination(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-arrival-save")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    eng.tap(next(e.id for e in res.elements if e.text == "Apps"), observe=False)

    preview = eng.flow_save("to_apps")
    assert preview["arrival_proof"] == {
        "status": "verified",
        "screen": "apps",
        "reason": "current destination was freshly recognized as a mapped screen",
    }
    assert "arrival_screen: apps" in preview["preview"]
    assert "arrival_status: mapped" in preview["preview"]


def test_flow_save_does_not_persist_stale_mapped_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    eng = _engine(tmp_path, ScriptedDevice([HOME], package=P, serial="emu-save-stale-arrival"))
    current = eng.analyze(source="hierarchy")
    eng.key("back", observe=False)
    app_map = store.load(P)
    assert app_map is not None
    app_map.screens["home"].stale = True
    store.save(app_map)
    monkeypatch.setattr(eng, "analyze", lambda **_kwargs: current)

    preview = eng.flow_save("stale_arrival_capture")

    assert preview["arrival_proof"]["status"] == "unverified"
    assert "no fresh map record" in preview["arrival_proof"]["reason"]
    assert "arrival_screen:" not in preview["preview"]


def test_flow_run_fails_when_mapped_arrival_screen_does_not_match(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(
            name="wrong_arrival",
            app=P,
            arrival_screen="apps",
            arrival_status="mapped",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    eng = _engine(tmp_path, ScriptedDevice([HOME], package=P, serial="emu-arrival-run"))

    result = eng.flow_run("wrong_arrival")
    assert result["ok"] is False
    assert result["code"] == "arrival_screen_unverified"
    assert result["arrival_screen"] == {
        "expected": "apps",
        "recognized": "home",
        "verified": False,
    }


def test_explicit_legacy_flow_reports_execution_without_claiming_arrival(tmp_path: Path) -> None:
    flow_file = tmp_path / "legacy_recipe.yaml"
    flow_file.write_text(
        f"name: legacy_recipe\napp: {P}\nsteps:\n  - key: back\n",
        encoding="utf-8",
    )
    result = _engine(
        tmp_path,
        ScriptedDevice([HOME], package=P, serial="emu-legacy-unverified"),
    ).flow_run(file=str(flow_file))

    assert result["ok"] is True  # the explicitly requested step executed
    assert result["arrival_verified"] is False
    assert result["arrival_status"] == "unverified"


def test_flow_run_refuses_wrong_foreground_before_first_action(tmp_path: Path) -> None:
    other = _hier(
        _node(
            "android.widget.Button",
            text="Unrelated",
            rid="other:id/unrelated",
            clk=True,
            pkg="com.example.other",
        )
    )
    flow_file = tmp_path / "owned.yaml"
    flow_file.write_text(
        f"name: owned\napp: {P}\nsteps:\n  - tap: Apps\n",
        encoding="utf-8",
    )
    device = ScriptedDevice(
        [other],
        package="com.example.other",
        serial="emu-wrong-foreground",
    )

    with pytest.raises(UsageError, match="foreground package"):
        _engine(tmp_path, device).flow_run(file=str(flow_file))
    assert not any(call[0] == "click" for call in device.calls)


def test_leading_launch_may_establish_flow_origin_before_other_steps(tmp_path: Path) -> None:
    other = _hier(
        _node(
            "android.widget.TextView",
            text="Other",
            rid="other:id/title",
            pkg="com.example.other",
        )
    )

    class LaunchingDevice(ScriptedDevice):
        def launch_app(self, package: str, *, activity: str | None = None) -> None:
            super().launch_app(package, activity=activity)
            self._xml = HOME

    flow_file = tmp_path / "launch_owned.yaml"
    flow_file.write_text(
        f"name: launch_owned\napp: {P}\nsteps:\n  - launch_app: {P}\n  - assert_visible: Home\n",
        encoding="utf-8",
    )
    device = LaunchingDevice(
        [other],
        package="com.example.other",
        serial="emu-launch-establishes",
        text_index={"Home": (0, 0, 10, 10)},
    )
    result = _engine(tmp_path, device).flow_run(file=str(flow_file))

    assert result["ok"] is True
    assert ("launch_app", (P,)) in device.calls


def test_flow_execution_forces_fresh_runtime_flag_context(tmp_path: Path) -> None:
    flag_name = "catalog_variant"
    expected_context = context_id_for_flags({flag_name: "b"})
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        daemon={"enabled": False},
        flags={
            "prefs_files": {P: "catalog_flags.xml"},
            "context_keys": {P: [flag_name]},
            "context_refresh_s": 9999,
        },
    )
    FlowStore(cfg.memory).save(
        Flow(
            name="variant_recipe",
            app=P,
            context_id=expected_context,
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    device = ScriptedDevice(
        [HOME],
        package=P,
        serial="emu-context-refresh",
        prefs={"catalog_flags.xml": {flag_name: "b"}},
    )
    AppMemoryStore(cfg.memory).save_session(
        device.serial,
        SessionState(package=P, active_context_id="default"),
    )
    eng = Engine(cfg, device=device)
    # Model a recent ordinary refresh: only flow entry's forced read may discover the change.
    eng._flag_context_checked_at[P] = float("inf")

    result = eng.flow_run("variant_recipe")

    assert result["ok"] is True
    assert (
        AppMemoryStore(cfg.memory).load_session(device.serial).active_context_id == expected_context
    )


def test_combined_arrival_checks_mapped_screen_on_predicate_terminal_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(
            name="combined_proof",
            app=P,
            arrival="text:Ready",
            arrival_screen="apps",
            arrival_status="mapped",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    eng = _engine(tmp_path, ScriptedDevice([HOME, APPS], package=P, serial="emu-combined"))
    terminal = eng.analyze(source="hierarchy")
    terminal.meta.known_screen = "home"
    monkeypatch.setattr(
        eng,
        "await_predicate",
        lambda *_args, **_kwargs: ActionResult(
            ok=True,
            action="await",
            await_outcome="satisfied",
            observation=terminal,
        ),
    )

    result = eng.flow_run("combined_proof")

    assert result["ok"] is False
    assert result["code"] == "arrival_screen_unverified"
    assert result["arrival_screen"]["recognized"] == "home"


def test_mapped_arrival_rejects_a_stale_map_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    eng = _engine(tmp_path, ScriptedDevice([HOME], package=P, serial="emu-stale-arrival"))
    terminal = eng.analyze(source="hierarchy")
    app_map = store.load(P)
    assert app_map is not None
    app_map.screens["home"].stale = True
    store.save(app_map)
    FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory).save(
        Flow(
            name="stale_destination",
            app=P,
            arrival_screen="home",
            arrival_status="mapped",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    # Preserve a terminal observation whose name was recognized before the record became stale;
    # a stale cursor/name must not prove arrival without a fresh compatible map record.
    eng.analyze = lambda **_kwargs: terminal  # type: ignore[method-assign]

    with pytest.raises(UsageError, match="unavailable mapped arrival"):
        eng.flow_run("stale_destination")
    assert not any(call[0] == "press" for call in eng.device.calls)


def test_nested_flow_enforces_its_own_app_before_any_substep(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    FlowStore(cfg.memory).save(
        Flow(
            name="foreign_child",
            app="com.example.foreign",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    parent = tmp_path / "parent_foreign.yaml"
    parent.write_text(
        f"name: parent_foreign\napp: {P}\nsteps:\n  - flow: foreign_child\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-nested-foreign")
    with pytest.raises(UsageError, match="not parent app"):
        Engine(cfg, device=device).flow_run(file=str(parent))
    assert not any(call[0] == "press" for call in device.calls)


def test_nested_explicit_context_is_refused_before_an_earlier_parent_step(
    tmp_path: Path,
) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    FlowStore(cfg.memory).save(
        Flow(
            name="context_child",
            app=P,
            context_id="flags-other",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    parent = tmp_path / "parent_context.yaml"
    parent.write_text(
        f"name: parent_context\napp: {P}\nsteps:\n  - key: back\n  - flow: context_child\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-nested-context")

    with pytest.raises(UsageError, match="uses context"):
        Engine(cfg, device=device).flow_run(file=str(parent))

    assert not any(call[0] == "press" for call in device.calls)


def test_nested_flow_arrival_failure_fails_the_parent_step(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.record_screen(package=P, elements=_elements(APPS), name_hint="apps")
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    FlowStore(cfg.memory).save(
        Flow(
            name="child_with_proof",
            app=P,
            arrival_screen="apps",
            arrival_status="mapped",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    parent = tmp_path / "parent_proof.yaml"
    parent.write_text(
        f"name: parent_proof\napp: {P}\nsteps:\n  - flow: child_with_proof\n",
        encoding="utf-8",
    )
    result = Engine(
        cfg,
        device=ScriptedDevice([HOME], package=P, serial="emu-nested-proof"),
    ).flow_run(file=str(parent))

    assert result["ok"] is False
    assert result["code"] == "arrival_screen_unverified"
    assert result["failed_step"]["display"] == "flow child_with_proof"


@pytest.mark.parametrize("arrival", ["!text:Loading", "unknown:state"])
def test_nested_flow_invalid_arrival_is_refused_before_substeps(
    tmp_path: Path,
    arrival: str,
) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    child_path = FlowStore(cfg.memory).path("child_with_invalid_proof")
    child_path.parent.mkdir(parents=True, exist_ok=True)
    child_path.write_text(
        f"name: child_with_invalid_proof\napp: {P}\narrival: {arrival!r}\nsteps:\n  - key: back\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent_invalid_proof.yaml"
    parent.write_text(
        f"name: parent_invalid_proof\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - flow: child_with_invalid_proof\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-nested-invalid-proof")

    with pytest.raises(UsageError):
        Engine(cfg, device=device).flow_run(file=str(parent))

    assert not any(call[0] == "press" for call in device.calls)


def test_missing_nested_flow_is_refused_before_earlier_parent_action(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    parent = tmp_path / "parent_missing_child.yaml"
    parent.write_text(
        f"name: parent_missing_child\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - flow: child_that_does_not_exist\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-nested-missing-child")

    with pytest.raises(UsageError, match="no flow named"):
        Engine(cfg, device=device).flow_run(file=str(parent))

    assert device.calls == []


def test_nested_missing_params_are_refused_before_earlier_parent_action(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    child = Flow(
        name="child_missing_param",
        app=P,
        params={"TARGET": ""},
        steps=[RouteStep(kind="tap", label="${TARGET}")],
    )
    FlowStore(cfg.memory).save(child)
    parent = tmp_path / "parent_missing_param.yaml"
    parent.write_text(
        f"name: parent_missing_param\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - flow: child_missing_param\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-nested-missing-param")

    with pytest.raises(UsageError, match="missing flow param"):
        Engine(cfg, device=device).flow_run(file=str(parent))

    assert device.calls == []


def test_nested_cycle_is_refused_before_any_flow_action(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = FlowStore(cfg.memory)
    store.save(
        Flow(
            name="cycle_a",
            app=P,
            steps=[
                RouteStep(kind="key", arg="back"),
                RouteStep(kind="flow", arg="cycle_b"),
            ],
        )
    )
    store.save(
        Flow(
            name="cycle_b",
            app=P,
            steps=[
                RouteStep(kind="key", arg="menu"),
                RouteStep(kind="flow", arg="cycle_a"),
            ],
        )
    )
    device = ScriptedDevice([HOME], package=P, serial="emu-nested-cycle")

    with pytest.raises(UsageError, match="nested flow cycle"):
        Engine(cfg, device=device).flow_run("cycle_a")

    assert device.calls == []


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
    assert listed[0]["context_compatible"] is None
    compatible = store.list(active_package=P, active_context_id="default")
    assert compatible[0]["context_compatible"] is True
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
    gone_again = runner.invoke(app, ["flow", "delete", "cli_flow"])
    assert gone_again.exit_code == 0
    assert json.loads(gone_again.stdout)["status"] == "already_absent"


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


def test_cli_flow_run_inline_yaml_and_artifact_options(tmp_path: Path, monkeypatch) -> None:
    import android_ui_analyser.engine as engine_mod

    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda serial=None: ScriptedDevice([HOME], package=P, serial="emu-inline-dry"),
    )
    artifacts = tmp_path / "cli-artifacts"
    out = runner.invoke(
        app,
        [
            "--format",
            "compact",
            "flow",
            "run",
            "--yaml",
            "steps:\n  - assert: {text: Ready, count: 1}\n",
            "--dry-run",
            "--artifacts-dir",
            str(artifacts),
            "--evidence",
            "none",
            "--junit",
        ],
    )
    assert out.exit_code == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["source"] == "inline_yaml" and data["dry_run"] is True
    assert Path(data["artifacts"]["junit"]).is_file()


def test_cli_flow_run_param_parsing_error() -> None:
    out = runner.invoke(app, ["flow", "run", "x", "--param", "not-a-pair"])
    assert out.exit_code != 0


def test_cli_flow_save_help_is_preview_first() -> None:
    out = runner.invoke(app, ["flow", "save", "--help"])
    assert out.exit_code == 0
    assert "--save" in out.stdout
    assert "writes nothing" in out.stdout
    assert "Deprecated compatibility alias" in out.stdout


def test_cli_flow_save_selector_refusal_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    from android_ui_analyser import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_route",
        lambda *_args, **_kwargs: {
            "ok": False,
            "action": "flow-save-preview",
            "saved": False,
            "selector_warnings": ["step 1 has no safe selector"],
        },
    )
    out = runner.invoke(app, ["flow", "save", "unsafe"])
    assert out.exit_code == 1
    assert json.loads(out.stdout)["saved"] is False


# --------------------------------------------------------------------------- daemon


def test_daemon_dispatch_flow_run_and_save() -> None:
    class FakeEng:
        def flow_run(self, **kw: object) -> dict[str, object]:
            return {"ok": True, "flow": kw.get("name")}

        def flow_save(self, **kw: object) -> dict[str, object]:
            return {
                "ok": True,
                "flow": kw.get("name"),
                "action": "flow-save",
                "saved": kw.get("save"),
            }

    r = dispatch(FakeEng(), {"cmd": "flow_run", "args": {"name": "f", "dry_run": True}})
    assert r["ok"] and r["result"]["flow"] == "f"
    r2 = dispatch(FakeEng(), {"cmd": "flow_save", "args": {"name": "f", "save": True}})
    assert r2["ok"] and r2["result"]["action"] == "flow-save"
    assert r2["result"]["saved"] is True


def test_engine_flow_save_uses_session_store(tmp_path: Path) -> None:
    """flow_save reads the same session journal observe_action writes."""
    dev = ScriptedDevice([HOME, APPS], package=P, serial="emu-journal")
    eng = _engine(tmp_path, dev)
    res = eng.analyze(source="hierarchy")
    apps_id = next(e.id for e in res.elements if e.text == "Apps")
    eng.tap(apps_id, observe=False)
    eng.analyze(source="hierarchy")

    out = eng.flow_save("journeyed", last=5, save=True)
    assert out["ok"]
    flow = FlowStore(eng.config.memory).load("journeyed")
    assert flow.app == P
    assert any(s.kind == "tap" and s.resource_id == "nav_apps" for s in flow.steps)
    # the session store agrees
    sess = AppMemoryStore(eng.config.memory).load_session("emu-journal")
    assert sess.recent


# --------------------------------------------------------------- deeplinks + playbook


def test_open_link_action_records_deeplink(tmp_path) -> None:
    dev = ScriptedDevice([HOME], package=P, serial="emu-dl")
    eng = _engine(tmp_path, dev)
    eng.analyze(source="hierarchy")  # seed a cached package
    out = eng.open_link("myapp://set-flags?foo=a")
    assert out.ok and out.action == "open-link"
    assert ("open_link", ("myapp://set-flags?foo=a", P)) in dev.calls
    app = AppMemoryStore(eng.config.memory).load(P)
    assert app is not None
    assert [d.uri for d in app.deeplinks] == ["myapp://set-flags?foo=a"]


def test_remember_deeplink_dedups_and_counts(tmp_path) -> None:
    store = _store(tmp_path)
    store.remember_deeplink(P, "myapp://x", note="do x")
    store.remember_deeplink(P, "myapp://x")
    app = store.load(P)
    assert len(app.deeplinks) == 1
    assert app.deeplinks[0].count == 2 and app.deeplinks[0].note == "do x"


def test_playbook_notes_recipes_description(tmp_path) -> None:
    store = _store(tmp_path)
    store.set_description(P, "Example App (dev build)")
    store.remember_note(P, "gamification pill needs a feature flag")
    store.remember_recipe(P, "login_full", "tap 'Login with test user'")
    store.remember_recipe(P, "login_full", "tap testUserLogin")  # updates in place
    app = store.load(P)
    assert app.description == "Example App (dev build)"
    assert app.notes == ["gamification pill needs a feature flag"]
    assert len(app.recipes) == 1 and app.recipes[0].note == "tap testUserLogin"


def test_playbook_projection_hides_stale_facts_and_deduplicates_replacements(tmp_path) -> None:
    from android_ui_analyser.memory import playbook_view

    store = _store(tmp_path)
    store.remember_recipe(P, "open_library", "tap the old shelf")
    store.remember_recipe(P, "open_library", "tap the catalog tab")
    store.remember_note(P, "The archive is on the old toolbar")
    app_map = store.load(P)
    stale = next(
        item for item in app_map.knowledge if item.text == "The archive is on the old toolbar"
    )
    stale.status = "stale"
    store.save(app_map)

    view = playbook_view(store.load(P))

    assert [(recipe.name, recipe.note) for recipe in view["recipes"]] == [
        ("open_library", "tap the catalog tab")
    ]
    assert view["notes"] == []
    assert view["counts"]["stale_or_scoped_out"] == 1


def test_flow_open_link_and_bare_stop_app(tmp_path) -> None:
    # The real set-feature-flags recipe: open a deeplink, restart the app.
    text = """
name: set_flags
app: com.example.app
steps:
  - open_link: "myapp://set-flags?chat_v5=treatment"
  - stop_app
  - launch_app
  - wait_stable
"""
    flow = parse_flow_yaml(text, name="set_flags")
    kinds = [s.kind for s in flow.steps]
    assert kinds == ["open-link", "stop-app", "launch-app", "wait-stable"]
    assert flow.steps[0].arg == "myapp://set-flags?chat_v5=treatment"
    assert flow.steps[1].arg is None  # bare stop_app → defaults to flow.app at run

    dev = ScriptedDevice([HOME, HOME], package=P, serial="emu-flow-dl")
    eng = _engine(tmp_path, dev)
    flow_file = tmp_path / "set_flags.yaml"
    flow_file.write_text(text, encoding="utf-8")
    out = eng.flow_run(file=str(flow_file))
    assert out["ok"] is True, out
    assert ("open_link", ("myapp://set-flags?chat_v5=treatment", "com.example.app")) in dev.calls
    assert ("stop_app", ("com.example.app",)) in dev.calls  # bare → flow.app
    assert ("launch_app", ("com.example.app",)) in dev.calls


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
    store.set_description(P, "Example App")
    store.remember_recipe(P, "login_full", "tap 'Login with test user'")
    store.remember_deeplink(P, "myapp://set-flags?x=a", note="set feature flags then restart")
    store.remember_note(P, "the Apps tab is bottomBarTools (Tools=Apps)")
    text = render_map(store.load(P))
    assert "## Playbook" in text
    assert "Example App" in text
    assert "recipe `login_full`" in text
    assert "myapp://set-flags" in text
    assert "Tools=Apps" in text


def test_orient_surfaces_playbook(tmp_path) -> None:
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    store.set_description(P, "Example App")
    store.remember_recipe(P, "login_full", "tap testUserLogin")
    store.remember_deeplink(P, "myapp://set-flags?x=a", note="flags")
    dev = FakeDevice(hierarchy_xml=HOME, package=P)
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    out = eng.orient()
    assert out["known"] is True
    assert out["description"] == "Example App"
    assert out["recipes"]["login_full"] == "tap testUserLogin"
    assert out["deeplinks"][0]["uri"] == "myapp://set-flags?x=a"


def test_orient_caps_large_playbooks_and_points_to_about(tmp_path) -> None:
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    store.set_description(P, "Fictional catalog")
    for index in range(15):
        store.remember_note(P, f"Catalog note {index}")
        store.remember_deeplink(P, f"fiction://catalog/{index}")
    eng = Engine(
        cfg,
        device=FakeDevice(hierarchy_xml=HOME, package=P),
        factory=ProviderFactory(cfg),
    )

    out = eng.orient()

    assert len(out["notes"]) == 10
    assert len(out["deeplinks"]) == 8
    assert out["playbook_more"]["notes"] == 5
    assert out["playbook_more"]["deeplinks"] == 7
    assert "aua about" in out["playbook_more"]["hint"]


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

    r2 = runner.invoke(app, ["remember", "--app", P, "--deeplink", "myapp://x", "--note", "z"])
    assert r2.exit_code == 0

    about = runner.invoke(app, ["--format", "compact", "about", "--app", P])
    assert about.exit_code == 0
    data = json.loads(about.stdout)
    assert data["recipes"]["login_full"] == "tap testUserLogin"
    assert data["deeplinks"][0]["uri"] == "myapp://x"


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
    store.save(
        Flow(
            name="go_apps",
            app=P,
            steps=[RouteStep(kind="tap", label="Apps", resource_id="nav_apps")],
        )
    )

    parent = tmp_path / "parent.yaml"
    parent.write_text(
        'name: parent\napp: com.example.app\nsteps:\n  - flow: go_apps\n  - assert_visible: "Images"\n',
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
    nested_tap = next(row for row in out["steps_run"] if row.get("flow_path"))
    assert nested_tap["index"] == 0
    assert nested_tap["path"] == [0, 0]
    assert nested_tap["flow_path"] == ["go_apps"]


def test_nested_composite_steps_keep_the_full_audit_path(tmp_path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = FlowStore(cfg.memory)
    store.save(
        Flow(
            name="nested_composite",
            app=P,
            steps=[
                RouteStep(
                    kind="repeat",
                    repeat=1,
                    substeps=[
                        RouteStep(
                            kind="retry",
                            max_retries=1,
                            substeps=[RouteStep(kind="key", arg="back")],
                        )
                    ],
                )
            ],
        )
    )
    parent = tmp_path / "parent-composite.yaml"
    parent.write_text(
        f"name: parent_composite\napp: {P}\nsteps:\n  - flow: nested_composite\n",
        encoding="utf-8",
    )
    device = ScriptedDevice([HOME, HOME], package=P, serial="emu-composite-audit")

    result = Engine(cfg, device=device).flow_run(file=str(parent))

    leaf = next(row for row in result["steps_run"] if row["step"] == "key 'back'")
    assert leaf["index"] == 0
    assert leaf["path"] == [0, 0, 0, 0, 0, 0]
    assert leaf["flow_path"] == ["nested_composite"]


def test_flow_composition_run_flow_alias_and_render(tmp_path) -> None:
    flow = parse_flow_yaml("steps:\n  - run_flow: login\n", name="x")
    assert flow.steps[0].kind == "flow" and flow.steps[0].arg == "login"
    # canonical render key is `flow`
    assert "flow: login" in render_flow_yaml(flow)


def test_flow_composition_missing_sub_flow_is_preflighted(tmp_path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    f = tmp_path / "p.yaml"
    f.write_text("name: p\napp: com.example.app\nsteps:\n  - flow: nonexistent\n", encoding="utf-8")
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory

    dev = ScriptedDevice([HOME], package=P, serial="emu-nosub")
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    with pytest.raises(UsageError, match="no flow named 'nonexistent'"):
        eng.flow_run(file=str(f))
    assert dev.calls == []


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
        resource_index={"com.example.app:id/containerDetail": (0, 0, 100, 100)},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    # by text: absent
    assert eng.has("containerDetail").found is False
    # by id (bare tail): found
    r = eng.has("containerDetail", by="id")
    assert r.found is True and r.source == "hierarchy"


def test_flow_assert_visible_by_id(tmp_path) -> None:
    text = (
        "name: t\napp: com.example.app\nsteps:\n"
        "  - assert_visible: {id: containerDetail}\n"
        "  - wait_for: {id: inputBar, timeout_ms: 2000}\n"
    )
    flow = parse_flow_yaml(text, name="t")
    assert flow.steps[0].kind == "assert-visible" and flow.steps[0].by == "id"
    assert flow.steps[0].arg == "containerDetail"
    assert flow.steps[1].kind == "wait-for" and flow.steps[1].by == "id"

    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    dev = FakeDevice(
        hierarchy_xml=HOME,
        package=P,
        resource_index={"x:id/containerDetail": (0, 0, 9, 9), "x:id/inputBar": (0, 0, 9, 9)},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    f = tmp_path / "t.yaml"
    f.write_text(text, encoding="utf-8")
    out = eng.flow_run(file=str(f))
    assert out["ok"] is True, out


def test_cli_has_by_id(tmp_path, monkeypatch) -> None:
    import android_ui_analyser.engine as engine_mod
    from conftest import FakeDevice

    dev = FakeDevice(
        hierarchy_xml=HOME, package=P, resource_index={"x:id/containerHome": (0, 0, 9, 9)}
    )
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)
    ok = runner.invoke(app, ["has", "containerHome", "--by", "id"])
    assert ok.exit_code == 0, ok.stderr
    miss = runner.invoke(app, ["has", "containerHome"])  # by text → absent
    assert miss.exit_code == 1
