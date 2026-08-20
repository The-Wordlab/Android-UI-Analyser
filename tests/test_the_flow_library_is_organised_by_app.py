"""The flow library is filed per app, and a bare name two apps claim is refused.

Flows used to be flat files in ``<memory.dir>/flows/`` with the owning package recorded only
*inside* each one, so a machine testing several apps had one undifferentiated directory and no
way to ask "what can I replay against this app". Filing them under ``flows/<package>/`` answers
that, and costs the single guarantee the flat layout gave for free: a name is no longer unique.
These tests pin both halves — the layout, and the refusal to guess which app's journey a shared
name meant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import Flow, FlowStore, parse_flow_yaml
from android_ui_analyser.memory import RouteStep
from conftest import make_config
from test_memory import APPS, HOME, P
from test_navigation import ScriptedDevice

APP_A = "com.example.app"
APP_B = "com.example.other"

runner = CliRunner()


def _store(tmp_path: Path) -> FlowStore:
    return FlowStore(make_config(memory={"dir": str(tmp_path / "home")}).memory)


def _flow(name: str, app: str | None, *, label: str = "Apps") -> Flow:
    return Flow(name=name, app=app, steps=[RouteStep(kind="tap", label=label)])


def test_a_saved_flow_lands_under_its_own_app(tmp_path: Path) -> None:
    store = _store(tmp_path)

    path = store.save(_flow("cold_start", APP_A))

    assert path == store.flows_dir() / APP_A / "cold_start.yaml"
    assert store.load("cold_start").app == APP_A


def test_an_app_agnostic_flow_stays_in_the_library_root(tmp_path: Path) -> None:
    store = _store(tmp_path)

    path = store.save(_flow("dismiss_system_dialog", None))

    assert path == store.flows_dir() / "dismiss_system_dialog.yaml"


def test_listing_is_filterable_by_app(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_flow("a_only", APP_A))
    store.save(_flow("b_only", APP_B))
    store.save(_flow("either", None))

    assert {item["storage_name"] for item in store.list(app=APP_A)} == {"a_only", "either"}
    assert {item["storage_name"] for item in store.list(app=APP_B)} == {"b_only", "either"}
    assert {item["storage_name"] for item in store.list()} == {"a_only", "b_only", "either"}


def test_a_pre_existing_flat_file_is_still_found_and_attributed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.flows_dir().mkdir(parents=True)
    flat = store.flows_dir() / "legacy_reset.yaml"
    flat.write_text(f"name: legacy_reset\napp: {APP_A}\nsteps:\n  - key: back\n", encoding="utf-8")

    assert store.resolve("legacy_reset") == flat
    assert store.load("legacy_reset").app == APP_A
    assert [item["storage_name"] for item in store.list(app=APP_A)] == ["legacy_reset"]
    assert store.delete("legacy_reset") is True
    assert store.delete("legacy_reset") is False  # already-absent stays success


def test_re_saving_a_flat_flow_updates_it_where_it_lies(tmp_path: Path) -> None:
    """A user's flow directory may be checked in; a save must not leave a second copy."""
    store = _store(tmp_path)
    store.flows_dir().mkdir(parents=True)
    flat = store.flows_dir() / "reset.yaml"
    flat.write_text(f"name: reset\napp: {APP_A}\nsteps:\n  - key: back\n", encoding="utf-8")

    path = store.save(_flow("reset", APP_A, label="Sign out"), force=True)

    assert path == flat
    assert not (store.flows_dir() / APP_A / "reset.yaml").exists()
    assert "Sign out" in flat.read_text(encoding="utf-8")


def test_two_apps_may_each_own_a_flow_of_the_same_name(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.save(_flow("cold_start", APP_A))
    second = store.save(_flow("cold_start", APP_B))

    assert first != second
    assert store.resolve(f"{APP_A}:cold_start") == first
    assert store.resolve(f"{APP_B}:cold_start") == second
    assert store.load(f"{APP_B}:cold_start").app == APP_B


def test_an_ambiguous_bare_name_is_refused_with_every_candidate_named(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_flow("cold_start", APP_A))
    store.save(_flow("cold_start", APP_B))

    with pytest.raises(UsageError) as err:
        store.resolve("cold_start")

    message = f"{err.value} {err.value.hint or ''}"
    assert "ambiguous" in message
    assert f"{APP_A}:cold_start" in message
    assert f"{APP_B}:cold_start" in message


def test_an_ambiguous_name_is_never_deleted_by_guessing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_flow("cold_start", APP_A))
    store.save(_flow("cold_start", APP_B))

    with pytest.raises(UsageError, match="ambiguous"):
        store.delete("cold_start")

    assert len(store.find("cold_start")) == 2


def test_a_nested_bare_name_means_the_sibling_in_the_same_app(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_flow("sign_in", APP_A, label="A sign in"))
    store.save(_flow("sign_in", APP_B, label="B sign in"))
    parent = store.save(
        Flow(name="checkout", app=APP_B, steps=[RouteStep(kind="flow", arg="sign_in")])
    )

    resolved = store.resolve("sign_in", referring_dir=parent.parent)

    assert resolved == store.flows_dir() / APP_B / "sign_in.yaml"
    assert "B sign in" in resolved.read_text(encoding="utf-8")


def test_listing_offers_a_reference_that_actually_loads(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(_flow("cold_start", APP_A))
    store.save(_flow("cold_start", APP_B))
    store.save(_flow("unique_one", APP_A))

    refs = {item["path"]: item["ref"] for item in store.list()}

    assert refs[str(store.flows_dir() / APP_A / "unique_one.yaml")] == "unique_one"
    assert refs[str(store.flows_dir() / APP_A / "cold_start.yaml")] == f"{APP_A}:cold_start"
    for ref in refs.values():
        assert store.resolve(ref).is_file()


def test_the_engine_runs_and_deletes_a_namespaced_flow_by_qualified_name(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = FlowStore(cfg.memory)
    store.save(parse_flow_yaml(f"name: reach_apps\napp: {P}\nsteps:\n  - tap: 'Apps'\n"))
    engine = Engine(cfg, device=ScriptedDevice([HOME, APPS], package=P, serial="emu-ns"))

    run = engine.flow_run(f"{P}:reach_apps")
    assert run["ok"] is True, run

    deleted = engine.flow_delete(f"{P}:reach_apps")
    assert deleted["deleted"] is True
    assert deleted["path"] == str(store.flows_dir() / P / "reach_apps.yaml")
    assert engine.flow_delete(f"{P}:reach_apps")["status"] == "already_absent"


def test_cli_flow_list_filters_by_app_and_show_takes_a_qualified_name() -> None:
    # The CLI reads memory.dir from the autouse-isolated env config.
    store = FlowStore(make_config().memory)
    store.save(Flow(name="a_only", app=APP_A, steps=[RouteStep(kind="key", arg="back")]))
    store.save(Flow(name="b_only", app=APP_B, steps=[RouteStep(kind="key", arg="back")]))

    listed = runner.invoke(cli_app, ["flow", "list", "--app", APP_A])
    assert listed.exit_code == 0, listed.output
    payload = json.loads(listed.stdout)
    assert [item["storage_name"] for item in payload["flows"]] == ["a_only"]
    assert payload["app"] == APP_A

    shown = runner.invoke(cli_app, ["flow", "show", f"{APP_B}:b_only"])
    assert shown.exit_code == 0, shown.output
    assert "key: back" in shown.stdout
