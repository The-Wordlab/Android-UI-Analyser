"""Schema-v3 contexts, knowledge provenance, audit, and transactional correction."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import (
    LEGACY_CONTEXT_ID,
    AppMemoryStore,
    RouteStep,
    _shortest_path,
    context_id_for_flags,
    context_view,
    render_map,
    resolve_goal,
)
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.reconcile import (
    CorrectionOperation,
    ReconciliationStore,
    ResearchReport,
    audit_map,
)
from conftest import FakeDevice, make_config
from test_memory import APPS, HOME, SETTINGS_XML, P, _elements, _store

runner = CliRunner()


def test_v2_map_migrates_to_trusted_legacy_context(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="home")
    store.remember_note(P, "Apps differs by feature flag")
    index = store.index_path(P)
    raw = json.loads(index.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw.pop("contexts", None)
    raw.pop("knowledge", None)
    for screen in raw["screens"].values():
        for key in (
            "id",
            "canonical_name",
            "aliases",
            "logical_name",
            "variant",
            "state",
            "context_id",
            "name_source",
        ):
            screen.pop(key, None)
    index.write_text(json.dumps(raw), encoding="utf-8")

    migrated = store.load(P)

    assert migrated is not None and migrated.schema_version == 3
    assert migrated.contexts[LEGACY_CONTEXT_ID].verified is True
    assert migrated.screens["home"].context_id == LEGACY_CONTEXT_ID
    assert migrated.screens["home"].id
    assert any(
        item.text == "Apps differs by feature flag"
        and item.source == "legacy"
        and item.status == "accepted"
        for item in migrated.knowledge
    )


def test_same_ui_is_recorded_as_distinct_flag_variants(tmp_path) -> None:
    store = _store(tmp_path)
    flags_a = {"catalog_experiment": "a"}
    flags_b = {"catalog_experiment": "b"}
    context_a = context_id_for_flags(flags_a)
    context_b = context_id_for_flags(flags_b)

    first = store.record_screen(
        package=P,
        elements=_elements(HOME),
        name_hint="apps",
        context_id=context_a,
        context_flags=flags_a,
        context_verified=True,
    )
    second = store.record_screen(
        package=P,
        elements=_elements(HOME),
        name_hint="apps",
        context_id=context_b,
        context_flags=flags_b,
        context_verified=True,
    )
    revisit = store.record_screen(
        package=P,
        elements=_elements(HOME),
        context_id=context_a,
        context_flags=flags_a,
        context_verified=True,
    )
    app = store.load(P)

    assert first.name != second.name
    assert revisit.was_known and revisit.name == first.name
    assert app is not None
    assert {app.screens[first.name].context_id, app.screens[second.name].context_id} == {
        context_a,
        context_b,
    }
    text = render_map(app, all_contexts=True)
    assert "apps" in text and context_a in text and context_b in text
    projected = context_view(app, context_a)
    assert second.name not in projected.screens
    assert first.name in projected.screens


def test_navigation_never_crosses_feature_flag_contexts(tmp_path) -> None:
    store = _store(tmp_path)
    context_a = context_id_for_flags({"catalog_experiment": "a"})
    context_b = context_id_for_flags({"catalog_experiment": "b"})
    home_a = store.record_screen(
        package=P,
        elements=_elements(HOME),
        name_hint="home_a",
        context_id=context_a,
        context_flags={"catalog_experiment": "a"},
    )
    target_a = store.record_screen(
        package=P,
        elements=_elements(APPS),
        name_hint="catalog_a",
        context_id=context_a,
        context_flags={"catalog_experiment": "a"},
    )
    home_b = store.record_screen(
        package=P,
        elements=_elements(HOME),
        name_hint="home_b",
        context_id=context_b,
        context_flags={"catalog_experiment": "b"},
    )
    target_b = store.record_screen(
        package=P,
        elements=_elements(SETTINGS_XML),
        name_hint="settings_b",
        context_id=context_b,
        context_flags={"catalog_experiment": "b"},
    )
    store.record_route(
        P, home_a.name, target_a.name, "tap 'Open'", context_id=context_a
    )
    store.record_route(
        P, home_b.name, target_b.name, "tap 'Open'", context_id=context_b
    )
    app = store.load(P)

    assert _shortest_path(
        app, target_b.name, start=home_a.name, context_id=context_a
    ) == []
    assert resolve_goal(
        app, target_b.name, start=home_a.name, context_id=context_a
    ) is None


def test_flag_context_activation_scopes_routes_and_resets_cursor(tmp_path) -> None:
    store = _store(tmp_path)
    serial = "context-device"
    store.observe_screen(serial, package=P, elements=_elements(HOME))
    store.observe_action(serial, RouteStep(kind="tap", label="Apps"))

    context_id = store.activate_flag_context(
        serial,
        P,
        {"catalog_experiment": "a"},
        app_version="5.37.0",
        verified=True,
    )
    session = store.load_session(serial)

    assert session.active_context_id == context_id
    assert session.current_screen is None
    assert session.pending == []
    assert session.context_verified is True
    assert store.load(P).contexts[context_id].flags == {"catalog_experiment": "a"}
    assert store.latest_session(P).active_context_id == context_id


def test_raw_flag_deeplink_is_promoted_after_manual_cold_launch(tmp_path) -> None:
    cfg = make_config(
        memory={"dir": str(tmp_path / "home")},
        flags={"templates": {P: "demo-test://set-flags?{query}"}},
        daemon={"enabled": False},
    )
    device = FakeDevice(hierarchy_xml=HOME, package=P, serial="raw-flags")
    engine = Engine(cfg, device=device, factory=ProviderFactory(cfg))

    engine.open_link(
        "demo-test://set-flags?catalog_experiment=a",
        package=P,
        observe=False,
    )
    pending = AppMemoryStore(cfg.memory).load_session(device.serial)
    assert pending.pending_flags == {"catalog_experiment": "a"}

    engine.app("stop", package=P)
    engine.app("launch", package=P)
    active = AppMemoryStore(cfg.memory).load_session(device.serial)
    assert active.active_flags == {"catalog_experiment": "a"}
    assert active.context_verified is False


def test_audit_emits_agent_research_questions(tmp_path) -> None:
    store = _store(tmp_path)
    outcome = store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    app = store.load(P)
    app.screens[outcome.name].stale = True
    store.save(app)

    audit = audit_map(store.load(P))

    assert {issue.type for issue in audit.issues} >= {"poor_name", "stale_screen"}
    assert all(issue.questions for issue in audit.issues)


def test_agent_apply_is_atomic_and_rollback_restores_snapshot(tmp_path) -> None:
    store = _store(tmp_path)
    outcome = store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    reconciliation = ReconciliationStore(store)
    task = next(task for task in reconciliation.plan(P) if task.issue_type == "poor_name")
    report = ResearchReport(
        task_id=task.id,
        agent="codex",
        session="research-1",
        verdict="apply",
        rationale="The source route and title identify the destination as settings.",
        operations=[CorrectionOperation(op="rename", screen_id=outcome.name, value="settings")],
        knowledge=[
            {
                "kind": "claim",
                "text": "Settings is opened by buttonSettings.",
                "context_id": task.context_id,
            }
        ],
    )

    result = reconciliation.submit(P, report)
    corrected = store.load(P)

    assert result["status"] == "applied"
    assert "settings" in corrected.screens and outcome.name not in corrected.screens
    assert any(item.agent == "codex" for item in corrected.knowledge)
    event = result["event"]
    rollback_id = event["rollback_id"]
    reconciliation.rollback(P, rollback_id)
    restored = store.load(P)
    assert outcome.name in restored.screens and "settings" not in restored.screens


def test_review_verdict_is_queued_without_mutating_map(tmp_path) -> None:
    store = _store(tmp_path)
    outcome = store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")
    reconciliation = ReconciliationStore(store)
    task = next(task for task in reconciliation.plan(P) if task.issue_type == "poor_name")
    result = reconciliation.submit(
        P,
        ResearchReport(
            task_id=task.id,
            agent="codex",
            verdict="review",
            rationale="Runtime evidence is incomplete.",
            operations=[CorrectionOperation(op="rename", screen_id=outcome.name, value="settings")],
        ),
    )

    assert result["status"] == "review"
    assert outcome.name in store.load(P).screens
    assert reconciliation.status(P)["reports"]


def test_cli_knowledge_and_reconcile_contract(tmp_path) -> None:
    store = AppMemoryStore(make_config().memory)
    store.record_screen(package=P, elements=_elements(HOME), name_hint="screen")

    added = runner.invoke(
        cli_app,
        [
            "knowledge",
            "add",
            "--app",
            P,
            "--kind",
            "claim",
            "--text",
            "Catalog is controlled by a remote experiment.",
            "--agent",
            "codex",
        ],
    )
    assert added.exit_code == 0, added.stderr
    listed = runner.invoke(cli_app, ["knowledge", "list", "--app", P])
    assert listed.exit_code == 0
    assert json.loads(listed.stdout)["knowledge"]

    planned = runner.invoke(cli_app, ["reconcile", "plan", "--app", P])
    assert planned.exit_code == 0, planned.stderr
    task = next(
        item
        for item in json.loads(planned.stdout)["tasks"]
        if item["issue_type"] == "poor_name"
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "task_id": task["id"],
                "agent": "codex",
                "verdict": "apply",
                "rationale": "The source destination is catalog.",
                "operations": [
                    {
                        "op": "rename",
                        "screen_id": task["affected_ids"][0],
                        "value": "catalog",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    submitted = runner.invoke(
        cli_app,
        ["reconcile", "submit", "--app", P, str(report_path)],
    )
    assert submitted.exit_code == 0, submitted.stderr
    assert json.loads(submitted.stdout)["status"] == "applied"
    assert "catalog" in store.load(P).screens
