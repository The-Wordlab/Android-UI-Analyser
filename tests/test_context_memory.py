"""Schema-v4 contexts, route trust, knowledge provenance, and correction."""

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
from test_memory import APPS, HOME, SETTINGS_XML, P, _elements, _hier, _node, _store

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

    assert migrated is not None and migrated.schema_version == 4
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
    assert "catalog_experiment_b" in second.name
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


def test_memory_update_renames_the_screen_in_its_active_flag_context(tmp_path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    serial = "rename-context"
    flags = {"layout_experiment": "a"}
    context_id = store.activate_flag_context(serial, P, flags, verified=True)
    old = store.record_screen(
        package=P,
        elements=_elements(HOME),
        name_hint="inbox",
        context_id=context_id,
        context_flags=flags,
        context_verified=True,
    )
    sess = store.load_session(serial)
    sess.current_screen = old.name
    store.save_session(serial, sess)
    engine = Engine(
        cfg,
        device=FakeDevice(hierarchy_xml=HOME, package=P, serial=serial),
        factory=ProviderFactory(cfg),
    )

    result = engine.memory_update("workspace_alerts_inbox")

    app = AppMemoryStore(cfg.memory).load(P)
    assert result["known"] is True
    assert result["created"] is False
    assert result["screen"] == "workspace_alerts_inbox"
    assert old.name not in app.screens
    renamed = app.screens["workspace_alerts_inbox"]
    assert renamed.context_id == context_id
    assert renamed.name_source == "explicit"


def test_stable_route_namespaces_disambiguate_a_short_destination_title(tmp_path) -> None:
    store = _store(tmp_path)
    serial = "contextual-name"
    origin = store.record_screen(
        package=P,
        elements=_elements(HOME),
        name_hint="workspace",
        screen_height=800,
    )
    sess = store.load_session(serial)
    sess.package = P
    sess.current_screen = origin.name
    store.save_session(serial, sess)
    store.observe_action(
        serial,
        RouteStep(kind="tap", label="Messages", resource_id="workspaceTabMESSAGES"),
    )
    store.observe_action(
        serial,
        RouteStep(kind="tap", label="Alerts", resource_id="workspaceAlerts"),
    )
    destination = _hier(
        _node("android.widget.TextView", text="Inbox", b="[40,120][1040,210]"),
        _node("android.widget.Button", text="Create", clk=True, b="[40,300][1040,400]"),
    )

    store.observe_screen(
        serial,
        package=P,
        elements=_elements(destination),
        screen_height=800,
    )

    app = store.load(P)
    record = app.screens["workspace_messages_inbox"]
    assert record.name_source == "resource"
    assert record.logical_name == "workspace_messages_inbox"
    assert any(route.to_screen == record.name for route in app.routes)


def test_resource_namespace_keeps_name_stable_across_locales(tmp_path) -> None:
    store = _store(tmp_path)
    english = _hier(
        _node(
            "android.widget.TextView",
            text="Saved Items",
            rid="x:id/savedItemsLibraryRoot",
            b="[40,120][1040,210]",
        ),
        _node(
            "android.widget.Button",
            text="Create an item",
            rid="x:id/savedItemsLibraryCreateItem",
            clk=True,
            b="[40,300][1040,400]",
        ),
    )
    spanish = _hier(
        _node(
            "android.widget.TextView",
            text="Elementos guardados",
            rid="x:id/savedItemsLibraryRoot",
            b="[40,120][1040,210]",
        ),
        _node(
            "android.widget.Button",
            text="Crear un elemento",
            rid="x:id/savedItemsLibraryCreateItem",
            clk=True,
            b="[40,300][1040,400]",
        ),
    )

    first = store.record_screen(package=P, elements=_elements(english), screen_height=800)
    second = store.record_screen(package=P, elements=_elements(spanish), screen_height=800)
    record = store.load(P).screens[first.name]

    assert first.name == "saved_items_library"
    assert second.was_known and second.name == first.name
    assert record.name_source == "resource"
    assert "elementos_guardados" in record.aliases


def test_generic_shell_roots_do_not_override_destination_resource(tmp_path) -> None:
    store = _store(tmp_path)
    conversation = _hier(
        _node(
            "android.widget.FrameLayout",
            rid="x:id/toolbarRoot",
            b="[0,80][1080,220]",
        ),
        _node(
            "android.widget.TextView",
            text="A dynamic title",
            rid="x:id/conversationTitleView",
            b="[40,120][1040,210]",
        ),
    )

    outcome = store.record_screen(
        package=P, elements=_elements(conversation), screen_height=800
    )

    assert outcome.name == "conversation"


def test_long_toolbar_title_beats_short_action_and_inbound_banner(tmp_path) -> None:
    store = _store(tmp_path)
    detail = _hier(
        _node(
            "android.widget.TextView",
            text="Personal Budget Counter Demonstration",
            b="[160,90][900,170]",
        ),
        _node(
            "android.widget.Button",
            text="Share",
            rid="x:id/detailShare",
            clk=True,
            b="[850,200][1040,300]",
        ),
    )

    outcome = store.record_screen(
        package=P,
        elements=_elements(detail),
        inbound_label="Tap to open",
        inbound_resource_id="catalogBuildBannerReady",
        inbound_kind="tap",
        screen_height=800,
    )
    record = store.load(P).screens[outcome.name]

    assert outcome.name == "personal_budget_counter"
    assert record.name_source == "title"


def test_repeated_resource_namespace_names_detail_without_top_title(tmp_path) -> None:
    store = _store(tmp_path)
    detail = _hier(
        _node(
            "android.view.View",
            text="Open app",
            rid="x:id/articleDetailOpenApp",
            clk=True,
            b="[40,400][1040,500]",
        ),
        _node(
            "android.view.View",
            text="Share",
            rid="x:id/articleDetailShare",
            clk=True,
            b="[40,600][1040,700]",
        ),
        _node(
            "android.view.View",
            text="Save",
            rid="x:id/articleDetailSave",
            clk=True,
            b="[40,720][1040,820]",
        ),
    )

    outcome = store.record_screen(
        package=P,
        elements=_elements(detail),
        inbound_label="Tap to open",
        inbound_resource_id="catalogBuildBannerReady",
        inbound_kind="tap",
        screen_height=1000,
    )

    assert outcome.name == "article_detail"
    assert store.load(P).screens[outcome.name].name_source == "resource"


def test_loading_and_error_are_grouped_as_states_of_one_screen(tmp_path) -> None:
    store = _store(tmp_path)
    loading = _hier(
        _node(
            "android.widget.TextView",
            text="Loading",
            rid="x:id/catalogGridContent",
            b="[40,120][1040,210]",
        )
    )
    error = _hier(
        _node(
            "android.widget.TextView",
            text="Oops! Something went wrong. Try again",
            rid="x:id/catalogGridContent",
            b="[40,120][1040,210]",
        )
    )

    first = store.record_screen(package=P, elements=_elements(loading), screen_height=800)
    second = store.record_screen(package=P, elements=_elements(error), screen_height=800)
    app_map = store.load(P)
    records = [app_map.screens[first.name], app_map.screens[second.name]]

    assert first.name != second.name
    assert {record.logical_name for record in records} == {"catalog_grid"}
    assert {record.state for record in records} == {"loading", "error"}
    rendered = render_map(app_map)
    assert "state: loading" in rendered and "state: error" in rendered


def test_inbound_resource_names_webview_runtime_without_visible_title(tmp_path) -> None:
    store = _store(tmp_path)
    serial = "semantic-route"
    runtime = _hier(
        _node(
            "android.webkit.WebView",
            rid="x:id/containerCounterRuntime",
            b="[0,80][1080,2300]",
        )
    )
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial,
        RouteStep(
            kind="tap",
            label="Open app",
            resource_id="openCounterRuntimeButton",
        ),
    )
    store.observe_screen(serial, package=P, elements=_elements(runtime), screen_height=800)

    app_map = store.load(P)
    runtime_record = app_map.screens["counter_runtime"]
    edge = next(route for route in app_map.routes if route.to_screen == "counter_runtime")
    assert runtime_record.surface == "webview"
    assert runtime_record.name_source == "resource"
    assert edge.status == "provisional"


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


def test_research_hints_are_scoped_to_active_flag_context(tmp_path) -> None:
    store = _store(tmp_path)
    flags_a = {"catalog_experiment": "a"}
    flags_b = {"catalog_experiment": "b"}
    context_a = context_id_for_flags(flags_a)
    context_b = context_id_for_flags(flags_b)
    store.record_screen(
        package=P,
        elements=_elements(HOME),
        context_id=context_a,
        context_flags=flags_a,
    )
    app_map = store.load(P)
    app_map.research_tasks = [
        {
            "id": "task-a",
            "context_id": context_a,
            "status": "open",
            "questions": ["Research variant A"],
        },
        {
            "id": "task-b",
            "context_id": context_b,
            "status": "open",
            "questions": ["Research variant B"],
        },
    ]
    store.save(app_map)
    store.activate_flag_context("research-device", P, flags_b, verified=True)

    hints = store.navigation_hints("research-device", P)

    assert hints.research_tasks == ["research task-b: Research variant B"]


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


def test_auto_routes_are_provisional_until_observed_twice(tmp_path) -> None:
    store = _store(tmp_path)
    serial = "route-verification"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial, RouteStep(kind="tap", label="Apps", resource_id="nav_apps")
    )
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)

    first_map = store.load(P)
    edge = next(route for route in first_map.routes if route.to_screen == "apps")
    assert edge.status == "provisional"
    assert _shortest_path(first_map, "apps", start=edge.from_screen) == []
    assert any(
        task["issue_type"] == "provisional_route"
        for task in first_map.research_tasks
    )

    store.observe_action(serial, RouteStep(kind="key", arg="back"))
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(
        serial, RouteStep(kind="tap", label="Apps", resource_id="nav_apps")
    )
    store.observe_screen(serial, package=P, elements=_elements(APPS), screen_height=800)

    verified_map = store.load(P)
    edge = next(route for route in verified_map.routes if route.to_screen == "apps")
    assert edge.status == "verified"
    assert edge.verification_count == 1
    assert len(_shortest_path(verified_map, "apps", start=edge.from_screen)) == 1
    assert not any(
        task["issue_type"] == "provisional_route"
        and edge.id in task["affected_ids"]
        for task in verified_map.research_tasks
    )


def test_unlabeled_route_is_rejected_and_pushed_as_research(tmp_path) -> None:
    store = _store(tmp_path)
    serial = "route-rejection"
    store.observe_screen(serial, package=P, elements=_elements(HOME), screen_height=800)
    store.observe_action(serial, RouteStep(kind="tap"))
    store.observe_screen(
        serial, package=P, elements=_elements(SETTINGS_XML), screen_height=800
    )
    app_map = store.load(P)
    edge = app_map.routes[0]
    hints = store.navigation_hints(serial, P)

    assert edge.status == "rejected"
    assert edge.rejection_reason == "destination action has no durable selector"
    assert "tap [unlabeled]" not in render_map(app_map)
    task = next(
        task for task in app_map.research_tasks if task["issue_type"] == "unreplayable_route"
    )
    assert edge.id in task["affected_ids"]
    assert any(str(task["id"]) in prompt for prompt in hints.research_tasks)
    assert any(issue.type == "unreplayable_route" for issue in audit_map(app_map).issues)


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
