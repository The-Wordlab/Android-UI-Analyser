"""Flow files reject coercions, round-trip composites, and write atomically."""

from __future__ import annotations

import json
import threading
import tomllib
from pathlib import Path

import pytest

import android_ui_analyser.atomic as atomic_mod
from android_ui_analyser import __version__
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import (
    Flow,
    FlowStore,
    check_saveable,
    parse_flow_yaml,
    recorded_step_blockers,
    render_flow_yaml,
    resolve_params,
    steps_from_recent,
    validate_resolved_steps,
)
from android_ui_analyser.memory import RouteStep, route_step_risks
from conftest import make_config


@pytest.mark.parametrize(
    "yaml_text",
    [
        "name: {bad: type}\nsteps: [{key: back}]\n",
        "app: [bad]\nsteps: [{key: back}]\n",
        "context_id: {bad: type}\nsteps: [{key: back}]\n",
        "arrival: [bad]\nsteps: [{key: back}]\n",
        "aliases: [valid, 3]\nsteps: [{key: back}]\n",
        "params: {COUNT: 3}\nsteps: [{key: back}]\n",
        "schema_version: true\nsteps: [{key: back}]\n",
        "unknown: value\nsteps: [{key: back}]\n",
        "steps: [{input: {id: field, text: value, submit: 'false'}}]\n",
        "steps: [{wait_for: {text: ready, timeout_ms: '3'}}]\n",
        "steps: [{wait_for: {text: ready, timeout_ms: -1}}]\n",
        "steps: [{repeat: {times: '2', steps: [{key: back}]}}]\n",
        "steps: [{repeat: {times: 0, steps: [{key: back}]}}]\n",
        "steps: [{retry: {max_retries: false, steps: [{key: back}]}}]\n",
        "arrival_status: mapped\nsteps: [{key: back}]\n",
        "arrival_status: predicate_verified\nsteps: [{key: back}]\n",
        "arrival_status: unverified\narrival_screen: home\nsteps: [{key: back}]\n",
        "arrival_status: maybe\nsteps: [{key: back}]\n",
    ],
)
def test_malformed_flow_yaml_is_always_a_usage_error(yaml_text: str) -> None:
    with pytest.raises(UsageError):
        parse_flow_yaml(yaml_text)


@pytest.mark.parametrize(
    "step",
    [
        "key: unsupported",
        "key: -1",
        'tap_point: "not-a-point"',
        'tap_point: "-1,2"',
        "swipe: diagonal",
        "scroll: forward",
        "a11y_scroll: {id: list, direction: sideways}",
        "dev_profile: battery",
        "network_profile: airplane",
    ],
)
def test_finite_executor_arguments_are_rejected_while_loading(step: str) -> None:
    with pytest.raises(UsageError):
        parse_flow_yaml(f"steps:\n  - {step}\n")


@pytest.mark.parametrize(
    ("step", "kind", "arg"),
    [
        ("key: BACK", "key", "BACK"),
        ("key: KEYCODE_ESCAPE", "key", "KEYCODE_ESCAPE"),
        ("key: 4", "key", "4"),
        ('tap_point: "1.6, 2.4"', "tap-point", "2,2"),
        ("swipe: UP", "swipe", "up"),
        ("scroll: left", "scroll", "left"),
        ("a11y_scroll: {id: list, direction: FWD}", "a11y-scroll", "fwd"),
        ("dev_profile: AC", "dev-profile", "ac"),
        ("network_profile: wifi_only", "network-profile", "wifi-only"),
        ("network_profile: cellular-only", "network-profile", "cellular-only"),
        ("network_profile: slow", "network-profile", "slow"),
        ("network_profile: lossy", "network-profile", "lossy"),
    ],
)
def test_finite_executor_arguments_parse_to_runnable_steps(step: str, kind: str, arg: str) -> None:
    parsed = parse_flow_yaml(f"steps:\n  - {step}\n").steps[0]
    assert parsed.kind == kind and parsed.arg == arg


@pytest.mark.parametrize(
    ("step", "value", "kind", "arg"),
    [
        ("key: '${VALUE}'", "back", "key", "back"),
        ("tap_point: '${VALUE}'", "1.6, 2.4", "tap-point", "2,2"),
        ("swipe: '${VALUE}'", "UP", "swipe", "up"),
        ("scroll: '${VALUE}'", "left", "scroll", "left"),
        (
            "a11y_scroll: {id: list, direction: '${VALUE}'}",
            "FWD",
            "a11y-scroll",
            "fwd",
        ),
        ("dev_profile: '${VALUE}'", "AC", "dev-profile", "ac"),
        ("network_profile: '${VALUE}'", "wifi_only", "network-profile", "wifi-only"),
    ],
)
def test_finite_executor_arguments_validate_after_parameter_resolution(
    step: str, value: str, kind: str, arg: str
) -> None:
    flow = parse_flow_yaml(f"params: {{VALUE: ''}}\nsteps:\n  - {step}\n")
    assert flow.steps[0].arg == "${VALUE}"

    resolved = resolve_params(flow, {"VALUE": value})[0]

    assert resolved.kind == kind and resolved.arg == arg


@pytest.mark.parametrize(
    ("step", "value"),
    [
        ("key: '${VALUE}'", "unsupported"),
        ("tap_point: '${VALUE}'", "-1,2"),
        ("swipe: '${VALUE}'", "diagonal"),
        ("scroll: '${VALUE}'", "forward"),
        ("a11y_scroll: {id: list, direction: '${VALUE}'}", "sideways"),
        ("dev_profile: '${VALUE}'", "battery"),
        ("network_profile: '${VALUE}'", "airplane"),
    ],
)
def test_invalid_finite_parameter_is_rejected_after_resolution(step: str, value: str) -> None:
    flow = parse_flow_yaml(f"params: {{VALUE: ''}}\nsteps:\n  - {step}\n")
    with pytest.raises(UsageError):
        resolve_params(flow, {"VALUE": value})


def test_resolved_step_validation_is_pure_and_recursive() -> None:
    steps = [
        RouteStep(
            kind="repeat",
            repeat=1,
            substeps=[RouteStep(kind="network-profile", arg="cellular_only")],
        )
    ]

    validated = validate_resolved_steps(steps)

    assert steps[0].substeps[0].arg == "cellular_only"
    assert validated[0].substeps[0].arg == "cellular-only"
    with pytest.raises(UsageError):
        validate_resolved_steps(
            [
                RouteStep(
                    kind="retry",
                    max_retries=1,
                    substeps=[RouteStep(kind="swipe", arg="diagonal")],
                )
            ]
        )


@pytest.mark.parametrize(
    "step",
    [
        'tap: ""',
        'long_press: ""',
        'clear: ""',
        'tap_point: ""',
        'key: ""',
        'swipe: ""',
        'scroll: ""',
        'scroll_to: ""',
        'wait_for: ""',
        'assert_visible: ""',
        'assert_not_visible: ""',
        'launch_app: ""',
        'stop_app: ""',
        'open_link: ""',
        'goto: ""',
        'flow: ""',
        'dev_profile: ""',
        'flags_apply: ""',
        'mock_replay: ""',
        'network_profile: ""',
    ],
)
def test_empty_scalar_selector_or_argument_is_rejected(step: str) -> None:
    with pytest.raises(UsageError, match="must not be empty"):
        parse_flow_yaml(f"steps:\n  - {step}\n")


@pytest.mark.parametrize(
    "step",
    [
        'tap: "   "',
        'open_link: "   "',
        'goto: {screen: "   "}',
        'tap: {id: "   "}',
        'tap: {desc: "   "}',
        'launch_app: {package: "   "}',
    ],
)
def test_whitespace_only_selector_or_argument_is_rejected(step: str) -> None:
    with pytest.raises(UsageError, match="must not be empty"):
        parse_flow_yaml(f"steps:\n  - {step}\n")


@pytest.mark.parametrize(
    "header",
    [
        'name: "   "',
        'app: "   "',
        'context_id: "   "',
        'arrival: "   "',
        'aliases: ["   "]',
        'params: {"   ": value}',
    ],
)
def test_whitespace_only_top_level_identity_or_evidence_is_rejected(header: str) -> None:
    with pytest.raises(UsageError, match="non-empty|must not be empty"):
        parse_flow_yaml(f"{header}\nsteps: [{{key: back}}]\n")


def test_nonempty_scalar_content_is_preserved_without_trimming() -> None:
    flow = parse_flow_yaml(
        'name: " padded name "\n'
        'app: " com.example.catalog "\n'
        'context_id: " feature-on "\n'
        'arrival: " text:Ready "\n'
        'aliases: [" open catalog "]\n'
        'params: {" TARGET ": " value "}\n'
        'steps: [{tap: " Continue "}]\n'
    )

    assert flow.name == " padded name "
    assert flow.app == " com.example.catalog "
    assert flow.context_id == " feature-on "
    assert flow.arrival == " text:Ready "
    assert flow.aliases == [" open catalog "]
    assert flow.params == {" TARGET ": " value "}
    assert flow.steps[0].label == " Continue "


def test_composites_round_trip_and_resolve_nested_params() -> None:
    flow = parse_flow_yaml(
        """
name: nested
params: {TARGET: Save, PACKAGE: com.example.catalog}
steps:
  - repeat:
      times: 2
      steps:
        - tap: {id: "button${TARGET}", text: "${TARGET}", by: id}
        - retry:
            max_retries: 4
            steps:
              - launch_app: {package: "${PACKAGE}", activity: ".Main${TARGET}"}
"""
    )
    reparsed = parse_flow_yaml(render_flow_yaml(flow))
    assert reparsed.steps == flow.steps
    resolved = resolve_params(reparsed, {})
    nested_tap = resolved[0].substeps[0]
    launch = resolved[0].substeps[1].substeps[0]
    assert nested_tap.resource_id == "buttonSave" and nested_tap.label == "Save"
    assert launch.arg == "com.example.catalog" and launch.activity == ".MainSave"
    with pytest.raises(UsageError, match="TARGET"):
        resolve_params(
            parse_flow_yaml(
                "params: {TARGET: ''}\nsteps: [{repeat: {times: 2, steps: [{tap: '${TARGET}'}]}}]\n"
            ),
            {},
        )


def test_nested_materialization_and_risk_survive_round_trip() -> None:
    steps, params = steps_from_recent(
        [
            RouteStep(
                kind="repeat",
                repeat=2,
                substeps=[
                    RouteStep(kind="input", resource_id="field"),
                    RouteStep(kind="tap", resource_id="deleteAccount", by="id"),
                ],
            )
        ]
    )
    flow = Flow(name="nested", params=params, steps=steps)
    check_saveable(flow)
    reparsed = parse_flow_yaml(render_flow_yaml(flow))
    assert reparsed.steps[0].repeat == 2
    assert reparsed.steps[0].substeps[0].text == "${PARAM_1}"
    risks = route_step_risks(reparsed.steps[0], origin_package=None, destructive_labels=["delete"])
    assert any(risk["code"] == "destructive" for risk in risks)


@pytest.mark.parametrize(
    ("step", "field"),
    [
        (RouteStep(kind="tap", resource_id="${RID}", by="id"), "RID"),
        (RouteStep(kind="tap", label="Continue", package="${PACKAGE}"), "PACKAGE"),
        (
            RouteStep(kind="launch-app", arg="com.example.app", activity="${ACTIVITY}"),
            "ACTIVITY",
        ),
        (RouteStep(kind="swipe", direction="${DIRECTION}"), "DIRECTION"),
    ],
)
def test_check_saveable_rejects_unbound_params_in_every_materialized_field(
    step: RouteStep,
    field: str,
) -> None:
    with pytest.raises(UsageError, match=rf"{field}|{field.lower()}"):
        check_saveable(Flow(name=f"missing_{field.lower()}", steps=[step]))


@pytest.mark.parametrize(
    "step",
    [
        RouteStep(kind="double-tap", resource_id="row"),
        RouteStep(kind="a11y-action", resource_id="row", arg="CLICK"),
        RouteStep(kind="swipe", arg="coords"),
        RouteStep(kind="swipe", arg="up"),
        RouteStep(kind="scroll", arg="up"),
        RouteStep(kind="scroll-to", arg="Target"),
        RouteStep(kind="long-press", resource_id="row"),
        RouteStep(kind="paste"),
        RouteStep(kind="open-link", arg="fiction://item"),
    ],
)
def test_lossy_recorded_action_is_a_structured_save_blocker(step: RouteStep) -> None:
    blockers = recorded_step_blockers([step])
    assert len(blockers) == 1 and step.kind in blockers[0]


def test_recorded_a11y_scroll_description_round_trips_strictly() -> None:
    step = RouteStep(kind="a11y-scroll", content_desc="Catalog list", by="desc", arg="down")
    assert recorded_step_blockers([step]) == []
    loaded = parse_flow_yaml(render_flow_yaml(Flow(name="a11y", steps=[step]))).steps[0]
    assert loaded.content_desc == "Catalog list" and loaded.by == "desc" and loaded.arg == "down"


def test_flow_store_non_force_creation_is_race_safe(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def save(label: str) -> None:
        barrier.wait()
        try:
            store.save(Flow(name="race", steps=[RouteStep(kind="tap", label=label)]))
            outcomes.append("saved")
        except UsageError:
            outcomes.append("exists")

    threads = [threading.Thread(target=save, args=(label,)) for label in ("First", "Second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["exists", "saved"]
    assert parse_flow_yaml(store.path("race").read_text()).steps[0].label in {"First", "Second"}


def test_flow_store_force_failure_preserves_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.save(Flow(name="atomic", steps=[RouteStep(kind="tap", label="Old")]))
    old = store.path("atomic").read_text()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("fictional replace failure")

    monkeypatch.setattr(atomic_mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="fictional"):
        store.save(Flow(name="atomic", steps=[RouteStep(kind="tap", label="New")]), force=True)
    assert store.path("atomic").read_text() == old
    assert not list(store.flows_dir().glob("*.tmp"))


def test_flow_store_list_contains_even_non_parser_file_error(tmp_path: Path) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.flows_dir().mkdir(parents=True)
    store.path("broken").write_bytes(b"\xff")
    listed = store.list()
    assert listed[0]["name"] == "broken" and "error" in listed[0]


def test_flow_store_list_does_not_guess_compatibility_without_active_context(
    tmp_path: Path,
) -> None:
    store = FlowStore(make_config(memory={"dir": str(tmp_path)}).memory)
    store.save(
        Flow(
            name="contextual",
            app="com.example.catalog",
            context_id="feature-on",
            steps=[RouteStep(kind="key", arg="back")],
        )
    )
    assert store.list(active_package="com.example.catalog")[0]["context_compatible"] is None
    assert (
        store.list(active_package="com.example.catalog", active_context_id="feature-on")[0][
            "context_compatible"
        ]
        is True
    )


def test_runtime_and_plugin_versions_match() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    plugin = json.loads((root / ".claude-plugin/plugin.json").read_text())
    marketplace = json.loads((root / ".claude-plugin/marketplace.json").read_text())
    assert __version__ == project["project"]["version"] == plugin["version"] == "0.11.5"
    assert marketplace["plugins"][0]["version"] == __version__
