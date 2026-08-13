"""Regressions for recursive flow disclosure and immutable preflight resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_ui_analyser import flags as flags_mod
from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from android_ui_analyser.flows import Flow, FlowStore
from android_ui_analyser.memory import RouteStep
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from conftest import FakeDevice, make_config

P = "com.example.app"


def _engine(tmp_path: Path, *, memory: bool = True) -> Engine:
    return Engine(
        make_config(
            memory={"enabled": memory, "dir": str(tmp_path / "memory")},
            cache={"dir": str(tmp_path / "cache")},
            daemon={"enabled": False},
        ),
        device=FakeDevice(package=P, serial="resolved-preflight"),
    )


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=P, source="hierarchy"),
        elements=[],
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="home",
            device_serial="resolved-preflight",
        ),
    )


def test_dry_run_discloses_recursive_child_graph_risks_and_paths(tmp_path: Path) -> None:
    root = tmp_path / "flows"
    root.mkdir()
    (root / "flags.yaml").write_text("flags:\n  catalog_variant: preview\n", encoding="utf-8")
    (root / "grandchild.yaml").write_text(
        f"name: grandchild\napp: {P}\nsteps:\n  - network_offline\n",
        encoding="utf-8",
    )
    (root / "child.yaml").write_text(
        f"name: child\napp: {P}\nsteps:\n"
        "  - flags_apply: flags.yaml\n"
        "  - input: {id: prompt, text: hello}\n"
        "  - tap: Delete account\n"
        "  - flow: grandchild.yaml\n",
        encoding="utf-8",
    )
    parent = root / "parent.yaml"
    parent.write_text(
        f"name: parent\napp: {P}\narrival: rid:ready\nsteps:\n  - flow: child.yaml\n",
        encoding="utf-8",
    )

    result = _engine(tmp_path).flow_run(file=str(parent), dry_run=True)

    assert result["ok"] is True and result["would_execute"] is False
    assert set(result["effects"]) >= {
        "nested_execution",
        "settings_mutation",
        "data_mutation",
        "destructive",
        "environment_mutation",
    }
    by_code = {risk["code"]: risk["path"] for risk in result["risks"]}
    assert by_code["nested_execution"] == "steps[0].resolved_flow.steps[3]"
    assert by_code["settings_mutation"] == "steps[0].resolved_flow.steps[0]"
    assert by_code["data_mutation"] == "steps[0].resolved_flow.steps[1]"
    assert by_code["destructive"] == "steps[0].resolved_flow.steps[2]"
    assert by_code["environment_mutation"] == (
        "steps[0].resolved_flow.steps[3].resolved_flow.steps[0]"
    )
    assert [edge["reference"] for edge in result["flow_graph"]] == [
        "child.yaml",
        "grandchild.yaml",
    ]
    child = result["steps"][0]["resolved_flow"]
    assert child["steps"][3]["resolved_flow"]["steps"][0]["step"] == "network-offline"
    assert result["steps"][0]["destructive"] is True


def test_goal_candidate_discloses_resolved_child_risk_but_cannot_authorize_it(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    store = FlowStore(engine.config.memory)
    store.save(
        Flow(
            name="offline_child",
            app=P,
            steps=[RouteStep(kind="network-offline")],
        )
    )
    store.save(
        Flow(
            name="open_catalog_offline",
            app=P,
            aliases=["prepare offline catalog"],
            arrival="rid:catalogReady",
            steps=[RouteStep(kind="flow", arg="offline_child")],
        )
    )

    plan = engine._goal_session_plan("prepare offline catalog", _observation())

    candidate = next(item for item in plan.candidates if item.id == "flow:open_catalog_offline")
    assert {risk["code"] for risk in candidate.risks} == {
        "nested_execution",
        "environment_mutation",
    }
    assert candidate.evidence["resolved_effects"] == [
        "environment_mutation",
        "nested_execution",
    ]
    child = candidate.evidence["resolved_steps"][0]["resolved_flow"]
    assert child["steps"][0]["risks"][0]["code"] == "environment_mutation"
    assert candidate.safe is False
    assert candidate.call.executes is False
    assert candidate.call.mcp["arguments"]["dry_run"] is True
    assert plan.selected_candidate is None


def test_flow_execution_uses_preflighted_flags_after_file_changes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, memory=False)
    root = tmp_path / "flows"
    root.mkdir()
    flags_path = root / "flags.yaml"
    flags_path.write_text(f"app: {P}\nflags:\n  variant: original\n", encoding="utf-8")
    flow_path = root / "flags_parent.yaml"
    flow_path.write_text(
        f"name: flags_parent\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - flags_apply: flags.yaml\n",
        encoding="utf-8",
    )
    original_loader = flags_mod.load_flags_file
    loads: list[str] = []

    def load(path: str | Path) -> tuple[str | None, dict[str, str]]:
        loads.append(str(path))
        return original_loader(path)

    monkeypatch.setattr(flags_mod, "load_flags_file", load)
    applied: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        engine,
        "flags_set",
        lambda package, pairs, **_kwargs: applied.append((package, dict(pairs))) or {"ok": True},
    )
    journal: list[RouteStep] = []
    monkeypatch.setattr(engine, "_record_action_safe", journal.append)
    original_key = engine.key

    def key(name: str, **kwargs: Any) -> Any:
        result = original_key(name, **kwargs)
        flags_path.write_text(
            "app: com.example.changed\nflags:\n  variant: changed\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(engine, "key", key)

    result = engine.flow_run(file=str(flow_path))

    assert result["ok"] is True
    assert loads == [str(flags_path.resolve())]
    assert applied == [(P, {"variant": "original"})]
    assert [step.kind for step in journal] == ["key", "flags-apply"]
    assert journal[-1].arg == str(flags_path.resolve())
    assert ("press", ("back",)) in engine.device.calls


def test_flow_execution_uses_preflighted_cassette_after_file_breaks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    engine = _engine(tmp_path, memory=False)
    cassette = pm.cassette_dir(engine.config.memory.dir) / "original.yaml"
    original_rule = pm.map_rule("GET", "/original", body={"source": "preflight"})
    pm.save_cassette(cassette, "original", [original_rule])
    flow_path = tmp_path / "cassette_parent.yaml"
    flow_path.write_text(
        f"name: cassette_parent\napp: {P}\nsteps:\n"
        "  - key: back\n"
        "  - mock_replay: original\n",
        encoding="utf-8",
    )
    original_loader = pm.load_cassette
    loads: list[Path] = []

    def load(path: Path) -> list[dict[str, Any]]:
        loads.append(path)
        return original_loader(path)

    monkeypatch.setattr(pm, "load_cassette", load)
    journal: list[RouteStep] = []
    monkeypatch.setattr(engine, "_record_action_safe", journal.append)
    original_key = engine.key

    def key(name: str, **kwargs: Any) -> Any:
        result = original_key(name, **kwargs)
        cassette.write_text("entries: [", encoding="utf-8")
        return result

    monkeypatch.setattr(engine, "key", key)

    result = engine.flow_run(file=str(flow_path))

    assert result["ok"] is True
    assert loads == [cassette]
    assert pm.load_rules(pm.rules_path(engine.config.cache.dir)) == [original_rule]
    assert [step.kind for step in journal] == ["key", "mock-replay"]
    assert journal[-1].arg == "original"
    assert ("press", ("back",)) in engine.device.calls
