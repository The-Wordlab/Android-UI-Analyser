"""Rich flow assertions and portable execution evidence."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import parse_flow_yaml, render_flow_yaml, resolve_params
from conftest import FakeDevice, make_config

_CATALOG = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.TextView" text="Starter"
        resource-id="com.example.app:id/productTitle" enabled="true"
        bounds="[20,100][220,160]"/>
  <node index="1" class="android.widget.TextView" text="Professional"
        resource-id="com.example.app:id/productTitle" enabled="true"
        bounds="[300,100][540,160]"/>
  <node index="2" class="android.widget.Switch" text="Show available"
        resource-id="com.example.app:id/availability" checkable="true" checked="true"
        enabled="true" bounds="[20,300][540,380]"/>
</hierarchy>
"""


_RICH_FLOW = """name: catalog_regression
params:
  FIRST: Starter
steps:
  - assert: {id: productTitle, count: 2}
  - assert: {id: availability, checked: true, enabled: true}
  - assert_order:
      axis: horizontal
      selectors:
        - {text: "${FIRST}"}
        - {text: Professional}
  - screenshot: sorted_catalog
"""


def _engine(tmp_path: Path) -> tuple[Engine, FakeDevice]:
    config = make_config(
        memory={"dir": str(tmp_path / "memory")},
        cache={"dir": str(tmp_path / "cache")},
        daemon={"enabled": False},
    )
    device = FakeDevice(hierarchy_xml=_CATALOG)
    return Engine(config, device=device), device


def test_rich_flow_steps_parse_render_and_substitute() -> None:
    flow = parse_flow_yaml(_RICH_FLOW)
    assert [step.kind for step in flow.steps] == [
        "assert",
        "assert",
        "assert-order",
        "screenshot",
    ]
    assert flow.steps[0].assertion == {"count": 2}
    assert flow.steps[1].assertion == {"checked": True, "enabled": True}
    resolved = resolve_params(flow, {})
    assert resolved[2].assertion["selectors"][0]["text"] == "Starter"

    rendered = render_flow_yaml(flow)
    reparsed = parse_flow_yaml(rendered)
    assert [step.model_dump() for step in reparsed.steps] == [
        step.model_dump() for step in flow.steps
    ]


@pytest.mark.parametrize(
    "body",
    [
        "steps:\n  - assert: {text: Ready, exists: false}\n",
        "steps:\n  - assert: {text: Ready, exists: true, count: 0}\n",
        "steps:\n  - assert: {text: Ready, checked: true, count: 0}\n",
    ],
)
def test_rich_assertion_rejects_contradictory_predicates(body: str) -> None:
    with pytest.raises(UsageError):
        parse_flow_yaml(body)


def test_flow_run_requires_one_source_and_artifact_dir_for_evidence(tmp_path: Path) -> None:
    engine, _device = _engine(tmp_path)
    body = "steps:\n  - assert: {text: Starter}\n"
    with pytest.raises(UsageError, match="exactly one"):
        engine.flow_run(name="saved", yaml=body)
    with pytest.raises(UsageError, match="artifacts-dir"):
        engine.flow_run(yaml=body, evidence="all")


def test_inline_flow_writes_complete_artifact_and_junit_bundle(tmp_path: Path) -> None:
    engine, device = _engine(tmp_path)
    requested = tmp_path / "artifacts"

    result = engine.flow_run(
        yaml=_RICH_FLOW,
        artifacts_dir=str(requested),
        evidence="all",
        junit=True,
    )

    assert result["ok"] is True, result
    assert result["source"] == "inline_yaml"
    assert result["run_id"].startswith("flow-")
    assert result["duration_ms"] >= 0
    assert len(result["steps_run"]) == 4
    assert all(row["evidence_id"].startswith(result["run_id"]) for row in result["steps_run"])
    assert all("duration_ms" in row for row in result["steps_run"])
    assert device.screenshot_calls == 4  # three automatic step frames + named checkpoint

    artifact_dir = Path(result["artifacts"]["dir"])
    for relative in (
        "flow.yaml",
        "result.json",
        "manifest.json",
        "report.md",
        "junit.xml",
        "screenshots/001-sorted_catalog.png",
        "steps/001-after.png",
        "steps/001-observation.json",
    ):
        assert (artifact_dir / relative).is_file(), relative

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == result["run_id"]
    assert manifest["ok"] is True
    assert manifest["capture_errors"] == []

    junit = ET.parse(artifact_dir / "junit.xml").getroot()
    assert junit.attrib["tests"] == "4"
    assert junit.attrib["failures"] == "0"


def test_failed_assertion_keeps_diagnostic_and_failure_evidence(tmp_path: Path) -> None:
    engine, device = _engine(tmp_path)
    device.log_now(tag="Catalog", msg="refresh failed")

    result = engine.flow_run(
        yaml="steps:\n  - assert: {text: Missing, count: 1}\n",
        artifacts_dir=str(tmp_path / "failed"),
        evidence="failures",
        junit=True,
    )

    assert result["ok"] is False
    assert result["code"] == "assert_failed"
    assert "expected=1" in result["failure_detail"]
    assert result["resume_from_step"] == 0
    assert result["failure_evidence_id"].startswith(result["run_id"])
    artifact_dir = Path(result["artifacts"]["dir"])
    assert (artifact_dir / "failure.png").is_file()
    assert (artifact_dir / "failure-observation.json").is_file()
    assert "refresh failed" in (artifact_dir / "failure-diagnostics.txt").read_text(
        encoding="utf-8"
    )
    failure = ET.parse(artifact_dir / "junit.xml").find(".//failure")
    assert failure is not None
    assert failure.attrib["type"] == "assert_failed"


def test_assert_order_failure_reports_observed_centers(tmp_path: Path) -> None:
    engine, _device = _engine(tmp_path)
    result = engine.flow_run(
        yaml="""steps:
  - assert_order:
      axis: horizontal
      selectors:
        - {text: Professional}
        - {text: Starter}
"""
    )
    assert result["ok"] is False
    assert result["code"] == "assert_failed"
    assert "got [420, 120]" in result["failure_detail"]
