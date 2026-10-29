"""Agent-facing failures, evidence delivery and journal privacy across the new surfaces."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jsonschema
import pytest
import typer
from typer.testing import CliRunner

from android_ui_analyser import journal
from android_ui_analyser.cli_development_tools import register
from android_ui_analyser.errors import UsageError
from android_ui_analyser.mcp_development_tools import tool_definitions
from android_ui_analyser.mcp_server import _dispatch_tool, _image_block


@pytest.mark.parametrize("args", [
    ["preview", "check", "tree.json", "--checks", '[{"property":"exists"}]'],
    ["journey", "check"],
    ["journey", "record", "--fingerprint", "fixture", "--evidence", "Absent", "--failed"],
    ["journey", "finish"],
])
def test_unsatisfied_evidence_is_printed_and_returns_assertion_exit(args):
    app = typer.Typer()
    register(app, lambda ctx, fn: fn(None, None),
             lambda *args, **kwargs: {"ok": False, "verified": False},
             lambda value, fmt: typer.echo(json.dumps(value)))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 8, result.output
    assert json.loads(result.stdout) == {"ok": False, "verified": False}


def test_mcp_journey_source_and_preview_checks_describe_valid_requests():
    schemas = {tool.name: tool.inputSchema for tool in tool_definitions()}
    for name in ("journey_start", "journey_validate"):
        jsonschema.validate({"file": "/tmp/journey.yaml"}, schemas[name])
        jsonschema.validate({"yaml": "name: Example"}, schemas[name])
        for invalid in ({}, {"file": "one.yaml", "yaml": "name: two"}):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(invalid, schemas[name])
    request = {"path": "tree.json", "checks": [
        {"selector": {"text": "Catalog"}, "property": "exists", "expected": True},
    ]}
    jsonschema.validate(request, schemas["preview_check"])
    request["checks"][0]["property"] = "invented"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(request, schemas["preview_check"])


def test_new_mcp_operations_cannot_bypass_an_active_job(monkeypatch):
    from android_ui_analyser import jobs

    monkeypatch.setattr(jobs, "manager_for", lambda engine: object())

    def blocked(engine, name):
        raise UsageError("another job owns this target")

    monkeypatch.setattr(jobs, "reject_if_active", blocked)
    engine = SimpleNamespace(development_tool=lambda **kwargs: pytest.fail("bypassed job"))
    with pytest.raises(UsageError, match="another job"):
        _dispatch_tool(engine, "development_tool", {"operation": "input.tap"})


def test_mcp_render_includes_first_png_and_never_a_failed_preview(tmp_path):
    import base64

    png = tmp_path / "preview.png"
    png.write_bytes(b"fixture PNG bytes")
    payload = {"ok": True, "previews": [{"imagePath": str(png)}]}
    image = _image_block("preview_render", payload)
    assert image is not None
    assert base64.b64decode(image.data) == png.read_bytes()
    assert image.mimeType == "image/png"
    assert _image_block("preview_render", {**payload, "ok": False}) is None


def test_native_parameters_and_echoed_values_are_not_retained_in_journals(tmp_path):
    secret = "fictional-private-input-7f983"
    journal.record(
        cache_dir=tmp_path, serial="fixture-target", source="cli", cmd="development_tool",
        args={"operation": "input.text", "arguments": [secret], "options": {"target": secret}},
        result={"ok": True, "stdout": f"Typed: {secret}", "data": {"text": secret}},
    )
    files = list((tmp_path / "journal").glob("*.jsonl"))
    assert len(files) == 2
    for path in files:
        assert secret not in path.read_text()
        assert "input.text" in path.read_text()


def test_android_cold_boot_maps_to_native_option_in_adapter(monkeypatch, tmp_path):
    from android_ui_analyser import emulator
    from android_ui_analyser.platforms.virtual_targets import VirtualTargetStartRequest

    calls = []

    def start(*args, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "serial": "emulator-5592", "avd": "fixture",
                "instance": "fixture.p5592", "pid": 5592, "owner": "fixture-worker"}

    monkeypatch.setattr(emulator, "start", start)
    emulator.start_virtual_target(VirtualTargetStartRequest(
        definition_id="fixture", cache_dir=tmp_path, options={"cold": True},
    ))
    assert calls[0]["extra_args"] == ["-no-snapshot-load"]
    with pytest.raises(UsageError, match="cold must be boolean"):
        emulator.start_virtual_target(VirtualTargetStartRequest(options={"cold": "yes"}))
    assert len(calls) == 1
