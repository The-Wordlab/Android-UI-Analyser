"""A missing handle retains explicitly requested full evidence at every boundary."""

from __future__ import annotations

import json

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ElementNotFoundError
from android_ui_analyser.mcp_server import build_server
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from conftest import FakeDevice, make_config

MISSING = "el:" + "f" * 32
PARENT = "el:" + "a" * 32
EDITOR = "el:" + "b" * 32
DISABLED = "el:" + "c" * 32
VISUAL = "el:" + "d" * 32


def observation():
    return AnalyzeResult(
        screen=Screen(width=400, height=800, package="com.example.fiction", source="mixed"),
        elements=[
            Element(
                id=0,
                handle=PARENT,
                type="Group",
                resource_id="fiction:group",
                bounds=(0, 0, 400, 800),
                center=(200, 400),
                window="app",
            ),
            Element(
                id=1,
                handle=EDITOR,
                parent=0,
                type="EditText",
                text="",
                resource_id="fiction:draft",
                bounds=(20, 100, 380, 200),
                center=(200, 150),
                clickable=True,
                enabled=True,
                focused=True,
                checkable=False,
                checked=False,
                scrollable=False,
                long_clickable=True,
                password=False,
                window="app",
            ),
            Element(
                id=2,
                handle=DISABLED,
                parent=0,
                type="Switch",
                text="Alerts",
                bounds=(20, 300, 380, 400),
                center=(200, 350),
                enabled=False,
                checkable=True,
                checked=True,
                selected=False,
                window="app",
            ),
            Element(
                id=3,
                handle=VISUAL,
                type="Text",
                text="Opaque caption",
                source=Source.ocr,
                bounds=(20, 500, 380, 600),
                center=(200, 550),
                checkable=None,
                checked=None,
                scrollable=None,
                long_clickable=None,
                password=None,
            ),
        ],
        meta=Meta(
            duration_ms=1,
            tier_used="hierarchy",
            path="hierarchy",
            fingerprint="existing-resolution-frame",
            device_serial="fixture-target",
        ),
    )


def config(tmp_path, fields="all", meta="all"):
    return make_config(
        cache={"dir": str(tmp_path / "cache")},
        lease={"enabled": False, "registry_dir": str(tmp_path / "leases")},
        memory={"enabled": False, "dir": str(tmp_path / "maps")},
        capture={"enabled": False},
        daemon={"enabled": False},
        output={"with_image": False, "observation_fields": fields, "observation_meta": meta},
    )


def resolution(monkeypatch, observed):
    calls = []

    def analyze(self, *args, **kwargs):
        calls.append((args, kwargs))
        return observed

    monkeypatch.setattr(Engine, "analyze", analyze)
    return calls


def assert_full(payload, observed):
    error = payload["error"]
    assert error["code"] == "element_not_found"
    assert error["hint"].startswith("No action was sent.")
    assert error["observation_present"] is True
    actual = error["observation"]
    assert actual == observed.as_dict("json")
    rows = {e["id"]: e for e in actual["elements"]}
    assert rows[EDITOR]["parent"] == PARENT
    assert rows[EDITOR]["enabled"] is True and rows[EDITOR]["source"] == "hierarchy"
    assert rows[EDITOR]["checkable"] is False and rows[EDITOR]["password"] is False
    assert rows[DISABLED]["enabled"] is False and rows[DISABLED]["checked"] is True
    assert rows[VISUAL]["source"] == "ocr" and rows[VISUAL]["checkable"] is None
    assert rows[VISUAL]["parent"] is None and rows[VISUAL]["password"] is None


def assert_one_resolution(calls, device):
    assert len(calls) == 1
    assert calls[0][1]["record"] is False and calls[0][1]["record_ids"] is False
    assert not [
        c
        for c in device.calls
        if c[0] in {"click", "long_click", "send_keys", "screenshot", "dump_hierarchy"}
    ]


def test_engine_full_missing_handle_observation_survives_error_serialization(tmp_path, monkeypatch):
    observed = observation()
    calls = resolution(monkeypatch, observed)
    device = FakeDevice(serial="fixture-target")
    engine = Engine(config(tmp_path), device=device)
    try:
        with pytest.raises(ElementNotFoundError) as caught:
            engine.tap(MISSING, observe=False)
        assert_full(caught.value.to_dict(), observed)
        assert_one_resolution(calls, device)
    finally:
        engine.close()


def test_cli_full_missing_handle_observation_uses_the_shared_resolution(tmp_path, monkeypatch):
    observed = observation()
    calls = resolution(monkeypatch, observed)
    device = FakeDevice(serial="fixture-target")
    monkeypatch.setattr(Engine, "_connect_target", lambda self, serial=None: device)
    path = tmp_path / "aua.json"
    path.write_text(config(tmp_path).model_dump_json())
    monkeypatch.setenv("AUA_CONFIG", str(path))
    result = CliRunner().invoke(app, ["--format", "json", "tap-and-analyze", MISSING])
    assert result.exit_code == 2, result.output
    assert_full(json.loads(result.stderr), observed)
    assert_one_resolution(calls, device)


def test_mcp_full_missing_handle_observation_uses_the_shared_resolution(tmp_path, monkeypatch):
    observed = observation()
    calls = resolution(monkeypatch, observed)
    device = FakeDevice(serial="fixture-target")
    engine = Engine(config(tmp_path), device=device)
    server = build_server(engine)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(
                "tap_and_analyze",
                {
                    "id": MISSING,
                    "observe_fields": "all",
                    "observe_meta": "all",
                },
            )
            assert len(result.content) == 1 and result.content[0].type == "text"
            return json.loads(result.content[0].text)

    try:
        assert_full(anyio.run(run), observed)
        assert_one_resolution(calls, device)
    finally:
        engine.close()


def test_default_missing_handle_projection_remains_small(tmp_path, monkeypatch):
    observed = observation()
    calls = resolution(monkeypatch, observed)
    device = FakeDevice(serial="fixture-target")
    defaults = make_config().output
    engine = Engine(
        config(tmp_path, defaults.observation_fields, defaults.observation_meta), device=device
    )
    try:
        with pytest.raises(ElementNotFoundError) as caught:
            engine.tap(MISSING, observe=False)
        rows = caught.value.to_dict()["error"]["observation"]["elements"]
        editor = next(e for e in rows if e["id"] == EDITOR)
        assert "source" not in editor and "parent" not in editor and "enabled" not in editor
        assert next(e for e in rows if e["id"] == DISABLED)["enabled"] is False
        assert_one_resolution(calls, device)
    finally:
        engine.close()


def test_explicit_subset_keeps_only_requested_fields_and_values(tmp_path, monkeypatch):
    observed = observation()
    calls = resolution(monkeypatch, observed)
    device = FakeDevice(serial="fixture-target")
    engine = Engine(config(tmp_path, "id,enabled,source,parent,checkable", "all"), device=device)
    try:
        with pytest.raises(ElementNotFoundError) as caught:
            engine.tap(MISSING, observe=False)
        rows = caught.value.to_dict()["error"]["observation"]["elements"]
        assert all(set(e) <= {"id", "enabled", "source", "parent", "checkable"} for e in rows)
        assert next(e for e in rows if e["id"] == EDITOR)["parent"] == PARENT
        assert next(e for e in rows if e["id"] == DISABLED)["enabled"] is False
        assert next(e for e in rows if e["id"] == VISUAL)["source"] == "ocr"
        assert "checkable" not in next(e for e in rows if e["id"] == VISUAL)
        assert_one_resolution(calls, device)
    finally:
        engine.close()
