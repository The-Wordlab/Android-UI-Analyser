"""Session evidence prerequisites are discoverable and enforced before MCP dispatch."""

import json

import anyio
import jsonschema
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from android_ui_analyser.engine import Engine
from android_ui_analyser.mcp_server import _tool_definitions, build_server
from conftest import FakeDevice, make_config

VALID_OPTIONS = [
    {},
    {"evidence": "failures"},
    {"evidence": "none"},
    {"junit": False},
    {"evidence": "none", "junit": False},
    {"evidence": "failures", "artifacts_dir": ""},
    {"evidence": "all", "artifacts_dir": "/tmp/fictional-evidence"},
    {"junit": True, "artifacts_dir": "/tmp/fictional-evidence"},
    {"evidence": "none", "junit": True, "artifacts_dir": "/tmp/fictional-evidence"},
    {"evidence": "all", "junit": True, "artifacts_dir": "/tmp/fictional-evidence"},
]
INVALID_OPTIONS = [
    {"evidence": "all"},
    {"evidence": "all", "junit": False},
    {"evidence": "all", "artifacts_dir": ""},
    {"junit": True},
    {"evidence": "none", "junit": True},
    {"evidence": "failures", "junit": True},
    {"junit": True, "artifacts_dir": ""},
]


def _schema():
    return next(tool.inputSchema for tool in _tool_definitions() if tool.name == "session_start")


@pytest.mark.parametrize("name", ["tap_and_analyze", "long_press_and_analyze", "copy_text"])
def test_action_schemas_distinguish_observation_id_from_app_resource_id(name):
    props = next(
        tool.inputSchema["properties"] for tool in _tool_definitions() if tool.name == name
    )
    assert props["id"]["type"] == ["integer", "string"]
    assert "latest observation" in props["id"]["description"]
    assert "el:..." in props["id"]["description"]
    assert "Do not put this id in rid" in props["id"]["description"]
    assert "resource/test id" in props["rid"]["description"]
    assert "not the observation's element id" in props["rid"]["description"]


@pytest.mark.parametrize("name", ["input_and_analyze", "clear_and_analyze", "inspect"])
def test_id_only_tools_describe_returned_stable_element_ids(name):
    prop = next(
        tool.inputSchema["properties"]["id"] for tool in _tool_definitions() if tool.name == name
    )
    assert prop["type"] == ["integer", "string"]
    assert "latest observation" in prop["description"]
    assert "el:..." in prop["description"]


def test_schema_documents_defaults_and_artifact_requirements():
    schema = _schema()
    jsonschema.Draft7Validator.check_schema(schema)
    props = schema["properties"]
    assert props["evidence"]["default"] == "failures"
    assert props["junit"]["default"] is False
    assert "Neither needs artifacts_dir" in props["evidence"]["description"]
    assert "'all' requires artifacts_dir" in props["evidence"]["description"]
    assert "true requires artifacts_dir" in props["junit"]["description"]
    assert "non-empty" in props["artifacts_dir"]["description"]


@pytest.mark.parametrize("options", VALID_OPTIONS)
def test_schema_accepts_existing_engine_contract(options):
    jsonschema.validate({"goal": "Inspect a fictional page", **options}, _schema())


@pytest.mark.parametrize("options", INVALID_OPTIONS)
def test_schema_rejects_artifact_outputs_without_a_directory(options):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"goal": "Inspect a fictional page", **options}, _schema())


@pytest.mark.parametrize("profile", ["full", "web"])
def test_mcp_validates_artifact_prerequisites_before_dispatch(monkeypatch, profile):
    engine = Engine(make_config(), device=FakeDevice())
    calls = []

    def start(goal, **kwargs):
        calls.append((goal, kwargs))
        return {"ok": True, "goal": goal}

    monkeypatch.setattr(engine, "session_start", start)

    async def run():
        async with create_connected_server_and_client_session(
            build_server(engine, tool_profile=profile)
        ) as client:
            listed = await client.list_tools()
            assert (
                next(t.inputSchema for t in listed.tools if t.name == "session_start") == _schema()
            )
            for options in INVALID_OPTIONS:
                result = await client.call_tool(
                    "session_start", {"goal": "Inspect page", **options}
                )
                assert result.isError
                assert "Input validation error" in result.content[0].text
                assert (
                    "artifacts_dir" in result.content[0].text
                    or "non-empty" in result.content[0].text
                )
            assert calls == []
            for options in VALID_OPTIONS:
                result = await client.call_tool(
                    "session_start", {"goal": "Inspect page", **options}
                )
                assert not result.isError
                assert json.loads(result.content[0].text)["ok"] is True
                assert calls[-1][0] == "Inspect page"
                assert all(calls[-1][1][key] == value for key, value in options.items())
            assert len(calls) == len(VALID_OPTIONS)

    try:
        anyio.run(run)
    finally:
        engine.close()
