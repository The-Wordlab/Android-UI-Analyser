from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.route_coordinator import MAX_ITEMS, discover

APP = "com.example.demo"
# Shapes emitted by Engine.orient and FlowStore.list, not imagined screen arrays.
ORIENT = {
    "package": APP, "known": True, "screens": 3, "routes": 2,
    "suggested_gotos": ["goto catalog"], "research_tasks": ["Verify details arrival"],
    "description": "A fictional catalogue", "recipes": {"entry": "Use the entry button"},
    "deeplinks": [{"uri": "demo://catalog?token=private", "note": "catalogue"}],
    "notes": ["Some results load asynchronously"],
    "recommended_call": {"tool": "flow_run", "arguments": {"allow_destructive": True}},
    "launch_uri": "demo://catalog?token=private", "device_serial": "private-device",
}
LISTED = {
    "app": APP, "active_package": APP, "active_context_id": "private-context",
    "flows": [{"name": "Visit catalogue", "storage_name": "catalogue", "ref": APP + ":catalogue",
               "app": APP, "context_id": "private-context", "steps": 3, "params": [],
               "description": "Open catalogue and verify arrival", "aliases": ["products"],
               "arrival": "rid:catalog", "arrival_screen": "catalog", "arrival_status": "verified",
               "context_compatible": True, "path": "/private/flows/catalogue.yaml"}],
}


class Native:
    def __init__(self, *, orient=None, listed=None, flow=None, preview=None):
        # The exported list is the same catalogue returned by MCP list_tools;
        # validation is offline and constructs neither Engine nor Device.
        from android_ui_analyser.mcp_server import _tool_definitions

        self.schemas = {tool.name: tool.inputSchema for tool in _tool_definitions()}
        self.calls = []
        self.replies = {
            "orient": ORIENT if orient is None else orient,
            "flow_list": LISTED if listed is None else listed,
            "flow_run": {"ok": True, "observation": {"screen": {}, "elements": []}} if flow is None else flow,
            "session_candidate_flow": {
                "ok": True, "name": "native-candidate", "yaml": "steps: []\n", "source_steps": [1],
                "checkpoint_ids": ["arrival"], "saved": False, "replayed": False,
                "artifact": "/private/session/candidate-flow.yaml",
            } if preview is None else preview,
        }

    async def __call__(self, name, arguments):
        jsonschema.validate(arguments, self.schemas[name])
        self.calls.append((name, copy.deepcopy(arguments)))
        return copy.deepcopy(self.replies[name])


def approved(tmp_path):
    path = tmp_path / "approved.yaml"
    path.write_text("name: catalogue\nsteps:\n  - assert: 'rid:catalog'\n")
    return path


def coordinator(native, flows=None):
    return asyncio.run(discover(native, app=APP, approved_flows=flows or {}))


def test_actual_read_only_shapes_do_not_authorize_memory_routes():
    native = Native()
    routes = coordinator(native)
    view = routes.model_view()
    assert native.calls == [("orient", {}), ("flow_list", {"app": APP})]
    assert view["candidates"] == []
    assert view["knowledge"]["orient"]["screens"] == 3
    assert view["knowledge"]["orient"]["suggested_gotos"] == ["goto catalog"]
    assert view["knowledge"]["known_flows"][0]["arrival_status"] == "verified"
    assert view["knowledge"]["known_flows"][0]["execution_approved"] is False
    rendered = json.dumps(view)
    for forbidden in ("recommended_call", "launch_uri", "demo://", "private-", "/private/", '"ref"'):
        assert forbidden not in rendered
    assert asyncio.run(routes.execute("catalogue", native))["executed"] is False
    assert len(native.calls) == 2


def test_only_opaque_candidate_executes_with_safe_native_knobs(tmp_path):
    native = Native()
    path = approved(tmp_path)
    routes = coordinator(native, {"Approved catalogue": path})
    candidate = routes.model_view()["candidates"][0]
    assert candidate["candidate_id"].startswith("route-")
    assert str(path) not in json.dumps(routes.model_view())
    assert asyncio.run(routes.execute(str(path), native))["executed"] is False
    result = asyncio.run(routes.execute(candidate["candidate_id"], native))
    assert native.calls[-1] == ("flow_run", {"file": str(path.resolve()),
                                            "allow_destructive": False, "assist": False})
    assert result["ok"] is True
    assert result["result"] == native.replies["flow_run"]
    assert len(result["source_sha256"]) == 64
    with pytest.raises(TypeError):
        asyncio.run(routes.execute(candidate["candidate_id"], native, allow_destructive=True))


def test_parameter_values_are_bound_by_host_and_cannot_change_after_discovery(tmp_path):
    native = Native()
    values = {"entry": {"QUERY": "synthetic example"}}
    path = approved(tmp_path)
    routes = asyncio.run(discover(native, app=APP, approved_flows={"entry": path}, flow_parameters=values))
    values["entry"]["QUERY"] = "mutated later"
    candidate_id = routes.model_view()["candidates"][0]["candidate_id"]
    assert "synthetic example" not in json.dumps(routes.model_view())
    asyncio.run(routes.execute(candidate_id, native))
    assert native.calls[-1][1]["params"] == {"QUERY": "synthetic example"}
    assert native.calls[-1][1]["allow_destructive"] is False


def test_parameters_cannot_create_an_unapproved_flow():
    native = Native()
    with pytest.raises(ValueError, match="approved flow"):
        asyncio.run(discover(native, app=APP, approved_flows={}, flow_parameters={"unknown": {"QUERY": "x"}}))
    assert not native.calls


@pytest.mark.parametrize("change", ["bytes", "missing", "symlink"])
def test_changed_approved_source_never_executes(tmp_path, change):
    native = Native()
    path = approved(tmp_path)
    offered = tmp_path / "link.yaml" if change == "symlink" else path
    if change == "symlink":
        offered.symlink_to(path)
    routes = coordinator(native, {"entry": offered})
    candidate_id = routes.model_view()["candidates"][0]["candidate_id"]
    if change == "bytes":
        path.write_text("name: changed\nsteps: []\n")
    elif change == "missing":
        path.unlink()
    else:
        other = tmp_path / "other.yaml"
        other.write_bytes(path.read_bytes())
        offered.unlink()
        offered.symlink_to(other)
    result = asyncio.run(routes.execute(candidate_id, native))
    assert result == {"ok": False, "executed": False, "code": "approved_flow_changed",
                      "candidate_id": candidate_id}
    assert [name for name, _ in native.calls] == ["orient", "flow_list"]


def test_native_divergence_is_preserved_without_retry_or_assistance(tmp_path):
    failure = {"ok": False, "code": "flow_diverged", "step": 2, "remaining": ["assert"]}
    native = Native(flow=failure)
    routes = coordinator(native, {"entry": approved(tmp_path)})
    candidate_id = routes.model_view()["candidates"][0]["candidate_id"]
    result = asyncio.run(routes.execute(candidate_id, native))
    assert result["executed"] is True and result["ok"] is False
    assert result["result"] == failure
    assert len(native.calls) == 3


def test_draft_is_explicit_preview_and_never_promoted(tmp_path):
    native = Native()
    routes = coordinator(native, {"entry": approved(tmp_path)})
    before = routes.model_view()
    result = asyncio.run(routes.draft(native, "host-owned-session"))
    name, arguments = native.calls[-1]
    assert name == "session_candidate_flow"
    assert arguments == {"name": arguments["name"], "session_id": "host-owned-session",
                         "save": False, "replay": False}
    assert arguments["name"].startswith("candidate-")
    assert result["ok"] is True and result["execution_approved"] is False
    assert result["verified"] is False
    assert result["native_preview"] == native.replies[name]
    assert routes.model_view() == before


@pytest.mark.parametrize("preview", [
    {"ok": False, "error": {"code": "usage_error", "message": "contract required"}},
    {"ok": True, "saved": True, "replayed": False},
    {"ok": True, "saved": False, "replayed": True},
    {"ok": True},
])
def test_unproven_or_unexpected_preview_never_succeeds(preview):
    native = Native(preview=preview)
    routes = coordinator(native)
    result = asyncio.run(routes.draft(native, "session"))
    assert result["ok"] is False
    assert result["execution_approved"] is False
    assert result["native_preview"] == preview
    assert routes.model_view()["candidates"] == []


def test_empty_session_rejects_before_preview():
    native = Native()
    routes = coordinator(native)
    with pytest.raises(ValueError, match="explicit session"):
        asyncio.run(routes.draft(native, ""))
    assert len(native.calls) == 2


def test_discovery_ignores_wrong_foreground_and_reports_unavailable_library():
    native = Native(orient={**ORIENT, "package": "com.example.other"},
                    listed={"ok": False, "error": {"code": "unsupported"}})
    view = coordinator(native).model_view()
    assert view["knowledge"]["orient"]["available"] is False
    assert view["knowledge"]["known_flows"] == []
    assert view["knowledge"]["flow_list_available"] is False


def test_knowledge_is_bounded_and_model_view_cannot_mutate_approvals(tmp_path):
    native = Native(orient={**ORIENT, "notes": ["x" * 10000] * 200},
                    listed={**LISTED, "flows": LISTED["flows"] * 200})
    routes = coordinator(native, {"entry": approved(tmp_path)})
    view = routes.model_view()
    assert len(view["knowledge"]["known_flows"]) <= MAX_ITEMS
    assert view["knowledge"]["known_flows_total"] == 200
    assert len(view["knowledge"]["orient"]["notes"]) <= MAX_ITEMS
    assert len(view["knowledge"]["orient"]["notes"][0]) == 320
    assert view["knowledge"]["truncated"] is True
    count = len(view["knowledge"]["known_flows"])
    view["candidates"].clear()
    view["knowledge"]["known_flows"].clear()
    assert len(routes.model_view()["candidates"]) == 1
    assert len(routes.model_view()["knowledge"]["known_flows"]) == count


def test_large_unicode_native_shapes_fit_session_knowledge_budget(tmp_path):
    heavy = "🟢" * 1000
    orient = {**ORIENT, "recipes": {heavy + str(i): heavy for i in range(100)},
              "notes": [heavy] * 100, "description": heavy,
              "suggested_gotos": [heavy] * 100, "research_tasks": [heavy] * 100}
    row = {**LISTED["flows"][0], "name": heavy, "description": heavy,
           "arrival_screen": heavy, "params": [heavy] * 100}
    native = Native(orient=orient, listed={**LISTED, "flows": [row] * 100})
    path = approved(tmp_path)
    routes = coordinator(native, {heavy + str(i): path for i in range(MAX_ITEMS)})
    view = routes.model_view()
    assert len(view["candidates"]) == MAX_ITEMS
    assert len(json.dumps(view, ensure_ascii=False, separators=(",", ":")).encode()) <= 7500
    assert len(json.dumps(view["knowledge"], ensure_ascii=False, separators=(",", ":")).encode()) <= 5000
    assert view["knowledge"]["truncated"] is True
    assert view["knowledge"]["truncated_fields"]
    assert view["knowledge"]["known_flows_total"] == 100
    assert view["knowledge"]["execution_approved"] is False


def test_bad_approval_fails_before_any_native_discovery(tmp_path):
    native = Native()
    with pytest.raises(FileNotFoundError):
        coordinator(native, {"missing": tmp_path / "missing.yaml"})
    assert native.calls == []


def test_flow_throw_preserves_failure_for_caller_lifecycle(tmp_path):
    native = Native()
    routes = coordinator(native, {"entry": approved(tmp_path)})
    candidate_id = routes.model_view()["candidates"][0]["candidate_id"]

    async def unavailable(name, arguments):
        raise TimeoutError("unknown native execution outcome")

    with pytest.raises(TimeoutError, match="unknown native execution"):
        asyncio.run(routes.execute(candidate_id, unavailable))
