from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.run_live import (
    COMPACT_PROPERTIES,
    MODEL_TOOLS,
    SYSTEM,
    RunConfig,
    RunError,
    compact_refusal,
    completion,
    offered_schema,
    run_live,
    scripted_sender,
)


class FakeMCP:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.starts = 0
        self.finish_passes = True
        self.reset_passes = True
        self.same_target = True
        self.missing_tool = False
        self.with_image: bool | str = False

    async def list_tools(self) -> dict[str, Any]:
        names = [*MODEL_TOOLS, "session_start", "flow_run", "configure"]
        if self.missing_tool:
            names.remove("tap_and_analyze")
        return {"tools": [{"name": name, "description": "Public fake tool", "inputSchema": {
            "type": "object", "properties": {
                "id": {"type": "string"}, "session_id": {"type": "string"},
                "allow_incomplete": {"type": "boolean"}, "phase_done": {"type": "object"},
                **({"with_image": {"type": ["boolean", "string"]}}
                   if "analyze" in name else {}),
            }, "additionalProperties": False,
        }} for name in names]}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, copy.deepcopy(arguments)))
        if name == "configure":
            self.with_image = arguments.get("with_image", self.with_image)
            result = {"ok": True, "with_image": self.with_image}
        elif name == "session_start":
            self.starts += 1
            result = {
                "ok": True, "session_id": f"session-{self.starts}",
                "serial": "fictional-target" if self.same_target or self.starts == 1 else "other-target",
                "artifacts_dir": str(self.output / "aua" / "actual-bundle"),
                "observation": {
                    "screen": {"package": "dev.aua.fixture"},
                    "elements": [{"id": "el:fresh-start", "text": "AUA Agent Loop Fixture"}],
                    "meta": {"observation_contract": {"evidence_id": "frame-start", "analyze_needed": False}},
                },
                "goal_progress": {"completed": 0, "current": {
                    "id": "default_order", "assertions": [{"text": "HIDDEN_EXPECTED_ORDER"}],
                }},
                "next_call": {"tool": "tap_and_analyze", "arguments": {"id": "HIDDEN_HINT"}},
                "candidates": [{"call": {"id": "HIDDEN_CANDIDATE"}}],
            }
        elif name == "flow_run":
            result = {"ok": self.reset_passes}
        elif name == "session_finish":
            aborted = arguments.get("allow_incomplete") is True
            result = {"ok": True, "finished": self.finish_passes and not aborted,
                      "terminated": self.finish_passes or aborted}
        else:
            result = {"ok": True, "observation": {
                "elements": [{"id": "el:only-after-tap", "text": "Fictional response"}],
                "meta": {"observation_contract": {"evidence_id": "frame-after", "analyze_needed": False}},
            }}
        if self.with_image and isinstance(result.get("observation"), dict):
            result["observation"].setdefault("meta", {})["raw_image"] = "/private/evidence-only.png"
        blocks = [{"type": "text", "text": json.dumps(result)}]
        if self.with_image and name != "configure":
            blocks.append({"type": "image", "data": "FICTIONAL_IMAGE_BYTES", "mimeType": "image/png"})
        return {"content": blocks, "isError": False}


def config(tmp_path: Path, **kwargs: Any) -> RunConfig:
    return RunConfig(
        model="fictional/controller", scenario_id="classic-sort",
        goal="Verify fictional products and restore the fixture.",
        contract_yaml="hidden assertion contract", reset_yaml="fixture reset flow",
        output=tmp_path / "run", **kwargs,
    )


def native(name: str, arguments: Any = None) -> dict[str, Any]:
    return {"model": "fictional/controller", "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "reasoning_content": "Remember the observed state.",
                "tool_calls": [{"id": "native-call", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments or {}),
                }}],
            }}]}


def verifier(bundle: Path, scenario: str) -> dict[str, Any]:
    return {"passed": True, "verified": True, "cleanup_verified": True}


def test_public_mcp_controller_preserves_history_and_uses_independent_verifier(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)
    requests = []
    verified = []

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(payload))
        encoded = json.dumps(payload)
        assert "HIDDEN_EXPECTED_ORDER" not in encoded
        assert "HIDDEN_HINT" not in encoded
        assert "HIDDEN_CANDIDATE" not in encoded
        assert "hidden assertion contract" not in encoded
        assert cfg.goal in encoded
        if len(requests) == 1:
            assert "el:fresh-start" in encoded
            return native("tap_and_analyze", {"id": "el:fresh-start"})
        messages = payload["messages"]
        assert messages[-2]["reasoning_content"] == "Remember the observed state."
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "native-call"
        assert "el:only-after-tap" in messages[-1]["content"]
        return native("session_finish")

    def verify(bundle: Path, scenario: str) -> dict[str, Any]:
        verified.append((bundle, scenario))
        assert mcp.calls[-1] == ("session_finish", {
            "session_id": "session-2", "allow_incomplete": False, "summary": False,
        })
        return verifier(bundle, scenario)

    report = asyncio.run(run_live(mcp, send, cfg, verify=verify))

    assert report["passed"] is True
    assert report["valid_start"] is True and report["aua_finished"] is True
    assert report["cleanup_attempted"] is False
    assert report["model_requests"] == 2 and report["model_tool_calls"] == 2
    assert verified == [(cfg.output / "aua" / "actual-bundle", "classic-sort")]
    lifecycle = [(name, args) for name, args in mcp.calls if name != "configure"]
    assert [name for name, _ in lifecycle[:4]] == ["session_start", "flow_run", "session_finish", "session_start"]
    assert "contract_yaml" not in lifecycle[0][1]
    assert lifecycle[3][1]["contract_yaml"] == cfg.contract_yaml
    assert all("serial" not in args and "device" not in args for _, args in mcp.calls)
    assert (cfg.output / "model-turns.jsonl").exists()
    assert "HIDDEN_EXPECTED_ORDER" in (cfg.output / "mcp-calls.jsonl").read_text()


@pytest.mark.parametrize("verified", [False, True])
def test_ok_without_finished_never_passes_and_cleanup_cannot_reclassify_failure(
    tmp_path: Path, verified: bool,
) -> None:
    cfg = config(tmp_path, max_steps=1)
    mcp = FakeMCP(cfg.output)
    mcp.finish_passes = False

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=lambda *_: {"passed": verified}))

    assert report["passed"] is False and report["aua_finished"] is False
    assert report["verifier"] is None
    assert report["rejected_finish_attempts"] == 1
    assert report["cleanup_attempted"] is True
    assert mcp.calls[-2][1]["allow_incomplete"] is True


def test_independent_verifier_failure_overrules_finished(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=lambda *_: {"passed": False}))

    assert report["aua_finished"] is True
    assert report["passed"] is False
    assert "independent evidence" in report["error"]


@pytest.mark.parametrize("evidence", [
    {"passed": True, "verified": False, "cleanup_verified": True},
    {"passed": True, "verified": True, "cleanup_verified": False},
    {"passed": True},
])
def test_success_requires_usable_independent_evidence_and_cleanup(
    tmp_path: Path, evidence: dict[str, Any],
) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=lambda *_: evidence))

    assert report["passed"] is False
    if evidence.get("verified") is not True:
        assert report["false_pass"] is None


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (native("shell", {"command": "anything"}), "unavailable tool"),
        (native("tap_and_analyze", {"id": 7}), "public MCP schema"),
        (native("tap_and_analyze", {"phase_done": {"id": "pretend"}}), "public MCP schema"),
        (native("session_finish", {"allow_incomplete": True}), "public MCP schema"),
        (native("session_progress", {"session_id": "foreign"}), "public MCP schema"),
    ],
)
def test_disallowed_or_malformed_calls_never_reach_mcp(
    tmp_path: Path, response: dict[str, Any], error: str,
) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return response

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is False and error in report["error"]
    lifecycle = [name for name, _ in mcp.calls if name != "configure"]
    assert lifecycle[4:] == ["flow_run", "session_finish"]


@pytest.mark.parametrize("failure", ["truncated", "multi-call", "bad-json", "fake-text-call"])
def test_native_protocol_failures_do_not_execute_guessed_actions(tmp_path: Path, failure: str) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)
    response = native("tap_and_analyze", {"id": "el:fresh-start"})
    choice = response["choices"][0]
    if failure == "truncated":
        choice["finish_reason"] = "length"
    elif failure == "multi-call":
        choice["message"]["tool_calls"] *= 2
    elif failure == "bad-json":
        choice["message"]["tool_calls"][0]["function"]["arguments"] = "{broken"
    else:
        choice["finish_reason"] = "stop"
        choice["message"] = {"role": "assistant", "content": "tap_and_analyze(id='el:fresh-start')"}
        mcp.finish_passes = False

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return response

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is False
    assert not any(name == "tap_and_analyze" for name, _ in mcp.calls)


@pytest.mark.parametrize("failure", ["missing-tool", "reset", "different-target", "request-budget"])
def test_invalid_setup_and_context_budget_prevent_model_requests(tmp_path: Path, failure: str) -> None:
    cfg = config(tmp_path, max_request_bytes=1 if failure == "request-budget" else 100_000)
    mcp = FakeMCP(cfg.output)
    mcp.missing_tool = failure == "missing-tool"
    mcp.reset_passes = failure != "reset"
    mcp.same_target = failure != "different-target"

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("model must not be called")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is False and report["model_requests"] == 0
    if failure == "missing-tool":
        assert not mcp.calls
    else:
        assert report["cleanup_attempted"] is True


def test_endpoint_errors_are_sanitized_and_trigger_cleanup(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)
    secret = "fictional-private-api-token"

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        raise httpx.ConnectError(f"Cannot connect to https://user:{secret}@endpoint.invalid")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["error"] == "ConnectError"
    assert report["cleanup_attempted"] is True
    assert secret not in json.dumps(report)
    assert secret not in (cfg.output / "model-turns.jsonl").read_text()


@pytest.mark.parametrize("body_kind", ["nested", "direct", "long", "html"])
def test_http_error_records_status_and_bounded_redacted_message(
    tmp_path: Path, body_kind: str,
) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)
    secret = "fictional-private-api-token"
    header_only = "fictional-header-only-secret"
    request = httpx.Request(
        "POST", "https://endpoint.invalid/chat/completions",
        headers={"Authorization": f"Bearer {secret}", "X-Private": header_only},
    )
    message = f"Maximum context length exceeded; echoed key {secret}"
    if body_kind == "long":
        message += "x" * 4_096
    if body_kind == "html":
        response = httpx.Response(502, request=request, text=f"<html>{secret}</html>")
    else:
        error = {"message": message, "private_unused_field": header_only}
        response = httpx.Response(
            400, request=request, json={"error": error} if body_kind == "nested" else error,
        )

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        response.raise_for_status()
        pytest.fail("HTTP failure must raise")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is False and report["cleanup_attempted"] is True
    assert report["model_requests"] == 1 and report["model_tool_calls"] == 0
    prefix = f"HTTPStatusError (HTTP {response.status_code})"
    assert report["error"].startswith(prefix)
    assert len(report["error"]) <= len(prefix) + 2 + 2_048
    if body_kind == "html":
        assert report["error"] == prefix
    else:
        assert "Maximum context length exceeded" in report["error"]
        assert "[REDACTED]" in report["error"]
    if body_kind == "long":
        assert report["error"].endswith("...")
    turn = json.loads((cfg.output / "model-turns.jsonl").read_text())
    assert turn["error"] == report["error"]
    artifacts = json.dumps(report) + (cfg.output / "model-turns.jsonl").read_text()
    assert secret not in artifacts and header_only not in artifacts


class CompactMCP(FakeMCP):
    fresh_fingerprint = "fresh-fixture-screen"

    async def list_tools(self) -> dict[str, Any]:
        listing = await super().list_tools()
        for tool in listing["tools"]:
            schema = tool["inputSchema"]
            for field in COMPACT_PROPERTIES.get(tool["name"], set()):
                schema["properties"].setdefault(field, {})
            if tool["name"] == "tap_and_analyze":
                schema["properties"].update(text={"type": "string"}, desc={"type": "string"})
                schema["oneOf"] = [{"required": [key]} for key in ("id", "text", "desc")]
        return listing

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "key_and_analyze" and arguments.get("name") == "back":
            self.finish_passes = True
        result = await super().call_tool(name, arguments)
        if name == "analyze_screen":
            body = json.loads(result["content"][0]["text"])
            body["observation"]["meta"]["fingerprint"] = self.fresh_fingerprint
            result["content"][0]["text"] = json.dumps(body)
        return result


@pytest.mark.parametrize("explicit", [False, True])
def test_baseline_profile_keeps_original_system_tools_and_finish_path(tmp_path: Path, explicit: bool) -> None:
    cfg = config(tmp_path, **({"profile": "baseline"} if explicit else {}))
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["messages"][0] == {"role": "system", "content": SYSTEM}
        assert [tool["function"]["name"] for tool in payload["tools"]] == list(MODEL_TOOLS)
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is True and report["profile"] == "baseline"
    assert report["schema_repair_budget"] == report["schema_repairs"] == 0
    assert not any(name == "analyze_screen" for name, _ in mcp.calls)


@pytest.mark.parametrize("bad_call", [
    native("tap_and_analyze", {"id": "current", "text": "alias"}),
    native("analyze_screen", {"query": "partial"}),
    native("tap_and_analyze", {"id": "current", "observe_fields": "all"}),
    native("tap_and_analyze", {"id": "current", "phase_done": {"id": "pretend"}}),
    native("session_finish", {"allow_incomplete": True}),
    native("expect_and_analyze", {"text": "unavailable"}),
])
def test_compact_repairs_invalid_calls_without_executing_them(tmp_path: Path, bad_call: dict) -> None:
    cfg = config(tmp_path, profile="compact-v1")
    mcp = CompactMCP(cfg.output)
    sent = 0

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal sent
        sent += 1
        assert [tool["function"]["name"] for tool in payload["tools"]] == list(COMPACT_PROPERTIES)
        if sent == 1:
            return bad_call
        feedback = json.loads(payload["messages"][-1]["content"])
        assert feedback["error"]["executed"] is False
        assert len(feedback["error"]["message"]) <= 768
        assert payload["messages"][-1]["tool_call_id"] == "native-call"
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is True
    assert report["schema_repairs"] == 1 and report["model_tool_calls"] == 1
    records = list(map(json.loads, (cfg.output / "mcp-calls.jsonl").read_text().splitlines()))
    assert [row["tool"] for row in records if row["actor"] == "model"] == ["session_finish"]
    finish_read = next(row for row in records if row["actor"] == "finish_observation")
    assert finish_read["arguments"] == {"source": "hierarchy", "no_cache": True, "with_image": True}
    assert records.index(finish_read) < next(i for i, row in enumerate(records) if row["actor"] == "model")


def test_compact_schema_repair_budget_is_three(tmp_path: Path) -> None:
    cfg = config(tmp_path, profile="compact-v1")
    mcp = CompactMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return native("analyze_screen", {"query": "still unavailable"})

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is False and report["error"] == "schema repair budget exhausted"
    assert report["schema_repairs"] == report["schema_repair_budget"] == 3
    assert report["model_requests"] == 4 and report["model_tool_calls"] == 0
    assert len((cfg.output / "controller-feedback.jsonl").read_text().splitlines()) == 3
    assert not any(name == "analyze_screen" for name, _ in mcp.calls)


def test_compact_refusal_preserves_errors_current_ui_and_freshness_without_phase_history() -> None:
    result = {
        "ok": False, "finished": False, "terminated": False,
        "code": "incomplete", "errors": ["cleanup pending"], "warnings": ["screen changed"],
        "hint": "use forbidden allow_incomplete", "goal_progress": {
            "completed": 3, "total": 4, "done": False,
            "current": {"id": "cleanup", "objective": "Return home", "assertions": ["hidden"]},
            "phases": ["large prior state" * 2000],
        },
        "observation": {"screen": {"package": "dev.aua.fixture"}, "elements": [
            {"id": "current", "text": "Back", "enabled": False, "bounds": [0, 0, 10, 10],
             "checkable": True, "window": "system-overlay", "source": "hierarchy", "confidence": 0.7,
             "redundant_accounting": "x" * 20000},
        ], "meta": {"fingerprint": "fresh", "stale_risk": False, "observation_contract": {
            "evidence_fresh": True, "evidence_id": "current-frame"}, "caller": {"samples": 200},
            "warnings": [{"readiness": "unconfirmed", "unknown_outcome": True}],
            "readiness": {"unknown": True}}},
    }
    visible = compact_refusal(result)
    assert len(json.dumps(visible)) < 1200
    assert visible["errors"] == result["errors"] and visible["warnings"] == result["warnings"]
    assert visible["goal_progress"]["current"] == {"id": "cleanup", "objective": "Return home"}
    assert visible["observation"]["elements"][0]["enabled"] is False
    for key in ("checkable", "window", "source", "confidence"):
        assert visible["observation"]["elements"][0][key] == result["observation"]["elements"][0][key]
    assert visible["observation"]["meta"]["fingerprint"] == "fresh"
    assert visible["observation"]["meta"]["warnings"] == result["observation"]["meta"]["warnings"]
    assert visible["observation"]["meta"]["readiness"] == {"unknown": True}
    assert "hint" not in visible and "phases" not in visible["goal_progress"]
    assert result["goal_progress"]["phases"]  # Raw MCP evidence was not mutated.


def test_compact_rejected_finish_can_navigate_then_finish_with_fresh_evidence(tmp_path: Path) -> None:
    cfg = config(tmp_path, profile="compact-v1")
    mcp = CompactMCP(cfg.output)
    mcp.finish_passes = False
    planned = iter([native("session_finish"), native("key_and_analyze", {"name": "back"}), native("session_finish")])

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return next(planned)

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is True
    assert report["finish_attempts"] == 2 and report["rejected_finish_attempts"] == 1
    assert sum(name == "analyze_screen" for name, _ in mcp.calls) == 2


def test_compact_finish_does_not_accept_missing_fresh_fingerprint(tmp_path: Path) -> None:
    cfg = config(tmp_path, profile="compact-v1")
    mcp = CompactMCP(cfg.output)
    mcp.fresh_fingerprint = None

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is False and "trustworthy" in report["error"]
    assert report["cleanup_attempted"] is True
    assert not any(name == "session_finish" and args.get("allow_incomplete") is False for name, args in mcp.calls)


def test_model_timeout_is_bounded_and_cleanup_still_runs(tmp_path: Path) -> None:
    cfg = config(tmp_path, request_timeout_s=0.001)
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(5)
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["error"] == "TimeoutError"
    assert report["cleanup_attempted"] is True


def test_existing_artifact_directory_is_not_reused(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cfg.output.mkdir()
    (cfg.output / "prior-result.json").write_text("{}")
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("model must not be called")

    with pytest.raises(RunError, match="empty"):
        asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert not mcp.calls


@pytest.mark.parametrize("finish_reason", [None, "content_filter", "error", "unknown"])
def test_native_calls_need_a_normal_finish_reason(finish_reason: Any) -> None:
    response = native("session_finish")
    response["choices"][0]["finish_reason"] = finish_reason
    with pytest.raises(RunError, match="stop normally"):
        completion(response)


@pytest.mark.parametrize("call_id", [None, 1, [], {}, "", " "])
def test_native_call_id_must_be_a_nonempty_string(call_id: Any) -> None:
    response = native("session_finish")
    response["choices"][0]["message"]["tool_calls"][0]["id"] = call_id
    with pytest.raises(RunError, match="invalid native"):
        completion(response)


def test_schema_subset_retains_selector_and_wait_contracts_without_hidden_overrides() -> None:
    from android_ui_analyser.mcp_server import _tool_definitions

    original = {tool.name: tool.inputSchema for tool in _tool_definitions()}
    schema = offered_schema("tap_and_analyze", original["tap_and_analyze"])
    assert schema["oneOf"] == original["tap_and_analyze"]["oneOf"]
    for key in ("id", "rid", "text", "desc", "stable_key", "index", "observe_fields", "until"):
        assert schema["properties"][key] == original["tap_and_analyze"]["properties"][key]
    assert "phase_done" not in schema["properties"]
    assert "with_image" not in schema["properties"]
    assert offered_schema("session_finish", original["session_finish"])["properties"] == {}
    assert "with_image" in original["tap_and_analyze"]["properties"]


def test_scripted_smoke_uses_same_controller_but_is_not_a_model_measurement(tmp_path: Path) -> None:
    cfg = config(tmp_path, mode="scripted_harness_smoke")
    mcp = FakeMCP(cfg.output)
    send = scripted_sender([
        {"tool": "tap_and_analyze", "arguments": {"id": "el:fresh-start"}},
        {"tool": "session_finish", "arguments": {}},
    ])

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is True
    assert report["model_measured"] is False
    assert report["mode"] == "scripted_harness_smoke"


@pytest.mark.parametrize("prior_setting", [False, True, "/private/prior-image-destination.png"])
def test_screenshot_capture_is_harness_owned_and_never_exposed_to_text_model(
    tmp_path: Path, prior_setting: bool | str,
) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)
    mcp.with_image = prior_setting
    requests = []

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(payload))
        assert "FICTIONAL_IMAGE_BYTES" not in json.dumps(payload)
        assert "/private/evidence-only.png" not in json.dumps(payload)
        assert mcp.with_image is True
        if len(requests) == 1:
            return native("analyze_screen")
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is True
    assert report["capture_configuration_restored"] is True
    assert mcp.calls[:2] == [("configure", {}), ("configure", {"with_image": True})]
    assert ("analyze_screen", {"with_image": True}) in mcp.calls
    assert mcp.calls[-1] == ("configure", {"with_image": prior_setting})
    assert mcp.with_image == prior_setting
    assert "/private/evidence-only.png" in (cfg.output / "mcp-calls.jsonl").read_text()


def test_screenshot_configuration_failure_prevents_device_session(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    mcp = FakeMCP(cfg.output)
    original = mcp.call_tool

    async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "configure" and arguments.get("with_image") is True:
            return {"ok": False, "with_image": False}
        return await original(name, arguments)

    mcp.call_tool = call

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("model should not run without evidence capture")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))

    assert report["passed"] is False
    assert "screenshot evidence" in report["error"]
    assert report["capture_configuration_restored"] is True
    assert mcp.starts == 0


def hosted_config() -> dict[str, Any]:
    return {"reasoning": {"enabled": True, "exclude": False}, "provider": {
        "only": ["fictional"], "allow_fallbacks": False, "require_parameters": True,
        "max_price": {"prompt": 0.3, "completion": 1.2}}, "temperature": 0.5}


def test_hosted_request_is_archived_exactly_with_native_reasoning_and_private_projection(tmp_path: Path) -> None:
    cfg = config(tmp_path, backend="openrouter", observation_profile="hosted-v1",
                 request_config=hosted_config(), chat_template_kwargs={"enable_thinking": True})
    mcp = FakeMCP(cfg.output)
    requests = []
    details = [{"type": "reasoning.encrypted", "data": "opaque-signature==", "index": 0}]

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(payload))
        assert "chat_template_kwargs" not in payload and "parallel_tool_calls" not in payload
        assert payload["temperature"] == 0.5 and payload["provider"] == hosted_config()["provider"]
        encoded = json.dumps(payload)
        assert "fictional-target" not in encoded and "session-2" not in encoded
        assert str(cfg.output) not in encoded and "el:fresh-start" in encoded
        response = native("tap_and_analyze", {"id": "el:fresh-start"}) if len(requests) == 1 else native("session_finish")
        response["usage"]["cost"] = 0.01
        response["choices"][0]["message"]["reasoning_details"] = copy.deepcopy(details)
        if len(requests) == 2:
            assert payload["messages"][-2]["reasoning_details"] == details
        return response

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is True
    assert report["cost_accounting"]["reported_usd"] == 0.02
    assert report["temperature"] == 0.5 and report["chat_template_kwargs"] == {}
    rows = [json.loads(line) for line in (cfg.output / "model-turns.jsonl").read_text().splitlines()]
    assert [row["request"] for row in rows] == requests
    assert "fictional-target" in (cfg.output / "mcp-calls.jsonl").read_text()


@pytest.mark.parametrize("cost", [None, 0.1])
def test_hosted_missing_cost_or_limit_stops_next_request_and_keeps_cleanup(tmp_path: Path, cost: Any) -> None:
    cfg = config(tmp_path, backend="openrouter", observation_profile="hosted-v1", request_config=hosted_config())
    mcp = FakeMCP(cfg.output)
    requests = []

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(payload)
        response = native("tap_and_analyze", {"id": "el:fresh-start"})
        if cost is not None:
            response["usage"]["cost"] = cost
        return response

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is False and len(requests) == 1
    assert report["cleanup_attempted"] is True
    assert "usage.cost" in report["error"] if cost is None else "cost limit reached" in report["error"]
    taps = [call for call in mcp.calls if call[0] == "tap_and_analyze"]
    assert len(taps) == (0 if cost is None else 1)


def test_hosted_still_rejects_multiple_native_calls_without_executing(tmp_path: Path) -> None:
    cfg = config(tmp_path, backend="openrouter", observation_profile="hosted-v1", request_config=hosted_config())
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        response = native("tap_and_analyze", {"id": "el:fresh-start"})
        response["usage"]["cost"] = 0.01
        response["choices"][0]["message"]["tool_calls"] *= 2
        return response

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is False and "one native tool call" in report["error"]
    assert not any(name == "tap_and_analyze" for name, _ in mcp.calls)


def test_local_backend_can_use_hosted_projection_without_cost_metadata(tmp_path: Path) -> None:
    cfg = config(tmp_path, observation_profile="hosted-v1", chat_template_kwargs={"enable_thinking": True})
    mcp = FakeMCP(cfg.output)

    async def send(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["parallel_tool_calls"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert "fictional-target" not in json.dumps(payload)
        return native("session_finish")

    report = asyncio.run(run_live(mcp, send, cfg, verify=verifier))
    assert report["passed"] is True and "cost_accounting" not in report


def test_hosted_cli_rejects_missing_key_before_opening_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.aua_controller import run_live as module

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"id": "fake", "repository": "fictional/model", "backend": "openrouter",
                                               "request_config": hosted_config()}]}))
    monkeypatch.delenv("AUA_HOSTED_TEST_MISSING", raising=False)
    monkeypatch.setattr(sys, "argv", ["runner", "--base-url", "https://openrouter.ai/api/v1", "--model", "fake",
                                    "--scenario", "classic-sort", "--manifest", str(manifest), "--output", str(tmp_path / "out"),
                                    "--api-key-env", "AUA_HOSTED_TEST_MISSING", "--observation-profile", "hosted-v1"])
    monkeypatch.setattr(module, "stdio_client", lambda *_: pytest.fail("MCP must not open without hosted credentials"))
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2
