from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.run_live import COMPACT_SYSTEM, RunError
from experiments.aua_controller.run_realapp import (
    ASYNC_UI_WAIT_MAX_SECONDS,
    ASYNC_UI_WAIT_TOOL,
    CONTROLLER_TOOLS,
    WALL_CLOCK_WAIT_SECONDS,
    WALL_CLOCK_WAIT_TOOL,
    async_ui_wait_spec,
    controller_observation_arguments,
    controller_tool_timeouts,
    normalize_element_id_argument,
    realapp_tools,
    run_async_ui_wait,
    run_realapp,
)

SETTINGS = {"provider": {"only": ["fictional"], "allow_fallbacks": False,
                         "max_price": {"prompt": 0.3, "completion": 1.2}},
            "reasoning": {"effort": "low"}}
TOOLS = [
    {"type": "function", "function": {"name": "tap", "parameters": {
        "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish", "parameters": {
        "type": "object", "properties": {}, "additionalProperties": False}}},
]


def frame(fingerprint, texts=("Settings",), **extra):
    return {"ok": True, "observation": {
        "screen": {"package": "com.example.fictional", "width": 720, "height": 1280},
        "elements": [{"id": f"el:{fingerprint}-{i}", "text": text, "clickable": True, "bounds": [0, i, 9, i + 1],
                      "long_clickable": False, "center": [4, i]} for i, text in enumerate(texts)],
        "meta": {"fingerprint": fingerprint, "elapsed_ms": 12},
    }, **extra}


def model_call(name, arguments, *, cost=0.001, call_id="native-1"):
    return {"model": "fictional/model", "provider": "fictional",
            "usage": {"cost": cost, "prompt_tokens": 100, "completion_tokens": 8},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"type": "function", "id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def run_loop(tmp_path, responses, results, **kwargs):
    replies, outcomes, calls = iter(responses), iter(results), []

    async def send(payload):
        return copy.deepcopy(next(replies))

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        return copy.deepcopy(next(outcomes))

    options = {"send": send, "call_tool": call_tool, "tools": TOOLS, "system_prompt": "s", "user_prompt": "u",
               "initial_observation": frame("fp-0"), "model": "fictional/model", "output": tmp_path / "controller",
               "request_config": SETTINGS, "terminal_tools": frozenset({"finish"})}
    options.update(kwargs)
    return asyncio.run(run_agent(**options)), calls


def test_terminal_claim_limit_stops_after_rejected_finishes(tmp_path):
    rejected = {"ok": False, "finished": False, "terminated": False}
    report, calls = run_loop(
        tmp_path, [model_call("finish", {}), model_call("finish", {}, call_id="native-2"), model_call("tap", {"id": "el:x"})],
        [rejected, rejected, frame("fp-1")], terminal_claim_limit=2)
    assert report["stop_reason"] == "terminal_claimed"
    assert report["terminal_claims"] == 2 and report["error"] is None
    assert report["terminal_submission"]["accepted"] is False and report["terminal_submission"]["tool"] == "finish"
    assert [name for name, _ in calls] == ["finish", "finish"], "the loop stopped before the third model turn"


def test_realapp_judges_saved_frames_when_controller_reaches_step_budget(tmp_path):
    aua = FakeAua()
    model = FakeModel(
        controller=[model_call("tap_and_analyze", {"id": "el:fp-home-1"})],
        judgements={
            "record_verdict": [
                verdict("pass", "goal state is visible"),
                verdict("pass", "goal state is independently visible"),
            ]
        },
    )

    result = run(tmp_path, aua, model, max_steps=1)

    assert result["controller"]["stop_reason"] == "step_budget"
    assert result["verdict"]["verdict"] == "pass_with_warning"
    assert any("step budget" in warning for warning in result["warnings"])


def test_accepted_terminal_result_still_wins_over_the_claim_limit(tmp_path):
    report, _ = run_loop(tmp_path, [model_call("finish", {})], [{"ok": True, "finished": True, "terminated": True}],
                         terminal_claim_limit=1)
    assert report["stop_reason"] == "terminal_tool" and report["terminal_claims"] == 0


def test_no_progress_limit_stops_on_a_frozen_fingerprint(tmp_path):
    responses = [model_call("tap", {"id": f"el:{i}"}, call_id=f"native-{i}") for i in range(6)]
    results = [frame("fp-1"), frame("fp-1"), frame("fp-1"), frame("fp-1"), frame("fp-2"), frame("fp-2")]
    report, calls = run_loop(tmp_path, responses, results, no_progress_limit=3)
    assert report["stop_reason"] == "no_progress" and report["no_progress_streak"] == 3
    assert len(calls) == 4, "first fp-1 frame starts the streak; three repeats trip it"
    changing, _ = run_loop(tmp_path / "b", responses, [frame(f"fp-{i}") for i in range(6)], no_progress_limit=3, max_steps=6)
    assert changing["stop_reason"] == "step_budget" and changing["error"] is None
    assert "step budget" in changing["warnings"][0]


@pytest.mark.parametrize("kwargs", [
    {"terminal_claim_limit": 0}, {"no_progress_limit": -1}, {"terminal_claim_limit": 1.5},
    {"terminal_claim_limit": 1, "terminal_tools": frozenset()},
])
def test_breaker_limits_are_validated(tmp_path, kwargs):
    with pytest.raises(RunError):
        run_loop(tmp_path, [], [], **kwargs)


# --- run_realapp end to end with fake AUA and fake model -----------------------------------

MCP_SCHEMAS = {
    "analyze_screen": {"type": "object", "properties": {"no_cache": {"type": "boolean"}, "source": {"type": "string"},
                                                        "with_image": {"type": "boolean"}}, "description": "Analyze."},
    "tap_and_analyze": {"type": "object", "properties": {"id": {"type": "string"}, "coords": {"type": "array"},
                                                         "phase_done": {"type": "string"}}, "description": "Tap."},
    "long_press_and_analyze": {"type": "object", "properties": {"id": {"type": "string"},
                                                                "ms": {"type": "integer"},
                                                                "until": {"type": "string"}},
                               "required": ["id"], "description": "Long press."},
    "input_and_analyze": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                                           "submit": {"type": "boolean"}}, "required": ["text"]},
    "swipe_and_analyze": {"type": "object", "properties": {"direction": {"type": "string"}}, "required": ["direction"]},
    # Mirrors the server's `scroll` tool: a direction plus an optional percent of the scrollable
    # container. That container is why a scroll replays and a raw swipe does not.
    "scroll_and_analyze": {"type": "object",
                           "properties": {"direction": {"type": "string",
                                                        "enum": ["up", "down", "left", "right"]},
                                          "percent": {"type": "integer"}},
                           "required": ["direction"]},
    # Mirrors the server's `open_link`: a URI, with package pinning left at its default so the
    # VIEW intent stays on the app under test.
    "open_link_and_analyze": {"type": "object",
                              "properties": {"uri": {"type": "string"},
                                             "package": {"type": "string"},
                                             "pin_package": {"type": "boolean"}},
                              "required": ["uri"]},
    "wait_and_analyze": {"type": "object", "properties": {"for_": {"type": "string"}, "idle": {"type": "boolean"},
                                                          "timeout": {"type": "integer"}}},
    "key_and_analyze": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "back_gesture_and_analyze": {"type": "object", "properties": {"observe_fields": {"type": "string"},
                                                                    "until": {"type": "string"}},
                                 "description": "Edge back."},
    "session_progress": {"type": "object", "properties": {"session_id": {"type": "string"}}},
    "session_finish": {"type": "object", "properties": {"session_id": {"type": "string"}, "allow_incomplete": {"type": "boolean"},
                                                        "summary": {"type": "boolean"}}},
    "expect_and_analyze": {"type": "object", "properties": {"rid": {"type": "string"}, "text": {"type": "string"},
                                                            "desc": {"type": "string"}, "exists": {"type": "boolean"},
                                                            "absent": {"type": "boolean"},
                                                            "text_contains": {"type": "string"}}},
    "network_offline": {"type": "object", "properties": {"verify": {"type": "boolean"}}},
    "network_restore": {"type": "object", "properties": {"timeout_ms": {"type": "integer"}}},
    "job_start": {"type": "object", "properties": {"operation": {"type": "string"}}},
    "job_status": {"type": "object", "properties": {"job_id": {"type": "string"}}},
    "job_cancel": {"type": "object", "properties": {"job_id": {"type": "string"}}},
}


class FakeAua:
    """Enough of AUA's MCP surface for the runner: a session, three screens, one tap."""

    def __init__(self, *, accept_finish=False, knowledge=None):
        self.calls = []
        self.accept_finish = accept_finish
        self.knowledge = knowledge
        self.screen = frame("fp-home", ("Chats", "Settings"))

    async def call_tool(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        if name == "session_start":
            start = {"ok": True, "session_id": "sess-1", "serial": "emulator-0000", "recommended_call": {"secret": 1}}
            if self.knowledge is not None:
                start["relevant_knowledge"] = copy.deepcopy(self.knowledge)
            return start
        if name == "app_launch_and_analyze":
            return copy.deepcopy(self.screen)
        if name == "analyze_screen":
            return copy.deepcopy(self.screen)
        if name == "tap_and_analyze":
            self.screen = frame("fp-theme", ("Theme", "Light", "Dark"))
            return copy.deepcopy(self.screen)
        if name == "session_finish":
            if self.accept_finish and arguments.get("allow_incomplete") is False:
                return {"ok": True, "finished": True, "terminated": True}
            return {"ok": arguments.get("allow_incomplete") is True, "finished": False, "terminated": arguments.get("allow_incomplete") is True}
        raise AssertionError(f"unexpected tool {name}")

    async def list_tools(self):
        return copy.deepcopy(MCP_SCHEMAS)


class FakeModel:
    """Controller turns come in order; each judgement role answers by the forced tool name."""

    def __init__(self, controller, judgements):
        self.controller = list(controller)
        self.judgements = {name: list(replies) for name, replies in judgements.items()}
        self.payloads = []

    async def send(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        choice = payload.get("tool_choice")
        if isinstance(choice, dict):
            return copy.deepcopy(self.judgements[choice["function"]["name"]].pop(0))
        return copy.deepcopy(self.controller.pop(0))


def verdict(value, reason):
    return model_call("record_verdict", {"verdict": value, "confidence": 0.8, "reasons": [reason]}, cost=0.0004)


def screen_name(name, kind="settings"):
    return model_call("record_screen_name", {"logical_name": name, "kind": kind, "purpose": f"{name} screen.",
                                             "landmarks": ["Theme"]}, cost=0.0002)


def run(tmp_path, aua, model, **kwargs):
    options = {"call_tool": aua.call_tool, "list_tools": aua.list_tools, "send": model.send,
               "goal": "Switch the app theme to Light", "package": "com.example.fictional",
               "output": tmp_path / "run", "model": "fictional/model", "request_config": SETTINGS}
    options.update(kwargs)
    return asyncio.run(run_realapp(**options))


def test_realapp_tools_offer_compact_schemas_plus_an_outcome_claim():
    tools = realapp_tools(MCP_SCHEMAS)
    assert [tool["function"]["name"] for tool in tools] == [
        name for name in CONTROLLER_TOOLS if name != "session_progress"
    ]
    by_name = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    assert set(by_name["tap_and_analyze"]["properties"]) == {"id"} and by_name["tap_and_analyze"]["required"] == ["id"]
    assert set(by_name["long_press_and_analyze"]["properties"]) == {"id"}
    assert by_name["long_press_and_analyze"]["required"] == ["id"]
    assert by_name["back_gesture_and_analyze"]["properties"] == {}
    assert by_name["back_gesture_and_analyze"]["additionalProperties"] is False
    assert "coords" not in json.dumps(by_name)
    assert set(by_name["session_finish"]["properties"]) == {"outcome", "note"}
    assert "session_id" not in json.dumps(by_name)
    with pytest.raises(RunError):
        realapp_tools({name: schema for name, schema in MCP_SCHEMAS.items() if name != "session_finish"})


def test_compact_input_guidance_distinguishes_typing_submit_and_app_send():
    tools = {t["function"]["name"]: t["function"] for t in realapp_tools(MCP_SCHEMAS)}
    schema = tools["input_and_analyze"]["parameters"]
    assert schema["properties"]["submit"]["type"] == "boolean"
    assert "IME action" in schema["properties"]["submit"]["description"]
    assert "submit=true" in tools["input_and_analyze"]["description"]
    assert "not a chat-send shortcut" in tools["key_and_analyze"]["description"]
    for phrase in ("submit=true", "CANCEL", "Close", "Do not type the same text again",
                   "submitted=false", "editable", "IME Enter"):
        assert phrase in COMPACT_SYSTEM
    assert "send_key" not in schema["properties"], "Do not advertise unsupported compact arguments"


def test_controller_requests_semantic_fields_only_when_public_schema_supports_it():
    from android_ui_analyser.mcp_server import _tool_definitions

    schemas = {tool.name: dict(tool.inputSchema) for tool in _tool_definitions()}
    args = {"id": "current"}
    result = controller_observation_arguments("tap_and_analyze", args, schemas)
    assert {"id", "type", "resource_id", "bounds", "window", "focused"} <= set(result["observe_fields"].split(","))
    assert args == {"id": "current"}, "Never mutate model arguments or its action journal"
    assert controller_observation_arguments("session_finish", {}, schemas) == {}
    assert controller_observation_arguments("unknown", args, schemas) == args


@pytest.mark.parametrize("preserve", [False, True])
def test_realapp_end_state_guidance_is_explicit_and_does_not_change_cleanup(tmp_path, preserve):
    from experiments.aua_controller.run_live import (
        HOME_FINISH_INSTRUCTION,
        PRESERVE_FINISH_INSTRUCTION,
    )

    aua = FakeAua()
    model = FakeModel([model_call("session_finish", {"outcome": "achieved"})], {})
    result = run(tmp_path, aua, model, judge=False, preserve_end_state=preserve)
    prompt = model.payloads[0]["messages"][0]["content"]
    assert (PRESERVE_FINISH_INSTRUCTION if preserve else HOME_FINISH_INSTRUCTION) in prompt
    assert (HOME_FINISH_INSTRUCTION if preserve else PRESERVE_FINISH_INSTRUCTION) not in prompt
    assert result.get("cleanup_error") is None
    assert sum(name == "session_finish" for name, _ in aua.calls) == 1


def test_loading_capture_reaches_judges_but_never_becomes_action_safe(tmp_path):
    from experiments.aua_controller.session_state import observation_frame

    class LoadingAua(FakeAua):
        async def call_tool(self, name, arguments):
            result = await super().call_tool(name, arguments)
            if name == "tap_and_analyze":
                result = frame("fp-loading", ("Working...",), observation_present=True,
                               observation_contract={"fingerprint": "fp-loading", "reusable": False,
                                                     "evidence_fresh": False})
                result["observation"]["meta"].update(arrival_state="loading", stale_risk=True)
                assert observation_frame(result) is None
            return result

    model = FakeModel([model_call("tap_and_analyze", {"id": "el:fp-home-1"}),
                       model_call("session_finish", {"outcome": "achieved"})],
                      {"record_verdict": [verdict("pass", "observed"), verdict("pass", "observed")]})
    run(tmp_path, LoadingAua(), model)
    judges = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)]
    assert len(judges) == 2
    for payload in judges:
        content = json.dumps(payload["messages"])
        assert "Working..." in content
        assert "observed_transient_state" in content
        assert "proves_settled_destination" in content


def test_realapp_tools_accept_the_current_public_mcp_schemas():
    from android_ui_analyser.mcp_server import _tool_definitions

    schemas = {}
    for tool in _tool_definitions():
        schema = dict(tool.inputSchema)
        schema.setdefault("description", tool.description)
        schemas[tool.name] = schema

    offered = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in realapp_tools(schemas)
    }

    assert set(offered["long_press_and_analyze"]["properties"]) == {"id"}
    assert offered["long_press_and_analyze"]["required"] == ["id"]
    assert offered["back_gesture_and_analyze"]["properties"] == {}


def test_bare_element_uuid_is_repaired_without_rewriting_labels_or_stable_keys():
    bare = "af09101646e54c9aaa2cd4538b65197e"

    assert normalize_element_id_argument({"id": bare}) == ({"id": f"el:{bare}"}, True)
    for value in ("Continue with limited access", "buttonContinueAsGuest", "rid:button", "el:abc"):
        arguments = {"id": value}
        assert normalize_element_id_argument(arguments) == (arguments, False)


def test_contract_mode_keeps_session_progress_for_authored_checkpoints():
    names = [tool["function"]["name"] for tool in realapp_tools(MCP_SCHEMAS, contract=True)]
    assert "session_progress" in names
    assert "expect_and_analyze" in names


def test_realapp_tools_add_only_requested_resilience_capabilities():
    tools = realapp_tools(
        MCP_SCHEMAS,
        capabilities=["network", "wall-clock-wait", "async-ui-wait", "app-lifecycle"],
    )
    names = [tool["function"]["name"] for tool in tools]
    assert names[-6:] == [
        "network_offline",
        "network_restore",
        "wait_uninterrupted_620_seconds",
        "wait_for_ui_condition",
        "app_force_stop",
        "app_relaunch_and_analyze",
    ]
    async_schema = next(
        tool["function"]["parameters"]
        for tool in tools
        if tool["function"]["name"] == ASYNC_UI_WAIT_TOOL
    )
    assert async_schema["required"] == ["anchor", "pending_text"]
    assert async_schema["properties"]["timeout_seconds"]["maximum"] == ASYNC_UI_WAIT_MAX_SECONDS
    with pytest.raises(RunError, match="unknown controller capabilities"):
        realapp_tools(MCP_SCHEMAS, capabilities=["shell"])
    with pytest.raises(RunError, match="does not offer job_cancel"):
        realapp_tools(
            {name: schema for name, schema in MCP_SCHEMAS.items() if name != "job_cancel"},
            capabilities=["async-ui-wait"],
        )


def test_only_explicit_harness_waits_get_long_controller_tool_timeouts():
    configured = controller_tool_timeouts(
        ["network", "wall-clock-wait", "async-ui-wait", "app-lifecycle"]
    )
    assert set(configured) == {WALL_CLOCK_WAIT_TOOL, ASYNC_UI_WAIT_TOOL}
    assert configured[WALL_CLOCK_WAIT_TOOL] > WALL_CLOCK_WAIT_SECONDS
    assert configured[ASYNC_UI_WAIT_TOOL] > ASYNC_UI_WAIT_MAX_SECONDS
    assert controller_tool_timeouts(["network", "app-lifecycle"]) == {}


def test_async_ui_wait_builds_one_bounded_positive_and_negative_predicate():
    assert async_ui_wait_spec({
        "anchor": r"text:Report\, ready",
        "pending_text": r"Working, please wait",
        "timeout_seconds": 620,
    }) == (r"text:Report\, ready,!text:Working\, please wait", 620)
    for arguments in (
        {"anchor": "!text:Working", "pending_text": "Working"},
        {"anchor": "net:GET /result", "pending_text": "Working"},
        {"anchor": "text:Ready,text:Extra", "pending_text": "Working"},
        {"anchor": "text:Working", "pending_text": "Working"},
        {
            "anchor": "text:Ready",
            "pending_text": "Working",
            "timeout_seconds": ASYNC_UI_WAIT_MAX_SECONDS + 1,
        },
    ):
        with pytest.raises(RunError, match="async UI wait"):
            async_ui_wait_spec(arguments)


def test_async_ui_wait_uses_one_model_call_then_host_polls_the_durable_job(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "experiments.aua_controller.run_realapp.ASYNC_UI_WAIT_STATUS_POLL_SECONDS", 0
    )

    class DeferredAua(FakeAua):
        async def call_tool(self, name, arguments):
            if name == "job_start":
                self.calls.append((name, copy.deepcopy(arguments)))
                return {"ok": True, "job_id": "job-1", "status": "running", "terminal": False}
            if name == "job_status":
                self.calls.append((name, copy.deepcopy(arguments)))
                self.screen = frame("fp-ready", ("Report ready",))
                completed = {
                    **copy.deepcopy(self.screen),
                    "action": "await",
                    "await_outcome": "satisfied",
                    "capture_evidence": {"ref": "job:job-1", "finished": True},
                }
                return {
                    "ok": True,
                    "job_id": "job-1",
                    "status": "succeeded",
                    "terminal": True,
                    "run_ok": True,
                    "result": completed,
                }
            return await super().call_tool(name, arguments)

    aua = DeferredAua()
    model = FakeModel(
        controller=[
            model_call(
                ASYNC_UI_WAIT_TOOL,
                {
                    "anchor": "text:Report ready",
                    "pending_text": "Working",
                    "timeout_seconds": 620,
                },
            ),
            model_call(
                "session_finish",
                {"outcome": "achieved", "note": "The report is ready"},
                call_id="native-2",
            ),
        ],
        judgements={
            "record_verdict": [
                verdict("pass", "ready frame captured"),
                verdict("pass", "fresh evidence agrees"),
            ]
        },
    )

    result = run(
        tmp_path,
        aua,
        model,
        controller_capabilities=["async-ui-wait"],
        time_limit_s=700,
    )

    controller_payloads = [
        payload for payload in model.payloads if not isinstance(payload.get("tool_choice"), dict)
    ]
    assert len(controller_payloads) == 2, "the wait itself spent no model polling turns"
    assert [name for name, _ in aua.calls].count("job_start") == 1
    assert [name for name, _ in aua.calls].count("job_status") == 1
    start = next(arguments for name, arguments in aua.calls if name == "job_start")
    assert start == {
        "operation": "await",
        "predicate": "text:Report ready,!text:Working",
        "timeout_ms": 620_000,
        "poll_ms": 500,
        "observe": True,
    }
    assert result["deferred_waits"][0]["model_calls_during_wait"] == 0
    assert result["deferred_waits"][0]["capture_evidence"]["ref"] == "job:job-1"
    wait_call = json.loads(
        (tmp_path / "run/controller/tool-calls.jsonl").read_text().splitlines()[0]
    )
    assert wait_call["tool"] == ASYNC_UI_WAIT_TOOL
    assert wait_call["result"]["observation"]["meta"]["fingerprint"] == "fp-ready"
    assert json.loads((tmp_path / "run/deferred-waits.json").read_text())[0]["status"] == "succeeded"


def test_async_ui_wait_cancels_its_owned_job_when_the_controller_is_cancelled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "experiments.aua_controller.run_realapp.ASYNC_UI_WAIT_STATUS_POLL_SECONDS", 0
    )
    output = tmp_path / "run"
    output.mkdir()
    calls = []
    status_started = asyncio.Event()
    never = asyncio.Event()

    async def call(name, arguments, actor):
        calls.append((name, copy.deepcopy(arguments), actor))
        if name == "job_start":
            return {"ok": True, "job_id": "job-cancel", "status": "running", "terminal": False}
        if name == "job_status":
            status_started.set()
            await never.wait()
        if name == "job_cancel":
            return {
                "ok": True,
                "job_id": "job-cancel",
                "status": "cancelled",
                "terminal": True,
            }
        raise AssertionError(name)

    async def exercise():
        result = {"deferred_waits": []}
        task = asyncio.create_task(
            run_async_ui_wait(
                call=call,
                arguments={
                    "anchor": "rid:result_card",
                    "pending_text": "Working",
                    "timeout_seconds": 30,
                },
                result=result,
                output=output,
            )
        )
        await status_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return result

    result = asyncio.run(exercise())
    assert [name for name, _, _ in calls] == ["job_start", "job_status", "job_cancel"]
    assert calls[-1][1] == {"job_id": "job-cancel", "wait_ms": 10_000}
    assert result["deferred_waits"][0]["cancelled"] is True


def test_async_ui_wait_cancels_its_owned_job_when_status_poll_times_out(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    calls = []

    async def call(name, arguments, actor):
        calls.append((name, copy.deepcopy(arguments), actor))
        if name == "job_start":
            return {"ok": True, "job_id": "job-timeout", "status": "running", "terminal": False}
        if name == "job_status":
            raise TimeoutError("status deadline")
        if name == "job_cancel":
            return {
                "ok": True,
                "job_id": "job-timeout",
                "status": "cancelled",
                "terminal": True,
            }
        raise AssertionError(name)

    result = {"deferred_waits": []}
    with pytest.raises(TimeoutError, match="status deadline"):
        asyncio.run(
            run_async_ui_wait(
                call=call,
                arguments={
                    "anchor": "desc:Result ready",
                    "pending_text": "Still processing",
                    "timeout_seconds": 30,
                },
                result=result,
                output=output,
            )
        )

    assert [name for name, _, _ in calls] == ["job_start", "job_status", "job_cancel"]
    assert calls[-1][1] == {"job_id": "job-timeout", "wait_ms": 10_000}
    assert result["deferred_waits"][0]["cancelled"] is True


def test_realapp_claim_stops_the_loop_and_two_judges_decide(tmp_path):
    aua = FakeAua()
    model = FakeModel(
        controller=[model_call("tap_and_analyze", {"id": "el:fp-home-1"}),
                    model_call("session_finish", {"outcome": "achieved", "note": "Light is selected"}, call_id="native-2")],
        judgements={"record_verdict": [verdict("pass", "final frame lists Light"), verdict("pass", "Light visible")]},
    )
    result = run(tmp_path, aua, model)

    assert result["error"] is None
    assert result["verdict"]["verdict"] == "pass" and result["verdict"]["oracle"] == "model_judgement_v1"
    assert result["verdict"]["verified"] is False and result["verdict"]["agreement"] is True
    assert result["claim"] == {"outcome": "achieved", "note": "Light is selected"}
    assert result["controller"]["stop_reason"] == "terminal_claimed" and result["controller"]["terminal_claims"] == 1
    assert result["controller"]["steps_consumed"] == 2, "no wasted evidence hunt after the claim"

    names = [name for name, _ in aua.calls]
    assert names == [
        "session_start", "app_launch_and_analyze", "tap_and_analyze", "analyze_screen", "session_finish"
    ]
    assert aua.calls[4][1]["allow_incomplete"] is True, (
        "the model's completion claim must not release the lease before evidence and judgement"
    )

    controller_payloads = [p for p in model.payloads if not isinstance(p.get("tool_choice"), dict)]
    judge_payloads = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)]
    assert len(controller_payloads) == 2 and len(judge_payloads) == 2
    first_user = controller_payloads[0]["messages"][1]["content"]
    assert "Initial observation" in first_user and "bounds" in first_user and "recommended_call" not in first_user
    assert "el:fp-home-1" in first_user, "controller keeps ids so it can act"
    assert all(len(p["messages"]) == 2 for p in judge_payloads), "judges start from a fresh window"
    assert all("Light is selected" not in json.dumps(p["messages"][0]) for p in judge_payloads)
    assert "controller_claim_untrusted" in judge_payloads[0]["messages"][1]["content"]

    assert result["cost"]["controller"]["usd"] == pytest.approx(0.002)
    assert result["cost"]["judge"]["usd"] == pytest.approx(0.0008)
    assert result["cost"]["total_usd"] == pytest.approx(0.0028)
    assert result["cost"]["controller"]["provider"] == "fictional"

    output = tmp_path / "run"
    written = json.loads((output / "result.json").read_text())
    assert written["verdict"]["verdict"] == "pass"
    text = (output / "verdict.md").read_text()
    assert text.startswith("# PASS") and "model_judgement_v1" in text and "| total |" in text
    assert (output / "controller" / "controller-result.json").exists()
    assert json.loads((output / "final-observation.json").read_text())["observation"]["meta"]["fingerprint"] == "fp-theme"
    assert (output / "judge" / "judgements.jsonl").exists()
    assert not (output / "screens.json").exists(), "map building is opt-in"


def test_setup_proof_uses_run_mark_and_rechecks_latest_state_before_cleanup(tmp_path):
    class ProofAua(FakeAua):
        reads = 0

        async def call_tool(self, name, arguments):
            if name == "logcat_mark":
                self.calls.append((name, copy.deepcopy(arguments)))
                return {"ok": True}
            if name == "logcat_dump":
                self.calls.append((name, copy.deepcopy(arguments)))
                self.reads += 1
                return {"ok": True, "lines": ['{"tier": "premium"}' if self.reads == 1
                                               else '{"tier": "free"}']}
            return await super().call_tool(name, arguments)

    aua = ProofAua()
    model = FakeModel(
        controller=[model_call("session_finish", {"outcome": "achieved", "note": "observed"})],
        judgements={"record_verdict": [verdict("pass", "observed"), verdict("pass", "observed")]},
    )
    result = run(tmp_path, aua, model, setup_proof=(
        "setup_tier", r'"tier"\s*:\s*"(?P<value>[^"]+)"', "premium"), setup_proof_regex=True)
    assert result["setup_tier"] == {"verified": False, "actual": None, "source": "logcat"}
    names = [name for name, _ in aua.calls]
    assert names.index("session_start") < names.index("logcat_mark") < names.index("app_launch_and_analyze")
    marker = next(args["name"] for name, args in aua.calls if name == "logcat_mark")
    assert all(args["since"] == marker for name, args in aua.calls if name == "logcat_dump")
    assert names[-1] == "session_finish"


@pytest.mark.parametrize("latest", [None, "error", "exception"])
def test_database_setup_proof_rechecks_and_never_logs_private_results(tmp_path, monkeypatch, latest):
    import functools

    from experiments.aua_controller import run_realapp as module

    monkeypatch.setattr(module, "capture_database_setup_proof", functools.partial(
        module.capture_database_setup_proof, poll_timeout_s=0,
    ))
    secret = "private-account@example.test"

    class ProofAua(FakeAua):
        reads = 0

        async def call_tool(self, name, arguments):
            if name == "logcat_mark":
                self.calls.append((name, copy.deepcopy(arguments)))
                return {"ok": True, "clock": "device", "unix_ms": 100}
            if name == "database_query":
                self.calls.append((name, copy.deepcopy(arguments)))
                self.reads += 1
                if self.reads > 1 and latest == "exception":
                    raise RuntimeError(secret)
                if self.reads > 1 and latest == "error":
                    return {"ok": False, "error": secret, "responseBody": secret}
                return {"ok": True, "columns": ["actual"],
                        "rows": [["premium" if self.reads == 1 else None]],
                        "untrusted_extra": secret}
            return await super().call_tool(name, arguments)

    aua = ProofAua()
    model = FakeModel(
        controller=[model_call("session_finish", {"outcome": "achieved", "note": "observed"})],
        judgements={"record_verdict": [verdict("pass", "observed"), verdict("pass", "observed")]},
    )
    result = run(tmp_path, aua, model, setup_proof=("setup_tier", "database", "premium"),
                 setup_proof_query={"database": "proof.db", "sql":
                     "SELECT actual, observed_at_ms FROM evidence WHERE observed_at_ms >= :since_unix_ms"})
    assert aua.reads == 2
    assert result["setup_tier"] == ({"verified": False, "actual": None, "source": "database"}
                                      if latest is None else None)
    for path in (tmp_path / "run").rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(), path
    calls = [args for name, args in aua.calls if name == "database_query"]
    assert all(args["parameters"]["since_unix_ms"] == 100 for args in calls)
    assert aua.calls[-1][0] == "session_finish"


@pytest.mark.parametrize("saveable", [True, False])
@pytest.mark.parametrize("evidence_gap", [True, False])
@pytest.mark.parametrize("cleanup_fails", [True, False])
def test_primary_flow_preview_runs_before_cleanup_and_cannot_hide_failure(tmp_path, saveable, evidence_gap, cleanup_fails):
    class FlowAua(FakeAua):
        async def call_tool(self, name, arguments):
            if name == "session_finish" and cleanup_fails:
                self.calls.append((name, copy.deepcopy(arguments)))
                return {"ok": False, "finished": False, "error": "cleanup refused"}
            if name == "flow_save":
                self.calls.append((name, copy.deepcopy(arguments)))
                return {"ok": saveable, "steps": 1, "scope": {"requested_last": 1,
                        "selected": 1, "boundary_omitted": 0},
                        "preview": "schema_version: 1\nsteps: []\n"}
            return await super().call_tool(name, arguments)

    aua = FlowAua()
    judgement = verdict("pass", "observed")
    if evidence_gap:
        judgement = model_call("record_verdict", {
            "verdict": "unverified", "confidence": 0.8, "reasons": ["device default not observed"],
            "criteria": [{"criterion": "Device default follows system.", "result": "not_verified",
                          "evidence": "system mode unavailable"}],
        })
    model = FakeModel(
        controller=[model_call("tap_and_analyze", {"id": "el:fp-home-1"}),
                    model_call("session_finish", {"outcome": "achieved", "note": "observed"})],
        judgements={"record_verdict": [judgement, judgement]},
    )
    result = run(tmp_path, aua, model, save_primary_flow=True,
                 contract="- Device default follows system." if evidence_gap else None)
    names = [name for name, _ in aua.calls]
    assert names.index("flow_save") < names.index("session_finish")
    if saveable:
        assert result["primary_flow"] == "flow.yaml"
        assert result["route_action_count"] == 1
        assert (tmp_path / "run/flow.yaml").is_file()
    else:
        assert result["primary_flow_warning"]
        assert result["primary_flow_export"] == {"status": "unavailable", "reason": "proof_unavailable"}
        assert result["route_action_count"] == 1
        assert not (tmp_path / "run/flow.yaml").exists()
    if cleanup_fails:
        assert result["verdict"]["verdict"] == "unverified"
        assert result["verdict"]["cleanup_verified"] is False
        assert result["cleanup_error"]
    else:
        assert result["verdict"]["verdict"] == ("unverified" if evidence_gap else "pass")
        assert not result["error"]
        assert result["verdict"].get("cleanup_verified") is not False


def test_corrupt_execution_is_not_downgraded_to_optional_flow_warning(tmp_path, monkeypatch):
    from experiments.aua_controller import run_realapp as module

    async def corrupt(*args):
        raise module.ControllerJournalError("unknown execution outcome")

    monkeypatch.setattr(module, "export_primary_flow", corrupt)
    model = FakeModel(
        controller=[model_call("session_finish", {"outcome": "achieved", "note": "observed"})],
        judgements={"record_verdict": [verdict("pass", "observed"), verdict("pass", "observed")]},
    )
    result = run(tmp_path, FakeAua(), model, save_primary_flow=True)
    assert result["verdict"]["verdict"] == "unverified" and result["execution_error"]
    assert not result.get("primary_flow_warning")
    assert result["verdict"].get("cleanup_verified") is not False


def test_failed_judgement_spend_survives_in_result_and_markdown(tmp_path):
    aua = FakeAua()
    invalid = model_call("record_verdict", {"verdict": "invalid"}, cost=0.005)
    model = FakeModel(
        controller=[model_call("session_finish", {"outcome": "achieved", "note": "observed"}, cost=0.003)],
        judgements={"record_verdict": [invalid, invalid]},
    )
    result = run(tmp_path, aua, model)
    assert result["verdict"]["verdict"] == "unverified" and result["error"]
    assert result["cost"]["controller"]["usd"] == pytest.approx(0.003)
    assert result["cost"]["judge"]["usd"] == pytest.approx(0.010)
    assert result["cost"]["total_usd"] == pytest.approx(0.013)
    assert result["cost"]["judge"]["decider"]["requests"] == 2
    saved = json.loads((tmp_path / "run/result.json").read_text())
    assert saved["cost"]["total_usd"] == pytest.approx(0.013)
    assert "0.013000" in (tmp_path / "run/verdict.md").read_text()


def test_cancel_during_judge_preserves_unknown_cost_and_still_finishes_session(tmp_path):
    aua = FakeAua()

    async def exercise():
        judging = asyncio.Event()

        async def send(payload):
            if isinstance(payload.get("tool_choice"), dict):
                judging.set()
                await asyncio.Event().wait()
            return model_call("session_finish", {"outcome": "achieved", "note": "observed"})

        task = asyncio.create_task(run_realapp(
            call_tool=aua.call_tool, list_tools=aua.list_tools, send=send,
            goal="Observe theme", package="com.example.fictional", output=tmp_path / "run",
            model="fictional/model", request_config=SETTINGS,
        ))
        await asyncio.wait_for(judging.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert aua.calls[-1][0] == "session_finish"
    saved = json.loads((tmp_path / "run/result.json").read_text())
    assert saved["verdict"]["verdict"] == "unverified"
    assert "cancelled" in saved["error"]
    assert saved["cost"]["complete"] is False
    assert saved["cost"]["judge"]["decider"]["unreported_cost_requests"] == 1
    assert "provider charges are unknown" in (tmp_path / "run/verdict.md").read_text()
    assert "decision_cancelled" in (tmp_path / "run/judge/judge-events.jsonl").read_text()


def test_realapp_puts_host_knowledge_into_the_first_user_message(tmp_path):
    """A live run started with three accurate facts in the store and re-derived all of them by hand."""
    fact = {"id": "knowledge_theme", "kind": "claim", "name": None, "aliases": ["switch theme"], "score": 70,
            "text": "The appearance setting lives in a preferences file; a fresh install forces dark."}
    recipe = {"id": "knowledge_recipe", "kind": "recipe", "name": "open-theme", "score": 40, "text": "Tap Settings, then Theme."}
    junk = {"id": "knowledge_blank", "kind": "note", "text": "   "}

    def turns():
        return [model_call("tap_and_analyze", {"id": "el:fp-home-1"}),
                model_call("session_finish", {"outcome": "achieved", "note": "Light is selected"}, call_id="native-2")]

    def judges():
        return {"record_verdict": [verdict("pass", "final frame lists Light"), verdict("pass", "Light visible")]}

    informed = FakeModel(controller=turns(), judgements=judges())
    result = run(tmp_path, FakeAua(knowledge=[fact, recipe, junk]), informed)
    first_user = informed.payloads[0]["messages"][1]["content"]
    assert first_user.startswith("Goal: Switch the app theme to Light\n\nRecorded knowledge")
    assert "- [claim] The appearance setting lives in a preferences file" in first_user
    assert "- [recipe open-theme] Tap Settings, then Theme." in first_user
    assert first_user.index("Recorded knowledge") < first_user.index("Initial observation:")
    assert "knowledge_theme" not in first_user and "secret" not in first_user, "ids and hidden keys stay host-side"
    assert result["knowledge_shown"] == ["knowledge_theme", "knowledge_recipe"]
    for payload in informed.payloads[1:]:
        if isinstance(payload.get("tool_choice"), dict):
            assert "preferences file" not in payload["messages"][1]["content"], "judges decide from frames alone"

    uninformed = FakeModel(controller=turns(), judgements=judges())
    plain = run(tmp_path, FakeAua(), uninformed, output=tmp_path / "plain")
    assert uninformed.payloads[0]["messages"][1]["content"].startswith("Goal: Switch the app theme to Light\n\nInitial observation:")
    assert plain["knowledge_shown"] == []


def test_model_finish_claim_still_requires_independent_judgement(tmp_path):
    aua = FakeAua(accept_finish=True)
    model = FakeModel(
        controller=[model_call("session_finish", {"outcome": "already_satisfied"})],
        judgements={
            "record_verdict": [
                verdict("pass", "final frame confirms the goal"),
                verdict("pass", "fresh evidence agrees"),
            ]
        },
    )
    result = run(tmp_path, aua, model)
    assert result["verdict"]["oracle"] == "model_judgement_v1"
    assert result["verdict"]["verdict"] == "pass"
    assert result["controller"]["stop_reason"] == "terminal_claimed"
    assert result["cost"]["judge"]["usd"] > 0


def test_realapp_disagreement_is_unverified_and_stall_downgrades(tmp_path):
    aua = FakeAua()
    model = FakeModel(
        controller=[model_call("session_finish", {"outcome": "achieved"})],
        judgements={"record_verdict": [verdict("pass", "looks done"), verdict("fail", "Light not selected")]},
    )
    result = run(tmp_path, aua, model)
    assert result["verdict"]["verdict"] == "unverified" and result["verdict"]["agreement"] is False
    assert (tmp_path / "run" / "verdict.md").read_text().startswith("# UNVERIFIED")

    stalled = FakeAua()
    stalled_model = FakeModel(
        controller=[model_call("analyze_screen", {"no_cache": True}, call_id=f"n{i}") for i in range(3)],
        judgements={"record_verdict": [verdict("pass", "already Light"), verdict("pass", "already Light")]},
    )
    result = run(tmp_path / "stall", stalled, stalled_model, no_progress_limit=2)
    assert result["controller"]["stop_reason"] == "no_progress"
    assert result["verdict"]["verdict"] == "pass_with_warning"
    assert result["verdict"]["reasons"][0].startswith("Controller stalled")


def test_realapp_names_screens_once_each_and_summarises_the_route(tmp_path):
    aua = FakeAua()
    model = FakeModel(
        controller=[model_call("tap_and_analyze", {"id": "el:fp-home-1"}),
                    model_call("session_finish", {"outcome": "achieved"}, call_id="native-2")],
        judgements={
            "record_verdict": [verdict("pass", "ok"), verdict("pass", "ok")],
            "record_screen_name": [screen_name("home_feed", "home"), screen_name("settings_theme")],
            "record_route_summary": [model_call("record_route_summary", {"summary": "Home, then Settings theme.",
                                                                          "landmarks": ["Theme"], "pitfalls": []}, cost=0.0003)],
        },
    )
    result = run(tmp_path, aua, model, name_screens=True)
    assert [screen["logical_name"] for screen in result["screens"]] == ["home_feed", "settings_theme"]
    assert all(screen["oracle"] == "model_judgement_v1" and screen["verified"] is False for screen in result["screens"])
    assert result["route"]["summary"].startswith("Home")
    assert [(t.get("from"), t["to"]) for t in result["route"]["transitions"]] == [(None, "home_feed"), ("home_feed", "settings_theme")]
    assert result["cost"]["map"]["usd"] == pytest.approx(0.0007)
    assert not model.judgements["record_screen_name"], "three frames but two fingerprints: the repeat was cached"
    screens = json.loads((tmp_path / "run" / "screens.json").read_text())
    assert screens[1]["fingerprint"] == "fp-theme"
    assert (tmp_path / "run" / "route.json").exists()
    assert "settings_theme" in (tmp_path / "run" / "verdict.md").read_text()


def test_realapp_records_setup_failures_and_still_cleans_up(tmp_path):
    class BrokenAua(FakeAua):
        async def call_tool(self, name, arguments):
            if name == "flow_run":
                self.calls.append((name, arguments))
                return {"ok": False, "error": {"code": "step_failed", "message": "login button missing"}}
            return await super().call_tool(name, arguments)

    aua = BrokenAua()
    result = run(
        tmp_path,
        aua,
        FakeModel([], {}),
        setup_flows=[("steps: []", {})],
        retain_started_target=False,
    )
    # The model here answers nothing, so there is no verdict to reach; what this pins is that a
    # diverged setup flow is recorded as an adaptation rather than swallowing the run, and that
    # cleanup still happens either way.
    assert result["verdict"]["verdict"] == "unverified"
    assert result["error"] is None
    assert any("setup flow 0 diverged" in warning for warning in result["warnings"])
    assert any(item.get("flow_run_ok") is False for item in result["setup"])
    assert aua.calls[-1][0] == "session_finish"
    assert aua.calls[-1][1]["allow_incomplete"] is True
    assert aua.calls[-1][1]["retain_started_target"] is False
    assert (tmp_path / "run" / "result.json").exists()


def test_realapp_refuses_a_dirty_output_directory(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "old.txt").write_text("x")
    with pytest.raises(RunError, match="must be empty"):
        run(tmp_path, FakeAua(), FakeModel([], {}))
