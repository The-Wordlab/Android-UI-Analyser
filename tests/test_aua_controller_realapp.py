from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.agent_loop import run_agent
from experiments.aua_controller.run_live import RunError
from experiments.aua_controller.run_realapp import (
    CONTROLLER_TOOLS,
    WALL_CLOCK_WAIT_SECONDS,
    WALL_CLOCK_WAIT_TOOL,
    controller_tool_timeouts,
    normalize_element_id_argument,
    realapp_tools,
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
    "input_and_analyze": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                                           "submit": {"type": "boolean"}}, "required": ["text"]},
    "swipe_and_analyze": {"type": "object", "properties": {"direction": {"type": "string"}}, "required": ["direction"]},
    "wait_and_analyze": {"type": "object", "properties": {"for_": {"type": "string"}, "idle": {"type": "boolean"},
                                                          "timeout": {"type": "integer"}}},
    "key_and_analyze": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "session_progress": {"type": "object", "properties": {"session_id": {"type": "string"}}},
    "session_finish": {"type": "object", "properties": {"session_id": {"type": "string"}, "allow_incomplete": {"type": "boolean"},
                                                        "summary": {"type": "boolean"}}},
    "expect_and_analyze": {"type": "object", "properties": {"rid": {"type": "string"}, "text": {"type": "string"},
                                                            "desc": {"type": "string"}, "exists": {"type": "boolean"},
                                                            "absent": {"type": "boolean"},
                                                            "text_contains": {"type": "string"}}},
    "network_offline": {"type": "object", "properties": {"verify": {"type": "boolean"}}},
    "network_restore": {"type": "object", "properties": {"timeout_ms": {"type": "integer"}}},
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
    assert set(by_name["session_finish"]["properties"]) == {"outcome", "note"}
    assert "session_id" not in json.dumps(by_name)
    with pytest.raises(RunError):
        realapp_tools({name: schema for name, schema in MCP_SCHEMAS.items() if name != "session_finish"})


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
        capabilities=["network", "wall-clock-wait", "app-lifecycle"],
    )
    names = [tool["function"]["name"] for tool in tools]
    assert names[-5:] == [
        "network_offline",
        "network_restore",
        "wait_uninterrupted_620_seconds",
        "app_force_stop",
        "app_relaunch_and_analyze",
    ]
    with pytest.raises(RunError, match="unknown controller capabilities"):
        realapp_tools(MCP_SCHEMAS, capabilities=["shell"])


def test_only_wall_clock_wait_gets_a_long_controller_tool_timeout():
    configured = controller_tool_timeouts(["network", "wall-clock-wait", "app-lifecycle"])
    assert set(configured) == {WALL_CLOCK_WAIT_TOOL}
    assert configured[WALL_CLOCK_WAIT_TOOL] > WALL_CLOCK_WAIT_SECONDS
    assert controller_tool_timeouts(["network", "app-lifecycle"]) == {}


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
    assert "Initial observation" in first_user and "bounds" not in first_user and "recommended_call" not in first_user
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
    result = run(tmp_path, aua, FakeModel([], {}), setup_flows=[("steps: []", {})])
    # The model here answers nothing, so there is no verdict to reach; what this pins is that a
    # diverged setup flow is recorded as an adaptation rather than swallowing the run, and that
    # cleanup still happens either way.
    assert result["verdict"]["verdict"] == "unverified"
    assert result["error"] is None
    assert any("setup flow 0 diverged" in warning for warning in result["warnings"])
    assert any(item.get("flow_run_ok") is False for item in result["setup"])
    assert aua.calls[-1][0] == "session_finish"
    assert aua.calls[-1][1]["allow_incomplete"] is True
    assert (tmp_path / "run" / "result.json").exists()


def test_realapp_refuses_a_dirty_output_directory(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "old.txt").write_text("x")
    with pytest.raises(RunError, match="must be empty"):
        run(tmp_path, FakeAua(), FakeModel([], {}))
