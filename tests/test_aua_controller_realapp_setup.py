"""Setup contract for the real-app runner: flags, chained setup flows, contract, vision.

Every case here came from driving a real application and finding the runner could not express
what the run needed: a feature-flag arm, a second parameterised setup flow, the authored
acceptance criteria, or a question about appearance that element text cannot answer.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.aua_controller.judgement import encode_image, frame_fingerprint, screenshot_index
from experiments.aua_controller.run_live import RunError
from experiments.aua_controller.run_realapp import build_setup_flows, parse_pairs, run_realapp

# The sibling is imported by its bare module name: pytest puts tests/ on sys.path, while a
# dependency's stray top-level `tests` package in the venv shadows `tests.<module>`.
from test_aua_controller_realapp import (
    MCP_SCHEMAS,
    FakeModel,
    frame,
    model_call,
    verdict,
)

SETTINGS = {"provider": {"only": ["fictional"], "allow_fallbacks": False,
                         "max_price": {"prompt": 0.3, "completion": 1.2}},
            "reasoning": {"effort": "low"}}
VISION_SETTINGS = {"provider": {"only": ["seeing"], "order": ["seeing"],
                                "allow_fallbacks": False,
                                "max_price": {"prompt": 0.4, "completion": 1.6}},
                   "reasoning": {"effort": "low"}}


class SetupAua:
    """AUA's MCP surface for the setup phase, recording what each stage was asked to do."""

    def __init__(
        self,
        *,
        flags_ok=True,
        flow_ok=True,
        evidence_colours=None,
        session_starts=None,
        record_stop_ok=True,
        finish_ok=True,
    ):
        self.calls: list[tuple[str, dict]] = []
        self.flags_ok, self.flow_ok = flags_ok, flow_ok
        self.record_stop_ok, self.finish_ok = record_stop_ok, finish_ok
        self.evidence_colours = evidence_colours or {}
        self.session_starts = list(session_starts or [])
        self.screen = frame("fp-home", ("Chats", "Settings"))

    def _write_evidence(self, artifacts_dir):
        from PIL import Image

        evidence = artifacts_dir / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        entries = []
        for fingerprint, colour in self.evidence_colours.items():
            shot = evidence / f"{fingerprint}.png"
            Image.new("RGB", (720, 1280), colour).save(shot)
            entries.append({"command": "analyze_screen", "screenshot": str(shot),
                            "evidence_id": f"session-1:observation:{fingerprint}"})
        (artifacts_dir / "manifest.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")

    async def call_tool(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        if name == "session_start":
            # AUA writes its evidence bundle as the session runs, so the runner's own output
            # directory is empty when it starts. The fake writes it at the same moment.
            if self.evidence_colours:
                self._write_evidence(Path(arguments["artifacts_dir"]))
            if self.session_starts:
                return copy.deepcopy(self.session_starts.pop(0))
            return {"ok": True, "session_id": "sess-1", "serial": "emulator-0000"}
        if name == "install_app":
            return {"ok": True, "app_install": {"installed": True, "uninstalled_first": True}}
        if name == "app":
            return {"ok": True, "action": "app-grant"}
        if name == "screen_record_start":
            return {"ok": True, "action": "screen-record-start"}
        if name == "screen_record_stop":
            if not self.record_stop_ok:
                return {"ok": False, "error": {"code": "record_stop_failed"}}
            Path(arguments["path"]).write_bytes(b"fake-mp4")
            return {"ok": True, "action": "screen-record-stop", "path": arguments["path"]}
        if name == "app_launch_and_analyze":
            return copy.deepcopy(self.screen)
        if name == "flags_apply_and_analyze":
            return {"ok": True, "verified": True} if self.flags_ok else {"ok": False, "error": {"code": "flag_ignored"}}
        if name == "flow_run":
            return {"ok": True} if self.flow_ok else {"ok": False, "error": {"code": "flow_step_failed"}}
        if name == "analyze_screen":
            return copy.deepcopy(self.screen)
        if name == "tap_and_analyze":
            self.screen = frame("fp-theme", ("Theme", "Light Mode Selected"))
            return copy.deepcopy(self.screen)
        if name == "session_finish":
            return {
                "ok": self.finish_ok and arguments.get("allow_incomplete") is True,
                "finished": False,
                "terminated": self.finish_ok and arguments.get("allow_incomplete") is True,
            }
        raise AssertionError(f"unexpected tool {name}")

    async def list_tools(self):
        return copy.deepcopy(MCP_SCHEMAS)

    def named(self, name):
        return [args for called, args in self.calls if called == name]


def two_step_model(criteria=None):
    neutral = verdict("pass", "Light Mode Selected is visible")
    skeptical = verdict("pass", "the theme row agrees")
    if criteria:
        neutral["criteria"] = [
            {"criterion": criterion, "result": "verified", "evidence": "visible in the captured frame"}
            for criterion in criteria
        ]
        skeptical["criteria"] = [
            {"criterion": criterion, "result": "verified", "evidence": "independently confirmed"}
            for criterion in criteria
        ]
    return FakeModel(
        controller=[model_call("tap_and_analyze", {"id": "el:fp-home-1"}),
                    model_call("session_finish", {"outcome": "achieved", "note": "done"}, call_id="native-2")],
        judgements={"record_verdict": [neutral, skeptical]},
    )


def run(tmp_path, aua, model, **kwargs):
    options = {"call_tool": aua.call_tool, "list_tools": aua.list_tools, "send": model.send,
               "goal": "Switch the app theme to Light", "package": "com.example.fictional",
               "output": tmp_path / "run", "model": "fictional/model", "request_config": SETTINGS,
               "fresh_app": True, "apk": "/tmp/example.apk"}
    options.update(kwargs)
    return asyncio.run(run_realapp(**options))


# --- argument shapes -----------------------------------------------------------------


def test_mcp_timeout_covers_both_sequential_lease_waits():
    from datetime import timedelta

    from experiments.aua_controller.run_realapp import mcp_read_timeout

    assert mcp_read_timeout(0.0, 0.0) == timedelta(seconds=180.0)
    assert mcp_read_timeout(30.0, 600.0) == timedelta(seconds=750.0)


@pytest.mark.parametrize("given,expected", [
    (["a=1"], {"a": "1"}),
    (["a=1,b=2"], {"a": "1", "b": "2"}),
    (["a=1", "b=2"], {"a": "1", "b": "2"}),
    ([" a = 1 "], {"a": "1"}),
    (["a="], {"a": ""}),
    ([""], {}),
])
def test_parse_pairs_accepts_repeated_and_comma_joined_values(given, expected):
    assert parse_pairs(given, what="--flags") == expected


@pytest.mark.parametrize("bad", [["novalue"], ["=1"]])
def test_parse_pairs_rejects_a_pair_with_no_name_or_no_equals(bad):
    with pytest.raises(RunError):
        parse_pairs(bad, what="--flags")


def test_setup_params_bind_to_flows_by_position(tmp_path):
    first, second = tmp_path / "one.yaml", tmp_path / "two.yaml"
    first.write_text("steps: [a]", encoding="utf-8")
    second.write_text("steps: [b]", encoding="utf-8")
    flows = build_setup_flows([first, second], ["ENVIRONMENT=Curie", ""])
    assert flows[0] == ("steps: [a]", {"ENVIRONMENT": "Curie"})
    assert flows[1] == ("steps: [b]", {}), "an empty params slot means the flow takes none"
    # A flow needing no parameters can be skipped entirely by giving fewer params than flows.
    assert build_setup_flows([first, second], ["ENVIRONMENT=Curie"])[1][1] == {}
    with pytest.raises(RunError):
        build_setup_flows([first], ["A=1", "B=2"])


# --- deterministic lifecycle ----------------------------------------------------------


def test_permission_grant_does_not_require_a_fresh_reinstall(tmp_path):
    aua = SetupAua()

    run(
        tmp_path,
        aua,
        two_step_model(),
        fresh_app=False,
        apk="/tmp/example.apk",
        grant_permissions=True,
    )

    start = aua.named("session_start")[0]
    assert start["grant_permissions"] is True
    assert start.get("fresh") is not True


def test_session_bootstrap_owns_install_and_recording_wraps_launch_to_cleanup(tmp_path):
    aua = SetupAua()

    result = run(
        tmp_path,
        aua,
        two_step_model(),
        launch=True,
        record=True,
        lease_wait_s=0,
        fallback_lease_wait_s=45,
    )

    started = aua.named("session_start")
    assert len(started) == 1
    assert started[0] == {
        "goal": "Switch the app theme to Light",
        "package": "com.example.fictional",
        "headed": False,
        "artifacts_dir": str((tmp_path / "run" / "aua").resolve()),
        "evidence": "all",
        "apk": "/tmp/example.apk",
        "fresh": True,
        "confirmed": True,
        "grant_permissions": False,
        "launch_app": False,
        "wait_for_lease_s": 0,
        "provision_target": True,
    }
    assert aua.named("install_app") == []
    assert aua.named("app") == []
    order = [name for name, _ in aua.calls]
    assert order.index("screen_record_start") < order.index("app_launch_and_analyze")
    assert order.index("screen_record_stop") < max(
        index for index, name in enumerate(order) if name == "session_finish"
    )
    assert Path(result["recording"]["path"]).read_bytes() == b"fake-mp4"


def test_recording_cleanup_failure_forces_an_unverified_result(tmp_path):
    result = run(
        tmp_path,
        SetupAua(record_stop_ok=False),
        two_step_model(),
        record=True,
    )

    assert result["verdict"]["verdict"] == "unverified"
    assert result["verdict"]["oracle"] == "model_judgement_v1"
    assert result["verdict"]["reasons"], "judge evidence survives cleanup invalidation"
    assert "recording cleanup" in (result["error"] or "")
    assert result["recording"]["stop_ok"] is False


def test_failed_extra_emulator_falls_back_to_waiting_for_a_released_lease(tmp_path):
    aua = SetupAua(session_starts=[
        {"ok": False, "error": {"code": "virtual_target_start_failed", "message": "insufficient disk"}},
        {"ok": True, "session_id": "sess-waited", "serial": "emulator-5556"},
    ])

    result = run(
        tmp_path,
        aua,
        two_step_model(),
        lease_wait_s=0,
        fallback_lease_wait_s=45,
    )

    attempts = aua.named("session_start")
    assert len(attempts) == 2
    assert attempts[0]["provision_target"] is True
    assert attempts[0]["wait_for_lease_s"] == 0
    assert attempts[1]["provision_target"] is False
    assert attempts[1]["wait_for_lease_s"] == 45
    assert result["session_id"] == "sess-waited"
    assert result["lifecycle"]["lease_strategy"] == "waited_after_provision_failure"


# --- feature flags -------------------------------------------------------------------

def test_prelaunch_setup_runs_before_flags_and_product_launch(tmp_path):
    aua = SetupAua()
    run(
        tmp_path,
        aua,
        two_step_model(),
        prelaunch_setup_flows=[("name: environment", {"ENVIRONMENT": "Curie"})],
        flags={"simplification_experiment": "a"},
        setup_flows=[("name: login", {})],
    )

    ordered = [
        (name, arguments.get("yaml"))
        for name, arguments in aua.calls
        if name in {"flow_run", "flags_apply_and_analyze", "app_launch_and_analyze"}
    ]
    assert ordered == [
        ("flow_run", "name: environment"),
        ("flags_apply_and_analyze", None),
        ("app_launch_and_analyze", None),
        ("flow_run", "name: login"),
    ]


def test_the_judge_can_use_a_different_model_from_the_controller(tmp_path):
    aua, model = SetupAua(), two_step_model()
    result = run(
        tmp_path,
        aua,
        model,
        setup_flows=[],
        judge_model="vision/model",
        judge_request_config=VISION_SETTINGS,
    )
    assert result["judge_model"] == "vision/model"
    controller_turns = [p for p in model.payloads if not isinstance(p.get("tool_choice"), dict)]
    judge_turns = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)]
    assert {p["model"] for p in controller_turns} == {"fictional/model"}
    assert {p["model"] for p in judge_turns} == {"vision/model"}


def test_unverified_flag_readback_stops_before_product_journey(tmp_path):
    class UnverifiedFlagsAua(SetupAua):
        async def call_tool(self, name: str, arguments: dict):
            if name == "flags_apply_and_analyze":
                self.calls.append((name, arguments))
                return {"ok": True, "verified": False, "verification_error": "prefs unreadable"}
            return await SetupAua.call_tool(self, name, arguments)

    aua = UnverifiedFlagsAua()
    result = run(
        tmp_path,
        aua,
        two_step_model(),
        flags={"experiment": "a"},
    )

    assert "verified" in result["error"]
    assert all(name != "app_launch_and_analyze" for name, _arguments in aua.calls)


def test_flags_are_written_verified_and_recorded_before_the_setup_flow(tmp_path):
    aua = SetupAua()
    flows = [("steps: [login]", {})]
    result = run(tmp_path, aua, two_step_model(),
                 flags={"simplification_experiment": "a"}, setup_flows=flows)

    applied = aua.named("flags_apply_and_analyze")
    assert len(applied) == 1 and applied[0]["verify"] is True and applied[0]["restart"] is True
    written = Path(applied[0]["path"]).read_text(encoding="utf-8")
    assert "app: com.example.fictional" in written
    assert "simplification_experiment: a" in written
    assert result["flag_context"] == {"simplification_experiment": "a"}
    order = [name for name, _ in aua.calls]
    assert order.index("session_start") < order.index("flags_apply_and_analyze") < order.index("flow_run"), (
        "session bootstrap needs to install the app before flags, and setup needs the flag arm")


def test_an_ignored_flag_fails_the_run_instead_of_judging_the_wrong_arm(tmp_path):
    aua = SetupAua(flags_ok=False)
    result = run(tmp_path, aua, two_step_model(), flags={"gone_key": "a"}, setup_flows=[("steps: [x]", {})])
    assert result["error"] is not None and "feature flags not applied" in result["error"]
    assert aua.named("flow_run") == [], "no setup flow runs once the arm could not be established"
    assert result["verdict"]["verdict"] == "unverified"


def test_no_flags_means_no_flag_call_at_all(tmp_path):
    aua = SetupAua()
    result = run(tmp_path, aua, two_step_model(), setup_flows=[("steps: [login]", {})])
    assert aua.named("flags_apply_and_analyze") == [] and "flag_context" not in result


# --- chained setup flows --------------------------------------------------------------

def test_setup_flows_run_in_order_each_with_its_own_parameters(tmp_path):
    aua = SetupAua()
    run(tmp_path, aua, two_step_model(), setup_flows=[
        ("name: environment", {"ENVIRONMENT": "Curie"}),
        ("name: login", {}),
    ])
    ran = aua.named("flow_run")
    assert [call["yaml"] for call in ran] == ["name: environment", "name: login"]
    assert ran[0]["params"] == {"ENVIRONMENT": "Curie"}
    assert "params" not in ran[1], "a flow with no parameters is not sent an empty mapping"


def test_a_failing_setup_flow_stops_before_the_controller(tmp_path):
    aua = SetupAua(flow_ok=False)
    result = run(
        tmp_path,
        aua,
        two_step_model(),
        setup_flows=[("name: environment", {"ENVIRONMENT": "Curie"})],
    )

    assert "setup flow 0 failed" in result["error"]
    assert any(
        item.get("flow_run_ok") is False for item in result["setup"]
    )
    assert result["controller"] is None


def test_a_failing_prelaunch_flow_stops_before_flags_and_product_launch(tmp_path):
    aua = SetupAua(flow_ok=False)
    result = run(
        tmp_path,
        aua,
        two_step_model(),
        prelaunch_setup_flows=[("name: environment", {"ENVIRONMENT": "Curie"})],
        flags={"experiment": "a"},
    )

    assert "prelaunch setup flow 0 failed" in result["error"]
    assert all(name not in {"flags_apply_and_analyze", "app_launch_and_analyze"} for name, _ in aua.calls)


# --- authored contract and vision ------------------------------------------------------

def test_the_authored_contract_reaches_the_judge_and_element_text_does_not_replace_it(tmp_path):
    aua = SetupAua()
    contract = "- The theme list marks the chosen option as selected.\n- Surfaces render light."
    model = two_step_model([
        "The theme list marks the chosen option as selected.",
        "Surfaces render light.",
    ])
    run(tmp_path, aua, model, setup_flows=[], contract=contract)
    judged = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)]
    assert judged, "the judge was asked"
    body = json.dumps(judged[0]["messages"])
    assert "authored_contract" in body and "Surfaces render light" in body
    assert "authority" in body, "the judge is told the contract decides, not its own reading"


def test_vision_pairs_each_judged_frame_with_its_own_screenshot(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    run_dir = tmp_path / "run"
    evidence = run_dir / "aua" / "evidence"
    evidence.mkdir(parents=True)
    shots = {}
    for fingerprint in ("fp-home", "fp-theme"):
        path = evidence / f"{fingerprint}.png"
        Image.new("RGB", (720, 1280), (9, 9, 9) if fingerprint == "fp-home" else (250, 250, 250)).save(path)
        shots[fingerprint] = path
    (run_dir / "aua" / "manifest.json").write_text(json.dumps({"entries": [
        {"command": "analyze_screen", "evidence_id": f"session-1:observation:{fp}", "screenshot": str(path)}
        for fp, path in shots.items()
    ]}), encoding="utf-8")

    index = screenshot_index(run_dir / "aua" / "manifest.json")
    assert set(index) == {"fp-home", "fp-theme"}
    assert frame_fingerprint(frame("fp-theme")) == "fp-theme"

    encoded = encode_image(shots["fp-theme"])
    assert encoded.startswith("data:image/jpeg;base64,")
    assert len(encoded) < 40_000, "a judged screenshot is downscaled, not sent at full size"


def test_vision_sends_images_to_the_judge_and_records_how_many(tmp_path):
    pytest.importorskip("PIL")
    aua = SetupAua(evidence_colours={"fp-home": (9, 9, 9), "fp-theme": (250, 250, 250)})
    model = two_step_model()

    result = run(tmp_path, aua, model, setup_flows=[], vision=True)

    assert result["vision"]["images_attached"] >= 1
    judged = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)][0]
    content = judged["messages"][-1]["content"]
    assert isinstance(content, list), "the judge turn carries image parts, not a plain string"
    kinds = [part["type"] for part in content]
    assert kinds[0] == "text" and "image_url" in kinds
    assert content[-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_without_vision_the_judge_turn_stays_plain_text(tmp_path):
    aua, model = SetupAua(), two_step_model()
    run(tmp_path, aua, model, setup_flows=[])
    judged = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)][0]
    assert isinstance(judged["messages"][-1]["content"], str)
