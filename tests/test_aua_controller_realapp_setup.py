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
from experiments.aua_controller.run_realapp import (
    build_setup_flows,
    judge_intermediate_frame_limit,
    parse_pairs,
    run_realapp,
)

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
        record_stop_results=None,
        finish_ok=True,
        installed_packages=None,
    ):
        self.calls: list[tuple[str, dict]] = []
        self.flags_ok, self.flow_ok = flags_ok, flow_ok
        self.record_stop_ok, self.finish_ok = record_stop_ok, finish_ok
        self.record_stop_results = list(record_stop_results or [])
        self.evidence_colours = evidence_colours or {}
        self.session_starts = list(session_starts or [])
        self.installed_packages = set(installed_packages or [])
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
            return {"ok": True, "action": f"app-{arguments.get('action')}"}
        if name == "network_offline":
            return {"ok": True, "action": "network-offline", "verified": True}
        if name == "network_restore":
            return {"ok": True, "action": "network-restore", "verified": True}
        if name == "shell_read_only":
            return {"ok": True, "stdout": "1789370000\n", "stderr": ""}
        if name == "screen_record_start":
            return {"ok": True, "action": "screen-record-start"}
        if name == "screen_record_stop":
            if self.record_stop_results:
                outcome = copy.deepcopy(self.record_stop_results.pop(0))
                if outcome.get("ok") is True:
                    Path(arguments["path"]).write_bytes(b"fake-mp4")
                return outcome
            if not self.record_stop_ok:
                return {"ok": False, "error": {"code": "record_stop_failed"}}
            Path(arguments["path"]).write_bytes(b"fake-mp4")
            return {"ok": True, "action": "screen-record-stop", "path": arguments["path"]}
        if name == "app_launch_and_analyze":
            return copy.deepcopy(self.screen)
        if name == "app_status":
            package = arguments["package"]
            return {
                "ok": True,
                "package": package,
                "installed": package in self.installed_packages,
                "serial": "emulator-0000",
            }
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


def test_session_goal_can_describe_a_continuous_group_without_polluting_the_row_goal(tmp_path):
    aua = SetupAua()
    model = two_step_model()

    result = run(
        tmp_path,
        aua,
        model,
        session_goal="Keep one session for the complete ordered QA group",
    )

    assert aua.named("session_start")[0]["goal"] == "Keep one session for the complete ordered QA group"
    assert result["goal"] == "Switch the app theme to Light"
    assert result["session_goal"] == "Keep one session for the complete ordered QA group"
    assert "Goal: Switch the app theme to Light" in model.payloads[0]["messages"][-1]["content"]


def test_group_row_reuses_session_without_launch_or_cleanup(tmp_path):
    aua = SetupAua()

    result = run(
        tmp_path,
        aua,
        two_step_model(),
        fresh_app=False,
        apk=None,
        existing_session_id="sess-group",
        finish_session=False,
        launch_app=False,
        inherited_setup_facts=["Curie and the experiment flag were verified once."],
    )

    assert result["session_id"] == "sess-group"
    assert result["lifecycle"]["lease_strategy"] == "existing_session"
    assert aua.named("session_start") == []
    assert aua.named("app_launch_and_analyze") == []
    assert aua.named("session_finish") == []


def test_group_recording_starts_after_host_setup_and_before_product_launch(tmp_path):
    aua = SetupAua()

    run(
        tmp_path,
        aua,
        two_step_model(),
        record=True,
        record_after_host_setup=True,
        prelaunch_setup_flows=[("name: environment", {"ENVIRONMENT": "Curie"})],
        flags={"simplification_experiment": "a"},
    )

    order = [name for name, _ in aua.calls]
    assert order.index("flow_run") < order.index("flags_apply_and_analyze")
    assert order.index("flags_apply_and_analyze") < order.index("screen_record_start")
    assert order.index("screen_record_start") < order.index("app_launch_and_analyze")


def test_group_session_can_place_aua_evidence_outside_the_row_directory(tmp_path):
    aua = SetupAua()
    shared = tmp_path / "group" / "aua-session"

    run(
        tmp_path,
        aua,
        two_step_model(),
        session_artifacts_dir=shared,
    )

    assert aua.named("session_start")[0]["artifacts_dir"] == str(shared.resolve())


def test_reused_group_row_loads_judge_images_from_the_shared_session_directory(tmp_path):
    pytest.importorskip("PIL")
    aua = SetupAua(evidence_colours={"fp-home": (9, 9, 9), "fp-theme": (250, 250, 250)})
    shared = tmp_path / "group" / "aua-session"
    aua._write_evidence(shared)
    model = two_step_model()

    result = run(
        tmp_path,
        aua,
        model,
        fresh_app=False,
        apk=None,
        existing_session_id="sess-group",
        finish_session=False,
        launch_app=False,
        session_artifacts_dir=shared,
        vision=True,
    )

    assert aua.named("session_start") == []
    assert result["vision"]["images_attached"] >= 1
    judged = [p for p in model.payloads if isinstance(p.get("tool_choice"), dict)][0]
    assert isinstance(judged["messages"][-1]["content"], list)


def test_headed_request_reaches_session_bootstrap(tmp_path):
    aua = SetupAua()

    run(tmp_path, aua, two_step_model(), headed=True)

    started = aua.named("session_start")
    assert len(started) == 1
    assert started[0]["headed"] is True


def test_target_capability_requirements_reach_session_bootstrap(tmp_path):
    aua = SetupAua()

    run(tmp_path, aua, two_step_model(), needs=["root"])

    started = aua.named("session_start")
    assert len(started) == 1
    assert started[0]["needs"] == ["root"]


def test_forbidden_package_guard_blocks_before_recording_or_navigation(tmp_path):
    aua = SetupAua(installed_packages={"com.example.production"})

    result = run(
        tmp_path,
        aua,
        two_step_model(),
        forbidden_packages=["com.example.production"],
        record=True,
    )

    assert "forbidden package is installed" in result["error"]
    assert aua.named("screen_record_start") == []
    assert aua.named("app_launch_and_analyze") == []
    assert aua.named("session_finish"), "the leased session is still cleaned up"


def test_package_pinned_app_lifecycle_capability_maps_to_safe_aua_calls(tmp_path):
    aua = SetupAua()
    neutral = verdict("pass", "state persisted")
    skeptical = verdict("pass", "state persisted after relaunch")
    model = FakeModel(
        controller=[
            model_call("app_force_stop", {}),
            model_call("app_relaunch_and_analyze", {}),
            model_call(
                "session_finish",
                {"outcome": "achieved", "note": "persisted"},
                call_id="native-3",
            ),
        ],
        judgements={"record_verdict": [neutral, skeptical]},
    )

    run(tmp_path, aua, model, controller_capabilities=["app-lifecycle"])

    assert aua.named("app") == [
        {"action": "stop", "package": "com.example.fictional"}
    ]
    launches = aua.named("app_launch_and_analyze")
    assert len(launches) == 2
    assert launches[-1] == {"package": "com.example.fictional"}


def test_network_capability_maps_to_reversible_aua_calls(tmp_path):
    aua = SetupAua()
    neutral = verdict("pass", "retry worked")
    skeptical = verdict("pass", "message preserved")
    model = FakeModel(
        controller=[
            model_call("network_offline", {"verify": True}),
            model_call("network_restore", {}),
            model_call(
                "session_finish",
                {"outcome": "achieved", "note": "retried"},
                call_id="native-3",
            ),
        ],
        judgements={"record_verdict": [neutral, skeptical]},
    )

    run(tmp_path, aua, model, controller_capabilities=["network"])

    assert aua.named("network_offline") == [{"verify": True}]
    assert aua.named("network_restore") == [{}]


def test_controller_is_told_which_host_owned_setup_facts_are_already_verified(tmp_path):
    aua = SetupAua()
    model = two_step_model()

    run(
        tmp_path,
        aua,
        model,
        forbidden_packages=["com.example.production"],
        prelaunch_setup_flows=[("name: environment", {"ENVIRONMENT": "Curie"})],
        flags={"simplification_experiment": "a"},
        authored_context="Curie is mandatory; a mismatch is BLOCKED.",
    )

    prompt = model.payloads[0]["messages"][1]["content"]
    assert "Harness-owned setup already completed and verified" in prompt
    assert "ENVIRONMENT=Curie" in prompt
    assert "simplification_experiment=a" in prompt
    assert "com.example.production is not installed" in prompt
    assert "Authored precondition context" in prompt
    assert "Curie is mandatory" in prompt


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


def test_optional_recording_failure_keeps_the_product_verdict_and_frame_evidence(tmp_path):
    result = run(
        tmp_path,
        SetupAua(record_stop_ok=False),
        two_step_model(),
        record=True,
        recording_required=False,
    )

    assert result["verdict"]["verdict"] == "pass"
    assert result["recording"]["stop_ok"] is False
    assert result["recording_warning"].startswith("recording cleanup failed")
    assert result["error"] is None
    assert "frame evidence" in " ".join(result["warnings"])


def test_recording_stop_retries_one_adb_read_timeout(tmp_path):
    aua = SetupAua(record_stop_results=[
        {"ok": False, "error": {"code": "device", "message": "shell failed: adb read timeout"}},
        {"ok": True, "action": "screen-record-stop"},
    ])

    result = run(tmp_path, aua, two_step_model(), record=True)

    assert result["verdict"]["verdict"] == "pass"
    assert result["recording"]["stop_ok"] is True
    assert result["recording"]["stop_attempts"] == 2
    assert len(aua.named("screen_record_stop")) == 2
    assert "retried once" in " ".join(result["warnings"])


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
    fallback_settings = VISION_SETTINGS
    result = run(
        tmp_path,
        aua,
        model,
        setup_flows=[],
        judge_model="vision/model",
        judge_request_config=VISION_SETTINGS,
        judge_fallbacks=[("backup/vision-model", fallback_settings)],
    )
    assert result["judge_model"] == "vision/model"
    assert result["judge_request_config"]["reasoning"] == VISION_SETTINGS["reasoning"]
    assert result["judge_request_config"]["provider"] == VISION_SETTINGS["provider"]
    assert result["judge_fallbacks"] == [
        {"model": "backup/vision-model", "request_config": fallback_settings}
    ]
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

def test_a_session_contract_is_given_to_aua_not_only_to_the_judge(tmp_path):
    """The checkpoints have to reach `session_start` to be able to decide anything.

    `--contract` alone is the judge's reading material. Only `contract_yaml` on the session
    makes AUA hold the run to fresh assertion proof, which is the difference between
    `oracle: aua_session_contract, verified: true` and a model's reading of the frames.
    """
    aua = SetupAua()
    run(tmp_path, aua, two_step_model(), session_contract="version: 1\ncheckpoints: []\n")

    started = aua.named("session_start")[0]
    assert started["contract_yaml"] == "version: 1\ncheckpoints: []\n"


def test_without_one_no_contract_is_invented_for_the_session(tmp_path):
    aua = SetupAua()
    run(tmp_path, aua, two_step_model())

    assert "contract_yaml" not in aua.named("session_start")[0]


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


def test_a_diverging_setup_flow_hands_the_controller_the_screen_it_reached(tmp_path):
    """A setup flow is the fast path to a precondition, not the oracle.

    It used to end the run. Measured on 2026-09-14 against a real app: a guest entry that had
    plainly succeeded returned `unverified`, because the committed flow's arrival marker had
    moved - a whole device spent to say nothing about the product. The app is still running
    and still on a screen, so the controller drives on and establishes the rest semantically.
    The adaptation is recorded, because a verdict reached this way is not the same verdict.
    """
    aua = SetupAua(flow_ok=False)
    result = run(
        tmp_path,
        aua,
        two_step_model(),
        setup_flows=[("name: environment", {"ENVIRONMENT": "Curie"})],
    )

    assert result["error"] is None
    assert any(item.get("flow_run_ok") is False for item in result["setup"])
    assert result["controller"] is not None
    assert any("setup flow 0 diverged" in warning for warning in result["warnings"]), result[
        "warnings"
    ]


def test_the_model_is_told_which_part_of_setup_did_not_finish():
    """Driving on silently is worse than not driving on.

    The model starts from a screen the harness expected to be somewhere else; without the
    note it has no way to know which half of the precondition it still owes.
    """
    from experiments.aua_controller.run_realapp import goal_prompt

    prompt = goal_prompt(
        "Switch the app theme to Light",
        [],
        ["the setup flow stopped at step 4 (wait_timeout) on screen chat__1; it still owed: "
         "wait-for 'containerDetail'"],
    )

    assert "Setup did not finish as written" in prompt
    assert "wait_timeout" in prompt
    assert "containerDetail" in prompt


def test_a_goal_with_nothing_outstanding_says_nothing_about_setup():
    from experiments.aua_controller.run_realapp import goal_prompt

    assert "Setup did not finish" not in goal_prompt("Switch the app theme to Light", [])


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

def test_vision_keeps_text_frames_independent_from_the_rendered_image_cap():
    assert judge_intermediate_frame_limit(8, vision=True) == 8
    assert judge_intermediate_frame_limit(2, vision=True) == 2
    assert judge_intermediate_frame_limit(8, vision=False) == 8

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
