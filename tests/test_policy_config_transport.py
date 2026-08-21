"""Host-only transport tests for the optional local policy configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

import android_ui_analyser.cli as cli_mod
import android_ui_analyser.daemon as daemon_mod
import android_ui_analyser.engine as engine_mod
import android_ui_analyser.mcp_server as mcp_mod
import android_ui_analyser.providers.policy.functiongemma as functiongemma_mod
from android_ui_analyser.cli import app
from android_ui_analyser.config import Config, load_config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError


def _policy_config(tmp_path: Path) -> Config:
    config = Config()
    config.policy.enabled = True
    config.policy.chain = ["functiongemma"]
    config.policy.mode = "shadow"
    config.policy.max_candidates = 4
    config.models["functiongemma"].update(
        {
            "model_path": str(tmp_path / "local-base"),
            "adapter_path": str(tmp_path / "local-adapter"),
            "max_tokens": 31,
            "model_sha256": "a" * 64,
            "adapter_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        }
    )
    config.daemon.socket = str(tmp_path / "daemon.sock")
    config.cache.dir = str(tmp_path / "cache")
    return config


def _ready_artifact_config(tmp_path: Path) -> Config:
    config = _policy_config(tmp_path)
    model = Path(config.models["functiongemma"]["model_path"])
    adapter = Path(config.models["functiongemma"]["adapter_path"])
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    weights = b"tiny fictional adapter"
    (adapter / "adapters.safetensors").write_bytes(weights)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "model": str(model)}), encoding="utf-8"
    )
    config.models["functiongemma"]["model_sha256"] = None
    config.models["functiongemma"]["adapter_sha256"] = hashlib.sha256(weights).hexdigest()
    config.models["functiongemma"]["manifest_sha256"] = None
    return config


def test_policy_candidate_bound_rejects_more_than_four() -> None:
    with pytest.raises(ValueError):
        Config.model_validate({"policy": {"max_candidates": 5}})


def test_mcp_command_passes_the_effective_explicit_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "explicit.yaml"
    config_path.write_text(
        """
policy:
  enabled: true
  mode: shadow
  max_candidates: 4
models:
  functiongemma:
    model_path: /fictional/local-base
    adapter_path: /fictional/local-adapter
    max_tokens: 31
""".lstrip(),
        encoding="utf-8",
    )
    captured: dict[str, Config] = {}
    monkeypatch.setattr(mcp_mod, "run_stdio", lambda config: captured.setdefault("config", config))

    result = CliRunner().invoke(app, ["--config", str(config_path), "mcp"])

    assert result.exit_code == 0, result.stderr
    effective = captured["config"]
    assert effective.policy.enabled is True
    assert effective.policy.mode == "shadow"
    assert effective.policy.max_candidates == 4
    assert effective.models["functiongemma"]["model_path"] == "/fictional/local-base"


def test_build_default_engine_uses_supplied_config_without_reloading(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    monkeypatch.setattr(
        mcp_mod,
        "load_config",
        lambda: (_ for _ in ()).throw(AssertionError("must not rediscover config")),
    )

    engine = mcp_mod.build_default_engine(config)

    assert engine.config is config


def test_daemon_environment_roundtrips_only_the_effective_policy_slice(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    config.memory.destructive_labels = ["archive fictional record"]
    config.models["openai"]["model"] = "must-not-be-serialized"
    monkeypatch.setenv("AUA_POLICY__OBSOLETE", "stale")
    monkeypatch.setenv("AUA_MODELS__FUNCTIONGEMMA__OBSOLETE", "stale")
    monkeypatch.setenv("INHERITED_MARKER", "preserved")

    env = daemon_mod._daemon_environment(config)

    assert env["INHERITED_MARKER"] == "preserved"
    assert "AUA_POLICY__OBSOLETE" not in env
    assert "AUA_MODELS__FUNCTIONGEMMA__OBSOLETE" not in env
    assert env["AUA_POLICY__ENABLED"] == "true"
    assert env["AUA_POLICY__CHAIN"] == "functiongemma"
    assert env["AUA_POLICY__MODE"] == "shadow"
    assert env["AUA_POLICY__MAX_CANDIDATES"] == "4"
    assert env["AUA_MODELS__FUNCTIONGEMMA__MODEL_PATH"].endswith("local-base")
    assert env["AUA_MODELS__FUNCTIONGEMMA__ADAPTER_PATH"].endswith("local-adapter")
    assert env["AUA_MODELS__FUNCTIONGEMMA__MAX_TOKENS"] == "31"
    assert env["AUA_MODELS__FUNCTIONGEMMA__MODEL_SHA256"] == "a" * 64
    assert env["AUA_MODELS__FUNCTIONGEMMA__ADAPTER_SHA256"] == "b" * 64
    assert env["AUA_MODELS__FUNCTIONGEMMA__MANIFEST_SHA256"] == "c" * 64
    assert env["AUA_MEMORY__DESTRUCTIVE_LABELS"] == "archive fictional record"
    assert "must-not-be-serialized" not in env.values()

    child = load_config(env=env, cwd=tmp_path)
    assert child.policy == config.policy
    assert child.models["functiongemma"] == config.models["functiongemma"]
    assert child.memory.destructive_labels == config.memory.destructive_labels


def test_daemon_environment_roundtrips_selective_hybrid_reviewer(tmp_path: Path) -> None:
    config = _policy_config(tmp_path)
    config.policy.chain = ["functiongemma", "gemma4"]
    config.policy.strategy = "selective_hybrid"
    config.policy.primary_reviews = 2
    config.policy.reviewer_reviews = 3
    config.policy.candidate_scope = "safe_visible"
    config.models["gemma4"].update(
        {
            "model_path": str(tmp_path / "local-gemma4"),
            "revision": "fictional-revision",
            "max_tokens": 384,
            "max_mode": "advisory",
        }
    )

    env = daemon_mod._daemon_environment(config)
    child = load_config(env=env, cwd=tmp_path)

    assert env["AUA_POLICY__CHAIN"] == "functiongemma,gemma4"
    assert env["AUA_POLICY__STRATEGY"] == "selective_hybrid"
    assert env["AUA_POLICY__PRIMARY_REVIEWS"] == "2"
    assert env["AUA_POLICY__REVIEWER_REVIEWS"] == "3"
    assert env["AUA_POLICY__CANDIDATE_SCOPE"] == "safe_visible"
    assert env["AUA_MODELS__GEMMA4__MODEL_PATH"].endswith("local-gemma4")
    assert env["AUA_MODELS__GEMMA4__REVISION"] == "fictional-revision"
    assert env["AUA_MODELS__GEMMA4__MAX_TOKENS"] == "384"
    assert env["AUA_MODELS__GEMMA4__MAX_MODE"] == "advisory"
    assert child.policy == config.policy
    assert child.models["gemma4"] == config.models["gemma4"]


def test_policy_fingerprint_is_opaque_stable_and_config_sensitive(tmp_path: Path) -> None:
    config = _policy_config(tmp_path)

    first = daemon_mod.policy_config_fingerprint(config)
    again = daemon_mod.policy_config_fingerprint(config)
    config.policy.mode = "advisory"
    changed = daemon_mod.policy_config_fingerprint(config)
    config.policy.mode = "shadow"
    config.memory.destructive_labels.append("archive fictional record")
    safety_changed = daemon_mod.policy_config_fingerprint(config)

    assert first == again
    assert len(first) == 64
    assert str(tmp_path) not in first
    assert changed != first
    assert safety_changed != first


def test_daemon_ping_reports_policy_identity_without_exposing_paths(tmp_path: Path) -> None:
    config = _policy_config(tmp_path)
    engine = SimpleNamespace(config=config)

    response = daemon_mod.dispatch(engine, {"cmd": "ping", "args": {}})

    assert response["ok"] is True
    result = response["result"]
    assert result["policy_fingerprint"] == daemon_mod.policy_config_fingerprint(config)
    assert str(tmp_path) not in str(result)


def test_daemon_revalidates_policy_fingerprint_before_dispatch(tmp_path: Path) -> None:
    config = _policy_config(tmp_path)
    calls: list[str] = []
    engine = SimpleNamespace(config=config, analyze=lambda **_kwargs: calls.append("analyze"))

    response = daemon_mod.dispatch(
        engine,
        {"cmd": "analyze", "args": {}, "policy_fingerprint": "0" * 64},
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "policy_config_mismatch"
    assert calls == []


def test_route_refuses_a_policy_mismatched_daemon_without_cold_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    config.device.serial = "fictional-5554"
    config.perf.auto_daemon = False
    mutations: list[str] = []
    engine = SimpleNamespace(
        config=config,
        _lease_serial="fictional-5554",
        _lease_owner=None,
        _lease_owner_resolved=None,
        tap=lambda **_kwargs: mutations.append("tap"),
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _config: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _config: daemon_mod._aua_version())
    monkeypatch.setattr(daemon_mod, "running_policy_fingerprint", lambda _config: "stale")
    monkeypatch.setattr(cli_mod, "_replace_policy_mismatched_daemon", lambda *_args: False)

    with pytest.raises(UsageError) as caught:
        cli_mod._route(engine, "tap", element_id=1)

    assert getattr(caught.value, "code", None) == "policy_config_mismatch"
    assert mutations == []


def test_policy_off_caller_refuses_a_daemon_started_with_policy_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    config.policy.enabled = False
    config.policy.mode = "off"
    config.device.serial = "fictional-5554"
    config.perf.auto_daemon = False
    mutations: list[str] = []
    engine = SimpleNamespace(
        config=config,
        _lease_serial="fictional-5554",
        _lease_owner=None,
        _lease_owner_resolved=None,
        tap=lambda **_kwargs: mutations.append("tap"),
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _config: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _config: daemon_mod._aua_version())
    monkeypatch.setattr(daemon_mod, "running_policy_fingerprint", lambda _config: "enabled")
    monkeypatch.setattr(cli_mod, "_replace_policy_mismatched_daemon", lambda *_args: False)

    with pytest.raises(UsageError) as caught:
        cli_mod._route(engine, "tap", element_id=1)

    assert getattr(caught.value, "code", None) == "policy_config_mismatch"
    assert mutations == []


def test_policy_mismatch_restart_is_refused_while_capture_is_live(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    calls: list[str] = []
    daemon = SimpleNamespace(
        stop=lambda _config: calls.append("stop"),
        start=lambda _config, serial=None: calls.append(f"start:{serial}"),
        running_policy_fingerprint=lambda _config: "fresh",
    )
    monkeypatch.setattr(cli_mod, "_capture_session_live", lambda *_args: True)

    replaced = cli_mod._replace_policy_mismatched_daemon(daemon, config, "fresh")

    assert replaced is False
    assert calls == []


def test_policy_status_reports_a_stale_warm_daemon(tmp_path: Path, monkeypatch) -> None:
    config = _policy_config(tmp_path)
    monkeypatch.setattr(daemon_mod, "is_running", lambda _config: True)
    monkeypatch.setattr(daemon_mod, "running_policy_fingerprint", lambda _config: "stale")

    status = daemon_mod.policy_runtime_status(config)

    assert status["daemon"]["running"] is True
    assert status["daemon"]["compatible"] is False
    assert "restart required" in status["daemon"]["reason"]


def test_route_carries_matching_policy_identity_on_the_real_request(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    config.device.serial = "fictional-5554"
    config.perf.auto_daemon = False
    expected = daemon_mod.policy_config_fingerprint(config)
    client_options: dict[str, Any] = {}

    class Client:
        def __init__(self, _socket: str, **kwargs: Any) -> None:
            client_options.update(kwargs)

        def call(self, _command: str, **_kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "result": {"routed": True}}

    engine = SimpleNamespace(
        config=config,
        _lease_serial="fictional-5554",
        _lease_owner=None,
        _lease_owner_resolved=None,
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _config: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _config: daemon_mod._aua_version())
    monkeypatch.setattr(daemon_mod, "running_policy_fingerprint", lambda _config: expected)
    monkeypatch.setattr(daemon_mod, "DaemonClient", Client)

    result = cli_mod._route(engine, "analyze")

    assert result == {"routed": True}
    assert client_options["policy_fingerprint"] == expected
    assert client_options["decorate_response"] is True


def test_warm_daemon_decorates_once_before_the_client_returns(tmp_path: Path, monkeypatch) -> None:
    config = _policy_config(tmp_path)
    config.device.serial = "fictional-5554"
    config.perf.auto_daemon = False
    expected = daemon_mod.policy_config_fingerprint(config)
    warm_result = {"routed": True, "goal_progress": {"policy": {"status": "selected"}}}

    class Client:
        def __init__(self, _socket: str, **kwargs: Any) -> None:
            assert kwargs["decorate_response"] is True

        def call(self, _command: str, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "result": warm_result,
                "response_decorated": True,
            }

    engine = SimpleNamespace(
        config=config,
        _lease_serial="fictional-5554",
        _lease_owner=None,
        _lease_owner_resolved=None,
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _config: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _config: daemon_mod._aua_version())
    monkeypatch.setattr(daemon_mod, "running_policy_fingerprint", lambda _config: expected)
    monkeypatch.setattr(daemon_mod, "DaemonClient", Client)
    monkeypatch.setattr(
        "android_ui_analyser.coaching.decorate_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a marked warm-daemon response must not be decorated in the CLI")
        ),
    )

    assert cli_mod._route(engine, "analyze") is warm_result


def test_warm_daemon_detail_is_revised_at_cli_emit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from android_ui_analyser import journal
    from android_ui_analyser.schema import OutputFormat

    config = _policy_config(tmp_path)
    config.device.serial = "fictional-5554"
    config.perf.auto_daemon = False
    expected = daemon_mod.policy_config_fingerprint(config)
    warm_result = {"ok": True, "agent_visible": "daemon response"}

    class Client:
        def __init__(self, _socket: str, **_kwargs: Any) -> None:
            pass

        def call(self, _command: str, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "result": warm_result,
                "response_decorated": True,
                "journal_detail_id": "daemon-detail-id",
            }

    engine = SimpleNamespace(
        config=config,
        _lease_serial="fictional-5554",
        _lease_owner=None,
        _lease_owner_resolved=None,
    )
    revised: dict[str, Any] = {}
    monkeypatch.setattr(daemon_mod, "is_running", lambda _config: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _config: daemon_mod._aua_version())
    monkeypatch.setattr(daemon_mod, "running_policy_fingerprint", lambda _config: expected)
    monkeypatch.setattr(daemon_mod, "DaemonClient", Client)
    monkeypatch.setattr(
        journal,
        "record_emitted_response",
        lambda **kwargs: revised.update(kwargs) or True,
    )
    monkeypatch.setattr(cli_mod, "_INVOCATION_ID", "daemon-visible-response")
    monkeypatch.setattr(cli_mod, "_ENGINE", None)
    monkeypatch.setattr(cli_mod, "_UNTIL", None)
    monkeypatch.setattr(cli_mod, "_OBSERVATION_VIEW", None)
    monkeypatch.setattr(cli_mod, "_CLI_OUTPUT_FORMAT", OutputFormat.json)
    monkeypatch.setattr(cli_mod, "_CLI_OBSERVE_FIELDS_SPEC", None)
    monkeypatch.setattr(cli_mod, "_ANNOTATION_WARNINGS", [])
    cli_mod._CLI_JOURNAL_CONTEXTS.clear()

    result = cli_mod._route(engine, "analyze")
    cli_mod._emit(result, OutputFormat.json)

    assert json.loads(capsys.readouterr().out) == warm_result
    assert revised["detail_id"] == "daemon-detail-id"
    assert revised["invocation_id"] == "daemon-visible-response"
    assert revised["cmd"] == "analyze"
    assert revised["result"] == warm_result


def test_in_process_route_journals_the_decorated_agent_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_ui_analyser import coaching, journal

    config = Config()
    config.daemon.enabled = False
    config.cache.dir = str(tmp_path)
    config.device.serial = "fictional-5554"
    raw = {"ok": True, "raw": True}
    decorated = {"ok": True, "raw": True, "agent_visible": "decorated"}
    engine = SimpleNamespace(
        config=config,
        device=SimpleNamespace(serial="fictional-5554"),
        _lease_serial="fictional-5554",
        _lease_owner_resolved="agent-a",
        analyze=lambda: raw,
    )
    captured: dict[str, Any] = {}

    def decorate(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["current_recorded"] is False
        return decorated

    monkeypatch.setattr(cli_mod, "_warm", lambda _engine: None)
    monkeypatch.setattr(coaching, "decorate_result", decorate)
    monkeypatch.setattr(journal, "record", lambda **kwargs: captured.update(kwargs))

    result = cli_mod._route(engine, "analyze")

    assert result is decorated
    assert captured["result"] is decorated


def test_cli_detail_matches_projected_until_response_emitted_to_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from android_ui_analyser import coaching, journal
    from android_ui_analyser.projection import Projection
    from android_ui_analyser.schema import (
        ActionResult,
        AnalyzeResult,
        Element,
        Meta,
        OutputFormat,
        Screen,
    )

    def observation(text: str) -> AnalyzeResult:
        return AnalyzeResult(
            screen=Screen(
                width=1080,
                height=2400,
                package="com.example",
                source="hierarchy",
            ),
            elements=[
                Element(
                    id=7,
                    type="Button",
                    text=text,
                    resource_id="com.example:id/continue",
                    clickable=True,
                    bounds=[0, 100, 500, 200],
                    center=[250, 150],
                    window="app",
                )
            ],
            meta=Meta(duration_ms=3, tier_used="hierarchy", path="hierarchy"),
        )

    early = ActionResult(
        ok=True,
        action="tap",
        id=7,
        observation=observation("Early"),
        observation_present=True,
    )
    arrived = ActionResult(
        ok=True,
        action="await-predicate",
        observation=observation("Arrived"),
        observation_present=True,
        await_outcome="satisfied",
        await_terms=[{"term": "text:Arrived", "satisfied": True}],
        elapsed_ms=25,
    )
    config = Config()
    config.daemon.enabled = False
    config.cache.dir = str(tmp_path)
    config.device.serial = "fictional-5554"
    engine = SimpleNamespace(
        config=config,
        device=SimpleNamespace(serial="fictional-5554"),
        _lease_serial="fictional-5554",
        _lease_owner_resolved="agent-a",
        tap=lambda **_kwargs: early,
        await_predicate=lambda **_kwargs: arrived,
    )

    monkeypatch.setattr(cli_mod, "_warm", lambda _engine: None)
    monkeypatch.setattr(
        coaching,
        "decorate_result",
        lambda _engine, _cmd, result, **_kwargs: result,
    )
    monkeypatch.setattr(cli_mod, "_INVOCATION_ID", "cli-visible-response")
    monkeypatch.setattr(cli_mod, "_ENGINE", engine)
    monkeypatch.setattr(cli_mod, "_UNTIL", ("text:Arrived", 1_000, 50))
    monkeypatch.setattr(cli_mod, "_CLI_OUTPUT_FORMAT", OutputFormat.json)
    monkeypatch.setattr(cli_mod, "_CLI_OBSERVE_FIELDS_SPEC", "id,text")
    monkeypatch.setattr(
        cli_mod,
        "_OBSERVATION_VIEW",
        Projection.for_observation("id,text"),
    )
    monkeypatch.setattr(cli_mod, "_ANNOTATION_WARNINGS", [])
    cli_mod._CLI_JOURNAL_CONTEXTS.clear()

    result = cli_mod._route(engine, "tap", element_id=7, observe=True)
    cli_mod._emit(result, OutputFormat.json)

    emitted = json.loads(capsys.readouterr().out)
    event = next(
        row
        for row in journal.read_since(tmp_path, "fictional-5554", limit=10)
        if row["cmd"] == "tap"
    )
    detail = journal.read_detail(tmp_path, "fictional-5554", event["detail_id"])
    assert detail is not None
    assert detail["response"]["result"] == emitted
    assert detail["request"]["client"]["format"] == "json"
    assert detail["request"]["client"]["until"] == "text:Arrived"
    assert detail["request"]["client"]["until_timeout_ms"] == 1_000
    assert detail["request"]["client"]["until_poll_ms"] == 50
    assert detail["request"]["client"]["observe_fields"] == "id,text"
    assert detail["request"]["client"]["observation_projection"]["fields"] == [
        "id",
        "text",
    ]
    assert emitted["await_outcome"] == "satisfied"
    # Ids are published as stable ids, so the projected row names the element that way too.
    assert [e["text"] for e in emitted["observation"]["elements"]] == ["Arrived"]
    assert all(isinstance(e["id"], str) for e in emitted["observation"]["elements"])


def test_private_cli_until_row_inherits_input_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from android_ui_analyser import coaching, journal
    from android_ui_analyser.projection import Projection
    from android_ui_analyser.schema import (
        ActionResult,
        AnalyzeResult,
        Element,
        Meta,
        OutputFormat,
        Screen,
    )

    private_input = "correct horse, battery staple"

    def observation(parts: list[str]) -> AnalyzeResult:
        return AnalyzeResult(
            screen=Screen(
                width=1080,
                height=2400,
                package="com.example",
                source="hierarchy",
            ),
            elements=[
                Element(
                    id=index,
                    type="TextView",
                    text=text,
                    bounds=[0, 0, 10, 10],
                    center=[5, 5],
                )
                for index, text in enumerate(parts)
            ],
            meta=Meta(duration_ms=3, tier_used="hierarchy", path="hierarchy"),
        )

    early = ActionResult(
        ok=True,
        action="input",
        detail=private_input,
        observation=observation(["correct horse", "battery staple"]),
        observation_present=True,
    )
    arrived = ActionResult(
        ok=True,
        action="await-predicate",
        observation=observation(["correct horse", "battery staple"]),
        observation_present=True,
        await_outcome="satisfied",
    )
    config = Config()
    config.daemon.enabled = False
    config.cache.dir = str(tmp_path)
    config.device.serial = "fictional-5554"
    engine = SimpleNamespace(
        config=config,
        device=SimpleNamespace(serial="fictional-5554"),
        _lease_serial="fictional-5554",
        _lease_owner_resolved="agent-a",
        input_text=lambda **_kwargs: early,
        await_predicate=lambda **_kwargs: arrived,
    )

    monkeypatch.setattr(cli_mod, "_warm", lambda _engine: None)
    monkeypatch.setattr(
        coaching,
        "decorate_result",
        lambda _engine, _cmd, result, **_kwargs: result,
    )
    monkeypatch.setattr(cli_mod, "_INVOCATION_ID", "private-until-response")
    monkeypatch.setattr(cli_mod, "_ENGINE", engine)
    monkeypatch.setattr(
        cli_mod,
        "_UNTIL",
        ("text:correct horse\\, battery staple", 1_000, 50),
    )
    monkeypatch.setattr(cli_mod, "_CLI_OUTPUT_FORMAT", OutputFormat.json)
    monkeypatch.setattr(cli_mod, "_CLI_OBSERVE_FIELDS_SPEC", "id,text")
    monkeypatch.setattr(
        cli_mod,
        "_OBSERVATION_VIEW",
        Projection.for_observation("id,text"),
    )
    monkeypatch.setattr(cli_mod, "_ANNOTATION_WARNINGS", [])
    cli_mod._CLI_JOURNAL_CONTEXTS.clear()

    result = cli_mod._route(
        engine,
        "input_text",
        element_id=7,
        text=private_input,
        observe=True,
    )
    cli_mod._emit(result, OutputFormat.json)
    capsys.readouterr()

    events = journal.read_since(tmp_path, "fictional-5554", limit=10)
    await_event = next(row for row in events if row["cmd"] == "await_predicate")
    await_detail = journal.read_detail(
        tmp_path,
        "fictional-5554",
        await_event["detail_id"],
    )
    input_event = next(row for row in events if row["cmd"] == "input")
    input_detail = journal.read_detail(
        tmp_path,
        "fictional-5554",
        input_event["detail_id"],
    )
    assert await_detail is not None
    assert input_detail is not None
    serialized = json.dumps([await_event, await_detail, input_event, input_detail])
    assert private_input not in serialized
    assert "correct horse" not in serialized
    assert "battery staple" not in serialized
    assert await_event["args"]["predicate"] == "<redacted post-input text>"
    assert await_detail["request"]["args"]["predicate"] == (
        "<redacted post-input text>"
    )
    assert input_detail["request"]["client"]["until"] == (
        "<redacted post-input text>"
    )


def test_daemon_response_decoration_uses_the_warm_engine(monkeypatch) -> None:
    engine = SimpleNamespace(config=Config())
    calls: list[tuple[str, Any]] = []

    def decorate(warm_engine: Any, cmd: str, result: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(("decorate", (warm_engine, cmd, result, kwargs)))
        return {**result, "warm_decorated": True}

    monkeypatch.setattr("android_ui_analyser.coaching.decorate_result", decorate)
    request = {
        "cmd": "tap",
        "args": {"element_id": 7},
        "decorate_response": True,
    }
    response = daemon_mod._decorate_requested_response(
        engine,
        request,
        {"ok": True, "result": {"action": "tap"}},
    )

    assert response == {
        "ok": True,
        "result": {"action": "tap", "warm_decorated": True},
        "response_decorated": True,
    }
    assert calls == [
        (
            "decorate",
            (
                engine,
                "tap",
                {"action": "tap"},
                {"args": {"element_id": 7}, "current_recorded": False},
            ),
        )
    ]


def test_daemon_decorates_before_journaling_and_serializing(monkeypatch) -> None:
    events: list[tuple[str, Any]] = []
    sent: list[dict[str, Any]] = []
    request = {"cmd": "tap", "args": {"element_id": 7}, "decorate_response": True}

    class Connection:
        def __init__(self) -> None:
            self._chunks = iter([(json.dumps(request) + "\n").encode(), b""])

        def settimeout(self, _value: float) -> None:
            pass

        def recv(self, _size: int) -> bytes:
            return next(self._chunks)

        def sendall(self, value: bytes) -> None:
            sent.append(json.loads(value))

    def dispatch(_engine: Any, _request: dict[str, Any]) -> dict[str, Any]:
        events.append(("dispatch", None))
        return {"ok": True, "result": {"action": "tap"}}

    def decorate(_engine: Any, _cmd: str, result: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append(("decorate", None))
        return {**result, "warm_decorated": True}

    def journal(
        _engine: Any,
        _request: dict[str, Any],
        response: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        events.append(("journal", response))

    monkeypatch.setattr(daemon_mod, "dispatch", dispatch)
    monkeypatch.setattr("android_ui_analyser.coaching.decorate_result", decorate)
    monkeypatch.setattr(daemon_mod, "_journal_dispatch", journal)

    daemon_mod._handle_connection(SimpleNamespace(config=Config()), Connection())

    assert [name for name, _value in events] == ["dispatch", "decorate", "journal"]
    decorated = {
        "ok": True,
        "result": {"action": "tap", "warm_decorated": True},
        "response_decorated": True,
    }
    assert events[-1][1] == decorated
    assert sent == [decorated]


def test_daemon_client_serializes_the_decoration_request(monkeypatch) -> None:
    sent: list[dict[str, Any]] = []

    class Socket:
        def settimeout(self, _value: float) -> None:
            pass

        def connect(self, _path: str) -> None:
            pass

        def sendall(self, value: bytes) -> None:
            sent.append(json.loads(value))

        def recv(self, _size: int) -> bytes:
            return b'{"ok":true,"result":{},"response_decorated":true}\n'

        def close(self) -> None:
            pass

    monkeypatch.setattr(daemon_mod.socket, "socket", lambda *_args, **_kwargs: Socket())

    response = daemon_mod.DaemonClient(
        "/fictional/daemon.sock",
        decorate_response=True,
        journal_privacy_cmd="input",
    ).call("analyze")

    assert response["response_decorated"] is True
    assert sent == [
        {
            "cmd": "analyze",
            "args": {},
            "decorate_response": True,
            "journal_privacy_cmd": "input",
        }
    ]


def test_cli_and_mcp_policy_status_are_host_only_and_have_identical_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    config = _ready_artifact_config(tmp_path)
    model_settings = config.models["functiongemma"]
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "policy:",
                "  enabled: true",
                "  mode: shadow",
                "  chain: [functiongemma]",
                "  max_candidates: 4",
                "models:",
                "  functiongemma:",
                f"    model_path: {model_settings['model_path']}",
                f"    adapter_path: {model_settings['adapter_path']}",
                f"    max_tokens: {model_settings['max_tokens']}",
                "    model_sha256: null",
                f"    adapter_sha256: {model_settings['adapter_sha256']}",
                "daemon:",
                f"  socket: {config.daemon.socket}",
                "cache:",
                f"  dir: {config.cache.dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(functiongemma_mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        engine_mod,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("policy status must not touch a device")
        ),
    )

    cli_result = CliRunner().invoke(app, ["--config", str(config_path), "policy", "status"])
    assert cli_result.exit_code == 0, cli_result.stderr
    cli_payload = json.loads(cli_result.stdout)

    server = mcp_mod.build_server(Engine(config))

    async def call_status() -> tuple[set[str], dict[str, Any]]:
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            response = await client.call_tool("policy_status", {})
            text = next(block.text for block in response.content if block.type == "text")
            return {tool.name for tool in listed.tools}, json.loads(text)

    tools, mcp_payload = anyio.run(call_status)

    assert "policy_status" in tools
    assert cli_payload == mcp_payload
    provider = cli_payload["providers"][0]
    assert provider["runtime"]["ready"] is False
    assert provider["artifacts"]["ready"] is True
    assert provider["provenance"]["adapter_hash_verified"] is True
    assert provider["loaded"] is False
    assert cli_payload["daemon"]["running"] is False


def test_daemon_start_passes_the_sanitized_effective_environment(
    tmp_path: Path, monkeypatch
) -> None:
    config = _policy_config(tmp_path)
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(daemon_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_mod, "reap", lambda _config: None)
    alive = iter([False, True])
    monkeypatch.setattr(daemon_mod, "_socket_alive", lambda _socket: next(alive, True))

    result = daemon_mod.start(config)

    assert result["status"] == "started"
    child_env = captured["env"]
    assert child_env["AUA_POLICY__MODE"] == "shadow"
    assert child_env["AUA_MODELS__FUNCTIONGEMMA__ADAPTER_SHA256"] == "b" * 64
    assert child_env["AUA_MODELS__FUNCTIONGEMMA__MANIFEST_SHA256"] == "c" * 64
    assert "--socket" in captured["command"]
