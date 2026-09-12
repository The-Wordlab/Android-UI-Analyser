"""Run context survives separate CLI calls without extra device work or scope fallback."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import agent_run, leases
from android_ui_analyser.cli import app
from android_ui_analyser.config import Config, load_config
from android_ui_analyser.errors import UsageError
from android_ui_analyser.session import create_session_state, finish_session_state


@pytest.fixture
def run_context(tmp_path, monkeypatch):
    monkeypatch.setattr(leases, "_proc_started", lambda pid: "start" if pid == 123 else "new")
    monkeypatch.setattr(
        leases,
        "resolve_owner",
        lambda explicit=None: leases.LeaseOwner(explicit or "worker", pid=123, started="start"),
    )
    cfg = Config()
    cfg.cache.dir = str(tmp_path / "original-cache")
    cfg.lease.registry_dir = str(tmp_path / "shared-leases")
    cfg.memory.dir = str(tmp_path / "memory")
    cfg.daemon.enabled = False
    path = tmp_path / "run.json"
    context = agent_run.create_run(path, cfg, needs=["root"])
    return path, context


def bind_session(path, context):
    state = create_session_state(
        context.cache_dir,
        goal="Verify Settings opens",
        serial="emulator-5554",
        owner=context.owner,
        platform=context.platform,
        recommended_kind="ui",
        recommended_cli="aua analyze",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    context.session_id = state.session_id
    context.target_id = state.serial
    path.write_text(context.model_dump_json())
    return state


def test_init_is_private_isolated_and_does_not_snapshot_secrets(run_context, monkeypatch):
    path, context = run_context
    original = path.read_bytes()
    monkeypatch.setenv("PROVIDER_API_KEY", "private-test-value")
    assert path.stat().st_mode & 0o077 == 0
    assert context.config_path.stat().st_mode & 0o077 == 0
    assert context.config_path.parent.stat().st_mode & 0o077 == 0
    config = json.loads(context.config_path.read_text())
    assert config["lease"]["registry_dir"] == str(path.parent / "shared-leases")
    assert config["cache"]["dir"] != str(path.parent / "original-cache")
    assert "private-test-value" not in path.read_text() + context.config_path.read_text()
    with pytest.raises(UsageError, match="already exists"):
        agent_run.create_run(path, Config())
    assert path.read_bytes() == original


def test_start_freezes_run_config_then_followup_reuses_identity_once(run_context, monkeypatch):
    path, context = run_context
    calls = []
    frame = {
        "screen": {"package": "com.example.demo"},
        "elements": [{"id": "el:one", "text": "Settings"}],
        "meta": {"raw_image": "/already/returned.png"},
    }

    def invoke(args, *, env, cwd):
        calls.append((args, env, cwd))
        if "start" in args:
            state = create_session_state(
                context.cache_dir,
                goal="Verify Settings opens",
                serial="emulator-5554",
                owner=context.owner,
                recommended_kind="ui",
                recommended_cli="aua analyze",
                network_backup_preexisting=False,
                network_profile_preexisting=False,
            )
            data = {
                "session_id": state.session_id,
                "serial": state.serial,
                "owner": state.owner,
                "observation": frame,
            }
        else:
            data = {"ok": True, "action": "tap", "observation": frame}
        return subprocess.CompletedProcess(args, 0, json.dumps(data), "")

    monkeypatch.setattr(agent_run, "_invoke_cli", invoke)
    first, code = agent_run.execute_run(
        path, ["session", "start", "--goal", "Verify Settings opens"]
    )
    assert code == 0 and first["ok"]
    assert first["context"]["session_id"]
    assert first["context"]["target_id"] == "emulator-5554"
    assert "--serial" not in calls[0][0]
    monkeypatch.setenv("AUA_OWNER", "foreign")
    monkeypatch.setenv("AUA_SERIAL", "emulator-9998")
    monkeypatch.setenv("AUA_CACHE__DIR", str(path.parent / "wrong"))
    monkeypatch.setenv("AUA_PROFILE", "unknown")
    monkeypatch.setenv("AUA_DAEMON_SOCKET", "foreign.sock")
    monkeypatch.setenv("PROVIDER_API_KEY", "ephemeral-key")
    monkeypatch.setenv("AUA_PROVIDER_TOKEN", "prefixed-ephemeral-key")
    env_before = dict(os.environ)
    second, code = agent_run.execute_run(path, ["tap-and-analyze", "el:one"])
    assert code == 0 and second["ok"]
    assert second["context"]["session_id"] == first["context"]["session_id"]
    assert second["observation"] == frame
    assert second["observation_contract"]["image_path"] == "/already/returned.png"
    assert len(calls) == 2
    args, env, cwd = calls[1]
    assert args[args.index("--serial") + 1] == "emulator-5554"
    assert env["AUA_OWNER"] == context.owner
    assert env["AUA_CACHE__DIR"] == str(context.cache_dir)
    assert env["AUA_WORKER_SCOPE"] == context.run_id
    assert "AUA_DAEMON_SOCKET" not in env and "AUA_PROFILE" not in env
    assert env["PROVIDER_API_KEY"] == "ephemeral-key"
    assert env["AUA_PROVIDER_TOKEN"] == "prefixed-ephemeral-key"
    assert cwd == context.cwd and dict(os.environ) == env_before
    loaded = load_config(explicit_path=context.config_path, env=env, cwd=cwd)
    assert loaded.lease.registry_dir == str(path.parent / "shared-leases")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--owner", "foreign", "analyze"],
        ["analyze", "--serial", "emulator-9998"],
        ["--profile", "different", "analyze"],
        ["--config", "other.yaml", "analyze"],
        ["--platform", "other", "analyze"],
        ["--no-lease", "analyze"],
        ["--format", "compact", "analyze"],
        ["session", "progress", "--session-id", "wrong"],
        ["emulator", "stop", "--serial", "emulator-9998"],
        ["session", "start", "--goal", "another"],
        ["lease", "release"],
        ["mcp"],
        ["virtual-target", "stop", "--target-id", "foreign"],
        ["virtual-target", "stop", "--target-id", "emulator-5554", "--owner", "other"],
        ["virtual-target", "stop", "--all"],
        ["emulator", "stop", "--all"],
        ["emulator", "stop", "--avd", "other"],
        ["emulator", "stop", "--mine"],
        ["--needs", "play", "analyze"],
    ],
)
def test_context_changes_refuse_before_dispatch(run_context, monkeypatch, arguments):
    path, context = run_context
    bind_session(path, context)
    monkeypatch.setattr(agent_run, "_invoke_cli", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(UsageError):
        agent_run.execute_run(path, arguments)


def test_missing_or_replaced_goal_never_falls_back(run_context, monkeypatch):
    path, context = run_context
    bind_session(path, context)
    create_session_state(
        context.cache_dir,
        goal="Verify Profile opens",
        serial=context.target_id,
        owner=context.owner,
        recommended_kind="ui",
        recommended_cli="aua analyze",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    monkeypatch.setattr(agent_run, "_invoke_cli", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(UsageError, match="active goal changed"):
        agent_run.execute_run(path, ["analyze"])
    context.session_id = "missing-id"
    path.write_text(context.model_dump_json())
    with pytest.raises(UsageError, match="missing or mismatched"):
        agent_run.execute_run(path, ["session", "finish"])


def test_another_process_cannot_resume_saved_owner(run_context, monkeypatch):
    path, context = run_context
    monkeypatch.setattr(
        leases,
        "resolve_owner",
        lambda explicit=None: leases.LeaseOwner(context.owner, pid=456, started="new"),
    )
    monkeypatch.setattr(agent_run, "_invoke_cli", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(UsageError, match="different caller"):
        agent_run.execute_run(path, ["analyze"])


def test_finished_goal_allows_review_but_no_new_actions(run_context, monkeypatch):
    path, context = run_context
    state = bind_session(path, context)
    finish_session_state(context.cache_dir, state)
    calls = []

    def invoke(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"ok":true}', "")

    monkeypatch.setattr(agent_run, "_invoke_cli", invoke)
    with pytest.raises(UsageError, match="goal has ended"):
        agent_run.execute_run(path, ["tap-and-analyze", "el:old"])
    assert agent_run.execute_run(path, ["session", "review"])[0]["ok"]
    assert len(calls) == 1


def test_capture_id_and_error_observation_cannot_replace_goal(run_context, monkeypatch):
    path, context = run_context
    bind_session(path, context)
    calls = []
    replies = [
        (0, '{"ok":true,"session_id":"capture-buffer-id"}', ""),
        (
            6,
            "",
            json.dumps(
                {
                    "error": {
                        "code": "element_not_found",
                        "message": "No action was sent.",
                        "observation": {
                            "screen": {},
                            "elements": [{"id": "el:returned"}],
                            "meta": {},
                        },
                    }
                }
            ),
        ),
        (0, '{"ok":true,"detail":"artifact read"}', ""),
    ]

    def invoke(args, **kwargs):
        calls.append(args)
        code, out, err = replies.pop(0)
        return subprocess.CompletedProcess(args, code, out, err)

    monkeypatch.setattr(agent_run, "_invoke_cli", invoke)
    first, _ = agent_run.execute_run(path, ["capture", "status"])
    assert first["result"]["session_id"] == "capture-buffer-id"
    second, code = agent_run.execute_run(path, ["tap-and-analyze", "el:previous"])
    assert code == 6 and second["ok"] is False
    assert second["error"]["code"] == "element_not_found"
    assert second["observation"]["elements"][0]["id"] == "el:returned"
    third, _ = agent_run.execute_run(path, ["capture", "status"])
    assert third["observation"] is None
    assert agent_run.load_run(path).session_id == context.session_id
    assert len(calls) == 3


def test_cli_entrypoints_normalize_context_errors(run_context):
    path, _ = run_context
    result = CliRunner().invoke(app, ["run", "exec", str(path), "--", "analyze"])
    assert result.exit_code == 2, result.stdout
    data = json.loads(result.stdout)
    assert data["ok"] is False and data["error"]["code"] == "run_session_missing"
    assert data["observation"] is None


def test_missing_context_does_not_start_a_new_run(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_run, "_invoke_cli", lambda *a, **k: pytest.fail("must not dispatch"))
    path = tmp_path / "missing.json"
    result = CliRunner().invoke(
        app, ["run", "exec", str(path), "--", "session", "start", "--goal", "Verify Settings opens"]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "run_context_invalid"
    assert not path.exists()


def test_snapshot_modification_refuses_before_dispatch(run_context, monkeypatch):
    path, context = run_context
    snapshot = json.loads(context.config_path.read_text())
    snapshot["device"]["platform"] = "other"
    context.config_path.write_text(json.dumps(snapshot))
    monkeypatch.setattr(agent_run, "_invoke_cli", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(UsageError, match="configuration is missing or changed"):
        agent_run.execute_run(path, ["session", "start", "--goal", "Verify Settings opens"])


def test_cli_init_freezes_requested_output_defaults_without_an_engine(tmp_path, monkeypatch):
    from android_ui_analyser.engine import Engine

    monkeypatch.setattr(
        Engine, "__init__", lambda *a, **k: pytest.fail("init must not build an engine")
    )
    monkeypatch.setattr(leases, "_proc_started", lambda pid: "start")
    monkeypatch.setattr(
        leases,
        "resolve_owner",
        lambda explicit=None: leases.LeaseOwner(explicit or "worker", pid=123, started="start"),
    )
    config = Config()
    config.lease.registry_dir = str(tmp_path / "leases")
    config_path = tmp_path / "source.json"
    config_path.write_text(config.model_dump_json())
    path = tmp_path / "run.json"
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "--observe-fields",
            "all",
            "--observe-meta",
            "all",
            "--log-level",
            "debug",
            "run",
            "init",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True
    context = agent_run.load_run(path)
    saved = json.loads(context.config_path.read_text())
    assert saved["output"]["observation_fields"] == "all"
    assert saved["output"]["observation_meta"] == "all"
    assert saved["log_level"] == "debug"
    calls = []

    def invoke(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 2, "", '{"error":{"code":"usage","message":"test refusal"}}'
        )

    monkeypatch.setattr(agent_run, "_invoke_cli", invoke)
    agent_run.execute_run(path, ["session", "start", "--goal", "Verify Settings opens"])
    assert calls[0][calls[0].index("--log-level") + 1] == "debug"


def test_actual_child_cli_failure_is_one_attempt_without_device_access(run_context, monkeypatch):
    from android_ui_analyser import journal

    path, context = run_context
    # An actual second Python process exercises argv, frozen config, stderr and journal.
    # The invalid flag is rejected by Click before any engine/device action.
    monkeypatch.setenv("PYTHONPATH", str(Path(agent_run.__file__).resolve().parents[1]))
    before = path.read_bytes()
    response = CliRunner().invoke(
        app,
        [
            "run",
            "exec",
            str(path),
            "--",
            "session",
            "start",
            "--goal",
            "Verify Settings opens",
            "--unknown-example-flag",
        ],
    )
    assert response.exit_code == 2, response.stdout
    data = json.loads(response.stdout)
    assert data["ok"] is False and data["observation"] is None
    assert "unknown-example-flag" in data["error"].get("diagnostic", "")
    assert path.read_bytes() == before
    events = journal.read_since(context.cache_dir, None, limit=10)
    assert len(events) == 1
    assert events[0]["ok"] is False


@pytest.mark.parametrize(
    "payload,command",
    [
        ({"ok": True, "session_id": "other-goal"}, ["session", "progress"]),
        ({"ok": True, "owner": "other-owner"}, ["session", "review"]),
        (
            {
                "ok": True,
                "action": "tap",
                "observation": {
                    "screen": {},
                    "elements": [{"id": "el:foreign"}],
                    "meta": {"device_serial": "emulator-9998"},
                },
            },
            ["tap-and-analyze", "el:one"],
        ),
    ],
)
def test_mismatched_response_is_not_accepted_as_this_run(
    run_context, monkeypatch, payload, command
):
    path, context = run_context
    bind_session(path, context)
    before = path.read_bytes()
    calls = []

    def invoke(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(agent_run, "_invoke_cli", invoke)
    result, code = agent_run.execute_run(path, command)
    assert code == 1 and result["ok"] is False
    assert result["error"]["code"] == "run_context_mismatch"
    assert result["observation_contract"]["reusable"] is False
    assert path.read_bytes() == before and len(calls) == 1


def test_failed_context_save_preserves_started_identity_without_replaying(run_context, monkeypatch):
    path, context = run_context
    state = create_session_state(
        context.cache_dir,
        goal="Verify Settings opens",
        serial="emulator-5554",
        owner=context.owner,
        recommended_kind="ui",
        recommended_cli="aua analyze",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    calls = []

    def invoke(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {"session_id": state.session_id, "serial": state.serial, "owner": state.owner}
            ),
            "",
        )

    monkeypatch.setattr(agent_run, "_invoke_cli", invoke)
    monkeypatch.setattr(
        agent_run, "_write_private", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    result, code = agent_run.execute_run(
        path, ["session", "start", "--goal", "Verify Settings opens"]
    )
    assert code == 1 and result["error"]["code"] == "run_context_persist_failed"
    assert result["context"]["session_id"] == state.session_id
    assert result["result"]["serial"] == state.serial
    assert len(calls) == 1
