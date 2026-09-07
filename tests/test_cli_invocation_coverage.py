"""One host invocation is accounted for even when parsing or bypassing the router."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import cli, journal, leases
from android_ui_analyser.config import load_config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.session import SessionState, create_session_state, review_session_events
from conftest import FakeDevice, make_config

runner = CliRunner()


def events():
    return journal.read_since(load_config().cache.dir, None, limit=50)


@pytest.mark.parametrize(
    "argv", [["clock", "--help"], ["capture", "sheet", "--help"], ["session", "finish", "--help"]]
)
def test_nested_help_is_one_invocation_without_a_device(monkeypatch, argv):
    monkeypatch.setattr(Engine, "_connect_target", lambda *_a, **_kw: pytest.fail("help connected"))
    result = runner.invoke(cli.app, argv)
    assert result.exit_code == 0, result.output
    rows = events()
    assert len(rows) == 1
    assert rows[0]["cmd"] == "cli_help"
    assert rows[0]["invocation_id"]
    assert rows[0]["args"]["command"].endswith(" ".join(argv[:-1]))


def test_nested_parse_failure_redacts_untyped_values(monkeypatch):
    monkeypatch.setattr(
        Engine, "_connect_target", lambda *_a, **_kw: pytest.fail("parse connected")
    )
    result = runner.invoke(cli.app, ["clock", "set", "--ms", "private-token-abc"])
    assert result.exit_code == 2
    rows = events()
    assert len(rows) == 1
    assert rows[0]["cmd"] == "cli_usage_error"
    assert rows[0]["ok"] is False
    assert "private-token-abc" not in json.dumps(rows)
    assert "private-token-abc" not in "".join(
        p.read_text()
        for p in Path(load_config().cache.dir).rglob("*.jsonl")
    )


def test_direct_callback_and_routed_callback_each_record_once(monkeypatch):
    cfg = make_config()
    cfg.cache.dir = load_config().cache.dir
    engine = Engine(cfg, device=FakeDevice())
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
    monkeypatch.setattr(engine, "clock_set", lambda **kw: {"ok": True, "verified": True})
    first = runner.invoke(cli.app, ["clock", "set", "--ms", "123"])
    assert first.exit_code == 0, first.output
    assert len(events()) == 1
    assert events()[0]["cmd"] == "clock_set"
    assert events()[0]["result"]["verified"] is True
    monkeypatch.setattr(engine, "flow_list", lambda **kw: {"ok": True, "flows": []})
    second = runner.invoke(cli.app, ["flow", "list"])
    assert second.exit_code == 0, second.output
    rows = events()
    assert len(rows) == 2
    assert rows[0]["invocation_id"] != rows[1]["invocation_id"]


def test_known_wrong_command_returns_one_short_correction():
    result = runner.invoke(cli.app, ["screen"])
    error = json.loads(result.stderr.splitlines()[-1])["error"]
    assert error["recommended_call"] == "aua analyze --help"
    assert "available_commands" not in error
    assert "how_to_drive" not in error


@pytest.mark.parametrize("argv", [["private-token-abc"], ["clock", "--private-token-abc"]])
def test_unknown_syntax_cannot_leak_a_secret_as_a_command_or_option_name(argv):
    runner.invoke(cli.app, argv)
    retained = "".join(path.read_text() for path in Path(load_config().cache.dir).rglob("*.jsonl"))
    assert "private-token-abc" not in retained


def test_direct_callback_journals_outcome_without_private_output(monkeypatch):
    cfg = load_config()
    engine = Engine(cfg, device=FakeDevice())
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
    monkeypatch.setattr(
        engine, "clock_set", lambda **kw: {"ok": True, "detail": "private-token-abc"}
    )
    result = runner.invoke(cli.app, ["clock", "set", "--ms", "123"])
    assert result.exit_code == 0
    assert "private-token-abc" in result.stdout  # caller receives its own response unchanged
    retained = "".join(path.read_text() for path in Path(cfg.cache.dir).rglob("*.jsonl"))
    assert "private-token-abc" not in retained
    assert events()[0]["result"]["ok"] is True


@pytest.mark.parametrize("actual_error", ["clock_write_failed", "device_unavailable", None])
def test_direct_callback_expected_error_requires_the_exact_failure(monkeypatch, actual_error):
    cfg = load_config()
    cfg.lease.enabled = False
    cfg.daemon.enabled = False
    engine = Engine(cfg, device=FakeDevice())
    monkeypatch.setattr(cli.GlobalOpts, "engine", lambda self: engine)
    monkeypatch.setattr(Engine, "_list_targets", lambda *_a, **_kw: pytest.fail("discovery"))
    monkeypatch.setattr(Engine, "_connect_target", lambda *_a, **_kw: pytest.fail("connection"))

    def clock_set(**kwargs):
        if actual_error:
            raise DeviceError("private fixture failure", code=actual_error)
        return {"ok": True}

    monkeypatch.setattr(engine, "clock_set", clock_set)
    result = runner.invoke(
        cli.app, ["--expect-error", "clock_write_failed", "clock", "set", "--ms", "123"]
    )
    assert (result.exit_code != 0) is (actual_error is not None)
    rows = events()
    assert len(rows) == 1
    event = rows[0]
    matched = actual_error == "clock_write_failed"
    assert event["cmd"] == "clock_set"
    assert event["ok"] is (actual_error is None)
    assert event["extra"]["expected_error_code"] == "clock_write_failed"
    assert event["extra"]["expected_error_matched"] is matched
    assert "private fixture failure" not in json.dumps(rows)

    state = SessionState(
        session_id="fictional-probe-session",
        goal="Verify a declared clock refusal",
        goal_hash="fictional-probe-goal",
        serial="fake",
        started_ms=0,
        recommended_kind="manual",
        recommended_cli="reuse the returned result",
    )
    event["session_id"] = state.session_id
    review = review_session_events(state, rows)
    assert review["calls"] == 1
    assert review["accounting"]["expected_error_probes"] == 1
    assert review["accounting"]["expected_error_matches"] == int(matched)
    assert review["run_ok"] is matched


def test_host_help_joins_a_session_using_the_shared_lease_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("AUA_OWNER", "help-owner")
    cfg = load_config()
    assert cfg.cache.dir != cfg.lease.registry_dir
    owner = leases.resolve_owner(None)
    serial = "fictional-help-target"
    assert leases.acquire(cfg.lease.registry_dir, serial, owner=owner)
    state = create_session_state(
        cfg.cache.dir,
        goal="verify the fixture",
        serial=serial,
        owner=owner,
        recommended_kind="manual_observation",
        recommended_cli="reuse observation",
        network_backup_preexisting=False,
        network_profile_preexisting=False,
    )
    monkeypatch.setattr(Engine, "_connect_target", lambda *_a, **_kw: pytest.fail("help connected"))
    result = runner.invoke(cli.app, ["clock", "--help"])
    assert result.exit_code == 0
    rows = journal.read_since(cfg.cache.dir, serial, limit=5)
    assert len(rows) == 1
    assert rows[0]["session_id"] == state.session_id
    assert rows[0]["owner"] == owner


def test_eager_root_help_honors_explicit_target_without_a_device(monkeypatch):
    monkeypatch.setattr(Engine, "_connect_target", lambda *_a, **_kw: pytest.fail("help connected"))
    result = runner.invoke(cli.app, ["--serial", "fictional-explicit", "--help"])
    assert result.exit_code == 0
    rows = journal.read_since(load_config().cache.dir, "fictional-explicit", limit=5)
    assert len(rows) == 1
    assert rows[0]["cmd"] == "cli_help"
