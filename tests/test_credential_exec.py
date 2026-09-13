"""A saved credential reaches one consumer without replay or environment pollution."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys

import pytest

from android_ui_analyser import credential_exec as runner
from android_ui_analyser import credentials

NAME = "AUA_TEST_CONSUMER_TOKEN"
TOKEN = "dummy-only-token-${literal}-秘密"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    monkeypatch.delenv("AUA_TEST_UNSELECTED_TOKEN", raising=False)


def _consumer(marker, *, fail=False):
    digest = hashlib.sha256(TOKEN.encode()).hexdigest()
    return [sys.executable, "-c", (
        "import os, pathlib, hashlib, json, sys; "
        f"p=pathlib.Path({str(marker)!r}); "
        "p.write_text(p.read_text()+'run\\n' if p.exists() else 'run\\n'); "
        f"value=os.environ.get({NAME!r}, ''); "
        f"print(json.dumps({{'received': hashlib.sha256(value.encode()).hexdigest()=={digest!r}, "
        "'unselected': 'AUA_TEST_UNSELECTED_TOKEN' in os.environ})); "
        f"sys.exit({7 if fail else 0})"
    )]


def test_missing_key_save_launches_real_consumer_once(tmp_path, monkeypatch, capsys):
    destination, marker = tmp_path / ".env", tmp_path / "started"
    calls = []

    def save(name, path, timeout):
        calls.append((name, path, timeout))
        return "saved", TOKEN

    monkeypatch.setattr(credentials, "_dialog", save)
    result, code = runner.run_with_credentials(
        _consumer(marker), required=[NAME], env_file=destination,
    )
    captured = capsys.readouterr()
    assert result == {"ok": True, "status": "completed", "started": True, "exit_code": 0}
    assert code == 0
    assert json.loads(captured.out) == {"received": True, "unselected": False}
    assert not captured.err
    assert calls == [(NAME, destination, 300)]
    assert marker.read_text() == "run\n"
    assert NAME not in os.environ
    assert TOKEN not in captured.out + captured.err + json.dumps(result)


def test_saved_values_are_literal_and_only_selected_names_are_loaded(tmp_path, monkeypatch, capsys):
    destination, marker = tmp_path / ".env", tmp_path / "started"
    destination.write_text(f"{NAME}='{TOKEN}'\nAUA_TEST_UNSELECTED_TOKEN=must-stay-private\n")
    monkeypatch.setattr(credentials, "_dialog", lambda *_: pytest.fail("saved key needs no dialog"))
    result, code = runner.run_with_credentials(
        _consumer(marker), required=[NAME, NAME], env_file=destination, prompt=False,
    )
    assert result["ok"] and code == 0
    assert json.loads(capsys.readouterr().out) == {"received": True, "unselected": False}
    assert NAME not in os.environ


def test_process_credential_wins_without_reading_dotenv(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(NAME, TOKEN)
    monkeypatch.setattr(runner, "_file_values", lambda *_: pytest.fail("must not read file"))
    result, code = runner.run_with_credentials(
        _consumer(tmp_path / "started"), required=[NAME], env_file=tmp_path / "absent" / ".env",
    )
    assert code == 0 and result["ok"]
    assert json.loads(capsys.readouterr().out)["received"]
    assert os.environ[NAME] == TOKEN


@pytest.mark.parametrize("mode", ["cancelled", "timeout", "headless", "no_prompt"])
def test_setup_cancellation_or_failure_never_starts_consumer(tmp_path, monkeypatch, mode):
    def dialog(*_args, **_kwargs):
        if mode == "cancelled":
            return "cancelled", ""
        raise credentials._CredentialError("credential_dialog_" + mode, "The test dialog failed.")

    monkeypatch.setattr(credentials, "_dialog", dialog)
    marker = tmp_path / "started"
    result, code = runner.run_with_credentials(
        _consumer(marker), required=[NAME], env_file=tmp_path / ".env", prompt=mode != "no_prompt",
    )
    assert not result["ok"] and result["started"] is False
    assert code == (130 if mode == "cancelled" else 1)
    assert not marker.exists()
    assert not (tmp_path / ".env").exists()


def test_consumer_failure_is_not_replayed_or_reprompted(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(NAME, TOKEN)
    result, code = runner.run_with_credentials(
        _consumer(tmp_path / "started", fail=True), required=[NAME], env_file=tmp_path / ".env",
    )
    assert result == {"ok": False, "status": "completed", "started": True, "exit_code": 7}
    assert code == 7
    assert (tmp_path / "started").read_text() == "run\n"
    assert json.loads(capsys.readouterr().out)["received"]


def test_later_cancel_keeps_explicit_save_but_does_not_launch(tmp_path, monkeypatch):
    second = "AUA_TEST_SECOND_TOKEN"
    monkeypatch.delenv(second, raising=False)
    seen = []

    def dialog(name, *_args):
        seen.append(name)
        return ("saved", TOKEN) if name == NAME else ("cancelled", "")

    monkeypatch.setattr(credentials, "_dialog", dialog)
    path = tmp_path / ".env"
    marker = tmp_path / "started"
    result, code = runner.run_with_credentials(
        _consumer(marker), required=[NAME, second], env_file=path,
    )
    assert seen == [NAME, second]
    assert result["status"] == "cancelled" and code == 130
    assert not marker.exists()
    assert runner._file_values(path, [NAME]) == {NAME: TOKEN}
    assert NAME not in os.environ and second not in os.environ


def test_raw_values_on_both_consumer_streams_are_masked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(NAME, TOKEN)
    command = [sys.executable, "-c", (
        "import os,sys; "
        f"v=os.environ[{NAME!r}].encode(); "
        "os.write(1,b'out:'+v[:7]); os.write(1,v[7:]+b'\\n'); "
        "os.write(2,b'err:'+v+b'\\n')"
    )]
    result, code = runner.run_with_credentials(command, required=[NAME], env_file=tmp_path / ".env")
    captured = capsys.readouterr()
    assert result["ok"] and code == 0
    assert captured.out == "out:[REDACTED]\n"
    assert captured.err == "err:[REDACTED]\n"


@pytest.mark.parametrize("width", [1, 2, 4, 11, 64])
def test_redaction_handles_overlaps_unicode_and_every_chunk_boundary(width):
    text = "前:alpha-long/alpha/alphabet/秘密:後"
    encoded = text.encode()
    mask = runner._Redactor(["alpha", "alpha-long", "秘密"])
    chunks = [mask.feed(encoded[i:i+width]) for i in range(0, len(encoded), width)]
    chunks.append(mask.feed(b"", final=True))
    assert b"".join(chunks).decode() == "前:[REDACTED]/[REDACTED]/[REDACTED]bet/[REDACTED]:後"
    assert mask._pending == b""


def test_output_failure_drains_without_revealing_destination_exception(tmp_path, monkeypatch, capsys):
    class BrokenOutput(io.StringIO):
        def write(self, text):
            raise OSError(TOKEN)

    monkeypatch.setenv(NAME, TOKEN)
    marker = tmp_path / "finished"
    command = [sys.executable, "-c", (
        "import os,pathlib; "
        "os.write(1,b'x'*1048576); os.write(2,b'completed\\n'); "
        f"pathlib.Path({str(marker)!r}).write_text('finished')"
    )]
    with monkeypatch.context() as output:
        output.setattr(runner.sys, "stdout", BrokenOutput())
        result, code = runner.run_with_credentials(
            command, required=[NAME], env_file=tmp_path / ".env"
        )
    assert code == 1 and result["error"]["code"] == "credential_output_failed"
    assert result["started"] is True and marker.read_text() == "finished"
    captured = capsys.readouterr()
    assert captured.err == "completed\n"
    assert TOKEN not in json.dumps(result) + captured.out + captured.err


def test_malformed_file_error_does_not_reveal_existing_value(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text(f"{NAME}='{TOKEN}")
    result, code = runner.run_with_credentials(
        _consumer(tmp_path / "started"), required=[NAME], env_file=path, prompt=False,
    )
    assert code == 1 and result["error"]["code"] == "credential_invalid_env"
    assert TOKEN not in json.dumps(result) + capsys.readouterr().out


def test_setup_result_is_rechecked_before_consumer_launch(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(credentials, "request_secret", lambda *_a, **_kw: {"ok": True, "status": "saved"})
    result, code = runner.run_with_credentials(
        _consumer(tmp_path / "started"), required=[NAME], env_file=path,
    )
    assert code == 1 and result["error"]["code"] == "credential_missing"
    assert not (tmp_path / "started").exists()


def test_spawn_error_text_is_never_returned(tmp_path, monkeypatch):
    monkeypatch.setenv(NAME, TOKEN)

    def fail(*_args, **_kwargs):
        raise OSError(TOKEN)

    monkeypatch.setattr(runner.subprocess, "Popen", fail)
    result, code = runner.run_with_credentials(["consumer"], required=[NAME], env_file=tmp_path / ".env")
    assert code == 1 and result["started"] is False
    assert TOKEN not in json.dumps(result)


def test_explicit_credential_in_consumer_argv_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(NAME, TOKEN)
    result, code = runner.run_with_credentials(
        ["consumer", "--api-key=" + TOKEN], required=[NAME], env_file=tmp_path / ".env",
    )
    assert code == 1 and result["error"]["code"] == "credential_value_in_command"
    assert TOKEN not in json.dumps(result)
