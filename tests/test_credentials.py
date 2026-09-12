"""Credential values stay in local memory/file, never command or result channels."""
from __future__ import annotations

import io
import json
import os
import stat
import subprocess

import pytest
from dotenv import dotenv_values

from android_ui_analyser import credentials as c

SECRET = "test-only-credential-DoNotEcho"


def dialog(monkeypatch, value=SECRET, status="saved"):
    monkeypatch.setattr(c, "_dialog", lambda *args: (status, value))


def read(path):
    return dotenv_values(stream=io.StringIO(path.read_text()), interpolate=False)


def test_save_preserves_other_content_and_returns_only_status(tmp_path, monkeypatch, capsys, caplog):
    path = tmp_path / ".env"
    path.write_text('# Keep this\nOTHER="same" # comment\n')
    dialog(monkeypatch)
    result = c.request_secret("EXAMPLE_API_KEY", path)
    assert result == {"ok": True, "status": "saved", "name": "EXAMPLE_API_KEY", "env_file": str(path)}
    assert path.read_text().startswith('# Keep this\nOTHER="same" # comment\n')
    assert read(path) == {"OTHER": "same", "EXAMPLE_API_KEY": SECRET}
    assert SECRET not in json.dumps(result) + capsys.readouterr().out + caplog.text
    assert not list(tmp_path.glob("*.aua-lock"))
    assert not list(tmp_path.glob(".aua-credential-*"))
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("value", ["a'quote", r"back\slash", r"\'", 'a"b', " a b # c ", "${DO_NOT_EXPAND}", "dollar$HOME", "unicode-ñ-東京", "literal`cmd`$(nope)"])
def test_literal_values_roundtrip_without_shell_or_interpolation(tmp_path, monkeypatch, value):
    path = tmp_path / ".env.local"
    dialog(monkeypatch, value)
    assert c.request_secret("EXAMPLE_TOKEN", path)["ok"]
    assert read(path)["EXAMPLE_TOKEN"] == value


def test_replacement_removes_duplicates_and_preserves_multiline_other_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_bytes(b'\xef\xbb\xbf# note\r\nexport EXAMPLE_TOKEN="old\r\nmultiline"\r\nKEEP="first\r\nsecond"\r\nEXAMPLE_TOKEN=shadow\r\nTAIL=1')
    dialog(monkeypatch)
    assert c.request_secret("EXAMPLE_TOKEN", path, replace=True)["status"] == "saved"
    raw = path.read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf# note\r\nexport EXAMPLE_TOKEN=')
    assert raw.count(b"EXAMPLE_TOKEN=") == 1
    assert b'KEEP="first\r\nsecond"\r\n' in raw
    assert raw.endswith(b"TAIL=1")
    parsed = dotenv_values(stream=io.StringIO(raw.decode("utf-8-sig")), interpolate=False)
    assert parsed["EXAMPLE_TOKEN"] == SECRET
    assert parsed["KEEP"] == "first\r\nsecond"


def test_append_supplies_missing_final_newline(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("OTHER=1")
    dialog(monkeypatch)
    assert c.request_secret("EXAMPLE_TOKEN", path)["ok"]
    assert path.read_text().startswith("OTHER=1\nEXAMPLE_TOKEN=")


def test_already_set_never_opens_or_rewrites(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    original = b"EXAMPLE_TOKEN=original\n"
    path.write_bytes(original)
    monkeypatch.setattr(c, "_dialog", lambda *args: pytest.fail("dialog must not open"))
    assert c.request_secret("EXAMPLE_TOKEN", path)["status"] == "already_set"
    assert path.read_bytes() == original


def test_whitespace_only_binding_can_be_filled(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("EXAMPLE_TOKEN='   '\n")
    dialog(monkeypatch)
    assert c.request_secret("EXAMPLE_TOKEN", path)["status"] == "saved"
    assert read(path)["EXAMPLE_TOKEN"] == SECRET


@pytest.mark.parametrize("existing", [None, b"OTHER=1\n"])
def test_cancel_makes_no_files_or_changes(tmp_path, monkeypatch, existing):
    path = tmp_path / ".env"
    if existing:
        path.write_bytes(existing)
    dialog(monkeypatch, status="cancelled")
    assert c.request_secret("EXAMPLE_TOKEN", path)["status"] == "cancelled"
    assert path.read_bytes() == existing if existing else not path.exists()
    assert not list(tmp_path.glob("*.aua-lock"))


@pytest.mark.parametrize("value", ["", "   ", "line\nline", "line\rline", "nul\0value", "x" * 65537])
def test_bad_input_never_writes_or_echoes(tmp_path, monkeypatch, value):
    path = tmp_path / ".env"
    dialog(monkeypatch, value)
    result = c.request_secret("EXAMPLE_TOKEN", path)
    assert result["error"]["code"] == "credential_invalid_value"
    assert not path.exists()


@pytest.mark.parametrize("name", ["", "KEY=value", "BAD-NAME", "2KEY", "KEY\nLEAK", "x" * 257])
def test_invalid_name_is_rejected_before_dialog(tmp_path, monkeypatch, name):
    monkeypatch.setattr(c, "_dialog", lambda *args: pytest.fail("dialog must not open"))
    result = c.request_secret(name, tmp_path / ".env")
    assert result["error"]["code"] == "credential_invalid_name"
    assert "name" not in result


def test_malformed_file_is_preserved_without_echoing_content(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    original = ("KEY='" + SECRET).encode()
    path.write_bytes(original)
    monkeypatch.setattr(c, "_dialog", lambda *args: pytest.fail("dialog must not open"))
    result = c.request_secret("EXAMPLE_TOKEN", path)
    assert result["error"]["code"] == "credential_invalid_env"
    assert SECRET not in json.dumps(result)
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["symlink", "directory", "hardlink"])
def test_refuses_nonregular_or_linked_destinations(tmp_path, monkeypatch, kind):
    path, other = tmp_path / ".env", tmp_path / "other"
    other.write_text("KEEP=1")
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(other)
    else:
        os.link(other, path)
    monkeypatch.setattr(c, "_dialog", lambda *args: pytest.fail("dialog must not open"))
    assert c.request_secret("EXAMPLE_TOKEN", path)["error"]["code"] == "credential_unsafe_file"
    assert other.read_text() == "KEEP=1"


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_editor_change_is_preserved(tmp_path, monkeypatch, existing):
    path = tmp_path / ".env"
    if existing:
        path.write_text("OTHER=old")
    def edit(*args):
        path.write_text("OTHER=new")
        return "saved", SECRET
    monkeypatch.setattr(c, "_dialog", edit)
    result = c.request_secret("EXAMPLE_TOKEN", path)
    assert result["error"]["code"] == "credential_file_changed"
    assert path.read_text() == "OTHER=new"
    assert not list(tmp_path.glob(".aua-credential-*"))
    assert not list(tmp_path.glob("*.aua-lock"))


def test_atomic_failure_preserves_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("OTHER=old")
    dialog(monkeypatch)
    def fail(*args):
        raise OSError(SECRET)
    monkeypatch.setattr(c.os, "replace", fail)
    result = c.request_secret("EXAMPLE_TOKEN", path)
    assert result["error"]["code"] == "credential_file_error"
    assert SECRET not in json.dumps(result)
    assert path.read_text() == "OTHER=old"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_other_save_lock_is_not_removed(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    lock = tmp_path / ".env.aua-lock"
    lock.write_text("another writer")
    dialog(monkeypatch)
    assert c.request_secret("EXAMPLE_TOKEN", path)["error"]["code"] == "credential_save_busy"
    assert lock.read_text() == "another writer"
    assert not path.exists()


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_secret_only_returns_through_captured_private_pipe(tmp_path, monkeypatch, capsys, platform):
    path = tmp_path / ".env"
    monkeypatch.setattr(c.sys, "platform", platform)
    def fake_run(command, **kwargs):
        assert SECRET not in json.dumps(command) + (kwargs.get("input") or "")
        assert kwargs["capture_output"] is True and kwargs["check"] is False
        assert "shell" not in kwargs and "env" not in kwargs
        assert kwargs["timeout"] == 310
        script = kwargs.get("input") or command[2]
        assert "hidden answer" in script if platform == "darwin" else 'show="*"' in script
        return subprocess.CompletedProcess(command, 0, "saved:" + SECRET + "\n", "")
    monkeypatch.setattr(c.subprocess, "run", fake_run)
    result = c.request_secret("EXAMPLE_TOKEN", path)
    assert result["ok"] and read(path)["EXAMPLE_TOKEN"] == SECRET
    assert SECRET not in json.dumps(result) + capsys.readouterr().out


@pytest.mark.parametrize("mode", ["timeout", "child_error", "unavailable", "malformed", "decode_error"])
def test_child_failure_never_echoes_captured_secret(tmp_path, monkeypatch, mode):
    def fake_run(command, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 1, output=SECRET, stderr=SECRET)
        if mode == "decode_error":
            raise UnicodeError(SECRET)
        return subprocess.CompletedProcess(command, 1 if mode == "child_error" else 0,
            "unavailable:\n" if mode == "unavailable" else SECRET, SECRET)
    monkeypatch.setattr(c.subprocess, "run", fake_run)
    result = c.request_secret("EXAMPLE_TOKEN", tmp_path / ".env")
    assert not result["ok"]
    assert SECRET not in json.dumps(result)
    assert not (tmp_path / ".env").exists()
