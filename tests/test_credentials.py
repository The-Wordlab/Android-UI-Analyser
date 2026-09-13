"""Credential values stay in local memory/file, never command or result channels."""
from __future__ import annotations

import errno
import io
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
    assert path.with_name(".env.aua-lock").is_file()
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
    assert path.with_name(".env.aua-lock").is_file()


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
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env", ".env.aua-lock"]


def test_other_save_lock_is_not_removed(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    lock = tmp_path / ".env.aua-lock"
    lock.write_text("another writer")
    dialog(monkeypatch)
    with c._save_lock(path):
        identity = lock.stat().st_ino
        assert c.request_secret("EXAMPLE_TOKEN", path)["error"]["code"] == "credential_save_busy"
        assert lock.read_text() == "another writer"
        assert lock.stat().st_ino == identity
        assert not path.exists()
    assert c.request_secret("EXAMPLE_TOKEN", path)["status"] == "saved"
    assert lock.stat().st_ino == identity


@pytest.mark.parametrize("legacy_bytes", [b"", b"old lock contents"])
def test_orphaned_legacy_lock_is_reused_without_manual_recovery(tmp_path, monkeypatch, legacy_bytes):
    path = tmp_path / ".env"
    lock = tmp_path / ".env.aua-lock"
    lock.write_bytes(legacy_bytes)
    identity = lock.stat().st_ino
    dialog(monkeypatch)
    assert c.request_secret("EXAMPLE_TOKEN", path)["status"] == "saved"
    assert lock.stat().st_ino == identity
    assert lock.read_bytes() == (b"\0" if os.name == "nt" and not legacy_bytes else legacy_bytes)
    assert read(path)["EXAMPLE_TOKEN"] == SECRET


@pytest.mark.parametrize("kind", ["symlink", "directory", "hardlink", "dangling_symlink", "fifo"])
def test_refuses_unsafe_lock_without_touching_other_file(tmp_path, monkeypatch, kind):
    path, lock, other = tmp_path / ".env", tmp_path / ".env.aua-lock", tmp_path / "other"
    other.write_text("UNCHANGED")
    if kind == "directory":
        lock.mkdir()
    elif kind == "symlink":
        lock.symlink_to(other)
    elif kind == "dangling_symlink":
        lock.symlink_to(tmp_path / "absent")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs are unavailable")
        os.mkfifo(lock)
    else:
        os.link(other, lock)
    dialog(monkeypatch)
    result = c.request_secret("EXAMPLE_TOKEN", path)
    assert result["error"]["code"] == "credential_unsafe_lock"
    assert other.read_text() == "UNCHANGED"
    assert not path.exists()


def test_lock_replacement_during_save_is_preserved_and_stops_commit(tmp_path, monkeypatch):
    path, lock = tmp_path / ".env", tmp_path / ".env.aua-lock"
    path.write_text("OTHER=original\n")
    dialog(monkeypatch)
    original_fsync = c.os.fsync

    def replace_lock(fd):
        original_fsync(fd)
        lock.unlink()
        lock.write_bytes(b"replacement lock")

    monkeypatch.setattr(c.os, "fsync", replace_lock)
    assert c.request_secret("EXAMPLE_TOKEN", path)["error"]["code"] == "credential_lock_changed"
    assert lock.read_bytes() == b"replacement lock"
    assert path.read_text() == "OTHER=original\n"
    assert not list(tmp_path.glob(".aua-credential-*"))


def test_killed_process_releases_lock_and_next_save_succeeds(tmp_path, monkeypatch):
    path, lock, ready = tmp_path / ".env", tmp_path / ".env.aua-lock", tmp_path / "ready"
    script = """import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from android_ui_analyser.credentials import _save_lock
with _save_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready')
    time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(ready), str(Path(c.__file__).parents[1])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "The independent lock holder did not start"
        identity = lock.stat().st_ino
        dialog(monkeypatch)
        assert c.request_secret("EXAMPLE_TOKEN", path)["error"]["code"] == "credential_save_busy"
        child.kill()
        stdout, stderr = child.communicate(timeout=5)
        assert not stdout and not stderr
        assert lock.stat().st_ino == identity
        assert c.request_secret("EXAMPLE_TOKEN", path)["status"] == "saved"
        assert lock.stat().st_ino == identity
        assert read(path)["EXAMPLE_TOKEN"] == SECRET
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


@pytest.mark.parametrize("busy", [False, True])
def test_windows_byte_lock_initializes_only_a_nonsecret_marker(tmp_path, monkeypatch, busy):
    lock = tmp_path / ".env.aua-lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    calls = []

    def locking(actual_fd, mode, length):
        calls.append((actual_fd, mode, length, os.lseek(actual_fd, 0, os.SEEK_CUR)))
        if busy:
            raise PermissionError(errno.EACCES, SECRET)

    fake = SimpleNamespace(locking=locking, LK_NBLCK=2)
    try:
        with monkeypatch.context() as windows:
            windows.setattr(c.os, "name", "nt")
            windows.setitem(sys.modules, "msvcrt", fake)
            if busy:
                with pytest.raises(c._CredentialError) as failure:
                    c._acquire_lock(fd)
                assert failure.value.code == "credential_save_busy"
                assert SECRET not in str(failure.value)
            else:
                c._acquire_lock(fd)
        assert calls == [(fd, 2, 1, 0)]
        assert lock.read_bytes() == b"\0"
    finally:
        os.close(fd)


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
