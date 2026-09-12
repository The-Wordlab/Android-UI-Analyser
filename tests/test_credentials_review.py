"""Independent regression checks for the host credential privacy boundary."""

import io
import json
import random
import subprocess

from dotenv import dotenv_values

from android_ui_analyser import credentials


def test_arbitrary_single_line_values_round_trip_without_interpolation():
    randomizer = random.Random(8172)
    alphabet = "abcABC0123 '\\\"$={}#;é雪\t"
    values = ["${UNDEFINED_VALUE}", "'\\\\'", "  api value  "]
    values.extend(
        "sentinel" + "".join(randomizer.choices(alphabet, k=80)) for _ in range(100)
    )
    source = credentials._Snapshot(b"# untouched\nOTHER='multi\nline'\n", None)
    for value in values:
        result = credentials._updated(source, "TEST_API_KEY", value).decode("utf-8")
        parsed = dotenv_values(stream=io.StringIO(result), interpolate=False)
        assert parsed == {"OTHER": "multi\nline", "TEST_API_KEY": value}
        assert result.startswith("# untouched\nOTHER='multi\nline'\n")


def test_duplicate_multiline_target_cannot_shadow_saved_value():
    original = (
        "\ufeff# keep\r\nexport TEST_API_KEY='old\r\nvalue' # prior\r\n"
        "OTHER='still here'\r\n\r\nTEST_API_KEY=shadow\r\n# keep end"
    ).encode()
    result = credentials._updated(credentials._Snapshot(original, None), "TEST_API_KEY", "new")
    assert result.startswith("\ufeff# keep\r\nexport TEST_API_KEY='new'\r\n".encode())
    assert result.count(b"TEST_API_KEY=") == 1
    assert b"OTHER='still here'\r\n\r\n# keep end" in result


def test_invalid_existing_file_never_echoes_bytes(tmp_path, monkeypatch):
    sentinel = "OLD_VALUE_MUST_NEVER_APPEAR"
    destination = tmp_path / ".env"
    original = f"TEST_API_KEY='{sentinel}\n".encode()
    destination.write_bytes(original)
    monkeypatch.setattr(credentials, "_dialog", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    result = credentials.request_secret("TEST_API_KEY", destination, replace=True)
    assert result["error"]["code"] == "credential_invalid_env"
    assert sentinel not in json.dumps(result)
    assert destination.read_bytes() == original


def test_private_subprocess_exception_output_is_not_returned(tmp_path, monkeypatch):
    sentinel = "NEW_VALUE_MUST_NEVER_APPEAR"

    def timeout(command, **_kwargs):
        assert sentinel not in json.dumps(command)
        raise subprocess.TimeoutExpired(command, 1, output=sentinel, stderr=sentinel)

    monkeypatch.setattr(credentials.subprocess, "run", timeout)
    result = credentials.request_secret("TEST_API_KEY", tmp_path / ".env", timeout_s=1)
    assert result["error"]["code"] == "credential_dialog_timeout"
    assert sentinel not in json.dumps(result)
    assert not (tmp_path / ".env").exists()


def test_editor_change_after_temp_write_preserved_and_temp_removed(tmp_path, monkeypatch):
    destination = tmp_path / ".env"
    destination.write_text("OTHER=original\n")
    before = credentials._snapshot(destination)
    real_fsync = credentials.os.fsync

    def editor_change(fd):
        real_fsync(fd)
        destination.write_text("OTHER=editor-update\n")

    monkeypatch.setattr(credentials.os, "fsync", editor_change)
    try:
        credentials._write(destination, before, b"TEST_API_KEY=NEW_PRIVATE_SENTINEL\n")
    except credentials._CredentialError as error:
        assert error.code == "credential_file_changed"
    else:
        raise AssertionError("A concurrent edit must stop the save")
    assert destination.read_text() == "OTHER=editor-update\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]
