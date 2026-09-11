"""Publish fictional native recordings on volumes without hardlinks, without overwrites."""

import errno
import json
from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_recording
from test_native_recording_timeline import _fake_export, mp4
from test_recording_lifecycle_recovery import setup_recording


def _recording(tmp_path, monkeypatch):
    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.clock = 520
    monkeypatch.setattr(android_recording, "export_mp4", _fake_export)
    return target, dev, engine, root


def _assert_remote_retained(target, dev, engine, root):
    assert dev._recording_state_path().is_file()
    assert engine._pending_device_change("screen_recording").args["remote_path"] == root
    assert not any(command.startswith("rm -rf") for command in target.commands)


@pytest.mark.parametrize(
    "error", sorted({errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV, errno.EPERM})
)
def test_stop_publishes_complete_recording_when_hardlinks_are_unsupported(
    tmp_path,
    monkeypatch,
    error,
):
    target, dev, engine, root = _recording(tmp_path, monkeypatch)

    def unsupported(_source, _destination):
        raise OSError(error, "hardlinks unavailable on this volume")

    monkeypatch.setattr(android_recording.os, "link", unsupported)
    destination = tmp_path / "journey.mp4"
    result = engine.record_stop(str(destination))

    assert result.ok is True
    assert result.detail == str(destination)
    assert destination.read_bytes() == mp4(419)
    manifest = json.loads(Path(str(destination) + ".recording.json").read_text())
    assert manifest["cleanup_pending"] is False
    assert manifest["duration_check"] == "passed"
    for index, duration in enumerate([180, 180, 59]):
        assert (Path(str(destination) + ".segments") / f"segment-{index}.mp4").read_bytes() == mp4(
            duration
        )
    assert not dev._recording_state_path().exists()
    assert engine._pending_device_change("screen_recording") is None
    assert any(command.startswith("rm -rf " + root) for command in target.commands)


@pytest.mark.parametrize("error", [errno.EACCES, errno.ENOSPC, errno.EIO, errno.EEXIST])
def test_stop_does_not_mask_other_hardlink_errors_with_a_copy(tmp_path, monkeypatch, error):
    target, dev, engine, root = _recording(tmp_path, monkeypatch)

    def failed(_source, _destination):
        raise OSError(error, "publication refused")

    monkeypatch.setattr(android_recording.os, "link", failed)
    monkeypatch.setattr(
        android_recording.shutil,
        "copyfileobj",
        lambda *_a, **_kw: pytest.fail(
            "permission, storage and existing-file errors must propagate"
        ),
    )
    destination = tmp_path / "journey.mp4"
    with pytest.raises(DeviceError) as caught:
        engine.record_stop(str(destination))
    assert caught.value.code == "recording_export_failed"
    assert caught.value.__cause__.errno == error
    assert not destination.exists()
    assert list(Path(str(destination) + ".segments").iterdir()) == []
    _assert_remote_retained(target, dev, engine, root)


@pytest.mark.parametrize("denied_mode", ["rb", "xb"])
def test_eperm_fallback_still_enforces_source_and_destination_permissions(
    tmp_path,
    monkeypatch,
    denied_mode,
):
    target, dev, engine, root = _recording(tmp_path, monkeypatch)
    original_open = Path.open
    pending_copy = {}

    def unsupported(source, destination):
        pending_copy.update(rb=source, xb=destination)
        raise OSError(errno.EPERM, "filesystem cannot create hardlinks")

    def denied(path, mode="r", *args, **kwargs):
        if mode == denied_mode and path == pending_copy.get(mode):
            raise PermissionError(errno.EACCES, "copy access denied")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(android_recording.os, "link", unsupported)
    monkeypatch.setattr(Path, "open", denied)
    destination = tmp_path / "journey.mp4"
    with pytest.raises(DeviceError) as caught:
        engine.record_stop(str(destination))
    assert caught.value.code == "recording_export_failed"
    assert isinstance(caught.value.__cause__, PermissionError)
    assert caught.value.__cause__.errno == errno.EACCES
    assert not destination.exists()
    assert list(Path(str(destination) + ".segments").iterdir()) == []
    _assert_remote_retained(target, dev, engine, root)


@pytest.mark.parametrize("raced_name", ["journey.mp4", "journey.mp4.recording.json"])
@pytest.mark.parametrize("kind", ["file", "dangling_symlink", "existing_symlink"])
def test_fallback_refuses_file_or_symlink_created_after_preflight(
    tmp_path,
    monkeypatch,
    raced_name,
    kind,
):
    target, dev, engine, root = _recording(tmp_path, monkeypatch)
    destination = tmp_path / "journey.mp4"
    raced = tmp_path / raced_name
    symlink_target = tmp_path / "other-evidence"
    if kind == "existing_symlink":
        symlink_target.write_bytes(b"unrelated evidence")

    def race(_source, publication_path):
        if publication_path == raced:
            if kind == "file":
                raced.write_bytes(b"unrelated evidence")
            else:
                raced.symlink_to(symlink_target)
        raise OSError(errno.EOPNOTSUPP, "hardlinks unavailable")

    monkeypatch.setattr(android_recording.os, "link", race)
    with pytest.raises(DeviceError) as caught:
        engine.record_stop(str(destination))
    assert caught.value.code == "recording_export_failed"
    assert isinstance(caught.value.__cause__, FileExistsError)
    if kind == "file":
        assert raced.read_bytes() == b"unrelated evidence"
    else:
        assert raced.is_symlink()
        assert raced.readlink() == symlink_target
        if kind == "existing_symlink":
            assert symlink_target.read_bytes() == b"unrelated evidence"
        else:
            assert not symlink_target.exists()
    _assert_remote_retained(target, dev, engine, root)


def test_mid_copy_failure_keeps_originals_and_remote_undo(tmp_path, monkeypatch):
    target, dev, engine, root = _recording(tmp_path, monkeypatch)
    destination = tmp_path / "journey.mp4"
    original_copy = android_recording.shutil.copyfileobj

    def unsupported(_source, _destination):
        raise OSError(errno.EOPNOTSUPP, "hardlinks unavailable")

    def incomplete(source, output):
        if Path(output.name) == destination:
            output.write(b"partial")
            raise OSError(errno.ENOSPC, "volume filled during copy")
        return original_copy(source, output)

    monkeypatch.setattr(android_recording.os, "link", unsupported)
    monkeypatch.setattr(android_recording.shutil, "copyfileobj", incomplete)
    with pytest.raises(DeviceError) as caught:
        engine.record_stop(str(destination))
    assert caught.value.code == "recording_export_failed"
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert destination.read_bytes() == b"partial"
    assert len(list(Path(str(destination) + ".segments").glob("*.mp4"))) == 3
    assert Path(str(destination) + ".recording.json").is_file()
    _assert_remote_retained(target, dev, engine, root)
