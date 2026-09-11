"""MP4 compatibility and host-only export; fixtures contain no device captures."""

import subprocess
from pathlib import Path

import pytest

from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms import android_recording
from test_native_recording_timeline import mp4


def test_single_segment_is_copied_byte_for_byte_without_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "segment-0.mp4"
    source.write_bytes(mp4(12))
    target = tmp_path / "output.mp4"
    monkeypatch.setattr("shutil.which", lambda _: pytest.fail("single segment needs no executable"))
    assert android_recording.export_mp4([source], target) == "direct_copy"
    assert target.read_bytes() == source.read_bytes()


def test_concat_uses_every_segment_in_order_without_timing_overrides(tmp_path, monkeypatch):
    paths = [tmp_path / f"segment-{i}.mp4" for i in range(12)]
    for path in paths:
        path.write_bytes(mp4(3))
    monkeypatch.setattr("shutil.which", lambda _: "/fictional/ffmpeg")
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        playlist = Path(argv[argv.index("-i") + 1])
        assert playlist.read_text().splitlines() == [f"file '{p.name}'" for p in paths]
        assert Path(kwargs["cwd"]) == tmp_path
        assert kwargs["timeout"] > 0
        Path(argv[-1]).write_bytes(mp4(36))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    target = tmp_path / "output.mp4"
    assert android_recording.export_mp4(paths, target) == "bitstream_copy_concat"
    command = seen[0]
    assert command[command.index("-c") + 1] == "copy"
    assert command[command.index("-f") + 1] == "concat"
    assert "-n" in command and "-nostdin" in command
    assert not set(command) & {"-r", "-vf", "-filter_complex", "-itsscale", "-vsync", "-y"}
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize("failure", ["missing", "exit", "timeout", "invalid"])
def test_export_failure_keeps_originals_and_has_explicit_error(tmp_path, monkeypatch, failure):
    paths = [tmp_path / f"segment-{i}.mp4" for i in range(2)]
    for p in paths:
        p.write_bytes(mp4(3))
    monkeypatch.setattr("shutil.which", lambda _: None if failure == "missing" else "/fictional/ffmpeg")

    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        Path(argv[-1]).write_bytes(b"incomplete")
        return subprocess.CompletedProcess(argv, 1 if failure == "exit" else 0, "", "encoder failure")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(DeviceError) as exc:
        android_recording.export_mp4(paths, tmp_path / "output.mp4")
    assert exc.value.code == ("recording_export_unsupported" if failure == "missing" else "recording_export_failed")
    assert all(p.read_bytes() == mp4(3) for p in paths)


@pytest.mark.parametrize("symlink", [False, True])
def test_export_never_overwrites_existing_output(tmp_path, symlink):
    source = tmp_path / "segment-0.mp4"
    source.write_bytes(mp4(3))
    target = tmp_path / "output.mp4"
    if symlink:
        target.symlink_to(tmp_path / "missing.mp4")
    else:
        target.write_bytes(b"existing evidence")
    with pytest.raises(DeviceError, match="already exists"):
        android_recording.export_mp4([source], target)
    assert target.is_symlink() if symlink else target.read_bytes() == b"existing evidence"


@pytest.mark.parametrize("metadata_track", [False, True])
def test_real_host_ffmpeg_preserves_packets_order_and_pause_timestamps(tmp_path, monkeypatch, metadata_track):
    """Opt-in, synthetic color planes only. This is host export proof, not Android proof."""
    import json
    import os
    import shutil

    if os.environ.get("AUA_TEST_SYNTHETIC_FFMPEG") != "1":
        pytest.skip("opt in to bounded host-only synthetic ffmpeg validation")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("optional host ffmpeg/ffprobe unavailable")
    monkeypatch.setenv("AUA_SYNTHETIC_MEDIA_ROOT", str(tmp_path))
    paths = []
    for i, color in enumerate(["red", "green", "blue"]):
        path = tmp_path / f"segment-{i}.mp4"
        subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
             "-filter_threads", "1", "-f", "lavfi", "-i", f"color=c={color}:s=64x64:r=10:d=3",
             "-vf", "select='eq(n,0)+eq(n,1)+eq(n,2)+eq(n,22)+eq(n,23)+eq(n,24)+eq(n,29)'",
             "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "1", "-bf", "0", str(path)],
            check=True, capture_output=True, timeout=15,
        )
        if metadata_track:
            # Synthetic non-media track: reuse fictional color sample bytes, but mark
            # the cloned track as timed metadata. Never import a device capture.
            raw = path.read_bytes()
            moov = raw.index(b"moov") - 4
            size = int.from_bytes(raw[moov:moov + 4], "big")
            trak = raw.index(b"trak", moov) - 4
            length = int.from_bytes(raw[trak:trak + 4], "big")
            clone = bytearray(raw[trak:trak + length])
            track_id = clone.index(b"tkhd") + 16
            clone[track_id:track_id + 4] = (2).to_bytes(4, "big")
            clone = clone.replace(b"vide", b"meta", 1).replace(b"avc1", b"mett", 1)
            changed = (raw[:moov] + (size + len(clone)).to_bytes(4, "big")
                       + raw[moov + 4:moov + size] + clone + raw[moov + size:])
            path.write_bytes(changed)
        paths.append(path)

    def packets(path):
        result = subprocess.run(
            [ffprobe, "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0", "-show_packets",
             "-show_data_hash", "sha256", "-show_entries", "packet=pts_time,dts_time,duration_time,data_hash",
             "-of", "json", "-i", str(path)], check=True, capture_output=True, text=True, timeout=15,
        )
        return json.loads(result.stdout)["packets"]

    original_bytes = [p.read_bytes() for p in paths]
    originals = [packets(p) for p in paths]
    output = tmp_path / "synthetic-export.mp4"
    assert android_recording.export_mp4(paths, output) == "bitstream_copy_concat"
    combined = packets(output)
    def frame_hashes(path):
        # Concat can insert H.264 parameter sets without re-encoding. Compare decoded
        # pixels, and independently compare every packet timestamp/duration below.
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-threads", "1",
             "-i", str(path), "-fps_mode", "passthrough", "-f", "framemd5", "-"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return [line.rsplit(",", 1)[1].strip() for line in result.stdout.splitlines()
                if line and not line.startswith("#")]

    assert len(combined) == sum(map(len, originals))
    assert frame_hashes(output) == [value for path in paths for value in frame_hashes(path)]
    cursor = 0
    for original in originals:
        exported = combined[cursor:cursor + len(original)]
        for key in ("pts_time", "dts_time"):
            expected = [float(p[key]) - float(original[0][key]) for p in original]
            observed = [float(p[key]) - float(exported[0][key]) for p in exported]
            assert observed == pytest.approx(expected, abs=0.0001)
            assert max(b - a for a, b in zip(observed, observed[1:], strict=False)) == pytest.approx(2.0, abs=1e-9)
        assert [p["duration_time"] for p in exported] == [p["duration_time"] for p in original]
        cursor += len(original)
    assert android_recording.media_duration(output) == pytest.approx(sum(android_recording.media_duration(p) for p in paths), abs=0.001)
    assert [p.read_bytes() for p in paths] == original_bytes
    # Decode every exported frame to a null sink to check playable video without a viewer.
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-threads", "1",
         "-i", str(output), "-f", "null", "-"], check=True, capture_output=True, timeout=15,
    )
    single = tmp_path / "synthetic-single.mp4"
    assert android_recording.export_mp4(paths[:1], single) == "direct_copy"
    assert single.read_bytes() == original_bytes[0]


def test_no_playable_export_does_not_claim_streams_were_copied(tmp_path, monkeypatch):
    import json

    from test_recording_lifecycle_recovery import setup_recording

    target, dev, engine, root = setup_recording(tmp_path, monkeypatch)
    target.running = False
    target.crashed = True
    target.tail = 'all_corrupt'
    output = tmp_path / 'output.mp4'
    with pytest.raises(DeviceError):
        engine.record_stop(str(output))
    report = json.loads(output.with_name('output.mp4.recording.json').read_text())
    assert report['export']['stream_policy'] == 'No playable streams exported.'
    assert report['export']['path'] is None
