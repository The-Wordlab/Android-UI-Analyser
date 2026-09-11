"""Android-owned native recording supervisor and loss-aware evidence collection.

The supervisor runs on the target, independently of the CLI/daemon. The existing
screen_recording ledger owns its unique directory, processes and every segment. Original
MP4s are retained unchanged. A playable copy/concat export omits uncaptured gaps;
a sidecar describes segment placement and unverified wall-clock coverage.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import DeviceError

if TYPE_CHECKING:
    from .android_device import Uiautomator2Device

_ROOT = re.compile(r"/[A-Za-z0-9_./-]+\.aua-recording-[0-9a-f]{32}\Z")
_FINALIZE_S = 15.0


def destination(requested: str | None) -> str:
    root = f"{requested or '/sdcard/aua'}.aua-recording-{uuid.uuid4().hex}"
    _validate_root(root)
    return root


def _validate_root(root: str) -> None:
    if not _ROOT.fullmatch(root) or ".." in root.split("/"):
        raise DeviceError("invalid owned recording directory", code="recording_identity_invalid")


def supervisor_script() -> str:
    # /proc/uptime is monotonic across device clock changes. Keep encoder timestamps;
    # do not assume an idle tail was captured. No sampling or artificial frames.
    return r'''#!/system/bin/sh
umask 077
root=$1
total=$2
now() { read uptime rest < /proc/uptime; echo "$uptime"; }
cd "$root" || exit 1
child=
stop() { touch stop; [ -z "$child" ] || kill -2 "$child" 2>/dev/null; }
trap stop INT TERM
echo $$ > supervisor.pid
started=$(now)
deadline=$((${started%.*} + total))
i=0
reason=stopped
while [ ! -e stop ]; do
    stamp=$(now)
    remaining=$((deadline - ${stamp%.*}))
    if [ "$remaining" -le 0 ]; then reason=duration_limit; break; fi
    limit=180
    [ "$remaining" -ge 180 ] || limit=$remaining
    echo "begin $i $stamp" >> events
    screenrecord --time-limit "$limit" "$root/segment-$i.mp4" > "segment-$i.log" 2>&1 &
    child=$!
    wait "$child"
    result=$?
    # A trap can interrupt wait before screenrecord commits its muxer. Wait again.
    if [ -e stop ]; then wait "$child" 2>/dev/null; fi
    child=
    echo "end $i $(now) $result" >> events
    if [ "$result" -ne 0 ] && [ ! -e stop ]; then reason=encoder_failed; break; fi
    i=$((i + 1))
done
echo "finish $(now) $reason" >> events
'''


def _state(dev: Uiautomator2Device) -> dict[str, Any] | None:
    path = dev._recording_state_path()
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DeviceError("recording state is unreadable", code="recording_status_unknown") from exc
    if not isinstance(state, dict):
        raise DeviceError("recording state is invalid", code="recording_status_unknown")
    if state.get("mode") == "native_segments":
        _validate_root(str(state.get("remote", "")))
        if not state.get("boot_id"):
            raise DeviceError("recording boot identity is missing", code="recording_identity_mismatch")
        return state
    if _ROOT.fullmatch(str(state.get("remote", ""))):
        raise DeviceError("native recording state is incomplete", code="recording_status_unknown")
    return None


def _write_state(dev: Uiautomator2Device, state: dict[str, Any]) -> None:
    path = dev._recording_state_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)


def _process_table(dev: Uiautomator2Device) -> list[tuple[int, str]]:
    # Toybox can truncate ARGS even in a pipe. Request wide output, then resolve every
    # shell/encoder candidate from /proc in ONE batched transport call, never per PID.
    output = dev.shell("ps -A -w -o PID,ARGS")
    lines = output.splitlines()
    if not lines or lines[0].split() != ["PID", "ARGS"]:
        raise DeviceError("recording process inspection failed", code="recording_status_unknown")
    table: dict[int, str] = {}
    candidates = []
    for line in lines[1:]:
        fields = line.split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) in table:
            raise DeviceError("ambiguous recording process inspection", code="recording_status_unknown")
        pid, command = int(fields[0]), fields[1]
        table[pid] = command
        executable = command.split()[0].rsplit("/", 1)[-1].strip("[]")
        if executable in {"sh", "screenrecord"}:
            candidates.append(pid)
    if candidates:
        pids = " ".join(str(pid) for pid in candidates)
        script = (
            "# AUA_RECORDING_CMDLINES\n"
            f"for pid in {pids}; do "
            'if [ -r /proc/$pid/cmdline ]; then '
            'command=$(tr "\\000\\n" "  " < /proc/$pid/cmdline) || '
            '{ echo "$pid AUA_UNKNOWN"; continue; }; '
            'if [ -n "$command" ]; then printf "%s %s\\n" "$pid" "$command"; '
            'else stat=$(cat /proc/$pid/stat) || { echo "$pid AUA_UNKNOWN"; continue; }; '
            'printf "%s AUA_EMPTY %s\\n" "$pid" "$stat"; fi; '
            'elif [ ! -d /proc/$pid ]; then echo "$pid AUA_GONE"; '
            'else echo "$pid AUA_UNKNOWN"; fi; done; echo AUA_CMDLINES_COMPLETE'
        )
        resolved = dev.shell(script).splitlines()
        if not resolved or resolved[-1] != "AUA_CMDLINES_COMPLETE":
            raise DeviceError("full recording command lines unavailable", code="recording_status_unknown")
        seen = set()
        for line in resolved[:-1]:
            fields = line.split(None, 1)
            if (len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) not in candidates
                    or int(fields[0]) in seen or "AUA_UNKNOWN" in fields[1]):
                raise DeviceError("ambiguous recording command line", code="recording_status_unknown")
            pid, command = int(fields[0]), fields[1].strip()
            seen.add(pid)
            if command == "AUA_GONE":
                table.pop(pid)
            elif command.split()[0] == "AUA_EMPTY":
                # Empty cmdline alone also occurs during races/permission failures. Only
                # a matching, well-formed stat with zombie/dead state proves a non-writer.
                stat = command.partition(" ")[2]
                if not re.fullmatch(rf"{pid} \(.+\) [ZX](?: -?\d+){{19,}}", stat):
                    raise DeviceError("empty recording command line has unverified process state",
                                      code="recording_status_unknown")
                table.pop(pid)
            else:
                table[pid] = command
        if seen != set(candidates):
            raise DeviceError("incomplete recording command lines", code="recording_status_unknown")
    return list(table.items())


def _recording_role(command: str, root: str) -> str | None:
    # A path argument alone does not establish ownership: cat/tail and `sh -c`
    # readers can reference our footage without being part of the recorder.
    argv = command.split()
    if not argv:
        return None
    executable = argv[0].rsplit("/", 1)[-1]
    if (executable == "sh" and len(argv) == 4
            and argv[1:3] == [f"{root}/supervisor.sh", root] and argv[3].isdigit()):
        return "supervisor"
    if (executable == "screenrecord" and len(argv) > 1
            and re.fullmatch(re.escape(root) + r"/segment-\d+\.mp4", argv[-1])):
        return "encoder"
    return None


def _processes(dev: Uiautomator2Device, root: str) -> list[tuple[int, str]]:
    return [(pid, command) for pid, command in _process_table(dev)
            if _recording_role(command, root) is not None]


def _read_identity(dev: Uiautomator2Device, root: str) -> dict[str, Any]:
    try:
        identity = json.loads(dev.shell(f"cat {shlex.quote(root + '/identity.json')}"))
    except (ValueError, OSError) as exc:
        raise DeviceError("recording directory ownership is unverified", code="recording_identity_mismatch") from exc
    if (not isinstance(identity, dict) or identity.get("mode") != "native_segments"
            or identity.get("remote") != root or not _valid_boot(identity.get("boot_id"))):
        raise DeviceError("recording directory ownership is unverified", code="recording_identity_mismatch")
    return identity


def _valid_boot(value: Any) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _absent_directory(dev: Uiautomator2Device, root: str) -> bool:
    quoted, parent = shlex.quote(root), shlex.quote(str(Path(root).parent))
    # A failed traversal/permission check must not prove absence.
    return dev.shell(
        f"if [ -d {parent} ] && [ -x {parent} ] && [ ! -e {quoted} ] && [ ! -L {quoted} ]; then "
        f"missing=$(LC_ALL=C ls -ld {quoted} 2>&1); "
        'case "$missing" in *"No such file or directory"*) echo AUA_ABSENT;; esac; fi'
    ).strip() == "AUA_ABSENT"


def archive_stale(dev: Uiautomator2Device, root: str, boot: str) -> str | None:
    """Quarantine metadata only. Prior device footage and its pending undo are retained."""
    _validate_root(root)
    current = dev.instance_token()
    if not _valid_boot(boot) or not _valid_boot(current):
        raise DeviceError("recording belongs to another boot or boot identity is unverified", code="recording_identity_mismatch")
    if current == boot:
        return None
    if _processes(dev, root):
        raise DeviceError("prior recording path is active on current boot", code="recording_identity_mismatch")
    if not _absent_directory(dev, root) and _read_identity(dev, root)["boot_id"] != boot:
        raise DeviceError("prior recording ownership is unverified", code="recording_identity_mismatch")
    if dev.instance_token() != current:
        raise DeviceError("recording boot changed during inspection", code="recording_identity_mismatch")
    state = _state(dev)
    matching = state is not None and state.get("remote") == root and state.get("boot_id") == boot
    directory = dev._recording_state_path().parent / "quarantine"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{uuid.uuid4().hex}.json"
    with archive.open("x", encoding="utf-8") as handle:
        json.dump({"recording": state if matching else {"mode": "native_segments", "remote": root, "boot_id": boot},
                   "current_boot_id": current, "reason": "different verified boot; prior evidence retained",
                   "archived_at_s": time.time()}, handle, indent=2)
    if matching:
        dev._recording_state_path().unlink()
        dev._recording_remote = None
        dev._recording_report = None
    return str(archive)


def active_output(dev: Uiautomator2Device) -> str | None:
    for _pid, command in _process_table(dev):
        argv = command.split()
        if argv[0].rsplit("/", 1)[-1] == "screenrecord":
            if not argv[-1].startswith("/") or not argv[-1].endswith(".mp4"):
                raise DeviceError("active recording destination is ambiguous", code="recording_status_unknown")
            return argv[-1]
    return None


def recover(dev: Uiautomator2Device, *, quarantine_stale: bool = False) -> dict[str, Any] | None:
    state = _state(dev)
    if state:
        if not _valid_boot(state.get("boot_id")) or not _valid_boot(dev.instance_token()):
            raise DeviceError("recording belongs to another boot or boot identity is unverified", code="recording_identity_mismatch")
        if state.get("boot_id") != dev.instance_token():
            if not quarantine_stale:
                raise DeviceError("recording belongs to another boot", code="recording_identity_mismatch")
            archive_stale(dev, str(state["remote"]), str(state["boot_id"]))
        else:
            return state
    roots = set()
    for _pid, command in _process_table(dev):
        for arg in command.split():
            root, _, name = arg.rpartition("/")
            if ((name == "supervisor.sh" or re.fullmatch(r"segment-\d+\.mp4", name))
                    and _ROOT.fullmatch(root) and _recording_role(command, root) is not None):
                roots.add(root)
    if len(roots) > 1:
        raise DeviceError("multiple recording owners are active", code="recording_status_unknown")
    for root in roots:
        state = _read_identity(dev, root)
        if state["boot_id"] != dev.instance_token():
            raise DeviceError("recording identity mismatch", code="recording_identity_mismatch")
        _write_state(dev, state)
        return state
    return None


def start(dev: Uiautomator2Device, root: str, *, time_limit_s: int) -> str:
    _validate_root(root)
    boot = dev.instance_token()
    if not _valid_boot(boot):
        raise DeviceError("cannot identify recording boot", code="recording_identity_unavailable")
    if not 1 <= time_limit_s <= 86400:
        raise DeviceError("recording duration must be between 1 and 86400 seconds")
    if dev.shell("command -v setsid >/dev/null 2>&1 && echo AUA_SETSID").strip() != "AUA_SETSID":
        raise DeviceError("native recording requires session detachment (setsid)", code="recording_launch_unsupported")
    state = {
        "mode": "native_segments", "remote": root, "boot_id": boot,
        "state": "starting", "max_duration_s": time_limit_s, "segment_limit_s": 180,
        "host_start_requested_at_s": time.time(), "gapless_guaranteed": False,
        "continuous_coverage_verified": False,
        "requested_start_uptime_s": _uptime(dev),
    }
    _write_state(dev, state)  # recoverable even if the initiating process dies below
    quoted = shlex.quote(root)
    identity = shlex.quote(json.dumps(state))
    created = dev.shell(
        f"umask 077; mkdir {quoted} && printf %s {identity} > {quoted}/identity.json "
        "&& echo AUA_CREATED"
    )
    if created.strip() != "AUA_CREATED":
        raise DeviceError("recording directory already exists or cannot be created")
    dev.shell(f"printf %s {shlex.quote(supervisor_script())} > {quoted}/supervisor.sh")
    launched = dev.shell(
        # Establish HUP protection BEFORE the background fork: legacy shell PTY
        # teardown can otherwise kill the child before nohup installs its handler.
        # Detach the child session so it cannot hold the legacy PTY open. Keep
        # this parent alive until the actual supervisor publishes readiness.
        # The compound command remains grammatical with adbutils' exit suffix.
        f"(trap '' HUP; nohup setsid sh {quoted}/supervisor.sh {quoted} {time_limit_s} "
        f"> {quoted}/supervisor.log 2>&1 < /dev/null & "
        f"n=0; while [ ! -s {quoted}/supervisor.pid ] && [ $n -lt 30 ]; do "
        "sleep 0.1; n=$((n+1)); done; "
        f"if [ -s {quoted}/supervisor.pid ]; then echo AUA_LAUNCHED; "
        "else echo AUA_LAUNCH_TIMEOUT; fi)"
    )
    if launched.strip() != "AUA_LAUNCHED":
        raise DeviceError("native recording launch readiness was not confirmed", code="recording_start_unverified")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        events = dev.shell(f"cat {quoted}/events 2>/dev/null")
        if "begin 0 " in events and _processes(dev, root) and dev._wait_for_remote_file(
            f"{root}/segment-0.mp4", timeout_s=0.5
        ):
            state["state"] = "recording"
            _write_state(dev, state)
            dev._recording_remote = root
            dev._recording_report = state
            return root
        time.sleep(0.1)
    # Keep the write-ahead state and ledger. A failed start can still have a live encoder.
    raise DeviceError("native recording did not confirm startup", code="recording_start_unverified")


def _uptime(dev: Uiautomator2Device) -> float:
    try:
        value = float(dev.shell("cat /proc/uptime").split()[0])
    except (ValueError, IndexError) as exc:
        raise DeviceError("target monotonic clock is unreadable", code="recording_status_unknown") from exc
    if not math.isfinite(value) or value < 0:
        raise DeviceError("invalid target monotonic clock", code="recording_status_unknown")
    return value


def quiesce(dev: Uiautomator2Device, root: str) -> float:
    """Stop rotation first; signal only exact encoder paths, then await finalization."""
    _validate_root(root)
    stopped_at = _uptime(dev)
    if not _owned_directory(dev, root):
        return stopped_at
    dev.shell(f"touch {shlex.quote(root + '/stop')}")
    deadline = time.monotonic() + _FINALIZE_S
    signalled: set[int] = set()
    while True:
        processes = _processes(dev, root)
        if not processes:
            return stopped_at
        for pid, command in processes:
            if pid in signalled or _recording_role(command, root) != "encoder":
                continue
            # Verify the exact command again on target just before signalling. A recycled
            # pid must never turn teardown into a kill of somebody else's recording.
            expected = shlex.quote(command)
            dev.shell(
                f'actual=$(tr "\\000" " " < /proc/{pid}/cmdline 2>/dev/null); '
                f'[ "${{actual% }}" != {expected} ] || kill -2 {pid}'
            )
            signalled.add(pid)
        if time.monotonic() >= deadline:
            raise DeviceError("native recorder did not finalize", code="recording_cleanup_unverified")
        time.sleep(0.1)


def media_duration(path: Path) -> float | None:
    """Read the native movie timescale/duration without decoding or retiming frames."""
    total = path.stat().st_size
    with path.open("rb") as handle:
        def boxes(start: int, end: int) -> float | None:
            offset = start
            while offset + 8 <= end:
                handle.seek(offset)
                header = handle.read(8)
                size, kind = int.from_bytes(header[:4], "big"), header[4:]
                head = 8
                if size == 1:
                    size = int.from_bytes(handle.read(8), "big")
                    head = 16
                elif size == 0:
                    size = end - offset
                if size < head or offset + size > end:
                    return None
                if kind == b"moov":
                    value = boxes(offset + head, offset + size)
                    if value is not None:
                        return value
                elif kind == b"mvhd":
                    data = handle.read(min(32, size - head))
                    if len(data) < 20 or data[0] not in (0, 1):
                        return None
                    position = 20 if data[0] == 1 else 12
                    width = 8 if data[0] == 1 else 4
                    if len(data) < position + 4 + width:
                        return None
                    scale = int.from_bytes(data[position:position + 4], "big")
                    ticks = int.from_bytes(data[position + 4:position + 4 + width], "big")
                    return ticks / scale if scale and ticks != 2 ** (8 * width) - 1 else None
                offset += size
            return None
        return boxes(0, total)


def timeline(
    events: str, paths: list[Path], *, stop_uptime_s: float,
    start_uptime_s: float | None = None,
) -> dict[str, Any]:
    segments: dict[int, dict[str, Any]] = {}
    finish: dict[str, Any] | None = None
    def event_time(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid monotonic timestamp")
        return value

    for line in events.splitlines():
        fields = line.split()
        try:
            if len(fields) == 3 and fields[0] == "begin":
                index = int(fields[1])
                if index != len(segments) or finish is not None:
                    raise ValueError("non-sequential segment")
                started = event_time(fields[2])
                if segments and started < segments[index - 1].get("end_uptime_s", float("inf")):
                    raise ValueError("overlapping or incomplete previous segment")
                segments[index] = {"index": index, "start_uptime_s": started}
            elif len(fields) == 4 and fields[0] == "end":
                segment = segments[int(fields[1])]
                ended = event_time(fields[2])
                if "end_uptime_s" in segment or ended < segment["start_uptime_s"]:
                    raise ValueError("invalid segment end")
                segment.update(end_uptime_s=ended, exit_code=int(fields[3]))
            elif len(fields) == 3 and fields[0] == "finish":
                if finish is not None or fields[2] not in {"stopped", "duration_limit", "encoder_failed"}:
                    raise ValueError("invalid lifecycle completion")
                finished = event_time(fields[1])
                if segments and finished < segments[len(segments) - 1].get("end_uptime_s", float("inf")):
                    raise ValueError("finish before segment end")
                finish = {"uptime_s": finished, "reason": fields[2]}
            else:
                raise ValueError("invalid event")
        except (ValueError, KeyError) as exc:
            raise DeviceError("recording lifecycle log is incomplete", code="recording_timeline_invalid") from exc
    gaps: list[dict[str, Any]] = []
    media_total = 0.0
    failed = not segments or finish is None or finish.get("reason") == "encoder_failed"
    first = min((s["start_uptime_s"] for s in segments.values()), default=stop_uptime_s)
    if start_uptime_s is not None:
        if first > start_uptime_s:
            gaps.append({"start_uptime_s": start_uptime_s, "end_uptime_s": first, "reason": "encoder_startup"})
        first = start_uptime_s
    previous = first
    for index, segment in sorted(segments.items()):
        start = segment["start_uptime_s"]
        end = segment.get("end_uptime_s")
        path = next((p for p in paths if p.name == f"segment-{index}.mp4"), None)
        duration = media_duration(path) if path else None
        segment.update(file=path.name if path else None, media_duration_s=duration)
        if start > previous:
            gaps.append({"start_uptime_s": previous, "end_uptime_s": start, "reason": "segment_rotation"})
        if duration is not None:
            media_total += duration
        if end is None or duration is None or abs((end - start) - duration) > 2.0:
            failed = True
            gaps.append({"start_uptime_s": start, "end_uptime_s": end, "reason": "unverified_segment_coverage"})
        previous = end if end is not None else start
    if stop_uptime_s - previous > 0:
        gaps.append({"start_uptime_s": previous, "end_uptime_s": stop_uptime_s, "reason": "recording_ended_before_stop"})
    requested = max(0.0, stop_uptime_s - first)
    failed = failed or requested - media_total > 2.0
    return {
        "mode": "native_segments", "state": "finalized", "segments": list(segments.values()),
        "requested_duration_s": requested, "media_duration_s": media_total,
        "duration_check": "failed" if failed else "passed", "duration_tolerance_s": 2.0,
        "gaps": gaps, "finish": finish, "gapless_guaranteed": False,
        "continuous_coverage_verified": False,
        "limitations": "Process timestamps bound encoder activity, not exact first/last frame times. "
        "Rotation, encoder startup and idle tails may omit or leave footage unverified. "
        "Original timestamps are retained without stretching or filling. "
        "The playable export joins captured media and does not span wall-clock gaps. "
        "A passing duration check is not proof of gapless coverage.",
    }


def _owned_directory(dev: Uiautomator2Device, root: str) -> bool:
    """A write-ahead intent alone does not authorize touching an existing directory."""
    _validate_root(root)
    if _absent_directory(dev, root):
        return False
    if _read_identity(dev, root)["boot_id"] != dev.instance_token():
        raise DeviceError("recording directory ownership is unverified", code="recording_identity_mismatch")
    return True


def _require_new_output(path: Path) -> None:
    if os.path.lexists(path):
        raise DeviceError("recording output already exists; choose a new destination", code="recording_output_exists")


def _publish_file(source: Path, destination: Path) -> None:
    """Claim a new output without requiring hardlink support from its filesystem."""
    import errno

    try:
        os.link(source, destination)
        return
    except OSError as exc:
        if exc.errno not in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV, errno.EPERM}:
            raise
    # Linux also reports EPERM for filesystems without hardlinks. Source read and
    # exclusive destination creation independently enforce copy permissions. In
    # particular, xb refuses existing files and symlinks even after a preflight race.
    # Any read/write/close failure propagates: the caller retains remote evidence.
    with source.open("rb") as original, destination.open("xb") as published:
        shutil.copyfileobj(original, published)


def export_mp4(paths: list[Path], destination: Path) -> str:
    """Export native packets, preserving within-segment timing; never fill capture gaps.

    Called only by the Android runtime. ffmpeg is optional for multiple segments and
    receives an allowlisted local-file playlist, no shell, filters or FPS override.
    The caller stages the output and retains remote evidence until publication succeeds.
    """
    from .android_device import finalized_mp4

    _require_new_output(destination)
    if not paths:
        raise DeviceError("no finalized segments to export; remote retained", code="recording_export_failed")
    if len(paths) == 1:
        with destination.open("xb") as output, paths[0].open("rb") as source:
            shutil.copyfileobj(source, output)
        return "direct_copy"
    executable = shutil.which("ffmpeg")
    if not executable:
        raise DeviceError(
            "multi-segment MP4 export requires optional host ffmpeg on PATH; "
            "remote segments and lifecycle log retained; retry record stop after making ffmpeg available",
            code="recording_export_unsupported",
        )
    parent = paths[0].parent
    if any(p.parent != parent or not re.fullmatch(r"segment-\d+\.mp4", p.name) for p in paths):
        raise DeviceError("invalid segment export paths", code="recording_export_failed")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", dir=parent, encoding="utf-8") as playlist:
        playlist.write("".join(f"file '{path.name}'\n" for path in paths))
        playlist.flush()
        try:
            result = subprocess.run(
                [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                 "-protocol_whitelist", "file", "-f", "concat", "-safe", "1", "-i", playlist.name,
                 "-map", "0:v", "-map", "0:a?", "-c", "copy", "-movflags", "+faststart", "-f", "mp4", str(destination)],
                cwd=parent, capture_output=True, text=True, timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeviceError("MP4 export failed; remote segments retained for retry", code="recording_export_failed") from exc
    if result.returncode != 0 or not finalized_mp4(destination):
        raise DeviceError("MP4 export failed; remote segments retained for retry", code="recording_export_failed")
    return "bitstream_copy_concat"


def discard(dev: Uiautomator2Device, root: str) -> None:
    if _owned_directory(dev, root):
        quiesce(dev, root)
        dev.shell(f"rm -rf {shlex.quote(root)}")
        if dev.shell(f"[ -e {shlex.quote(root)} ] || echo AUA_REMOVED").strip() != "AUA_REMOVED":
            raise DeviceError("recording artifacts remain", code="recording_cleanup_unverified")
    state = _state(dev)
    if state and state.get("remote") == root:
        dev._recording_state_path().unlink(missing_ok=True)
    dev._recording_remote = None


def _file_status(dev: Uiautomator2Device, remote: str) -> str:
    path, parent = shlex.quote(remote), shlex.quote(str(Path(remote).parent))
    result = dev.shell(
        f"if [ -d {parent} ] && [ -x {parent} ]; then "
        f"if [ -f {path} ] && [ -r {path} ]; then echo AUA_FILE_PRESENT; "
        f"elif [ -e {path} ] || [ -L {path} ]; then echo AUA_FILE_UNKNOWN; "
        f"else missing=$(LC_ALL=C ls -ld {path} 2>&1); "
        'case "$missing" in *"No such file or directory"*) echo AUA_FILE_MISSING;; '
        '*) echo AUA_FILE_UNKNOWN;; esac; fi; else echo AUA_FILE_UNKNOWN; fi'
    ).strip()
    if result not in {"AUA_FILE_PRESENT", "AUA_FILE_MISSING"}:
        raise DeviceError("recording file inspection failed; remote retained", code="recording_status_unknown")
    return "present" if result == "AUA_FILE_PRESENT" else "missing"


def stop(dev: Uiautomator2Device, local_path: str, state: dict[str, Any]) -> str:
    from .android_device import finalized_mp4

    root = str(state["remote"])
    _validate_root(root)
    # Resolve the parent only: a dangling output symlink must count as existing evidence.
    requested = Path(local_path).expanduser().absolute()
    destination = requested.parent.resolve() / requested.name
    manifest = destination.with_name(destination.name + ".recording.json")
    output = destination.with_name(destination.name + ".segments")
    for path in (destination, manifest, output):
        _require_new_output(path)
    if "requested_stop_uptime_s" not in state:
        state["requested_stop_uptime_s"] = _uptime(dev)
        _write_state(dev, state)
    stop_time = float(state["requested_stop_uptime_s"])
    quiesce(dev, root)
    events = dev.shell(f"cat {shlex.quote(root + '/events')}")
    indices = [int(m.group(1)) for m in re.finditer(r"^begin (\d+) ", events, re.MULTILINE)]
    # Quiesce proved all writers exited. Failed/incomplete lifecycle permits salvage;
    # a transfer error or a corrupt download after a clean stop remains retryable.
    failed_encoder = (not re.search(r"^finish ", events, re.MULTILINE)
                      or bool(re.search(r"^finish [^ ]+ encoder_failed$", events, re.MULTILINE)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aua-recording-", dir=destination.parent) as temp:
        staged = Path(temp) / "segments"
        staged.mkdir()
        paths = []
        originals = []
        statuses = {}
        for index in indices:
            path = staged / f"segment-{index}.mp4"
            if failed_encoder and _file_status(dev, f"{root}/{path.name}") == "missing":
                statuses[index] = "missing"
                continue
            dev._call("pull", f"{root}/{path.name}", str(path))
            originals.append(path)
            if not finalized_mp4(path):
                if not failed_encoder:
                    raise DeviceError("native segment is not a finalized MP4; remote retained", code="recording_segment_incomplete")
                statuses[index] = "corrupt"
                continue
            statuses[index] = "finalized"
            paths.append(path)
        report = timeline(events, paths, stop_uptime_s=stop_time,
                          start_uptime_s=state.get("requested_start_uptime_s"))
        report.update(segment_directory=str(output), manifest=str(manifest),
                      max_duration_s=state.get("max_duration_s"),
                      host_start_requested_at_s=state.get("host_start_requested_at_s"))
        for segment in report["segments"]:
            index = segment["index"]
            segment.update(status=statuses[index], exported=statuses[index] == "finalized")
            if statuses[index] == "corrupt":
                segment["original_file"] = f"segment-{index}.mp4"
        partial = any(status != "finalized" for status in statuses.values()) or not paths
        report.update(cleanup_pending=partial, retained_remote=root if partial else None)
        if partial:
            report["duration_check"] = "failed"
        playable = Path(temp) / "export.mp4"
        method = export_mp4(paths, playable) if paths else "unavailable"
        report["export"] = {
            "path": str(destination) if paths else None, "method": method, "covers_wall_clock_gaps": False,
            "media_duration_s": media_duration(playable) if paths else None,
            "segment_indices": [segment["index"] for segment in report["segments"] if segment["exported"]],
            "stream_policy": ("Video and optional audio copied; non-media tracks retained in original segments only."
                              if method == "bitstream_copy_concat" else
                              "Original file copied unchanged." if method == "direct_copy" else
                              "No playable streams exported."),
            "timing": "Native packet timing within each segment; segments joined in index order. "
            "Inter-segment wall-clock gaps are omitted, not captured or filled.",
        }
        staged_manifest = Path(temp) / "manifest.json"
        staged_manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Claim each final name exclusively, including races after the preflight. Publish
        # the playable path last. On partial publication/cleanup failure remote evidence
        # remains retryable at a new destination; never overwrite existing evidence.
        try:
            output.mkdir()
            for path in originals:
                _publish_file(path, output / path.name)
            _publish_file(staged_manifest, manifest)
            if paths:
                _publish_file(playable, destination)
        except OSError as exc:
            raise DeviceError(
                "recording publication failed; local/remote evidence retained; retry at a new destination",
                code="recording_export_failed",
            ) from exc
    dev._recording_report = report
    if not paths:
        raise DeviceError(
            f"no finalized segments to export; originals/diagnostics retained at {output} and {manifest}; "
            "remote evidence and undo retained", code="recording_no_playable_segments",
        )
    if not report["cleanup_pending"]:
        discard(dev, root)
    return str(destination)
