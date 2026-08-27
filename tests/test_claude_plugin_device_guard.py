"""The Claude plugin prevents the exact raw-adb/ffmpeg escape hatch AUA already covers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / ".claude-plugin" / "hooks" / "guard_device_bypass.py"


def _run(command: str) -> dict[str, object] | None:
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def _permission(command: str) -> tuple[str, str]:
    payload = _run(command)
    assert payload is not None
    output = payload["hookSpecificOutput"]  # type: ignore[index]
    return output["permissionDecision"], output["permissionDecisionReason"]  # type: ignore[index]


def test_blocks_raw_coordinate_input_even_through_a_shell_wrapper() -> None:
    decision, reason = _permission(
        "bash -lc 'adb -s emulator-5556 shell input tap 540 2010'"
    )
    assert decision == "deny"
    assert "fresh returned id/rid" in reason


def test_blocks_known_device_operations_with_an_exact_aua_lane() -> None:
    decision, reason = _permission("/opt/android/platform-tools/adb logcat -d")
    assert decision == "deny"
    assert "aua logcat mark" in reason

    decision, reason = _permission("adb pull /sdcard/aua_recording.mp4 /tmp/evidence.mp4")
    assert decision == "deny"
    assert "aua record stop" in reason


def test_unknown_raw_adb_requires_user_approval_instead_of_silently_running() -> None:
    decision, reason = _permission("adb shell cmd vendor-special probe")
    assert decision == "ask"
    assert "aua capabilities" in reason


def test_only_redirects_ffmpeg_when_the_input_is_aua_evidence() -> None:
    assert _permission("ffmpeg -i /tmp/aua_expand.mp4 /tmp/frames/out.png")[0] == "deny"
    assert _run("ffmpeg -i family-movie.mp4 family-movie.gif") is None


def test_mentions_and_host_build_commands_are_not_blocked() -> None:
    assert _run("rg -n adb README.md") is None
    assert _run("./gradlew installDevDebug") is None
