"""Proving audio is playing is a read-only question, and `aua shell` refused it.

Measured 2026-09-01 running `mini-apps-beat-painter`. The scenario's own text names the technique —
"confirming an active audio output track on the worker's device (for example
`adb -s <serial> shell dumpsys audio` / `media.audio_flinger` showing an active track for the app
while the beat plays)" — and it is how that run proved a real defect: the UI's pause icon flipped
instantly while `dumpsys audio` still reported `state:started` at +2s and +35s.

`aua shell dumpsys audio` returned `shell_mutation_refused`, because the `dumpsys` branch carries a
per-service allow-list and `audio` was not on it. So the run fell back to raw `adb shell` — exactly
the platform-boundary bypass this surface exists to prevent. A read-only surface that refuses the
sanctioned read does not make the run safer; it makes the run go around it.

A bare `dumpsys <service>` invokes that service's `dump()` and nothing else. Arguments are where
mutation lives (`battery set`, `deviceidle force-idle`), so these two services are admitted with no
arguments at all rather than the branch being opened up generally.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_ui_analyser.device import _android_shell_is_read_only  # noqa: E402


def test_a_bare_audio_dump_is_read_only() -> None:
    assert _android_shell_is_read_only(["dumpsys", "audio"])


def test_the_audio_flinger_service_is_named_too() -> None:
    """The scenario text names both, and they report different halves of the same question."""

    assert _android_shell_is_read_only(["dumpsys", "media.audio_flinger"])


def test_arguments_are_where_mutation_lives_so_they_are_refused() -> None:
    """`dumpsys battery set level 5` mutates. Admitting a service must not admit its subcommands."""

    assert not _android_shell_is_read_only(["dumpsys", "audio", "set"])
    assert not _android_shell_is_read_only(["dumpsys", "media.audio_flinger", "--reset"])


def test_an_unlearned_service_is_still_refused() -> None:
    """Allow-list, not deny-list. A service AUA has not reasoned about stays out, even bare —
    `dumpsys battery` is harmless and `dumpsys deviceidle force-idle` is not, and the branch cannot
    tell the difference from the service name alone."""

    assert not _android_shell_is_read_only(["dumpsys", "deviceidle"])
    assert not _android_shell_is_read_only(["dumpsys"])


def test_the_services_that_already_worked_still_work() -> None:
    """The existing per-service rules are unchanged, including the ones that require an argument."""

    assert _android_shell_is_read_only(["dumpsys", "package", "com.example.app"])
    assert _android_shell_is_read_only(["dumpsys", "activity", "activities"])
    assert not _android_shell_is_read_only(["dumpsys", "package"])
    assert not _android_shell_is_read_only(["dumpsys", "cpuinfo", "--reset"])
