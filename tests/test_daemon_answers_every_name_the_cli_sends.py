"""Every command name the CLI can dispatch must exist in the daemon's table.

These are two namespaces that look like one. The public CLI command was renamed to
`tap-and-analyze`; the *daemon* command name it dispatches (`tap_point`) was never added. Because
`_route` deliberately raises a structured daemon error rather than falling back in-process — "it is
the answer, so it must not be swallowed" — a name the daemon does not know is a hard failure, not a
slow path. So the feature works when tested cold and fails for every agent following the skill
guide's advice to run a warm daemon.

Found the honest way: a sweep4 runner hit `unknown_command: tap_point` on a real device and dropped
to `adb shell input tap`, losing the recorded step. Diffing the two sets then turned up a *second*
missing name, `input_text`, that nobody had reported — which is the argument for checking the whole
set rather than the one instance someone tripped over.

Static on purpose: it needs no device, no daemon and no emulator, so it runs in CI on every change.
"""

from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "android_ui_analyser"


def _names_the_cli_sends() -> set[str]:
    cli = (SRC / "cli.py").read_text()
    return set(re.findall(r'_route\(\s*engine,\s*"([a-z_]+)"', cli, re.S))


def _names_the_daemon_answers() -> set[str]:
    dae = (SRC / "daemon.py").read_text()
    single = set(re.findall(r'cmd\s*==\s*"([a-z_]+)"', dae))
    grouped: set[str] = set()
    for group in re.findall(r'cmd\s+in\s*\(([^)]*)\)', dae):
        grouped |= set(re.findall(r'"([a-z_]+)"', group))
    return single | grouped


def test_the_daemon_answers_every_name_the_cli_can_send() -> None:
    sent, handled = _names_the_cli_sends(), _names_the_daemon_answers()
    missing = sorted(sent - handled)
    assert not missing, (
        "the CLI dispatches these names but the daemon has no branch for them, so each fails with "
        f"`unknown_command` whenever a daemon is warm: {missing}"
    )


def test_the_scan_can_actually_see_something() -> None:
    """Guard the guard: an empty scan would make the check above vacuously green."""
    sent = _names_the_cli_sends()
    assert len(sent) > 20, f"only found {len(sent)} dispatched names — the regex has drifted"
    assert "tap_point" in sent, "the name this test was written for is no longer being dispatched"
    assert "analyze" in _names_the_daemon_answers(), "daemon scan found no analyze branch"
