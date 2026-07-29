"""The daemon-skew fingerprint must track code, not file timestamps.

The fingerprint was the newest `.py` mtime. But the CLI and the daemon routinely run from
two different trees — an installed copy vs. the repo — and `install.sh` rewrites mtimes
without changing a line. So byte-identical code reported skew, and `_route` quietly dropped
every call onto the in-process path: a device connect per call (~6x slower) announced only
in a stderr log line that a caller reading stdout never sees.

Observed: `daemon runs aua 0.6.0+src1785356950 but this CLI is 0.6.0+src1785358302` straight
after a reinstall, with both sides on the same commit.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from android_ui_analyser import cli
from android_ui_analyser import daemon as daemon_mod


def test_touching_a_source_file_does_not_change_the_fingerprint() -> None:
    """The exact regression: a rewrite that changes mtimes but no bytes must still match."""
    victim = Path(daemon_mod.__file__)
    before = daemon_mod._source_fingerprint()
    original = victim.stat()
    try:
        os.utime(victim, (original.st_atime + 10_000, original.st_mtime + 10_000))
        assert daemon_mod._source_fingerprint() == before, "mtime must not affect the fingerprint"
    finally:
        os.utime(victim, (original.st_atime, original.st_mtime))


def test_changing_source_bytes_does_change_the_fingerprint(tmp_path: Path) -> None:
    """The property the check exists for: different code must still be detected."""
    before = daemon_mod._source_fingerprint()
    added = Path(daemon_mod.__file__).parent / "_fingerprint_probe.py"
    try:
        added.write_text("PROBE = 1\n", encoding="utf-8")
        assert daemon_mod._source_fingerprint() != before, "edited code must read as skew"
    finally:
        with contextlib.suppress(OSError):
            added.unlink()
    assert daemon_mod._source_fingerprint() == before, "removing it must restore the identity"


class FakeClient:
    def __init__(self, capturing: bool) -> None:
        self._capturing = capturing

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def call(self, cmd: str, **_: Any) -> dict[str, Any]:
        assert cmd == "capture_status"
        return {"ok": True, "result": {"running": self._capturing}}


class FakeDaemonModule:
    """Stands in for the `daemon` module: records lifecycle calls, scripts the version."""

    def __init__(self, *, capturing: bool = False, restart_matches: bool = True) -> None:
        self.capturing = capturing
        self.restart_matches = restart_matches
        self.calls: list[str] = []
        self._restarted = False

    def DaemonClient(self, _path: object, timeout: float = 0.0) -> FakeClient:  # noqa: N802
        return FakeClient(self.capturing)

    def socket_path(self, _cfg: object) -> str:
        return "/tmp/fake.sock"

    def stop(self, _cfg: object) -> dict[str, Any]:
        self.calls.append("stop")
        return {"ok": True}

    def start(self, _cfg: object, serial: str | None = None) -> dict[str, Any]:
        self.calls.append("start")
        self._restarted = True
        return {"ok": True}

    def running_version(self, _cfg: object) -> str:
        if self._restarted and self.restart_matches:
            return self._aua_version()
        return "0.6.0+srcOLD"

    def _aua_version(self) -> str:
        return "0.6.0+srcNEW"


class _Cfg:
    class device:  # noqa: N801
        serial = None


def test_a_skewed_daemon_is_restarted_rather_than_bypassed() -> None:
    fake = FakeDaemonModule()
    assert cli._replace_skewed_daemon(fake, _Cfg(), "0.6.0+srcOLD") is True
    assert fake.calls == ["stop", "start"], "the warm path is restored, not abandoned"


def test_a_live_capture_session_is_never_restarted_away() -> None:
    """Frames only exist in that process — losing them is worse than a slow call."""
    fake = FakeDaemonModule(capturing=True)
    assert cli._replace_skewed_daemon(fake, _Cfg(), "0.6.0+srcOLD") is False
    assert fake.calls == [], "a recording daemon must not be signalled"


def test_a_restart_that_does_not_resolve_skew_reports_failure() -> None:
    """If the replacement still serves other code, say so — the caller must fall back."""
    fake = FakeDaemonModule(restart_matches=False)
    assert cli._replace_skewed_daemon(fake, _Cfg(), "0.6.0+srcOLD") is False
