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
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import cli
from android_ui_analyser import daemon as daemon_mod
from android_ui_analyser.config import Config
from android_ui_analyser.errors import DaemonBusyError, DaemonOutcomeUnknownError


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


def _route_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.daemon.enabled = True
    cfg.daemon.socket = str(tmp_path / "daemon.sock")
    cfg.cache.dir = str(tmp_path / "cache")
    cfg.device.serial = "fictional-5554"
    return cfg


def test_route_never_replays_a_request_whose_daemon_outcome_is_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _route_config(tmp_path)
    mutations: list[int] = []

    class UnknownClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def call(self, _cmd: str, **_kwargs: Any) -> dict[str, Any]:
            raise DaemonOutcomeUnknownError("response timed out")

    engine = SimpleNamespace(
        config=cfg,
        _lease_owner=None,
        _lease_owner_resolved=None,
        tap=lambda **_kwargs: mutations.append(1),
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _cfg: daemon_mod._aua_version())
    monkeypatch.setattr(daemon_mod, "DaemonClient", UnknownClient)

    with pytest.raises(DaemonOutcomeUnknownError):
        cli._route(engine, "tap", element_id=1)

    assert mutations == []


def test_route_refuses_in_process_fallback_while_daemon_pid_is_live(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _route_config(tmp_path)
    mutations: list[int] = []

    class BusyClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def call(self, _cmd: str, **_kwargs: Any) -> dict[str, Any]:
            raise OSError("busy socket")

    engine = SimpleNamespace(
        config=cfg,
        _lease_owner=None,
        _lease_owner_resolved=None,
        tap=lambda **_kwargs: mutations.append(1),
    )
    monkeypatch.setattr(daemon_mod, "is_running", lambda _cfg: True)
    monkeypatch.setattr(daemon_mod, "running_version", lambda _cfg: None)
    monkeypatch.setattr(daemon_mod, "DaemonClient", BusyClient)
    monkeypatch.setattr(daemon_mod, "process_running", lambda _cfg: True)

    with pytest.raises(DaemonBusyError):
        cli._route(engine, "tap", element_id=1)

    assert mutations == []
