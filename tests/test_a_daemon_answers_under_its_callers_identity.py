"""A warm daemon must hold the lease as the caller that started it, not as a new agent.

``aua shell`` starts a daemon on demand. The daemon then acquires and renews the lease for the
work it serves.  If the daemon derives its own worker scope, that scope can differ from the
caller's — and the caller reads back a lease it no longer recognises as its own, refusing the
device it is holding.  This was a real regression: ``aua --serial X shell dumpsys audio``
answered ``device_leased`` naming the caller itself as the holder.

The cause was ``_daemon_environment`` pinning ``AUA_CACHE__DIR`` into the child even when the
launching caller had never set it, which is right for cache isolation and wrong as an identity.
So identity is passed down explicitly instead of re-derived from a directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import leases
from android_ui_analyser.config import Config
from android_ui_analyser.daemon import _daemon_environment


def _scope_under(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> str:
    """The worker scope a process launched with *env* would compute for itself."""

    for key in ("AUA_CACHE__DIR", "AUA_WORKER_SCOPE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if key in ("AUA_CACHE__DIR", "AUA_WORKER_SCOPE"):
            monkeypatch.setenv(key, value)
    return leases._worker_scope()


def test_a_daemon_started_by_a_plain_caller_carries_the_plain_caller_s_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression itself: no override in, so no scope out."""

    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    monkeypatch.delenv("AUA_WORKER_SCOPE", raising=False)
    caller_scope = leases._worker_scope()
    assert caller_scope == ""

    env = _daemon_environment(Config())

    assert _scope_under(env, monkeypatch) == caller_scope


def test_a_daemon_started_by_a_scoped_worker_carries_that_worker_s_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A parallel worker's daemon must not become a third agent on that worker's device."""

    monkeypatch.delenv("AUA_WORKER_SCOPE", raising=False)
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "worker-one"))
    caller_scope = leases._worker_scope()
    assert caller_scope

    env = _daemon_environment(Config())

    assert _scope_under(env, monkeypatch) == caller_scope


def test_two_workers_daemons_still_differ_from_each_other(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Carrying the caller's identity must not flatten two callers into one."""

    scopes = []
    for name in ("worker-one", "worker-two"):
        monkeypatch.delenv("AUA_WORKER_SCOPE", raising=False)
        monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / name))
        env = _daemon_environment(Config())
        scopes.append(_scope_under(env, monkeypatch))

    assert scopes[0] != scopes[1]


def test_the_caller_recognises_the_lease_its_own_daemon_wrote(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End of the chain: the lease the daemon persists reads back as the caller's own."""

    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    monkeypatch.delenv("AUA_WORKER_SCOPE", raising=False)
    owner = "claude-1708-2:242026"

    env = _daemon_environment(Config())
    daemon_scope = _scope_under(env, monkeypatch)
    written: dict[str, Any] = {
        "serial": "emulator-5556",
        "owner": owner,
        "owner_pid": None,
        "owner_started": None,
    }
    if daemon_scope:
        written["scope"] = daemon_scope

    monkeypatch.delenv("AUA_CACHE__DIR", raising=False)
    monkeypatch.delenv("AUA_WORKER_SCOPE", raising=False)
    assert leases._entry_matches_owner(written, owner)


def test_an_explicit_scope_survives_even_when_it_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """"Set to empty" must beat "derive from the cache directory", or the bug returns."""

    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "somewhere"))
    monkeypatch.setenv("AUA_WORKER_SCOPE", "")

    assert leases._worker_scope() == ""
