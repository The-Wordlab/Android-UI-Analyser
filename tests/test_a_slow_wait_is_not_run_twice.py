"""A request carrying its own deadline gets a socket budget above it — always, not by name.

`DaemonClient.call` raised the socket timeout only for a hand-listed set of long-poll commands.
`await_predicate` — the engine call behind every global `--until` — was not in it, so it kept the
5s default. Any predicate slower than that timed out at the socket, `_route` caught the socket
error as "daemon route unavailable", and ran the entire wait a second time in-process.

Measured 2026-08-10 on one run: a `--until 'rid:<name>'` that named no element on screen was
journaled twice for the same command — `source=daemon` 31,011ms and `source=cli` 31,702ms, two
different pids. 62s of a 99s run, from one mistake that should have cost 30. Nothing surfaced the
duplication; the agent reported a single ~30s wait and had no way to see the other one.

Keying on `timeout_ms` rather than the command name removes the drift: this is the second list in
this codebase to silently fall behind the thing it was tracking.
"""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.daemon import _LONG_POLL_COMMANDS, DaemonClient


def _socket_timeout(cmd: str, **args: Any) -> float:
    """The timeout `call` would set, without opening a socket."""
    client = DaemonClient("/nonexistent.sock", timeout=5.0)
    captured: dict[str, float] = {}

    import socket as socket_mod

    class _Probe:
        def settimeout(self, value: float) -> None:
            captured["timeout"] = value

        def connect(self, _path: str) -> None:
            raise OSError("probe: the timeout is already set by here")

        def close(self) -> None:
            pass

    original = socket_mod.socket
    socket_mod.socket = lambda *a, **k: _Probe()  # type: ignore[assignment]
    try:
        with pytest.raises(OSError, match="probe"):
            client.call(cmd, **args)
    finally:
        socket_mod.socket = original  # type: ignore[assignment]
    return captured["timeout"]


def test_an_until_gets_a_budget_above_its_own_deadline() -> None:
    assert _socket_timeout("await_predicate", predicate="rid:x", timeout_ms=30_000) >= 30.0


def test_a_short_until_still_clears_its_deadline() -> None:
    assert _socket_timeout("await_predicate", predicate="rid:x", timeout_ms=8_000) >= 8.0


def test_a_deadline_below_the_default_does_not_lower_it() -> None:
    assert _socket_timeout("await_predicate", predicate="rid:x", timeout_ms=500) >= 5.0


def test_a_deadline_is_honoured_whatever_the_command_is_called() -> None:
    """The point of keying on the number is that a new long command needs no list edit."""
    assert _socket_timeout("some_future_command", timeout_ms=45_000) >= 45.0


def test_a_long_poll_without_a_deadline_still_gets_the_generous_default() -> None:
    for cmd in _LONG_POLL_COMMANDS:
        assert _socket_timeout(cmd) >= 60.0, cmd


def test_await_predicate_is_in_the_long_poll_set() -> None:
    """It is the engine call behind every `--until`; a deadline-less one must not use 5s."""
    assert "await_predicate" in _LONG_POLL_COMMANDS


def test_an_ordinary_command_keeps_the_short_default() -> None:
    assert _socket_timeout("analyze") == 5.0
