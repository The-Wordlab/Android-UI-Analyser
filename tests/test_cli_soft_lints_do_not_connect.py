"""Optional CLI advice must never reconnect after a daemon-routed command.

Measured 2026-08-20 during a headed QA run: deep vision finished inside the daemon in 294ms,
but the CLI emitted nothing for minutes.  The redundant-analyze lint read ``engine.device``
after routing, opening a second local device user that waited behind the daemon/session fence.
The same mistake existed in the wait lint before its routed command.

Both lints need only the selected serial to read the per-device journal.  Routing already stores
that identity on the short-lived Engine, so advice must remain a host-only journal read.
"""

from __future__ import annotations

import sys
import types

import pytest

import android_ui_analyser
import android_ui_analyser.cli as cli_mod


class _NoConnectEngine:
    def __init__(self, *, leased: str | None, configured: str | None) -> None:
        self._lease_serial = leased
        self._lease_owner_resolved = "agent-a:1"
        self.config = types.SimpleNamespace(
            cache=types.SimpleNamespace(dir="/tmp/aua-soft-lint"),
            device=types.SimpleNamespace(serial=configured),
        )
        self.device_reads = 0

    @property
    def device(self) -> object:
        self.device_reads += 1
        return types.SimpleNamespace(serial="must-not-connect")


def _journal(monkeypatch: pytest.MonkeyPatch, events: list[dict]) -> list[str | None]:
    seen: list[str | None] = []

    def read_since(_cache: object, serial: str | None, **_kwargs: object) -> list[dict]:
        seen.append(serial)
        return events

    fake_journal = types.SimpleNamespace(
        read_since=read_since,
        review_events=lambda _cache, _serial, rows, **_kwargs: rows,
    )
    monkeypatch.setattr(android_ui_analyser, "journal", fake_journal, raising=False)
    monkeypatch.setitem(sys.modules, "android_ui_analyser.journal", fake_journal)
    return seen


def test_redundant_analyze_lint_uses_routed_lease_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _journal(monkeypatch, [])
    engine = _NoConnectEngine(leased="emulator-5558", configured="emulator-5554")

    cli_mod._warn_if_redundant_analyze(engine, {"cmd": "analyze"})

    assert seen == ["emulator-5558"]
    assert engine.device_reads == 0


def test_wait_lint_uses_configured_serial_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _journal(monkeypatch, [])
    engine = _NoConnectEngine(leased=None, configured="emulator-5556")

    cli_mod._warn_if_wait_could_have_been_until(engine, None)

    assert seen == ["emulator-5556"]
    assert engine.device_reads == 0


def test_unpinned_soft_lint_stays_host_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _journal(monkeypatch, [])
    engine = _NoConnectEngine(leased=None, configured=None)

    cli_mod._warn_if_redundant_analyze(engine)

    assert seen == [None]
    assert engine.device_reads == 0


@pytest.mark.parametrize("changed", ["same", "different", "unmet", "projected_empty", "diagnostic"])
def test_redundant_read_warning_requires_matching_usable_emitted_frames(
    monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    import time

    first = {
        "cmd": "tap",
        "ok": True,
        "owner": "agent-a:1",
        "ts_ms": int(time.time() * 1000),
        "result": {
            "action": "tap",
            "observation": {
                "elements": [{"id": "el:ready", "text": "Ready"}],
                "meta": {"fingerprint": "ready-frame"},
            },
        },
    }
    latest = {
        "cmd": "analyze",
        "ok": True,
        "args": {"source": "auto"},
        "result": {"meta": {"fingerprint": "ready-frame"}},
    }
    if changed == "different":
        latest["result"]["meta"]["fingerprint"] = "changed-frame"
    elif changed == "unmet":
        first["result"]["settled_unmet"] = True
    elif changed == "projected_empty":
        first["result"]["observation"]["elements"] = []
    elif changed == "diagnostic":
        latest["args"]["fields"] = "id,text"
    _journal(monkeypatch, [first, latest])
    warnings: list[str] = []
    monkeypatch.setattr(
        cli_mod,
        "logger",
        types.SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )
    engine = _NoConnectEngine(leased="example-target", configured=None)

    cli_mod._warn_if_redundant_analyze(engine)

    assert bool(warnings) is (changed == "same")
    assert engine.device_reads == 0
