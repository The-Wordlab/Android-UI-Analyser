"""The soft lints fire on your own last command, not on whoever else is driving this device.

The journal is per-device, and running several agents against one host is a supported setup, so
all of them append to the same file. Both lints read the newest entries and assumed the newest
one was theirs.

Measured 2026-08-10: an agent's very first `analyze` was told "redundant analyze right after
wait". The `wait` belonged to a different process and had run minutes earlier. The agent reported
the warning as misleading and guessed at "prior/shared AUA session state" — the lint had accused
it of a command it never ran. A lint that does that teaches you to ignore it, which costs more
than the lint ever saved.

`pid` cannot be the discriminator: everything routed through the warm daemon carries the daemon's
pid. The lease owner can, so the journal now records it, and entries written before it did fall
back to an age check — a gap that large is not the "immediately after" the message claims.
"""

from __future__ import annotations

import time
from typing import Any

from android_ui_analyser.cli import _SAME_TURN_MS, _same_caller


class _FakeEngine:
    def __init__(self, owner: str | None) -> None:
        self._lease_owner_resolved = owner


def _entry(*, owner: str | None = None, age_ms: float = 0.0) -> dict[str, Any]:
    event: dict[str, Any] = {"ts_ms": time.time() * 1000.0 - age_ms, "ok": True}
    if owner is not None:
        event["owner"] = owner
    return event


def test_my_own_previous_command_is_still_linted() -> None:
    assert _same_caller(_FakeEngine("agent-a:1"), _entry(owner="agent-a:1"))


def test_another_agents_command_is_not_mine() -> None:
    assert not _same_caller(_FakeEngine("agent-b:2"), _entry(owner="agent-a:1"))


def test_owner_wins_over_age() -> None:
    """A parallel agent's command is not mine even when it landed a second ago."""
    assert not _same_caller(_FakeEngine("agent-b:2"), _entry(owner="agent-a:1", age_ms=200))


def test_my_own_command_is_mine_however_long_i_took_to_think() -> None:
    fresh = _entry(owner="agent-a:1", age_ms=_SAME_TURN_MS * 5)

    assert _same_caller(_FakeEngine("agent-a:1"), fresh), "a slow model still deserves the lint"


def test_an_entry_without_an_owner_falls_back_to_age() -> None:
    engine = _FakeEngine("agent-a:1")

    assert _same_caller(engine, _entry(age_ms=1_000))
    assert not _same_caller(engine, _entry(age_ms=_SAME_TURN_MS + 1_000))


def test_an_entry_with_neither_owner_nor_timestamp_is_given_the_benefit_of_the_doubt() -> None:
    """Silently dropping the lint on an unreadable entry would regress what it does catch."""
    assert _same_caller(_FakeEngine("agent-a:1"), {"ok": True})


def test_an_unleased_caller_falls_back_to_age() -> None:
    """`--no-lease` and single-agent scripts have no owner; they keep the old behaviour."""
    engine = _FakeEngine(None)

    assert _same_caller(engine, _entry(owner="agent-a:1", age_ms=1_000))
    assert not _same_caller(engine, _entry(owner="agent-a:1", age_ms=_SAME_TURN_MS + 1_000))
