"""The warm daemon claims the device for whoever typed the command, not for itself.

`resolve_owner` falls back to `derive_identity`, which walks up to the first non-shell ancestor.
That answers one name in the CLI process and a different one inside the detached daemon, and
neither `--owner` nor `$AUA_OWNER` crossed the socket — so the daemon leased under its own name.
When the CLI had already taken the lease in-process, the daemon was then refused by its own
caller's lease, and `device_leased` named an owner the caller recognised as itself.

Measured 2026-08-10: an agent hit exactly that, tried `--owner <the named holder>`, was refused
again, and went reading `leases.py`, `cli.py`, `engine.py` and `config.py` to find out why. It
got working by setting `AUA_DAEMON__ENABLED=false` — switching the warm path off. Roughly nine
of its ten minutes went on this, against four UI commands of actual work.

The refusal itself is right; a held device must stay held. What was wrong is who the daemon
said it was, and a hint that offered `omit --serial to auto-pick` when nothing was free to pick.
"""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.daemon import DaemonClient, _adopt_client_owner
from android_ui_analyser.errors import DeviceLeasedError, UsageError
from android_ui_analyser.leases import DEFAULT_TTL_S, acquire, choose_device


class _FakeEngine:
    def __init__(self, resolved: str | None = None) -> None:
        self._lease_owner: str | None = None
        self._lease_serial: str | None = "emulator-5554" if resolved else None
        self._lease_owner_resolved: str | None = resolved
        self.claims = 0

    def _lease_device(self) -> str | None:
        self.claims += 1
        self._lease_owner_resolved = self._lease_owner
        return "emulator-5554"


def test_the_client_keeps_the_owner_it_resolved_in_its_own_process() -> None:
    assert DaemonClient("/nonexistent.sock", owner="agent-a:1")._owner == "agent-a:1"
    assert DaemonClient("/nonexistent.sock")._owner is None


def test_an_unnamed_client_sends_no_owner_and_the_daemon_keeps_its_own() -> None:
    engine = _FakeEngine(resolved="daemon-self:1")

    _adopt_client_owner(engine, None)

    assert engine.claims == 0, "an old client without the field must keep working"
    assert engine._lease_owner_resolved == "daemon-self:1"


def test_a_named_client_makes_the_daemon_reclaim_under_that_name() -> None:
    engine = _FakeEngine(resolved="daemon-self:1")

    _adopt_client_owner(engine, "cli-caller:2")

    assert engine.claims == 1, "the daemon's own lease identity must not be reused"
    assert engine._lease_owner_resolved == "cli-caller:2"


def test_the_same_owner_twice_does_not_reclaim() -> None:
    """The claim is an adb round-trip; the warm path exists to not pay it every call."""
    engine = _FakeEngine(resolved="cli-caller:2")

    _adopt_client_owner(engine, "cli-caller:2")

    assert engine.claims == 0


def test_a_connected_daemon_refuses_to_claim_a_different_serial() -> None:
    engine = _FakeEngine(resolved="daemon-self:1")
    engine._device = type("Device", (), {"serial": "emulator-5554"})()
    engine._lease_device = lambda: "emulator-5558"  # type: ignore[method-assign]

    with pytest.raises(UsageError, match="bound to emulator-5554") as caught:
        _adopt_client_owner(engine, "cli-caller:2")

    assert caught.value.code == "daemon_device_mismatch"
    assert engine._lease_serial is None


def test_a_held_device_is_still_refused(tmp_path: Any) -> None:
    acquire(tmp_path, "emulator-5554", owner="someone-else:1", ttl_s=DEFAULT_TTL_S)

    with pytest.raises(DeviceLeasedError):
        choose_device(
            tmp_path,
            owner="me:1",
            explicit="emulator-5554",
            candidates=[("emulator-5554", {})],
        )


def test_the_refusal_offers_a_move_that_can_actually_work(tmp_path: Any) -> None:
    """`omit --serial to auto-pick` lands on the no-free-device branch when nothing is free."""
    acquire(tmp_path, "emulator-5554", owner="someone-else:1", ttl_s=DEFAULT_TTL_S)

    with pytest.raises(DeviceLeasedError) as caught:
        choose_device(
            tmp_path,
            owner="me:1",
            explicit="emulator-5554",
            candidates=[("emulator-5554", {})],
        )

    hint = caught.value.hint or ""
    assert "omit --serial" not in hint, "there is nothing to auto-pick"
    assert "expires" in hint, "waiting it out is the cheapest move and was never mentioned"
    assert "aua emulator start" in hint, "booting your own is the move that always works"
    assert "--owner someone-else:1" in hint, "adopting the holder is the move for a split identity"


def test_the_free_device_hint_survives_when_one_is_free(tmp_path: Any) -> None:
    acquire(tmp_path, "emulator-5554", owner="someone-else:1", ttl_s=DEFAULT_TTL_S)

    with pytest.raises(DeviceLeasedError) as caught:
        choose_device(
            tmp_path,
            owner="me:1",
            explicit="emulator-5554",
            candidates=[("emulator-5554", {}), ("emulator-5556", {})],
        )

    assert "emulator-5556" in (caught.value.hint or ""), "routing is still the first answer"
