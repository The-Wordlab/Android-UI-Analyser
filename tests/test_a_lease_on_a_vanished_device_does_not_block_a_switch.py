"""Asking for a device that exists must not be refused by a lease on one that does not.

Observed 2026-09-01. A lease file for `emulator-5554` survived the emulator being stopped —
process-bound leases do not age out while the owning process lives, and the owner here was a
long-running agent. Every later `aua --serial emulator-5556 shell ...` then failed with

    lease_switch_required: <owner> already leases emulator-5554;
                           switching to emulator-5556 would release the current lease

naming a device that was not attached at all. The suggested remedy — acknowledge the handoff and
release — is real work aimed at a device with nothing left to clean up.

The sticky case is deliberately *not* changed. When a caller names no device and its own target
has vanished, `LeasedTargetUnavailableError` is right: another free device is not a substitute for
a screen whose ids and app state belong to the retained lease. That is pinned by
`test_missing_sticky_target_retains_lease_without_suggesting_a_switch`. This is the opposite case:
the caller named a different device, so it has already said it is not waiting for the old one.
"""

from __future__ import annotations

import pytest

from android_ui_analyser import leases as L
from android_ui_analyser.errors import LeaseSwitchRequiredError

ONLINE = [
    ("emulator-5556", {"root": False, "play": True, "proxy": False}),
    ("emulator-5558", {"root": False, "play": True, "proxy": False}),
]


def test_a_lease_on_a_device_that_is_not_online_does_not_refuse_an_explicit_switch(
    tmp_path,
) -> None:
    """The regression itself."""

    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True

    serial, why = L.choose_device(
        tmp_path, owner="claude", explicit="emulator-5556", candidates=ONLINE
    )

    assert serial == "emulator-5556"
    assert L.holder(tmp_path, "emulator-5556") == "claude"


def test_a_lease_on_a_device_that_is_still_online_does_refuse_the_switch(tmp_path) -> None:
    """The guard still exists. A live device holds real state and must be acknowledged."""

    assert L.acquire(tmp_path, "emulator-5556", owner="claude") is True

    with pytest.raises(LeaseSwitchRequiredError) as caught:
        L.choose_device(
            tmp_path,
            owner="claude",
            explicit="emulator-5558",
            candidates=ONLINE,
        )

    assert "emulator-5556" in str(caught.value)
    assert L.holder(tmp_path, "emulator-5558") is None


def test_the_error_never_names_a_device_the_caller_cannot_see(tmp_path) -> None:
    """A held device that is online is nameable; one that is not online must not be named."""

    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    assert L.acquire(tmp_path, "emulator-5556", owner="claude", allow_additional=True) is True

    with pytest.raises(LeaseSwitchRequiredError) as caught:
        L.choose_device(
            tmp_path,
            owner="claude",
            explicit="emulator-5558",
            candidates=ONLINE,
        )

    message = str(caught.value)
    assert "emulator-5556" in message
    assert "emulator-5554" not in message, "named a device that is not attached"


def test_the_sticky_case_is_unchanged(tmp_path) -> None:
    """Naming no device must still retain a vanished lease rather than reroute."""

    from android_ui_analyser.errors import LeasedTargetUnavailableError

    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True

    with pytest.raises(LeasedTargetUnavailableError):
        L.choose_device(tmp_path, owner="claude", explicit=None, candidates=ONLINE)

    assert L.holder(tmp_path, "emulator-5554") == "claude"


def test_the_vanished_lease_is_dropped_so_the_next_unpinned_call_still_works(tmp_path) -> None:
    """Ignoring the dead lease is not enough — two leases break the very next call.

    `_choose_device_unlocked`'s sticky branch refuses an owner holding more than one device with
    "multiple legacy leases". So merely skipping the vanished serial in the switch check would
    trade one refusal for a worse one on the following unpinned command.
    """

    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True

    L.choose_device(tmp_path, owner="claude", explicit="emulator-5556", candidates=ONLINE)

    assert L.holder(tmp_path, "emulator-5554") is None
    serial, why = L.choose_device(
        tmp_path, owner="claude", explicit=None, candidates=ONLINE
    )
    assert (serial, why) == ("emulator-5556", "sticky")


def test_a_vanished_lease_belonging_to_someone_else_is_left_alone(tmp_path) -> None:
    """Only the caller's own dead lease is dropped. Another agent's stays theirs."""

    assert L.acquire(tmp_path, "emulator-5554", owner="cursor") is True

    L.choose_device(tmp_path, owner="claude", explicit="emulator-5556", candidates=ONLINE)

    assert L.holder(tmp_path, "emulator-5554") == "cursor"


def test_nothing_is_dropped_when_the_requested_device_is_also_offline(tmp_path) -> None:
    """A caller naming an absent device has not moved on; do not tidy up on its behalf."""

    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True

    from android_ui_analyser.errors import DeviceLeasedError

    # The message this raises ("lease changed while it was being selected") is a poor
    # description of "you asked for a device that is not attached", but it is what the path
    # already said and is not this change's business. Only the retained lease is asserted.
    with pytest.raises(DeviceLeasedError):
        L.choose_device(
            tmp_path, owner="claude", explicit="emulator-9999", candidates=ONLINE
        )

    assert L.holder(tmp_path, "emulator-5554") == "claude"
