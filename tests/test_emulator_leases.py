"""Leases: parallel agents must stop landing on the same emulator.

Three emulators attached, Claude testing search and Cursor testing delete — and both drive
`emulator-5554`, because `connect(serial=None)` takes "the only/first device". Each mutates
the screen the other is reading. Nothing errors; the results are just quietly wrong.

The two properties worth pinning are not features:

* **No deadlock is reachable.** Expiry is evaluated when a lease is read, not swept by a
  reaper, so a crashed agent leaves nothing behind that can block anyone and there is no
  cleanup step to forget. A "watchdog that frees stuck locks" would itself become a process
  that can die holding the world.
* **Identity is stable across an agent's calls.** Measured: session ids are *not* — two
  consecutive tool calls from one agent reported 40966 then 40979, because each shell
  invocation gets its own session. Keying on that would hand the agent a different emulator
  every command, turning stickiness into churn.
"""

from __future__ import annotations

import json
import os
import time

from android_ui_analyser import leases as L


def test_a_second_agent_cannot_take_a_held_device(tmp_path):
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    assert L.acquire(tmp_path, "emulator-5554", owner="cursor") is False
    assert L.holder(tmp_path, "emulator-5554") == "claude"


def test_reacquire_by_the_same_owner_is_sticky_not_a_conflict(tmp_path):
    """Every command re-acquires; that must renew, never fail."""
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    assert L.held_by(tmp_path, "claude") == ["emulator-5554"]


def test_an_expired_lease_is_not_a_lease(tmp_path):
    """A crashed agent must not hold a device forever — this is the anti-deadlock property."""
    L.acquire(tmp_path, "emulator-5556", owner="claude", ttl_s=1)
    time.sleep(1.1)
    assert L.read_lease(tmp_path, "emulator-5556") is None
    assert L.holder(tmp_path, "emulator-5556") is None
    assert L.acquire(tmp_path, "emulator-5556", owner="cursor") is True
    assert L.holder(tmp_path, "emulator-5556") == "cursor"


def test_expired_leases_are_never_reported_as_holders(tmp_path):
    L.acquire(tmp_path, "emulator-5554", owner="claude", ttl_s=1)
    L.acquire(tmp_path, "emulator-5556", owner="cursor", ttl_s=9999)
    time.sleep(1.1)
    live = {e["serial"]: e["owner"] for e in L.list_leases(tmp_path)}
    assert live == {"emulator-5556": "cursor"}
    assert L.held_by(tmp_path, "claude") == []


def test_one_agent_cannot_renew_or_release_anothers_lease(tmp_path):
    L.acquire(tmp_path, "emulator-5554", owner="claude")
    assert L.renew(tmp_path, "emulator-5554", owner="cursor") is False
    assert L.release(tmp_path, "emulator-5554", owner="cursor") is False
    assert L.holder(tmp_path, "emulator-5554") == "claude"


def test_renew_keeps_a_long_running_wait_alive(tmp_path):
    """`--until` legitimately blocks 90-120s; the lease must outlive a single call."""
    L.acquire(tmp_path, "emulator-5554", owner="claude", ttl_s=2)
    time.sleep(1.2)
    assert L.renew(tmp_path, "emulator-5554", owner="claude") is True
    time.sleep(1.2)
    assert L.holder(tmp_path, "emulator-5554") == "claude", "renew must reset the idle clock"


def test_default_ttl_outlives_the_longest_blocking_call():
    assert L.DEFAULT_TTL_S == 120


def test_agent_can_extend_its_lease_without_next_command_shrinking_it(tmp_path):
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    assert L.renew(tmp_path, "emulator-5554", owner="claude", ttl_s=600) is True
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    entry = L.read_lease(tmp_path, "emulator-5554")
    assert entry["ttl_s"] == 600


def test_release_frees_the_device_for_others(tmp_path):
    L.acquire(tmp_path, "emulator-5554", owner="claude")
    assert L.release(tmp_path, "emulator-5554", owner="claude") is True
    assert L.acquire(tmp_path, "emulator-5554", owner="cursor") is True


def test_force_acquire_replaces_a_live_holder(tmp_path):
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True
    assert L.acquire(tmp_path, "emulator-5554", owner="cursor", force=True) is True
    assert L.holder(tmp_path, "emulator-5554") == "cursor"


def test_lease_expires_immediately_when_derived_owner_process_is_gone(tmp_path, monkeypatch):
    owner = f"codex-{os.getpid()}-session-start"
    monkeypatch.setattr(L, "_proc_started", lambda pid: "session-start")
    assert L.acquire(tmp_path, "emulator-5554", owner=owner) is True
    monkeypatch.setattr(L.os, "kill", lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError()))
    assert L.read_lease(tmp_path, "emulator-5554") is None


def test_lease_records_what_it_is_for(tmp_path):
    """`--needs` and the app under test are provenance: which agent used it, for what."""
    L.acquire(tmp_path, "emulator-5554", owner="claude", needs=["root"], app="co.example.dev")
    entry = L.read_lease(tmp_path, "emulator-5554")
    assert entry["needs"] == ["root"]
    assert entry["app"] == "co.example.dev"
    assert L.idle_seconds(entry) < 5


def test_corrupt_lease_file_reads_as_free(tmp_path):
    """A half-written file must not wedge a device — free is the safe interpretation."""
    path = L.lease_dir(tmp_path) / "emulator-5554.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"serial": "emulator-5554", "own', encoding="utf-8")
    assert L.read_lease(tmp_path, "emulator-5554") is None
    assert L.acquire(tmp_path, "emulator-5554", owner="claude") is True


def test_serial_is_sanitised_into_the_filename(tmp_path):
    L.acquire(tmp_path, "192.168.1.5:5555", owner="claude")
    assert L.holder(tmp_path, "192.168.1.5:5555") == "claude"
    written = list(L.lease_dir(tmp_path).glob("*.json"))
    assert len(written) == 1
    assert "/" not in written[0].name


# --------------------------------------------------------------------------- identity


def test_explicit_owner_wins_over_everything(monkeypatch):
    monkeypatch.setenv("AUA_OWNER", "from-env")
    assert L.resolve_owner("explicit") == "explicit"


def test_env_owner_wins_over_derivation(monkeypatch):
    monkeypatch.setenv("AUA_OWNER", "from-env")
    assert L.resolve_owner(None) == "from-env"


def test_derived_identity_is_stable_within_a_process(monkeypatch):
    """The whole scheme rests on this: an unstable id makes stickiness churn."""
    monkeypatch.delenv("AUA_OWNER", raising=False)
    assert L.derive_identity() == L.derive_identity()
    assert L.resolve_owner(None) == L.derive_identity()


def test_derived_identity_is_never_empty(monkeypatch):
    monkeypatch.delenv("AUA_OWNER", raising=False)
    monkeypatch.setenv("AUA_OWNER", "   ")  # whitespace must not count as set
    assert L.resolve_owner(None).strip() != ""


def test_derived_identity_skips_shells(monkeypatch):
    """A shell is spawned per call, so keying on one would change identity every command."""
    monkeypatch.delenv("AUA_OWNER", raising=False)
    ident = L.derive_identity()
    assert not any(ident.startswith(f"{sh}-") for sh in ("zsh", "bash", "sh"))


# --------------------------------------------------------------------------- selection

import pytest  # noqa: E402

from android_ui_analyser.errors import DeviceLeasedError  # noqa: E402

CANDIDATES = [
    ("emulator-5554", {"root": True, "play": True, "proxy": True}),
    ("emulator-5556", {"root": False, "play": True, "proxy": False}),
    ("emulator-5558", {"root": False, "play": True, "proxy": False}),
]


def _choose(tmp_path, owner, explicit=None, needs=None):
    return L.choose_device(
        tmp_path, owner=owner, explicit=explicit, candidates=CANDIDATES, needs=needs
    )


def test_two_agents_get_different_devices(tmp_path):
    """The whole point: Claude on search and Cursor on delete must not share a screen."""
    a, _ = _choose(tmp_path, "claude")
    b, _ = _choose(tmp_path, "cursor")
    assert a != b


def test_an_agent_stays_on_its_device(tmp_path):
    """Element ids, app state and the learned screen map are all per-device."""
    first, why1 = _choose(tmp_path, "claude")
    second, why2 = _choose(tmp_path, "claude")
    assert (first, why1) == (second, "assigned") or why1 == "assigned"
    assert second == first
    assert why2 == "sticky"


def test_explicit_serial_on_a_held_device_is_refused_not_redirected(tmp_path):
    """Silently moving a test pinned to a device's state would invalidate it invisibly."""
    held, _ = _choose(tmp_path, "claude")
    with pytest.raises(DeviceLeasedError) as err:
        _choose(tmp_path, "cursor", explicit=held)
    assert "claude" in err.value.message
    assert err.value.hint and "free now" in err.value.hint
    assert int(err.value.exit_code) == 9


def test_leased_exit_code_is_distinct_from_device_error(tmp_path):
    """A runner must tell "busy, try another" from "nothing is reachable"."""
    from android_ui_analyser.errors import DeviceError, ExitCode

    assert int(ExitCode.LEASED) == 9
    assert int(DeviceError.exit_code) != int(DeviceLeasedError.exit_code)


def test_needs_routes_to_a_capable_device(tmp_path):
    serial, _ = _choose(tmp_path, "claude", needs=["proxy"])
    assert serial == "emulator-5554", "only 5554 is rootable, so only it can proxy HTTPS"


def test_explicit_device_that_cannot_meet_needs_is_refused(tmp_path):
    with pytest.raises(DeviceLeasedError) as err:
        _choose(tmp_path, "claude", explicit="emulator-5556", needs=["root"])
    assert "root" in err.value.message


def test_unknown_capability_is_unmet_not_ignored(tmp_path):
    """Silently satisfying `--needs jellybean` hands back a device that cannot do it."""
    assert L.unmet_needs({"root": True}, ["jellybean"]) == ["jellybean"]
    with pytest.raises(DeviceLeasedError):
        _choose(tmp_path, "claude", needs=["jellybean"])


def test_exhaustion_names_who_holds_what(tmp_path):
    """"No free device" is useless without saying which agent to go ask."""
    for owner in ("a", "b", "c"):
        _choose(tmp_path, owner)
    with pytest.raises(DeviceLeasedError) as err:
        _choose(tmp_path, "d")
    for owner in ("a", "b", "c"):
        assert owner in err.value.message


def test_an_expired_holder_frees_the_device_for_selection(tmp_path):
    """Anti-deadlock, at the selection layer: a crashed agent must not starve the pool."""
    for owner in ("a", "b", "c"):
        L.choose_device(
            tmp_path, owner=owner, explicit=None, candidates=CANDIDATES, needs=None, ttl_s=1
        )
    time.sleep(1.1)
    serial, why = _choose(tmp_path, "late")
    assert why == "assigned" and serial in dict(CANDIDATES)


def test_lease_file_is_readable_json(tmp_path):
    """Operators debug this by eye; it must not be an opaque blob."""
    L.acquire(tmp_path, "emulator-5554", owner="claude")
    raw = (L.lease_dir(tmp_path) / "emulator-5554.json").read_text(encoding="utf-8")
    entry = json.loads(raw)
    assert entry["owner"] == "claude"
    assert {"serial", "owner", "acquired", "last_activity", "ttl_s"} <= set(entry)
