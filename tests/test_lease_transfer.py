"""A normal agent owns one sticky device; explicit handoff moves that lease safely."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest

from android_ui_analyser import leases
from android_ui_analyser.errors import (
    DeviceLeasedError,
    LeaseHandoffPendingError,
    LeaseSwitchRequiredError,
    UsageError,
)

CANDIDATES = [("emulator-5554", {}), ("emulator-5556", {})]


def _process_acquire(cache_dir: str, serial: str, barrier: Any, results: Any) -> None:
    barrier.wait()
    results.put((serial, leases.acquire(cache_dir, serial, owner="agent-a")))


def _process_hold_command(cache_dir: str, entered: Any, release: Any) -> None:
    with leases.device_command(cache_dir, "emulator-5554"):
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("parent did not release foreground command")


def _process_enter_command(cache_dir: str, attempted: Any, entered: Any) -> None:
    attempted.set()
    with leases.device_command(cache_dir, "emulator-5554"):
        entered.set()


def _process_enter_background(cache_dir: str, entered: Any) -> None:
    with leases.device_use(cache_dir, "emulator-5554"):
        entered.set()


def _process_hold_background(cache_dir: str, entered: Any, release: Any) -> None:
    with leases.device_use(cache_dir, "emulator-5554"):
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("parent did not release background reader")


def _process_hold_transaction(
    cache_dir: str,
    attempted: Any,
    order: Any,
    release: Any,
) -> None:
    attempted.set()
    with leases.device_transaction(cache_dir, "emulator-5554"):
        order.put("writer")
        if not release.wait(timeout=5):
            raise AssertionError("parent did not release ownership transition")


def _process_report_background(cache_dir: str, attempted: Any, order: Any) -> None:
    attempted.set()
    with leases.device_use(cache_dir, "emulator-5554"):
        order.put("late-reader")


def _bound_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[leases.LeaseOwner, leases.LeaseOwner, leases.LeaseOwner]:
    starts = {111: "source", 222: "child", 333: "other"}
    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    monkeypatch.setattr(leases.os, "kill", lambda _pid, _signal: None)
    return (
        leases.LeaseOwner("orchestrator", pid=111, started="source"),
        leases.LeaseOwner("child-agent", pid=222, started="child"),
        leases.LeaseOwner("other-child", pid=333, started="other"),
    )


def test_direct_acquire_cannot_give_one_owner_two_devices(tmp_path) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")

    assert not leases.acquire(tmp_path, "emulator-5556", owner="agent-a")
    assert leases.held_by(tmp_path, "agent-a") == ["emulator-5554"]


def test_acquire_cannot_succeed_without_persisting_the_lease(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_create(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only lease store")

    monkeypatch.setattr(leases, "atomic_create_text", fail_create)

    assert not leases.acquire(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.held_by(tmp_path, "agent-a") == []


def test_concurrent_acquires_leave_one_primary_lease(tmp_path) -> None:
    barrier = Barrier(2)

    def claim(serial: str) -> bool:
        barrier.wait()
        return leases.acquire(tmp_path, serial, owner="agent-a")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["emulator-5554", "emulator-5556"]))

    assert sorted(results) == [False, True]
    assert len(leases.primary_held_by(tmp_path, "agent-a")) == 1


def test_concurrent_processes_leave_one_primary_lease(tmp_path) -> None:
    pytest.importorskip("fcntl")
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    workers = [
        context.Process(
            target=_process_acquire,
            args=(str(tmp_path), serial, barrier, results),
        )
        for serial in ("emulator-5554", "emulator-5556")
    ]
    try:
        for worker in workers:
            worker.start()
        acquired = [results.get(timeout=5), results.get(timeout=5)]
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        results.close()
        results.join_thread()

    assert sorted(success for _serial, success in acquired) == [False, True]
    assert len(leases.primary_held_by(tmp_path, "agent-a")) == 1


def test_interrupted_replacement_never_becomes_an_ambiguous_bare_route(tmp_path) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.choose_device(
        tmp_path,
        owner="agent-a",
        explicit="emulator-5556",
        candidates=CANDIDATES,
        allow_replacement=True,
    ) == ("emulator-5556", "replacement_reserved")
    reserved = leases.read_lease(tmp_path, "emulator-5556")
    assert reserved is not None and reserved["role"] == "replacement"

    with pytest.raises(LeaseSwitchRequiredError, match="interrupted replacement"):
        leases.choose_device(
            tmp_path,
            owner="agent-a",
            explicit=None,
            candidates=CANDIDATES,
        )

    # Crash after the old clean/release but before promotion: the next ordinary selection can
    # safely finish the metadata transition because only the reserved target remains.
    assert leases.release(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.choose_device(
        tmp_path,
        owner="agent-a",
        explicit=None,
        candidates=CANDIDATES,
    ) == ("emulator-5556", "sticky")
    assert leases.primary_held_by(tmp_path, "agent-a") == ["emulator-5556"]


def test_replacement_cannot_be_promoted_until_the_old_primary_is_gone(tmp_path) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.acquire(
        tmp_path,
        "emulator-5556",
        owner="agent-a",
        allow_additional=True,
    )

    assert not leases.promote_replacement(tmp_path, "emulator-5556", owner="agent-a")
    assert leases.primary_held_by(tmp_path, "agent-a") == ["emulator-5554"]


def test_legacy_primary_survivor_is_an_idempotent_promotion(tmp_path) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")
    assert leases._acquire_unlocked(  # noqa: SLF001 - construct a pre-one-lease registry
        tmp_path,
        "emulator-5556",
        owner="agent-a",
        role="primary",
    )
    assert not leases.promote_replacement(tmp_path, "emulator-5554", owner="agent-a")

    assert leases.release(tmp_path, "emulator-5556", owner="agent-a")
    assert leases.promote_replacement(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.primary_held_by(tmp_path, "agent-a") == ["emulator-5554"]


def test_one_owner_cannot_accumulate_multiple_replacement_reservations(tmp_path) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.acquire(
        tmp_path,
        "emulator-5556",
        owner="agent-a",
        allow_additional=True,
    )

    assert not leases.acquire(
        tmp_path,
        "emulator-5558",
        owner="agent-a",
        allow_additional=True,
    )
    assert leases.held_by(tmp_path, "agent-a") == ["emulator-5554", "emulator-5556"]


def test_new_replacement_is_published_with_its_non_routable_role(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")

    def refuse_followup_rewrite(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replacement must not need a second publication")

    monkeypatch.setattr(leases, "_write", refuse_followup_rewrite)
    assert leases.acquire(
        tmp_path,
        "emulator-5556",
        owner="agent-a",
        allow_additional=True,
    )
    reserved = leases.read_lease(tmp_path, "emulator-5556")
    assert reserved is not None
    assert reserved["role"] == "replacement"
    assert reserved["replacement_from"] == ["emulator-5554"]


def test_release_reports_an_unlink_failure_and_keeps_the_lease(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert leases.acquire(tmp_path, "emulator-5554", owner="agent-a")
    target = leases.lease_dir(tmp_path) / "emulator-5554.json"
    original_unlink = Path.unlink

    def fail_target(path: Path, *args: object, **kwargs: object) -> None:
        if path == target:
            raise OSError("read-only lease store")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_target)

    assert not leases.release(tmp_path, "emulator-5554", owner="agent-a")
    assert leases.holder(tmp_path, "emulator-5554") == "agent-a"


def test_explicit_switch_warns_without_changing_either_lease(tmp_path) -> None:
    assert leases.choose_device(
        tmp_path,
        owner="agent-a",
        explicit=None,
        candidates=CANDIDATES,
    ) == ("emulator-5554", "assigned")

    with pytest.raises(LeaseSwitchRequiredError) as caught:
        leases.choose_device(
            tmp_path,
            owner="agent-a",
            explicit="emulator-5556",
            candidates=CANDIDATES,
        )

    assert caught.value.code == "lease_switch_required"
    assert "--replace" in (caught.value.hint or "")
    assert leases.held_by(tmp_path, "agent-a") == ["emulator-5554"]
    assert leases.holder(tmp_path, "emulator-5556") is None


def test_handoff_freezes_source_then_rebinds_same_lease_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    before = leases.read_lease(tmp_path, "emulator-5554")
    assert before is not None
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source)

    assert offered["token"].startswith("aua1_")
    with pytest.raises(LeaseHandoffPendingError):
        leases.choose_device(
            tmp_path,
            owner=source,
            explicit=None,
            candidates=CANDIDATES,
        )

    accepted = leases.accept_handoff(tmp_path, offered["token"], owner=child)
    after = leases.read_lease(tmp_path, "emulator-5554")
    assert accepted["serial"] == "emulator-5554"
    assert leases.held_by(tmp_path, source) == []
    assert leases.held_by(tmp_path, child) == ["emulator-5554"]
    assert after is not None
    assert after["generation"] != before["generation"]
    assert after["owner_pid"] == 222
    assert after["owner_started"] == "child"
    assert "handoff" not in after

    with pytest.raises(UsageError, match="invalid, expired, or already used"):
        leases.accept_handoff(tmp_path, offered["token"], owner=other)


def test_same_owner_roundtrip_rejects_work_from_the_old_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, _other = _bound_owners(monkeypatch)
    serial = "emulator-5554"
    assert leases.acquire(tmp_path, serial, owner=source)
    original = leases.read_lease(tmp_path, serial)
    assert original is not None

    to_child = leases.create_handoff(tmp_path, serial, owner=source)
    leases.accept_handoff(tmp_path, to_child["token"], owner=child)
    to_source = leases.create_handoff(tmp_path, serial, owner=child)
    leases.accept_handoff(tmp_path, to_source["token"], owner=source)

    with pytest.raises(DeviceLeasedError, match="generation changed"):
        leases.validate_use(
            tmp_path,
            serial,
            owner=source,
            expected_generation=str(original["generation"]),
            renew=False,
        )


def test_handoff_requires_a_live_process_bound_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, _child, _other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner="ttl-only-source")

    with pytest.raises(UsageError, match="process-bound source ownership"):
        leases.create_handoff(tmp_path, "emulator-5554", owner="ttl-only-source")


def test_same_process_cannot_accept_its_own_handoff_under_another_label(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _child, _other = _bound_owners(monkeypatch)
    same_process = leases.LeaseOwner("renamed-source", pid=111, started="source")
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source)

    with pytest.raises(UsageError, match="different agent process"):
        leases.accept_handoff(tmp_path, offered["token"], owner=same_process)
    assert leases.holder(tmp_path, "emulator-5554") == "orchestrator"


def test_only_one_of_two_receivers_can_accept_the_token(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source)
    barrier = Barrier(2)

    def accept(owner: leases.LeaseOwner) -> bool:
        barrier.wait()
        try:
            leases.accept_handoff(tmp_path, offered["token"], owner=owner)
        except UsageError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = list(pool.map(accept, [child, other]))

    assert sorted(accepted) == [False, True]
    assert leases.holder(tmp_path, "emulator-5554") in {"child-agent", "other-child"}


def test_handoff_waits_for_an_inflight_device_command(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, _other = _bound_owners(monkeypatch)
    serial = "emulator-5554"
    assert leases.acquire(tmp_path, serial, owner=source)
    command_started = Event()
    transfer_started = Event()
    let_command_finish = Event()

    def run_command() -> None:
        with leases.device_use(tmp_path, serial):
            leases.validate_use(tmp_path, serial, owner=source)
            command_started.set()
            assert let_command_finish.wait(timeout=2)

    def offer_transfer() -> dict[str, object]:
        transfer_started.set()
        return leases.create_handoff(tmp_path, serial, owner=source)

    with ThreadPoolExecutor(max_workers=2) as pool:
        command = pool.submit(run_command)
        assert command_started.wait(timeout=2)
        transfer = pool.submit(offer_transfer)
        assert transfer_started.wait(timeout=2)
        assert not transfer.done()
        let_command_finish.set()
        command.result(timeout=2)
        offered = transfer.result(timeout=2)

    assert leases.accept_handoff(tmp_path, str(offered["token"]), owner=child)[
        "serial"
    ] == serial


def test_foreground_commands_serialize_while_background_read_coexists(tmp_path) -> None:
    first_entered = Event()
    second_attempted = Event()
    second_entered = Event()
    background_entered = Event()
    release = Event()

    def foreground_one() -> None:
        with leases.device_command(tmp_path, "emulator-5554"):
            first_entered.set()
            assert release.wait(timeout=2)

    def foreground_two() -> None:
        second_attempted.set()
        with leases.device_command(tmp_path, "emulator-5554"):
            second_entered.set()

    def background() -> None:
        with leases.device_use(tmp_path, "emulator-5554"):
            background_entered.set()

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(foreground_one)
        assert first_entered.wait(timeout=2)
        second = pool.submit(foreground_two)
        reader = pool.submit(background)
        try:
            assert second_attempted.wait(timeout=2)
            assert background_entered.wait(timeout=2)
            assert not second_entered.is_set()
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        reader.result(timeout=2)
    assert second_entered.is_set()


def test_foreground_command_lock_is_cross_process_but_background_can_read(tmp_path) -> None:
    pytest.importorskip("fcntl")
    context = mp.get_context("spawn")
    first_entered = context.Event()
    second_attempted = context.Event()
    second_entered = context.Event()
    background_entered = context.Event()
    release = context.Event()
    workers = [
        context.Process(
            target=_process_hold_command,
            args=(str(tmp_path), first_entered, release),
        ),
        context.Process(
            target=_process_enter_command,
            args=(str(tmp_path), second_attempted, second_entered),
        ),
        context.Process(
            target=_process_enter_background,
            args=(str(tmp_path), background_entered),
        ),
    ]
    try:
        workers[0].start()
        assert first_entered.wait(timeout=5)
        workers[1].start()
        workers[2].start()
        assert second_attempted.wait(timeout=5)
        assert background_entered.wait(timeout=5)
        assert not second_entered.is_set()
        release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)

    assert second_entered.is_set()


def test_waiting_cross_process_transition_blocks_late_background_readers(tmp_path) -> None:
    fcntl = pytest.importorskip("fcntl")
    context = mp.get_context("spawn")
    first_entered = context.Event()
    release_first = context.Event()
    writer_attempted = context.Event()
    release_writer = context.Event()
    reader_attempted = context.Event()
    order = context.Queue()
    workers = [
        context.Process(
            target=_process_hold_background,
            args=(str(tmp_path), first_entered, release_first),
        ),
        context.Process(
            target=_process_hold_transaction,
            args=(str(tmp_path), writer_attempted, order, release_writer),
        ),
        context.Process(
            target=_process_report_background,
            args=(str(tmp_path), reader_attempted, order),
        ),
    ]
    try:
        workers[0].start()
        assert first_entered.wait(timeout=5)
        workers[1].start()
        assert writer_attempted.wait(timeout=5)

        gate_path = (
            leases.lease_dir(tmp_path)
            / ".locks"
            / "device-gate-emulator-5554.lock"
        )
        deadline = time.monotonic() + 5
        writer_has_turnstile = False
        while time.monotonic() < deadline:
            with gate_path.open("a+", encoding="utf-8") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    writer_has_turnstile = True
                    break
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            time.sleep(0.01)
        assert writer_has_turnstile

        workers[2].start()
        assert reader_attempted.wait(timeout=5)
        release_first.set()
        assert order.get(timeout=5) == "writer"
        release_writer.set()
        assert order.get(timeout=5) == "late-reader"
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
    finally:
        release_first.set()
        release_writer.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        order.close()
        order.join_thread()


def test_pending_handoff_survives_source_death_only_until_expiry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = {111: "source", 222: "child"}
    alive = {111, 222}

    def check_process(pid: int, _signal: int) -> None:
        if pid not in alive:
            raise ProcessLookupError

    monkeypatch.setattr(leases, "_proc_started", lambda pid: starts.get(pid, ""))
    monkeypatch.setattr(leases.os, "kill", check_process)
    source = leases.LeaseOwner("orchestrator", pid=111, started="source")
    child = leases.LeaseOwner("child", pid=222, started="child")
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    offered = leases.create_handoff(
        tmp_path,
        "emulator-5554",
        owner=source,
        ttl_s=10,
    )
    alive.remove(111)

    assert leases.read_lease(tmp_path, "emulator-5554") is not None
    accepted = leases.accept_handoff(tmp_path, offered["token"], owner=child)
    assert accepted["owner"] == "child"
    assert leases.holder(tmp_path, "emulator-5554") == "child"

    # A separate expired offer releases a dead source lazily, preserving the process-bound
    # contract once the explicitly requested transfer window ends.
    alive.add(111)
    assert leases.release(tmp_path, "emulator-5554", owner=child)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    leases.create_handoff(tmp_path, "emulator-5554", owner=source, ttl_s=1)
    second_raw = json.loads(
        (leases.lease_dir(tmp_path) / "emulator-5554.json").read_text(encoding="utf-8")
    )
    alive.remove(111)
    expires = float(second_raw.get("handoff", {}).get("expires") or 0)
    monkeypatch.setattr(leases, "_now", lambda: max(expires + 1, 10**12))
    assert leases.read_lease(tmp_path, "emulator-5554") is None


def test_live_process_bound_owner_never_loses_lease_to_idle_ttl(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _child, _other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source, ttl_s=1)
    entry = leases.read_lease(tmp_path, "emulator-5554")
    assert entry is not None
    monkeypatch.setattr(
        leases,
        "_now",
        lambda: float(entry["last_activity"]) + 10_000,
    )

    assert leases.read_lease(tmp_path, "emulator-5554") is not None


def test_expired_offer_returns_to_its_still_live_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, _other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source, ttl_s=1)
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source, ttl_s=1)
    raw = leases.read_lease(tmp_path, "emulator-5554")
    assert raw is not None
    expires = float(raw["handoff"]["expires"])
    monkeypatch.setattr(leases, "_now", lambda: expires)

    assert leases.pending_handoff(raw) is None
    with pytest.raises(UsageError, match="invalid, expired, or already used"):
        leases.accept_handoff(tmp_path, offered["token"], owner=child)
    assert leases.choose_device(
        tmp_path,
        owner=source,
        explicit=None,
        candidates=CANDIDATES,
    ) == ("emulator-5554", "sticky")
    resumed = leases.read_lease(tmp_path, "emulator-5554")
    assert resumed is not None and "handoff" not in resumed


def test_receiver_must_release_its_existing_device_before_accepting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, _other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source)
    assert leases.acquire(tmp_path, "emulator-5556", owner=child)

    with pytest.raises(LeaseSwitchRequiredError):
        leases.accept_handoff(tmp_path, offered["token"], owner=child)
    assert leases.holder(tmp_path, "emulator-5554") == "orchestrator"

    assert leases.release(tmp_path, "emulator-5556", owner=child)
    assert leases.accept_handoff(tmp_path, offered["token"], owner=child)[
        "serial"
    ] == "emulator-5554"


def test_handoff_rejects_a_ttl_only_recipient_without_consuming_token(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, _other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source)

    with pytest.raises(UsageError, match="process-bound receiving agent"):
        leases.accept_handoff(tmp_path, offered["token"], owner="ttl-only-child")

    assert leases.accept_handoff(tmp_path, offered["token"], owner=child)[
        "serial"
    ] == "emulator-5554"


def test_source_can_cancel_without_releasing_the_device(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, child, _other = _bound_owners(monkeypatch)
    assert leases.acquire(tmp_path, "emulator-5554", owner=source)
    offered = leases.create_handoff(tmp_path, "emulator-5554", owner=source)

    assert leases.cancel_handoff(tmp_path, "emulator-5554", owner=source)
    assert leases.holder(tmp_path, "emulator-5554") == "orchestrator"
    with pytest.raises(UsageError, match="invalid, expired, or already used"):
        leases.accept_handoff(tmp_path, offered["token"], owner=child)
