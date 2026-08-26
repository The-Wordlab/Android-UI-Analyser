"""Every emulator stop must record *who* asked for it.

Observed: a headless instance disappeared between one command and the next, mid-scenario, under a
worker that had been issuing commands continuously (so the idle timer had been reset all along,
and only ~55-60 min had passed against its own 3600s `--idle-stop`). The emulator's own log showed
AUA's **graceful** shutdown sequence — "Wait for emulator (pid …) 20 seconds to shutdown gracefully
before kill" — beginning at that exact second, so this was a *requested* stop, not a crash and not
an abrupt external `pkill`.

And that is where the investigation stopped. No log anywhere recorded the origin of a stop request:
not the requester's owner string, not its pid, not whether it came from a serial-scoped stop, an
owner-scoped stop, or the idle watchdog firing. A search of every log touched in the preceding 90
minutes found nothing referencing that serial and a shutdown. The stop was visible; its author was
not.

It cost a real verdict — the worker was one bounded check from separating "the daily allowance is
silently exhausted with no upsell" from "one thread got stuck". Worse, an unattributable stop cannot
be *ruled out* as coordinator error, which is exactly the ambiguity that makes a shared pool
untrustworthy: on the same night a coordinator legitimately stopped two instances belonging to a
duplicate worker, and there was no way to prove those calls had not also taken this one.

So these tests pin three things: the origin is written down durably, the code path is named, and an
owner-scoped stop states exactly which instances it matched rather than only how many died.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import emulator as emulator_mod
from android_ui_analyser import emulator_watchdog as watchdog_mod

# Outside the real emulator port range (5554-5682, even ports only), so a slip in this file's
# stubbing still cannot reach a device somebody is using.
SER_A = "emulator-9998"
SER_B = "emulator-9996"

RUNNING = [
    {"serial": SER_A, "model": "pixel", "android_version": "16", "state": "device"},
    {"serial": SER_B, "model": "pixel", "android_version": "16", "state": "device"},
]


@pytest.fixture(autouse=True)
def _no_real_kills(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Autouse: `stop` exists to terminate emulators, and this module calls it repeatedly."""
    monkeypatch.setattr(emulator_mod, "running_emulators", lambda: list(RUNNING))
    killed: list[str] = []
    monkeypatch.setattr(emulator_mod, "_adb_emu_kill", killed.append)
    # `stop` also signals recorded process groups; never let that reach a real pid.
    monkeypatch.setattr(emulator_mod.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(emulator_mod, "_kill_watchdog", lambda meta: None)
    return killed


def _write_record(cache: Path, *, instance: str, serial: str, owner: str | None) -> Path:
    directory = cache / "emulator"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{instance}.json"
    path.write_text(
        json.dumps(
            {
                "avd": instance.split(".")[0],
                "instance": instance,
                "serial": serial,
                "owner": owner,
                "pid": 4242,
                "started_by_aua": True,
                "last_activity": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _stop_log(cache: Path) -> list[dict[str, Any]]:
    path = cache / "emulator" / emulator_mod.STOP_LOG_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_serial_scoped_stop_is_written_down_with_its_origin(tmp_path: Path) -> None:
    """The record the 90-minute log search needed and did not find."""
    cache = tmp_path / "cache"
    out = emulator_mod.stop(serial=SER_A, cache_dir=cache)

    entries = _stop_log(cache)
    assert len(entries) == 1, f"exactly one stop, exactly one record: {entries!r}"
    entry = entries[0]
    assert entry["requested_via"] == "serial"
    assert entry["stopped"] == [SER_A]
    assert entry["origin"]["pid"] == os.getpid(), "the requester's pid is the whole point"
    assert entry["origin"]["requested_by"] == "cli"
    assert entry["ts"], "a stop with no timestamp cannot be lined up against the emulator log"

    # The same attribution must come back to the caller, not only reach the file.
    assert out["requested_via"] == "serial"
    assert out["origin"]["pid"] == os.getpid()


def test_the_requesters_owner_is_recorded_even_when_it_is_not_the_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A coordinator stopping a worker's device is the case that could not be ruled out.

    The requester's own `$AUA_OWNER` and the owner a stop is *scoped* to are different facts, and
    only recording the second one leaves the accident invisible.
    """
    cache = tmp_path / "cache"
    _write_record(cache, instance="pool_1.p9998", serial=SER_A, owner="worker-a")
    monkeypatch.setenv("AUA_OWNER", "coordinator")

    out = emulator_mod.stop(owner="worker-a", cache_dir=cache)

    assert out["origin"]["requester_owner"] == "coordinator"
    assert out["owner"] == "worker-a", "the scoped-to owner is a different fact from the requester"
    entry = _stop_log(cache)[0]
    assert entry["origin"]["requester_owner"] == "coordinator"
    assert entry["request"]["owner"] == "worker-a"


def test_an_owner_scoped_stop_states_which_instances_it_matched(tmp_path: Path) -> None:
    """"stopped: [one serial]" does not prove it considered only one candidate."""
    cache = tmp_path / "cache"
    _write_record(cache, instance="pool_1.p9998", serial=SER_A, owner="worker-a")
    _write_record(cache, instance="pool_2.p9996", serial=SER_B, owner="worker-b")

    out = emulator_mod.stop(owner="worker-a", cache_dir=cache)

    assert [m["serial"] for m in out["matched"]] == [SER_A]
    assert out["matched"][0]["instance"] == "pool_1.p9998"
    assert out["matched"][0]["owner"] == "worker-a"
    # And it must be visible that a second instance existed and was deliberately left alone.
    assert sorted(m["serial"] for m in out["considered"]) == sorted([SER_B, SER_A])
    assert out["stopped"] == [SER_A]


def test_an_owner_scoped_stop_that_matched_nothing_says_so_and_is_still_recorded(
    tmp_path: Path,
) -> None:
    """A no-op teardown must leave a trace too — "nobody asked" and "nothing matched" differ."""
    cache = tmp_path / "cache"
    _write_record(cache, instance="pool_2.p9996", serial=SER_B, owner="worker-b")

    out = emulator_mod.stop(owner="worker-a", cache_dir=cache)

    assert out["stopped"] == []
    assert out["matched"] == []
    assert [m["serial"] for m in out["considered"]] == [SER_B]
    entry = _stop_log(cache)[0]
    assert entry["requested_via"] == "owner-scope"
    assert entry["stopped"] == []


def test_the_idle_watchdog_names_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _no_real_kills: list[str]
) -> None:
    """The watchdog is one of the three suspects, so it must be distinguishable from the other two.

    A watchdog stop is a timeout to lengthen; a coordinator stop at the same moment is a bug. They
    were indistinguishable, which is why ~55-60 min against a 3600s idle limit could not be used to
    exonerate the watchdog.
    """
    cache = tmp_path / "cache"
    _write_record(cache, instance="pool_1.p9998", serial=SER_A, owner="worker-a")
    meta_path = cache / "emulator" / "pool_1.p9998.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["idle_timeout_s"] = 1
    meta["last_activity"] = time.time() - 9999
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(watchdog_mod, "_still_running", lambda serial, pid: True)

    assert watchdog_mod.run_watchdog(cache_dir=str(cache), instance="pool_1.p9998") == 0

    entries = _stop_log(cache)
    assert entries, "the watchdog's own stop must be recorded like any other"
    assert entries[-1]["origin"]["requested_by"] == "idle-watchdog"
    assert entries[-1]["stopped"] == [SER_A]


def test_the_audit_line_is_best_effort_and_never_breaks_a_stop(tmp_path: Path) -> None:
    """A stop must not fail because its record could not be written.

    Deliberate direction: losing the audit line is bad, but refusing to release a device because
    the log is unwritable would strand the pool — the failure this list exists to avoid.
    """
    cache = tmp_path / "cache"
    # Occupy the log's own path with a directory, so the append raises IsADirectoryError.
    (cache / "emulator" / emulator_mod.STOP_LOG_NAME).mkdir(parents=True)

    out = emulator_mod.stop(serial=SER_A, cache_dir=cache)

    assert out["ok"] is True
    assert out["stopped"] == [SER_A]
    assert out["origin"]["pid"] == os.getpid(), "the response still carries the attribution"
