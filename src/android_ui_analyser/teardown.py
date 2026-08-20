"""The reaper: replays :mod:`device_ledger` undos for devices nobody holds any more.

Two nets, because either one alone has a hole:

**An opportunistic sweep on every command.** Cheap (a directory glob), and it covers the common
case — an agent walks away, the next agent's first ``analyze`` cleans up before it starts. But it
only fires if *somebody* runs ``aua`` again, and the last agent of the day is exactly the one
that leaves the device dirty overnight.

**A detached watchdog per dirty device.** Spawned by the command that made the first mutation, so
the guarantee does not depend on anyone coming back. It outlives the agent, the daemon, and the
shell, polls until the holder is provably gone, replays, and exits when the ledger is empty. This
is the net for the failure the user actually hit: SIGKILL, and no further ``aua`` command ever.

Both call :func:`reap`, which is idempotent and refuses to touch a device with a live holder.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import device_ledger

logger = logging.getLogger(__name__)


def _connect(platform: Any, serial: str) -> Any | None:
    try:
        return platform.connect(serial)
    except Exception as exc:
        logger.debug("teardown could not connect to %s: %s", serial, exc)
        return None


def reap(
    serial: str,
    *,
    platform: Any,
    cache_dir: str | Path | None = None,
    lease_registry_dir: str | Path | None = None,
    grace_s: float = device_ledger.DEFAULT_GRACE_S,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Undo everything pending on *serial*, when it is safe to.

    ``force`` skips only the holder check — a human saying "clean this device now". It never
    skips the reboot check, because replaying a stale undo against a fresh boot is how a
    restore lands on a device that never had the mutation.
    """
    entries = device_ledger.read_ledger(serial)
    if not entries:
        return {"serial": serial, "skipped": "nothing pending", "undone": [], "failed": []}

    why = device_ledger.reapable(
        serial,
        entries=entries,
        cache_dir=cache_dir,
        lease_registry_dir=lease_registry_dir,
        grace_s=grace_s,
    )
    if why is None and not force:
        return {
            "serial": serial,
            "skipped": "a live holder still owns these changes",
            "undone": [],
            "failed": [],
        }

    device = _connect(platform, serial)
    token: str | None = None
    if device is not None:
        with contextlib.suppress(Exception):
            token = device.instance_token()

    needs_device = any(
        device_ledger.UNDO_OPS[e.op].needs_device for e in entries if e.op in device_ledger.UNDO_OPS
    )
    if device is None and needs_device:
        # An offline device forgot its settings anyway; the host-side residue has not. Replay
        # only what needs no target, and leave the rest for when the device comes back.
        entries = [
            e
            for e in entries
            if e.op in device_ledger.UNDO_OPS and not device_ledger.UNDO_OPS[e.op].needs_device
        ]
        if not entries:
            return {
                "serial": serial,
                "skipped": "target unreachable; device-side undos deferred",
                "undone": [],
                "failed": [],
            }

    context = device_ledger.UndoContext(
        serial=serial,
        device=device,
        capability=platform.capability,
        instance_token=token,
    )
    report = device_ledger.replay(serial, entries=entries, context=context, dry_run=dry_run)
    report["reason"] = why or "forced"
    if report["undone"]:
        logger.info(
            "teardown on %s (%s): %s",
            serial,
            report["reason"],
            ", ".join(f"{d['kind']}: {d['result']}" for d in report["undone"]),
        )
    return report


def sweep(
    *,
    platform: Any,
    cache_dir: str | Path | None = None,
    lease_registry_dir: str | Path | None = None,
    grace_s: float = device_ledger.DEFAULT_GRACE_S,
    skip: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Reap every dirty device with no live holder.

    *skip* is the caller's own serial: its changes are live by definition, and re-checking them
    on every command would be pure cost.
    """
    out = []
    for serial in device_ledger.pending_serials():
        if skip and serial == skip:
            continue
        report = reap(
            serial,
            platform=platform,
            cache_dir=cache_dir,
            lease_registry_dir=lease_registry_dir,
            grace_s=grace_s,
            dry_run=dry_run,
        )
        if report.get("undone") or report.get("failed"):
            out.append(report)
    return out


# --------------------------------------------------------------------------- the watchdog


def watchdog_pid_path(serial: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(serial))
    return device_ledger.ledger_dir() / f"{safe}.watchdog.pid"


def watchdog_alive(serial: str) -> int | None:
    """Pid of the live teardown watchdog for *serial*, or ``None``."""
    path = watchdog_pid_path(serial)
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if pid <= 1:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    except (PermissionError, OSError):
        return pid
    return pid


def ensure_watchdog(
    serial: str,
    *,
    cache_dir: str | Path,
    lease_registry_dir: str | Path | None = None,
    platform_name: str,
    grace_s: float = device_ledger.DEFAULT_GRACE_S,
    poll_s: float = 15.0,
) -> int | None:
    """Make sure a detached watchdog is guarding *serial*. Returns its pid, or ``None``.

    Called by the code that records a mutation, not by device boot: a physical phone whose clock
    was moved has no emulator process to hang a watchdog off, and it is the target where a
    left-behind change hurts most, because nobody can fix it by killing an emulator.
    """
    existing = watchdog_alive(serial)
    if existing is not None:
        return existing
    log = device_ledger.ledger_dir() / "watchdog.log"
    cmd = [
        sys.executable,
        "-m",
        "android_ui_analyser.teardown_watchdog",
        "--serial",
        str(serial),
        "--cache",
        str(cache_dir),
        "--lease-registry",
        str(lease_registry_dir or cache_dir),
        "--platform",
        str(platform_name),
        "--grace-s",
        str(float(grace_s)),
        "--poll-s",
        str(float(poll_s)),
    ]
    try:
        with open(log, "a", encoding="utf-8") as fh:  # noqa: SIM115
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=fh,
                stderr=fh,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        logger.warning("could not spawn teardown watchdog for %s: %s", serial, exc)
        return None
    with contextlib.suppress(OSError):
        watchdog_pid_path(serial).write_text(str(proc.pid), encoding="utf-8")
    return int(proc.pid)


__all__ = [
    "ensure_watchdog",
    "reap",
    "sweep",
    "watchdog_alive",
    "watchdog_pid_path",
]
