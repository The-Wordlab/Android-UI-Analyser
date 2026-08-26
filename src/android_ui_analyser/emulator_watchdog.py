"""Idle watchdog for aua-started headless emulators.

Spawned by ``aua emulator start --headless``. If no aua activity touches the instance's
serial for ``idle_timeout_s`` seconds, stops that emulator so agents that forget
``aua emulator stop --mine`` do not leave fans spinning overnight.

Meta files are keyed by *instance* (``{avd}`` or ``{avd}.p{port}`` for parallel boots).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("android_ui_analyser.emulator_watchdog")

_POLL_S = 30.0


def _meta_path(cache_dir: Path, instance: str) -> Path:
    return cache_dir.expanduser() / "emulator" / f"{instance}.json"


def _load_meta(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _last_activity(meta: dict, cache_dir: Path, serial: str | None) -> float:
    last = float(meta.get("last_activity") or meta.get("started_at") or 0.0)
    if serial:
        from . import journal as journal_mod

        jpath = journal_mod.journal_path(cache_dir, serial)
        if jpath.is_file():
            with __import__("contextlib").suppress(OSError):
                last = max(last, jpath.stat().st_mtime)
    return last


def _leased(
    cache_dir: Path,
    serial: str | None,
    *,
    lease_registry_dir: str | Path | None = None,
) -> str | None:
    """The live lease holder for *serial*, or ``None``.

    Wall-clock idleness alone is a weak reason to stop an emulator: an agent can legitimately
    sit in a 90-120s ``--until`` wait, and a long-running orchestrator can pause between steps.
    A lease is the stronger signal, and it is *fast* in the direction that matters — a lease
    whose owner process is gone reads as expired immediately, before its TTL is even consulted.
    So "idle AND unleased" means the agent is really gone, which is what makes a short idle
    timeout safe to use.
    """
    from . import leases

    try:
        entry = leases.read_lease(lease_registry_dir or cache_dir, serial) if serial else None
    except Exception:
        return "unknown"  # cannot tell — treat as held, never guess a device is free
    return str(entry.get("owner")) if entry else None


def _still_running(serial: str | None, pid: int | None) -> bool:
    if serial:
        from .emulator import running_emulators

        with __import__("contextlib").suppress(Exception):
            if any(d.get("serial") == serial for d in running_emulators()):
                return True
    if isinstance(pid, int):
        try:
            import os

            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False
    return False


def _reset_device_changes(cache: Path, serial: str | None) -> None:
    """Replay this device's pending undos while it is still reachable."""
    if not serial:
        return
    from . import device_ledger, teardown

    if not device_ledger.read_ledger(serial):
        return
    from .config import load_config
    from .platforms.registry import PlatformFactory

    config = load_config()
    config.cache.dir = str(cache)
    config.lease.enabled = False
    platform = PlatformFactory(config).create()
    report = teardown.reap(serial, platform=platform, cache_dir=cache, force=True)
    for item in report.get("undone", []):
        logger.warning("undid %s on %s: %s", item["kind"], serial, item["result"])


def run_watchdog(
    *,
    cache_dir: str,
    instance: str,
    lease_registry_dir: str | None = None,
) -> int:
    cache = Path(cache_dir).expanduser()
    path = _meta_path(cache, instance)
    lease_registry = lease_registry_dir or cache_dir
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("emulator watchdog watching instance=%s cache=%s", instance, cache)

    while True:
        meta = _load_meta(path)
        if not meta or not meta.get("started_by_aua"):
            logger.info("watchdog exit: no aua meta for %s", instance)
            return 0
        idle_s = float(meta.get("idle_timeout_s") or 0)
        if idle_s <= 0:
            logger.info("watchdog exit: idle stop disabled for %s", instance)
            return 0
        serial = meta.get("serial") if isinstance(meta.get("serial"), str) else None
        pid = meta.get("pid") if isinstance(meta.get("pid"), int) else None
        if not _still_running(serial, pid):
            path.unlink(missing_ok=True)
            logger.info("watchdog exit: emulator already gone (%s)", instance)
            return 0
        last = _last_activity(meta, cache, serial)
        idle_for = time.time() - last
        if idle_for >= idle_s:
            holder = _leased(cache, serial, lease_registry_dir=lease_registry)
            if holder is not None:
                # Idle but still leased: an agent is between steps, or paused inside a long
                # wait. Killing its emulator here would fail a test that was working.
                logger.info(
                    "idle %.0fs but %s is leased by %s — leaving it running",
                    idle_for,
                    serial,
                    holder,
                )
                time.sleep(_POLL_S)
                continue
            # Hand the device's own changes back before the device goes away: the host-side
            # residue (an orphan mitmdump holding its listen port) outlives the emulator.
            with __import__("contextlib").suppress(Exception):
                _reset_device_changes(cache, serial)
            logger.warning(
                "idle %.0fs >= %.0fs — auto-stopping aua-started headless instance %s (%s)",
                idle_for,
                idle_s,
                instance,
                serial,
            )
            # Drop our pid so stop() does not try to signal this process mid-cleanup.
            meta["watchdog_pid"] = None
            with __import__("contextlib").suppress(OSError):
                path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            from .emulator import stop_spawned_instance

            # Name ourselves in the stop record. A device that vanished under a live worker
            # could not be attributed to the watchdog or to a coordinator, and the two have
            # very different consequences: one is a timeout to lengthen, the other is a bug.
            with __import__("contextlib").suppress(Exception):
                stop_spawned_instance(
                    instance=instance,
                    pid=pid,
                    cache_dir=cache,
                    requested_by="idle-watchdog",
                    lease_registry_dir=lease_registry,
                    owner=str(meta.get("owner")) if meta.get("owner") else None,
                )
            return 0
        time.sleep(_POLL_S)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aua-emulator-watchdog")
    p.add_argument("--cache", required=True)
    p.add_argument("--lease-registry", required=True)
    p.add_argument(
        "--instance",
        help="Meta file stem ({avd} or {avd}.p{port}). Preferred over --avd.",
    )
    p.add_argument(
        "--avd",
        help="Deprecated alias for --instance (single-instance boots).",
    )
    args = p.parse_args(argv)
    instance = args.instance or args.avd
    if not instance:
        p.error("one of --instance / --avd is required")
    return run_watchdog(
        cache_dir=args.cache,
        instance=instance,
        lease_registry_dir=args.lease_registry,
    )


if __name__ == "__main__":
    sys.exit(main())
