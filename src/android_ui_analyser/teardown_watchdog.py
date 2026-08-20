"""Detached per-device teardown watchdog.

Spawned by :func:`teardown.ensure_watchdog` the first time a device gets a pending undo. Polls
until the holder is provably gone, replays the undos, and exits when the ledger is empty.

Why a separate process rather than a thread in the agent's process: the failure being covered is
*that process dying*. A thread cannot survive its own SIGKILL. This one is started with
``start_new_session=True`` so it also survives the shell and the daemon that spawned it.

It is deliberately dumb and cheap — a glob, a ``kill(pid, 0)``, a sleep — so leaving one running
per dirty device costs nothing measurable, and it exits as soon as there is nothing to guard.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("android_ui_analyser.teardown_watchdog")

# Give up rather than poll forever: if the undos have not become replayable in this long, either
# the holder is genuinely still working (fine — the next command's sweep will get it) or the
# target is gone for good and the record is a museum piece.
_MAX_LIFETIME_S = 24 * 3600.0


def run_watchdog(
    *,
    serial: str,
    cache_dir: str,
    platform_name: str = "android",
    grace_s: float = 120.0,
    poll_s: float = 15.0,
    max_lifetime_s: float = _MAX_LIFETIME_S,
) -> int:
    from . import device_ledger, teardown

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info("teardown watchdog watching %s (grace %.0fs)", serial, grace_s)
    started = time.monotonic()
    platform = None

    try:
        while True:
            if not device_ledger.read_ledger(serial):
                logger.info("teardown watchdog exit: nothing pending for %s", serial)
                return 0
            if time.monotonic() - started > max_lifetime_s:
                logger.warning(
                    "teardown watchdog exit: %s still not reapable after %.0fh",
                    serial,
                    max_lifetime_s / 3600.0,
                )
                return 0
            why = device_ledger.reapable(serial, cache_dir=cache_dir, grace_s=grace_s)
            if why is not None:
                if platform is None:
                    platform = _build_platform(cache_dir, platform_name)
                if platform is None:
                    return 1
                report = teardown.reap(
                    serial, platform=platform, cache_dir=cache_dir, grace_s=grace_s
                )
                for item in report.get("undone", []):
                    logger.warning("undid %s on %s: %s", item["kind"], serial, item["result"])
                for item in report.get("failed", []):
                    logger.error("could not undo %s on %s: %s", item["kind"], serial, item["error"])
                if not device_ledger.read_ledger(serial):
                    logger.info("teardown watchdog exit: %s is clean", serial)
                    return 0
            time.sleep(max(1.0, poll_s))
    finally:
        with contextlib.suppress(OSError):
            path = teardown.watchdog_pid_path(serial)
            if path.is_file():
                path.unlink()


def _build_platform(cache_dir: str, platform_name: str):  # noqa: ANN202
    """A minimal adapter for this serial. Config is loaded, not invented, so overrides apply."""
    try:
        from .config import load_config
        from .platforms.registry import PlatformFactory

        config = load_config()
        config.cache.dir = str(Path(cache_dir).expanduser())
        # Leasing off: this process is the reaper, not a competitor for the device. It has
        # already proved, via device_ledger.reapable, that nobody holds the target.
        config.lease.enabled = False
        return PlatformFactory(config).create(platform_name)
    except Exception as exc:
        logger.error("teardown watchdog cannot build a %s adapter: %s", platform_name, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aua-teardown-watchdog")
    p.add_argument("--serial", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--platform", default="android")
    p.add_argument("--grace-s", type=float, default=120.0)
    p.add_argument("--poll-s", type=float, default=15.0)
    args = p.parse_args(argv)
    return run_watchdog(
        serial=args.serial,
        cache_dir=args.cache,
        platform_name=args.platform,
        grace_s=args.grace_s,
        poll_s=args.poll_s,
    )


if __name__ == "__main__":
    sys.exit(main())
