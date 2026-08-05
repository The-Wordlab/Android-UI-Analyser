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


def run_watchdog(*, cache_dir: str, instance: str) -> int:
    cache = Path(cache_dir).expanduser()
    path = _meta_path(cache, instance)
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
            from .emulator import stop

            # Name ourselves in the stop record. A device that vanished under a live worker
            # could not be attributed to the watchdog or to a coordinator, and the two have
            # very different consequences: one is a timeout to lengthen, the other is a bug.
            with __import__("contextlib").suppress(Exception):
                if serial:
                    stop(serial=serial, cache_dir=cache, requested_by="idle-watchdog")
                else:
                    stop(
                        avd=str(meta.get("avd") or instance),
                        cache_dir=cache,
                        requested_by="idle-watchdog",
                    )
            return 0
        time.sleep(_POLL_S)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aua-emulator-watchdog")
    p.add_argument("--cache", required=True)
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
    return run_watchdog(cache_dir=args.cache, instance=instance)


if __name__ == "__main__":
    sys.exit(main())
