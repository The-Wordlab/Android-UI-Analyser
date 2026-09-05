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
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import device_ledger
from .atomic import atomic_write_text
from .platforms.identity import LEGACY_PLATFORM, TargetLike, TargetRef, target_ref
from .platforms.options_transport import (
    encode_platform_options,
    platform_options_fingerprint,
    scrub_platform_option_environment,
)

logger = logging.getLogger(__name__)


def cleanup_complete(result: Any) -> bool:
    """Whether a :meth:`Engine.teardown_run` result proves nothing was deferred.

    Handing a device back — to the next agent, or to a human breaking a wedged lease — only
    counts if the undos actually replayed. A skipped or partial reap means the device still
    carries somebody else's proxy, clock or radio state, so the caller must keep the lease
    rather than advertise a clean device.
    """

    if not isinstance(result, dict) or not result.get("ok", False):
        return False
    reports = result.get("reports")
    if not isinstance(reports, list):
        return False
    for report in reports:
        if not isinstance(report, dict):
            return False
        skipped = report.get("skipped")
        if skipped not in (None, "", "nothing pending"):
            return False
        remaining = report.get("remaining", 0)
        if not isinstance(remaining, (int, float)) or remaining != 0:
            return False
    return True


def _connect(platform: Any, serial: str) -> Any | None:
    try:
        return platform.validate_runtime(platform.connect(serial))
    except Exception as exc:
        logger.debug("teardown could not connect to %s: %s", serial, exc)
        return None


def _active_options_fingerprint(platform: Any, platform_name: str) -> str | None:
    """Keyed identity of the adapter config currently available for recovery."""

    config = getattr(platform, "config", None)
    getter = getattr(config, "platform_options", None)
    if not callable(getter):
        return None
    options = getter(platform_name)
    if not isinstance(options, Mapping):
        return None
    return platform_options_fingerprint(options, key_dir=device_ledger.ledger_dir())


def reap(
    serial: TargetLike,
    *,
    platform: Any,
    cache_dir: str | Path | None = None,
    lease_registry_dir: str | Path | None = None,
    grace_s: float = device_ledger.DEFAULT_GRACE_S,
    force: bool = False,
    dry_run: bool = False,
    platform_name: str = LEGACY_PLATFORM,
    options_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Undo everything pending on *serial*, when it is safe to.

    ``force`` skips only the holder check — a human saying "clean this device now". It never
    skips the reboot check, because replaying a stale undo against a fresh boot is how a
    restore lands on a device that never had the mutation.
    """
    ref = target_ref(serial, platform=platform_name)
    adapter_name = getattr(platform, "name", None)
    if isinstance(adapter_name, str) and adapter_name.strip().lower() != ref.platform:
        return {
            **ref.to_json(),
            "skipped": (
                f"adapter platform {adapter_name.strip().lower()!r} does not match "
                f"recorded platform {ref.platform!r}"
            ),
            "undone": [],
            "failed": [],
            "remaining": len(device_ledger.read_ledger(ref)),
        }
    entries = device_ledger.read_ledger(ref)
    if not entries:
        return {
            **ref.to_json(),
            "skipped": "nothing pending",
            "undone": [],
            "failed": [],
        }

    try:
        active_fingerprint = options_fingerprint or _active_options_fingerprint(platform, ref.platform)
    except Exception:
        return {
            **ref.to_json(),
            "code": "platform_options_identity_unavailable",
            "skipped": "cannot verify platform options; restore the original local identity key",
            "identity_key": str(device_ledger.ledger_dir() / ".platform-options-hmac-key"),
            "undone": [], "failed": [], "remaining": len(entries),
        }
    if not device_ledger.options_match(entries, active_fingerprint):
        return {
            **ref.to_json(),
            "code": "platform_options_recovery_mismatch",
            "skipped": (
                "selected platform options do not match the adapter configuration that "
                "recorded these changes; restore both the original configuration and its "
                "local identity key"
            ),
            "identity_key": str(device_ledger.ledger_dir() / ".platform-options-hmac-key"),
            "undone": [],
            "failed": [],
            "remaining": len(entries),
        }

    why = device_ledger.reapable(
        ref,
        entries=entries,
        cache_dir=cache_dir,
        lease_registry_dir=lease_registry_dir,
        grace_s=grace_s,
    )
    if why is None and not force:
        return {
            **ref.to_json(),
            "skipped": "a live holder still owns these changes",
            "undone": [],
            "failed": [],
        }

    device = _connect(platform, ref.target_id)
    token: str | None = None
    if device is not None:
        with contextlib.suppress(Exception):
            token = device.instance_token()

    needs_device = any(
        device_ledger.UNDO_OPS[e.op].needs_target for e in entries if e.op in device_ledger.UNDO_OPS
    )
    if device is None and needs_device:
        # Being offline does not erase persistent settings. Replay only host-side residue,
        # leaving target mutations pending until the original instance can be verified.
        entries = [
            e
            for e in entries
            if e.op in device_ledger.UNDO_OPS and not device_ledger.UNDO_OPS[e.op].needs_target
        ]
        if not entries:
            return {
                **ref.to_json(),
                "skipped": "target unreachable; device-side undos deferred",
                "undone": [],
                "failed": [],
            }

    context = device_ledger.UndoContext(
        serial=ref.target_id,
        device=device,
        capability=platform.capability,
        runtime_capability=platform.runtime_capability,
        instance_token=token,
        platform=ref.platform,
    )
    report = device_ledger.replay(ref, entries=entries, context=context, dry_run=dry_run)
    report["reason"] = why or "forced"
    if report["undone"]:
        logger.info(
            "teardown on %s (%s): %s",
            ref.target_id,
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
    skip: TargetLike | None = None,
    dry_run: bool = False,
    platform_factory: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    """Reap every dirty device with no live holder.

    *skip* is the caller's own serial: its changes are live by definition, and re-checking them
    on every command would be pure cost.
    """
    out = []
    current_name = str(getattr(platform, "name", LEGACY_PLATFORM) or LEGACY_PLATFORM).lower()
    skip_ref = target_ref(skip, platform=current_name) if skip is not None else None
    adapters: dict[str, Any] = {current_name: platform}
    for ref in device_ledger.pending_targets():
        if skip_ref == ref:
            continue
        selected = adapters.get(ref.platform)
        if selected is None and platform_factory is not None:
            try:
                selected = platform_factory(ref.platform)
            except Exception as exc:
                logger.warning("cannot build %s adapter for teardown: %s", ref.platform, exc)
            else:
                adapters[ref.platform] = selected
        if selected is None:
            out.append(
                {
                    **ref.to_json(),
                    "skipped": f"no adapter available for platform {ref.platform!r}",
                    "undone": [],
                    "failed": [],
                    "remaining": len(device_ledger.read_ledger(ref)),
                }
            )
            continue
        report = reap(
            ref,
            platform=selected,
            cache_dir=cache_dir,
            lease_registry_dir=lease_registry_dir,
            grace_s=grace_s,
            dry_run=dry_run,
        )
        if report.get("undone") or report.get("failed"):
            out.append(report)
    return out


# --------------------------------------------------------------------------- the watchdog


def watchdog_pid_path(
    serial: TargetLike, *, platform: str = LEGACY_PLATFORM
) -> Path:
    ref = target_ref(serial, platform=platform)
    return device_ledger.ledger_dir() / f"{ref.storage_key}.watchdog.pid"


def _read_watchdog_registration(
    serial: TargetLike, *, platform: str = LEGACY_PLATFORM
) -> tuple[int, str] | None:
    """Read current JSON metadata, accepting the legacy file containing only a pid."""

    path = watchdog_pid_path(serial, platform=platform)
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw
    if isinstance(payload, dict):
        pid = payload.get("pid")
        fingerprint = str(payload.get("platform_options_fingerprint") or "").strip()
    else:
        try:
            pid = int(payload)
        except (TypeError, ValueError):
            return None
        fingerprint = ""
    if not isinstance(pid, int) or pid <= 1:
        return None
    return pid, fingerprint


def _write_watchdog_registration(
    serial: TargetLike,
    *,
    pid: int,
    options_fingerprint: str,
    platform: str = LEGACY_PLATFORM,
) -> None:
    ref = target_ref(serial, platform=platform)
    atomic_write_text(
        watchdog_pid_path(ref),
        json.dumps(
            {
                **ref.to_json(),
                "pid": int(pid),
                "platform_options_fingerprint": str(options_fingerprint),
            },
            sort_keys=True,
        )
        + "\n",
    )


def _clear_watchdog_registration(
    serial: TargetLike,
    *,
    pid: int,
    options_fingerprint: str,
    platform: str = LEGACY_PLATFORM,
) -> None:
    """Remove only this process's metadata, never a replacement watchdog's record."""

    ref = target_ref(serial, platform=platform)
    if _read_watchdog_registration(ref) != (int(pid), str(options_fingerprint)):
        return
    with contextlib.suppress(OSError):
        watchdog_pid_path(ref).unlink()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _is_target_watchdog(pid: int, ref: TargetRef) -> bool | None:
    """Positive process identity before any signal; ``None`` means it could not be read."""

    try:
        result = subprocess.run(  # noqa: S603
            ["ps", "-ww", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    command = (result.stdout or "").strip()
    if not command:
        return False if result.returncode == 0 else None
    with contextlib.suppress(ValueError):
        argv = shlex.split(command)
        module_matches = any(
            argv[index : index + 2] == ["-m", "android_ui_analyser.teardown_watchdog"]
            for index in range(max(0, len(argv) - 1))
        )

        def _arg(name: str) -> str | None:
            try:
                return argv[argv.index(name) + 1]
            except (ValueError, IndexError):
                return None

        return bool(
            module_matches
            and _arg("--serial") == ref.target_id
            and _arg("--platform") == ref.platform
        )
    return None


def watchdog_alive(
    serial: TargetLike, *, platform: str = LEGACY_PLATFORM
) -> int | None:
    """Pid of the live teardown watchdog for *serial*, or ``None``."""
    ref = target_ref(serial, platform=platform)
    registration = _read_watchdog_registration(ref)
    if registration is None:
        return None
    pid, _fingerprint = registration
    if not _pid_exists(pid):
        with contextlib.suppress(OSError):
            watchdog_pid_path(ref).unlink()
        return None
    identity = _is_target_watchdog(pid, ref)
    if identity is False:
        # A recycled pid is not a watchdog and must never be signalled or reused.
        with contextlib.suppress(OSError):
            watchdog_pid_path(ref).unlink()
        return None
    return pid


def _retire_watchdog(ref: TargetRef, pid: int, fingerprint: str) -> bool:
    """Stop a proven stale watchdog so another adapter config can replace it."""

    identity = _is_target_watchdog(pid, ref)
    if identity is False:
        _clear_watchdog_registration(
            ref, pid=pid, options_fingerprint=fingerprint
        )
        return True
    if identity is None:
        logger.warning(
            "cannot verify stale teardown watchdog pid %s for %s; leaving it alone",
            pid,
            ref.target_id,
        )
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError) as exc:
        logger.warning("cannot stop stale teardown watchdog pid %s: %s", pid, exc)
        return False
    deadline = time.monotonic() + 2.0
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_exists(pid):
        logger.warning("stale teardown watchdog pid %s did not stop", pid)
        return False
    _clear_watchdog_registration(ref, pid=pid, options_fingerprint=fingerprint)
    return True


def ensure_watchdog(
    serial: TargetLike,
    *,
    cache_dir: str | Path,
    lease_registry_dir: str | Path | None = None,
    platform_name: str,
    platform_options: Mapping[str, Any] | None = None,
    grace_s: float = device_ledger.DEFAULT_GRACE_S,
    poll_s: float = 15.0,
) -> int | None:
    """Make sure a detached watchdog is guarding *serial*. Returns its pid, or ``None``.

    Called by the code that records a mutation, not by device boot: a physical phone whose clock
    was moved has no emulator process to hang a watchdog off, and it is the target where a
    left-behind change hurts most, because nobody can fix it by killing an emulator.
    """
    ref = target_ref(serial, platform=platform_name)
    options_fingerprint = platform_options_fingerprint(
        platform_options or {}, key_dir=device_ledger.ledger_dir()
    )
    registration = _read_watchdog_registration(ref)
    if registration is not None:
        existing, recorded_fingerprint = registration
        if not _pid_exists(existing):
            _clear_watchdog_registration(
                ref,
                pid=existing,
                options_fingerprint=recorded_fingerprint,
            )
        elif _is_target_watchdog(existing, ref) is False:
            # Stale metadata naming an unrelated recycled pid: forget the file, never signal.
            _clear_watchdog_registration(
                ref,
                pid=existing,
                options_fingerprint=recorded_fingerprint,
            )
        elif recorded_fingerprint == options_fingerprint:
            return existing
        elif not _retire_watchdog(ref, existing, recorded_fingerprint):
            return None
    log = device_ledger.ledger_dir() / "watchdog.log"
    try:
        with tempfile.TemporaryFile(mode="w+b") as options_file:
            options_file.write(encode_platform_options(platform_options or {}))
            options_file.seek(0)
            options_fd = options_file.fileno()
            cmd = [
                sys.executable,
                "-m",
                "android_ui_analyser.teardown_watchdog",
                "--serial",
                ref.target_id,
                "--cache",
                str(cache_dir),
                "--lease-registry",
                str(lease_registry_dir or cache_dir),
                "--platform",
                ref.platform,
                "--platform-options-fd",
                str(options_fd),
                "--grace-s",
                str(float(grace_s)),
                "--poll-s",
                str(float(poll_s)),
            ]
            with open(log, "a", encoding="utf-8") as fh:  # noqa: SIM115
                proc = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=fh,
                    stderr=fh,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(options_fd,),
                    env=scrub_platform_option_environment(),
                )
    except OSError as exc:
        logger.warning("could not spawn teardown watchdog for %s: %s", ref.target_id, exc)
        return None
    with contextlib.suppress(OSError):
        _write_watchdog_registration(
            ref,
            pid=int(proc.pid),
            options_fingerprint=options_fingerprint,
        )
    return int(proc.pid)


__all__ = [
    "ensure_watchdog",
    "reap",
    "sweep",
    "watchdog_alive",
    "watchdog_pid_path",
]
