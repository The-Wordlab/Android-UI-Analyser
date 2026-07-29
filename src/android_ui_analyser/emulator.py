"""Boot / list / stop Android AVDs for unattended (headless) agent runs.

``aua`` does not create system images or AVDs — that stays with ``sdkmanager`` /
``avdmanager`` / Android Studio. This module only wraps the host ``emulator`` binary
and waits until ``adb`` reports ``state=device``, so agents can verify a feature without
opening an emulator window on the user's desktop.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .errors import DeviceError, UsageError

logger = logging.getLogger(__name__)

_DEFAULT_WAIT_S = 120.0
_POLL_S = 1.0


def sdk_root() -> Path | None:
    """Resolve ``$ANDROID_HOME`` / ``$ANDROID_SDK_ROOT`` when set."""
    for key in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw).expanduser()
            if p.is_dir():
                return p
    # Common macOS default when env is unset but Android Studio is installed.
    mac = Path.home() / "Library/Android/sdk"
    if mac.is_dir():
        return mac
    return None


def emulator_bin() -> str:
    """Path to the ``emulator`` binary, or raise :class:`UsageError`."""
    which = shutil.which("emulator")
    if which:
        return which
    root = sdk_root()
    if root is not None:
        candidate = root / "emulator" / "emulator"
        if candidate.is_file():
            return str(candidate)
    raise UsageError(
        "Android emulator binary not found",
        hint="Install Android SDK emulator tools and put `emulator` on PATH "
        "(or set ANDROID_HOME). Create an AVD once with Android Studio / avdmanager.",
    )


def adb_bin() -> str:
    which = shutil.which("adb")
    if which:
        return which
    root = sdk_root()
    if root is not None:
        candidate = root / "platform-tools" / "adb"
        if candidate.is_file():
            return str(candidate)
    raise UsageError(
        "adb not found on PATH",
        hint="Install Android SDK platform-tools (or ensure `adb` is on PATH).",
    )


def _pid_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir).expanduser() / "emulator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_avds() -> dict[str, Any]:
    """Return configured AVD names (does not start anything)."""
    bin_path = emulator_bin()
    try:
        proc = subprocess.run(  # noqa: S603
            [bin_path, "-list-avds"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DeviceError(
            "could not list AVDs (`emulator -list-avds` failed)",
            hint="Check the Android SDK emulator install.",
        ) from exc
    names = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return {
        "ok": True,
        "action": "emulator-list",
        "emulator": bin_path,
        "avds": names,
        "count": len(names),
        "hint": None
        if names
        else "No AVDs yet — create one with Android Studio or `avdmanager create avd …`.",
    }


def running_emulators() -> list[dict[str, Any]]:
    """Emulator serials currently visible to adb (``emulator-*``)."""
    from .device import list_devices

    out: list[dict[str, Any]] = []
    for d in list_devices():
        serial = d.serial or ""
        if serial.startswith("emulator-"):
            out.append(
                {
                    "serial": serial,
                    "model": d.model,
                    "android_version": d.android_version,
                    "state": d.state,
                }
            )
    return out


def status(*, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """What AUA knows about host emulator tooling + live emulator serials."""
    info: dict[str, Any] = {
        "ok": True,
        "action": "emulator-status",
        "sdk_root": str(sdk_root()) if sdk_root() else None,
    }
    try:
        info["emulator"] = emulator_bin()
        info["emulator_ok"] = True
    except UsageError as exc:
        info["emulator"] = None
        info["emulator_ok"] = False
        info["emulator_error"] = exc.message
    try:
        listed = list_avds()
        info["avds"] = listed["avds"]
    except (UsageError, DeviceError):
        info["avds"] = []
    info["running"] = running_emulators()
    if cache_dir is not None:
        started = []
        for path in _pid_dir(cache_dir).glob("*.json"):
            with path.open(encoding="utf-8") as fh:
                try:
                    started.append(json.load(fh))
                except json.JSONDecodeError:
                    continue
        info["started_by_aua"] = started
    return info


def start(
    avd: str | None = None,
    *,
    headless: bool = True,
    wait_s: float = _DEFAULT_WAIT_S,
    cache_dir: str | Path,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Boot an AVD (headless by default) and wait until adb sees ``state=device``.

    If *avd* is omitted and exactly one AVD exists, that one is used. Never creates AVDs.
    Does not steal focus when ``headless=True`` (``-no-window``).
    """
    listed = list_avds()
    names: list[str] = list(listed["avds"])
    if not names:
        raise UsageError(
            "no Android Virtual Devices configured",
            hint="Create an AVD once (Android Studio Device Manager, or avdmanager), "
            "then re-run `aua emulator start --headless`.",
        )
    if avd is None:
        if len(names) != 1:
            raise UsageError(
                f"multiple AVDs available ({', '.join(names)}); pass --avd <name>",
                hint="`aua emulator list` shows names.",
            )
        avd = names[0]
    elif avd not in names:
        raise UsageError(
            f"unknown AVD {avd!r}",
            hint=f"Known AVDs: {', '.join(names)} (`aua emulator list`).",
        )

    before = {d["serial"] for d in running_emulators()}
    bin_path = emulator_bin()
    cmd = [bin_path, "-avd", avd]
    if headless:
        cmd += ["-no-window", "-no-audio", "-gpu", "swiftshader_indirect"]
    if extra_args:
        cmd += list(extra_args)

    log_path = _pid_dir(cache_dir) / f"{avd}.log"
    log_fh = open(log_path, "a")  # noqa: SIM115 — kept for child lifetime
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    meta = {
        "avd": avd,
        "pid": proc.pid,
        "headless": headless,
        "cmd": cmd,
        "log": str(log_path),
        "started_at": time.time(),
    }
    (_pid_dir(cache_dir) / f"{avd}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    serial = _wait_for_new_emulator(before, timeout_s=wait_s)
    if serial is None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGTERM)
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-800:]
        raise DeviceError(
            f"emulator {avd!r} did not become ready within {int(wait_s)}s",
            hint=f"Check the log: {log_path}\n{tail}",
        )

    meta["serial"] = serial
    (_pid_dir(cache_dir) / f"{avd}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "ok": True,
        "action": "emulator-start",
        "avd": avd,
        "serial": serial,
        "pid": proc.pid,
        "headless": headless,
        "log": str(log_path),
        "hint": f"Use `aua --serial {serial} analyze` (or `aua daemon start`) next.",
    }


def _wait_for_new_emulator(before: set[str], *, timeout_s: float) -> str | None:
    deadline = time.monotonic() + max(5.0, timeout_s)
    while time.monotonic() < deadline:
        for d in running_emulators():
            if d.get("state") == "device" and d["serial"] not in before:
                return str(d["serial"])
        if not before:
            ready = [d for d in running_emulators() if d.get("state") == "device"]
            if ready:
                return str(ready[0]["serial"])
        time.sleep(_POLL_S)
    ready = [d for d in running_emulators() if d.get("state") == "device"]
    return str(ready[0]["serial"]) if ready else None


def stop(
    *,
    serial: str | None = None,
    avd: str | None = None,
    all_devices: bool = False,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Stop a running emulator (``adb emu kill``), scoped by serial/AVD.

    An untargeted call is refused rather than treated as "all". `aua emulator stop` reads
    like "stop the one I started", but an emulator can be holding a logged-in session, a
    seeded database, or belong to something else entirely on the same machine — and killing
    it is not undoable. ``--all`` says so explicitly, matching how ``app clear`` requires
    ``--yes`` for the same reason.
    """
    targets = running_emulators()
    if not serial and avd is None and not all_devices:
        raise UsageError(
            "emulator stop needs a target: --serial, --avd, or --all",
            hint=(
                "running: "
                + (", ".join(t["serial"] for t in targets) or "none")
                + ". Killing an emulator is not undoable — it may hold a signed-in session."
            ),
        )
    if serial:
        targets = [t for t in targets if t["serial"] == serial]

    if avd is not None:
        meta_path = _pid_dir(cache_dir) / f"{avd}.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            ser = meta.get("serial")
            if ser:
                targets = [t for t in running_emulators() if t["serial"] == ser] or targets
            pid = meta.get("pid")
            if isinstance(pid, int) and not targets:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(pid, signal.SIGTERM)
                meta_path.unlink(missing_ok=True)
                return {
                    "ok": True,
                    "action": "emulator-stop",
                    "stopped": [],
                    "detail": f"signalled pid {pid} for avd {avd}",
                }

    if not targets and serial is None and avd is None:
        stopped_pids: list[int] = []
        for path in list(_pid_dir(cache_dir).glob("*.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
                continue
            pid = meta.get("pid")
            if isinstance(pid, int):
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(pid, signal.SIGTERM)
                    stopped_pids.append(pid)
            path.unlink(missing_ok=True)
        return {
            "ok": True,
            "action": "emulator-stop",
            "stopped": [],
            "detail": "no running emulator-* devices; cleared aua pid records if any",
            "signalled_pids": stopped_pids,
        }

    if not targets:
        raise DeviceError(
            "no matching running emulator to stop",
            hint="`aua emulator status` lists live serials; pass --serial emulator-5554.",
        )

    adb = adb_bin()
    stopped: list[str] = []
    for t in targets:
        ser = t["serial"]
        try:
            subprocess.run(  # noqa: S603
                [adb, "-s", ser, "emu", "kill"],
                check=False,
                capture_output=True,
                timeout=20,
            )
            stopped.append(ser)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.debug("emu kill %s failed: %s", ser, exc)

    for path in list(_pid_dir(cache_dir).glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            continue
        if meta.get("serial") in stopped or (avd and path.stem == avd):
            path.unlink(missing_ok=True)

    return {"ok": True, "action": "emulator-stop", "stopped": stopped}
