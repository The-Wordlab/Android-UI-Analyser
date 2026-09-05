"""Boot / list / stop Android AVDs for unattended (headless) agent runs.

Most lifecycle is boot/stop only. Exception: ``ensure_proxy_avd`` can install a small
**Google APIs** (non–Play Store) system image and create a rootable AVD — needed so
``aua proxy`` can ``adb root`` and install the mitm CA as a system trust anchor.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

from .atomic import atomic_write_text
from .errors import DeviceError, UsageError
from .platforms.virtual_targets import (
    OwnedVirtualTargetStopRequest,
    VirtualTargetCreateRequest,
    VirtualTargetCreateResult,
    VirtualTargetDefinition,
    VirtualTargetDeleteRequest,
    VirtualTargetDeleteResult,
    VirtualTargetInstance,
    VirtualTargetList,
    VirtualTargetProvisionRequest,
    VirtualTargetReclaimRequest,
    VirtualTargetStartRequest,
    VirtualTargetStatus,
    VirtualTargetStopRequest,
    VirtualTargetStopResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_WAIT_S = 120.0
_POLL_S = 1.0
# AVDs aua starts auto-stop after this much idle (no journal / no touch) *and* no live lease.
# Agents must still call `stop --mine`; this is the safety net when they forget — or when they
# are killed and never get the chance. Windowed AVDs are covered too: an emulator nobody has
# touched for twenty minutes is nobody's session, and it burns the same battery as a headless one.
#
# `idle` here means "no AUA activity", which is not the same as "no human activity" — a person
# clicking around a windowed AVD by hand is invisible to us. Twenty minutes is what makes that
# safe: a human at the keyboard does not go twenty minutes without touching the thing they are
# using, while a forgotten emulator crosses it immediately.
_DEFAULT_IDLE_STOP_S = 1200.0  # 20 minutes

# Small + rootable default for HTTPS proxy / system-CA work. API 30 boots faster than
# 34/36 and still supports classic /system/etc/security/cacerts remount.
PROXY_AVD_NAME = "aua_proxy"
PROXY_API_DEFAULT = 30
PROXY_DEVICE = "pixel_3a"  # compact; we still shrink lcd/ram in config.ini
_PROXY_LCD = (720, 1280, 320)  # w, h, density — matches "Small Phone" class
_PROXY_RAM_MB = 1536
_PROXY_DATA_PARTITION = "2G"


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


def ensure_adb_on_path() -> str | None:
    """Publish the SDK's ``adb`` on ``PATH`` so PATH-only consumers agree with us.

    ``adb_bin`` already falls back to ``$ANDROID_HOME``/the macOS default SDK, but
    third-party libraries do not: ``adbutils`` (which backs device listing) and any
    subprocess we spawn only search ``PATH``. A stock Android Studio install keeps
    ``adb`` inside the SDK and off a non-interactive shell's ``PATH``, so on a perfectly
    working machine ``aua doctor`` reported three failures — adb, devices, emulator —
    while every AUA-resolved call succeeded. That false negative is what unattended
    setup gates on, so normalise it once, before any command runs.

    Returns the resolved ``adb`` path, or ``None`` when there is genuinely none.
    """
    found = shutil.which("adb")
    if found:
        return found
    root = sdk_root()
    if root is None:
        return None
    candidate = root / "platform-tools" / "adb"
    if not (candidate.is_file() and os.access(candidate, os.X_OK)):
        return None
    os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    # adbutils honours this explicitly; set it too so it never re-scans a stale PATH.
    os.environ.setdefault("ADBUTILS_ADB_PATH", str(candidate))
    return str(candidate)


def _sdk_tool(name: str) -> str | None:
    """Resolve ``sdkmanager`` / ``avdmanager``.

    Prefer ``$ANDROID_HOME/cmdline-tools/latest`` over PATH — Homebrew's
    ``android-commandlinetools`` is often too old for current SDK repository XML.
    """
    root = sdk_root()
    candidates: list[Path] = []
    if root is not None:
        candidates.append(root / "cmdline-tools" / "latest" / "bin" / name)
        ct = root / "cmdline-tools"
        if ct.is_dir():
            for child in sorted(ct.iterdir(), reverse=True):
                if child.name == "latest":
                    continue
                candidates.append(child / "bin" / name)
        candidates.append(root / "tools" / "bin" / name)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    which = shutil.which(name)
    return which


def sdkmanager_bin() -> str:
    path = _sdk_tool("sdkmanager")
    if path:
        return path
    raise UsageError(
        "sdkmanager not found",
        hint=(
            "Install Android SDK Command-line Tools (Android Studio → SDK Tools, or "
            "`sdkmanager` package), then re-run. Needed only for `aua emulator ensure-proxy`."
        ),
    )


def avdmanager_bin() -> str:
    path = _sdk_tool("avdmanager")
    if path:
        return path
    raise UsageError(
        "avdmanager not found",
        hint=(
            "Install Android SDK Command-line Tools (same package as sdkmanager), "
            "then re-run `aua emulator ensure-proxy`."
        ),
    )


def preferred_abi() -> str:
    """Host ABI for the emulator system image (Apple Silicon → arm64-v8a)."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64-v8a"
    return "x86_64"


def avd_dir(name: str) -> Path:
    return Path.home() / ".android" / "avd" / f"{name}.avd"


def inspect_avd(name: str) -> dict[str, Any]:
    """Read ``config.ini`` and classify Play Store vs rootable Google APIs."""
    cfg_path = avd_dir(name) / "config.ini"
    info: dict[str, Any] = {
        "name": name,
        "config": str(cfg_path) if cfg_path.is_file() else None,
        "rootable": None,
        "play_store": None,
        "tag": None,
        "image": None,
        "api": None,
        "lcd": None,
        "ram_mb": None,
    }
    if not cfg_path.is_file():
        return info
    raw = cfg_path.read_text(encoding="utf-8", errors="replace")
    kv: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        kv[k.strip()] = v.strip()
    tag = kv.get("tag.id") or kv.get("tag.ids") or ""
    image = kv.get("image.sysdir.1") or ""
    play = (
        tag == "google_apis_playstore"
        or "playstore" in tag.lower()
        or "playstore" in image.lower()
        or kv.get("PlayStore.enabled", "").lower() == "true"
    )
    # google_apis (no playstore) and AOSP `default` / `google_apis` are rootable.
    rootable = (not play) and (
        "google_apis" in tag
        or tag in ("default", "aosp", "")
        or ("google_apis" in image and "playstore" not in image.lower())
    )
    api: int | None = None
    m = re.search(r"android-(\d+)", image) or re.search(r"android-(\d+)", kv.get("target", ""))
    if m:
        api = int(m.group(1))
    lcd = None
    if kv.get("hw.lcd.width") and kv.get("hw.lcd.height"):
        with contextlib.suppress(ValueError):
            lcd = {
                "width": int(kv["hw.lcd.width"]),
                "height": int(kv["hw.lcd.height"]),
                "density": int(kv.get("hw.lcd.density") or 0) or None,
            }
    ram = None
    with contextlib.suppress(ValueError):
        if kv.get("hw.ramSize"):
            ram = int(kv["hw.ramSize"])
    info.update(
        {
            "rootable": rootable,
            "play_store": play,
            "tag": tag or None,
            "image": image or None,
            "api": api,
            "lcd": lcd,
            "ram_mb": ram,
        }
    )
    return info


def _list_avd_names() -> list[str]:
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
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def recommend_proxy_avd(*, api: int = PROXY_API_DEFAULT, name: str = PROXY_AVD_NAME) -> dict[str, Any]:
    """Suggest a small rootable system image + create/start commands (no side effects)."""
    abi = preferred_abi()
    package = f"system-images;android-{api};google_apis;{abi}"
    existing: list[dict[str, Any]] = []
    try:
        existing = [inspect_avd(n) for n in _list_avd_names()]
    except (UsageError, DeviceError):
        existing = []
    rootable = [d for d in existing if d.get("rootable") is True]
    return {
        "ok": True,
        "action": "emulator-recommend-proxy",
        "name": name,
        "api": api,
        "abi": abi,
        "package": package,
        "device": PROXY_DEVICE,
        "why": (
            "Google Play AVDs refuse `adb root`, so system mitm CA install fails and HTTPS "
            "apps that only trust system CAs produce empty proxy cassettes. A small "
            "google_apis (non-Play) image is rootable and boots faster."
        ),
        "existing_rootable": [d["name"] for d in rootable],
        "existing_avds": existing,
        "create": f"aua emulator ensure-proxy --name {name} --api {api}",
        "start": f"aua emulator start --avd {name} --headless",
        "hint": (
            f"Rootable AVD already present: {', '.join(d['name'] for d in rootable)}. "
            f"`aua emulator start --avd {rootable[0]['name']} --headless` then `aua proxy start`."
            if rootable
            else f"No rootable AVD yet — run `aua emulator ensure-proxy` "
            f"(downloads {package}, creates {name})."
        ),
    }


def _image_installed(package: str) -> bool:
    root = sdk_root()
    if root is None:
        return False
    # system-images;android-30;google_apis;arm64-v8a → system-images/android-30/google_apis/arm64-v8a
    parts = package.split(";")
    if len(parts) != 4:
        return False
    path = root.joinpath(*parts)
    return path.is_dir() and any(path.iterdir())


def _run_sdk(
    cmd: list[str],
    *,
    timeout: float,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Avoid interactive license / progress TTY issues in agent shells.
    env.setdefault("ANDROID_SDK_ROOT", str(sdk_root() or ""))
    return subprocess.run(  # noqa: S603
        cmd,
        input=stdin,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _patch_proxy_avd_config(name: str) -> Path:
    """Shrink LCD/RAM after create so the AVD stays fast."""
    cfg = avd_dir(name) / "config.ini"
    if not cfg.is_file():
        raise DeviceError(
            f"AVD config missing after create: {cfg}",
            hint="avdmanager may have failed; check cmdline-tools install.",
        )
    w, h, dens = _PROXY_LCD
    lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
    overrides = {
        "hw.lcd.width": str(w),
        "hw.lcd.height": str(h),
        "hw.lcd.density": str(dens),
        "hw.ramSize": str(_PROXY_RAM_MB),
        "disk.dataPartition.size": _PROXY_DATA_PARTITION,
        "PlayStore.enabled": "false",
        "hw.keyboard": "yes",
    }
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if "=" not in line or line.strip().startswith("#"):
            out.append(line)
            continue
        k, _, _ = line.partition("=")
        key = k.strip()
        if key in overrides:
            out.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, val in overrides.items():
        if key not in seen:
            out.append(f"{key}={val}")
    cfg.write_text("\n".join(out) + "\n", encoding="utf-8")
    return cfg


def ensure_proxy_avd(
    *,
    name: str = PROXY_AVD_NAME,
    api: int = PROXY_API_DEFAULT,
    force: bool = False,
    accept_licenses: bool = True,
) -> dict[str, Any]:
    """Install a small google_apis system image (if needed) and create a rootable AVD.

    Idempotent: if *name* already exists and is rootable, returns it unless ``force``.
    If another rootable AVD already exists and *name* is missing, still creates *name*
    only when the agent asked for this helper — so proxy work has a known small target.
    """
    rec = recommend_proxy_avd(api=api, name=name)
    package = str(rec["package"])
    detail = inspect_avd(name)
    if detail.get("config") and detail.get("rootable") and not force:
        return {
            "ok": True,
            "action": "emulator-ensure-proxy",
            "created": False,
            "avd": name,
            "detail": detail,
            "package": package,
            "hint": f"Already ready. `aua emulator start --avd {name} --headless` then `aua proxy start`.",
        }
    if detail.get("config") and detail.get("play_store") and not force:
        raise UsageError(
            f"AVD {name!r} exists but is a Google Play image (not rootable)",
            hint=f"Pass --force to recreate as google_apis, or use another name. {rec['create']}",
        )

    sm = sdkmanager_bin()
    am = avdmanager_bin()
    installed = _image_installed(package)
    steps: list[str] = []

    if accept_licenses:
        lic = _run_sdk([sm, "--licenses"], timeout=120, stdin="y\n" * 40)
        steps.append(f"licenses exit={lic.returncode}")

    if not installed or force:
        # Large download — give it room. Agents should expect several minutes.
        inst = _run_sdk([sm, package], timeout=1800, stdin="y\n" * 10)
        steps.append(f"sdkmanager {package} exit={inst.returncode}")
        if inst.returncode != 0 and not _image_installed(package):
            err = ((inst.stderr or "") + (inst.stdout or ""))[-1200:]
            raise DeviceError(
                f"failed to install system image {package}",
                hint=err or "Check network + Android SDK licenses (`sdkmanager --licenses`).",
            )

    # Create (or recreate) the AVD.
    create_cmd = [
        am,
        "create",
        "avd",
        "--force",
        "--name",
        name,
        "--package",
        package,
        "--device",
        PROXY_DEVICE,
    ]
    created = _run_sdk(create_cmd, timeout=120, stdin="no\n")
    steps.append(f"avdmanager create exit={created.returncode}")
    if created.returncode != 0 and not (avd_dir(name) / "config.ini").is_file():
        # Some avdmanager builds dislike --device; retry without.
        created = _run_sdk(
            [am, "create", "avd", "--force", "--name", name, "--package", package],
            timeout=120,
            stdin="no\n",
        )
        steps.append(f"avdmanager create (no device) exit={created.returncode}")
    if not (avd_dir(name) / "config.ini").is_file():
        err = ((created.stderr or "") + (created.stdout or ""))[-1200:]
        raise DeviceError(
            f"failed to create AVD {name!r}",
            hint=err or f"Try manually: avdmanager create avd -n {name} -k {package!r}",
        )

    cfg = _patch_proxy_avd_config(name)
    detail = inspect_avd(name)
    return {
        "ok": True,
        "action": "emulator-ensure-proxy",
        "created": True,
        "avd": name,
        "detail": detail,
        "package": package,
        "config": str(cfg),
        "steps": steps,
        "hint": (
            "Start goal work with `aua session start --needs root,proxy --goal <goal>`; it "
            "selects or boots this compatible image and leases it automatically. Then run "
            "`aua proxy start` without a serial "
            "(system CA needs this rootable image)."
        ),
    }


def _pid_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir).expanduser() / "emulator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_instance_record(meta: object) -> bool:
    """True for an emulator instance record, false for anything else in the pid directory.

    ``_pid_dir`` is not exclusively ours: ``anim_off`` writes a settings snapshot beside the instance
    records as ``<inst>.anim.json``, so the device's animation and dialog settings can be restored
    later. Two readers globbed ``*.json`` and treated every file as an instance, so on a machine that
    had been running for two days ``emulator status`` reported **24 "started" emulators that were all
    settings snapshots** - and a caller deciding whether an AVD was free could not parse any of it.

    Nothing was cleaned up either: a snapshot carries no ``serial``/``pid``/``avd``, so the stop and
    cleanup paths skip it, which is why they accumulate. That is a leak rather than a correctness bug,
    and it is what made this visible.

    The three other globs over this directory are guarded by accident rather than design - they match
    on ``serial``, ``pid`` or ``avd``, none of which a snapshot carries. Stating the distinction here
    means a future reader does not have to work out which sites were safe and why.

    An instance record always carries ``avd``; see the reservation payload written at start.
    """
    return isinstance(meta, dict) and bool(meta.get("avd"))


# Console ports the Android emulator binds (even numbers only → serial emulator-{port}).
_EMULATOR_PORT_MIN = 5554
_EMULATOR_PORT_MAX = 5682
# How long a port reservation is honoured. Only has to cover the gap between choosing a
# port and the emulator binding it; after that the instance is visible to adb.
_RESERVATION_TTL_S = 300


def instance_id(avd: str, port: int) -> str:
    """Stable meta-file stem for one running instance (``{avd}.p{port}``)."""
    return f"{avd}.p{int(port)}"


def _serial_for_port(port: int) -> str:
    return f"emulator-{int(port)}"


def _port_from_serial(serial: str) -> int | None:
    m = re.fullmatch(r"emulator-(\d+)", serial.strip())
    if not m:
        return None
    port = int(m.group(1))
    return port if port % 2 == 0 else None


def _used_console_ports(*, cache_dir: str | Path | None = None) -> set[int]:
    used: set[int] = set()
    with contextlib.suppress(Exception):
        for d in running_emulators():
            p = _port_from_serial(str(d.get("serial") or ""))
            if p is not None:
                used.add(p)
    if cache_dir is not None:
        for meta in _aua_started_records(cache_dir):
            raw = meta.get("port")
            if isinstance(raw, int):
                used.add(raw)
            elif isinstance(meta.get("serial"), str):
                p = _port_from_serial(str(meta["serial"]))
                if p is not None:
                    used.add(p)
    return used


def _reservation_dir() -> Path:
    """Global port-reservation directory, deliberately NOT under ``AUA_CACHE__DIR``.

    Parallel agents are told to keep separate caches so proxy rules cannot leak between
    them, which means a per-cache record is invisible to every other worker. Port
    allocation is the one thing that must be coordinated process-wide, so its bookkeeping
    lives at a fixed path regardless of the caller's cache.
    """
    d = Path.home() / ".cache/android-ui-analyser/portlocks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reserved_console_ports() -> set[int]:
    """Ports claimed by a starting instance, dropping reservations that went stale.

    A reservation only has to survive the gap between choosing a port and the emulator
    binding it. Anything older than the window is either booted (and therefore visible to
    adb) or dead, so it must not block the range forever.
    """
    import time as _time

    out: set[int] = set()
    for f in _reservation_dir().glob("*.port"):
        try:
            port = int(f.stem)
        except ValueError:
            with contextlib.suppress(Exception):
                f.unlink()
            continue
        try:
            age = _time.time() - f.stat().st_mtime
        except OSError:
            continue
        if age > _RESERVATION_TTL_S:
            with contextlib.suppress(Exception):
                f.unlink()
            continue
        out.add(port)
    return out


def _claim_console_port(port: int) -> bool:
    """Atomically claim ``port``. False means another process already holds it."""
    target = _reservation_dir() / f"{int(port)}.port"
    try:
        fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError:
        return True  # cannot coordinate; do not block the caller
    with contextlib.suppress(Exception):
        os.write(fd, f"{os.getpid()}\n".encode())
    with contextlib.suppress(Exception):
        os.close(fd)
    return True


def release_console_port(port: int | None) -> None:
    """Drop a port reservation once the instance owns the port (or failed to start)."""
    if port is None:
        return
    with contextlib.suppress(Exception):
        (_reservation_dir() / f"{int(port)}.port").unlink()


def _leased_console_ports(lease_registry_dir: str | Path | None) -> set[int]:
    """Console ports of serials the lease registry says are held by a live owner.

    The adb snapshot is not the authority on which ports are occupied: a live failure saw
    adb briefly omit a foreign worker's emulator-5554 while a second agent allocated a
    port, so 5554 looked free, the new boot answered on the foreign device, and the
    claim-failure rollback stopped it. The registry is host-wide and process-bound, so a
    live lease marks its port used regardless of what adb happens to report — and a dead
    owner's lease reads as expired, freeing the port immediately.
    """
    if lease_registry_dir is None:
        return set()
    out: set[int] = set()
    try:
        from . import leases

        for serial in leases.live_leased_serials(lease_registry_dir):
            p = _port_from_serial(serial)
            if p is not None:
                out.add(p)
    except OSError as exc:
        raise DeviceError(
            "cannot read the host-wide lease registry; refusing emulator provisioning",
            hint=(
                f"Lease authority {Path(lease_registry_dir).expanduser()} is inaccessible: "
                f"{exc}. Restore access, then retry."
            ),
        ) from exc
    return out


def allocate_console_port(
    preferred: int | None = None,
    *,
    cache_dir: str | Path | None = None,
    lease_registry_dir: str | Path | None = None,
) -> int:
    """Pick a free even emulator console port (5554–5682), and reserve it.

    The reservation matters: adb cannot see an emulator that is still booting, so two
    agents starting at the same moment used to read an identical "used" set, pick the same
    port, and race. The loser bound no console port at all - it stayed alive but invisible
    to adb, while its serial actually belonged to the winner's AVD. Two agents then drove
    one device believing they each had their own.
    """
    used = (
        _used_console_ports(cache_dir=cache_dir)
        | _reserved_console_ports()
        | _leased_console_ports(lease_registry_dir)
    )
    if preferred is not None:
        port = int(preferred)
        if port % 2 != 0:
            raise UsageError(
                f"emulator console port must be even (got {port})",
                hint="Use 5554, 5556, … — serial becomes emulator-{port}.",
            )
        if port < _EMULATOR_PORT_MIN or port > _EMULATOR_PORT_MAX:
            raise UsageError(
                f"emulator port {port} out of range "
                f"{_EMULATOR_PORT_MIN}–{_EMULATOR_PORT_MAX}",
            )
        if port in used:
            raise DeviceError(
                f"emulator port {port} already in use",
                hint="Omit --port to auto-allocate, or pick a free even port "
                f"(used: {', '.join(str(p) for p in sorted(used)) or 'none'}).",
            )
        _claim_console_port(port)
        return port
    for port in range(_EMULATOR_PORT_MIN, _EMULATOR_PORT_MAX + 1, 2):
        if port in used:
            continue
        # Claim before returning: the check above is a snapshot, and a concurrent caller
        # may be between its own check and its emulator binding the port.
        if _claim_console_port(port):
            return port
    raise DeviceError(
        "no free emulator console ports left",
        hint=f"All even ports {_EMULATOR_PORT_MIN}–{_EMULATOR_PORT_MAX} are taken — "
        "`aua emulator stop --mine` (or `--owner`) to free some.",
    )


def _live_foreign_lease(
    lease_registry_dir: str | Path | None,
    serial: str | None,
    owner: str | None,
) -> dict[str, Any] | None:
    """The live lease that makes *serial* untouchable for this caller, or ``None``.

    A stop must never guess: when the registry cannot be read, the answer is "held by
    unknown", because killing a device someone else is driving is the unrecoverable
    direction. A lease whose owner process is dead reads as expired and does not block.
    """
    if lease_registry_dir is None or not serial:
        return None
    try:
        from . import leases

        entry = leases.read_lease(lease_registry_dir, serial)
        if entry is None:
            return None
        if owner is not None and leases.entry_owned_by(entry, owner):
            return None
        return entry
    except Exception:
        return {"serial": serial, "owner": "unknown", "inaccessible": True}


@contextlib.contextmanager
def _stop_lease_transaction(
    lease_registry_dir: str | Path | None,
    serial: str | None,
) -> Iterator[None]:
    """Fence one stop decision and kill against device use and ownership transitions."""

    if lease_registry_dir is None or not serial:
        yield
        return
    from . import leases

    with leases.device_transaction(lease_registry_dir, serial):
        yield


def resolve_owner(explicit: str | None = None) -> str | None:
    """Owner tag for parallel agents: ``--owner``, else ``$AUA_OWNER``, else None."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = (os.environ.get("AUA_OWNER") or "").strip()
    return env or None


def _default_parallel_owner() -> str:
    import uuid

    return f"aua-{uuid.uuid4().hex[:8]}"


def list_avds() -> dict[str, Any]:
    """Return configured AVD names (does not start anything)."""
    bin_path = emulator_bin()
    names = _list_avd_names()
    details = [inspect_avd(n) for n in names]
    rootable = [d["name"] for d in details if d.get("rootable")]
    play = [d["name"] for d in details if d.get("play_store")]
    hint = None
    if not names:
        hint = (
            "No AVDs yet — `aua emulator ensure-proxy` creates a small rootable Google APIs "
            "AVD (needed for HTTPS proxy / system CA), or use Android Studio Device Manager."
        )
    elif play and not rootable:
        hint = (
            "Only Google Play AVDs found (not rootable — `adb root` / system CA install fail). "
            "For proxy/mock HTTPS: `aua emulator ensure-proxy` then "
            f"`aua emulator start --avd {PROXY_AVD_NAME} --headless`."
        )
    return {
        "ok": True,
        "action": "emulator-list",
        "emulator": bin_path,
        "avds": names,
        "details": details,
        "rootable": rootable,
        "play_store": play,
        "count": len(names),
        "hint": hint,
        "recommend_proxy": recommend_proxy_avd() if (not rootable) else None,
    }


def select_avd_for_session(
    avd: str | None = None,
    *,
    needs: list[str] | None = None,
) -> str:
    """Choose a configured AVD whose offline image facts satisfy session requirements."""

    listed = list_avds()
    details = [item for item in listed.get("details", []) if isinstance(item, dict)]
    if not details:
        raise UsageError(
            "no Android Virtual Devices configured",
            hint="`aua emulator ensure-proxy` creates a rootable default AVD.",
        )
    requested = [str(item).strip().lower() for item in (needs or []) if str(item).strip()]
    supported = {"root", "play", "proxy"}
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise UsageError(
            f"cannot select an AVD for unknown requirement(s): {', '.join(unknown)}",
            hint="Supported device requirements are root, play, and proxy.",
        )

    def matches(item: dict[str, Any]) -> bool:
        return all(
            {
                "root": item.get("rootable") is True,
                "proxy": item.get("rootable") is True,
                "play": item.get("play_store") is True,
            }[need]
            for need in requested
        )

    if avd is not None:
        selected = next((item for item in details if item.get("name") == avd), None)
        if selected is None:
            names = ", ".join(str(item.get("name")) for item in details)
            raise UsageError(f"unknown AVD {avd!r}", hint=f"Known AVDs: {names}.")
        if not matches(selected):
            raise UsageError(
                f"AVD {avd!r} does not satisfy: {', '.join(requested)}",
                hint="Omit --avd to let session start choose a compatible image.",
            )
        return avd

    compatible = [item for item in details if matches(item)]
    if not compatible:
        available = ", ".join(
            f"{item.get('name')}"
            f"(root={bool(item.get('rootable'))},play={bool(item.get('play_store'))})"
            for item in details
        )
        raise UsageError(
            f"no configured AVD satisfies: {', '.join(requested) or 'default'}",
            hint=f"Configured images: {available}.",
        )
    # Preserve the emulator tool's configured order. It is stable, user-controlled, and keeps
    # an explicit local default without inventing a second AVD preference configuration.
    return str(compatible[0]["name"])


def running_emulators() -> list[dict[str, Any]]:
    """Emulator serials currently visible to adb (``emulator-*``)."""
    from .device import list_devices
    from .platforms import android_transport

    out: list[dict[str, Any]] = []
    # Emulator boot/wait/cleanup used to bypass AndroidPlatform.prepare_host. When the shared
    # daemon disappeared mid-boot, every waiting process entered adbutils' implicit autostart at
    # once. Inventory now stays inside the same endpoint-global transaction as adapter listing.
    inventory = android_transport.run_adb_inventory_operation(None, list_devices)
    for d in inventory:
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
        info["avd_details"] = listed.get("details") or []
        info["rootable"] = listed.get("rootable") or []
        info["play_store"] = listed.get("play_store") or []
        info["hint"] = listed.get("hint")
        if listed.get("recommend_proxy"):
            info["recommend_proxy"] = listed["recommend_proxy"]
    except (UsageError, DeviceError):
        info["avds"] = []
    info["running"] = running_emulators()
    if cache_dir is not None:
        started = []
        for path in _pid_dir(cache_dir).glob("*.json"):
            with path.open(encoding="utf-8") as fh:
                try:
                    meta = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if _is_instance_record(meta):
                started.append(meta)
        info["started_by_aua"] = started
    return info


def _serial_shell(serial: str) -> Callable[[str], str]:
    """A ``devopts.ShellFn`` bound to one serial, for use before an Engine exists."""

    def shell(cmd: str) -> str:
        proc = subprocess.run(  # noqa: S603
            [adb_bin(), "-s", serial, "shell", cmd],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.stdout or ""

    return shell


def _wait_for_boot(shell: Callable[[str], str], *, timeout_s: float = 90.0) -> bool:
    """Block until the device is actually usable, not merely booted. False on timeout.

    ``sys.boot_completed`` flips before PackageManager will answer queries, so a caller that
    trusted it got empty results from its very first `pm`/`dumpsys package` call — reported as
    "the app is not installed" rather than "ask again in a moment". Requiring PackageManager to
    name a package too costs one extra shell round trip and removes that whole class of
    start-up flake.
    """
    deadline = time.monotonic() + max(5.0, timeout_s)
    booted = False
    while time.monotonic() < deadline:
        if not booted and (shell("getprop sys.boot_completed") or "").strip() == "1":
            booted = True
        if booted and "package:" in (shell("pm path android") or ""):
            return True
        time.sleep(_POLL_S)
    return False


def _clear_inherited_blackholed_proxy(
    serial: str, *, cache_dir: str | Path
) -> dict[str, Any]:
    """Clear a confirmed dead proxy inherited from an AVD's persisted device state.

    A read-only/parallel boot inherits Android's global ``http_proxy`` setting from the base
    image, but it cannot inherit the host process or ``adb reverse`` tunnel that made that
    setting useful.  When no AUA ownership record claims the endpoint and both reachability
    checks are red, preserving it only guarantees that every app starts offline.  Clear that
    one proven-safe state before an app can launch; never touch an owned, reachable foreign,
    or externally-addressed proxy.

    This lives in the Android virtual-device service rather than in the CLI so direct CLI,
    MCP, and ``session start --start-emulator`` boots share the same recovery.
    """
    from . import proxy_mock

    try:
        before = proxy_mock.proxy_health(serial, cache_dir, self_heal=False)
    except Exception as exc:  # pragma: no cover - defensive; boot must still be attributable
        return {
            "ok": False,
            "checked": False,
            "cleared": False,
            "detail": f"could not inspect inherited proxy state: {exc}",
        }

    state_before = str(before.get("state") or "unknown")
    result: dict[str, Any] = {
        "ok": bool(before.get("ok", False)),
        "checked": True,
        "cleared": False,
        "state_before": state_before,
        "state_after": state_before,
    }
    if state_before != "blackholed" or before.get("owned") is not False:
        result["detail"] = "inherited proxy state needed no automatic cleanup"
        return result

    shell = _serial_shell(serial)
    # Match Device.set_http_proxy(None): ':0' clears on modern Android and deleting the key is
    # the fallback.  Verification below is authoritative; command stdout is not.
    shell("settings put global http_proxy :0")
    shell("settings delete global http_proxy")
    # A mismatched stale record can accompany an unowned target.  It cannot describe the
    # endpoint the device was actually using, so retaining it would make the clean device look
    # owned/degraded on its next status call.
    proxy_mock.clear_state(serial)

    try:
        after = proxy_mock.proxy_health(serial, cache_dir, self_heal=False)
    except Exception as exc:  # pragma: no cover - defensive
        result.update(
            ok=False,
            detail=f"cleared inherited proxy setting but could not verify recovery: {exc}",
        )
        return result

    state_after = str(after.get("state") or "unknown")
    cleared = state_after == "unproxied"
    result.update(
        ok=cleared,
        cleared=cleared,
        state_after=state_after,
        detail=(
            "cleared an unowned blackholed proxy inherited from the AVD"
            if cleared
            else "attempted to clear an unowned blackholed proxy, but verification failed"
        ),
    )
    if cleared:
        logger.warning("cleared inherited blackholed proxy on %s before app launch", serial)
    return result


def _adb_emu_kill(serial: str) -> None:
    """Terminate one emulator. A named seam, so a test can stub it and MEAN it.

    This used to be an inline `subprocess.run` inside `stop`, with no way to intercept it.
    A test that patched `_adb_emu_kill` (which did not exist) silently patched nothing and
    ran the real kill against the serials in its fixture — `emulator-5554`/`emulator-5556`,
    i.e. whatever the developer actually had running. Every full-suite run killed them.
    """
    subprocess.run(  # noqa: S603
        [adb_bin(), "-s", serial, "emu", "kill"],
        check=False,
        capture_output=True,
        timeout=20,
    )


def touch_activity(cache_dir: str | Path, serial: str | None) -> None:
    """Bump ``last_activity`` on any aua-started emulator record matching *serial*."""
    if not serial:
        return
    for path in _pid_dir(cache_dir).glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        if meta.get("serial") != serial and path.stem != serial:
            continue
        meta["last_activity"] = time.time()
        with contextlib.suppress(OSError):
            path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _kill_watchdog(meta: dict[str, Any] | None) -> None:
    if not isinstance(meta, dict):
        return
    wpid = meta.get("watchdog_pid")
    # Skip self: the idle watchdog may call stop() and must not SIGTERM mid-cleanup.
    if isinstance(wpid, int) and wpid != os.getpid():
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(wpid, signal.SIGTERM)


def _rollback_failed_start(
    *,
    pid: int,
    meta: dict[str, Any],
    meta_path: Path,
    console_port: int | None,
) -> None:
    """Reap only the process this start spawned, without consulting shared ADB state."""

    _kill_watchdog(meta)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, signal.SIGTERM)
    meta_path.unlink(missing_ok=True)
    release_console_port(console_port)


def _spawn_idle_watchdog(
    *,
    cache_dir: Path,
    instance: str,
    lease_registry_dir: str | Path | None = None,
) -> int | None:
    """Detach a watchdog that stops this instance after idle_timeout_s. Returns pid or None."""
    log = cache_dir / "emulator" / f"{instance}.watchdog.log"
    cache_dir.joinpath("emulator").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "android_ui_analyser.emulator_watchdog",
        "--cache",
        str(cache_dir),
        "--instance",
        instance,
        "--lease-registry",
        str(lease_registry_dir or cache_dir),
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
        return int(proc.pid)
    except OSError as exc:
        logger.warning("could not spawn emulator watchdog: %s", exc)
        return None


def default_gpu_mode(*, headless: bool) -> str:
    """Pick a GPU backend that does not melt the host CPU.

    Older aua always used ``swiftshader_indirect`` for ``-no-window``. That is a
    *software* GLES renderer — fine for Linux CI without a GPU, but on a Mac laptop it
    pegs the CPU, spins the fans, and drains the battery while a windowed emulator
    (``-gpu host`` / Metal) stays quiet. Desktop hosts can use hardware GPU even with
    ``-no-window``; only fall back to SwiftShader when there is clearly no display.
    """
    if not headless:
        return "auto"
    if sys.platform in ("darwin", "win32"):
        return "host"
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "host"
    return "swiftshader"


def start(
    avd: str | None = None,
    *,
    headless: bool = True,
    animations: bool = False,
    audio: bool = False,
    wait_s: float = _DEFAULT_WAIT_S,
    cache_dir: str | Path,
    extra_args: list[str] | None = None,
    gpu: str | None = None,
    idle_timeout_s: float | None = None,
    port: int | None = None,
    read_only: bool | None = None,
    parallel: bool = False,
    owner: str | None = None,
    lease_registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Boot an AVD (headless by default) and wait until adb sees ``state=device``.

    If *avd* is omitted and exactly one AVD exists, that one is used. Does not steal
    focus when ``headless=True`` (``-no-window``). Creating AVDs is ``ensure_proxy_avd``.

    *gpu* defaults via :func:`default_gpu_mode` — ``host`` on Mac/Windows so headless
    does not fall back to CPU SwiftShader.

    *idle_timeout_s*: for headless starts, auto-stop after this many seconds without
    aua activity (default 900). Pass ``0`` to disable. Windowed defaults to disabled.

    *parallel*: allocate a free console port, pass ``-read-only`` so multiple agents can
    boot the same AVD name concurrently, tag with *owner* (or ``$AUA_OWNER``, or an
    auto id). Standalone start only provisions: when this agent has no lease, the returned
    ``serial`` can select its one sticky device; switching an existing lease requires the
    explicit, warned ``lease acquire <serial> --replace`` path. Stop with ``stop --serial`` /
    ``--owner`` / ``--mine`` (scoped by ``$AUA_OWNER``).
    """
    listed = list_avds()
    names: list[str] = list(listed["avds"])
    if not names:
        raise UsageError(
            "no Android Virtual Devices configured",
            hint="`aua emulator ensure-proxy` creates a small rootable Google APIs AVD, "
            "or create one in Android Studio — then `aua emulator start --headless`.",
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

    # Parallel / multi-instance: unique port + read-only so the AVD file lock does not block.
    same_avd_running = any(m.get("avd") == avd for m in _aua_started_records(cache_dir))
    if (parallel or same_avd_running) and read_only is None:
        read_only = True
    if not parallel and port is None and same_avd_running:
        # Implicit parallel when re-starting an AVD that aua already has recorded.
        parallel = True

    console_port: int | None = None
    if parallel or port is not None:
        console_port = allocate_console_port(
            port, cache_dir=cache_dir, lease_registry_dir=lease_registry_dir
        )

    if read_only is None:
        read_only = False

    owner_tag = resolve_owner(owner)
    if parallel and not owner_tag:
        owner_tag = _default_parallel_owner()

    inst = instance_id(avd, console_port) if console_port is not None else avd
    expected_serial = _serial_for_port(console_port) if console_port is not None else None

    # Reserve the instance meta early so a concurrent --parallel start does not pick the
    # same console port between allocate and Popen.
    meta_path = _pid_dir(cache_dir) / f"{inst}.json"
    wrote_reserve = False
    if console_port is not None and not meta_path.is_file():
        reserve = {
            "avd": avd,
            "instance": inst,
            "port": console_port,
            "serial": expected_serial,
            "started_by_aua": True,
            "reserving": True,
            "started_at": time.time(),
            "last_activity": time.time(),
        }
        if owner_tag:
            reserve["owner"] = owner_tag
        with contextlib.suppress(OSError):
            meta_path.write_text(json.dumps(reserve, indent=2) + "\n", encoding="utf-8")
            wrote_reserve = True

    def rollback_reservation() -> None:
        release_console_port(console_port)
        if wrote_reserve:
            meta_path.unlink(missing_ok=True)

    def reservation_step(operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except BaseException:
            rollback_reservation()
            raise

    before = reservation_step(lambda: {d["serial"] for d in running_emulators()})
    if expected_serial is not None and expected_serial in before:
        # The allocated port already answers as a live device: the used-port snapshot
        # missed it (adb blink, foreign worker's cache). Booting onto it would collide,
        # and any cleanup keyed by this serial would hit that foreign instance — so fail
        # here, before spawning anything, without having touched the device.
        release_console_port(console_port)
        if wrote_reserve:
            meta_path.unlink(missing_ok=True)
        raise DeviceError(
            f"console port {console_port} already belongs to a running emulator "
            f"({expected_serial})",
            hint="Another instance owns this serial; retry to allocate a different port.",
        )
    bin_path = reservation_step(emulator_bin)
    gpu_mode = (gpu or default_gpu_mode(headless=headless)).strip() or "auto"
    cmd = [bin_path, "-avd", avd]
    if headless:
        cmd += ["-no-window", "-no-boot-anim", "-gpu", gpu_mode]
        # A headless start silenced the device unconditionally, so the pool had no audio
        # device at all and a scenario about sound could not ask for one. Off stays the
        # default - it is one less subsystem on a machine running five workers - but it is
        # now a choice. Note this never made audio *unverifiable*: `dumpsys audio` reports
        # AudioPlaybackConfiguration state within ~30ms of a play tap, and
        # `dumpsys media.audio_flinger` distinguishes real output from an idle stream.
        # Only microphone input is genuinely unobservable.
        if not audio:
            cmd += ["-no-audio"]
    else:
        cmd += ["-gpu", gpu_mode]
    if console_port is not None:
        cmd += ["-port", str(console_port)]
    if read_only:
        cmd += ["-read-only"]
    if extra_args:
        cmd += list(extra_args)

    # `None` means "no opinion" → the default applies, headless or not. An explicit 0 stays 0 and
    # is remembered as a deliberate choice, so the orphan sweep does not later hand this instance
    # a watchdog its owner asked not to have.
    idle_stop_explicit = idle_timeout_s is not None
    if idle_timeout_s is None:
        idle_timeout_s = _DEFAULT_IDLE_STOP_S
    idle_timeout_s = max(0.0, float(idle_timeout_s))

    log_path = _pid_dir(cache_dir) / f"{inst}.log"
    log_fh = reservation_step(lambda: open(log_path, "a"))  # noqa: SIM115
    try:
        proc = reservation_step(
            lambda: subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                close_fds=True,
            )
        )
    finally:
        log_fh.close()

    now = time.time()
    from .leases import _proc_started

    boot_token = f"{inst}@{secrets.token_hex(16)}"
    meta: dict[str, Any] = {
        "avd": avd,
        "instance": inst,
        "instance_token": boot_token,
        "pid": proc.pid,
        "process_started": _proc_started(proc.pid),
        "headless": headless,
        "gpu": gpu_mode,
        "cmd": cmd,
        "log": str(log_path),
        "started_at": now,
        "last_activity": now,
        "started_by_aua": True,
        "idle_timeout_s": idle_timeout_s,
        "read_only": bool(read_only),
        "parallel": bool(parallel),
    }
    if console_port is not None:
        meta["port"] = console_port
        meta["serial"] = expected_serial
    if owner_tag:
        meta["owner"] = owner_tag

    cleanup_done = False

    def rollback_failed_start() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        _rollback_failed_start(
            pid=proc.pid,
            meta=meta,
            meta_path=meta_path,
            console_port=console_port,
        )

    def startup_step(operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except BaseException:
            rollback_failed_start()
            raise

    startup_step(
        lambda: meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    )

    if expected_serial is not None:
        serial = startup_step(
            lambda: _wait_for_serial(expected_serial, timeout_s=wait_s, expect_avd=avd)
        )
    else:
        serial = startup_step(lambda: _wait_for_new_emulator(before, timeout_s=wait_s))
    if serial is None:
        rollback_failed_start()
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-800:]
        raise DeviceError(
            f"emulator {avd!r} did not become ready within {int(wait_s)}s",
            hint=f"Check the log: {log_path}\n{tail}",
        )

    if proc.poll() is not None:
        # The serial answered, but the process this call spawned is already dead — so the
        # device that satisfied the wait belongs to someone else (a port race, or an adb
        # blink that hid a running instance during allocation). Fail before the proxy
        # cleanup below can mutate settings on a foreign target, and clean only our own
        # bookkeeping: the serial is explicitly not ours to stop.
        rollback_failed_start()
        raise DeviceError(
            f"emulator {avd!r} exited during startup while {serial} stayed online; "
            "that serial belongs to another instance",
            hint="Retry the start; a different console port will be allocated.",
        )

    # The instance is up and adb can see it, so the reservation has done its job and the
    # normal used-port detection takes over from here.
    release_console_port(console_port)

    # Check once as soon as adb exposes the serial. Some images publish their persisted global
    # settings this early, and removing a dead proxy before NetworkMonitor's first probe avoids
    # a transient offline classification. Android may restore the setting later in boot, though,
    # so the settled check below remains authoritative.
    startup_step(lambda: _clear_inherited_blackholed_proxy(serial, cache_dir=cache_dir))

    meta["serial"] = serial
    meta["last_activity"] = time.time()
    watchdog_pid = None
    meta["idle_stop_explicit"] = idle_stop_explicit
    if idle_timeout_s > 0:
        watchdog_pid = startup_step(
            lambda: _spawn_idle_watchdog(
                cache_dir=Path(cache_dir).expanduser(),
                instance=inst,
                lease_registry_dir=lease_registry_dir,
            )
        )
        meta["watchdog_pid"] = watchdog_pid
    startup_step(
        lambda: meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    )
    shell = startup_step(lambda: _serial_shell(serial))
    # `adb state=device` precedes Android restoring persisted global settings. The live failure
    # behind this recovery read unproxied here, then became blackholed seconds later. Always wait
    # for the settled boot and check again; this is the result exposed to callers.
    startup_step(lambda: _wait_for_boot(shell, timeout_s=min(90.0, wait_s)))
    proxy_cleanup = startup_step(
        lambda: _clear_inherited_blackholed_proxy(serial, cache_dir=cache_dir)
    )

    # Animations off by default. Measured on a windowed AVD: a tap settles in 272ms instead
    # of 357ms, and the spread narrows from 225ms to 69ms — the predictability matters more
    # than the mean, because every wait-for-settle is sized by the worst case. Scoped to an
    # AVD WE booted: `dev anim off` writes global settings that outlive the process, so doing
    # it to a device someone else started would silently change their manual QA or break a
    # visual/animation test. `--animations` keeps them.
    animations_disabled = False
    if not animations:
        with contextlib.suppress(Exception):
            from . import devopts

            state = devopts.anim_off(shell, _pid_dir(cache_dir) / f"{inst}.anim.json")
            # Read back rather than assume: claiming this without checking is the same
            # false-success the wait above was hiding.
            anim = (state or {}).get("anim") or {}
            animations_disabled = bool(anim) and all(
                str(v) in ("0", "0.0") for v in anim.values()
            )
    stop_hint = (
        f"`aua emulator stop --serial {serial}`"
        if owner_tag
        else f"`aua emulator stop --mine` (or `--avd {avd}` / `--serial {serial}`)"
    )
    if owner_tag:
        stop_hint = (
            f"`AUA_OWNER={owner_tag} aua emulator stop --mine` "
            f"or `aua emulator stop --serial {serial}` / `--owner {owner_tag}`"
        )
    return {
        "ok": True,
        "action": "emulator-start",
        "avd": avd,
        "instance": inst,
        "instance_token": boot_token,
        "serial": serial,
        "port": console_port,
        "owner": owner_tag,
        "read_only": bool(read_only),
        "parallel": bool(parallel),
        "pid": proc.pid,
        "headless": headless,
        "gpu": gpu_mode,
        "animations_disabled": animations_disabled,
        "proxy_cleanup": proxy_cleanup,
        "idle_timeout_s": idle_timeout_s,
        "watchdog_pid": watchdog_pid,
        "log": str(log_path),
        "hint": (
            "Standalone start only provisions the emulator. Agents should begin goal work with "
            "`aua session start --goal <goal>`, which selects and leases automatically; then omit "
            "`--serial` from ordinary commands. "
            f"**Required when finished:** {stop_hint}. "
            + (
                f"Safety net: auto-stops after {int(idle_timeout_s)}s idle."
                if idle_timeout_s > 0
                else "Idle auto-stop disabled."
            )
        ),
    }


def avd_name_of_serial(serial: str) -> str | None:
    """Ask the running emulator which AVD it is, via its console (``emu avd name``)."""
    try:
        out = subprocess.run(  # noqa: S603
            [adb_bin(), "-s", serial, "emu", "avd", "name"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except Exception:
        return None
    for line in (out or "").splitlines():
        line = line.strip()
        if line and line.upper() != "OK":
            return line
    return None


def _wait_for_serial(serial: str, *, timeout_s: float, expect_avd: str | None = None) -> str | None:
    """Wait for ``serial`` to appear, optionally proving it is the AVD we launched.

    ``expect_avd`` guards against answering to somebody else's device: if two instances
    ever contend for one console port, the port's real owner satisfies this wait and the
    caller would happily drive a device it does not own. Verifying the AVD name turns that
    silent mix-up into a hard failure.
    """

    def _match(d: dict[str, Any]) -> bool:
        if d.get("state") != "device" or d.get("serial") != serial:
            return False
        if expect_avd is None:
            return True
        actual = avd_name_of_serial(serial)
        # An emulator that will not answer its console yet is not a mismatch; keep waiting.
        return actual is None or actual == expect_avd

    deadline = time.monotonic() + max(5.0, timeout_s)
    while time.monotonic() < deadline:
        for d in running_emulators():
            if _match(d):
                return serial
        time.sleep(_POLL_S)
    for d in running_emulators():
        if _match(d):
            return serial
    return None


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


def _aua_started_records(cache_dir: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _pid_dir(cache_dir).glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_instance_record(meta):
            continue
        meta = dict(meta)
        meta["_path"] = str(path)
        out.append(meta)
    return out


def adopt_idle_watchdogs(
    *,
    cache_dir: str | Path,
    idle_timeout_s: float,
    lease_registry_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Give a watchdog back to any aua-started emulator that is running without a live one.

    The watchdog is a *process*, spawned once at boot. Reboot the host, `pkill -f python`, or let
    it crash, and the emulator it guarded becomes immortal — nothing re-spawns it, and nothing
    else looks for aua-started emulators running unguarded. Observed on a dev host: an instance
    recorded with `idle_timeout_s: 900` and `watchdog_pid: None`, plus a live emulator whose
    record said `0`, i.e. two different ways to end up with no safety net at all.

    Deliberately narrow:

    * Only instances AUA itself booted (`started_by_aua`). An emulator someone opened in Android
      Studio is their session, and stopping it would be inexcusable.
    * Only ones actually visible to adb, so a stale record does not spawn a watchdog for a device
      that is already gone.
    * An explicit ``--idle-stop 0`` is honoured. "Never stop this" is a real instruction, and
      quietly overriding it a day later would be worse than the leak.

    Returns one entry per adopted instance, for the caller to log.
    """
    root = Path(cache_dir).expanduser()
    timeout = max(0.0, float(idle_timeout_s))
    if timeout <= 0:
        return []
    try:
        live = {d.get("serial") for d in running_emulators()}
    except Exception:
        return []
    adopted: list[dict[str, Any]] = []
    for meta in _aua_started_records(root):
        # `_aua_started_records` is named for its directory, not for this flag: it filters on
        # `avd` alone (see `_is_instance_record`), so it will return a record that never claimed
        # AUA booted it. `run_watchdog` refuses such a record anyway, which would make this a
        # wasted process rather than a stopped emulator — but a function whose contract is "only
        # instances AUA booted" has to check that itself, not inherit it.
        if not meta.get("started_by_aua"):
            continue
        serial = meta.get("serial")
        if not isinstance(serial, str) or serial not in live:
            continue
        if meta.get("idle_stop_explicit") and float(meta.get("idle_timeout_s") or 0) <= 0:
            continue  # its owner asked for no auto-stop
        wpid = meta.get("watchdog_pid")
        if isinstance(wpid, int) and wpid > 1:
            try:
                os.kill(wpid, 0)
                continue  # already guarded
            except (PermissionError, OSError) as exc:
                if isinstance(exc, PermissionError):
                    continue  # alive, just not ours to signal
            except ProcessLookupError:
                pass  # gone — fall through and replace it
        path = Path(str(meta.get("_path") or ""))
        if not path.is_file():
            continue
        instance = path.stem
        fresh = {k: v for k, v in meta.items() if k != "_path"}
        # A record that never had a timeout gets the configured one; never shorten a longer
        # deliberate choice, because that would be a silent policy change on someone's boot.
        fresh["idle_timeout_s"] = max(timeout, float(meta.get("idle_timeout_s") or 0))
        # No activity has been *observed* since whenever this record was last touched, so the
        # existing stamp is the honest baseline — resetting it here would hand every adopted
        # emulator a fresh 20 minutes on every command and it would never age out at all.
        fresh.setdefault("last_activity", float(meta.get("started_at") or time.time()))
        pid = _spawn_idle_watchdog(
            cache_dir=root,
            instance=instance,
            lease_registry_dir=lease_registry_dir,
        )
        fresh["watchdog_pid"] = pid
        with contextlib.suppress(OSError):
            atomic_write_text(path, json.dumps(fresh, indent=2) + "\n")
        adopted.append(
            {
                "instance": instance,
                "serial": serial,
                "idle_timeout_s": fresh["idle_timeout_s"],
                "watchdog_pid": pid,
            }
        )
    return adopted


def discards_writes(serial: str, *, cache_dir: str | Path | None = None) -> bool:
    """Was *serial* booted ``-read-only``, so on-disk changes land in a throwaway overlay?

    A ``-read-only`` emulator accepts an install, prints ``Success``, and loses the app when it
    stops — the failure mode `aua emulator start --parallel` documents. Anything that writes to
    the device image (installing a build above all) has to ask first, because "reported success,
    changed nothing" is the one outcome a caller cannot detect afterwards.

    Answers ``False`` when AUA has no record of booting *serial*: an emulator someone else
    started, or a physical device, is not knowably read-only, and refusing on a guess would block
    the ordinary case.
    """

    if cache_dir is None:
        return False
    for meta in _aua_started_records(cache_dir):
        if str(meta.get("serial") or "") != serial:
            continue
        if meta.get("read_only") or meta.get("parallel"):
            return True
    return False


STOP_LOG_NAME = "stops.log"


def _stop_origin(requested_by: str | None) -> dict[str, Any]:
    """Identify the process asking for a stop.

    ``owner`` here is the *requester's* own ``$AUA_OWNER``, which is not the same thing as the
    owner a stop is scoped to: the accident this exists to detect is a coordinator stopping a
    worker's device, where those two differ.
    """
    return {
        "requested_by": requested_by or "cli",
        "requester_owner": resolve_owner(None),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        # The command line settles "which path asked" when the branch name is not enough --
        # e.g. a `--mine` in a shell that had a stale $AUA_OWNER exported.
        "argv": [str(a) for a in sys.argv[:16]],
    }


def _log_stop(
    cache_dir: str | Path,
    *,
    origin: dict[str, Any],
    requested_via: str,
    request: dict[str, Any],
    matched: list[dict[str, Any]],
    stopped: list[str],
    skipped_leased: list[dict[str, Any]] | None = None,
) -> None:
    """Append an attributable record of one stop, and say it out loud on stderr.

    A device died under a live worker mid-scenario and the cause could never be settled. The
    emulator's own log showed AUA's *graceful* shutdown sequence beginning at that exact second,
    so the stop had certainly been requested — but no log anywhere recorded the requester's
    owner, its pid, or which code path asked (explicit serial, owner match, or the idle
    watchdog). Coordinator error could therefore not be ruled out, and an unattributable stop is
    what makes a shared pool untrustworthy: on the same night a coordinator legitimately stopped
    two instances, and there was no way to prove those calls had not also taken this one.

    The record lands beside ``{instance}.log`` — the file that investigation already read — and
    is best-effort: a stop must never fail because its audit line could not be written.
    """
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": time.time(),
        "action": "emulator-stop",
        "requested_via": requested_via,
        "origin": origin,
        "request": request,
        "matched": matched,
        "stopped": stopped,
    }
    if skipped_leased:
        entry["skipped_leased"] = skipped_leased
        logger.warning(
            "emulator stop skipped leased target(s): %s",
            ", ".join(f"{s.get('serial')} (held by {s.get('holder')})" for s in skipped_leased),
        )
    logger.warning(
        "emulator stop via=%s stopped=%s by pid=%s owner=%s request=%s",
        requested_via,
        stopped or "nothing",
        origin.get("pid"),
        origin.get("requester_owner"),
        {k: v for k, v in request.items() if v},
    )
    with contextlib.suppress(OSError):
        directory = _pid_dir(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / STOP_LOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _instance_identity(meta: dict[str, Any]) -> dict[str, Any]:
    """The subset of a pid record that names *which* instance a scoped stop matched."""
    return {
        "instance": meta.get("instance") or Path(str(meta.get("_path") or "")).stem or None,
        "serial": meta.get("serial"),
        "avd": meta.get("avd"),
        "owner": meta.get("owner"),
        "pid": meta.get("pid"),
    }


def _owned_instance_pid(meta: dict[str, Any]) -> int | None:
    """Recorded process id only for an exact, positive AUA instance identity."""

    pid = meta.get("pid")
    path = Path(str(meta.get("_path") or ""))
    instance = meta.get("instance")
    if (
        meta.get("started_by_aua") is True
        and isinstance(pid, int)
        and pid > 1
        and isinstance(instance, str)
        and bool(instance)
        and path.stem == instance
    ):
        return pid
    return None


def stop_spawned_instance(
    *,
    instance: str,
    pid: int | None,
    cache_dir: str | Path,
    lease_registry_dir: str | Path | None = None,
    owner: str | None = None,
    requested_by: str | None = None,
    requested_via: str = "spawned-instance",
    expected_instance_token: str | None = None,
) -> dict[str, Any]:
    """Roll back exactly one boot this process performed — never an ambiguous serial.

    The claim-failure rollback that motivated this used ``stop(serial=...)``. When
    provisioning had collided with a foreign worker's emulator (adb briefly omitted it, so
    its console port looked free), that serial named *their* device, and the rollback shut
    it down mid-run. A rollback owns only what it demonstrably created: the recorded
    emulator process and its instance record under this cache.

    The recorded process group is stopped only when its instance identity is exact and no newer
    live foreign lease may have claimed it. The shared ADB server is never involved. When a
    foreign lease exists, the timeline decides what is still ours:

    * lease acquired **before** our boot record — the serial always belonged to the other
      worker, so our spawned process is an unbound loser we may (and must) terminate;
    * lease acquired **after** our boot — another agent legitimately claimed the fresh
      device between our boot and our failed claim, so nothing here is ours to touch.
    """
    origin = _stop_origin(requested_by)
    if instance and (Path(instance).name != instance or instance in {".", ".."}):
        raise UsageError("invalid Android instance identity")
    meta_path = _pid_dir(cache_dir) / f"{instance}.json" if instance else None
    meta: dict[str, Any] = {}
    if meta_path is not None and meta_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
    positive_pid = _owned_instance_pid({**meta, "_path": str(meta_path or "")})
    record_is_ours = (
        positive_pid is not None
        and (pid is None or positive_pid == pid)
        and (expected_instance_token is None or meta.get("instance_token") == expected_instance_token)
    )
    serial = (
        str(meta["serial"])
        if record_is_ours and isinstance(meta.get("serial"), str) and meta.get("serial")
        else None
    )
    skipped: list[dict[str, Any]] = []
    stopped: list[str] = []
    own_pid = positive_pid if record_is_ours else None
    process_gone = False
    with _stop_lease_transaction(lease_registry_dir, serial):
        # Re-read after obtaining the target fence. A new boot can replace a record while
        # cleanup was waiting; neither its process nor its bookkeeping belongs to this call.
        if record_is_ours and meta_path is not None:
            try:
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            identity_fields = ("instance", "instance_token", "pid", "started_at", "serial")
            if not isinstance(current, dict) or any(current.get(k) != meta.get(k) for k in identity_fields):
                record_is_ours = False
                own_pid = None
            started = meta.get("process_started")
            if record_is_ours and started:
                from .leases import _proc_started

                if _proc_started(int(positive_pid or 0)) != started:
                    own_pid = None
                    try:
                        os.kill(int(positive_pid or 0), 0)
                    except ProcessLookupError:
                        process_gone = True
                    except OSError:
                        record_is_ours = False  # Inaccessible is not proof of death.
                    else:
                        record_is_ours = False  # A live replacement pid is never ours.
        # The lease read and every process/device kill stay inside the same exclusive
        # target fence used by acquire/release/transfer. No process can claim this serial
        # after the check and before the kill.
        foreign = _live_foreign_lease(lease_registry_dir, serial, owner)
        may_kill_own_process = True
        if foreign is not None and serial:
            skipped.append(
                {"serial": serial, "holder": str(foreign.get("owner") or "unknown")}
            )
            lease_acquired = float(foreign.get("acquired") or 0)
            boot_started = float(meta.get("started_at") or 0)
            if not lease_acquired or not boot_started or lease_acquired >= boot_started:
                # The lease may cover the device we just booted (claimed by another agent in
                # the gap before our own claim). Terminating our process would kill it.
                may_kill_own_process = False
        if may_kill_own_process and isinstance(own_pid, int) and own_pid > 1:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(own_pid, signal.SIGTERM)
                if serial and foreign is None:
                    stopped.append(serial)
        if record_is_ours and may_kill_own_process and meta_path is not None:
            if not process_gone:
                _kill_watchdog(meta)
            meta_path.unlink(missing_ok=True)
            port = meta.get("port")
            if isinstance(port, int) and not process_gone:
                release_console_port(port)
    matched = (
        [_instance_identity({**meta, "_path": str(meta_path)})] if record_is_ours else []
    )
    _log_stop(
        cache_dir,
        origin=origin,
        requested_via=requested_via,
        request={"instance": instance, "pid": pid, "serial": serial},
        matched=matched,
        stopped=stopped,
        skipped_leased=skipped or None,
    )
    return {
        "ok": True,
        "action": "emulator-stop",
        "stopped": stopped,
        "matched": matched,
        "skipped_leased": skipped,
        "requested_via": requested_via,
        "origin": origin,
        "instance": instance or None,
    }


def stop(
    *,
    serial: str | None = None,
    avd: str | None = None,
    all_devices: bool = False,
    mine: bool = False,
    owner: str | None = None,
    cache_dir: str | Path,
    requested_by: str | None = None,
    lease_registry_dir: str | Path | None = None,
    lease_owner: str | None = None,
) -> dict[str, Any]:
    """Stop a running emulator (``adb emu kill``), scoped by serial/AVD/owner/mine.

    When *lease_registry_dir* is given, any matched target holding a live lease that
    *lease_owner* does not own is skipped rather than killed, and reported under
    ``skipped_leased``: a device another live agent is driving is never AUA's to stop,
    however it was scoped. Free it first with ``aua lease release <serial>`` (its holder),
    or wait for the holder process to exit.

    An untargeted call is refused rather than treated as "all". ``--all`` kills every
    emulator; ``--mine`` kills AVDs recorded under aua's cache (optionally filtered by
    ``owner`` / ``$AUA_OWNER`` so parallel agents only tear down their own).

    Every stop is attributed: ``requested_by`` names the code path (``cli`` by default,
    ``idle-watchdog`` from the watchdog), and the returned ``origin`` / ``requested_via`` are
    also appended to ``<cache>/emulator/stops.log``. See :func:`_log_stop`.
    """
    owner_tag = resolve_owner(owner)
    origin = _stop_origin(requested_by)
    request = {
        "serial": serial,
        "avd": avd,
        "owner": owner_tag,
        "mine": bool(mine),
        "all": bool(all_devices),
    }
    if not serial and avd is None and not all_devices and not mine and not owner_tag:
        targets = running_emulators()
        raise UsageError(
            "emulator stop needs a target: --serial, --avd, --owner, --mine, or --all",
            hint=(
                "running: "
                + (", ".join(t["serial"] for t in targets) or "none")
                + ". Parallel agents: `aua emulator stop --serial <yours>` or "
                "`AUA_OWNER=… aua emulator stop --mine`."
            ),
        )

    # --owner alone (or with --mine): stop matching aua records.
    if (mine or owner_tag) and not serial and avd is None and not all_devices:
        all_records = _aua_started_records(cache_dir)
        owned_records = [meta for meta in all_records if _owned_instance_pid(meta) is not None]
        if owner_tag:
            records = [m for m in owned_records if m.get("owner") == owner_tag]
            detail = f"stopped emulators owned by {owner_tag!r}"
        else:
            records = owned_records
            detail = "stopped emulators recorded as started by aua"
        # An owner-scoped stop must say exactly which instances it matched, so a caller can see
        # it hit one device and not several — the question nobody could answer after a worker
        # lost its emulator mid-scenario.
        matched = [_instance_identity(m) for m in records]
        considered = [_instance_identity(m) for m in all_records]
        if not records:
            payload = {
                "ok": True,
                "action": "emulator-stop",
                "stopped": [],
                "owner": owner_tag,
                "matched": [],
                "considered": considered,
                "requested_via": "owner-scope",
                "origin": origin,
                "detail": (
                    "no aua-started emulator records"
                    + (f" for owner {owner_tag!r}" if owner_tag else " in cache")
                ),
            }
            _log_stop(
                cache_dir,
                origin=origin,
                requested_via="owner-scope",
                request=request,
                matched=[],
                stopped=[],
            )
            return payload
        stopped_mine: list[str] = []
        skipped_mine: list[dict[str, Any]] = []
        for meta in records:
            ser = meta.get("serial")
            target_serial = ser if isinstance(ser, str) else None
            with _stop_lease_transaction(lease_registry_dir, target_serial):
                foreign = _live_foreign_lease(
                    lease_registry_dir, target_serial, lease_owner
                )
                if foreign is not None:
                    # A record AUA wrote does not outrank a live lease: the instance may have
                    # been handed to another agent, and its pid is that agent's device now.
                    skipped_mine.append(
                        {"serial": ser, "holder": str(foreign.get("owner") or "unknown")}
                    )
                    continue
                pid = _owned_instance_pid(meta)
                if pid is not None:
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        os.killpg(pid, signal.SIGTERM)
                        if target_serial:
                            stopped_mine.append(target_serial)
                _kill_watchdog(meta)
                path = Path(str(meta.get("_path") or ""))
                if path.is_file():
                    path.unlink(missing_ok=True)
        _log_stop(
            cache_dir,
            origin=origin,
            requested_via="owner-scope",
            request=request,
            matched=matched,
            stopped=stopped_mine,
            skipped_leased=skipped_mine or None,
        )
        return {
            "ok": True,
            "action": "emulator-stop",
            "stopped": stopped_mine,
            "skipped_leased": skipped_mine,
            "owner": owner_tag,
            "matched": matched,
            "considered": considered,
            "requested_via": "owner-scope",
            "origin": origin,
            "detail": detail,
        }

    if serial and avd is None and not all_devices:
        serial_records = [
            meta
            for meta in _aua_started_records(cache_dir)
            if meta.get("serial") == serial
            and _owned_instance_pid(meta) is not None
        ]
        if len(serial_records) == 1:
            record = serial_records[0]
            return stop_spawned_instance(
                instance=str(record.get("instance") or Path(str(record["_path"])).stem),
                pid=_owned_instance_pid(record),
                cache_dir=cache_dir,
                lease_registry_dir=lease_registry_dir,
                owner=lease_owner,
                requested_by=requested_by,
                requested_via="serial",
            )

    avd_has_owned_records = avd is not None and any(
        _owned_instance_pid(meta) is not None
        and (meta.get("avd") == avd or Path(str(meta.get("_path") or "")).stem == avd)
        for meta in _aua_started_records(cache_dir)
    )
    targets = [] if avd_has_owned_records else running_emulators()
    if serial:
        targets = [t for t in targets if t["serial"] == serial]

    if avd is not None:
        # Match legacy `{avd}.json` and parallel `{avd}.p{port}.json` records.
        avd_records = [
            m
            for m in _aua_started_records(cache_dir)
            if _owned_instance_pid(m) is not None
            and (m.get("avd") == avd or Path(str(m.get("_path") or "")).stem == avd)
        ]
        if avd_records:
            stopped_avd: list[str] = []
            skipped_avd: list[dict[str, Any]] = []
            for meta in avd_records:
                ser = meta.get("serial")
                target_serial = ser if isinstance(ser, str) else None
                with _stop_lease_transaction(lease_registry_dir, target_serial):
                    foreign = _live_foreign_lease(
                        lease_registry_dir, target_serial, lease_owner
                    )
                    if foreign is not None:
                        skipped_avd.append(
                            {"serial": ser, "holder": str(foreign.get("owner") or "unknown")}
                        )
                        continue
                    pid = _owned_instance_pid(meta)
                    if pid is not None:
                        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                            os.killpg(pid, signal.SIGTERM)
                            if target_serial:
                                stopped_avd.append(target_serial)
                    _kill_watchdog(meta)
                    path = Path(str(meta.get("_path") or ""))
                    if path.is_file():
                        path.unlink(missing_ok=True)
            avd_matched = [_instance_identity(m) for m in avd_records]
            _log_stop(
                cache_dir,
                origin=origin,
                requested_via="avd-records",
                request=request,
                matched=avd_matched,
                stopped=stopped_avd,
                skipped_leased=skipped_avd or None,
            )
            return {
                "ok": True,
                "action": "emulator-stop",
                "stopped": stopped_avd,
                "skipped_leased": skipped_avd,
                "matched": avd_matched,
                "requested_via": "avd-records",
                "origin": origin,
                "detail": f"stopped aua instances of avd {avd}",
            }
        # Fall through: maybe a live emulator whose serial we don't have recorded.
        meta_path = _pid_dir(cache_dir) / f"{avd}.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            ser = meta.get("serial")
            if ser:
                targets = [t for t in running_emulators() if t["serial"] == ser] or targets

    if not targets and serial is None and avd is None:
        stopped_pids: list[int] = []
        for path in list(_pid_dir(cache_dir).glob("*.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
                continue
            ser = meta.get("serial") if isinstance(meta, dict) else None
            target_serial = ser if isinstance(ser, str) else None
            with _stop_lease_transaction(lease_registry_dir, target_serial):
                if (
                    _live_foreign_lease(lease_registry_dir, target_serial, lease_owner)
                    is not None
                ):
                    continue  # a record whose device another live agent leases is not residue
                pid = _owned_instance_pid({**meta, "_path": str(path)})
                if pid is not None:
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        os.killpg(pid, signal.SIGTERM)
                        stopped_pids.append(pid)
                _kill_watchdog(meta)
                path.unlink(missing_ok=True)
        _log_stop(
            cache_dir,
            origin=origin,
            requested_via="registry-cleanup",
            request=request,
            matched=[],
            stopped=[],
        )
        return {
            "ok": True,
            "action": "emulator-stop",
            "stopped": [],
            "matched": [],
            "requested_via": "registry-cleanup",
            "origin": origin,
            "detail": "no running emulator-* devices; cleared aua pid records if any",
            "signalled_pids": stopped_pids,
        }

    if not targets:
        raise DeviceError(
            "no matching running emulator to stop",
            hint="`aua emulator status` lists live serials; pass --serial emulator-5554 "
            "or `aua emulator stop --mine` / `--owner`.",
        )

    stopped: list[str] = []
    skipped_live: list[dict[str, Any]] = []
    for t in targets:
        ser = t["serial"]
        with _stop_lease_transaction(lease_registry_dir, ser):
            foreign = _live_foreign_lease(lease_registry_dir, ser, lease_owner)
            if foreign is not None:
                skipped_live.append(
                    {"serial": ser, "holder": str(foreign.get("owner") or "unknown")}
                )
                continue
            try:
                _adb_emu_kill(ser)
                stopped.append(ser)
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                logger.debug("emu kill %s failed: %s", ser, exc)

    for path in list(_pid_dir(cache_dir).glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            continue
        if meta.get("serial") in stopped or (avd and meta.get("avd") == avd):
            _kill_watchdog(meta)
            path.unlink(missing_ok=True)

    requested_via = "serial" if serial else ("all" if all_devices else "running-list")
    live_matched = [{"instance": None, "serial": t["serial"]} for t in targets]
    _log_stop(
        cache_dir,
        origin=origin,
        requested_via=requested_via,
        request=request,
        matched=live_matched,
        stopped=stopped,
        skipped_leased=skipped_live or None,
    )
    result: dict[str, Any] = {
        "ok": True,
        "action": "emulator-stop",
        "stopped": stopped,
        "skipped_leased": skipped_live,
        "matched": live_matched,
        "requested_via": requested_via,
        "origin": origin,
    }
    if skipped_live:
        result["hint"] = (
            "Skipped target(s) hold a live lease owned by another agent. A leased device is "
            "never stopped; its holder must release it (`aua lease release <serial>`), or its "
            "lease frees the moment the holder process exits."
        )
    return result


# --------------------------------------------------------------------------- neutral service API


def _virtual_definition(detail: dict[str, Any]) -> VirtualTargetDefinition:
    """Translate one Android AVD record into the platform-neutral definition contract."""

    name = str(detail.get("name") or "").strip()
    return VirtualTargetDefinition(
        definition_id=name,
        capabilities={
            "root": detail.get("rootable") is True,
            "proxy": detail.get("rootable") is True,
            "play": detail.get("play_store") is True,
        },
        details=detail,
    )


def _virtual_instance(
    payload: dict[str, Any], *, require_owned_token: bool
) -> VirtualTargetInstance:
    target_id = str(payload.get("serial") or payload.get("target_id") or "").strip()
    token = str(payload.get("instance_token") or payload.get("instance") or "").strip() or None
    if require_owned_token and token is None:
        raise DeviceError(
            "Android virtual-target startup returned no owned instance token",
            code="platform_capability_invalid",
            hint="Update AUA; rollback cannot safely stop this boot by serial alone.",
        )
    known = {
        "ok",
        "action",
        "serial",
        "target_id",
        "instance",
        "instance_token",
        "avd",
        "definition_id",
        "owner",
        "pid",
    }
    return VirtualTargetInstance(
        target_id=target_id,
        instance_token=token,
        definition_id=(
            str(payload.get("avd") or payload.get("definition_id") or "").strip() or None
        ),
        owner=str(payload["owner"]) if payload.get("owner") is not None else None,
        pid=payload.get("pid") if isinstance(payload.get("pid"), int) else None,
        details={key: value for key, value in payload.items() if key not in known},
        legacy_result=payload,
    )


def list_virtual_targets() -> VirtualTargetList:
    """List reusable Android definitions through the neutral virtual-target contract."""

    payload = list_avds()
    definitions = tuple(
        _virtual_definition(item)
        for item in payload.get("details", [])
        if isinstance(item, dict) and item.get("name")
    )
    details = {
        key: value
        for key, value in payload.items()
        if key not in {"ok", "action", "avds", "details", "count"}
    }
    return VirtualTargetList(
        definitions=definitions,
        details=details,
        legacy_result=payload,
    )


def select_virtual_target(
    definition_id: str | None = None,
    *,
    needs: list[str] | tuple[str, ...] | None = None,
) -> VirtualTargetDefinition:
    """Select an AVD by neutral definition id and capability requirements."""

    selected = select_avd_for_session(definition_id, needs=list(needs or ()))
    detail = inspect_avd(selected)
    if not detail.get("name"):
        detail = {**detail, "name": selected}
    return _virtual_definition(detail)


def start_virtual_target(request: VirtualTargetStartRequest) -> VirtualTargetInstance:
    """Start one Android definition and return its exact owned-instance token."""

    options = dict(request.options)
    allowed = {"extra_args", "gpu", "idle_timeout_s", "port", "read_only"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise UsageError(
            f"unsupported Android virtual-target start option(s): {', '.join(unknown)}",
            hint=f"Supported options: {', '.join(sorted(allowed))}.",
        )
    extra_args = options.get("extra_args")
    if extra_args is not None and not isinstance(extra_args, list):
        raise UsageError("virtual-target option extra_args must be a list of strings")
    payload = start(
        request.definition_id,
        headless=request.headless,
        animations=request.animations,
        audio=request.audio,
        wait_s=request.wait_s,
        cache_dir=request.cache_dir,
        extra_args=[str(item) for item in extra_args] if extra_args is not None else None,
        gpu=str(options["gpu"]) if options.get("gpu") is not None else None,
        idle_timeout_s=(
            float(options["idle_timeout_s"])
            if options.get("idle_timeout_s") is not None
            else None
        ),
        port=int(options["port"]) if options.get("port") is not None else None,
        read_only=(
            bool(options["read_only"]) if options.get("read_only") is not None else None
        ),
        parallel=request.parallel,
        owner=str(request.owner) if request.owner is not None else None,
        lease_registry_dir=request.lease_registry_dir,
    )
    return _virtual_instance(payload, require_owned_token=True)


def provision_virtual_target(
    request: VirtualTargetProvisionRequest,
) -> VirtualTargetInstance:
    """Select a compatible Android definition and start it as one semantic operation."""

    # Visibility/audio describe the boot we are about to create, not immutable AVD-image facts.
    # Android's image selector owns root/Play/proxy; the common start flags own the other two.
    image_needs = [item for item in request.needs if item in {"root", "play", "proxy"}]
    unknown = sorted(set(request.needs) - {"root", "play", "proxy", "headed", "audio"})
    if unknown:
        raise UsageError(
            f"cannot provision Android for unknown requirement(s): {', '.join(unknown)}",
            hint="Supported provisioning requirements are root, play, proxy, headed, and audio.",
        )
    selected = select_virtual_target(request.definition_id, needs=image_needs)
    return start_virtual_target(
        VirtualTargetStartRequest(
            definition_id=selected.definition_id,
            headless=request.headless,
            audio=request.audio,
            animations=request.animations,
            wait_s=request.wait_s,
            cache_dir=request.cache_dir,
            lease_registry_dir=request.lease_registry_dir,
            owner=request.owner,
            parallel=request.parallel,
            options=request.options,
        )
    )


def virtual_target_status(
    *, cache_dir: str | Path | None = None
) -> VirtualTargetStatus:
    """Return configured, running, and AUA-owned Android virtual targets."""

    payload = status(cache_dir=cache_dir)
    definitions = tuple(
        _virtual_definition(item)
        for item in payload.get("avd_details", [])
        if isinstance(item, dict) and item.get("name")
    )
    records = {
        str(item.get("serial")): item
        for item in payload.get("started_by_aua", [])
        if isinstance(item, dict) and item.get("serial")
    }
    running: list[VirtualTargetInstance] = []
    for raw in payload.get("running", []):
        if not isinstance(raw, dict) or not raw.get("serial"):
            continue
        serial = str(raw["serial"])
        record = records.get(serial, {})
        combined = {**raw, **record, "serial": serial}
        running.append(_virtual_instance(combined, require_owned_token=False))
    owned = tuple(
        _virtual_instance(item, require_owned_token=False)
        for item in records.values()
        if item.get("serial")
    )
    details = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "ok",
            "action",
            "avds",
            "avd_details",
            "running",
            "started_by_aua",
        }
    }
    return VirtualTargetStatus(
        definitions=definitions,
        running=tuple(running),
        owned=owned,
        details=details,
        legacy_result=payload,
    )


def stop_virtual_targets(request: VirtualTargetStopRequest) -> VirtualTargetStopResult:
    """Stop matching Android virtual targets through neutral selection vocabulary."""

    payload = stop(
        serial=request.target_id,
        avd=request.definition_id,
        all_devices=request.all_targets,
        mine=request.mine,
        owner=request.owner,
        cache_dir=request.cache_dir,
        requested_by=request.requested_by,
        lease_registry_dir=request.lease_registry_dir,
        # Preserve LeaseOwner's process identity; stringifying it would make this caller look
        # foreign to its own live lease during the stop transaction.
        lease_owner=cast(str | None, request.lease_owner),
    )
    stopped = tuple(str(item) for item in payload.get("stopped", []) if str(item))
    preserved = tuple(
        str(item.get("serial"))
        for item in payload.get("skipped_leased", [])
        if isinstance(item, dict) and item.get("serial")
    )
    details = {
        key: value
        for key, value in payload.items()
        if key not in {"ok", "action", "stopped", "skipped_leased"}
    }
    if payload.get("skipped_leased"):
        details["skipped_leased"] = payload["skipped_leased"]
    return VirtualTargetStopResult(
        stopped_target_ids=stopped,
        preserved_target_ids=preserved,
        details=details,
        legacy_result=payload,
    )


def stop_virtual_target_instance(
    request: OwnedVirtualTargetStopRequest,
) -> VirtualTargetStopResult:
    """Roll back exactly one Android boot using its opaque owned-instance token."""

    stem, separator, nonce = request.instance_token.rpartition("@")
    scoped_token = bool(separator and stem and re.fullmatch(r"[0-9a-f]{32}", nonce))
    if not scoped_token and request.expected_pid is None:
        # Old metadata stems identify a reusable AVD/port, not a boot. They need an explicit
        # captured pid; current starts always carry a unique token instead.
        raise UsageError("legacy Android instance cleanup requires the original process id")
    payload = stop_spawned_instance(
        instance=stem if scoped_token else request.instance_token,
        pid=request.expected_pid,
        expected_instance_token=request.instance_token if scoped_token else None,
        cache_dir=request.cache_dir,
        lease_registry_dir=request.lease_registry_dir,
        # The opaque owner may be a LeaseOwner carrying pid/start metadata used by the lease gate.
        owner=cast(str | None, request.owner),
        requested_by=request.requested_by,
    )
    stopped = tuple(str(item) for item in payload.get("stopped", []) if str(item))
    preserved = tuple(
        str(item.get("serial"))
        for item in payload.get("skipped_leased", [])
        if isinstance(item, dict) and item.get("serial")
    )
    return VirtualTargetStopResult(
        stopped_target_ids=stopped,
        preserved_target_ids=preserved,
        details={
            key: value
            for key, value in payload.items()
            if key not in {"ok", "action", "stopped", "skipped_leased"}
        },
        legacy_result=payload,
    )


def reclaim_virtual_targets(
    request: VirtualTargetReclaimRequest,
) -> tuple[VirtualTargetInstance, ...]:
    """Re-arm retirement supervision only for Android instances AUA owns."""

    adopted = adopt_idle_watchdogs(
        cache_dir=request.cache_dir,
        idle_timeout_s=request.idle_timeout_s,
        lease_registry_dir=request.lease_registry_dir,
    )
    return tuple(
        _virtual_instance(item, require_owned_token=False)
        for item in adopted
        if item.get("serial")
    )


def create_virtual_target(request: VirtualTargetCreateRequest) -> VirtualTargetCreateResult:
    """Create Android's supported rootable virtual-target definition.

    Android's existing creation surface intentionally provisions the small non-Play profile used
    for proxy/system-CA work.  The neutral contract does not freeze that policy: another adapter
    owns and validates its own ``options``.
    """

    options = dict(request.options)
    allowed = {"accept_licenses", "api", "profile"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise UsageError(
            f"unsupported Android virtual-target create option(s): {', '.join(unknown)}",
            hint=f"Supported options: {', '.join(sorted(allowed))}.",
        )
    profile = str(options.get("profile") or "proxy").strip().lower()
    if profile not in {"proxy", "rootable"}:
        raise UsageError(
            f"unsupported Android virtual-target profile {profile!r}",
            hint="Android currently creates the rootable/proxy profile; existing AVDs remain usable.",
        )
    payload = ensure_proxy_avd(
        name=request.definition_id,
        api=int(options.get("api", PROXY_API_DEFAULT)),
        force=request.replace,
        accept_licenses=bool(options.get("accept_licenses", True)),
    )
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        detail = inspect_avd(request.definition_id)
    if not detail.get("name"):
        detail = {**detail, "name": request.definition_id}
    return VirtualTargetCreateResult(
        definition=_virtual_definition(detail),
        created=bool(payload.get("created")),
        details={
            key: value
            for key, value in payload.items()
            if key not in {"ok", "action", "created", "detail"}
        },
        legacy_result=payload,
    )


def delete_virtual_target(request: VirtualTargetDeleteRequest) -> VirtualTargetDeleteResult:
    """Delete one stopped Android AVD definition after explicit confirmation."""

    if request.options:
        unknown = ", ".join(sorted(str(key) for key in request.options))
        raise UsageError(f"unsupported Android virtual-target delete option(s): {unknown}")
    if not request.confirmed:
        raise UsageError(
            "deleting a virtual-target definition needs explicit confirmation",
            hint="Re-run with --yes after confirming the target definition and its data may be removed.",
        )
    name = request.definition_id
    detail = inspect_avd(name)
    if not detail.get("config"):
        return VirtualTargetDeleteResult(
            definition_id=name,
            deleted=False,
            details={"reason": "not_found"},
        )
    running = [
        str(item.get("serial"))
        for item in running_emulators()
        if item.get("serial") and avd_name_of_serial(str(item["serial"])) == name
    ]
    if running:
        raise UsageError(
            f"virtual-target definition {name!r} is running on {', '.join(running)}",
            hint="Stop the exact target first; deleting a live definition is refused.",
        )
    proc = _run_sdk(
        [avdmanager_bin(), "delete", "avd", "--name", name],
        timeout=120,
    )
    if proc.returncode != 0 or inspect_avd(name).get("config"):
        detail_text = ((proc.stderr or "") + (proc.stdout or ""))[-1200:]
        raise DeviceError(
            f"failed to delete virtual-target definition {name!r}",
            hint=detail_text or "Check the Android SDK avdmanager installation.",
        )
    return VirtualTargetDeleteResult(
        definition_id=name,
        deleted=True,
        details={"previous": detail},
    )
