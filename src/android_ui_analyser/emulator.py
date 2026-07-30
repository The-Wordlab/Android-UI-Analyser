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
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import DeviceError, UsageError

logger = logging.getLogger(__name__)

_DEFAULT_WAIT_S = 120.0
_POLL_S = 1.0

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
            f"`aua emulator start --avd {name} --headless` then "
            "`aua --serial <serial> proxy start` (system CA needs this rootable image)."
        ),
    }


def _pid_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir).expanduser() / "emulator"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
                    started.append(json.load(fh))
                except json.JSONDecodeError:
                    continue
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
    """Block until ``sys.boot_completed`` is 1. Returns False on timeout."""
    deadline = time.monotonic() + max(5.0, timeout_s)
    while time.monotonic() < deadline:
        if (shell("getprop sys.boot_completed") or "").strip() == "1":
            return True
        time.sleep(_POLL_S)
    return False


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


def start(
    avd: str | None = None,
    *,
    headless: bool = True,
    animations: bool = False,
    wait_s: float = _DEFAULT_WAIT_S,
    cache_dir: str | Path,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Boot an AVD (headless by default) and wait until adb sees ``state=device``.

    If *avd* is omitted and exactly one AVD exists, that one is used. Does not steal
    focus when ``headless=True`` (``-no-window``). Creating AVDs is ``ensure_proxy_avd``.
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

            shell = _serial_shell(serial)
            # Wait for sys.boot_completed, not just adb `state=device`. Those are ~15s apart,
            # and a `settings put` made in between is undone as the system finishes booting —
            # which looked like success while the scales stayed at 1.0.
            _wait_for_boot(shell, timeout_s=min(90.0, wait_s))
            state = devopts.anim_off(shell, _pid_dir(cache_dir) / f"{avd}.anim.json")
            # Read back rather than assume: claiming this without checking is the same
            # false-success the wait above was hiding.
            anim = (state or {}).get("anim") or {}
            animations_disabled = bool(anim) and all(
                str(v) in ("0", "0.0") for v in anim.values()
            )
    return {
        "ok": True,
        "action": "emulator-start",
        "avd": avd,
        "serial": serial,
        "pid": proc.pid,
        "headless": headless,
        "animations_disabled": animations_disabled,
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

    stopped: list[str] = []
    for t in targets:
        ser = t["serial"]
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
        if meta.get("serial") in stopped or (avd and path.stem == avd):
            path.unlink(missing_ok=True)

    return {"ok": True, "action": "emulator-stop", "stopped": stopped}
