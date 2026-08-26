"""Built-in Android platform strategy."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .. import hierarchy
from ..config import Config
from ..device import Device
from ..errors import DeviceError
from ..memory import matches_any
from ..providers.base import ScreenImage
from ..schema import DeviceInfo, Element
from ..scroll_geom import _iter_nodes, _node_box
from . import android_apk, android_transport
from .base import AppBundle, InstalledApp, NormalizedTree, PlatformAdapter
from .registry import register_platform

logger = logging.getLogger(__name__)

_PACKAGE_RE = re.compile(r'package="([^"]+)"')
_CAPS_TTL_S = 3600
# Long enough that a burst of actions pays one `pidof`, short enough that a process replaced
# behind AUA's back costs at most this many seconds of empty windows.
_PID_CACHE_TTL_S = 30.0


def _runtime_emulator_capabilities(target_id: str) -> dict[str, bool]:
    """Visibility/audio facts for an emulator process, conservative when attribution is unclear."""

    if not target_id.startswith("emulator-"):
        # These requirements describe host-controlled virtual-device facilities. A USB phone's
        # own display must never satisfy ``--headed``: that flag promises the caller an emulator
        # window, and treating the handset as equivalent can fresh-install a QA build on a real
        # device before the caller has any visible window to stop it.
        return {"emulator": False, "headed": False, "audio": False}
    try:
        port = int(target_id.rsplit("-", 1)[1])
        result = subprocess.run(  # noqa: S603
            ["ps", "-axo", "command="],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return {"emulator": True, "headed": False, "audio": False}
    port_pattern = re.compile(rf"(?:^|\s)-port\s+{port}(?:\s|$)")
    command = next(
        (
            line
            for line in (result.stdout or "").splitlines()
            if port_pattern.search(line) and ("emulator" in line or "qemu" in line)
        ),
        "",
    )
    if not command:
        # A default-port emulator may not disclose ``-port``. Unknown is not proof that it can
        # satisfy a requested visible/audio session, so AUA will provision a known-good instance.
        return {"emulator": True, "headed": False, "audio": False}
    return {
        "emulator": True,
        "headed": "-no-window" not in command,
        "audio": "-no-audio" not in command,
    }


def _android_shell(target_id: str, command: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            ["adb", "-s", target_id, "shell", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return (result.stdout or "").strip()
    except Exception:
        return ""


def probe_android_capabilities(cache_dir: str | Path, target_id: str) -> dict[str, object]:
    """Android runtime facts used by generic lease requirements."""

    path = Path(cache_dir).expanduser() / "caps" / f"{target_id.replace(':', '_')}.json"
    now = time.time()
    with contextlib.suppress(Exception):
        cached = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and (now - float(cached.get("probed") or 0)) < _CAPS_TTL_S:
            return {**cached, **_runtime_emulator_capabilities(target_id)}

    tags = _android_shell(target_id, "getprop ro.build.tags")
    debuggable = _android_shell(target_id, "getprop ro.debuggable")
    vending = _android_shell(target_id, "pm list packages com.android.vending")
    rootable = "test-keys" in tags or debuggable.strip() == "1"
    capabilities: dict[str, object] = {
        "serial": target_id,
        "root": rootable,
        "play": "com.android.vending" in vending,
        "proxy": rootable,
        "probed": now,
        **_runtime_emulator_capabilities(target_id),
    }
    with contextlib.suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(capabilities, indent=2) + "\n", encoding="utf-8")
    return capabilities


def package_from_tree(
    raw_tree: str, ignore: Sequence[str] = ("com.android.systemui",)
) -> str | None:
    """Guess Android's foreground package from its hierarchy XML."""

    packages = _PACKAGE_RE.findall(raw_tree)
    if not packages:
        return None
    counts = Counter(
        package for package in packages if package and not matches_any(package, ignore)
    )
    if not counts:
        counts = Counter(packages)
    return counts.most_common(1)[0][0]


@register_platform("android")
class AndroidPlatform(PlatformAdapter):
    """Android implementation backed by adb and uiautomator2."""

    capabilities = frozenset(
        {
            "app.files",
            "app.install",
            "app.lifecycle",
            "app.status",
            "device.logs",
            "device.network",
            "device.proxy",
            "device.shell",
            "emulator",
            "app_database",
            "device_agent",
            "developer_settings",
            "feature_flags",
            "microphone",
            "network",
            "network_profiles",
            "proxy",
            "virtual_devices",
            "webview",
            "ui.input",
            "ui.screenshot",
            "ui.tree",
        }
    )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # (serial, app_id) -> (pid or None, monotonic stamp). See `_app_pid`.
        self._pid_cache: dict[tuple[str, str], tuple[str | None, float]] = {}

    def prepare_host(self) -> None:
        from ..emulator import ensure_adb_on_path

        ensure_adb_on_path()
        android_transport.ensure_adb_server_ready(self.config.lease.registry_dir)

    def connect(self, target_id: str | None = None) -> Device:
        from .. import device as device_mod
        from .android_runtime import AndroidDeviceRuntime

        self.prepare_host()
        return AndroidDeviceRuntime(device_mod.resolve_serial(target_id))

    def list_targets(self) -> list[DeviceInfo]:
        from .. import device as device_mod

        self.prepare_host()
        try:
            return device_mod.list_devices()
        except DeviceError:
            # The server can disappear after the readiness probe (SDK upgrades and external
            # tools sometimes replace it). Re-enter the same coordinated bootstrap and retry
            # enumeration once; never turn a transport failure into an empty target pool.
            android_transport.ensure_adb_server_ready(self.config.lease.registry_dir)
            return device_mod.list_devices()

    def target_preference(self, target: DeviceInfo) -> int:
        # Prefer a disposable emulator over a physical USB phone when the user did not pin one.
        return 0 if target.serial.startswith("emulator-") else 1

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        self.prepare_host()
        count = max(1, min(int(limit), 500))
        pid_args: list[str] = []
        if app_id:
            pid_result = subprocess.run(  # noqa: S603
                ["adb", "-s", target_id, "shell", "pidof", app_id],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            pids = [item for item in (pid_result.stdout or "").split() if item.isdigit()]
            if not pids:
                return []
            # Android logcat's --pid lane is exact and substantially cheaper than downloading
            # the global buffer and hoping the package name appears in each message.
            pid_args = ["-v", "threadtime", "--pid", pids[0]]
        result = subprocess.run(  # noqa: S603
            ["adb", "-s", target_id, "logcat", "-d", *pid_args, "-t", str(count)],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return [line for line in (result.stdout or "").splitlines() if line.strip()][-count:]

    def probe_target_capabilities(self, target_id: str) -> dict[str, object]:
        return probe_android_capabilities(self.config.cache.dir, target_id)

    _CAPABILITY_MODULES = {
        "app_database": "app_database",
        "device_agent": "device_agent",
        "developer_settings": "devopts",
        "feature_flags": "flags",
        "microphone": "mic",
        "network": "network",
        "network_profiles": "network_profiles",
        "proxy": "proxy_mock",
        "virtual_devices": "emulator",
        "webview": "webview",
    }

    def load_capability(self, capability: str) -> object | None:
        module = self._CAPABILITY_MODULES.get(capability)
        if module is None:
            return None
        self.prepare_host()
        return importlib.import_module(f"android_ui_analyser.{module}")

    def doctor_checks(self) -> dict[str, object]:
        self.prepare_host()
        adb = shutil.which("adb")
        try:
            u2 = importlib.util.find_spec("uiautomator2")
            u2_check: dict[str, object] = {
                "ok": u2 is not None,
                "detail": "importable" if u2 is not None else "not installed",
            }
        except Exception as exc:  # pragma: no cover - defensive
            u2_check = {"ok": False, "detail": f"error: {exc}"}

        emulator = self.capability("virtual_devices")
        try:
            status = emulator.status(cache_dir=self.config.cache.dir)
            emulator_check: dict[str, object] = {
                "ok": bool(status.get("emulator_ok")),
                "detail": {
                    "binary": status.get("emulator"),
                    "avds": status.get("avds") or [],
                    "rootable": status.get("rootable") or [],
                    "play_store": status.get("play_store") or [],
                    "running": status.get("running") or [],
                },
            }
            if status.get("hint"):
                emulator_check["hint"] = status["hint"]
            elif (status.get("play_store") or []) and not (status.get("rootable") or []):
                emulator_check["hint"] = (
                    "Only Google Play AVDs — HTTPS proxy needs a rootable image: "
                    "`aua emulator ensure-proxy`."
                )
        except Exception as exc:  # pragma: no cover - defensive
            emulator_check = {"ok": False, "detail": str(exc)}

        return {
            "platform": {
                "ok": True,
                "detail": self.name,
                "capabilities": sorted(self.capabilities),
            },
            "adb": {"ok": adb is not None, "detail": adb or "adb not found on PATH"},
            "uiautomator2": u2_check,
            "emulator": emulator_check,
        }

    def diagnostic_logs(
        self,
        runtime: Device,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        pid: str | None = None
        if app_id:
            pid = self._app_pid(runtime, app_id)
            if pid is None:
                # The process is gone. That is a fact about the app, not a reason to hand back
                # the whole device buffer and let it read as the app's own output.
                return ""
        raw = runtime.logcat(since_ms=since_ms, dump=True, pid=pid)
        return "\n".join(raw.splitlines()[-max(1, lines) :])

    def _app_pid(self, runtime: Device, app_id: str) -> str | None:
        """First pid of *app_id*, or None when it is not running.

        ``logcat --pid`` is exact and cheaper than downloading the global buffer: measured on an
        emulator, the scoped dump ran in 46 ms against 48 ms unscoped, while resolving the pid
        costs a further 34 ms. That second round trip is the one worth removing, because this
        runs after every action rather than once: hence the short-lived cache, which turns a
        burst of actions into one lookup.

        The TTL is what keeps a stale pid harmless. If the app's process is replaced without AUA
        acting, a stale pid selects nothing and the window reads as "the app logged nothing" —
        wrong, but self-healing within seconds, and never another app's output attributed to
        this one.
        """
        from shlex import quote

        key = (runtime.serial, app_id)
        cached = self._pid_cache.get(key)
        now = time.monotonic()
        if cached is not None and now - cached[1] < _PID_CACHE_TTL_S:
            return cached[0]
        try:
            out = runtime.shell(f"pidof {quote(app_id)}")
        except Exception as exc:  # noqa: BLE001 — a missing process is normal, not an error
            logger.debug("pidof %s failed: %s", app_id, exc)
            return None
        pids = [item for item in (out or "").split() if item.isdigit()]
        pid = pids[0] if pids else None
        self._pid_cache[key] = (pid, now)
        return pid

    def forget_app_process(self, app_id: str | None = None) -> None:
        """Drop cached pids after a lifecycle event that replaces the app's process.

        Called by the core when it launches, restarts, clears, or reinstalls an app. Without
        this the next action after a relaunch would read the *old* process's window, which is
        empty — and an empty window is exactly the answer a caller cannot tell from the truth.
        """
        if app_id is None:
            self._pid_cache.clear()
            return
        for key in [k for k in self._pid_cache if k[1] == app_id]:
            self._pid_cache.pop(key, None)

    def capture_screenshot(self, runtime: Device) -> ScreenImage:
        return runtime.screenshot()

    def inspect_app_bundle(self, bundle: Path) -> AppBundle:
        info = android_apk.inspect_bundle(bundle)
        return AppBundle(
            app_id=info.package,
            version_name=info.version_name,
            version_code=info.version_code,
        )

    def installed_app(self, runtime: Device, app_id: str) -> InstalledApp:
        state = android_apk.installed_app(runtime.serial, app_id)
        return InstalledApp(
            app_id=state.package,
            installed=state.installed,
            version_name=state.version_name,
            version_code=state.version_code,
        )

    def install_app_bundle(
        self,
        runtime: Device,
        bundle: Path,
        *,
        replace: bool = True,
        grant_permissions: bool = False,
        timeout_s: float = 300.0,
    ) -> None:
        self.prepare_host()
        android_apk.install_bundle(
            runtime.serial,
            bundle,
            reinstall=replace,
            grant_permissions=grant_permissions,
            timeout_s=timeout_s,
        )

    def uninstall_app(self, runtime: Device, app_id: str) -> None:
        self.prepare_host()
        android_apk.uninstall(runtime.serial, app_id)

    def install_persistence_warning(self, runtime: Device) -> str | None:
        # A `-read-only` emulator (what `--parallel` implies) puts disk writes in an overlay it
        # throws away on stop. The install genuinely works for the life of this instance, so
        # refusing would break the ordinary parallel-agent run — but saying nothing is worse: a
        # caller who boots read-only, installs, and expects the build to still be there next
        # session gets `Success` now and "not installed" later, with nothing explaining the gap.
        from ..emulator import discards_writes

        if not discards_writes(runtime.serial, cache_dir=self.config.cache.dir):
            return None
        return (
            f"{runtime.serial} was booted -read-only (--parallel implies it), so this install "
            "lives only until the emulator stops. Boot with `--parallel --no-read-only` if the "
            "build must survive a restart."
        )

    def element_state(self, raw_tree: str, element: Element) -> dict[str, object]:
        state = super().element_state(raw_tree, element)
        node = next(
            (node for node in _iter_nodes(raw_tree) if _node_box(node) == tuple(element.bounds)),
            None,
        )
        if node is not None:
            holder = node
            if node.get("checkable") != "true":
                holder = next(
                    (child for child in node.iter("node") if child.get("checkable") == "true"),
                    node,
                )
            state.update(
                checkable=holder.get("checkable") == "true",
                checked=holder.get("checked") == "true",
                enabled=node.get("enabled") == "true",
                selected=node.get("selected") == "true",
                focused=node.get("focused") == "true",
            )
        if element.checkable:
            state["checkable"] = True
            state["checked"] = bool(element.checked)
        if element.selected is not None:
            state["selected"] = element.selected
        return state

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree(
            elements=hierarchy.parse_hierarchy(raw_tree, screen_size),
            app_id=package_from_tree(raw_tree, ignored_app_ids),
        )
