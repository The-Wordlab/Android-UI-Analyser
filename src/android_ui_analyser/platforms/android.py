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
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .. import hierarchy
from ..config import Config
from ..errors import UsageError
from ..memory import matches_any
from ..providers.base import ScreenImage
from ..schema import AppContext, Element
from . import android_apk, android_transport
from .base import AppBundle, DiscoveredTarget, InstalledApp, NormalizedTree, PlatformAdapter
from .diagnostics import AppExitEvidence, DiagnosticSourcePolicy, DiagnosticWindow
from .geometry import DisplayGeometry
from .identity import TargetRef
from .registry import register_platform
from .runtime import TargetRuntime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .android_device import AndroidRuntimeBase

logger = logging.getLogger(__name__)

_PACKAGE_RE = re.compile(r'package="([^"]+)"')
_CAPS_TTL_S = 3600
# Long enough that a burst of actions pays one `pidof`, short enough that a process replaced
# behind AUA's back costs at most this many seconds of empty windows.
_PID_CACHE_TTL_S = 30.0
_ANDROID_KEY_NAMES = frozenset(
    {
        "home",
        "back",
        "left",
        "right",
        "up",
        "down",
        "center",
        "menu",
        "search",
        "enter",
        "delete",
        "del",
        "recent",
        "recents",
        "volume_up",
        "volume_down",
        "volume_mute",
        "camera",
        "power",
    }
)
_XML_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _node_box(node: ET.Element) -> tuple[int, int, int, int] | None:
    match = _XML_BOUNDS_RE.search(node.get("bounds") or "")
    if not match:
        return None
    x1, y1, x2, y2 = (int(group) for group in match.groups())
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _iter_nodes(raw_tree: str) -> list[ET.Element]:
    """Return Android UIAutomator nodes, or an empty list for a malformed dump."""

    if not raw_tree or not raw_tree.strip():
        return []
    try:
        return list(ET.fromstring(raw_tree).iter("node"))
    except ET.ParseError as exc:  # pragma: no cover - malformed device response
        logger.warning("could not parse Android hierarchy dump: %s", exc)
        return []


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
            "app.links",
            "app.status",
            "device.accessibility",
            "device.airplane",
            "device.clipboard",
            "device.clock",
            "device.keyboard",
            "device.location",
            "device.logs",
            "device.media",
            "device.orientation",
            "device.proxy",
            "device.recording",
            "device.shell",
            "device.touch",
            "app_database",
            "device_agent",
            "developer_settings",
            "feature_flags",
            "microphone",
            "network",
            "network_profiles",
            "proxy",
            "target_supervision",
            "virtual_targets",
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

    def normalize_key(self, name: str) -> str:
        """Keep Android's historical keycode aliases inside the Android strategy."""

        candidate = super().normalize_key(name)
        known = (
            candidate.casefold() in _ANDROID_KEY_NAMES
            or candidate.upper().startswith("KEYCODE_")
            or candidate.isdigit()
        )
        if not known:
            raise UsageError(
                f"unknown key '{name}'",
                hint=(
                    "Valid: "
                    + ", ".join(sorted(_ANDROID_KEY_NAMES))
                    + ", KEYCODE_*, or a keycode number."
                ),
            )
        return candidate

    def prepare_host(self) -> None:
        from ..emulator import ensure_adb_on_path

        ensure_adb_on_path()
        android_transport.ensure_adb_server_ready(self.config.lease.registry_dir)

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        from .. import device as device_mod
        from .android_runtime import AndroidDeviceRuntime

        self.prepare_host()
        return android_transport.run_adb_server_operation(
            self.config.lease.registry_dir,
            lambda: AndroidDeviceRuntime(device_mod.resolve_serial(target_id)),
        )

    def list_targets(self) -> list[DiscoveredTarget]:
        from .. import device as device_mod

        self.prepare_host()
        devices = android_transport.run_adb_inventory_operation(
            self.config.lease.registry_dir,
            device_mod.list_devices,
        )
        return list(devices)

    def target_preference(self, target: DiscoveredTarget) -> int:
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
        "target_supervision": "platforms.android_supervision",
        "virtual_targets": "emulator",
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

        emulator = self.capability("virtual_targets")
        try:
            target_status = emulator.virtual_target_status(cache_dir=self.config.cache.dir)
            # Android's doctor response predates the neutral virtual-target schema. The typed
            # service result deliberately retains this adapter-owned payload so this Android-only
            # renderer can preserve its public fields without bypassing the service contract.
            status = dict(target_status.legacy_result or target_status.to_dict())
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
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        """Compatibility text rendering over the normalized Android diagnostic window."""

        return self.diagnostic_window(
            runtime,
            lines=lines,
            # The compatibility API historically meant "the latest N" when no boundary was
            # supplied; zero keeps that unbounded-before-tail behavior.
            since=since_ms if since_ms is not None else 0,
            app_id=app_id,
        ).text

    def diagnostic_window(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        """Parse Android logcat into platform-neutral diagnostic evidence."""

        from .. import logcat as logcat_mod

        target = TargetRef(platform=self.name, target_id=runtime.target_id)
        clock = logcat_mod.resolve_clock(
            runtime,
            self.config.cache.dir,
            target=target,
        )
        marks = logcat_mod.load_marks(logcat_mod.marks_path(self.config.cache.dir, target))
        since_ms, since_label = logcat_mod.resolve_since_ms(marks, since, clock=clock)
        pid: str | None = None
        if app_id:
            pid = self._app_pid(runtime, app_id)
        android_runtime = cast("AndroidRuntimeBase", runtime)
        raw = android_runtime.logcat(since_ms=since_ms, dump=True, pid=pid)
        bounded = "\n".join(raw.splitlines()[-max(1, int(lines)) :])
        try:
            timezone_offset = android_runtime.utc_offset_minutes() or 0
        except Exception:  # noqa: BLE001 - timestamps remain useful without a known zone
            timezone_offset = 0
        return logcat_mod.parse_diagnostic_window(
            bounded,
            target=target,
            since=since_label,
            since_unix_ms=since_ms,
            clock=clock.name,
            skew_ms=clock.skew_ms,
            app_id=app_id,
            # With no live pid, an app-scoped compatibility dump must stay empty. The global
            # scan is retained only as normalized crash evidence for the just-dead process.
            include_events=app_id is None or pid is not None,
            tz_offset_minutes=timezone_offset,
        )

    def mark_diagnostics(
        self,
        runtime: TargetRuntime,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, object]:
        """Persist an Android device-clock cursor under its platform-scoped target identity."""

        from .. import logcat as logcat_mod

        if clear:
            self.clear_diagnostics(runtime)
        target = TargetRef(platform=self.name, target_id=runtime.target_id)
        clock = logcat_mod.resolve_clock(
            runtime,
            self.config.cache.dir,
            target=target,
            force=refresh_clock,
        )
        return logcat_mod.set_mark(
            self.config.cache.dir,
            target,
            name or "default",
            clock=clock,
        )

    def clear_diagnostics(self, runtime: TargetRuntime) -> None:
        android_runtime = cast("AndroidRuntimeBase", runtime)
        android_runtime.logcat(dump=False)

    def diagnostic_source_policy(
        self, app_id: str | None = None
    ) -> DiagnosticSourcePolicy:
        from ..logcat import android_source_policy

        return android_source_policy(app_id)

    def app_exit_evidence(
        self,
        before: AppContext | str | None,
        after: AppContext | str | None,
        elements: Sequence[Element],
    ) -> AppExitEvidence | None:
        """Interpret Android launcher and app-error surfaces behind the adapter boundary."""

        def app_id(value: AppContext | str | None) -> str | None:
            if isinstance(value, AppContext):
                return value.app_id
            if not value or "/" not in value:
                return None
            return value.split("/", 1)[0] or None

        before_app = app_id(before)
        after_app = app_id(after)
        if not before_app or not after_app or before_app == after_app:
            return None
        crash_dialog = any("aerr_" in str(element.resource_id or "") for element in elements)
        to_launcher = any(hint in after_app.casefold() for hint in ("launcher", "home"))
        if not crash_dialog and not to_launcher:
            return None
        return AppExitEvidence(
            from_app_id=before_app,
            to_app_id=after_app,
            crash_dialog=crash_dialog,
        )

    def link_chooser_visible(self, runtime: TargetRuntime) -> bool:
        """Normalize Android ResolverActivity/SystemUI evidence for the shared link flow."""

        try:
            context = AppContext.coerce(runtime.current_app())
        except Exception:
            return False
        app_id = (context.app_id or "").casefold()
        surface_id = (context.surface_id or "").casefold()
        if (
            "resolver" in surface_id
            or "intentresolver" in app_id
            or app_id in {"android", "com.android.intentresolver", "com.android.internal.app"}
        ):
            return True
        with contextlib.suppress(Exception):
            raw_tree = self.dump_tree(runtime)
            if "Open with" in raw_tree or ("Just once" in raw_tree and "Always" in raw_tree):
                return True
        return False

    def link_chooser_candidates(
        self,
        elements: Sequence[Element],
        *,
        preferred_app_id: str | None = None,
    ) -> list[Element]:
        chrome = {"Just once", "Always", "Open with", "Cancel", "Open"}
        candidates = [
            element
            for element in elements
            if element.clickable
            and (element.text or element.content_desc or "").strip()
            and (element.text or element.content_desc or "").strip() not in chrome
        ]
        preferred = (preferred_app_id or "").casefold()
        tail = preferred.rsplit(".", 1)[-1] if preferred else ""
        if not preferred:
            return candidates
        matching = [
            element
            for element in candidates
            if preferred in f"{element.text or ''} {element.content_desc or ''}".casefold()
            or tail in f"{element.text or ''} {element.content_desc or ''}".casefold()
        ]
        return [*matching, *(element for element in candidates if element not in matching)]

    def link_chooser_confirmation(self, elements: Sequence[Element]) -> Element | None:
        return next(
            (
                element
                for element in elements
                if element.clickable and (element.text or "").strip() == "Just once"
            ),
            None,
        )

    def _app_pid(self, runtime: TargetRuntime, app_id: str) -> str | None:
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
            android_runtime = cast("AndroidRuntimeBase", runtime)
            out = android_runtime.shell(f"pidof {quote(app_id)}")
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

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        if self.config.perf.capture_adb_screencap:
            return runtime.screencap_png()
        return runtime.screenshot()

    def inspect_app_bundle(self, bundle: Path) -> AppBundle:
        info = android_apk.inspect_bundle(bundle)
        return AppBundle(
            app_id=info.package,
            version_name=info.version_name,
            version_code=info.version_code,
        )

    def installed_app(self, runtime: TargetRuntime, app_id: str) -> InstalledApp:
        state = android_apk.installed_app(runtime.serial, app_id)
        return InstalledApp(
            app_id=state.package,
            installed=state.installed,
            version_name=state.version_name,
            version_code=state.version_code,
        )

    def install_app_bundle(
        self,
        runtime: TargetRuntime,
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

    def uninstall_app(self, runtime: TargetRuntime, app_id: str) -> None:
        self.prepare_host()
        android_apk.uninstall(runtime.serial, app_id)

    def install_persistence_warning(self, runtime: TargetRuntime) -> str | None:
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
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree(
            elements=hierarchy.parse_hierarchy(raw_tree, screen_size),
            app_id=package_from_tree(raw_tree, ignored_app_ids),
        )
