"""Built-in iOS simulator platform strategy: ``simctl`` for the device, AXe for the UI.

Select it with ``--platform ios`` (or ``AUA_PLATFORM=ios`` / ``device.platform: ios``). The
agent-facing surface is unchanged: the same ``analyze``, id-based taps, waits, flows and maps
work on an iPhone or iPad simulator. Under the hood the accessibility tree comes from
``axe describe-ui`` and the frame from ``xcrun simctl io screenshot``; both are read-only, so
the dashboard can peek at a simulator another agent is driving.

Physical iPhones are out of scope: AXe drives simulators only.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ConfigError, DeviceError, UsageError
from ..providers.base import ScreenImage
from ..schema import AppContext, Element, TargetInfo, TargetStatus
from . import ios_tree
from .base import AppBundle, DiscoveredTarget, InstalledApp, NormalizedTree, PlatformAdapter
from .diagnostics import AppExitEvidence
from .geometry import DisplayGeometry
from .ios_runtime import KEY_NAMES, IOSSimulatorRuntime, is_known_key
from .ios_tools import (
    SYSTEM_APP_IDS,
    CommandRunner,
    HostCommandRunner,
    IOSTools,
    SimulatorInfo,
    png_size,
    read_app_bundle_info,
)
from .registry import register_platform
from .runtime import TargetRuntime

logger = logging.getLogger(__name__)

_DEFAULT_BOOT_TIMEOUT_S = 120.0
_STATUS = {
    "Booted": TargetStatus.online,
    "Shutdown": TargetStatus.offline,
    "Booting": TargetStatus.booting,
    "Shutting Down": TargetStatus.booting,
}


@register_platform("ios")
class IOSPlatform(PlatformAdapter):
    """iOS simulators through Apple's ``simctl`` and the AXe accessibility CLI."""

    capabilities = frozenset(
        {
            "app.install",
            "app.lifecycle",
            "app.links",
            "app.status",
            "app_database",
            "feature_flags",
            "device.clipboard",
            "device.location",
            "device.touch",
            "ui.input",
            "ui.peek",
            "ui.read_deadline",
            "ui.screenshot",
            "ui.tree",
        }
    )

    def __init__(self, config: Config, runner: CommandRunner | None = None) -> None:
        super().__init__(config)
        self._runner: CommandRunner = runner or HostCommandRunner()
        self._tools: IOSTools | None = None

    # -- configuration ------------------------------------------------------------------------

    def validate_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        known = {"axe_path", "boot_timeout_s"}
        unknown = sorted(str(key) for key in options if key not in known)
        if unknown:
            raise ConfigError(
                f"platform 'ios' does not accept options: {', '.join(unknown)}",
                hint="Known options: axe_path (str), boot_timeout_s (number).",
            )
        normalized: dict[str, Any] = {}
        axe_path = options.get("axe_path")
        if axe_path is not None:
            if not isinstance(axe_path, str) or not axe_path.strip():
                raise ConfigError("platform 'ios' option axe_path must be a non-empty string")
            normalized["axe_path"] = axe_path.strip()
        timeout = options.get("boot_timeout_s")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
                raise ConfigError("platform 'ios' option boot_timeout_s must be a positive number")
            normalized["boot_timeout_s"] = float(timeout)
        return normalized

    @property
    def tools(self) -> IOSTools:
        if self._tools is None:
            self._tools = IOSTools(self._runner, axe_path=self.options.get("axe_path"))
        return self._tools

    def prepare_host(self) -> None:
        self.tools.resolve_xcrun()
        self.tools.resolve_axe()

    def load_capability(self, capability: str) -> object | None:
        if capability == "app_database":
            from .ios_database import IOSDatabase

            return IOSDatabase(self.tools)
        if capability == "feature_flags":
            from .ios_preferences import IOSPreferences

            return IOSPreferences(self.tools)
        return None

    def normalize_key(self, name: str) -> str:
        candidate = super().normalize_key(name)
        if not is_known_key(candidate):
            raise UsageError(
                f"unknown key '{name}'",
                hint="Valid: " + ", ".join(sorted(KEY_NAMES)) + ", or hid:<usage-code>.",
            )
        return candidate

    # -- discovery and connection -----------------------------------------------------------------

    def list_targets(self) -> list[DiscoveredTarget]:
        self.prepare_host()
        return [
            TargetInfo(
                target_id=sim.udid,
                platform=self.name,
                status=_STATUS.get(sim.state, TargetStatus.unknown),
                model=sim.name,
                os_name="ios",
                os_version=sim.os_version,
            )
            for sim in self.tools.list_simulators()
            if sim.is_available
        ]

    def target_preference(self, target: DiscoveredTarget) -> int:
        # Phones first: most apps under test are phone layouts, and an unpinned agent should
        # land on the smaller, faster-to-read screen.
        return 0 if (target.model or "").startswith("iPhone") else 1

    def _select(self, target_id: str | None) -> SimulatorInfo:
        simulators = [sim for sim in self.tools.list_simulators() if sim.is_available]
        if target_id is None:
            booted = [sim for sim in simulators if sim.booted]
            if len(booted) == 1:
                return booted[0]
            if not booted:
                raise DeviceError(
                    "no booted iOS simulator",
                    code="no_target",
                    hint="Boot one (`xcrun simctl boot <udid>`) or pass --serial <udid|name>; `aua devices` lists them.",
                )
            names = ", ".join(f"{sim.name} ({sim.udid})" for sim in booted)
            raise DeviceError(
                f"multiple booted iOS simulators: {names}",
                code="multiple_targets",
                hint="Pass --serial <udid> (or a unique simulator name).",
            )
        wanted = target_id.strip()
        by_udid = [sim for sim in simulators if sim.udid.casefold() == wanted.casefold()]
        if by_udid:
            return by_udid[0]
        by_name = [sim for sim in simulators if sim.name.casefold() == wanted.casefold()]
        booted_by_name = [sim for sim in by_name if sim.booted]
        if len(by_name) == 1 or len(booted_by_name) == 1:
            return (booted_by_name or by_name)[0]
        if by_name:
            udids = ", ".join(f"{sim.udid} (iOS {sim.os_version or '?'})" for sim in by_name)
            raise DeviceError(
                f"{len(by_name)} simulators are named {wanted!r}: {udids}",
                code="multiple_targets",
                hint="Pass the UDID instead of the name.",
            )
        raise DeviceError(
            f"no iOS simulator named or identified {wanted!r}",
            code="ios_target_not_found",
            hint="`aua --platform ios devices` lists UDIDs and names.",
        )

    def _boot(self, sim: SimulatorInfo) -> SimulatorInfo:
        timeout_s = float(self.options.get("boot_timeout_s") or _DEFAULT_BOOT_TIMEOUT_S)
        result = self.tools.simctl("boot", sim.udid, timeout_s=timeout_s, check=False)
        if not result.ok and "already booted" not in result.error_text.casefold():
            raise self.tools.error(result, "simctl boot")
        self.tools.simctl("bootstatus", sim.udid, "-b", timeout_s=timeout_s)
        refreshed = self.tools.find_simulator(sim.udid)
        return refreshed or sim

    def _probe_geometry(self, udid: str) -> DisplayGeometry:
        """Pixels from one screenshot, points from the accessibility root: the scale is theirs."""

        with tempfile.TemporaryDirectory(prefix="aua-ios-") as tmp:
            pixels = png_size(self.tools.screenshot_png(udid, Path(tmp) / "probe.png"))
        if pixels is None:
            raise DeviceError(f"could not picture simulator {udid}", code="screencap_failed")
        deadline = time.monotonic() + 15.0
        points: tuple[float, float] | None = None
        while points is None:
            roots = ios_tree.parse_envelope(
                self.tools.axe("describe-ui", "--udid", udid, timeout_s=30.0).text
            )[0]
            points = ios_tree.root_frame_size(roots)
            if points is None and time.monotonic() >= deadline:
                raise DeviceError(
                    f"simulator {udid} exposes no accessibility root yet",
                    code="ios_target_not_ready",
                    hint="Wait for the home screen to appear, then retry.",
                )
            if points is None:
                time.sleep(0.5)
        # Portrait screenshots of a landscape-reported root (or vice versa) would mean rotation,
        # which this adapter does not model; align the axes and let the scale be uniform.
        if (points[0] > points[1]) != (pixels[0] > pixels[1]):
            points = (points[1], points[0])
        return DisplayGeometry.scaled(native_size=points, canonical_size=pixels)

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        self.prepare_host()
        sim = self._select(target_id)
        if not sim.booted:
            if target_id is None:  # pragma: no cover - _select only returns booted here
                raise DeviceError("no booted iOS simulator", code="no_target")
            logger.info("booting iOS simulator %s (%s)", sim.name, sim.udid)
            sim = self._boot(sim)
        return IOSSimulatorRuntime(
            self.tools,
            sim.udid,
            geometry=self._probe_geometry(sim.udid),
            boot_token=sim.last_booted_at or self.tools.boot_identity(sim.udid),
            data_path=sim.data_path,
        )

    # -- ui.tree / ui.screenshot ------------------------------------------------------------------

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return ios_tree.normalize(
            raw_tree,
            screen_size,
            geometry=geometry or DisplayGeometry.identity(*screen_size),
            ignored_app_ids=ignored_app_ids,
        )

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        return runtime.screenshot()

    # -- ui.peek: read-only, never attaches automation --------------------------------------------

    def peek_foreground_app(self, target_id: str) -> AppContext | None:
        self.prepare_host()
        result = self.tools.axe(
            "describe-ui", "--udid", target_id, "--point", "100,300", timeout_s=10.0, check=False
        )
        if not result.ok:
            return None
        roots, _ = ios_tree.parse_envelope(result.text)
        pid = next((int(root["pid"]) for root in roots if str(root.get("pid", "")).isdigit()), None)
        if pid is None:
            return None
        app_id = self.tools.running_apps(target_id).get(pid)
        return AppContext(app_id=app_id) if app_id else None

    def peek_screenshot(self, target_id: str) -> ScreenImage:
        self.prepare_host()
        with tempfile.TemporaryDirectory(prefix="aua-ios-") as tmp:
            data = self.tools.screenshot_png(target_id, Path(tmp) / "peek.png")
        size = png_size(data)
        if size is None:
            raise DeviceError(f"could not picture {target_id!r}", code="screencap_failed")
        return ScreenImage(data, width=size[0], height=size[1])

    # -- app.status / app.install -------------------------------------------------------------------

    def inspect_app_bundle(self, bundle: Path) -> AppBundle:
        info = read_app_bundle_info(bundle)
        return AppBundle(
            app_id=str(info["CFBundleIdentifier"]),
            version_name=_optional_str(info.get("CFBundleShortVersionString")),
            version_code=_optional_str(info.get("CFBundleVersion")),
        )

    def installed_app(self, runtime: TargetRuntime, app_id: str) -> InstalledApp:
        info = self.tools.app_info(runtime.target_id, app_id)
        installed = bool(info.get("CFBundleVersion") or info.get("Path") or info.get("Bundle"))
        return InstalledApp(
            app_id=app_id,
            installed=installed,
            version_name=_optional_str(info.get("CFBundleShortVersionString"))
            if installed
            else None,
            version_code=_optional_str(info.get("CFBundleVersion")) if installed else None,
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
        del replace  # simctl install always replaces an existing build and keeps its data
        self.prepare_host()
        path = bundle.expanduser()
        info = read_app_bundle_info(path)
        result = self.tools.simctl(
            "install", runtime.target_id, str(path), timeout_s=timeout_s, check=False
        )
        if not result.ok:
            raise DeviceError(
                f"install of {path.name} failed: {result.error_text}",
                code="install_failed",
                hint="Is this an iphonesimulator build matching the simulator's architecture?",
            )
        if grant_permissions:
            self.tools.simctl(
                "privacy",
                runtime.target_id,
                "grant",
                "all",
                str(info["CFBundleIdentifier"]),
                timeout_s=30.0,
            )

    def uninstall_app(self, runtime: TargetRuntime, app_id: str) -> None:
        self.prepare_host()
        result = self.tools.simctl(
            "uninstall", runtime.target_id, app_id, timeout_s=120.0, check=False
        )
        if not result.ok and "not installed" not in result.error_text.casefold():
            raise self.tools.error(result, "simctl uninstall")

    # -- evidence ------------------------------------------------------------------------------------

    def app_exit_evidence(
        self,
        before: AppContext | str | None,
        after: AppContext | str | None,
        elements: Sequence[Element],
    ) -> AppExitEvidence | None:
        before_app = _app_id(before)
        after_app = _app_id(after)
        if not before_app or not after_app or before_app == after_app:
            return None
        if before_app.casefold() in SYSTEM_APP_IDS or after_app.casefold() not in SYSTEM_APP_IDS:
            return None
        return AppExitEvidence(from_app_id=before_app, to_app_id=after_app, crash_dialog=False)

    def doctor_checks(self) -> dict[str, Any]:
        checks: dict[str, Any] = {
            "platform": {"ok": True, "detail": self.name, "capabilities": sorted(self.capabilities)}
        }
        try:
            xcrun = self.tools.resolve_xcrun()
            checks["xcrun"] = {"ok": True, "detail": xcrun}
        except DeviceError as exc:
            checks["xcrun"] = {"ok": False, "detail": exc.message, "hint": exc.hint}
        try:
            axe = self.tools.resolve_axe()
            version = self.tools.axe_version()
            checks["axe"] = {"ok": True, "detail": f"{axe} ({version})" if version else axe}
        except DeviceError as exc:
            checks["axe"] = {"ok": False, "detail": exc.message, "hint": exc.hint}
        if checks["xcrun"]["ok"]:
            try:
                simulators = [sim for sim in self.tools.list_simulators() if sim.is_available]
                booted = [f"{sim.name} ({sim.udid})" for sim in simulators if sim.booted]
                checks["simulators"] = {
                    "ok": bool(booted),
                    "detail": {"available": len(simulators), "booted": booted},
                }
                if not booted:
                    checks["simulators"]["hint"] = (
                        "Boot one: `xcrun simctl boot <udid>` (see `aua --platform ios devices`)."
                    )
            except DeviceError as exc:
                checks["simulators"] = {"ok": False, "detail": exc.message}
        return checks


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _app_id(value: AppContext | str | None) -> str | None:
    if isinstance(value, AppContext):
        return value.app_id
    if not value:
        return None
    return str(value).split("/", 1)[0] or None


__all__ = ["IOSPlatform"]
