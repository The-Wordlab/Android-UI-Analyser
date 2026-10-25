"""Read-only screenshots of physical iOS devices through Xcode's CoreDevice CLI.

This adapter deliberately claims only ui.screenshot. AXe's simulator hierarchy and
input operations do not work on physical phones. No app is launched or navigated.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import DeviceError, UnsupportedPlatformCapabilityError
from ..providers.base import ScreenImage
from ..schema import TargetInfo, TargetStatus
from .base import DiscoveredTarget, NormalizedTree, PlatformAdapter
from .geometry import DisplayGeometry
from .ios_tools import CommandRunner, HostCommandRunner, png_size
from .registry import register_platform
from .runtime import TargetRuntime

_SCREENSHOT_FEATURE = "com.apple.coredevice.feature.capturescreenshot"
_SETUP_HINT = (
    "Connect and unlock the iPhone, trust this Mac, and enable Developer Mode. "
    "Use an Xcode version that supports `devicectl device capture screenshot`."
)


def _property(device: dict[str, Any], section: str, legacy: str) -> dict[str, Any]:
    """Xcode 27's properties envelope, with older CoreDevice JSON compatibility."""
    return device.get("properties", {}).get(section) or device.get(legacy) or {}


def _physical_ios(device: dict[str, Any]) -> bool:
    hardware = _property(device, "hardware", "hardwareProperties")
    return hardware.get("reality") == "physical" and hardware.get("platform") in {"iOS", "iPadOS"}


def _target_info(device: dict[str, Any]) -> TargetInfo:
    hardware = _property(device, "hardware", "hardwareProperties")
    state = _property(device, "state", "deviceProperties")
    software = _property(device, "software", "deviceProperties")
    connection = _property(device, "connection", "connectionProperties")
    version = software.get("osVersionNumber")
    if isinstance(version, dict):
        version = version.get("stringValue")
    connected = connection.get("state", connection.get("tunnelState")) == "connected"
    paired = connection.get("pairingState") == "paired"
    return TargetInfo(
        target_id=str(hardware.get("udid") or device["identifier"]),
        platform="ios-device",
        model=state.get("name") or hardware.get("marketingName"),
        os_name="ios",
        os_version=str(version) if version else None,
        status=TargetStatus.online if connected and paired else TargetStatus.offline,
    )


class IOSDeviceTools:
    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def xcrun(self) -> str:
        executable = shutil.which("xcrun")
        if not executable:
            raise DeviceError(
                "Xcode's xcrun was not found", code="ios_tool_missing", hint=_SETUP_HINT
            )
        return executable

    def command(self, *args: str, timeout_s: float = 30) -> dict[str, Any]:
        # Machine JSON goes to a private file: stdout also contains progress messages.
        with tempfile.TemporaryDirectory(prefix="aua-ios-device-") as directory:
            output = Path(directory) / "result.json"
            result = self.runner.run(
                [
                    self.xcrun(),
                    "devicectl",
                    *args,
                    "--timeout",
                    str(int(timeout_s)),
                    "--json-output",
                    str(output),
                ],
                timeout_s=timeout_s + 5,
            )
            payload: dict[str, Any] = {}
            if output.is_file():
                try:
                    payload = json.loads(output.read_text())
                    if not isinstance(payload, dict):
                        raise ValueError("expected an object")
                except (ValueError, OSError) as exc:
                    raise DeviceError(
                        "Invalid CoreDevice JSON response", code="ios_tool_failed"
                    ) from exc
            if not result.ok or payload.get("info", {}).get("outcome") == "failed":
                detail = (
                    json.dumps(payload.get("error", {}))
                    if payload.get("error")
                    else result.error_text
                )
                folded = detail.casefold()
                code = "ios_tool_failed"
                if "locked" in folded or "passcode" in folded:
                    code = "ios_device_locked"
                elif (
                    "not supported" in folded
                    or "unrecognized" in folded
                    or "unknown subcommand" in folded
                ):
                    code = "ios_screenshot_unsupported"
                elif any(
                    word in folded
                    for word in (
                        "not connected",
                        "disconnected",
                        "not found",
                        "not available",
                        "unavailable",
                    )
                ):
                    code = "ios_device_unavailable"
                raise DeviceError(
                    f"CoreDevice command failed: {detail}", code=code, hint=_SETUP_HINT
                )
            if not payload or not isinstance(payload.get("result"), dict):
                raise DeviceError("CoreDevice returned no result", code="ios_tool_failed")
            return payload["result"]

    def devices(self) -> list[dict[str, Any]]:
        return [
            device
            for device in self.command("list", "devices").get("devices", [])
            if _physical_ios(device)
        ]

    def details(self, target: str, *, timeout_s: float = 30) -> dict[str, Any]:
        device = self.command("device", "info", "details", "--device", target, timeout_s=timeout_s)
        if not _physical_ios(device):
            raise DeviceError(
                "Target is not a physical iOS device",
                code="ios_target_not_found",
                hint="Use --platform ios for an iOS simulator.",
            )
        return device

    def screenshot(self, target: str) -> ScreenImage:
        with tempfile.TemporaryDirectory(prefix="aua-ios-device-") as directory:
            path = Path(directory) / "screen.png"
            self.command(
                "device", "capture", "screenshot", "--device", target, "--destination", str(path)
            )
            if not path.is_file():
                raise DeviceError("CoreDevice did not write a screenshot", code="screencap_failed")
            data = path.read_bytes()
        size = png_size(data)
        if size is None:
            raise DeviceError("CoreDevice returned an invalid PNG", code="screencap_failed")
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError) as exc:
            raise DeviceError("CoreDevice returned an invalid PNG", code="screencap_failed") from exc
        return ScreenImage(data, width=size[0], height=size[1])


class IOSDeviceRuntime(TargetRuntime):
    def __init__(self, tools: IOSDeviceTools, target_id: str) -> None:
        self._tools = tools
        self.target_id = target_id

    def screenshot(self) -> ScreenImage:
        return self._tools.screenshot(self.target_id)


@register_platform("ios-device")
class IOSDevicePlatform(PlatformAdapter):
    """Physical iPhone/iPad capture, without simulator automation or installed helpers."""

    capabilities = frozenset({"ui.screenshot"})

    def __init__(self, config: Config, runner: CommandRunner | None = None) -> None:
        super().__init__(config)
        self.tools = IOSDeviceTools(runner or HostCommandRunner())

    def list_targets(self) -> list[DiscoveredTarget]:
        devices = self.tools.devices()
        requested = self.config.device.serial
        if requested:
            matches = [
                device
                for device in devices
                if requested.casefold()
                in {
                    str(device.get("identifier", "")).casefold(),
                    _target_info(device).target_id.casefold(),
                    str(_target_info(device).model or "").casefold(),
                }
            ]
            if len(matches) > 1:
                raise DeviceError(
                    "Multiple physical iOS devices match this name",
                    code="multiple_targets",
                    hint="Pass the phone's UDID with --serial.",
                )
        else:
            matches = devices
        targets: list[DiscoveredTarget] = []
        for device in devices:
            info = _target_info(device)
            connection = _property(device, "connection", "connectionProperties")
            if (
                device in matches
                and info.status != TargetStatus.online
                and connection.get("pairingState") == "paired"
            ):
                # A CoreDevice tunnel becomes dormant between commands. Probe reachability
                # using bounded read-only metadata, including on ordinary unpinned calls.
                # Cached inventory alone must not report a remembered, unplugged phone online.
                try:
                    info = _target_info(self.tools.details(info.target_id, timeout_s=5))
                except DeviceError as exc:
                    if exc.code not in {"ios_device_unavailable", "ios_tool_timeout"}:
                        raise
            targets.append(info)
        return targets

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        if target_id is None:
            available = [item for item in self.list_targets() if item.status == TargetStatus.online]
            if len(available) != 1:
                raise DeviceError(
                    "Select one connected physical iOS device",
                    code="multiple_targets" if available else "no_target",
                    hint="Pass --serial <UDID>; `aua --platform ios-device devices` lists phones.",
                )
            target_id = available[0].target_id
        device = self.tools.details(target_id)
        info = _target_info(device)
        if info.status != TargetStatus.online:
            raise DeviceError(
                "Physical iOS device is not connected and paired",
                code="ios_device_unavailable",
                hint=_SETUP_HINT,
            )
        if _SCREENSHOT_FEATURE not in {
            item.get("featureIdentifier") for item in device.get("capabilities", [])
        }:
            raise DeviceError(
                "This device/Xcode combination does not expose screenshot capture",
                code="ios_screenshot_unsupported",
                hint=_SETUP_HINT,
            )
        return IOSDeviceRuntime(self.tools, info.target_id)

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        return runtime.screenshot()

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        raise UnsupportedPlatformCapabilityError(self.name, "ui.tree")

    def doctor_checks(self) -> dict[str, Any]:
        checks: dict[str, Any] = {
            "platform": {"ok": True, "detail": self.name, "capabilities": sorted(self.capabilities)}
        }
        try:
            checks["xcrun"] = {"ok": True, "detail": self.tools.xcrun()}
            targets = self.list_targets()
            checks["physical_devices"] = {
                "ok": any(t.status == TargetStatus.online for t in targets),
                "detail": [t.model_dump(mode="json") for t in targets],
            }
        except DeviceError as exc:
            checks["physical_devices"] = {"ok": False, "detail": exc.message, "hint": exc.hint}
        return checks
