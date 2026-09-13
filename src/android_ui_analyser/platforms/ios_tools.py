"""Host-side command plumbing for the built-in iOS simulator platform.

Every native call the iOS adapter makes goes through a :class:`CommandRunner`, so tests can
substitute a fake and never spawn ``xcrun`` or ``axe``. Two tools split the work:

* Apple's ``xcrun simctl`` owns simulator inventory, boot state, app lifecycle, screenshots,
  clipboard, links, permissions and location.
* AXe (``axe``, https://github.com/cameroncooke/AXe) reads the accessibility tree of a booted
  simulator and drives HID input (taps, swipes, keys, hardware buttons).

Nothing here is imported by the generic engine; it is reached only through ``IOSPlatform``.
"""

from __future__ import annotations

import json
import plistlib
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .. import read_budget
from ..errors import DeviceError, UsageError

AXE_INSTALL_HINT = (
    "Install AXe: `brew tap cameroncooke/axe && brew install axe` (macOS with Xcode 26 or newer)."
)
XCRUN_HINT = "Install Xcode with an iOS simulator runtime so `xcrun simctl` is available."
SPRINGBOARD_APP_ID = "com.apple.springboard"
# Apps that own the screen between third-party apps: the home screen itself and its search.
SYSTEM_APP_IDS = frozenset({SPRINGBOARD_APP_ID, "com.apple.spotlight"})


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def text(self) -> str:
        return self.stdout.decode("utf-8", "replace")

    @property
    def error_text(self) -> str:
        """The most useful one-line explanation a failed tool left behind."""

        detail = self.stderr.decode("utf-8", "replace").strip() or self.text.strip()
        lines = [line.strip() for line in detail.splitlines() if line.strip()]
        return lines[0] if lines else f"exit status {self.returncode}"

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    """One host process invocation; the only seam between the adapter and real tools."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        input_bytes: bytes | None = None,
    ) -> CommandResult: ...


class HostCommandRunner:
    """Run the real tool, bounded by the caller's UI-read deadline when one is active."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        budget = read_budget.current()
        if budget is not None:
            # ``remaining()`` raises ReadDeadlineExceeded itself once the deadline has passed.
            timeout_s = min(timeout_s, budget.remaining())
        try:
            proc = subprocess.run(  # noqa: S603 - argv is built from typed adapter inputs
                list(argv),
                capture_output=True,
                timeout=timeout_s,
                input=input_bytes,
                check=False,
            )
        except FileNotFoundError as exc:
            raise DeviceError(
                f"{argv[0]} is not installed or not on PATH",
                code="ios_tool_missing",
                hint=AXE_INSTALL_HINT if Path(argv[0]).name == "axe" else XCRUN_HINT,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            if budget is not None:
                raise read_budget.ReadDeadlineExceeded("UI-read deadline reached") from exc
            raise DeviceError(
                f"{Path(argv[0]).name} {argv[1] if len(argv) > 1 else ''} timed out after "
                f"{timeout_s:.0f}s",
                code="ios_tool_timeout",
            ) from exc
        return CommandResult(
            argv=tuple(argv),
            returncode=proc.returncode,
            stdout=proc.stdout or b"",
            stderr=proc.stderr or b"",
        )


@dataclass(frozen=True)
class SimulatorInfo:
    """One simulator as ``simctl list -j`` reports it, with its runtime version resolved."""

    udid: str
    name: str
    state: str
    runtime_id: str
    os_version: str | None
    is_available: bool
    data_path: str | None
    last_booted_at: str | None

    @property
    def booted(self) -> bool:
        return self.state == "Booted"


class IOSTools:
    """Typed wrappers over ``simctl`` and ``axe`` sharing one :class:`CommandRunner`."""

    def __init__(
        self,
        runner: CommandRunner,
        *,
        axe_path: str | None = None,
        xcrun_path: str | None = None,
    ) -> None:
        self._runner = runner
        self._axe_path = axe_path
        self._xcrun_path = xcrun_path

    # -- tool discovery -------------------------------------------------------------------

    def resolve_axe(self) -> str:
        candidate = self._axe_path or shutil.which("axe")
        if not candidate:
            raise DeviceError(
                "AXe (`axe`) was not found on PATH", code="ios_tool_missing", hint=AXE_INSTALL_HINT
            )
        return candidate

    def resolve_xcrun(self) -> str:
        candidate = self._xcrun_path or shutil.which("xcrun")
        if not candidate:
            raise DeviceError(
                "`xcrun` was not found on PATH", code="ios_tool_missing", hint=XCRUN_HINT
            )
        return candidate

    # -- raw invocations ------------------------------------------------------------------

    def simctl(
        self,
        *args: str,
        timeout_s: float = 30.0,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> CommandResult:
        result = self._runner.run(
            [self.resolve_xcrun(), "simctl", *args], timeout_s=timeout_s, input_bytes=input_bytes
        )
        if check and not result.ok:
            raise self.error(result, f"simctl {args[0] if args else ''}".strip())
        return result

    def axe(
        self,
        *args: str,
        timeout_s: float = 30.0,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> CommandResult:
        result = self._runner.run(
            [self.resolve_axe(), *args], timeout_s=timeout_s, input_bytes=input_bytes
        )
        if check and not result.ok:
            raise self.error(result, f"axe {args[0] if args else ''}".strip())
        return result

    def axe_version(self) -> str | None:
        result = self.axe("--version", timeout_s=10.0, check=False)
        return result.text.strip() or None if result.ok else None

    @staticmethod
    def error(result: CommandResult, what: str) -> DeviceError:
        """Translate a failed tool run into AUA's typed device error."""

        detail = result.error_text
        folded = detail.casefold()
        if "not booted" in folded:
            return DeviceError(
                f"{what} failed: {detail}",
                code="ios_target_not_booted",
                hint="Boot the simulator first (`aua devices` lists it; `xcrun simctl boot <udid>`).",
            )
        if "no simulator with udid" in folded or "invalid device" in folded:
            return DeviceError(
                f"{what} failed: {detail}",
                code="ios_target_not_found",
                hint="Pass a UDID or unique name from `aua devices`.",
            )
        return DeviceError(f"{what} failed: {detail}", code="ios_tool_failed")

    # -- simctl readers -------------------------------------------------------------------

    def list_simulators(self) -> list[SimulatorInfo]:
        payload = json.loads(self.simctl("list", "-j", timeout_s=20.0).text or "{}")
        versions: dict[str, str] = {}
        for runtime in payload.get("runtimes") or []:
            identifier = str(runtime.get("identifier") or "")
            version = runtime.get("version")
            if identifier and version:
                versions[identifier] = str(version)
        simulators: list[SimulatorInfo] = []
        for runtime_id, devices in (payload.get("devices") or {}).items():
            for device in devices or []:
                udid = str(device.get("udid") or "")
                if not udid:
                    continue
                simulators.append(
                    SimulatorInfo(
                        udid=udid,
                        name=str(device.get("name") or udid),
                        state=str(device.get("state") or "Unknown"),
                        runtime_id=str(runtime_id),
                        os_version=versions.get(str(runtime_id))
                        or _version_from_runtime_id(runtime_id),
                        is_available=bool(device.get("isAvailable", True)),
                        data_path=device.get("dataPath"),
                        last_booted_at=device.get("lastBootedAt"),
                    )
                )
        return simulators

    def find_simulator(self, udid: str) -> SimulatorInfo | None:
        return next((sim for sim in self.list_simulators() if sim.udid == udid), None)

    def plist_json(self, data: bytes) -> Any:
        """Convert simctl's OpenStep-style plist output into JSON via ``plutil``."""

        if not data.strip():
            return {}
        result = self._runner.run(
            ["plutil", "-convert", "json", "-o", "-", "-"], timeout_s=10.0, input_bytes=data
        )
        if not result.ok:
            raise DeviceError(
                f"could not decode simctl output: {result.error_text}", code="ios_tool_failed"
            )
        return json.loads(result.text or "{}")

    def app_info(self, udid: str, app_id: str) -> dict[str, Any]:
        """``simctl appinfo``: an absent app yields only its identifier back."""

        result = self.simctl("appinfo", udid, app_id, timeout_s=15.0, check=False)
        if not result.ok:
            return {}
        info = self.plist_json(result.stdout)
        return info if isinstance(info, dict) else {}

    def installed_apps(self, udid: str) -> dict[str, dict[str, Any]]:
        result = self.simctl("listapps", udid, timeout_s=20.0)
        apps = self.plist_json(result.stdout)
        return apps if isinstance(apps, dict) else {}

    def running_apps(self, udid: str) -> dict[int, str]:
        """pid -> bundle id for every UIKit app process, plus the home screen itself."""

        result = self.simctl("spawn", udid, "launchctl", "list", timeout_s=10.0, check=False)
        if not result.ok:
            return {}
        return parse_launchctl_list(result.text)

    def screenshot_png(self, udid: str, path: Path) -> bytes:
        self.simctl("io", udid, "screenshot", "--type", "png", str(path), timeout_s=20.0)
        try:
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)


def parse_launchctl_list(text: str) -> dict[int, str]:
    """Map ``launchctl list`` rows (pid, status, label) to UIKit bundle identifiers."""

    apps: dict[int, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].strip().isdigit():
            continue
        pid = int(parts[0])
        label = parts[2].strip()
        if label.startswith("UIKitApplication:"):
            bundle = label.split(":", 1)[1].split("[", 1)[0]
            if bundle:
                apps[pid] = bundle
        elif label == "com.apple.SpringBoard":
            apps[pid] = SPRINGBOARD_APP_ID
    return apps


def _version_from_runtime_id(runtime_id: str) -> str | None:
    # com.apple.CoreSimulator.SimRuntime.iOS-26-5 -> 26.5
    tail = runtime_id.rsplit(".", 1)[-1]
    if "-" not in tail:
        return None
    _, _, version = tail.partition("-")
    return version.replace("-", ".") or None


def read_app_bundle_info(bundle: Path) -> dict[str, Any]:
    """Info.plist of a simulator ``.app`` bundle on the host, via the standard library."""

    path = bundle.expanduser()
    if path.suffix.lower() == ".ipa":
        raise UsageError(
            f"{path.name} is an .ipa archive; simulators install .app bundles",
            hint="Build for the iphonesimulator SDK and pass the resulting .app directory.",
        )
    plist = path / "Info.plist" if path.is_dir() else path
    if not plist.is_file():
        raise UsageError(
            f"{path} is not an iOS .app bundle (no Info.plist)",
            hint="Pass the .app directory produced by an iphonesimulator build.",
        )
    try:
        info = plistlib.loads(plist.read_bytes())
    except Exception as exc:  # noqa: BLE001 - any malformed plist is the same usage error
        raise UsageError(f"could not read {plist}: {exc}") from exc
    if not isinstance(info, dict) or not info.get("CFBundleIdentifier"):
        raise UsageError(f"{plist} has no CFBundleIdentifier")
    return info


def png_size(data: bytes) -> tuple[int, int] | None:
    """Width/height from a PNG header without decoding the image."""

    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return (width, height) if width > 0 and height > 0 else None


__all__ = [
    "AXE_INSTALL_HINT",
    "SPRINGBOARD_APP_ID",
    "SYSTEM_APP_IDS",
    "XCRUN_HINT",
    "CommandResult",
    "CommandRunner",
    "HostCommandRunner",
    "IOSTools",
    "SimulatorInfo",
    "parse_launchctl_list",
    "png_size",
    "read_app_bundle_info",
]
