"""The built-in iOS adapter drives simctl and AXe through one seam, never Android tooling."""

from __future__ import annotations

import contextlib
import io
import json
import os
import plistlib
import sqlite3
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from android_ui_analyser.config import Config
from android_ui_analyser.errors import (
    ConfigError,
    DeviceError,
    UnsupportedPlatformCapabilityError,
    UsageError,
)
from android_ui_analyser.platforms import (
    AttachedTargetCase,
    PlatformFactory,
    registry,
    run_attached_target_conformance,
)
from android_ui_analyser.platforms.ios import IOSPlatform
from android_ui_analyser.platforms.ios_tools import CommandResult
from android_ui_analyser.schema import TargetStatus
from test_ios_tree import APP_ID, fixture_roots

UDID = "11111111-2222-3333-4444-555555555555"
OTHER_UDID = "66666666-7777-8888-9999-000000000000"
NAME = "iPhone Fixture"


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeSimulatorHost:
    """A scripted macOS host: answers `xcrun simctl`, `axe` and `plutil` from memory."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.inputs: dict[int, bytes] = {}
        self.states = {UDID: "Booted", OTHER_UDID: "Shutdown"}
        self.scroll_offset = 0.0
        self.running = {4242: APP_ID, 77: "com.apple.springboard"}
        self.clipboard = ""
        self.installed = {
            APP_ID: {
                "CFBundleIdentifier": APP_ID,
                "CFBundleVersion": "12",
                "CFBundleShortVersionString": "1.2.0",
                "Path": "/fixture/App.app",
            }
        }
        self.png = _png(300, 600)
        self.describe_override: list[dict] | None = None
        self.data_path = "/fixture/data"
        self.container: Path | None = None

    # -- the seam --------------------------------------------------------------------------

    def run(
        self, argv: Sequence[str], *, timeout_s: float, input_bytes: bytes | None = None
    ) -> CommandResult:
        argv = tuple(argv)
        self.calls.append(argv)
        if input_bytes is not None:
            self.inputs[len(self.calls) - 1] = input_bytes
        tool = Path(argv[0]).name
        if tool == "plutil":
            return self._ok(argv, input_bytes or b"{}")
        if tool == "xcrun":
            assert argv[1] == "simctl"
            return self._simctl(argv, argv[2:], input_bytes)
        if tool == "axe":
            return self._axe(argv, argv[1:], input_bytes)
        if tool == "ps":
            assert argv[1:] == ("-p", "1234", "-o", "lstart=")
            return self._ok(argv, b"Fri Sep 18 10:00:00 2026\n")
        raise AssertionError(f"unexpected host tool {argv!r}")

    @staticmethod
    def _ok(argv: tuple[str, ...], stdout: bytes = b"") -> CommandResult:
        return CommandResult(argv=argv, returncode=0, stdout=stdout, stderr=b"")

    @staticmethod
    def _fail(argv: tuple[str, ...], message: str, code: int = 1) -> CommandResult:
        return CommandResult(argv=argv, returncode=code, stdout=b"", stderr=message.encode())

    def _simctl(
        self, argv: tuple[str, ...], args: tuple[str, ...], input_bytes: bytes | None
    ) -> CommandResult:
        verb = args[0]
        if verb == "list":
            payload = {
                "runtimes": [
                    {"identifier": "com.apple.CoreSimulator.SimRuntime.iOS-26-5", "version": "26.5"}
                ],
                "devices": {
                    "com.apple.CoreSimulator.SimRuntime.iOS-26-5": [
                        {
                            "udid": UDID,
                            "name": NAME,
                            "state": self.states[UDID],
                            "isAvailable": True,
                            "dataPath": self.data_path,
                            "lastBootedAt": "2026-09-13T10:00:00Z",
                        },
                        {
                            "udid": OTHER_UDID,
                            "name": "iPad Fixture",
                            "state": self.states[OTHER_UDID],
                            "isAvailable": True,
                            "dataPath": "/fixture/other",
                        },
                        {
                            "udid": "unavailable",
                            "name": "Broken",
                            "state": "Shutdown",
                            "isAvailable": False,
                        },
                    ]
                },
            }
            return self._ok(argv, json.dumps(payload).encode())
        udid = args[1]
        if udid not in self.states:
            return self._fail(argv, f"Invalid device: {udid}", 164)
        if verb == "boot":
            self.states[udid] = "Booted"
            return self._ok(argv)
        if verb == "bootstatus":
            return self._ok(argv, b"Finished\n")
        if self.states[udid] != "Booted" and verb in {
            "io",
            "spawn",
            "launch",
            "terminate",
            "openurl",
            "pbcopy",
            "pbpaste",
        }:
            return self._fail(argv, "Unable to lookup in current state: Shutdown", 149)
        if verb == "io":
            assert args[2:5] == ("screenshot", "--type", "png")
            Path(args[5]).write_bytes(self.png)
            return self._ok(argv)
        if verb == "spawn" and args[2:] == ("launchctl", "list"):
            rows = "".join(
                f"{pid}\t0\t{'com.apple.SpringBoard' if bundle == 'com.apple.springboard' else f'UIKitApplication:{bundle}[0e2c][rb-legacy]'}\n"
                for pid, bundle in self.running.items()
            )
            return self._ok(argv, rows.encode())
        if verb == "spawn" and args[2:] == ("launchctl", "managerpid"):
            return self._ok(argv, b"1234\n")
        if verb == "spawn" and args[2:] == ("defaults", "read", "-g", "AppleLocale"):
            return self._ok(argv, b"en_US@rg=eszzzz\n")
        if verb == "appinfo":
            info = self.installed.get(args[2], {"CFBundleIdentifier": args[2]})
            return self._ok(argv, json.dumps(info).encode())
        if verb == "listapps":
            return self._ok(argv, json.dumps(self.installed).encode())
        if verb == "launch":
            if args[2] not in self.installed:
                return self._fail(argv, f"Simulator device failed to launch {args[2]}.", 4)
            return self._ok(argv, f"{args[2]}: 4242\n".encode())
        if verb == "terminate":
            return (
                self._ok(argv)
                if args[2] in self.installed
                else self._fail(argv, "found nothing to terminate", 3)
            )
        if verb == "openurl":
            return (
                self._ok(argv)
                if args[2].startswith("https://")
                else self._fail(argv, f"failed to open {args[2]}", 115)
            )
        if verb == "pbcopy":
            self.clipboard = (input_bytes or b"").decode()
            return self._ok(argv)
        if verb == "pbpaste":
            return self._ok(argv, self.clipboard.encode())
        if verb == "get_app_container":
            if args[2] in self.installed and self.container is not None:
                return self._ok(argv, f"{self.container}\n".encode())
            return self._fail(argv, "No such file or directory", 2)
        if verb in {"location", "privacy", "install", "uninstall"}:
            return self._ok(argv)
        raise AssertionError(f"unexpected simctl call {args!r}")

    def _axe(
        self, argv: tuple[str, ...], args: tuple[str, ...], input_bytes: bytes | None
    ) -> CommandResult:
        verb = args[0]
        if verb == "--version":
            return self._ok(argv, b"1.8.0\n")
        udid = args[args.index("--udid") + 1]
        if udid not in self.states:
            return self._fail(argv, f"Error: No simulator with UDID {udid} was found.")
        if self.states[udid] != "Booted":
            return self._fail(
                argv, f"Error: Cannot run accessibility commands against {udid} as it is not booted"
            )
        if verb == "describe-ui":
            roots = self.describe_override or fixture_roots(self.scroll_offset)
            if "--point" in args:
                return self._ok(argv, json.dumps(roots[0]["children"][0]["children"][2]).encode())
            return self._ok(argv, json.dumps(roots).encode())
        if verb == "swipe":
            self.scroll_offset += 10.0
        if verb in {
            "tap",
            "touch",
            "swipe",
            "type",
            "key",
            "key-combo",
            "key-sequence",
            "button",
            "gesture",
        }:
            return self._ok(argv, b"done\n")
        raise AssertionError(f"unexpected axe call {args!r}")

    def argv_of(self, *prefix: str) -> list[tuple[str, ...]]:
        """Recorded calls starting with *prefix*, with the tool path reduced to its name."""

        named = [(Path(call[0]).name, *call[1:]) for call in self.calls]
        return [call for call in named if call[: len(prefix)] == prefix]

    def input_for(self, *prefix: str) -> bytes | None:
        """Stdin handed to the first recorded call starting with *prefix*."""

        for index, call in enumerate(self.calls):
            if (Path(call[0]).name, *call[1:])[: len(prefix)] == prefix:
                return self.inputs.get(index)
        return None


def _config(tmp_path: Path, **platform_options) -> Config:
    return Config.model_validate(
        {
            "device": {"platform": "ios"},
            "platforms": {"ios": platform_options},
            "cache": {"dir": str(tmp_path / "cache")},
            "memory": {"enabled": False},
            "lease": {"enabled": False},
            "teardown": {"enabled": False},
            "ocr": {"enabled": False},
            "perf": {"prefetch": False, "predictive_prefetch": False, "auto_daemon": False},
        }
    )


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> FakeSimulatorHost:
    fake = FakeSimulatorHost()
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail(f"real process invoked: {a!r}")
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail(f"real process invoked: {a!r}")
    )
    return fake


@pytest.fixture
def adapter(
    tmp_path: Path, host: FakeSimulatorHost, monkeypatch: pytest.MonkeyPatch
) -> IOSPlatform:
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")
    platform = IOSPlatform(_config(tmp_path), runner=host)
    platform.options = platform.validate_options({})
    platform.validate_declared_capabilities()
    return platform


def test_ios_is_a_built_in_platform_selected_without_loading_android(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    loaded: list[str] = []
    original = registry._load_builtin

    def spy(name: str) -> None:
        loaded.append(name)
        original(name)

    monkeypatch.setattr(registry, "_load_builtin", spy)

    assert "ios" in registry.available_platforms()
    platform = PlatformFactory(_config(tmp_path)).create()

    assert isinstance(platform, IOSPlatform)
    assert platform.name == "ios"
    # Selecting iOS never imports the Android strategy (this module may already hold "ios").
    assert "android" not in loaded


def test_ios_options_are_closed_and_typed(tmp_path: Path) -> None:
    platform = IOSPlatform(_config(tmp_path))

    assert dict(platform.validate_options({"axe_path": " /opt/axe ", "boot_timeout_s": 30})) == {
        "axe_path": "/opt/axe",
        "boot_timeout_s": 30.0,
    }
    with pytest.raises(ConfigError, match="does not accept options: endpoint"):
        platform.validate_options({"endpoint": "x"})
    with pytest.raises(ConfigError, match="boot_timeout_s"):
        platform.validate_options({"boot_timeout_s": True})


def test_the_attached_target_profile_passes_through_the_engine(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    report = run_attached_target_conformance(
        adapter,
        AttachedTargetCase(
            target_id=UDID,
            element_text="Continue",
            expected_bounds=(30, 60, 90, 150),
            expected_app_id=APP_ID,
            require_non_identity_geometry=True,
            input_element_text="me@example.test",
            key_name="home",
            expected_scrollable_bounds=(15, 150, 285, 570),
            unsupported_capability="device.logs",
        ),
    )

    assert report.platform == "ios"
    assert report.geometry == (3.0, 0.0, 0.0, 3.0, 0.0, 0.0)
    assert report.screenshot_size == (300, 600)
    assert report.tap_point == (60, 105)
    assert "engine-verified-swipe" in report.checks
    # Canonical pixels went out as logical points.
    assert ("axe", "tap", "-x", "20", "-y", "35", "--udid", UDID) in host.argv_of("axe", "tap")
    assert host.input_for("axe", "type") == b"aua conformance"
    assert ("axe", "button", "home", "--udid", UDID) in host.argv_of("axe", "button")
    assert host.argv_of("axe", "swipe")
    assert not [call for call in host.calls if Path(call[0]).name == "adb"]


def test_discovery_reports_every_available_simulator_with_its_boot_state(
    adapter: IOSPlatform,
) -> None:
    targets = adapter.list_targets()

    assert [(t.target_id, t.status, t.model, t.os_name, t.os_version) for t in targets] == [
        (UDID, TargetStatus.online, NAME, "ios", "26.5"),
        (OTHER_UDID, TargetStatus.offline, "iPad Fixture", "ios", "26.5"),
    ]
    assert targets[0].state == "device" and targets[1].state == "offline"
    assert adapter.target_preference(targets[0]) < adapter.target_preference(targets[1])


def test_connecting_unpinned_picks_the_only_booted_simulator(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(None)

    assert runtime.target_id == UDID
    assert runtime.window_size() == (300, 600)
    assert runtime.instance_token() == "2026-09-13T10:00:00Z"
    assert runtime.device_locale() == "en-US"
    assert not host.argv_of("xcrun", "simctl", "boot")


def test_connecting_a_named_shut_down_simulator_boots_it_first(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect("iPad Fixture")

    assert runtime.target_id == OTHER_UDID
    assert runtime.instance_token() == "launchd:1234:Fri Sep 18 10:00:00 2026"
    assert host.argv_of("xcrun", "simctl", "boot", OTHER_UDID)
    assert host.argv_of("xcrun", "simctl", "bootstatus", OTHER_UDID, "-b")
    assert host.states[OTHER_UDID] == "Booted"


def test_connection_errors_are_typed(adapter: IOSPlatform, host: FakeSimulatorHost) -> None:
    with pytest.raises(DeviceError) as missing:
        adapter.connect("no-such-simulator")
    assert missing.value.code == "ios_target_not_found"

    host.states[UDID] = "Shutdown"
    with pytest.raises(DeviceError) as none_booted:
        adapter.connect(None)
    assert none_booted.value.code == "no_target"

    host.states[UDID] = host.states[OTHER_UDID] = "Booted"
    with pytest.raises(DeviceError) as many:
        adapter.connect(None)
    assert many.value.code == "multiple_targets"


def test_current_app_reads_the_process_under_the_screen_without_a_full_tree(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(UDID)
    host.calls.clear()

    assert runtime.current_app().app_id == APP_ID
    describe = host.argv_of("axe", "describe-ui")
    assert len(describe) == 1 and "--point" in describe[0]


def test_keys_map_to_hid_codes_buttons_and_the_back_gesture(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(UDID)
    host.calls.clear()

    runtime.press("enter")
    runtime.press("back")
    runtime.press("lock")
    runtime.press("hid:58")
    assert ("axe", "key", "40", "--udid", UDID) in host.argv_of("axe", "key")
    assert ("axe", "key", "58", "--udid", UDID) in host.argv_of("axe", "key")
    assert ("axe", "button", "lock", "--udid", UDID) in host.argv_of("axe", "button")
    gesture = host.argv_of("axe", "gesture")[0]
    assert gesture[2] == "swipe-from-left-edge" and gesture[3:7] == (
        "--screen-width",
        "100",
        "--screen-height",
        "200",
    )
    with pytest.raises(UsageError):
        adapter.normalize_key("volume_up")
    with pytest.raises(UsageError):
        runtime.press("volume_up")
    assert adapter.normalize_key("Home") == "Home"


def test_non_ascii_text_goes_through_the_pasteboard(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(UDID)
    host.calls.clear()

    runtime.send_text("héllo wörld", clear=False)

    assert not host.argv_of("axe", "type")
    assert host.clipboard == "héllo wörld"
    assert (
        "axe",
        "key-combo",
        "--modifiers",
        "227",
        "--key",
        "25",
        "--udid",
        UDID,
    ) in host.argv_of("axe", "key-combo")


def test_clear_then_type_selects_all_deletes_and_types_ascii_via_stdin(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(UDID)
    host.calls.clear()

    runtime.send_text("plain", clear=True)
    runtime.send_ime_action("search")

    names = [
        tuple(Path(c[0]).name if i == 0 else p for i, p in enumerate(c))[:2] for c in host.calls
    ]
    assert names == [("axe", "key-combo"), ("axe", "key"), ("axe", "type"), ("axe", "key")]
    assert host.inputs[2] == b"plain"


def test_lifecycle_links_clipboard_and_location_use_simctl(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(UDID)
    host.calls.clear()

    runtime.launch_app(APP_ID, activity="ignored")
    runtime.stop_app("com.example.absent")  # nothing to terminate is not an error
    runtime.open_link("https://example.test/path")
    runtime.set_clipboard("copied")
    assert runtime.get_clipboard() == "copied"
    runtime.paste()
    runtime.set_location(48.85, 2.35)
    runtime.grant_permissions(APP_ID)
    assert runtime.app_version(APP_ID) == "1.2.0"

    assert host.argv_of("xcrun", "simctl", "launch", UDID, APP_ID)
    assert host.argv_of("xcrun", "simctl", "openurl", UDID, "https://example.test/path")
    assert host.argv_of("xcrun", "simctl", "location", UDID, "set", "48.85,2.35")
    assert host.argv_of("xcrun", "simctl", "privacy", UDID, "grant", "all", APP_ID)
    with pytest.raises(DeviceError) as launch:
        runtime.launch_app("com.example.absent")
    assert launch.value.code == "app_launch_failed"
    with pytest.raises(DeviceError) as link:
        runtime.open_link("nosuchscheme-zz://x")
    assert link.value.code == "link_unhandled"


def test_clear_app_wipes_only_the_data_container_then_resets_permissions(
    adapter: IOSPlatform, host: FakeSimulatorHost, tmp_path: Path
) -> None:
    container = (
        tmp_path / "Containers" / "Data" / "Application" / "ABCDEF01-0000-0000-0000-000000000001"
    )
    for name in ("Documents", "Library/Preferences", "SystemData", "tmp"):
        (container / name).mkdir(parents=True)
    (container / "Library" / "Preferences" / "settings.plist").write_bytes(b"<plist/>")
    (container / ".com.apple.mobile_container_manager.metadata.plist").write_bytes(b"<plist/>")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    os.symlink(outside, container / "link")
    host.container = container
    runtime = adapter.connect(UDID)
    host.calls.clear()

    assert runtime.clear_app(APP_ID)

    assert sorted(path.name for path in container.iterdir()) == [
        ".com.apple.mobile_container_manager.metadata.plist"
    ]
    assert (outside / "keep").exists()  # the symlink was unlinked, never followed
    assert [call[2] for call in host.argv_of("xcrun", "simctl")] == [
        "terminate",
        "get_app_container",
        "privacy",
    ]
    assert host.argv_of("xcrun", "simctl", "privacy", UDID, "reset", "all", APP_ID)
    with pytest.raises(DeviceError) as exc:
        runtime.clear_app("com.example.absent")
    assert exc.value.code == "app_not_installed"


def test_permission_snapshot_reads_tcc_and_restore_replays_it(
    adapter: IOSPlatform, host: FakeSimulatorHost, tmp_path: Path
) -> None:
    host.data_path = str(tmp_path)
    db = tmp_path / "Library" / "TCC" / "TCC.db"
    db.parent.mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE access (service TEXT, client TEXT, auth_value INTEGER)")
        conn.executemany(
            "INSERT INTO access VALUES (?, ?, ?)",
            [
                ("kTCCServicePhotos", APP_ID, 2),
                ("kTCCServiceMicrophone", APP_ID, 0),
                ("kTCCServiceCamera", APP_ID, 2),  # no simctl service name: not restorable
                ("kTCCServiceCalendar", "com.example.other", 2),
            ],
        )
        conn.commit()
    runtime = adapter.connect(UDID)
    host.calls.clear()

    assert runtime.granted_permissions(APP_ID) == ["photos"]
    runtime.restore_permissions(APP_ID, ["photos", "not-a-service"])

    privacy = [call[4:] for call in host.argv_of("xcrun", "simctl", "privacy", UDID)]
    assert privacy == [("reset", "all", APP_ID), ("grant", "photos", APP_ID)]


def test_app_status_and_install_read_bundles_with_the_standard_library(
    adapter: IOSPlatform, host: FakeSimulatorHost, tmp_path: Path
) -> None:
    runtime = adapter.connect(UDID)
    bundle = tmp_path / "Fixture.app"
    bundle.mkdir()
    (bundle / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "com.example.newbuild",
                "CFBundleShortVersionString": "2.0",
                "CFBundleVersion": "20",
            }
        )
    )

    info = adapter.inspect_app_bundle(bundle)
    assert (info.app_id, info.version_name, info.version_code) == (
        "com.example.newbuild",
        "2.0",
        "20",
    )
    assert adapter.installed_app(runtime, APP_ID).installed is True
    assert adapter.installed_app(runtime, APP_ID).version_name == "1.2.0"
    assert adapter.installed_app(runtime, "com.example.absent").installed is False

    adapter.install_app_bundle(runtime, bundle, grant_permissions=True)
    assert host.argv_of("xcrun", "simctl", "install", UDID, str(bundle))
    assert host.argv_of("xcrun", "simctl", "privacy", UDID, "grant", "all", "com.example.newbuild")
    adapter.uninstall_app(runtime, "com.example.newbuild")
    assert host.argv_of("xcrun", "simctl", "uninstall", UDID, "com.example.newbuild")
    with pytest.raises(UsageError, match=r"\.ipa"):
        adapter.inspect_app_bundle(tmp_path / "Fixture.ipa")
    with pytest.raises(UsageError, match="Info.plist"):
        adapter.inspect_app_bundle(tmp_path / "Missing.app")


def test_peek_is_read_only_and_answers_for_a_target_nobody_connected(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    context = adapter.peek_foreground_app(UDID)
    frame = adapter.peek_screenshot(UDID)

    assert context is not None and context.app_id == APP_ID
    assert (frame.width, frame.height) == (300, 600)
    verbs = {(Path(c[0]).name, c[1] if len(c) > 1 else "") for c in host.calls}
    assert ("axe", "describe-ui") in verbs
    assert not {
        v for v in verbs if v[1] in {"tap", "touch", "swipe", "type", "key", "boot", "launch"}
    }
    assert adapter.peek_foreground_app(OTHER_UDID) is None


def test_optional_android_only_capabilities_refuse_with_a_typed_error(adapter: IOSPlatform) -> None:
    for name in (
        "device.logs",
        "virtual_targets",
        "device.shell",
        "device.orientation",
    ):
        with pytest.raises(UnsupportedPlatformCapabilityError) as exc:
            adapter.capability(name)
        assert exc.value.code == "platform_capability_unsupported"


@pytest.mark.parametrize("already_stopped", [True, False])
def test_stop_handles_multiline_simctl_errors_without_hiding_other_failures(
    adapter: IOSPlatform, host: FakeSimulatorHost, monkeypatch, already_stopped: bool,
) -> None:
    runtime = adapter.connect(UDID)
    original = host._simctl

    def simctl(argv, args, input_bytes):
        if args[0] == "terminate":
            return host._fail(argv,
                "An error was encountered processing the command (domain=NSPOSIXErrorDomain, code=3):\n"
                "Simulator device failed to terminate the app.\n"
                + ("found nothing to terminate" if already_stopped else "Permission denied"), 3)
        return original(argv, args, input_bytes)

    monkeypatch.setattr(host, "_simctl", simctl)
    if already_stopped:
        runtime.stop_app(APP_ID)
    else:
        with pytest.raises(DeviceError):
            runtime.stop_app(APP_ID)


def test_app_exit_evidence_recognises_a_fall_back_to_the_home_screen(adapter: IOSPlatform) -> None:
    from android_ui_analyser.schema import AppContext

    evidence = adapter.app_exit_evidence(
        AppContext(app_id=APP_ID), AppContext(app_id="com.apple.springboard"), []
    )
    assert evidence is not None and evidence.from_app_id == APP_ID and not evidence.crash_dialog
    assert (
        adapter.app_exit_evidence(
            AppContext(app_id=APP_ID), AppContext(app_id="com.example.other"), []
        )
        is None
    )
    assert (
        adapter.app_exit_evidence(
            AppContext(app_id="com.apple.springboard"), AppContext(app_id=APP_ID), []
        )
        is None
    )


def test_tool_failures_become_device_errors_with_hints(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    host.states[UDID] = "Shutdown"
    runtime_tools = adapter.tools

    with pytest.raises(DeviceError) as exc:
        runtime_tools.axe("describe-ui", "--udid", UDID)
    assert exc.value.code == "ios_target_not_booted"
    with pytest.raises(DeviceError) as missing:
        runtime_tools.axe("describe-ui", "--udid", "nope")
    assert missing.value.code == "ios_target_not_found"


def test_doctor_reports_tools_and_booted_simulators(adapter: IOSPlatform) -> None:
    checks = adapter.doctor_checks()

    assert checks["platform"]["detail"] == "ios"
    assert checks["xcrun"]["ok"] and checks["axe"]["ok"]
    assert "1.8.0" in checks["axe"]["detail"]
    assert checks["simulators"] == {
        "ok": True,
        "detail": {"available": 2, "booted": [f"{NAME} ({UDID})"]},
    }


def test_a_missing_axe_binary_is_a_clear_setup_error(
    tmp_path: Path, host: FakeSimulatorHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None if name == "axe" else f"/fake/bin/{name}")
    platform = IOSPlatform(_config(tmp_path), runner=host)
    platform.options = platform.validate_options({})

    with pytest.raises(DeviceError) as exc:
        platform.list_targets()
    assert exc.value.code == "ios_tool_missing" and "brew" in (exc.value.hint or "")
    assert platform.doctor_checks()["axe"]["ok"] is False


def test_process_names_are_looked_up_once_per_pid(
    adapter: IOSPlatform, host: FakeSimulatorHost
) -> None:
    runtime = adapter.connect(UDID)
    host.calls.clear()

    runtime.dump_hierarchy()
    runtime.current_app()
    runtime.dump_hierarchy()
    assert len(host.argv_of("xcrun", "simctl", "spawn", UDID, "launchctl", "list")) == 1

    host.running[9001] = "com.example.relaunched"
    host.scroll_offset = 0.0
    roots = fixture_roots()
    pending = list(roots)
    while pending:
        current = pending.pop()
        current["pid"] = 9001
        pending.extend(current.get("children") or [])
    host.describe_override = roots
    assert runtime.current_app().app_id == "com.example.relaunched"
    assert len(host.argv_of("xcrun", "simctl", "spawn", UDID, "launchctl", "list")) == 2
