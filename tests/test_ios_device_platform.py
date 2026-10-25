"""Physical capture uses CoreDevice only and never touches simulator/Android UI tools."""

from __future__ import annotations

import copy
import io
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError, UnsupportedPlatformCapabilityError
from android_ui_analyser.platforms import PlatformFactory
from android_ui_analyser.platforms.ios_device import IOSDevicePlatform
from android_ui_analyser.platforms.ios_tools import CommandResult
from android_ui_analyser.schema import TargetStatus

UDID = "00000000-0000000000000001"
IDENTIFIER = "11111111-2222-3333-4444-555555555555"


def config(tmp_path):
    return Config.model_validate(
        {
            "device": {"platform": "ios-device", "serial": UDID},
            "cache": {"dir": str(tmp_path / "cache")},
            "memory": {"enabled": False},
            "lease": {"enabled": False},
            "teardown": {"enabled": False},
            "capture": {"enabled": False},
            "perf": {"auto_daemon": False, "prefetch": False},
        }
    )


class FakeCoreDevice:
    def __init__(self):
        self.calls = []
        self.device = {
            "identifier": IDENTIFIER,
            "properties": {
                "hardware": {"reality": "physical", "platform": "iOS", "udid": UDID},
                "state": {"name": "Fixture Phone"},
                "software": {"osVersionNumber": {"stringValue": "27.0"}},
                "connection": {"state": "connected", "pairingState": "paired"},
            },
            "capabilities": [
                {"featureIdentifier": "com.apple.coredevice.feature.capturescreenshot"}
            ],
        }
        self.inventory = [self.device]
        output = io.BytesIO()
        Image.new("RGB", (117, 253), "navy").save(output, format="PNG")
        self.png = output.getvalue()
        self.failure = None
        self.details_failure = None
        self.missing_png = False

    def run(self, argv, *, timeout_s, input_bytes=None):
        self.calls.append(tuple(argv))
        assert Path(argv[0]).name == "xcrun" and argv[1] == "devicectl"
        assert 0 < timeout_s <= 35
        result = {}
        error = None
        if argv[2:4] == ["list", "devices"]:
            result = {"devices": self.inventory}
        elif argv[2:5] == ["device", "info", "details"]:
            result = self.device
            error = self.details_failure
        elif argv[2:5] == ["device", "capture", "screenshot"]:
            assert argv[argv.index("--device") + 1] == UDID
            error = self.failure
            if not error and not self.missing_png:
                Path(argv[argv.index("--destination") + 1]).write_bytes(self.png)
        else:
            pytest.fail(f"unexpected native call: {argv}")
        payload = {"info": {"outcome": "failed" if error else "success"}, "result": result}
        if error:
            payload["error"] = {"localizedDescription": error}
        Path(argv[argv.index("--json-output") + 1]).write_text(json.dumps(payload))
        return CommandResult(tuple(argv), 1 if error else 0, b"human progress", b"")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/fake/bin/xcrun" if name == "xcrun" else None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected real process"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected real process"))
    host = FakeCoreDevice()
    platform = IOSDevicePlatform(config(tmp_path), host)
    return host, platform


def test_registered_adapter_only_promises_capture(tmp_path):
    platform = PlatformFactory(config(tmp_path)).create()
    assert isinstance(platform, IOSDevicePlatform)
    assert platform.capabilities == {"ui.screenshot"}


def test_discovery_filters_simulators_and_non_ios_and_understands_new_json(setup):
    host, platform = setup
    simulator = copy.deepcopy(host.device)
    simulator["properties"]["hardware"]["reality"] = "simulated"
    watch = copy.deepcopy(host.device)
    watch["properties"]["hardware"]["platform"] = "watchOS"
    host.inventory.extend([simulator, watch])
    targets = platform.list_targets()
    assert len(targets) == 1
    assert targets[0].target_id == UDID
    assert targets[0].model == "Fixture Phone"
    assert targets[0].status == TargetStatus.online
    assert targets[0].os_version == "27.0"


def test_older_coredevice_json_and_dormant_requested_tunnel(setup):
    host, platform = setup
    host.inventory = [
        {
            "identifier": IDENTIFIER,
            "hardwareProperties": host.device["properties"]["hardware"],
            "deviceProperties": {"name": "Fixture Phone", "osVersionNumber": "26.0"},
            "connectionProperties": {"tunnelState": "disconnected", "pairingState": "paired"},
        }
    ]
    assert platform.list_targets()[0].status == TargetStatus.online
    assert any(call[2:5] == ("device", "info", "details") for call in host.calls)


def test_unpinned_followup_probes_dormant_tunnel_and_keeps_unplugged_phone_offline(setup):
    host, platform = setup
    platform.config.device.serial = None
    dormant = copy.deepcopy(host.device)
    dormant["properties"]["connection"]["state"] = "disconnected"
    host.inventory = [dormant]
    assert platform.list_targets()[0].status == TargetStatus.online
    host.details_failure = "Device is not available"
    assert platform.list_targets()[0].status == TargetStatus.offline


def test_capture_is_fresh_and_private_temporaries_are_removed(setup, tmp_path):
    host, platform = setup
    runtime = platform.validate_runtime(platform.connect(IDENTIFIER))
    assert runtime.target_id == UDID
    engine = Engine(config(tmp_path), device=runtime, platform=platform)
    try:
        output = tmp_path / "current.png"
        result = engine.screenshot(str(output))
        assert result.ok and output.read_bytes() == host.png
        assert Image.open(output).size == (117, 253)
        for call in host.calls:
            assert not Path(call[call.index("--json-output") + 1]).exists()
            if "--destination" in call:
                assert not Path(call[call.index("--destination") + 1]).exists()
        with pytest.raises(UnsupportedPlatformCapabilityError):
            engine.analyze(source="hierarchy")
        with pytest.raises(UnsupportedPlatformCapabilityError):
            platform.runtime_capability("ui.input", runtime)
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("Device is locked; enter passcode", "ios_device_locked"),
        ("Device disconnected", "ios_device_unavailable"),
        ("Capture is not supported", "ios_screenshot_unsupported"),
    ],
)
def test_native_failures_are_typed_without_retry_or_fallback(setup, failure, code):
    host, platform = setup
    runtime = platform.connect(UDID)
    host.failure = failure
    with pytest.raises(DeviceError) as raised:
        platform.capture_screenshot(runtime)
    assert raised.value.code == code
    assert sum(call[2:5] == ("device", "capture", "screenshot") for call in host.calls) == 1
    assert all(not Path(call[call.index("--json-output") + 1]).exists() for call in host.calls)


@pytest.mark.parametrize("missing", [True, False])
def test_no_success_for_missing_or_invalid_png(setup, missing):
    host, platform = setup
    runtime = platform.connect(UDID)
    host.missing_png = missing
    host.png = b"not a png"
    with pytest.raises(DeviceError, match="screenshot|PNG") as raised:
        platform.capture_screenshot(runtime)
    assert raised.value.code == "screencap_failed"


def test_no_capture_when_device_does_not_support_it(setup):
    host, platform = setup
    host.device["capabilities"] = []
    with pytest.raises(DeviceError) as raised:
        platform.connect(UDID)
    assert raised.value.code == "ios_screenshot_unsupported"
    assert not any(call[2:5] == ("device", "capture", "screenshot") for call in host.calls)


def test_valid_header_does_not_hide_a_truncated_png(setup):
    host, platform = setup
    runtime = platform.connect(UDID)
    host.png = host.png[:24]
    with pytest.raises(DeviceError) as raised:
        platform.capture_screenshot(runtime)
    assert raised.value.code == "screencap_failed"


def test_ambiguous_devices_are_not_chosen(setup):
    host, platform = setup
    other = copy.deepcopy(host.device)
    other["properties"]["hardware"]["udid"] = "00000000-0000000000000002"
    host.inventory.append(other)
    with pytest.raises(DeviceError) as raised:
        platform.connect()
    assert raised.value.code == "multiple_targets"
