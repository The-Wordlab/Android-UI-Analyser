from __future__ import annotations

import base64
import io
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from PIL import Image

from android_ui_analyser.chrome_extension_setup import (
    EXTENSION_ID,
    chrome_extension_status,
    install_chrome_extension,
    native_manifest_path,
)
from android_ui_analyser.chrome_native_host import read_native_message, write_native_message
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import ConfigError, UnsupportedPlatformCapabilityError
from android_ui_analyser.platforms.chrome_extension import (
    ATTACHED_TARGET_ID,
    BRIDGE_PROTOCOL,
    ChromeAttachOptions,
    ChromeBridgeServer,
    ChromeExtensionConnection,
)
from android_ui_analyser.platforms.registry import PlatformFactory
from android_ui_analyser.platforms.web import WebPlatform


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(stream, format="PNG")
    return stream.getvalue()


class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.events = [
            {
                "kind": "console",
                "level": "warning",
                "message": "fixture warning",
                "url": "https://example.test/app",
                "timestamp_ms": 50,
            }
        ]

    def request(self, method, params=None, *, timeout_ms):
        self.calls.append((method, dict(params or {})))
        if method == "page_state":
            return {
                "page_id": "tab-42",
                "url": "https://example.test/app",
                "title": "Fixture",
                "viewport": {"width": 320, "height": 240},
            }
        if method == "evaluate":
            return {
                "format": "aua-web-dom/1",
                "url": "https://example.test/app",
                "title": "Fixture",
                "viewport": {"width": 320, "height": 240},
                "nodes": [
                    {
                        "tag": "button",
                        "text": "Continue",
                        "resource_id": "continue",
                        "bounds": [10, 20, 120, 60],
                        "clickable": True,
                    }
                ],
            }
        if method == "capture_screenshot":
            return {"data": base64.b64encode(_png()).decode("ascii")}
        return {}

    def event_snapshot(self):
        return [dict(event) for event in self.events]

    def clear_events(self):
        count = len(self.events)
        self.events.clear()
        return count

    def close(self):
        self.closed = True


class FakeExtensionLauncher:
    def __init__(self, connection: ChromeExtensionConnection) -> None:
        self.connection = connection
        self.calls = []

    def launch(self, target, options):
        self.calls.append((target, options))
        return self.connection


def _config(tmp_path: Path, **web_options) -> Config:
    return Config.model_validate(
        {
            "device": {"platform": "web", "serial": ATTACHED_TARGET_ID},
            "platforms": {"web": {"connection": "existing-chrome", **web_options}},
            "cache": {"dir": str(tmp_path / "cache")},
            "memory": {"enabled": False},
            "lease": {"enabled": False},
            "teardown": {"enabled": False},
            "capture": {"enabled": False},
            "ocr": {"enabled": False},
        }
    )


def test_attached_connection_drives_only_the_approved_tab() -> None:
    bridge = FakeBridge()
    connection = ChromeExtensionConnection(ChromeAttachOptions(), bridge=bridge)

    assert connection.url == "https://example.test/app"
    assert connection.viewport_size() == (320, 240)
    assert json.loads(connection.snapshot())["nodes"][0]["resource_id"] == "continue"
    assert connection.screenshot_png().startswith(b"\x89PNG")
    connection.click(25, 40)
    connection.type_text("hello")
    connection.scroll(100, 200, 0, 300)
    assert connection.pages()["pages"] == [
        {
            "id": "tab-42",
            "index": 0,
            "active": True,
            "attached": True,
            "url": "https://example.test/app",
            "title": "Fixture",
            "frames": [],
        }
    ]
    assert connection.diagnostics(limit=10, kinds=(), since_ms=None)["count"] == 1

    with pytest.raises(UnsupportedPlatformCapabilityError, match="browser.storage"):
        connection.storage(include_values=False)
    with pytest.raises(ConfigError, match="was not approved"):
        connection.page_select("tab-99")

    connection.close()
    assert bridge.calls[-1][0] == "detach"
    assert bridge.closed is True


def test_attached_screenshot_is_normalized_to_css_viewport() -> None:
    class RetinaBridge(FakeBridge):
        def request(self, method, params=None, *, timeout_ms):
            if method == "capture_screenshot":
                stream = io.BytesIO()
                Image.new("RGB", (640, 480), "white").save(stream, format="PNG")
                return {
                    "data": base64.b64encode(stream.getvalue()).decode("ascii"),
                    "viewport": {"width": 320, "height": 240},
                }
            return super().request(method, params, timeout_ms=timeout_ms)

    connection = ChromeExtensionConnection(ChromeAttachOptions(), bridge=RetinaBridge())
    with Image.open(io.BytesIO(connection.screenshot_png())) as screenshot:
        assert screenshot.size == (320, 240)


def test_web_platform_keeps_isolated_mode_but_can_select_existing_chrome(tmp_path: Path) -> None:
    bridge = FakeBridge()
    connection = ChromeExtensionConnection(ChromeAttachOptions(), bridge=bridge)
    launcher = FakeExtensionLauncher(connection)
    platform = WebPlatform(_config(tmp_path), extension_launcher=launcher)
    platform.options = platform.validate_options(
        platform.config.platform_options("web")
    )

    assert platform.list_targets()[0].target_id == ATTACHED_TARGET_ID
    assert platform.supports("ui.tree")
    assert platform.supports("browser.diagnostics")
    assert not platform.supports("browser.storage")
    assert not platform.supports("browser.network")
    assert platform.supports("session.state")

    runtime = platform.connect()
    assert runtime.target_id == ATTACHED_TARGET_ID
    assert runtime.current_app().app_id == "example.test"
    assert launcher.calls[0][0] == ATTACHED_TARGET_ID
    runtime.close()


def test_platform_factory_normalizes_existing_chrome_without_playwright(tmp_path: Path) -> None:
    platform = PlatformFactory(_config(tmp_path)).create("web")

    assert platform.options["connection"] == "existing-chrome"
    assert platform.list_targets()[0].target_id == ATTACHED_TARGET_ID
    assert not platform.supports("browser.storage")


def test_attached_tab_reuses_engine_analysis_and_action_path(tmp_path: Path) -> None:
    bridge = FakeBridge()
    connection = ChromeExtensionConnection(ChromeAttachOptions(), bridge=bridge)
    launcher = FakeExtensionLauncher(connection)
    platform = WebPlatform(_config(tmp_path), extension_launcher=launcher)
    platform.options = platform.validate_options(platform.config.platform_options("web"))
    engine = Engine(platform.config, platform=platform)

    try:
        result = engine.analyze(source="hierarchy", with_ocr=False)
        assert result.screen.package == "example.test"
        assert result.elements[0].stable_key == "rid:continue"

        action = engine.tap(selector={"rid": "continue"}, observe=False)
        assert action.ok is True
        assert ("click", {"x": 65, "y": 40}) in bridge.calls
    finally:
        engine.close()


def test_existing_chrome_rejects_profile_mutating_launch_options(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not accept isolated-browser options"):
        platform = WebPlatform(_config(tmp_path, storage_state="private.json"))
        platform.validate_options(platform.config.platform_options("web"))


def test_bridge_authenticates_and_round_trips_requests(tmp_path: Path) -> None:
    socket_path = Path("/tmp") / f"aua-chrome-test-{uuid.uuid4().hex[:8]}.sock"
    config_path = tmp_path / "bridge.json"
    server = ChromeBridgeServer(
        socket_path=socket_path,
        config_path=config_path,
        attach_timeout_ms=2_000,
    )

    def extension_side() -> None:
        deadline = time.monotonic() + 2
        while not config_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.005)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.connect(config["socket"])
        writer = peer.makefile("w", encoding="utf-8", newline="\n")
        reader = peer.makefile("r", encoding="utf-8")
        writer.write(
            json.dumps(
                {
                    "type": "host_hello",
                    "protocol": BRIDGE_PROTOCOL,
                    "token": config["token"],
                }
            )
            + "\n"
        )
        writer.write(
            json.dumps(
                {"type": "hello", "protocol": BRIDGE_PROTOCOL, "attached": True}
            )
            + "\n"
        )
        writer.flush()
        request = json.loads(reader.readline())
        writer.write(
            json.dumps(
                {"reply_to": request["id"], "ok": True, "result": {"pong": True}}
            )
            + "\n"
        )
        writer.flush()
        peer.close()

    thread = threading.Thread(target=extension_side)
    thread.start()
    server.start()
    assert server.request("ping", timeout_ms=1_000) == {"pong": True}
    server.close()
    thread.join(timeout=2)
    assert not socket_path.exists()
    assert not config_path.exists()


def test_native_message_framing_round_trip() -> None:
    stream = io.BytesIO()
    write_native_message(stream, {"hello": "world"})
    raw = stream.getvalue()
    assert struct.unpack("=I", raw[:4])[0] == len(raw) - 4
    stream.seek(0)
    assert read_native_message(stream) == {"hello": "world"}
    assert read_native_message(stream) is None


def test_native_host_waits_for_authenticated_bridge_before_reporting_ready(
    tmp_path: Path,
) -> None:
    socket_path = Path("/tmp") / f"aua-chrome-host-test-{uuid.uuid4().hex[:8]}.sock"
    config_path = tmp_path / "bridge.json"
    server = ChromeBridgeServer(
        socket_path=socket_path,
        config_path=config_path,
        attach_timeout_ms=2_000,
    )
    errors: list[BaseException] = []

    def start_server() -> None:
        try:
            server.start()
        except BaseException as exc:
            errors.append(exc)

    server_thread = threading.Thread(target=start_server)
    server_thread.start()
    deadline = time.monotonic() + 2
    while not config_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.005)

    environment = dict(os.environ)
    environment["AUA_CHROME_BRIDGE_CONFIG"] = str(config_path)
    host = subprocess.Popen(
        [sys.executable, "-m", "android_ui_analyser.chrome_native_host"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert host.stdin is not None
    assert host.stdout is not None
    ready = read_native_message(host.stdout)
    assert ready == {"type": "host_ready", "protocol": BRIDGE_PROTOCOL}
    write_native_message(
        host.stdin,
        {"type": "hello", "protocol": BRIDGE_PROTOCOL, "attached": True},
    )
    server_thread.join(timeout=2)
    assert not errors
    assert not server_thread.is_alive()

    replies: list[object] = []

    def request() -> None:
        replies.append(server.request("ping", timeout_ms=1_000))

    request_thread = threading.Thread(target=request)
    request_thread.start()
    forwarded = read_native_message(host.stdout)
    assert forwarded is not None
    write_native_message(
        host.stdin,
        {"reply_to": forwarded["id"], "ok": True, "result": {"pong": True}},
    )
    request_thread.join(timeout=2)
    assert replies == [{"pong": True}]

    server.close()
    host.stdin.close()
    try:
        host.wait(timeout=2)
    except subprocess.TimeoutExpired:
        host.terminate()
        host.wait(timeout=2)


def test_extension_installer_registers_only_the_stable_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr("android_ui_analyser.chrome_extension_setup.sys.platform", "darwin")

    result = install_chrome_extension("chrome")
    status = chrome_extension_status("chrome")
    manifest = json.loads(Path(result["native_host_manifest"]).read_text(encoding="utf-8"))
    extension_manifest = json.loads(
        (Path(result["extension_path"]) / "manifest.json").read_text(encoding="utf-8")
    )

    assert status["ok"] is True
    assert manifest["allowed_origins"] == [f"chrome-extension://{EXTENSION_ID}/"]
    assert extension_manifest["permissions"] == ["activeTab", "debugger", "nativeMessaging"]
    assert Path(manifest["path"]).stat().st_mode & 0o100
    assert str(Path(sys.executable).absolute()) in Path(manifest["path"]).read_text(
        encoding="utf-8"
    )
    assert "Google/ChromeForTesting/NativeMessagingHosts" in str(
        native_manifest_path("chrome-for-testing")
    )
