"""Run the shipped extension worker, then consume its events through the real engine."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.chrome_extension import (
    ChromeAttachOptions,
    ChromeBridgeServer,
    ChromeExtensionConnection,
)
from android_ui_analyser.platforms.web import WebPlatform
from test_chrome_extension import FakeBridge, FakeExtensionLauncher, _config

WORKER = Path(__file__).parents[1] / "src/android_ui_analyser/chrome_extension/worker.js"


def _worker(script: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    harness = """
    let eventListener;
    const messages = [];
    globalThis.chrome = {
      debugger: {
        onEvent: {addListener(fn) { eventListener = fn; }},
        onDetach: {addListener() {}},
        async detach() {}
      },
      tabs: {onRemoved: {addListener() {}}},
      runtime: {onMessage: {addListener() {}}}
    };
    """
    source = (
        harness
        + WORKER.read_text(encoding="utf-8")
        + """
    attachedTabId = 42;
    nativePort = {postMessage(message) { messages.push(message); }};
    function emit(method, params, tabId = 42) {
      eventListener({tabId}, method, params);
    }
    (async () => {
    """
        + script
        + """
      console.log(JSON.stringify({messages, pending: pendingRequests.size}));
    })().catch(error => { console.error(error); process.exitCode = 1; });
    """
    )
    result = subprocess.run(
        [node, "-"], input=source, capture_output=True, text=True, check=True, timeout=10
    )
    return json.loads(result.stdout)


def _server(tmp_path: Path, messages: list[dict]) -> ChromeBridgeServer:
    server = ChromeBridgeServer(
        socket_path=tmp_path / "unused.sock",
        config_path=tmp_path / "unused.json",
        attach_timeout_ms=100,
    )
    payload = "".join(json.dumps(message) + "\n" for message in messages)
    server._read_messages(io.BytesIO(payload.encode()))
    return server


def test_extension_network_events_are_structured_private_and_inline(tmp_path: Path) -> None:
    result = _worker("""
      const url = "https://user:secret@api.example.test/data?token=secret#private";
      emit("Network.requestWillBeSent", {
        requestId: "one", type: "Fetch",
        request: {url, method: "POST", headers: {Authorization: "secret"}, postData: "secret"}
      });
      emit("Network.responseReceived", {
        requestId: "one", type: "Fetch", response: {url, status: 503, headers: {"Set-Cookie": "secret"}}
      });
      emit("Network.loadingFailed", {requestId: "one", type: "Fetch", errorText: "net::ERR_FAILED"});
      emit("Network.webSocketCreated", {url: "wss://user:secret@api.example.test/socket?token=secret"});
      emit("Runtime.consoleAPICalled", {type: "log", args: [{value: "hello"}]});
      emit("Network.requestWillBeSent", {requestId: "foreign", request: {url, method: "GET"}}, 99);
    """)
    assert result["pending"] == 0
    assert "secret" not in json.dumps(result)
    server = _server(tmp_path, result["messages"])
    bridge = FakeBridge()
    bridge.events = server.event_snapshot()
    connection = ChromeExtensionConnection(ChromeAttachOptions(), bridge=bridge)
    platform = WebPlatform(_config(tmp_path), extension_launcher=FakeExtensionLauncher(connection))
    platform.options = platform.validate_options(platform.config.platform_options("web"))
    engine = Engine(platform.config, platform=platform)
    try:
        diagnostic = engine.analyze(source="hierarchy", with_ocr=False).meta.browser_diagnostics
        assert diagnostic is not None and diagnostic["count"] == 5
        request, response, failed, websocket, console = diagnostic["events"]
        for event in (request, response, failed):
            assert event["url"] == "https://api.example.test/data"
            assert event["method"] == "POST"
            assert event["resource_type"] == "fetch"
            assert "headers" not in event and "postData" not in event
        assert request["message"] == "POST https://api.example.test/data"
        assert response["status"] == 503 and response["level"] == "error"
        assert response["message"] == "503 https://api.example.test/data"
        assert failed["kind"] == "request_failed"
        assert websocket["url"] == "wss://api.example.test/socket"
        assert console["message"] == "hello"
    finally:
        engine.close()


def test_request_correlation_is_bounded_and_reset_when_attachment_ends() -> None:
    result = _worker("""
      for (let i = 0; i < 2001; i++) {
        emit("Network.requestWillBeSent", {
          requestId: String(i), type: "XHR", request: {url: "https://example.test/data", method: "GET"}
        });
      }
      if (pendingRequests.size !== 2000) throw new Error("unbounded request tracking");
      emit("Network.loadingFinished", {requestId: "2000"});
      if (pendingRequests.size !== 1999) throw new Error("completed request retained");
      await detachCurrent();
      emit("Network.requestWillBeSent", {requestId: "later", request: {url: "https://example.test/ignored"}});
    """)
    assert result["pending"] == 0
    assert len(result["messages"]) == 2001


def test_bridge_redacts_older_worker_urls_and_reports_scoped_overflow(tmp_path: Path) -> None:
    url = "https://user:secret@[::1]:8000/data?token=secret#private"
    messages = [
        {
            "type": "event",
            "event": {
                "kind": "response",
                "level": "warning",
                "message": f"401 {url}",
                "url": url,
                "timestamp_ms": index,
            },
        }
        for index in range(1, 2002)
    ]
    server = _server(tmp_path, messages)
    connection = ChromeExtensionConnection(ChromeAttachOptions(), bridge=server)
    diagnostic = connection.diagnostics(limit=10, kinds=(), since_ms=1)
    assert diagnostic["total_count"] == 2000 and diagnostic["count"] == 10
    assert diagnostic["buffer_overflow"] and diagnostic["truncated"]
    assert diagnostic["events"][-1]["url"] == "https://[::1]:8000/data"
    assert diagnostic["events"][-1]["message"] == "401 https://[::1]:8000/data"
    assert "secret" not in json.dumps(diagnostic)
    assert not connection.diagnostics(limit=10, kinds=(), since_ms=2)["buffer_overflow"]
    assert connection.diagnostics_clear()["cleared"] == 2000
    assert not connection.diagnostics(limit=10, kinds=(), since_ms=None)["buffer_overflow"]


@pytest.mark.parametrize(
    ("kind", "prefix"), [("request", "GET"), ("response", "401"), ("websocket", "opened")]
)
def test_bridge_redacts_truncated_legacy_network_messages(
    tmp_path: Path, kind: str, prefix: str
) -> None:
    url = "https://user:secret@example.test/data?token=secret&padding=" + "x" * 3000
    message = {
        "type": "event",
        "event": {"kind": kind, "url": url, "message": f"{prefix} {url}"[:2000]},
    }
    events = _server(tmp_path, [message]).event_snapshot()
    assert "secret" not in json.dumps(events)
    assert events[0]["url"] == "https://example.test/data"
