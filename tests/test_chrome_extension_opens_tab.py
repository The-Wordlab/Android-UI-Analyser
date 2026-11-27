"""AUA can open its own tab in the user's Chrome; it still never drives a tab it did not open."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.parse
import uuid
from pathlib import Path

import pytest

from android_ui_analyser.chrome_extension_setup import EXTENSION_ID
from android_ui_analyser.errors import DeviceError
from android_ui_analyser.platforms.chrome_extension import ChromeBridgeServer
from android_ui_analyser.platforms.registry import PlatformFactory
from test_chrome_extension import _config

WORKER = Path(__file__).parents[1] / "src/android_ui_analyser/chrome_extension/worker.js"

HARNESS = """
const calls = [];
let onMessage, onUpdated;
globalThis.chrome = {
  debugger: {
    onEvent: {addListener() {}},
    onDetach: {addListener() {}},
    async attach(target) { calls.push(["attach", target.tabId]); },
    async detach() {},
    async sendCommand(_target, method, params) {
      if (method !== "Runtime.evaluate") return {};
      const value = params.expression === "location.protocol" ? "https:" :
        {width: 1280, height: 800, title: "Example", url: "https://example.com/"};
      return {result: {value}};
    }
  },
  tabs: {
    onRemoved: {addListener() {}},
    onUpdated: {addListener(fn) { onUpdated = fn; }, removeListener() {}},
    async update(tabId, props) {
      calls.push(["update", tabId, props.url]);
      setTimeout(() => { onUpdated(tabId, {status: "loading"}); onUpdated(tabId, {status: "complete"}); });
    },
    async get(tabId) { return {id: tabId}; },
    async query() { return []; }
  },
  action: {setIcon: async () => {}, setTitle: async () => {}},
  runtime: {
    id: "aua",
    getURL: path => "chrome-extension://aua/" + path,
    getManifest: () => ({version: "1"}),
    onMessage: {addListener(fn) { onMessage = fn; }},
    connectNative() {
      const port = {
        onMessage: {addListener(fn) { setTimeout(() => fn({type: "host_ready", protocol: 1})); }},
        onDisconnect: {addListener() {}},
        postMessage(message) { calls.push(["native", message.type]); }
      };
      return port;
    }
  }
};
"""


def _send(message: dict, sender: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    source = HARNESS + WORKER.read_text(encoding="utf-8") + f"""
    onMessage({json.dumps(message)}, {json.dumps(sender)}, response => {{
      console.log(JSON.stringify({{response, calls}}));
    }});
    """
    result = subprocess.run(
        [node, "-"], input=source, capture_output=True, text=True, check=True, timeout=10
    )
    return json.loads(result.stdout)


def test_the_extension_opens_the_url_in_its_own_tab_and_attaches_it() -> None:
    sender = {"id": "aua", "tab": {"id": 7}, "url": "chrome-extension://aua/open.html#x"}

    result = _send({"action": "open", "url": "https://example.com/"}, sender)

    assert result["response"]["attached"] is True
    assert ["update", 7, "https://example.com/"] in result["calls"]
    assert ["attach", 7] in result["calls"]
    assert ["native", "hello"] in result["calls"]


@pytest.mark.parametrize(
    ("url", "sender_url"),
    [
        ("https://example.com/", "chrome-extension://aua/popup.html"),
        ("file:///etc/passwd", "chrome-extension://aua/open.html#x"),
    ],
)
def test_only_the_open_page_can_open_and_only_http_pages(url: str, sender_url: str) -> None:
    sender = {"id": "aua", "tab": {"id": 7}, "url": sender_url}

    result = _send({"action": "open", "url": url}, sender)

    assert "error" in result["response"]
    assert not any(call[0] == "attach" for call in result["calls"])


def test_aua_asks_chrome_to_open_the_configured_url(tmp_path, monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        "android_ui_analyser.platforms.chrome_extension.subprocess.run",
        lambda command, **_kwargs: commands.append(command),
    )
    server = ChromeBridgeServer(
        socket_path=Path("/tmp") / f"aua-open-test-{uuid.uuid4().hex[:8]}.sock",
        config_path=tmp_path / "bridge.json",
        attach_timeout_ms=100,
        open_url="https://example.com/a?b=1",
    )

    with pytest.raises(DeviceError, match="timed out") as error:
        server.start()

    page = commands[0][-1]
    assert page.startswith(f"chrome-extension://{EXTENSION_ID}/open.html#")
    assert urllib.parse.unquote(page.split("#", 1)[1]) == "https://example.com/a?b=1"
    assert "https://example.com/a?b=1" in str(error.value.hint)


def test_attached_mode_accepts_a_url_to_open(tmp_path) -> None:
    platform = PlatformFactory(_config(tmp_path, url="https://example.com/")).create("web")

    assert platform._attach_options().open_url == "https://example.com/"


OPEN_PAGE = Path(__file__).parents[1] / "src/android_ui_analyser/chrome_extension/open.js"


@pytest.mark.parametrize(("answer", "reloads"), [("Unknown popup action", True), ("", False)])
def test_the_open_page_reloads_a_copy_too_old_to_open_tabs(answer: str, reloads: bool) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    source = f"""
    const calls = [];
    globalThis.location = {{hash: "#https%3A%2F%2Fexample.com%2F"}};
    globalThis.document = {{querySelector: () => ({{}})}};
    globalThis.chrome = {{runtime: {{
      reload: () => calls.push("reload"),
      sendMessage: async message => {{
        calls.push(message.action);
        return {json.dumps({"error": answer} if answer else {"attached": True})};
      }}
    }}}};
    """ + OPEN_PAGE.read_text(encoding="utf-8") + """
    setTimeout(() => console.log(JSON.stringify(calls)));
    """
    result = subprocess.run(
        [node, "-"], input=source, capture_output=True, text=True, check=True, timeout=10
    )

    assert json.loads(result.stdout) == (["open", "reload"] if reloads else ["open"])
