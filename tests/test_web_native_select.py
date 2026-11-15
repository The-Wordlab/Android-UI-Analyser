"""Native choice selection must be targeted and dispatch normal frontend events.

Unlike editable text, a native select ignores keyboard.insert_text. A global key
sequence can also lose focus when action observations run. This regression uses
the real AUA engine and isolated browser, with no external app or account.
"""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import DeviceError

pytest.importorskip("playwright.sync_api")


@pytest.mark.parametrize("choice", [{"value": "two"}, {"label": "Second"}])
def test_native_select_by_stable_id_updates_choice_and_frontend_state(tmp_path: Path, choice):
    (tmp_path / "index.html").write_text('''<!doctype html>
<label for="choice">Choice</label>
<select id="choice"><option value="">Choose</option>
<option value="one">First</option><option value="two">Second</option></select>
<label for="other">Other choice</label>
<select id="other"><option>Untouched</option><option>Second</option></select>
<p id="result">No change</p>
<script>
let inputCount = 0;
choice.addEventListener('input', () => inputCount++);
choice.addEventListener('change', () => {
  const value = choice.value;
  // Rerender from frontend state: a DOM-only value assignment is insufficient.
  choice.outerHTML = `<select id="choice"><option value="">Choose</option>
    <option value="two" selected>Second</option></select>`;
  result.textContent = `Selected ${value}; input events ${inputCount}`;
});
</script>''')
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = Config.model_validate({
        "device": {"platform": "web"},
        "platforms": {"web": {"url": f"http://127.0.0.1:{server.server_port}/index.html"}},
        "cache": {"dir": str(tmp_path / "cache")},
        "lease": {"registry_dir": str(tmp_path / "leases")},
        "memory": {"enabled": False}, "capture": {"enabled": False},
        "teardown": {"enabled": False}, "ocr": {"enabled": False},
        "perf": {"auto_daemon": False, "prefetch": False, "predictive_prefetch": False,
                 "stable_delay_ms": {"default": 0, "tap": 0}},
    })
    engine = Engine(config)
    started = False
    try:
        try:
            engine.session_start("Verify native choice selection", provision_target=False)
            started = True
        except DeviceError as exc:
            if "Executable doesn't exist" in str(exc) or exc.code == "web_driver_missing":
                pytest.skip(f"Playwright Chromium unavailable: {exc}")
            raise
        before = engine.analyze(source="hierarchy", with_ocr=False)
        target = next(el for el in before.elements if el.resource_id == "choice")
        assert target.handle is not None
        assert target.handle.startswith("el:")
        action = engine.select_option(target.handle, **choice)
        assert action.ok is True
        assert action.observation is not None
        observed = {el.resource_id: el for el in action.observation.elements if el.resource_id}
        assert observed["choice"].text == "Second"
        assert observed["choice"].handle == target.handle
        assert observed["other"].text == "Untouched"
        assert observed["result"].text == "Selected two; input events 1"
    finally:
        if started:
            engine.session_finish(allow_incomplete=True)
        engine.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
