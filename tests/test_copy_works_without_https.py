"""The dashboard's copy buttons must work over plain http.

``aua dashboard`` is served at ``http://aua.local/`` by default. That is not a browser
"secure context" (only https, ``localhost`` and ``127.0.0.1`` are), so
``navigator.clipboard`` is not merely restricted there — it does not exist at all. A
copy path that only knows about ``navigator.clipboard`` therefore reports "copy failed"
on every click of the default deployment, on the host and on a phone alike.

Asserting on the page source cannot prove the fallback actually copies, so the guard
runs the real function in node with a clipboard-less ``navigator``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from android_ui_analyser import dashboard as dash

_HARNESS = """
%(source)s

globalThis.navigator = {};   // plain http: no clipboard API whatsoever
let written = null;
const area = {
  value: '',
  style: {},
  setAttribute() {},
  focus() {},
  select() {},
  setSelectionRange() {},
  remove() {},
};
globalThis.document = {
  createElement() { return area; },
  body: { appendChild() {}, removeChild() {} },
  execCommand(command) {
    if (command !== 'copy') return false;
    written = area.value;
    return true;
  },
};

Promise.resolve(copyText(%(payload)s)).then(ok => {
  console.log(JSON.stringify({ ok: ok, written: written }));
}, err => {
  console.log(JSON.stringify({ ok: 'threw', written: String(err) }));
});
"""


def _function_source(name: str) -> str:
    """The named JS function, brace-matched out of the page so node can run it."""
    start = dash._DASHBOARD_HTML.find("function " + name + "(")
    assert start != -1, f"the page no longer defines {name}()"
    depth = 0
    for index in range(dash._DASHBOARD_HTML.find("{", start), len(dash._DASHBOARD_HTML)):
        char = dash._DASHBOARD_HTML[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return dash._DASHBOARD_HTML[start : index + 1]
    raise AssertionError(f"unbalanced braces in {name}()")


def test_the_copy_path_has_a_non_secure_context_fallback() -> None:
    """Runs without node, so a bare CI image still fails on a clipboard-only copy."""
    source = _function_source("copyText")
    helpers = re.findall(r"\b([A-Za-z_$][\w$]*)\(", source)
    reachable = source + "".join(
        _function_source(helper)
        for helper in dict.fromkeys(helpers)
        if helper not in {"copyText", "then", "resolve", "catch"}
        and ("function " + helper + "(") in dash._DASHBOARD_HTML
    )
    assert "execCommand" in reachable, (
        "copyText() only knows navigator.clipboard, which does not exist on "
        "http://aua.local/ — every copy button reports 'copy failed' there"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_copy_text_copies_when_navigator_clipboard_is_absent() -> None:
    source = _function_source("copyText")
    helpers = re.findall(r"\b([A-Za-z_$][\w$]*)\(", source)
    for helper in dict.fromkeys(helpers):
        if helper == "copyText":
            continue
        if ("function " + helper + "(") in dash._DASHBOARD_HTML:
            source += "\n" + _function_source(helper)
    payload = "session log line: ok=false"
    script = _HARNESS % {"source": source, "payload": json.dumps(payload)}
    proc = subprocess.run(
        [shutil.which("node") or "node", "-"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"] is True, f"copyText() reported failure over plain http: {result}"
    assert result["written"] == payload, f"nothing reached the clipboard: {result}"
