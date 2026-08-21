"""The dashboard page's inline script must actually parse.

A single unterminated string literal takes the whole script down, and because both
`#grid-view` and `#detail-view` start hidden and are only revealed by that script, the
result is a completely blank page under a working header — no error, no partial render,
nothing. Shipped exactly that: a `\\n` written into the non-raw Python string holding the
page became a real newline inside a JS string literal.

Substring assertions cannot see this (every expected snippet is still present) and neither
can an HTTP test (the server returns 200 with the broken page). Parsing is what catches it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from android_ui_analyser import dashboard as dash

_SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.S)


def _inline_scripts() -> list[str]:
    html = dash._DASHBOARD_HTML.replace("__POLL_MS__", "500")
    html = html.replace("__MODE_JSON__", '"grid"').replace("__SERIAL_JSON__", '""')
    html = html.replace("__PHONE_ACCESS_URL_JSON__", '""')
    html = html.replace("__DATABASE_TOKEN__", "test-token")
    return [body for body in _SCRIPT.findall(html) if body.strip()]


def test_the_page_has_an_inline_script_to_check() -> None:
    scripts = _inline_scripts()
    assert scripts, "no inline script found; this guard would silently pass forever"


def test_no_javascript_string_literal_is_broken_by_a_real_newline() -> None:
    """Runs without node, so the guard still bites in a bare CI image.

    Scans with a real (small) state machine rather than counting quotes: a line like
    `const q = \'"\' + name.replaceAll(\'"\', \'""\')` has an odd number of double quotes and
    is perfectly valid, so counting alone reports failures that are not there.
    """
    for body in _inline_scripts():
        quote: str | None = None       # which quote opened the string we are inside
        opened_on = 0
        line = 1
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "\n":
                assert quote is None, (
                    f"line {opened_on}: {quote} string is still open at the end of the line "
                    f"— a literal newline inside a JS string kills the whole script"
                )
                line += 1
                i += 1
                continue
            if quote is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                i += 1
                continue
            # outside a string
            if ch in "\'\"":
                quote = ch
                opened_on = line
                i += 1
                continue
            if ch == "`":  # template literals may legitimately span lines
                i += 1
                while i < len(body) and body[i] != "`":
                    if body[i] == "\\":
                        i += 1
                    elif body[i] == "\n":
                        line += 1
                    i += 1
                i += 1
                continue
            if body.startswith("//", i):
                while i < len(body) and body[i] != "\n":
                    i += 1
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i)
                end = len(body) if end < 0 else end + 2
                line += body.count("\n", i, end)
                i = end
                continue
            i += 1
        assert quote is None, f"line {opened_on}: unterminated {quote} string at end of script"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_inline_script_parses_as_javascript() -> None:
    for body in _inline_scripts():
        result = subprocess.run(  # noqa: S603
            [shutil.which("node") or "node", "--check", "-"],
            input=body,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"dashboard script does not parse:\n{result.stderr}"
