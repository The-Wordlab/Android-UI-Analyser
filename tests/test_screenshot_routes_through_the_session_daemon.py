"""Screenshots share the session daemon's device fence instead of opening a second device user.

Measured 2026-08-20: a legal-page screenshot waited silently for 70 seconds after the preceding
daemon-routed point action.  The CLI called ``engine.screenshot`` directly, so its short-lived
Engine competed with the warm session daemon for the same target.  Cropping had the same direct
``engine.device`` path.  The daemon already implements ``screenshot``; every CLI variant must use
it, then any crop/scale is a host-only transform of the written PNG.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI = Path(__file__).resolve().parent.parent / "src" / "android_ui_analyser" / "cli.py"


def test_every_screenshot_variant_routes_and_never_opens_the_cli_device() -> None:
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    command = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "screenshot"
    )
    command_source = ast.get_source_segment(source, command) or ""

    assert '_route(engine, "screenshot"' in command_source
    assert "engine.screenshot(" not in command_source
    assert "engine.device" not in command_source
