"""Capture state lives ONLY in the warm daemon, so its CLI commands must route there.

Regression guard for a bug that made the whole feature unreachable: the daemon dispatched
``capture_*`` correctly and wrote frames to disk, but every CLI command built a fresh
in-process engine — a process with no buffer — so ``capture status`` reported
``running: false`` and ``capture last`` raised "not running" while frames were piling up
next to them. Unlike a stateless call (``tap``, ``analyze``), an in-process fallback cannot
produce the right answer here: it is not slower, it is wrong.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI = Path(__file__).resolve().parent.parent / "src" / "android_ui_analyser" / "cli.py"

CAPTURE_COMMANDS = {
    "capture_status_cmd",
    "capture_last_cmd",
    "capture_export_cmd",
    "capture_sheet_cmd",
    "capture_explain_cmd",
    "capture_on_cmd",
    "capture_off_cmd",
    "capture_prune_cmd",
}


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in cli.py — was the command renamed?")


def test_every_capture_command_routes_through_the_daemon() -> None:
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for name in sorted(CAPTURE_COMMANDS):
        src = ast.get_source_segment(CLI.read_text(encoding="utf-8"), _function(tree, name)) or ""
        routed = "_route(" in src
        direct = "engine.capture_" in src
        if direct or not routed:
            offenders.append(f"{name}: routed={routed} direct_engine_call={direct}")
    assert not offenders, "capture commands must call _route(engine, 'capture_…'): " + "; ".join(
        offenders
    )


def test_daemon_dispatches_every_capture_command() -> None:
    """The other half: routing is useless if the daemon has no branch to answer it."""
    daemon_src = (CLI.parent / "daemon.py").read_text(encoding="utf-8")
    for cmd in (
        "capture_status",
        "capture_last",
        "capture_export",
        "capture_sheet",
        "capture_explain",
        "capture_on",
        "capture_off",
        "capture_prune",
    ):
        assert f'"{cmd}"' in daemon_src, f"daemon.py has no dispatch branch for {cmd}"


def test_sidecar_is_local_not_daemon_routed() -> None:
    """Sidecar starts a host process — it must talk to the local engine, not the daemon buffer."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    src = ast.get_source_segment(CLI.read_text(encoding="utf-8"), _function(tree, "capture_sidecar_cmd")) or ""
    assert "engine.capture_sidecar_" in src
    assert "_route(" not in src
