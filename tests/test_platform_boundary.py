"""Architecture guard: reusable layers may not grow a native Android escape hatch."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "android_ui_analyser"
GENERIC_MODULES = (
    "engine.py",
    "cli.py",
    "mcp_server.py",
    "dashboard.py",
    "capture_sidecar.py",
    "daemon.py",
    # The undo ledger and its reaper describe device changes as data and replay them through the
    # adapter. Keeping them here is what lets an iOS or web plugin inherit the whole cleanup
    # guarantee without an Android dependency — the undo would otherwise be the one place a
    # native call quietly crept back into the reusable layer.
    "device_ledger.py",
    "teardown.py",
    "teardown_watchdog.py",
)
ANDROID_SERVICE_MODULES = {
    "app_database",
    "device_agent",
    "devopts",
    "emulator",
    "flags",
    "mic",
    "network",
    "network_profiles",
    "proxy_mock",
    "webview",
}
NATIVE_COMMAND_PREFIXES = ("adb", "dumpsys", "run-as", "settings ", "svc ")
ANDROID_BACKENDS = {
    "app_database.py",
    "device.py",
    "device_agent.py",
    "devopts.py",
    "emulator.py",
    "flags.py",
    "mic.py",
    "network.py",
    "network_profiles.py",
    "platforms/android.py",
    "platforms/android_runtime.py",
    "platforms/android_transport.py",
    "proxy_mock.py",
    "webview.py",
}


def _relative_import_leaf(node: ast.ImportFrom) -> str:
    return (node.module or "").rsplit(".", 1)[-1]


def test_generic_layers_reach_android_services_only_through_platform_gate() -> None:
    violations: list[str] = []
    for filename in GENERIC_MODULES:
        path = ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                native = {
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in {"adbutils", "uiautomator2"}
                }
                if native:
                    violations.append(f"{filename}:{node.lineno} imports {sorted(native)}")
            elif isinstance(node, ast.ImportFrom):
                leaf = _relative_import_leaf(node)
                imported = {alias.name for alias in node.names}
                error_types = {
                    "MicDeliveredReleaseError",
                    "MicDeliveryUncertainError",
                    "MicToggleStartUncertainError",
                    "MicToggleStopUncertainError",
                }
                pure_error_import = (
                    filename == "cli.py" and leaf == "mic" and imported <= error_types
                )
                if (leaf in ANDROID_SERVICE_MODULES and not pure_error_import) or leaf == "android":
                    violations.append(f"{filename}:{node.lineno} imports {node.module}")
                if leaf == "device":
                    direct = imported & {"connect", "list_devices", "Uiautomator2Device"}
                    # Engine's two names are a marked downstream-test monkeypatch shim. Production
                    # calls still route through PlatformAdapter; do not extend this allow-list.
                    allowed = {"connect", "list_devices"} if filename == "engine.py" else set()
                    if direct - allowed:
                        violations.append(
                            f"{filename}:{node.lineno} imports device symbols {sorted(direct)}"
                        )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"adb_reverse", "adb_reverse_remove"}:
                    violations.append(f"{filename}:{node.lineno} calls {node.func.attr}")
                if (
                    node.func.attr == "shell"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    command = str(node.args[0].value).strip().lower()
                    if command.startswith(NATIVE_COMMAND_PREFIXES):
                        violations.append(f"{filename}:{node.lineno} executes {command!r}")
            elif isinstance(node, (ast.List, ast.Tuple)) and node.elts:
                first = node.elts[0]
                if isinstance(first, ast.Constant) and str(first.value).lower() == "adb":
                    violations.append(f"{filename}:{node.lineno} constructs an adb command")

    assert violations == []


def test_native_android_dependencies_are_confined_to_android_backends() -> None:
    violations: list[str] = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative in ANDROID_BACKENDS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in {"adbutils", "uiautomator2"}:
                        violations.append(f"{relative}:{node.lineno} imports {alias.name}")
            elif isinstance(node, (ast.List, ast.Tuple)) and node.elts:
                first = node.elts[0]
                if isinstance(first, ast.Constant) and str(first.value).lower() == "adb":
                    violations.append(f"{relative}:{node.lineno} constructs an adb command")

    assert violations == []
