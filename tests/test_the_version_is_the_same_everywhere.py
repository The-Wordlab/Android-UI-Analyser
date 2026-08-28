"""A release cannot have different answers to "which version is this?".

The CLI, Python package, Claude/Codex plugins, marketplace, uvx MCP source, and README install
commands travel through different paths. A stale plugin manifest or copy-paste command fails
silently: an update may never appear or may start an older server. Name every disagreement so a
maintainer fixes the source instead of the consumer.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from android_ui_analyser import __version__

REPO = Path(__file__).resolve().parents[1]
SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def _published_versions() -> dict[str, str]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    claude_plugin = json.loads(
        (REPO / ".claude-plugin/plugin.json").read_text(encoding="utf-8")
    )
    codex_plugin = json.loads(
        (REPO / ".codex-plugin/plugin.json").read_text(encoding="utf-8")
    )
    marketplace = json.loads(
        (REPO / ".claude-plugin/marketplace.json").read_text(encoding="utf-8")
    )
    mcp = json.loads((REPO / ".mcp.json").read_text(encoding="utf-8"))
    source = mcp["mcpServers"]["android-ui-analyser"]["args"][2]
    tag = re.search(r"@v([^@\s]+)$", source)
    return {
        "src/android_ui_analyser/__init__.py": __version__,
        "pyproject.toml": project["project"]["version"],
        ".claude-plugin/plugin.json": claude_plugin["version"],
        ".claude-plugin/marketplace.json": marketplace["plugins"][0]["version"],
        ".codex-plugin/plugin.json": codex_plugin["version"],
        ".mcp.json pinned git tag": tag.group(1) if tag else "<no vX.Y.Z source tag>",
    }


def test_the_version_is_the_same_everywhere() -> None:
    versions = _published_versions()
    expected = versions["pyproject.toml"]
    disagree = {path: version for path, version in versions.items() if version != expected}
    details = ", ".join(f"{path} says {version}" for path, version in disagree.items())
    assert not disagree, f"pyproject.toml says {expected}; {details}"


def test_the_published_version_is_semver() -> None:
    version = _published_versions()["pyproject.toml"]
    assert SEMVER.fullmatch(version), f"pyproject.toml version {version!r} is not valid SemVer"


def test_readme_install_examples_pin_the_published_release() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    pins = re.findall(
        r"Android-UI-Analyser\.git@v([^'\"\s]+)", readme
    ) + re.findall(r"^git checkout v([^\s]+)$", readme, re.MULTILINE)
    expected = _published_versions()["pyproject.toml"]

    assert pins, "README.md must include a pinned release installation example"
    assert set(pins) == {expected}, (
        f"README.md install examples must pin v{expected}; found "
        f"{', '.join(f'v{version}' for version in sorted(set(pins)))}"
    )


def test_both_plugins_share_the_pinned_uvx_mcp_server() -> None:
    claude_plugin = json.loads(
        (REPO / ".claude-plugin/plugin.json").read_text(encoding="utf-8")
    )
    codex_plugin = json.loads(
        (REPO / ".codex-plugin/plugin.json").read_text(encoding="utf-8")
    )
    mcp = json.loads((REPO / ".mcp.json").read_text(encoding="utf-8"))
    server = mcp["mcpServers"]["android-ui-analyser"]

    assert claude_plugin["name"] == codex_plugin["name"] == "android-ui-analyser"
    assert codex_plugin["skills"] == "./skills/"
    assert codex_plugin["mcpServers"] == "./.mcp.json"
    assert server["command"] == "uvx"
    assert server["args"][-2:] == ["aua", "mcp"]
    assert "@v" in server["args"][2], "the plugin must never follow moving main"
