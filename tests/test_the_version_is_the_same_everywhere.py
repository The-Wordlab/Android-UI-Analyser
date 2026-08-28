"""A release cannot have four different answers to "which version is this?".

The CLI, Python package, Claude plugin, and marketplace are installed through different paths.
A stale plugin manifest fails silently: `/plugin update` simply never offers the release. Name
every disagreement here so a maintainer fixes the source instead of debugging the consumer.
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
    plugin = json.loads((REPO / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    marketplace = json.loads(
        (REPO / ".claude-plugin/marketplace.json").read_text(encoding="utf-8")
    )
    return {
        "src/android_ui_analyser/__init__.py": __version__,
        "pyproject.toml": project["project"]["version"],
        ".claude-plugin/plugin.json": plugin["version"],
        ".claude-plugin/marketplace.json": marketplace["plugins"][0]["version"],
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
