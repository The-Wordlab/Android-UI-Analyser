"""A tag without release notes recreates the exact problem releases are meant to solve.

The release workflow lifts its GitHub Release body from the current version's changelog section.
If the section is absent or empty, users get a version number with no explanation of what changed.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from android_ui_analyser import __version__

REPO = Path(__file__).resolve().parents[1]


def _changelog() -> str:
    path = REPO / "CHANGELOG.md"
    assert path.is_file(), "CHANGELOG.md is missing; the release workflow has no notes to publish"
    return path.read_text(encoding="utf-8")


def test_the_current_release_has_non_empty_notes() -> None:
    changelog = _changelog()
    match = re.search(
        rf"^## \[{re.escape(__version__)}\] - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s*\n"
        rf"(?P<body>.*?)(?=^## \[|^\[[^]]+\]:|\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, (
        f"CHANGELOG.md has no '## [{__version__}] - YYYY-MM-DD' section; "
        "run scripts/bump-version.sh before tagging"
    )
    date.fromisoformat(match.group("date"))
    assert match.group("body").strip(), (
        f"CHANGELOG.md section {__version__} is empty; describe what users receive"
    )


def test_the_next_release_has_an_unreleased_section() -> None:
    assert re.search(r"^## \[Unreleased\]\s*$", _changelog(), re.MULTILINE), (
        "CHANGELOG.md has no ## [Unreleased] heading for the next release"
    )


def test_the_current_release_link_points_at_this_repository() -> None:
    expected = (
        f"[{__version__}]: https://github.com/The-Wordlab/Android-UI-Analyser/"
        f"releases/tag/v{__version__}"
    )
    assert expected in _changelog(), (
        f"CHANGELOG.md must link {__version__} to this repository's GitHub Release: {expected}"
    )
