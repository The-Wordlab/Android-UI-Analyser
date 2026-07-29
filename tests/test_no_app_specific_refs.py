"""Guard: no specific app may be named in tracked files — this repo is PUBLIC.

`aua` is app-agnostic. Examples must use obviously fictional placeholders, and real per-app
knowledge (package ids, private debug deeplink schemes) belongs in the user's own config or
in the local playbook — never committed here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The last character is bracketed so this file never matches its own pattern: a plain
# `git grep` for any banned term must come back empty across the whole tree, this guard
# included.
_BANNED = (r"luzi[a]", r"thewordla[b]", r"theworldla[b]")

# Repo-relative path -> why it may still name an app. Expected to stay empty; entries are
# self-expiring (see test_allowlist_has_no_stale_entries), so an exemption cannot outlive
# its reason.
_ALLOWLIST: dict[str, str] = {
    "src/android_ui_analyser/flags.py": (
        "being rewritten in a parallel change that drops its hardcoded package->scheme map"
    ),
}

_WHY = """
`android-ui-analyser` is a PUBLIC repo and an app-agnostic tool: it must not name a
specific app, its package ids, or its private debug deeplink scheme.

Use the fictional placeholders instead:
  package   com.example.app   (com.example.app.dev for a dev build)
  scheme    myapp://          (e.g. myapp://set-flags?flag=value, myapp://home)

Targeting a real app is still fully supported — that knowledge belongs in the USER's
config (`flags.templates`) or the local playbook `aua remember` builds, not in this repo.
"""


def _pattern() -> re.Pattern[str]:
    return re.compile("|".join(_BANNED), re.IGNORECASE)


def _tracked_files() -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [path for path in out.split("\0") if path]


def _references_in(rel: str, pattern: re.Pattern[str]) -> list[str]:
    try:
        text = (REPO / rel).read_bytes().decode("utf-8", errors="ignore")
    except OSError:  # tracked but not on disk (staged deletion), or unreadable
        return []
    return [
        f"{rel}:{lineno}: {line.strip()[:120]}"
        for lineno, line in enumerate(text.splitlines(), 1)
        if pattern.search(line)
    ]


def test_no_app_specific_references() -> None:
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout (or git unavailable) — nothing to enumerate")

    pattern = _pattern()
    this_file = Path(__file__).resolve().relative_to(REPO).as_posix()
    hits: list[str] = []
    for rel in tracked:
        if rel == this_file or rel in _ALLOWLIST:
            continue
        hits += _references_in(rel, pattern)

    assert not hits, f"{_WHY}\nFound {len(hits)} reference(s):\n" + "\n".join(hits)


def test_allowlist_has_no_stale_entries() -> None:
    """A clean allowlisted file must lose its entry, so no exemption is granted silently."""
    pattern = _pattern()
    stale = [rel for rel in _ALLOWLIST if not _references_in(rel, pattern)]
    assert not stale, (
        "These files no longer name an app, so their _ALLOWLIST entries are obsolete — "
        f"delete them from {Path(__file__).name}: " + ", ".join(stale)
    )
