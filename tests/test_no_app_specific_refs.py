"""Guard: no specific app may be named in tracked files — this repo is PUBLIC.

`aua` is app-agnostic. Examples must use obviously fictional placeholders, and real per-app
knowledge (package ids, private debug deeplink schemes) belongs in the user's own config or
in the local playbook — never committed here.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# One-way fingerprints keep the deny-list useful without publishing the private names and
# selectors it protects. Each pair is ``(case-folded character count, sha256)``. The scanner
# checks token substrings, so a forbidden company or app name is still caught inside a package,
# URL, scheme, or camelCase selector.
_BANNED_FINGERPRINTS = {
    (5, "d54e4818de1e48e28cc125c048d4b06dbc40adea1dae4650ed974618bbf6ccb7"),
    (7, "7e72b4554a267071e253746b1ca258eeca155433fccda474ed9ee379914a9d75"),
    (8, "6645e460b204a6c7db03e3cdd3cb49630ffa8f00aee2bea0d964ebe3ed038a25"),
    (10, "696bd1d0c0c80724544014b7dbde1e53120e756363d927aa220659512bda9872"),
    (10, "f35b26bcea2fe57c42445a1880712b0a803d66aa5b47c806d5971302d269d46e"),
    (11, "2eaf666c5a63366ea559ee11728c3405befad80ad3120a86803f0ceb683be17b"),
    (16, "e7d691c42dfc5890abfe3957e9a64831b41b5874d21fba706d18c805916115b1"),
    (17, "8dc188f9feb1a90361d19e1f330b550dc1e542a60cf4cc511f792bdf7077c942"),
    (18, "330cc94b8228255b9ac9f9f978f1c2ba8d003db2e3042e3825f88917aad56e43"),
    (19, "109f617282c5b653ebba9a1d62db225e160a9fb10cb8535755614bcc0cbd2b26"),
    (36, "3e9b41124244962294a4ab1ff610c716aa39e74b00cc26caeb90ab98b0f6cf1a"),
}

_TOKEN = re.compile(r"[A-Za-z0-9_@.:/+%-]+")

# Repo-relative path -> why it may still name an app. Expected to stay empty; entries are
# self-expiring (see test_allowlist_has_no_stale_entries), so an exemption cannot outlive
# its reason.
_ALLOWLIST: dict[str, str] = {}

_WHY = """
`android-ui-analyser` is a PUBLIC repo and an app-agnostic tool: it must not name a
specific app — not its package ids, not its private debug deeplink scheme, and not
resource-ids or flag keys copied from its UI (those leak unreleased product structure).

Use fictional placeholders instead:
  package     com.example.app    (com.example.app.dev for a dev build)
  scheme      myapp://           (e.g. myapp://set-flags?flag=value, myapp://home)
  tab         homeTabBROWSE      notification entry   notificationsButton
  container   containerDetail    switch/toggle        settingsSwitch

Invented camelCase ids are welcome — only ids taken from a real app are banned.

Targeting a real app is still fully supported — that knowledge belongs in the USER's
config (`flags.templates`) or the local playbook `aua remember` builds, not in this repo.
"""


def _tracked_files() -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [path for path in out.split("\0") if path]


def _fingerprint(value: str) -> tuple[int, str]:
    folded = value.casefold()
    return len(folded), hashlib.sha256(folded.encode()).hexdigest()


def _references_in(rel: str) -> list[str]:
    try:
        text = (REPO / rel).read_bytes().decode("utf-8", errors="ignore")
    except OSError:  # tracked but not on disk (staged deletion), or unreadable
        return []
    hits: list[str] = []
    by_length: dict[int, set[str]] = {}
    for length, digest in _BANNED_FINGERPRINTS:
        by_length.setdefault(length, set()).add(digest)
    for lineno, line in enumerate(text.splitlines(), 1):
        for token in _TOKEN.findall(line):
            folded = token.casefold()
            found = False
            for length, digests in by_length.items():
                for start in range(len(folded) - length + 1):
                    if _fingerprint(folded[start : start + length])[1] in digests:
                        hits.append(f"{rel}:{lineno}: forbidden app-specific fingerprint")
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return hits


def test_no_app_specific_references() -> None:
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout (or git unavailable) — nothing to enumerate")

    this_file = Path(__file__).resolve().relative_to(REPO).as_posix()
    hits: list[str] = []
    for rel in tracked:
        if rel == this_file or rel in _ALLOWLIST:
            continue
        hits += _references_in(rel)

    assert not hits, f"{_WHY}\nFound {len(hits)} reference(s):\n" + "\n".join(hits)


def test_allowlist_has_no_stale_entries() -> None:
    """A clean allowlisted file must lose its entry, so no exemption is granted silently."""
    stale = [rel for rel in _ALLOWLIST if not _references_in(rel)]
    assert not stale, (
        "These files no longer name an app, so their _ALLOWLIST entries are obsolete — "
        f"delete them from {Path(__file__).name}: " + ", ".join(stale)
    )
