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
    (7, "5153f687c17119c01e3edb32616ec838a3567a25da6029e8ada21eb4e2262579"),
    (7, "7e72b4554a267071e253746b1ca258eeca155433fccda474ed9ee379914a9d75"),
    (8, "0163da5d4139a781689e50e6587befcc72b5111d89e65a0374133025dbc6c83d"),
    (8, "6645e460b204a6c7db03e3cdd3cb49630ffa8f00aee2bea0d964ebe3ed038a25"),
    (9, "36afad554e7d6142231d95293308c5241bc8b677f6e50d250a2c7ebdbd1c351c"),
    (10, "696bd1d0c0c80724544014b7dbde1e53120e756363d927aa220659512bda9872"),
    (10, "cec8488078bf691aa2df7517ca80b0e06e454334a9ade4188fb421ed98e91734"),
    (10, "f35b26bcea2fe57c42445a1880712b0a803d66aa5b47c806d5971302d269d46e"),
    (11, "2eaf666c5a63366ea559ee11728c3405befad80ad3120a86803f0ceb683be17b"),
    (12, "3efddd390ef5bcfab3522a9d28eabc0af50c40a298452e069c75a7002a84c043"),
    (13, "1012e2130017784ce03735e09bc5e4c17065b84be67a623459877809a941d4a5"),
    (13, "349de0068cbcd9534e5f5d27fb1b7930b469a37a1086972a2bd83e8af84168fc"),
    (13, "97459a303e53430f99c4ee1c65673f99f39e4a54194a81b347c01043dd8823ce"),
    (13, "ba3ee3cad0b42abe97777172f3b92d3d4d751919c634b53cc76ac7f1cbdb16fb"),
    (14, "7b86ca4e7bdc2cf5370099c28ec5f5a670f67758da319250d793adfa8fd4bee5"),
    (14, "d156ef879234a173be1ce47bd100d8c93e0544edba7d274e7678b4b7b46f6a1a"),
    (16, "e7d691c42dfc5890abfe3957e9a64831b41b5874d21fba706d18c805916115b1"),
    (17, "5b68b8bc26eddc9d10d6b54dd01f139398c87b325bd3d989311f92f4b54f3888"),
    (17, "8bc2b46820b7f967a79e35f78867f65790256c42393fbb7291abdb7b1f09865b"),
    (17, "8dc188f9feb1a90361d19e1f330b550dc1e542a60cf4cc511f792bdf7077c942"),
    (18, "330cc94b8228255b9ac9f9f978f1c2ba8d003db2e3042e3825f88917aad56e43"),
    (18, "9bd7e122ea82a39929dc139bdd7c47516667f86a34263e492564dee9defc7d6a"),
    (19, "109f617282c5b653ebba9a1d62db225e160a9fb10cb8535755614bcc0cbd2b26"),
    (23, "2d365ee7f74e51a04949f040216e4a9c48725c4506edf7d28f3559e3b2663d15"),
    (36, "3e9b41124244962294a4ab1ff610c716aa39e74b00cc26caeb90ab98b0f6cf1a"),
}

# Only phrase-shaped fingerprints belong here. Compacting every deny-list entry would turn a
# legitimate hyphenated repository owner or a long binary line into accidental adjacent text.
_BANNED_COMPACT_FINGERPRINTS = {
    (6, "d18bb58fed5530267cce8cd03e64aec1eb9efc44e2a87ecababd437ecee509d5"),
    (7, "7e72b4554a267071e253746b1ca258eeca155433fccda474ed9ee379914a9d75"),
}

_TOKEN = re.compile(r"[A-Za-z0-9_@.:/+%-]+")
_SEPARATORS = re.compile(r"[^a-z0-9]+")
_PHRASE = re.compile(r"[A-Za-z0-9]+(?:[ _-]+[A-Za-z0-9]+)+")

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
            [
                "git",
                "-C",
                str(REPO),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
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


def _by_length(fingerprints: set[tuple[int, str]]) -> dict[int, set[str]]:
    grouped: dict[int, set[str]] = {}
    for length, digest in fingerprints:
        grouped.setdefault(length, set()).add(digest)
    return grouped


def _contains_fingerprint(value: str, fingerprints: dict[int, set[str]]) -> bool:
    folded = value.casefold()
    for length, digests in fingerprints.items():
        for start in range(len(folded) - length + 1):
            if _fingerprint(folded[start : start + length])[1] in digests:
                return True
    return False


def _value_has_banned_fingerprint(
    value: str,
    fingerprints: dict[int, set[str]],
    compact_fingerprints: dict[int, set[str]],
) -> bool:
    if any(_contains_fingerprint(token, fingerprints) for token in _TOKEN.findall(value)):
        return True
    for phrase in _PHRASE.findall(value):
        compact = _SEPARATORS.sub("", phrase.casefold())
        if _contains_fingerprint(compact, compact_fingerprints):
            return True
    return False


def _references_in(rel: str) -> list[str]:
    fingerprints = _by_length(_BANNED_FINGERPRINTS)
    compact_fingerprints = _by_length(_BANNED_COMPACT_FINGERPRINTS)
    if _value_has_banned_fingerprint(rel, fingerprints, compact_fingerprints):
        return [f"{rel}: forbidden app-specific fingerprint in path"]
    try:
        text = (REPO / rel).read_bytes().decode("utf-8", errors="ignore")
    except OSError:  # tracked but not on disk (staged deletion), or unreadable
        return []
    hits: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _value_has_banned_fingerprint(line, fingerprints, compact_fingerprints):
            hits.append(f"{rel}:{lineno}: forbidden app-specific fingerprint")
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


def test_scanner_catches_names_split_by_separators() -> None:
    marker = "fictionalmarker"
    fingerprints = _by_length({_fingerprint(marker)})

    assert _value_has_banned_fingerprint("fictional marker", {}, fingerprints)
    assert _value_has_banned_fingerprint("FICTIONAL_MARKER", {}, fingerprints)
    assert not _value_has_banned_fingerprint("unrelated placeholder", {}, fingerprints)
