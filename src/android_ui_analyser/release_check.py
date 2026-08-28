"""Ask GitHub whether a newer AUA has been released — cheaply, and without a clone.

Automations run `aua` on every job, and "is there a new version?" used to mean pulling all of
`main`: slow, and an answer to a different question, because `main` moves between releases.
This module reads the one endpoint that answers it directly (`/releases/latest`) and caches
the reply, because the unauthenticated GitHub API allows 60 requests per hour per IP — a
fleet of runners behind one NAT would otherwise spend that whole budget on version checks.

A failed check is a normal outcome here, never an exception. A laptop on a plane, a
rate-limited CI runner and a repository that has cut no releases yet all report `error` and
let the caller carry on: an update check must never be the thing that breaks a pipeline.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .atomic import atomic_write_text

__all__ = ["LATEST_RELEASE_URL", "UpdateStatus", "check_for_update", "format_status"]

LATEST_RELEASE_URL = "https://api.github.com/repos/The-Wordlab/Android-UI-Analyser/releases/latest"

# Release notes are meant to be read by a human at a terminal, not archived. The cap keeps a
# pathological body out of the cache file and out of an automation's log.
MAX_NOTES_CHARS = 8000

_CACHE_FILENAME = "update-check.json"

_VERSION_RE = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True)
class UpdateStatus:
    """What the check found. `latest is None` means the question could not be answered."""

    installed: str
    latest: str | None
    update_available: bool
    tag: str | None
    published_at: str | None
    release_url: str | None
    notes: str | None
    # When the release data was actually fetched — so a cached answer reports its own age
    # rather than pretending it just came off the wire.
    checked_at: str
    from_cache: bool
    error: str | None


def check_for_update(
    *, timeout: float = 5.0, force: bool = False, cache_ttl: float = 3600.0
) -> UpdateStatus:
    """Report whether a newer release exists. Never raises for a check that could not run."""

    if not force:
        cached = _read_cache(cache_ttl)
        if cached is not None:
            release, fetched_at = cached
            return _status(release, checked_at=fetched_at, from_cache=True)

    try:
        release = _fetch_latest(timeout)
    except urllib.error.HTTPError as exc:
        return _failed(_http_message(exc))
    except (OSError, ValueError) as exc:
        # URLError (DNS, refused, TLS) and socket timeouts are OSError; a body that is not
        # the JSON object we expect arrives as ValueError.
        return _failed(f"could not reach GitHub: {exc}")

    now = _now()
    _write_cache(release, now)
    return _status(release, checked_at=now, from_cache=False)


def format_status(status: UpdateStatus, *, repo_dir: str | None = None) -> str:
    """Render the human-readable block, including the exact upgrade commands."""

    if status.error is not None:
        return f"aua {status.installed} — update check unavailable: {status.error}"

    if not status.update_available:
        latest = status.latest or status.installed
        return f"aua {status.installed} — up to date (latest release: {latest})."

    tag = status.tag or f"v{status.latest}"
    clone = repo_dir or "<your aua clone>"
    lines = [f"aua {status.installed} — update available: {status.latest}"]
    if status.published_at:
        lines[0] += f" (released {status.published_at[:10]})"
    if status.release_url:
        lines.append(status.release_url)
    if status.notes:
        lines += ["", "What's new:", status.notes.strip()]
    lines += [
        "",
        "Upgrade:",
        f"  cd {clone} && git fetch --tags && git checkout {tag} && ./install.sh",
    ]
    return "\n".join(lines)


# --- version comparison -------------------------------------------------------------------


def _version_key(
    text: str,
) -> tuple[tuple[int, int, int], int, tuple[tuple[int, int | str], ...]] | None:
    """Return a comparison key for the SemVer form used by release tags.

    `packaging` is not a declared dependency, so the small comparison stays local. Numeric
    prerelease identifiers sort numerically and before text identifiers; any prerelease sorts
    before the final release. Build metadata is accepted and ignored for precedence, as SemVer
    requires. Anything outside SemVer is reported rather than guessed at.
    """
    match = _VERSION_RE.fullmatch(text.strip().removeprefix("v").removeprefix("V"))
    if match is None:
        return None
    numbers = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    prerelease = match.group(4)
    if prerelease is None:
        return (numbers, 1, ())
    identifiers_list: list[tuple[int, int | str]] = []
    for part in prerelease.split("."):
        identifiers_list.append((0, int(part)) if part.isdigit() else (1, part))
    identifiers = tuple(identifiers_list)
    return (numbers, 0, identifiers)


# --- GitHub ------------------------------------------------------------------------------


def _fetch_latest(timeout: float) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        # GitHub rejects requests without one.
        "User-Agent": f"android-ui-analyser/{__version__}",
    }
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if token:
        # Lifts a CI runner from 60 requests/hour per shared IP to 5000 per account. The token
        # is never stored, echoed or returned — it only ever leaves as this header.
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(LATEST_RELEASE_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    payload = json.loads(body.decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError("GitHub returned an unexpected payload")
    return payload


def _http_message(exc: urllib.error.HTTPError) -> str:
    if exc.code == 404:
        return "no releases published yet"
    if exc.code in (403, 429):
        return (
            "GitHub rate limit reached; set GITHUB_TOKEN to raise it, "
            "or retry after the limit resets"
        )
    return f"GitHub returned HTTP {exc.code}"


# --- cache -------------------------------------------------------------------------------


def _cache_path() -> Path:
    configured = os.environ.get("AUA_CACHE__DIR")
    if configured:
        root = Path(configured).expanduser()
    elif xdg_cache := os.environ.get("XDG_CACHE_HOME"):
        root = Path(xdg_cache).expanduser() / "android-ui-analyser"
    else:
        root = Path("~/.cache/android-ui-analyser").expanduser()
    return root / _CACHE_FILENAME


def _read_cache(cache_ttl: float) -> tuple[dict[str, Any], str] | None:
    """Return the cached release and its fetch time, or None when there is nothing usable."""
    try:
        raw = json.loads(_cache_path().read_text(encoding="utf-8"))
        fetched_at = str(raw["fetched_at"])
        release = raw["release"]
        age = (datetime.now(UTC) - datetime.fromisoformat(fetched_at)).total_seconds()
    except (OSError, ValueError, KeyError, TypeError):
        # A truncated, hand-edited or older-format cache is not an error worth reporting: the
        # next fetch overwrites it.
        return None
    if not isinstance(release, dict) or age < 0 or age > cache_ttl:
        return None
    return release, fetched_at


def _write_cache(release: dict[str, Any], fetched_at: str) -> None:
    # Only successes are cached. A failure is usually transient (offline, rate limit), and
    # remembering it would keep reporting a stale problem after the network came back.
    payload = {
        "fetched_at": fetched_at,
        "release": {
            "tag_name": release.get("tag_name"),
            "published_at": release.get("published_at"),
            "html_url": release.get("html_url"),
            "body": release.get("body"),
        },
    }
    # A read-only or full cache directory costs a round trip next time, nothing more.
    with contextlib.suppress(OSError):
        atomic_write_text(_cache_path(), json.dumps(payload, ensure_ascii=False))


# --- assembly ----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: Any, *, limit: int | None = None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value[:limit] if limit is not None else value


def _failed(
    error: str, *, checked_at: str | None = None, from_cache: bool = False
) -> UpdateStatus:
    return UpdateStatus(
        installed=__version__,
        latest=None,
        update_available=False,
        tag=None,
        published_at=None,
        release_url=None,
        notes=None,
        checked_at=checked_at or _now(),
        from_cache=from_cache,
        error=error,
    )


def _status(release: dict[str, Any], *, checked_at: str, from_cache: bool) -> UpdateStatus:
    tag = _text(release.get("tag_name"))
    if tag is None:
        return _failed(
            "GitHub returned a release with no tag",
            checked_at=checked_at,
            from_cache=from_cache,
        )

    latest = tag.removeprefix("v").removeprefix("V")
    installed_key = _version_key(__version__)
    latest_key = _version_key(latest)
    error = None
    if installed_key is None or latest_key is None:
        error = f"cannot compare version {__version__} with release {latest}"

    return UpdateStatus(
        installed=__version__,
        latest=latest,
        update_available=(
            installed_key is not None and latest_key is not None and latest_key > installed_key
        ),
        tag=tag,
        published_at=_text(release.get("published_at")),
        release_url=_text(release.get("html_url")),
        notes=_text(release.get("body"), limit=MAX_NOTES_CHARS),
        checked_at=checked_at,
        from_cache=from_cache,
        error=error,
    )
