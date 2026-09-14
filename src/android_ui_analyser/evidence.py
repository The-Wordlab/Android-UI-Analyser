"""One list of everything a run produced, so the result can be handed on without re-globbing.

A verdict on its own is a claim.  What makes it reviewable is the screenshots, the video and the
written report beside it - and the caller usually wants to publish exactly that set: a bundle for
a reviewer, a link on a pull request.  Today every caller re-discovers those files by walking the
artifact directory and guessing which ones matter, which is how a full logcat ends up in a
published bundle.

So this walks the directory once and answers three questions the caller actually has: what is
there, what is it (image, video, text, data), and what must a person look at before any of it
leaves the machine.  It never deletes, never uploads, and never redacts in place - deciding what
is publishable is the caller's call, and a module that quietly dropped files would make a bundle
look complete when it was not.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

KIND_BY_SUFFIX: dict[str, str] = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "video",
    ".mp4": "video",
    ".webm": "video",
    ".mkv": "video",
    ".md": "text",
    ".txt": "text",
    ".log": "text",
    ".json": "data",
    ".jsonl": "data",
    ".yaml": "data",
    ".yml": "data",
    ".xml": "data",
    ".har": "data",
}

# Name fragments whose contents are device or network records rather than authored evidence.
# These are not excluded - a run's raw calls are exactly what you want when a verdict is disputed -
# but nothing here should reach a shared link without a person having read it first.
_NEEDS_REVIEW = ("logcat", "calls.jsonl", "proxy", "mock", "har", "flows.jsonl", ".log")

_SKIP_DIRS = {"__pycache__", ".aua-cache", ".git"}
_MAX_FILES = 2000


def classify(path: Path) -> str:
    """image | video | text | data | other, by extension alone."""

    return KIND_BY_SUFFIX.get(path.suffix.casefold(), "other")


def needs_review(path: Path) -> bool:
    """Would publishing this file, unread, risk leaking something?"""

    lowered = str(path).casefold()
    return any(fragment in lowered for fragment in _NEEDS_REVIEW)


def _entry(path: Path, root: Path) -> dict[str, Any]:
    size = 0
    with contextlib.suppress(OSError):
        size = path.stat().st_size
    return {
        "path": str(path),
        "name": str(path.relative_to(root)),
        "kind": classify(path),
        "bytes": size,
        "review_before_publishing": needs_review(path),
    }


def _walk(root: Path) -> Iterable[Path]:
    stack = [root]
    seen = 0
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                if child.name not in _SKIP_DIRS and not child.is_symlink():
                    stack.append(child)
                continue
            if not child.is_file():
                continue
            seen += 1
            if seen > _MAX_FILES:
                return
            yield child


def collect(root: str | Path, *, extra: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Group everything under *root* (plus any *extra* files) by kind.

    ``extra`` exists because a screen recording is often written where the caller asked for it
    rather than inside the bundle, and a recording left out of the list is a recording nobody
    publishes.
    """

    base = Path(root).expanduser()
    entries: list[dict[str, Any]] = []
    if base.is_dir():
        entries = [_entry(path, base) for path in _walk(base)]
    for item in extra:
        path = Path(item).expanduser()
        if not path.is_file() or any(entry["path"] == str(path) for entry in entries):
            continue
        entry = _entry(path, path.parent)
        entry["name"] = path.name
        entries.append(entry)

    grouped: dict[str, list[dict[str, Any]]] = {
        kind: [entry for entry in entries if entry["kind"] == kind]
        for kind in ("image", "video", "text", "data", "other")
    }
    review = sorted(entry["name"] for entry in entries if entry["review_before_publishing"])
    return {
        "dir": str(base),
        "counts": {kind: len(items) for kind, items in grouped.items()},
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "images": grouped["image"],
        "videos": grouped["video"],
        "text": grouped["text"],
        "data": grouped["data"],
        "other": grouped["other"],
        "review_before_publishing": review,
        "truncated": len(entries) >= _MAX_FILES,
    }


def handback(
    result: dict[str, Any],
    *,
    root: str | Path,
    extra: Iterable[str | Path] = (),
    report: str | Path | None = None,
) -> dict[str, Any]:
    """The verdict and its evidence in one payload, with the publishing caveat attached.

    The caller is told, every time, that the review list has not been read for it.  A bundle is
    published by a person or by another agent acting for one, and "AUA said it was fine" is not a
    sentence anyone should be able to write.
    """

    bundle = collect(root, extra=extra)
    if report is not None and Path(report).expanduser().is_file():
        bundle["report"] = str(Path(report).expanduser())
    payload = dict(result)
    payload["evidence"] = bundle
    if bundle["review_before_publishing"]:
        payload["publishing"] = (
            "Read the files in evidence.review_before_publishing before sharing this bundle: "
            "device and network records can carry tokens, account addresses and identifiers. "
            "AUA has not read them for you."
        )
    else:
        payload["publishing"] = (
            "No device or network records were captured, so nothing in this bundle is "
            "flagged for review. Screenshots and video still show whatever was on screen."
        )
    return payload
