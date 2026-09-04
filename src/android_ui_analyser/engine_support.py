"""Helpers shared by ``engine.py`` and its domain modules.

Module-level constants, small pure functions and record types that more than one engine module
reads. Nothing here touches a device; nothing here may import ``engine`` or an ``engine_*`` module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

from .errors import UsageError

logger = logging.getLogger("android_ui_analyser.engine")


_ASSIST_MAX_STEPS = 6  # bound on planner actions per recovery attempt (opt-in only)


# A bottom system bar starts within this fraction of the screen height. Wide enough for a
# tall three-button bar, narrow enough that a systemui sheet or the expanded notification
# shade can never be mistaken for it (see Engine._system_bar_top).
_SYSTEM_BAR_BAND = 0.85


# Terms that describe the surrounding UI rather than a user's intended control. A single match
# on one of these is not enough to turn a visible multi-word control into an execution proposal.
_GENERIC_MANUAL_MATCH_TERMS = frozenset(
    {"action", "button", "control", "item", "menu", "option", "page", "settings", "ui", "view"}
)


_AWAIT_PREFIXES = {
    "text": "text",
    "rid": "rid",
    "id": "rid",
    "desc": "desc",
    # Off-screen evidence. A UI predicate answers "is it drawn yet"; these answer "did the
    # work actually happen", which is a different question and sometimes the only answerable
    # one — a streamed LaTeX answer reaches the hierarchy as U+FFFD, so no `text:` term can
    # confirm it arrived. Terms are ANDed, so `net:POST /v1/chat,text:x =` reads as "the
    # backend replied *and* the screen shows it".
    "net": "net",
    "log": "log",
}


class _AwaitTerm(NamedTuple):
    """One condition in an ``await`` predicate: a selector, and whether it must be absent."""

    text: str  # as written, for echoing back
    by: str  # text | rid | desc — the same vocabulary every selector uses
    value: str
    negated: bool


class _ActionSite(NamedTuple):
    """Where one action was spent, for cost bookkeeping: ``(screen, control, package)``.

    All three come from the *same* pre-action cache read. The package has to travel with the
    site because the id cache is deleted the moment the device is touched, so the settle path
    that records the measurement can no longer look it up.
    """

    screen: str
    control: str
    package: str | None


class _ResolvedFlagsResource(NamedTuple):
    """Parsed flags file retained across flow preflight and execution."""

    source_path: str
    app: str | None
    pairs: dict[str, str]


class _ResolvedCassetteResource(NamedTuple):
    """Parsed cassette retained across flow preflight and execution."""

    name: str
    source_path: Path
    entries: list[dict[str, Any]]


def _split_await_terms(predicate: str) -> list[str]:
    r"""Split comma-separated terms while allowing a literal comma as ``\,``.

    Shell quotes protect spaces from the shell; they cannot tell this grammar whether a comma
    belongs to a label or separates two terms because the quote characters are already gone by
    the time Python receives the argument.  A small explicit escape keeps the grammar usable:
    ``--until 'text:Hello\, friend,!text:Loading'``.  Only comma and backslash are special, so
    values such as Windows-looking paths or regular apostrophes are not accidentally rewritten.
    """
    chunks: list[str] = []
    current: list[str] = []
    escaped = False
    for char in predicate:
        if escaped:
            if char in {",", "\\"}:
                current.append(char)
            else:
                # Preserve an escape that does not belong to this tiny grammar.  In particular,
                # regex-like text remains byte-for-byte what the caller supplied.
                current.extend(("\\", char))
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == ",":
            chunks.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        raise UsageError(
            "await predicate ends with an incomplete escape",
            hint="Use `\\,` for a literal comma and `\\\\` for a literal backslash.",
        )
    chunks.append("".join(current))
    return chunks


def _parse_await_terms(predicate: str, *, require_positive: bool = False) -> list[_AwaitTerm]:
    """``"rid:resultCard,!text:Generating"`` → two terms, ANDed.

    A deliberately tiny grammar rather than an expression language. What a lane needs is
    "this appeared and that went away"; a general evaluator would add a second place for a
    predicate to be quietly wrong about the screen, which is the failure this list exists to
    remove. Unknown prefixes are refused rather than treated as literal text, for the same
    reason an unrecognised ``--by`` token is.
    """
    raw = (predicate or "").strip()
    if not raw:
        raise UsageError(
            "await needs a predicate",
            hint="e.g. `aua await-and-analyze 'rid:resultCard,!text:Generating'` — comma-separated terms, "
            "all of which must hold; `!` means must be absent; `\\,` is a literal comma.",
        )
    terms: list[_AwaitTerm] = []
    for chunk in _split_await_terms(raw):
        piece = chunk.strip()
        if not piece:
            continue
        negated = piece.startswith("!")
        body = piece[1:].strip() if negated else piece
        prefix, sep, value = body.partition(":")
        if not sep or not value.strip():
            raise UsageError(
                f"await term {piece!r} needs a <field>:<value> form",
                hint="fields: "
                + ", ".join(sorted(_AWAIT_PREFIXES))
                + " (prefix with ! for absent)",
            )
        by = _AWAIT_PREFIXES.get(prefix.strip().lower())
        if by is None:
            raise UsageError(
                f"await term {piece!r} names an unknown field {prefix.strip()!r}",
                hint="fields: "
                + ", ".join(sorted(_AWAIT_PREFIXES))
                + " (prefix with ! for absent)",
            )
        terms.append(_AwaitTerm(text=piece, by=by, value=value.strip(), negated=negated))
    if not terms:
        raise UsageError("await needs at least one term", hint="e.g. `text:Done`")
    if require_positive and not any(not term.negated for term in terms):
        raise UsageError(
            "an action-bound await needs at least one positive arrival term",
            hint=(
                "Add the text:, rid:, desc:, net:, or log: evidence that proves the action "
                "arrived. Keep absence-only checks such as `!text:Loading` in a standalone "
                "wait/await."
            ),
        )
    return terms


def _is_resource_id_lookup(by: str) -> bool:
    """Both public spellings select the locale-independent resource-id field."""
    return (by or "text").lower() in {"id", "rid"}


def _label(text: str) -> str:
    """A one-line label for a summary row — normalised, not shortened.

    These used to be cut at 60 characters, which bought nothing: the same string is already in
    `elements[].text` at full length in the same response, and on the densest screen measured
    (2026-08-10) every label together came to 149 characters inside a 9,915-character payload,
    with none reaching the limit. What it did cost was legibility — a heading past the limit came
    back as a sentence that simply stops, so two agent runs read it as complete and spent an extra
    `analyze` recovering text they had already been sent.
    """
    return text.replace("\n", " ").strip()


def detail_tokens(outcome: str, **fields: Any) -> str:
    """``"moved steps=3 dy=1420"`` — outcome first, then ``k=v`` pairs.

    ``ActionResult`` is a frozen schema owned elsewhere, so scroll/expect verdicts ride in
    ``detail``. Outcome-first keeps it greppable (``grep -q target-not-found``) and the
    tokens keep it parseable; the exit code stays the primary signal.
    """
    parts = [outcome]
    parts += [f"{k}={v}" for k, v in fields.items() if v is not None]
    return " ".join(parts)


class _HandoverRefused(Exception):
    """The UiAutomation slot could not be handed to the helper, and why.

    Raised rather than returned so :meth:`Engine._device_agent_borrowed` can refuse from any depth
    without every caller unpacking a union. It is already journalled by the time it is raised; the
    caller's only remaining decision is whether a refusal is fatal for what it was doing.
    """

    def __init__(self, reason: str, serial: str | None, detail: str | None = None) -> None:
        super().__init__(reason if detail is None else f"{reason}: {detail}")
        self.reason = reason
        self.serial = serial
        self.detail = detail
