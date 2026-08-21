"""Persistent per-app memory — the tool's long-term knowledge of an app (PRD §6b).

The tool builds and maintains a map of each app **itself** as it navigates: what screens
exist, what durable elements are on them, and the routes between them. An agent reads it
back at session start (``aua map``) so it knows the layout before navigating, and on every
``analyze`` the tool recognises the current screen (``meta.known_screen``) and records new
screens / route edges passively — no extra agent calls.

Design notes
------------
- **Durable skeleton only.** We store screens, routes, and *stable* elements (tabs,
  buttons, tool names, input *shapes*) — never volatile per-user content. A list of recent
  chats is recorded as a *shape* ("list (dynamic)"), not its items; the agent fetches live
  contents with ``analyze`` when it actually needs them. This keeps the map small, fresh,
  and free of PII.
- **Recognition by anchors.** A screen's identity is a set of durable *anchors* (stable
  resource-ids + short chrome labels + content-descriptions). Revisits are matched by
  Jaccard overlap (robust to dynamic content); large divergence or an app-version bump
  flags the screen ``stale`` so the agent re-verifies.
- **Privacy.** Local-only, never transmitted. ``EditText`` *values* are stored as a shape
  (``"<filled>"``), and secret / PII-looking text is redacted (``memory.redact``).
- **Single renderer.** :func:`render_map` produces both the on-disk ``MAP.md`` and every
  ``aua map`` view, so there is never any drift between them.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from collections import Counter, deque
from collections.abc import Sequence
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .atomic import atomic_write_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import MemoryCfg
    from .schema import Element

logger = logging.getLogger("android_ui_analyser.memory")

MEMORY_SCHEMA_VERSION = 4
LEGACY_CONTEXT_ID = "legacy-default"
DEFAULT_CONTEXT_ID = "default"

# Session pending buffer: cap on steps accumulated between analyzes (a hit DROPS the
# eventual edge — a truncated step list would replay wrong), and a TTL so an abandoned
# journey (app killed mid-flow) can't smear a stale edge onto a later navigation.
_PENDING_CAP = 12
_PENDING_TTL_S = 600
_RECENT_CAP = 40  # rolling action journal (feeds `aua flow save --last N`)
# Diagnostic access log (`CallRecord`). Held apart from `_RECENT_CAP` because it also holds
# waits and failures, i.e. more lines than there were actions. The session file is rewritten
# on every recorded call, so this cap is a history/size trade-off: 120 lines covers a long
# run end to end (the session that motivated the log was 40 actions plus its waits) while
# keeping the file in the tens of kilobytes.
_CALLS_CAP = 120

# A same-package edge with this many destination-producing actions is almost certainly a
# recognition failure that kept the cursor on one sparse screen while the user crossed several
# real screens. Long cross-package sign-in journeys are exempt (their package markers prove the
# steps are an intentional transit route). Agents can still author longer flows explicitly.
_MAX_SAME_PACKAGE_DESTINATION_STEPS = 4

# Jaccard overlap (after a small activity bonus) at/above which a screen is recognised as
# a known one rather than treated as new. Below the drift band it is a fresh screen.
_RECOGNIZE_MIN = 0.34
# How many times a drifted shape must be seen before it becomes the screen's identity. One
# is too few: a screen caught with the keyboard up, a video playing or a banner open diverges
# hard for exactly one frame, and adopting that pins the screen to furniture that is normally
# absent. Two is enough to tell a rebuild from a moment.
_READOPT_SIGHTINGS = 2

# Shared chrome/renderers describe *how* a screen is drawn, not *which* screen it is. They may
# remain in stored signatures for drift diagnostics, but recognition must have at least one
# app-authored, discriminative anchor in common. Without this guard, two sparse surfaces that
# only shared a back button and a framework root scored 0.8 and collapsed into one screen.
_GENERIC_IDENTITY_RESOURCE_IDS = frozenset(
    {
        "action_bar_root",
        "actionbar_root",
        "button_back",
        "button_nav_back",
        "buttonnavback",
        "latex_view",
        "latexview",
        "nav_back",
        "navigate_up",
        "screen_title",
        "screentitle",
        "surface_title",
        "surfacetitle",
        "title",
        "title_view",
        "titleview",
        "header",
        "toolbar_back",
    }
)
_GENERIC_IDENTITY_LABELS = frozenset(
    {
        "back",
        "close",
        "dismiss",
        "navigate up",
        "up",
    }
)

# A screen matching the session's last goto/find target is floated to the top of
# suggestions regardless of frequency (a large additive boost, not a hard pin).
_LAST_GOAL_BOOST = 1_000_000.0

REDACT_TOKENS = {"<filled>", "<empty>", "<redacted>"}


def matches_any(package: str | None, globs: Sequence[str]) -> bool:
    """True if *package* matches any fnmatch glob (a bare literal matches exactly)."""
    if not package:
        return False
    p = package.lower()
    return any(fnmatch(p, g.lower()) for g in globs)


# Inbound labels too generic (or too destructive) to name a screen after — a screen
# reached via "Continue" or "Delete" is named from its title/activity instead. Compared
# via slug(label).
GENERIC_INBOUND = frozenset(
    {
        "continue",
        "next",
        "ok",
        "okay",
        "done",
        "skip",
        "cancel",
        "back",
        "close",
        "yes",
        "no",
        "not_now",
        "later",
        "ask_me_later",
        "got_it",
        "accept",
        "agree",
        "confirm",
        "allow",
        "deny",
        "submit",
        "save",
        "retry",
        "get_started",
        "delete",
        "sign_out",
        "log_out",
        "login",
        "log_in",
        "sign_in",
        "just_once",
        "while_using_the_app",
        "only_this_time",
        "always_allow",
        "open",
        "create",
    }
)


# --------------------------------------------------------------------------- models


class KeyElement(BaseModel):
    """A durable, actionable element worth remembering (nav target, button, input)."""

    model_config = ConfigDict(extra="ignore")
    type: str
    label: str | None = None
    resource_id: str | None = None
    clickable: bool = False
    input: bool = False
    value: str | None = None  # shape only, never the literal value (e.g. "<filled>")


class ContextRecord(BaseModel):
    """A reproducible UI configuration, normally created from verified feature flags."""

    model_config = ConfigDict(extra="ignore")
    id: str
    flags: dict[str, str] = Field(default_factory=dict)
    app_version: str | None = None
    shell_anchors: list[str] = Field(default_factory=list)
    source: Literal["legacy", "default", "flags_verified", "flags_unverified", "agent"] = "default"
    verified: bool = False
    evidence: list[str] = Field(default_factory=list)
    first_seen: str
    last_seen: str


class KnowledgeEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["source", "runtime", "agent", "user"] = "agent"
    ref: str | None = None
    detail: str | None = None


class KnowledgeScope(BaseModel):
    model_config = ConfigDict(extra="ignore")
    package: str
    app_version: str | None = None
    context_id: str | None = None
    flags: dict[str, str] = Field(default_factory=dict)


class KnowledgeItem(BaseModel):
    """A provenance-bearing fact learned from a person, agent, runtime, or source."""

    model_config = ConfigDict(extra="ignore")
    id: str
    kind: Literal["description", "note", "recipe", "deeplink", "claim"] = "note"
    text: str
    name: str | None = None
    scope: KnowledgeScope
    source: Literal["legacy", "user", "agent", "runtime", "source"] = "agent"
    agent: str | None = None
    session: str | None = None
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)
    status: Literal["accepted", "proposed", "stale", "rejected"] = "accepted"
    created_at: str
    last_verified: str | None = None


class ActionTiming(BaseModel):
    """How long one control on one screen has historically taken to produce its next screen.

    Kept per (screen, control) rather than per action kind, because that is the granularity the
    cost actually varies at: on one screen a tap settles in 40ms and on another the same kind of
    tap waits on a network round-trip. The coarse per-kind EMA has to average those together,
    which makes it simultaneously too slow for the cheap screen and far too fast for the
    expensive one — ``perf.py`` caps it at 1.6s while real screens here take 18-60s.

    ``max_ms`` is kept beside the average deliberately: a deadline set from the mean is wrong
    half the time by construction. ``n`` is what tells a reader whether to believe any of it.
    """

    model_config = ConfigDict(extra="ignore")
    n: int = 0
    ema_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    last_seen: str | None = None
    # What ended the wait the last time: a real change, or the deadline running out. A control
    # whose history is all timeouts has no useful average — it has an unsolved problem.
    last_outcome: str | None = None


class ScreenRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    id: str | None = None
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    logical_name: str | None = None
    variant: str | None = None
    state: str | None = None
    surface: str | None = None
    context_id: str = DEFAULT_CONTEXT_ID
    name_source: Literal["explicit", "resource", "title", "route", "activity", "legacy"] = "legacy"
    activity: str | None = None
    signature: str
    anchors: list[str] = Field(default_factory=list)
    tier: str = "hierarchy"
    # Perception evidence for experience-based OCR skip (not the same as latest ``tier`` —
    # a hierarchy-tier observation may still have run Apple Vision augmentation).
    hierarchy_only_ok: int = 0  # visits where hierarchy alone was enough (no kept OCR els)
    ocr_helped: int = 0  # visits where OCR contributed kept elements
    # Per-control settle history for this screen, keyed by the control's stable key (or its
    # resource-id tail when it has none). See :class:`ActionTiming`.
    timings: dict[str, ActionTiming] = Field(default_factory=dict)
    key_elements: list[KeyElement] = Field(default_factory=list)
    dynamic: list[str] = Field(default_factory=list)  # shapes, e.g. "row list (dynamic)"
    notes: list[str] = Field(default_factory=list)
    app_version: str | None = None
    first_seen: str
    last_seen: str
    last_verified: str
    visit_count: int = 1
    stale: bool = False
    # A drifted shape waiting to be corroborated before it replaces ``anchors``. See
    # ``_READOPT_SIGHTINGS`` and the re-anchoring branch in ``record_screen``.
    pending_anchors: list[str] = Field(default_factory=list)
    pending_anchor_hits: int = 0


def screen_skips_ocr(rec: ScreenRecord, *, min_hierarchy_ok: int = 3) -> bool:
    """True when past visits say hierarchy alone is enough — skip parallel OCR.

    Conservative: only native/form surfaces with enough hierarchy-only successes and
    zero recorded OCR contributions. Canvas/webview and any prior OCR help stay on.
    """
    if rec.stale:
        return False
    if rec.ocr_helped > 0:
        return False
    if rec.hierarchy_only_ok < min_hierarchy_ok:
        return False
    surface = (rec.surface or "native").lower()
    return surface in ("native", "form")


class CaptureArrivalTerm(BaseModel):
    """One privacy-checked UI fact from a satisfied action-bound ``--until``.

    This is capture provenance, not an authored flow step.  Only the three UI selector lanes
    whose truth can be tied to the returned hierarchy fingerprint are retained.  Network and
    log predicates remain valid for authored flows, but cannot certify that a later
    ``flow save`` snapshot is the same destination.
    """

    model_config = ConfigDict(extra="forbid")
    by: Literal["text", "rid", "desc"]
    value: str
    negated: bool = False

    @field_validator("value")
    @classmethod
    def _nonempty_value(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("capture arrival term value must not be empty")
        return normalized


class CaptureArrivalProof(BaseModel):
    """Capture-only proof that the newest recorded action reached a specific UI frame.

    Every provenance field is duplicated deliberately.  A later ``flow save`` may use the
    predicate only when it still agrees with the selected action, current session segment, app
    context, foreground package, and fresh hierarchy fingerprint.  Rendering a flow drops this
    object; only its validated predicate is promoted to the authored artifact.
    """

    model_config = ConfigDict(extra="forbid")
    source: Literal["action_until"] = "action_until"
    terms: list[CaptureArrivalTerm]
    fingerprint: str
    package: str
    origin_package: str
    context_id: str
    capture_segment: int
    capture_order: int

    @field_validator("terms")
    @classmethod
    def _positive_arrival_required(
        cls, terms: list[CaptureArrivalTerm]
    ) -> list[CaptureArrivalTerm]:
        if not terms or not any(not term.negated for term in terms):
            raise ValueError("capture arrival proof needs a positive UI term")
        return terms

    @field_validator("fingerprint", "package", "origin_package", "context_id")
    @classmethod
    def _nonempty_provenance(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("capture arrival provenance must not be empty")
        return normalized


class RouteStep(BaseModel):
    """One replayable action of a route edge or flow (the shared step model).

    Auto-recorded steps carry a durable selector (``resource_id`` tail first, redacted
    ``label`` second) and NEVER a typed value; ``text`` exists for agent-authored flows
    only. ``package`` is the package the step ran in — ``None`` means the journey's
    origin app (transit steps through e.g. Chrome keep their package).
    """

    model_config = ConfigDict(extra="ignore")
    kind: str  # tap | long-press | input | clear | key | swipe | scroll-to
    #            (flows also: launch-app | wait-for | wait-stable | assert-visible | goto |
    #             repeat | retry | hide-keyboard | paste | …)
    label: str | None = None  # redacted-safe element label (may be '<redacted>')
    # New recordings keep descriptions distinct from visible text.  Older routes/flows used
    # ``label`` for either and remain valid; ``by is None`` preserves that tolerant matching.
    content_desc: str | None = None
    resource_id: str | None = None  # resource-id TAIL, the primary replay selector
    arg: str | None = None  # kind-specific: key name / swipe direction / query / package
    text: str | None = None  # input value — FLOWS ONLY, never set by auto-recording
    submit: bool = False  # input: fire the IME action after typing
    package: str | None = None  # package the step ran in; None = the journey's origin
    activity: str | None = None  # launch-app: pin the entry Activity on multi-launcher builds
    timeout_ms: int | None = None  # wait-for / wait-stable / assert-visible override
    by: str | None = None  # match target by: text (default) | id (resource-id) | desc
    index: int | None = None  # nth (0-based) of several matches, as the CLI's --index
    # Capture-only provenance.  Flow rendering deliberately omits these per-action fields: the
    # selected homogeneous segment promotes origin/context to the Flow itself.
    origin_package: str | None = None
    context_id: str | None = None
    capture_segment: int | None = None
    capture_order: int | None = None
    # A satisfied action-bound arrival predicate can be reused by a later ``flow save`` only
    # while it remains bound to this exact action/frame/segment.  Like the fields above, this is
    # session capture state and is stripped before route learning or flow rendering.
    arrival_proof: CaptureArrivalProof | None = None
    # What the call COST, as opposed to ``timeout_ms`` above, which is what the caller was
    # willing to WAIT. ``started_at_ms`` is wall-clock epoch milliseconds on purpose: the CLI,
    # the daemon and the on-device helper all record into the same journal, so a monotonic
    # reading would only be comparable inside whichever process took it, and epoch ms also
    # prints as a timestamp a human can line up against a logcat dump.
    started_at_ms: int | None = None
    elapsed_ms: int | None = None
    # What came back: "ok", an error code, or an await outcome ("satisfied" / "timeout").
    # ``None`` means unmeasured (recorded before timing existed, or by a caller with nothing
    # to report) — never "succeeded".
    outcome: str | None = None
    # scroll-to: which way to swipe WHILE SEARCHING, as the CLI's --direction (default "up",
    # i.e. look further down the list). A grid that opens already scrolled past its target
    # needs "down" — searching the wrong way reads as a missing element, not a missed search.
    direction: str | None = None
    # Composite flow blocks (Maestro ``repeat`` / ``retry``).
    substeps: list[RouteStep] = Field(default_factory=list)
    repeat: int | None = None
    max_retries: int | None = None
    # Flow-only assertion payload.  The YAML parser validates the exact vocabulary before this
    # shared route model sees it; keeping it in one mapping avoids turning every new semantic
    # predicate into navigation-memory state that auto-recorded routes could accidentally use.
    assertion: dict[str, Any] = Field(default_factory=dict)
    # Flow-only *data* payload, for steps whose argument is structured but is not a predicate —
    # currently `db-execute` (database / sql / parameters / restart). Kept separate from
    # `assertion` so neither field's name has to lie about what it holds, and so a new data step
    # never widens the assertion vocabulary that `goto` replay inspects.
    data: dict[str, Any] = Field(default_factory=dict)


class RouteEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = None
    from_screen: str
    to_screen: str
    action: str  # human label, e.g. "tap 'Apps'" (derived from steps when present)
    context_id: str = DEFAULT_CONTEXT_ID
    guards: dict[str, str] = Field(default_factory=dict)
    steps: list[RouteStep] = Field(default_factory=list)  # [] = legacy pre-v2 edge
    count: int = 1
    status: Literal["provisional", "verified", "rejected"] = "verified"
    verification_count: int = 0
    rejection_reason: str | None = None
    last_seen: str


class Deeplink(BaseModel):
    """A known deeplink URI for an app + what it does (a latency shortcut to suggest)."""

    model_config = ConfigDict(extra="ignore")
    uri: str
    note: str | None = None
    count: int = 1
    probed: bool = False  # True once actually opened on device (mined links start False)
    landed: str | None = None  # screen it landed on when last opened (if recognized)
    last_seen: str | None = None


class Recipe(BaseModel):
    """A named app-specific procedure the agent should reuse (e.g. login_full)."""

    model_config = ConfigDict(extra="ignore")
    name: str
    note: str


class LaunchEntry(BaseModel):
    """How to cold-start this app — the MAIN/LAUNCHER Activity to pin.

    Dev builds often declare more than one launcher (product entry + Dev Tools). Without
    a pin, ``am start`` / ``app_start`` resolves that nondeterministically. Once set,
    every bare ``aua app launch <pkg>`` (and flag/proxy restart fallback) uses this
    Activity instead of flipping a coin.
    """

    model_config = ConfigDict(extra="ignore")
    activity: str
    # user = `aua remember --launch-activity`; explicit = successful `app launch --activity`;
    # resolved = only one MAIN/LAUNCHER was declared so we auto-pinned it.
    source: Literal["user", "explicit", "resolved"] = "user"
    alternatives: list[str] = Field(default_factory=list)  # other MAIN/LAUNCHER components
    last_seen: str | None = None


class AppMap(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: int = MEMORY_SCHEMA_VERSION
    package: str
    label: str | None = None
    app_version: str | None = None
    last_verified: str | None = None
    # Playbook — durable app-level knowledge the tool learns and surfaces to the agent.
    description: str | None = None  # what this app is / how it behaves (one-liner)
    deeplinks: list[Deeplink] = Field(default_factory=list)  # shortcuts (set flags, jump)
    recipes: list[Recipe] = Field(default_factory=list)  # login_full, etc.
    notes: list[str] = Field(default_factory=list)  # quirks worth remembering
    launch: LaunchEntry | None = None  # pinned cold-start Activity (multi-launcher builds)
    # Last-seen MAIN/LAUNCHER set when no pin exists yet (surfaces ambiguity in the playbook).
    launcher_activities: list[str] = Field(default_factory=list)
    contexts: dict[str, ContextRecord] = Field(default_factory=dict)
    knowledge: list[KnowledgeItem] = Field(default_factory=list)
    research_tasks: list[dict[str, object]] = Field(default_factory=list)
    pending_reports: list[dict[str, object]] = Field(default_factory=list)
    screens: dict[str, ScreenRecord] = Field(default_factory=dict)
    routes: list[RouteEdge] = Field(default_factory=list)


class CallRecord(BaseModel):
    """One line of the tool's own access log: a call, when it ran, and what it answered.

    Deliberately **not** a :class:`RouteStep`, and kept in its own session list. ``pending``
    and ``recent`` are *replayable*: ``pending`` becomes a route edge's steps and ``recent``
    becomes a saved flow, so appending to either asserts "replaying this reproduces the
    journey". The records most worth timing are the least safe there — a ``wait-for`` that
    timed out would replay as a satisfied precondition, its :func:`step_display` would join
    :func:`derive_action` and so change the edge identity route dedup is keyed on, and it
    would consume the ``capture_order`` watermark :meth:`AppMemoryStore.record_action_arrival`
    uses to bind a satisfied ``--until`` to the newest *action*. A separate type makes that
    mistake a type error instead of silently corrupted route learning.

    Everything here is redaction-safe by construction: callers pass the same display strings
    the map and dry runs show, and free-form phrases go through :func:`log_phrase` first.
    """

    model_config = ConfigDict(extra="ignore")
    kind: str  # the call: tap | input | key | wait-for | wait-stable | analyze | goto | flow
    target: str | None = None  # what it was called on/with; the log prints ``kind`` itself
    started_at_ms: int | None = None  # wall-clock epoch ms (comparable across processes)
    elapsed_ms: int | None = None  # measured cost, never a configured budget
    outcome: str | None = None  # ok | satisfied | timeout | <error code>
    detail: str | None = None  # one short phrase of the answer (why it failed, what it hit)
    package: str | None = None
    screen: str | None = None  # recognized screen the call left the device on, if any
    # Ties a log line to the replayable journal entry it describes; ``None`` for calls that
    # never enter it (waits, read-only analyzes).
    capture_order: int | None = None


class SessionState(BaseModel):
    """Cross-process navigation cursor (per device serial) used to draw route edges."""

    model_config = ConfigDict(extra="ignore")
    package: str | None = None
    current_screen: str | None = None
    pending: list[RouteStep] = Field(default_factory=list)  # steps since last analyze
    pending_since: str | None = None  # ISO ts of the first pending step (TTL guard)
    pending_overflow: bool = False  # cap hit → the eventual edge is dropped, never garbled
    recent: list[RouteStep] = Field(default_factory=list)  # rolling journal (`flow save`)
    # Diagnostic access log — see :class:`CallRecord`. Nothing here is ever replayed, which is
    # what lets it hold the calls ``recent`` must refuse: waits, failures, read-only analyzes.
    calls: list[CallRecord] = Field(default_factory=list)
    # A boundary advances when the foreground changes to a non-transit app or the deterministic
    # memory context changes.  The journal is retained so ``flow save --last N`` can disclose a
    # requested crossing, but only actions stamped with the newest segment may be selected.
    capture_segment: int = 0
    capture_boundary_reason: str | None = None
    # Monotonic action watermark. Unlike ``len(recent)``, this keeps identifying new actions
    # after the rolling journal reaches its cap and drops one old entry for every new one.
    next_capture_order: int = 0
    last_goal: str | None = None  # last `goto`/find target (boosts ranked suggestions)
    active_context_id: str = DEFAULT_CONTEXT_ID
    active_flags: dict[str, str] = Field(default_factory=dict)
    pending_flags: dict[str, str] = Field(default_factory=dict)
    context_verified: bool = False
    # Which boot of the device this cursor describes; see MemoryStore.claim_session.
    instance: str | None = None

    @field_validator("pending", "recent", mode="before")
    @classmethod
    def _drop_legacy_strings(cls, v: object) -> object:
        """Pre-v2 sessions stored plain strings; drop them (pending is ephemeral)."""
        if isinstance(v, list):
            return [item for item in v if isinstance(item, dict | RouteStep)]
        return v


class RecordOutcome(NamedTuple):
    name: str
    was_known: bool
    stale: bool
    created: bool


class NavHints(NamedTuple):
    """Navigation affordances for the current screen, pushed inline into ``analyze`` meta."""

    known_routes: list[str]  # outgoing edges: ["tap 'Catalog' → catalog", ...]
    suggested_gotos: list[str]  # ranked ready-to-run: ["goto product_detail", ...]
    suggested_deeplinks: list[str]  # shortcut jumps: ["open myapp://dashboard", ...]
    map_hint: str | None  # nudge when there's a map but nothing actionable from here
    research_tasks: list[str]  # unresolved map questions ready for an external agent
    ask: dict[str, str] | None = None  # one question about THIS screen, answerable inline


# --------------------------------------------------------------------------- steps


def step_display(step: RouteStep) -> str:
    """Human/display form of one step (also the searchable text for ``--find``)."""
    kind = step.kind
    if kind in ("tap", "long-press", "clear"):
        if step.label:
            return f"{kind} '{step.label}'"
        if step.content_desc:
            return f"{kind} [desc:{step.content_desc}]"
        if step.resource_id:
            return f"{kind} [#{step.resource_id}]"
        return f"{kind} [unlabeled]"
    if kind == "input":
        return "input '<filled>'" + (" + send" if step.submit else "")
    if kind in ("key", "scroll-to", "wait-for", "assert-visible", "screenshot"):
        return f"{kind} '{step.arg}'"
    if kind == "assert":
        selector = (
            f"#{step.resource_id}"
            if step.resource_id
            else step.content_desc or step.label or "<missing selector>"
        )
        checks = ",".join(sorted(step.assertion)) or "exists"
        return f"assert '{selector}' ({checks})"
    if kind == "assert-order":
        selectors = step.assertion.get("selectors", [])
        axis = step.assertion.get("axis", "vertical")
        return f"assert-order {axis} ({len(selectors)} selectors)"
    if kind in ("swipe", "scroll"):
        return f"{kind} {step.arg}"
    if kind == "prefs-write":
        # `flow run --dry-run` is the last review before persisted app data is overwritten, so
        # the file and the keys belong in the display; the values may be app configuration and
        # are deliberately not shown.
        keys = ", ".join(sorted(step.data.get("values") or {}))
        return f"prefs-write {step.arg} ({keys})" if keys else f"prefs-write {step.arg}"
    if kind in ("launch-app", "stop-app", "open-link", "goto", "flow"):
        # Show a pinned entry Activity: on a multi-launcher build it is the difference
        # between landing in the product and landing in a developer menu, so a dry run
        # must not hide it.
        if kind == "launch-app" and step.activity:
            return f"{kind} {step.arg or ''}/{step.activity}".lstrip()
        # A bare launch_app/stop_app targets the flow's own app, so arg is legitimately
        # unset — rendering it as the literal "None" reads as a broken step in a dry run.
        return f"{kind} {step.arg}" if step.arg else kind
    return kind  # wait-stable and future kinds


def derive_action(steps: list[RouteStep], origin_package: str | None = None) -> str:
    """The edge's display string, derived deterministically from its steps.

    Join rules match the legacy format (≤3 steps → ``a + b``, else ``first … last``) so
    single-tap edges keep their exact old identity; a transit leg appends a
    ``⇢ (via <pkg>)`` suffix naming the foreign package(s) crossed.
    """
    displays = [step_display(s) for s in steps]
    if not displays:
        return ""
    base = " + ".join(displays) if len(displays) <= 3 else f"{displays[0]} … {displays[-1]}"
    vias: list[str] = []
    for s in steps:
        if s.package and s.package != origin_package and s.package not in vias:
            vias.append(s.package)
    if vias:
        base += f" ⇢ (via {', '.join(vias)})"
    return base


def _step_target(step: RouteStep) -> str | None:
    """The selector half of :func:`step_display` — an access log prints the kind separately."""
    display = step_display(step)
    prefix = f"{step.kind} "
    if display.startswith(prefix):
        return display[len(prefix) :]
    return None if display == step.kind else display


def _unquote(value: str | None) -> str | None:
    """Compare a log target with a log detail without the quoting getting in the way."""
    return value.strip("'\"") if value else value


def log_phrase(value: str | None, *, limit: int = 90) -> str | None:
    """Make one free-form phrase safe and short enough for an access-log line.

    The log records what a *call* asked for, and a wait predicate or a failure message is
    prose the tool did not choose. PII becomes ``<redacted>`` under the same test the durable
    label path uses, so the diagnostic log can never become the one place a phone number
    survives; length is bounded because a log line that wraps is a log line nobody reads.
    """
    if not value:
        return None
    phrase = " ".join(value.split())
    if not phrase:
        return None
    if _looks_pii(phrase):
        return "<redacted>"
    return phrase if len(phrase) <= limit else phrase[: limit - 1] + "…"


def as_call_record(
    step: RouteStep,
    *,
    screen: str | None = None,
    detail: str | None = None,
) -> CallRecord:
    """Project a recorded action onto its access-log line (the one conversion point).

    Keeps the two journals from drifting: a log line names its action with the same phrase
    ``--find`` and a dry run use, so it can be matched back to the recorded step without a
    second naming convention to learn.
    """
    return CallRecord(
        kind=step.kind,
        target=_step_target(step),
        started_at_ms=step.started_at_ms,
        elapsed_ms=step.elapsed_ms,
        outcome=step.outcome,
        detail=log_phrase(detail),
        package=step.package,
        screen=screen,
        capture_order=step.capture_order,
    )


def _epoch_ms_iso(epoch_ms: int | None) -> str:
    """Epoch ms → local ISO-8601 with milliseconds; ``-`` when the caller had no clock."""
    if epoch_ms is None:
        return "-"
    return datetime.fromtimestamp(epoch_ms / 1000).astimezone().isoformat(timespec="milliseconds")


def render_call_log(entries: Sequence[CallRecord | RouteStep]) -> list[str]:
    """Render a session journal as server-style access log lines, newest last.

    Request and response share one line, the way a combined web-server log writes them, so
    "what was called, when, and what came back" reads top to bottom with no join::

        2026-08-06T11:14:03.412+02:00 tap 'Continue' -> ok 412ms pkg=com.example.app screen=home
        2026-08-06T11:14:03.912+02:00 wait-for 'Welcome' -> timeout 8003ms pkg=com.example.app

    Takes either journal: :class:`CallRecord` diagnostics and the replayable
    :class:`RouteStep` records both carry the same timing fields. Pure — no I/O and no clock
    read — so the caller decides whether it is rendering a live cursor, a session file, or a
    list it built itself. An unmeasured record prints ``-`` instead of vanishing: a call that
    predates timing still happened, and hiding it would misrepresent the run.
    """
    lines: list[str] = []
    for entry in entries:
        record = entry if isinstance(entry, CallRecord) else as_call_record(entry)
        request = f"{record.kind} {record.target}" if record.target else record.kind
        answer = [
            record.outcome or "-",
            f"{record.elapsed_ms}ms" if record.elapsed_ms is not None else "-",
        ]
        answer += [
            f"{key}={value}"
            for key, value in (
                ("pkg", record.package),
                ("screen", record.screen),
                ("order", record.capture_order),
            )
            if value is not None
        ]
        detail = f" ({record.detail})" if record.detail else ""
        lines.append(
            f"{_epoch_ms_iso(record.started_at_ms)} {request} -> {' '.join(answer)}{detail}"
        )
    return lines


#: Steps that destroy state by their nature, with no label to judge them on. A tap is only
#: destructive because of what it says; wiping app data, writing to the app's database or
#: rewriting its shared preferences is destructive whatever it is called, so these can never be
#: replayed speculatively.
INHERENTLY_DESTRUCTIVE_KINDS = frozenset({"clear-data", "db-execute", "prefs-write"})


def is_destructive_step(step: RouteStep, lexicon: Sequence[str]) -> bool:
    """True when auto-replaying *step* could destroy state (guards ``goto``).

    Two rules, because there are two kinds of danger. Some steps are dangerous by *kind*: there
    is no reading of ``clear_data`` that is safe to replay while probing a route. The rest are
    dangerous by *label*, and only tap/long-press can act at all (scrolling *to* "Delete" is
    harmless).
    """
    if step.kind in INHERENTLY_DESTRUCTIVE_KINDS:
        return True
    if step.kind not in ("tap", "long-press"):
        return False
    labels = [
        value.strip().lower()
        for value in (step.label, step.content_desc)
        if value and value not in REDACT_TOKENS
    ]
    if resource := _resource_slug(step.resource_id):
        labels.append(resource.replace("_", " "))
    return any(
        re.search(rf"\b{re.escape(word.lower())}\b", label) for label in labels for word in lexicon
    )


_SAFE_GOTO_STEP_KINDS = frozenset(
    {
        "tap",
        "swipe",
        "scroll",
        "scroll-to",
        "wait-for",
        "wait-stable",
        "assert-visible",
        "assert-not-visible",
        "assert",
        "assert-order",
        "screenshot",
        "hide-keyboard",
        "a11y-scroll",
    }
)
_SETTINGS_STEP_KINDS = frozenset({"flags-apply", "dev-profile"})
_DATA_STEP_KINDS = frozenset({"input", "clear", "paste"})
_ENVIRONMENT_STEP_KINDS = frozenset(
    {
        "proxy-start",
        "proxy-stop",
        "mock-replay",
        "network-offline",
        "network-restore",
        "network-profile",
        "network-profile-restore",
    }
)
_LIFECYCLE_STEP_KINDS = frozenset({"launch-app", "stop-app"})
_EXTERNAL_URI_SCHEMES = frozenset(
    {"http", "https", "intent", "market", "mailto", "tel", "sms", "geo", "file", "content"}
)
_SETTINGS_URI_TOKENS = frozenset(
    {
        "flag",
        "flags",
        "feature",
        "features",
        "experiment",
        "experiments",
        "config",
        "configuration",
        "setting",
        "settings",
        "preference",
        "preferences",
        "prefs",
    }
)


def _deeplink_changes_settings(uri: str) -> bool:
    """Whether a URI advertises configuration mutation in its structured components."""
    try:
        parsed = urlsplit(uri)
        fields = [parsed.netloc, parsed.path, parsed.fragment]
        fields.extend(key for key, _value in parse_qsl(parsed.query, keep_blank_values=True))
    except ValueError:
        fields = [uri]
    tokens = {
        token for field in fields for token in re.split(r"[^a-z0-9]+", field.casefold()) if token
    }
    return bool(tokens & _SETTINGS_URI_TOKENS)


def route_step_risks(
    step: RouteStep,
    *,
    origin_package: str | None,
    destructive_labels: Sequence[str],
    path: str = "",
) -> list[dict[str, str]]:
    """Classify side effects that an auto-learned ``goto`` must not replay silently.

    A route edge proves that an action preceded a recognized screen; it does not prove the
    action was *only* navigation.  In particular, Android reports a configuration deeplink and
    a screen deeplink through the same ``open-link`` primitive.  This classifier is deliberately
    independent of any private app vocabulary and conservative where the route cannot prove
    intent.  Authored flows remain the explicit surface for setup/mutation journeys.
    """
    risks: list[dict[str, str]] = []

    def add(code: str, reason: str, *, where: str = path) -> None:
        item = {"code": code, "reason": reason}
        if where:
            item["path"] = where
        if item not in risks:
            risks.append(item)

    kind = step.kind.strip().lower()
    if step.package and origin_package and step.package != origin_package:
        add(
            "external_package",
            "step targets a package outside the app whose map is being replayed",
        )
    if is_destructive_step(step, destructive_labels):
        add("destructive", "label matches the configured destructive-action vocabulary")

    if kind in _SAFE_GOTO_STEP_KINDS:
        return risks
    if kind == "key":
        if (step.arg or "").strip().casefold() != "back":
            add("system_navigation", "only the Back key is a navigation-only goto step")
        return risks
    if kind == "open-link":
        uri = (step.arg or "").strip()
        try:
            scheme = urlsplit(uri).scheme.casefold() if uri else ""
        except ValueError:
            scheme = ""
        if uri and _deeplink_changes_settings(uri):
            add(
                "settings_mutation",
                "deeplink contains configuration/feature-setting parameters",
            )
        elif scheme in _EXTERNAL_URI_SCHEMES:
            add("external_navigation", "link can hand control to another app or system handler")
        else:
            add(
                "deeplink_effect",
                "a custom deeplink may navigate or mutate app state; the recorded edge "
                "cannot distinguish them",
            )
        return risks
    if kind in INHERENTLY_DESTRUCTIVE_KINDS:
        # A second, differently-coded risk on top of the `destructive` one above: the label
        # opt-in exists so an agent can allow a *known* destructive tap, and it must not double
        # as consent to rewrite persisted app data on the way to a screen.
        add("data_mutation", f"{kind} rewrites persisted app data")
        return risks
    if kind in _SETTINGS_STEP_KINDS:
        add("settings_mutation", f"{kind} changes app configuration")
        return risks
    if kind in _DATA_STEP_KINDS:
        add("data_mutation", f"{kind} changes user-visible field or clipboard state")
        return risks
    if kind in _ENVIRONMENT_STEP_KINDS:
        add("environment_mutation", f"{kind} changes the test/device environment")
        return risks
    if kind in _LIFECYCLE_STEP_KINDS:
        add("app_lifecycle", f"{kind} starts or stops an application process")
        return risks
    if kind == "tap-point":
        add("unbound_coordinate", "coordinate action has no stable semantic navigation target")
        return risks
    if kind == "long-press":
        # Long-press commonly opens a context menu, but it can also immediately mutate content;
        # unlike a normal tap, Android exposes that stronger gesture explicitly.
        add("content_interaction", "long-press is not provably navigation-only")
        return risks
    if kind in {"repeat", "retry"}:
        if not step.substeps:
            add("unknown_action", f"{kind} has no inspectable substeps")
            return risks
        for index, substep in enumerate(step.substeps):
            nested_path = f"{path}.substeps[{index}]" if path else f"substeps[{index}]"
            risks.extend(
                item
                for item in route_step_risks(
                    substep,
                    origin_package=origin_package,
                    destructive_labels=destructive_labels,
                    path=nested_path,
                )
                if item not in risks
            )
        return risks
    if kind in {"goto", "flow"}:
        add("nested_execution", f"{kind} delegates to another learned or authored journey")
        return risks
    add("unknown_action", f"{kind or 'empty'} is not a navigation-only goto step")
    return risks


def route_edge_is_safe(
    edge: RouteEdge,
    *,
    origin_package: str | None,
    destructive_labels: Sequence[str],
) -> bool:
    """Whether an edge is structured and every recorded step is navigation-only."""
    return bool(edge.steps) and not any(
        route_step_risks(
            step,
            origin_package=origin_package,
            destructive_labels=destructive_labels,
        )
        for step in edge.steps
    )


# --------------------------------------------------------------------------- redaction


def _is_input(el: Element) -> bool:
    t = (el.type or "").lower()
    return any(k in t for k in ("edittext", "textfield", "autocomplete", "searchview"))


_SECRET_HINT = re.compile(
    r"pass(word|code)|secret|otp|cvv|\bpin\b|token|credit.?card|card.?number|security.?code",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)")
_LONGNUM = re.compile(r"\d{6,}")


def _is_secret_field(el: Element) -> bool:
    rid_tail = _id_tail(el.resource_id) or ""
    rid_words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", rid_tail)
    rid_words = re.sub(r"[^A-Za-z0-9]+", " ", rid_words)
    parts = " ".join(p for p in (el.resource_id, rid_words, el.content_desc, el.type) if p)
    return bool(_SECRET_HINT.search(parts))


def _looks_pii(text: str) -> bool:
    return bool(_EMAIL.search(text) or _PHONE.search(text) or _LONGNUM.search(text))


def _looks_dynamic(text: str) -> bool:
    """Heuristic: volatile content (counts, prices, clocks, ids) we must not anchor on."""
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"v?\d+(?:[._:/-]\d+){1,}(?:[-+._a-z0-9]*)?", t, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{1,4}(?:[./-]\d{1,4}){1,2}", t):
        return True
    digits = sum(c.isdigit() for c in t)
    if digits and digits / len(t) > 0.30:
        return True
    return bool(
        re.search(r"\d{1,2}:\d{2}", t)
        or re.search(r"\b(today|yesterday|\d+\s*(min|hour|day|week)s?\s+ago)\b", t, re.I)
    )


def _capture_arrival_term_is_safe(term: CaptureArrivalTerm) -> bool:
    """Whether an automatically retained arrival term is durable and privacy-safe.

    The proof lives in the rolling local session journal, but it may later be promoted into an
    authored flow.  Apply the same conservative policy as map anchors: never promote PII,
    redaction sentinels, or obviously volatile labels/ids.  Callers can still author a more
    specific arrival predicate explicitly when volatility is intentional.
    """
    value = term.value.strip()
    return bool(
        value
        and len(value) <= 128
        and value not in REDACT_TOKENS
        and not _looks_pii(value)
        and not _looks_dynamic(value)
    )


def capture_arrival_predicate(proof: CaptureArrivalProof) -> str:
    r"""Render a capture proof into the tiny flow-arrival predicate grammar.

    Commas and backslashes are the grammar's only separators and are escaped here, so a value
    cannot smuggle an extra condition into the promoted predicate.
    """

    def render(term: CaptureArrivalTerm) -> str:
        value = term.value.replace("\\", "\\\\").replace(",", "\\,")
        prefix = "!" if term.negated else ""
        return f"{prefix}{term.by}:{value}"

    return ",".join(render(term) for term in proof.terms)


class CaptureArrivalCheck(NamedTuple):
    """Pure validation result for a capture proof considered by ``flow save``."""

    proof: CaptureArrivalProof | None
    reason: str


def capture_arrival_for_current(
    selected: Sequence[RouteStep],
    *,
    session: SessionState,
    observation_package: str | None,
    observation_fingerprint: str | None,
) -> CaptureArrivalCheck:
    """Return a captured predicate only when every binding still describes this frame.

    ``flow save`` takes a fresh hierarchy snapshot before selecting a journal suffix.  This
    helper is the single comparison point between that snapshot and a prior action-bound
    ``--until``: the proof must be on the newest selected action, the whole suffix must be from
    the same provenance tuple, the current session must still own that tuple, and the exact
    hierarchy fingerprint/package must match.  A mismatch is ordinary unverified arrival, never
    a partially trusted proof.
    """
    if not selected:
        return CaptureArrivalCheck(None, "no selected action can carry arrival proof")
    newest = selected[-1]
    proof = newest.arrival_proof
    if proof is None:
        return CaptureArrivalCheck(
            None, "newest selected action has no satisfied action-bound arrival proof"
        )
    action_provenance = (
        newest.capture_segment,
        newest.capture_order,
        newest.origin_package,
        newest.context_id,
    )
    proof_provenance = (
        proof.capture_segment,
        proof.capture_order,
        proof.origin_package,
        proof.context_id,
    )
    if action_provenance != proof_provenance:
        return CaptureArrivalCheck(None, "arrival proof does not belong to the selected action")
    segment_provenance = (
        newest.capture_segment,
        newest.origin_package,
        newest.context_id,
    )
    if any(
        (step.capture_segment, step.origin_package, step.context_id) != segment_provenance
        for step in selected
    ):
        return CaptureArrivalCheck(None, "selected actions do not share one capture provenance")
    if (
        newest.capture_segment != session.capture_segment
        or newest.origin_package != session.package
        or newest.context_id != session.active_context_id
    ):
        return CaptureArrivalCheck(None, "arrival proof belongs to an older session segment")
    if not observation_package or proof.package != observation_package:
        return CaptureArrivalCheck(None, "arrival proof package does not match the fresh snapshot")
    if not observation_fingerprint or proof.fingerprint != observation_fingerprint:
        return CaptureArrivalCheck(
            None, "arrival proof fingerprint does not match the fresh snapshot"
        )
    if not all(_capture_arrival_term_is_safe(term) for term in proof.terms):
        return CaptureArrivalCheck(None, "arrival proof contains volatile or private evidence")
    return CaptureArrivalCheck(proof, "fresh snapshot matches satisfied action-bound arrival proof")


def _id_tail(resource_id: str | None) -> str | None:
    if not resource_id:
        return None
    tail = resource_id.split("/")[-1].strip()
    return tail or None


# System chrome (status bar, nav bar) is shared across every screen, so it is a poor
# identity signal — exclude it from signatures/names. We keep it out of the *signature*
# only; ``analyze`` output is unchanged.
_SYSTEM_ID_PREFIXES = ("com.android.systemui:", "android:")


def _system_chrome(el: Element, height: int | None) -> bool:
    """True for status-bar / framework chrome (never an app's own content)."""
    rid = el.resource_id or ""
    if rid.startswith(_SYSTEM_ID_PREFIXES):
        return True
    # Status-bar band (battery/wifi/clock icons often carry only a content-desc).
    return height is not None and el.center[1] < 0.035 * height


def _bottom_nav(el: Element, height: int | None) -> bool:
    """True for the persistent bottom navigation band (shared across top-level tabs)."""
    return height is not None and el.center[1] > 0.90 * height


def redact_label(el: Element, *, redact: bool = True) -> str | None:
    """Durable, privacy-safe label for an element (never a typed value)."""
    if _is_input(el):
        hint = el.content_desc or (_id_tail(el.resource_id) or "").replace("_", " ").strip() or None
        if redact and hint and _is_secret_field(el):
            return "<redacted>"
        return hint  # the field VALUE (el.text) is deliberately never read here
    text = (el.text or el.content_desc or "").strip()
    if not text:
        return None
    if redact and (_is_secret_field(el) or _looks_pii(text)):
        return "<redacted>"
    return text[:60]


def recorded_selector(el: Element, *, elements: Sequence[Element] = ()) -> dict[str, str | None]:
    """The safest reusable selector for a newly recorded action.

    Resource ids are locale-independent and win when their tail is stable.  A content
    description is next, kept distinct so replay does not accidentally match visible copy.
    Visible text is the last semantic fallback.  PII, dynamic values and input *values* are
    never selectors; returning no selector is intentional and makes ``flow save`` refuse the
    brittle capture with edit guidance.
    """
    tail = _id_tail(el.resource_id)
    secret_field = _is_secret_field(el)
    rid_matches = [other for other in elements if _id_tail(other.resource_id) == tail]
    if (
        tail
        and not _looks_dynamic(tail)
        and not _looks_pii(tail)
        and (not elements or len(rid_matches) == 1)
    ):
        # Keep safe copy/description as human/search/destructive-risk evidence. ``by=id``
        # makes it non-selector evidence even though YAML preserves it for review.
        supplemental_desc = (el.content_desc or "").strip()
        desc = (
            supplemental_desc[:60]
            if supplemental_desc
            and not secret_field
            and not _looks_dynamic(supplemental_desc)
            and not _looks_pii(supplemental_desc)
            else None
        )
        supplemental = "" if _is_input(el) else (el.text or "").strip()
        label = (
            supplemental[:60]
            if supplemental
            and not secret_field
            and not _looks_dynamic(supplemental)
            and not _looks_pii(supplemental)
            else None
        )
        return {"resource_id": tail, "content_desc": desc, "label": label, "by": "id"}

    desc = (el.content_desc or "").strip()
    persisted_desc = desc[:60]
    desc_matches = [
        other for other in elements if (other.content_desc or "").strip()[:60] == persisted_desc
    ]
    if (
        desc
        and not _is_secret_field(el)
        and not _looks_dynamic(desc)
        and not _looks_pii(desc)
        and (not elements or len(desc_matches) == 1)
    ):
        return {
            "resource_id": None,
            "content_desc": persisted_desc,
            "label": None,
            "by": "desc",
        }

    # An EditText's visible text is the value the user typed, not the field's identity.
    text = "" if _is_input(el) else (el.text or "").strip()
    persisted_text = text[:60]
    text_matches = [
        other for other in elements if (other.text or "").strip()[:60] == persisted_text
    ]
    if (
        text
        and not _is_secret_field(el)
        and not _looks_dynamic(text)
        and not _looks_pii(text)
        and (not elements or len(text_matches) == 1)
    ):
        return {
            "resource_id": None,
            "content_desc": None,
            "label": persisted_text,
            "by": "text",
        }

    return {"resource_id": None, "content_desc": None, "label": None, "by": None}


# --------------------------------------------------------- signatures & key elements


def screen_anchors(
    elements: list[Element], *, redact: bool = True, height: int | None = None
) -> set[str]:
    """Durable identity tokens for a screen (stable ids + chrome labels + descriptions).

    System chrome (status bar) and the persistent bottom-nav band are excluded so the
    signature is driven by the screen *body* — otherwise every tab looks alike.
    """
    anchors: set[str] = set()
    for el in elements:
        if _system_chrome(el, height) or _bottom_nav(el, height):
            continue
        tail = _id_tail(el.resource_id)
        if tail and not _looks_dynamic(tail):
            anchors.add("id:" + tail.lower())
            if el.selected is True or el.checked is True:
                anchors.add("sel:" + tail.lower())
        cd = (el.content_desc or "").strip()
        if (
            cd
            and len(cd) <= 30
            and not _looks_dynamic(cd)
            and not (redact and (_is_secret_field(el) or _looks_pii(cd)))
        ):
            anchors.add("cd:" + cd.lower())
        if el.text and not _is_input(el):
            t = el.text.strip()
            if (
                1 <= len(t) <= 24
                and not _looks_dynamic(t)
                and not (redact and (_is_secret_field(el) or _looks_pii(t)))
            ):
                anchors.add("tx:" + t.lower())
    return anchors


def shell_anchors(elements: list[Element], *, height: int | None = None) -> set[str]:
    """Stable persistent-navigation anchors used to describe a context's app shell."""
    anchors: set[str] = set()
    for el in elements:
        if not _bottom_nav(el, height):
            continue
        tail = _id_tail(el.resource_id)
        label = (el.text or el.content_desc or "").strip()
        if tail and not _looks_dynamic(tail):
            anchors.add("id:" + tail.lower())
            if el.selected is True or el.checked is True:
                anchors.add("sel:" + tail.lower())
        elif label and len(label) <= 24 and not _looks_dynamic(label):
            anchors.add("tx:" + label.lower())
    return anchors


def signature(activity: str | None, anchors: set[str]) -> str:
    base = (activity or "") + "|" + "\n".join(sorted(anchors))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def anchor_similarity(a: set[str], b: set[str]) -> float:
    """Weighted overlap: resource and selected-state anchors carry more identity."""
    if not a and not b:
        return 1.0
    weights = {"sel": 5.0, "id": 4.0, "cd": 2.0, "tx": 1.0}

    def weight(anchor: str) -> float:
        return weights.get(anchor.split(":", 1)[0], 1.0)

    union = a | b
    if not union:
        return 0.0
    return sum(weight(item) for item in a & b) / sum(weight(item) for item in union)


def _is_discriminative_anchor(anchor: str) -> bool:
    """Whether an identity anchor says more than shared navigation/rendering chrome."""
    kind, _, value = anchor.partition(":")
    if kind == "id":
        return _resource_slug(value) not in _GENERIC_IDENTITY_RESOURCE_IDS
    if kind in {"cd", "tx"}:
        return value.strip().lower() not in _GENERIC_IDENTITY_LABELS
    return True


def key_elements(
    elements: list[Element], *, redact: bool = True, cap: int = 40, height: int | None = None
) -> list[KeyElement]:
    """The durable, actionable subset of a screen (nav, buttons, inputs) — deduped.

    Status-bar chrome is dropped; the bottom nav is kept (it is useful app navigation).
    """
    out: list[KeyElement] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for el in elements:
        if _system_chrome(el, height):
            continue
        is_in = _is_input(el)
        if not (el.clickable or is_in or el.resource_id):
            continue
        label = redact_label(el, redact=redact)
        tail = _id_tail(el.resource_id)
        key = (el.type, label, tail)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            KeyElement(
                type=el.type,
                label=label,
                resource_id=tail,
                clickable=el.clickable,
                input=is_in,
                value=("<filled>" if (is_in and (el.text or "").strip()) else None),
            )
        )
        if len(out) >= cap:
            break
    return out


def detect_dynamic(elements: list[Element]) -> list[str]:
    """Recognise repeated-row shapes (lists) and record them as shapes, not contents."""
    counts: Counter[str] = Counter()
    for el in elements:
        tail = _id_tail(el.resource_id)
        if tail and not _looks_dynamic(tail):
            counts[tail] += 1
    shapes = [f"{tail} list (dynamic, {n}+ items)" for tail, n in counts.items() if n >= 4]
    return shapes[:5]


def title_of(elements: list[Element], height: int | None = None) -> str | None:
    """Topmost compact, non-dynamic heading text (below the status bar).

    App and document titles routinely exceed the old 24-character anchor limit.
    Keeping those titles eligible prevents a short toolbar action such as ``Mute``
    or ``Share`` from becoming the screen name.
    """
    if not elements:
        return None
    h = height or max((e.bounds[3] for e in elements), default=1) or 1
    cands: list[tuple[int, int, str]] = []
    for el in elements:
        if _is_input(el) or _system_chrome(el, height):
            continue
        t = (el.text or el.content_desc or "").strip()
        if not t or _looks_dynamic(t) or not (2 <= len(t) <= 64):
            continue
        if el.center[1] <= 0.22 * h:
            cands.append((el.center[1], -len(t), t))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def slug(text: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return s.strip("_")[:40]


def _short(text: str | None, tokens: int = 3) -> str:
    """A short slug: first ``tokens`` words only (so a long card label → a clean name)."""
    parts = [p for p in slug(text).split("_") if p and p not in {"a", "an", "the"}]
    return "_".join(parts[:tokens])


_GENERIC_TITLES = frozenset(
    {
        "create",
        "new",
        "details",
        "detail",
        "result",
        "screen",
        "home",
        "open",
        "crear",
        "nouveau",
        "nuovo",
        # A screen is never named after the control that leaves it. `title_of` reads the
        # topmost text in the upper fifth of the display, and on a bottom sheet that band is
        # the dismiss chrome, not the heading — so a sign-up sheet reading "Create your
        # account" was recorded as the screen `cancel`. Measured 2026-08-10: an agent called
        # that "a misleading map name" and fell back to reading the raw text itself.
        "cancel",
        "close",
        "dismiss",
        "back",
        "done",
        "skip",
        "next",
        "ok",
        "x",
        "cancelar",
        "cerrar",
        "atras",
        "listo",
        "hecho",
        "omitir",
        "siguiente",
        "voltar",
        "fechar",
        "pular",
        "annuler",
        "fermer",
        "retour",
        "termine",
        "passer",
        "annulla",
        "chiudi",
        "indietro",
        "fatto",
        "avanti",
    }
)

_GENERIC_RESOURCE_FAMILIES = frozenset(
    {
        "action_bar",
        "app",
        "body",
        "content",
        "main",
        "nav_bar",
        "navigation",
        "root",
        "scaffold",
        "toolbar",
    }
)


def _resource_slug(resource_id: str | None) -> str:
    tail = _id_tail(resource_id) or ""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", tail)
    return slug(words)


def _semantic_resource_family(resource_id: str | None) -> tuple[str | None, int]:
    """Turn app-authored resource namespaces into locale-independent destinations."""
    value = _resource_slug(resource_id)
    if not value or _looks_dynamic(value):
        return None, 0
    for prefix in ("container_", "screen_"):
        if value.startswith(prefix) and len(value) > len(prefix):
            family = value.removeprefix(prefix)
            if family not in _GENERIC_RESOURCE_FAMILIES:
                return family, 5
    title_view = re.match(r"^(.+?)_title_view$", value)
    if title_view and title_view.group(1) not in _GENERIC_RESOURCE_FAMILIES:
        return title_view.group(1), 4
    structural = re.match(
        r"^(?:screen_|container_)?(.+?)_(?:root|content|screen|container|page|layout)$",
        value,
    )
    if structural:
        family = structural.group(1)
        if family not in _GENERIC_RESOURCE_FAMILIES:
            return family, 4
    return None, 0


def _destination_from_resource(resource_id: str | None) -> str | None:
    """Infer the destination represented by an action's stable resource id."""
    value = _resource_slug(resource_id)
    if not value:
        return None
    if value.startswith("bottom_bar_"):
        return value.removeprefix("bottom_bar_")
    if value.startswith("button_"):
        candidate = value.removeprefix("button_")
        if candidate in {"settings", "notifications", "profile"}:
            return candidate
    tokens = [
        token
        for token in value.split("_")
        if token
        not in {
            "button",
            "btn",
            "cta",
            "link",
            "card",
            "action",
            "open",
            "launch",
            "show",
            "view",
            "go",
            "to",
        }
    ]
    if len(tokens) >= 2:
        return "_".join(tokens)
    family, _score = _semantic_resource_family(resource_id)
    return family


_ROUTE_CONTROL_TOKENS = {
    "action",
    "bar",
    "bottom",
    "btn",
    "button",
    "card",
    "cta",
    "link",
    "nav",
    "navigation",
    "tab",
}


def _contextual_name_from_inbound(pending: list[RouteStep], title: str | None) -> str | None:
    """Derive context for a short title from stable, app-authored route selectors.

    A title such as ``Inbox`` or ``Details`` is often reused in unrelated parts of an app.
    Selectors can distinguish them without any app-specific vocabulary. For example, the
    neutral sequence ``workspaceTabMESSAGES`` -> ``workspaceAlerts`` -> title ``Inbox``
    contains the durable name ``workspace_messages_inbox``.

    The final selector must end in its action label or destination title. A random resource
    id therefore cannot outrank a good visible title.
    """
    title_name = _short(title)
    if not title_name:
        return None
    steps = [
        step
        for step in pending
        if step.kind in {"tap", "long-press", "open-link", "goto"} and step.resource_id
    ]
    if not steps:
        return None

    def semantic_tokens(step: RouteStep) -> list[str]:
        return [
            token
            for token in _resource_slug(step.resource_id).split("_")
            if token and token not in _ROUTE_CONTROL_TOKENS
        ]

    def without_suffix(tokens: list[str], value: str | None) -> tuple[list[str], bool]:
        suffix = [token for token in slug(value).split("_") if token]
        if suffix and len(tokens) > len(suffix) and tokens[-len(suffix) :] == suffix:
            return tokens[: -len(suffix)], True
        return tokens, False

    final = steps[-1]
    original = semantic_tokens(final)
    context, trimmed = without_suffix(original, final.label)
    if not trimmed:
        context, trimmed = without_suffix(original, title_name)
    if not trimmed or not context:
        return None

    for step in steps[:-1]:
        tokens = semantic_tokens(step)
        if tokens[: len(context)] == context:
            context.extend(tokens[len(context) :])

    title_tokens = title_name.split("_")
    if context[-len(title_tokens) :] != title_tokens:
        context.extend(title_tokens)
    return slug("_".join(context)) or None


def propose_name(
    *,
    hint: str | None = None,
    resource_name: str | None = None,
    inbound_label: str | None = None,
    inbound_resource_id: str | None = None,
    inbound_kind: str | None = None,
    title: str | None = None,
    activity: str | None = None,
    is_first: bool = False,
) -> str:
    # A generic/confirm/destructive inbound label ("Continue", "Delete", "Ask me later")
    # says nothing about the DESTINATION — the login screen must not be named "delete"
    # just because a Delete tap led there. Demote such labels below title/activity.
    generic = bool(inbound_label) and slug(inbound_label) in GENERIC_INBOUND
    cands: list[str] = []
    title_slug = _short(title)
    if hint:
        cands.append(slug(hint))  # an explicit name is used verbatim
    if resource_name:
        cands.append(slug(resource_name))
    if title_slug and title_slug not in _GENERIC_TITLES:
        cands.append(_short(title))
    if inbound_kind == "tap":
        cands.append(_destination_from_resource(inbound_resource_id) or "")
    if inbound_label and not generic:
        cands.append(_short(inbound_label))
    if is_first:
        cands.append("home")
    if activity:
        cands.append(_short(activity.rsplit(".", 1)[-1].replace("Activity", "")))
    if inbound_label and generic:
        cands.append(_short(inbound_label))  # last resort, still better than "screen"
    for c in cands:
        if c:
            return c
    return "screen"


def context_id_for_flags(flags: dict[str, str]) -> str:
    """A stable context id; app version is metadata, not part of the identity."""
    if not flags:
        return DEFAULT_CONTEXT_ID
    normalized = "&".join(f"{key}={flags[key]}" for key in sorted(flags))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    readable = slug(next(iter(sorted(flags))))[:18] or "flags"
    return f"flags-{readable}-{digest}"


def _stable_id(kind: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{kind}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _knowledge_id(package: str, kind: str, name: str | None, text: str) -> str:
    return _stable_id("knowledge", package, kind, name or "", text)


def _infer_state(elements: list[Element]) -> str | None:
    labels = " ".join(
        (el.text or el.content_desc or "").strip().lower()
        for el in elements
        if el.text or el.content_desc
    )
    resources = " ".join(_resource_slug(el.resource_id) for el in elements if el.resource_id)
    if re.search(r"\b(loading|please wait|creating|creando|generating)\b", labels):
        return "loading"
    if re.search(r"\b(oops|something went wrong|try again|failed|error|reintentar)\b", labels):
        return "error"
    if re.search(r"\b(no .+ yet|nothing here|empty|nada que mostrar|sin resultados)\b", labels):
        return "empty"
    if "ready" in resources or re.search(
        r"\b(tap to open|ready to open|listo para abrir)\b", labels
    ):
        return "ready"
    return None


def _screen_surface(elements: list[Element]) -> str:
    kinds = {(el.type or "").lower() for el in elements}
    if any("webview" in kind for kind in kinds):
        return "webview"
    if any(kind in {"image", "canvas"} or "surfaceview" in kind for kind in kinds):
        return "canvas"
    if any(_is_input(el) for el in elements):
        return "form"
    return "native"


def _resource_name(elements: list[Element], height: int | None = None) -> str | None:
    """Prefer stable resource namespaces over volatile or localized visible copy."""
    selected: list[str] = []
    candidates: Counter[str] = Counter()
    resource_slugs: set[str] = set()
    for el in elements:
        if _system_chrome(el, height):
            continue
        tail = _id_tail(el.resource_id)
        if not tail or _looks_dynamic(tail):
            continue
        if el.selected is True or el.checked is True:
            destination = _destination_from_resource(tail)
            if destination:
                selected.append(destination)
        if _bottom_nav(el, height):
            continue
        resource_slugs.add(_resource_slug(tail))
        family, score = _semantic_resource_family(tail)
        if family:
            candidates[family] += score
    if selected:
        return selected[0]
    if candidates:
        return candidates.most_common(1)[0][0]
    # Compose/WebView bridges often expose a semantic namespace rather than a
    # structural ``*Root`` id: e.g. ``articleDetailPoster``,
    # ``articleDetailShare``, and ``articleDetailSave``. A repeated two-token
    # prefix is a durable screen family; count distinct ids so repeated list rows
    # cannot win merely through volume.
    namespace_counts: Counter[str] = Counter()
    generic_prefixes = {
        "button",
        "icon",
        "image",
        "item",
        "label",
        "text",
        "view",
    }
    for value in resource_slugs:
        tokens = value.split("_")
        if len(tokens) >= 3 and tokens[0] not in generic_prefixes:
            namespace_counts["_".join(tokens[:2])] += 1
    namespaces = [
        (namespace, count)
        for namespace, count in namespace_counts.items()
        if count >= 2 and namespace not in _GENERIC_RESOURCE_FAMILIES
    ]
    return max(namespaces, key=lambda item: item[1])[0] if namespaces else None


# --------------------------------------------------------------------------- store


class AppStrings(BaseModel):
    """Per-locale UI labels mined from the app's string resources (`aua explore mine`).

    The cross-locale bridge for text lookups: a key (``basket_edit``) names the same
    label in every locale, so a query written in one language can be resolved to what
    the device actually renders in its own.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    package: str
    locales: list[str] = Field(default_factory=list)
    entries: dict[str, dict[str, str]] = Field(default_factory=dict)  # key → locale → text


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_activity(package: str, activity: str) -> str:
    """Expand ``pkg/.Foo`` / ``.Foo`` into a fully-qualified class name."""
    act = (activity or "").strip()
    if not act:
        return ""
    if "/" in act:
        pkg, cls = act.split("/", 1)
        if cls.startswith("."):
            return f"{pkg}{cls}"
        return cls
    if act.startswith("."):
        return f"{package}{act}"
    return act


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def upgrade_app_map(app: AppMap) -> AppMap:
    """Upgrade v1/v2 maps in memory without discarding their learned routes."""
    now = app.last_verified or _now_iso()
    original_version = app.schema_version
    pre_context = original_version < 3
    if pre_context:
        app.contexts.setdefault(
            LEGACY_CONTEXT_ID,
            ContextRecord(
                id=LEGACY_CONTEXT_ID,
                source="legacy",
                verified=True,
                first_seen=now,
                last_seen=now,
                app_version=app.app_version,
            ),
        )
    for key, rec in app.screens.items():
        if pre_context or not rec.context_id:
            rec.context_id = LEGACY_CONTEXT_ID
            rec.name_source = "legacy"
        rec.name = key
        rec.canonical_name = rec.canonical_name or key
        rec.logical_name = rec.logical_name or rec.canonical_name
        rec.id = rec.id or _stable_id("screen", app.package, rec.context_id, key)
    for index, edge in enumerate(app.routes):
        if pre_context or not edge.context_id:
            edge.context_id = LEGACY_CONTEXT_ID
        # Routes learned before provisional verification existed remain trusted.
        if original_version < 4:
            edge.status = "verified"
            edge.verification_count = max(edge.verification_count, edge.count)
        edge.id = edge.id or _stable_id(
            "route",
            app.package,
            edge.context_id,
            edge.from_screen,
            edge.action,
            edge.to_screen,
            index,
        )
    existing = {item.id for item in app.knowledge}

    def add_legacy(kind: str, text: str, *, name: str | None = None) -> None:
        kid = _knowledge_id(app.package, kind, name, text)
        if kid in existing:
            return
        app.knowledge.append(
            KnowledgeItem(
                id=kid,
                kind=kind,  # type: ignore[arg-type]
                text=text,
                name=name,
                scope=KnowledgeScope(package=app.package, context_id=LEGACY_CONTEXT_ID),
                source="legacy",
                evidence=[KnowledgeEvidence(kind="user", detail="migrated from app playbook")],
                created_at=now,
                last_verified=now,
            )
        )
        existing.add(kid)

    if pre_context:
        if app.description:
            add_legacy("description", app.description)
        for note in app.notes:
            add_legacy("note", note)
        for recipe in app.recipes:
            add_legacy("recipe", recipe.note, name=recipe.name)
        for deeplink in app.deeplinks:
            add_legacy("deeplink", deeplink.note or deeplink.uri, name=deeplink.uri)
    app.schema_version = MEMORY_SCHEMA_VERSION
    return app


class AppMemoryStore:
    """Read/write the per-app maps and the per-device navigation session."""

    def __init__(self, cfg: MemoryCfg) -> None:
        self.cfg = cfg
        self._sqlite = None
        if cfg.backend == "sqlite":
            from .memory_sqlite import SqliteMemoryBackend

            self._sqlite = SqliteMemoryBackend(
                Path(cfg.sqlite_path).expanduser(),
                migrate_from=self.memory_root(),
            )

    # -- paths (everything stays under memory.dir) ------------------------

    @property
    def base(self) -> Path:
        return Path(self.cfg.dir).expanduser()

    def memory_root(self) -> Path:
        return self.base / "memory"

    def app_dir(self, package: str) -> Path:
        return self.memory_root() / _safe(package)

    def index_path(self, package: str) -> Path:
        return self.app_dir(package) / "index.json"

    def map_path(self, package: str) -> Path:
        return self.app_dir(package) / "MAP.md"

    def session_path(self, serial: str) -> Path:
        return self.base / "state" / f"session_{_safe(serial)}.json"

    # -- app map I/O ------------------------------------------------------

    def load(self, package: str) -> AppMap | None:
        if self._sqlite is not None:
            app = self._sqlite.load_app(package)
            return upgrade_app_map(app) if app is not None else None
        path = self.index_path(package)
        if not path.is_file():
            return None
        try:
            return upgrade_app_map(AppMap.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:  # pragma: no cover - corrupt file → treat as absent
            return None

    def save(self, app: AppMap) -> None:
        app = upgrade_app_map(app)
        if self._sqlite is not None:
            self._sqlite.save_app(app)
            # Keep the human-readable MAP.md next to the legacy tree for `aua map` browsing.
            d = self.app_dir(app.package)
            d.mkdir(parents=True, exist_ok=True)
            self.map_path(app.package).write_text(
                render_map(app, detail="default"), encoding="utf-8"
            )
            return
        d = self.app_dir(app.package)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.index_path(app.package), app.model_dump_json(indent=2))
        self.map_path(app.package).write_text(render_map(app, detail="default"), encoding="utf-8")

    def list_apps(self) -> list[str]:
        if self._sqlite is not None:
            return self._sqlite.list_apps()
        root = self.memory_root()
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "index.json").is_file())

    # -- mined strings I/O --------------------------------------------------
    # Deliberately file-based in both backends: the mined-label table is a bulk artifact
    # regenerated wholesale by `explore mine`, not per-walk incremental state.

    def strings_path(self, package: str) -> Path:
        return self.app_dir(package) / "strings.json"

    def load_strings(self, package: str) -> AppStrings | None:
        path = self.strings_path(package)
        if not path.is_file():
            return None
        try:
            return AppStrings.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - corrupt file → treat as absent
            return None

    def save_strings(self, strings: AppStrings) -> None:
        d = self.app_dir(strings.package)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.strings_path(strings.package), strings.model_dump_json())

    # -- session I/O ------------------------------------------------------

    def load_session(self, serial: str) -> SessionState:
        if self._sqlite is not None:
            return self._sqlite.load_session(serial)
        path = self.session_path(serial)
        if path.is_file():
            try:
                return SessionState.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                pass
        return SessionState()

    def save_session(self, serial: str, sess: SessionState) -> None:
        if self._sqlite is not None:
            self._sqlite.save_session(serial, sess)
            return
        atomic_write_text(self.session_path(serial), sess.model_dump_json())

    def claim_session(self, serial: str, instance: str | None) -> bool:
        """Bind *serial*'s cursor to a device instance, discarding another instance's.

        `aua flow save --last N` returned steps **no lane had performed** — an `open_link`
        to a screen it never opened, a consent tap, a flag cycle it never ran, a tap
        labelled from a different test's content. Three lanes reported it independently,
        and one nearly filed a hand-written flow containing another test's journey.

        The action journal is keyed by serial, and serials come from a small pool that is
        handed out and reclaimed as workers come and go: `emulator-5554` may be one AVD,
        then a second, then that second one again for a *different* run, inside an hour.
        Per-worker `AUA_CACHE__DIR` isolation cannot help, because the memory directory is
        deliberately shared so learned routes accumulate across workers — and it should stay
        that way. Only the *session* is instance-specific.

        So the cursor now records which boot it belongs to, and a mismatch discards it.
        Everything in SessionState is instance state (cursor, pending/recent steps, active
        flag context); the learned AppMap lives elsewhere and is untouched, so a fresh
        worker still inherits every route the pool has learned.

        Returns True when a foreign session was discarded. An unreadable *instance* (None)
        changes nothing: the same rule as everywhere else here — never destroy state on the
        strength of an observation we could not make.
        """
        if instance is None:
            return False
        sess = self.load_session(serial)
        if sess.instance == instance:
            return False
        if sess.instance is None and not sess.recent and not sess.pending and not sess.package:
            # An empty cursor is nobody's; just stamp it so the next call is a no-op.
            sess.instance = instance
            self.save_session(serial, sess)
            return False
        logger.info(
            "%s is a new device instance; discarding a cursor from %s (%d recorded steps)",
            serial,
            sess.instance or "an unstamped session",
            len(sess.recent),
        )
        self.save_session(serial, SessionState(instance=instance))
        return True

    def latest_session(self, package: str) -> SessionState | None:
        """Most recently written cursor for *package* (keeps offline ``map --app`` contextual)."""
        if self._sqlite is not None:
            return self._sqlite.latest_session(package)
        root = self.base / "state"
        if not root.is_dir():
            return None
        paths = sorted(
            root.glob("session_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                session = SessionState.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if session.package == package:
                return session
        return None

    @staticmethod
    def _advance_capture_segment(sess: SessionState, reason: str) -> None:
        """Start an empty recorded-flow segment without erasing boundary evidence."""
        sess.capture_segment += 1
        sess.capture_boundary_reason = reason

    @staticmethod
    def _route_step(step: RouteStep, origin_package: str) -> RouteStep:
        """Drop session-only capture provenance and timing before a step enters the map.

        A route edge is a claim about *what to replay*, not about how long one occurrence of it
        took. Keeping the measurement would rewrite ``index.json`` with fresh latency on every
        re-walk and invite a reader to treat a single sample as a budget; the diagnostic access
        log is where a duration stays comparable to the run it came from.
        """
        update: dict[str, object] = {
            "origin_package": None,
            "context_id": None,
            "capture_segment": None,
            "capture_order": None,
            "arrival_proof": None,
            "started_at_ms": None,
            "elapsed_ms": None,
            "outcome": None,
        }
        if step.package == origin_package:
            update["package"] = None
        if step.substeps:
            update["substeps"] = [
                AppMemoryStore._route_step(substep, origin_package) for substep in step.substeps
            ]
        return step.model_copy(update=update)

    def activate_flag_context(
        self,
        serial: str,
        package: str,
        flags: dict[str, str],
        *,
        app_version: str | None = None,
        verified: bool,
        replace: bool = False,
        evidence: list[str] | None = None,
    ) -> str:
        """Promote a flag set to the active observation context after flags-set/restart."""
        sess = self.load_session(serial)
        merged = (
            {} if replace else dict(sess.active_flags) if sess.package in (None, package) else {}
        )
        merged.update(flags)
        context_id = context_id_for_flags(merged)
        context_changed = sess.active_context_id != context_id
        package_changed = sess.package not in (None, package)
        if context_changed or package_changed:
            if sess.package is not None:
                if package_changed:
                    reason = f"origin app changed from {sess.package} to {package}"
                else:
                    reason = f"memory context changed from {sess.active_context_id} to {context_id}"
                self._advance_capture_segment(sess, reason)
            sess.current_screen = None
            sess.pending = []
            sess.pending_since = None
            sess.pending_overflow = False
        sess.package = package
        sess.active_flags = merged
        sess.pending_flags = {}
        sess.active_context_id = context_id
        sess.context_verified = verified
        self.save_session(serial, sess)

        now = _now_iso()
        app = self.load(package) or AppMap(package=package)
        existing = app.contexts.get(context_id)
        if existing is None:
            app.contexts[context_id] = ContextRecord(
                id=context_id,
                flags=merged,
                app_version=app_version,
                source="flags_verified" if verified else "flags_unverified",
                verified=verified,
                evidence=evidence or [],
                first_seen=now,
                last_seen=now,
            )
        else:
            existing.flags = merged
            existing.last_seen = now
            existing.app_version = app_version or existing.app_version
            existing.verified = existing.verified or verified
            if verified:
                existing.source = "flags_verified"
            if evidence:
                existing.evidence = sorted(set(existing.evidence) | set(evidence))
        self.save(app)
        return context_id

    def set_pending_flags(self, serial: str, package: str, flags: dict[str, str]) -> None:
        """Remember a raw set-flags deeplink until the app is cold-launched."""
        if not flags:
            return
        sess = self.load_session(serial)
        if sess.package not in (None, package):
            previous_package = sess.package
            self._advance_capture_segment(
                sess,
                f"origin app changed from {previous_package} to {package}",
            )
            # A flag deeplink can retarget another app without producing an observation.
            # Treat that as the same hard journey boundary as a foreign screen so older
            # actions/pending flags cannot be attributed to the new origin.
            sess.current_screen = None
            sess.pending = []
            sess.pending_since = None
            sess.pending_overflow = False
            sess.active_flags = {}
            sess.pending_flags = {}
            sess.active_context_id = DEFAULT_CONTEXT_ID
            sess.context_verified = False
        sess.package = package
        sess.pending_flags.update(flags)
        self.save_session(serial, sess)

    def promote_pending_context(
        self, serial: str, package: str, *, app_version: str | None = None
    ) -> str | None:
        """Activate flags captured from a raw deeplink after a manual cold launch."""
        sess = self.load_session(serial)
        if sess.package != package or not sess.pending_flags:
            return None
        return self.activate_flag_context(
            serial,
            package,
            dict(sess.pending_flags),
            app_version=app_version,
            verified=False,
        )

    def clear_context(self, serial: str, package: str) -> None:
        """Forget session flag identity after app data is explicitly cleared."""
        sess = self.load_session(serial)
        owned = sess.package == package
        transit = sess.package is not None and matches_any(package, self.cfg.transit_packages)
        if not (owned or transit):
            return
        # Clearing app data is a hard, non-recorded journey boundary even when the app was
        # already in the default/no-flags context.  Actions captured before and after it must
        # never be materialized into one reusable flow.
        self._advance_capture_segment(sess, f"app data cleared for {package}")
        sess.current_screen = None
        sess.pending = []
        sess.pending_since = None
        sess.pending_overflow = False
        if owned:
            sess.active_context_id = DEFAULT_CONTEXT_ID
            sess.active_flags = {}
            sess.pending_flags = {}
            sess.context_verified = False
        self.save_session(serial, sess)

    def mark_capture_boundary(self, serial: str, package: str, reason: str) -> None:
        """Separate recorded actions around an external mutation we do not journal."""
        sess = self.load_session(serial)
        if sess.package != package and not (
            sess.package is not None and matches_any(package, self.cfg.transit_packages)
        ):
            return
        self._advance_capture_segment(sess, reason)
        sess.current_screen = None
        sess.pending = []
        sess.pending_since = None
        sess.pending_overflow = False
        self.save_session(serial, sess)

    # -- recording --------------------------------------------------------

    def _recognize(
        self,
        app: AppMap,
        anchors: set[str],
        activity: str | None,
        sig: str,
        context_id: str = DEFAULT_CONTEXT_ID,
        *,
        state: str | None = None,
        surface: str | None = None,
    ) -> tuple[str | None, float]:
        best: str | None = None
        best_sim = 0.0
        exact = [(name, rec) for name, rec in app.screens.items() if rec.context_id == context_id]
        legacy = [
            (name, rec) for name, rec in app.screens.items() if rec.context_id == LEGACY_CONTEXT_ID
        ]
        groups = [exact]
        if context_id != LEGACY_CONTEXT_ID:
            groups.append(legacy)
        for group_index, candidates in enumerate(groups):
            group_best: str | None = None
            group_sim = 0.0
            for name, rec in candidates:
                state_compatible = rec.state == state or (rec.state is None and state is None)
                surface_compatible = not rec.surface or not surface or rec.surface == surface
                recorded_anchors = set(rec.anchors)
                shared_identity = {
                    anchor
                    for anchor in anchors & recorded_anchors
                    if _is_discriminative_anchor(anchor)
                }
                # An identical hash made only from framework roots/back buttons is not screen
                # identity. Require a shared app-authored title, description, state selector,
                # or resource id even on the exact-signature fast path.
                if (
                    rec.signature == sig
                    and state_compatible
                    and surface_compatible
                    and shared_identity
                ):
                    return name, 1.0
                if not state_compatible:
                    continue
                if not shared_identity:
                    continue
                sim = anchor_similarity(anchors, recorded_anchors)
                if rec.activity and activity:
                    sim += 0.05 if rec.activity == activity else -0.10
                if not surface_compatible:
                    sim -= 0.45
                if sim > group_sim:
                    group_best, group_sim = name, sim
            threshold = _RECOGNIZE_MIN if group_index == 0 else 0.72
            if group_best is not None and group_sim >= threshold:
                return group_best, group_sim
            if group_sim > best_sim:
                best, best_sim = group_best, group_sim
        if best is not None and best_sim >= _RECOGNIZE_MIN:
            return best, best_sim
        return None, best_sim

    def _unique_name(
        self,
        app: AppMap,
        base: str,
        *,
        context_id: str = DEFAULT_CONTEXT_ID,
        context_flags: dict[str, str] | None = None,
        state: str | None = None,
    ) -> str:
        base = base or "screen"
        if base not in app.screens:
            return base
        flags = context_flags
        if flags is None and context_id in app.contexts:
            flags = app.contexts[context_id].flags
        parts: list[str] = []
        if context_id != DEFAULT_CONTEXT_ID:
            if flags:
                key = sorted(flags)[0]
                parts.append(f"{key}_{flags[key]}")
            else:
                parts.append(context_id)
        if state:
            parts.append(state)
        if parts:
            candidate = f"{base}__{slug('_'.join(parts))[:32]}"
            if candidate not in app.screens:
                return candidate
        return f"{base}__{_stable_id('variant', context_id, base)[-8:]}"

    def _rename(self, app: AppMap, old: str, new: str) -> str:
        if new == old or not new:
            return old
        new = self._unique_name(app, new, context_id=app.screens[old].context_id)
        rec = app.screens.pop(old)
        if old not in rec.aliases:
            rec.aliases.append(old)
        rec.name = new
        rec.canonical_name = new
        app.screens[new] = rec
        for e in app.routes:
            if e.from_screen == old:
                e.from_screen = new
            if e.to_screen == old:
                e.to_screen = new
        return new

    def record_screen(
        self,
        *,
        package: str,
        elements: list[Element],
        label: str | None = None,
        activity: str | None = None,
        app_version: str | None = None,
        tier: str = "hierarchy",
        name_hint: str | None = None,
        inbound_label: str | None = None,
        inbound_resource_id: str | None = None,
        inbound_kind: str | None = None,
        route_name: str | None = None,
        screen_height: int | None = None,
        context_id: str = DEFAULT_CONTEXT_ID,
        context_flags: dict[str, str] | None = None,
        context_verified: bool = False,
        ocr_helped: bool | None = None,
    ) -> RecordOutcome:
        """Record/update the current screen; return how it was identified."""
        if not self.cfg.enabled:
            return RecordOutcome(name="", was_known=False, stale=False, created=False)
        now = _now_iso()
        app = self.load(package) or AppMap(package=package)
        anchors = screen_anchors(elements, redact=self.cfg.redact, height=screen_height)
        sig = signature(activity, anchors)
        state = _infer_state(elements)
        surface = _screen_surface(elements)
        name, _sim = self._recognize(
            app,
            anchors,
            activity,
            sig,
            context_id,
            state=state,
            surface=surface,
        )
        was_known = name is not None
        created = False
        stale = False

        if name is None:
            created = True
            resource_name = _resource_name(elements, screen_height)
            title = title_of(elements, screen_height)
            base = (
                name_hint
                or resource_name
                or route_name
                or propose_name(
                    resource_name=resource_name,
                    inbound_label=inbound_label,
                    inbound_resource_id=inbound_resource_id,
                    inbound_kind=inbound_kind,
                    title=title,
                    activity=activity,
                    is_first=len(app.screens) == 0,
                )
            )
            name_source: Literal["explicit", "resource", "title", "route", "activity", "legacy"] = (
                "explicit"
                if name_hint
                else "resource"
                if resource_name or route_name
                else "title"
                if title and _short(title) not in _GENERIC_TITLES
                else "resource"
                if inbound_kind == "tap" and _destination_from_resource(inbound_resource_id)
                else "route"
                if inbound_label
                else "activity"
            )
            logical_name = slug(base) or "screen"
            name = self._unique_name(
                app,
                logical_name,
                context_id=context_id,
                context_flags=context_flags,
                state=state,
            )
            app.screens[name] = ScreenRecord(
                name=name,
                id=_stable_id("screen", package, context_id, name, sig),
                canonical_name=name,
                logical_name=logical_name,
                variant=context_id if context_id != DEFAULT_CONTEXT_ID else None,
                state=state,
                surface=surface,
                context_id=context_id,
                name_source=name_source,
                activity=activity,
                signature=sig,
                anchors=sorted(anchors),
                tier=tier,
                key_elements=key_elements(elements, redact=self.cfg.redact, height=screen_height),
                dynamic=detect_dynamic(elements),
                app_version=app_version,
                first_seen=now,
                last_seen=now,
                last_verified=now,
                hierarchy_only_ok=1 if ocr_helped is False else 0,
                ocr_helped=1 if ocr_helped is True else 0,
            )
        else:
            if name_hint and slug(name_hint) and slug(name_hint) != name:
                name = self._rename(app, name, slug(name_hint))
            rec = app.screens[name]
            resource_name = _resource_name(elements, screen_height)
            title = title_of(elements, screen_height)
            stronger_name = resource_name or (
                _short(title) if title and _short(title) not in _GENERIC_TITLES else None
            )
            source_rank = {
                "legacy": 0,
                "activity": 1,
                "route": 2,
                "title": 3,
                "resource": 4,
                "explicit": 5,
            }
            stronger_source: Literal["resource", "title"] = "resource" if resource_name else "title"
            if (
                stronger_name
                and slug(stronger_name) != (rec.logical_name or rec.name)
                and source_rank[stronger_source] > source_rank[rec.name_source]
            ):
                name = self._rename(app, name, slug(stronger_name))
                rec = app.screens[name]
                rec.logical_name = slug(stronger_name)
                rec.name_source = stronger_source
            version_changed = bool(
                app_version and rec.app_version and app_version != rec.app_version
            )
            drifted = (
                1.0 - anchor_similarity(anchors, set(rec.anchors))
            ) > self.cfg.drift_threshold
            stale = version_changed or drifted
            rec.last_seen = now
            rec.visit_count += 1
            if app_version:
                rec.app_version = app_version

            # Drift has to be recoverable. Anchors are refreshed only on a healthy visit, and
            # once a screen has drifted that visit can never arrive: every later observation is
            # compared against the same superseded anchors, diverges again, and re-flags stale.
            # A repeatedly visited screen can otherwise stay pinned to superseded anchors and
            # confirm its own staleness forever.
            #
            # So a drifted shape is held aside instead of discarded, and adopted once it turns
            # up again. Corroboration rather than the first sighting is the whole point: a
            # single odd frame is usually a keyboard, a video or a banner, and taking that as
            # identity is what produced the unmatchable anchors in the first place.
            if drifted:
                pending = set(rec.pending_anchors)
                repeats = (
                    pending
                    and (1.0 - anchor_similarity(anchors, pending)) <= self.cfg.drift_threshold
                )
                if repeats:
                    rec.pending_anchor_hits += 1
                else:
                    rec.pending_anchors = sorted(anchors)
                    rec.pending_anchor_hits = 1
                if rec.pending_anchor_hits >= _READOPT_SIGHTINGS and not version_changed:
                    stale = False
            else:
                rec.pending_anchors = []
                rec.pending_anchor_hits = 0

            rec.stale = stale
            if not stale:
                rec.last_verified = now
                rec.signature = sig
                rec.anchors = sorted(anchors)
                rec.pending_anchors = []
                rec.pending_anchor_hits = 0
                rec.tier = tier
                rec.surface = surface
                if ocr_helped is True:
                    rec.ocr_helped += 1
                elif ocr_helped is False:
                    rec.hierarchy_only_ok += 1
                if ke := key_elements(elements, redact=self.cfg.redact, height=screen_height):
                    rec.key_elements = ke
                if dyn := detect_dynamic(elements):
                    rec.dynamic = dyn
                rec.state = state
            if title:
                title_alias = _short(title)
                if (
                    title_alias
                    and title_alias not in _GENERIC_TITLES
                    and title_alias not in {rec.name, rec.logical_name, *rec.aliases}
                ):
                    rec.aliases.append(title_alias)

        context = app.contexts.get(context_id)
        if context is None:
            source: Literal["legacy", "default", "flags_verified", "flags_unverified", "agent"] = (
                "flags_verified"
                if context_flags and context_verified
                else "flags_unverified"
                if context_flags
                else "default"
            )
            context = ContextRecord(
                id=context_id,
                flags=context_flags or {},
                app_version=app_version,
                source=source,
                verified=context_verified,
                first_seen=now,
                last_seen=now,
            )
            app.contexts[context_id] = context
        context.last_seen = now
        context.app_version = app_version or context.app_version
        context.verified = context.verified or context_verified
        shell = shell_anchors(elements, height=screen_height)
        if shell:
            context.shell_anchors = sorted(set(context.shell_anchors) | shell)

        if app_version:
            app.app_version = app_version
        if label:
            app.label = label
        app.last_verified = now
        self.save(app)
        return RecordOutcome(name=name, was_known=was_known, stale=stale, created=created)

    def record_route(
        self,
        package: str,
        from_screen: str,
        to_screen: str,
        action: str | None = None,
        *,
        steps: list[RouteStep] | None = None,
        context_id: str = DEFAULT_CONTEXT_ID,
        verified: bool | None = None,
    ) -> RouteEdge | None:
        if not self.cfg.enabled or from_screen == to_screen:
            return None
        app = self.load(package)
        if app is None:
            return None
        # The session cursor keeps the name a screen had when the caller arrived, and a rename
        # moves it. Recording the stale name produced an edge whose source was not a screen —
        # marked `verified=True`, invisible until a correction refused to commit because of it.
        # A rename leaves the old name in `aliases`, so the cursor is recoverable rather than lost.
        from_screen = _resolve_screen_name(app, from_screen) or from_screen
        to_screen = _resolve_screen_name(app, to_screen) or to_screen
        if from_screen not in app.screens or to_screen not in app.screens:
            return None
        if from_screen == to_screen:
            return None
        steps = steps or []
        if action is None:
            action = derive_action(steps, package)
        if not action:
            return None
        # Explicit/manual edges without structured steps pre-date automatic observation
        # and remain trusted. Auto-recorded edges always pass an explicit verification bit.
        # ``verified is None`` is an explicitly authored/legacy edge. Automatic observation
        # always passes a bool and must be corroborated by a second consistent recording; a
        # destination merely being a known screen is not independent evidence that recognition
        # named it correctly.
        explicitly_trusted = verified is None
        is_verified = explicitly_trusted or bool(verified)
        independently_corroborated = bool(verified) and any(step.package for step in steps)
        rejection_reason = _route_rejection_reason(steps) if steps else None
        now = _now_iso()
        route_context = app.contexts.get(context_id)
        for e in app.routes:
            if (
                e.from_screen == from_screen
                and e.to_screen == to_screen
                and e.action == action
                and e.context_id == context_id
            ):
                e.count += 1
                e.last_seen = now
                if steps and not e.steps:
                    e.steps = steps  # re-walking a legacy edge upgrades it in place
                if rejection_reason is None:
                    has_conflict = _has_route_conflict(app, e)
                    if (
                        explicitly_trusted
                        or independently_corroborated
                        or (is_verified and e.count >= 2)
                    ) and not has_conflict:
                        e.status = "verified"
                        e.verification_count += 1
                        e.rejection_reason = None
                    elif e.status == "rejected" and not has_conflict:
                        e.status = "provisional"
                        e.rejection_reason = None
                else:
                    e.status = "rejected"
                    e.rejection_reason = rejection_reason
                _demote_contradicting_edges(app, e)
                self.save(app)
                return e
        edge = RouteEdge(
            id=_stable_id("route", package, context_id, from_screen, action, to_screen),
            from_screen=from_screen,
            to_screen=to_screen,
            action=action,
            context_id=context_id,
            guards=dict(route_context.flags) if route_context else {},
            steps=steps,
            status=(
                "rejected"
                if rejection_reason
                else "verified"
                if explicitly_trusted or independently_corroborated
                else "provisional"
            ),
            verification_count=(
                1
                if (explicitly_trusted or independently_corroborated) and not rejection_reason
                else 0
            ),
            rejection_reason=rejection_reason,
            last_seen=now,
        )
        app.routes.append(edge)
        _demote_contradicting_edges(app, edge)
        self.save(app)
        return edge

    def record_route_outcome(
        self, package: str, route_id: str, *, ok: bool, reached: str | None
    ) -> None:
        """Write back what replaying an edge actually did.

        A miss demotes rather than rejects, for the same reason a single sighting does not
        promote: one replay can fail on timing, a permission dialog, or an app that was simply
        somewhere else. `provisional` takes it out of pathfinding until it proves itself again,
        which is recoverable; `rejected` is a verdict this has not earned from one attempt.
        The reason carries where it really went, so the next reader can see the conflict.
        """
        if not self.cfg.enabled:
            return
        app = self.load(package)
        if app is None:
            return
        edge = next((e for e in app.routes if e.id == route_id), None)
        if edge is None:
            return
        if ok:
            if not _has_route_conflict(app, edge):
                edge.rejection_reason = None
                edge.verification_count += 1
                edge.status = "verified"
            else:
                edge.status = "provisional"
                edge.rejection_reason = "conflicting destination for the same origin and action"
        elif edge.status == "verified":
            edge.status = "provisional"
            edge.rejection_reason = f"replay landed on {reached or 'an unrecognised screen'}"
        else:
            edge.rejection_reason = f"replay landed on {reached or 'an unrecognised screen'}"
        edge.last_seen = _now_iso()
        self.save(app)

    def refresh_research_tasks(
        self, package: str, *, context_id: str | None = None
    ) -> list[dict[str, object]]:
        """Materialize current map uncertainties for two-way agent reconciliation."""
        if not self.cfg.auto_research:
            return []
        # Local import avoids a module cycle: reconciliation is layered on map memory.
        from .reconcile import ReconciliationStore

        return [
            task.model_dump(mode="json")
            for task in ReconciliationStore(self).plan(package, context_id=context_id)
        ]

    # -- playbook (durable app-level knowledge the agent reuses) ----------

    def remember_knowledge(
        self,
        package: str,
        *,
        kind: Literal["description", "note", "recipe", "deeplink", "claim"],
        text: str,
        name: str | None = None,
        context_id: str | None = None,
        app_version: str | None = None,
        flags: dict[str, str] | None = None,
        source: Literal["legacy", "user", "agent", "runtime", "source"] = "agent",
        agent: str | None = None,
        session: str | None = None,
        evidence: list[KnowledgeEvidence] | None = None,
        status: Literal["accepted", "proposed", "stale", "rejected"] = "accepted",
    ) -> KnowledgeItem | None:
        if not self.cfg.enabled or not text:
            return None
        app = self.load(package) or AppMap(package=package)
        kid = _knowledge_id(package, kind, name, text)
        now = _now_iso()
        item = next((known for known in app.knowledge if known.id == kid), None)
        if item is None:
            item = KnowledgeItem(
                id=kid,
                kind=kind,
                text=text,
                name=name,
                scope=KnowledgeScope(
                    package=package,
                    app_version=app_version,
                    context_id=context_id,
                    flags=flags or {},
                ),
                source=source,
                agent=agent,
                session=session,
                evidence=evidence or [],
                status=status,
                created_at=now,
                last_verified=now if status == "accepted" else None,
            )
            app.knowledge.append(item)
        else:
            item.status = status
            item.last_verified = now if status == "accepted" else item.last_verified
            if evidence:
                item.evidence = evidence
        self.save(app)
        return item

    def remember_deeplink(
        self,
        package: str,
        uri: str,
        note: str | None = None,
        *,
        probed: bool = False,
        landed: str | None = None,
    ) -> None:
        """Record a deeplink for an app (dedup by uri; bump count on re-use).

        *probed* marks that it was actually opened on device (mined links start False, so
        ``explore plan`` can suggest probing them); *landed* is the screen it reached.
        """
        if not self.cfg.enabled or not uri:
            return
        app = self.load(package) or AppMap(package=package)
        now = _now_iso()
        for d in app.deeplinks:
            if d.uri == uri:
                d.count += 1
                d.last_seen = now
                if note:
                    d.note = note
                if probed:
                    d.probed = True
                if landed:
                    d.landed = landed
                self._upsert_knowledge_in_map(
                    app, "deeplink", note or uri, name=uri, source="runtime" if probed else "user"
                )
                self.save(app)
                return
        app.deeplinks.append(
            Deeplink(uri=uri, note=note, probed=probed, landed=landed, last_seen=now)
        )
        self._upsert_knowledge_in_map(
            app, "deeplink", note or uri, name=uri, source="runtime" if probed else "user"
        )
        self.save(app)

    # -- learned action cost (per screen, per control) ---------------------

    def record_action_timing(
        self,
        package: str,
        *,
        screen: str,
        control: str,
        ms: float,
        outcome: str | None = None,
        context_id: str = DEFAULT_CONTEXT_ID,
        half_life: float = 4.0,
    ) -> None:
        """Remember how long *control* on *screen* took to produce its next screen.

        Scoped to ``context_id`` because the same control under a different flag set is a
        different control for timing purposes — an experiment arm that adds a network call
        changes the cost of the tap, and averaging the arms together describes neither.
        """
        if not self.cfg.enabled or ms < 0 or not (screen and control):
            return
        app = self.load(package)
        if app is None:
            return
        rec = app.screens.get(screen)
        if rec is None or rec.context_id != context_id:
            return
        prev = rec.timings.get(control)
        now = _now_iso()
        if prev is None:
            rec.timings[control] = ActionTiming(
                n=1, ema_ms=ms, max_ms=ms, last_ms=ms, last_seen=now, last_outcome=outcome
            )
        else:
            alpha = 2.0 / (half_life + 1.0)
            prev.ema_ms = alpha * ms + (1 - alpha) * prev.ema_ms
            prev.max_ms = max(prev.max_ms, ms)
            prev.last_ms = ms
            prev.n += 1
            prev.last_seen = now
            prev.last_outcome = outcome
        self.save(app)

    def action_timing(
        self,
        package: str,
        *,
        screen: str,
        control: str,
        context_id: str = DEFAULT_CONTEXT_ID,
    ) -> ActionTiming | None:
        """What we know about this control's cost, or ``None`` when we have never timed it."""
        if not self.cfg.enabled or not (screen and control):
            return None
        app = self.load(package)
        if app is None:
            return None
        rec = app.screens.get(screen)
        if rec is None or rec.context_id != context_id:
            return None
        return rec.timings.get(control)

    def slow_controls(
        self,
        package: str,
        *,
        screen: str,
        context_id: str = DEFAULT_CONTEXT_ID,
        min_ms: float = 1000.0,
        min_samples: int = 1,
        # `object` values could not be sorted or handed to `Meta.slow_controls`
        # (`list[dict[str, Any]]`) without suppressing both, so the annotation named a
        # stricter contract than either this method or its only caller actually keeps.
    ) -> list[dict[str, Any]]:
        """Controls on this screen known to be slow, worst first — for the agent, on arrival.

        ``min_ms`` deliberately excludes the ordinary sub-second tap: a list of everything is
        not a warning. ``n`` rides along so a reader can tell one observation from ten.
        """
        if not self.cfg.enabled or not screen:
            return []
        app = self.load(package)
        if app is None:
            return []
        rec = app.screens.get(screen)
        if rec is None or rec.context_id != context_id:
            return []
        # Ranked on the timing itself rather than on the rendered row: `max_ms` inside a row is
        # one of several value types, so sorting there could only be made to type-check by
        # loosening the row or suppressing the check. `t.max_ms` is a float.
        ranked = sorted(
            (
                (key, t)
                for key, t in rec.timings.items()
                if t.n >= min_samples and max(t.ema_ms, t.max_ms) >= min_ms
            ),
            key=lambda pair: pair[1].max_ms,
            reverse=True,
        )
        return [
            {
                "control": key,
                "avg_ms": round(t.ema_ms),
                "max_ms": round(t.max_ms),
                "n": t.n,
                "last_outcome": t.last_outcome,
            }
            for key, t in ranked
        ]

    def remember_note(self, package: str, note: str) -> None:
        if not self.cfg.enabled or not note:
            return
        app = self.load(package) or AppMap(package=package)
        if note not in app.notes:
            app.notes.append(note)
            self._upsert_knowledge_in_map(app, "note", note, source="user")
            self.save(app)

    def remember_launch_entry(
        self,
        package: str,
        activity: str,
        *,
        source: Literal["user", "explicit", "resolved"] = "user",
        alternatives: Sequence[str] | None = None,
    ) -> LaunchEntry | None:
        """Pin the cold-start Activity for *package*.

        A ``user`` pin is never overwritten by ``explicit`` / ``resolved`` — once someone
        (or the agent via ``remember --launch-activity``) teaches the right entry, autodetection
        must not flip it back to a coin-toss candidate.
        """
        if not self.cfg.enabled or not activity:
            return None
        app = self.load(package) or AppMap(package=package)
        existing = app.launch
        if existing is not None and existing.source == "user" and source != "user":
            # Still refresh the alternatives list when we learn more about the package.
            if alternatives is not None:
                alts = [a for a in alternatives if a and a != existing.activity]
                if alts and alts != existing.alternatives:
                    existing.alternatives = list(alts)
                    existing.last_seen = _now_iso()
                    self.save(app)
            return existing
        pinned = _normalize_activity(package, activity)
        alts = [
            _normalize_activity(package, a)
            for a in (alternatives or [])
            if a and _normalize_activity(package, a) != pinned
        ]
        # Preserve prior alternatives when the caller didn't re-query.
        if not alts and existing is not None:
            alts = [a for a in existing.alternatives if a != pinned]
        app.launch = LaunchEntry(
            activity=pinned,
            source=source,
            alternatives=alts,
            last_seen=_now_iso(),
        )
        # Ambiguity list is obsolete once we have a pin.
        if app.launcher_activities:
            app.launcher_activities = []
        self.save(app)
        return app.launch

    def record_launcher_activities(
        self, package: str, activities: Sequence[str]
    ) -> LaunchEntry | None:
        """Remember the package's MAIN/LAUNCHER set; auto-pin when unambiguous.

        Returns the (possibly new) :class:`LaunchEntry` when a pin exists after this call,
        else ``None`` — caller should surface ambiguity when ``None`` and ``len > 1``.
        """
        if not self.cfg.enabled:
            return None
        normalized = []
        seen: set[str] = set()
        for raw in activities:
            act = _normalize_activity(package, raw)
            if not act or act in seen:
                continue
            seen.add(act)
            normalized.append(act)
        app = self.load(package) or AppMap(package=package)
        if app.launch is not None:
            # Keep alternatives fresh under an existing pin.
            app.launch.alternatives = [a for a in normalized if a != app.launch.activity]
            app.launch.last_seen = _now_iso()
            app.launcher_activities = []
            self.save(app)
            return app.launch
        if len(normalized) == 1:
            app.launch = LaunchEntry(
                activity=normalized[0],
                source="resolved",
                alternatives=[],
                last_seen=_now_iso(),
            )
            app.launcher_activities = []
            self.save(app)
            return app.launch
        # 0 or many — store the set so the playbook can shout about it.
        if normalized != app.launcher_activities:
            app.launcher_activities = normalized
            self.save(app)
        return None

    def launch_activity(self, package: str) -> str | None:
        """Pinned cold-start Activity for *package*, or ``None`` if unset."""
        if not self.cfg.enabled:
            return None
        app = self.load(package)
        if app is None or app.launch is None:
            return None
        return app.launch.activity

    def remember_recipe(self, package: str, name: str, note: str) -> None:
        if not self.cfg.enabled or not (name and note):
            return
        app = self.load(package) or AppMap(package=package)
        for r in app.recipes:
            if r.name == name:
                r.note = note
                self._upsert_knowledge_in_map(app, "recipe", note, name=name, source="user")
                self.save(app)
                return
        app.recipes.append(Recipe(name=name, note=note))
        self._upsert_knowledge_in_map(app, "recipe", note, name=name, source="user")
        self.save(app)

    def set_description(self, package: str, description: str) -> None:
        if not self.cfg.enabled or not description:
            return
        app = self.load(package) or AppMap(package=package)
        app.description = description
        self._upsert_knowledge_in_map(app, "description", description, source="user")
        self.save(app)

    def _upsert_knowledge_in_map(
        self,
        app: AppMap,
        kind: Literal["description", "note", "recipe", "deeplink", "claim"],
        text: str,
        *,
        name: str | None = None,
        source: Literal["legacy", "user", "agent", "runtime", "source"] = "agent",
    ) -> KnowledgeItem:
        kid = _knowledge_id(app.package, kind, name, text)
        existing = next((item for item in app.knowledge if item.id == kid), None)
        if existing is not None:
            existing.status = "accepted"
            existing.last_verified = _now_iso()
            return existing
        now = _now_iso()
        item = KnowledgeItem(
            id=kid,
            kind=kind,
            text=text,
            name=name,
            scope=KnowledgeScope(package=app.package),
            source=source,
            evidence=[KnowledgeEvidence(kind="user" if source == "user" else "runtime")],
            created_at=now,
            last_verified=now,
        )
        app.knowledge.append(item)
        return item

    # -- auto-record orchestration (engine + daemon call these) -----------

    def observe_action(
        self,
        serial: str,
        step: RouteStep,
        *,
        started_at_ms: int | None = None,
        elapsed_ms: int | None = None,
        outcome: str | None = None,
        screen: str | None = None,
    ) -> None:
        """Remember the last state-changing action so the next analyze can draw an edge.

        The keyword arguments are what the call COST and what it answered. They are accepted
        here rather than expected on *step* because the caller that builds the step (a
        selector resolved against the hierarchy) and the caller that stops the clock (after
        the device has settled) are different places in the same command; a measurement given
        here wins over one already on the step. Timing lands both on the replayable record and
        on the diagnostic access log, so ``flow save`` and the log agree about one action.

        A caller with nothing to report gets no log line here — the engine closes its own line
        on the way out, once the cost is actually known, and a duplicate half-empty line above
        it would only make the log harder to read. See :meth:`record_call`.
        """
        if not (self.cfg.enabled and self.cfg.auto_record):
            return
        # Privacy invariant (PRD §6b): auto-recorded steps never carry a typed value.
        if step.text is not None:
            step = step.model_copy(update={"text": None})
        measured = {
            key: value
            for key, value in (
                ("started_at_ms", started_at_ms),
                ("elapsed_ms", elapsed_ms),
                ("outcome", outcome),
            )
            if value is not None
        }
        if measured:
            step = step.model_copy(update=measured)
        sess = self.load_session(serial)
        # Stamp the action before it enters either journal.  ``step.package`` is where the
        # control was physically acted on; origin/context describe the app journey that owns
        # it.  Transit packages therefore remain inside one capture segment by design.
        next_order = max(
            sess.next_capture_order,
            max(
                (
                    recorded.capture_order + 1
                    for recorded in sess.recent
                    if recorded.capture_order is not None
                ),
                default=0,
            ),
        )
        step = step.model_copy(
            update={
                "origin_package": sess.package or step.package,
                "context_id": sess.active_context_id,
                "capture_segment": sess.capture_segment,
                "capture_order": next_order,
            }
        )
        sess.next_capture_order = next_order + 1
        sess.recent.append(step)
        sess.recent = sess.recent[-_RECENT_CAP:]
        if measured:
            sess.calls.append(as_call_record(step, screen=screen))
            sess.calls = sess.calls[-_CALLS_CAP:]
        # A persistent top-level navigation control expresses a fresh destination. If sparse
        # recognition failed to advance the cursor after an earlier action, carrying the whole
        # exploratory history into this tap creates a compound route that no agent intended.
        # Keep the journal, but let the newest top-level intent supersede unresolved pending.
        if sess.pending and _is_top_level_navigation_step(step):
            sess.pending = []
            sess.pending_since = None
            sess.pending_overflow = False
        if not sess.pending:
            sess.pending_since = _now_iso()
        if len(sess.pending) >= _PENDING_CAP:
            # Stop accumulating: a truncated step list must never become an edge.
            sess.pending_overflow = True
        else:
            sess.pending.append(step)
        self.save_session(serial, sess)

    def record_call(self, serial: str, record: CallRecord) -> None:
        """Append one line to the session's diagnostic access log.

        A separate entry point from :meth:`observe_action` because most of what a diagnostic
        log is for — waits, refusals, errors, read-only analyzes — must never reach the
        replayable journal (see :class:`CallRecord`). Deliberately touches neither ``pending``
        nor the ``capture_order`` watermark: logging a call is an observation about the run,
        not a step in the journey being learned.
        """
        if not (self.cfg.enabled and self.cfg.auto_record):
            return
        sess = self.load_session(serial)
        sess.calls.append(record)
        sess.calls = sess.calls[-_CALLS_CAP:]
        self.save_session(serial, sess)

    def record_call_cost(
        self,
        serial: str,
        *,
        kind: str,
        elapsed_ms: int,
        started_at_ms: int | None = None,
        target: str | None = None,
        outcome: str | None = None,
        screen: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Say what a call answered and what it cost, on the way out.

        A combined access log writes the request and its response as one line, and it can only
        do that at the end — the cost is not knowable when the call starts. So an action opens
        its line before the device is touched (:meth:`observe_action`, which is the only moment
        the start instant and the resolved selector are both in hand) and this closes it.
        ``started_at_ms`` is the identity: it matches only the line this same call opened, so a
        call that died before reporting leaves an honestly unmeasured line rather than donating
        it to the next one.

        A call with no replayable step — every wait, and a refusal — has no open line and gets
        a whole one. That is the point: waits must never enter ``recent``/``pending`` (see
        :class:`CallRecord`), so this log is the only place a wait's cost can be recorded, and
        an unjournalled wait is usually the largest missing number in a slow run.
        """
        if not (self.cfg.enabled and self.cfg.auto_record):
            return
        sess = self.load_session(serial)
        open_line = (
            sess.calls[-1]
            if sess.calls
            and sess.calls[-1].elapsed_ms is None
            and started_at_ms is not None
            and sess.calls[-1].started_at_ms == started_at_ms
            else None
        )
        if open_line is None:
            sess.calls.append(
                CallRecord(
                    kind=kind,
                    target=log_phrase(target),
                    started_at_ms=started_at_ms,
                    elapsed_ms=elapsed_ms,
                    outcome=outcome,
                    detail=log_phrase(detail),
                    package=sess.package,
                    screen=screen,
                )
            )
            sess.calls = sess.calls[-_CALLS_CAP:]
        else:
            open_line.elapsed_ms = elapsed_ms
            open_line.outcome = outcome or open_line.outcome
            open_line.screen = screen or open_line.screen
            # Name the app even when the step itself did not: a step omits ``package`` to mean
            # "the journey's origin", which is a saving in a route and a blank in a log.
            open_line.package = open_line.package or sess.package
            # A detail that merely repeats the target says nothing twice, and a log line nobody
            # can skim is a log line nobody reads.
            phrase = log_phrase(detail)
            if phrase and _unquote(phrase) != _unquote(open_line.target):
                open_line.detail = phrase
            # Keep the replayable record and the log agreeing about the same action: `flow save`
            # and a latency review must not disagree about what one tap cost.
            if sess.recent and sess.recent[-1].capture_order == open_line.capture_order:
                sess.recent[-1] = sess.recent[-1].model_copy(
                    update={
                        "elapsed_ms": elapsed_ms,
                        "outcome": outcome or sess.recent[-1].outcome,
                    }
                )
        self.save_session(serial, sess)

    def call_log(self, serial: str, *, since_ms: int | None = None) -> list[str]:
        """The session's access log, rendered, oldest first.

        Reads the cursor rather than taking a journal argument so a caller answering "which
        call cost the eight seconds" needs one call and no knowledge of where the log lives.
        ``since_ms`` scopes it to one goal session; a line with no clock is kept either way,
        because dropping an unmeasured call would quietly shorten the run being reviewed.
        """
        if not self.cfg.enabled:
            return []
        calls = self.load_session(serial).calls
        if since_ms is not None:
            calls = [
                call
                for call in calls
                if call.started_at_ms is None or call.started_at_ms >= since_ms
            ]
        return render_call_log(calls)

    def record_action_arrival(
        self,
        serial: str,
        *,
        terms: Sequence[CaptureArrivalTerm | dict[str, object]],
        fingerprint: str | None,
        package: str | None,
    ) -> CaptureArrivalProof | None:
        """Bind satisfied UI ``--until`` evidence to the exact newest recorded action.

        This method is intentionally fail-closed and non-throwing: arrival capture is a bonus
        after an action and must never turn that already-completed action into an error.  The
        engine should first wait for outstanding memory writers, then call this only for
        ``await_outcome=satisfied`` with the typed terms from the predicate and the returned
        observation's fingerprint/package.
        """
        if not (self.cfg.enabled and self.cfg.auto_record and fingerprint and package):
            return None
        try:
            parsed_terms = [
                term
                if isinstance(term, CaptureArrivalTerm)
                else CaptureArrivalTerm.model_validate(term)
                for term in terms
            ]
        except Exception:  # malformed optional evidence must not fail the completed action
            return None
        if not parsed_terms or not all(
            _capture_arrival_term_is_safe(term) for term in parsed_terms
        ):
            return None

        sess = self.load_session(serial)
        if not sess.recent:
            return None
        newest = sess.recent[-1]
        if (
            newest.capture_order is None
            or newest.capture_segment is None
            or not newest.origin_package
            or not newest.context_id
            or newest.capture_order != sess.next_capture_order - 1
            or newest.capture_segment != sess.capture_segment
            or newest.origin_package != sess.package
            or newest.context_id != sess.active_context_id
            or package != newest.origin_package
        ):
            return None
        try:
            proof = CaptureArrivalProof(
                terms=parsed_terms,
                fingerprint=fingerprint,
                package=package,
                origin_package=newest.origin_package,
                context_id=newest.context_id,
                capture_segment=newest.capture_segment,
                capture_order=newest.capture_order,
            )
        except Exception:  # Pydantic protects the persisted journal shape
            return None

        updated = newest.model_copy(update={"arrival_proof": proof})
        sess.recent[-1] = updated
        # ``observe_screen`` normally consumes the pending step before this runs.  Preserve the
        # same proof if a passive/no-record observation left it pending, without guessing by
        # list position.
        sess.pending = [
            pending.model_copy(update={"arrival_proof": proof})
            if pending.capture_order == proof.capture_order
            else pending
            for pending in sess.pending
        ]
        self.save_session(serial, sess)
        return proof

    def observe_screen(
        self,
        serial: str,
        *,
        package: str,
        elements: list[Element],
        label: str | None = None,
        activity: str | None = None,
        app_version: str | None = None,
        tier: str = "hierarchy",
        screen_height: int | None = None,
        ocr_helped: bool | None = None,
    ) -> str | None:
        """Record the current screen + any pending route edge. Returns ``known_screen``.

        The journey cursor follows a small state machine (all state lives in the
        session file, so daemon + CLI processes agree):

        - **ignored** package (IME/system chrome) → nothing recorded, session untouched.
        - **transit** package (Google auth, permission dialogs — ``transit_packages``)
          while a different origin app owns the journey → the screen still records into
          the transit app's OWN map, but the cursor/pending stay on the origin app, so
          the eventual return records the whole excursion as ONE replayable edge.
        - **same package** → the classic case: draw the pending edge (unless it
          overflowed or went stale) and advance the cursor.
        - **foreign non-transit package** → a genuine app switch: reset the cursor,
          drop pending cleanly (never a cross-app edge).
        """
        if not (self.cfg.enabled and self.cfg.auto_record):
            return None
        if matches_any(package, self.cfg.ignore_packages):
            return None  # keyboards / system chrome never get a map of their own
        sess = self.load_session(serial)
        # Post-action loading shells are not destinations. Preserve the cursor and pending
        # action so the settled observation records one direct edge to the final screen.
        if sess.package == package and sess.pending and _infer_state(elements) == "loading":
            return None
        in_transit = (
            sess.package is not None
            and package != sess.package
            and matches_any(package, self.cfg.transit_packages)
        )
        same_context_owner = sess.package in (None, package)
        context_id = sess.active_context_id if same_context_owner else DEFAULT_CONTEXT_ID
        context_flags = sess.active_flags if same_context_owner else {}
        context_verified = sess.context_verified if same_context_owner else False
        inbound_label, inbound_kind, inbound_resource_id = _parse_inbound(list(sess.pending))
        route_name = _contextual_name_from_inbound(
            list(sess.pending), title_of(elements, screen_height)
        )
        outcome = self.record_screen(
            package=package,
            elements=elements,
            label=label,
            activity=activity,
            app_version=app_version,
            tier=tier,
            inbound_label=inbound_label,
            inbound_kind=inbound_kind,
            inbound_resource_id=inbound_resource_id,
            route_name=route_name,
            screen_height=screen_height,
            context_id=context_id,
            context_flags=context_flags,
            context_verified=context_verified,
            ocr_helped=ocr_helped,
        )
        if in_transit:
            # The origin journey continues through this screen — session untouched.
            return outcome.name if outcome.was_known else None
        prev, prev_pkg, pending = sess.current_screen, sess.package, list(sess.pending)
        if prev == outcome.name and prev_pkg == package:
            # Same screen: keep pending. An input (or a transitional frame recognised as
            # the screen we left — auth returns often flash one) must not eat the steps
            # of a journey still in flight; the cap/TTL bound any accumulation.
            self.save_session(serial, sess)
            return outcome.name if outcome.was_known else None
        route: RouteEdge | None = None
        if (
            pending
            and prev
            and prev_pkg == package
            and prev != outcome.name
            and not sess.pending_overflow
            and _pending_fresh(sess.pending_since)
        ):
            steps = [self._route_step(step, package) for step in pending]
            route = self.record_route(
                package,
                prev,
                outcome.name,
                steps=steps,
                context_id=context_id,
                verified=(
                    (outcome.was_known and not outcome.stale)
                    # A completed cross-package journey is already corroborated by
                    # multiple package-scoped observations before it returns home.
                    or any(step.package and step.package != package for step in steps)
                ),
            )
        sess.current_screen = outcome.name
        if not same_context_owner:
            old_package = sess.package
            self._advance_capture_segment(
                sess,
                f"foreground app changed from {old_package or 'unknown'} to {package}",
            )
            sess.active_context_id = DEFAULT_CONTEXT_ID
            sess.active_flags = {}
            sess.pending_flags = {}
            sess.context_verified = False
        sess.package = package
        sess.pending = []
        sess.pending_since = None
        sess.pending_overflow = False
        self.save_session(serial, sess)
        if self.cfg.auto_research and (outcome.created or outcome.stale or route is not None):
            self.refresh_research_tasks(package, context_id=context_id)
        return outcome.name if outcome.was_known else None

    def recognize_screen(
        self,
        serial: str,
        *,
        package: str | None,
        elements: list[Element],
        activity: str | None = None,
        screen_height: int | None = None,
    ) -> str | None:
        """The name this screen already has in *package*'s map, or None. Writes nothing.

        Exists so a caller that records asynchronously can still name the screen it is
        looking at. Returning a remembered name from the last observation instead would name
        the PREVIOUS screen — and after a transition that is the one answer that must never
        be given.
        """
        if not (self.cfg.enabled and self.cfg.auto_record):
            return None
        if not package or matches_any(package, self.cfg.ignore_packages):
            return None
        app = self.load(package)
        if app is None or not app.screens:
            return None
        anchors = screen_anchors(elements, redact=self.cfg.redact, height=screen_height)
        name, _sim = self._recognize(
            app,
            anchors,
            activity,
            signature(activity, anchors),
            self.load_session(serial).active_context_id,
            state=_infer_state(elements),
            surface=_screen_surface(elements),
        )
        return name

    def observe_screen_passive(
        self,
        serial: str,
        *,
        package: str | None,
        elements: list[Element],
        activity: str | None = None,
        screen_height: int | None = None,
    ) -> str | None:
        """Recognition-only recording for post-action observation snapshots.

        A snapshot may be mid-transition, so this NEVER creates screens or mutates
        records — it only recognises against the existing map. Landing on a *different*
        known screen draws the pending edge immediately (so agents that act on
        observation ids produce single-action edges, not compound monsters) and advances
        the cursor. The *same* screen keeps pending — ``input`` then ``tap Send`` must
        record as one honest two-step edge. Anything unrecognised defers to the next
        plain analyze. Returns the recognised name, if any.
        """
        if not (self.cfg.enabled and self.cfg.auto_record):
            return None
        if not package or matches_any(package, self.cfg.ignore_packages):
            return None
        sess = self.load_session(serial)
        if sess.package != package:
            if sess.package is not None and matches_any(package, self.cfg.transit_packages):
                return None  # configured transit remains part of the origin journey
            old_package = sess.package
            if old_package is not None:
                self._advance_capture_segment(
                    sess,
                    f"foreground app changed from {old_package} to {package}",
                )
            sess.package = package
            sess.current_screen = None
            sess.pending = []
            sess.pending_since = None
            sess.pending_overflow = False
            sess.active_context_id = DEFAULT_CONTEXT_ID
            sess.active_flags = {}
            sess.pending_flags = {}
            sess.context_verified = False
            self.save_session(serial, sess)
            return None
        if sess.pending and _infer_state(elements) == "loading":
            return None  # awaited/final observation must consume the route, not this shell
        app = self.load(package)
        if app is None or not app.screens:
            return None
        anchors = screen_anchors(elements, redact=self.cfg.redact, height=screen_height)
        sig = signature(activity, anchors)
        name, _sim = self._recognize(
            app,
            anchors,
            activity,
            sig,
            sess.active_context_id,
            state=_infer_state(elements),
            surface=_screen_surface(elements),
        )
        if name is None:
            return None
        prev = sess.current_screen
        if prev and name != prev:
            if sess.pending and not sess.pending_overflow and _pending_fresh(sess.pending_since):
                steps = [self._route_step(step, package) for step in sess.pending]
                self.record_route(
                    package,
                    prev,
                    name,
                    steps=steps,
                    context_id=sess.active_context_id,
                    verified=True,
                )
            sess.current_screen = name
            sess.pending = []
            sess.pending_since = None
            sess.pending_overflow = False
            self.save_session(serial, sess)
        return name

    def navigation_hints(
        self,
        serial: str,
        package: str,
        *,
        max_suggest: int = 4,
        max_research: int = 3,
        include_navigation: bool = True,
        half_life_days: float = 3.0,
        now: datetime | None = None,
    ) -> NavHints:
        """Affordances for the current screen (read-only): outgoing routes + ranked gotos.

        Reads the session cursor (just updated by :meth:`observe_screen`) for the current
        screen, then derives routes/suggestions from the stored map. Empty when no map.
        """
        empty = NavHints(
            known_routes=[],
            suggested_gotos=[],
            suggested_deeplinks=[],
            map_hint=None,
            research_tasks=[],
        )
        app = self.load(package)
        if app is None:
            return empty
        sess = self.load_session(serial)
        research_tasks = _research_prompts(app, max_research, context_id=sess.active_context_id)
        deeplinks = _suggest_deeplinks(app, max_suggest) if include_navigation else []
        if not app.screens:
            # Playbook-only (e.g. freshly mined): still offer the deeplink shortcuts.
            hint = (
                f"{len(app.deeplinks)} deeplink shortcut(s) known — see `aua about`"
                if app.deeplinks
                else None
            )
            return NavHints(
                known_routes=[],
                suggested_gotos=[],
                suggested_deeplinks=deeplinks,
                map_hint=hint,
                research_tasks=research_tasks,
            )
        current = sess.current_screen
        now = now or datetime.now().astimezone()
        adj = _adjacency(app, sess.active_context_id)
        known_routes = (
            [
                f"{e.action} → {e.to_screen}"
                for e in sorted(adj.get(current or "", []), key=lambda x: x.to_screen)
                if e.to_screen in app.screens
                and not app.screens[e.to_screen].stale
                and route_edge_is_safe(
                    e,
                    origin_package=app.package,
                    destructive_labels=self.cfg.destructive_labels,
                )
            ]
            if include_navigation
            else []
        )
        # Rank destinations by usage; prefer ones reachable from here so `goto` will work,
        # else fall back to the app's top screens (reachable from a root).
        reachable = _reachable(app, current, sess.active_context_id)
        exact_context = {
            name
            for name, rec in app.screens.items()
            if rec.context_id == sess.active_context_id and not rec.stale
        }
        legacy_fallback = {
            name
            for name, rec in app.screens.items()
            if rec.context_id == LEGACY_CONTEXT_ID and not rec.stale
        }
        visible = exact_context or legacy_fallback
        if current is None:
            reachable = {
                name
                for name in visible
                if _shortest_path(
                    app,
                    name,
                    context_id=sess.active_context_id,
                    destructive_labels=self.cfg.destructive_labels,
                    safe_only=True,
                )
            }
        # Suggestions are advertised as ready-to-run commands. Never suggest a stale screen or
        # an unreachable fallback merely because it ranks highly; `goto` would reject it and the
        # agent would learn that map hints are guesses. Legacy screens are fallback-only when the
        # active context has no healthy screens of its own.
        pool = [
            name
            for name in visible
            if name != current
            and name in reachable
            and _shortest_path(
                app,
                name,
                start=current,
                context_id=sess.active_context_id,
                destructive_labels=self.cfg.destructive_labels,
                safe_only=True,
            )
        ]
        ranked = sorted(
            pool,
            key=lambda n: _rank_score(
                app.screens[n], now=now, half_life_days=half_life_days, last_goal=sess.last_goal
            ),
            reverse=True,
        )
        suggested = (
            [f"goto {n}" for n in ranked[: max(0, max_suggest)]] if include_navigation else []
        )
        map_hint = None
        if not known_routes and not suggested and not deeplinks:
            n = len(app.screens)
            map_hint = f"{n} screen{'s' * (n != 1)} mapped — run `aua map`"
        return NavHints(
            known_routes=known_routes,
            suggested_gotos=suggested,
            suggested_deeplinks=deeplinks,
            map_hint=map_hint,
            research_tasks=research_tasks,
            ask=ask_about_current_screen(app, current, context_id=sess.active_context_id),
        )

    def set_last_goal(self, serial: str, goal: str | None) -> None:
        """Remember the last goto/find target on the session cursor (ranking boost)."""
        sess = self.load_session(serial)
        sess.last_goal = goal
        self.save_session(serial, sess)

    # -- management -------------------------------------------------------

    def forget(self, package: str, screen: str | None = None) -> dict[str, str | None]:
        if screen is None:
            deleted = False
            if self._sqlite is not None:
                deleted = self._sqlite.delete_app(package)
            d = self.app_dir(package)
            if d.is_dir():
                shutil.rmtree(d)
                deleted = True
            return {"forgot": package if deleted else None}
        app = self.load(package)
        if app and screen in app.screens:
            del app.screens[screen]
            app.routes = [
                e for e in app.routes if e.from_screen != screen and e.to_screen != screen
            ]
            self.save(app)
            return {"forgot": f"{package}/{screen}"}
        return {"forgot": None}


def _suggest_deeplinks(app: AppMap, cap: int) -> list[str]:
    """Ready-to-run deeplink shortcuts for inline `analyze` hints — concrete URIs only
    (templated ones need a value), proven (probed) ones first."""
    concrete = [
        d
        for d in app.deeplinks
        if "$" not in d.uri and "{" not in d.uri and not d.uri.rstrip().endswith("/")
    ]
    concrete.sort(key=lambda d: (not d.probed, d.uri))  # probed (known-good) first
    return [f"open {d.uri}" for d in concrete[: max(0, cap)]]


def _demote_contradicting_edges(app: AppMap, edge: RouteEdge) -> None:
    """Two destinations for one deterministic action cannot both be verified.

    The same stable selector on the same origin in the same context is deterministic. A second,
    different destination means at least one recognition was wrong, and neither edge has earned
    trust. Demoting rather than deleting keeps the evidence — the existing ``route_conflict``
    research task is what resolves which destination is real.
    """
    conflicts = [
        other
        for other in app.routes
        if other.from_screen == edge.from_screen
        and other.action == edge.action
        and other.context_id == edge.context_id
        and other.to_screen != edge.to_screen
    ]
    if not conflicts:
        return
    reason = "conflicting destination for the same origin and action"
    edge.status = "provisional"
    edge.rejection_reason = reason
    for other in conflicts:
        other.status = "provisional"
        other.rejection_reason = reason


def _has_route_conflict(app: AppMap, edge: RouteEdge) -> bool:
    return any(
        other.from_screen == edge.from_screen
        and other.action == edge.action
        and other.context_id == edge.context_id
        and other.to_screen != edge.to_screen
        for other in app.routes
    )


def _resolve_screen_name(app: AppMap, name: str | None) -> str | None:
    """The name a screen goes by now, following a rename through its aliases."""
    if not name:
        return None
    if name in app.screens:
        return name
    for key, rec in app.screens.items():
        if name in rec.aliases:
            return key
    return None


def ask_about_current_screen(
    app: AppMap, current: str | None, *, context_id: str | None = None
) -> dict[str, str] | None:
    """The one open question about the screen the caller is looking at, or nothing.

    `research_tasks` lists whatever is open, in map order, so it offers questions about screens
    the caller has never seen and cannot answer. Measured 2026-08-10: 970 open questions had
    accumulated at roughly 130 a day and not one had ever been answered.

    The agent standing on a screen is the one who knows what it is, and it is about to run
    another command anyway. So ask it here, about this screen only, and let the answer ride
    along on whatever it runs next.
    """
    resolved = _resolve_screen_name(app, current)
    if resolved is None:
        return None
    current = resolved
    rec = app.screens[current]
    # A rename stamps `explicit`, and a screen can carry several open naming tasks (one per
    # app version or flag context). Without this, answering re-asked the same question about
    # the name it had just accepted — the fastest way to teach an agent to ignore the line.
    if rec.name_source == "explicit":
        return None
    for task in app.research_tasks:
        if task.get("status") != "open" or task.get("issue_type") != "poor_name":
            continue
        affected = task.get("affected_ids")
        if rec.id not in (affected if isinstance(affected, list) else []):
            continue
        if context_id is not None and task.get("context_id") not in (
            None,
            context_id,
            LEGACY_CONTEXT_ID,
        ):
            continue
        task_id = str(task.get("id"))
        return {
            "id": task_id,
            "about": current,
            "q": f"`{current}` is a generated name — what is this screen actually for?",
            "how": f'add `--answers {task_id}="<name>"` to your next command (any command)',
        }
    return None


def _research_prompts(app: AppMap, cap: int, *, context_id: str | None = None) -> list[str]:
    prompts: list[str] = []
    for task in app.research_tasks:
        if task.get("status") != "open":
            continue
        if context_id is not None and task.get("context_id") not in (
            None,
            context_id,
            LEGACY_CONTEXT_ID,
        ):
            continue
        questions = task.get("questions")
        question = (
            str(questions[0])
            if isinstance(questions, list) and questions
            else str(task.get("observations") or "Research this map uncertainty.")
        )
        prompts.append(f"research {task.get('id')}: {question}")
        if len(prompts) >= max(0, cap):
            break
    return prompts


def _route_rejection_reason(steps: list[RouteStep]) -> str | None:
    """Why an automatically observed edge cannot safely be replayed."""
    destination_steps = [
        step for step in steps if step.kind in {"tap", "long-press", "open-link", "key", "goto"}
    ]
    if not destination_steps:
        return "no destination-producing action"
    if len(destination_steps) > _MAX_SAME_PACKAGE_DESTINATION_STEPS and not any(
        step.package for step in steps
    ):
        return "too many destination actions accumulated without a recognized screen"
    for step in destination_steps:
        if step.kind in {"tap", "long-press"}:
            if (
                step.resource_id
                or step.content_desc
                or (step.label and step.label not in REDACT_TOKENS)
            ):
                return None
        elif step.arg:
            return None
    return "destination action has no durable selector"


def _is_top_level_navigation_step(step: RouteStep) -> bool:
    """Conservative resource-id heuristic for persistent app navigation controls."""
    if step.kind != "tap" or not step.resource_id:
        return False
    resource = _resource_slug(step.resource_id)
    return resource.startswith(
        ("bottom_bar_", "bottom_nav_", "navigation_bar_", "tab_")
    ) or resource.endswith("_tab")


def _pending_fresh(since: str | None, *, now: datetime | None = None) -> bool:
    """False when the pending buffer is older than the TTL (an abandoned journey)."""
    if not since:
        return True
    try:
        started = datetime.fromisoformat(since)
    except (ValueError, TypeError):
        return True
    now = now or datetime.now().astimezone()
    return (now.timestamp() - started.timestamp()) <= _PENDING_TTL_S


def _parse_inbound(
    pending: list[RouteStep],
) -> tuple[str | None, str | None, str | None]:
    """Pull a durable (label, kind, resource-id) from destination-producing steps."""
    label: str | None = None
    kind: str | None = None
    resource_id: str | None = None
    for s in pending:
        if s.kind not in {"tap", "long-press", "open-link", "key", "goto"}:
            continue
        if s.label and s.label not in REDACT_TOKENS:
            label = s.label
        elif s.content_desc:
            label = s.content_desc
        if s.resource_id:
            resource_id = s.resource_id
        if s.label or s.content_desc or s.resource_id or s.arg:
            kind = s.kind
    return label, kind, resource_id


# --------------------------------------------------------------------------- rendering


def _routes_for_context(
    app: AppMap,
    context_id: str | None,
    *,
    include_provisional: bool = True,
    include_rejected: bool = False,
) -> list[RouteEdge]:
    def eligible(edge: RouteEdge) -> bool:
        return (include_rejected or edge.status != "rejected") and (
            include_provisional or edge.status == "verified"
        )

    if context_id is None:
        return [edge for edge in app.routes if eligible(edge)]
    exact = [edge for edge in app.routes if edge.context_id == context_id and eligible(edge)]
    if context_id == LEGACY_CONTEXT_ID:
        return exact
    exact_keys = {(edge.from_screen, edge.action) for edge in exact}
    legacy = [
        edge
        for edge in app.routes
        if edge.context_id == LEGACY_CONTEXT_ID
        and eligible(edge)
        and (edge.from_screen, edge.action) not in exact_keys
    ]
    return [*exact, *legacy]


def context_view(app: AppMap, context_id: str) -> AppMap:
    """A JSON-safe map projection with exact context plus trusted legacy fallback."""
    view = app.model_copy(deep=True)
    view.screens = {
        name: rec
        for name, rec in view.screens.items()
        if rec.context_id in (context_id, LEGACY_CONTEXT_ID)
    }
    view.routes = _routes_for_context(view, context_id)
    view.contexts = {
        key: value for key, value in view.contexts.items() if key in (context_id, LEGACY_CONTEXT_ID)
    }
    view.knowledge = [
        item
        for item in view.knowledge
        if item.scope.context_id in (None, context_id, LEGACY_CONTEXT_ID)
    ]
    return view


def _adjacency(
    app: AppMap,
    context_id: str | None = None,
    *,
    include_provisional: bool = False,
) -> dict[str, list[RouteEdge]]:
    adj: dict[str, list[RouteEdge]] = {}
    routes = _routes_for_context(app, context_id, include_provisional=include_provisional)
    destinations: dict[tuple[str, str, str], set[str]] = {}
    for edge in routes:
        key = (edge.from_screen, edge.action, edge.context_id)
        destinations.setdefault(key, set()).add(edge.to_screen)
    conflicted = {key for key, targets in destinations.items() if len(targets) > 1}
    for e in routes:
        if (e.from_screen, e.action, e.context_id) in conflicted:
            continue
        adj.setdefault(e.from_screen, []).append(e)
    return adj


def _roots(
    app: AppMap,
    context_id: str | None = None,
    *,
    include_provisional: bool = False,
) -> list[str]:
    routes = _routes_for_context(app, context_id, include_provisional=include_provisional)
    visible = {
        name
        for name, rec in app.screens.items()
        if context_id is None or rec.context_id in (context_id, LEGACY_CONTEXT_ID)
    }
    targets = {edge.to_screen for edge in routes}
    roots = [name for name in visible if name not in targets]
    if not roots:
        roots = ["home"] if "home" in visible else list(visible)[:1]
    roots.sort(key=lambda n: (0 if n == "home" else 1, app.screens[n].first_seen))
    return roots


def _rank_score(
    rec: ScreenRecord,
    *,
    now: datetime,
    half_life_days: float,
    last_goal: str | None = None,
) -> float:
    """Recency-decayed visit frequency (+last-goal boost) — 'what you're into' lately.

    ``visit_count`` weighted by an exponential decay on time since ``last_seen``, so a
    short half-life makes today's activity dominate without storing any history.
    """
    try:
        last = datetime.fromisoformat(rec.last_seen)
        age_days = max(0.0, (now.timestamp() - last.timestamp()) / 86400.0)
    except (ValueError, TypeError):
        age_days = 0.0  # missing/corrupt timestamp → treat as fresh, rank by raw frequency
    hl = half_life_days if half_life_days > 0 else 1.0
    score = rec.visit_count * (0.5 ** (age_days / hl))
    if last_goal and rec.name == last_goal:
        score += _LAST_GOAL_BOOST
    return score


def _reachable(app: AppMap, start: str | None, context_id: str | None = None) -> set[str]:
    """Screens reachable from *start* via known routes (excluding *start* itself)."""
    if not start or start not in app.screens:
        return set()
    adj = _adjacency(app, context_id)
    seen: set[str] = set()
    dq: deque[str] = deque([start])
    while dq:
        node = dq.popleft()
        for e in adj.get(node, []):
            if e.to_screen not in seen and e.to_screen != start:
                seen.add(e.to_screen)
                dq.append(e.to_screen)
    return seen


def _summarize_keys(rec: ScreenRecord) -> list[str]:
    actions: list[str] = []
    for ke in rec.key_elements:
        if ke.clickable and not ke.input and ke.label and ke.label not in actions:
            actions.append(ke.label)
    out: list[str] = []
    if actions:
        shown = " | ".join(actions[:10]) + (" …" if len(actions) > 10 else "")
        out.append(f"actions: {shown}")
    for ke in [k for k in rec.key_elements if k.input][:3]:
        val = f" ({ke.value})" if ke.value else ""
        out.append(f"input: {ke.label or 'field'}{val}")
    return out


def _header(app: AppMap, context_id: str | None = None) -> list[str]:
    lines = [f"# {app.label or app.package}  ({app.package})"]
    meta: list[str] = []
    if app.app_version:
        meta.append(f"version {app.app_version}")
    if app.last_verified:
        meta.append(f"last verified {app.last_verified}")
    if context_id:
        n = sum(rec.context_id in (context_id, LEGACY_CONTEXT_ID) for rec in app.screens.values())
        r = len(_routes_for_context(app, context_id))
    else:
        n, r = len(app.screens), len(_routes_for_context(app, None))
    meta.append(f"{n} screen{'s' * (n != 1)}, {r} route{'s' * (r != 1)}")
    if context_id:
        meta.append(f"context {context_id}")
    lines.append("_" + " · ".join(meta) + "_")
    return lines


def playbook_view(
    app: AppMap,
    *,
    context_id: str | None = None,
    max_deeplinks: int | None = None,
    max_notes: int | None = None,
) -> dict[str, Any]:
    """Current, deduplicated playbook facts for an agent-facing response.

    The legacy lists remain the storage-compatible projection, while ``knowledge`` carries
    provenance, status, version and context. Rendering the lists directly made a superseded
    recipe appear beside its replacement and made ``knowledge stale`` ineffective: the stale
    sentence was still echoed from ``app.notes``. This view merges both stores by semantic key,
    lets the newest accepted scoped fact win, and suppresses stale/out-of-scope exact copies
    without deleting their evidence.
    """

    def normal(value: str | None) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).casefold()

    def exact_key(kind: str, name: str | None, text: str) -> tuple[str, str, str]:
        return kind, normal(name), normal(text)

    def in_scope(item: KnowledgeItem) -> bool:
        scope = item.scope
        if scope.app_version and app.app_version and scope.app_version != app.app_version:
            return False
        return context_id is None or scope.context_id in (None, context_id, LEGACY_CONTEXT_ID)

    ordered = sorted(
        app.knowledge,
        key=lambda item: (item.last_verified or item.created_at, item.created_at, item.id),
    )
    suppressed = {
        exact_key(item.kind, item.name, item.text)
        for item in ordered
        if item.status != "accepted" or not in_scope(item)
    }
    active = [item for item in ordered if item.status == "accepted" and in_scope(item)]

    description = (
        app.description
        if app.description and exact_key("description", None, app.description) not in suppressed
        else None
    )
    recipes: dict[str, Recipe] = {
        normal(recipe.name): recipe
        for recipe in app.recipes
        if exact_key("recipe", recipe.name, recipe.note) not in suppressed
    }
    deeplinks: dict[str, Deeplink] = {
        normal(link.uri): link
        for link in app.deeplinks
        if exact_key("deeplink", link.uri, link.note or link.uri) not in suppressed
    }
    notes: dict[str, str] = {
        normal(note): note for note in app.notes if exact_key("note", None, note) not in suppressed
    }

    for item in active:
        if item.kind == "description":
            description = item.text
        elif item.kind == "recipe" and item.name:
            recipes[normal(item.name)] = Recipe(name=item.name, note=item.text)
        elif item.kind == "deeplink" and item.name:
            prior = deeplinks.get(normal(item.name))
            deeplinks[normal(item.name)] = Deeplink(
                uri=item.name,
                note=None if item.text == item.name else item.text,
                count=prior.count if prior else 1,
                probed=prior.probed if prior else False,
                landed=prior.landed if prior else None,
                last_seen=prior.last_seen if prior else item.last_verified,
            )
        elif item.kind in {"note", "claim"}:
            notes[normal(item.text)] = item.text

    links = sorted(
        deeplinks.values(),
        key=lambda link: (
            not link.probed,
            "$" in link.uri or "{" in link.uri,
            -link.count,
            link.uri,
        ),
    )
    note_values = list(notes.values())
    total_links, total_notes = len(links), len(note_values)
    if max_deeplinks is not None:
        links = links[: max(0, max_deeplinks)]
    if max_notes is not None:
        note_values = note_values[: max(0, max_notes)]
    return {
        "description": description,
        "recipes": list(recipes.values()),
        "deeplinks": links,
        "notes": note_values,
        "counts": {
            "recipes": len(recipes),
            "deeplinks": total_links,
            "notes": total_notes,
            "stale_or_scoped_out": len(suppressed),
        },
    }


def _launch_lines(app: AppMap) -> list[str]:
    """How this app cold-starts: the pinned entry Activity, or the unresolved ambiguity.

    Kept out of :func:`playbook_view` because a launch pin is not a knowledge item — it has no
    provenance/staleness lifecycle, it is a single durable setting that changes what bare
    ``aua app launch`` does. Surfaced first because every journey starts with a launch.
    """
    if app.launch is not None:
        alts = (
            f" (also declares: {', '.join(f'`{a}`' for a in app.launch.alternatives)})"
            if app.launch.alternatives
            else ""
        )
        return [
            f"- launch: `{app.launch.activity}`"
            f" — pinned ({app.launch.source}); bare `aua app launch` uses this{alts}"
        ]
    if len(app.launcher_activities) > 1:
        listed = ", ".join(f"`{a}`" for a in app.launcher_activities)
        return [
            f"- launch AMBIGUOUS: {listed} — pin with "
            "`aua remember --launch-activity <Activity>` or "
            "`aua app launch <pkg> --activity <Activity>` before trusting cold starts"
        ]
    return []


def launch_payload(app: AppMap) -> dict[str, Any]:
    """The cold-start entry as ``about``-style JSON, so agents can branch on it structurally.

    ``{"launch": {...}}`` once an entry is pinned, ``{"launch_ambiguous": [...]}`` while a
    multi-launcher build has none, and ``{}`` when there is nothing to report — callers merge it
    into their payload. The markdown twin is :func:`_launch_lines`.
    """
    if app.launch is not None:
        return {
            "launch": {
                "activity": app.launch.activity,
                "source": app.launch.source,
                "alternatives": list(app.launch.alternatives),
            }
        }
    if len(app.launcher_activities) > 1:
        return {"launch_ambiguous": list(app.launcher_activities)}
    return {}


def _playbook_lines(app: AppMap, *, context_id: str | None = None) -> list[str]:
    """The current app playbook, deduplicated through :func:`playbook_view`."""
    view = playbook_view(app, context_id=context_id)
    launch = _launch_lines(app)
    # A launch pin (or an unresolved multi-launcher build) is worth a playbook on its own —
    # it decides which screen every later step starts from.
    if not (any(view[key] for key in ("description", "deeplinks", "recipes", "notes")) or launch):
        return []
    lines = ["## Playbook"]
    if view["description"]:
        lines.append(str(view["description"]))
    lines.extend(launch)
    for r in view["recipes"]:
        lines.append(f"- recipe `{r.name}`: {r.note}")
    for d in view["deeplinks"]:
        lines.append(f"- deeplink `{d.uri}`" + (f" — {d.note}" if d.note else ""))
    for note in view["notes"]:
        lines.append(f"- note: {note}")
    return lines


def render_map(
    app: AppMap,
    *,
    detail: str = "default",
    find: str | None = None,
    screen: str | None = None,
    depth: int | None = None,
    context_id: str | None = None,
    all_contexts: bool = True,
) -> str:
    """The single source for ``MAP.md`` and every ``aua map`` view (PRD §6b)."""
    selected_context = None if all_contexts else (context_id or DEFAULT_CONTEXT_ID)
    if find:
        return _render_find(app, find, selected_context)
    if screen:
        return _render_screen_detail(app, screen)

    lines = [*_header(app, selected_context), ""]
    contexts = [
        context
        for context in app.contexts.values()
        if selected_context is None or context.id in (selected_context, LEGACY_CONTEXT_ID)
    ]
    if contexts:
        lines.append("## Contexts")
        for context in sorted(contexts, key=lambda item: item.id):
            status = "verified" if context.verified else "unverified"
            flags = (
                ", ".join(f"{key}={value}" for key, value in sorted(context.flags.items()))
                or "no flags"
            )
            lines.append(f"- {context.id}  ({status}, {context.source}) — {flags}")
            if detail != "brief" and context.shell_anchors:
                lines.append(f"    - shell: {' | '.join(context.shell_anchors)}")
            if detail != "brief" and context.evidence:
                lines.append(f"    - evidence: {' | '.join(context.evidence)}")
        lines.append("")
    if playbook := _playbook_lines(app, context_id=selected_context):
        lines.extend([*playbook, ""])
    if not app.screens:
        lines.append("_(no screens recorded yet — run `aua analyze` while navigating)_")
        return "\n".join(lines) + "\n"

    visible = [
        rec
        for rec in app.screens.values()
        if selected_context is None or rec.context_id in (selected_context, LEGACY_CONTEXT_ID)
    ]
    lines.append("## Screens")
    grouped: dict[str, list[ScreenRecord]] = {}
    for rec in visible:
        grouped.setdefault(rec.logical_name or rec.canonical_name or rec.name, []).append(rec)
    for logical, variants in sorted(grouped.items()):
        multiple = len(variants) > 1
        if multiple:
            lines.append(f"{logical}")
        for rec in sorted(variants, key=lambda item: (item.context_id, item.name)):
            indent = "    " if multiple else ""
            labels = [f"tier: {rec.tier}", f"context: {rec.context_id}"]
            if rec.variant:
                labels.append(f"variant: {rec.variant}")
            if rec.state:
                labels.append(f"state: {rec.state}")
            if rec.surface:
                labels.append(f"surface: {rec.surface}")
            if rec.stale:
                labels.append("STALE")
            display = f"- {rec.name}" if multiple else rec.name
            lines.append(f"{indent}{display}  ({', '.join(labels)})")
            if detail != "brief":
                lines.extend(f"{indent}    - {summary}" for summary in _summarize_keys(rec))
                lines.extend(f"{indent}    - {shape}" for shape in rec.dynamic)

    routes = _routes_for_context(app, selected_context)
    if routes:
        lines.append("")
        lines.append("## Routes")
        by_context: dict[str, list[RouteEdge]] = {}
        for edge in routes:
            by_context.setdefault(edge.context_id, []).append(edge)
        for route_context, edges in sorted(by_context.items()):
            if len(by_context) > 1 or selected_context is None:
                lines.append(f"### {route_context}")
            for edge in sorted(
                edges, key=lambda item: (item.from_screen, item.action, item.to_screen)
            ):
                guard = (
                    "  [" + ", ".join(f"{k}={v}" for k, v in sorted(edge.guards.items())) + "]"
                    if edge.guards
                    else ""
                )
                status = "  [provisional]" if edge.status == "provisional" else ""
                lines.append(
                    f"{edge.from_screen} --{edge.action}--> {edge.to_screen}{guard}{status}"
                )
    open_tasks = [
        task
        for task in app.research_tasks
        if task.get("status") == "open"
        and (
            selected_context is None
            or task.get("context_id") in (None, selected_context, LEGACY_CONTEXT_ID)
        )
    ]
    if open_tasks and detail != "brief":
        lines.extend(["", "## Research needed"])
        for task in open_tasks:
            questions = task.get("questions")
            question = (
                str(questions[0])
                if isinstance(questions, list) and questions
                else str(task.get("observations") or "Research this map uncertainty.")
            )
            lines.append(f"- `{task.get('id')}` ({task.get('issue_type')}): {question}")
    return "\n".join(lines).rstrip() + "\n"


def _shortest_path(
    app: AppMap,
    target: str,
    start: str | None = None,
    context_id: str | None = None,
    *,
    include_provisional: bool = False,
    exclude_route_ids: set[str] | None = None,
    destructive_labels: Sequence[str] = (),
    safe_only: bool = False,
) -> list[RouteEdge]:
    """Safest shortest route to *target*; ``[]`` if already there / no path.

    With *start* given, search only from that screen (used by ``goto`` from the agent's
    current position); otherwise search from roots, then any screen (the original behaviour).
    Navigation-only routes are searched first, even when an unsafe route is shorter or more
    travelled. If none exists, the ordinary shortest route remains discoverable so callers can
    return a plan/risk preview. ``safe_only`` suppresses that fallback for surfaces which
    advertise a route as immediately executable.

    Pre-v2 routes have no structured steps to classify, so they participate only in the
    compatibility fallback; execution-time parsing and guards still decide whether they run.
    """
    adj = _adjacency(app, context_id, include_provisional=include_provisional)
    excluded = exclude_route_ids or set()
    if start is not None:
        if start == target or start not in app.screens:
            return []
        starts = [start]
    else:
        roots = _roots(app, context_id, include_provisional=include_provisional)
        if target in roots:
            return []
        visible = [
            name
            for name, rec in app.screens.items()
            if context_id is None or rec.context_id in (context_id, LEGACY_CONTEXT_ID)
        ]
        starts = roots + [name for name in visible if name not in roots]

    def edge_is_safe(edge: RouteEdge) -> bool:
        return route_edge_is_safe(
            edge,
            origin_package=app.package,
            destructive_labels=destructive_labels,
        )

    def search(*, require_safe: bool) -> list[RouteEdge] | None:
        for origin in starts:
            visited = {origin}
            queue: deque[tuple[str, list[RouteEdge]]] = deque([(origin, [])])
            while queue:
                node, path = queue.popleft()
                # Among parallel edges to the same screen, prefer a replayable
                # (steps-bearing) one, then the most-travelled. Safety is a whole-path
                # constraint, not a tie-breaker: the safe BFS completes before fallback.
                edges = sorted(
                    (
                        edge
                        for edge in adj.get(node, [])
                        if edge.id not in excluded and (not require_safe or edge_is_safe(edge))
                    ),
                    key=lambda e: (not e.steps, -e.count, e.action),
                )
                for edge in edges:
                    if edge.to_screen in visited:
                        continue
                    new_path = path + [edge]
                    if edge.to_screen == target:
                        return new_path
                    visited.add(edge.to_screen)
                    queue.append((edge.to_screen, new_path))
        return None

    safe = search(require_safe=True)
    if safe is not None:
        return safe
    if safe_only:
        return []
    fallback = search(require_safe=False)
    if fallback is not None:
        return fallback
    return []


def _format_path(path: list[RouteEdge]) -> str:
    parts = [path[0].from_screen]
    parts += [f"--{e.action}--> {e.to_screen}" for e in path]
    return " ".join(parts)


_ARRIVAL_NOISE = frozenset(
    {
        "a",
        "an",
        "and",
        "app",
        "navigate",
        "open",
        "page",
        "reach",
        "screen",
        "the",
        "to",
        "verify",
    }
)


def _search_words(value: str) -> list[str]:
    return [word for word in re.split(r"\W+", value.casefold()) if word]


# Words that describe *doing* something rather than the thing being reached. A goal is a
# sentence, so it carries far more of these than a keyword query does, and every one of them
# used to be a hard requirement.
# Words that describe the *asking* rather than the thing asked for: auxiliaries, the verbs of
# testing, and the words that mean "some screen" rather than any screen in particular. A goal is
# a sentence, so it carries dozens of these; a keyword query carries none. Every one of them used
# to be a hard requirement, which is what made a goal unsearchable.
#
# Stripping is safe by construction: it only ever *removes* a conjunct from an all-terms-must-
# match test, so it can never turn a hit into a miss. Only precision is at stake, and precision
# is still held by requiring every surviving term.
_SEARCH_NOISE = _ARRIVAL_NOISE | frozenset(
    {
        "a",
        "actually",
        "again",
        "an",
        "and",
        "app",
        "appear",
        "appears",
        "are",
        "as",
        "assert",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "check",
        "checks",
        "choose",
        "click",
        "clicks",
        "confirm",
        "confirms",
        "correctly",
        "did",
        "displayed",
        "displays",
        "do",
        "does",
        "ensure",
        "enter",
        "exist",
        "exists",
        "expect",
        "expects",
        "find",
        "finds",
        "for",
        "from",
        "get",
        "gets",
        "go",
        "goes",
        "got",
        "had",
        "has",
        "have",
        "in",
        "inside",
        "inspect",
        "into",
        "is",
        "it",
        "its",
        "load",
        "loaded",
        "loading",
        "loads",
        "look",
        "looks",
        "must",
        "navigate",
        "of",
        "on",
        "once",
        "open",
        "opens",
        "page",
        "present",
        "press",
        "presses",
        "properly",
        "reach",
        "reaches",
        "really",
        "renders",
        "return",
        "returns",
        "screen",
        "see",
        "seen",
        "sees",
        "select",
        "selects",
        "should",
        "show",
        "showing",
        "shown",
        "shows",
        "so",
        "still",
        "tap",
        "taps",
        "test",
        "tests",
        "that",
        "the",
        "their",
        "then",
        "there",
        "this",
        "through",
        "to",
        "ui",
        "up",
        "using",
        "validate",
        "verify",
        "via",
        "view",
        "visible",
        "visit",
        "was",
        "were",
        "while",
        "will",
        "with",
        "within",
        "work",
        "working",
        "works",
    }
)

NO_TARGET_MATCH = 10_000


def search_terms(query: str) -> list[str]:
    """The content words of *query* — what the caller is looking for, minus how they asked.

    ``_query_match`` used to require every word of the query to appear, which is right for a
    keyword query and fatal for a goal. Measured 2026-08-18 on a 6-screen map: ``resolve_goal``
    answered 2 of 9 plainly-worded goals, and returned *nothing* for "open the settings screen"
    against a map holding a screen named ``settings`` — because "open" appears nowhere on it.
    Falls back to the raw terms when a query is nothing but noise, so ``--find "open"`` still
    searches for the word rather than for everything.
    """
    terms = arrival_destination_terms(query)
    return [term for term in terms if term not in _SEARCH_NOISE] or terms


def _query_match(haystack: str, query: str, terms: Sequence[str]) -> int | None:
    """Return phrase/term match quality for one search lane; lower is stronger.

    Every term still has to appear. What changed is which words count as terms: ``search_terms``
    drops the ones that describe the asking rather than the target.
    """
    candidate = haystack.casefold()
    if query and query in candidate:
        return 0
    if terms and all(term in candidate for term in terms):
        return 1
    return None


def _arrival_value_matches(value: str, goal_terms: Sequence[str]) -> bool:
    """Whether one identity value is specifically represented in a broader goal."""
    value_terms = [term for term in _search_words(value) if term not in _ARRIVAL_NOISE]
    if not value_terms or not goal_terms:
        return False
    value_phrase = " ".join(value_terms)
    goal_phrase = " ".join(goal_terms)
    if value_phrase in goal_phrase or goal_phrase in value_phrase:
        return True
    value_set = set(value_terms)
    goal_set = set(goal_terms)
    return value_set <= goal_set or goal_set <= value_set


def arrival_destination_terms(goal: str) -> list[str]:
    """Prefer the object of the requested navigation/assertion over origin context.

    A goal such as ``open saved items in Settings`` contains two screen-like phrases. The
    destination is ``saved items``; letting the trailing origin/container name prove arrival
    is how a clickable launcher row was previously mistaken for an already-open destination.
    """
    match = re.search(
        r"\b(?:open|tap|press|click|reach|enter|visit|view|inspect|verify|select|choose|"
        r"navigate(?:\s+once)?\s+to|go\s+to|return\s+to)\s+(?:the\s+)?"
        r"(?P<destination>.+?)"
        r"(?=\s+(?:and\s+(?:verify|prove|confirm|assert|preview|check|open|tap|select|then)\b|"
        r"then\b|while\b|using\b|through\b|via\b|from\b|among\b|within\b|inside\b|in\b)|"
        r"[;,.]|$)",
        goal,
        flags=re.IGNORECASE,
    )
    value = match.group("destination") if match is not None else goal
    raw_terms = _search_words(value)
    return [term for term in raw_terms if term not in _ARRIVAL_NOISE] or raw_terms


def _target_search_lanes(
    app: AppMap,
    name: str,
    context_id: str | None,
) -> tuple[str, str, str, str, str]:
    """Identity, passive anchors, incoming routes, controls, and fallback shape.

    A clickable row named after another screen describes how to *leave* the current screen;
    it is not current-screen identity. Keep it searchable, but rank the incoming route whose
    destination it names ahead of that launcher row. This distinction is also what prevents a
    fuzzy ``goto`` from resolving its destination to the screen that merely links there.
    """
    rec = app.screens[name]
    identities = [
        name,
        rec.canonical_name or "",
        rec.logical_name or "",
        *rec.aliases,
    ]
    passive_keys = [
        value
        for key in rec.key_elements
        if not key.clickable
        for value in (key.label or "", key.resource_id or "")
        if value
    ]
    controls = [
        value
        for key in rec.key_elements
        if key.clickable
        for value in (key.label or "", key.resource_id or "")
        if value
    ]
    clickable_tokens = {" ".join(_search_words(value)) for value in controls if value}
    passive_anchors: list[str] = []
    clickable_anchors: list[str] = []
    for anchor in rec.anchors:
        _kind, _sep, value = anchor.partition(":")
        normalized = " ".join(_search_words(value))
        if normalized and normalized in clickable_tokens:
            clickable_anchors.append(anchor)
        else:
            passive_anchors.append(anchor)
    incoming = [
        edge.action for edge in _routes_for_context(app, context_id) if edge.to_screen == name
    ]
    return (
        " ".join(identities),
        " ".join([*identities, *passive_keys, *passive_anchors]),
        " ".join([*identities, *passive_keys, *passive_anchors, *incoming]),
        " ".join(
            [
                *identities,
                *passive_keys,
                *passive_anchors,
                *incoming,
                *controls,
                *clickable_anchors,
            ]
        ),
        " ".join(
            [
                *identities,
                *passive_keys,
                *passive_anchors,
                *incoming,
                *controls,
                *clickable_anchors,
                *rec.dynamic,
            ]
        ),
    )


def _target_match_rank(
    app: AppMap,
    name: str,
    query: str,
    context_id: str | None,
    terms: Sequence[str] | None = None,
) -> int | None:
    if terms is None:
        terms = search_terms(query)
    phrase = " ".join(terms)
    raw = query.casefold().strip()
    for lane, haystack in enumerate(_target_search_lanes(app, name, context_id)):
        quality = _query_match(haystack, phrase, terms)
        if quality is None and raw and raw in haystack.casefold():
            quality = 0
        if quality is not None:
            return lane * 2 + quality
    return None


def target_arrival_evidence(
    app: AppMap,
    target: str,
    goal: str,
    elements: Sequence[Element],
    *,
    screen_height: int | None = None,
) -> dict[str, str] | None:
    """Return target-specific proof for an already-recognised mapped screen.

    Callers must still establish that current recognition named *target*. This pure helper
    answers the second, independent question: does that target identity actually describe the
    requested *goal*? A fresh canonical/logical/name/alias can answer yes. So can a current,
    non-clickable title or durable anchor. A clickable matching control never can: it normally
    names the destination reached *after* tapping it, which was the source of false
    ``already_there`` results.
    """
    rec = app.screens.get(target)
    terms = arrival_destination_terms(goal)
    if rec is None or not terms:
        return None

    clickable_goal_match = any(
        _arrival_value_matches(value, terms)
        for element in elements
        if element.clickable is True
        for value in (
            (element.text or "").strip(),
            (element.content_desc or "").strip(),
            (_id_tail(element.resource_id) or "").replace("_", " "),
        )
        if value
    )

    # A learned canonical name can itself come from a prior title heuristic. When the only
    # current goal match is clickable, do not let that persisted name launder a launcher row
    # into identity proof; require the passive title/anchor path below instead.
    if not rec.stale:
        identities = [
            rec.name,
            rec.canonical_name or "",
            rec.logical_name or "",
            *rec.aliases,
        ]
        for identity in identities:
            if (
                identity
                and not clickable_goal_match
                and _arrival_value_matches(identity.replace("_", " "), terms)
            ):
                return {
                    "source": "mapped_identity",
                    "target": target,
                    "value": identity,
                }

    visible = [
        element
        for element in elements
        if element.clickable is not True
        and not _is_input(element)
        and not _system_chrome(element, screen_height)
        and not _bottom_nav(element, screen_height)
    ]
    title = title_of(visible, screen_height)
    if title and _arrival_value_matches(title, terms):
        return {"source": "visible_title", "target": target, "value": title}

    remembered = set(rec.anchors)
    for element in visible:
        candidates = (
            ("id", _id_tail(element.resource_id) or ""),
            ("cd", (element.content_desc or "").strip()),
            ("tx", (element.text or "").strip()),
        )
        for kind, value in candidates:
            anchor = f"{kind}:{value.casefold()}"
            if (
                value
                and anchor in remembered
                and _arrival_value_matches(value.replace("_", " "), terms)
            ):
                return {
                    "source": "visible_anchor",
                    "target": target,
                    "value": anchor,
                }
    return None


def screen_is_root(app: AppMap, screen: str, context_id: str | None = None) -> bool:
    """Whether *screen* is a mapped route root in the selected capture context."""
    rec = app.screens.get(screen)
    if rec is None:
        return False
    if context_id is not None and rec.context_id not in {context_id, LEGACY_CONTEXT_ID}:
        return False
    return screen in _roots(app, context_id)


def _find_targets(app: AppMap, query: str, context_id: str | None = None) -> list[str]:
    """Screens matching *query* by name, key-element label, anchor, dynamic shape, or a
    route action that leads to them (so ``--find "report"`` finds the reports screen).

    Every term has to appear somewhere in the screen, but they do not have to appear together.
    Matching the query as one literal substring meant only a caller who already knew the
    phrasing could find anything: `--find "search"` and `--find "catalog"` each answered, while
    `--find "catalog search"` — a goal, which is how anyone actually asks — answered "no matching
    screen in memory" about a synthetic map that held the route.
    """
    targets, _terms = _find_targets_and_terms(app, query, context_id)
    return targets


def _search_passes(query: str) -> list[list[str]]:
    """Strictest first: the query exactly as typed, then only the words that name a target.

    Two passes rather than one so that loosening is purely additive. A keyword query already
    answered correctly and keeps answering identically -- it never reaches the second pass.
    A synthetic regression proves that collapsing this into a single stripped pass can move a
    keyword answer to a sibling screen. Two passes keep the strict answer when one exists.
    """
    raw = _search_words(query.casefold().strip())
    stripped = search_terms(query)
    return [raw] if stripped == raw else [raw, stripped]


def _find_targets_and_terms(
    app: AppMap, query: str, context_id: str | None = None
) -> tuple[list[str], list[str]]:
    """Matching screens plus the terms that matched them, so callers rank on the same basis."""
    for terms in _search_passes(query):
        scored: list[tuple[bool, bool, int, str]] = []
        for name, rec in app.screens.items():
            if context_id is not None and rec.context_id not in (context_id, LEGACY_CONTEXT_ID):
                continue
            score = _target_match_rank(app, name, query, context_id, terms)
            if score is not None:
                scored.append((rec.stale, rec.context_id == LEGACY_CONTEXT_ID, score, name))
        if scored:
            scored.sort()
            return [name for _stale, _legacy, _score, name in scored], list(terms)
    return [], []


def find_result(app: AppMap, query: str, context_id: str | None = None) -> dict[str, object]:
    """Structured ``--find --json`` payload: matching screens + the route to each."""
    results = []
    targets = _find_targets(app, query, context_id)
    for t in targets[:8]:
        rec = app.screens.get(t)
        path = [] if rec and rec.stale else _shortest_path(app, t, context_id=context_id)
        results.append(
            {
                "screen": t,
                "tier": rec.tier if rec else None,
                "stale": rec.stale if rec else None,
                "route": [
                    {"from": e.from_screen, "action": e.action, "to": e.to_screen} for e in path
                ],
                "key_elements": [ke.model_dump() for ke in rec.key_elements] if rec else [],
            }
        )
    return {
        "query": query,
        "package": app.package,
        "results": results,
        "total_matches": len(targets),
        "truncated": max(0, len(targets) - len(results)),
    }


def resolve_goal(
    app: AppMap,
    goal: str,
    *,
    start: str | None = None,
    now: datetime | None = None,
    half_life_days: float = 3.0,
    last_goal: str | None = None,
    context_id: str | None = None,
    destructive_labels: Sequence[str] = (),
) -> str | None:
    """Best screen name for a fuzzy *goal* (powers ``aua goto``).

    Exact name wins; otherwise prefer a screen reachable from *start*, then higher rank
    score, then a shorter route.
    """
    targets, matched_terms = _find_targets_and_terms(app, goal, context_id)
    if not targets:
        return None
    g = goal.lower().strip()
    exact = [t for t in targets if t.lower() == g]
    if exact:
        return exact[0]
    now = now or datetime.now().astimezone()

    def key(name: str) -> tuple[int, bool, bool, bool, bool, float, int]:
        path = _shortest_path(
            app,
            name,
            start=start,
            context_id=context_id,
            destructive_labels=destructive_labels,
        )
        reachable = (
            bool(path) or name == start or (start is None and name in _roots(app, context_id))
        )
        score = _rank_score(
            app.screens[name], now=now, half_life_days=half_life_days, last_goal=last_goal
        )
        # A screen *named* for the goal beats one that merely contains a matching element.
        rec = app.screens[name]
        match_rank = _target_match_rank(app, name, goal, context_id, matched_terms)
        return (
            -(match_rank if match_rank is not None else NO_TARGET_MATCH),
            not rec.stale,
            rec.context_id != LEGACY_CONTEXT_ID,
            g in name.lower(),
            reachable,
            score,
            -(len(path) or 99),
        )

    return sorted(targets, key=key, reverse=True)[0]


def _render_find(app: AppMap, query: str, context_id: str | None = None) -> str:
    targets = _find_targets(app, query, context_id)
    lines = [f"# find: {query}  ({app.package})"]
    if not targets:
        lines.append("")
        lines.append("_(no matching screen in memory — navigate there once so it is recorded)_")
        return "\n".join(lines) + "\n"
    for t in targets[:8]:
        rec = app.screens.get(t)
        lines.append("")
        lines.append(f"## {t}" + (f"  (tier: {rec.tier})" if rec else ""))
        path = [] if rec and rec.stale else _shortest_path(app, t, context_id=context_id)
        roots = _roots(app, context_id)
        has_untrusted_incoming = any(
            edge.to_screen == t
            for edge in _routes_for_context(app, context_id, include_provisional=True)
        )
        if rec and rec.stale:
            route_text = "(stale screen; re-observe before routing)"
        elif path:
            route_text = _format_path(path)
        elif t in roots and not has_untrusted_incoming:
            route_text = "(start here)"
        else:
            route_text = "(no verified route)"
        lines.append("route: " + route_text)
        if rec:
            lines.extend(f"  - {s}" for s in _summarize_keys(rec))
    if len(targets) > 8:
        lines.extend(("", f"_({len(targets) - 8} lower-ranked matches omitted)_"))
    return "\n".join(lines).rstrip() + "\n"


def _render_screen_detail(app: AppMap, screen: str) -> str:
    rec = app.screens.get(screen)
    if rec is None:
        avail = ", ".join(sorted(app.screens)) or "(none)"
        return f"# {screen}\n\n_(unknown screen; known: {avail})_\n"
    lines = [f"# {screen}  ({app.package})"]
    meta = [f"tier: {rec.tier}", f"visits: {rec.visit_count}", f"last verified {rec.last_verified}"]
    if rec.activity:
        meta.insert(0, f"activity: {rec.activity}")
    if rec.stale:
        meta.append("STALE")
    lines.append("_" + " · ".join(meta) + "_")
    if rec.key_elements:
        lines.append("")
        lines.append("## Elements")
        for ke in rec.key_elements:
            bits = [ke.type]
            if ke.label:
                bits.append(f"“{ke.label}”")
            if ke.resource_id:
                bits.append(f"#{ke.resource_id}")
            flags = [f for f, on in (("clickable", ke.clickable), ("input", ke.input)) if on]
            if ke.value:
                flags.append(ke.value)
            if flags:
                bits.append(f"[{', '.join(flags)}]")
            lines.append("- " + " ".join(bits))
    if rec.dynamic:
        lines.append("")
        lines.append("## Dynamic")
        lines.extend(f"- {d}" for d in rec.dynamic)
    incoming = [e for e in app.routes if e.to_screen == screen]
    outgoing = [e for e in app.routes if e.from_screen == screen]
    if incoming or outgoing:
        lines.append("")
        lines.append("## Routes")
        for e in incoming:
            lines.append(f"← {e.from_screen} --{e.action}-->")
        for e in outgoing:
            lines.append(f"→ {e.to_screen} ({e.action})")
    return "\n".join(lines).rstrip() + "\n"
