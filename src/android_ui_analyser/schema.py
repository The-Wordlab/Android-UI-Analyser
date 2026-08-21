"""Canonical, versioned output schema (PRD §8).

This module is the **single source of truth** for the shape of everything the CLI and
MCP server emit. Pydantic models here are imported by the engine, the CLI, the MCP
wrapper, and the tests. Do not duplicate these shapes elsewhere.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- enums


class Source(str, Enum):
    """Where a single element came from."""

    hierarchy = "hierarchy"
    detection = "detection"
    ocr = "ocr"
    grounding = "grounding"
    webview = "webview"


class ScreenSource(str, Enum):
    """Aggregate provenance of the whole screen result."""

    hierarchy = "hierarchy"
    vision = "vision"
    mixed = "mixed"


class PathKind(str, Enum):
    """Which high-level perception path produced the result."""

    hierarchy = "hierarchy"
    vision = "vision"


class Tier(str, Enum):
    """Escalation ladder rungs (PRD §6a), cheapest → most expensive."""

    text = "text"
    selector = "selector"
    hierarchy = "hierarchy"
    vision = "vision"
    grounding = "grounding"


# Canonical ordering for the escalation ladder. Index == cost rank.
TIER_ORDER: tuple[Tier, ...] = (
    Tier.text,
    Tier.selector,
    Tier.hierarchy,
    Tier.vision,
    Tier.grounding,
)


def tier_rank(tier: Tier | str) -> int:
    """Return the cost rank of a tier (lower == cheaper)."""
    t = Tier(tier)
    return TIER_ORDER.index(t)


class OutputFormat(str, Enum):
    json = "json"
    pretty = "pretty"
    compact = "compact"
    tsv = "tsv"  # one element per line, tab-separated (see :mod:`projection`)
    delta = "delta"  # omit elements when meta.unchanged; else compact + element_diff
    msgpack = "msgpack"  # binary dump (host-side; see binary_dump.py / hierarchy.fbs)


class MatchMode(str, Enum):
    exact = "exact"
    contains = "contains"
    regex = "regex"


# --------------------------------------------------------------------------- models

Bounds = tuple[int, int, int, int]
Center = tuple[int, int]


def center_of(bounds: Bounds) -> Center:
    """Geometric center of an ``[x1, y1, x2, y2]`` box."""
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) // 2, (y1 + y2) // 2)


class Element(BaseModel):
    """One actionable thing on screen, identified by a stable integer ``id``.

    The interaction-state flags (``checkable``/``checked``/``selected``/``scrollable``/
    ``long_clickable``/``password``) are **tri-state**: ``True``/``False`` when the
    accessibility node reported the attribute, ``None`` when it is genuinely unknown (a
    vision-derived element has no a11y attributes at all). A caller reading a toggle must
    be able to tell "off" from "unknown", so we never coerce a missing attribute to
    ``False``. ``clickable``/``enabled``/``focused`` predate this rule and keep their
    long-standing plain-``bool`` semantics.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    type: str
    text: str | None = None
    resource_id: str | None = None
    content_desc: str | None = None
    bounds: Bounds
    center: Center
    clickable: bool = False
    enabled: bool = True
    focused: bool = False
    checkable: bool | None = None
    checked: bool | None = None
    selected: bool | None = None
    scrollable: bool | None = None
    long_clickable: bool | None = None
    password: bool | None = None
    source: Source = Source.hierarchy
    confidence: float | None = None
    # Cross-frame fingerprint — survives re-analyze ID churn (see ``identity.stable_key``).
    stable_key: str | None = None
    # Window layer: app | ime | system | overlay (hierarchy package heuristics).
    window: str | None = None
    # ``id`` of the nearest *collected* ancestor, or None for a root / vision element. Kept
    # because the acting control is often not a geometric container of the label that names
    # it: a design-system tile puts the click on an inner Box and renders the title as a
    # sibling **outside** those bounds, so containment cannot find one from the other and
    # only the tree can. See :func:`selectors.acting_node`.
    parent: int | None = None

    def compact(self) -> dict[str, Any]:
        """Token-minimal dict: drop nulls and default-valued verbose fields."""
        out: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "bounds": list(self.bounds),
            "center": list(self.center),
        }
        if self.text is not None:
            out["text"] = self.text
        if self.resource_id is not None:
            out["resource_id"] = self.resource_id
        if self.content_desc is not None:
            out["content_desc"] = self.content_desc
        if self.clickable:
            out["clickable"] = True
        if not self.enabled:
            out["enabled"] = False
        if self.focused:
            out["focused"] = True
        out.update(self._compact_state())
        if self.source is not Source.hierarchy:
            out["source"] = self.source.value
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        if self.stable_key is not None:
            out["stable_key"] = self.stable_key
        if self.window is not None:
            out["window"] = self.window
        if self.parent is not None:
            out["parent"] = self.parent
        return out

    def _compact_state(self) -> dict[str, Any]:
        """Interaction-state flags worth their tokens.

        ``checked`` rides along whenever the node is ``checkable`` even when it is
        ``False``, because on a checkable node the *off* state is the payload — dropping
        it as a "default" would make a switch unreadable, which is the whole point of
        these fields.
        """
        out: dict[str, Any] = {}
        if self.checkable:
            out["checkable"] = True
            out["checked"] = self.checked
        elif self.checked:
            out["checked"] = True
        for name in ("selected", "scrollable", "long_clickable", "password"):
            if getattr(self, name):
                out[name] = True
        return out


class Screen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    package: str | None = None
    activity: str | None = None
    source: ScreenSource


class ObservationContract(BaseModel):
    """Machine-readable guidance for safely reusing one caller-visible frame."""

    model_config = ConfigDict(extra="forbid")

    fingerprint: str | None = None
    evidence_id: str | None = None
    produced_by: str
    reusable: bool
    analyze_needed: bool
    reason: str


class CallerTurn(BaseModel):
    """What the caller's own latency cost, and whether the screen it was holding is still up.

    aua knows both ends of every gap — the stamp it wrote when its last call returned, and the
    clock when this one started — so the caller's think time is free to compute. Measured over
    one 13-call agent session it was 75% of the elapsed wall time: the dominant term, and until
    now the one number aua discarded.

    It is reported because it is what the wait ceiling is sized from, and a caller that can see
    `ema_ms` can see why its budget is what it is. Note the arithmetic runs one way only: the
    estimate can lower the ceiling below `perf.max_wait_ms` but never raise it, so a caller
    reading a large `ema_ms` next to a 5000ms `wait_ceiling_ms` is being told that calling again
    is its cheaper move, not that aua will wait that long for it.
    """

    model_config = ConfigDict(extra="forbid")

    # Think time between the previous call returning and this one starting.
    gap_ms: int | None = None
    # Present when this gap was measured but deliberately not learned from: `idle` (longer than
    # any caller turn could be — someone walked away) or `clock` (time went backwards).
    gap_ignored: str | None = None
    ema_ms: int | None = None
    spread_ms: int | None = None
    samples: int | None = None
    # The ceiling one observation wait may not exceed, and which policy produced it:
    # `pinned` / `fixed` (reproducible) or `cold` / `adaptive` (sized from the measurement).
    wait_ceiling_ms: int | None = None
    wait_ceiling_mode: str | None = None
    # True when the screen described by the caller's *previous* result is no longer on the
    # device. A stale element id already raises; a wholly replaced screen said nothing at all,
    # so a caller could reason about a screen that had been gone for its entire thinking gap.
    # Deliberately a field and not prose, and deliberately not an error: acting anyway is a
    # legitimate choice and only the caller can make it.
    previous_screen_gone: bool | None = None
    previous_screen_age_ms: int | None = None

    @model_serializer
    def _only_what_was_measured(self) -> dict[str, Any]:
        """Drop unset fields on the way out, at every nesting depth.

        This block rides on every response, so the difference between an absent key and a null
        one is not cosmetic twice over: nulls are pure token cost on the path this project is
        most sensitive about, and `previous_screen_gone: null` actively misreads as "checked,
        the screen is fine" when the truth is "there was nothing to compare". The enclosing
        `Meta`/`ActionResult` dumps filter their own top-level Nones but cannot see inside a
        nested model, which is why this lives here rather than at the call site.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_ms: int
    tier_used: Tier
    path: PathKind
    providers_used: list[str] = Field(default_factory=list)
    known_screen: str | None = None  # recognised app-map screen name (PRD §6b, §8)
    # Navigation affordances pushed inline from app memory (§6b) so an agent gets them on
    # the analyze it already runs, instead of having to remember to call `aua map`.
    known_routes: list[str] = Field(default_factory=list)  # ["tap 'Catalog' → catalog", ...]
    suggested_gotos: list[str] = Field(default_factory=list)  # ["goto product_detail", ...]
    suggested_deeplinks: list[str] = Field(  # ["open myapp://dashboard", ...] — shortcuts
        default_factory=list
    )
    research_tasks: list[str] = Field(default_factory=list)
    # Saved journeys for THIS app, as ``name(PARAM, …)``. A flow replays a whole sequence in one
    # call, so it belongs next to the routes rather than in a manual nobody opens.
    flows: list[str] = Field(default_factory=list)
    # One question about THIS screen, answerable on the caller's next command via `--answers`.
    # Scoped to where the caller is standing, because that is the only question they can answer
    # from what is in front of them.
    ask: dict[str, str] | None = None
    map_hint: str | None = None  # e.g. "12 screens mapped — run `aua map`"
    # Controls on THIS screen that history says are slow, worst first:
    # [{"control": "buttonContinue", "avg_ms": 4800, "max_ms": 6100, "n": 3, ...}].
    # Told to the agent on arrival so a long wait is planned rather than discovered as a timeout.
    slow_controls: list[dict[str, Any]] = Field(default_factory=list)
    capture_hint: str | None = None  # rolling buffer saw post-action change
    # The hierarchy handed us text it could not represent (U+FFFD). Silence here is worse
    # than slowness: an agent reads "Divide both sides by 2 to solve for <?>: <?>", cannot
    # see the formula, and either reports a wrong observation or falls back to eyeballing
    # screenshots on its own. Say so, and name the flag that recovers it.
    lossy_text: bool = False
    lossy_hint: str | None = None
    # How many broken labels OCR successfully repaired in place on this analyze.
    ocr_repaired: int = 0
    annotated_image: str | None = None
    raw_image: str | None = None  # unannotated screenshot saved on request (--with-image)
    device_serial: str | None = None
    device_locale: str | None = None  # BCP-47 UI locale (e.g. "es-ES"); labels render in it
    observation_contract: ObservationContract | None = None
    # Optional token-cheap delta vs the previous analyze (perf.differential).
    element_diff: dict[str, Any] | None = None
    # Host-side incremental a11y: True when hierarchy XML matched the previous analyze.
    unchanged: bool = False
    # Why `unchanged` / `element_diff` must NOT be read as evidence about an action's effect:
    # set on a post-action observation whose settle wait never confirmed the screen had moved,
    # so the frame it describes may predate the action. A *string* rather than a False flag
    # because `compact` drops falsey values and `delta` allowlists keys — a boolean would
    # vanish from output in exactly the cases that need it.
    stale_risk: str | None = None
    # SHA1 of the raw hierarchy XML (or elements fingerprint for vision paths).
    fingerprint: str | None = None
    # How the result was produced (e.g. hierarchy, hierarchy-unchanged, vision).
    via: str | None = None
    caller: CallerTurn | None = None
    # Active goal-session phase and one exact next call. Attached best-effort after perception;
    # absent outside a goal session.
    goal_progress: dict[str, Any] | None = None


class AnalyzeResult(BaseModel):
    """Top-level ``analyze`` payload (PRD §8)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    screen: Screen
    elements: list[Element]
    meta: Meta

    # -- rendering ---------------------------------------------------------

    def as_dict(self, fmt: OutputFormat | str = OutputFormat.json) -> dict[str, Any]:
        """The serialisable payload for *fmt* (``compact`` trims to the smallest footprint).

        Shared by :meth:`render` and by :class:`ActionResult` so an embedded ``observation``
        renders in the same format as a standalone ``analyze``.
        """
        fmt = OutputFormat(fmt)
        if fmt is OutputFormat.delta:
            base = self.as_dict(OutputFormat.compact)
            if self.meta.unchanged:
                base["elements"] = []
                base["meta"] = {
                    k: v
                    for k, v in base.get("meta", {}).items()
                    if k
                    in {
                        "duration_ms",
                        "tier_used",
                        "path",
                        "unchanged",
                        "fingerprint",
                        "via",
                        "element_diff",
                        "device_serial",
                        "known_screen",
                        # Must survive the delta trim: `delta` fires precisely when
                        # `unchanged` is True, which is the case whose caveat matters.
                        "stale_risk",
                    }
                    and v not in (None, [], False)
                }
                base["meta"]["unchanged"] = True
            return base
        if fmt is OutputFormat.compact:
            return {
                "schema_version": self.schema_version,
                "screen": {
                    k: v for k, v in self.screen.model_dump(mode="json").items() if v is not None
                },
                "elements": [e.compact() for e in self.elements],
                "meta": {
                    k: v
                    for k, v in self.meta.model_dump(mode="json").items()
                    # `v not in (None, [], False)` compares with ==, and 0 == False, so a
                    # `duration_ms` of 0 was dropped — the payload then failed its own schema
                    # (duration_ms is required), and only on FAST calls, i.e. the ones the
                    # trimming exists for. Test `False` by identity.
                    if v is not None and v != [] and v is not False
                },
            }
        return self.model_dump(mode="json")

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        """Serialise to one of the output formats (PRD §8)."""
        fmt = OutputFormat(fmt)
        if fmt is OutputFormat.msgpack:
            from .binary_dump import pack_analyze_b64

            return pack_analyze_b64(self)
        data = self.as_dict(fmt)
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)

    def element_by_id(self, element_id: int) -> Element | None:
        for e in self.elements:
            if e.id == element_id:
                return e
        return None


# --------------------------------------------------------- lightweight result models


class HasResult(BaseModel):
    """Result of the ``has`` quick-check (PRD §5 quick checks)."""

    model_config = ConfigDict(extra="forbid")

    found: bool
    source: str | None = None  # "hierarchy" | "ocr"
    bounds: Bounds | None = None
    text: str | None = None
    # On a miss only: the device UI locale plus, for text lookups, a hint that labels
    # render in that locale — match observed labels or go --by id. Language-neutral:
    # the query may be written in any language, so no locale is treated as "safe".
    # On a hit through the mined-strings bridge, `hint` names the key and rendering
    # that matched (and `text` carries the rendering that was actually found).
    device_locale: str | None = None
    hint: str | None = None
    wait_clamped_from_ms: int | None = None
    wait_ceiling_ms: int | None = None
    wait_ceiling_mode: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class AppStatusResult(BaseModel):
    """Read-only package-manager status for one app on the selected target."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    action: str = "app-status"
    package: str
    installed: bool
    serial: str
    version_name: str | None = None
    version_code: str | None = None
    mode: str = "read-only"

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ShellResult(BaseModel):
    """Structured result of one bounded, read-only command on the selected target."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str = "shell"
    serial: str
    argv: list[str]
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_limit_bytes: int = 256 * 1024
    exit_code: int
    duration_ms: int
    mode: str = "read-only"

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = self.model_dump(mode="json")
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ActionResult(BaseModel):
    """Result of an action command (tap/input/swipe/key/...)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str
    id: int | None = None
    target: list[int] | None = None  # coords or bounds acted on
    detail: str | None = None
    # The screen right after the action (when called with observe=True), so an agent gets
    # fresh element ids without a separate `analyze` round-trip (act + observe in one call).
    observation: AnalyzeResult | None = None
    # Stable contract fields for downstream agents. `observation_present` is always returned on
    # an *action* response, so callers can branch without checking for key existence. Both are
    # None by default rather than False/[] because `render()` only strips None: a plain default
    # would emit these on commands that perform no action at all, and "observation_present:
    # false" on a `screenshot` claims the effect of an action was not observed when there was no
    # action to observe. `_observe` sets `observation_present` on both of its branches, which is
    # what makes the field reliably present exactly where the contract promises it.
    observation_present: bool | None = None
    # Route context on the action response itself.
    known_screen: str | None = None
    # Stable identifiers for the returned ids (ID churn does happen, stable_key usually survives).
    stable_elements: list[dict[str, Any]] | None = None
    # Compact diff summary from the folded observation (`meta.element_diff` transformed).
    action_diff_summary: dict[str, Any] | None = None
    # Inline hint when an action already returns usable screen state.
    note: str | None = None
    # Structured one-call efficiency recommendation (CLI and MCP share the same ids/text).
    advice: list[dict[str, str]] | None = None
    # What an install actually did to the target, as data rather than prose: {"package": ...,
    # "installed": bool, "pushed": bool, "reason": "already-present"|"missing"|…, "version_name":
    # …, "bundle_version_name": …}. `pushed` is the field that matters to a caller deciding
    # whether app state survived — an idempotent skip preserves data, a push may not — and a
    # sentence in `detail` cannot be branched on.
    app_install: dict[str, Any] | None = None
    goal_progress: dict[str, Any] | None = None
    observation_contract: ObservationContract | None = None
    # Why the folded observation may not show the action's effect — e.g. the post-action screen was
    # byte-identical to the pre-action one. Carries the reason, not a bare flag, so a reader can
    # judge it. Absent when the observation is trustworthy, so its presence is the signal.
    stale_risk: str | None = None
    # What the post-action wait cost: {"ms": int, "via": str, "anim": ...}. Answers "why did that
    # tap take 300 ms" as data rather than as a tag glued onto `detail`.
    settle: dict[str, Any] | None = None
    # Real wall time for the whole call, device round trips included. `settle.ms` and the
    # observation's `meta.duration_ms` each measure one phase, and in the field their sum
    # understated a 5.26s call as 247ms — a caller budgeting from those numbers is budgeting
    # from a fiction. This is the number a stopwatch would show.
    wall_ms: int | None = None
    # The deliberate post-action pause actually spent, so a latency sweep can attribute cost
    # to the knob rather than inferring it (see `perf.stable_delay_ms`).
    stable_delay_ms: int | None = None
    # Set when the caller asked to wait longer than `perf.max_wait_ms` allows. Carries the
    # *requested* value so the response explains itself: the wait was shortened on purpose and
    # "nothing yet" must not be read as "nothing there". `wait_ceiling_ms` is what was enforced.
    wait_clamped_from_ms: int | None = None
    wait_ceiling_ms: int | None = None
    # Whether that ceiling was measured (`cold`/`adaptive`) or pinned (`fixed`/`pinned`).
    wait_ceiling_mode: str | None = None
    # What the caller's own latency cost, and whether its previous screen is still there.
    caller: CallerTurn | None = None
    # True when the folded observation came back with no visible elements. An action that
    # reports success while handing back an empty screen is the defect this names: downstream
    # the agent burns a turn, and the screen it wanted may already have arrived.
    observation_empty: bool | None = None
    # A bounded wait reached its ceiling with the screen still moving. Distinct from an error:
    # the screen is returned, nothing is known to be wrong, and the caller may simply ask again.
    settled_unmet: bool | None = None
    # What can be done from the screen this action landed on — the point being that an agent should
    # not have to scan `observation.elements` to pick its next id. Each entry carries the control's
    # own learned cost when we have one, so "tap 26 next, and it historically takes 4.8s" is a
    # single read: [{"id": 26, "label": "Submit", "rid": "submitButton", "avg_ms": 4800, "n": 3}].
    # Capped, because a list of everything is a dump, not guidance.
    next_actions: list[dict[str, Any]] | None = None
    # Navigation shortcuts out of here that memory already knows — merged `known_routes` and
    # `suggested_gotos`, hoisted so they are visible without opening `observation.meta`.
    routes: list[str] | None = None
    # Rolling capture saw a post-action pixel change — pull `aua capture last`.
    capture_hint: str | None = None
    # Did the action's effect get *observed*, as opposed to merely attempted?
    # True  = read back and confirmed, False = read back and provably did not happen,
    # None  = not checked, or checked and genuinely ambiguous. Absent from output when None,
    # so `ok` keeps its meaning for actions that cannot verify themselves.
    verified: bool | None = None
    # Which node actually received the interaction, and how it relates to the one named. The
    # control a label belongs to is frequently a different node — a design-system tile puts
    # the click on an inner Box and renders the title outside it — so "I acted on what you
    # named" and "I acted on its sibling" must not look the same in output.
    acting: dict[str, Any] | None = None
    # `await`: which terminal condition ended the wait.  In addition to satisfied /
    # screen-changed / timeout, an action-bound wait can return settled-unmet when the action
    # demonstrably reached a different, stable destination but the caller's positive arrival
    # term names something that is not there.  Standalone waits deliberately retain their
    # package/activity-only three-outcome semantics.
    # What actually changed, rather than a claim that the action dispatched: activity before and
    # after, node-count delta, focus movement, text added/removed. Deliberately carries no
    # confidence score — a number invites trusting a figure over evidence, and "a command
    # reporting success is not evidence of effect" is this project's first lesson. `changed` is
    # an explicit boolean (None = no baseline to compare) so "nothing changed" is checkable.
    change: dict[str, Any] | None = None
    # A bounded fatal/ANR/error block read automatically from the action's own diagnostic-log
    # window when the app leaves the foreground. This is structured and top-level so CLI and MCP
    # callers receive the cause without parsing the warning or spending another tool call.
    crash_evidence: dict[str, Any] | None = None
    # What the app itself logged in this action's own window, scoped to its process and reduced
    # to a priority set plus a line budget. The screen is the app's conclusion; this is its
    # reasoning, and between them sits the failure an agent otherwise cannot see: the tap
    # landed, the screen looks plausible, and the app quietly logged the refusal. Absent when
    # the window was empty, so a quiet action costs nothing.
    app_logs: dict[str, Any] | None = None
    await_outcome: str | None = None
    # Per-term results, reported satisfied or not: *which* term is missing is how a reader
    # tells a failed load from a slow one.
    await_terms: list[dict[str, Any]] | None = None
    # Structured correction for ``await_outcome=settled-unmet``.  Keeping this separate from
    # prose lets adapters and agents offer a corrected predicate without parsing ``detail``.
    arrival_mismatch: dict[str, Any] | None = None
    # Unmet positive ``rid:`` terms that no mapped screen of this app has ever carried. An
    # unmet term reads as "not there yet" whether or not it *can* be there; naming the ones
    # that cannot is what stops an agent inventing a second id and waiting on that too.
    unknown_selectors: list[dict[str, Any]] | None = None
    elapsed_ms: int | None = None
    # Bounded multi-step actions expose why they stopped and a compact semantic hop trace.
    stop_reason: str | None = None
    steps_run: list[dict[str, Any]] | None = None
    # Locale-bridge diagnosis: on a translated hit, which string key and rendering matched;
    # on a text miss, what the target renders as in the device locale (see Meta.device_locale).
    hint: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        fmt = OutputFormat(fmt)
        data = {
            k: v
            for k, v in self.model_dump(mode="json").items()
            if v is not None and k != "observation"
        }
        if self.observation is not None:
            data["observation"] = self.observation.as_dict(fmt)
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class NetworkState(BaseModel):
    """Observed Android connectivity controls and the active default network."""

    model_config = ConfigDict(extra="forbid")

    airplane_mode: bool | None = None
    wifi_supported: bool | None = None
    wifi_enabled: bool | None = None
    cellular_supported: bool | None = None
    mobile_data_enabled: bool | None = None
    active_network: bool | None = None
    active_network_id: str | None = None
    active_transports: list[str] = Field(default_factory=list)
    internet_validated: bool | None = None
    offline: bool | None = None


class NetworkShaping(BaseModel):
    """Observed bandwidth, latency, or packet-loss shaping for a network profile."""

    model_config = ConfigDict(extra="forbid")

    mechanism: str
    upload_bps: int | None = None
    download_bps: int | None = None
    min_latency_ms: int | None = None
    max_latency_ms: int | None = None
    interface: str | None = None
    loss_percent: float | None = None
    qdisc: str | None = None
    root_enabled: bool | None = None


class NetworkResult(BaseModel):
    """Result of deterministic network status/offline/restore operations."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: str
    state: NetworkState
    saved_state: NetworkState | None = None
    profile: str | None = None
    shaping: NetworkShaping | None = None
    verified: bool | None = None
    detail: str | None = None
    goal_progress: dict[str, Any] | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class ResolveResult(BaseModel):
    """Remap a previous-frame id or ``stable_key`` onto the current screen."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    from_id: int | None = None
    to_id: int | None = None
    stable_key: str | None = None
    element: Element | None = None
    detail: str | None = None

    def render(self, fmt: OutputFormat | str = OutputFormat.json) -> str:
        data = {k: v for k, v in self.model_dump(mode="json").items() if v is not None}
        indent = 2 if OutputFormat(fmt) is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        return json.dumps(data, indent=indent, separators=sep, ensure_ascii=False)


class DeviceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serial: str
    model: str | None = None
    android_version: str | None = None
    locale: str | None = None
    state: str = "device"
