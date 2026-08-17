"""The interface-agnostic perception + action engine (PRD §6, §6a).

The engine orchestrates the analyze pipeline and the cost-aware escalation ladder. It
depends only on: the schema, the config, the device ABC, the provider *factory* +
interfaces, and the routing helpers. It NEVER imports a concrete provider, and the
hierarchy/gate/merge/annotate modules are imported lazily so a fresh checkout imports
cleanly. The CLI, MCP server, and daemon are all thin adapters over this class.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import shlex
import threading
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from copy import deepcopy
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from . import routing
from .assertions import (
    Selector,
    apply_structural_filters,
    check_contains_all,
    evaluate_order,
    normalize_selector,
)
from .config import Config
from .device import Device, connect, list_devices
from .errors import (
    AuaError,
    DeviceError,
    ElementNotFoundError,
    JobCancelledError,
    ProviderError,
    SelectorAmbiguousError,
    SelectorNotFoundError,
    StabilityTimeout,
    StaleElementIdError,
    UsageError,
)
from .memory import (
    DEFAULT_CONTEXT_ID,
    LEGACY_CONTEXT_ID,
    AppMap,
    AppMemoryStore,
    NavHints,
    RouteEdge,
    RouteStep,
    _id_tail,
    _shortest_path,
    arrival_destination_terms,
    context_view,
    is_destructive_step,
    launch_payload,
    matches_any,
    playbook_view,
    recorded_selector,
    redact_label,
    resolve_goal,
    route_step_risks,
    screen_is_root,
    screen_skips_ocr,
    step_display,
    target_arrival_evidence,
)
from .platforms import AppBundle, InstalledApp, PlatformAdapter, PlatformFactory
from .providers.base import DetBox, OcrProvider, Point, ScreenAnalysisResult, ScreenImage, TextBox
from .providers.registry import ProviderFactory, registered_names, run_chain
from .schema import (
    ActionResult,
    AnalyzeResult,
    DeviceInfo,
    Element,
    HasResult,
    MatchMode,
    Meta,
    NetworkResult,
    PathKind,
    ResolveResult,
    Screen,
    ScreenSource,
    Source,
    Tier,
    center_of,
)
from .scroll_geom import (
    Box,
    Sample,
    _contains,
    region_probe,
    scrollable_boxes,
    travel,
)
from .selectors import (
    _MAX_CANDIDATES,
    _match_step,
    app_elements,
    drop_redundant_ocr,
    element_digest,
    is_back_resource_id,
    match_selector,
    nearest_elements,
    ocr_added_app_content,
    selector_label,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .flags import PrefsRead
    from .policy import PolicyMode, PolicySelector

logger = logging.getLogger("android_ui_analyser.engine")

# Keep the historical module-level monkeypatch seams working for downstream tests and
# integrations while production construction moves behind PlatformAdapter.
_DEFAULT_ANDROID_CONNECT = connect
_DEFAULT_ANDROID_LIST_DEVICES = list_devices

# A run of text one line tall measures roughly two to three average character widths
# in height; a wrapped paragraph measures many. Used to decide whether aiming at a
# phrase inside an element can only move horizontally.
_SINGLE_LINE_HEIGHT_RATIO = 3.5

QUERY_CONFIDENT = 1.0  # all salient tokens / exact phrase present
QUERY_SOFT = 0.5  # best-effort threshold when escalation is exhausted
_ASSIST_MAX_STEPS = 6  # bound on planner actions per recovery attempt (opt-in only)
# A bottom system bar starts within this fraction of the screen height. Wide enough for a
# tall three-button bar, narrow enough that a systemui sheet or the expanded notification
# shade can never be mistaken for it (see Engine._system_bar_top).
_SYSTEM_BAR_BAND = 0.85
_MAX_FLOW_DEPTH = 5  # bound on nested `flow:` sub-flow composition (cycle backstop)
# A hierarchy dump quicker than this can outrun the screen it is reading, so a post-action
# sample may catch a half-attached tree; a slower one cannot (the render has finished by the
# time it returns). Measured ~150ms headless vs 600-1200ms windowed on the same emulator.
_FAST_DUMP_MS = 250.0
_CHANGE_TEXT_CAP = 12  # text deltas echoed back per direction; a list screen would dump hundreds
_FLAGS_VERIFY_DEADLINE_S = 2.0  # how long a flag write gets to reach the app's prefs file
_FLAGS_ENTRY_TIMEOUT_S = 3.0  # how long a pinned entry Activity gets before the default one
_FLAGS_FOREGROUND_TIMEOUT_S = 6.0  # how long the relaunched app gets to reach the foreground
# Foreground ownership can lead accessibility-window attachment briefly on a cold launch. Retry
# only while the requested package demonstrably remains foreground, and never beyond this budget.
_LAUNCH_HIERARCHY_SETTLE_S = 2.0
_LAUNCH_HIERARCHY_POLL_S = 0.05
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


class _ResolvedFlowNode(NamedTuple):
    """One immutable nested-flow snapshot validated before device mutation."""

    flow: Any
    directory: Path | None
    source_id: str
    steps: list[RouteStep]


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


class _ResolvedFlowPlan(NamedTuple):
    """The complete filesystem snapshot authorized by one flow preflight."""

    flow_graph: dict[tuple[str | None, str], _ResolvedFlowNode]
    flags: dict[int, _ResolvedFlagsResource]
    cassettes: dict[int, _ResolvedCassetteResource]


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


def _regex_literal_hint(predicate: str) -> str | None:
    """Explain regex-looking action predicates, which deliberately use literal matching."""
    with contextlib.suppress(AuaError):
        for term in _parse_await_terms(predicate):
            if term.by not in {"text", "rid", "desc"}:
                continue
            value = term.value
            if (
                value.startswith("^")
                or value.endswith("$")
                or any(token in value for token in (".*", ".+", "\\d", "\\s", "\\w", "(?"))
            ):
                return (
                    f"{term.text!r} looks regex-like, but action `until` terms use literal "
                    "contains matching. Use exact text/resource-id, or run "
                    "`aua await-and-analyze '<predicate>' --match regex` as a standalone wait."
                )
    return None


def _safe_adopted_change(
    previous: dict[str, Any] | None,
    adopted: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Never replace a valid action delta with a false claim made without a baseline."""
    if not isinstance(adopted, dict):
        return adopted
    if adopted.get("node_count_before") is not None or adopted.get("changed") is not False:
        return adopted
    if isinstance(previous, dict) and previous.get("node_count_before") is not None:
        return previous
    uncertain = dict(adopted)
    uncertain["changed"] = None
    return uncertain


class _PendingOcr(NamedTuple):
    """Apple Vision work running while the main thread captures the hierarchy."""

    image: ScreenImage
    provider: OcrProvider
    future: Future[list[TextBox]]
    executor: ThreadPoolExecutor
    started_at: float


class _HierarchyObservation(NamedTuple):
    elements: list[Element]
    package: str | None
    xml_hash: str
    ocr_texts: list[TextBox]
    ocr_elements: list[Element]
    ocr_provider: str | None
    image: ScreenImage | None


def _parse_point(arg: str | None) -> tuple[int, int] | None:
    """``"x,y"`` → ``(x, y)``; None when it is not a usable pair of coordinates."""
    if not arg:
        return None
    parts = arg.replace(" ", "").split(",")
    if len(parts) != 2:
        return None
    try:
        x, y = (int(round(float(p))) for p in parts)
    except ValueError:
        return None
    return (x, y) if x >= 0 and y >= 0 else None


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


def _install_versions_differ(installed: InstalledApp, bundle: AppBundle) -> bool:
    """Should ``install --if-needed`` re-push, given what the target already has?

    Compared as strings on purpose: a versionName is not a number (``"1.0-rc2+abc"`` is normal),
    and an ordering invented here would decide "newer" wrong on the first build that used a
    suffix. Only *difference* is knowable, and difference is the whole question — the caller
    asked for this bundle, not for a newer one.

    Fails **open** (returns ``True``) when the target reports no version at all: an unanswerable
    "is this the same build?" must not resolve to "yes, skip it", or a run silently verifies the
    previous build.
    """

    for target, source in (
        (installed.version_code, bundle.version_code),
        (installed.version_name, bundle.version_name),
    ):
        if source is None:
            continue
        if target is None:
            return True
        if str(target).strip() != str(source).strip():
            return True
    return False


def _action_mark(verb: str, el: Element) -> str:
    """Compact capture timeline label — verb + best human/id token."""
    label = el.text or el.content_desc or _id_tail(el.resource_id) or el.id
    # Keep marks short for timeline readability.
    text = str(label).replace("\n", " ").strip()
    if len(text) > 40:
        text = text[:37] + "…"
    return f"{verb}:{text}"


def _region_from_point(cx: int, cy: int, width: int, height: int) -> str:
    """Map a screen point onto the same 3×3 names used by ``diff_summary``."""
    gx = 0 if cx < width / 3 else (2 if cx >= 2 * width / 3 else 1)
    gy = 0 if cy < height / 3 else (2 if cy >= 2 * height / 3 else 1)
    names = (
        ("upper-left", "upper", "upper-right"),
        ("left", "center", "right"),
        ("lower-left", "lower", "lower-right"),
    )
    return names[gy][gx]


def _package_from_xml(xml: str, ignore: Sequence[str] = ("com.android.systemui",)) -> str | None:
    """Cheap foreground-package guess from a hierarchy dump (avoids an app_current RPC).

    Picks the most common ``package=`` among nodes, excluding *ignore* globs — system
    chrome and IMEs overlay every app, so an open keyboard must never win the vote.
    Falls back to the overall majority when every node is ignorable.
    """
    from .platforms.android import package_from_tree

    return package_from_tree(xml, ignore)


def _parse_legacy_steps(action: str) -> list[RouteStep] | None:
    """Replay steps for a pre-v2 string-only edge: strictly a single ``tap 'X'``.

    Anything else — compound joins, ``tap [View]``, key/input/swipe — is unreplayable
    and returns ``None`` (a clean ``unsupported_action``, never a garbage label).
    """
    m = re.fullmatch(r"tap '([^']+)'", action)
    if m is None:
        return None
    return [RouteStep(kind="tap", label=m.group(1))]


class Restart(NamedTuple):
    """Whether the app was confirmed back up, through which entry, and why not."""

    ok: bool
    activity: str | None
    error: str | None


class StepFailure(NamedTuple):
    """Why (and where) a step sequence stopped — the executor's divergence signal."""

    code: str  # destructive_step | input_required | input_not_applied | element_not_found |
    #            unsupported_action | wait_timeout | assert_failed
    at: int  # failing step index within the executed list
    step: RouteStep
    detail: str | None = None


def _goto_handoff(
    goal: str,
    target: str,
    code: str,
    hops: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    res: AnalyzeResult,
    *,
    failed_step: RouteStep | None = None,
    remaining_steps: list[RouteStep] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Stop driving and return enough state for the caller to continue manually."""
    out = {
        "ok": False,
        "code": code,
        "goal": goal,
        "target": target,
        "arrived": False,
        "hops": hops,
        "remaining_route": remaining,
        "current_screen": res.meta.known_screen,
        "suggested_gotos": res.meta.suggested_gotos,
        "elements": [
            {"id": e.id, "label": e.text or e.content_desc, "clickable": e.clickable}
            for e in res.elements
            if (e.text or e.content_desc)
        ][:20],
        "hint": hint
        or 'route diverged — finish the failed step, then `aua goto "…" --from-here` '
        "(or continue with `aua analyze` + `aua tap-and-analyze`)",
    }
    if failed_step is not None:
        out["step"] = {"display": step_display(failed_step), **failed_step.model_dump()}
    if remaining_steps:
        out["remaining_steps"] = [step_display(s) for s in remaining_steps]
        pkg = next((s.package for s in remaining_steps if s.package), None)
        if pkg:
            out["expected_package"] = pkg
    return out


def detail_tokens(outcome: str, **fields: Any) -> str:
    """``"moved steps=3 dy=1420"`` — outcome first, then ``k=v`` pairs.

    ``ActionResult`` is a frozen schema owned elsewhere, so scroll/expect verdicts ride in
    ``detail``. Outcome-first keeps it greppable (``grep -q target-not-found``) and the
    tokens keep it parseable; the exit code stays the primary signal.
    """
    parts = [outcome]
    parts += [f"{k}={v}" for k, v in fields.items() if v is not None]
    return " ".join(parts)


# u2 accepts these names (plus KEYCODE_* / a numeric keycode); anything else reaches the
# device as a no-op-or-crash, so it is rejected up front rather than looking like it worked.
_KEY_NAMES = frozenset(
    {
        "home",
        "back",
        "left",
        "right",
        "up",
        "down",
        "center",
        "menu",
        "search",
        "enter",
        "delete",
        "del",
        "recent",
        "recents",
        "volume_up",
        "volume_down",
        "volume_mute",
        "camera",
        "power",
    }
)


def _detect_lossy_text(elements: list[Any]) -> tuple[bool, str | None]:
    """Did the accessibility tree hand us text it could not represent?

    A U+FFFD in a label means the real glyph never reached us. It happens on
    formula/equation rendering, some custom fonts, and WebView content: prose survives,
    the interesting part becomes "?". Returning that silently is the worst outcome - the
    agent believes it read the screen, reports an observation that omits the very thing
    under test, or starts eyeballing screenshots on its own. Flag it and name the recovery.
    """
    hits = 0
    for e in elements:
        for attr in ("text", "content_desc"):
            v = getattr(e, attr, None)
            if isinstance(v, str) and "\ufffd" in v:
                hits += v.count("\ufffd")
    if not hits:
        return False, None
    return True, (
        f"{hits} unrepresentable character(s) in hierarchy text (U+FFFD): formula/WebView "
        "content did not survive the accessibility tree. Re-read with "
        "`aua analyze --source vision` (OCR) before judging anything that depends on that text."
    )


class Engine:
    def __init__(
        self,
        config: Config,
        *,
        device: Device | None = None,
        factory: ProviderFactory | None = None,
        platform: PlatformAdapter | None = None,
    ) -> None:
        self.config = config
        self._device = device
        self._platform = platform
        self._platform_factory = PlatformFactory(config)
        self.factory = factory or ProviderFactory(config)
        self._mem: AppMemoryStore | None = None
        self._version_cache: dict[str, str | None] = {}
        self._flag_context_checked_at: dict[str, float] = {}
        # Session default for --with-image (CLI global / MCP configure); per-call wins.
        self._default_with_image: bool | str | None = (
            config.output.with_image if config.output.with_image else None
        )
        self._capture: Any = None  # CaptureBuffer | None — set by capture_start
        # Pixel signature taken just before a state-changing action; consumed by
        # ``_await_post_action_ready`` so observe does not return a mid-transition tree.
        self._pre_action_sig: tuple[float, ...] | None = None
        self._pre_action_tree_fp: tuple[str, ...] | None = None
        # Pre-action screen shape for the change summary, and the last activity we managed to
        # read. The activity is chained across observations rather than sampled before each
        # action, so a sequence of actions gets its before/after comparison at no extra cost.
        self._pre_action_state: dict[str, Any] | None = None
        # The first folded observation consumes `_pre_action_state`. An action-bound await then
        # re-observes the settled destination and must compare it with that same original screen,
        # not with an absent baseline that can falsely report `changed: false`.
        self._action_observation_baseline: dict[str, Any] | None = None
        self._last_activity: str | None = None
        # Lease context: which agent this engine speaks for, what it needs, and what it got.
        # Set by the CLI/MCP layer before the device is first touched.
        self._flows_cache: dict[str, list[str]] = {}
        self._lease_owner: str | None = None
        self._lease_needs: list[str] | None = None
        self._lease_serial: str | None = None
        self._lease_owner_resolved: str | None = None
        self._lease_wait_s: float = 0.0
        self._lease_waited_ms: int = 0
        from .perf import GateCache, HierarchyPrefetch, SettleProfiles

        self._prefetch = HierarchyPrefetch()
        self._settle_profiles = SettleProfiles()
        self._gate_cache = GateCache()
        self._last_mem_fp: str | None = None
        self._last_known_screen: str | None = None
        self._last_action_kind: str | None = None
        self._last_action_site: tuple[str, str] | None = None
        self._last_analyze_elements: list[Element] | None = None
        self._last_hierarchy_hash: str | None = None
        self._last_analyze_result: AnalyzeResult | None = None
        self._mem_lock = threading.Lock()
        self._mem_threads_lock = threading.Lock()
        self._mem_thread: threading.Thread | None = None
        self._mem_threads: list[threading.Thread] = []
        self._claimed_instance_token: str | None = None
        self._action_recording_suppression = 0
        # Set only by the warm daemon/MCP job manager. Supported wait loops consult the event
        # between device reads; the manager object is transport state, intentionally typed Any
        # here to avoid making the interface-agnostic Engine import its adapter.
        self._job_cancel_event: threading.Event | None = None
        self._aua_job_manager: Any = None

    def _effective_with_image(self, with_image: bool | str | None) -> bool | str | None:
        """Per-call ``with_image`` overrides the session default; ``False`` forces off."""
        if with_image is False:
            return None
        if with_image is not None:
            return with_image
        return self._default_with_image

    # ----------------------------------------------------------------- device

    @property
    def platform(self) -> PlatformAdapter:
        """The selected platform strategy, created only when first needed."""

        if self._platform is None:
            self._platform = self._platform_factory.create()
        return self._platform

    def _connect_target(self, target_id: str | None) -> Device:
        # AUA historically exposed ``engine.connect`` as an informal injection seam. Preserve
        # it during this migration so existing embedders do not have to move atomically.
        if self.platform.name == "android" and connect is not _DEFAULT_ANDROID_CONNECT:
            return connect(target_id)
        return self.platform.connect(target_id)

    def _list_targets(self) -> list[DeviceInfo]:
        if self.platform.name == "android" and list_devices is not _DEFAULT_ANDROID_LIST_DEVICES:
            return list_devices()
        return self.platform.list_targets()

    @property
    def device(self) -> Device:
        """Lazily connect; doctor/devices/config work without ever touching this."""
        if self._device is None:
            self._device = self._connect_target(self._lease_device())
            self._claim_memory_session()
        return self._device

    def _claim_memory_session(self) -> None:
        """Bind already-open memory to a device connected later in this Engine lifetime."""
        if self._mem is None or self._device is None:
            return
        with contextlib.suppress(Exception), self._mem_lock:
            token = self._device.instance_token()
            if token is None:
                # A transient unreadable boot id is not proof and must be retried at the next
                # session boundary; never mark the cached Device as safely claimed.
                return
            if token == self._claimed_instance_token:
                return
            self._mem.claim_session(self._device.serial, token)
            self._claimed_instance_token = token

    def _lease_device(self) -> str | None:
        """The serial this engine may use, claiming a lease on it.

        Without this, ``connect(None)`` takes "the only/first device" and two agents working
        in parallel silently drive the same emulator — each mutating the screen the other is
        reading. Nothing errors; the results are just wrong.

        Returns the configured serial untouched when leasing is off, so a single-agent setup
        and every existing script behave exactly as before.
        """
        cfg = self.config
        explicit = cfg.device.serial
        if not getattr(cfg.lease, "enabled", True):
            return explicit

        from . import leases

        needs = list(self._lease_needs or [])

        def candidates() -> list[tuple[str, dict[str, Any]]]:
            infos = [device for device in self._list_targets() if device.state == "device"]
            # Preserve each platform's preference (Android, for example, favours a disposable
            # emulator over a USB phone) without teaching the engine platform-specific identities.
            infos.sort(key=self.platform.target_preference)
            return [
                (
                    info.serial,
                    self.platform.probe_target_capabilities(info.serial) if needs else {},
                )
                for info in infos
                if info.serial
            ]

        try:
            initial = candidates()
        except Exception:
            return explicit  # cannot enumerate; let connect() produce the real error
        if not initial and not self._lease_wait_s:
            return explicit
        owner = leases.resolve_owner(self._lease_owner)
        if self._lease_wait_s:
            serial, _why, waited_ms = leases.wait_for_device(
                cfg.cache.dir,
                owner=owner,
                explicit=explicit,
                candidates=candidates,
                needs=needs,
                ttl_s=int(getattr(cfg.lease, "ttl_s", leases.DEFAULT_TTL_S)),
                wait_s=self._lease_wait_s,
            )
            self._lease_waited_ms = waited_ms
        else:
            serial, _why = leases.choose_device(
                cfg.cache.dir,
                owner=owner,
                explicit=explicit,
                candidates=initial,
                needs=needs,
                ttl_s=int(getattr(cfg.lease, "ttl_s", leases.DEFAULT_TTL_S)),
            )
            self._lease_waited_ms = 0
        self._lease_serial = serial
        self._lease_owner_resolved = owner
        return serial

    def _reset_owner_transient_state(self) -> None:
        """Drop observations and transport state when a warm daemon changes caller owner.

        Device caches are valid only for the owner that produced them.  Keeping them across a
        daemon hand-off can make a fresh numeric id, session id, or prefetched hierarchy from
        the previous owner look current to the next one even when both use the same emulator.
        Durable map knowledge stays shared; only invocation/session-local state is cleared.
        """
        self._last_activity = None
        self._pre_action_sig = None
        self._pre_action_tree_fp = None
        self._pre_action_state = None
        self._action_observation_baseline = None
        self._last_mem_fp = None
        self._last_known_screen = None
        self._last_action_kind = None
        self._last_action_site = None
        self._last_analyze_elements = None
        self._last_hierarchy_hash = None
        self._last_analyze_result = None
        self._session_id: str | None = None
        self._prefetch.invalidate()
        self._gate_cache = type(self._gate_cache)()

    def _flows_for(self, package: str | None) -> list[str]:
        """Saved journeys for *package*, as ``name(PARAM, …)``.

        A flow replays a whole sequence — launch, taps, waits, cross-app auth — in one call,
        and one had been sitting saved and parameterised for this project with no agent ever
        running it. `flow` appeared 19 times in the long guide and zero times in anything an
        agent actually reads: not the orientation block, not the analyze header. That is the
        same omission that kept `goto` unused across five runs.

        Cached per package/context and directory fingerprint. Flows are deliberately editable
        YAML, so a long-lived daemon must notice save/delete/rename/manual edits without a
        restart; statting the small flat directory is cheaper than parsing it every frame.
        """
        if not package:
            return []
        # A package with no package-matching cursor is in its default context until runtime
        # flags prove otherwise. Treating that as None hid every auto-recorded default-context
        # flow on the first frame after returning from a foreign app.
        context_id: str | None = DEFAULT_CONTEXT_ID
        if self._memory is not None and self._device is not None:
            with contextlib.suppress(Exception):
                session = self._memory.load_session(self._device.serial)
                if session.package == package:
                    context_id = session.active_context_id
        from .flows import FlowStore

        store = FlowStore(self.config.memory)
        fingerprint: tuple[tuple[str, int, int], ...] = ()
        with contextlib.suppress(OSError):
            fingerprint = tuple(
                (path.name, stat.st_mtime_ns, stat.st_size)
                for path in sorted(store.flows_dir().glob("*.yaml"))
                if (stat := path.stat())
            )
        cache_key = f"{package}\0{context_id or ''}\0{fingerprint!r}"
        cached = self._flows_cache.get(cache_key)
        if cached is not None:
            return cached
        names: list[str] = []
        with contextlib.suppress(Exception):
            for flow in store.list():
                if flow.get("error"):
                    continue
                if flow.get("app") not in (None, package):
                    continue
                if flow.get("context_id") not in (None, context_id):
                    continue
                params = ", ".join(flow.get("params") or [])
                runnable = flow.get("storage_name") or flow["name"]
                names.append(f"{runnable}({params})" if params else str(runnable))
        self._flows_cache[cache_key] = names
        return names

    def renew_lease(self) -> None:
        """Heartbeat the current lease — called from inside long waits.

        A single ``--until`` can block 90-120s. Without a heartbeat mid-wait, a shorter TTL
        would expire while the holder is still actively driving the device.
        """
        serial = getattr(self, "_lease_serial", None)
        owner = getattr(self, "_lease_owner_resolved", None)
        if not serial or not owner:
            return
        from . import leases

        with contextlib.suppress(Exception):
            leases.renew(self.config.cache.dir, serial, owner=owner)

    def list_devices(self) -> list[DeviceInfo]:
        return self._list_targets()

    # ----------------------------------------------------------------- capture

    def _context(self) -> tuple[Device, int, int]:
        # window_size is memoized on the device; no app_current RPC on the hot path.
        device = self.device
        w, h = device.window_size()
        return device, w, h

    def _capture_hierarchy(
        self, device: Device, w: int, h: int
    ) -> tuple[list[Element], str | None, str]:
        perf = self.config.perf
        if perf.prefetch:
            slot = self._prefetch.take()
            if slot is not None:
                xml_hash = hashlib.sha1(slot.xml.encode()).hexdigest()
                return slot.elements, slot.package, xml_hash

        compressed = bool(self.config.device.compressed_hierarchy)
        raw_tree = self.platform.dump_tree(device, compact=compressed)
        tree_hash = hashlib.sha1(raw_tree.encode()).hexdigest()
        normalized = self.platform.normalize_tree(
            raw_tree,
            (w, h),
            ignored_app_ids=self.config.memory.ignore_packages,
        )
        return normalized.elements, normalized.app_id, tree_hash

    def _kick_hierarchy_prefetch(self) -> None:
        """Speculatively dump+parse the hierarchy for the next analyze."""
        if not self.config.perf.prefetch:
            return
        if self._device is None:
            return
        device = self._device
        platform = self.platform
        compressed = bool(self.config.device.compressed_hierarchy)
        try:
            w, h = device.window_size()
        except Exception:  # pragma: no cover - device mid-disconnect
            return

        def dump() -> str:
            return platform.dump_tree(device, compact=compressed)

        def parse(raw_tree: str) -> tuple[list[Element], str | None]:
            normalized = platform.normalize_tree(
                raw_tree,
                (w, h),
                ignored_app_ids=self.config.memory.ignore_packages,
            )
            return normalized.elements, normalized.app_id

        self._prefetch.kick(dump, parse)

    def _screenshot(self, *, max_reuse_ms: float = 50.0) -> ScreenImage:
        """Prefer a fresh-enough capture-buffer frame; else take a device screenshot."""
        perf = self.config.perf
        if perf.reuse_capture_frames and self._capture is not None:
            with contextlib.suppress(Exception):
                age = self._capture.latest_age_ms()
                img = self._capture.latest_frame()
                if img is not None and age is not None and age <= max_reuse_ms:
                    return img
        return self.device.screenshot()

    def _start_hierarchy_ocr(self, *, with_ocr: bool | None) -> _PendingOcr | None:
        """Start the macOS OCR augmenter before the hierarchy capture begins.

        This intentionally selects only Apple Vision from the configured OCR chain. A
        heavyweight cross-platform OCR fallback must not silently run on every hierarchy
        call; those providers remain available to the ordinary vision fallback.
        """
        want_ocr = self.config.ocr.enabled if with_ocr is None else with_ocr
        if (
            not want_ocr
            or not self.config.ocr.augment_hierarchy
            or not self.factory.is_enabled("ocr")
        ):
            return None
        chain = self.factory.build_chain("ocr")
        provider = next(
            (
                item
                for item in chain.providers
                if item.name == "apple_vision" and isinstance(item, OcrProvider)
            ),
            None,
        )
        if provider is None or not provider.is_available().ok:
            return None
        try:
            # The daemon's rolling capture normally makes this a memory read. Keeping the
            # screenshot on the caller thread avoids concurrent ADB/uiautomator RPCs.
            image = self._screenshot(max_reuse_ms=250.0)
        except Exception as exc:
            logger.info("parallel hierarchy OCR could not capture a screenshot: %s", exc)
            return None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aua-apple-ocr")
        started_at = time.perf_counter()
        future = executor.submit(provider.recognize, image)
        return _PendingOcr(image, provider, future, executor, started_at)

    def _finish_hierarchy_ocr(
        self, pending: _PendingOcr | None
    ) -> tuple[list[TextBox], str | None, ScreenImage | None]:
        if pending is None:
            return [], None, None
        timed_out = False
        try:
            elapsed = time.perf_counter() - pending.started_at
            timeout = max(0.0, self.config.timeouts.vision_ms / 1000.0 - elapsed)
            texts = pending.future.result(timeout=timeout)
            return texts, pending.provider.name, pending.image
        except FuturesTimeout:
            timed_out = True
            pending.future.cancel()
            logger.warning("parallel hierarchy OCR timed out")
            return [], None, pending.image
        except Exception as exc:
            logger.info("parallel hierarchy OCR unavailable: %s", exc)
            return [], None, pending.image
        finally:
            pending.executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

    def _fuse_hierarchy_ocr(
        self,
        elements: list[Element],
        package: str | None,
        xml_hash: str | None,
        pending: _PendingOcr | None,
    ) -> _HierarchyObservation:
        """Finish a pending OCR job and fuse kept boxes onto the hierarchy elements."""
        texts, provider, image = self._finish_hierarchy_ocr(pending)
        ocr_elements: list[Element] = []
        if provider is not None:
            from . import merge

            start_id = max((element.id for element in elements), default=-1) + 1
            ocr_elements = merge.merge_vision([], texts, start_id=start_id)
            if self.config.ocr.drop_redundant:
                # Withhold readings of text the tree already reports. Provenance is worth
                # keeping where OCR *adds* something - web content, a lossy-text repair, a
                # surface the tree cannot see - but a second copy of text already described
                # is not evidence to reconcile. Measured on one app screen: 14 of 16
                # readings were pure duplication, and one of the remaining two was a misread
                # ("Al" for "AI") that survived only because it differed from the truth.
                # Those cost tokens on every observation and let a wrong label be quoted as
                # fact. See selectors.drop_redundant_ocr for what counts as redundant.
                keep = {id(el) for el in drop_redundant_ocr([*elements, *ocr_elements])}
                ocr_elements = [el for el in ocr_elements if id(el) in keep]
        return _HierarchyObservation(
            elements,
            package,
            xml_hash,
            texts,
            ocr_elements,
            provider,
            image,
        )

    def _capture_hierarchy_with_ocr(
        self,
        device: Device,
        w: int,
        h: int,
        *,
        with_ocr: bool | None,
    ) -> _HierarchyObservation:
        """Capture hierarchy, optionally fused with Apple OCR.

        ``with_ocr=True`` keeps the parallel overlap (OCR starts before hierarchy).
        Auto mode (``None``) captures hierarchy first, then skips OCR entirely when the
        map already knows this screen is hierarchy-sufficient — experience-based cheap
        analyze without risking unknown screens. Forced ``False`` is hierarchy-only.
        """
        if with_ocr is False:
            elements, package, xml_hash = self._capture_hierarchy(device, w, h)
            return _HierarchyObservation(elements, package, xml_hash, [], [], None, None)

        if with_ocr is True:
            # Caller forced OCR — overlap screenshot OCR with the hierarchy dump.
            pending = self._start_hierarchy_ocr(with_ocr=True)
            try:
                elements, package, xml_hash = self._capture_hierarchy(device, w, h)
            except BaseException:
                if pending is not None:
                    pending.future.cancel()
                    pending.executor.shutdown(wait=False, cancel_futures=True)
                raise
            return self._fuse_hierarchy_ocr(elements, package, xml_hash, pending)

        # Auto: hierarchy first so we can consult the map before paying for OCR.
        elements, package, xml_hash = self._capture_hierarchy(device, w, h)
        if self._map_skips_ocr(device, package, elements, h):
            return _HierarchyObservation(elements, package, xml_hash, [], [], None, None)
        pending = self._start_hierarchy_ocr(with_ocr=True)
        return self._fuse_hierarchy_ocr(elements, package, xml_hash, pending)

    def _map_skips_ocr(
        self,
        device: Device,
        package: str | None,
        elements: list[Element],
        height: int,
    ) -> bool:
        """Skip parallel OCR only when map evidence says this known screen never needed it."""
        mem = self._memory
        if mem is None or not package:
            return False
        try:
            name = mem.recognize_screen(
                device.serial,
                package=package,
                elements=elements,
                screen_height=height,
            )
            if not name:
                return False
            app = mem.load(package)
            rec = app.screens.get(name) if app else None
            return bool(rec and screen_skips_ocr(rec))
        except Exception as exc:
            logger.debug("map OCR-skip check failed: %s", exc)
            return False

    def _run_vision(
        self,
        device: Device,
        *,
        with_ocr: bool | None,
        start_id: int = 0,
        image: ScreenImage | None = None,
        ocr_result: tuple[list[TextBox], str] | None = None,
    ) -> tuple[list[Element], list[str], ScreenImage]:
        from . import merge

        img = image or self._screenshot(max_reuse_ms=80.0)
        providers_used: list[str] = []
        detections: list[DetBox] = []
        if self.factory.is_enabled("detection"):
            chain = self.factory.build_chain("detection")
            if chain.providers:
                try:
                    detections, name = run_chain(
                        chain,
                        lambda p: p.detect(img),  # type: ignore[attr-defined]
                        timeout_s=self.config.timeouts.detection_ms / 1000.0,
                    )
                    providers_used.append(name)
                except ProviderError as exc:
                    logger.info("detection unavailable, continuing OCR-only: %s", exc)

        texts: list[TextBox] = []
        want_ocr = self.config.ocr.enabled if with_ocr is None else with_ocr
        if ocr_result is not None:
            texts, name = ocr_result
            providers_used.append(name)
        elif want_ocr and self.factory.is_enabled("ocr"):
            chain = self.factory.build_chain("ocr")
            if chain.providers:
                try:
                    texts, name = run_chain(
                        chain,
                        lambda p: p.recognize(img),  # type: ignore[attr-defined]
                        timeout_s=self.config.timeouts.vision_ms / 1000.0,
                    )
                    providers_used.append(name)
                except ProviderError as exc:
                    logger.info("ocr unavailable: %s", exc)

        elements = merge.merge_vision(detections, texts, iou_threshold=0.5, start_id=start_id)
        return elements, providers_used, img

    def _repair_lossy_text(self, device: Device, elements: list[Element]) -> tuple[int, str | None]:
        """Fill in hierarchy labels the accessibility tree could not represent, using OCR.

        The tree sometimes hands back U+FFFD instead of the real glyphs - formula and
        WebView content especially. Element *structure* is fine, only the text is lost, so
        replacing the whole observation with a vision pass would be wasteful and would
        double the payload. Instead run OCR once and graft the recognised text onto the
        elements that are broken, matched by geometric overlap.

        Costs one OCR pass (~145ms with apple_vision) and only when something is actually
        broken. Returns how many labels were repaired and which provider performed it.
        """
        broken = [e for e in elements if e.text is not None and "\ufffd" in e.text]
        if not broken:
            return 0, None
        try:
            img = self._screenshot(max_reuse_ms=250.0)
            chain = self.factory.build_chain("ocr")
            if not chain.providers:
                return 0, None
            texts, name = run_chain(
                chain,
                lambda p: p.recognize(img),  # type: ignore[attr-defined]
                timeout_s=self.config.timeouts.vision_ms / 1000.0,
            )
        except Exception as exc:  # never let a repair attempt break the analyze
            logger.info("lossy-text repair unavailable: %s", exc)
            return 0, None

        def overlap(a: Any, b: Any) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = ix * iy
            if inter <= 0:
                return 0.0
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / area_b

        repaired = 0
        for el in broken:
            box = getattr(el, "bounds", None)
            if not box:
                continue
            hits = []
            for tb in texts:
                tbb = getattr(tb, "bounds", None) or getattr(tb, "box", None)
                if not tbb:
                    continue
                # Keep OCR text whose box sits mostly inside the broken element.
                if overlap(tuple(box), tuple(tbb)) >= 0.5 and (tb.text or "").strip():
                    hits.append((tbb[1], tbb[0], tb.text.strip()))
            if not hits:
                continue
            hits.sort()  # reading order: top-to-bottom, then left-to-right
            merged = " ".join(h[2] for h in hits)
            if merged and merged != el.text:
                el.text = merged
                repaired += 1
        if repaired:
            logger.info("repaired %d lossy label(s) with OCR (%s)", repaired, name)
        return repaired, (name if repaired else None)

    def ask_screen(self, question: str) -> dict[str, Any]:
        """Ask the configured grounding model about screenshot + current element graph."""
        question = question.strip()
        if not question:
            raise UsageError("ask needs a non-empty screen question")
        if not self.factory.is_enabled("grounding"):
            raise UsageError(
                "screen questions need grounding.enabled: true",
                hint="Enable grounding and configure a screen-analysis provider such as gemini or openai.",
            )

        t0 = time.perf_counter()
        device, width, height = self._context()
        observation = self._capture_hierarchy_with_ocr(device, width, height, with_ocr=None)
        elements = observation.elements + observation.ocr_elements
        package = observation.package
        app = device.current_app()
        package = app.get("package") or package
        activity = app.get("activity") or None
        image = observation.image or self._screenshot(max_reuse_ms=80.0)
        graph: list[dict[str, Any]] = []
        for element in elements:
            item: dict[str, Any] = {
                "id": element.id,
                "type": element.type,
                "bounds": list(element.bounds),
            }
            if element.text:
                item["text"] = element.text
            if element.content_desc:
                item["desc"] = element.content_desc
            if element.resource_id:
                item["rid"] = _id_tail(element.resource_id)
            if element.source is not Source.hierarchy:
                item["source"] = element.source.value
            if element.confidence is not None:
                item["confidence"] = round(element.confidence, 4)
            for flag in ("clickable", "checkable", "checked", "selected", "scrollable"):
                value = getattr(element, flag)
                if value:
                    item[flag] = True
            graph.append(item)
        chain = self.factory.build_chain("grounding")
        result, provider = run_chain(
            chain,
            lambda item: item.ask(image, question, graph),  # type: ignore[attr-defined]
            timeout_s=self.config.timeouts.grounding_ms / 1000.0,
        )
        if not isinstance(result, ScreenAnalysisResult):  # pragma: no cover - contract guard
            raise ProviderError("grounding", [(provider, "invalid screen-analysis result")])
        return {
            "question": question,
            "screen": {
                "width": width,
                "height": height,
                "package": package,
                "activity": activity,
            },
            "provider": provider,
            "model": result.model,
            "perception_providers": (
                [observation.ocr_provider] if observation.ocr_provider else []
            ),
            "duration_ms": round((time.perf_counter() - t0) * 1000),
            "usage": result.usage,
            "input_image": result.input_image,
            "graph_elements": len(graph),
            "analysis": result.analysis,
        }

    # ----------------------------------------------------------------- analyze

    def _resolve_pins(self, source: str | None, strategy: str | None) -> tuple[bool, bool, bool]:
        """Return (force_hierarchy, force_vision, pin_grounding). strategy > source."""
        s = (strategy or "").lower()
        if s in ("text", "selector", "hierarchy"):
            return True, False, False
        if s == "vision":
            return False, True, False
        if s == "grounding":
            return False, True, True
        src = (source or "auto").lower()
        if src == "hierarchy":
            return True, False, False
        if src == "vision":
            return False, True, False
        return False, False, False

    def analyze(
        self,
        *,
        source: str = "auto",
        with_ocr: bool | None = None,
        query: str | None = None,
        annotate: bool | str | None = None,
        with_image: bool | str | None = None,
        strategy: str | None = None,
        cheap: bool = False,
        deep: bool = False,
        no_cache: bool = False,
        record: bool = True,
    ) -> AnalyzeResult:
        wi = self._effective_with_image(with_image)
        if wi:
            return self._with_raw_image(
                self.analyze(
                    source=source,
                    with_ocr=with_ocr,
                    query=query,
                    annotate=annotate,
                    strategy=strategy,
                    cheap=cheap,
                    deep=deep,
                    no_cache=no_cache,
                    record=record,
                    with_image=False,  # already applying session/per-call image below
                ),
                wi,
            )
        ceiling = routing.resolve_ceiling(self.config.routing.max_tier, cheap=cheap, deep=deep)
        force_hier, force_vis, pin_grounding = self._resolve_pins(source, strategy)
        # An explicit --strategy pin is a per-call opt-in: raise the ceiling so the pinned
        # tier is actually reachable even if routing.max_tier is lower (still never an
        # *implicit* paid escalation — the user named the tier).
        if pin_grounding:
            ceiling = Tier.grounding
        elif force_vis and not routing.allows(Tier.vision, ceiling):
            ceiling = Tier.vision
        if query:
            return self._analyze_query(
                query,
                ceiling=ceiling,
                force_hierarchy=force_hier,
                force_vision=force_vis,
                pin_grounding=pin_grounding,
                with_ocr=with_ocr,
                annotate=annotate,
                no_cache=no_cache,
            )
        return self._analyze_screen(
            ceiling=ceiling,
            force_hierarchy=force_hier,
            force_vision=force_vis,
            with_ocr=with_ocr,
            annotate=annotate,
            no_cache=no_cache,
            record=record,
        )

    def _analyze_screen(
        self,
        *,
        ceiling: Tier,
        force_hierarchy: bool,
        force_vision: bool,
        with_ocr: bool | None,
        annotate: bool | str | None,
        no_cache: bool,
        record: bool = True,
    ) -> AnalyzeResult:
        t0 = time.perf_counter()
        device, w, h = self._context()
        if self.config.memory.enabled:
            # Re-check a long-lived daemon's boot identity before it reads or writes any
            # serial-scoped cursor. This is a cheap no-op for the same token and retries a
            # transiently unreadable first token; a reboot on the same serial clears history.
            _ = self._memory
            self._claim_memory_session()
        providers_used: list[str] = []
        img: ScreenImage | None = None
        package: str | None = None
        activity: str | None = None

        elements: list[Element] = []
        hierarchy_elements: list[Element] = []
        hierarchy_observation: _HierarchyObservation | None = None
        screen_source = ScreenSource.hierarchy
        tier_used = Tier.hierarchy
        path = PathKind.hierarchy

        if not force_vision:
            hierarchy_observation = self._capture_hierarchy_with_ocr(
                device, w, h, with_ocr=with_ocr
            )
            hierarchy_elements = hierarchy_observation.elements
            elements = hierarchy_elements + hierarchy_observation.ocr_elements
            package = hierarchy_observation.package
            xml_hash = hierarchy_observation.xml_hash
            img = hierarchy_observation.image
            if hierarchy_observation.ocr_provider:
                providers_used.append(hierarchy_observation.ocr_provider)
                screen_source = ScreenSource.mixed
            if (
                self.config.perf.skip_unchanged_analyze
                and xml_hash
                and xml_hash == self._last_hierarchy_hash
                and self._last_analyze_result is not None
                and not no_cache
                # A warm daemon can move between apps whose accessibility dump happens to
                # hash the same during a transition. Reusing the previous payload in that case
                # creates an impossible observation: the new hierarchy under the old package.
                # Package identity is part of a screen observation, not optional metadata.
                and package == self._last_analyze_result.screen.package
                # Identical accessibility XML does not imply identical pixels (canvas,
                # charts, video, custom rendering). Current OCR must reach the caller.
                and not hierarchy_observation.ocr_provider
            ):
                prev = self._last_analyze_result
                # Reusing the PAYLOAD is fine — the tree really is identical — but the memory
                # side-effects are not the tree's properties and must still run:
                #  - `known_screen`: the map learns between calls, so the first analyze of a new
                #    screen answers None and every later one would repeat that None forever.
                #  - a pending route deferred by a mid-transition observe snapshot is drawn by
                #    the NEXT recording analyze; skipping it dropped the edge silently, so the
                #    map stopped learning exactly when the screen sat still.
                # `_record_screen_safe` has its own unchanged-screen fast path, so this is a
                # map read rather than a re-record.
                known = prev.meta.known_screen
                if record:
                    known, _hints = self._record_screen_safe(
                        device, package, activity, prev.elements, tier_used, h
                    )
                reused = prev.model_copy(
                    update={
                        "meta": prev.meta.model_copy(
                            update={
                                "duration_ms": int((time.perf_counter() - t0) * 1000),
                                "unchanged": True,
                                "fingerprint": xml_hash,
                                "known_screen": known,
                                "via": "hierarchy-unchanged",
                                "element_diff": {
                                    "added": [],
                                    "removed": [],
                                    "changed": [],
                                    "prev_count": len(prev.elements),
                                    "curr_count": len(prev.elements),
                                }
                                if self.config.perf.differential
                                else prev.meta.element_diff,
                            }
                        )
                    }
                )
                # Reusing the payload must not skip the side effect callers depend on: every
                # action invalidates the id cache, so an unchanged screen returned straight
                # from memory left NOTHING on disk and the next `tap <id>` died with "no
                # cached analyze result". The ids are only usable because analyze persists them.
                if not no_cache:
                    self._write_cache(reused)
                return reused
        else:
            xml_hash = None

        use_vision = force_vision
        xml_dump: str | None = None
        if not force_vision and not force_hierarchy:
            decision = self._gate_decide(hierarchy_elements, package=package, activity=activity)
            if decision.use_vision and routing.allows(Tier.vision, ceiling):
                # Prefer WebView DOM/a11y enrichment over OCR when the tree looks hollow.
                wv_cfg = self.config.perception.webview
                if wv_cfg.enabled:
                    from . import webview as webview_mod

                    xml_dump = self.platform.dump_tree(
                        device,
                        compact=bool(self.config.device.compressed_hierarchy),
                    )
                    if webview_mod.should_try_webview(hierarchy_elements, xml_dump):
                        shell = None
                        if wv_cfg.cdp:
                            shell = lambda cmd: str(  # noqa: E731
                                device.shell(cmd) if hasattr(device, "shell") else ""
                            )
                        enriched = webview_mod.enrich(
                            xml_dump,
                            screen_size=(w, h),
                            shell=shell,
                            cdp=wv_cfg.cdp,
                        )
                        if len(enriched) >= wv_cfg.min_elements:
                            hierarchy_elements = enriched
                            elements = enriched + (
                                hierarchy_observation.ocr_elements
                                if hierarchy_observation is not None
                                else []
                            )
                            screen_source = (
                                ScreenSource.mixed
                                if hierarchy_observation is not None
                                and hierarchy_observation.ocr_provider
                                else ScreenSource.hierarchy
                            )
                            path = PathKind.hierarchy
                            providers_used.append("webview")
                            logger.info(
                                "webview enrichment: %d elements (skipping vision)", len(enriched)
                            )
                            use_vision = False
                        else:
                            use_vision = True
                            logger.info("gate → vision: %s", decision.reason)
                    else:
                        use_vision = True
                        logger.info("gate → vision: %s", decision.reason)
                else:
                    use_vision = True
                    logger.info("gate → vision: %s", decision.reason)
            elif decision.use_vision:
                logger.info("gate wants vision but ceiling=%s; staying hierarchy", ceiling.value)

        if use_vision:
            # slow fallback path: fetch full app context (incl. activity)
            app = device.current_app()
            package = app.get("package") or package
            activity = app.get("activity") or None
            precomputed_ocr = None
            if hierarchy_observation is not None and hierarchy_observation.ocr_provider:
                precomputed_ocr = (
                    hierarchy_observation.ocr_texts,
                    hierarchy_observation.ocr_provider,
                )
            start_id = max((element.id for element in elements), default=-1) + 1
            vis_elements, vision_providers, img = self._run_vision(
                device,
                with_ocr=with_ocr,
                start_id=start_id,
                image=img,
                ocr_result=precomputed_ocr,
            )
            for provider in vision_providers:
                if provider not in providers_used:
                    providers_used.append(provider)
            if hierarchy_observation is not None:
                # OCR already exists as its own raw pool. Keep detection elements (including
                # labels associated from OCR), but do not append the same OCR boxes a third time.
                vis_elements = [el for el in vis_elements if el.source is not Source.ocr]
                elements = elements + vis_elements
                screen_source = ScreenSource.mixed
            else:
                elements = vis_elements
                screen_source = ScreenSource.vision
            tier_used = Tier.vision
            path = PathKind.vision

        if record:
            ocr_helped: bool | None = None
            if hierarchy_observation is not None and with_ocr is not False:
                # Evidence for experience-based OCR skip: only count visits where OCR was
                # allowed. Forced hierarchy-only paths (goto hops) must not inflate the score.
                if hierarchy_observation.ocr_provider is not None:
                    # Not `bool(ocr_elements)`: the status-bar clock is read on every
                    # screen and never matches the tree's digits, so it survives every
                    # redundancy test and would score every visit as "OCR helped" -
                    # pinning hierarchy_only_ok at zero and disabling the skip forever.
                    ocr_helped = ocr_added_app_content(
                        [*hierarchy_observation.elements, *hierarchy_observation.ocr_elements]
                    )
                else:
                    ocr_helped = False  # skipped (map) or unavailable → hierarchy alone
            known_screen, hints = self._record_screen_safe(
                device,
                package,
                activity,
                elements,
                tier_used,
                h,
                ocr_helped=ocr_helped,
            )
        else:
            # An observe snapshot taken right after an action can be mid-transition; never
            # let it pollute memory with a transient screen (it's just fresh ids for the agent).
            known_screen, hints = None, None
        annotated = self._maybe_annotate(annotate, device, elements, img)
        ediff = None
        if self.config.perf.differential and self._last_analyze_elements is not None:
            from .perf import element_diff as _element_diff

            with contextlib.suppress(Exception):
                ediff = _element_diff(self._last_analyze_elements, elements)
        self._last_analyze_elements = list(elements)

        from .perf import elements_fingerprint

        fp = (
            None
            if hierarchy_observation is not None and hierarchy_observation.ocr_provider
            else xml_hash
        )
        if not fp:
            with contextlib.suppress(Exception):
                fp = elements_fingerprint(elements)

        # Auto-recover before reporting: a fast answer that is not usable is not an answer.
        # Only pays the OCR cost when the tree actually handed back broken text.
        # When parallel OCR ran, keep the hierarchy text untouched and expose raw OCR boxes
        # alongside it. The older repair fallback remains for hosts without Apple Vision.
        _repaired, _repair_provider = (
            self._repair_lossy_text(device, elements)
            if not use_vision
            and not (hierarchy_observation is not None and hierarchy_observation.ocr_provider)
            else (0, None)
        )
        if _repair_provider and _repair_provider not in providers_used:
            # Provenance matters: text that came from OCR is not text the app exposed.
            providers_used.append(_repair_provider)
        _lossy, _lossy_hint = _detect_lossy_text(elements)
        if _lossy and hierarchy_observation is not None and hierarchy_observation.ocr_provider:
            _lossy_hint = (
                "The hierarchy contains unrepresentable text; raw Apple Vision OCR elements "
                "are included alongside it with source=ocr. Compare both observations."
            )
        result = AnalyzeResult(
            screen=Screen(
                width=w, height=h, package=package, activity=activity, source=screen_source
            ),
            elements=elements,
            meta=Meta(
                duration_ms=int((time.perf_counter() - t0) * 1000),
                tier_used=tier_used,
                path=path,
                providers_used=providers_used,
                known_screen=known_screen,
                known_routes=hints.known_routes if hints else [],
                suggested_gotos=hints.suggested_gotos if hints else [],
                slow_controls=self._slow_controls_safe(known_screen),
                suggested_deeplinks=hints.suggested_deeplinks if hints else [],
                research_tasks=hints.research_tasks if hints else [],
                flows=self._flows_for(package),
                ask=hints.ask if hints else None,
                map_hint=hints.map_hint if hints else None,
                capture_hint=self._capture_hint(),
                lossy_text=_lossy,
                lossy_hint=_lossy_hint,
                ocr_repaired=_repaired,
                annotated_image=annotated,
                device_serial=device.serial,
                element_diff=ediff,
                unchanged=False,
                fingerprint=fp,
                via=path.value if hasattr(path, "value") else str(path),
            ),
        )
        if xml_hash:
            self._last_hierarchy_hash = xml_hash
        self._last_analyze_result = result
        if not no_cache:
            self._write_cache(result)
        if self.config.perf.prefetch:
            self._kick_hierarchy_prefetch()
        return result

    def _gate_decide(
        self,
        elements: list[Element],
        *,
        package: str | None,
        activity: str | None,
    ) -> Any:
        from . import gate
        from .perf import GateCache

        cfg = self.config.perception.gate
        if self.config.perf.gate_cache:
            key = GateCache.key(elements, package=package, activity=activity)
            hit = self._gate_cache.get(key)
            if hit is not None:
                return hit
            decision = gate.decide(elements, package=package, activity=activity, cfg=cfg)
            self._gate_cache.put(key, decision)
            return decision
        return gate.decide(elements, package=package, activity=activity, cfg=cfg)

    def _analyze_query(
        self,
        query: str,
        *,
        ceiling: Tier,
        force_hierarchy: bool,
        force_vision: bool,
        pin_grounding: bool,
        with_ocr: bool | None,
        annotate: bool | str | None,
        no_cache: bool,
    ) -> AnalyzeResult:
        t0 = time.perf_counter()
        device, w, h = self._context()
        package: str | None = None
        activity: str | None = None
        providers_used: list[str] = []
        pool: list[Element] = []
        hierarchy_elements: list[Element] = []
        hierarchy_observation: _HierarchyObservation | None = None
        img: ScreenImage | None = None
        tier_used = Tier.hierarchy
        screen_source = ScreenSource.hierarchy
        path = PathKind.hierarchy
        best: Element | None = None
        best_score = 0.0
        known_screen: str | None = None
        hints: NavHints | None = None

        # --- T1/T2: satisfy from the hierarchy first (cheap-first) ---
        if not force_vision:
            hierarchy_observation = self._capture_hierarchy_with_ocr(
                device, w, h, with_ocr=with_ocr
            )
            hierarchy_elements = hierarchy_observation.elements
            pool = hierarchy_elements + hierarchy_observation.ocr_elements
            package = hierarchy_observation.package
            img = hierarchy_observation.image
            if hierarchy_observation.ocr_provider:
                providers_used.append(hierarchy_observation.ocr_provider)
                screen_source = ScreenSource.mixed
            tier_used = Tier.selector
            known_screen, hints = self._record_screen_safe(
                device, package, activity, pool, Tier.hierarchy, h
            )
            cand, score = self._match_query(query, pool)
            if cand is not None and score > best_score:
                best, best_score = cand, score
            if best_score >= QUERY_CONFIDENT and not pin_grounding:
                return self._finish_query(
                    best,
                    w,
                    h,
                    package,
                    activity,
                    screen_source,
                    Tier.selector,
                    PathKind.hierarchy,
                    providers_used,
                    device,
                    annotate,
                    img,
                    no_cache,
                    t0,
                    known_screen,
                    hints,
                )

        # --- T3: vision, if allowed and useful ---
        want_vision = force_vision
        if not force_vision and routing.allows(Tier.vision, ceiling):
            decision = self._gate_decide(hierarchy_elements, package=package, activity=activity)
            kind = routing.classify_query(query)
            want_vision = decision.use_vision or kind is routing.QueryKind.visual or pin_grounding

        if want_vision and routing.allows(Tier.vision, ceiling):
            app = device.current_app()
            package = app.get("package") or package
            activity = app.get("activity") or None
            precomputed_ocr = None
            if hierarchy_observation is not None and hierarchy_observation.ocr_provider:
                precomputed_ocr = (
                    hierarchy_observation.ocr_texts,
                    hierarchy_observation.ocr_provider,
                )
            start_id = max((element.id for element in pool), default=-1) + 1
            vis_elements, vprov, img = self._run_vision(
                device,
                with_ocr=with_ocr,
                start_id=start_id,
                image=img,
                ocr_result=precomputed_ocr,
            )
            for provider in vprov:
                if provider not in providers_used:
                    providers_used.append(provider)
            if hierarchy_observation is not None:
                vis_elements = [el for el in vis_elements if el.source is not Source.ocr]
            pool = pool + vis_elements
            screen_source = ScreenSource.mixed if pool and vis_elements else ScreenSource.vision
            tier_used = Tier.vision
            path = PathKind.vision
            if force_vision:  # hierarchy block was skipped → record the screen from vision
                known_screen, hints = self._record_screen_safe(
                    device, package, activity, pool, Tier.vision, h
                )
            cand, score = self._match_query(query, vis_elements)
            if cand is not None and score > best_score:
                best, best_score = cand, score
            if best_score >= QUERY_CONFIDENT and not pin_grounding:
                return self._finish_query(
                    best,
                    w,
                    h,
                    package,
                    activity,
                    screen_source,
                    Tier.vision,
                    path,
                    providers_used,
                    device,
                    annotate,
                    img,
                    no_cache,
                    t0,
                    known_screen,
                    hints,
                )

        # --- T4: grounding VLM, only if explicitly allowed (never silent/paid by default) ---
        grounding_ok = (
            routing.allows(Tier.grounding, ceiling)
            and self.factory.is_enabled("grounding")
            and (
                pin_grounding or routing.classify_query(query) is not routing.QueryKind.resource_id
            )
        )
        if best_score < QUERY_CONFIDENT and grounding_ok:
            chain = self.factory.build_chain("grounding")
            if chain.providers:
                if img is None:
                    img = device.screenshot()
                try:
                    loc, name = run_chain(
                        chain,
                        lambda p: p.locate(img, query),  # type: ignore[attr-defined]
                        is_empty=lambda r: r is None,
                        timeout_s=self.config.timeouts.grounding_ms / 1000.0,
                    )
                    providers_used.append(name)
                    grounded = self._map_grounding(loc, pool, w, h)
                    if grounded is not None:
                        return self._finish_query(
                            grounded,
                            w,
                            h,
                            package,
                            activity,
                            ScreenSource.mixed,
                            Tier.grounding,
                            PathKind.vision,
                            providers_used,
                            device,
                            annotate,
                            img,
                            no_cache,
                            t0,
                        )
                except ProviderError as exc:
                    logger.info("grounding unavailable: %s", exc)
        elif best_score < QUERY_CONFIDENT and self.factory.is_enabled("grounding"):
            logger.info(
                "not escalating to grounding: ceiling=%s (use --deep or raise routing.max_tier)",
                ceiling.value,
            )

        # --- best-effort or not-found ---
        chosen = best if best is not None and best_score >= QUERY_SOFT else None
        return self._finish_query(
            chosen,
            w,
            h,
            package,
            activity,
            screen_source,
            tier_used,
            path,
            providers_used,
            device,
            annotate,
            img,
            no_cache,
            t0,
            known_screen,
            hints,
        )

    def _finish_query(
        self,
        element: Element | None,
        w: int,
        h: int,
        package: str | None,
        activity: str | None,
        screen_source: ScreenSource,
        tier_used: Tier,
        path: PathKind,
        providers_used: list[str],
        device: Device,
        annotate: bool | str | None,
        img: ScreenImage | None,
        no_cache: bool,
        t0: float,
        known_screen: str | None = None,
        hints: NavHints | None = None,
    ) -> AnalyzeResult:
        elements = [element] if element is not None else []
        annotated = self._maybe_annotate(annotate, device, elements, img)
        result = AnalyzeResult(
            screen=Screen(
                width=w, height=h, package=package, activity=activity, source=screen_source
            ),
            elements=elements,
            meta=Meta(
                duration_ms=int((time.perf_counter() - t0) * 1000),
                tier_used=tier_used,
                path=path,
                providers_used=providers_used,
                known_screen=known_screen,
                known_routes=hints.known_routes if hints else [],
                suggested_gotos=hints.suggested_gotos if hints else [],
                slow_controls=self._slow_controls_safe(known_screen),
                suggested_deeplinks=hints.suggested_deeplinks if hints else [],
                research_tasks=hints.research_tasks if hints else [],
                flows=self._flows_for(package),
                ask=hints.ask if hints else None,
                map_hint=hints.map_hint if hints else None,
                capture_hint=self._capture_hint(),
                annotated_image=annotated,
                device_serial=device.serial,
            ),
        )
        if not no_cache:
            self._write_cache(result)
        return result

    # ----------------------------------------------------------------- query match

    def _match_query(self, query: str, elements: list[Element]) -> tuple[Element | None, float]:
        tokens = routing.salient_tokens(query)
        phrase = " ".join(tokens)
        ql = query.strip().lower()
        best: Element | None = None
        best_score = -1.0
        for el in elements:
            parts: list[str] = []
            if el.text:
                parts.append(el.text)
            if el.content_desc:
                parts.append(el.content_desc)
            if el.resource_id:
                parts.append(el.resource_id.split("/")[-1].replace("_", " "))
            hay = " ".join(parts).lower().strip()
            if not hay:
                continue
            if el.text and el.text.strip().lower() == ql or phrase and phrase in hay:
                score = 1.0
            elif tokens:
                score = sum(1 for t in tokens if t in hay) / len(tokens)
            else:
                score = 0.0
            # tie-break: prefer clickable, then smaller area
            adj = score + (0.001 if el.clickable else 0.0)
            if adj > best_score:
                best, best_score = el, adj
        if best is None:
            return None, 0.0
        return best, min(1.0, best_score)

    def _map_grounding(
        self, loc: Point | DetBox | None, pool: list[Element], w: int, h: int
    ) -> Element | None:
        from . import merge

        if loc is None:
            return None
        if isinstance(loc, Point):
            px, py = loc.x, loc.y
            # element containing the point, else nearest center
            containing = [
                e
                for e in pool
                if e.bounds[0] <= px <= e.bounds[2] and e.bounds[1] <= py <= e.bounds[3]
            ]
            if containing:
                return min(
                    containing,
                    key=lambda e: (e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]),
                )
            if pool:
                return min(pool, key=lambda e: (e.center[0] - px) ** 2 + (e.center[1] - py) ** 2)
            box = (max(0, px - 24), max(0, py - 24), min(w, px + 24), min(h, py + 24))
            return Element(
                id=0,
                type="GroundedPoint",
                bounds=box,
                center=(px, py),
                source=Source.grounding,
                confidence=loc.confidence,
                clickable=True,
            )
        # DetBox
        if pool:
            scored = [(merge.iou(loc.bounds, e.bounds), e) for e in pool]
            scored.sort(key=lambda t: t[0], reverse=True)
            if scored and scored[0][0] > 0.1:
                return scored[0][1]
        return Element(
            id=len(pool),
            type="GroundedBox",
            text=loc.label,
            bounds=loc.bounds,
            center=center_of(loc.bounds),
            source=Source.grounding,
            confidence=loc.confidence,
            clickable=loc.interactable,
        )

    # ----------------------------------------------------------------- memory (§6b)

    @property
    def _memory(self) -> AppMemoryStore | None:
        if not self.config.memory.enabled:
            return None
        if self._mem is None:
            self._mem = AppMemoryStore(self.config.memory)
            # Claim the serial's cursor for *this* device instance before anything reads it.
            # Session state is keyed by serial and serials are recycled from a small pool,
            # so without this a worker inherits its predecessor's action journal and
            # `flow save` hands back steps from another scenario. One ~10ms device read per
            # invocation, done here because this is the only place the store is built.
            #
            # `self._device`, deliberately, not `self.device`: the latter would *connect*,
            # and offline commands (`aua map --app …`) read memory with no device attached —
            # making them wait out a uiautomator2 connect timeout to learn nothing. Every
            # path that has a session worth protecting has already connected, because the
            # earliest readers here are handed a live `device` by their caller.
            self._claim_memory_session()
        return self._mem

    def _version_for(self, device: Device, package: str) -> str | None:
        """App versionName, fetched at most once per package (kept off the hot path)."""
        if package not in self._version_cache:
            try:
                self._version_cache[package] = device.app_version(package)
            except Exception:  # pragma: no cover - best effort
                self._version_cache[package] = None
        return self._version_cache[package]

    def _sync_runtime_flag_context(
        self,
        device: Device,
        package: str,
        mem: AppMemoryStore,
        *,
        force: bool = False,
    ) -> bool:
        """Discover already-active feature flags before assigning a screen context."""
        cfg = self.config.flags
        configured = package in cfg.prefs_files or package in cfg.context_keys
        if not cfg.auto_context or not configured:
            return False
        now = time.monotonic()
        if not force and now - self._flag_context_checked_at.get(package, float("-inf")) < max(
            0.0, cfg.context_refresh_s
        ):
            return False
        self._flag_context_checked_at[package] = now
        from .flags import read_context_flags

        result = read_context_flags(
            device,
            package,
            prefs_file=cfg.prefs_files.get(package),
            keys=cfg.context_keys.get(package),
            key_patterns=cfg.context_key_patterns,
        )
        if not result.verified:
            logger.debug("runtime flag context unavailable for %s: %s", package, result.reason)
            return False
        previous = mem.load_session(device.serial)
        previous_identity = (
            previous.package,
            previous.active_context_id,
            tuple(sorted(previous.active_flags.items())),
        )
        mem.activate_flag_context(
            device.serial,
            package,
            result.flags,
            app_version=self._version_for(device, package),
            verified=True,
            replace=True,
            evidence=[f"shared_prefs:{name}" for name in result.files],
        )
        current = mem.load_session(device.serial)
        changed = previous_identity != (
            current.package,
            current.active_context_id,
            tuple(sorted(current.active_flags.items())),
        )
        if changed:
            self._last_mem_fp = None
            self._last_known_screen = None
        return changed

    def _record_screen_safe(
        self,
        device: Device,
        package: str | None,
        activity: str | None,
        elements: list[Element],
        tier: Tier,
        height: int | None = None,
        *,
        ocr_helped: bool | None = None,
    ) -> tuple[str | None, NavHints | None]:
        """Auto-record the current screen + derive navigation hints; never break analyze.

        Returns ``(known_screen, hints)``. ``hints`` carries the inline affordances
        (known_routes / suggested_gotos / map_hint) so the agent gets them on the analyze
        it already runs, instead of having to remember to call ``aua map``.
        ``ocr_helped`` records whether parallel OCR contributed kept elements (for
        experience-based OCR skip on later visits).
        """
        mem = self._memory
        if mem is None or not package:
            return None, None
        perf = self.config.perf
        try:
            # Context discovery precedes the unchanged-screen fast path: a flag may have
            # changed outside AUA while the rendered hierarchy stayed temporarily equal.
            context_changed = self._sync_runtime_flag_context(device, package, mem)
            from .perf import elements_fingerprint

            fp = elements_fingerprint(elements)
            if perf.skip_unchanged_memory and not context_changed and fp == self._last_mem_fp:
                mcfg = self.config.memory
                hints = (
                    mem.navigation_hints(
                        device.serial,
                        package,
                        max_suggest=mcfg.suggest_max,
                        max_research=mcfg.research_suggest_max,
                        include_navigation=mcfg.suggest,
                        half_life_days=mcfg.rank_half_life_days,
                    )
                    if mcfg.suggest or mcfg.auto_research
                    else None
                )
                return self._last_known_screen, hints

            def _do_record() -> str | None:
                return mem.observe_screen(
                    device.serial,
                    package=package,
                    elements=elements,
                    activity=activity,
                    app_version=self._version_for(device, package),
                    tier=tier.value,
                    screen_height=height,
                    ocr_helped=ocr_helped,
                )

            mcfg = self.config.memory
            if perf.async_memory:
                # Hints come from the map as it stands; the write happens off-path.
                hints = (
                    mem.navigation_hints(
                        device.serial,
                        package,
                        max_suggest=mcfg.suggest_max,
                        max_research=mcfg.research_suggest_max,
                        include_navigation=mcfg.suggest,
                        half_life_days=mcfg.rank_half_life_days,
                    )
                    if mcfg.suggest or mcfg.auto_research
                    else None
                )

                def _bg() -> None:
                    with self._mem_lock:
                        try:
                            known = _do_record()
                            self._last_known_screen = known
                            self._last_mem_fp = fp
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("async memory record failed: %s", exc)

                t = threading.Thread(target=_bg, name="aua-mem-record", daemon=True)
                # Register the writer before exposing it to an action/save caller.  Pruning
                # after append but before start used to immediately drop the new thread
                # because ``is_alive()`` is false until ``start()`` returns, recreating the
                # exact provenance race this list is meant to prevent.
                with self._mem_threads_lock:
                    self._mem_threads = [
                        thread for thread in self._mem_threads if thread.is_alive()
                    ]
                    self._mem_thread = t
                    self._mem_threads.append(t)
                    t.start()
                # Recognise synchronously rather than reusing the remembered name: the write
                # above has not landed yet, so `self._last_known_screen` still holds the
                # PREVIOUS screen's name. Reporting it labelled the device launcher and a
                # system ANR dialog with names from the app under test's own map, and told a
                # caller that had just navigated back that it was still on the screen it left.
                # Recognition is a read of a map `navigation_hints` just loaded on this same
                # path, so it costs ~nothing; an unmapped screen answers None, which is honest
                # rather than wrong.
                return (
                    mem.recognize_screen(
                        device.serial,
                        package=package,
                        elements=elements,
                        activity=activity,
                        screen_height=height,
                    ),
                    hints,
                )

            known = _do_record()
            self._last_known_screen = known
            self._last_mem_fp = fp
            hints = (
                mem.navigation_hints(
                    device.serial,
                    package,
                    max_suggest=mcfg.suggest_max,
                    max_research=mcfg.research_suggest_max,
                    include_navigation=mcfg.suggest,
                    half_life_days=mcfg.rank_half_life_days,
                )
                if mcfg.suggest or mcfg.auto_research
                else None
            )
            return known, hints
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("memory record_screen failed: %s", exc)
            return None, None

    def _join_memory_writers(self, *, timeout_s: float = 5.0) -> bool:
        """Wait for every queued async screen write within one bounded deadline.

        ``_mem_thread`` retained only the newest writer.  When observations arrived faster
        than their asynchronous map writes, an older queued writer could outlive it and stamp
        a package/context boundary *after* the next action had already been journaled.  Keep
        all outstanding writers ordered ahead of action capture and artifact materialisation.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._mem_threads_lock:
                pending = [
                    thread
                    for thread in self._mem_threads
                    if thread is not threading.current_thread() and thread.is_alive()
                ]
                latest = self._mem_thread
                if (
                    latest is not None
                    and latest is not threading.current_thread()
                    and latest.is_alive()
                    and latest not in pending
                ):
                    # Compatibility for integrations/tests that set the historical singular
                    # writer handle directly. Production writers are also kept in the list.
                    pending.append(latest)
            if not pending:
                with self._mem_threads_lock:
                    self._mem_threads = [
                        thread for thread in self._mem_threads if thread.is_alive()
                    ]
                return True
            for thread in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                thread.join(timeout=remaining)

    def _record_action_safe(self, step: RouteStep) -> None:
        if self._action_recording_suppression:
            return
        mem = self._memory
        if mem is None or self._device is None:
            return
        try:
            self._claim_memory_session()
            # Async screen recording and the action journal both update SessionState.  Let the
            # screen writer finish first so the action is stamped with its newly established
            # origin/context/segment and cannot be overwritten by a stale save.
            if not self._join_memory_writers(timeout_s=5.0):
                raise RuntimeError("memory screen provenance is still being finalized")
            with self._mem_lock:
                mem.observe_action(self._device.serial, step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("memory record_action failed: %s", exc)

    @contextlib.contextmanager
    def _without_action_recording(self) -> Iterator[None]:
        """Collapse an internally composed operation into its one public journal step."""
        self._action_recording_suppression += 1
        try:
            yield
        finally:
            self._action_recording_suppression -= 1

    def _mark_logcat(self, name: str) -> None:
        """Best-effort device-clock logcat mark (never fails the action that triggered it)."""
        try:
            if self._device is None:
                return
            from . import logcat as logcat_mod

            clock = logcat_mod.resolve_clock(self._device, self.config.cache.dir)
            logcat_mod.set_mark(self.config.cache.dir, self._device.serial, name, clock=clock)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("logcat mark %r failed: %s", name, exc)

    def _cached_package(self) -> str | None:
        """Package of the last analyze (call BEFORE the action invalidates the cache)."""
        cached = self._read_cache()
        return cached.screen.package if cached else None

    def _next_actions(self, obs: AnalyzeResult, *, limit: int = 12) -> list[dict[str, Any]] | None:
        """What can be done from here, decision-ready, with each control's learned cost attached.

        This exists to remove a *reasoning* step, not a call: the post-action screen already came
        back inline, but an agent still had to scan `observation.elements` to find which of 50
        nodes it could act on — and scanning was expensive enough that agents preferred
        `--no-observe` plus a filtered `analyze`, which is two calls to avoid one read.

        Ordered by how likely a caller wants it (labelled controls first, then the rest) and
        capped: a list of every tappable node is a dump, and a dump is what `analyze` is for.
        Cost rides on the entry it belongs to, so "tap 26 next, and it takes ~4.8s" is one read
        rather than a cross-reference against `slow_controls`.
        """
        rows: list[dict[str, Any]] = []
        timings = self._screen_timings_safe(obs.meta.known_screen if obs.meta else None)
        for e in obs.elements:
            if not (e.clickable or e.checkable or e.long_clickable or e.scrollable):
                continue
            label = (e.text or e.content_desc or "").strip()
            rid = _id_tail(e.resource_id)
            row: dict[str, Any] = {"id": e.id}
            if label:
                row["label"] = _label(label)
            if rid:
                row["rid"] = rid
            if e.checkable is not None:
                row["checked"] = e.checked
            if e.selected is not None:
                row["selected"] = e.selected
            known = timings.get(e.stable_key or rid or "")
            if known is not None:
                row["avg_ms"] = round(known.ema_ms)
                row["max_ms"] = round(known.max_ms)
                row["n"] = known.n
            rows.append(row)
        if not rows:
            return None
        # Labelled first: an unlabelled container is kept (it may be the only thing that acts —
        # see `keep_actionable`) but it is not what a caller reaches for first.
        rows.sort(key=lambda r: 0 if r.get("label") or r.get("rid") else 1)
        return rows[:limit]

    def _screen_timings_safe(self, screen: str | None) -> dict[str, Any]:
        """The timing map for *screen*, keyed by control — empty when unknown."""
        mem = self._memory
        if not screen or mem is None:
            return {}
        with contextlib.suppress(Exception):
            package = self._package_of_last_analyze()
            if package:
                app = mem.load(package)
                rec = app.screens.get(screen) if app else None
                if rec is not None:
                    return dict(rec.timings)
        return {}

    def _slow_controls_safe(self, screen: str | None) -> list[dict[str, Any]]:
        """Slow controls on *screen*, for `meta` — told on arrival, not discovered on timeout.

        This is the half the coarse profile could never provide: a per-kind average cannot say
        *which* control on the screen in front of you costs 6s. An agent that knows before acting
        can plan the wait (or pick `--until`) instead of reading a timeout as a broken product.
        """
        mem = self._memory
        if not screen or mem is None:
            return []
        with contextlib.suppress(Exception):
            package = self._package_of_last_analyze() or (
                self.device.current_app().get("package") if self._device is not None else None
            )
            if package:
                return mem.slow_controls(package, screen=screen)
        return []

    def _record_action_timing_safe(self, ms: float, *, outcome: str) -> None:
        """Never let bookkeeping break an action — the same contract as the observation itself."""
        site = getattr(self, "_last_action_site", None)
        mem = self._memory
        if not site or mem is None:
            return
        with contextlib.suppress(Exception):
            package = self._package_of_last_analyze()
            if package:
                mem.record_action_timing(
                    package, screen=site[0], control=site[1], ms=ms, outcome=outcome
                )

    def _action_site(self, element: Element | None) -> tuple[str, str] | None:
        """``(screen, control)`` for cost bookkeeping, or None when either is unknown.

        The control key prefers ``stable_key`` — the cross-frame fingerprint that exists exactly
        because element ids churn between analyses — and falls back to the resource-id tail. A
        node with neither is not keyed at all: a timing filed under a key that means something
        different next run is worse than no timing, because it would be spent as a deadline.
        """
        if element is None:
            return None
        cached = self._read_cache()
        screen = cached.meta.known_screen if cached and cached.meta else None
        control = element.stable_key or _id_tail(element.resource_id)
        if not (screen and control):
            return None
        return str(screen), str(control)

    def _learned_action_budget(self, default_total_ms: int) -> int | None:
        """The deadline this control has earned from history, or None to use the coarse profile.

        Built from ``max_ms`` rather than the average: a deadline set from the mean is by
        construction too short half the time, and the cost of being too short is a false
        "nothing changed" — which this suite has already mistaken for a product defect. Padded
        by 50% and floored at the caller's default so history can only ever *extend* the wait.
        """
        site = getattr(self, "_last_action_site", None)
        mem = self._memory
        if not site or mem is None or self._device is None:
            return None
        screen, control = site
        with contextlib.suppress(Exception):
            package = self._package_of_last_analyze()
            if not package:
                return None
            timing = mem.action_timing(package, screen=screen, control=control)
            if timing is None or timing.n < 1:
                return None
            return max(default_total_ms, int(timing.max_ms * 1.5))
        return None

    def _step(
        self,
        kind: str,
        element: Element | None = None,
        *,
        arg: str | None = None,
        submit: bool = False,
    ) -> RouteStep:
        """The structured record of one action (selector + redacted label, never a value)."""
        self._last_action_kind = kind
        # Where this action is happening, so its cost can be learned per (screen, control) and
        # not merely per action kind. Captured here because `_step` runs *before* the action,
        # while the cache still describes the screen we are acting from — afterwards it
        # describes the destination, which is not where the click was spent.
        self._last_action_site = self._action_site(element)
        selector = (
            recorded_selector(
                element,
                elements=(self._last_analyze_result.elements if self._last_analyze_result else ()),
            )
            if element
            else {
                "label": None,
                "content_desc": None,
                "resource_id": None,
                "by": None,
            }
        )
        return RouteStep(
            kind=kind,
            label=selector["label"],
            content_desc=selector["content_desc"],
            resource_id=selector["resource_id"],
            by=selector["by"],
            arg=arg,
            submit=submit,
            package=self._cached_package(),
        )

    def current_package(self) -> str | None:
        """Best-effort foreground package (for ``aua map`` without ``--app``)."""
        try:
            pkg = self.device.current_app().get("package")
        except Exception:  # pragma: no cover - device hiccup
            pkg = None
        if pkg:
            return pkg
        try:
            device, w, h = self._context()
            raw_tree = self.platform.dump_tree(device)
            return self.platform.normalize_tree(
                raw_tree,
                (w, h),
                ignored_app_ids=self.config.memory.ignore_packages,
            ).app_id
        except Exception:  # pragma: no cover
            return None

    def memory_update(self, screen_name: str | None = None) -> dict[str, Any]:
        """Force-record the current screen now (PRD §5 ``aua memory update``)."""
        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        device, w, h = self._context()
        elements, package, _xml_hash = self._capture_hierarchy(device, w, h)
        app = device.current_app()
        package = app.get("package") or package
        if not package:
            raise UsageError("could not determine the foreground package to record")
        # ``memory update --screen`` is the explicit correction path for a bad generated
        # name. Keep that correction in the same feature-flag context as normal analyze;
        # recording into ``default`` creates a disconnected duplicate and leaves every
        # route pointing at the bad name.
        self._sync_runtime_flag_context(device, package, mem)
        sess = mem.load_session(device.serial)
        same_context = sess.package in (None, package)
        context_id = sess.active_context_id if same_context else DEFAULT_CONTEXT_ID
        context_flags = sess.active_flags if same_context else {}
        outcome = mem.record_screen(
            package=package,
            elements=elements,
            activity=app.get("activity") or None,
            app_version=self._version_for(device, package),
            tier="hierarchy",
            name_hint=screen_name,
            screen_height=h,
            context_id=context_id,
            context_flags=context_flags,
            context_verified=sess.context_verified if same_context else False,
        )
        sess = mem.load_session(device.serial)
        sess.current_screen = outcome.name
        sess.package = package
        sess.pending = []
        mem.save_session(device.serial, sess)
        return {
            "ok": True,
            "action": "memory-update",
            "package": package,
            "screen": outcome.name,
            "known": outcome.was_known,
            "stale": outcome.stale,
            "created": outcome.created,
        }

    # ----------------------------------------------------------------- step executor

    def _source_for(self, steps: list[RouteStep], index: int, origin_package: str | None) -> str:
        """Analyze source between steps: ``auto`` when the NEXT step runs in a foreign
        (transit) package — its screen may be vision-tier — else the fast hierarchy path."""
        nxt = steps[index] if index < len(steps) else None
        if nxt is not None and nxt.package and nxt.package != origin_package:
            return "auto"
        return "hierarchy"

    def _analyze_route_step(
        self,
        steps: list[RouteStep],
        index: int,
        origin_package: str | None,
        *,
        hierarchy_ocr: bool,
    ) -> AnalyzeResult:
        """Observe between route steps without taxing ordinary native hops with OCR.

        Foreign/transit screens keep ``source=auto`` and its normal OCR behavior. Inside
        the origin app, ``goto`` can request hierarchy-only observations and explicitly
        retry OCR only when a remembered selector is absent.
        """
        source = self._source_for(steps, index, origin_package)
        with_ocr = None if hierarchy_ocr or source == "auto" else False
        return self.analyze(source=source, with_ocr=with_ocr)

    def _run_flow_assertion(self, step: RouteStep) -> ActionResult:
        """Evaluate the rich flow ``assert:`` step through the public expect primitive."""

        predicates = dict(step.assertion)
        first = bool(predicates.pop("first", False))
        count = predicates.pop("count", None)
        return self.expect(
            rid=step.resource_id,
            text=step.label,
            desc=step.content_desc,
            exists=bool(predicates.pop("exists", False)),
            absent=bool(predicates.pop("absent", False)),
            text_is=predicates.pop("text_is", None),
            text_contains=predicates.pop("text_contains", None),
            checked=predicates.pop("checked", None),
            enabled=predicates.pop("enabled", None),
            selected=predicates.pop("selected", None),
            focused=predicates.pop("focused", None),
            count=count,
            within=predicates.pop("within", None),
            same_parent_as=predicates.pop("same_parent_as", None),
            contains_all=predicates.pop("contains_all", None),
            index=step.index,
            first=first,
            timeout_ms=step.timeout_ms or 0,
            observe=False,
        )

    def _run_flow_order_assertion(self, step: RouteStep) -> tuple[bool, str]:
        """Assert explicit horizontal/vertical ordering without guessing grid semantics."""

        assertion = step.assertion
        axis = assertion.get("axis")
        selectors = assertion.get("selectors")
        if axis not in {"horizontal", "vertical", "reading"} or not isinstance(selectors, list):
            return False, "invalid assert_order payload"
        deadline = time.monotonic() + max(0, step.timeout_ms or 0) / 1000.0
        while True:
            raw_tree = self.platform.dump_tree(self.device)
            elements = self.platform.normalize_tree(
                raw_tree,
                self.device.window_size(),
                ignored_app_ids=self.config.memory.ignore_packages,
            ).elements
            order = evaluate_order(elements, axis=axis, selectors=selectors)
            if order.ok:
                return True, order.detail
            detail = order.detail
            if time.monotonic() >= deadline:
                return False, detail
            self._job_sleep(0.25)

    def _run_steps(
        self,
        steps: list[RouteStep],
        *,
        origin_package: str | None,
        allow_destructive: bool,
        allow_goto_steps: bool = False,
        scroll_fallback: bool = False,
        res: AnalyzeResult | None = None,
        executed: list[dict[str, Any]] | None = None,
        flow_depth: int = 0,
        hierarchy_ocr: bool = True,
        flow_dir: Path | None = None,
        allow_unsafe_route_effects: bool = True,
        flow_plan: _ResolvedFlowPlan | None = None,
        flow_artifacts: Any | None = None,
    ) -> tuple[StepFailure | None, AnalyzeResult]:
        """Execute *steps* with selector matching, settle waits, and re-perception.

        The single replay engine behind ``goto`` edge replay and ``flow run``. Between
        state-changing steps it settles (suppressed ``wait_stable``) and re-analyzes with
        a package-aware source (:meth:`_source_for`). Verification is lazy — a wrong
        screen surfaces as the next step's ``element_not_found`` — terminal verification
        (``known_screen`` / asserts) is the caller's job. Returns
        ``(failure | None, last analyze result)``.

        ``flow_dir`` is the directory of the flow file these steps came from, and is what makes
        a nested ``flow:`` reference resolvable by path — "next to me" has no meaning without
        it. It is passed down each nesting level, so a sub-flow's own references are relative
        to the sub-flow.
        """
        if res is None:
            res = self._analyze_route_step(steps, 0, origin_package, hierarchy_ocr=hierarchy_ocr)
        lexicon = self.config.memory.destructive_labels
        for i, s in enumerate(steps):
            step_started = time.perf_counter()
            step_extra: dict[str, Any] = {}
            if is_destructive_step(s, lexicon) and not allow_destructive:
                return StepFailure("destructive_step", i, s), res
            non_destructive_risks = [
                risk
                for risk in route_step_risks(
                    s,
                    origin_package=origin_package,
                    destructive_labels=lexicon,
                )
                if risk["code"] != "destructive"
            ]
            if non_destructive_risks and not allow_unsafe_route_effects:
                return StepFailure("unsafe_route_step", i, s), res
            kind = s.kind
            reanalyze = True  # most kinds change state → settle + re-perceive
            settle = True
            if kind in ("tap", "long-press", "clear", "input"):
                if kind == "input" and s.text is None:
                    # auto-recorded inputs never store the value — the caller supplies it
                    return StepFailure("input_required", i, s), res
                el = _match_step(res.elements, s)
                if el is None and not hierarchy_ocr:
                    source = self._source_for(steps, i, origin_package)
                    if source == "hierarchy":
                        # Known native routes stay hierarchy-fast. OCR is paid only when
                        # the remembered selector is not present in accessibility data.
                        retry = self.analyze(source="hierarchy", with_ocr=True)
                        retry_el = _match_step(retry.elements, s)
                        if retry_el is not None:
                            res, el = retry, retry_el
                selector_value = s.resource_id or s.content_desc or s.label
                if el is None and scroll_fallback and selector_value:
                    self.scroll_to(
                        selector_value,
                        observe=False,
                        by=s.by
                        or ("id" if s.resource_id else "desc" if s.content_desc else "text"),
                    )
                    res = self._analyze_route_step(
                        steps, i, origin_package, hierarchy_ocr=hierarchy_ocr
                    )
                    el = _match_step(res.elements, s)
                if el is None:
                    return StepFailure("element_not_found", i, s), res
                if kind == "tap":
                    self.tap(el.id, observe=False)
                elif kind == "long-press":
                    self.long_press(el.id, observe=False)
                elif kind == "clear":
                    self.clear(el.id, observe=False)
                else:
                    # A step that typed nothing must diverge, not pass quietly: a flow whose
                    # input never landed goes on to assert against a screen it never reached,
                    # and reports the app's fault instead of its own.
                    if not self.input_text(el.id, s.text or "", submit=s.submit, observe=False).ok:
                        return StepFailure("input_not_applied", i, s), res
            elif kind == "tap-point":
                point = _parse_point(s.arg)
                if point is None:
                    return StepFailure("unsupported_action", i, s), res
                self.tap_point(*point, observe=False)
            elif kind == "key":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                self.key(s.arg, observe=False)
            elif kind == "swipe":
                if s.arg not in ("up", "down", "left", "right"):
                    return StepFailure("unsupported_action", i, s), res
                self.swipe(s.arg, observe=False)
            elif kind == "scroll":
                # A recorded scroll used to be unsaveable *and* unreplayable: the engine
                # records kind="scroll", the flow schema had no such kind, and rendering it
                # raised KeyError('scroll') - surfacing as `internal_error: 'scroll'` out of
                # `flow save`. Any journey containing a scroll therefore could not be
                # captured at all, which is most journeys worth capturing.
                if s.arg not in ("up", "down", "left", "right"):
                    return StepFailure("unsupported_action", i, s), res
                self.scroll(s.arg, observe=False)
            elif kind == "scroll-to":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if not self.scroll_to(
                    s.arg,
                    observe=False,
                    by=s.by or "text",
                    # Default matches the CLI's `--direction up`: keep looking further down.
                    direction=s.direction or "up",
                ).ok:
                    return StepFailure("element_not_found", i, s), res
            elif kind == "launch-app":
                pkg = s.arg or origin_package  # bare launch_app → the flow's own app
                if not pkg:
                    return StepFailure("unsupported_action", i, s), res
                # `activity:` pins the entry component on multi-launcher builds.
                self.app("launch", package=pkg, activity=s.activity)
            elif kind == "stop-app":
                pkg = s.arg or origin_package
                if not pkg:
                    return StepFailure("unsupported_action", i, s), res
                self.app("stop", package=pkg)
                reanalyze = False  # app is gone; nothing to perceive until relaunch
            elif kind == "open-link":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                self.open_link(s.arg, observe=False)
            elif kind == "wait-for":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if not self.wait(
                    for_=s.arg, timeout_ms=s.timeout_ms or 10000, by=s.by or "text"
                ).ok:
                    return StepFailure("wait_timeout", i, s), res
                settle = False  # the wait already absorbed the transition
            elif kind == "wait-stable":
                try:
                    self.wait_stable(settle_ms=600, timeout_ms=s.timeout_ms or 15000)
                except StabilityTimeout:
                    return StepFailure("wait_timeout", i, s), res
                settle = False
            elif kind == "assert-visible":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if not self.has(s.arg, timeout_ms=s.timeout_ms or 0, by=s.by or "text").found:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False  # pure check, screen unchanged
            elif kind == "assert-not-visible":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if self.has(s.arg, timeout_ms=s.timeout_ms or 0, by=s.by or "text").found:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False
            elif kind == "assert":
                assertion = self._run_flow_assertion(s)
                if not assertion.ok:
                    return StepFailure("assert_failed", i, s, assertion.detail), res
                step_extra["assertion"] = assertion.detail
                reanalyze = False
            elif kind == "assert-order":
                ok, detail = self._run_flow_order_assertion(s)
                if not ok:
                    return StepFailure("assert_failed", i, s, detail), res
                step_extra["assertion"] = detail
                reanalyze = False
            elif kind == "screenshot":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                try:
                    if flow_artifacts is not None:
                        screenshot_path = flow_artifacts.capture_checkpoint(s.arg)
                    else:
                        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", s.arg).strip("-._")
                        path = self._default_annotate_path(
                            self.device.serial,
                            suffix=f"flow-{safe_name or 'checkpoint'}",
                            timestamped=True,
                        )
                        screenshot_path = self.screenshot(path).detail
                except Exception as exc:  # noqa: BLE001 - preserve a resumable flow failure
                    return StepFailure(
                        "screenshot_failed", i, s, f"{type(exc).__name__}: {exc}"
                    ), res
                if screenshot_path:
                    step_extra["screenshot"] = screenshot_path
                reanalyze = False
                settle = False
            elif kind == "hide-keyboard":
                self.hide_keyboard(observe=False)
            elif kind == "paste":
                self.paste(observe=False)
            elif kind == "dev-profile":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                self.dev_profile(s.arg)
                reanalyze = False
            elif kind == "a11y-scroll":
                el = _match_step(res.elements, s)
                if el is None:
                    return StepFailure("element_not_found", i, s), res
                direction = (s.arg or "forward").lower()
                self.a11y_scroll(el.id, direction=direction, observe=False)
            elif kind == "flags-apply":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                flags_snapshot = flow_plan.flags.get(id(s)) if flow_plan is not None else None
                if flow_plan is not None and flags_snapshot is None:
                    return StepFailure("resource_snapshot_missing", i, s), res
                if not self.flags_apply(
                    s.arg,
                    observe=False,
                    _snapshot=flags_snapshot,
                ).get("ok", True):
                    return StepFailure("assert_failed", i, s), res
            elif kind == "network-offline":
                if not self.network_offline(verify=True, timeout_ms=s.timeout_ms or 10_000).ok:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False
            elif kind == "network-restore":
                if not self.network_restore(timeout_ms=s.timeout_ms or 15_000).ok:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False
            elif kind == "network-profile":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                if not self.network_profile_apply(s.arg, timeout_ms=s.timeout_ms or 20_000).ok:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False
            elif kind == "network-profile-restore":
                if not self.network_profile_restore(timeout_ms=s.timeout_ms or 20_000).ok:
                    return StepFailure("assert_failed", i, s), res
                reanalyze = False
            elif kind == "proxy-start":
                self.proxy_start()
                reanalyze = False
            elif kind == "proxy-stop":
                self.proxy_stop()
                reanalyze = False
            elif kind == "mock-replay":
                if not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                cassette_snapshot = (
                    flow_plan.cassettes.get(id(s)) if flow_plan is not None else None
                )
                if flow_plan is not None and cassette_snapshot is None:
                    return StepFailure("resource_snapshot_missing", i, s), res
                self.mock_replay(s.arg, _snapshot=cassette_snapshot)
                reanalyze = False
            elif kind == "repeat":
                times = max(1, s.repeat or 1)
                for iteration in range(times):
                    sub_executed: list[dict[str, Any]] = []
                    subfail, res = self._run_steps(
                        s.substeps,
                        origin_package=origin_package,
                        allow_destructive=allow_destructive,
                        allow_goto_steps=allow_goto_steps,
                        scroll_fallback=scroll_fallback,
                        res=res,
                        executed=sub_executed,
                        flow_depth=flow_depth,
                        hierarchy_ocr=hierarchy_ocr,
                        # substeps came from the same file, so "next to me" is unchanged
                        flow_dir=flow_dir,
                        allow_unsafe_route_effects=allow_unsafe_route_effects,
                        flow_plan=flow_plan,
                        flow_artifacts=flow_artifacts,
                    )
                    if executed is not None:
                        for row in sub_executed:
                            child_path = row.get("path")
                            if not isinstance(child_path, list):
                                child_path = [row.get("index")]
                            executed.append(
                                {
                                    **row,
                                    "index": i,
                                    "path": [i, iteration, *child_path],
                                }
                            )
                    if subfail is not None:
                        return StepFailure(subfail.code, i, s, subfail.detail), res
                reanalyze = False
                settle = False
            elif kind == "retry":
                attempts = max(1, s.max_retries or 3)
                subfail = StepFailure("assert_failed", i, s)
                for attempt in range(attempts):
                    sub_executed = []
                    subfail, res = self._run_steps(
                        s.substeps,
                        origin_package=origin_package,
                        allow_destructive=allow_destructive,
                        allow_goto_steps=allow_goto_steps,
                        scroll_fallback=scroll_fallback,
                        res=res,
                        executed=sub_executed,
                        flow_depth=flow_depth,
                        hierarchy_ocr=hierarchy_ocr,
                        # substeps came from the same file, so "next to me" is unchanged
                        flow_dir=flow_dir,
                        allow_unsafe_route_effects=allow_unsafe_route_effects,
                        flow_plan=flow_plan,
                        flow_artifacts=flow_artifacts,
                    )
                    if executed is not None:
                        for row in sub_executed:
                            child_path = row.get("path")
                            if not isinstance(child_path, list):
                                child_path = [row.get("index")]
                            executed.append(
                                {
                                    **row,
                                    "index": i,
                                    "path": [i, attempt, *child_path],
                                }
                            )
                    if subfail is None:
                        break
                if subfail is not None:
                    return StepFailure(subfail.code, i, s, subfail.detail), res
                reanalyze = False
                settle = False
            elif kind == "goto":
                if not allow_goto_steps or not s.arg:
                    return StepFailure("unsupported_action", i, s), res
                out = self.goto(
                    s.arg,
                    allow_destructive=allow_destructive,
                    # A goto nested in an explicitly authored flow is already deliberate
                    # execution; keep flow semantics while standalone learned goto stays safe.
                    allow_unsafe=True,
                )
                if not out.get("ok"):
                    return StepFailure(str(out.get("code") or "route_unknown"), i, s), res
                settle = False  # goto verified arrival; just refresh our view
            elif kind == "flow":
                # Run a saved flow inline (Maestro's runFlow) — reuse shared recipes.
                if not allow_goto_steps or not s.arg or flow_depth >= _MAX_FLOW_DEPTH:
                    return StepFailure("unsupported_action", i, s), res
                try:
                    key = self._flow_ref_key(s.arg, flow_dir)
                    node = flow_plan.flow_graph.get(key) if flow_plan is not None else None
                    if node is None:
                        node = self._resolve_nested_flow_node(s.arg, flow_dir)
                except UsageError:
                    return StepFailure("route_unknown", i, s), res
                sub, sub_dir, _source_id, sub_steps = node
                nested_executed: list[dict[str, Any]] = []
                subfail, res = self._execute_flow_steps(
                    sub,
                    sub_steps,
                    res=res,
                    allow_destructive=allow_destructive,
                    scroll_fallback=scroll_fallback,
                    executed=nested_executed,
                    flow_depth=flow_depth + 1,
                    hierarchy_ocr=hierarchy_ocr,
                    flow_dir=sub_dir,
                    allow_unsafe_route_effects=allow_unsafe_route_effects,
                    flow_plan=flow_plan,
                    flow_artifacts=flow_artifacts,
                )
                if executed is not None:
                    for row in nested_executed:
                        child_path = row.get("path")
                        if not isinstance(child_path, list):
                            child_path = [row.get("index")]
                        prior_flow_path = row.get("flow_path")
                        if not isinstance(prior_flow_path, list):
                            prior_flow_path = []
                        executed.append(
                            {
                                **row,
                                "index": i,
                                "path": [i, *child_path],
                                "flow_path": [s.arg, *prior_flow_path],
                            }
                        )
                if subfail is not None:
                    return StepFailure(subfail.code, i, s, subfail.detail), res
                arrival_verified, arrival_code, res, _arrival_evidence = (
                    self._flow_arrival_evidence(sub, res)
                )
                if arrival_verified is False:
                    return StepFailure(arrival_code or "arrival_unverified", i, s), res
                settle = False  # the sub-flow already settled
            else:
                return StepFailure("unsupported_action", i, s), res

            if reanalyze:
                if settle:
                    nxt = steps[i + 1] if i + 1 < len(steps) else None
                    if not self._settle_for_next_step(nxt):
                        with contextlib.suppress(StabilityTimeout):
                            self.wait_stable(settle_ms=500, timeout_ms=8000)
                res = self._analyze_route_step(
                    steps, i + 1, origin_package, hierarchy_ocr=hierarchy_ocr
                )
            if executed is not None:
                step_row: dict[str, Any] = {
                    "index": i,
                    "step": step_display(s),
                    "duration_ms": max(0, int((time.perf_counter() - step_started) * 1000)),
                    **step_extra,
                }
                if flow_artifacts is not None:
                    flow_artifacts.record_step(step_row, kind=kind, observation=res)
                executed.append(step_row)
        return None, res

    @staticmethod
    def _flow_ref_key(ref: str, flow_dir: Path | None) -> tuple[str | None, str]:
        return (str(flow_dir.resolve()) if flow_dir is not None else None, ref)

    def _resolve_nested_flow_node(self, ref: str, flow_dir: Path | None) -> _ResolvedFlowNode:
        """Load and resolve one nested flow as an immutable execution snapshot.

        Nested flows resolved by *name* from AUA's own memory directory only, so a promoted
        flow that referenced a sibling broke for anyone whose memory directory did not happen
        to contain a flow of that name. Factoring shared preconditions into ``flows/common/``
        was therefore impossible: nine shared routes had to be inlined into ~35 derived flows,
        so a fix to one does not propagate.

        A path-looking reference that resolves nowhere is **refused**, not retried as a name.
        Falling back would look up a sanitised spelling of the path in the memory directory
        (``common/auth.yaml`` → ``common_auth.yaml``), where a chance match would silently run
        a different journey. Failing to find the file the author named is recoverable; running
        somebody else's flow instead is not. The searched candidates are logged, because the
        executor's ``StepFailure`` carries a code and a step but no message.
        """
        from .flows import (
            FlowStore,
            anchor_paths,
            looks_like_path,
            nested_flow_candidates,
            parse_flow_yaml,
            resolve_params,
        )

        store = FlowStore(self.config.memory)
        if not looks_like_path(ref):
            path = store.path(ref).resolve()
            flow = store.load(ref)
            directory = path.parent
            steps = anchor_paths(resolve_params(flow, {}), directory)
            return _ResolvedFlowNode(flow, directory, str(path), steps)
        candidates = nested_flow_candidates(ref, flow_dir, store.flows_dir())
        for cand in candidates:
            if cand.is_file():
                path = cand.resolve()
                flow = parse_flow_yaml(path.read_text(encoding="utf-8"), name=path.stem)
                directory = path.parent
                steps = anchor_paths(resolve_params(flow, {}), directory)
                return _ResolvedFlowNode(flow, directory, str(path), steps)
        logger.warning(
            "nested flow %r not found; looked in: %s",
            ref,
            ", ".join(str(c) for c in candidates) or "(nowhere — no referring directory)",
        )
        raise UsageError(
            f"no flow file for nested reference {ref!r}",
            hint="Tried: " + ", ".join(str(c) for c in candidates),
        )

    def _resolve_nested_flow(self, ref: str, flow_dir: Path | None) -> tuple[Any, Path | None]:
        """Compatibility wrapper returning the parsed flow and its source directory."""
        node = self._resolve_nested_flow_node(ref, flow_dir)
        return node.flow, node.directory

    @staticmethod
    def _flow_graph_identity(flow: Any, flow_dir: Path | None) -> str:
        directory = str(flow_dir.resolve()) if flow_dir is not None else "<memory>"
        return f"{directory}::{flow.name}"

    def _preflight_nested_flow_graph(
        self,
        steps: list[RouteStep],
        *,
        flow_dir: Path | None,
        flow_app: str | None = None,
        context_id: str | None = None,
        flow_depth: int = 0,
        ancestors: tuple[str, ...] = (),
        plan: _ResolvedFlowPlan | None = None,
        goto_allowed: bool = True,
    ) -> _ResolvedFlowPlan:
        """Resolve and validate every composed flow before the first device mutation.

        Nested flows used to be loaded only when execution reached their step. A missing file,
        unbound parameter, invalid arrival predicate, or cycle could therefore be discovered
        after earlier parent steps had already changed the device. This filesystem-only walk
        proves the whole graph is runnable first; runtime app/context checks still happen on
        the fresh observation at the point each child begins.
        """
        if plan is None:
            plan = _ResolvedFlowPlan({}, {}, {})

        for index, step in enumerate(steps):
            if step.substeps:
                self._preflight_nested_flow_graph(
                    step.substeps,
                    flow_dir=flow_dir,
                    flow_app=flow_app,
                    context_id=context_id,
                    flow_depth=flow_depth,
                    ancestors=ancestors,
                    plan=plan,
                    goto_allowed=False,
                )
            if step.kind == "flags-apply":
                if not step.arg:
                    raise UsageError("flags_apply step needs a flags file")
                from .flags import load_flags_file

                app, pairs = load_flags_file(step.arg)
                plan.flags[id(step)] = _ResolvedFlagsResource(
                    str(Path(step.arg).expanduser().resolve()),
                    app,
                    deepcopy(pairs),
                )
            elif step.kind == "mock-replay":
                if not step.arg:
                    raise UsageError("mock_replay step needs a cassette name or path")
                from . import proxy_mock as pm

                cassette = pm.cassette_dir(self.config.memory.dir) / f"{step.arg}.yaml"
                alternate = Path(step.arg).expanduser()
                selected = alternate if alternate.is_file() else cassette
                plan.cassettes[id(step)] = _ResolvedCassetteResource(
                    step.arg,
                    selected.resolve(),
                    deepcopy(pm.load_cassette(selected)),
                )
            elif step.kind == "goto":
                if not step.arg:
                    raise UsageError("goto step needs a mapped screen goal")
                mem = self._memory
                mapped_app = mem.load(flow_app) if mem is not None and flow_app else None
                if (
                    mapped_app is None
                    or resolve_goal(
                        mapped_app,
                        step.arg,
                        context_id=context_id,
                        destructive_labels=self.config.memory.destructive_labels,
                    )
                    is None
                ):
                    raise UsageError(
                        f"goto target {step.arg!r} is not mapped for {flow_app or 'this flow'}",
                        hint="record/map the destination before composing it into a flow",
                    )
                if not goto_allowed or flow_depth > 0 or index != 0:
                    raise UsageError(
                        "goto inside a flow must be its first top-level step",
                        hint=(
                            "A later goto's route origin depends on earlier mutations and cannot "
                            "be authorized atomically; capture the exact route steps instead."
                        ),
                    )
            if step.kind != "flow":
                continue
            if not step.arg:
                raise UsageError("nested flow step needs a flow name or path")
            if flow_depth >= _MAX_FLOW_DEPTH:
                raise UsageError(
                    f"nested flow depth exceeds {_MAX_FLOW_DEPTH}",
                    hint="remove the recursive reference or flatten the composed flow",
                )
            key = self._flow_ref_key(step.arg, flow_dir)
            node = plan.flow_graph.get(key)
            if node is None:
                node = self._resolve_nested_flow_node(step.arg, flow_dir)
                plan.flow_graph[key] = node
            child, child_dir, identity, child_steps = node
            child_app = child.app or flow_app
            child_context = child.context_id or context_id
            if child.app is not None and child.app != flow_app:
                raise UsageError(
                    f"nested flow {step.arg!r} belongs to {child.app}, not parent app {flow_app}",
                    hint=(
                        "Keep cross-app transit as explicit package-stamped steps in one flow "
                        "so its entry contract can be verified before the first mutation."
                    ),
                )
            if child.context_id is not None and child.context_id != context_id:
                raise UsageError(
                    f"nested flow {step.arg!r} uses context {child.context_id}, not {context_id}",
                    hint="compose only flows recorded for the same app context",
                )
            self._validate_flow_arrival_screen(child, child_app, child_context)
            if identity in ancestors:
                chain = " -> ".join((*ancestors, identity))
                raise UsageError(
                    f"nested flow cycle detected: {chain}",
                    hint="remove one of the recursive flow references",
                )
            if child.arrival:
                _parse_await_terms(child.arrival, require_positive=True)
            self._preflight_nested_flow_graph(
                child_steps,
                flow_dir=child_dir,
                flow_app=child_app,
                context_id=child_context,
                flow_depth=flow_depth + 1,
                ancestors=(*ancestors, identity),
                plan=plan,
                goto_allowed=False,
            )
        return plan

    def _resolved_flow_disclosure(
        self,
        steps: list[RouteStep],
        *,
        flow_dir: Path | None,
        flow_app: str | None,
        plan: _ResolvedFlowPlan,
        path_prefix: str = "steps",
        index_offset: int = 0,
    ) -> dict[str, Any]:
        """Describe the exact recursively resolved graph without exposing resource payloads.

        A ``flow`` step is itself conservatively non-authorizable, but that generic fact is not
        enough for review: the referenced child may change settings, data, or the environment.
        The preflight plan already pins every child file, so disclosure walks those same nodes
        instead of reopening YAML. Parsed flag values and cassette bodies deliberately remain
        private to the execution plan and never enter this result (or rendered flow YAML).
        """
        lexicon = self.config.memory.destructive_labels
        all_risks: list[dict[str, str]] = []
        graph: list[dict[str, Any]] = []

        def add_risks(items: Sequence[dict[str, str]]) -> None:
            for item in items:
                if item not in all_risks:
                    all_risks.append(item)

        def walk(
            current: list[RouteStep],
            *,
            directory: Path | None,
            origin_package: str | None,
            prefix: str,
            offset: int = 0,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for index, step in enumerate(current):
                absolute_index = index + offset
                path = f"{prefix}[{absolute_index}]"
                risks = route_step_risks(
                    step,
                    origin_package=origin_package,
                    destructive_labels=lexicon,
                    path=path,
                )
                add_risks(risks)
                row: dict[str, Any] = {
                    "index": absolute_index,
                    "path": path,
                    "step": step_display(step),
                    "kind": step.kind,
                    "risks": risks,
                }
                if step.substeps:
                    row["substeps"] = walk(
                        step.substeps,
                        directory=directory,
                        origin_package=origin_package,
                        prefix=f"{path}.substeps",
                    )
                if step.kind == "flow" and step.arg:
                    node = plan.flow_graph.get(self._flow_ref_key(step.arg, directory))
                    if node is not None:
                        child, child_dir, source_id, child_steps = node
                        edge = {
                            "path": path,
                            "reference": step.arg,
                            "name": child.name,
                            "source": source_id,
                            "app": child.app or origin_package,
                            "context_id": child.context_id,
                        }
                        graph.append(edge)
                        child_rows = walk(
                            child_steps,
                            directory=child_dir,
                            origin_package=child.app or origin_package,
                            prefix=f"{path}.resolved_flow.steps",
                        )
                        row["resolved_flow"] = {
                            **edge,
                            "arrival": child.arrival,
                            "arrival_screen": child.arrival_screen,
                            "arrival_status": child.arrival_status or "unverified",
                            "steps": child_rows,
                        }
                nested_rows = [
                    *row.get("substeps", []),
                    *((row.get("resolved_flow") or {}).get("steps") or []),
                ]
                row["destructive"] = any(
                    risk.get("code") == "destructive" for risk in risks
                ) or any(bool(nested.get("destructive")) for nested in nested_rows)
                row["effects"] = sorted(
                    {str(risk.get("code")) for risk in risks if risk.get("code") is not None}
                    | {
                        str(effect)
                        for nested in nested_rows
                        for effect in nested.get("effects", [])
                    }
                )
                rows.append(row)
            return rows

        resolved_steps = walk(
            steps,
            directory=flow_dir,
            origin_package=flow_app,
            prefix=path_prefix,
            offset=index_offset,
        )
        return {
            "steps": resolved_steps,
            "risks": all_risks,
            "effects": sorted(
                {str(risk.get("code")) for risk in all_risks if risk.get("code") is not None}
            ),
            "flow_graph": graph,
        }

    def _validate_flow_arrival_screen(
        self,
        flow: Any,
        package: str | None,
        context_id: str | None,
    ) -> None:
        """Prove a declared mapped arrival exists, is fresh, and fits the known context."""
        if not flow.arrival_screen:
            return
        mem = self._memory
        app = mem.load(package) if mem is not None and package else None
        record = app.screens.get(flow.arrival_screen) if app is not None else None
        context_ok = bool(
            record is not None
            and (context_id is None or record.context_id in {context_id, LEGACY_CONTEXT_ID})
        )
        if record is None or record.stale or not context_ok:
            raise UsageError(
                f"flow '{flow.name}' claims unavailable mapped arrival {flow.arrival_screen!r}",
                hint="record a fresh same-context destination or use a positive arrival predicate",
            )

    @staticmethod
    def _flow_leading_launch_establishes_origin(flow: Any, steps: list[RouteStep]) -> bool:
        """Whether the first step explicitly brings the flow's owning app to foreground."""
        if not flow.app or not steps or steps[0].kind != "launch-app":
            return False
        return (steps[0].arg or flow.app) == flow.app

    def _flow_runtime_state(
        self,
        flow: Any,
        observation: AnalyzeResult,
        *,
        refresh_context: bool,
        transit_step: RouteStep | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Return active context and an entry-contract mismatch, if any.

        The observation is the foreground truth.  Session state contributes a feature-flag
        context only when it belongs to that same package; a cursor from another app must not
        authorize a replay.  Explicit flow execution forces one runtime flag read so a recent
        out-of-band flag change cannot be hidden by the normal short refresh cache.  A resumed
        flow may continue inside configured transit only when the origin-owned session and the
        next step's explicit package both corroborate that exact foreground.
        """
        observed_package = observation.screen.package or self.current_package()
        mem = self._memory
        active_context: str | None = DEFAULT_CONTEXT_ID if mem is None else None
        owner_mismatch = bool(flow.app and observed_package != flow.app)
        transit_resume = False
        if mem is not None and self._device is not None and observed_package:
            # Serialize with async screen recording: otherwise an older background session write
            # can land after this forced flag read and put the stale context back.
            with self._mem_lock:
                session = mem.load_session(self._device.serial)
                transit_resume = bool(
                    owner_mismatch
                    and transit_step is not None
                    and transit_step.package == observed_package
                    and matches_any(observed_package, self.config.memory.transit_packages)
                    and session.package == flow.app
                )
                if refresh_context and not owner_mismatch:
                    self._sync_runtime_flag_context(
                        self._device,
                        observed_package,
                        mem,
                        force=True,
                    )
                    session = mem.load_session(self._device.serial)
                if session.package == observed_package or transit_resume:
                    active_context = session.active_context_id

        if owner_mismatch and not transit_resume:
            return None, {
                "code": "flow_app_mismatch",
                "expected_package": flow.app,
                "observed_package": observed_package,
                "reason": "the flow's owning app is not in the foreground",
            }

        if flow.context_id and active_context != flow.context_id:
            return active_context, {
                "code": "flow_context_mismatch",
                "expected_context_id": flow.context_id,
                "active_context_id": active_context,
                "observed_package": observed_package,
                "reason": "the active app context does not match the recorded flow context",
            }
        return active_context, None

    def _execute_flow_steps(
        self,
        flow: Any,
        steps: list[RouteStep],
        *,
        res: AnalyzeResult,
        allow_destructive: bool,
        scroll_fallback: bool,
        flow_depth: int,
        hierarchy_ocr: bool,
        flow_dir: Path | None,
        allow_unsafe_route_effects: bool,
        executed: list[dict[str, Any]],
        allow_transit_resume: bool = False,
        flow_plan: _ResolvedFlowPlan | None = None,
        flow_artifacts: Any | None = None,
    ) -> tuple[StepFailure | None, AnalyzeResult]:
        """Execute a resolved flow after enforcing its package/context entry contract."""

        def run_chunk(
            chunk: list[RouteStep], start: int, current: AnalyzeResult
        ) -> tuple[StepFailure | None, AnalyzeResult]:
            chunk_executed: list[dict[str, Any]] = []
            failure, latest = self._run_steps(
                chunk,
                origin_package=flow.app,
                allow_destructive=allow_destructive,
                allow_goto_steps=True,
                scroll_fallback=scroll_fallback,
                res=current,
                executed=chunk_executed,
                flow_depth=flow_depth,
                hierarchy_ocr=hierarchy_ocr,
                flow_dir=flow_dir,
                allow_unsafe_route_effects=allow_unsafe_route_effects,
                flow_plan=flow_plan,
                flow_artifacts=flow_artifacts,
            )
            for row in chunk_executed:
                row["index"] += start
            executed.extend(chunk_executed)
            if failure is not None:
                failure = StepFailure(
                    failure.code, failure.at + start, failure.step, failure.detail
                )
            return failure, latest

        _active_context, mismatch = self._flow_runtime_state(
            flow,
            res,
            refresh_context=True,
            transit_step=steps[0] if allow_transit_resume and steps else None,
        )
        start = 0
        if mismatch is not None:
            if not (
                mismatch["code"] == "flow_app_mismatch"
                and self._flow_leading_launch_establishes_origin(flow, steps)
            ):
                return StepFailure(mismatch["code"], 0, steps[0]), res

            # A wrong foreground is allowed only for this explicit establishing launch.  Verify
            # its observed result (including flags) before a second flow action is authorized.
            failure, res = run_chunk(steps[:1], 0, res)
            if failure is not None:
                return failure, res
            _active_context, mismatch = self._flow_runtime_state(flow, res, refresh_context=True)
            if mismatch is not None:
                return StepFailure(mismatch["code"], 0, steps[0]), res
            start = 1

        if start == len(steps):
            return None, res
        return run_chunk(steps[start:], start, res)

    def _flow_arrival_evidence(
        self,
        flow: Any,
        res: AnalyzeResult,
    ) -> tuple[bool | None, str | None, AnalyzeResult, dict[str, Any]]:
        """Verify every declared arrival condition against one terminal observation.

        ``None`` means the flow declared no arrival proof.  That remains executable when the
        caller names the flow directly, but it is never presented as verified arrival.
        """
        declared = bool(flow.arrival_screen or flow.arrival)
        predicate_ok = True
        predicate_code: str | None = None
        evidence: dict[str, Any] = {}
        terminal = res
        terminal_is_fresh = False
        if flow.arrival:
            arrival = self.await_predicate(
                flow.arrival,
                timeout_ms=30_000,
                poll_ms=300,
                observe=True,
            )
            evidence["arrival"] = arrival.model_dump(mode="json")
            predicate_ok = arrival.ok
            if not arrival.ok:
                predicate_code = f"arrival_{arrival.await_outcome or 'unverified'}"
            # When a legacy predicate and mapped screen are both present, this exact folded
            # observation is the screen proof too.  Checking the earlier last-step frame would
            # combine two different moments and could accept a transient destination.
            if arrival.observation is not None:
                terminal = arrival.observation
                terminal_is_fresh = True

        screen_ok = True
        if flow.arrival_screen:
            if not terminal_is_fresh:
                # Some valid flow steps intentionally do not re-perceive (notably stop_app,
                # because the app has just disappeared).  Their returned ``res`` describes the
                # screen from before the step and must never satisfy mapped arrival proof.  A
                # no-cache hierarchy read makes the proof about the device now; predicate-based
                # arrivals already supplied their own terminal observation above.
                terminal = self.analyze(
                    source="hierarchy",
                    with_ocr=False,
                    no_cache=True,
                )
            expected = flow.arrival_screen
            recognized = terminal.meta.known_screen
            _active_context, runtime_mismatch = self._flow_runtime_state(
                flow,
                terminal,
                refresh_context=True,
            )
            record_ok = runtime_mismatch is None
            mem = self._memory
            package = terminal.screen.package
            if record_ok and mem is not None and package:
                from .memory import LEGACY_CONTEXT_ID

                app = mem.load(package)
                record = app.screens.get(expected) if app is not None else None
                session = mem.load_session(self.device.serial)
                allowed_contexts = {session.active_context_id, LEGACY_CONTEXT_ID}
                record_ok = bool(
                    record is not None
                    and not record.stale
                    and record.context_id in allowed_contexts
                    and session.package == package
                )
            elif record_ok:
                # ``known_screen`` is map-derived. With no usable map/session there is no way
                # to prove that a supplied or cached name is fresh for this app context.
                record_ok = False
            screen_ok = recognized == expected and record_ok
            evidence["arrival_screen"] = {
                "expected": expected,
                "recognized": recognized,
                "verified": screen_ok,
            }
        else:
            evidence["arrival_screen"] = {
                "expected": None,
                "recognized": terminal.meta.known_screen,
                "verified": False,
                "status": flow.arrival_status or "unverified",
            }

        verified = declared and predicate_ok and screen_ok
        evidence["arrival_verified"] = bool(verified)
        evidence["arrival_status"] = "verified" if verified else "unverified"
        code = None
        if declared and not screen_ok:
            code = "arrival_screen_unverified"
        elif declared and not predicate_ok:
            code = predicate_code
        return (bool(verified) if declared else None), code, terminal, evidence

    def _settle_for_next_step(self, nxt: RouteStep | None) -> bool:
        """Synchronize on the next step's known selector instead of a full pixel settle.

        Returns True when the next target already appeared (caller skips ``wait_stable``).
        Falls back to False for swipes/keys/unknown labels so the conservative settle runs.
        """
        if nxt is None:
            return False
        timeout_ms = min(int(nxt.timeout_ms or 3000), 4000)
        if nxt.kind in ("wait-for", "assert-visible") and nxt.arg:
            return self.has(nxt.arg, timeout_ms=timeout_ms, by=nxt.by or "text").found
        if nxt.kind in ("tap", "long-press", "input", "clear", "a11y-scroll"):
            if nxt.resource_id:
                return self.has(nxt.resource_id, timeout_ms=timeout_ms, by="id").found
            if nxt.content_desc:
                return self.has(nxt.content_desc, timeout_ms=timeout_ms, by="desc").found
            label = nxt.label
            if label and label not in ("<filled>", "<redacted>"):
                return self.has(label, timeout_ms=timeout_ms, by="text").found
        return False

    def _mid_edge_path(
        self,
        app: AppMap,
        target: str,
        elements: list[Element],
        *,
        context_id: str | None = None,
    ) -> tuple[list[RouteEdge], int] | None:
        """Find a multi-step edge to *target* whose steps already match the current UI.

        Used by ``--from-here`` when recognition has already named a mid-journey screen
        (so shortest-path from that screen is empty) but a remembered edge still has
        remaining steps visible — e.g. edge home→images ``[Apps, Images]`` while the
        map now says ``apps``.
        """
        from .memory import DEFAULT_CONTEXT_ID, LEGACY_CONTEXT_ID

        best: tuple[RouteEdge, int, int] | None = None  # edge, resume_from, remaining
        for edge in app.routes:
            if edge.to_screen != target:
                continue
            if context_id and edge.context_id not in (
                context_id,
                DEFAULT_CONTEXT_ID,
                LEGACY_CONTEXT_ID,
            ):
                continue
            steps = edge.steps or _parse_legacy_steps(edge.action)
            if not steps:
                continue
            matches = [j for j, s in enumerate(steps) if _match_step(elements, s)]
            if not matches:
                continue
            resume = matches[-1]
            remaining = len(steps) - resume
            if best is None or remaining < best[2] or (remaining == best[2] and resume > best[1]):
                best = (edge, resume, remaining)
        if best is None:
            return None
        return [best[0]], best[1]

    # ----------------------------------------------------------------- planner (§7.3)

    def _planner_view(self, res: AnalyzeResult) -> tuple[list[dict[str, Any]], ScreenImage | None]:
        """Token-light element list for the planner (+ a screenshot only if weakly labelled)."""
        elements = [
            {
                "id": e.id,
                "label": e.text or e.content_desc,
                "clickable": e.clickable,
                "input": "edittext" in (e.type or "").lower(),
            }
            for e in res.elements
        ]
        labeled = sum(1 for e in res.elements if e.text or e.content_desc)
        img: ScreenImage | None = None
        if res.elements and (labeled < 3 or labeled / len(res.elements) < 0.3):
            with contextlib.suppress(Exception):  # image is a bonus; text-only still works
                img = self.device.screenshot()
        return elements, img

    def _drive_with_planner(
        self,
        objective: str,
        *,
        res: AnalyzeResult,
        max_steps: int,
        allow_destructive: bool,
        until: str | None = None,
    ) -> tuple[bool, AnalyzeResult]:
        """Let the opt-in planner choose actions toward *objective* until done/until/cap.

        Bounded and safe: the planner may only target an id from the list we hand it
        (validated here), its taps pass the destructive guard, and it runs at most
        *max_steps* times. Returns ``(reached, last analyze result)``. Never the happy
        path — callers gate on ``factory.is_enabled("planner")`` + an explicit opt-in.
        """
        if not self.factory.is_enabled("planner"):
            return False, res
        chain = self.factory.build_chain("planner")
        if not chain.providers:
            return False, res
        lexicon = self.config.memory.destructive_labels
        for _ in range(max(1, max_steps)):
            if until and self.has(until).found:
                return True, res
            elements, img = self._planner_view(res)
            try:
                decision, name = run_chain(
                    chain,
                    lambda p: p.decide(objective, elements, img),  # type: ignore[attr-defined]  # noqa: B023
                    is_empty=lambda r: r is None,
                    timeout_s=self.config.timeouts.planner_ms / 1000.0,
                )
            except ProviderError as exc:
                logger.info("planner unavailable: %s", exc)
                return False, res
            action = decision.action
            if action == "done":
                return True, res
            if action == "give-up":
                return False, res
            el = res.element_by_id(decision.target_id) if decision.target_id is not None else None
            if action in ("tap", "input") and el is None:
                return False, res  # invalid/off-screen id → hand off rather than guess
            if el is not None:  # destructive guard applies to the planner too
                probe = RouteStep(
                    kind="tap",
                    # This probe is transient policy evidence, not persisted memory. Include
                    # every semantic surface so a resource-only `deleteAccount`/`signOut`
                    # control cannot bypass a label-only guard (including when copy redacts).
                    label=el.text,
                    content_desc=el.content_desc,
                    resource_id=el.resource_id,
                )
                if is_destructive_step(probe, lexicon) and not allow_destructive:
                    return False, res
            if action == "tap" and el is not None:
                self.tap(el.id, observe=False)
            elif action == "input" and el is not None:
                self.input_text(el.id, decision.text or "", observe=False)
            elif action == "key" and decision.arg:
                self.key(decision.arg, observe=False)
            elif action == "swipe" and decision.arg in ("up", "down", "left", "right"):
                self.swipe(decision.arg, observe=False)
            elif action == "scroll-to" and decision.arg:
                self.scroll_to(decision.arg, observe=False)
            else:
                return False, res  # unusable decision → hand off
            with contextlib.suppress(StabilityTimeout):
                self.wait_stable(settle_ms=500, timeout_ms=8000)
            res = self.analyze(source="auto")  # planner may land on unlabeled screens
        return False, res

    def _goto_assist_recover(
        self, target: str, res: AnalyzeResult, *, allow_destructive: bool
    ) -> tuple[bool, AnalyzeResult]:
        """On a diverged goto, let the planner try to reach *target*. Verified by
        target-specific mapped identity plus fresh screen evidence, not the planner verdict."""
        objective = (
            f"Reach the app screen named '{target}'. If a dialog, permission prompt, or "
            "popup is blocking the screen, dismiss it (Allow, Not now, Skip, Close, "
            "Continue) to make progress toward that screen."
        )
        _, res = self._drive_with_planner(
            objective, res=res, max_steps=_ASSIST_MAX_STEPS, allow_destructive=allow_destructive
        )
        memory = self._memory
        app = memory.load(res.screen.package) if memory is not None and res.screen.package else None
        proof = (
            target_arrival_evidence(
                app,
                target,
                target,
                res.elements,
                screen_height=res.screen.height,
            )
            if app is not None and res.meta.known_screen == target
            else None
        )
        return proof is not None, res

    def _assist_suggestion(self, assist: bool) -> str | None:
        """Handoff hint: suggest --assist when it wasn't used; note it was tried if it was."""
        if not assist:
            return (
                "route diverged — continue manually, or re-run with `--assist` to let a "
                "fast model try to recover (needs `planner.enabled` + its API key)"
            )
        return "route diverged and assisted recovery could not reach the target — continue manually"

    def goto(
        self,
        goal: str,
        *,
        plan: bool = False,
        max_steps: int = 8,
        allow_destructive: bool = False,
        allow_unsafe: bool = False,
        assist: bool = False,
        from_here: bool = False,
        _attempted_route_ids: set[str] | None = None,
        _observation: AnalyzeResult | None = None,
    ) -> dict[str, Any]:
        """Drive to a remembered screen via the app map (PRD §6b).

        Resolves *goal* to a known screen, then replays the recorded steps of each edge
        on the shortest route, re-analyzing and verifying ``known_screen`` after each hop.
        On any mismatch it stops and hands back the remaining route/steps + the current
        screen, so the caller can continue manually. ``plan=True`` returns the annotated
        route without acting. Destructive steps (config ``memory.destructive_labels``)
        are refused unless *allow_destructive*. Deeplinks, cross-package actions, settings/data
        mutation, app lifecycle changes, environment changes, and other actions not provably
        limited to navigation are refused unless *allow_unsafe*. A refusal includes the full
        route/risk preview and occurs before the first state-changing step.

        ``from_here=True`` (``--from-here``): you already opened part of the journey —
        scan the first edge for the last step that still matches the *current* screen and
        resume from there (same idea as mid-auth transit resume, but for any route). When
        recognition already named a mid-journey screen so shortest-path is empty, also
        search multi-step edges that still lead to the goal and resume mid-edge.
        """
        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        # Known routes normally replay stable hierarchy selectors. Keep the happy path free
        # of OCR; `_run_steps` retries with it only when a remembered label is absent.
        # ``reach`` already paid for one bootstrap observation.  Accept it through a private
        # seam so the high-level one-call path does not immediately read the same screen again.
        # Public CLI/MCP goto behavior is unchanged.
        res = _observation or self.analyze(source="hierarchy", with_ocr=False)
        serial = res.meta.device_serial or self.device.serial
        package = res.screen.package or self.current_package()
        if not package:
            return {
                "ok": False,
                "code": "no_package",
                "goal": goal,
                "hint": "could not determine the foreground app",
            }
        # Transit resume: stranded mid-auth (foreground is a transit package while the
        # session journey belongs to another app) → resolve the goal against the ORIGIN
        # app's map and continue its transit edge from the first step that matches here.
        transit_resume = False
        sess_probe = mem.load_session(serial)
        if (
            sess_probe.package
            and package != sess_probe.package
            and matches_any(package, self.config.memory.transit_packages)
            and mem.load(sess_probe.package) is not None
        ):
            package = sess_probe.package
            transit_resume = True
        app = mem.load(package)
        if app is None or not app.screens:
            return {
                "ok": False,
                "code": "route_unknown",
                "goal": goal,
                "package": package,
                "hint": "no map for this app yet — explore with `aua analyze`",
            }
        sess = mem.load_session(serial)
        # The cursor is a memory of the last screen aua *wrote down*; the analyze above is the
        # screen the device is on now. They diverge whenever a write was lost, so replaying a
        # route planned from the cursor pressed `back` twice on the Android home screen. Mid-
        # transit the observed screen belongs to another app, and there the cursor is correct.
        current = sess.current_screen
        if not transit_resume and res.meta.known_screen:
            current = res.meta.known_screen
        lexicon = self.config.memory.destructive_labels
        target = resolve_goal(
            app,
            goal,
            start=current,
            half_life_days=self.config.memory.rank_half_life_days,
            last_goal=sess.last_goal,
            context_id=sess.active_context_id,
            destructive_labels=lexicon,
        )
        if target is None:
            return {
                "ok": False,
                "code": "route_unknown",
                "goal": goal,
                "package": package,
                "known_screens": list(context_view(app, sess.active_context_id).screens),
                "hint": "no known screen matches; explore with `aua analyze`",
            }

        def arrival_proof(observation: AnalyzeResult) -> dict[str, str] | None:
            if observation.meta.known_screen != target:
                return None
            return target_arrival_evidence(
                app,
                target,
                goal,
                observation.elements,
                screen_height=observation.screen.height,
            )

        mem.set_last_goal(serial, goal)  # remember intent for ranking even if we divert
        if current == target and not transit_resume:  # mid-transit we are NOT on target
            proof = arrival_proof(res)
            if proof is None:
                return {
                    "ok": False,
                    "code": "arrival_unproven",
                    "goal": goal,
                    "target": target,
                    "arrived": False,
                    "package": package,
                    "current_screen": current,
                    "elements": [element.compact() for element in res.elements],
                    "hint": (
                        "The map cursor names this screen, but the requested destination is "
                        "not proven by its mapped identity or a fresh non-clickable title/anchor. "
                        "A matching clickable row is navigation evidence, not arrival."
                    ),
                }
            return {
                "ok": True,
                "goal": goal,
                "target": target,
                "arrived": True,
                "already_there": True,
                "package": package,
                "route": [],
                "hops": [],
                "arrival_proof": proof,
            }
        path = _shortest_path(
            app,
            target,
            start=current,
            context_id=sess.active_context_id,
            exclude_route_ids=_attempted_route_ids,
            destructive_labels=lexicon,
        )
        resume_from = 0
        from_here_preset = False
        if not path and from_here and not transit_resume:
            # Recognition may already name a mid-journey screen while a multi-step edge
            # toward the goal still has remaining selectors on screen.
            mid = self._mid_edge_path(app, target, res.elements, context_id=sess.active_context_id)
            if mid is not None:
                path, resume_from = mid
                from_here_preset = True

        def edge_preview(edge: RouteEdge) -> dict[str, Any]:
            steps = edge.steps or _parse_legacy_steps(edge.action)
            risk_rows: list[dict[str, Any]] = []
            for step_index, step in enumerate(steps or []):
                for risk in route_step_risks(
                    step,
                    origin_package=package,
                    destructive_labels=lexicon,
                    path=f"steps[{step_index}]",
                ):
                    risk_rows.append(
                        {
                            "step_index": step_index,
                            "step": step_display(step),
                            **risk,
                        }
                    )
            return {
                "from": edge.from_screen,
                "action": edge.action,
                "to": edge.to_screen,
                "steps": [step_display(step) for step in (steps or [])],
                "replayable": steps is not None,
                "legacy": not edge.steps,
                "risk": "requires_opt_in" if risk_rows else "safe_navigation",
                "risks": risk_rows,
                # Kept for compatibility with callers that consumed the original plan field.
                "destructive": [
                    step.label for step in (steps or []) if is_destructive_step(step, lexicon)
                ],
            }

        route = [edge_preview(edge) for edge in path]
        if not path:
            return {
                "ok": False,
                "code": "route_unknown",
                "goal": goal,
                "target": target,
                "package": package,
                "current_screen": current,
                "hint": (
                    'no known route from here — try `aua goto "…" --from-here` if you '
                    "already opened part of a remembered edge, or explore with `aua analyze`"
                    if not from_here
                    else "no known route / mid-edge match from here — explore with `aua analyze`"
                ),
            }
        if plan:
            return {
                "ok": True,
                "goal": goal,
                "target": target,
                "plan": True,
                "package": package,
                "route": route,
                "note": "not executed (--plan)",
            }

        # Preflight the WHOLE learned route before even trying to resume within it. An observed
        # edge proves that its actions preceded the destination; it does not prove that a
        # deeplink, cross-package action, or configuration step was navigation-only. Doing this
        # before transit/from-here selector matching also guarantees a blind caller sees the
        # side-effect reason rather than an unrelated element-miss from inside a risky route.
        blocked: list[dict[str, Any]] = []
        for edge_index, edge in enumerate(path):
            edge_steps = edge.steps or _parse_legacy_steps(edge.action) or []
            start = resume_from if edge_index == 0 and from_here_preset else 0
            for step_index, step in enumerate(edge_steps[start:], start=start):
                for risk in route_step_risks(
                    step,
                    origin_package=package,
                    destructive_labels=lexicon,
                    path=f"route[{edge_index}].steps[{step_index}]",
                ):
                    code = risk["code"]
                    # Learned routes never execute another route/flow. This is not an opt-in
                    # side effect: `_run_steps` has no safe semantics for it, so author an
                    # explicit flow instead of mutating earlier hops and failing late.
                    if code == "nested_execution":
                        blocked.append(
                            {
                                "edge_index": edge_index,
                                "step_index": step_index,
                                "step": step_display(step),
                                **risk,
                            }
                        )
                        continue
                    if code == "destructive" and allow_destructive:
                        continue
                    if code != "destructive" and allow_unsafe:
                        continue
                    blocked.append(
                        {
                            "edge_index": edge_index,
                            "step_index": step_index,
                            "step": step_display(step),
                            **risk,
                        }
                    )
        if blocked:
            codes = {str(item["code"]) for item in blocked}
            required: list[str] = []
            if codes - {"destructive"}:
                required.append("--allow-unsafe")
            if "destructive" in codes:
                required.append("--allow-destructive")
            first = blocked[0]
            first_edge = path[int(first["edge_index"])]
            blocked_first_steps = first_edge.steps or _parse_legacy_steps(first_edge.action) or []
            first_step = blocked_first_steps[int(first["step_index"])]
            return {
                "ok": False,
                "code": "destructive_step" if codes == {"destructive"} else "unsafe_route",
                "goal": goal,
                "target": target,
                "package": package,
                "current_screen": current,
                "route": route,
                "risks": blocked,
                "required_opt_in": required,
                "step": {"display": step_display(first_step), **first_step.model_dump()},
                "hint": (
                    "No route step was executed. Review `route[].risks`, then re-run with "
                    + " and ".join(required)
                    + " only if every disclosed side effect is intended. For setup or mutation, "
                    "prefer an explicitly authored `flow run` journey."
                ),
            }
        if not from_here_preset:
            resume_from = 0
            if transit_resume or from_here:
                first_steps = path[0].steps or _parse_legacy_steps(path[0].action)
                if first_steps is None:
                    return _goto_handoff(
                        goal,
                        target,
                        "unsupported_action",
                        [],
                        route,
                        res,
                        hint=(
                            "mid-transit on a pre-v2 edge — finish manually, then re-run goto"
                            if transit_resume
                            else "first edge is not replayable — walk it once to re-record, "
                            "or author a flow"
                        ),
                    )
                if transit_resume:
                    res = self.analyze(source="auto")  # transit screens may be vision-tier
                matches = [j for j, s in enumerate(first_steps) if _match_step(res.elements, s)]
                if not matches:
                    if transit_resume:
                        return _goto_handoff(
                            goal,
                            target,
                            "element_not_found",
                            [],
                            route,
                            res,
                            remaining_steps=first_steps,
                            hint="mid-transit, but no remembered step matches this screen — "
                            "finish it manually (`aua analyze` + `aua tap-and-analyze`), then re-run `aua goto`",
                        )
                    # --from-here with no matching step: still try from the start of the edge
                    # (current screen is the edge's from_screen). Agents that are mid-edge
                    # with no selector visible yet fall through to full replay.
                    resume_from = 0
                else:
                    # Transit: first match is the auth step to perform now.
                    # --from-here: last match skips already-passed prefix taps when several
                    # remembered selectors are still on screen (e.g. Apps + Settings).
                    resume_from = matches[0] if transit_resume else matches[-1]

        hops: list[dict[str, Any]] = []
        attempted_route_ids = set(_attempted_route_ids or ())

        def arrived_result(*, early: bool = False) -> dict[str, Any]:
            proof = arrival_proof(res)
            if proof is None:
                return {
                    "ok": False,
                    "code": "arrival_unproven",
                    "goal": goal,
                    "target": target,
                    "arrived": False,
                    "package": package,
                    "final_screen": res.meta.known_screen,
                    "hops": hops,
                    "route": route,
                    "elements": [element.compact() for element in res.elements],
                    "hint": (
                        "Recognition named the target, but this frame does not prove the "
                        "goal-specific destination. Inspect the returned observation instead "
                        "of treating a clickable destination label as arrival."
                    ),
                }
            out: dict[str, Any] = {
                "ok": True,
                "goal": goal,
                "target": target,
                "arrived": True,
                "package": package,
                "final_screen": res.meta.known_screen,
                "hops": hops,
                "route": route,
                "elements": [e.compact() for e in res.elements],
                "arrival_proof": proof,
            }
            if early:
                out["early_arrival"] = True
            return out

        def replan_from(
            reached: str | None, *, attempted_route: list[dict[str, Any]]
        ) -> dict[str, Any] | None:
            """Continue from a recognized divergence without replaying an attempted edge."""
            remaining_budget = max_steps - len(hops)
            if not reached or remaining_budget <= 0:
                return None
            latest = mem.load(package)
            latest_sess = mem.load_session(serial)
            if latest is None or not _shortest_path(
                latest,
                target,
                start=reached,
                context_id=latest_sess.active_context_id,
                exclude_route_ids=attempted_route_ids,
                destructive_labels=lexicon,
            ):
                return None
            follow = self.goto(
                target,
                max_steps=remaining_budget,
                allow_destructive=allow_destructive,
                allow_unsafe=allow_unsafe,
                assist=assist,
                _attempted_route_ids=attempted_route_ids,
            )
            follow["goal"] = goal
            follow["replanned_from"] = reached
            follow["hops"] = [*hops, *follow.get("hops", [])]
            follow["route"] = [*attempted_route, *follow.get("route", [])]
            return follow

        for i, edge in enumerate(path):
            if i >= max_steps:
                return _goto_handoff(goal, target, "max_steps", hops, route[i:], res)
            all_steps = edge.steps or _parse_legacy_steps(edge.action)
            if all_steps is None:
                return _goto_handoff(
                    goal,
                    target,
                    "unsupported_action",
                    hops,
                    route[i:],
                    res,
                    hint="edge recorded before v2 — walk it once to re-record it "
                    "(or author a flow), then goto can replay it",
                )
            steps = all_steps[resume_from:] if i == 0 else all_steps
            if edge.id:
                attempted_route_ids.add(edge.id)
            edge_executed: list[dict[str, Any]] = []
            fail, res = self._run_steps(
                steps,
                origin_package=package,
                allow_destructive=allow_destructive,
                allow_unsafe_route_effects=allow_unsafe,
                res=res,
                executed=edge_executed,
                hierarchy_ocr=False,
            )
            if fail is not None:
                reached = res.meta.known_screen
                if edge_executed:
                    hops.append(
                        {
                            "action": edge.action,
                            "expected": edge.to_screen,
                            "known_screen": reached,
                            "ok": reached == target,
                            "partial": True,
                            "executed_steps": edge_executed,
                            "failed_step": step_display(fail.step),
                        }
                    )
                # A policy refusal or a manual handoff mid-transit did not test the edge.
                # Demote only when a mutation produced a different recognized screen inside
                # the origin app; that is actual contradictory route evidence.
                if (
                    edge_executed
                    and edge.id
                    and reached
                    and reached != edge.from_screen
                    and res.screen.package == package
                ):
                    with contextlib.suppress(Exception):
                        mem.record_route_outcome(package, edge.id, ok=False, reached=reached)
                if reached == target:
                    return arrived_result(early=True)
                if edge_executed:
                    replanned = replan_from(reached, attempted_route=route[: i + 1])
                    if replanned is not None:
                        return replanned
                if assist:
                    recovered, res = self._goto_assist_recover(
                        target, res, allow_destructive=allow_destructive
                    )
                    if recovered:
                        break  # post-loop confirms arrival from known_screen
                return _goto_handoff(
                    goal,
                    target,
                    fail.code,
                    hops,
                    route[i:],
                    res,
                    failed_step=fail.step,
                    remaining_steps=steps[fail.at :],
                    hint=self._assist_suggestion(assist),
                )
            reached = res.meta.known_screen
            if reached != edge.to_screen and self._observation_is_loading(res):
                # An analyzed action can legitimately return the app's settled loading shell.
                # That is evidence the tap landed, not evidence the learned route diverged.
                # Reuse the read-only mapped-screen recognizer for one bounded arrival wait
                # before demoting the route or asking an agent to recover manually.
                with contextlib.suppress(UsageError):
                    awaited = self._await_known_screen(
                        edge.to_screen,
                        timeout_ms=5_000,
                        poll_ms=200,
                    )
                    if awaited.ok and awaited.observation is not None:
                        res = awaited.observation
                        reached = res.meta.known_screen
            if reached != edge.to_screen and "apple_vision" not in res.meta.providers_used:
                # A custom-rendered destination may not be recognisable from accessibility
                # alone. Pay for one OCR retry before declaring that the route diverged.
                retry = self.analyze(source="hierarchy", with_ocr=True)
                if retry.meta.known_screen == edge.to_screen:
                    res = retry
                    reached = retry.meta.known_screen
            hops.append(
                {
                    "action": edge.action,
                    "expected": edge.to_screen,
                    "known_screen": reached,
                    "ok": reached in {edge.to_screen, target},
                    **({"executed_steps": edge_executed} if len(edge_executed) > 1 else {}),
                }
            )
            # Replaying an edge IS the check on it, and the device is the ground truth. This
            # outcome was computed and then thrown away on every hop of every `goto` ever run,
            # so a route that had stopped working stayed `verified` forever and no amount of
            # driving could clean the map. Measured 2026-08-10: 118 of 636 rows contradicted
            # another row on the same origin+action+context. Nothing here needs an agent.
            if edge.id:
                with contextlib.suppress(Exception):
                    mem.record_route_outcome(
                        package, edge.id, ok=reached == edge.to_screen, reached=reached
                    )
            if reached == target:
                return arrived_result(early=reached != edge.to_screen)
            if reached != edge.to_screen:
                replanned = replan_from(reached, attempted_route=route[: i + 1])
                if replanned is not None:
                    return replanned
                if assist:
                    recovered, res = self._goto_assist_recover(
                        target, res, allow_destructive=allow_destructive
                    )
                    if recovered:
                        break
                return _goto_handoff(
                    goal,
                    target,
                    "wrong_screen",
                    hops,
                    route[i + 1 :],
                    res,
                    hint=self._assist_suggestion(assist),
                )
        arrived = res.meta.known_screen == target
        if arrived:
            return arrived_result()
        return {
            "ok": False,
            "goal": goal,
            "target": target,
            "arrived": False,
            "package": package,
            "final_screen": res.meta.known_screen,
            "hops": hops,
            "route": route,
            # destination elements (ids) so the caller can act without a re-analyze;
            # the id cache is already warm from goto's final analyze.
            "elements": [e.compact() for e in res.elements],
        }

    # ----------------------------------------------------------------- flows (§6b)

    def flow_run(
        self,
        name: str | None = None,
        *,
        file: str | None = None,
        yaml: str | None = None,
        params: dict[str, str] | None = None,
        dry_run: bool = False,
        from_step: int = 0,
        allow_destructive: bool = True,
        assist: bool = False,
        allow_unsafe: bool = True,
        artifacts_dir: str | None = None,
        evidence: str = "failures",
        junit: bool = False,
        _observation: AnalyzeResult | None = None,
    ) -> dict[str, Any]:
        """Replay a named (or ``--file``) flow in one call — the whole journey.

        Runs through the same executor as ``goto``; on divergence returns the failing
        step's index + the remaining steps so the caller can fix or finish manually and
        resume with ``from_step``. Authored flows are deliberate intent, so destructive
        steps are ALLOWED by default (unlike goto's auto-learned replay). With *assist*
        (opt-in planner), a divergence triggers one recovery attempt (dismiss a blocking
        dialog) then resumes from the failed step before handing off.
        """
        from .flow_artifacts import FlowArtifactWriter, validate_evidence_mode
        from .flows import (
            FlowStore,
            anchor_paths,
            parse_flow_yaml,
            render_flow_yaml,
            resolve_params,
        )

        run_started = time.perf_counter()
        try:
            evidence = validate_evidence_mode(evidence)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        if junit and not artifacts_dir:
            raise UsageError("--junit needs --artifacts-dir")
        if not artifacts_dir and evidence != "failures":
            raise UsageError("--evidence needs --artifacts-dir")
        sources = [name is not None, file is not None, yaml is not None]
        if sum(sources) != 1:
            raise UsageError(
                "flow run needs exactly one of NAME, --file, or --yaml",
                hint="use a saved name, a YAML path, or an inline YAML body",
            )

        base_dir: Path | None = None
        flow_file: Path | None = None
        if file is not None:
            path = Path(file).expanduser()
            if not path.is_file():
                # Name the absolute location, always. Reporting the relative path back is
                # what hid this bug: "no flow file at flows/x.yaml" looks like a typo, while
                # "no flow file at /Users/daemon-was-started-here/flows/x.yaml" tells you
                # immediately that the lookup happened somewhere you did not expect.
                raise UsageError(
                    f"no flow file at {path.resolve()}",
                    hint=(
                        "That is where a relative path resolves for the process running the "
                        "flow, which is not always the shell you typed in."
                    )
                    if not path.is_absolute()
                    else None,
                )
            flow = parse_flow_yaml(path.read_text(encoding="utf-8"), name=path.stem)
            flow_file = path.resolve()
            root_source_id = str(flow_file)
            base_dir = flow_file.parent
        elif yaml is not None:
            flow = parse_flow_yaml(yaml, name="inline")
            root_source_id = "inline:" + hashlib.sha256(yaml.encode("utf-8")).hexdigest()
        elif name is not None:
            store = FlowStore(self.config.memory)
            flow = store.load(name)
            root_source_id = str(store.path(name).resolve())
            base_dir = store.flows_dir()
        # A flow's optional YAML `name:` is display metadata. Named replay is addressed by
        # its storage key (the filename), while file replay must keep using the exact file.
        # Returning or suggesting the display name made failed journeys impossible to resume
        # whenever those identities differed.
        runnable_name = name or flow.name
        resume_prefix: str | None
        if flow_file is not None:
            resume_prefix = f"aua flow run --file {shlex.quote(str(flow_file))}"
        elif yaml is not None:
            resume_prefix = None
        else:
            resume_prefix = f"aua flow run {shlex.quote(runnable_name)}"
        identity: dict[str, Any] = {"flow": runnable_name}
        if flow.name != runnable_name:
            identity["declared_name"] = flow.name
        if flow_file is not None:
            identity["file"] = str(flow_file)
        if yaml is not None:
            identity["source"] = "inline_yaml"

        artifact_writer: FlowArtifactWriter | None = None
        if artifacts_dir:
            artifact_writer = FlowArtifactWriter(
                artifacts_dir,
                flow_name=runnable_name,
                evidence=evidence,
                junit=junit,
                screenshot=lambda path: str(self.screenshot(str(path)).detail or path),
                diagnostics=lambda: (
                    self.platform.diagnostic_logs(self.device, lines=400)
                    if self.platform.supports("device.logs")
                    else None
                ),
            )

        def finish(
            result: dict[str, Any], observation: AnalyzeResult | None = None
        ) -> dict[str, Any]:
            duration_ms = max(0, int((time.perf_counter() - run_started) * 1000))
            result["duration_ms"] = duration_ms
            if artifact_writer is None:
                return result
            if result.get("ok") is False:
                if observation is not None:
                    artifact_writer.record_failure(result, observation)
                else:
                    artifact_writer.record_preflight_failure(result)
            return artifact_writer.finalize(
                result,
                canonical_flow_yaml=render_flow_yaml(flow),
                duration_ms=duration_ms,
            )

        active_context: str | None = None
        if flow.context_id and self._memory is not None and self._device is not None:
            session = self._memory.load_session(self._device.serial)
            # Dry-run is intentionally device-read-free. Disclose compatibility only when the
            # persisted context belongs to this flow's app; another foreground's cursor is not
            # evidence either way.
            if flow.app is None or session.package == flow.app:
                active_context = session.active_context_id
        if flow.arrival:
            _parse_await_terms(flow.arrival, require_positive=True)
        self._validate_flow_arrival_screen(flow, flow.app, flow.context_id)
        steps = resolve_params(flow, params or {})
        if base_dir is not None:
            # A path *inside* a flow belongs to the flow, not to the caller's cwd.
            steps = anchor_paths(steps, base_dir)
        if not 0 <= from_step < len(steps):
            raise UsageError(f"--from-step {from_step} out of range (flow has {len(steps)} steps)")
        steps_slice = steps[from_step:]
        flow_plan = self._preflight_nested_flow_graph(
            steps_slice,
            flow_dir=base_dir,
            flow_app=flow.app,
            context_id=flow.context_id,
            ancestors=(root_source_id,),
        )
        disclosure = self._resolved_flow_disclosure(
            steps_slice,
            flow_dir=base_dir,
            flow_app=flow.app,
            plan=flow_plan,
            index_offset=from_step,
        )
        lexicon = self.config.memory.destructive_labels

        def step_is_destructive(step: RouteStep, directory: Path | None) -> bool:
            if is_destructive_step(step, lexicon):
                return True
            if any(step_is_destructive(child, directory) for child in step.substeps):
                return True
            if step.kind == "flow" and step.arg:
                node = flow_plan.flow_graph.get(self._flow_ref_key(step.arg, directory))
                return bool(
                    node and any(step_is_destructive(child, node.directory) for child in node.steps)
                )
            return False

        destructive_indices = [
            from_step + index
            for index, step in enumerate(steps_slice)
            if step_is_destructive(step, base_dir)
        ]
        if dry_run:
            return finish(
                {
                    "ok": True,
                    **identity,
                    "dry_run": True,
                    "app": flow.app,
                    "context_id": flow.context_id,
                    "arrival": flow.arrival,
                    "arrival_screen": flow.arrival_screen,
                    "arrival_status": flow.arrival_status or "unverified",
                    "active_context_id": active_context,
                    "context_compatible": (
                        None
                        if flow.context_id is not None and active_context is None
                        else flow.context_id in (None, active_context)
                    ),
                    "would_execute": False,
                    "params_declared": sorted(flow.params),
                    "steps": disclosure["steps"],
                    "risks": disclosure["risks"],
                    "effects": disclosure["effects"],
                    "flow_graph": disclosure["flow_graph"],
                    "note": "not executed (--dry-run)",
                }
            )

        if destructive_indices and not allow_destructive:
            index = destructive_indices[0]
            return finish(
                {
                    "ok": False,
                    "code": "destructive_step",
                    **identity,
                    "step_index": index,
                    "failed_step": {
                        "display": step_display(steps[index]),
                        **steps[index].model_dump(),
                    },
                    "steps_run": [],
                    "remaining_steps": [step_display(step) for step in steps[index:]],
                    "hint": "review the full flow, then rerun with --allow-destructive",
                }
            )

        # Execution always begins from a current foreground observation. ``reach`` hands its
        # just-captured frame through the private seam; direct flow_run pays for exactly one.
        res = _observation or self.analyze(source="hierarchy", with_ocr=False)
        active_context, entry_mismatch = self._flow_runtime_state(
            flow,
            res,
            refresh_context=True,
            transit_step=steps_slice[0] if from_step > 0 else None,
        )
        if entry_mismatch is not None and not (
            entry_mismatch["code"] == "flow_app_mismatch"
            and self._flow_leading_launch_establishes_origin(flow, steps_slice)
        ):
            if entry_mismatch["code"] == "flow_context_mismatch":
                raise UsageError(
                    f"flow '{flow.name}' was recorded for context {flow.context_id}, but the "
                    f"active context is {active_context}",
                    hint="activate the recorded feature/flag context before replaying this flow",
                )
            raise UsageError(
                f"flow '{flow.name}' belongs to {flow.app}, but the foreground package is "
                f"{res.screen.package or 'unknown'}",
                hint=(
                    "bring the owning app to foreground, or make launch_app for that exact "
                    "package the flow's first step"
                ),
            )
        executed: list[dict[str, Any]] = []

        def _exec(
            slice_start: int, res_in: AnalyzeResult | None
        ) -> tuple[Any, AnalyzeResult, int | None]:
            ex: list[dict[str, Any]] = []
            # ``res_in`` is always present: either the fresh top-level entry observation or
            # the fresh handoff returned by a failed/assisted attempt.
            assert res_in is not None
            f, r = self._execute_flow_steps(
                flow,
                steps[slice_start:],
                res=res_in,
                allow_destructive=allow_destructive,
                scroll_fallback=True,
                executed=ex,
                flow_depth=0,
                hierarchy_ocr=True,
                flow_dir=base_dir,
                allow_unsafe_route_effects=allow_unsafe,
                allow_transit_resume=slice_start > 0,
                flow_plan=flow_plan,
                flow_artifacts=artifact_writer,
            )
            for e in ex:
                e["index"] += slice_start  # absolute flow indices
                path = e.get("path")
                if isinstance(path, list) and path and isinstance(path[0], int):
                    e["path"] = [path[0] + slice_start, *path[1:]]
            executed.extend(ex)
            return f, r, (slice_start + f.at if f is not None else None)

        fail, res, idx = _exec(from_step, res)
        if fail is not None and assist and self.factory.is_enabled("planner"):
            objective = (
                f"A UI automation step could not run: {step_display(fail.step)}. If a "
                "dialog, permission prompt, or popup is blocking the screen, dismiss it "
                "(Allow, Not now, Skip, Close, Continue) so the flow can proceed."
            )
            recovered, res = self._drive_with_planner(
                objective, res=res, max_steps=_ASSIST_MAX_STEPS, allow_destructive=allow_destructive
            )
            if recovered and idx is not None:
                fail, res, idx = _exec(idx, res)  # resume from the failed step
        if fail is not None:
            assert idx is not None
            if resume_prefix is not None:
                hint = (
                    "fix the flow or finish the step manually, then resume with "
                    f"`{resume_prefix} --from-step {idx}`"
                )
            else:
                hint = (
                    "fix the flow or finish the step manually, then submit the same inline "
                    f"YAML again with from_step={idx}"
                )
            if not assist:
                hint += (
                    "; or add `--assist` to let a fast model clear blockers "
                    "(needs `planner.enabled` + its API key)"
                )
            failure_result = {
                "ok": False,
                "code": fail.code,
                **identity,
                "step_index": idx,
                "failed_step": {"display": step_display(fail.step), **fail.step.model_dump()},
                "steps_run": executed,
                "remaining_steps": [step_display(s) for s in steps[idx:]],
                "current_screen": res.meta.known_screen,
                "elements": [
                    {"id": e.id, "label": e.text or e.content_desc, "clickable": e.clickable}
                    for e in res.elements
                    if (e.text or e.content_desc)
                ][:20],
                "hint": hint,
            }
            if fail.detail:
                failure_result["failure_detail"] = fail.detail
            if resume_prefix is not None:
                failure_result["resume_call"] = f"{resume_prefix} --from-step {idx}"
            else:
                failure_result["resume_from_step"] = idx
            return finish(failure_result, res)
        arrival_verified, arrival_code, res, arrival_evidence = self._flow_arrival_evidence(
            flow,
            res,
        )
        out = {
            "ok": True,
            **identity,
            "steps_run": executed,
            "final_screen": res.meta.known_screen,
            # destination elements (ids) so the caller can act without a re-analyze
            "elements": [e.compact() for e in res.elements],
            **arrival_evidence,
        }
        if arrival_verified is False:
            out["ok"] = False
            out["code"] = arrival_code or "arrival_unverified"
        return finish(out, res if out.get("ok") is False else None)

    def explore_mine(
        self, source: str, *, package: str | None = None, save: bool = True
    ) -> dict[str, Any]:
        """Mine an app's source tree for deeplinks and save them to its playbook (§6b).

        Deeplinks are shortcuts — jump straight to a screen instead of navigating — and
        the app declares them in its source. Found links are recorded so the agent can
        reuse them (`aua open-and-analyze <uri>`); templated ones (``$id``/``{id}``) are flagged.
        """
        from .explore import mine_deeplinks

        result = mine_deeplinks(Path(source))
        pkg = package or self.current_package()
        saved = 0
        mem = self._memory
        if save and pkg and mem is not None:
            for d in result.deeplinks:
                note = "mined from source" + (f" ({d.source})" if d.source else "")
                if d.templated:
                    note += " — templated, fill the placeholder"
                mem.remember_deeplink(pkg, d.uri, note=note)
                saved += 1
        return {
            "ok": True,
            "action": "explore-mine",
            "package": pkg,
            "source": source,
            "schemes": result.schemes,
            "found": len(result.deeplinks),
            "saved": saved,
            "deeplinks": result.as_dict()["deeplinks"],
        }

    def explore_plan(self, *, package: str | None = None, max_tasks: int = 12) -> dict[str, Any]:
        """A prioritized exploration worklist for the calling agent (the offline-agent mode).

        Reads the app's map + playbook and returns concrete next actions. Existing map debt is
        more valuable than speculative shortcuts: unresolved research comes first, then dead-end
        screens, then unprobed deeplinks. Every task declares whether it can leave the app or
        mutate state so an agent never turns "explore" into accidental authorization.
        """
        from .explore import _is_templated as _is_templated_uri
        from .reconcile import ReconciliationStore

        mem = self._memory
        pkg = package or self.current_package()
        out: dict[str, Any] = {"ok": True, "action": "explore-plan", "package": pkg, "tasks": []}
        if mem is None or not pkg:
            out["hint"] = "no memory/package — run on a device with memory enabled"
            return out
        app = mem.load(pkg)
        tasks: list[dict[str, Any]] = []
        if app is None or (not app.screens and not app.deeplinks):
            out["known"] = {"screens": 0, "routes": 0, "deeplinks": 0}
            out["bootstrap"] = (
                "no map yet — mine deeplinks first (`aua explore mine <repo> --app "
                f"{pkg}`), then launch + log in (`aua about` for the recipe) and `aua open-and-analyze` "
                "each concrete deeplink, analyzing after each to seed screens fast."
            )
            out["hint"] = "then re-run `aua explore plan`"
            return out
        out["known"] = {
            "screens": len(app.screens),
            "routes": len(app.routes),
            "deeplinks": len(app.deeplinks),
        }

        def add_task(
            *,
            kind: str,
            do: str,
            why: str,
            risk: str,
            external: bool = False,
            destructive: bool = False,
        ) -> None:
            tasks.append(
                {
                    "kind": kind,
                    "do": do,
                    "why": why,
                    "risk": risk,
                    "external": external,
                    "destructive": destructive,
                    "requires_explicit_authorization": destructive,
                }
            )

        # 1) Current structural questions. `plan` also refreshes stale task materialization, so
        # this worklist cannot ignore hundreds of audit findings merely because deeplinks exist.
        latest = mem.latest_session(pkg)
        active_context = latest.active_context_id if latest else DEFAULT_CONTEXT_ID
        research = ReconciliationStore(mem).plan(pkg, context_id=active_context)
        issue_rank = {
            "orphan_route": 0,
            "route_conflict": 1,
            "unreplayable_route": 2,
            "stale_screen": 3,
            "duplicate_screen": 4,
            "poor_name": 5,
            "provisional_route": 6,
            "unverified_context": 7,
            "legacy_context": 8,
        }
        open_research = sorted(
            (task for task in research if task.status == "open"),
            key=lambda task: (issue_rank.get(task.issue_type, 99), task.id),
        )
        for task in open_research:
            affected = ", ".join(task.affected_ids[:3]) or "map"
            add_task(
                kind="resolve_map_issue",
                do=(
                    f'aua reconcile plan --app "{pkg}"; investigate task {task.id} '
                    "with source or a fresh runtime observation, then submit evidence"
                ),
                why=f"{task.issue_type} affects {affected}",
                risk="read_only_research",
            )

        # 2) Dead ends before shortcuts. The instruction is explicitly non-destructive: a screen
        # with an unknown Delete/Pay/Send control is not permission to press it.
        outgoing = {edge.from_screen for edge in app.routes if edge.status == "verified"}
        for name in app.screens:
            if name not in outgoing:
                add_task(
                    kind="expand_screen",
                    do=(
                        f'aua goto "{name}"; aua analyze; inspect unvisited controls one at a '
                        "time, skipping destructive or external actions unless authorized"
                    ),
                    why="screen has no verified routes out — map safe navigation from it",
                    risk="interactive_navigation",
                )

        # 3) Deeplinks are speculative external intents, not the first exploration strategy.
        # Flag obviously destructive URI vocabulary and refuse to turn it into a ready-to-run
        # command. Other links remain runnable, but their metadata tells an orchestrator that the
        # action crosses the normal UI boundary and arrival must be verified.
        destructive_words = {
            "delete",
            "remove",
            "logout",
            "signout",
            "purchase",
            "subscribe",
            "payment",
            "erase",
        }
        destructive_words.update(
            word.casefold().replace(" ", "") for word in self.config.memory.destructive_labels
        )
        for d in app.deeplinks:
            if d.probed:
                continue
            normalized_uri = d.uri.casefold().replace("-", "").replace("_", "")
            destructive = any(word and word in normalized_uri for word in destructive_words)
            templated = _is_templated_uri(d.uri)
            if destructive:
                do = (
                    f'inspect the source/handler for "{d.uri}"; do not open it without explicit '
                    "authorization for the destructive effect"
                )
                risk = "destructive_external_intent"
            elif templated:
                do = (
                    f'fill the placeholder in "{d.uri}" with a non-sensitive fixture, then '
                    "`aua open-and-analyze` it and verify arrival"
                )
                risk = "templated_external_intent"
            else:
                do = f'aua open-and-analyze "{d.uri}"'
                risk = "external_intent"
            add_task(
                kind="probe_template" if templated else "probe_deeplink",
                do=do,
                why=(
                    "unprobed deeplink — delivered intent is not proof of arrival"
                    if not templated
                    else "templated deeplink needs a safe fixture and verified destination"
                ),
                risk=risk,
                external=True,
                destructive=destructive,
            )

        out["tasks"] = tasks[: max(0, max_tasks)]
        out["remaining"] = len(tasks)
        out["remaining_by_kind"] = dict(Counter(task["kind"] for task in tasks))
        out["hint"] = (
            "resolve map debt and safe dead ends before speculative intents; results auto-record. "
            "Re-run `aua explore plan` as the map improves, and never treat a listed task as "
            "authorization for destructive or external side effects."
        )
        return out

    def flow_save(
        self,
        name: str,
        *,
        last: int = 12,
        force: bool = False,
        save: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Preview or save a trustworthy homogeneous suffix of recorded actions.

        Redacted inputs/labels become required ``${PARAM_n}`` placeholders — typed
        values are never recorded, so the agent fills them in the saved YAML.
        """
        from .flows import (
            Flow,
            FlowStore,
            check_saveable,
            recorded_selector_resilience,
            recorded_step_blockers,
            render_flow_yaml,
            steps_from_recent,
        )
        from .memory import capture_arrival_for_current, capture_arrival_predicate

        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        if force and not save:
            raise UsageError(
                "--force only applies when --save writes the previewed flow",
                hint="preview first, then add --save --force to replace the existing file",
            )
        if dry_run and save:
            raise UsageError("--dry-run and --save cannot be combined")

        # Terminal proof must describe the screen on the device now, never the cursor left by
        # an older observation.  This read also applies any foreground/context boundary before
        # the journal suffix is selected.
        current = self.analyze(source="hierarchy", with_ocr=False)
        if not self._join_memory_writers(timeout_s=5.0):
            raise UsageError(
                "recorded-flow provenance is still being finalized",
                hint="retry `aua flow save` after the current memory update completes",
            )
        sess = mem.load_session(self.device.serial)
        if last < 1:
            raise UsageError("flow save --last must be at least 1")
        requested = last
        journal = list(sess.recent)
        if not journal:
            raise UsageError(
                "no recorded actions to save",
                hint="drive the app first (tap/input/…), then `aua flow save <name>`",
            )

        warnings: list[str] = []
        newest = journal[-1]
        segment_id: int | None
        origin: str | None
        context_id: str | None
        if newest.capture_segment is not None:
            if not newest.origin_package:
                raise UsageError(
                    "recorded actions have no proven origin package",
                    hint="analyze the app first, then repeat the intended actions before saving",
                )
            if newest.capture_segment != sess.capture_segment:
                raise UsageError(
                    "no actions exist in the current capture segment",
                    hint=(
                        f"the segment was reset because {sess.capture_boundary_reason}; drive the "
                        "intended app/context again before saving"
                        if sess.capture_boundary_reason
                        else "drive the intended app/context again before saving"
                    ),
                )
            segment_id = newest.capture_segment
            segment = [step for step in journal if step.capture_segment == segment_id]
            origin = newest.origin_package
            context_id = newest.context_id
        else:
            # One-release compatibility for journals captured before per-action provenance.
            # Transit packages remain folded into the owning app; a foreign non-transit action
            # starts the suffix.  The uncertainty is disclosed rather than presented as proof.
            segment_id = None
            origin = (
                sess.package
                if newest.package is None
                or matches_any(newest.package, self.config.memory.transit_packages)
                else newest.package
            )
            suffix: list[RouteStep] = []
            for step in reversed(journal):
                pkg = step.package
                if pkg not in (None, origin) and not matches_any(
                    pkg, self.config.memory.transit_packages
                ):
                    break
                suffix.append(step)
            segment = list(reversed(suffix))
            # A legacy step has no capture-time context.  The session's *current* context may
            # have changed since the action, so attaching it would turn uncertainty into false
            # typed provenance (and could make the flow replay in the wrong variant).
            context_id = None
            warnings.append(
                "legacy recorded actions have no per-action origin/context provenance; "
                "the newest package-compatible suffix was selected conservatively"
            )

        selected = segment[-requested:]
        boundary_omitted = max(0, min(requested, len(journal)) - len(selected))
        if boundary_omitted:
            warnings.append(
                f"requested --last {requested} crosses a capture boundary; omitted "
                f"{boundary_omitted} older action(s) and used only the newest homogeneous suffix"
            )
        if not selected:
            raise UsageError(
                "no actions exist in the current capture segment",
                hint=(
                    f"the segment was reset because {sess.capture_boundary_reason}; drive the "
                    "intended app/context again before saving"
                    if sess.capture_boundary_reason
                    else "drive the intended app/context again before saving"
                ),
            )
        if segment_id is not None:
            expected_provenance = (segment_id, origin, context_id)
            if any(
                (step.capture_segment, step.origin_package, step.context_id) != expected_provenance
                for step in selected
            ):
                raise UsageError(
                    "selected recorded actions have mixed origin/context provenance",
                    hint=(
                        "nothing was saved; drive the journey again after a clean app/context "
                        "boundary, or request a smaller homogeneous --last suffix"
                    ),
                )

        captured_arrival = capture_arrival_for_current(
            selected,
            session=sess,
            observation_package=current.screen.package,
            observation_fingerprint=current.meta.fingerprint,
        )
        captured_predicate = (
            capture_arrival_predicate(captured_arrival.proof)
            if captured_arrival.proof is not None
            else None
        )
        arrival_screen: str | None = None
        arrival_reason: str
        if current.screen.package != origin:
            arrival_reason = (
                f"current package {current.screen.package or 'unknown'} is not the selected "
                f"segment origin {origin or 'unknown'}"
            )
        elif sess.active_context_id != context_id:
            arrival_reason = (
                f"current context {sess.active_context_id} does not match selected context "
                f"{context_id}"
            )
        elif current.meta.known_screen:
            from .memory import LEGACY_CONTEXT_ID

            app = mem.load(origin) if origin else None
            mapped = app.screens.get(current.meta.known_screen) if app is not None else None
            if (
                mapped is not None
                and not mapped.stale
                and mapped.context_id in {context_id, LEGACY_CONTEXT_ID}
            ):
                arrival_screen = current.meta.known_screen
                arrival_reason = "current destination was freshly recognized as a mapped screen"
            else:
                arrival_reason = (
                    "current known_screen has no fresh map record in the selected capture context"
                )
        else:
            arrival_reason = "current destination is not a mapped known_screen"
        if arrival_screen is None and captured_predicate is not None:
            arrival_reason = captured_arrival.reason

        def arrival_payload() -> dict[str, Any]:
            payload: dict[str, Any] = {
                "status": "verified" if arrival_screen or captured_predicate else "unverified",
                "screen": arrival_screen,
                "reason": arrival_reason,
            }
            if captured_predicate is not None and arrival_screen is None:
                payload.update(
                    predicate=captured_predicate,
                    source="satisfied_action_until",
                    fingerprint=current.meta.fingerprint,
                )
            return payload

        selector_resilience = [
            item.model_dump(mode="json") for item in recorded_selector_resilience(selected)
        ]

        store = FlowStore(self.config.memory)
        path = store.path(name)
        existed_before = path.is_file()
        required_save_mode = "force" if existed_before else "create"
        save_call = f"aua flow save {shlex.quote(name)} --last {requested} --save"
        if existed_before:
            save_call += " --force"
        invalid_force_probe = {
            "case": "force_without_save",
            "error_code": "usage",
            "cli": (
                f"aua --expect-error usage flow save {shlex.quote(name)} --last {requested} --force"
            ),
            "mcp": {
                "tool": "flow_save",
                "arguments": {
                    "name": name,
                    "last": requested,
                    "force": True,
                    "expect_error": "usage",
                },
            },
        }
        capture_blockers = recorded_step_blockers(selected)
        if capture_blockers:
            return {
                "ok": False,
                "action": "flow-save-preview",
                "flow": name,
                "path": str(path),
                "exists": existed_before,
                "collision": existed_before,
                "status": "not_saveable",
                "required_save_mode": required_save_mode,
                "saved": False,
                "saveable": False,
                "steps": len(selected),
                "scope": {
                    "requested_last": requested,
                    "selected": len(selected),
                    "origin_package": origin,
                    "context_id": context_id,
                    "capture_segment": segment_id,
                    "boundary_omitted": boundary_omitted,
                },
                "capture_warnings": capture_blockers,
                # One-release response alias for callers that handled selector refusals.
                "selector_warnings": capture_blockers,
                "arrival_proof": arrival_payload(),
                "selector_resilience": selector_resilience,
                "hint": (
                    "Nothing was written. Re-record with fully captured replay arguments and "
                    "a unique stable, privacy-safe selector, or author the step explicitly in YAML."
                ),
                **({"warnings": warnings} if warnings else {}),
            }

        materialized = [
            step.model_copy(update={"package": None}) if step.package == origin else step
            for step in selected
        ]
        steps, params = steps_from_recent(materialized)

        flow = Flow(
            name=name,
            app=origin,
            context_id=context_id,
            description=f"Recorded from the last {len(steps)} session actions",
            arrival_screen=arrival_screen,
            arrival=(captured_predicate if arrival_screen is None else None),
            arrival_status=(
                "mapped"
                if arrival_screen
                else "predicate_verified"
                if captured_predicate
                else "unverified"
            ),
            params=params,
            steps=steps,
        )
        preview = render_flow_yaml(flow)
        warnings.extend(check_saveable(flow))
        should_save = save and not dry_run
        if should_save:
            path = store.save(flow, force=force)
            self._flows_cache.clear()
        out = {
            "ok": True,
            "action": "flow-save" if should_save else "flow-save-preview",
            "flow": name,
            "path": str(path),
            "exists": path.is_file(),
            "collision": existed_before,
            "status": (
                "overwritten"
                if should_save and existed_before
                else "created"
                if should_save
                else "preview_existing"
                if existed_before
                else "preview_new"
            ),
            "required_save_mode": required_save_mode,
            "steps": len(steps),
            "params_needed": sorted(params),
            "saved": should_save,
            "saveable": True,
            "dry_run": dry_run,
            "scope": {
                "requested_last": requested,
                "selected": len(selected),
                "origin_package": origin,
                "context_id": context_id,
                "capture_segment": segment_id,
                "boundary_omitted": boundary_omitted,
            },
            "arrival_proof": arrival_payload(),
            "arrival_status": flow.arrival_status or "unverified",
            "selector_resilience": selector_resilience,
            "preview": preview,
            "hint": (
                "saved; edit/fill ${PARAM_n}, then preview replay with the run_preview_call"
                if should_save
                else "nothing written; review the scope, selectors, and arrival proof, then run save_call"
            ),
        }
        if should_save:
            out["run_preview_call"] = f"aua flow run {shlex.quote(name)} --dry-run"
        else:
            out["save_call"] = save_call
            if existed_before:
                out["invalid_mode_probe"] = invalid_force_probe
        if dry_run:
            warnings.append(
                "--dry-run remains a deprecated non-writing alias; flow save previews by default"
            )
        if warnings:
            out["warnings"] = warnings
        return out

    def flow_delete(self, name: str) -> dict[str, Any]:
        """Idempotently delete one named flow through the shared engine boundary."""
        from .flows import FlowStore

        store = FlowStore(self.config.memory)
        path = store.path(name)
        deleted = store.delete(name)
        if deleted:
            self._flows_cache.clear()
        return {
            "ok": True,
            "action": "flow-delete",
            "flow": name,
            "path": str(path),
            "deleted": deleted,
            "status": "deleted" if deleted else "already_absent",
        }

    def flow_list(self) -> dict[str, Any]:
        """List flows and disclose compatibility with the attached session when known."""
        from .flows import FlowStore

        package: str | None = None
        context_id: str | None = None
        mem = self._memory
        can_observe_foreground = self._device is not None
        if not can_observe_foreground:
            # CLI invocations and a newly constructed MCP engine are fresh here.  Discover an
            # already-online target before using the lazy device property: an absent/offline
            # device must leave this read-only listing available rather than paying a failed u2
            # connection (or changing the device state to make one available).
            with contextlib.suppress(Exception):
                configured_serial = self.config.device.serial
                can_observe_foreground = any(
                    info.state == "device"
                    and (configured_serial is None or info.serial == configured_serial)
                    for info in self.list_devices()
                )
        if can_observe_foreground:
            with contextlib.suppress(Exception):
                package = self.current_package()
                device = self._device
                if mem is not None and device is not None and package is not None:
                    session = mem.load_session(device.serial)
                    if session.package == package:
                        context_id = session.active_context_id
        return {
            "flows": FlowStore(self.config.memory).list(
                active_package=package if context_id is not None else None,
                active_context_id=context_id,
            ),
            "active_package": package,
            "active_context_id": context_id,
        }

    def _goal_session_plan(self, goal: str, observation: AnalyzeResult) -> Any:
        """Build the shared CLI/MCP goal plan from an observation already in hand."""
        from .capabilities import capabilities_for_goal
        from .flows import Flow, FlowStore, anchor_paths, resolve_params
        from .session import plan_goal_session

        mem = self._memory
        app: AppMap | None = None
        current_screen = observation.meta.known_screen
        context_id = DEFAULT_CONTEXT_ID
        package = observation.screen.package
        if mem is not None and package:
            app = mem.load(package)
            session = mem.load_session(observation.meta.device_serial or self.device.serial)
            if session.package == package:
                current_screen = observation.meta.known_screen or session.current_screen
                context_id = session.active_context_id

        flows: list[Flow] = []
        resolved_flow_evidence: dict[str, dict[str, Any]] = {}
        # A malformed flow must not prevent a new agent from starting a session.  It stays
        # visible through `flow list`, whose error is the right repair surface.
        store = FlowStore(self.config.memory)
        for item in store.list():
            storage_name = item.get("storage_name")
            if not isinstance(storage_name, str) or item.get("error"):
                continue
            try:
                flow = store.load(storage_name)
            except Exception:
                # Isolate each artifact: one renamed/corrupt flow must not hide every valid
                # recommendation that follows it alphabetically.
                continue
            if flow.app in (None, package) and flow.context_id in (None, context_id):
                source = store.path(storage_name).resolve()
                with contextlib.suppress(Exception):
                    resolved_steps = anchor_paths(resolve_params(flow, {}), source.parent)
                    resolved_plan = self._preflight_nested_flow_graph(
                        resolved_steps,
                        flow_dir=source.parent,
                        flow_app=flow.app,
                        context_id=flow.context_id,
                        ancestors=(str(source),),
                    )
                    resolved_flow_evidence[storage_name] = self._resolved_flow_disclosure(
                        resolved_steps,
                        flow_dir=source.parent,
                        flow_app=flow.app,
                        plan=resolved_plan,
                    )
                # `flow run` loads by storage key, not by the optional declared display name.
                # Keep aliases/description for goal matching while emitting an executable call.
                declared_name = flow.name
                aliases = list(flow.aliases)
                if declared_name != storage_name and declared_name not in aliases:
                    aliases.append(declared_name)
                flows.append(flow.model_copy(update={"name": storage_name, "aliases": aliases}))

        return plan_goal_session(
            goal,
            observation,
            app=app,
            context_id=context_id,
            current_screen=current_screen,
            flows=flows,
            destructive_labels=self.config.memory.destructive_labels,
            relevant_capabilities=capabilities_for_goal(goal),
            resolved_flow_evidence=resolved_flow_evidence,
        )

    def session_start(
        self,
        goal: str,
        *,
        observation: AnalyzeResult | None = None,
        contract_file: str | None = None,
        contract_yaml: str | None = None,
        artifacts_dir: str | None = None,
        evidence: str = "failures",
        junit: bool = False,
        wait_for_lease_s: float = 0,
        start_emulator: bool = False,
        headed: bool = False,
        audio: bool = False,
        avd: str | None = None,
        package: str | None = None,
        activity: str | None = None,
        apk: str | None = None,
        reinstall: bool = False,
        fresh: bool = False,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Observe once and return the safest goal-specific CLI and MCP next call.

        Supplying *observation* is an internal composition seam used by ``reach``. A caller may
        explicitly name *package*/*activity* to launch into the intended app first; the launch's
        folded observation is reused, so bootstrap still performs exactly one screen read.

        *apk* makes this the single bootstrap call: boot an emulator if asked, put the build on it
        (skipping the push when that version is already there), launch it, observe, and plan. The
        bundle also names the package, so *package* is optional when *apk* is given.
        """
        if not goal.strip():
            raise UsageError("session start needs a non-empty goal")
        from .session_artifacts import validate_session_evidence_mode
        from .session_contracts import load_session_contract, render_session_contract_yaml

        contract = (
            load_session_contract(file=contract_file, yaml=contract_yaml)
            if contract_file is not None or contract_yaml is not None
            else None
        )
        canonical_contract_yaml = render_session_contract_yaml(contract) if contract else None
        try:
            evidence = validate_session_evidence_mode(evidence)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        if junit and not artifacts_dir:
            raise UsageError("--junit needs --artifacts-dir")
        if not artifacts_dir and evidence != "failures":
            raise UsageError("--evidence needs --artifacts-dir")
        if wait_for_lease_s < 0:
            raise UsageError("--wait-for-lease must not be negative")
        if wait_for_lease_s and observation is not None:
            raise UsageError("wait_for_lease_s cannot be combined with an injected observation")
        self._lease_wait_s = float(wait_for_lease_s)
        self._lease_waited_ms = 0
        emulator_started = False
        if observation is None and start_emulator and self._device is None:
            online = [device for device in self.list_devices() if device.state == "device"]
            if not online:
                from . import emulator as emulator_mod
                from . import leases

                boot_owner = leases.resolve_owner(getattr(self, "_lease_owner", None))
                boot = emulator_mod.start(
                    avd,
                    headless=not headed,
                    audio=audio,
                    cache_dir=self.config.cache.dir,
                    owner=boot_owner,
                )
                serial = str(boot["serial"])
                self.config.device.serial = serial
                self._lease_serial = serial
                self._lease_owner_resolved = boot_owner
                emulator_started = True
        installed_bundle: dict[str, Any] | None = None
        try:
            if observation is None and apk:
                # Install before the launch, not after: `--app` names the package to open, and a
                # bootstrap that launched first would either open the previous build or fail on a
                # device that has never had this app. Folding it in here is what lets one
                # `session start` cover boot, install, launch, observe, and plan.
                bundled = self.install_app(
                    apk,
                    package=package,
                    mode="fresh" if fresh else "reinstall" if reinstall else "if-needed",
                    confirmed=confirmed,
                    launch=False,
                    observe=False,
                )
                installed_bundle = bundled.app_install
                if package is None and installed_bundle:
                    # The bundle names the app, so `--apk` alone is enough to know what to open.
                    package = str(installed_bundle.get("package") or "") or None
            if observation is None and package:
                launched = self.app(
                    "launch",
                    package=package,
                    activity=activity,
                    observe=True,
                )
                observation = launched.observation
                # ``app launch`` deliberately withholds ``next_actions`` when its folded
                # hierarchy came from a one-sample/timeout/unchanged settle path.  Reusing that
                # explicitly unstable frame here makes the goal planner answer
                # ``manual_observation`` even though the immediately following hierarchy is
                # actionable.  Session bootstrap owns the launch, so pay for that one bounded
                # authoritative read now instead of handing every agent a redundant analyze.
                if (
                    observation is not None
                    and launched.next_actions is None
                    and isinstance(launched.note, str)
                    and "has not produced a stable readback yet" in launched.note
                ):
                    observation = self._await_launch_hierarchy(package)
            observed = observation or self.analyze(source="hierarchy", with_ocr=False)
            if package and observed.screen.package != package:
                # A launch readback must never combine the requested package with a hierarchy
                # captured from the app we just left. Discard every speculative/cached seam and
                # take one authoritative hierarchy-only sample. If Android still reports a
                # different package, stop before creating a goal plan from impossible state.
                self._prefetch.invalidate()
                self._last_hierarchy_hash = None
                self._last_analyze_result = None
                observed = self.analyze(
                    source="hierarchy",
                    with_ocr=False,
                    no_cache=True,
                )
                if observed.screen.package != package:
                    raise DeviceError(
                        (
                            f"launch foreground was {package}, but the authoritative hierarchy "
                            f"belongs to {observed.screen.package or 'an unknown package'}"
                        ),
                        code="launch_observation_mismatch",
                        hint=(
                            "The window may still be attaching. Re-run session start once the "
                            "requested app is settled; AUA did not create a plan from this frame."
                        ),
                    )
        except Exception:
            if emulator_started:
                from . import emulator as emulator_mod

                with contextlib.suppress(Exception):
                    emulator_mod.stop(
                        serial=self.config.device.serial,
                        cache_dir=self.config.cache.dir,
                        requested_by="session-start-rollback",
                    )
                self.close()
            raise
        finally:
            self._lease_wait_s = 0.0
        plan = self._goal_session_plan(goal, observed)
        from . import network, network_profiles
        from .session import complete_current_ui_phase_from_observation, create_session_state

        serial = observed.meta.device_serial or self.device.serial
        session_owner = getattr(self, "_lease_owner_resolved", None)
        capture_package = observed.screen.package
        capture_context_id: str | None = None
        capture_segment: int | None = None
        capture_start_order: int | None = None
        mem = self._memory
        if mem is not None:
            self._join_memory_writers(timeout_s=5.0)
            with contextlib.suppress(Exception):
                cursor = mem.load_session(serial)
                capture_context_id = cursor.active_context_id
                capture_segment = cursor.capture_segment
                capture_start_order = cursor.next_capture_order
        state = create_session_state(
            self.config.cache.dir,
            goal=goal,
            serial=serial,
            owner=session_owner,
            recommended_kind=plan.recommended_call.kind,
            recommended_cli=plan.recommended_call.cli,
            network_backup_preexisting=network.backup_path(self.config.cache.dir, serial).is_file(),
            network_profile_preexisting=network_profiles.profile_path(
                self.config.cache.dir, serial
            ).is_file(),
            emulator_started=emulator_started,
            contract=contract,
            contract_yaml=canonical_contract_yaml,
            artifact_dir=None,
            evidence=cast(Literal["none", "failures", "all"], evidence),
            junit=junit,
            capture_package=capture_package,
            capture_context_id=capture_context_id,
            capture_segment=capture_segment,
            capture_start_order=capture_start_order,
        )
        if artifacts_dir:
            from .session import update_session_state
            from .session_artifacts import SessionArtifactStore

            artifact_store = SessionArtifactStore.create(
                artifacts_dir,
                session_id=state.session_id,
                goal=goal,
                evidence=evidence,
                junit=junit,
                contract_yaml=canonical_contract_yaml,
            )
            state = update_session_state(
                self.config.cache.dir,
                state,
                artifact_dir=str(artifact_store.root.resolve()),
            )
        contract_verdict: dict[str, Any] | None = None
        if contract is not None:
            state, contract_verdict = self._complete_contract_phase_from_observation(
                state,
                observed,
            )
        else:
            state = complete_current_ui_phase_from_observation(
                self.config.cache.dir,
                state,
                observation=observed,
            )
        self._session_id = state.session_id
        # Recommend only the active checkpoint from this frame. Future phases must be planned
        # lazily from the observation that activates them; projecting a launcher frame onto every
        # later checkpoint produced stale and sometimes misleading calls.
        active_phase = next(
            (phase for phase in state.phases if phase.status != "completed"),
            None,
        )
        if active_phase is not None:
            call = self._phase_recommended_call(state, active_phase, observed)
            if call is not None:
                from .session import update_phase_recommendation

                state = update_phase_recommendation(
                    self.config.cache.dir,
                    state,
                    phase_id=active_phase.id,
                    call=call,
                )
        out = plan.model_dump(mode="json")
        from .session import phase_progress

        # Bootstrap is a routing response, not a second copy of the persisted session document.
        # Keep the current checkpoint and terse upcoming list; `session progress` exposes the full
        # durable phase record on explicit reconnect/debug requests.
        progress = phase_progress(state, compact=True)
        phase_call = progress.get("next_call")
        out.update(
            session_id=state.session_id,
            goal_hash=state.goal_hash,
            owner=state.owner,
            serial=state.serial,
            cleanup=[
                "network_restore",
                "network_profile_restore",
                *(["owned_emulator_stop"] if emulator_started else []),
            ],
            cleanup_call={
                "cli": f"aua --serial {state.serial} session finish",
                "mcp": {
                    "tool": "session_finish",
                    "arguments": {"session_id": state.session_id},
                },
                "reason": (
                    "Run this once when finished. It restores only session-owned reversible "
                    "state and returns the efficiency review; do not restore the network "
                    "separately first."
                ),
            },
            emulator_started=emulator_started,
            lease_waited_ms=self._lease_waited_ms,
            artifacts_dir=state.artifact_dir,
            goal_progress=progress,
        )
        # Session bootstrap embeds an AnalyzeResult rather than an ActionResult, so it does not
        # otherwise carry the latter's `next_actions`. Derive them from this same fresh frame:
        # manual handoff can now choose a stable selector without a redundant analyze/capability
        # call, and every numeric id is guaranteed to belong to the observation just returned.
        next_actions = self._next_actions(observed)
        if next_actions:
            out["next_actions"] = next_actions
        if installed_bundle is not None:
            # Whether bootstrap pushed a build or reused the one already there decides whether app
            # data survived, so it belongs in the session's own record rather than only in the log.
            out["app_install"] = installed_bundle
        if contract_verdict is not None:
            out["contract_verdict"] = contract_verdict
        if isinstance(phase_call, dict):
            # The active typed checkpoint is the actual next action for both single- and
            # multi-phase goals. The whole-goal planner remains useful for candidate evidence,
            # but must never contradict a deterministic phase such as verified network status.
            out["recommended_call"] = phase_call
        elif progress.get("done") is True:
            # Structured proof on the bootstrap frame can complete a single UI goal before
            # any action is needed. Never leave the whole-goal planner's stale navigation call
            # at the top level; the only remaining lifecycle action is the existing cleanup.
            out["recommended_call"] = {
                "kind": "session_finish",
                "cli": f"aua --serial {state.serial} session finish",
                "mcp": {
                    "tool": "session_finish",
                    "arguments": {"session_id": state.session_id},
                },
                "reason": (
                    "The bootstrap observation already proves the goal. Finish the session "
                    "once to release its lifecycle and collect the review."
                ),
                "executes": True,
            }
        if active_phase is not None:
            out.update(
                self._session_policy_output(
                    state,
                    active_phase,
                    observed,
                    recommended_call=out.get("recommended_call"),
                )
            )
        return out

    def _session_policy_mode(self) -> PolicyMode:
        """Resolve the opt-in selector mode without making policy a base dependency."""
        section = getattr(self.config, "policy", None)
        if section is None or not bool(getattr(section, "enabled", False)):
            return "off"
        mode = str(getattr(section, "mode", "off") or "off").strip().casefold()
        return cast("PolicyMode", mode) if mode in {"off", "shadow", "advisory"} else "off"

    @staticmethod
    def _policy_selector_arguments(
        element: Element,
        elements: Sequence[Element],
    ) -> tuple[dict[str, str], str] | None:
        """Return one reusable, privacy-filtered selector and its safe display label."""
        selector = recorded_selector(element, elements=elements)
        by = selector.get("by")
        redacted = redact_label(element)
        if redacted == "<redacted>":
            return None
        # A stable id can make the *action* reusable even when its adjacent copy is volatile.
        # For policy selection we additionally need safe semantic evidence, so withhold the
        # whole candidate when the durable-selector filter refused non-empty copy/description.
        if by == "id" and (
            (bool((element.text or "").strip()) and not selector.get("label"))
            or (bool((element.content_desc or "").strip()) and not selector.get("content_desc"))
        ):
            return None
        if by == "id" and selector.get("resource_id"):
            value = str(selector["resource_id"])
            args = {"rid": value}
        elif by == "desc" and selector.get("content_desc"):
            value = str(selector["content_desc"])
            args = {"desc": value}
        elif by == "text" and selector.get("label"):
            value = str(selector["label"])
            args = {"text": value}
        else:
            return None

        # `recorded_selector` already refuses secrets, PII, typed values, dynamic values, and
        # ambiguous selectors. `redact_label` is an independent final check, but its result is
        # used only when the stricter durable-selector filter preserved the same copy. Dynamic
        # copy beside a safe resource id must not sneak back into the prompt through prose.
        persisted_label = selector.get("label") or selector.get("content_desc")
        safe_label = (
            str(persisted_label)
            if persisted_label and redacted not in {None, "<redacted>"}
            else value
        )
        return args, safe_label

    def _policy_tap_candidates(
        self,
        state: Any,
        phase: Any,
        observation: AnalyzeResult,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Compile guard-owned exact calls from one fresh frame.

        The optional model never sees the hierarchy and never authors arguments. It receives
        only enabled app controls with a unique durable selector, locally-proved semantic
        relevance, and a clean destructive-risk check. Toggle/input/system controls stay out of
        this first integration because a generic tap can mutate state without proving progress.
        """
        from .policy import PolicyCandidate
        from .session import _goal_terms, _match_score

        fingerprint = observation.meta.fingerprint
        package = observation.screen.package
        # A compound goal may enumerate the visible alternatives after naming the requested
        # destination (``Open History from these choices: Grammar, History, Physics``).  Those
        # alternatives are context, not evidence that every row is goal-relevant.  Reuse the
        # same destination-object extraction as arrival proof so candidate recall and model
        # selection are both conditioned on the requested target only.
        policy_goal = " ".join(arrival_destination_terms(phase.objective)) or phase.objective
        goal_terms = set(_goal_terms(policy_goal))
        ranked: list[tuple[int, str, Any]] = []
        max_candidates = max(1, int(getattr(self.config.policy, "max_candidates", 4)))
        stage_counts = {
            "elements": len(observation.elements),
            "enabled_clickable": 0,
            "safe_control": 0,
            "stable_selector": 0,
            "non_destructive": 0,
            "target_matched": 0,
            "offered": 0,
        }

        for element in observation.elements:
            if not element.clickable or element.enabled is False:
                continue
            stage_counts["enabled_clickable"] += 1
            if element.checkable or element.selected is True or element.window not in {None, "app"}:
                continue
            element_type = element.type.casefold()
            if "edittext" in element_type or "textfield" in element_type or "input" in element_type:
                continue
            stage_counts["safe_control"] += 1

            selector_value = self._policy_selector_arguments(element, observation.elements)
            if selector_value is None:
                continue
            stage_counts["stable_selector"] += 1
            arguments, safe_label = selector_value

            rid_label = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])",
                " ",
                (element.resource_id or "").rsplit("/", 1)[-1],
            ).replace("_", " ")
            # Raw copy is used only by deterministic in-process classification. It is never
            # placed in the PolicyContext, response, journal, or model prompt.
            risk_label = " ".join(
                value for value in (element.text, element.content_desc, rid_label) if value
            )
            if is_destructive_step(
                RouteStep(kind="tap", label=risk_label),
                self.config.memory.destructive_labels,
            ):
                continue
            stage_counts["non_destructive"] += 1

            semantic_label = " ".join(value for value in (safe_label, rid_label) if value)
            matched_terms = goal_terms & set(_goal_terms(semantic_label))
            if not matched_terms or matched_terms <= _GENERIC_MANUAL_MATCH_TERMS:
                continue
            stage_counts["target_matched"] += 1
            score = _match_score(policy_goal, semantic_label)
            call = {"tool": "tap_and_analyze", "arguments": arguments}
            material = json.dumps(
                {
                    "session_id": state.session_id,
                    "phase_id": phase.id,
                    "fingerprint": fingerprint,
                    "package": package,
                    "call": call,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            candidate = PolicyCandidate(
                # Dense opaque IDs are assigned after the bounded candidate set is known.
                # FunctionGemma v3 was trained on exactly this 0..N-1 ID vocabulary.
                candidate_id=0,
                call=call,
                model_arguments=arguments,
                purpose=f"Tap the current-frame {safe_label!r} control and observe the result.",
                proof="The exact call returns a folded post-action observation.",
                safe=True,
                authorized=True,
                redundant=False,
                session_id=state.session_id,
                phase=phase.id,
                observation_fingerprint=fingerprint,
                package=package,
            )
            ranked.append((score, material, candidate))

        # Keep the most goal-relevant guarded calls. Assign a dense hidden permutation of
        # 0..N-1 because those are the only opaque IDs in the frozen adapter's vocabulary;
        # use a separate stable permutation for display order so neither position nor source
        # hierarchy order leaks a preference.
        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = ranked[:max_candidates]
        id_rows = sorted(
            selected,
            key=lambda row: hashlib.sha256(f"policy-id\0{row[1]}".encode()).hexdigest(),
        )
        ids = {row[1]: candidate_id for candidate_id, row in enumerate(id_rows)}
        ordered = sorted(
            selected,
            key=lambda row: hashlib.sha256(f"policy-order\0{row[1]}".encode()).hexdigest(),
        )
        candidates = [dataclass_replace(row[2], candidate_id=ids[row[1]]) for row in ordered]
        stage_counts["offered"] = len(candidates)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "schema_version": 1,
                    "target_term_count": len(goal_terms),
                    "stages": stage_counts,
                }
            )
        return candidates

    @staticmethod
    def _policy_selection_goal(objective: str, candidates: Sequence[Any]) -> str:
        """Keep safe disambiguating evidence without reintroducing alternative-list bias.

        Candidate filtering intentionally uses only the requested destination.  The selector can
        still need a qualifier when several safe rows share that destination (for example four
        ``History`` rows with different summaries).  Preserve only objective terms that also
        occur in privacy-screened candidate prose, after removing explicit alternative lists.
        User text, typed values, and unrelated private vocabulary therefore cannot enter the
        local-model prompt through this seam.
        """

        target_terms = arrival_destination_terms(objective)
        target_goal = " ".join(target_terms) or objective
        cleaned = re.sub(
            r"\s+(?:from|among)\s+(?:the\s+)?(?:these\s+)?(?:visible\s+)?"
            r"(?:[A-Za-z]+\s+)?(?:destinations|choices)\s*:\s*.*$",
            "",
            objective,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+(?:rather\s+than|choices\s+are|the\s+alternatives\s+are|"
            r"available\s+destinations\s+are)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        from .session import _goal_terms

        candidate_terms: set[str] = set()
        for candidate in candidates:
            candidate_terms.update(_goal_terms(str(getattr(candidate, "purpose", ""))))
        generic = {
            "action",
            "call",
            "control",
            "current",
            "exact",
            "folded",
            "frame",
            "observation",
            "observe",
            "post",
            "result",
            "returns",
            "tap",
        }
        target_set = set(target_terms)
        qualifiers: list[str] = []
        for term in _goal_terms(cleaned):
            if (
                term in candidate_terms
                and term not in target_set
                and term not in generic
                and term not in qualifiers
            ):
                qualifiers.append(term)
        if not qualifiers:
            return target_goal
        return f"Requested destination: {target_goal}. Matching evidence: {' '.join(qualifiers)}."

    @staticmethod
    def _policy_suggestion(candidate: Any) -> dict[str, Any]:
        """Render a selected guard-owned call for advisory mode only."""
        call: dict[str, Any] = {
            "tool": str(candidate.call["tool"]),
            "arguments": dict(candidate.call["arguments"]),
        }
        arguments = call["arguments"]
        if "rid" in arguments:
            cli = f"aua tap-and-analyze --rid {shlex.quote(str(arguments['rid']))}"
        elif "desc" in arguments:
            cli = f"aua tap-and-analyze --desc {shlex.quote(str(arguments['desc']))}"
        else:
            cli = f"aua tap-and-analyze --text {shlex.quote(str(arguments['text']))}"
        return {
            "kind": "policy_advisory",
            "candidate_id": candidate.candidate_id,
            "cli": cli,
            "mcp": call,
            "reason": (
                "The optional local policy selected this guard-approved current-frame call. "
                "AUA has not executed it and has not replaced the deterministic recommendation."
            ),
            "executes": True,
        }

    def _policy_context_is_current(
        self,
        state: Any,
        phase: Any,
        observation: AnalyzeResult,
        candidate: Any,
    ) -> tuple[bool, str | None]:
        """Revalidate session, phase, and frame provenance after model latency."""
        fingerprint = observation.meta.fingerprint
        package = observation.screen.package
        try:
            current_state = self._session_state(state.session_id)
        except Exception as exc:  # pragma: no cover - defensive, surfaced as policy metadata
            return False, f"session revalidation failed: {type(exc).__name__}"
        current_phase = next(
            (item for item in current_state.phases if item.status != "completed"),
            None,
        )
        if current_state.session_id != state.session_id or current_state.finished_ms is not None:
            return False, "the goal session changed or finished during policy evaluation"
        if current_phase is None or current_phase.id != phase.id:
            return False, "the active goal phase changed during policy evaluation"
        if candidate.observation_fingerprint != fingerprint or candidate.package != package:
            return False, "the selected candidate is not bound to the supplied observation"
        if observation.meta.device_serial not in {None, current_state.serial}:
            return False, "the supplied observation belongs to another device"

        # A warm Engine can observe a newer frame while a slow policy call is in flight. Never
        # expose a selector from the older frame in that case. A short-lived Engine may have no
        # cache here; session/phase/candidate provenance still provides the binding.
        latest = self._last_analyze_result
        if latest is not None:
            latest_fingerprint = latest.meta.fingerprint
            if latest_fingerprint and latest_fingerprint != fingerprint:
                return False, "a newer observation replaced the policy input frame"
            if latest.screen.package != package:
                return False, "the foreground package changed during policy evaluation"
        return True, None

    def _session_policy_output(
        self,
        state: Any,
        phase: Any,
        observation: AnalyzeResult | None,
        *,
        recommended_call: Any,
    ) -> dict[str, Any]:
        """Evaluate the optional policy as a non-fatal, non-executing side channel."""
        mode = self._session_policy_mode()
        if mode == "off":
            return {}

        deterministic_kind = (
            recommended_call.get("kind") if isinstance(recommended_call, dict) else None
        )
        if phase.kind != "verify" or deterministic_kind not in {
            "manual_action",
            "manual_observation",
        }:
            audit: dict[str, Any] = {
                "mode": mode,
                "status": "skipped_deterministic",
                "provider": None,
                "model_used": False,
                "candidate_count": 0,
                "eligible_candidate_ids": [],
                "error": None,
            }
            return {"policy": audit}
        if (
            observation is None
            or not observation.meta.fingerprint
            or bool(observation.meta.stale_risk)
            or not observation.screen.package
        ):
            audit = {
                "mode": mode,
                "status": "skipped_unbound_observation",
                "provider": None,
                "model_used": False,
                "candidate_count": 0,
                "eligible_candidate_ids": [],
                "error": "policy requires a fresh fingerprinted observation",
            }
            return {"policy": audit}

        try:
            from .policy import PolicyContext, evaluate_policy, guard_candidates

            compiler_audit: dict[str, Any] = {}
            candidates = self._policy_tap_candidates(
                state,
                phase,
                observation,
                diagnostics=compiler_audit,
            )
            deterministic_mcp = (
                recommended_call.get("mcp") if isinstance(recommended_call, dict) else None
            )
            compiler_audit["recommended_call_offered"] = (
                any(candidate.trusted_call() == deterministic_mcp for candidate in candidates)
                if isinstance(deterministic_mcp, dict)
                else None
            )
            policy_goal = self._policy_selection_goal(phase.objective, candidates)
            context = PolicyContext(
                goal=policy_goal,
                phase=phase.id,
                session_id=state.session_id,
                candidates=tuple(candidates),
                observation={
                    "fresh": True,
                    **(
                        {"known_screen": observation.meta.known_screen}
                        if observation.meta.known_screen
                        else {}
                    ),
                },
                constraints=(
                    "Select only a supplied guard-approved candidate.",
                    "Do not invent or execute a call.",
                ),
                recent_outcomes=(
                    "session_active=true",
                    "outcome=known",
                    "goal_checkpoint_reached=false",
                ),
                observation_fingerprint=observation.meta.fingerprint,
                package=observation.screen.package,
            )
            max_candidates = max(1, int(getattr(self.config.policy, "max_candidates", 4)))
            eligible = guard_candidates(context, max_candidates=max_candidates)
            selector: PolicySelector | None = None
            if len(eligible) > 1:
                chain = self.factory.build_chain("policy")
                selector = cast(
                    "PolicySelector | None",
                    chain.providers[0] if chain.providers else None,
                )
            decision = evaluate_policy(
                context,
                selector,
                mode=mode,
                max_candidates=max_candidates,
            )
            audit = decision.as_json()
            audit["compiler"] = compiler_audit
            # Exact calls belong only in the separate advisory field, never in shadow/audit.
            audit.pop("recommended_call", None)
            selected = decision.selected_candidate
            suggestion = None
            if selected is not None:
                current, stale_reason = self._policy_context_is_current(
                    state,
                    phase,
                    observation,
                    selected,
                )
                if not current:
                    audit["status"] = "rejected_stale_context"
                    audit["error"] = stale_reason
                    audit.pop("selected_candidate_id", None)
                elif mode == "advisory" and decision.model_used:
                    suggestion = self._policy_suggestion(selected)
            out: dict[str, Any] = {"policy": audit}
            if suggestion is not None:
                out["policy_suggestion"] = suggestion
            return out
        except Exception as exc:  # policy is optional and must never break a UI result
            logger.warning("optional policy evaluation failed: %s", exc)
            audit = {
                "mode": mode,
                "status": "error",
                "provider": None,
                "model_used": False,
                "candidate_count": 0,
                "eligible_candidate_ids": [],
                "error": f"policy evaluation failed: {type(exc).__name__}",
            }
            return {"policy": audit}

    def _phase_recommended_call(
        self,
        state: Any,
        phase: Any,
        observation: AnalyzeResult | None,
        *,
        avoid_deeplinks: bool = False,
    ) -> dict[str, Any] | None:
        """Return one safe exact call for a phase, using only the supplied fresh frame."""
        avoid_deeplinks = avoid_deeplinks or any(
            re.search(r"\bdeep[ -]?links?\b", constraint, flags=re.IGNORECASE)
            for constraint in getattr(phase, "constraints", [])
        )
        if phase.kind == "environment":
            if getattr(phase, "satisfaction", None) == "verified_network_status":
                return {
                    "kind": "network_status",
                    "cli": "aua network status --verify",
                    "mcp": {"tool": "network_status", "arguments": {"verify": True}},
                    "reason": (
                        "This phase records the verified current network transport before "
                        "any reversible environment change."
                    ),
                    "executes": False,
                }
            return {
                "kind": "network_offline",
                "cli": "aua network offline --verify",
                "mcp": {"tool": "network_offline", "arguments": {"verify": True}},
                "reason": "This phase requires verified reversible network isolation.",
                "executes": True,
            }
        if phase.kind == "cleanup" and getattr(phase, "satisfaction", None) != "fresh_assertions":
            return {
                "kind": "session_finish",
                "cli": f"aua --serial {state.serial} session finish",
                "mcp": {
                    "tool": "session_finish",
                    "arguments": {"session_id": state.session_id},
                },
                "reason": "This is the final phase; restore only session-owned reversible state.",
                "executes": True,
            }
        if observation is None:
            if phase.recommended_call is not None:
                return phase.recommended_call
            # Deterministic host/device transitions (notably verified network isolation) do not
            # carry an Android hierarchy. Once one activates a UI checkpoint, return the one
            # read-only call that will both observe that frame and lazily plan the phase. A null
            # next_call strands a fresh agent; replaying the pre-transition frame risks stale ids.
            return {
                "kind": "refresh_observation",
                "cli": f"aua --serial {state.serial} analyze --source hierarchy",
                "mcp": {"tool": "analyze_screen", "arguments": {"source": "hierarchy"}},
                "reason": (
                    "The active UI phase began after a non-UI transition. Read one fresh "
                    "hierarchy frame; its goal_progress will contain the exact next action."
                ),
                "executes": False,
            }

        # Never turn an explicitly caveated post-action frame into another mutation.  A stale
        # hierarchy can still contain perfectly plausible controls from the screen we just left;
        # the only safe next step is one authoritative read that produces a new fingerprint.
        if observation.meta.stale_risk:
            return {
                "kind": "refresh_observation",
                "cli": f"aua --serial {state.serial} analyze --source hierarchy --no-cache",
                "mcp": {
                    "tool": "analyze_screen",
                    "arguments": {"source": "hierarchy", "no_cache": True},
                },
                "reason": (
                    "This frame is explicitly marked stale-risk and cannot authorize another "
                    "action. Read one uncached hierarchy frame before replanning."
                ),
                "executes": False,
            }

        # Loading is not a navigation failure.  When the hierarchy names the loading marker,
        # wait for that evidence to disappear; otherwise wait for one tree change.  Both calls
        # return the resulting analyzed frame and are bounded, so the agent does not busy-loop or
        # guess at a control while content is attaching.
        if self._observation_is_loading(observation):
            loading_predicate: str | None = None
            for element in observation.elements:
                label = " ".join(
                    value for value in (element.text, element.content_desc) if value
                ).strip()
                if re.search(r"\bloading\b", label, re.IGNORECASE):
                    loading_predicate = "!text:Loading"
                    break
                if re.search(r"\bplease wait\b", label, re.IGNORECASE):
                    loading_predicate = "!text:Please wait"
                    break
            if loading_predicate is not None:
                return {
                    "kind": "await_loading",
                    "cli": (
                        f"aua --serial {state.serial} await-and-analyze "
                        f"{shlex.quote(loading_predicate)} --timeout-ms 15000 --poll-ms 200 "
                        "--ignore-case --observe"
                    ),
                    "mcp": {
                        "tool": "await_and_analyze",
                        "arguments": {
                            "predicate": loading_predicate,
                            "timeout_ms": 15000,
                            "poll_ms": 200,
                            "ignore_case": True,
                        },
                    },
                    "reason": (
                        "The current hierarchy explicitly reports loading. Wait once for that "
                        "marker to disappear and reuse the returned analyzed frame."
                    ),
                    "executes": False,
                }
            return {
                "kind": "wait_for_change",
                "cli": (
                    f"aua --serial {state.serial} wait-and-analyze --changed "
                    "--timeout-ms 15000 --interval 150 --observe"
                ),
                "mcp": {
                    "tool": "wait_changed_and_analyze",
                    "arguments": {"timeout_ms": 15000, "interval_ms": 150},
                },
                "reason": (
                    "The current frame contains an unlabeled loading/progress state. Wait once "
                    "for the hierarchy to change and reuse the returned analyzed frame."
                ),
                "executes": False,
            }

        # Offline is already its own deterministic phase. Removing that word here prevents the
        # UI checkpoint planner from recommending network isolation again after it completed.
        ui_goal = re.sub(r"\boffline\b|\bairplane mode\b", " ", phase.objective, flags=re.I)
        ui_goal = " ".join(ui_goal.split()) or phase.objective
        from .session import _goal_terms, _match_score

        goal_terms = set(_goal_terms(ui_goal))
        destination_term_list = arrival_destination_terms(ui_goal)
        destination_terms = set(destination_term_list)
        target_goal = " ".join(destination_term_list) or ui_goal

        # A stable mapped screen plus an exact multi-word title from the requested destination
        # is arrival evidence, not permission to descend into a child row that happens to share
        # one word. This mattered for a destination titled "Network & internet": the old fallback
        # immediately proposed its nested "Internet" row after the requested screen had arrived.
        visible_arrival = next(
            (
                label
                for element in observation.elements
                if not element.clickable
                and (label := (element.text or element.content_desc or "").strip())
                and len(set(_goal_terms(label)) & destination_terms) >= 2
                and label.casefold() in ui_goal.casefold()
            ),
            None,
        )
        if observation.meta.known_screen and visible_arrival:
            preview = re.search(
                r"\bpreview\s+(?:(?:the|a)\s+)?(?:(?:flow)\s+)?"
                r"(?P<name>[A-Za-z0-9_.-]+)(?:\s+--last\s+(?P<last>[0-9]+))?",
                ui_goal,
                flags=re.IGNORECASE,
            )
            if preview is not None:
                name = preview.group("name")
                last = int(preview.group("last") or 12)
                return {
                    "kind": "flow_save_preview",
                    "cli": f"aua flow save {shlex.quote(name)} --last {last}",
                    "mcp": {"tool": "flow_save", "arguments": {"name": name, "last": last}},
                    "reason": (
                        f"The current mapped screen visibly matches {visible_arrival!r}; "
                        "continue with the requested non-writing flow preview instead of "
                        "navigating into a weaker one-word match."
                    ),
                    "executes": False,
                    "arrival": {
                        "status": "observed",
                        "known_screen": observation.meta.known_screen,
                        "visible_title": visible_arrival,
                        **(
                            {"fingerprint": observation.meta.fingerprint}
                            if observation.meta.fingerprint
                            else {}
                        ),
                    },
                }
            return {
                "kind": "arrived",
                "cli": "No call: reuse this result's observation; the destination is visible",
                "mcp": None,
                "reason": (
                    f"The current mapped screen visibly matches {visible_arrival!r}; do not "
                    "navigate into a weaker one-word child match."
                ),
                "executes": False,
                "arrival": {
                    "status": "observed",
                    "known_screen": observation.meta.known_screen,
                    "visible_title": visible_arrival,
                    **(
                        {"fingerprint": observation.meta.fingerprint}
                        if observation.meta.fingerprint
                        else {}
                    ),
                },
            }

        # Only after current-frame evidence has been considered may an older remembered route,
        # flow, or shortcut become the next call. This prevents a dubious child route from
        # outranking stronger visible evidence on the exact frame the caller already has.
        plan = self._goal_session_plan(ui_goal, observation)
        if plan.recommended_call.kind not in {"network_offline", "map_find"} and not (
            avoid_deeplinks and plan.recommended_call.kind.startswith("deeplink")
        ):
            return plan.recommended_call.model_dump(mode="json")
        ranked: list[tuple[int, Any]] = []
        for element in observation.elements:
            if not element.clickable or element.enabled is False:
                continue
            resource_label = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])",
                " ",
                (element.resource_id or "").rsplit("/", 1)[-1],
            ).replace("_", " ")
            label = " ".join(
                value for value in (element.text, element.content_desc, resource_label) if value
            )
            # A configured destructive control is never an execution recommendation. A bare
            # one-token control sharing only one word with a longer goal is weak evidence too.
            # The same applies to a multi-word control whose sole overlap is generic UI context
            # (for example, "Search Settings" matching only "settings"). Keep it visible in the
            # observation instead of turning it into an execution call.
            if is_destructive_step(
                RouteStep(kind="tap", label=label),
                self.config.memory.destructive_labels,
            ):
                continue
            semantic_label = element.text or element.content_desc or resource_label
            control_terms = set(_goal_terms(semantic_label))
            # Alternative labels mentioned later in a compound goal must not compete with the
            # object of its navigation verb.  Use the requested destination for current-frame
            # matching and ranking, just as the optional policy compiler does.
            target_terms = destination_terms or goal_terms
            matched_terms = target_terms & control_terms
            if not matched_terms:
                continue
            exact_goal_match = target_goal.casefold().strip() in semantic_label.casefold()
            explicit_control_request = bool(
                re.search(
                    rf"\b(?:open|tap|select|choose|launch|enter|view|inspect)\s+"
                    rf"(?:the\s+)?{re.escape(semantic_label)}\b",
                    ui_goal,
                    flags=re.IGNORECASE,
                )
            )
            weak_one_token = (
                len(matched_terms) == 1
                and len(target_terms) > 1
                and (len(control_terms) == 1 or matched_terms <= _GENERIC_MANUAL_MATCH_TERMS)
                and not exact_goal_match
                and not explicit_control_request
            )
            if weak_one_token:
                continue
            score = _match_score(target_goal, label)
            ranked.append((score, element))
        if ranked:
            _score, element = max(ranked, key=lambda item: (item[0], -item[1].id))
            mcp_arguments: dict[str, Any]
            if element.resource_id:
                rid = element.resource_id.rsplit("/", 1)[-1]
                if len(match_selector(observation.elements, rid=rid)) == 1:
                    cli = f"aua tap-and-analyze --rid {shlex.quote(rid)}"
                    mcp_arguments = {"rid": rid}
                else:
                    cli = f"aua tap-and-analyze {element.id}"
                    mcp_arguments = {"id": element.id}
            elif (
                element.content_desc
                and len(match_selector(observation.elements, desc=element.content_desc)) == 1
            ):
                cli = f"aua tap-and-analyze --desc {shlex.quote(element.content_desc)}"
                mcp_arguments = {"desc": element.content_desc}
            elif element.text and len(match_selector(observation.elements, text=element.text)) == 1:
                cli = f"aua tap-and-analyze --text {shlex.quote(element.text)}"
                mcp_arguments = {"text": element.text}
            else:
                cli = f"aua tap-and-analyze {element.id}"
                mcp_arguments = {"id": element.id}
            return {
                "kind": "manual_action",
                "cli": cli,
                "mcp": {"tool": "tap_and_analyze", "arguments": mcp_arguments},
                "reason": (
                    f"The current frame has one goal-relevant enabled control: "
                    f"{(element.text or element.content_desc or element.resource_id or element.id)!r}."
                ),
                "executes": True,
            }

        # A target may simply be below the fold.  An app-owned accessibility node that explicitly
        # reports scrollable=true is stronger evidence than another analyze, but it does not prove
        # which hidden row exists.  Move exactly one page and let the folded observation replan.
        if any(
            element.scrollable is True
            and element.enabled is not False
            and element.window in {None, "app"}
            for element in observation.elements
        ):
            return {
                "kind": "scroll_action",
                "cli": (
                    f"aua --serial {state.serial} scroll-and-analyze up --pages 1 --percent 70"
                ),
                "mcp": {
                    "tool": "scroll_and_analyze",
                    "arguments": {"direction": "up", "percent": 70},
                },
                "reason": (
                    "No goal-labelled control is visible, but the app exposes a scrollable "
                    "container. Scroll one page and replan from the returned analyzed frame."
                ),
                "executes": True,
            }
        return {
            "kind": "manual_observation",
            "cli": "No call: inspect this result's next_actions and choose deliberately",
            "mcp": None,
            "reason": (
                "No verified route, matching flow, or unambiguous goal-labelled control is "
                "available on this frame. The result already includes the reusable observation "
                "and its available next_actions; another capabilities/analyze call would add no evidence."
            ),
            "executes": False,
        }

    def session_mark_phase(
        self,
        phase_id: str,
        evidence: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Acknowledge current-phase evidence without adding a dedicated device read."""
        from .session import mark_phase_complete, phase_progress

        state = self._session_state(session_id)
        try:
            state = mark_phase_complete(
                self.config.cache.dir,
                state,
                phase_id=phase_id,
                evidence=evidence,
            )
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        return {"ok": True, "goal_progress": phase_progress(state)}

    def _complete_contract_phase_from_observation(
        self,
        state: Any,
        observation: AnalyzeResult,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Prove at most one authored checkpoint from one exact settled frame."""

        if state.contract is None:
            return state, None
        current = next((phase for phase in state.phases if phase.status != "completed"), None)
        if current is None:
            return state, None
        if observation.meta.stale_risk:
            return state, {
                "checkpoint_id": current.id,
                "ok": False,
                "code": "stale_observation",
                "detail": observation.meta.stale_risk,
            }
        fingerprint = observation.meta.fingerprint
        if not fingerprint:
            return state, {
                "checkpoint_id": current.id,
                "ok": False,
                "code": "missing_fingerprint",
                "detail": "contract proof requires a fingerprinted settled observation",
            }

        from .assertions import evaluate_assertion_step

        results: list[dict[str, Any]] = []
        for index, assertion in enumerate(current.assertions):
            verdict = evaluate_assertion_step(assertion, observation.elements)
            results.append(
                {
                    "index": index,
                    "kind": assertion.kind,
                    "ok": verdict.ok,
                    "detail": verdict.detail,
                }
            )
        diagnostics = {
            "checkpoint_id": current.id,
            "ok": bool(results) and all(item["ok"] for item in results),
            "assertions": results,
            "fingerprint": fingerprint,
        }
        if not diagnostics["ok"]:
            diagnostics["code"] = "contract_assertions_failed"
            return state, diagnostics

        capture_order: int | None = None
        mem = self._memory
        if mem is not None:
            self._join_memory_writers(timeout_s=5.0)
            with contextlib.suppress(Exception):
                cursor = mem.load_session(state.serial)
                matching = [
                    step.capture_order
                    for step in cursor.recent
                    if step.capture_order is not None
                    and (
                        state.capture_segment is None
                        or step.capture_segment == state.capture_segment
                    )
                    and (
                        state.capture_start_order is None
                        or step.capture_order >= state.capture_start_order
                    )
                ]
                capture_order = max(matching) if matching else None

        from .session import ObservationProvenance, PhaseProof, mark_phase_complete
        from .session_artifacts import observation_evidence_id

        evidence_id = observation_evidence_id(
            state.session_id,
            observation.model_dump(mode="json"),
        )
        proof = PhaseProof(
            source="contract_assertions",
            command="contract_assertions",
            verified=True,
            observation=ObservationProvenance(
                fingerprint=fingerprint,
                source=observation.screen.source.value,
                via=observation.meta.via,
                device_serial=observation.meta.device_serial or state.serial,
                package=observation.screen.package or "unknown",
            ),
            evidence_id=evidence_id,
            assertions_verified=len(results),
            capture_order=capture_order,
        )
        try:
            updated = mark_phase_complete(
                self.config.cache.dir,
                state,
                phase_id=current.id,
                evidence=f"all {len(results)} authored assertions passed on {fingerprint}",
                _proof=proof,
            )
        except ValueError as exc:
            diagnostics.update(ok=False, code="contract_proof_rejected", detail=str(exc))
            return state, diagnostics
        diagnostics["evidence_id"] = evidence_id
        diagnostics["capture_order"] = capture_order
        return updated, diagnostics

    def session_progress(
        self,
        session_id: str | None = None,
        *,
        observation: AnalyzeResult | None = None,
        _avoid_deeplinks: bool = False,
    ) -> dict[str, Any]:
        """Return and, when possible, refresh the current phase's one exact next call."""
        from .session import (
            complete_current_ui_phase_from_observation,
            phase_progress,
            update_phase_recommendation,
        )

        state = self._session_state(session_id)
        if state.finished_ms is not None:
            # A terminated session is immutable. Do not run the route planner or manufacture a
            # nested recommendation that phase_progress will then have to hide.
            return {"ok": True, "goal_progress": phase_progress(state)}
        contract_verdict: dict[str, Any] | None = None
        if observation is not None:
            if state.contract is not None:
                state, contract_verdict = self._complete_contract_phase_from_observation(
                    state,
                    observation,
                )
            else:
                state = complete_current_ui_phase_from_observation(
                    self.config.cache.dir,
                    state,
                    observation=observation,
                )
        current = next((phase for phase in state.phases if phase.status != "completed"), None)
        call: dict[str, Any] | None = None
        if current is not None:
            call = self._phase_recommended_call(
                state,
                current,
                observation,
                avoid_deeplinks=_avoid_deeplinks,
            )
            if call is not None and call != current.recommended_call:
                state = update_phase_recommendation(
                    self.config.cache.dir,
                    state,
                    phase_id=current.id,
                    call=call,
                )
        out: dict[str, Any] = {"ok": True, "goal_progress": phase_progress(state)}
        if contract_verdict is not None:
            out["contract_verdict"] = contract_verdict
        if current is not None:
            out.update(
                self._session_policy_output(
                    state,
                    current,
                    observation,
                    recommended_call=call or current.recommended_call,
                )
            )
        return out

    def _session_state(self, session_id: str | None = None) -> Any:
        from .session import load_session_state

        resolved = session_id or getattr(self, "_session_id", None)
        state = load_session_state(self.config.cache.dir, session_id=resolved) if resolved else None
        if state is None:
            serial = (
                getattr(self, "_lease_serial", None)
                or self.config.device.serial
                or self.device.serial
            )
            state = load_session_state(
                self.config.cache.dir,
                serial=serial,
                owner=getattr(self, "_lease_owner_resolved", None),
            )
        if state is None:
            raise UsageError(
                "no active AUA goal session",
                hint='Start one with `aua session start --goal "<goal>"`.',
            )
        self._session_id = state.session_id
        return state

    def session_review(self, session_id: str | None = None) -> dict[str, Any]:
        """Return owner-isolated call efficiency and concrete next-run improvements."""
        from . import journal as journal_mod
        from .session import review_session_events

        state = self._session_state(session_id)
        events = journal_mod.read_since(
            self.config.cache.dir,
            state.serial,
            since_ms=state.started_ms,
            limit=2_000,
        )
        return review_session_events(state, events)

    def _session_candidate(self, state: Any, *, name: str) -> Any:
        """Build one unverified candidate from this contract's correlated action window."""

        if state.contract is None:
            raise UsageError("candidate flows require an authored session contract")
        incomplete = [phase.id for phase in state.phases if phase.status != "completed"]
        if incomplete:
            raise UsageError(
                "candidate flow requires every contract checkpoint to be complete",
                hint="incomplete: " + ", ".join(incomplete),
            )
        if (
            state.capture_package is None
            or state.capture_segment is None
            or state.capture_start_order is None
        ):
            raise UsageError(
                "session capture provenance is incomplete; candidate cannot be trusted"
            )
        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled; the session action path was not recorded")
        self._join_memory_writers(timeout_s=5.0)
        cursor = mem.load_session(state.serial)
        if cursor.capture_segment != state.capture_segment:
            raise UsageError(
                "the session crossed an app/context capture boundary",
                hint="repeat the journey in one package/context segment before promotion",
            )
        checkpoints = [
            {
                "id": phase.id,
                "capture_order": phase.proof.capture_order if phase.proof else None,
                "assertions": phase.assertions,
            }
            for phase in state.phases
        ]
        from .candidate_flows import build_candidate_flow

        return build_candidate_flow(
            name=name,
            app=state.capture_package,
            context_id=state.capture_context_id,
            recent=cursor.recent,
            start_capture_order=state.capture_start_order,
            capture_segment=state.capture_segment,
            checkpoints=checkpoints,
        )

    def session_candidate_flow(
        self,
        name: str,
        *,
        session_id: str | None = None,
        reset_flow: str | None = None,
        replay: bool = False,
        save: bool = False,
    ) -> dict[str, Any]:
        """Preview, replay, and only then promote a verified session action path."""

        if not name.strip():
            raise UsageError("candidate flow needs a non-empty name")
        if save:
            replay = True
        if replay and not reset_flow:
            raise UsageError(
                "candidate replay needs an explicit reset flow",
                hint="Pass --reset-flow NAME; AUA will not guess or mutate setup state.",
            )
        state = self._session_state(session_id)
        candidate = self._session_candidate(state, name=name)
        out: dict[str, Any] = {
            "ok": True,
            "name": candidate.flow.name,
            "yaml": candidate.yaml,
            "source_steps": candidate.source_steps,
            "checkpoint_ids": list(candidate.checkpoint_ids),
            "replayed": False,
            "saved": False,
        }
        if state.artifact_dir:
            from .atomic import atomic_write_text

            candidate_path = Path(state.artifact_dir) / "candidate-flow.yaml"
            atomic_write_text(candidate_path, candidate.yaml)
            out["artifact"] = str(candidate_path)
        if not replay:
            return out

        assert reset_flow is not None  # replay validation above requires it
        reset_path = Path(reset_flow).expanduser()
        reset = (
            self.flow_run(file=str(reset_path.resolve()))
            if reset_path.is_file()
            else self.flow_run(name=reset_flow)
        )
        out["reset"] = reset
        if reset.get("ok") is not True:
            out.update(ok=False, code="candidate_reset_failed")
            return out
        replayed = self.flow_run(yaml=candidate.yaml)
        out["replay"] = replayed
        out["replayed"] = replayed.get("ok") is True
        if replayed.get("ok") is not True:
            out.update(ok=False, code="candidate_replay_failed")
            return out
        if save:
            from .flows import FlowStore

            path = FlowStore(self.config.memory).save(candidate.flow, force=False)
            out["saved"] = True
            out["path"] = str(path)
        return out

    def session_finish(
        self,
        session_id: str | None = None,
        *,
        allow_incomplete: bool = False,
    ) -> dict[str, Any]:
        """Restore only reversible state created after this session started, then review it."""
        from . import network, network_profiles
        from .session import finish_session_state, phase_progress

        state = self._session_state(session_id)
        contract_verdict: dict[str, Any] | None = None
        contract_observation: AnalyzeResult | None = None
        if state.contract is not None and any(
            phase.status != "completed" for phase in state.phases
        ):
            fresh = self.analyze(source="hierarchy", with_ocr=False, no_cache=True)
            contract_observation = fresh
            state, contract_verdict = self._complete_contract_phase_from_observation(state, fresh)
            incomplete = [
                {
                    "id": phase.id,
                    "objective": phase.objective,
                    "status": phase.status,
                }
                for phase in state.phases
                if phase.status != "completed"
            ]
            if incomplete and not allow_incomplete:
                progress = phase_progress(state)
                return {
                    "ok": False,
                    "code": "contract_incomplete",
                    "session_id": state.session_id,
                    "finished": False,
                    "terminated": False,
                    "verdict": "incomplete",
                    "missing_checkpoints": incomplete,
                    "contract_verdict": contract_verdict,
                    "observation": fresh.model_dump(mode="json"),
                    "goal_progress": progress,
                    "next_call": progress.get("next_call"),
                    "cleanup": [],
                    "errors": [],
                    "hint": (
                        "The authored contract is still active. Satisfy the current checkpoint "
                        "and retry session finish, or explicitly pass --allow-incomplete."
                    ),
                }
        candidate_payload: dict[str, Any] | None = None
        if state.contract is not None and all(
            phase.status == "completed" for phase in state.phases
        ):
            try:
                candidate = self._session_candidate(
                    state,
                    name=f"session-{state.session_id[:8]}",
                )
                candidate_payload = {
                    "name": candidate.flow.name,
                    "yaml": candidate.yaml,
                    "source_steps": candidate.source_steps,
                    "checkpoint_ids": list(candidate.checkpoint_ids),
                    "verified": False,
                    "hint": "Replay with an explicit reset flow before saving.",
                }
            except UsageError as exc:
                candidate_payload = {
                    "verified": False,
                    "error": exc.to_dict().get("error"),
                }
        cleanup: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        def restore(name: str, fn: Any) -> dict[str, Any] | None:
            try:
                result = fn()
                payload = (
                    result.model_dump(mode="json") if hasattr(result, "model_dump") else result
                )
                cleanup.append(
                    {"action": name, "ok": bool(payload.get("ok", True)), "result": payload}
                )
                if not payload.get("ok", True):
                    errors.append(
                        {"action": name, "message": str(payload.get("detail") or "restore failed")}
                    )
                return payload
            except AuaError as exc:
                errors.append({"action": name, "message": exc.message})
                return None

        if (
            not state.network_profile_preexisting
            and network_profiles.profile_path(self.config.cache.dir, state.serial).is_file()
        ):
            restore("network_profile_restore", self.network_profile_restore)
        if (
            not state.network_backup_preexisting
            and network.backup_path(self.config.cache.dir, state.serial).is_file()
        ):
            restore("network_restore", self.network_restore)

        if state.emulator_started:
            from . import emulator as emulator_mod

            stopped = restore(
                "owned_emulator_stop",
                lambda: emulator_mod.stop(
                    serial=state.serial,
                    cache_dir=self.config.cache.dir,
                    requested_by="session-finish",
                ),
            )
            stopped_serials = stopped.get("stopped", []) if stopped is not None else []
            if (
                state.serial in stopped_serials
                and self._device is not None
                and self._device.serial == state.serial
            ):
                # Closing is tied to the owned stop itself, not unrelated restore errors. A
                # failed network cleanup must never leave a dead emulator cached in this Engine.
                self.close()

        if not errors:
            state = finish_session_state(self.config.cache.dir, state)
        progress = phase_progress(state)
        review = self.session_review(state.session_id)
        result = {
            "ok": not errors,
            "session_id": state.session_id,
            # ``finished`` means every requested checkpoint completed. ``terminated`` means the
            # session lifecycle/cleanup ended successfully. Keeping those distinct prevents a
            # closed session with incomplete phases from claiming both finished=true and done=false.
            "finished": not errors and bool(progress["done"]),
            "terminated": not errors,
            "goal_progress": progress,
            "cleanup": cleanup,
            "errors": errors,
            "review": review,
            "hint": (
                (
                    "session completed; only session-owned reversible state was restored"
                    if progress["done"]
                    else "session terminated and cleanup completed; unfinished goal phases remain incomplete"
                )
                if not errors
                else "cleanup is incomplete; fix the reported device access and run session finish again"
            ),
        }
        if contract_verdict is not None:
            result["contract_verdict"] = contract_verdict
        if contract_observation is not None:
            result["observation"] = contract_observation.model_dump(mode="json")
        if candidate_payload is not None:
            result["candidate_flow"] = candidate_payload
        if state.artifact_dir:
            result["artifacts_dir"] = state.artifact_dir
        return result

    def reach(
        self,
        goal: str,
        *,
        until: str | None = None,
        timeout_ms: int = 30_000,
        interval_ms: int = 300,
        allow_unsafe: bool = False,
        allow_destructive: bool = False,
        assist: bool = False,
    ) -> dict[str, Any]:
        """Use the safest known route or flow, then optionally verify arrival evidence.

        Selection is the same pure plan returned by :meth:`session_start`.  A safe verified
        goto wins, followed by a safe matching flow.  A deeplink or risky journey is never
        selected unless its exact risk class was explicitly authorized.  The initial
        observation is reused by route/flow execution instead of being immediately repeated.
        """
        if not goal.strip():
            raise UsageError("reach needs a non-empty goal")
        if until is not None:
            _parse_await_terms(until, require_positive=True)  # preflight before any action
        observation = self.analyze(source="hierarchy", with_ocr=False)
        plan = self._goal_session_plan(goal, observation)

        def authorized(candidate: Any) -> bool:
            if candidate.safe:
                return True
            codes = {risk.get("code") for risk in candidate.risks}
            # A positive caller-owned predicate is stronger than an old flow's absent arrival
            # metadata. It authorizes only that missing-proof risk; every side effect retains
            # its normal opt-in requirement.
            if until is not None:
                codes.discard("arrival_unverified")
            if (
                "required_params" in codes
                or "legacy_route" in codes
                or "arrival_unverified" in codes
                or "arrival_invalid" in codes
                or "arrival_screen_invalid" in codes
                or "nested_execution" in codes
            ):
                return False
            if "destructive" in codes and not allow_destructive:
                return False
            return not (codes - {"destructive"}) or allow_unsafe

        candidate = next(
            (
                item
                for item in plan.candidates
                if item.kind in {"arrived", "goto", "flow", "deeplink"} and authorized(item)
            ),
            None,
        )
        if candidate is None:
            return {
                "ok": False,
                "code": "navigation_unavailable",
                "goal": goal,
                "observation": observation.model_dump(mode="json"),
                "candidates": [item.model_dump(mode="json") for item in plan.candidates],
                "recommended_call": plan.recommended_call.model_dump(mode="json"),
                "warnings": plan.warnings,
            }

        navigation: dict[str, Any]
        if candidate.kind == "arrived":
            proof = candidate.evidence.get("arrival_proof")
            navigation = {
                "ok": isinstance(proof, dict),
                "arrived": isinstance(proof, dict),
                "already_there": isinstance(proof, dict),
                "target": candidate.target,
                "final_screen": observation.meta.known_screen,
                "elements": [element.compact() for element in observation.elements],
                "arrival_proof": proof,
            }
            if not isinstance(proof, dict):
                navigation.update(
                    code="arrival_unproven",
                    hint=(
                        "A mapped cursor alone is not arrival proof; inspect this observation "
                        "for a non-clickable destination title/anchor."
                    ),
                )
        elif candidate.kind == "goto":
            navigation = self.goto(
                goal,
                allow_unsafe=allow_unsafe,
                allow_destructive=allow_destructive,
                assist=assist,
                _observation=observation,
            )
        elif candidate.kind == "flow":
            navigation = self.flow_run(
                candidate.name,
                allow_destructive=allow_destructive,
                allow_unsafe=allow_unsafe,
                assist=assist,
                _observation=observation,
            )
        else:
            action = self.open_link(candidate.name, observe=True)
            landed = action.observation.meta.known_screen if action.observation else None
            expected = candidate.target
            proven = expected is not None and landed == expected
            navigation = {
                "ok": action.ok and (proven or until is not None),
                "action": action.model_dump(mode="json"),
                "expected_screen": expected,
                "final_screen": landed,
                "arrival_proven": proven,
            }
            if action.ok and not proven and until is None:
                navigation.update(
                    code="arrival_unproven",
                    hint="Intent delivery is not arrival; provide --until semantic evidence.",
                )

        out: dict[str, Any] = {
            "ok": bool(navigation.get("ok")),
            "goal": goal,
            "strategy": candidate.kind,
            "candidate": candidate.model_dump(mode="json"),
            "navigation": navigation,
        }
        if out["ok"] and until is not None:
            awaited = self.await_predicate(
                until,
                timeout_ms=timeout_ms,
                poll_ms=interval_ms,
                observe=True,
            )
            out["await"] = awaited.model_dump(mode="json")
            out["ok"] = awaited.ok
            if not awaited.ok:
                out["code"] = f"arrival_{awaited.await_outcome or 'unverified'}"
        return out

    def navigate(
        self,
        goal: str,
        *,
        max_steps: int = 12,
        allow_destructive: bool = False,
        until: str | None = None,
        save_flow: str | None = None,
    ) -> dict[str, Any]:
        """Drive to *goal* from scratch with the opt-in planner — the self-improving path.

        No prior map needed: the planner chooses each action; because those actions run
        through the normal tap/input/… methods, the journey is **recorded into memory**,
        so a later ``aua goto <that screen>`` replays it deterministically for free. Stop
        early on ``until`` text. ``save_flow`` also materializes the taken path as a flow.
        Requires ``planner.enabled`` (this command IS the explicit opt-in).
        """
        if not self.factory.is_enabled("planner"):
            raise UsageError(
                "navigate needs the planner enabled",
                hint="set `planner.enabled: true` + the model's API key (e.g. GEMINI_API_KEY)",
            )
        mem = self._memory
        serial = self.device.serial
        capture_before = mem.load_session(serial).next_capture_order if mem else None
        res = self.analyze(source="auto")  # perceive + record the starting screen
        arrived, res = self._drive_with_planner(
            goal,
            res=res,
            max_steps=max_steps,
            allow_destructive=allow_destructive,
            until=until,
        )

        def save_refusal(code: str, reason: str, hint: str) -> dict[str, Any]:
            """Report that navigation finished but its requested artifact was not trustworthy."""
            return {
                "ok": False,
                "code": code,
                "goal": goal,
                "arrived": arrived,
                "final_screen": res.meta.known_screen,
                "package": res.screen.package,
                "elements": [e.compact() for e in res.elements],
                "flow_save": {
                    "name": save_flow,
                    "saved": False,
                    "reason": reason,
                },
                "hint": hint,
            }

        flow_saved: str | None = None
        if save_flow:
            from .flows import Flow, FlowStore, recorded_step_blockers, steps_from_recent

            if mem is None:
                return save_refusal(
                    "flow_capture_memory_disabled",
                    "memory is disabled, so the planner path was not recorded",
                    "Enable memory before using `navigate --save-flow`.",
                )
            if not self._join_memory_writers(timeout_s=5.0):
                return save_refusal(
                    "flow_capture_pending",
                    "recorded-flow provenance is still being finalized",
                    "Retry `navigate --save-flow` after the current memory update completes.",
                )

            # The final observation can complete asynchronously and establish an app/context
            # boundary.  Read the journal only after that write has landed, then require every
            # selected action to belong to the finalized current segment.
            session = mem.load_session(serial)
            retained_orders = sorted(
                step.capture_order for step in session.recent if step.capture_order is not None
            )
            if (
                capture_before is not None
                and retained_orders
                and retained_orders[0] > capture_before
            ):
                return save_refusal(
                    "flow_capture_overflow",
                    "the planner journey exceeded the rolling action journal and its beginning was dropped",
                    "Capture a shorter journey (40 actions or fewer), or split it into composed flows.",
                )
            taken = [
                step
                for step in session.recent
                if capture_before is not None
                and step.capture_order is not None
                and step.capture_order >= capture_before
            ]
            if not taken:
                return save_refusal(
                    "flow_capture_empty",
                    "the finalized planner journal contains no new replayable actions",
                    "Drive at least one action, or omit --save-flow when already at the goal.",
                )
            newest = taken[-1]
            homogeneous = bool(
                newest.capture_segment is not None
                and newest.origin_package is not None
                and all(
                    step.capture_segment == newest.capture_segment
                    and step.origin_package == newest.origin_package
                    and step.context_id == newest.context_id
                    for step in taken
                )
            )
            if not homogeneous:
                return save_refusal(
                    "flow_capture_mixed",
                    "the planner path crosses an app/context boundary or lacks provenance",
                    "Save a smaller homogeneous journey with `flow save`.",
                )
            if not (
                newest.capture_segment == session.capture_segment
                and newest.origin_package == session.package
                and newest.context_id == session.active_context_id
            ):
                boundary = session.capture_boundary_reason or "the foreground app/context changed"
                return save_refusal(
                    "flow_capture_boundary",
                    f"the recorded actions belong to an older capture segment ({boundary})",
                    "Drive the intended app/context again before saving a flow.",
                )
            blockers = recorded_step_blockers(taken)
            if blockers:
                return save_refusal(
                    "flow_capture_lossy",
                    "the planner path cannot be replayed exactly: " + "; ".join(blockers),
                    "Author the missing replay details explicitly in a flow YAML file.",
                )
            origin = newest.origin_package
            materialized = [
                step.model_copy(update={"package": None}) if step.package == origin else step
                for step in taken
            ]
            steps, params = steps_from_recent(materialized)
            arrival_screen: str | None = None
            if (
                arrived
                and res.screen.package == origin
                and session.package == origin
                and session.active_context_id == newest.context_id
                and res.meta.known_screen
            ):
                from .memory import LEGACY_CONTEXT_ID

                app = mem.load(origin) if origin else None
                record = app.screens.get(res.meta.known_screen) if app is not None else None
                if (
                    record is not None
                    and not record.stale
                    and record.context_id in {newest.context_id, LEGACY_CONTEXT_ID}
                ):
                    arrival_screen = res.meta.known_screen
            flow_store = FlowStore(self.config.memory)
            if flow_store.path(save_flow).exists():
                return save_refusal(
                    "flow_capture_exists",
                    f"flow '{save_flow}' already exists and was not overwritten",
                    "Choose a new name or explicitly manage the existing flow first.",
                )
            try:
                path = flow_store.save(
                    Flow(
                        name=save_flow,
                        app=origin,
                        context_id=newest.context_id,
                        description=f"Recorded by `aua navigate`: {goal}",
                        arrival_screen=arrival_screen,
                        arrival_status="mapped" if arrival_screen else "unverified",
                        params=params,
                        steps=steps,
                    ),
                    force=False,
                )
            except UsageError as exc:
                return save_refusal(
                    "flow_capture_save_refused",
                    str(exc),
                    "Nothing was overwritten; choose another name or repair the existing flow.",
                )
            flow_saved = str(path)
            # Long-lived daemon/MCP engines may already have rendered flow hints for this app.
            # The newly saved journey must be discoverable on the very next observation.
            self._flows_cache.clear()
        out: dict[str, Any] = {
            "ok": arrived,
            "goal": goal,
            "arrived": arrived,
            "final_screen": res.meta.known_screen,
            "package": res.screen.package,
            "elements": [e.compact() for e in res.elements],
            "hint": (
                "goal reached — the path was recorded; next time use `aua goto` (free/fast)"
                if arrived
                else "planner could not confirm the goal — finish manually or refine the goal"
            ),
        }
        if flow_saved:
            out["flow_saved"] = flow_saved
        return out

    def close(self) -> None:
        """Release the device (and its on-device uiautomator2 server). Idempotent."""
        with contextlib.suppress(Exception):
            self.capture_stop()
        # An async observation may still be reading this Device and finalising the session
        # provenance.  Flush it before closing the transport or letting a daemon process exit.
        self._join_memory_writers(timeout_s=5.0)
        dev = self._device
        if dev is not None:
            with contextlib.suppress(Exception):
                dev.close()
            self._device = None
            self._claimed_instance_token = None

    def orient(self) -> dict[str, Any]:
        """What the tool already knows about the foreground app (for ``daemon start``).

        Surfaces the app **playbook** (description, deeplinks, login recipes, quirks) up
        front so the agent starts informed — the durable knowledge the tool learned.
        """
        mem = self._memory
        pkg = self.current_package()
        out: dict[str, Any] = {"package": pkg, "known": False}
        if mem is None or not pkg:
            return out
        app = mem.load(pkg)
        if app is None:
            return out
        session = mem.load_session(self.device.serial)
        playbook = playbook_view(
            app,
            context_id=session.active_context_id,
            max_deeplinks=8,
            max_notes=10,
        )
        launch = launch_payload(app)
        has_playbook = bool(
            any(playbook[key] for key in ("description", "deeplinks", "recipes", "notes")) or launch
        )
        if not app.screens and not has_playbook:
            return out
        hints = mem.navigation_hints(
            self.device.serial,
            pkg,
            max_suggest=self.config.memory.suggest_max,
            max_research=self.config.memory.research_suggest_max,
            include_navigation=self.config.memory.suggest,
            half_life_days=self.config.memory.rank_half_life_days,
        )
        out.update(
            known=True,
            screens=len(app.screens),
            routes=len(app.routes),
            suggested_gotos=hints.suggested_gotos,
            research_tasks=hints.research_tasks,
        )
        if playbook["description"]:
            out["description"] = playbook["description"]
        out.update(launch)
        if playbook["recipes"]:
            out["recipes"] = {r.name: r.note for r in playbook["recipes"]}
        if playbook["deeplinks"]:
            out["deeplinks"] = [
                {"uri": link.uri, "note": link.note} for link in playbook["deeplinks"]
            ]
        if playbook["notes"]:
            out["notes"] = playbook["notes"]
        counts = playbook["counts"]
        if counts["deeplinks"] > len(playbook["deeplinks"]) or counts["notes"] > len(
            playbook["notes"]
        ):
            out["playbook_more"] = {
                "deeplinks": counts["deeplinks"] - len(playbook["deeplinks"]),
                "notes": counts["notes"] - len(playbook["notes"]),
                "hint": "Run `aua about` for the complete current playbook.",
            }
        if counts["stale_or_scoped_out"]:
            out["playbook_filtered"] = counts["stale_or_scoped_out"]
        return out

    def map_find(self, goal: str, *, package: str | None = None) -> dict[str, Any]:
        """Return a context-compatible route preview for a goal without executing it."""
        mem = self._memory
        if mem is None:
            raise UsageError("memory is disabled", hint="Set `memory.enabled: true` in config.")
        pkg = package or self.current_package()
        if not pkg:
            raise UsageError("could not determine the foreground package")
        app = mem.load(pkg)
        if app is None:
            return {"ok": False, "goal": goal, "package": pkg, "code": "map_unknown"}
        session = mem.load_session(self.device.serial)
        context_id = session.active_context_id
        start = session.current_screen
        lexicon = self.config.memory.destructive_labels
        target = resolve_goal(
            app,
            goal,
            start=start,
            context_id=context_id,
            destructive_labels=lexicon,
        )
        path = (
            _shortest_path(
                app,
                target,
                start=start,
                context_id=context_id,
                destructive_labels=lexicon,
            )
            if target
            else None
        )
        if not target or not path:
            return {
                "ok": False,
                "goal": goal,
                "package": pkg,
                "current_screen": start,
                "context_id": context_id,
                "code": "route_unknown",
            }
        risks: list[dict[str, Any]] = []
        route: list[dict[str, Any]] = []
        for edge_index, edge in enumerate(path):
            edge_risks: list[dict[str, Any]] = []
            if not edge.steps:
                edge_risks.append(
                    {
                        "code": "legacy_route",
                        "reason": "route has no inspectable structured steps",
                        "path": f"route[{edge_index}]",
                    }
                )
            else:
                for step_index, step in enumerate(edge.steps):
                    edge_risks.extend(
                        route_step_risks(
                            step,
                            origin_package=app.package,
                            destructive_labels=lexicon,
                            path=f"route[{edge_index}].steps[{step_index}]",
                        )
                    )
            risks.extend(edge_risks)
            route.append(
                {
                    "from": edge.from_screen,
                    "to": edge.to_screen,
                    "status": edge.status,
                    "steps": [step_display(step) for step in edge.steps],
                    "risk": "requires_review" if edge_risks else "safe_navigation",
                    "risks": edge_risks,
                }
            )
        safe = not risks
        required_opt_in: list[str] = []
        codes = {str(item["code"]) for item in risks}
        if codes - {"destructive", "legacy_route"}:
            required_opt_in.append("--allow-unsafe")
        if "destructive" in codes:
            required_opt_in.append("--allow-destructive")
        arguments: dict[str, Any] = {"goal": goal}
        if not safe:
            arguments["plan"] = True
        return {
            "ok": True,
            "goal": goal,
            "package": pkg,
            "current_screen": start,
            "target": target,
            "context_id": context_id,
            "safe": safe,
            "status": "ready" if safe else "requires_review",
            "route": route,
            "risks": risks,
            "required_opt_in": required_opt_in,
            "recommended_call": {
                "cli": f"aua goto {goal!r}" + (" --plan" if not safe else ""),
                "mcp": {"tool": "goto", "arguments": arguments},
                "executes": safe,
                "reason": (
                    "A safe structured navigation route is ready to run."
                    if safe
                    else "Review the route risks before authorizing any disclosed side effect."
                ),
            },
        }

    def _job_checkpoint(self) -> None:
        """Abort a supported background wait at the next safe device-read boundary."""
        event = getattr(self, "_job_cancel_event", None)
        if event is not None and event.is_set():
            raise JobCancelledError("background wait cancelled")

    def _job_sleep(self, seconds: float) -> None:
        """Sleep interruptibly when this Engine is executing a background job."""
        event = getattr(self, "_job_cancel_event", None)
        if event is None:
            time.sleep(seconds)
            return
        if event.wait(max(0.0, seconds)):
            raise JobCancelledError("background wait cancelled")

    def _job_requires_warm_transport(self) -> None:
        raise UsageError(
            "background jobs require a warm AUA daemon or MCP server",
            hint="Enable the daemon and retry, or use the normal foreground wait command.",
        )

    # These adapters are reached only when a CLI job command cannot route to the warm daemon.
    # The actual job manager lives at the daemon/MCP boundary so the worker survives the short
    # client process and status/cancel calls reconnect to the same Engine.
    def job_start(self, **_kwargs: Any) -> None:
        self._job_requires_warm_transport()

    def job_status(self, **_kwargs: Any) -> None:
        self._job_requires_warm_transport()

    def job_wait(self, **_kwargs: Any) -> None:
        self._job_requires_warm_transport()

    def job_cancel(self, **_kwargs: Any) -> None:
        self._job_requires_warm_transport()

    def job_list(self, **_kwargs: Any) -> None:
        self._job_requires_warm_transport()

    # ----------------------------------------------------------------- wait --for-stable

    def wait_stable(
        self,
        *,
        interval_ms: int = 120,
        settle_ms: int = 200,
        timeout_ms: int = 30000,
        observe: bool = False,
        ignore_animation: bool = True,
    ) -> ActionResult:
        """Return once the screen stops changing for ``settle_ms`` (PRD §5, AC14).

        Cheap perceptual-hash over screenshots only — NO OCR, NO hierarchy parse. Works on
        opaque/Compose/video screens; ideal for waiting on image generation / loading.
        ``observe`` folds in a post-settle ``analyze`` — because the screen is settled, the
        returned ids are reliable (fixes the "premature observation" trap on transitions).

        When ``ignore_animation`` is True (default), per-cell grid hashing is used so that
        regions with continuous looping animation (spinners, videos, Lottie) are auto-masked
        and don't prevent settling. The screen is "settled" when all non-animated cells stop
        changing.
        """
        from . import imaging

        device = self.device
        deadline = time.monotonic() + timeout_ms / 1000.0
        samples = 0

        if ignore_animation:
            gs = imaging.GridSettle(streak=imaging.ANIMATION_STREAK)
            stable_since: float | None = None
            while True:
                self._job_checkpoint()
                img = device.screenshot()
                samples += 1
                now = time.monotonic()
                grid_stable = gs.feed(img)
                if grid_stable:
                    if stable_since is None:
                        stable_since = now
                    if (now - stable_since) * 1000.0 >= settle_ms:
                        masked = gs.masked_cells
                        detail = f"settled after {samples} samples"
                        if masked:
                            detail += f" (ignored {len(masked)} animated cells)"
                        return self._observe(
                            ActionResult(ok=True, action="wait-stable", detail=detail),
                            observe,
                            settle=False,  # already settled
                        )
                else:
                    stable_since = None
                if now >= deadline:
                    masked = gs.masked_cells
                    hint = "Increase --timeout/--settle, or the screen is still animating."
                    if masked:
                        hint = (
                            f"{len(masked)} cell(s) flagged as animation and excluded; "
                            "remaining content still changing. " + hint
                        )
                    raise StabilityTimeout(
                        f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                        hint=hint,
                    )
                self._job_sleep(interval_ms / 1000.0)
        else:
            last: int | None = None
            stable_since_legacy: float | None = None
            while True:
                self._job_checkpoint()
                current = imaging.dhash(device.screenshot())
                samples += 1
                now = time.monotonic()
                if last is not None and imaging.is_stable(current, last):
                    if stable_since_legacy is None:
                        stable_since_legacy = now
                    if (now - stable_since_legacy) * 1000.0 >= settle_ms:
                        return self._observe(
                            ActionResult(
                                ok=True,
                                action="wait-stable",
                                detail=f"settled after {samples} samples",
                            ),
                            observe,
                            settle=False,
                        )
                else:
                    stable_since_legacy = None
                last = current
                if now >= deadline:
                    raise StabilityTimeout(
                        f"screen did not settle within {timeout_ms} ms ({samples} samples)",
                        hint="Increase --timeout/--settle, or the screen is still animating.",
                    )
                self._job_sleep(interval_ms / 1000.0)

    # ----------------------------------------------------------------- has (T0)

    def has(
        self,
        text: str,
        *,
        match: str = "contains",
        ignore_case: bool = False,
        ocr_fallback: bool = True,
        source: str = "auto",
        timeout_ms: int = 0,
        by: str = "text",
    ) -> HasResult:
        """Quick presence check — NOT the full pipeline (PRD §5, §6a T0).

        ``by="id"`` matches a resource-id (a bare tail like ``containerDetail`` too) —
        this can confirm containers the parsed element list prunes (Maestro-style
        ``assertVisible: id:``). OCR fallback only applies to text lookups.
        """
        mode = MatchMode(match)
        device = self.device
        src = (source or "auto").lower()

        # T0: hierarchy selector (short-circuits on first hit)
        if src in ("auto", "hierarchy"):
            if timeout_ms and timeout_ms > 0:
                bounds = device.wait_for(
                    text, match=mode, ignore_case=ignore_case, timeout_ms=timeout_ms, by=by
                )
            else:
                bounds = device.find_text(text, match=mode, ignore_case=ignore_case, by=by)
            if bounds is not None:
                return HasResult(found=True, source="hierarchy", bounds=bounds, text=text)
            if src == "hierarchy" or by == "id":
                return HasResult(found=False, source="hierarchy")

        # T0→T3: OCR fallback (only on a hierarchy miss)
        if (src in ("auto", "vision")) and (ocr_fallback or src == "vision"):
            hit = self._ocr_contains(device, text, mode, ignore_case)
            if hit is not None:
                return HasResult(found=True, source="ocr", bounds=hit, text=text)

        return HasResult(found=False, source="hierarchy" if src != "vision" else "ocr")

    def _ocr_contains(
        self, device: Device, text: str, mode: MatchMode, ignore_case: bool
    ) -> tuple[int, int, int, int] | None:
        if not self.factory.is_enabled("ocr"):
            return None
        chain = self.factory.build_chain("ocr")
        if not chain.providers:
            return None
        img = device.screenshot()
        try:
            boxes, _ = run_chain(
                chain,
                lambda p: p.recognize(img),  # type: ignore[attr-defined]
                timeout_s=self.config.timeouts.vision_ms / 1000.0,
            )
        except ProviderError as exc:
            logger.info("ocr fallback unavailable: %s", exc)
            return None
        import re as _re

        needle = text if not ignore_case else text.lower()
        for tb in boxes:
            hay = tb.text if not ignore_case else tb.text.lower()
            ok = False
            if mode is MatchMode.exact:
                ok = hay.strip() == needle.strip()
            elif mode is MatchMode.regex:
                flags = _re.IGNORECASE if ignore_case else 0
                ok = _re.search(text, tb.text, flags) is not None
            else:
                ok = needle in hay
            if ok:
                return tb.bounds
        return None

    # ----------------------------------------------------------------- inspect

    def inspect(self, element_id: int) -> Element:
        return self._resolve(element_id)

    def screenshot(self, path: str | None = None, *, annotate: bool = False) -> ActionResult:
        device = self.device
        img = self.platform.capture_screenshot(device)
        if annotate:
            cached = self._read_cache()
            elements = cached.elements if cached else []
            out = path or self._default_annotate_path(device.serial)
            from . import annotate as annotate_mod

            saved = annotate_mod.annotate(img, elements, out)
            return ActionResult(ok=True, action="screenshot", detail=saved)
        out = path or self._default_annotate_path(device.serial, suffix="screenshot")
        img.save(out)
        return ActionResult(ok=True, action="screenshot", detail=out)

    # ----------------------------------------------------------------- actions

    @staticmethod
    def _compact_action_diff(element_diff: dict[str, Any] | None) -> dict[str, Any] | None:
        """Keep inline diffs token-cheap and machine-readable."""
        if not isinstance(element_diff, dict):
            return None
        added = element_diff.get("added", [])
        removed = element_diff.get("removed", [])
        changed = element_diff.get("changed", [])
        out: dict[str, Any] = {
            "added": len(added) if isinstance(added, list) else added,
            "removed": len(removed) if isinstance(removed, list) else removed,
            "changed": len(changed) if isinstance(changed, list) else changed,
        }
        if "prev_count" in element_diff:
            out["prev_count"] = element_diff["prev_count"]
        if "curr_count" in element_diff:
            out["curr_count"] = element_diff["curr_count"]
        if element_diff.get("unchanged") is not None:
            out["unchanged"] = bool(element_diff["unchanged"])
        return out

    @staticmethod
    def _stable_elements(elements: list[Element]) -> list[dict[str, Any]]:
        """A compact stable-key map for the ids in the folded observation."""
        out: list[dict[str, Any]] = []
        for e in elements:
            if e.stable_key is not None:
                out.append({"id": e.id, "stable_key": e.stable_key})
            else:
                out.append({"id": e.id})
        return out

    def _analyze_post_action(
        self,
        with_image: bool | str | None,
        *,
        record_screen: bool = False,
    ) -> AnalyzeResult:
        """Read the post-action screen, escalating thin trees exactly as ``analyze`` would."""
        obs = self.analyze(
            source="hierarchy",
            record=record_screen,
            with_image=self._effective_with_image(with_image),
        )
        # Hierarchy first because it is tens of milliseconds and answers for most screens.
        # But pinning the folded observation to hierarchy made every Compose/canvas/WebView
        # caller pay for a second explicit analyze. Escalate once through the normal gate.
        if self.config.perception.observe_escalates_to_vision:
            with contextlib.suppress(Exception):
                decision = self._gate_decide(
                    obs.elements,
                    package=obs.screen.package,
                    activity=obs.screen.activity,
                )
                if decision.use_vision:
                    richer = self.analyze(
                        source="auto",
                        record=record_screen,
                        with_image=self._effective_with_image(with_image),
                    )
                    # Keep the hierarchy answer unless escalation actually found more.
                    if len(richer.elements) > len(obs.elements):
                        obs = richer
        return obs

    @staticmethod
    def _change_has_semantic_effect(change: dict[str, Any] | None) -> bool:
        """Whether the readback names an effect beyond node/layout churn."""
        if not change:
            return False
        return bool(
            change.get("activity_changed") is True
            or change.get("focus_moved") is True
            or change.get("text_added")
            or change.get("text_removed")
        )

    @staticmethod
    def _tap_settle_needs_confirmation(
        action_kind: str | None, ready: dict[str, Any] | None
    ) -> bool:
        """Whether an early tap settle needs a longer quiet window before analysis.

        The live traces defeated content heuristics in three different ways: an old screen plus
        one pager node, OCR destination text over the old hierarchy, and a mixed hierarchy with
        both old and new screens. Therefore no single early frame certifies arrival. Confirm every
        tap-like fast hierarchy/pixel settle; slower double-sampled hierarchy settles keep their
        existing path.
        """
        if action_kind not in {"tap", "tap-point", "double-tap", "long-press"}:
            return False
        if not ready or ready.get("timeout") or not ready.get("changed"):
            return False
        return ready.get("via") in {"hierarchy-fast", "pixels"}

    def _observe(
        self,
        result: ActionResult,
        observe: bool,
        with_image: bool | str | None = None,
        *,
        settle: bool = True,
        record_screen: bool = False,
        hierarchy_only: bool = False,
        adopt_action: bool = False,
    ) -> ActionResult:
        """Attach the post-action screen so callers skip a separate ``analyze`` round-trip.

        The folded ``analyze`` also re-populates the id cache, so the agent can act on an id
        from ``result.observation`` immediately (e.g. type → tap send) in one fewer call.

        When ``settle`` is True (default for actions), wait until pixels differ from the
        pre-action frame and the non-animated region is idle — otherwise agents get the
        previous screen and burn a second ``wait --for-stable`` + re-analyze.
        """
        if observe:
            with contextlib.suppress(Exception):  # observation is a bonus; never fail the action
                # Read before the settle consumes the pre-action bookkeeping.
                if adopt_action:
                    before_state = self._action_observation_baseline
                    self._action_observation_baseline = None
                else:
                    before_state = self._pre_action_state
                    self._pre_action_state = None
                    if before_state is not None:
                        self._action_observation_baseline = before_state
                ready: dict[str, Any] | None = None
                confirmed_stable = False
                if settle:
                    settle_ms, total_ms = 45, 1100
                    if self.config.perf.settle_profiles and self._last_action_kind:
                        settle_ms, total_ms = self._settle_profiles.budget(
                            self._last_action_kind,
                            total_max_ms=self.config.perf.settle_total_max_ms,
                        )
                    # A control with its own history overrides the per-kind guess. This is the
                    # only path that can exceed `settle_total_max_ms`: that ceiling exists to stop
                    # a *blind* timer taxing every same-screen tap, and it is the wrong instrument
                    # for a control we have actually measured at 18s.
                    learned = self._learned_action_budget(total_ms)
                    if learned is not None:
                        total_ms = learned
                    ready = self._await_post_action_ready(
                        settle_ms=settle_ms, total_timeout_ms=total_ms
                    )
                    # Only learn from real transitions — same-screen / timeouts poison the EMA
                    # and made subsequent taps ~2× slower (450→900ms) in the field.
                    if (
                        ready
                        and self.config.perf.settle_profiles
                        and self._last_action_kind
                        and ready.get("changed")
                        and ready.get("via") in {"hierarchy-fast", "hierarchy", "pixels"}
                        and ready.get("ms") is not None
                    ):
                        self._settle_profiles.observe(
                            self._last_action_kind,
                            min(float(ready["ms"]), self.config.perf.settle_learn_cap_ms),
                        )
                    # Persist the per-site cost regardless of whether the EMA accepted it: the
                    # coarse profile refuses timeouts because they poison an app-wide average,
                    # but "this control timed out" is precisely what a future run needs told.
                    if ready is not None and ready.get("ms") is not None:
                        self._record_action_timing_safe(
                            float(ready["ms"]),
                            outcome=("changed" if ready.get("changed") else "unchanged"),
                        )
                if self._tap_settle_needs_confirmation(self._last_action_kind, ready):
                    confirm_t0 = time.monotonic()
                    try:
                        # A 350ms quiet window outlives a ripple and a single Compose pager
                        # frame, while the 1.4s ceiling keeps a looping surface bounded.
                        self.wait_stable(
                            interval_ms=80,
                            settle_ms=350,
                            timeout_ms=1400,
                            observe=False,
                        )
                        confirmed_stable = True
                    except Exception:  # noqa: BLE001 — observation remains best-effort
                        if ready is not None:
                            ready["confirmation_timeout"] = True
                    if ready is not None:
                        ready["confirmation_ms"] = int((time.monotonic() - confirm_t0) * 1000)

                # Analyze only after the confirmation. Besides being safer, this avoids paying
                # for and returning an OCR-enriched read of a frame we already distrust.
                if hierarchy_only:
                    obs = self.analyze(
                        source="hierarchy",
                        with_ocr=False,
                        record=record_screen,
                        with_image=self._effective_with_image(with_image),
                    )
                else:
                    obs = self._analyze_post_action(with_image, record_screen=record_screen)
                change: dict[str, Any] | None = None
                if not hierarchy_only:
                    with contextlib.suppress(Exception):
                        change = self._change_summary(before_state, obs)
                if ready is not None and (confirmed_stable or ready.get("confirmation_timeout")):
                    ready["semantic_confirmation"] = self._change_has_semantic_effect(change)

                # Post-action analyzes deliberately do not write memory because their frame may
                # still be transitional. Recognition against the existing map is safe, though,
                # and is strong destination evidence. Do it before evaluating stale risk so a
                # looping animation cannot make a correctly recognised, semantically different
                # destination look unsafe merely because its extended quiet-window timed out.
                mem = self._memory
                if mem is not None and self._device is not None:
                    with contextlib.suppress(Exception):
                        known = mem.observe_screen_passive(
                            self._device.serial,
                            package=obs.screen.package,
                            elements=obs.elements,
                            activity=obs.screen.activity,
                            screen_height=obs.screen.height,
                        )
                        if known:
                            obs.meta.known_screen = known

                before_known = (before_state or {}).get("known_screen")
                destination_confirmed = bool(
                    obs.meta.known_screen and obs.meta.known_screen != before_known
                )

                result.observation = obs
                result.change = change
                caveat = self._stale_observation_risk(
                    settle,
                    ready,
                    destination_confirmed=destination_confirmed,
                    semantic_change_confirmed=self._change_has_semantic_effect(change),
                )
                if caveat:
                    obs.meta.stale_risk = caveat
                    # Also at the top level of the action result, because a runner reading only the
                    # terse form must not have to know the caveat exists to find it. It gets its
                    # own field rather than being appended to `detail`: `detail` carries a
                    # *semantic value* for several actions — `app launch` puts the launched
                    # package/activity there — and appending a marker to it corrupts the thing a
                    # caller parses. The caveat text, not a bare flag, so the reason travels too.
                    result.stale_risk = caveat
                launch_next_actions_unstable = bool(
                    result.action == "app-launch"
                    and (
                        ready is None
                        or ready.get("timeout")
                        or not ready.get("changed")
                        # hierarchy-fast proves departure from the old tree with one sample. It
                        # is enough for an observation, but not for advertising numeric ids that
                        # may disappear before the next command reaches them.
                        or ready.get("via") not in {"pixels", "hierarchy"}
                    )
                )
                if ready and ready.get("ms") is not None and ready.get("via") != "unchanged":
                    # Surface settle cost so agents/tests can see why a tap took >50 ms — in its own
                    # field. This used to be appended to `detail` as a "settle=295ms via=pixels"
                    # tag, which corrupts the semantic value `detail` carries for some actions:
                    # `app launch` puts the launched package/activity there, so an observed launch
                    # answered `detail: "<pkg>/<activity> settle=295ms via=pixels"` and a caller
                    # parsing it got the timing glued onto the component name. Structured here, so
                    # it can be read as a number rather than scraped out of prose.
                    settle: dict[str, Any] = {"ms": ready["ms"]}
                    if ready.get("via"):
                        settle["via"] = ready["via"]
                    if ready.get("masked"):
                        settle["anim"] = ready["masked"]
                    if ready.get("confirmation_ms") is not None:
                        settle["confirmation_ms"] = ready["confirmation_ms"]
                    if ready.get("semantic_confirmation") is not None:
                        settle["semantic_confirmation"] = ready["semantic_confirmation"]
                    result.settle = settle
                result.observation_present = True
                result.next_actions = (
                    None if launch_next_actions_unstable else self._next_actions(obs)
                )
                nav = list(obs.meta.known_routes or []) + list(obs.meta.suggested_gotos or [])
                result.routes = nav or None
                result.known_screen = obs.meta.known_screen
                result.stable_elements = self._stable_elements(obs.elements)
                result.action_diff_summary = self._compact_action_diff(obs.meta.element_diff)
                if launch_next_actions_unstable:
                    result.note = (
                        "The app is foreground, but its launch screen has not produced a stable "
                        "readback yet, so numeric next actions are withheld. Run `aua analyze` "
                        "once before acting on an id."
                    )
                elif ready and ready.get("timeout") and self._change_has_semantic_effect(change):
                    result.note = (
                        "Fresh hierarchy confirms the action changed the screen. Use this "
                        "observation; if an exact destination is still absent, run one exact "
                        "predicate wait instead of a predicate-less settle wait."
                    )
                else:
                    result.note = "No separate analyze needed; state is in observation."
                # Say it in the note too, not only in `change`: the screen this observation
                # describes belongs to a different app, so every id in it is a dead end for
                # whatever the caller was doing.
                left = change.get("app_left_foreground") if isinstance(change, dict) else None
                if left:
                    dialog = (
                        " A system crash dialog is on screen." if left.get("crash_dialog") else ""
                    )
                    result.note = (
                        f"WARNING: {left['from']} left the foreground — {left['to']} is in front "
                        f"now, so this observation is NOT your app.{dialog} Check "
                        '`aua logcat --grep "FATAL EXCEPTION|ANR in" --since last-action` for the '
                        "cause, then relaunch with `aua app restart-and-analyze "
                        f"{left['from']}` instead of navigating this screen. {result.note}"
                    )
        else:
            self._pre_action_sig = None
            if self.config.perf.prefetch:
                self._kick_hierarchy_prefetch()
            result.observation_present = False
        hint = self._capture_hint()
        if hint:
            result.capture_hint = hint
        return result

    @staticmethod
    def _stale_observation_risk(
        settle: bool,
        ready: dict[str, Any] | None,
        *,
        destination_confirmed: bool = False,
        semantic_change_confirmed: bool = False,
    ) -> str | None:
        """Why this post-action observation may describe the screen as it was *before* the action.

        Observed: a `tap` succeeded and the device advanced, yet the returned observation reported
        an empty `element_diff` with `unchanged=true` — measured against a snapshot taken before
        the screen changed. A screenshot plus a fresh `analyze` showed the app had in fact moved
        on.

        The mechanism is the settle wait giving up early. `_await_post_action_ready` returns
        `via=unchanged` after ~80ms of identical frames, and `via=hierarchy-same` on two matching
        trees; the folded `analyze` then dumps a tree that still matches the previous one, so
        `skip_unchanged_analyze` reuses the *previous* payload and stamps `unchanged=true`. The
        device advances a few milliseconds later. Nothing was wrong with any individual step —
        the claim is just older than it looks.

        This is the dangerous direction. The other ways a tap can look inert risk an agent giving
        up too early; this one risks it **repeating an action that already happened** — a second
        submit, a second message, a second purchase attempt. So the engine cannot report
        `unchanged` as a fact here: it genuinely cannot tell "no effect" from "not yet", and
        saying which one it is would be a guess presented as evidence.

        Deliberately conservative: only a *confirmed* transition (the wait saw the screen change
        and then stop, without timing out) clears the caveat. A real in-screen no-op therefore
        carries it too — correct, because the engine cannot distinguish that case either, and the
        expensive mistake is the other direction.
        """
        if not settle:
            # No wait was performed, so there is nothing to be stale relative to; the caller
            # asked for a raw read.
            return None
        if ready is None:
            return (
                "post-action wait did not run, so `unchanged` / `element_diff` may describe the "
                "pre-action screen. Re-analyze before concluding the action had no effect."
            )
        via = str(ready.get("via") or "?")
        if (
            ready.get("confirmation_timeout")
            and destination_confirmed
            and ready.get("semantic_confirmation") is True
        ):
            # The stability wait answers whether every pixel stopped moving. Recognition plus a
            # semantic before/after delta answers the question agents actually need: whether the
            # action reached a different known destination. Persistent animation must not turn
            # that stronger evidence into a false stale warning.
            return None
        if ready.get("confirmation_timeout"):
            return (
                "post-action read looked transitional and its extended stability confirmation "
                "timed out. The later observation is safer but may still be in flight — wait or "
                "re-analyze; never repeat a mutating action from this readback alone."
            )
        if ready.get("semantic_confirmation") is False:
            return (
                "post-action screen stabilized, but AUA observed no semantic destination beyond "
                "layout/node movement. The action may have a visual-only effect — do not repeat a "
                "mutating action from this readback alone."
            )
        if ready.get("timeout") and semantic_change_confirmed:
            # The visual settle budget expired, but the fresh hierarchy has different text,
            # focus, or Activity. It may still be rendering, yet it demonstrably does not
            # predate the action. Calling that stale made fresh agents start a predicate-less
            # 60s wait even when the requested content was already present.
            return None
        if ready.get("timeout"):
            return (
                f"post-action wait timed out (via={via}) — this observation may be mid-transition "
                "or predate the action. Re-analyze before concluding anything from it."
            )
        if not ready.get("changed"):
            return (
                f"post-action wait saw no confirmed screen change (via={via}), so `unchanged` and "
                "`element_diff` may be measured against a frame that predates the action. NOT "
                "evidence the action had no effect — re-analyze, and never retry a mutating "
                "action on `unchanged` alone."
            )
        return None

    @staticmethod
    def _await_foreground(device: Device, package: str, *, timeout_ms: int = 20_000) -> bool:
        """Whether *package* owns the foreground within the budget.

        Returns as soon as it does, so a healthy launch pays only one `app_current` call. The
        budget is generous because a refused launch already failed loudly in ``launch_app``:
        what is left to catch is an app that starts and dies, and a cold start behind a long
        splash must not be mistaken for one. A splash counts as arrival — it is the app's own
        Activity — so this waits for arrival, not for readiness.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            with contextlib.suppress(Exception):
                if (device.current_app() or {}).get("package") == package:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def _await_launch_hierarchy(self, package: str) -> AnalyzeResult:
        """Return a hierarchy attributed to a launch whose foreground is already proven.

        Android can report the new Activity as focused before its accessibility window replaces a
        short-lived SystemUI tree. That is a read race, not evidence that the launch failed. Retry
        fresh hierarchy-only samples while the requested package remains foreground; if ownership
        changes or the bounded attachment window expires, refuse the mixed-package observation.
        """
        deadline = time.monotonic() + _LAUNCH_HIERARCHY_SETTLE_S
        last_package = ""
        while True:
            # `_observe()` has already cached its first readback. Once package attribution says
            # that tree belongs to another window, neither its on-disk ids nor its in-process
            # differential baseline may survive into a retry (or a typed failure).
            self._invalidate_launch_observation()
            try:
                fresh = self.analyze(
                    source="hierarchy",
                    with_ocr=False,
                    no_cache=True,
                    record=False,
                )
            except Exception:
                self._invalidate_launch_observation()
                raise
            last_package = fresh.screen.package or ""
            if not last_package:
                try:
                    foreground = str((self.device.current_app() or {}).get("package") or "")
                except Exception:  # noqa: BLE001 — absence of ownership proof must fail closed
                    foreground = ""
                if foreground != package:
                    self._invalidate_launch_observation()
                    raise DeviceError(
                        (
                            f"{package} reached the foreground, but ownership changed to "
                            f"{foreground or 'an unknown package'} while the hierarchy had no "
                            "package attribution"
                        ),
                        code="launch_observation_mismatch",
                        hint=(
                            "Inspect one fresh hierarchy before acting; AUA did not attribute an "
                            "unowned hierarchy to the launched app."
                        ),
                    )
                fresh.screen.package = package
                self._write_cache(fresh)
                return fresh
            if last_package == package:
                # `no_cache=True` prevents a retry sample from becoming authoritative merely by
                # being read. Persist only the sample whose ownership this method accepted.
                self._write_cache(fresh)
                return fresh

            try:
                foreground = str((self.device.current_app() or {}).get("package") or "")
            except Exception:  # noqa: BLE001 — absence of ownership proof must fail closed
                foreground = ""
            if foreground != package:
                self._invalidate_launch_observation()
                raise DeviceError(
                    (
                        f"{package} reached the foreground, but ownership changed to "
                        f"{foreground or 'an unknown package'} while the hierarchy belonged to "
                        f"{last_package}"
                    ),
                    code="launch_observation_mismatch",
                    hint=(
                        "Inspect one fresh hierarchy before acting; AUA did not return a mixed-"
                        "package launch observation."
                    ),
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._invalidate_launch_observation()
                raise DeviceError(
                    (
                        f"launch foreground was {package}, but the hierarchy still belonged to "
                        f"{last_package} after the attachment wait"
                    ),
                    code="launch_observation_mismatch",
                    hint=(
                        "The launched window did not attach consistently. Inspect one fresh "
                        "hierarchy before acting."
                    ),
                )
            time.sleep(min(_LAUNCH_HIERARCHY_POLL_S, remaining))

    def _invalidate_launch_observation(self) -> None:
        """Drop every cache layer that could still describe a rejected launch tree."""
        self._invalidate_cache()
        self._prefetch.invalidate()
        self._last_analyze_elements = None
        self._last_hierarchy_hash = None
        self._last_analyze_result = None

    def _adopt_recovered_launch_observation(
        self, launched: ActionResult, fresh: AnalyzeResult
    ) -> None:
        """Replace all fields derived from a transient launch readback with *fresh*."""
        launched.observation = fresh
        launched.observation_present = True
        launched.stable_elements = self._stable_elements(fresh.elements)
        launched.next_actions = self._next_actions(fresh)
        nav = list(fresh.meta.known_routes or []) + list(fresh.meta.suggested_gotos or [])
        launched.routes = nav or None
        launched.known_screen = fresh.meta.known_screen
        launched.action_diff_summary = self._compact_action_diff(fresh.meta.element_diff)
        # The original before/after comparison was computed against the rejected tree. A fresh
        # hierarchy alone cannot reconstruct it, so absence is more truthful than mixed evidence.
        launched.change = None
        launched.stale_risk = fresh.meta.stale_risk
        launched.note = "No separate analyze needed; state is in observation."

    def _await_post_action_ready(
        self,
        *,
        change_timeout_ms: int = 500,
        settle_ms: int = 45,
        total_timeout_ms: int = 1100,
        poll_ms: int = 28,
    ) -> dict[str, Any]:
        """Wait for post-action content change, then pixel-idle (animation-aware).

        Runs pixel settle and hierarchy double-sample in one loop so a Compose
        transition that updates the tree early can return before pixels fully idle.
        """
        from . import imaging

        device = self.device
        pre = self._pre_action_sig
        pre_tree = self._pre_action_tree_fp
        self._pre_action_sig = None
        self._pre_action_tree_fp = None
        t0 = time.monotonic()
        deadline = t0 + total_timeout_ms / 1000.0
        change_deadline = t0 + change_timeout_ms / 1000.0
        changed = pre is None
        gs = imaging.GridSettle(streak=imaging.ANIMATION_STREAK)
        stable_since: float | None = None
        last_tree: tuple[str, ...] | None = None
        next_hier_at = t0 + 0.04
        hier_checks = 0
        identical_polls = 0
        same_tree_hits = 0

        while time.monotonic() < deadline:
            try:
                # Fresh frames only — reusing a capture-buffer JPEG (~2 fps) falsely
                # reports idle / change and stretches same-screen taps.
                img = device.screenshot()
            except Exception:
                break
            now = time.monotonic()
            sig = imaging.frame_signature(img)
            if not changed and pre is not None:
                if imaging.frames_differ(pre, sig):
                    changed = True
                    identical_polls = 0
                else:
                    identical_polls += 1
                    # No pixel movement and no tree rewrite → action was a visual no-op
                    # (or FakeDevice). Don't burn the full change_timeout.
                    if identical_polls >= 3 and now - t0 >= 0.08:
                        return {
                            "changed": False,
                            "masked": 0,
                            "ms": int((now - t0) * 1000),
                            "timeout": False,
                            "via": "unchanged",
                        }
            if not changed and now > change_deadline:
                changed = True  # give up waiting for a pixel delta; settle what we have

            visually_idle = gs.feed(img)
            if visually_idle and changed:
                if stable_since is None:
                    stable_since = now
                if (now - stable_since) * 1000.0 >= settle_ms:
                    return {
                        "changed": changed,
                        "masked": len(gs.masked_cells),
                        "ms": int((now - t0) * 1000),
                        "timeout": False,
                        "via": "pixels",
                    }
            else:
                stable_since = None

            if pre_tree is not None and hier_checks < 8 and now >= next_hier_at:
                hier_checks += 1
                next_hier_at = now + 0.06
                with contextlib.suppress(Exception):
                    dump_started = time.monotonic()
                    xml = self.platform.dump_tree(
                        device,
                        compact=bool(self.config.device.compressed_hierarchy),
                    )
                    dump_ms = (time.monotonic() - dump_started) * 1000.0
                    w, h = device.window_size()
                    els = self.platform.normalize_tree(xml, (w, h)).elements
                    parts: list[str] = []
                    for e in els:
                        if getattr(e, "window", None) == "system":
                            continue
                        rid = (e.resource_id or "").split("/")[-1]
                        label = (e.text or e.content_desc or "")[:40]
                        if rid or label:
                            parts.append(f"{rid}:{label}")
                    cur = tuple(parts[:60])
                    if not cur:
                        pass
                    elif cur == pre_tree:
                        # Same accessibility tree as pre-action — in-screen tap / ripple /
                        # selected-state. Element IDs are still valid; don't wait for pixels
                        # (GridSettle stays busy on animations and was the 2× regression).
                        same_tree_hits += 1
                        if same_tree_hits >= 2 or (same_tree_hits >= 1 and now - t0 >= 0.12):
                            return {
                                "changed": False,
                                "masked": len(gs.masked_cells),
                                "ms": int((time.monotonic() - t0) * 1000),
                                "timeout": False,
                                "via": "hierarchy-same",
                            }
                    else:
                        same_tree_hits = 0
                        changed = True
                        s_cur, s_pre = set(cur), set(pre_tree)
                        delta = len(s_cur ^ s_pre)
                        union = max(1, len(s_cur | s_pre))
                        # A big delta is measured against the PRE-action tree, so it only says we
                        # LEFT the old screen — a header-only frame differs from where we came
                        # from maximally, and accepting it on sight handed the caller an
                        # observation with the list body missing (measured on a fast device: one
                        # run in four read zero rows off a screen that has five).
                        #
                        # "Arrived" is distinguished from "still painting" by the tree having
                        # stopped GROWING, which takes two samples to see. Whether that second
                        # sample is affordable depends entirely on the device: a hierarchy dump
                        # measured ~150ms headless but ~600-1200ms windowed, and on the slow one
                        # the render has always finished before the first dump even returns. So
                        # spend a confirming dump only when the remaining budget can absorb one —
                        # otherwise take what we have. Requiring it unconditionally turned 662ms
                        # actions into 1276ms ones with 9 of 12 hitting the deadline, which buys
                        # nothing on a device whose dumps are already slower than its rendering.
                        # A tree that is a small fraction of the one we left is the shape of a
                        # half-drawn screen — a header whose body has not been attached yet.
                        # That, not "differs from before", is what has to hold us.
                        #
                        # Only devices with FAST dumps can land in that state: a dump measured
                        # ~150ms headless but 600-1200ms windowed, and on the slow one the render
                        # has always finished before the first dump even returns (measured: rows
                        # present in the first sample every time, 807-1205ms after the tap). So
                        # the confirming sample is worth its cost exactly when dumps are cheap;
                        # spending it anywhere else buys nothing and cost +614ms per action.
                        thin = len(pre_tree) >= 6 and len(cur) * 2 < len(pre_tree)
                        settled = visually_idle or not thin or dump_ms > _FAST_DUMP_MS
                        if settled and delta >= max(4, union // 3):
                            return {
                                "changed": True,
                                "masked": len(gs.masked_cells),
                                "ms": int((time.monotonic() - t0) * 1000),
                                "timeout": False,
                                "via": "hierarchy-fast",
                            }
                        if settled and cur == last_tree:
                            return {
                                "changed": True,
                                "masked": len(gs.masked_cells),
                                "ms": int((time.monotonic() - t0) * 1000),
                                "timeout": False,
                                "via": "hierarchy",
                            }
                        last_tree = cur
            time.sleep(poll_ms / 1000.0)

        return {
            "changed": changed,
            "masked": len(gs.masked_cells),
            "ms": int((time.monotonic() - t0) * 1000),
            "timeout": True,
            "via": "timeout",
        }

    def resolve(
        self,
        target: str | int,
        *,
        fresh: bool = True,
    ) -> ResolveResult:
        """Remap a previous-frame id or ``stable_key`` onto the current screen.

        Integer ids die on every re-analyze; ``stable_key`` (and this remapper) survive.
        """
        from .identity import find_by_stable_key, remap_ids, stable_key

        cached = self._read_cache()
        current = self.analyze(source="auto", record=False) if fresh or cached is None else cached

        from_id: int | None = None
        key: str | None = None
        if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
            from_id = int(target)
            if cached is not None:
                prev = cached.element_by_id(from_id)
                if prev is not None:
                    key = prev.stable_key or stable_key(prev)
                    mapping = remap_ids(cached.elements, current.elements)
                    if from_id in mapping:
                        to_id = mapping[from_id]
                        el = current.element_by_id(to_id)
                        return ResolveResult(
                            ok=True,
                            from_id=from_id,
                            to_id=to_id,
                            stable_key=key,
                            element=el,
                        )
            # Fall through: treat as missing and try key from fresh screen? No — id unknown.
            raise ElementNotFoundError(
                f"could not resolve id {from_id} onto the current screen",
                hint="Re-analyze after the screen changes, or pass a stable_key (rid:…).",
            )

        key = str(target).strip()
        hits = find_by_stable_key(current.elements, key)
        if len(hits) == 1:
            el = hits[0]
            return ResolveResult(ok=True, from_id=from_id, to_id=el.id, stable_key=key, element=el)
        if not hits:
            raise ElementNotFoundError(
                f"no element with stable_key {key!r} on the current screen",
                hint="Run `aua analyze` and use the element's stable_key, or a prior id.",
            )
        raise SelectorAmbiguousError(
            f"stable_key {key!r} matched {len(hits)} elements",
            hint="Disambiguate with --rid/--text or inspect the screen.",
        )

    def resolve_selector(
        self,
        *,
        rid: str | None = None,
        text: str | None = None,
        desc: str | None = None,
        index: int | None = None,
        first: bool = False,
        fresh: bool = True,
        vision_fallback: bool = True,
        prefer_clickable: bool = False,
    ) -> Element:
        """Resolve a one-shot selector to a single element, in this one call.

        Element ids die the moment the screen changes, so ``analyze`` → grep → ``tap <id>``
        is three round-trips whose middle step the caller has to hand-write. Resolving
        server-side collapses that to one, and the re-analyze also refreshes the id cache
        so ids in the returned observation are immediately usable.

        Raises rather than guessing: :class:`SelectorNotFoundError` on no match (with the
        nearest candidates), :class:`SelectorAmbiguousError` on several (with all of them).
        A silent pick is indistinguishable from "the app ignored a valid tap".
        """
        selector = {"rid": rid, "text": text, "desc": desc}
        given = [field for field, value in selector.items() if value]
        if len(given) != 1:
            raise UsageError(
                "give exactly one selector: --rid <resource-id> | --text <label> | --desc <desc>",
                hint="e.g. `aua tap-and-analyze --rid notificationsButton` or `aua tap-and-analyze --text 'Create an app'`",
            )
        cached = None if fresh else self._read_cache()
        result = cached if cached is not None else self.analyze(source="hierarchy", record=False)
        elements = result.elements
        # Fused OCR readings of text the tree already reports would break tiering: see
        # drop_redundant_ocr. Must happen before matching, not after.
        matches = match_selector(drop_redundant_ocr(elements), rid=rid, text=text, desc=desc)
        label = selector_label(selector)
        if not matches and rid:
            # The element list prunes unlabeled, non-actionable containers, so a real
            # `containerDetail` is addressable without being listed. Ask the device
            # directly before giving up — the same lookup `has --by id` uses.
            container = self._resolve_container_rid(rid)
            if container is not None:
                return container
        if not matches and text and vision_fallback:
            matches, elements = self._match_by_vision(elements, text)
        if not matches:
            needle = rid or text or desc or ""
            near = nearest_elements(elements, needle)
            # Every row carries both an `id` (this analyze's ordinal) and a `rid` (the app's
            # resource-id), so `--rid 49` is the natural conflation of the two columns. Answering
            # it with "nearest: action_bar_root | System UI notification" sends the reader looking
            # for a spelling mistake that is not there, so name the actual mix-up first.
            if rid and rid.isdigit():
                hint = (
                    f"{rid} is an element id, not a resource-id — ids are positional: "
                    f"`aua tap-and-analyze {rid}`. Use --rid for the app's resource-id string "
                    "(the `rid` column), and prefer it: ids are renumbered by every analyze."
                )
            elif near:
                hint = "nearest: " + " | ".join(element_digest(el) for el in near)
            else:
                hint = "Run `aua analyze` to see what is on screen."
            raise SelectorNotFoundError(
                f"no element matches {label} "
                f"({len(app_elements(elements))} app elements on screen)",
                hint=hint,
            )
        if len(matches) > 1:
            if index is not None:
                if not 0 <= index < len(matches):
                    raise UsageError(
                        f"--index {index} out of range: {label} matches {len(matches)} elements",
                        hint="Indexes are 0-based and follow reading order (top-left first).",
                    )
                return matches[index]
            if prefer_clickable:
                # Compose routinely renders a clickable icon tile and a caption beneath it
                # carrying the same text, as two separate accessibility nodes with no
                # overlap - measured on a real sheet: the tile at y 840-1000 and the caption
                # at y 1016-1051. So `tap --text "Attach Photos"` matches two elements and
                # both are genuine; no dedup rule can help, because neither is a duplicate
                # reading of the other. But only one of them can be tapped, and the caller
                # said tap. Choosing it is not guessing - it is the only interpretation that
                # can be carried out. Two tappable matches stay ambiguous, as they should.
                tappable = [el for el in matches if el.clickable]
                if len(tappable) == 1:
                    logger.info(
                        "%s matched %d elements; taking the only clickable one (id=%s)",
                        label,
                        len(matches),
                        tappable[0].id,
                    )
                    return tappable[0]
            if not first:
                raise SelectorAmbiguousError(
                    f"{label} matches {len(matches)} elements — "
                    "disambiguate with --index <n> or take --first",
                    hint="candidates: "
                    + " | ".join(element_digest(el) for el in matches[:_MAX_CANDIDATES]),
                )
        return matches[0]

    def _match_by_vision(
        self, elements: list[Element], text: str
    ) -> tuple[list[Element], list[Element]]:
        """Look again with vision when the hierarchy has no element carrying ``text``.

        Web content publishes almost nothing to the accessibility tree, so `--text Continue`
        fails on an OAuth consent page, a Google sign-in form or an in-app Terms screen where
        "Continue" is plainly on screen. Because vision element ids do not survive between CLI
        invocations, the only way through used to be a raw coordinate tap — the single thing
        this tool exists to remove.

        Only a label falls back. A resource-id is a property of the tree and pixels cannot
        supply one; a content-desc is likewise unobservable, so `--desc` stays strict.

        Returns the matches and the element list to describe the screen with, so that on a
        miss the "nearest" hint names what is *visible* rather than an empty WebView node.
        """
        try:
            seen = self.analyze(source="vision", record=False)
        except AuaError as exc:  # vision unavailable is not a selector error
            logger.debug("vision fallback for %r unavailable: %s", text, exc)
            return [], elements
        matches = match_selector(seen.elements, text=text)
        if matches:
            logger.info("resolved --text %r by vision; the hierarchy had no match", text)
            return matches, seen.elements
        return [], elements + [el for el in seen.elements if el not in elements]

    def _resolve_container_rid(self, rid: str) -> Element | None:
        """A pruned container, addressed by resource-id via the device itself.

        Returns ``None`` when the device does not know it either, so the caller still
        raises :class:`SelectorNotFoundError` with its candidate list. ``id=-1`` marks an
        element that never came from an ``analyze``, so it is not in the id cache.
        """
        bounds = self.device.find_text(rid, match=MatchMode.exact, by="id")
        if bounds is None:
            return None
        return Element(
            id=-1,
            type="Container",
            resource_id=rid,
            bounds=bounds,
            center=center_of(bounds),
        )

    def _target(
        self, element_id: int | None, selector: dict[str, Any] | None, *, verb: str = "tap"
    ) -> Element:
        """The element an action addresses: a freshly-bound prior id, or a selector.

        Integer ids are reading-order ordinals, not identities.  A network update can renumber a
        dynamic list between ``analyze`` and ``long-press`` without any AUA command in between;
        resolving the integer straight out of the old cache then acts on a different row.  Before
        every numeric action, re-read and remap the cached element by stable identity.  When the
        evidence no longer names the same labelled control, refuse before touching the device.
        """
        if selector:
            # A verb that needs something tappable may break a tie on clickability.
            return self.resolve_selector(**selector, prefer_clickable=verb in ("tap", "long-press"))
        if element_id is None:
            raise UsageError(
                f"{verb} needs an element id or a selector",
                hint=f"`aua {verb} 9`, `aua {verb} --rid someId`, or `aua {verb} --text 'Label'`",
            )
        return self._resolve_action_id(element_id, verb=verb)

    @staticmethod
    def _binding_label(element: Element) -> tuple[str, str]:
        """Normalised semantic label used to detect a resource-id that changed owners."""

        def normalise(value: str | None) -> str:
            return re.sub(r"\s+", " ", (value or "").strip()).casefold()

        return normalise(element.text), normalise(element.content_desc)

    def _resolve_action_id(self, element_id: int, *, verb: str) -> Element:
        """Remap a frame-local id onto a fresh hierarchy, or raise ``stale_element_id``.

        Resource ids are normally the strongest binding, but reusable row layouts give every
        item the same rid.  The label check is therefore intentional even after ``remap_ids``:
        if cached id 8 said "Draft Orion" and the same row now says "Suggested reply", touching
        the overlapping node would silently perform the requested action on the wrong content.
        """
        from .identity import remap_ids, stable_key

        cached = self._read_cache()
        if cached is None:
            raise ElementNotFoundError(
                "no cached analyze result", hint="Run `aua analyze` first to assign element ids."
            )
        previous = cached.element_by_id(element_id)
        if previous is None:
            valid = ", ".join(str(e.id) for e in cached.elements[:20]) or "(none)"
            raise ElementNotFoundError(
                f"element id {element_id} is not in the last analyze (valid: {valid})",
                hint="Re-run `aua analyze`; ids change when the screen changes.",
            )

        # Freshness validation compares hierarchy bindings. Running optional OCR here adds an
        # unrelated provider call to every numeric action and defeats goto's hierarchy-first
        # policy; `_run_steps` already performs one explicit OCR retry on a selector miss.
        current = self.analyze(source="hierarchy", with_ocr=False, record=False)
        mapped = remap_ids(cached.elements, current.elements).get(element_id)
        candidate = current.element_by_id(mapped) if mapped is not None else None
        previous_key = previous.stable_key or stable_key(previous)
        labels_match = candidate is not None and self._binding_label(
            previous
        ) == self._binding_label(candidate)
        if candidate is None or not labels_match:
            selector_hint = "a stable --rid/--text/--desc selector"
            if previous.resource_id:
                selector_hint = f"--rid {(previous.resource_id or '').rsplit('/', 1)[-1]}"
            elif previous.content_desc:
                selector_hint = f"--desc {previous.content_desc!r}"
            elif previous.text:
                selector_hint = f"--text {previous.text!r}"
            current_label = None
            if candidate is not None:
                current_label = candidate.text or candidate.content_desc
            detail = f"; it now resolves to {current_label!r}" if current_label else ""
            raise StaleElementIdError(
                f"element id {element_id} is stale for {verb}: binding {previous_key!r} changed{detail}",
                hint=(
                    f"No action was sent. Use {selector_hint}, or re-run `aua analyze` and use "
                    "an id from that fresh observation."
                ),
            )
        return candidate

    def target_report(
        self,
        *,
        rid: str | None = None,
        text: str | None = None,
        desc: str | None = None,
        index: int | None = None,
        first: bool = False,
    ) -> dict[str, Any]:
        """Answer "what does this label actually address?" without touching the device.

        This is the half of the acting-node problem that cost a verdict. A lane did not tap
        and misread the result — it *read state*: the tile's title reported
        ``clickable:false, enabled:true`` and it concluded the control was broken. Nothing in
        the output said that node carries no click action, so its ``enabled`` flag described a
        caption rather than a control, and the pair looks **identical whether the real control
        is enabled or disabled**. So there has to be a way to ask before believing.

        Returns the node named, the node that acts, their relation, the state that actually
        belongs to the control, and the point a tap would use.
        """
        from .selectors import acting_node, acting_report

        named = self.resolve_selector(
            rid=rid, text=text, desc=desc, index=index, first=first, prefer_clickable=False
        )
        cached = self._read_cache()
        pool = list(cached.elements) if cached is not None else [named]
        if all(other.id != named.id for other in pool):
            pool.append(named)
        found = acting_node(pool, named)
        acted = found.element if found.redirected else named
        aim_x, _ = self._tap_point(acted, text if acted.id == named.id else None)
        _, aim_y = self._aim(acted)
        return {
            "ok": True,
            "action": "target",
            "named": named.compact(),
            "acts": found.relation == "self",
            "acting": acting_report(found),
            "control": {
                "id": acted.id,
                "type": acted.type,
                "clickable": bool(acted.clickable),
                "enabled": bool(acted.enabled),
                "checkable": acted.checkable,
                "checked": acted.checked,
                "long_clickable": acted.long_clickable,
                "bounds": list(acted.bounds),
            },
            "tap_point": [aim_x, aim_y],
            "hint": (
                None
                if found.relation == "self"
                else "`enabled`/`clickable` on the node you named describe a caption, not the "
                "control — read them off `control` instead."
            ),
        }

    def _acting_target(self, el: Element, *, verb: str = "tap") -> tuple[Element, dict[str, Any]]:
        """The node that will receive the interaction, plus the report of that choice.

        Retargets **only when the named node carries no interaction at all**, which is the
        deliberate boundary. A node that is itself clickable is left exactly as it was, so the
        overwhelming majority of existing taps are untouched — the danger a change to `tap`
        carries is silently wrong results, and this narrows the blast radius to the case that
        is already broken. Tapping a non-clickable node's centre today dispatches into
        whatever is under it and usually does nothing (the observed tile returned ok:true and
        did not act), so redirecting there can only improve on a coin flip.

        Also refuses to move when the choice would be a guess: several candidate controls, or
        none, leave the target alone and say so in the report.
        """
        from .selectors import acting_node, acting_report

        cached = self._read_cache()
        pool = list(cached.elements) if cached is not None else [el]
        if all(other.id != el.id for other in pool):
            pool.append(el)
        found = acting_node(pool, el)
        if verb == "long-press" and found.relation == "sibling-subtree":
            raise UsageError(
                "long-press will not retarget a label into a sibling control subtree",
                hint=(
                    "No gesture was sent. Inspect the acting candidates with `aua target`, then "
                    "long-press the actual control by --rid or a fresh numeric id."
                ),
                code="unsafe_action_target",
            )
        report = acting_report(found)
        return (found.element if found.redirected else el), report

    def _tap_point(self, el: Element, needle: str | None) -> tuple[int, int]:
        """Where to tap for *el*, aiming at *needle* when it is only part of one line.

        Two links can share a single line: "Terms of use and Privacy policy" is one
        underlined run where each phrase is its own tappable span. Android does not publish
        a ClickableSpan as a separate node, and OCR groups by visual line, so both the tree
        and the pixels describe that line as **one** element. Tapping its centre lands
        between the two links - on neither - and the only workaround left was a measured
        coordinate tap, which is exactly what this tool exists to remove.

        So when the matched element's text merely *contains* the phrase asked for, and the
        element is a single line, aim proportionally: estimate the phrase's horizontal share
        of the line and tap the middle of that. Character widths vary, so this is an
        estimate - but an estimate inside the right phrase beats the exact centre of the
        wrong one.

        Falls back to the element centre whenever the assumption does not hold: no needle,
        an exact match, a multi-line box, or a phrase not found in the text.
        """
        cx, cy = el.center
        text = (el.text or "").strip()
        want = (needle or "").strip()
        if not want or not text or len(want) >= len(text):
            return cx, cy
        start = text.casefold().find(want.casefold())
        if start < 0:
            return cx, cy
        x1, y1, x2, y2 = el.bounds
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return cx, cy
        # Is this one line? Aspect ratio does not answer that - a 640x200 paragraph is still
        # wider than it is tall. Comparing the height to the average character width does,
        # and is independent of screen density: one line of text is roughly two to three
        # character-widths tall, a paragraph is many. Guessing wrong here would aim at the
        # right horizontal offset on the wrong line, which is worse than the centre.
        avg_char_width = width / len(text)
        if height > _SINGLE_LINE_HEIGHT_RATIO * avg_char_width:
            return cx, cy
        mid_char = start + len(want) / 2.0
        x = int(x1 + width * (mid_char / len(text)))
        x = max(x1 + 1, min(x, x2 - 1))
        logger.info(
            "aiming at %r inside %r: x=%d instead of the line centre %d", want, text[:40], x, cx
        )
        return x, cy

    def tap(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        observe: bool = True,
        with_image: bool | str | None = None,
        _hierarchy_settle: bool = False,
    ) -> ActionResult:
        named = self._target(element_id, selector)
        # The label a caller names is often not the node that acts — see `_acting_target`.
        # Resolve that first, then aim, so both corrections apply to the control rather than
        # to the caption sitting below it.
        el, acting = self._acting_target(named)
        if element_id is not None and acting.get("relation") == "sibling-subtree":
            raise UsageError(
                f"numeric id {element_id} names a label, not the sibling control that would act",
                hint=(
                    "No gesture was sent. Numeric ids are exact frame bindings and are never "
                    "retargeted to a sibling. Use the acting control's fresh id/rid from `aua "
                    "target`, or use the visible-text selector deliberately."
                ),
                code="unsafe_action_target",
            )
        # Two independent corrections to the naive `el.center`, composed on separate axes.
        # `_tap_point` aims x at a named phrase inside a single line, so two links on one line
        # are separately reachable. `_aim` moves y out of the system navigation bar, whose
        # docstring states it never second-guesses x for exactly this reason. Take x from the
        # first and y from the second; `_aim` still raises when nothing of the element is
        # visible, which must survive rather than being swallowed here.
        # The phrase-aiming needle belongs to the *named* node's text, so it is only
        # meaningful when the named node is the one being tapped.
        needle = (selector or {}).get("text") if el.id == named.id else None
        cx, _ = self._tap_point(el, needle)
        _, cy = self._aim(el)
        step = self._step("tap", el)  # built pre-action: needs the cached package
        with self._acting(_action_mark("tap", el), capture_pre_action=not _hierarchy_settle):
            self.device.click(cx, cy)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="tap", id=el.id, target=[cx, cy], acting=acting),
            observe,
            with_image,
        )

    def tap_point(
        self,
        x: int,
        y: int,
        *,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        """Tap an exact screen coordinate, and record it like any other action.

        For a canvas, or a grid that publishes one accessibility node per row with nothing
        per cell, a coordinate is the *correct* address rather than a workaround — there is
        no element to name. The alternative a runner is left with is `adb shell input tap`,
        which cannot be recorded, so any journey crossing such a surface becomes uncapturable.

        Both of `tap`'s aim corrections are deliberately bypassed: `_tap_point` moves x
        toward a named phrase and `_aim` moves y out of the system navigation bar, and each
        exists to guess better than `el.center`. Here the caller has already said exactly
        where, so guessing on either axis would be wrong — including the nav-bar lift, since
        tapping the bar itself is a legitimate thing to ask for.
        """
        step = self._step("tap-point", arg=f"{x},{y}")
        with self._acting(f"tap-point:{x},{y}"):
            self.device.click(x, y)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="tap-point", target=[x, y]), observe, with_image
        )

    def long_press(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        ms: int = 600,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        named = self._target(element_id, selector, verb="long-press")
        el, acting = self._acting_target(named, verb="long-press")
        cx, cy = self._aim(el)
        step = self._step("long-press", el)
        with self._acting(_action_mark("long-press", el)):
            self.device.long_click(cx, cy, ms)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="long-press", id=el.id, target=[cx, cy], acting=acting),
            observe,
            with_image,
        )

    def mic_inject(
        self,
        wav_path: str | Path,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        control_mode: str = "hold",
        pre_roll_ms: int = 250,
        post_roll_ms: int = 250,
        observe: bool = True,
        with_image: bool | str | None = None,
        _action: str = "mic-inject",
        _source: str | None = None,
    ) -> ActionResult:
        """Inject PCM WAV audio with an optional hold or tap-to-toggle control."""

        if pre_roll_ms < 0 or post_roll_ms < 0:
            raise UsageError(
                "microphone pre-roll and post-roll must be zero or greater",
                code="mic_roll_invalid",
            )
        from . import mic as mic_mod

        has_control = element_id is not None or selector is not None
        control_mode = mic_mod.validate_control_mode(control_mode, has_target=has_control)
        prepared = mic_mod.prepare_injection(self.device.serial, wav_path)
        el: Element | None = None
        acting: dict[str, Any] | None = None
        target: list[int] | None = None
        toggle_owner: str | None = None
        if has_control:
            verb = "tap" if control_mode == "toggle" else "long-press"
            if control_mode == "toggle":
                owner_before_target = self.device.current_app()
                toggle_owner = str(owner_before_target.get("package") or "")
                if not toggle_owner:
                    raise DeviceError(
                        "could not prove which foreground app owns the toggle control",
                        code="mic_toggle_owner_unknown",
                        hint="No gesture or audio was sent. Observe the app again before retrying.",
                    )
            named = self._target(element_id, selector, verb=verb)
            el, acting = self._acting_target(named, verb=verb)
            if control_mode == "toggle":
                if element_id is not None and acting.get("relation") == "sibling-subtree":
                    raise UsageError(
                        f"numeric id {element_id} names a label, not the sibling toggle control",
                        hint=(
                            "No gesture was sent. Use the acting control's fresh id/rid from "
                            "`aua target`, or use the visible-text selector deliberately."
                        ),
                        code="unsafe_action_target",
                    )
                if not el.enabled or not el.clickable:
                    raise UsageError(
                        "toggle microphone control must be enabled and clickable",
                        code="mic_toggle_target_inactive",
                        hint="No gesture or audio was sent. Choose the active tap-to-start control.",
                    )
                if el.checked is True or el.selected is True:
                    raise UsageError(
                        "toggle microphone control is already active",
                        code="mic_toggle_already_active",
                        hint=(
                            "No gesture or audio was sent. Stop the active recording first, then "
                            "retry only after the control visibly returns to its off state."
                        ),
                    )
                needle = (selector or {}).get("text") if el.id == named.id else None
                cx, _ = self._tap_point(el, needle)
                _, cy = self._aim(el)
                owner = self.device.current_app()
                current_owner = str(owner.get("package") or "")
                if not current_owner or current_owner != toggle_owner:
                    raise DeviceError(
                        "the foreground app changed after resolving the toggle control",
                        code="mic_toggle_owner_changed",
                        hint=(
                            "No gesture or audio was sent. Observe the current app and resolve "
                            "the control again."
                        ),
                    )
            else:
                cx, cy = self._aim(el)
            target = [cx, cy]
        # Claim the affected-build one-attempt guard before either opening gesture. Burning an
        # attempt if control start subsequently fails is conservative; allowing another stream
        # can crash the emulator process on 36.4.10.
        prepared = mic_mod.claim_injection_attempt(prepared)

        # Settle timing still needs to know which public action ran, but microphone samples
        # (like typed values) must not silently become a replayable navigation route.
        self._step(_action, el)
        mark = _action if el is None else _action_mark("mic", el)
        control_started = False
        down_attempted = False
        toggle_stop_allowed = True
        action_error: BaseException | None = None
        terminal_mic_error: mic_mod.MicDeliveryUncertainError | None = None
        injection_attempted = False
        injection_completed = False

        def toggle_owner_failure(stage: str) -> DeviceError | None:
            try:
                current = self.device.current_app()
            except BaseException as exc:
                return DeviceError(
                    f"could not prove toggle control ownership {stage}: {type(exc).__name__}",
                    code="mic_toggle_owner_unknown",
                    hint="Do not tap or retry blindly; recording state may be unknown.",
                )
            current_package = str(current.get("package") or "")
            if not current_package or current_package != toggle_owner:
                return DeviceError(
                    f"the foreground app changed {stage}",
                    code="mic_toggle_owner_changed",
                    hint="AUA refused to tap the snapshotted point in a different app.",
                )
            return None

        try:
            with self._acting(mark):
                try:
                    if target is not None:
                        if control_mode == "hold":
                            # DOWN can land before its response is lost. Mark the attempt first
                            # so cleanup sends the harmless matching UP even when this raises.
                            down_attempted = True
                            self.device.touch_down(*target)
                            control_started = True
                        else:
                            owner_error = toggle_owner_failure("immediately before toggle START")
                            if owner_error is not None:
                                action_error = owner_error
                            else:
                                try:
                                    self.device.click_once(*target)
                                except BaseException as exc:
                                    # Never compensate for an ambiguous START: a second tap could
                                    # either stop a delivered first tap or start recording itself.
                                    terminal_mic_error = mic_mod.MicToggleStartUncertainError()
                                    logger.warning(
                                        "toggle START was not confirmed: %s", type(exc).__name__
                                    )
                                else:
                                    control_started = True
                        if control_started and pre_roll_ms:
                            time.sleep(pre_roll_ms / 1000.0)

                    if control_mode == "toggle" and control_started and terminal_mic_error is None:
                        owner_error = toggle_owner_failure("before audio injection")
                        if owner_error is not None:
                            toggle_stop_allowed = False
                            terminal_mic_error = mic_mod.MicToggleStopUncertainError(
                                "toggle START was confirmed, but the original app no longer "
                                "provably owned the screen before audio injection"
                            ).note_followup_failure("ownership_before_audio", owner_error)

                    if terminal_mic_error is None and action_error is None:
                        injection_attempted = True
                        try:
                            mic_mod.inject_prepared(prepared)
                            injection_completed = True
                        except mic_mod.MicDeliveryUncertainError as exc:
                            # INTERNAL can arrive after every sample was delivered. Finish the
                            # control and force one observation, but retain the typed outcome.
                            terminal_mic_error = exc
                        except BaseException as exc:
                            action_error = exc

                    if (
                        control_started
                        and action_error is None
                        and injection_attempted
                        and (injection_completed or terminal_mic_error is not None)
                        and post_roll_ms
                    ):
                        try:
                            time.sleep(post_roll_ms / 1000.0)
                        except BaseException as exc:
                            if terminal_mic_error is not None:
                                terminal_mic_error.note_followup_failure("post_roll", exc)
                            else:
                                action_error = exc
                except BaseException as exc:
                    if terminal_mic_error is not None:
                        terminal_mic_error.note_followup_failure("control_action", exc)
                    else:
                        action_error = exc
                finally:
                    should_finish_hold = (
                        control_mode == "hold" and down_attempted and target is not None
                    )
                    should_finish_toggle = (
                        control_mode == "toggle"
                        and control_started
                        and toggle_stop_allowed
                        and target is not None
                    )
                    if should_finish_hold or should_finish_toggle:
                        assert target is not None
                        try:
                            if should_finish_hold:
                                self.device.touch_up(*target)
                            else:
                                owner_error = toggle_owner_failure("before toggle STOP")
                                if owner_error is not None:
                                    raise owner_error
                                # Exact snapshotted point, exactly once; never re-resolve a
                                # label whose meaning may have changed from Start to Stop.
                                self.device.click_once(*target)
                        except BaseException as exc:
                            finish_stage = (
                                "touch_release" if control_mode == "hold" else "toggle_stop"
                            )
                            if terminal_mic_error is not None:
                                terminal_mic_error.note_followup_failure(finish_stage, exc)
                                logger.warning(
                                    "control cleanup also failed after ambiguous microphone action"
                                )
                            elif injection_completed and action_error is None:
                                error_type = (
                                    mic_mod.MicDeliveredReleaseError
                                    if control_mode == "hold"
                                    else mic_mod.MicToggleStopUncertainError
                                )
                                terminal_mic_error = error_type().note_followup_failure(
                                    finish_stage, exc
                                )
                                logger.warning("control cleanup failed after audio delivery")
                            elif action_error is None:
                                if control_mode == "toggle":
                                    terminal_mic_error = (
                                        mic_mod.MicToggleStopUncertainError().note_followup_failure(
                                            finish_stage, exc
                                        )
                                    )
                                else:
                                    action_error = exc
                            else:
                                if control_mode == "toggle":
                                    terminal_mic_error = mic_mod.MicToggleStopUncertainError()
                                    terminal_mic_error.note_followup_failure(
                                        "audio_action", action_error
                                    )
                                    terminal_mic_error.note_followup_failure(finish_stage, exc)
                                    action_error = None
                                else:
                                    # Preserve the known injection error for hold mode, whose
                                    # UP cleanup is idempotent and does not toggle recording on.
                                    logger.warning(
                                        "touch-up also failed after microphone injection failed"
                                    )
        except BaseException as exc:
            if terminal_mic_error is not None:
                terminal_mic_error.note_followup_failure("action_cleanup", exc)
            elif action_error is None:
                action_error = exc

        if terminal_mic_error is None and action_error is not None:
            raise action_error

        wav = prepared.wav
        channel_label = "mono" if wav.channels == 1 else "stereo"
        media_detail = (
            f"{wav.duration_s:.3f}s PCM {wav.sample_format} {channel_label} at {wav.sample_rate} Hz"
        )
        if injection_completed:
            detail = f"injected {media_detail}"
        elif injection_attempted:
            detail = f"audio delivery did not complete cleanly for {media_detail}"
        else:
            detail = f"audio was not injected ({media_detail})"
        if _source:
            detail = f"{_source}; {detail}"
        if target is not None:
            control_label = "push-to-talk hold" if control_mode == "hold" else "toggle control"
            detail += (
                f" with {control_label} ({pre_roll_ms}ms pre-roll, {post_roll_ms}ms post-roll)"
            )
        action_result = ActionResult(
            ok=terminal_mic_error is None,
            action=_action,
            id=el.id if el is not None else None,
            target=target,
            detail=detail,
            acting=acting,
        )
        if terminal_mic_error is not None:
            try:
                # A late transport/gesture failure is the one outcome where a fresh screen is
                # mandatory: without it, `--no-observe` would leave callers with no safe way
                # to decide whether the already-sent samples took effect.
                result = self._observe(action_result, True, with_image)
                result_payload = result.model_dump(mode="json")
            except BaseException as exc:
                raise terminal_mic_error.note_followup_failure("observation", exc) from exc
            raise terminal_mic_error.with_result(result_payload)
        return self._observe(action_result, observe, with_image)

    def mic_speak(
        self,
        text: str,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        control_mode: str = "hold",
        voice: str | None = None,
        rate: int | None = None,
        pre_roll_ms: int = 250,
        post_roll_ms: int = 250,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        """Synthesize *text* with macOS ``say`` and inject the resulting temporary WAV."""

        from . import mic as mic_mod

        has_control = element_id is not None or selector is not None
        control_mode = mic_mod.validate_control_mode(control_mode, has_target=has_control)
        if pre_roll_ms < 0 or post_roll_ms < 0:
            raise UsageError(
                "microphone pre-roll and post-roll must be zero or greater",
                code="mic_roll_invalid",
            )

        import tempfile

        with tempfile.TemporaryDirectory(prefix="aua-mic-") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            mic_mod.synthesize_speech(text, wav_path, voice=voice, rate=rate)
            return self.mic_inject(
                wav_path,
                element_id,
                selector=selector,
                control_mode=control_mode,
                pre_roll_ms=pre_roll_ms,
                post_roll_ms=post_roll_ms,
                observe=observe,
                with_image=with_image,
                _action="mic-speak",
                _source="generated with macOS say",
            )

    def double_tap(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        el = self._target(element_id, selector, verb="double-tap")
        cx, cy = self._aim(el)
        step = self._step("double-tap", el)
        with self._acting(_action_mark("double-tap", el)):
            self.device.double_click(cx, cy)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="double-tap", id=el.id, target=[cx, cy]),
            observe,
            with_image,
        )

    def input_text(
        self,
        element_id: int | None = None,
        text: str = "",
        *,
        selector: dict[str, Any] | None = None,
        submit: bool = False,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        el = self._target(element_id, selector, verb="input")
        cx, cy = self._aim(el)
        # The step records the field's SHAPE only — the typed value is never persisted
        # (PRD §6b privacy; observe_action strips `text` defensively too).
        step = self._step("input", el, submit=submit)
        before = el.text
        with self._acting(_action_mark("input", el)):
            self.device.input_text(cx, cy, text, clear=True, submit=submit)
        self._record_action_safe(step)
        verified = self._typed_text_landed(before, text, submit=submit)
        return self._observe(
            ActionResult(
                ok=verified is not False,
                action="input",
                id=el.id,
                detail=text,
                verified=verified,
            ),
            observe,
            with_image,
        )

    def _system_bar_top(self) -> int | None:
        """Top edge of the bottom system bar, or None when it cannot be established.

        Read from the elements of the last analyze, so this costs no device call. The nav
        bar's *background* node carries no label and does not survive parsing, but its
        buttons do — Back / Home / Overview are clickable nodes with content-descriptions
        on the ``system`` window layer, sitting flush against the bottom edge. The
        shallowest of those is the bar's top.

        Returns None on gesture navigation (no buttons to find) and whenever the analyze
        cache has no system elements — for instance after ``analyze --no-system``. None
        means "do not adjust anything", so the worst case is the old behaviour, never a
        differently-wrong aim point.
        """
        cached = self._read_cache()
        if cached is None or not cached.elements:
            return None
        try:
            _w, height = self.device.window_size()
        except Exception:  # pragma: no cover - best effort
            return None
        if height <= 0:
            return None
        # Flush with the bottom edge, and starting inside the bottom band. The band matters:
        # a full-screen system overlay (the expanded shade) or a systemui dialog must not be
        # mistaken for the bar, or we would "clamp" taps in the middle of the screen.
        floor = height * _SYSTEM_BAR_BAND
        tops = [
            el.bounds[1]
            for el in cached.elements
            if el.window == "system" and el.bounds[3] >= height - 2 and el.bounds[1] >= floor
        ]
        return min(tops) if tops else None

    def _aim(self, el: Element) -> tuple[int, int]:
        """Where to touch *el*: its centre, unless the centre is under the system nav bar.

        11 controls across 6 apps in one sweep published accessibility bounds extending
        *below* the nav bar window, which starts at y=1184 on this pool. A touch at the
        element's centre then lands on the bar, and does one of two things, neither of them
        the caller's intent:

        - it is delivered to **Home**, which backgrounds the app under test and loses any
          unsaved input — a state-corrupting side effect, reported as success;
        - it is swallowed while the app stays foregrounded, so "the app is still in the
          foreground" is not evidence that the touch landed.

        Both were silent, which is what makes them expensive: a run loses its journey and
        then reports that the *product* ignored it.

        So aim at the middle of the part of the element that is actually visible above the
        bar. Only ``y`` moves — the horizontal aim is never second-guessed here, so this
        composes with phrase-aiming, which only moves ``x``. When nothing of the element is
        visible, there is no honest aim point and this raises rather than touching the bar:
        an error the caller can act on beats a success that backgrounded the app.

        The element's own layer is checked first. A caller that resolved the system Back or
        Home button *means* the bar, and clamping that would break the one case where the
        centre is right.
        """
        cx, cy = el.center
        if el.window == "system":
            return cx, cy
        bar_top = self._system_bar_top()
        if bar_top is None or cy < bar_top:
            return cx, cy
        top = el.bounds[1]
        if top >= bar_top - 1:
            raise ElementNotFoundError(
                f"element {el.id} lies entirely under the system navigation bar "
                f"(element starts at y={top}, the bar starts at y={bar_top})",
                hint=(
                    "Nothing of it is touchable: a tap there goes to the navigation bar and "
                    "can background the app. Scroll it into view, dismiss what is covering "
                    "it, or act on a visible element instead."
                ),
            )
        aimed = (top + bar_top - 1) // 2
        logger.info(
            "element %s extends under the system bar (y=%d); aiming at y=%d instead of %d",
            el.id,
            bar_top,
            aimed,
            cy,
        )
        return cx, aimed

    def _typed_text_landed(self, before: str | None, text: str, *, submit: bool) -> bool | None:
        """Did the text actually go in? ``True`` / ``False`` / ``None`` for "cannot tell".

        `input` was the last command that could report `ok:true` having done nothing at all —
        the family the rest of this tool has already closed off (`tap` re-analyzes, `record`
        consults `ps`, `emulator stop` checks the running list). It was seen repeatedly across
        a sweep: a field that never took the text, reported as success, and the lane read the
        empty result as the product ignoring it.

        Only one situation is unambiguous enough to *fail*: the field is readable, it still
        holds exactly what it held before, and it does not contain what we typed. Everything
        else stays `ok` with an honest ``None``, because plenty of fields legitimately do not
        read back what you typed:

        - ``submit=True`` sends the value and leaves the field empty — the common case for a
          chat composer, and failing it would be far worse than the bug;
        - password fields report a mask, or nothing at all;
        - fields that reformat as you type (phone numbers, card numbers, dates) or truncate
          at ``maxLength`` hold a *transformed* value, which is still a successful input.

        An unreadable field is ``None``, never "unchanged" — the same rule as everywhere else
        here: never turn a failed observation into a verdict.
        """
        if not text or submit:
            return None
        after = self.device.focused_text()
        if after is None:
            return None
        if text in after:
            return True
        if after == (before or ""):
            logger.warning(
                "input did not change the field (still %d chars); reporting the truth",
                len(after),
            )
            return False
        return None

    def clear(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        el = self._target(element_id, selector, verb="clear")
        cx, cy = el.center
        step = self._step("clear", el)
        with self._acting(_action_mark("clear", el)):
            self.device.click(cx, cy)
            self.device.clear_text()
        self._record_action_safe(step)
        return self._observe(ActionResult(ok=True, action="clear", id=el.id), observe, with_image)

    # ------------------------------------------------------------- scroll internals

    def _dump(self) -> str:
        return self.platform.dump_tree(self.device)

    def _scroll_box(
        self, *, from_id: int | None = None, selector: dict[str, Any] | None = None, xml: str = ""
    ) -> tuple[Box, bool]:
        """``(box, is_real_container)`` — where a directional swipe should actually happen.

        Swiping the middle of the *screen* is why scrolling "did nothing": on a screen whose
        list occupies a sub-rectangle (a sheet, a tab body, a pane under a fixed header) the
        gesture lands outside the scrollable and gets thrown away. So aim at the scrollable
        container: the one under the anchor element when given, else the biggest on screen.
        """
        device = self.device
        w, h = device.window_size()
        screen: Box = (0, 0, w, h)
        boxes = scrollable_boxes(xml or self._dump(), (w, h))
        if not boxes:
            return screen, False
        anchor = None
        if from_id is not None or selector:
            anchor = self._target(from_id, selector, verb="swipe").center
        if anchor is not None:
            inside = [b for b in boxes if _contains(b, anchor)]
            if inside:  # innermost container under the anchor wins (nested scrollables)
                return min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), True
            return (anchor[0], anchor[1], anchor[0], anchor[1]), False
        return boxes[0], True

    def _swipe_path(self, box: Box, direction: str, percent: int) -> tuple[int, int, int, int]:
        """Swipe endpoints spanning *percent* of *box*, inset from its edges.

        The inset matters: a gesture that starts on the very edge of a list is grabbed by
        the system's back/notification gestures instead of the list.
        """
        w, h = self.device.window_size()
        x1b, y1b, x2b, y2b = box
        cx, cy = (x1b + x2b) // 2, (y1b + y2b) // 2
        span_x = max(1, int((x2b - x1b) * min(percent, 90) / 200))
        span_y = max(1, int((y2b - y1b) * min(percent, 90) / 200))
        d = direction.lower()
        if d == "up":
            path = (cx, cy + span_y, cx, cy - span_y)
        elif d == "down":
            path = (cx, cy - span_y, cx, cy + span_y)
        elif d == "left":
            path = (cx + span_x, cy, cx - span_x, cy)
        elif d == "right":
            path = (cx - span_x, cy, cx + span_x, cy)
        else:
            raise UsageError(f"unknown swipe direction '{direction}'", hint="up|down|left|right")
        x1, y1, x2, y2 = path
        clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
        return (
            clamp(x1, 1, w - 2),
            clamp(y1, 1, h - 2),
            clamp(x2, 1, w - 2),
            clamp(y2, 1, h - 2),
        )

    def _settle_after_swipe(self) -> None:
        """Let a fling finish before probing, or every scroll reads as "barely moved"."""
        from . import imaging

        device = self.device
        gs = imaging.GridSettle(streak=2)
        deadline = time.monotonic() + 0.9
        while time.monotonic() < deadline:
            try:
                if gs.feed(device.screenshot()):
                    return
            except Exception:
                break
            time.sleep(0.035)
        time.sleep(0.05)

    def _probe(self, box: Box) -> Sample:
        return region_probe(self._dump(), box, ignore_packages=self.config.memory.ignore_packages)

    def _swipe_once(self, box: Box, direction: str, percent: int) -> tuple[int, bool]:
        """One verified swipe inside *box*: ``(distance_along_axis, scrolled)``.

        ``travel``'s own ``moved`` is "the sample differs at all" — a repaint, a ripple, or an
        element appearing all set it, none of which mean the content scrolled. Reporting that
        as movement is how `scroll --direction down` came to answer ``moved steps=1`` with no
        distance at the very top of a list, where scrolling further is impossible. The verdict
        is therefore the measured shift ALONG THE AXIS the caller asked about; a changed sample
        with zero shift is honestly "did not scroll".
        """
        before = self._probe(box)
        x1, y1, x2, y2 = self._swipe_path(box, direction, percent)
        self.device.swipe(x1, y1, x2, y2)
        self._settle_after_swipe()
        dx, dy, _changed = travel(before, self._probe(box))
        distance = dx if direction in ("left", "right") else dy
        return distance, bool(distance)

    def swipe(
        self,
        direction: str | None = None,
        *,
        from_id: int | None = None,
        selector: dict[str, Any] | None = None,
        percent: int = 70,
        coords: tuple[int, int, int, int] | None = None,
        observe: bool = True,
        verify: bool = False,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        device = self.device
        if coords is not None:
            x1, y1, x2, y2 = coords
            step = self._step("swipe", arg="coords")
            with self._acting(f"swipe:{direction or 'coords'}"):
                device.swipe(x1, y1, x2, y2)
            self._record_action_safe(step)
            return self._observe(
                ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]),
                observe,
                with_image,
            )
        if direction is None:
            raise UsageError(
                "swipe needs a direction or --coords", hint="e.g. `aua swipe-and-analyze up`"
            )
        d = direction.lower()
        box, real = self._scroll_box(from_id=from_id, selector=selector)
        x1, y1, x2, y2 = self._swipe_path(box, d, percent)
        step = self._step("swipe", arg=d)
        if not verify:
            with self._acting(f"swipe:{d}"):
                device.swipe(x1, y1, x2, y2)
                # Only settle here when the caller skipped observe — otherwise
                # ``_observe`` already does change+idle and a second settle doubles latency.
                if not observe:
                    self._settle_after_swipe()
            self._record_action_safe(step)
            return self._observe(
                ActionResult(ok=True, action="swipe", target=[x1, y1, x2, y2]), observe, with_image
            )
        with self._acting(f"swipe:{d}"):
            distance, moved = self._swipe_once(box, d, percent)
        self._record_action_safe(step)
        # ok stays True — the gesture WAS performed, and a swipe is also used to dismiss or
        # page things where "the screen did not move" is the expected outcome. The verdict
        # is reported instead of swallowed; `aua scroll-and-analyze` is the strict-exit-code variant.
        return self._observe(
            ActionResult(
                ok=True,
                action="swipe",
                target=[x1, y1, x2, y2],
                detail=detail_tokens(
                    "moved" if moved else "no-change",
                    dy=abs(distance) if moved and distance else None,
                    scrollable=str(real).lower(),
                ),
            ),
            observe,
            with_image,
        )

    def scroll(
        self,
        direction: str | None = None,
        *,
        pages: int = 1,
        to_end: bool = False,
        to_start: bool = False,
        from_id: int | None = None,
        selector: dict[str, Any] | None = None,
        percent: int = 70,
        max_steps: int = 25,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        """Scroll a container and report what actually happened.

        Outcomes in ``detail`` (first token): ``moved`` · ``reached-end`` (moved, then ran
        out of content) · ``already-at-end`` (the very first swipe changed nothing). The
        first two are ``ok``; ``already-at-end`` is ``ok`` only for ``--to-end/--to-start``,
        where being at the end IS the postcondition. Everywhere else a scroll that moved
        nothing is a failure, because "nothing left to scroll" and "my swipe missed the
        list" must not look the same to a caller looping until something appears.
        """
        if to_end and to_start:
            raise UsageError("--to-end and --to-start are mutually exclusive")
        if direction is None:
            direction = "down" if to_start else "up"
        d = direction.lower()
        if d not in ("up", "down", "left", "right"):
            raise UsageError(f"unknown scroll direction '{direction}'", hint="up|down|left|right")
        limit = max_steps if (to_end or to_start) else max(1, pages)
        box, real = self._scroll_box(from_id=from_id, selector=selector)
        step = self._step("scroll", arg=d)
        travelled = 0
        steps = 0
        with self._acting():
            for _ in range(limit):
                dy, moved = self._swipe_once(box, d, percent)
                if not moved:
                    break
                steps += 1
                travelled += abs(dy)
        self._record_action_safe(step)
        at_end = steps < limit
        if steps == 0:
            outcome = "already-at-end"
        elif at_end:
            outcome = "reached-end"
        else:
            outcome = "moved"
        ok = steps > 0 or to_end or to_start
        return self._observe(
            ActionResult(
                ok=ok,
                action="scroll",
                target=list(box),
                detail=detail_tokens(
                    outcome,
                    steps=steps,
                    dy=travelled or None,
                    direction=d,
                    scrollable=str(real).lower(),
                ),
            ),
            observe,
            with_image,
        )

    def scroll_to(
        self,
        query: str,
        *,
        match: str = "contains",
        ignore_case: bool = False,
        observe: bool = True,
        by: str = "text",
        direction: str = "up",
        max_swipes: int = 10,
        percent: int = 70,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        """Scroll until *query* is on screen, verifying every swipe actually moved.

        Outcomes in ``detail`` (first token): ``already-visible`` · ``moved`` (found it,
        with ``dy``) · ``already-at-end`` (nothing scrolled, so the target is simply not on
        this screen) · ``target-not-found`` (scrolled the whole way and never saw it).
        Only the first two are ``ok`` — the old version returned ``ok:false`` with exit 0,
        which is the same as saying nothing at all to an automated caller.
        """
        mode = MatchMode(match)
        step = self._step("scroll-to", arg=query)

        def locate() -> tuple[int, int, int, int] | None:
            return self.device.find_text(query, match=mode, ignore_case=ignore_case, by=by)

        found = locate()
        if found is not None:
            return self._observe(
                ActionResult(
                    ok=True,
                    action="scroll-to",
                    detail=detail_tokens("already-visible", target=query),
                    target=list(found),
                ),
                observe,
                with_image,
            )
        box, real = self._scroll_box()
        travelled = 0
        steps = 0
        exhausted = True
        with self._acting():
            for _ in range(max(1, max_swipes)):
                dy, moved = self._swipe_once(box, direction, percent)
                if moved:
                    steps += 1
                    travelled += abs(dy)
                found = locate()
                if found is not None or not moved:
                    exhausted = False
                    break
        self._record_action_safe(step)
        if found is not None:
            outcome = "moved"
        elif steps == 0:
            outcome = "already-at-end"
        else:
            outcome = "target-not-found"
        return self._observe(
            ActionResult(
                ok=found is not None,
                action="scroll-to",
                detail=detail_tokens(
                    outcome,
                    target=query,
                    steps=steps,
                    dy=travelled or None,
                    scrollable=str(real).lower(),
                    exhausted="true" if (found is None and exhausted) else None,
                ),
                target=list(found) if found else None,
            ),
            observe,
            with_image,
        )

    def key(
        self,
        name: str,
        *,
        observe: bool = True,
        with_image: bool | str | None = None,
        _hierarchy_settle: bool = False,
    ) -> ActionResult:
        candidate = name.strip()
        known = (
            candidate.lower() in _KEY_NAMES
            or candidate.upper().startswith("KEYCODE_")
            or candidate.isdigit()
        )
        if not known:
            raise UsageError(
                f"unknown key '{name}'",
                hint="Valid: "
                + ", ".join(sorted(_KEY_NAMES))
                + ", KEYCODE_*, or a keycode number.",
            )
        step = self._step("key", arg=name)
        with self._acting(f"key:{name}", capture_pre_action=not _hierarchy_settle):
            self.device.press(name)
        self._record_action_safe(step)
        return self._observe(ActionResult(ok=True, action="key", detail=name), observe, with_image)

    def back_until(
        self,
        predicate: str,
        *,
        back_id: int | None = None,
        back_selector: dict[str, str] | None = None,
        max_steps: int = 4,
        step_timeout_ms: int = 1_200,
        poll_ms: int = 200,
    ) -> ActionResult:
        """Navigate back until mapped-screen or semantic UI evidence is present.

        Each fresh frame is checked for an unambiguous toolbar Back/Navigate-up affordance and
        that stable selector is preferred; hardware Back is the fallback. This matters on nested
        Compose screens that consume the hardware event. The predicate is validated before the
        first mutation, every step is observed, and leaving the starting package stops the
        journey. Cross-package traversal belongs in a risk-preflighted route or flow.
        """
        raw_destination = (predicate or "").strip()
        known_screen_target = (
            raw_destination if re.fullmatch(r"[A-Za-z0-9_.-]+", raw_destination or "") else None
        )
        terms = [] if known_screen_target else _parse_await_terms(predicate)
        unsupported = sorted({term.by for term in terms if term.by in {"net", "log"}})
        if unsupported:
            raise UsageError(
                "back-until needs screen evidence, not off-screen evidence",
                hint="Use text:, rid:, or desc: terms so AUA knows where Back arrived.",
            )
        if not known_screen_target and not any(not term.negated for term in terms):
            raise UsageError(
                "back-until needs at least one positive destination term",
                hint="Add text:, rid:, or desc: evidence for the screen you want to reach.",
            )
        if back_selector is not None and not isinstance(back_selector, dict):
            raise UsageError("back_selector must be an object with one of rid, text, or desc")
        selector = {key: value for key, value in (back_selector or {}).items() if value}
        if back_id is not None and selector:
            raise UsageError("choose either back_id or back_selector, not both")
        if back_id is not None and back_id < 0:
            raise UsageError("back_id must be a non-negative id from the current observation")
        if selector and (len(selector) != 1 or next(iter(selector)) not in {"rid", "text", "desc"}):
            raise UsageError("back_selector must contain exactly one of rid, text, or desc")
        if not 1 <= max_steps <= 12:
            raise UsageError("back-until --max-steps must be between 1 and 12")
        if step_timeout_ms < 0 or poll_ms < 10:
            raise UsageError("back-until timeouts must be non-negative and poll at least 10ms")

        # Bind an explicit ordinal to the caller's cached frame before the predicate precheck
        # analyzes again. The ordinal itself is never identity: it must be remapped from this
        # original element onto the fresh precheck observation before any action is authorized.
        back_binding: Element | None = None
        if back_id is not None:
            cached = self._read_cache()
            back_binding = cached.element_by_id(back_id) if cached is not None else None

        started_at = time.monotonic()
        device = self.device

        def wait_destination(timeout_ms: int) -> ActionResult:
            if known_screen_target:
                return self._await_known_screen(
                    known_screen_target, timeout_ms=timeout_ms, poll_ms=poll_ms
                )
            return self.await_predicate(
                predicate,
                timeout_ms=timeout_ms,
                poll_ms=poll_ms,
                observe=True,
                rich_ui=False,
                hierarchy_only=True,
            )

        def refresh_weak_terminal(
            result: ActionResult,
            before_observation: AnalyzeResult | None,
        ) -> ActionResult:
            """Replace a half-attached success frame with one authoritative hierarchy read."""
            if not result.ok or not self._back_terminal_frame_is_weak(
                before_observation, result.observation
            ):
                return result
            try:
                fresh = self.analyze(
                    source="hierarchy",
                    with_ocr=False,
                    no_cache=True,
                    record=False,
                )
            except Exception:  # noqa: BLE001 - the already-proven result remains valid evidence
                return result
            if known_screen_target:
                actual = self._recognize_screen_read_only(fresh)
                fresh.meta.known_screen = actual
                still_satisfied = bool(
                    actual and actual.casefold() == known_screen_target.casefold()
                )
                refreshed_terms = [
                    {
                        "term": f"screen:{known_screen_target}",
                        "present": still_satisfied,
                        "satisfied": still_satisfied,
                    }
                ]
            else:
                refreshed_terms = self._await_terms_on_observation(
                    terms,
                    [{} for _term in terms],
                    fresh,
                    mode=MatchMode.contains,
                    ignore_case=True,
                )
                still_satisfied = all(row["satisfied"] for row in refreshed_terms)
            if still_satisfied:
                result.observation = fresh
                result.observation_present = True
                result.await_terms = refreshed_terms
                return result
            # The tiny frame's positive evidence disappeared on the authoritative reread.
            # Returning the original ok=True would certify a transient title that is no longer
            # present, exactly the false-success this refresh exists to prevent.
            result.ok = False
            result.detail = (
                "authoritative terminal reread no longer satisfies the destination evidence"
            )
            result.observation = fresh
            result.observation_present = True
            result.await_terms = refreshed_terms
            result.await_outcome = "settled-unmet"
            result.verified = False
            return result

        current = wait_destination(0)
        origin_package = str(
            current.observation.screen.package if current.observation is not None else ""
        )
        if not origin_package:
            origin_package = str((device.current_app() or {}).get("package") or "")
        if current.ok:
            current = refresh_weak_terminal(current, None)
            if not current.ok:
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="terminal_evidence_unmet",
                    detail=current.detail or "terminal destination evidence disappeared",
                    steps_run=[],
                    started_at=started_at,
                )
            current.action = "back-until"
            current.detail = "destination already satisfied; steps=0"
            current.stop_reason = "already_satisfied"
            current.steps_run = []
            current.verified = True
            return current

        steps_run: list[dict[str, Any]] = []
        for steps in range(1, max_steps + 1):
            before_observation = current.observation
            before = self._back_observation_identity(current.observation)
            requested_id: int | None = None
            explicit_id_invalid = False
            if steps == 1 and back_id is not None:
                if back_binding is None or current.observation is None:
                    explicit_id_invalid = True
                else:
                    from .identity import remap_ids

                    requested_id = remap_ids([back_binding], current.observation.elements).get(
                        back_binding.id
                    )
                    explicit_id_invalid = requested_id is None
            status, selected, frame_id = self._semantic_back_selector(
                current.observation,
                selector or None,
                frame_id=requested_id,
            )
            if explicit_id_invalid:
                status = "invalid"
            if status == "ambiguous":
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="ambiguous_back_affordance",
                    detail="several Back/Navigate-up controls are visible; supply --back-rid/text/desc",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if status == "invalid":
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="no_back_affordance",
                    detail="the explicit Back id is not a fresh enabled app-owned control",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if selected is None and self._mapped_screen_is_root(current.observation):
                # Never spend a hardware Back from a mapped route root on the strength of a
                # transient predicate miss. Recheck the exact destination once on a fresh frame;
                # if it remains unmet, return the in-package boundary instead of crossing it.
                rechecked = wait_destination(0)
                package = self._back_observed_package(rechecked, device)
                if origin_package and package and package != origin_package:
                    return self._back_until_result(
                        rechecked,
                        ok=False,
                        reason="package_changed",
                        detail=f"foreground left {origin_package} for {package}",
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                if rechecked.ok:
                    rechecked = refresh_weak_terminal(rechecked, before_observation)
                    if not rechecked.ok:
                        return self._back_until_result(
                            rechecked,
                            ok=False,
                            reason="terminal_evidence_unmet",
                            detail=(
                                rechecked.detail or "terminal destination evidence disappeared"
                            ),
                            steps_run=steps_run,
                            started_at=started_at,
                        )
                    return self._back_until_result(
                        rechecked,
                        ok=True,
                        reason=("already_satisfied" if not steps_run else "predicate_satisfied"),
                        detail=(
                            "destination already satisfied; steps=0"
                            if not steps_run
                            else f"satisfied after {len(steps_run)} back-navigation step(s)"
                        ),
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                root_still_visible = self._mapped_screen_is_root(rechecked.observation)
                return self._back_until_result(
                    rechecked,
                    ok=False,
                    reason="package_boundary_risk" if root_still_visible else "screen_unstable",
                    detail=(
                        "destination evidence is still unmet on a mapped route root; stopped "
                        "before hardware Back could leave the app"
                        if root_still_visible
                        else "the mapped root changed during its safety recheck; stopped before "
                        "hardware Back could act on an unchecked frame"
                    ),
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if selected is not None:
                try:
                    if frame_id is not None:
                        # The unlabeled top-left navigation affordance has no semantic selector.
                        # Its id came from this exact observation, and `tap` immediately remaps
                        # that binding against a fresh hierarchy before touching the device.
                        self.tap(element_id=frame_id, observe=False, _hierarchy_settle=True)
                    else:
                        self.tap(selector=selected, observe=False, _hierarchy_settle=True)
                except SelectorAmbiguousError:
                    return self._back_until_result(
                        current,
                        ok=False,
                        reason="ambiguous_back_affordance",
                        detail="the selected Back affordance became ambiguous before action",
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                except (ElementNotFoundError, SelectorNotFoundError, StaleElementIdError):
                    return self._back_until_result(
                        current,
                        ok=False,
                        reason="no_back_affordance",
                        detail="the selected Back affordance disappeared before action",
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                via = "affordance"
            else:
                self.key("back", observe=False, _hierarchy_settle=True)
                via = "hardware"
            step_deadline = time.monotonic() + (step_timeout_ms / 1000.0)
            current = wait_destination(step_timeout_ms)

            # `await_predicate` deliberately returns screen-changed before trusting terms from
            # a newly resumed Activity. Re-evaluate on that Activity before sending another
            # navigation action, or a destination that just appeared could be overshot. A chain
            # of Activity transitions is observed only within this step's original deadline;
            # if it remains unstable, stop rather than act on an unchecked screen.
            transition_rechecks = 0
            while not current.ok and current.await_outcome == "screen-changed":
                package = self._back_observed_package(current, device)
                if origin_package and package and package != origin_package:
                    steps_run.append(
                        self._back_step_evidence(
                            index=steps - 1,
                            via=via,
                            selector=selected,
                            before=before,
                            observation=current.observation,
                        )
                    )
                    return self._back_until_result(
                        current,
                        ok=False,
                        reason="package_changed",
                        detail=f"foreground left {origin_package} for {package}",
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                remaining_ms = max(0, int((step_deadline - time.monotonic()) * 1000))
                if transition_rechecks >= 3 or (transition_rechecks and remaining_ms == 0):
                    steps_run.append(
                        self._back_step_evidence(
                            index=steps - 1,
                            via=via,
                            selector=selected,
                            before=before,
                            observation=current.observation,
                        )
                    )
                    return self._back_until_result(
                        current,
                        ok=False,
                        reason="screen_unstable",
                        detail="screen kept changing before destination evidence could be checked",
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                current = wait_destination(remaining_ms)
                transition_rechecks += 1
            package = self._back_observed_package(current, device)
            steps_run.append(
                self._back_step_evidence(
                    index=steps - 1,
                    via=via,
                    selector=selected,
                    before=before,
                    observation=current.observation,
                )
            )
            if origin_package and package and package != origin_package:
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="package_changed",
                    detail=f"foreground left {origin_package} for {package}",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            after = self._back_observation_identity(current.observation)
            if current.ok:
                current = refresh_weak_terminal(current, before_observation)
                if not current.ok:
                    return self._back_until_result(
                        current,
                        ok=False,
                        reason="terminal_evidence_unmet",
                        detail=current.detail or "terminal destination evidence disappeared",
                        steps_run=steps_run,
                        started_at=started_at,
                    )
                return self._back_until_result(
                    current,
                    ok=True,
                    reason="predicate_satisfied",
                    detail=f"satisfied after {steps} back-navigation step(s)",
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if known_screen_target and (
                current.observation is None or not current.observation.meta.known_screen
            ):
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="screen_unrecognized",
                    detail=(
                        "the post-Back frame was not recognized by the app map; stopped "
                        "rather than risk overshooting the requested screen"
                    ),
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if known_screen_target and self._mapped_screen_state(current.observation) == "loading":
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="screen_unstable",
                    detail=(
                        "the post-Back frame is still a mapped loading state; stopped rather "
                        "than navigate again before it settles"
                    ),
                    steps_run=steps_run,
                    started_at=started_at,
                )
            if before and after and before == after:
                retry_hint = (
                    "; reuse this returned observation and retry once with --back-id <fresh-id> "
                    "only if that id is visibly the app-owned unlabeled Back control, or use "
                    "--back-rid/--back-desc for a semantic Back control"
                    if via == "hardware"
                    else ""
                )
                return self._back_until_result(
                    current,
                    ok=False,
                    reason="no_progress",
                    detail=f"{via} Back produced no semantic screen change{retry_hint}",
                    steps_run=steps_run,
                    started_at=started_at,
                )

        return self._back_until_result(
            current,
            ok=False,
            reason="max_steps",
            detail=f"destination unmet after max_steps={max_steps}",
            steps_run=steps_run,
            started_at=started_at,
        )

    def _recognize_screen_read_only(self, observation: AnalyzeResult) -> str | None:
        """Recognize one hierarchy frame without recording or mutating app memory."""
        package = observation.screen.package or ""
        memory = self._memory
        if memory is None or not package or memory.load(package) is None:
            return None
        return memory.recognize_screen(
            self.device.serial,
            package=package,
            elements=observation.elements,
            activity=observation.screen.activity,
            screen_height=observation.screen.height,
        )

    def _await_known_screen(self, target: str, *, timeout_ms: int, poll_ms: int) -> ActionResult:
        """Observe hierarchy frames until memory recognizes *target*, within one Back step."""
        started_at = time.monotonic()
        deadline = started_at + max(0.0, timeout_ms / 1000.0)
        checks = 0
        while True:
            checks += 1
            observation = self.analyze(source="hierarchy", with_ocr=False, record=False)
            package = observation.screen.package or ""
            memory = self._memory
            app = memory.load(package) if memory is not None and package else None
            if memory is None or app is None or target not in app.screens:
                scope = package or "the foreground app"
                raise UsageError(
                    f"{target!r} is not a mapped screen for {scope}",
                    hint=(
                        "Use `aua map --find <goal>` to discover exact screen names, or pass "
                        "positive text:/rid:/desc: destination evidence."
                    ),
                )
            resolved_target = target
            actual = self._recognize_screen_read_only(observation)
            # `record=False` intentionally leaves map metadata blank. Surface the result of
            # this read-only anchor recognition so the caller can reuse the final frame and so
            # the next Back is allowed only from a stable mapped intermediate screen.
            observation.meta.known_screen = actual
            satisfied = bool(actual and actual.casefold() == resolved_target.casefold())
            elapsed = int((time.monotonic() - started_at) * 1000)
            if satisfied or time.monotonic() >= deadline:
                outcome = "satisfied" if satisfied else "timeout"
                return ActionResult(
                    ok=satisfied,
                    action="await",
                    detail=(
                        f"{outcome} after {elapsed}ms ({checks} checks)"
                        + ("" if satisfied else f"; current screen: {actual or 'unknown'}")
                    ),
                    observation=observation,
                    observation_present=True,
                    await_outcome=outcome,
                    await_terms=[
                        {
                            "term": f"screen:{resolved_target}",
                            "present": satisfied,
                            "satisfied": satisfied,
                        }
                    ],
                    elapsed_ms=elapsed,
                )
            time.sleep(max(0.01, poll_ms / 1000.0))

    def _mapped_screen_state(self, observation: AnalyzeResult | None) -> str | None:
        """Return a recognized screen's remembered state without changing app memory."""
        if observation is None or not observation.meta.known_screen:
            return None
        package = observation.screen.package or ""
        memory = self._memory
        app = memory.load(package) if memory is not None and package else None
        if app is None:
            return None
        record = app.screens.get(observation.meta.known_screen)
        return record.state if record is not None else None

    def _mapped_screen_is_root(self, observation: AnalyzeResult | None) -> bool:
        """Whether this exact frame is a recognized in-app route root.

        Hardware Back from a mapped root can leave the package. The route map gives us a
        conservative boundary without guessing from toolbar geometry; absent or incomplete
        memory returns False and preserves the existing bounded hardware behavior.
        """
        if observation is None or not observation.meta.known_screen:
            return False
        package = observation.screen.package or ""
        memory = self._memory
        app = memory.load(package) if memory is not None and package else None
        if memory is None or app is None:
            return False
        context_id: str | None = None
        with contextlib.suppress(Exception):
            session = memory.load_session(observation.meta.device_serial or self.device.serial)
            if session.package == package:
                context_id = session.active_context_id
        return screen_is_root(app, observation.meta.known_screen, context_id)

    @staticmethod
    def _back_terminal_frame_is_weak(
        before: AnalyzeResult | None,
        after: AnalyzeResult | None,
    ) -> bool:
        """Detect a half-attached terminal hierarchy without penalising truly sparse screens."""
        if after is None:
            return True
        before_count = len(before.elements) if before is not None else 0
        after_count = len(after.elements)
        if before_count >= 8 and after_count * 3 < before_count:
            return True
        if after_count == 0:
            return False
        usable = any(
            element.window in {None, "app"}
            and (
                element.clickable is True
                or bool((element.text or "").strip())
                or bool((element.content_desc or "").strip())
                or bool((element.resource_id or "").strip())
            )
            for element in after.elements
        )
        return not usable

    def _observation_is_loading(self, observation: AnalyzeResult | None) -> bool:
        """Conservative signal that a wrong-screen verdict would be premature."""
        if observation is None:
            return False
        if self._mapped_screen_state(observation) == "loading":
            return True
        for element in observation.elements:
            kind = (element.type or "").casefold()
            if kind.endswith("progressbar"):
                return True
            label = " ".join(
                value for value in (element.text, element.content_desc) if value
            ).strip()
            if re.search(r"\b(?:loading|please wait)\b", label, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _semantic_back_selector(
        observation: AnalyzeResult | None,
        override: dict[str, str] | None = None,
        *,
        frame_id: int | None = None,
    ) -> tuple[str, dict[str, str] | None, int | None]:
        """One app-owned Back selector, plus none/ambiguous status."""
        if override:
            return "one", override, None
        if observation is None:
            return "none", None, None
        if frame_id is not None:
            element = observation.element_by_id(frame_id)
            if element is None:
                return "invalid", None, None
            if (
                element.clickable is not True
                or element.enabled is False
                or element.window not in {None, "app"}
            ):
                return "invalid", None, None
            return "one", {"frame_id": str(frame_id)}, frame_id
        candidates: list[tuple[dict[str, str], int | None]] = []
        bottom = int(observation.screen.height * _SYSTEM_BAR_BAND)
        for element in observation.elements:
            if element.clickable is not True or element.enabled is False:
                continue
            if element.window not in {None, "app"}:
                continue
            rid = (element.resource_id or "").strip()
            if (
                ":" in rid
                and observation.screen.package
                and not rid.startswith(observation.screen.package + ":")
            ):
                continue
            if element.bounds[1] >= bottom:
                continue
            desc = (element.content_desc or "").strip()
            text = (element.text or "").strip()
            if is_back_resource_id(rid):
                candidates.append(({"rid": rid}, None))
            elif desc.casefold() in {"back", "navigate up", "up"}:
                candidates.append(({"desc": desc}, None))
            elif text.casefold() == "back":
                candidates.append(({"text": text}, None))
        if not candidates:
            return "none", None, None
        if len(candidates) > 1:
            return "ambiguous", None, None
        selected, frame_id = candidates[0]
        return "one", selected, frame_id

    @staticmethod
    def _back_observation_identity(observation: AnalyzeResult | None) -> str | None:
        if observation is None:
            return None
        labels = tuple(
            (
                (element.resource_id or "")[:80],
                (element.content_desc or "")[:80],
                (element.text or "")[:80],
                element.bounds,
            )
            for element in observation.elements
            if element.resource_id or element.content_desc or element.text
        )
        fingerprint = hashlib.sha256(
            repr((observation.screen.package, observation.meta.known_screen, labels)).encode()
        ).hexdigest()[:12]
        if observation.meta.known_screen:
            return f"{observation.meta.known_screen}:{fingerprint}"
        return fingerprint

    @staticmethod
    def _back_observed_package(current: ActionResult, device: Device) -> str:
        return str(
            (current.observation.screen.package if current.observation is not None else "")
            or (device.current_app() or {}).get("package")
            or ""
        )

    @classmethod
    def _back_step_evidence(
        cls,
        *,
        index: int,
        via: str,
        selector: dict[str, str] | None,
        before: str | None,
        observation: AnalyzeResult | None,
    ) -> dict[str, Any]:
        after = cls._back_observation_identity(observation)
        return {
            "index": index,
            "via": via,
            **({"selector": selector} if selector is not None else {}),
            "from_screen": before,
            "to_screen": after,
            "changed": bool(before and after and before != after),
        }

    @staticmethod
    def _back_until_result(
        current: ActionResult,
        *,
        ok: bool,
        reason: str,
        detail: str,
        steps_run: list[dict[str, Any]],
        started_at: float,
    ) -> ActionResult:
        current.ok = ok
        current.action = "back-until"
        current.detail = detail
        current.stop_reason = reason
        current.steps_run = steps_run
        current.elapsed_ms = int((time.monotonic() - started_at) * 1000)
        current.verified = ok
        # Every other observed action reports arrival in the top-level `known_screen`; this one
        # hid it inside `await_terms`, so a caller reading the documented field got None for a
        # call that fully succeeded. Fall back to the observation's own answer.
        if current.known_screen is None and current.observation is not None:
            meta = getattr(current.observation, "meta", None)
            if meta is not None:
                current.known_screen = meta.known_screen
        return current

    def hide_keyboard(
        self, *, observe: bool = True, with_image: bool | str | None = None
    ) -> ActionResult:
        """Dismiss the soft keyboard (Maestro ``hideKeyboard``).

        Prefer this over ``key back`` when the IME is covering the tree — back can
        leave the screen; hide-keyboard aims to only dismiss the keyboard.
        """
        step = self._step("hide-keyboard")
        with self._acting("hide-keyboard"):
            self.device.hide_keyboard()
        self._record_action_safe(step)
        # Verify, don't assume. This returned ok=True unconditionally, and an IME that stays
        # up is not a cosmetic miss: it covers the bottom of the screen, so the button the
        # caller is trying to reach is hidden while the command says the keyboard is gone.
        shown = self._ime_shown()
        detail = "hidden" if shown is False else ("still-shown" if shown else "unknown")
        return self._observe(
            ActionResult(ok=shown is not True, action="hide-keyboard", detail=detail),
            observe,
            with_image,
        )

    def _ime_shown(self) -> bool | None:
        """Is the soft keyboard up? ``None`` when the device will not say.

        Tri-state on purpose: "cannot tell" must not read as "hidden", or this check would
        recreate the very false-success it exists to catch.
        """
        try:
            out = self.device.shell("dumpsys input_method | grep -m1 mInputShown")
        except Exception:  # pragma: no cover - device/shell unavailable
            return None
        text = (out or "").strip()
        if "mInputShown=true" in text:
            return True
        if "mInputShown=false" in text:
            return False
        return None

    def open_link(
        self,
        uri: str,
        *,
        package: str | None = None,
        prefer: str | None = None,
        pin_package: bool = True,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        """Open a deeplink URI (jump straight to a screen / trigger an app action).

        By default pins the VIEW intent to the foreground/known package so Android's
        "Open with…" chooser never appears when both prod + dev builds are installed.
        Pass ``pin_package=False`` (CLI ``--no-package-pin``) to deliberately exercise
        the chooser. If a chooser still appears after open, raises :class:`DeviceError`
        naming the competing app rows — never leaves the caller stranded on the dialog.
        """
        target_pkg = package or prefer
        if pin_package and not target_pkg:
            target_pkg = self.current_package() or self._cached_package()
        step = self._step("open-link", arg=uri)
        with self._acting():
            self.device.open_link(uri, package=target_pkg if pin_package else None)
        time.sleep(0.35)  # chooser / activity settle
        detail = uri if not target_pkg else f"{uri} → {target_pkg}"
        if self._is_chooser():
            competitors = self._chooser_app_labels()
            if pin_package and target_pkg and self._dismiss_chooser(prefer=target_pkg):
                time.sleep(0.25)
            if self._is_chooser():
                listing = ", ".join(competitors) if competitors else "(unknown handlers)"
                raise DeviceError(
                    "deeplink opened the system 'Open with…' chooser",
                    hint=(
                        f"Competing apps on screen: {listing}. "
                        f"Re-run with `--package <id>` (e.g. the foreground "
                        f"`{self.current_package() or 'com.example.app'}`), or "
                        f"`--no-package-pin` only when you intentionally want the chooser."
                    ),
                )
            detail = f"{uri} (chooser→{target_pkg or 'picked'})"
        self._record_action_safe(step)
        self._remember_deeplink_safe(uri, package=target_pkg)
        self._remember_pending_flag_context(uri, target_pkg)
        result = self._observe(
            ActionResult(ok=True, action="open-link", detail=detail), observe, with_image
        )
        self._flag_deeplink_that_did_not_land(result, uri)
        return result

    def _flag_deeplink_that_did_not_land(self, result: ActionResult, uri: str) -> None:
        """Say so when the URI was accepted but the app stayed exactly where it was.

        `am start` returning cleanly only means the intent was delivered — an app is free to
        ignore it, and several only honour a deeplink across a restart. The result still read
        `ok: true` with `detail: "<uri> → <package>"`, which is indistinguishable from a jump
        that worked. Measured 2026-08-10: an agent fired a deeplink shortcut, got that response,
        discovered by eye that nothing had moved, called it "misleading", and never reached for
        a shortcut again for the rest of the run — it tapped its way through instead.
        """
        change = result.change if isinstance(result.change, dict) else None
        if not change or change.get("activity_changed") is not False:
            return
        if change.get("changed") is not False:
            return
        result.stale_risk = (
            f"the app accepted {uri} but did not move: same activity, identical tree. The intent "
            "was delivered; this app ignored it from the current state. Some deeplinks only apply "
            "across a restart — `aua app restart-and-analyze <pkg>` then re-open — otherwise "
            "navigate normally."
        )

    def _remember_pending_flag_context(self, uri: str, package: str | None) -> None:
        """Recognize configured raw set-flags links so a later manual launch is scoped."""
        if not package or self._memory is None:
            return
        template = self.config.flags.templates.get(package)
        if not template or "{query}" not in template:
            return
        prefix = template.split("{query}", 1)[0]
        if not uri.startswith(prefix):
            return
        from urllib.parse import parse_qsl, urlsplit

        flags = dict(parse_qsl(urlsplit(uri).query))
        if flags:
            self._memory.set_pending_flags(self.device.serial, package, flags)

    def _is_chooser(self) -> bool:
        """True when the system resolver / 'Open with…' UI is in the foreground."""
        device = self.device
        try:
            app = device.current_app() or {}
        except Exception:
            return False
        pkg = (app.get("package") or "").lower()
        activity = (app.get("activity") or "").lower()
        if (
            "resolver" in activity
            or "intentresolver" in pkg
            or pkg in {"android", "com.android.intentresolver", "com.android.internal.app"}
        ):
            return True
        with contextlib.suppress(Exception):
            xml = self.platform.dump_tree(device)
            if "Open with" in xml or ("Just once" in xml and "Always" in xml):
                return True
        return False

    def _chooser_app_labels(self) -> list[str]:
        """Clickable app-row labels on a chooser screen (best-effort)."""
        skip = {"Just once", "Always", "Open with", "Cancel", "Open"}
        labels: list[str] = []
        with contextlib.suppress(Exception):
            result = self.analyze(source="hierarchy", record=False)
            for el in result.elements:
                label = (el.text or el.content_desc or "").strip()
                if label and label not in skip and el.clickable:
                    labels.append(label)
        return labels

    def _dismiss_chooser(self, *, prefer: str | None = None) -> bool:
        """If the system 'Open with…' resolver is foreground, pick an app and continue."""
        if not self._is_chooser():
            return False
        device = self.device
        # Prefer an explicit package label match, else tap "Just once" on first row.
        try:
            result = self.analyze(source="hierarchy", record=False)
        except Exception:
            return False
        prefer_tail = (prefer or "").rsplit(".", 1)[-1].lower() if prefer else ""
        candidates = [
            el
            for el in result.elements
            if el.clickable
            and (
                (prefer_tail and prefer_tail in (el.text or el.content_desc or "").lower())
                or (el.text or "").strip() not in {"Just once", "Always", "Open with"}
            )
        ]
        target = None
        if prefer_tail:
            for el in candidates:
                hay = f"{el.text or ''} {el.content_desc or ''}".lower()
                if prefer_tail in hay or (prefer or "").lower() in hay:
                    target = el
                    break
        if target is None:
            # First non-chrome row that looks like an app
            for el in result.elements:
                label = (el.text or el.content_desc or "").strip()
                if (
                    label
                    and label not in {"Just once", "Always", "Open with", "Cancel"}
                    and el.clickable
                ):
                    target = el
                    break
        if target is None:
            return False
        x, y = target.center
        device.click(x, y)
        # Confirm "Just once" if still on chooser.
        time.sleep(0.3)
        with contextlib.suppress(Exception):
            again = self.analyze(source="hierarchy", record=False)
            for el in again.elements:
                if (el.text or "").strip() == "Just once" and el.clickable:
                    device.click(*el.center)
                    break
        # Cache only — deliberately NOT `_acting()`. This runs mid-`open_link`, whose window
        # is already open; re-stamping here would move it past the deeplink's own output.
        self._invalidate_cache()
        return True

    def _remember_deeplink_safe(self, uri: str, *, package: str | None = None) -> None:
        mem = self._memory
        if mem is None or self._device is None:
            return
        pkg = package or self._cached_package() or self.current_package()
        if not pkg:
            return
        with contextlib.suppress(Exception):  # playbook is a bonus; never fail the action
            mem.remember_deeplink(pkg, uri, probed=True)

    @staticmethod
    def _await_terms_on_observation(
        terms: list[_AwaitTerm],
        previous: list[dict[str, Any]],
        observation: AnalyzeResult,
        *,
        mode: MatchMode,
        ignore_case: bool,
    ) -> list[dict[str, Any]]:
        """Evaluate UI terms against one exact hierarchy frame.

        The ordinary poll uses ``Device.find_text`` because it is the cheapest possible check.
        Arrival-mismatch detection also needs to prove that the *stable frame* it inspected still
        misses the predicate.  Reusing results from an earlier selector RPC would combine two
        moments and could call a destination wrong while it was still rendering.

        Off-screen ``net:``/``log:`` terms retain their already evaluated value.  In practice the
        early mismatch path is intentionally disabled when a positive off-screen term is present,
        but retaining those rows keeps this helper total and the output order unchanged.
        """

        def matches(candidate: str, wanted: str) -> bool:
            hay = candidate.casefold() if ignore_case else candidate
            needle = wanted.casefold() if ignore_case else wanted
            if mode is MatchMode.exact:
                return hay == needle
            if mode is MatchMode.regex:
                flags = re.IGNORECASE if ignore_case else 0
                return re.search(wanted, candidate, flags) is not None
            return needle in hay

        refreshed: list[dict[str, Any]] = []
        for index, term in enumerate(terms):
            if term.by not in {"text", "rid", "desc"}:
                refreshed.append(dict(previous[index]))
                continue
            present = False
            for element in observation.elements:
                if term.by == "rid":
                    full = element.resource_id or ""
                    values = [full, _id_tail(full) or ""] if full else []
                elif term.by == "desc":
                    values = [element.content_desc or ""]
                else:
                    values = [element.text or "", element.content_desc or ""]
                if any(value and matches(value, term.value) for value in values):
                    present = True
                    break
            refreshed.append(
                {
                    "term": term.text,
                    "present": present,
                    "satisfied": (not present) if term.negated else present,
                }
            )
        return refreshed

    @staticmethod
    def _await_observation_identity(observation: AnalyzeResult) -> str | None:
        """Stable UI shape used only to confirm an action destination across fresh frames."""
        anchors = tuple(
            (
                _id_tail(element.resource_id) or "",
                element.content_desc or "",
                element.text or "",
                element.bounds,
            )
            for element in app_elements(observation.elements)
            if element.resource_id or element.content_desc or element.text
        )
        if not anchors:
            return None
        return hashlib.sha256(
            repr((observation.screen.package or "", anchors)).encode()
        ).hexdigest()[:16]

    @staticmethod
    def _await_destination_changed(
        observation: AnalyzeResult, baseline: dict[str, Any] | None
    ) -> bool:
        """Whether a hierarchy frame is semantically different from the pre-action screen."""
        if baseline is None:
            return False
        before_identity = str(baseline.get("arrival_identity") or "")
        after_identity = Engine._await_observation_identity(observation) or ""
        if before_identity and after_identity:
            return before_identity != after_identity
        before_package = str(baseline.get("package") or "")
        after_package = str(observation.screen.package or "")
        if before_package and after_package and before_package != after_package:
            return True
        before_known = str(baseline.get("known_screen") or "")
        after_known = str(observation.meta.known_screen or "")
        if before_known and after_known and before_known != after_known:
            return True
        before_labels = {str(value) for value in baseline.get("labels") or [] if value}
        after_labels = {
            _label(value)
            for element in app_elements(observation.elements)
            for value in (element.text, element.content_desc)
            if value and _label(value)
        }
        if before_labels != after_labels:
            return True
        before_rids = {str(value) for value in baseline.get("rids") or [] if value}
        after_rids = {
            rid
            for element in app_elements(observation.elements)
            if (rid := _id_tail(element.resource_id))
        }
        if before_rids and before_rids != after_rids:
            return True
        return int(baseline.get("count") or 0) != len(observation.elements)

    @staticmethod
    def _arrival_predicate_suggestions(
        observation: AnalyzeResult,
        baseline: dict[str, Any] | None,
        *,
        limit: int = 3,
    ) -> list[str]:
        """Stable positive predicates visible only after (or at least on) the destination.

        Resource ids are preferred because they survive copy changes and do not echo user content.
        Text/description is a fallback for apps that expose no ids.  Numeric frame ids are never
        suggested: they are observation-local and are exactly what this recovery is meant to make
        unnecessary.
        """

        def escaped(value: str) -> str:
            return value.replace("\\", "\\\\").replace(",", "\\,")

        before_rids = {str(value) for value in (baseline or {}).get("rids") or [] if value}
        before_labels = {str(value) for value in (baseline or {}).get("labels") or [] if value}
        elements = app_elements(observation.elements)
        # Actionable controls first, then the remaining anchors in visual order.
        ordered = sorted(
            elements,
            key=lambda element: (
                0 if element.enabled and (element.clickable or element.checkable) else 1,
                element.bounds[1],
                element.bounds[0],
            ),
        )
        suggestions: list[str] = []
        seen: set[str] = set()

        def add(prefix: str, value: str) -> None:
            value = _label(value)
            if not value or len(value) > 120:
                return
            predicate = f"{prefix}:{escaped(value)}"
            key = predicate.casefold()
            if key not in seen:
                seen.add(key)
                suggestions.append(predicate)

        # Prefer anchors introduced by the action, then fall back to any destination anchor.
        for new_only in (True, False):
            for element in ordered:
                rid = _id_tail(element.resource_id)
                if not rid or (new_only and rid in before_rids):
                    continue
                add("rid", rid)
                if len(suggestions) >= limit:
                    return suggestions
        for new_only in (True, False):
            for element in ordered:
                if element.password:
                    continue
                for prefix, value in (
                    ("text", element.text or ""),
                    ("desc", element.content_desc or ""),
                ):
                    label = _label(value)
                    if not label or label.isdigit() or (new_only and label in before_labels):
                        continue
                    add(prefix, label)
                    if len(suggestions) >= limit:
                        return suggestions
        return suggestions

    def _sample_action_destination(self) -> AnalyzeResult | None:
        """One fresh, hierarchy-only frame for action-arrival mismatch detection."""
        try:
            return self.analyze(
                source="hierarchy",
                with_ocr=False,
                no_cache=True,
                record=False,
            )
        except Exception as exc:  # noqa: BLE001 - a missed optimization must not fail the wait
            logger.debug("action arrival sample unavailable: %s", exc)
            return None

    def await_predicate(
        self,
        predicate: str,
        *,
        timeout_ms: int = 60_000,
        poll_ms: int = 500,
        match: str = "contains",
        ignore_case: bool = False,
        observe: bool = False,
        adopt_action: bool = False,
        rich_ui: bool = True,
        hierarchy_only: bool = False,
    ) -> ActionResult:
        """Wait until *predicate* holds, and say exactly what ended the wait.

        Observed: image-editing round trips run 2-4 minutes each; one lane watched a GET hang 48
        seconds before cancelling, another a 3-minute stall that resolved on retry. With nothing
        to wait *on*, lanes polled every 15s, waited fixed intervals, or concluded "stuck" from a
        stale frame. A coordinator twice had to ask a lane "is that a hang or a slow backend?"
        because nothing in the output distinguished them. That distinction is the whole value
        here, so the outcome is a named field rather than something inferred from `ok`:

        * ``satisfied`` — every term held.
        * ``screen-changed`` — the foreground activity or package moved while waiting and the
          predicate is still unmet. Returns immediately instead of burning the budget: the
          surface being waited on is gone, so more waiting cannot help. This is the outcome that
          separates "we got kicked out / an error dialog took over" from "still working".
        * ``settled-unmet`` — action-bound waits only: the action reached a stable, non-loading,
          semantically different destination in the same activity, but the caller's positive UI
          arrival term is not on it. This returns a structured ``arrival_mismatch`` rather than
          spending a long budget on a predicate that describes the screen left behind.
        * ``timeout`` — budget spent, predicate unmet, still on the same screen.

        **Not** network idle, which is what was originally asked for. This app is never network
        idle — opening the app hub fires a fire-and-forget play call, analytics post continuously,
        chat surfaces stream — so idleness would be a flaky proxy for "this is ready". A predicate
        says what is actually wanted.

        Standalone ``screen-changed`` remains keyed on the resumed activity/package and
        deliberately not on the element tree: a streaming surface rewrites its tree constantly,
        so a tree-change trigger would abort every legitimate wait on exactly the screens this
        exists for. The stable-tree check is reachable only when ``adopt_action`` says one action
        has already run, and requires two equal fresh destination frames.

        Per-term results are always returned, satisfied or not, because *which* term is missing is
        how a reader tells a failed load from a slow one: spinner gone but results absent is a
        failure, spinner still present is progress.
        """
        terms = _parse_await_terms(predicate, require_positive=adopt_action)
        device = self.device
        mode = MatchMode(match)
        action_baseline = deepcopy(self._action_observation_baseline) if adopt_action else None
        positive_terms = [term for term in terms if not term.negated]
        detect_arrival_mismatch = bool(
            adopt_action
            and action_baseline is not None
            and positive_terms
            # A stable UI cannot prove that an asynchronous network/log event will never arrive.
            # Preserve those waits rather than turning a quiet screen into a false mismatch.
            and all(term.by in {"text", "rid", "desc"} for term in positive_terms)
        )
        stable_destination_identity: str | None = None
        stable_destination_checks = 0

        def snapshot() -> tuple[str, str]:
            try:
                info = device.current_app() or {}
            except Exception:  # a device hiccup must not be read as a navigation
                return ("", "")
            return (str(info.get("package") or ""), str(info.get("activity") or ""))

        # Baseline for the off-screen terms, taken before the first evaluation so a `net:` /
        # `log:` term only ever matches evidence produced *after* the wait began. Without it
        # the previous turn's response would satisfy this turn's wait instantly.
        wall_baseline = time.time()

        def _log_baseline_ms() -> int:
            """Baseline as a **device**-clock epoch — `logcat(since_ms=…)` demands one.

            The host clock is not interchangeable: an emulator can sit seconds off, and a
            baseline in the wrong frame either drops the very lines we are waiting for or
            admits the previous turn's. The measured skew is cached, so this costs no adb
            round-trip on the poll path.
            """
            try:
                from . import logcat as logcat_mod

                clock = logcat_mod.resolve_clock(device, self.config.cache.dir)
                return int(clock.to_device(int(wall_baseline * 1000)))
            except Exception:
                return int(wall_baseline * 1000)

        log_baseline_ms = _log_baseline_ms() if any(t.by == "log" for t in terms) else 0

        def _net_present(spec: str) -> bool:
            try:
                from . import proxy_mock

                flows = proxy_mock.read_flows_since(self.config.cache.dir, wall_baseline)
            except Exception:  # proxy not running / extra not installed
                return False
            return any(proxy_mock.flow_matches(f, spec) for f in flows)

        def _log_present(spec: str) -> bool:
            try:
                lines = device.logcat(dump=True, since_ms=log_baseline_ms) or ""
            except TypeError:  # device implementations that take no since filter
                try:
                    lines = device.logcat(dump=True) or ""
                except Exception:
                    return False
            except Exception:
                return False
            if not isinstance(lines, str):
                lines = "\n".join(str(x) for x in lines)
            haystack = lines.lower() if ignore_case else lines
            needle = spec.lower() if ignore_case else spec
            return needle in haystack

        def evaluate() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for term in terms:
                if term.by == "net":
                    present = _net_present(term.value)
                elif term.by == "log":
                    present = _log_present(term.value)
                else:
                    present = (
                        device.find_text(
                            term.value, match=mode, ignore_case=ignore_case, by=term.by
                        )
                        is not None
                    )
                out.append(
                    {
                        "term": term.text,
                        "present": present,
                        "satisfied": (not present) if term.negated else present,
                    }
                )
            return out

        ui_terms = [term for term in terms if term.by in {"text", "desc"}]

        def evaluate_rich() -> list[dict[str, Any]] | None:
            """Verify UI text against hierarchy plus OCR before making a final claim.

            The device selector only sees accessibility text. That is cheap enough to poll, but
            it made ``!text:Loading`` succeed immediately on a visible canvas label and made a
            positive result time out even though OCR could read it. Rich verification is bounded:
            once before accepting a negated UI term and once at the deadline for a positive miss.
            """
            if not rich_ui or not ui_terms:
                return None
            try:
                observed = self.analyze(source="hierarchy", with_ocr=True, record=False)
            except Exception:  # noqa: BLE001 - unavailable OCR preserves hierarchy semantics
                return None
            base_present = {str(result["term"]): bool(result["present"]) for result in results}

            def matches(value: str, needle: str) -> bool:
                candidate = value.casefold() if ignore_case else value
                wanted = needle.casefold() if ignore_case else needle
                if mode is MatchMode.exact:
                    return candidate == wanted
                if mode is MatchMode.regex:
                    flags = re.IGNORECASE if ignore_case else 0
                    return re.search(needle, value, flags) is not None
                return wanted in candidate

            rich: list[dict[str, Any]] = []
            for term in terms:
                if term.by not in {"text", "desc"}:
                    present = base_present.get(term.text, False)
                else:
                    values: list[str] = []
                    for element in observed.elements:
                        if term.by == "text":
                            values.extend(
                                value for value in (element.text, element.content_desc) if value
                            )
                        elif element.content_desc:
                            values.append(element.content_desc)
                    # Rich analysis is an enrichment, never a replacement: a provider may return
                    # only OCR boxes while the cheap selector already proved a hierarchy term.
                    present = base_present.get(term.text, False) or any(
                        matches(value, term.value) for value in values
                    )
                rich.append(
                    {
                        "term": term.text,
                        "present": present,
                        "satisfied": (not present) if term.negated else present,
                    }
                )
            return rich

        started_at = time.monotonic()
        # Internal hierarchy-only navigation already obtains a fresh observation before the
        # next action. Android's `app_current` is unexpectedly expensive on some devices
        # (~5s per call), and polling it before/after each 1.2s Back step made the "bounded"
        # primitive take 31s. Package boundaries are verified from that observation by
        # `back_until`, so omit redundant activity RPCs on this private fast path.
        origin = ("", "") if hierarchy_only else snapshot()
        deadline = started_at + max(0.0, timeout_ms / 1000.0)
        next_negative_rich_at = started_at
        negative_ui_terms = any(term.negated for term in ui_terms)
        checks = 0
        self._job_checkpoint()
        results = evaluate()
        while True:
            self._job_checkpoint()
            checks += 1
            if all(t["satisfied"] for t in results):
                if not negative_ui_terms:
                    return self._await_result(
                        "satisfied",
                        results,
                        started_at,
                        checks,
                        origin,
                        origin,
                        observe,
                        adopt_action,
                        hierarchy_only=hierarchy_only,
                        capture_terms=terms,
                    )
                # A negated accessibility miss is not proof of visual absence. Verify with OCR,
                # but at most every two seconds while a canvas/loading label remains visible.
                if time.monotonic() >= next_negative_rich_at:
                    rich = evaluate_rich()
                    next_negative_rich_at = time.monotonic() + max(2.0, poll_ms / 250.0)
                    if rich is None or all(term["satisfied"] for term in rich):
                        return self._await_result(
                            "satisfied",
                            rich or results,
                            started_at,
                            checks,
                            origin,
                            origin,
                            observe,
                            adopt_action,
                            hierarchy_only=hierarchy_only,
                            capture_terms=terms,
                        )
                    results = rich
            if detect_arrival_mismatch:
                destination = self._sample_action_destination()
                if destination is not None:
                    destination_terms = self._await_terms_on_observation(
                        terms,
                        results,
                        destination,
                        mode=mode,
                        ignore_case=ignore_case,
                    )
                    if all(term["satisfied"] for term in destination_terms):
                        return self._await_result(
                            "satisfied",
                            destination_terms,
                            started_at,
                            checks,
                            origin,
                            origin,
                            observe,
                            adopt_action,
                            hierarchy_only=hierarchy_only,
                            capture_terms=terms,
                        )
                    unmet_positive = [
                        row["term"]
                        for term, row in zip(terms, destination_terms, strict=True)
                        if not term.negated and not row["satisfied"]
                    ]
                    negative_unmet = any(
                        term.negated and not row["satisfied"]
                        for term, row in zip(terms, destination_terms, strict=True)
                    )
                    identity = self._await_observation_identity(destination)
                    candidate = bool(
                        unmet_positive
                        and not negative_unmet
                        and identity
                        and not self._observation_is_loading(destination)
                        and self._await_destination_changed(destination, action_baseline)
                    )
                    if candidate:
                        if identity == stable_destination_identity:
                            stable_destination_checks += 1
                        else:
                            stable_destination_identity = identity
                            stable_destination_checks = 1
                        if stable_destination_checks >= 2:
                            suggestions = self._arrival_predicate_suggestions(
                                destination,
                                action_baseline,
                            )
                            satisfied_negatives = [
                                term.text
                                for term, row in zip(terms, destination_terms, strict=True)
                                if term.negated and row["satisfied"]
                            ]
                            corrected = ",".join([*suggestions[:1], *satisfied_negatives])
                            recommended_call = (
                                f"aua await-and-analyze {shlex.quote(corrected)} --observe"
                                if corrected
                                else None
                            )
                            mismatch: dict[str, Any] = {
                                "code": "arrival_mismatch",
                                "original_predicate": predicate,
                                "unmet_positive_terms": unmet_positive,
                                "suggested_positive_predicates": suggestions,
                                "stable_checks": stable_destination_checks,
                                "screen_changed": True,
                                "loading": False,
                                "action_repeated": False,
                            }
                            if destination.meta.known_screen:
                                mismatch["known_screen"] = destination.meta.known_screen
                            if recommended_call:
                                mismatch["recommended_call"] = recommended_call
                                mismatch["recommended_mcp_call"] = {
                                    "tool": "await_and_analyze",
                                    "arguments": {"predicate": corrected},
                                }
                            return self._await_result(
                                "settled-unmet",
                                destination_terms,
                                started_at,
                                checks,
                                origin,
                                origin,
                                observe,
                                adopt_action,
                                hierarchy_only=hierarchy_only,
                                arrival_mismatch=mismatch,
                                capture_terms=terms,
                            )
                    else:
                        stable_destination_identity = None
                        stable_destination_checks = 0
            now = origin if hierarchy_only else snapshot()
            if now != origin and any(now):
                return self._await_result(
                    "screen-changed",
                    results,
                    started_at,
                    checks,
                    origin,
                    now,
                    observe,
                    adopt_action,
                    hierarchy_only=hierarchy_only,
                    capture_terms=terms,
                )
            if time.monotonic() >= deadline:
                rich = evaluate_rich()
                if rich is not None and all(term["satisfied"] for term in rich):
                    return self._await_result(
                        "satisfied",
                        rich,
                        started_at,
                        checks,
                        origin,
                        origin,
                        observe,
                        adopt_action,
                        hierarchy_only=hierarchy_only,
                        capture_terms=terms,
                    )
                return self._await_result(
                    "timeout",
                    results,
                    started_at,
                    checks,
                    origin,
                    now,
                    observe,
                    adopt_action,
                    hierarchy_only=hierarchy_only,
                    capture_terms=terms,
                )
            self._job_sleep(max(0.01, poll_ms / 1000.0))
            results = evaluate()

    def _await_result(
        self,
        outcome: str,
        terms: list[dict[str, Any]],
        started_at: float,
        checks: int,
        origin: tuple[str, str],
        now: tuple[str, str],
        observe: bool,
        adopt_action: bool = False,
        *,
        hierarchy_only: bool = False,
        arrival_mismatch: dict[str, Any] | None = None,
        capture_terms: list[_AwaitTerm] | None = None,
    ) -> ActionResult:
        elapsed = int((time.monotonic() - started_at) * 1000)
        unmet = [t["term"] for t in terms if not t["satisfied"]]
        detail = f"{outcome} after {elapsed}ms ({checks} checks)"
        if unmet:
            detail += "; unmet: " + ", ".join(unmet)
        if outcome == "screen-changed":
            detail += f"; now on {now[0]}/{now[1]} (was {origin[0]}/{origin[1]})"
        result = ActionResult(
            ok=outcome == "satisfied",
            action="await",
            detail=detail,
            # `outcome` rides in `acting`-style structured form so a caller branches on a field
            # rather than parsing prose. `ok` alone cannot carry three states.
            await_outcome=outcome,
            await_terms=terms,
            arrival_mismatch=arrival_mismatch,
            elapsed_ms=elapsed,
        )
        # A standalone await is read-only. A global action ``--until`` is different: its final
        # evidence replaces the action's early loading-shell readback, so it must run the normal
        # recording path and consume the still-pending action into this destination. The CLI
        # opts into this explicitly; MCP/standalone waits retain their passive behaviour.
        observed = self._observe(
            result,
            observe,
            settle=False,
            # A timeout is explicitly not final evidence; recording it would merely replace an
            # early loading shell with a later loading shell and consume the action anyway.
            record_screen=adopt_action and outcome != "timeout",
            hierarchy_only=hierarchy_only,
            adopt_action=adopt_action,
        )
        if (
            adopt_action
            and outcome == "satisfied"
            and observed.observation is not None
            and capture_terms
        ):
            memory = self._memory
            if memory is not None and self._join_memory_writers(timeout_s=5.0):
                with contextlib.suppress(Exception):
                    memory.record_action_arrival(
                        self.device.serial,
                        terms=[
                            {
                                "by": term.by,
                                "value": term.value,
                                "negated": term.negated,
                            }
                            for term in capture_terms
                        ],
                        fingerprint=observed.observation.meta.fingerprint,
                        package=observed.observation.screen.package,
                    )
        if arrival_mismatch is not None:
            call = arrival_mismatch.get("recommended_call")
            observed.note = (
                "The action ran once and reached this stable destination, but its arrival "
                "predicate names content that is not here. Reuse this fresh observation and do "
                "not repeat the action."
            )
            if call:
                observed.note += f" If explicit validation is needed, use `{call}`."
        return observed

    def wait(
        self,
        *,
        for_: str | None = None,
        idle: bool = False,
        timeout_ms: int = 5000,
        match: str = "contains",
        ignore_case: bool = False,
        observe: bool = False,
        by: str = "text",
        absent: bool = False,
    ) -> ActionResult:
        device = self.device
        if idle:
            device.wait_idle(timeout_ms)
            return self._observe(
                ActionResult(ok=True, action="wait", detail="idle"), observe, settle=False
            )
        if not for_:
            raise UsageError("wait needs --for <text> or --idle")
        mode = MatchMode(match)
        if absent:
            # Wait until the target is NO LONGER present (loading spinners, transient
            # dialogs) — Maestro's `notVisible`. ok=True once it's gone.
            deadline = time.monotonic() + timeout_ms / 1000.0
            gone = False
            while True:
                if device.find_text(for_, match=mode, ignore_case=ignore_case, by=by) is None:
                    gone = True
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            if not gone:
                detail = self._wait_timeout_message(
                    for_, mode=mode, by=by, ignore_case=ignore_case, absent=True
                )
                return self._observe(
                    ActionResult(ok=False, action="wait", detail=detail), observe, settle=False
                )
            return self._observe(
                ActionResult(ok=True, action="wait", detail=f"absent:{for_}"), observe, settle=False
            )
        found = device.wait_for(
            for_, match=mode, ignore_case=ignore_case, timeout_ms=timeout_ms, by=by
        )
        if found is None:
            detail = self._wait_timeout_message(
                for_, mode=mode, by=by, ignore_case=ignore_case, absent=False
            )
            return self._observe(
                ActionResult(ok=False, action="wait", detail=detail), observe, settle=False
            )
        result = ActionResult(
            ok=True,
            action="wait",
            detail=for_,
            target=list(found),
        )
        # `--observe` returns the screen with fresh ids so the agent acts without a separate
        # `analyze` — attached even on a MISS, so a failed wait is diagnosable in one call.
        # settle=False: wait already blocked on the condition; don't pay pixel-settle again.
        return self._observe(result, observe, settle=False)

    def hierarchy_fingerprint(self) -> str | None:
        """Cheap SHA1 of the current hierarchy dump (no parse). Used by watch/push."""
        device = self.device
        compressed = bool(self.config.device.compressed_hierarchy)
        try:
            xml = self.platform.dump_tree(device, compact=compressed)
        except Exception:  # pragma: no cover
            return None
        return hashlib.sha1(xml.encode()).hexdigest()

    def wait_changed(
        self,
        *,
        timeout_ms: int = 15000,
        interval_ms: int | None = None,
        observe: bool = False,
    ) -> ActionResult:
        """Block until the hierarchy fingerprint changes (or timeout).

        Host-polled stand-in for AccessibilityEvent push (phase 2). Prefer this over
        busy ``analyze`` loops when waiting for *any* UI change.
        """
        interval = (
            interval_ms if interval_ms is not None else int(self.config.daemon.watch_interval_ms)
        )
        baseline = self.hierarchy_fingerprint()
        deadline = time.monotonic() + timeout_ms / 1000.0
        samples = 0
        while time.monotonic() < deadline:
            self._job_sleep(max(0.05, interval / 1000.0))
            samples += 1
            self._job_checkpoint()
            fp = self.hierarchy_fingerprint()
            if fp and baseline and fp != baseline:
                return self._observe(
                    ActionResult(
                        ok=True,
                        action="wait-changed",
                        detail=f"changed after {samples} samples fingerprint={fp[:12]}",
                    ),
                    observe,
                    settle=False,
                )
            if fp and baseline is None:
                baseline = fp
        raise StabilityTimeout(
            f"hierarchy did not change within {timeout_ms} ms ({samples} samples)",
            hint="Increase --timeout, or the screen is idle. Use `aua wait-and-analyze --for` for a label.",
        )

    def wait_after_change(
        self,
        *,
        timeout_ms: int = 60_000,
        interval_ms: int = 120,
        settle_ms: int = 1_200,
        confirmation_ms: int = 1_800,
        observe: bool = False,
    ) -> ActionResult:
        """Wait for a change, visual settle, and a bounded late-change confirmation.

        A loading shell can become visually quiet while its request is still running. Plainly
        composing :meth:`wait_changed` with :meth:`wait_stable` therefore accepts the first quiet
        spinner frame as the result. This contract adds a second, bounded phase: after visual
        settle, the hierarchy must stay unchanged for ``confirmation_ms``. If later content lands
        during that window, stability is measured again from the new frame.

        The confirmation uses the hierarchy rather than full-frame pixels so a looping spinner or
        video remains maskable by :meth:`wait_stable`. Opaque/canvas results should still use an
        explicit predicate, which is the only generic proof that particular content arrived.
        ``timeout_ms`` bounds the complete change + settle + confirmation sequence.
        """
        started = time.monotonic()
        deadline = started + max(0.0, timeout_ms / 1000.0)

        def remaining_ms() -> int:
            return max(1, int((deadline - time.monotonic()) * 1000))

        self.wait_changed(
            timeout_ms=remaining_ms(),
            interval_ms=interval_ms,
            observe=False,
        )
        late_changes = 0
        while True:
            if time.monotonic() >= deadline:
                raise StabilityTimeout(
                    f"screen did not finish changing within {timeout_ms} ms",
                    hint="Increase --timeout, or wait for a specific result with `--until`.",
                )
            self.wait_stable(
                interval_ms=interval_ms,
                settle_ms=max(1, settle_ms),
                timeout_ms=remaining_ms(),
                observe=False,
            )

            baseline = self.hierarchy_fingerprint()
            confirm_deadline = min(
                deadline,
                time.monotonic() + max(0, confirmation_ms) / 1000.0,
            )
            changed_again = False
            while time.monotonic() < confirm_deadline:
                self._job_sleep(max(0.01, interval_ms / 1000.0))
                current = self.hierarchy_fingerprint()
                if baseline and current and current != baseline:
                    changed_again = True
                    late_changes += 1
                    break
                if baseline is None and current:
                    baseline = current
            if changed_again:
                continue
            if confirm_deadline >= deadline and confirmation_ms > 0:
                raise StabilityTimeout(
                    f"screen did not remain confirmed within {timeout_ms} ms",
                    hint="Increase --timeout, or wait for a specific result with `--until`.",
                )
            elapsed = int((time.monotonic() - started) * 1000)
            detail = f"changed and confirmed settled after {elapsed}ms"
            if late_changes:
                detail += f" ({late_changes} late change(s) restabilized)"
            return self._observe(
                ActionResult(ok=True, action="wait-after-change", detail=detail),
                observe,
                settle=False,
            )

    def _wait_timeout_message(
        self,
        needle: str,
        *,
        mode: MatchMode,
        by: str,
        ignore_case: bool,
        absent: bool,
    ) -> str:
        """Rich timeout diagnosis — mode, fields, candidates, accidental-regex hint."""
        field = {"text": "text", "id": "resource-id", "desc": "content-desc"}.get(by, by)
        intent = "still present" if absent else "never appeared"
        parts = [
            f"wait timed out: {needle!r} {intent} "
            f"(match={mode.value}, by={by}, fields={field}"
            f"{', ignore_case' if ignore_case else ''})"
        ]
        # Accidental regex under contains — an observed agent failure mode.
        meta = set(r".*+?[](){}|^$\\")
        if mode is MatchMode.contains and any(c in needle for c in meta):
            parts.append(
                f"hint: pattern looks like regex but --match is '{mode.value}' "
                f"(matched literally as a substring). Use --match regex."
            )
        # Closest on-screen candidates.
        try:
            result = self.analyze(source="hierarchy", record=False)
            from .selectors import app_elements, nearest_elements

            near = nearest_elements(result.elements, needle, limit=5)
            if near:
                digests = []
                for el in near:
                    label = el.text or el.content_desc or (el.resource_id or "").split("/")[-1]
                    digests.append(f"id={el.id}:{label!r}")
                parts.append("closest on screen: " + "; ".join(digests))
            else:
                app_count = len(app_elements(result.elements))
                parts.append(f"screen has {app_count} app elements (no close text match)")
        except Exception as exc:  # pragma: no cover - diagnostic bonus
            parts.append(f"(could not snapshot screen: {exc})")
        return " — ".join(parts)

    # ----------------------------------------------------------------- expect

    def _node_state(self, xml: str, el: Element) -> dict[str, Any]:
        """Interaction state from the selected platform's native-tree adapter."""

        return dict(self.platform.element_state(xml, el))

    def _check_predicates(
        self, el: Element, state: dict[str, Any], predicates: dict[str, Any]
    ) -> list[str]:
        """Names of the predicates that do NOT hold, as ``expected!=actual`` strings."""
        labels = [v for v in (state["text"], state["content_desc"]) if v]
        failures: list[str] = []
        for name, want in predicates.items():
            if name in ("exists", "absent"):
                continue
            if name == "text_is":
                if not any(v.strip() == want for v in labels):
                    failures.append(f"text_is={want!r}!=actual={labels or None!r}")
            elif name == "text_contains":
                if not any(want.lower() in v.lower() for v in labels):
                    failures.append(f"text_contains={want!r}!=actual={labels or None!r}")
            else:
                actual = state.get(name)
                if bool(actual) is not bool(want):
                    failures.append(f"{name}={str(want).lower()}!=actual={str(actual).lower()}")
        return failures

    def _expect_once(
        self,
        selector: dict[str, Any],
        predicates: dict[str, Any],
        *,
        index: int | None,
        first: bool,
        count: int | None = None,
        within: Selector | None = None,
        same_parent_as: Selector | None = None,
        contains_all: Sequence[Selector] = (),
    ) -> tuple[bool, str]:
        """One evaluation pass: ``(ok, detail)``. One hierarchy dump, no screenshots."""
        xml = self.platform.dump_tree(self.device)
        w, h = self.device.window_size()
        elements = self.platform.normalize_tree(
            xml,
            (w, h),
            ignored_app_ids=self.config.memory.ignore_packages,
        ).elements
        label = selector_label(selector)
        matches = match_selector(elements, **selector)
        structural = apply_structural_filters(
            elements,
            matches,
            within=within,
            same_parent_as=same_parent_as,
        )
        if not structural.ok:
            return False, detail_tokens("fail", sought=label) + " | " + str(structural.detail)
        matches = list(structural.matches)
        if count is not None and len(matches) != count:
            detail = detail_tokens(
                "fail",
                sought=label,
                predicate="count",
                expected=count,
                actual=len(matches),
            )
            if matches:
                detail += " | found: " + " | ".join(
                    element_digest(el) for el in matches[:_MAX_CANDIDATES]
                )
            return False, detail
        if count == 0:
            return True, detail_tokens(
                "pass", sought=label, predicate="count", expected=0, actual=0
            )
        if predicates.get("absent"):
            if not matches:
                return True, detail_tokens("pass", sought=label, predicate="absent")
            return False, detail_tokens(
                "fail", sought=label, predicate="absent", actual="present"
            ) + " | found: " + " | ".join(element_digest(el) for el in matches[:_MAX_CANDIDATES])
        if not matches:
            near = nearest_elements(elements, selector.get("rid") or selector.get("text") or "")
            app_only = app_elements(elements)
            detail = detail_tokens(
                "fail",
                sought=label,
                predicate="exists",
                actual="absent",
                on_screen=len(app_only),
                system=len(elements) - len(app_only) or None,
            )
            if near:
                detail += " | nearest: " + " | ".join(element_digest(el) for el in near)
            return False, detail
        state_only = [k for k in predicates if k not in ("exists", "absent")]
        structural_only = bool(within or same_parent_as or contains_all)
        if len(matches) > 1 and (state_only or structural_only) and index is None and not first:
            raise SelectorAmbiguousError(
                f"{label} matches {len(matches)} elements — "
                "disambiguate with --index <n> or --first before asserting on its state",
                hint="candidates: "
                + " | ".join(element_digest(el) for el in matches[:_MAX_CANDIDATES]),
            )
        if index is not None and index >= len(matches):
            return False, detail_tokens(
                "fail",
                sought=label,
                predicate="index",
                expected=index,
                actual=f"{len(matches)} matches",
            )
        el = matches[index] if index is not None else matches[0]
        if contains_all:
            contains_ok, contains_detail = check_contains_all(elements, el, contains_all)
            if not contains_ok:
                return False, detail_tokens(
                    "fail", sought=label, id=el.id
                ) + " | " + contains_detail
        failures = self._check_predicates(el, self._node_state(xml, el), predicates)
        if failures:
            return False, detail_tokens("fail", sought=label, id=el.id) + " | " + "; ".join(
                failures
            )
        checks = ",".join(predicates) or "exists"
        if count is not None:
            checks += f",count={count}"
        return True, detail_tokens("pass", sought=label, id=el.id, checks=checks)

    def expect(
        self,
        *,
        rid: str | None = None,
        text: str | None = None,
        desc: str | None = None,
        exists: bool = False,
        absent: bool = False,
        text_is: str | None = None,
        text_contains: str | None = None,
        checked: bool | None = None,
        enabled: bool | None = None,
        selected: bool | None = None,
        focused: bool | None = None,
        count: int | None = None,
        within: dict[str, Any] | None = None,
        same_parent_as: dict[str, Any] | None = None,
        contains_all: Sequence[dict[str, Any]] | None = None,
        index: int | None = None,
        first: bool = False,
        timeout_ms: int = 0,
        poll_ms: int = 250,
        observe: bool = False,
    ) -> ActionResult:
        """Assert something about the screen; ``ok=False`` means the assertion failed.

        This is the primitive that turns an acceptance-criteria list into a script: one
        criterion per call, exit code per criterion. ``timeout_ms`` polls until the
        assertion holds, which is what replaces a ``sleep`` guess — the flakiness the
        project's own testing guidance warns about.
        """
        selector = {"rid": rid, "text": text, "desc": desc}
        if len([v for v in selector.values() if v]) != 1:
            raise UsageError(
                "expect needs exactly one of --rid / --text / --desc",
                hint="e.g. `aua expect-and-analyze --rid notificationsButton --exists`",
            )
        if absent and exists:
            raise UsageError("--exists and --absent are mutually exclusive")
        if count is not None and count < 0:
            raise UsageError("--count must not be negative")
        if index is not None and index < 0:
            raise UsageError("--index must not be negative")
        if absent and any(value is not None for value in (within, same_parent_as, contains_all)):
            raise UsageError("--absent cannot be combined with structural predicates")
        if count == 0 and any(
            value is not None
            for value in (
                text_is,
                text_contains,
                checked,
                enabled,
                selected,
                focused,
                within,
                same_parent_as,
                contains_all,
            )
        ):
            raise UsageError("--count 0 cannot be combined with element state/structure predicates")
        normalized_within = (
            normalize_selector(within, field="within") if within is not None else None
        )
        normalized_same_parent = (
            normalize_selector(same_parent_as, field="same_parent_as")
            if same_parent_as is not None
            else None
        )
        normalized_contains = tuple(
            normalize_selector(value, field=f"contains_all[{position}]")
            for position, value in enumerate(contains_all or ())
        )
        if contains_all is not None and not normalized_contains:
            raise UsageError("contains_all must not be empty")
        predicates: dict[str, Any] = {}
        if absent:
            predicates["absent"] = True
        for name, value in (
            ("text_is", text_is),
            ("text_contains", text_contains),
            ("checked", checked),
            ("enabled", enabled),
            ("selected", selected),
            ("focused", focused),
        ):
            if value is not None:
                predicates[name] = value
        if not predicates or exists:
            predicates.setdefault("exists", True)
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        while True:
            ok, detail = self._expect_once(
                selector,
                predicates,
                index=index,
                first=first,
                count=count,
                within=normalized_within,
                same_parent_as=normalized_same_parent,
                contains_all=normalized_contains,
            )
            if ok or time.monotonic() >= deadline:
                return self._observe(
                    ActionResult(ok=ok, action="expect", detail=detail), observe, None
                )
            time.sleep(max(0.05, poll_ms / 1000.0))

    # ----------------------------------------------------------------- device extras

    def clipboard_set(self, text: str) -> ActionResult:
        self.device.set_clipboard(text)
        return ActionResult(ok=True, action="clipboard-set", detail=text)

    def clipboard_get(self) -> ActionResult:
        text = self.device.get_clipboard()
        return ActionResult(ok=True, action="clipboard-get", detail=text)

    def paste(self, *, observe: bool = True, with_image: bool | str | None = None) -> ActionResult:
        with self._acting():
            self.device.paste()
        # The clipboard value is deliberately not captured. Keep a lossy journal marker so a
        # recorded-flow preview refuses to pretend the resulting journey is self-contained.
        self._record_action_safe(RouteStep(kind="paste"))
        return self._observe(ActionResult(ok=True, action="paste"), observe, with_image)

    def copy_text(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
    ) -> ActionResult:
        el = self._target(element_id, selector, verb="copy")
        text = (el.text or el.content_desc or "").strip()
        if not text:
            raise UsageError(
                "element has no text or content-desc to copy",
                hint="Pick a labelled element, or use `clipboard set` for a literal.",
            )
        self.device.set_clipboard(text)
        return ActionResult(ok=True, action="copy", id=el.id, detail=text)

    def location_set(self, lat: float, lon: float) -> ActionResult:
        self.device.set_location(lat, lon)
        return ActionResult(ok=True, action="location-set", detail=f"{lat},{lon}")

    def orientation_set(self, mode: str) -> ActionResult:
        with self._acting():
            self.device.set_orientation(mode)
        return ActionResult(ok=True, action="orientation-set", detail=mode)

    def orientation_get(self) -> ActionResult:
        mode = self.device.get_orientation()
        return ActionResult(ok=True, action="orientation-get", detail=mode)

    def airplane_set(self, enabled: bool) -> ActionResult:
        self.device.set_airplane_mode(enabled)
        return ActionResult(
            ok=True,
            action="airplane-set",
            detail="on" if enabled else "off",
            note=(
                "airplane mode is not proof of offline connectivity because Wi-Fi may remain "
                "active; for offline tests use `aua network offline --verify` and always "
                "restore with `aua network restore`"
                if enabled
                else None
            ),
        )

    def airplane_toggle(self) -> ActionResult:
        cur = self.device.get_airplane_mode()
        enabled = not cur if cur is not None else True
        self.device.set_airplane_mode(enabled)
        return ActionResult(
            ok=True,
            action="airplane-toggle",
            detail="on" if enabled else "off",
            note=(
                "airplane mode is not proof of offline connectivity; use verified reversible "
                "`network offline` for offline tests"
                if enabled
                else None
            ),
        )

    def network_status(self) -> NetworkResult:
        from . import network

        device = self.device
        path = network.backup_path(self.config.cache.dir, device.serial)
        backup = network.load_backup(path)
        state = network.read_network_state(device)
        return NetworkResult(
            ok=True,
            action="network-status",
            state=state,
            saved_state=backup.state if backup is not None else None,
            verified=state.active_network is not None,
            detail="restore point available" if backup is not None else "no restore point",
        )

    def network_offline(self, *, verify: bool = True, timeout_ms: int = 10_000) -> NetworkResult:
        from . import network, network_profiles

        device = self.device
        profile = network_profiles.load_profile(
            network_profiles.profile_path(self.config.cache.dir, device.serial)
        )
        if profile is not None and not network_profiles.stale_profile(profile, device):
            raise UsageError(
                f"network profile {profile.profile!r} is active",
                hint="Run `aua network profile restore` before entering offline mode.",
            )
        path = network.backup_path(self.config.cache.dir, device.serial)
        initial = network.read_network_state(device)
        backup = network.save_backup(path, device=device, state=initial)
        network.apply_offline_controls(device, initial)
        if verify:
            state, verified = network.wait_for_state(
                device,
                network.offline_verified,
                timeout_ms=timeout_ms,
            )
        else:
            state = network.read_network_state(device)
            verified = None
        result = NetworkResult(
            ok=bool(verified) if verify else True,
            action="network-offline",
            state=state,
            saved_state=backup.state,
            verified=verified,
            detail=(
                "offline verified"
                if verified
                else (
                    "offline controls applied without verification"
                    if not verify
                    else "offline verification timed out; restore point retained"
                )
            ),
        )
        self._record_action_safe(RouteStep(kind="network-offline"))
        return result

    def network_restore(self, *, timeout_ms: int = 15_000) -> NetworkResult:
        from . import network

        device = self.device
        path = network.backup_path(self.config.cache.dir, device.serial)
        backup = network.require_current_backup(path, device=device)
        network.restore_controls(device, backup.state)
        state, verified = network.wait_for_state(
            device,
            lambda current: network.restored_verified(current, backup.state),
            timeout_ms=timeout_ms,
        )
        if verified:
            path.unlink(missing_ok=True)
        result = NetworkResult(
            ok=verified,
            action="network-restore",
            state=state,
            saved_state=backup.state,
            verified=verified,
            detail=(
                "original network state restored"
                if verified
                else "restore verification timed out; restore point retained"
            ),
        )
        self._record_action_safe(RouteStep(kind="network-restore"))
        return result

    def network_profile_list(self) -> dict[str, Any]:
        from . import network_profiles

        return {
            "ok": True,
            "action": "network-profile-list",
            "profiles": [
                {
                    "name": "wifi-only",
                    "effect": "enable Wi-Fi and disable mobile data",
                    "needs": [],
                },
                {
                    "name": "cellular-only",
                    "effect": "disable Wi-Fi and enable mobile data",
                    "needs": [],
                },
                {
                    "name": "slow",
                    "effect": "EDGE bandwidth with 80-400 ms latency",
                    "needs": ["emulator"],
                },
                {
                    "name": "lossy",
                    "effect": "outbound packet loss on the active interface",
                    "needs": ["root"],
                },
            ],
            "names": list(network_profiles.PROFILE_NAMES),
        }

    def network_profile_status(self) -> NetworkResult:
        from . import network, network_profiles

        device = self.device
        path = network_profiles.profile_path(self.config.cache.dir, device.serial)
        backup = network_profiles.load_profile(path)
        state = network.read_network_state(device)
        if backup is None:
            return NetworkResult(
                ok=True,
                action="network-profile-status",
                state=state,
                verified=True,
                detail="no active network profile",
            )
        if network_profiles.stale_profile(backup, device):
            return NetworkResult(
                ok=True,
                action="network-profile-status",
                profile=backup.profile,
                state=state,
                saved_state=backup.network_state,
                verified=False,
                detail="profile restore point belongs to a previous device boot",
            )

        shaping = None
        if backup.profile in ("wifi-only", "cellular-only"):
            verified = network_profiles.profile_verified(backup.profile, state)
        elif backup.profile == "slow":
            current = network_profiles.read_emulator_shape(device.serial)
            shaping = current.evidence()
            verified = bool(
                current.upload_bps > 0
                and current.download_bps > 0
                and current.min_latency_ms >= 80
                and current.max_latency_ms >= 400
            )
        else:
            if backup.interface is None:
                verified = False
            else:
                shaping = network_profiles.qdisc_evidence(
                    device.serial,
                    backup.interface,
                    root=network_profiles.root_enabled(device.serial),
                )
                verified = bool(
                    shaping.qdisc == "netem"
                    and shaping.loss_percent is not None
                    and backup.loss_percent is not None
                    and abs(shaping.loss_percent - backup.loss_percent) < 0.01
                )
        return NetworkResult(
            ok=True,
            action="network-profile-status",
            profile=backup.profile,
            state=state,
            saved_state=backup.network_state,
            shaping=shaping,
            verified=verified,
            detail="profile verified" if verified else "profile could not be verified",
        )

    def network_profile_apply(
        self,
        profile: str,
        *,
        loss_percent: float = 10.0,
        timeout_ms: int = 15_000,
    ) -> NetworkResult:
        from . import network, network_profiles

        name = network_profiles.normalize_profile(profile)
        if not 0.1 <= loss_percent <= 100:
            raise UsageError("--loss-percent must be between 0.1 and 100")
        device = self.device
        path = network_profiles.profile_path(self.config.cache.dir, device.serial)
        initial = network.read_network_state(device)
        if (
            network.load_backup(network.backup_path(self.config.cache.dir, device.serial))
            is not None
        ):
            raise UsageError(
                "verified offline mode is active",
                hint="Run `aua network restore` before applying a network profile.",
            )
        active_profile = network_profiles.load_profile(path)
        if active_profile is not None and not network_profiles.stale_profile(
            active_profile,
            device,
        ):
            raise UsageError(
                f"network profile {active_profile.profile!r} is already active",
                hint="Run `aua network profile restore` before applying another profile.",
            )

        if name in ("wifi-only", "cellular-only"):
            backup = network_profiles.save_profile(
                path,
                device=device,
                profile=name,
                network_state=initial,
            )
            network_profiles.apply_radio_profile(device, name)
            state, verified = network_profiles.wait_for_radio_profile(
                device,
                name,
                timeout_ms=timeout_ms,
            )
            shaping = None
        elif name == "slow":
            original_shape = network_profiles.read_emulator_shape(device.serial)
            backup = network_profiles.save_profile(
                path,
                device=device,
                profile=name,
                network_state=initial,
                emulator_shape=original_shape,
            )
            current_shape = network_profiles.set_emulator_shape(
                device.serial,
                speed="edge",
                delay="edge",
            )
            state = network.read_network_state(device)
            shaping = current_shape.evidence()
            verified = bool(
                current_shape.upload_bps > 0
                and current_shape.download_bps > 0
                and current_shape.min_latency_ms >= 80
                and current_shape.max_latency_ms >= 400
            )
        else:
            interface, original_qdisc, was_root = network_profiles.prepare_loss(device.serial)
            try:
                backup = network_profiles.save_profile(
                    path,
                    device=device,
                    profile=name,
                    network_state=initial,
                    interface=interface,
                    original_qdisc=original_qdisc,
                    loss_percent=loss_percent,
                    root_was_enabled=was_root,
                )
            except Exception:
                network_profiles.safe_unroot_after_failed_apply(
                    device.serial,
                    was_root=was_root,
                )
                raise
            shaping = network_profiles.set_loss(
                device.serial,
                interface=interface,
                loss_percent=loss_percent,
            )
            state = network.read_network_state(device)
            verified = bool(
                shaping.qdisc == "netem"
                and shaping.loss_percent is not None
                and abs(shaping.loss_percent - loss_percent) < 0.01
            )

        result = NetworkResult(
            ok=verified,
            action="network-profile-apply",
            profile=name,
            state=state,
            saved_state=backup.network_state,
            shaping=shaping,
            verified=verified,
            detail=(
                f"{name} profile verified"
                if verified
                else f"{name} profile verification timed out; restore point retained"
            ),
        )
        self._record_action_safe(RouteStep(kind="network-profile", arg=name))
        return result

    def network_profile_restore(self, *, timeout_ms: int = 20_000) -> NetworkResult:
        from . import network, network_profiles

        device = self.device
        path = network_profiles.profile_path(self.config.cache.dir, device.serial)
        backup = network_profiles.require_current_profile(path, device=device)
        shaping = None
        if backup.profile in ("wifi-only", "cellular-only"):
            state, verified = network_profiles.restore_radio_profile(
                device,
                backup.network_state,
                timeout_ms=timeout_ms,
            )
        elif backup.profile == "slow":
            if backup.emulator_shape is None:
                raise UsageError("slow profile restore point has no original emulator shape")
            current = network_profiles.restore_emulator_shape(
                device.serial,
                backup.emulator_shape,
            )
            shaping = current.evidence()
            verified = network_profiles.shape_matches(current, backup.emulator_shape)
            state = network.read_network_state(device)
        else:
            shaping, verified = network_profiles.remove_loss(device.serial, backup)
            state = network.read_network_state(device)
        if verified:
            path.unlink(missing_ok=True)
        result = NetworkResult(
            ok=verified,
            action="network-profile-restore",
            profile=backup.profile,
            state=state,
            saved_state=backup.network_state,
            shaping=shaping,
            verified=verified,
            detail=(
                "original network conditions restored"
                if verified
                else "profile restore verification failed; restore point retained"
            ),
        )
        self._record_action_safe(RouteStep(kind="network-profile-restore"))
        return result

    def media_add(self, path: str, *, remote_dir: str = "/sdcard/DCIM/Camera") -> ActionResult:
        remote = self.device.add_media(path, remote_dir=remote_dir)
        return ActionResult(ok=True, action="media-add", detail=remote)

    def record_start(self, path: str | None = None) -> ActionResult:
        remote = self.device.start_recording(path or "/sdcard/aua_recording.mp4")
        return ActionResult(ok=True, action="record-start", detail=remote)

    def record_stop(self, local_path: str) -> ActionResult:
        saved = self.device.stop_recording(local_path)
        return ActionResult(ok=True, action="record-stop", detail=saved)

    def clock_set(self, *, timestamp_ms: int | None = None, restore: bool = False) -> ActionResult:
        """Set or restore the device wall clock (Maestro ``travel``).

        Moving the clock often invalidates auth tokens (401s). Always ``clock restore``
        (or ``clock set --restore``) when the test is done — never leave the device in
        a time-traveled state.
        """
        path = self._clock_backup_path()
        if restore:
            if not path.is_file():
                raise UsageError(
                    "no saved clock to restore",
                    hint="Run `aua clock set --ms …` first; it saves the prior wall clock.",
                )
            previous = int(path.read_text(encoding="utf-8").strip())
            self.device.set_clock(previous)
            path.unlink(missing_ok=True)
            return ActionResult(ok=True, action="clock-restore", detail=str(previous))
        if timestamp_ms is None:
            raise UsageError("clock set needs --ms <unix-ms> (or --restore)")
        # Save current clock once so restore is possible.
        current = self.device.get_clock_ms()
        if current is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                path.write_text(str(current), encoding="utf-8")
        self.device.set_clock(timestamp_ms)
        return ActionResult(
            ok=True,
            action="clock-set",
            detail=str(timestamp_ms),
        )

    def _clock_backup_path(self) -> Path:
        serial = self._device.serial if self._device else (self.config.device.serial or "default")
        safe = str(serial).replace(":", "_")
        return Path(self.config.cache.dir).expanduser() / f"clock_backup_{safe}.txt"

    def _dev_backup_path(self) -> Path:
        serial = self._device.serial if self._device else (self.config.device.serial or "default")
        safe = str(serial).replace(":", "_")
        return Path(self.config.cache.dir).expanduser() / f"devopts_backup_{safe}.json"

    def _proxy_port(self) -> int | None:
        """Listen port of the running mitm, or ``None`` if none has been chosen yet."""
        from . import proxy_mock as pm

        return pm.load_listen_port(Path(self.config.cache.dir).expanduser())

    def dev_show(self) -> dict[str, Any]:
        from . import devopts

        state = devopts.read_state(self.device.shell)
        return {"ok": True, "action": "dev-show", **state}

    def dev_anim(self, mode: str) -> dict[str, Any]:
        from . import devopts

        path = self._dev_backup_path()
        m = (mode or "").lower()
        if m == "off":
            state = devopts.anim_off(self.device.shell, path)
        elif m == "restore":
            state = devopts.anim_restore(self.device.shell, path)
        else:
            raise UsageError(
                f"unknown anim mode {mode!r}",
                hint="Use `aua dev anim off` or `aua dev anim restore`.",
            )
        return {"ok": True, "action": f"dev-anim-{m}", **state}

    def dev_crashes(self, enabled: bool) -> dict[str, Any]:
        from . import devopts

        state = devopts.crashes_set(self.device.shell, enabled, self._dev_backup_path())
        return {
            "ok": True,
            "action": "dev-crashes-on" if enabled else "dev-crashes-off",
            **state,
        }

    def dev_profile(self, name: str) -> dict[str, Any]:
        from . import devopts

        path = self._dev_backup_path()
        n = (name or "").lower()
        if n == "ac":
            state = devopts.profile_ac(self.device.shell, path)
        elif n == "default":
            state = devopts.profile_default(self.device.shell, path)
        else:
            raise UsageError(
                f"unknown dev profile {name!r}",
                hint="Use `ac` (anim off + crashes on) or `default` (restore).",
            )
        self._record_action_safe(RouteStep(kind="dev-profile", arg=n))
        return {"ok": True, "action": f"dev-profile-{n}", **state}

    def a11y_scroll(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        direction: str = "forward",
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        el = self._target(element_id, selector, verb="a11y scroll")
        d = (direction or "forward").lower()
        action = "SCROLL_FORWARD" if d in ("forward", "fwd", "down") else "SCROLL_BACKWARD"
        if d not in ("forward", "fwd", "down", "backward", "back", "up"):
            raise UsageError(
                f"unknown scroll direction {direction!r}",
                hint="Use --forward or --backward.",
            )
        cx, cy = el.center
        step = self._step("a11y-scroll", el, arg=d)
        with self._acting(f"a11y-scroll:{d}"):
            self.device.a11y_action(cx, cy, action)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="a11y-scroll", detail=f"{d} @{el.id}"),
            observe,
            with_image,
        )

    def a11y_action(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        action: str = "CLICK",
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        el = self._target(element_id, selector, verb="a11y action")
        cx, cy = el.center
        act = (action or "CLICK").strip().upper()
        step = self._step("a11y-action", el, arg=act)
        with self._acting(f"a11y:{act}"):
            self.device.a11y_action(cx, cy, act)
        self._record_action_safe(step)
        return self._observe(
            ActionResult(ok=True, action="a11y-action", detail=f"{act} @{el.id}"),
            observe,
            with_image,
        )

    def flags_set(
        self,
        package: str,
        assignments: list[str] | dict[str, str],
        *,
        observe: bool = True,
        with_image: bool | str | None = None,
        restart: bool = True,
        activity: str | None = None,
        verify: bool = True,
        prefs_file: str | None = None,
    ) -> dict[str, Any]:
        """Write flags via the package's deeplink, read them back, and restart the app.

        The restart is the default because flags read at cold start (a landing view-model
        building its tab list once) are invisible to the process that received the
        deeplink: without it the caller screenshots the OLD ui and blames the flag.
        """
        from .flags import build_uri, dump_result, parse_assignments

        pairs = (
            parse_assignments(list(assignments))
            if not isinstance(assignments, dict)
            else dict(assignments)
        )
        templates = dict(self.config.flags.templates)
        uri = build_uri(package, pairs, templates)
        entry = (activity or self._foreground_activity(package)) if restart else None
        mem = self._memory
        if mem is not None and not self._join_memory_writers(timeout_s=5.0):
            # `flags_apply` suppresses the internal open-link journal so the outer operation
            # is captured once. Preserve provenance ordering explicitly before that mutation.
            raise UsageError("memory provenance is still being finalized")
        self.open_link(uri, package=package, pin_package=True, observe=False)
        # Read back BEFORE the force-stop: the file on disk is the proof the app committed
        # the override, and killing a process with a pending async write would lose it.
        prefs = (
            self._verify_flags(
                package, pairs, prefs_file=prefs_file, deadline_s=_FLAGS_VERIFY_DEADLINE_S
            )
            if verify
            else None
        )
        restarted = self._restart_app(package, entry) if restart else Restart(False, None, None)
        payload = dump_result(
            package=package,
            uri=uri,
            flags=pairs,
            prefs=prefs,
            restarted=restarted.ok,
            activity=restarted.activity,
            restart_error=restarted.error,
        )
        if restarted.ok and mem is not None:
            active = prefs.applied if prefs is not None and prefs.verified else pairs
            fully_verified = bool(
                prefs is not None and prefs.verified and not prefs.ignored and not prefs.mismatched
            )
            if not self._join_memory_writers(timeout_s=5.0):
                raise UsageError("memory provenance is still being finalized")
            with self._mem_lock:
                mem.activate_flag_context(
                    self.device.serial,
                    package,
                    active,
                    app_version=self._version_for(self.device, package),
                    verified=fully_verified,
                )
                payload["context_id"] = mem.load_session(self.device.serial).active_context_id
        observed = self._observe(
            ActionResult(ok=True, action="flags-set"), observe, with_image
        ).model_dump(mode="json", exclude_none=True)
        # Keep the verification payload's own ok/detail, but expose the same folded-screen
        # contract as every other observed action. Previously this analysis was performed and
        # then discarded, so callers paid for it and still had to call ``analyze`` themselves.
        for key in (
            "observation",
            "observation_present",
            "known_screen",
            "stable_elements",
            "action_diff_summary",
            "next_actions",
            "routes",
            "note",
            "stale_risk",
            "settle",
        ):
            if key in observed:
                payload[key] = observed[key]
        return payload

    def _foreground_activity(self, package: str) -> str | None:
        """The activity of *package* if it is in the foreground — the one to relaunch."""
        with contextlib.suppress(Exception):
            app = self.device.current_app() or {}
            if (app.get("package") or "") == package:
                return app.get("activity") or None
        return None

    def _wait_foreground(self, package: str, timeout_s: float | None = None) -> bool:
        deadline = time.monotonic() + (
            _FLAGS_FOREGROUND_TIMEOUT_S if timeout_s is None else timeout_s
        )
        while True:
            with contextlib.suppress(Exception):
                if ((self.device.current_app() or {}).get("package") or "") == package:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.3)

    def _launch_entry(self, package: str, activity: str | None) -> tuple[str | None, str | None]:
        """Decide which Activity a cold start targets, teaching the app map on the way.

        Returns ``(activity_or_None, note_or_None)``. ``None`` means "let the platform resolve
        it" — the pre-existing behaviour. The note is set only when the choice stayed ambiguous,
        so the agent learns it should pin one instead of trusting the screen it happens to get.

        An explicit ``--activity`` wins outright. Otherwise a pin already in the map is reused,
        which is what makes a repeated journey open the same screen every time. With no pin, the
        declared MAIN/LAUNCHER set decides: exactly one is auto-pinned, several stay unpinned.
        """
        if activity:
            return activity, None
        mem = self._memory
        if mem is None:
            return None, None
        with self._mem_lock:
            pinned = mem.launch_activity(package)
        if pinned:
            return pinned, None
        try:
            declared = self.device.launcher_activities(package)
        except Exception as exc:  # noqa: BLE001 — a launch must not fail over a memory nicety
            logger.debug("could not read the launcher activities of %s: %s", package, exc)
            return None, None
        with self._mem_lock:
            entry = mem.record_launcher_activities(package, declared)
        if entry is not None:
            return entry.activity, None
        if len(declared) > 1:
            listed = ", ".join(declared)
            return None, (
                f"{package} declares {len(declared)} launcher activities ({listed}), so this "
                "multi-launcher build cold-started on whichever one the manifest lists first — "
                "possibly a Dev Tools entry rather than the product. Pin the right one with "
                "`aua remember --launch-activity <Activity>` to make later launches deterministic."
            )
        return None, None

    def _restart_app(self, package: str, activity: str | None) -> Restart:
        """Force-stop + relaunch, and confirm the app came back.

        A pinned entry Activity is usually NOT exported (a mid-flow screen never is), and
        ``am start -n`` then prints a SecurityException instead of failing — so the
        foreground has to be re-read rather than assumed, or this reports a restart that
        left the app dead.
        """
        device = self.device
        if not activity:
            # `flags set` restarts with no mid-flow Activity to return to. Without the learned
            # pin this fell through to an unpinned resolve — the coin flip that can reopen a Dev
            # Tools entry, so the flags the caller just set get verified against the wrong screen.
            mem = self._memory
            if mem is not None:
                with self._mem_lock:
                    activity = mem.launch_activity(package)
        with self._acting():
            device.stop_app(package)
            pinned = False
            if activity:
                # Only wait for something that was actually asked to start. `launch_app`
                # RAISES when `am start` refuses (a non-exported Activity is the usual case,
                # and the default pinned entry is simply whatever was in the foreground) —
                # waiting the entry timeout after a refusal is waiting for a process that was
                # never launched. That cost `_FLAGS_ENTRY_TIMEOUT_S` on every `flags set`
                # against an app whose entry Activity is not exported, before the fallback
                # had even started.
                launched = True
                try:
                    device.launch_app(package, activity=activity)
                except Exception as exc:  # noqa: BLE001 — any refusal means "did not start"
                    launched = False
                    logger.debug(
                        "%s/%s refused (%s); using the default entry", package, activity, exc
                    )
                if launched:
                    pinned = self._wait_foreground(package, _FLAGS_ENTRY_TIMEOUT_S)
                    if not pinned:
                        logger.debug(
                            "%s/%s did not come up; using the default entry", package, activity
                        )
            if not pinned:
                try:
                    device.launch_app(package)
                except Exception as exc:  # noqa: BLE001 — same reasoning as above
                    return Restart(False, None, f"{package} could not be relaunched: {exc}")
                if not self._wait_foreground(package):
                    return Restart(False, None, f"{package} did not come back after the restart")
            with contextlib.suppress(Exception):
                device.wait_idle(3000)
        # Where it LANDED, not where it was aimed: a build with two launcher activities
        # resolves the default entry ambiguously, and the caller analyzes that screen next.
        return Restart(True, self._foreground_activity(package), None)

    def _verify_flags(
        self,
        package: str,
        pairs: dict[str, str],
        *,
        prefs_file: str | None,
        deadline_s: float,
    ) -> PrefsRead:
        """Poll the app's prefs until every requested key is there, or time runs out."""
        from .flags import read_prefs

        name = prefs_file or self.config.flags.prefs_files.get(package)
        deadline = time.monotonic() + deadline_s
        while True:
            prefs = read_prefs(self.device, package, pairs, prefs_file=name)
            if not prefs.verified or not (prefs.ignored or prefs.mismatched):
                return prefs
            if time.monotonic() >= deadline:
                return prefs
            time.sleep(0.25)

    def flags_apply(
        self,
        path: str,
        *,
        package: str | None = None,
        observe: bool = True,
        with_image: bool | str | None = None,
        restart: bool = True,
        activity: str | None = None,
        verify: bool = True,
        prefs_file: str | None = None,
        _snapshot: _ResolvedFlagsResource | None = None,
    ) -> dict[str, Any]:
        from .flags import load_flags_file

        if _snapshot is None:
            app, pairs = load_flags_file(path)
            source_path = str(Path(path).expanduser().resolve())
        else:
            app, pairs = _snapshot.app, deepcopy(_snapshot.pairs)
            source_path = _snapshot.source_path
        pkg = package or app or self.current_package()
        if not pkg:
            raise UsageError(
                "flags apply needs a package",
                hint="Put `app: <pkg>` in the YAML or pass `--package`.",
            )
        with self._without_action_recording():
            result = self.flags_set(
                pkg,
                pairs,
                observe=observe,
                with_image=with_image,
                restart=restart,
                activity=activity,
                verify=verify,
                prefs_file=prefs_file,
            )
        if result.get("ok", True):
            self._record_action_safe(RouteStep(kind="flags-apply", arg=source_path))
        return result

    def proxy_start(
        self,
        *,
        port: int | None = None,
        install_ca: bool = True,
    ) -> dict[str, Any]:
        from . import proxy_mock as pm

        cache = Path(self.config.cache.dir).expanduser()
        # Touch the device first so a dead serial does not leave a stray mitmdump.
        device = self.device
        ca_info: dict[str, Any] | None = None
        if install_ca:
            try:
                ca_info = pm.install_system_ca(device.serial)
            except (DeviceError, UsageError) as exc:
                # Still start the proxy — but surface why HTTPS will likely fail.
                ca_info = {"ok": False, "error": str(exc), "hint": getattr(exc, "hint", None)}
                logger.warning("system CA install failed: %s", exc)
        # ``port<=0`` / omitted → random free high port (never hardcodes 8080).
        preferred = port if port and port > 0 else None
        pid, listen = pm.start_mitm(cache_dir=cache, port=preferred, mode="map")
        try:
            device.adb_reverse(listen, listen)
            device.set_http_proxy(f"127.0.0.1:{listen}")
        except Exception:
            with contextlib.suppress(Exception):
                pm.stop_mitm(cache)
            raise
        # Relaunching the foreground app makes it inherit Zygote CA mounts.
        pkg = None
        with contextlib.suppress(Exception):
            pkg = (device.current_app() or {}).get("package")
        if pkg and ca_info and ca_info.get("ok"):
            with contextlib.suppress(Exception):
                device.stop_app(pkg)
                device.launch_app(pkg)
        out: dict[str, Any] = {
            "ok": True,
            "action": "proxy-start",
            "pid": pid,
            "port": listen,
            "ca": ca_info,
            "hint": (
                f"Device http_proxy is 127.0.0.1:{listen} (via adb reverse). "
                + (
                    "System CA installed — force-stop/relaunch done for foreground app."
                    if ca_info and ca_info.get("ok")
                    else "CA install failed or skipped: HTTPS apps that only trust system "
                    "CAs will produce EMPTY cassettes until the mitm CA is a "
                    "system trust anchor. Fix: `aua emulator ensure-proxy` → start "
                    "`aua_proxy` → `aua proxy start` on that serial."
                )
            ),
        }
        if pkg:
            out["relaunched"] = pkg
        return out

    def proxy_stop(self) -> dict[str, Any]:
        from . import proxy_mock as pm

        cache = Path(self.config.cache.dir).expanduser()
        p = self._proxy_port()
        with contextlib.suppress(Exception):
            self.device.set_http_proxy(None)
        if p is not None:
            with contextlib.suppress(Exception):
                self.device.adb_reverse_remove(p)
        stopped = pm.stop_mitm(cache)
        return {"ok": True, "action": "proxy-stop", "stopped": stopped, "port": p}

    def mock_map(
        self,
        method: str,
        path: str,
        *,
        status: int = 200,
        body: str | None = None,
    ) -> dict[str, Any]:
        from . import proxy_mock as pm

        cache = Path(self.config.cache.dir).expanduser()
        rules_file = pm.rules_path(cache)
        rules = pm.load_rules(rules_file)
        rule = pm.map_rule(method, path, status=status, body=body)
        rules.append(rule)
        pm.write_rules(rules_file, rules)
        return {"ok": True, "action": "mock-map", "rule": rule, "count": len(rules)}

    def mock_record(self, action: str, name: str | None = None) -> dict[str, Any]:
        from . import proxy_mock as pm

        cache = Path(self.config.cache.dir).expanduser()
        a = (action or "").lower()
        if a == "start":
            if not name:
                raise UsageError("mock record start needs a NAME")
            pm.record_path(cache).write_text("[]", encoding="utf-8")
            # Restart mitm in record mode if running; otherwise just arm the sidecar.
            env_mode = cache / "mock_mode.txt"
            env_mode.write_text("record", encoding="utf-8")
            (cache / "mock_record_name.txt").write_text(name, encoding="utf-8")
            # Live addon reads AUA_MOCK_MODE from process env — restart to flip mode.
            # Keep the same listen port when one is already bound so adb reverse stays valid.
            prev = pm.load_listen_port(cache)
            pm.stop_mitm(cache)
            _pid, listen = pm.start_mitm(cache_dir=cache, port=prev, mode="record")
            with contextlib.suppress(Exception):
                self.device.adb_reverse(listen, listen)
                self.device.set_http_proxy(f"127.0.0.1:{listen}")
            return {
                "ok": True,
                "action": "mock-record-start",
                "name": name,
                "port": listen,
            }
        if a == "stop":
            name_path = cache / "mock_record_name.txt"
            rec_name = name or (
                name_path.read_text(encoding="utf-8").strip() if name_path.is_file() else ""
            )
            if not rec_name:
                raise UsageError("mock record stop needs the cassette NAME")
            entries: list[dict[str, Any]] = []
            rec = pm.record_path(cache)
            if rec.is_file():
                try:
                    import json as _json

                    data = _json.loads(rec.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        entries = data
                except Exception:
                    entries = []
            dest = pm.cassette_dir(self.config.memory.dir) / f"{rec_name}.yaml"
            pm.save_cassette(dest, rec_name, entries)
            # Flip back to map mode on the same port.
            prev = pm.load_listen_port(cache)
            pm.stop_mitm(cache)
            _pid, listen = pm.start_mitm(cache_dir=cache, port=prev, mode="map")
            with contextlib.suppress(Exception):
                self.device.adb_reverse(listen, listen)
                self.device.set_http_proxy(f"127.0.0.1:{listen}")
            out: dict[str, Any] = {
                "ok": True,
                "action": "mock-record-stop",
                "name": rec_name,
                "path": str(dest),
                "entries": len(entries),
                "port": listen,
            }
            if not entries:
                tls = pm.tls_failures_in_log(cache)
                out["ok"] = False
                out["code"] = "proxy_tls"
                out["hint"] = (
                    "Recorded 0 HTTP flows. Mitm saw CONNECTs but TLS failed — the app "
                    "does not trust the mitm CA (its NSC is system-only). "
                    "Re-run `aua proxy start` on a rootable emulator so the system CA "
                    "overlay is installed, then force-stop + relaunch the app."
                )
                if tls:
                    out["tls_errors"] = tls
            return out
        raise UsageError(
            f"unknown mock record action {action!r}",
            hint="Use `aua mock record start NAME` or `aua mock record stop`.",
        )

    def mock_replay(
        self,
        name: str,
        *,
        _snapshot: _ResolvedCassetteResource | None = None,
    ) -> dict[str, Any]:
        from . import proxy_mock as pm

        cache = Path(self.config.cache.dir).expanduser()
        if _snapshot is None:
            path = pm.cassette_dir(self.config.memory.dir) / f"{name}.yaml"
            if not path.is_file():
                # also accept a direct path
                alt = Path(name).expanduser()
                path = alt if alt.is_file() else path
            entries = pm.load_cassette(path)
        else:
            path = _snapshot.source_path
            entries = deepcopy(_snapshot.entries)
        pm.write_rules(pm.rules_path(cache), entries)
        self._record_action_safe(RouteStep(kind="mock-replay", arg=name))
        return {
            "ok": True,
            "action": "mock-replay",
            "name": name,
            "entries": len(entries),
            "path": str(path),
        }

    def erase(
        self,
        element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        chars: int | None = None,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        """Erase text in a field (Maestro ``eraseText``): focus + delete *chars* or clear all."""
        el = (
            self._target(element_id, selector, verb="erase")
            if (element_id is not None or selector)
            else None
        )
        with self._acting():
            if el is not None:
                cx, cy = el.center
                self.device.click(cx, cy)
            if chars is None or chars <= 0:
                self.device.clear_text()
            else:
                self.device.erase_chars(chars)
        detail = "all" if not chars or chars <= 0 else str(chars)
        return self._observe(
            ActionResult(ok=True, action="erase", id=el.id if el else None, detail=detail),
            observe,
            with_image,
        )

    def app(
        self,
        action: str,
        *,
        package: str | None = None,
        activity: str | None = None,
        clear_state: bool = False,
        confirmed: bool = False,
        observe: bool = True,
        with_image: bool | str | None = None,
    ) -> ActionResult:
        device = self.device
        a = action.lower()
        if a in ("foreground", "current"):
            info = device.current_app()
            return ActionResult(ok=True, action=f"app-{a}", detail=json.dumps(info))
        if a == "launch":
            if not package:
                raise UsageError("app launch needs a package name")
            if clear_state and not confirmed:
                raise UsageError(
                    "launch --clear wipes app data (flags + session) — pass --yes",
                    hint="`aua app launch <pkg> --clear --yes`",
                )
            mem = self._memory
            if mem is not None and not self._join_memory_writers(timeout_s=5.0):
                raise UsageError("memory provenance is still being finalized")
            # --activity pins the entry Activity — some builds have multiple launcher
            # activities (e.g. a Dev Tools menu) and default resolution picks whichever the
            # manifest lists first, which is not necessarily the product's own entry.
            entry, launch_note = self._launch_entry(package, activity)
            # Journal the launch. Without this it was invisible to `session review`, which then
            # reported 10 calls for an 18-call run — and the invisible ones were the crash
            # recovery, i.e. exactly the work its efficiency advice was reasoning about.
            step = self._step("app-launch", arg=package)
            with self._acting():
                if clear_state:
                    device.clear_app(package)
                device.launch_app(package, activity=entry)
            self._record_action_safe(step)
            if mem is not None:
                with self._mem_lock:
                    if clear_state:
                        mem.clear_context(device.serial, package)
                    else:
                        mem.mark_capture_boundary(
                            device.serial,
                            package,
                            f"app process launched for {package}",
                        )
                        mem.promote_pending_context(
                            device.serial,
                            package,
                            app_version=self._version_for(device, package),
                        )
            detail = f"{package}/{entry}" if entry else package
            if clear_state:
                detail = f"{detail} (cleared)"
            if not self._await_foreground(device, package):
                # uiautomator2's app_start swallows `am start` failures, so a launch that never
                # happened used to answer ok=True. The caller then drives a screen that is not
                # there and every selector fails with an unrelated "no element matches".
                raise DeviceError(
                    f"launched {detail} but {package} never reached the foreground",
                    hint=(
                        "That Activity may not be exported (`am start` denies it) — retry "
                        "without --activity."
                        if activity
                        # A wrong pin is silent otherwise: the caller never asked for this
                        # Activity, so "retry without --activity" would be misleading advice.
                        else (
                            "The remembered launch Activity may be wrong or unexported — re-pin "
                            "with `aua remember --launch-activity <Activity>`."
                            if entry
                            else "Check the package name, and that the device is unlocked."
                        )
                    ),
                )
            if mem is not None and activity:
                # Only an explicit --activity teaches a NEW pin here: a reused pin needs no
                # rewrite, and the single-launcher case was already pinned while resolving.
                with self._mem_lock:
                    mem.remember_launch_entry(package, activity, source="explicit")
            # `_acting()` starts a speculative hierarchy dump as soon as the launch command
            # returns. Foreground verification happens afterwards, so that speculative slot
            # may describe the app we just left or a half-attached transition window. Never
            # let the authoritative launch readback consume it, and never reuse the previous
            # app's unchanged-screen payload across this lifecycle boundary.
            self._prefetch.invalidate()
            self._last_hierarchy_hash = None
            self._last_analyze_result = None
            # `launch` is the first action of nearly every journey, and it used to answer with a
            # bare ok/detail: no fresh ids, and no statement of what the launch actually produced.
            # Callers then spent a separate `analyze` to learn where they had landed, and had
            # nothing structured to show for the step. `_await_foreground` above already proves the
            # package reached the foreground, so this adds the *screen* to that proof — the same
            # act-and-observe contract every other action honours.
            launched = self._observe(
                ActionResult(ok=True, action="app-launch", detail=detail), observe, with_image
            )
            if (
                observe
                and launched.observation is not None
                and not launched.observation.screen.package
            ):
                # A hierarchy provider may be unable to attribute nodes to a package. The
                # foreground check immediately above is authoritative for that missing field,
                # so bind the otherwise useful landing observation to the verified package.
                try:
                    foreground = str((self.device.current_app() or {}).get("package") or "")
                except Exception:  # noqa: BLE001 — absence of ownership proof must fail closed
                    foreground = ""
                if foreground != package:
                    self._invalidate_launch_observation()
                    raise DeviceError(
                        (
                            f"{package} reached the foreground, but ownership changed to "
                            f"{foreground or 'an unknown package'} while the hierarchy had no "
                            "package attribution"
                        ),
                        code="launch_observation_mismatch",
                        hint=(
                            "Inspect one fresh hierarchy before acting; AUA did not attribute an "
                            "unowned hierarchy to the launched app."
                        ),
                    )
                launched.observation.screen.package = package
                self._write_cache(launched.observation)
            elif (
                observe
                and launched.observation is not None
                and launched.observation.screen.package != package
            ):
                # Foreground verification and hierarchy capture are separate Android reads. A
                # transition race can satisfy the former while the latter still belongs to the
                # app we left or to a short-lived SystemUI attachment frame. Fresh hierarchy-only
                # reads may heal that race, but only while foreground ownership remains proven and
                # only inside a small bound; a persistent mismatch stays a typed failure.
                fresh = self._await_launch_hierarchy(package)
                self._adopt_recovered_launch_observation(launched, fresh)
            if launch_note:
                # `_observe` owns `note` when it attaches a screen, so the ambiguity warning is
                # prepended afterwards rather than passed in — it must not be silently dropped.
                launched.note = f"{launch_note} {launched.note}" if launched.note else launch_note
            return launched
        if a in ("kill", "force-stop"):
            if not package:
                raise UsageError("app kill needs a package name")
            mem = self._memory
            if mem is not None and not self._join_memory_writers(timeout_s=5.0):
                raise UsageError("memory provenance is still being finalized")
            with self._acting():
                device.stop_app(package)
            if mem is not None:
                with self._mem_lock:
                    mem.mark_capture_boundary(
                        device.serial,
                        package,
                        f"app process stopped for {package}",
                    )
            return ActionResult(ok=True, action="app-kill", detail=package)
        if a == "stop":
            if not package:
                raise UsageError("app stop needs a package name")
            mem = self._memory
            if mem is not None and not self._join_memory_writers(timeout_s=5.0):
                raise UsageError("memory provenance is still being finalized")
            with self._acting():
                device.stop_app(package)
            if mem is not None:
                with self._mem_lock:
                    mem.mark_capture_boundary(
                        device.serial,
                        package,
                        f"app process stopped for {package}",
                    )
            return ActionResult(ok=True, action="app-stop", detail=package)
        if a in ("clear", "clear-state", "clear_state"):
            if not package:
                raise UsageError("app clear needs a package name")
            if not confirmed:
                raise UsageError(
                    "app clear wipes ALL app data (feature flags, login session, local config) "
                    "— pass --yes / --yes-wipe-flags to confirm",
                    hint="Then re-apply flag overrides / re-login before asserting experiment UI.",
                )
            mem = self._memory
            if mem is not None and not self._join_memory_writers(timeout_s=5.0):
                raise UsageError("memory provenance is still being finalized")
            with self._acting():
                device.clear_app(package)
            if mem is not None:
                with self._mem_lock:
                    mem.clear_context(device.serial, package)
            return ActionResult(ok=True, action="app-clear", detail=package)
        if a in ("grant", "grant-permissions", "grant_permissions"):
            if not package:
                raise UsageError("app grant needs a package name")
            device.grant_permissions(package)
            return ActionResult(ok=True, action="app-grant", detail=package)
        raise UsageError(
            f"unknown app action '{action}'",
            hint="foreground|launch|stop|kill|clear|grant|current",
        )

    # ----------------------------------------------------------------- app bundle installs

    #: Install modes, narrowest first. ``if-needed`` is the default because the common case is a
    #: run that just wants the build present, and re-pushing an APK that is already there costs
    #: tens of seconds on an emulator for no change in state.
    INSTALL_MODES = ("if-needed", "reinstall", "fresh")

    def install_app(
        self,
        bundle: str,
        *,
        package: str | None = None,
        mode: str = "if-needed",
        confirmed: bool = False,
        grant_permissions: bool = False,
        launch: bool = False,
        activity: str | None = None,
        observe: bool = True,
        with_image: bool | str | None = None,
        # Milliseconds, not seconds, because the daemon sizes a request's socket budget from a
        # `timeout_ms` argument. An install that outran a 60s socket would come back as
        # `daemon_outcome_unknown` — the one error agents are told never to retry.
        timeout_ms: int = 300_000,
    ) -> ActionResult:
        """Put an app bundle on the target, optionally launching it, in one call.

        The three modes differ only in what they do when the app is *already* installed:
        ``if-needed`` leaves it alone unless the bundle's version differs, ``reinstall`` always
        pushes but keeps app data, and ``fresh`` uninstalls first — the only mode that survives a
        signing-key change, and the only one that destroys data, which is why it needs
        *confirmed*.

        ``launch=True`` folds :meth:`app` in afterwards so a caller gets bundle → installed →
        foreground → screen from a single request. That fold is the point: an install whose
        result has to be followed by a launch and then an analyze is three round-trips to learn
        one thing, and each extra call is another chance for the caller to skip the readback.
        """

        if mode not in self.INSTALL_MODES:
            raise UsageError(
                f"unknown install mode '{mode}'",
                hint=f"one of: {'|'.join(self.INSTALL_MODES)}",
            )
        platform = self.platform
        if not platform.supports("app.install"):
            raise DeviceError(
                f"platform '{platform.name}' cannot install app bundles",
                code="unsupported_capability",
            )
        path = Path(bundle).expanduser()
        info = platform.inspect_app_bundle(path)
        app_id = package or info.app_id
        if package and package != info.app_id:
            # A mismatch means the caller is about to install one app and then drive another;
            # every later selector would fail against a screen that is not the one named.
            raise UsageError(
                f"{path.name} declares package '{info.app_id}', not '{package}'",
                hint="Drop --package, or pass the bundle that really contains it.",
            )
        device = self.device
        before = platform.installed_app(device, app_id)
        pushed = False
        removed = False
        reason: str
        if not before.installed:
            reason = "missing"
            pushed = True
        elif mode == "fresh":
            reason = "fresh-requested"
            pushed = True
            removed = True
        elif mode == "reinstall":
            reason = "reinstall-requested"
            pushed = True
        elif _install_versions_differ(before, info):
            # `if-needed` still pushes on a version change: "the build under test is present" is
            # the request, and a stale build that merely shares a package id does not satisfy it.
            reason = "version-differs"
            pushed = True
        else:
            reason = "already-present"
        if removed and not confirmed:
            raise UsageError(
                f"install --fresh removes {app_id} and ALL its data (feature flags, login "
                "session, local config) — pass --yes to confirm",
                hint=f"Or keep the data: `aua app install {path} --reinstall`.",
            )
        started = time.perf_counter()
        if pushed:
            mem = self._memory
            if mem is not None and not self._join_memory_writers(timeout_s=5.0):
                raise UsageError("memory provenance is still being finalized")
            with self._acting():
                if removed:
                    platform.uninstall_app(device, app_id)
                platform.install_app_bundle(
                    device,
                    path,
                    replace=not removed,
                    grant_permissions=grant_permissions,
                    timeout_s=max(1.0, timeout_ms / 1000.0),
                )
            if mem is not None:
                with self._mem_lock:
                    # A new build is a new set of screens: element ids, copy, and routes learned
                    # from the previous one are no longer evidence about this one.
                    if removed:
                        mem.clear_context(device.serial, app_id)
                    else:
                        mem.mark_capture_boundary(
                            device.serial,
                            app_id,
                            f"app bundle installed for {app_id}",
                        )
            # Everything below describes the binary we just replaced. `_version_for` memoises a
            # versionName per package for memory provenance, and the analyze caches hold the
            # previous build's tree; leaving either in place makes the next read report the old
            # app's state under the new app's name.
            self._version_cache.pop(app_id, None)
            self._prefetch.invalidate()
            self._last_hierarchy_hash = None
            self._last_analyze_result = None
            after = platform.installed_app(device, app_id)
            if not after.installed:
                # adb can report a successful install for a package the manager never registered.
                raise DeviceError(
                    f"{path.name} installed without error but {app_id} is not on {device.serial}",
                    code="install_unverified",
                    hint="Check the bundle's package id and the device's remaining storage.",
                )
        else:
            after = before
        if grant_permissions and not pushed:
            # `-g` only applies to the install itself, so an idempotent skip would silently drop
            # the caller's permission request.
            device.grant_permissions(app_id)
        detail_info: dict[str, Any] = {
            "package": app_id,
            "installed": True,
            "pushed": pushed,
            "uninstalled_first": removed,
            "reason": reason,
            "mode": mode,
            "bundle": str(path),
            "bundle_version_name": info.version_name,
            "bundle_version_code": info.version_code,
            "version_name": after.version_name,
            "version_code": after.version_code,
            "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
        }
        summary = f"{app_id} {info.version_name or '?'}"
        summary += f" ({'installed' if pushed else 'already present — skipped'})"
        transient = platform.install_persistence_warning(device) if pushed else None
        if transient:
            detail_info["persists"] = False
            detail_info["persistence_note"] = transient
        if launch:
            launched = self.app(
                "launch",
                package=app_id,
                activity=activity,
                clear_state=False,
                confirmed=confirmed,
                observe=observe,
                with_image=with_image,
            )
            # `app restart` sets the precedent: a composed command returns the inner action's
            # result rather than renaming it, so a caller that branches on `action` still sees
            # the launch it is about to drive. What the install did travels in `app_install`.
            launched.app_install = detail_info
            launched.detail = f"{summary}; launched {launched.detail}"
            if transient:
                launched.note = f"{transient} {launched.note}" if launched.note else transient
            return launched
        # No observation: an install does not change what is on screen, so folding a hierarchy
        # dump in here would bill the caller for a read that tells them nothing. `app clear` and
        # `app stop` answer the same way.
        return ActionResult(
            ok=True,
            action="app-install",
            detail=summary,
            app_install=detail_info,
            note=transient,
        )

    # ----------------------------------------------------------------- app databases

    def database_list(self, package: str) -> dict[str, Any]:
        from . import app_database

        return app_database.list_databases(self.device, package)

    def database_schema(
        self,
        package: str,
        database: str,
        *,
        table: str | None = None,
        restart: bool = True,
    ) -> dict[str, Any]:
        from . import app_database

        return app_database.database_schema(
            self.device,
            package,
            database,
            table=table,
            restart=restart,
        )

    def database_query(
        self,
        package: str,
        database: str,
        sql: str,
        *,
        parameters: dict[str, Any] | list[Any] | None = None,
        limit: int = 100,
        timeout_ms: int = 5000,
        restart: bool = True,
    ) -> dict[str, Any]:
        from . import app_database

        return app_database.query_database(
            self.device,
            package,
            database,
            sql,
            parameters=parameters,
            limit=limit,
            timeout_ms=timeout_ms,
            restart=restart,
        )

    def database_execute(
        self,
        package: str,
        database: str,
        sql: str,
        *,
        parameters: dict[str, Any] | list[Any] | None = None,
        timeout_ms: int = 5000,
        restart: bool = True,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        from . import app_database

        return app_database.execute_database(
            self.device,
            self.config.cache.dir,
            package,
            database,
            sql,
            parameters=parameters,
            timeout_ms=timeout_ms,
            restart=restart,
            confirmed=confirmed,
        )

    def database_backup(
        self,
        package: str,
        database: str,
        *,
        restart: bool = True,
    ) -> dict[str, Any]:
        from . import app_database

        return app_database.backup_database(
            self.device,
            self.config.cache.dir,
            package,
            database,
            restart=restart,
        )

    def database_backups(self, package: str, database: str) -> dict[str, Any]:
        from . import app_database

        return app_database.list_backups(
            self.device,
            self.config.cache.dir,
            package,
            database,
        )

    def database_restore(
        self,
        package: str,
        database: str,
        backup_id: str,
        *,
        restart: bool = True,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        from . import app_database

        return app_database.restore_database(
            self.device,
            self.config.cache.dir,
            package,
            database,
            backup_id,
            restart=restart,
            confirmed=confirmed,
        )

    # ----------------------------------------------------------------- logcat / suite

    def logcat_mark(self, name: str = "default", *, clear: bool = False) -> dict[str, Any]:
        """Store a named device-clock mark (and optionally clear the device logcat buffer).

        Measures the skew fresh: this is the user-invoked entry point, the one place where
        an adb round-trip is affordable and where reporting real drift is the whole point.
        """
        from . import logcat as logcat_mod

        device = self.device
        if clear:
            device.logcat(dump=False)
        clock = logcat_mod.resolve_clock(device, self.config.cache.dir, force=True)
        entry = logcat_mod.set_mark(
            self.config.cache.dir, device.serial, name or "default", clock=clock
        )
        return {"ok": True, "action": "logcat-mark", **entry}

    def logcat(
        self,
        *,
        grep: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        lines: int | None = None,
    ) -> dict[str, Any]:
        """Dump recent logcat, filtered by mark / grep / tag / line count."""
        from . import logcat as logcat_mod

        device = self.device
        path = logcat_mod.marks_path(self.config.cache.dir, device.serial)
        marks = logcat_mod.load_marks(path)
        clock = logcat_mod.resolve_clock(device, self.config.cache.dir)
        try:
            since_ms, since_label = logcat_mod.resolve_since_ms(marks, since, clock=clock)
        except KeyError as exc:
            known = ", ".join(sorted(marks)) or "(none)"
            raise UsageError(
                f"unknown logcat mark {since!r}",
                hint=f"Known marks: {known}. Set one with `aua logcat mark <name>`.",
            ) from exc
        raw = device.logcat(since_ms=since_ms, dump=True)
        filtered = logcat_mod.filter_logcat(raw, grep=grep, tag=tag, lines=lines)
        return {
            "ok": True,
            "lines": filtered,
            "since": since_label,
            "since_unix_ms": since_ms,
            "clock": clock.name,
            "skew_ms": clock.skew_ms,
            "grep": grep,
            "tag": tag,
            "count": len(filtered),
        }

    def suite_run(
        self,
        path: str,
        *,
        continue_on_fail: bool = False,
        text: str | None = None,
    ) -> dict[str, Any]:
        """Run an AC checklist YAML (path, or *text* when path is ``-``)."""
        from . import suite as suite_mod

        if text is not None:
            suite = suite_mod.parse_suite(text, source=path or "<stdin>")
        elif path == "-":
            raise UsageError(
                "suite run from stdin needs the YAML body passed as text",
                hint="CLI reads stdin when PATH is `-`.",
            )
        else:
            suite = suite_mod.load_suite(path)
        result = suite_mod.run_suite(self, suite, continue_on_fail=continue_on_fail)
        return result.as_dict()

    # ----------------------------------------------------------------- doctor

    def provider_status(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for kind in ("ocr", "detection", "grounding", "planner"):
            chain_names = self.factory.chain_names(kind)
            enabled = self.factory.is_enabled(kind)
            items: list[dict[str, Any]] = []
            for name in registered_names(kind):
                try:
                    prov = self.factory.create(kind, name)
                    avail = prov.is_available()
                    items.append(
                        {
                            "name": name,
                            "available": avail.ok,
                            "reason": avail.reason,
                            "in_chain": name in chain_names,
                            "kind_enabled": enabled,
                        }
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    items.append(
                        {
                            "name": name,
                            "available": False,
                            "reason": f"init error: {exc}",
                            "in_chain": name in chain_names,
                            "kind_enabled": enabled,
                        }
                    )
            out[kind] = items
        return out

    # ----------------------------------------------------------------- annotate

    def _with_raw_image(self, result: AnalyzeResult, with_image: bool | str) -> AnalyzeResult:
        img = self._screenshot(max_reuse_ms=80.0)
        out = (
            with_image
            if isinstance(with_image, str)
            else self._default_annotate_path(self.device.serial, suffix="screen", timestamped=True)
        )
        img.save(out)
        result.meta.raw_image = out
        return result

    def _maybe_annotate(
        self,
        annotate: bool | str | None,
        device: Device,
        elements: list[Element],
        img: ScreenImage | None,
    ) -> str | None:
        if not annotate:
            return None
        from . import annotate as annotate_mod

        if img is None:
            img = self._screenshot(max_reuse_ms=80.0)
        out = annotate if isinstance(annotate, str) else self._default_annotate_path(device.serial)
        return annotate_mod.annotate(img, elements, out)

    def _default_annotate_path(
        self, serial: str, *, suffix: str = "annotated", timestamped: bool = False
    ) -> str:
        run_dir = Path(self.config.cache.dir).expanduser() / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        safe = serial.replace(":", "_")
        if timestamped:
            # Sequential captures (before/after an action) must never clobber each other.
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
            return str(run_dir / f"{safe}_{suffix}_{stamp}.png")
        return str(run_dir / f"{safe}_{suffix}.png")

    # ----------------------------------------------------------------- cache

    def _cache_path(self, serial: str | None = None) -> Path:
        # Resolve the real connected serial on reads (config serial may be null =
        # auto-detected) so a `tap`/`inspect` process keys the same file `analyze`
        # wrote. Writes pass the serial explicitly and never trigger a connect here.
        if serial is None:
            serial = self._device.serial if self._device else self.device.serial
        cache_dir = Path(self.config.cache.dir).expanduser()
        safe = str(serial).replace(":", "_")
        return cache_dir / f"analyze_{safe}.json"

    def _write_cache(self, result: AnalyzeResult) -> None:
        if not self.config.cache.enabled:
            return
        path = self._cache_path(result.meta.device_serial)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.model_dump_json(), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues
            logger.warning("could not write analyze cache: %s", exc)

    def _read_cache(self) -> AnalyzeResult | None:
        path = self._cache_path()
        if not path.is_file():
            return None
        try:
            return AnalyzeResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupt cache
            logger.warning("ignoring corrupt analyze cache: %s", exc)
            return None

    @contextlib.contextmanager
    def _acting(
        self, label: str | None = None, *, capture_pre_action: bool = True
    ) -> Iterator[None]:
        """Bracket a device interaction: open the log window, then drop the stale id cache.

        Wrap the interaction rather than following it, because the two halves belong on
        opposite sides of it. ``last-action`` has to be stamped BEFORE the device is
        touched — ``logcat --since last-action`` means "since just before the last action",
        so it must cover what the app logged *in response*. A stamp taken after
        ``device.click()`` returns excludes exactly those lines, and under-reporting a
        window looks identical to "the app logged nothing". The id cache, conversely, can
        only be known stale once the interaction has happened.

        Being a context manager is the point: an action cannot complete without having
        opened its window first, so the ordering cannot come apart per-action again.
        """
        self._mark_logcat("last-action")
        # Same reasoning as the log window: mark the capture timeline BEFORE the
        # interaction so the post-action burst records the transition itself, not just
        # whatever is on screen once it has already settled.
        buf = self._capture
        if buf is not None:
            with contextlib.suppress(Exception):
                buf.mark(label or "action")
        # Any speculative hierarchy dump is stale the moment we touch the device.
        self._prefetch.invalidate()
        self._action_observation_baseline = None
        # Pixel fingerprint for settle-then-observe: must be taken BEFORE the gesture.
        self._pre_action_sig = None
        if capture_pre_action:
            with contextlib.suppress(Exception):
                from . import imaging

                self._pre_action_sig = imaging.frame_signature(self._screenshot(max_reuse_ms=40.0))
        # Cheap tree fingerprint from the last analyze (no extra dump) — used to
        # early-accept observe once the accessibility tree has moved and stabilised.
        self._pre_action_tree_fp = self._tree_fingerprint()
        # Same cache read, richer shape: what the result will diff against to say what changed.
        self._pre_action_state = self._pre_action_snapshot()
        yield
        self._invalidate_cache()
        # Speculative dump while the UI is settling / the agent thinks.
        if self.config.perf.prefetch or self.config.perf.predictive_prefetch:
            self._kick_hierarchy_prefetch()

    def _pre_action_snapshot(self) -> dict[str, Any] | None:
        """What the screen was, cheaply, so the next result can say what *changed*.

        Observed need: an action reported that it dispatched, never what it did. The most reusable
        technique produced all week was reading the resumed activity — it names what is in front of
        the user, so after tapping something that should open a picker it says whether the picker
        opened. That is a fact about the system rather than a reading of the app's description of
        itself, and it is what settled a disputed critical failure.

        Deliberately **not** a confidence score, which is what was originally asked for. A number
        invites trusting a figure over evidence, and the founding lesson of this whole list is that
        a command reporting success is not evidence of effect. What lanes needed was never "how
        sure are you" but "what changed".

        Nearly free: the shape comes from the analyze already in cache, so no device call. The
        activity costs one call **only when it is not already known** — a plain hierarchy `analyze`
        never learns it (only the vision path fetches app context), so the first action has to ask.
        Every later action reuses the value the previous observation recorded, so a sequence pays
        once rather than per action.
        """
        cached = self._read_cache()
        if cached is None:
            return None
        labels: list[str] = []
        rids: list[str] = []
        focused: int | None = None
        for e in cached.elements:
            if e.focused and focused is None:
                focused = e.id
            label = (e.text or e.content_desc or "").strip()
            if label:
                labels.append(_label(label))
            rid = _id_tail(e.resource_id)
            if rid:
                rids.append(rid)
        return {
            "count": len(cached.elements),
            "focused": focused,
            "labels": labels,
            "rids": rids,
            "arrival_identity": self._await_observation_identity(cached),
            "package": cached.screen.package,
            "activity": self._last_activity or self._read_activity(),
            "known_screen": (
                cached.meta.known_screen if cached.meta is not None else self._last_known_screen
            ),
        }

    @staticmethod
    def _app_left_foreground(
        activity_before: str | None, activity_after: str | None, obs: AnalyzeResult
    ) -> dict[str, Any] | None:
        """Report the app under test vanishing from the foreground — nearly always a crash.

        A tap that kills the app answered ``ok: true`` with a cheerful observation of the
        launcher, leaving the caller to infer the crash from ``activity_after`` by hand.
        A weaker caller does not make that leap: it concludes the button "navigated home"
        and then spends its whole budget trying to navigate back inside a dead app.

        Both signals Android gives us are checked — the system's ``aerr_*`` crash dialog, and
        the foreground falling back to a launcher. An ordinary app-to-app hand-off (a share
        sheet, a browser) is deliberately NOT reported: the package changing is normal there.
        """

        def package_of(activity: str | None) -> str | None:
            if not activity or "/" not in activity:
                return None
            return activity.split("/", 1)[0] or None

        before_pkg = package_of(activity_before)
        after_pkg = package_of(activity_after)
        if not before_pkg or not after_pkg or before_pkg == after_pkg:
            return None
        crash_dialog = any("aerr_" in str(e.resource_id or "") for e in obs.elements)
        to_launcher = any(hint in after_pkg.lower() for hint in ("launcher", "home"))
        if not crash_dialog and not to_launcher:
            return None
        return {"from": before_pkg, "to": after_pkg, "crash_dialog": crash_dialog}

    def _change_summary(self, before: dict[str, Any] | None, obs: AnalyzeResult) -> dict[str, Any]:
        """Structured before/after deltas, with "nothing changed" stated rather than implied.

        ``changed`` is an explicit boolean so a caller can branch on it without re-deriving the
        answer from four other fields — and so "nothing changed" is machine-checkable, which is
        the half that was missing. An unknown baseline is reported as ``None``, never as False:
        "I could not compare" and "they are the same" are different claims.
        """
        # Not shortened. These become `change.text_added` / `text_removed`, which is the field a
        # caller reads to decide whether the action did what it was for — so a cut here lands in
        # the verdict itself. Saving a few dozen characters there is a false economy: the reader
        # cannot tell a clipped line from a complete one, and the way out is another `analyze`,
        # which costs a round trip and kilobytes to recover the tens of bytes that were withheld.
        after_labels = [
            (e.text or e.content_desc or "").strip()
            for e in obs.elements
            if (e.text or e.content_desc or "").strip()
        ]
        after_focus = next((e.id for e in obs.elements if e.focused), None)
        activity_before = (before or {}).get("activity") or self._last_activity
        activity_after = self._read_activity()
        if activity_after is not None:
            self._last_activity = activity_after

        out: dict[str, Any] = {
            "activity_before": activity_before,
            "activity_after": activity_after,
            "activity_changed": (
                None
                if activity_before is None or activity_after is None
                else activity_before != activity_after
            ),
            "node_count_after": len(obs.elements),
        }
        left = self._app_left_foreground(activity_before, activity_after, obs)
        if left is not None:
            out["app_left_foreground"] = left
        if before is None:
            # No baseline: say so instead of implying stability from silence.
            out.update(
                {
                    "changed": None if out["activity_changed"] is None else out["activity_changed"],
                    "node_count_before": None,
                    "node_count_delta": None,
                    "focus_moved": None,
                    "text_added": [],
                    "text_removed": [],
                    "detail": "no pre-action snapshot — deltas unavailable",
                }
            )
            return out

        added = [t for t in dict.fromkeys(after_labels) if t not in set(before["labels"])]
        removed = [t for t in dict.fromkeys(before["labels"]) if t not in set(after_labels)]
        focus_moved = before["focused"] != after_focus
        out.update(
            {
                "node_count_before": before["count"],
                "node_count_delta": len(obs.elements) - before["count"],
                "focus_moved": focus_moved,
                "text_added": added[:_CHANGE_TEXT_CAP],
                "text_removed": removed[:_CHANGE_TEXT_CAP],
            }
        )
        out["changed"] = bool(
            out["activity_changed"] or added or removed or out["node_count_delta"] or focus_moved
        )
        if not out["changed"]:
            out["detail"] = (
                "nothing changed: same activity, same node count, no text added or removed, "
                "focus unmoved"
            )
        return out

    def _read_activity(self) -> str | None:
        """``package/activity`` in front of the user, or None if it cannot be read.

        One device call, only on the observe path — which already spends a settle and a full
        hierarchy dump, so this is a small fraction of a cost the caller has already accepted.
        Never sampled before the action: the baseline is chained from the previous observation, so
        a sequence of actions gets its comparison for free.
        """
        with contextlib.suppress(Exception):
            info = self.device.current_app() or {}
            package = str(info.get("package") or "")
            activity = str(info.get("activity") or "")
            if package or activity:
                return f"{package}/{activity}"
        return None

    def _tree_fingerprint(self) -> tuple[str, ...] | None:
        """Stable-ish fingerprint of the last cached screen (rids + labels)."""
        cached = self._read_cache()
        if cached is None:
            return None
        parts: list[str] = []
        for e in cached.elements:
            if getattr(e, "window", None) == "system":
                continue
            rid = (e.resource_id or "").split("/")[-1]
            label = (e.text or e.content_desc or "")[:40]
            if rid or label:
                parts.append(f"{rid}:{label}")
        return tuple(parts[:60]) if parts else None

    def _capture_hint(self) -> str | None:
        buf = self._capture
        if buf is None or not self.config.capture.hint:
            return None
        if not buf.hint_ready():
            return None
        return "recent pixel change after last action — `aua capture last --since last-action`"

    def capture_start(self) -> dict[str, Any]:
        """Start the rolling capture buffer (daemon-warm sessions)."""
        from .capture import CaptureBuffer, CaptureCfgView

        if not self.config.capture.enabled and self._capture is None:
            # Explicit start still allowed even if config default is off.
            pass
        cfg = self.config.capture
        device = self.device
        root = Path(self.config.cache.dir).expanduser() / "captures"
        view = CaptureCfgView(
            enabled=True,
            idle_fps=cfg.idle_fps,
            burst_fps=cfg.burst_fps,
            burst_ms=cfg.burst_ms,
            extend_burst_on_change=cfg.extend_burst_on_change,
            ttl_s=cfg.ttl_s,
            max_mb=cfg.max_mb,
            jpeg_quality=cfg.jpeg_quality,
            hint=cfg.hint,
        )
        if self._capture is not None:
            self._capture.resume()
            if not self._capture.running:
                self._capture.start()
            return self._capture.status()
        # Default is the u2 path: it is ~2.2x faster than `adb exec-out screencap -p` (the
        # device encodes JPEG instead of a full-res PNG) and every frame is re-encoded to JPEG
        # on write anyway, so the lossless capture buys nothing here. The opt-in flag stays for
        # callers that need pixel-exact frames.
        shot = device.screenshot
        if self.config.perf.capture_adb_screencap:
            shot = getattr(device, "screencap_png", device.screenshot)
        buf = CaptureBuffer(
            root=root,
            serial=device.serial,
            cfg=view,
            screenshot=shot,
        )
        buf.start()
        self._capture = buf
        return buf.status()

    def capture_stop(self) -> dict[str, Any]:
        buf = self._capture
        if buf is None:
            return {"ok": True, "action": "capture-stop", "running": False}
        buf.stop()
        self._capture = None
        return {
            "ok": True,
            "action": "capture-stop",
            "running": False,
            "session_id": buf.session_id,
        }

    def capture_on(self) -> dict[str, Any]:
        if self._capture is None:
            return self.capture_start()
        self._capture.resume()
        return self._capture.status()

    def capture_off(self) -> dict[str, Any]:
        if self._capture is None:
            return {"ok": True, "action": "capture-status", "running": False, "paused": True}
        self._capture.pause()
        return self._capture.status()

    def capture_idle_pause(self) -> bool:
        """Stop sampling because the client went quiet; frames already kept stay readable."""
        buf = self._capture
        if buf is None or not buf.running or buf.paused:
            return False
        buf.pause("idle")
        return True

    def capture_idle_resume(self) -> bool:
        """Resume a buffer that idle-paused. An explicit ``capture off`` stays off."""
        buf = self._capture
        if buf is None or not buf.paused:
            return False
        buf.resume(only_if_idle=True)
        return not buf.paused

    def capture_status(self) -> dict[str, Any]:
        if self._capture is None:
            return {
                "ok": True,
                "action": "capture-status",
                "running": False,
                "paused": False,
                # Do NOT assert the daemon is down: this same answer comes back from
                # INSIDE a warm daemon whose buffer is simply off, and telling the caller
                # to start a daemon they already started sends them down a blind alley.
                "hint": (
                    "no capture buffer in this process — `aua capture on` if the daemon is "
                    "warm, otherwise `aua daemon start` (capture.enabled) first."
                ),
            }
        return self._capture.status()

    def capture_last(
        self,
        *,
        seconds: float | None = None,
        since: str | None = None,
        region: str | None = None,
        where_rid: str | None = None,
    ) -> dict[str, Any]:
        if self._capture is None:
            raise UsageError(
                "capture buffer is not running",
                hint=(
                    "`aua capture on` if the daemon is warm, otherwise `aua daemon start` "
                    "(capture.enabled) first — then re-run. Frames only exist while it runs."
                ),
            )
        since_ms: int | None = None
        if since:
            s = since.lower().strip()
            if s in ("last-action", "last_action", "action"):
                since_ms = self._capture.last_action_ms()
                if since_ms is None:
                    raise UsageError(
                        "no last-action mark in the capture buffer yet",
                        hint="Perform a tap/input/swipe first, then retry.",
                    )
            else:
                since_ms = self._capture.last_action_ms()
        resolved_region = region
        if where_rid and not resolved_region:
            resolved_region = self._region_for_rid(where_rid)
        return self._capture.last(
            seconds=seconds,
            since_ms=since_ms,
            region=resolved_region,
            where_rid=where_rid,
        )

    def _region_for_rid(self, rid: str) -> str | None:
        """Best-effort grid cell for a resource-id from the last analyze cache."""
        cached = self._read_cache()
        if cached is None:
            return None
        want = rid.strip()
        for el in cached.elements:
            if not el.resource_id:
                continue
            if (
                el.resource_id == want
                or el.resource_id.endswith("/" + want)
                or _id_tail(el.resource_id) == want
            ):
                w = cached.screen.width or 0
                h = cached.screen.height or 0
                if w <= 0 or h <= 0:
                    with contextlib.suppress(Exception):
                        w, h = self.device.window_size()
                if w > 0 and h > 0:
                    cx, cy = el.center
                    return _region_from_point(cx, cy, w, h)
        return None

    def capture_export(
        self,
        path: str,
        *,
        seconds: float | None = None,
        since: str | None = None,
        fmt: str = "gif",
        fps: float = 8.0,
    ) -> dict[str, Any]:
        if self._capture is None:
            raise UsageError(
                "capture buffer is not running",
                hint="`aua capture on` / `aua daemon start` first.",
            )
        since_ms = None
        if since and since.lower().strip() in ("last-action", "last_action", "action"):
            since_ms = self._capture.last_action_ms()
        try:
            return self._capture.export(path, seconds=seconds, since_ms=since_ms, fmt=fmt, fps=fps)
        except (ValueError, ImportError) as exc:
            raise UsageError(str(exc)) from exc

    def capture_explain(
        self,
        *,
        seconds: float | None = None,
        since: str | None = None,
        llm: bool = False,
    ) -> dict[str, Any]:
        """Narrate the recent capture window (local summary; optional LLM)."""
        if self._capture is None:
            raise UsageError(
                "capture buffer is not running",
                hint="`aua capture on` / `aua daemon start` first.",
            )
        since_ms = None
        if since and since.lower().strip() in ("last-action", "last_action", "action"):
            since_ms = self._capture.last_action_ms()
        out = self._capture.explain_local(seconds=seconds, since_ms=since_ms)
        if llm:
            out["llm"] = self._capture_explain_llm(out)
        return out

    def _capture_explain_llm(self, payload: dict[str, Any]) -> str | None:
        """Best-effort narration via the planner chain (opt-in)."""
        try:
            if not self.factory.is_enabled("planner"):
                return "(llm skipped: planner disabled — enable planner in config or use local narration)"
            names = self.factory.chain_names("planner")
            objective = (
                "Summarize this Android UI transition for a QA agent in 2-4 sentences.\n"
                f"{payload.get('narration')}\n"
                f"Diff lines: {payload.get('summary')}"
            )
            for name in names:
                try:
                    prov = self.factory.create("planner", name)
                    if not prov.is_available().ok:
                        continue
                    decision = prov.decide(objective, [])
                    if decision is None:
                        continue
                    text = getattr(decision, "reason", None) or getattr(decision, "thought", None)
                    return str(text or decision)[:2000]
                except Exception:
                    continue
        except Exception as exc:
            return f"(llm error: {exc})"
        return None

    def capture_prune(self) -> dict[str, Any]:
        if self._capture is None:
            return {"ok": True, "action": "capture-prune", "removed": 0, "running": False}
        return self._capture.prune()

    def capture_sidecar_start(self) -> dict[str, Any]:
        """Start a host-side capture sidecar (survives without the full daemon)."""
        from . import capture_sidecar as cs

        if not self.config.capture.sidecar:
            raise UsageError(
                "capture sidecar is disabled",
                hint="Set capture.sidecar: true in config.",
            )
        return cs.start(
            serial=self.device.serial,
            cache_dir=Path(self.config.cache.dir).expanduser(),
            cfg=self.config.capture,
        )

    def capture_sidecar_stop(self) -> dict[str, Any]:
        from . import capture_sidecar as cs

        return cs.stop(Path(self.config.cache.dir).expanduser())

    def _invalidate_cache(self) -> None:
        path = self._cache_path()
        with contextlib.suppress(OSError):  # pragma: no cover
            path.unlink(missing_ok=True)

    def _resolve(self, element_id: int) -> Element:
        cached = self._read_cache()
        if cached is None:
            raise ElementNotFoundError(
                "no cached analyze result", hint="Run `aua analyze` first to assign element ids."
            )
        el = cached.element_by_id(element_id)
        if el is None:
            valid = ", ".join(str(e.id) for e in cached.elements[:20]) or "(none)"
            raise ElementNotFoundError(
                f"element id {element_id} is not in the last analyze (valid: {valid})",
                hint="Re-run `aua analyze`; ids change when the screen changes.",
            )
        return el
