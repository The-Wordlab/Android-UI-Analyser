"""The interface-agnostic perception + action engine (PRD §6, §6a).

The engine orchestrates the analyze pipeline and the cost-aware escalation ladder. It
depends only on: the schema, the config, the device ABC, the provider *factory* +
interfaces, and the routing helpers. It NEVER imports a concrete provider, and the
hierarchy/gate/merge/annotate modules are imported lazily so a fresh checkout imports
cleanly. The CLI, MCP server, and daemon are all thin adapters over this class.
"""

from __future__ import annotations

import atexit
import contextlib
import re
import sys
import threading
import time
import weakref
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from . import (
    engine_actions,
    engine_analyze,
    engine_apps,
    engine_capture,
    engine_environment,
    engine_flows,
    engine_memory,
    engine_navigation,
    engine_observation,
    engine_policy,
    engine_sessions,
    engine_waits,
)
from .config import Config
from .device import Device, connect, list_devices
from .engine_apps import (
    _install_versions_differ,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
)
from .engine_capture import (
    DeviceStoodDownError,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
    _region_from_point,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
)
from .engine_flows import (
    _parse_point,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
)
from .engine_navigation import (
    _HANDOVER_HINTS,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
)
from .engine_support import (
    _ActionSite,
    _AwaitTerm,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
    _HandoverRefused,
    _label,
    _parse_await_terms,
    logger,
)
from .engine_waits import (
    _AWAIT_PREDICATE_HELD,  # noqa: F401  (re-exported: imported from here by tests or sibling modules)
)
from .errors import (
    AuaError,
    DeviceError,
    DeviceLeasedError,
    UsageError,
)
from .memory import (
    AppMemoryStore,
    AppStrings,
    _id_tail,
    matches_any,
)
from .platforms import PlatformAdapter, PlatformFactory
from .providers.base import (
    ScreenImage,
)
from .providers.registry import ProviderFactory
from .schema import (
    ActionResult,
    AnalyzeResult,
    DeviceInfo,
    Element,
    ElementId,
    ShellResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .policy import PolicyMode

# Keep the historical module-level monkeypatch seams working for downstream tests and
# integrations while production construction moves behind PlatformAdapter.
_DEFAULT_ANDROID_CONNECT = connect
_DEFAULT_ANDROID_LIST_DEVICES = list_devices

# A run of text one line tall measures roughly two to three average character widths
# in height; a wrapped paragraph measures many. Used to decide whether aiming at a
# phrase inside an element can only move horizontally.
# A bottom system bar starts within this fraction of the screen height. Wide enough for a
# tall three-button bar, narrow enough that a systemui sheet or the expanded notification
# shade can never be mistaken for it (see Engine._system_bar_top).
# A hierarchy dump quicker than this can outrun the screen it is reading, so a post-action
# sample may catch a half-attached tree; a slower one cannot (the render has finished by the
# time it returns). Measured ~150ms headless vs 600-1200ms windowed on the same emulator.
# Package-name fragments that mark a surface AUA is never *testing* — the system launcher and
# the system UI. Scoping an action's log window to one of these is not a smaller answer, it is a
# wrong one: measured on a real device, a Back to home attached 20 lines of `LauncherStateManager`
# animation state under a field that claims to be the app's own output.
# Foreground ownership can lead accessibility-window attachment briefly on a cold launch. Retry
# only while the requested package demonstrably remains foreground, and never beyond this budget.
# Terms that describe the surrounding UI rather than a user's intended control. A single match
# on one of these is not enough to turn a visible multi-word control into an execution proposal.


#: ``await_outcome`` values that mean the predicate held. Two names rather than one because
#: ``absence-satisfied`` holds on weaker evidence — every term was negated, so what the caller
#: left is gone and nothing here evidences what arrived. Anything that treats arrival as
#: *learnable* keeps comparing against ``satisfied`` alone: an absence-only predicate is not a
#: route's arrival proof and must never be recorded as one.


#: Leading flow steps that leave the flow's own app dead and the device on the launcher. A flow
#: opening with these cannot be expected to find that app already in the foreground — it is about
#: to put it there itself. See :meth:`Engine._flow_leading_launch_establishes_origin`.


# `wait --for` restricted to fields `find_text`/`wait_for` can actually search — `net:`/`log:`
# are off-screen evidence with no `by=` equivalent on this path, so they are deliberately not
# offered here even though `_AWAIT_PREFIXES` knows them.


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


def _package_from_xml(xml: str, ignore: Sequence[str] = ("com.android.systemui",)) -> str | None:
    """Cheap foreground-package guess from a hierarchy dump (avoids an app_current RPC).

    Picks the most common ``package=`` among nodes, excluding *ignore* globs — system
    chrome and IMEs overlay every app, so an open keyboard must never win the vote.
    Falls back to the overall majority when every node is ignorable.
    """
    packages = re.findall(r'package="([^"]+)"', xml)
    if not packages:
        return None
    counts = Counter(
        package for package in packages if package and not matches_any(package, ignore)
    )
    if not counts:
        counts = Counter(packages)
    return counts.most_common(1)[0][0]


# u2 accepts these names (plus KEYCODE_* / a numeric keycode); anything else reaches the
# device as a no-op-or-crash, so it is rejected up front rather than looking like it worked.


# Engines with possibly-unflushed async map writes. Weak so holding one here never keeps an
# Engine (and its device connection) alive; see :func:`_flush_memory_writers_at_exit`.
_LIVE_ENGINES: weakref.WeakSet[Engine] = weakref.WeakSet()


def _flush_memory_writers_at_exit() -> None:
    """Land queued map writes before the interpreter tears the writer threads down.

    Screen writes run on a background thread, and that thread is a daemon: at interpreter
    exit it is killed wherever it happens to be. Almost every AUA invocation is a process
    that starts a writer and exits milliseconds later, so on the daemon-less path a queued map
    update can be lost even while every call reports its screen correctly. The
    warm daemon hid it, because that process lives long enough for the write to land.

    ``atexit`` handlers run before daemon threads are killed, which is exactly the window
    this needs. The wait is bounded and failure is silent by design: a map update is worth a
    moment at shutdown, never a hang and never an error on the way out.
    """

    for engine in list(_LIVE_ENGINES):
        with contextlib.suppress(Exception):
            engine._join_memory_writers(timeout_s=2.0)


atexit.register(_flush_memory_writers_at_exit)


class _DeviceLoan(NamedTuple):
    """An open helper channel plus what the caller needs to report about the handover."""

    channel: Any
    serial: str
    purpose: str
    u2_was_connected: bool


#: What to do about each way the handover can be refused. Keyed by the reason
#: :meth:`Engine._device_agent_borrowed` journalled, so the advice and the diagnostic agree.


def _actionable_keys(elements: Sequence[Element]) -> frozenset[str]:
    """The app's own controls, named the way an observation publishes them.

    Deliberately narrow — the interaction flags and nothing else, matching
    ``projection._is_actionable``. This is the set a caller's next command can name, which is
    what makes it the right basis for "did the screen you are holding move": a label whose text
    ticked over has not taken anything away from you, and an arriving dialog has.
    """
    from .identity import stable_key as _sk
    from .selectors import app_elements

    return frozenset(
        key
        for el in app_elements(elements)
        if (el.clickable or el.checkable or el.long_clickable or el.scrollable)
        and (key := el.stable_key or _sk(el))
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
        if hasattr(self.factory, "model_control"):
            self.model_control = self.factory.model_control
        else:
            # Small injected test/plugin factories predate the dashboard control plane. Keep
            # their policy behavior unchanged while still giving Engine one shared switch API.
            from .model_control import ModelControlStore

            self.model_control = ModelControlStore(config)
        self._mem: AppMemoryStore | None = None
        self._version_cache: dict[str, str | None] = {}
        self._strings_cache: dict[str, AppStrings | None] = {}
        self._flag_context_checked_at: dict[str, float] = {}
        # Session default for --with-image (CLI global / MCP configure); per-call wins.
        self._default_with_image: bool | str | None = (
            config.output.with_image if config.output.with_image else None
        )
        self._capture: Any = None  # CaptureBuffer | None — set by capture_start
        # Capture auto-start happens on the daemon's background initializer. Lifecycle
        # commands can arrive on the socket while that initializer is still creating its
        # directories, so start/stop/on/off must be one atomic state transition. This lock is
        # deliberately capture-only: the background initializer is forbidden from connecting
        # to the device and therefore must not serialize the daemon's first real request.
        self._capture_lock = threading.RLock()
        # True only while the UiAutomation slot is on loan to the on-device helper.
        # Read by :meth:`_capture_screenshot`, which is the one device call that can
        # arrive from another thread during a handover.
        self._stood_down = False
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
        # ``(fingerprint the caller was holding, why it is gone)``, decided by the pre-action
        # resolution read and consumed by the response that read produced. Paired with the
        # fingerprint on purpose: an action that resolves nothing against the live screen
        # (`key`, `swipe`, `tap-point`) must not inherit a verdict about a screen that is no
        # longer the one this caller holds. See :meth:`_screen_moved_verdict`.
        self._screen_moved: tuple[str, str] | None = None
        # Which app this run is driving, and the log window already reported for it. The second
        # half stops a wait — which does not stamp a new `last-action` — from re-reporting the
        # previous action's lines as though the app had just said them again.
        self._app_under_test: str | None = None
        self._app_logs_reported_ms: int | None = None
        self._log_prefs_store: AppMemoryStore | None = None
        # `config.logs` fields the CALLER set on purpose this session (MCP `configure`, or a typed
        # `--app-logs` flag). Those beat a stored per-app preference: an agent that just asked for
        # 60 lines while chasing a library must get 60, or the per-turn control silently does
        # nothing for exactly the apps that have a preference.
        self._session_log_fields: set[str] = set()
        self._last_activity: str | None = None
        # Lease context: which agent this engine speaks for, what it needs, and what it got.
        # Set by the CLI/MCP layer before the device is first touched.
        self._flows_cache: dict[str, list[str]] = {}
        self._lease_owner: str | None = None
        self._lease_needs: list[str] | None = None
        self._lease_serial: str | None = None
        self._lease_owner_resolved: str | None = None
        self._lease_selection_reason: str | None = None
        self._lease_was_preexisting = False
        # Generation fences cached observations even when a lease leaves and later returns to
        # the same long-lived process identity (so owner equality alone cannot detect the gap).
        self._lease_generation_resolved: str | None = None
        # True only inside the explicit `lease acquire --replace` reservation. Normal device
        # commands may never leave one owner holding more than one sticky target.
        self._lease_allow_replacement = False
        # A warm Engine is shared across many daemon/MCP calls, and background jobs use another
        # thread. Each executing thread therefore owns its own command-lifetime device fence.
        self._device_use_context = threading.local()
        self._lease_wait_s: float = 0.0
        self._lease_waited_ms: int = 0
        # Serials whose helper setup has already been tried and refused. Without this a
        # target that can never run the helper would re-probe on every single run.
        self._helper_unavailable: set[str] = set()
        # (resolved?, serial) — a plain None serial is a legitimate answer, so the flag
        # is what distinguishes "not asked yet" from "asked, and there is no pin".
        self._leased_serial_resolved: tuple[bool, str | None] | None = None
        # Once per engine: a warm daemon must not re-glob the ledger on every request.
        self._swept_abandoned = False
        from .perf import GateCache, HierarchyPrefetch, SettleProfiles

        self._prefetch = HierarchyPrefetch()
        self._settle_profiles = SettleProfiles()
        self._gate_cache = GateCache()
        # What this caller costs to think, and which screen it was last handed. Both are
        # cross-process state (every CLI call is a new interpreter), so the engine only ever
        # holds the copy read at the start of this turn; see :meth:`open_caller_turn`.
        self._caller_turn: Any = None
        self._caller_latency_key: str | None = None
        # Memoised fallback read for the daemon path, where the turn was opened in another
        # process. `False` means "not looked up yet" — None is a real answer here.
        self._caller_profile_cache: Any = False
        # (clamped_from, ceiling) for the wait currently in flight; consumed by `_await_result`.
        self._pending_wait_clamp: tuple[int | None, int] | None = None
        self._last_mem_fp: str | None = None
        self._last_known_screen: str | None = None
        self._last_action_kind: str | None = None
        # Monotonic stamp for the wall clock of the current command, consumed by `_wall_ms`.
        # Declared here rather than inferred, so mypy sees the float it actually holds.
        self._call_started_at: float | None = None
        # The same instant on the shared clock, for the access log. A monotonic reading is
        # only comparable inside this process; a journal line has to line up with another
        # process's journal and with a logcat dump, so the instant is kept both ways.
        self._call_started_epoch_ms: int | None = None
        self._last_action_site: _ActionSite | None = None
        self._last_analyze_elements: list[Element] | None = None
        self._last_hierarchy_hash: str | None = None
        self._last_analyze_result: AnalyzeResult | None = None
        # Screenshot whose pixels produced the current analyze's px: element identities.
        # The outer --with-image wrapper saves this exact frame instead of taking a second,
        # potentially different screenshot after identity assignment.
        self._last_analyze_image: ScreenImage | None = None
        # Set only for the duration of an explicit `session autopilot` run; see
        # `_session_policy_mode`.
        self._policy_mode_override: PolicyMode | None = None
        self._mem_lock = threading.Lock()
        self._mem_threads_lock = threading.Lock()
        self._mem_thread: threading.Thread | None = None
        self._mem_threads: list[threading.Thread] = []
        # Reachable from the interpreter-exit flush, so a short-lived call does not drop the
        # map update it just queued.
        _LIVE_ENGINES.add(self)
        self._claimed_instance_token: str | None = None
        self._action_recording_suppression = 0
        # Set only by the warm daemon/MCP job manager. Supported wait loops consult the event
        # between device reads; the manager object is transport state, intentionally typed Any
        # here to avoid making the interface-agnostic Engine import its adapter.
        self._job_cancel_event: threading.Event | None = None
        # The daemon serves foreground calls while a job waits in another thread. Job identity
        # therefore belongs to the executing thread; a shared flag made every concurrent
        # foreground wait inherit the background job's unlimited budget.
        self._job_context = threading.local()
        self._aua_job_manager: Any = None

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

    def _lease_target_id(self, serial: str | None) -> str | None:
        """Map a connected runtime's canonical id back to the target AUA leased.

        Adapters may accept an alias and return a runtime whose canonical id differs. The lease
        remains keyed by the selected target, regardless of platform. A genuinely different id
        still fails the one-command/one-device check below.
        """

        if (
            serial is not None
            and self._device is not None
            and serial == self._device.serial
            and self._lease_serial is not None
        ):
            return self._lease_serial
        return serial

    @property
    def _lease_registry_dir(self) -> str:
        """Host-wide lease authority, independent of this run's artifact/cache directory."""

        return str(self.config.lease.registry_dir)

    def begin_device_use(self, serial: str | None = None) -> None:
        """Fence this thread's complete device command against release or lease transfer."""

        if not getattr(self.config.lease, "enabled", True):
            return
        serial = self._lease_target_id(serial)
        active = getattr(self._device_use_context, "guard", None)
        if active is not None:
            active_serial = getattr(self._device_use_context, "serial", None)
            if serial is not None and active_serial != serial:
                raise UsageError(
                    f"one command cannot drive both {active_serial} and {serial}",
                    code="device_transaction_mismatch",
                )
            return
        target = serial
        if target is None and self._device is not None:
            target = self._device.serial
        if target is None:
            target = self._leased_serial()
        owner = self._lease_owner_resolved
        # Host-only calls and injected test Devices deliberately have no registry entry.
        if not target or not owner:
            return

        from . import leases

        guard = leases.device_command(self._lease_registry_dir, target)
        guard.__enter__()
        try:
            generation = leases.validate_use(
                self._lease_registry_dir,
                target,
                owner=owner,
            )
            previous_generation = self._lease_generation_resolved
            if previous_generation is not None and generation != previous_generation:
                self._reset_owner_transient_state()
            self._lease_generation_resolved = generation
        except BaseException:
            guard.__exit__(*sys.exc_info())
            raise
        self._device_use_context.guard = guard
        self._device_use_context.serial = target
        self._device_use_context.generation = generation

    @contextlib.contextmanager
    def device_use_context(
        self,
        serial: str | None = None,
        *,
        owner: str | None = None,
        generation: str | None = None,
    ) -> Iterator[None]:
        """Fence one short background read without claiming the foreground command mutex."""

        if not getattr(self.config.lease, "enabled", True):
            yield
            return
        target = self._lease_target_id(
            serial or (self._device.serial if self._device is not None else None)
        )
        resolved_owner = owner or self._lease_owner_resolved
        if not target or not resolved_owner:
            yield
            return
        from . import leases

        with leases.device_use(self._lease_registry_dir, target):
            leases.validate_use(
                self._lease_registry_dir,
                target,
                owner=resolved_owner,
                expected_generation=generation,
                renew=False,
            )
            yield

    def release_device_use(self) -> None:
        """Release this thread's command-lifetime device fence, if one was acquired."""

        guard = getattr(self._device_use_context, "guard", None)
        if guard is None:
            return
        for field in ("guard", "serial", "generation"):
            with contextlib.suppress(AttributeError):
                delattr(self._device_use_context, field)
        guard.__exit__(None, None, None)

    @property
    def device(self) -> Device:
        """Lazily connect; doctor/devices/config work without ever touching this."""
        target = self._device.serial if self._device is not None else self._leased_serial()
        self.begin_device_use(target)
        if self._device is None:
            self._device = self._connect_target(target)
            self._claim_memory_session()
            # First connect is the one moment we know which device is ours and that an adapter
            # exists: the cheapest place to hand back every *other* device a dead agent left
            # proxied, offline or time-travelled.
            self._sweep_abandoned_devices(skip=self._device.serial)
        return self._device

    def _leased_serial(self) -> str | None:
        """Which target this engine may drive, resolved without connecting to it.

        Split out from :attr:`device` so a caller can learn *which* device it has before
        deciding whether to connect. That distinction is worth real time: handing a run to
        the on-device helper costs 2839ms when uiautomator2 is already attached (it has to
        let the slot go and wait for the helper to bind) and 682ms when it never attached at
        all — 2155ms of which is purely the release. Knowing the serial early is what makes
        the second path reachable.

        Cached so the lease is resolved once per engine, whoever asks first.

        A device this engine was *given* is the answer: leasing exists to choose a target when
        nobody has chosen one, so asking it here answered whichever emulator happened to be
        attached to the host instead. On a machine with three running that meant a serial
        belonging to a different device than the one being driven — and the caller that needs
        this cheap path is the helper offload, which would then hand a whole flow to the wrong
        target. It also made the offload tests pass only where a device was plugged in: with
        none attached, leasing answers None and the offload silently declined.
        """

        if self._leased_serial_resolved is None:
            given = self._device.serial if self._device is not None else None
            self._leased_serial_resolved = (True, given or self._lease_device())
        return self._leased_serial_resolved[1]

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

    def _lease_device(self, *, excluded: set[str] | None = None) -> str | None:
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

        excluded = excluded or set()

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
                if info.serial and info.serial not in excluded
            ]

        # Discovery failure is not an empty pool. Propagate the platform's typed transport error
        # so session bootstrap never provisions or switches devices because ADB disappeared.
        initial = candidates()
        owner = leases.resolve_owner(self._lease_owner)
        held_before = set(leases.held_by(self._lease_registry_dir, owner))
        if not initial and not self._lease_wait_s:
            if explicit is not None:
                # Explicit/injected targets retain the historical lazy-connect seam. The
                # platform runtime will produce the authoritative reachability error without
                # making a different target eligible.
                return explicit
            if not held_before:
                return None
        if self._lease_wait_s:
            serial, why, waited_ms = leases.wait_for_device(
                self._lease_registry_dir,
                owner=owner,
                explicit=explicit,
                candidates=candidates,
                needs=needs,
                ttl_s=int(getattr(cfg.lease, "ttl_s", leases.DEFAULT_TTL_S)),
                allow_replacement=self._lease_allow_replacement,
                wait_s=self._lease_wait_s,
            )
            self._lease_waited_ms = waited_ms
        else:
            serial, why = leases.choose_device(
                self._lease_registry_dir,
                owner=owner,
                explicit=explicit,
                candidates=initial,
                needs=needs,
                ttl_s=int(getattr(cfg.lease, "ttl_s", leases.DEFAULT_TTL_S)),
                allow_replacement=self._lease_allow_replacement,
            )
            self._lease_waited_ms = 0
        self._lease_serial = serial
        self._lease_owner_resolved = owner
        self._lease_selection_reason = why
        self._lease_was_preexisting = serial in held_before
        return serial

    def _selected_target_has_app(self, serial: str, package: str) -> bool:
        """Check app presence only after this caller has safely leased *serial*."""

        if not self.platform.supports("app.status"):
            raise DeviceError(
                f"platform '{self.platform.name}' cannot select a target by installed app",
                code="unsupported_capability",
            )
        runtime: Device | None = None
        with self.device_use_context(serial):
            try:
                runtime = self._connect_target(serial)
                return bool(self.platform.installed_app(runtime, package).installed)
            finally:
                if runtime is not None:
                    with contextlib.suppress(Exception):
                        runtime.close()

    def _release_failed_bootstrap_target(self, serial: str) -> None:
        """Release only a target this session bootstrap acquired and never acted on."""

        from . import leases

        owner = self._lease_owner_resolved
        if not owner or not leases.release(self._lease_registry_dir, serial, owner=owner):
            raise DeviceError(
                f"could not release unused bootstrap target {serial}",
                hint="Run `aua lease list`; no app action was attempted on the target.",
            )
        self._lease_serial = None
        self._lease_owner_resolved = None
        self._lease_selection_reason = None
        self._lease_was_preexisting = False
        self._lease_generation_resolved = None
        self._leased_serial_resolved = None

    def _prepare_session_target(
        self,
        *,
        wait_for_lease_s: float,
        start_emulator: bool,
        headed: bool,
        audio: bool,
        avd: str | None,
        animations: bool = False,
        package: str | None = None,
        app_will_be_installed: bool = False,
    ) -> dict[str, Any]:
        """Select/claim a compatible target, provisioning one only when the pool has none."""

        if self._device is not None:
            return {
                "serial": self._device.serial,
                "emulator_started": False,
                "lease_waited_ms": 0,
            }

        self._lease_wait_s = float(wait_for_lease_s)
        self._lease_waited_ms = 0
        # Window/audio requests are target requirements too. Probe every online candidate and use
        # one only when its actual emulator process satisfies them; otherwise provision a known
        # matching instance below.
        requested_needs = list(self._lease_needs or [])
        if headed and "headed" not in requested_needs:
            requested_needs.append("headed")
        if audio and "audio" not in requested_needs:
            requested_needs.append("audio")
        self._lease_needs = requested_needs
        required_app = package if package and not app_will_be_installed else None
        excluded_for_missing_app: set[str] = set()
        selection_error: DeviceLeasedError | None = None
        try:
            while True:
                try:
                    selected = self._lease_device(excluded=excluded_for_missing_app)
                except DeviceLeasedError as exc:
                    selection_error = exc
                    selected = None
                if not selected:
                    break
                if required_app:
                    try:
                        app_installed = self._selected_target_has_app(selected, required_app)
                    except Exception:
                        if not self._lease_was_preexisting:
                            self._release_failed_bootstrap_target(selected)
                        raise
                    if not app_installed:
                        if self._lease_was_preexisting:
                            raise DeviceError(
                                f"{required_app} is not installed on leased target {selected}",
                                code="required_app_not_installed_on_leased_target",
                                hint=(
                                    "The lease was retained. Supply `--apk <bundle>` to install "
                                    "the app there, or explicitly replace the target after cleanup."
                                ),
                            )
                        self._release_failed_bootstrap_target(selected)
                        excluded_for_missing_app.add(selected)
                        if self.config.device.serial:
                            raise DeviceError(
                                f"{required_app} is not installed on requested target {selected}",
                                code="required_app_not_installed",
                                hint="Supply `--apk <bundle>` or choose a target that has the app.",
                            )
                        continue
                self.config.device.serial = selected
                return {
                    "serial": selected,
                    "emulator_started": False,
                    "lease_waited_ms": self._lease_waited_ms,
                }
            if required_app and selection_error is None:
                detail = (
                    f"; checked without finding it on {', '.join(sorted(excluded_for_missing_app))}"
                    if excluded_for_missing_app
                    else ""
                )
                raise DeviceError(
                    f"no available target has required app {required_app}{detail}",
                    code="required_app_not_installed",
                    hint=(
                        "Supply `--apk <bundle>` so AUA can provision and install it, or attach "
                        "an unleased target where it is already installed."
                    ),
                )
            if self.config.device.serial:
                if selection_error is not None:
                    raise selection_error
                raise DeviceError(f"requested device {self.config.device.serial} is not online")
            if not start_emulator:
                if selection_error is not None:
                    raise selection_error
                raise DeviceError(
                    "no compatible unleased device is online",
                    hint="Allow automatic provisioning or attach a compatible Android target.",
                )

            from . import leases

            emulator_mod = self.platform.capability("virtual_devices")
            selected_avd = emulator_mod.select_avd_for_session(
                avd,
                needs=[
                    need for need in (self._lease_needs or []) if need in {"root", "play", "proxy"}
                ],
            )
            boot_owner = leases.resolve_owner(getattr(self, "_lease_owner", None))
            boot = emulator_mod.start(
                selected_avd,
                headless=not headed,
                animations=animations,
                audio=audio,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self._lease_registry_dir,
                owner=boot_owner,
                parallel=True,
            )
            serial = str(boot["serial"])
            self.config.device.serial = serial
            self._lease_serial = None
            self._leased_serial_resolved = None
            self._lease_owner_resolved = None
            claimed: str | None = None
            try:
                claimed = self._lease_device()
                if claimed != serial:
                    raise DeviceError(
                        f"automatic session provisioning started {serial} but leased {claimed}"
                    )
                if required_app and not self._selected_target_has_app(serial, required_app):
                    raise DeviceError(
                        f"{required_app} is not installed on provisioned target {serial}",
                        code="required_app_not_installed",
                        hint=(
                            "Supply `--apk <bundle>` so AUA can install the app while "
                            "provisioning. The unusable emulator was stopped."
                        ),
                    )
            except Exception:
                # Roll back only what this boot demonstrably created — its own spawned
                # process and instance record. A serial-scoped stop here once killed a
                # foreign worker's emulator: the claim had failed precisely because that
                # serial was somebody else's leased device.
                if claimed == serial:
                    self._release_failed_bootstrap_target(serial)
                with contextlib.suppress(Exception):
                    emulator_mod.stop_spawned_instance(
                        instance=str(boot.get("instance") or ""),
                        pid=boot.get("pid"),
                        cache_dir=self.config.cache.dir,
                        lease_registry_dir=self._lease_registry_dir,
                        owner=boot_owner,
                        requested_by="session-start-claim-rollback",
                    )
                raise
            return {
                **boot,
                "serial": serial,
                "emulator_started": True,
                "lease_waited_ms": self._lease_waited_ms,
            }
        finally:
            self._lease_wait_s = 0.0

    def _reset_owner_transient_state(self) -> None:
        """Drop observations and transport state when a warm daemon changes caller owner.

        Device caches are valid only for the owner that produced them.  Keeping them across a
        daemon hand-off can make a fresh numeric id, session id, or prefetched hierarchy from
        the previous owner look current to the next one even when both use the same emulator.
        Durable map knowledge stays shared; only invocation/session-local state is cleared.
        """
        if not self._join_memory_writers(timeout_s=5.0):
            raise UsageError(
                "the previous owner's screen-map write is still running",
                hint="Retry this call after the current device operation settles.",
                code="owner_handoff_busy",
            )
        self._last_activity = None
        self._pre_action_sig = None
        self._pre_action_tree_fp = None
        self._pre_action_state = None
        self._action_observation_baseline = None
        self._screen_moved = None
        self._last_mem_fp = None
        self._last_known_screen = None
        self._last_action_kind = None
        self._last_action_site = None
        self._last_analyze_elements = None
        self._last_hierarchy_hash = None
        self._last_analyze_result = None
        self._session_id: str | None = None
        # A latency estimate belongs to one caller. The warm daemon's Engine outlives the client
        # that built it, so a cached key here would price the next agent's waits from the
        # previous one's thinking speed.
        self._caller_turn = None
        self._caller_latency_key = None
        self._caller_profile_cache = False
        # Belt and braces: `await_predicate` always overwrites this before `_await_result` can
        # read it, so a leak is not reachable today — but it is per-command state on an object
        # that outlives the command, which is exactly the shape of bug this reset exists for.
        self._pending_wait_clamp = None
        self._prefetch.invalidate()
        self._gate_cache = type(self._gate_cache)()

    # ------------------------------------------------------- device change ledger

    def _ledger_identity(self) -> dict[str, Any]:
        """Who is making a change, so a stranger can tell later whether they are still alive."""
        from . import leases

        owner = getattr(self, "_lease_owner_resolved", None) or leases.resolve_owner(
            getattr(self, "_lease_owner", None)
        )
        process = leases.owner_caller(owner) or {}
        return {
            "owner": str(owner),
            "owner_pid": process.get("pid"),
            "owner_started": process.get("started"),
            "cache_dir": str(self.config.cache.dir),
            # With leasing on, a vanished lease is the signal that this agent is done with the
            # device. With it off there is no such signal and only the process can speak.
            "leased": bool(getattr(self.config.lease, "enabled", True)),
        }

    def record_device_change(
        self,
        *,
        key: str,
        kind: str,
        op: str,
        args: dict[str, Any] | None = None,
        detail: str = "",
        serial: str | None = None,
    ) -> None:
        """Journal how to undo a persistent device change — **before** making it.

        Every device mutation must come through here. The record is what lets another process
        clean up after this one is SIGKILLed, and writing it after the mutation would leave
        exactly the gap that makes a dirty device unrecoverable. See ``device_ledger``.
        """
        if not getattr(self.config.teardown, "enabled", True):
            return
        from . import device_ledger

        target = serial or (self._device.serial if self._device else self.config.device.serial)
        if not target:
            return
        token: str | None = None
        if self._device is not None:
            with contextlib.suppress(Exception):
                token = self._device.instance_token()
        identity = self._ledger_identity()
        device_ledger.record(
            target,
            key=key,
            kind=kind,
            op=op,
            args=args or {},
            detail=detail,
            instance_token=token,
            **identity,
        )
        self._ensure_teardown_watchdog(target)

    def _record_device_agent_change(self, serial: str) -> None:
        """The helper's accessibility service stays in the secure services list after we exit.

        Not inert: Android suppresses accessibility services only while uiautomator2 holds
        UiAutomation, so a left-enabled helper keeps binding on a device somebody else inherits.
        """
        with contextlib.suppress(Exception):
            self.record_device_change(
                key="device_agent_service",
                kind="device_agent_service",
                op="disable_device_agent",
                detail="on-device helper accessibility service enabled",
                serial=serial,
            )

    def forget_device_change(self, *keys: str, serial: str | None = None) -> None:
        """Drop records for changes this process has just undone itself."""
        from . import device_ledger

        target = serial or (self._device.serial if self._device else self.config.device.serial)
        if target:
            with contextlib.suppress(Exception):
                device_ledger.forget(target, *keys)

    def _ensure_teardown_watchdog(self, serial: str) -> None:
        if not getattr(self.config.teardown, "watchdog", True):
            return
        from . import teardown

        with contextlib.suppress(Exception):
            teardown.ensure_watchdog(
                serial,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self._lease_registry_dir,
                platform_name=self.platform.name,
                grace_s=float(self.config.teardown.grace_s),
                poll_s=float(self.config.teardown.watchdog_poll_s),
            )

    def _sweep_abandoned_devices(self, *, skip: str | None) -> None:
        """Undo other devices' orphaned changes — the cheap net, run once per Engine.

        A directory glob when nothing is pending, which is the normal case. Deliberately never
        raises: a stuck cleanup on some other emulator must not fail the command the caller
        actually asked for.
        """
        cfg = self.config.teardown
        if not (getattr(cfg, "enabled", True) and getattr(cfg, "sweep_on_command", True)):
            return
        if self._swept_abandoned:
            return
        self._swept_abandoned = True
        self._adopt_orphan_emulators()
        from . import device_ledger, teardown

        try:
            if not device_ledger.pending_serials():
                return
            reports = teardown.sweep(
                platform=self.platform,
                cache_dir=self.config.cache.dir,
                lease_registry_dir=self._lease_registry_dir,
                grace_s=float(cfg.grace_s),
                skip=skip,
            )
        except Exception as exc:
            logger.debug("teardown sweep skipped: %s", exc)
            return
        for report in reports:
            logger.warning(
                "reset abandoned changes on %s (%s): %s",
                report.get("serial"),
                report.get("reason"),
                ", ".join(f"{d['kind']}" for d in report.get("undone", [])) or "none",
            )

    def _adopt_orphan_emulators(self) -> None:
        """Re-arm the idle watchdog on aua-started emulators that lost theirs.

        The watchdog is a process spawned once at boot, and nothing re-spawns it: a host reboot,
        a stray ``pkill``, or a crash leaves that emulator immortal, because the only thing that
        would ever have stopped it is gone. Observed on a dev host — an instance recorded with
        ``idle_timeout_s: 900`` and ``watchdog_pid: None``.

        Never raises, and never touches an emulator AUA did not start.
        """
        cfg = self.config.teardown
        if not getattr(cfg, "enabled", True):
            return
        timeout = float(getattr(cfg, "emulator_idle_stop_s", 0.0))
        if timeout <= 0:
            return
        try:
            virtual = self.platform.capability("virtual_devices")
        except Exception:
            return  # platform cannot boot targets, so it cannot have orphaned any
        try:
            adopted = virtual.adopt_idle_watchdogs(
                cache_dir=self.config.cache.dir,
                idle_timeout_s=timeout,
                lease_registry_dir=self._lease_registry_dir,
            )
        except Exception as exc:
            logger.debug("emulator watchdog adoption skipped: %s", exc)
            return
        for item in adopted:
            logger.warning(
                "%s (%s) was running with no idle watchdog — re-armed at %.0fs",
                item.get("serial"),
                item.get("instance"),
                float(item.get("idle_timeout_s") or 0),
            )

    def teardown_status(self) -> dict[str, Any]:
        """What device changes are still pending an undo, and whether they can be run now."""
        from . import device_ledger

        pending = device_ledger.status(
            cache_dir=self.config.cache.dir,
            lease_registry_dir=self._lease_registry_dir,
            grace_s=float(self.config.teardown.grace_s),
        )
        return {
            "ok": True,
            "action": "teardown-status",
            "devices": pending,
            "detail": (
                "no device has pending changes"
                if not pending
                else f"{len(pending)} device(s) carry changes AUA can undo"
            ),
        }

    def teardown_run(
        self, *, serial: str | None = None, force: bool = False, dry_run: bool = False
    ) -> dict[str, Any]:
        """Undo pending changes now — for one serial, or every device with no live holder."""
        from . import device_ledger, teardown

        grace = float(self.config.teardown.grace_s)
        if serial:
            reports = [
                teardown.reap(
                    serial,
                    platform=self.platform,
                    cache_dir=self.config.cache.dir,
                    lease_registry_dir=self._lease_registry_dir,
                    grace_s=grace,
                    force=force,
                    dry_run=dry_run,
                )
            ]
        else:
            reports = [
                teardown.reap(
                    target,
                    platform=self.platform,
                    cache_dir=self.config.cache.dir,
                    lease_registry_dir=self._lease_registry_dir,
                    grace_s=grace,
                    force=force,
                    dry_run=dry_run,
                )
                for target in device_ledger.pending_serials()
            ]
        undone = sum(len(r.get("undone") or ()) for r in reports)
        failed = sum(len(r.get("failed") or ()) for r in reports)
        return {
            "ok": failed == 0,
            "action": "teardown-run",
            "dry_run": dry_run,
            "reports": reports,
            "detail": f"{undone} change(s) undone, {failed} failed",
        }

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
            leases.renew(self._lease_registry_dir, serial, owner=owner)

    def list_devices(self) -> list[DeviceInfo]:
        return self._list_targets()

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

    # ----------------------------------------------------------------- step executor

    # Step kinds the on-device helper can perform itself. Everything else — proxy, network
    # shaping, feature flags, launching apps, recursion into saved routes — is a host
    # operation, so the run stops there and the host takes over from that index.
    _DEVICE_STEP_KINDS = frozenset(
        {
            "tap",
            "long-press",
            "input",
            "clear",
            "key",
            "wait-for",
            "assert-visible",
            "assert-not-visible",
            # Gestures. These matter out of proportion to their number: the offload only ever
            # takes a *leading* prefix, and real flows scroll early and often, so without them
            # the handover stopped at the first swipe and almost never earned its cost.
            #
            # `hide-keyboard` is deliberately absent. The host dismisses the IME with
            # KEYCODE_ESCAPE precisely because Back finishes the Activity when no keyboard is
            # up, and accessibility cannot send a raw keycode. The device can only press Back
            # after confirming an input-method window exists — and uiautomator2 installs a
            # headless AdbKeyboard as the default IME, which exposes no such window. The step
            # would therefore report success having done nothing, on every AUA-driven device.
            "swipe",
            "scroll",
            "scroll-to",
            "tap-point",
            "paste",
            "wait-stable",
        }
    )
    # Only the global keys exist as accessibility actions; an arbitrary keycode needs input
    # injection, which the helper deliberately does not do.
    _DEVICE_KEY_ARGS = frozenset({"back", "home", "recents", "recent"})

    # ``wait-for`` and the asserts name their target with a predicate in ``arg`` (plus ``by``),
    # not with the element selectors the acting kinds use. Treating them the same way silently
    # disqualified every run containing one, because ``arg``-only steps looked selector-less.
    _DEVICE_PREDICATE_KINDS = frozenset({"wait-for", "assert-visible", "assert-not-visible"})
    _DEVICE_DIRECTIONS = frozenset({"up", "down", "left", "right"})
    # Matchers the helper implements; anything else (regex, a custom matcher) stays on the host.
    #
    # This must be a subset of ``Uiautomator2Device._BY_FIELDS``, and it had drifted in both
    # directions. ``content_desc`` and ``resource_id`` were listed here but are not spellings
    # the host knows at all — ``_fields_for`` refuses an unknown token rather than degrading
    # to a text search — so a step could run on the device and then be a hard usage error the
    # moment the host re-ran it. And ``id`` was *missing*, which is the spelling the flow
    # parser actually emits for a resource-id predicate, so every one of them in every saved
    # flow was silently disqualified and sent the rest of the run back to the host.
    _DEVICE_BY_FIELDS = frozenset({"text", "desc", "rid", "id"})

    # What the host itself waits for each checking step, so the device can be told rather
    # than guessing. Divergence here is not symmetric: a device check that waits *longer*
    # than the host can pass an assertion the host would have failed, and a device pass is
    # final. Kept as ``s.timeout_ms or <default>`` because that is exactly how the host's own
    # branches spell it, including that an authored 0 means "the default" for a wait.
    _HOST_STEP_TIMEOUT_MS = {
        "wait-for": 10000,
        "assert-visible": 0,
        "assert-not-visible": 0,
        "wait-stable": 15000,
    }

    def _device_is_spoken_for(self, serial: str) -> str | None:
        """Is anything else using this device? Returns a journal reason, or None if it is free.

        Only one thing can hold Android's UiAutomation slot, so a handover is safe exactly
        when this process is the only thing driving the device. Two callers are not, and both
        were found the hard way — the offload released the slot, something else took it back
        about six hundred milliseconds later, and the accessibility service was torn down in
        the middle of a run the device had already started. Measured on a 24-step flow with
        the helper on: 24 of 24 steps in 3.5s when the device was free, 3 or 4 steps in 22s
        when it was not, and never the same number twice.

        A background job is the first. It runs on its own thread inside a warm engine and
        genuinely overlaps other work, so there is nobody to ask to stand down.

        A daemon is the second, and it disqualifies its own engine too. Standing the device
        down (see :meth:`_device_stood_down`) was meant to make the in-daemon case work, and
        it went most of the way: under a warm daemon the offload went from never finishing a
        run to finishing 24 of 24 in 3.4s on roughly four runs in five. Roughly is the
        problem. The remaining failures still lose the accessibility service mid-run, they
        cost about 25s against a 17s host path, and after tracing every device call in the
        process they have no in-process cause left — so something outside it is still taking
        the slot, and naming that is the work this guard is waiting on.

        Being wrong here is only ever slower, never incorrect: the host finishes whatever the
        device did not. But an offload that pays off four times in five and taxes the fifth is
        not a good trade for a warm daemon, which is the default way AUA runs.
        """

        import os

        from . import daemon
        from .jobs import manager_for

        try:
            if manager_for(self).active() is not None:
                return "job_running"
        except Exception:  # noqa: BLE001 - unreadable job state is not evidence of a job
            pass

        # Ask, do not infer. `socket_path` appends the serial, and inside the daemon the
        # configured socket already carries it, so a pidfile lookup from in there lands on a
        # path that never exists and reports the device free — which is exactly backwards.
        if daemon.serving():
            return None if self.config.helper.offload_under_daemon else "daemon_owns_device"

        try:
            pid, _ = daemon.read_pidfile(daemon.socket_path(self.config, serial=serial) + ".pid")
        except Exception:  # noqa: BLE001 - no readable pidfile means nobody to conflict with
            return None
        if pid is None or pid == os.getpid():
            return None
        try:
            os.kill(pid, 0)  # liveness only; a stale pidfile must not block the handover
        except OSError:
            return None
        return "another_process_owns_device"

    @contextlib.contextmanager
    def _device_stood_down(self) -> Iterator[bool]:
        """Put the device down for the duration of a handover, and pick it up again after.

        Closing ``self._device`` is not enough, and believing it was is what made the offload
        unreliable inside a warm daemon. The rolling capture buffer is handed ``device.screenshot``
        when it starts, so it holds its own reference to the same uiautomator2 client; its
        sampling thread keeps firing, uiautomator2 silently restarts the server the call needs,
        and the slot is gone again while the device is still working through its steps. The
        buffer has to be paused, not just the handle released.

        Both are restored on the way out, including after an exception, because leaving a
        daemon with a stopped capture buffer would quietly break the next ``capture last``.

        Yields whether the device actually went quiet. False means a frame grab is still in
        flight and the caller must not hand the slot over: it will land mid-run and take the
        slot straight back. Ignoring that answer is what was left of the flakiness.
        """

        buffer = self._capture
        resume_capture = False
        settled = True
        if buffer is not None and buffer.running and not buffer.paused:
            # Wait for a frame already in flight. Setting the flag alone leaves the sampling
            # thread free to take one more screenshot, and one is enough: it reconnects
            # uiautomator2, which takes the slot straight back off the helper.
            # Generous against a normal frame (tens of milliseconds) and cheap when the
            # answer is no. It used to expire regularly — about three runs in ten, and only
            # back-to-back — because the buffer held a device-bound ``screenshot`` and its
            # next tick sat inside a uiautomator2 reconnect the previous handover had made
            # necessary. It samples through :meth:`_capture_screenshot` now, which never
            # connects, so a buffer that will not settle in two seconds is genuinely busy.
            settled = buffer.pause("handover", settle_s=2.0)
            resume_capture = True
        if self._device is not None:
            self._device.close()
            self._device = None
        # Pausing the buffer stops it *asking*; this stops it being *answered*. Both are
        # needed, because a tick already past the pause check would otherwise reconnect
        # uiautomator2 through the engine and take the slot back off a helper mid-run.
        self._stood_down = True
        try:
            yield settled
        finally:
            self._stood_down = False
            if resume_capture and buffer is not None:
                buffer.resume()

    def _quiesce_background_device_work(self, timeout_s: float = 5.0) -> bool:
        """Wait until nothing but this thread is talking to the device. True if that is so.

        AUA speculates in the background — a hierarchy prefetch, an async memory write — and
        both reach the device through uiautomator2. Handing the UiAutomation slot to the
        helper while one is in flight is what made the offload unreliable: the background call
        fails, uiautomator2 restarts its server to recover, and that restart suppresses the
        accessibility service the device is *currently* running steps through. The run then
        stops at a different step every time, which is exactly the failure that is hardest to
        read from a log.
        """

        idle = self._prefetch.quiesce(timeout_s)
        deadline = time.monotonic() + timeout_s
        with self._mem_threads_lock:
            threads = list(self._mem_threads)
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)
            if thread.is_alive():
                idle = False
        return idle

    def _journal_helper(
        self,
        outcome: str,
        serial: str | None,
        *,
        cmd: str | None = None,
        ok: bool | None = None,
        args: dict[str, Any] | None = None,
        result: Any = None,
        **fields: Any,
    ) -> None:
        """Record what the helper decided, so it can be diagnosed after the run.

        The helper is the one subsystem that can silently do nothing: it declines for half a
        dozen legitimate reasons and the run still succeeds, just slower. Without a line in
        the journal there is no way to answer "did it fire, and if not why" once the run is
        over — which is exactly the question worth asking while its thresholds are still
        being tuned on real flows.

        The keyword arguments exist because this started out serving one caller and now serves
        two with genuinely different needs, and the difference was visible in the dashboard as
        three separate wrong things:

        ``cmd``
            Outcomes are past-tense — ``offloaded``, ``skipped``, ``partial`` — which reads
            correctly for a decision the offload made on its own. Composed into ``helper.<outcome>``
            for a *user-invoked* command it produced ``helper.drove``, which the dashboard shows in
            the slot where the command the caller ran belongs. A reader sees a command name that
            does not exist and cannot be searched for.
        ``ok``
            Derived as ``outcome in {"offloaded", "partial"}``, so any outcome added later is false
            by construction. A drive that reached its goal was rendered with a red FAIL badge.
        ``args`` / ``result``
            Everything used to go to ``extra``, which the dashboard does not render. Its two panels
            read ``args`` and the response, so a successful drive showed an empty request and a bare
            ``{"ok": false}`` — the goal, the budget, the steps and the stop reason all present in
            the journal and none of them on screen.
        """

        from . import journal

        with contextlib.suppress(Exception):
            journal.record(
                cache_dir=self.config.cache.dir,
                serial=serial,
                source="helper",
                cmd=cmd or f"helper.{outcome}",
                args=args,
                ok=(outcome in {"offloaded", "partial"}) if ok is None else ok,
                result=result,
                extra=fields,
            )

    @contextlib.contextmanager
    def _device_agent_borrowed(self, *, purpose: str) -> Iterator[Any]:
        """Hand Android's UiAutomation slot to the on-device helper and yield an open channel.

        Every caller that runs work *inside* the device needs all of this, in this order, and none
        of it is optional — each step here is a failure that was diagnosed on a real device, and
        the comments say which. A second copy of this sequence would not learn the next fix, so
        there is one, and the two things callers genuinely differ on are left to them:

        * **What counts as enough work to be worth the handover.** The flow offload has
          ``helper.min_flow_steps``; see :meth:`drive_on_device` for why an autonomous drive has no
          equivalent floor.
        * **Whether a refusal is fatal.** For the flow offload it is not — it is an optimisation, so
          it journals and runs on the host. For a command the user asked for by name it is, because
          silently doing nothing is the wrong answer to an explicit request. So this raises
          :class:`_HandoverRefused` and the caller chooses.

        The refusal is journalled here, at the point of refusal, so the diagnostic line is identical
        whichever caller asked.
        """

        agent = None
        serial: str | None = None
        try:
            agent = self.platform.capability("device_agent")
        except Exception as exc:  # noqa: BLE001 - no helper on this platform
            self._journal_helper("skipped", None, reason="platform_has_no_helper")
            raise _HandoverRefused("platform_has_no_helper", None) from exc

        # Deliberately NOT `self.device`: touching that connects uiautomator2, and connecting
        # it is the single most expensive part of the handover. With the slot already taken
        # the device has to give it up and wait for the helper to rebind — measured 2155ms,
        # against 16ms when nothing ever attached. Resolving the serial on its own is what
        # keeps the cheap path reachable, and drops the whole fixed cost from 2839ms to 682ms.
        serial = self._leased_serial()
        if serial is None:
            self._journal_helper("skipped", None, reason="no_target_serial")
            raise _HandoverRefused("no_target_serial", None)
        self.begin_device_use(serial)
        # Whether uiautomator2 is already attached is the single biggest factor in what this
        # costs (682ms if not, 2839ms if so), so record it: a run that looks disappointing is
        # usually one where something connected before the handover got a chance.
        was_connected = self._device is not None

        # Order matters. ``is_bound`` cannot be asked yet: this engine is holding a
        # uiautomator2 connection, and Android suppresses every accessibility service
        # while UiAutomation is held, so the helper would look absent no matter how
        # healthy it is. ``is_enabled`` reads the setting, which suppression does not
        # touch, so it is the one question worth asking first — and asking it first also
        # means a device with no helper never pays for a pointless handover.
        if not agent.is_enabled(serial):
            if not self.config.helper.auto_setup:
                self._journal_helper("skipped", serial, reason="auto_setup_disabled")
                raise _HandoverRefused("auto_setup_disabled", serial)
            # Ask cheaply whether root is even plausible before doing anything with a
            # side effect: `adb root` restarts adbd and costs about a second, and on a
            # retail phone or a Play image the answer is always no. Remembering that
            # answer per serial is what keeps "just switch it on" from taxing every
            # single run on a device that can never run the helper.
            if serial in self._helper_unavailable:
                self._journal_helper("skipped", serial, reason="known_unavailable")
                raise _HandoverRefused("known_unavailable", serial)
            if not agent.rootable(serial):
                self._helper_unavailable.add(serial)
                logger.debug("helper: %s cannot run adbd as root; using the polling path", serial)
                self._journal_helper("skipped", serial, reason="not_rootable")
                raise _HandoverRefused("not_rootable", serial)
            try:
                self._record_device_agent_change(serial)
                agent.enable(serial)
            except Exception as exc:  # noqa: BLE001 - setup is best-effort by design
                self._helper_unavailable.add(serial)
                logger.debug("helper setup failed on %s (%s); polling instead", serial, exc)
                self._journal_helper("skipped", serial, reason="setup_failed", error=str(exc)[:160])
                raise _HandoverRefused("setup_failed", serial, str(exc)[:160]) from exc

        # Nothing else may be mid-call on this device when the slot changes hands.
        if not self._quiesce_background_device_work():
            self._journal_helper("skipped", serial, reason="device_busy_in_background")
            raise _HandoverRefused("device_busy_in_background", serial)

        blocker = self._device_is_spoken_for(serial)
        if blocker is not None:
            self._journal_helper("skipped", serial, reason=blocker)
            raise _HandoverRefused(blocker, serial)

        with self._device_stood_down() as device_is_quiet:
            if not device_is_quiet:
                # The capture buffer never went quiet, so a screenshot is still in flight
                # and will reconnect uiautomator2 the moment it lands — mid-run, taking
                # the slot back off the helper. Every observed failure of this kind looked
                # like a broken helper and was this: the handover cost 10-13s and lost the
                # accessibility service, against 1.7s when the buffer had settled.
                self._journal_helper("skipped", serial, reason="capture_would_not_settle")
                raise _HandoverRefused("capture_would_not_settle", serial)
            agent.release_uiautomation(serial)

            if not agent.is_bound(serial):
                # Record whether the slot is *still* held. "Not bound" has two very
                # different causes — a helper that will not start, and a uiautomator2
                # server that outlived the release — and only the second is AUA's own
                # doing.
                self._journal_helper(
                    "skipped",
                    serial,
                    reason="not_bound_after_release",
                    u2_was_connected=was_connected,
                    uiautomation_still_held=agent.uiautomation_held(serial),
                )
                raise _HandoverRefused("not_bound_after_release", serial)

            channel = agent.open_channel(serial, timeout=self.config.helper.connect_timeout_s)
            try:
                yield _DeviceLoan(
                    channel=channel, serial=serial, purpose=purpose, u2_was_connected=was_connected
                )
            finally:
                channel.close()

    # ------------------------------------------------------------------ the lane that always works

    # -- recording a human's journey ---------------------------------------

    # ----------------------------------------------------------------- flows (§6b)

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
        with contextlib.suppress(Exception):
            self.release_device_use()

    # ------------------------------------------------- what the caller costs to think

    def _caller_latency_store(self) -> Any:
        """This caller's cross-process latency record, or None when it cannot be identified.

        Keyed by lease owner rather than device serial: the gap is a property of whoever is
        generating the calls, and ``resolve_owner`` answers "which agent is asking" without
        touching a device — which matters because this is read before anything connects, and
        because the warm daemon adopts the client's owner per request, so daemon-routed and
        in-process calls resolve to the same record.
        """
        from .caller_latency import CallerLatencyStore

        key = self._caller_latency_key
        if key is None:
            from . import leases

            # Same precedence the lease layer uses, so a daemon-adopted client owner and an
            # in-process CLI owner resolve to one record rather than two halves of one estimate.
            with contextlib.suppress(Exception):
                key = str(
                    getattr(self, "_lease_owner_resolved", None)
                    or leases.resolve_owner(getattr(self, "_lease_owner", None))
                )
            if not key:
                return None
            self._caller_latency_key = key
        return CallerLatencyStore(Path(self.config.memory.dir).expanduser() / "state", key)

    def open_caller_turn(self) -> None:
        """Measure the caller's think time before this call does any work.

        Called by the adapters (CLI ``_run``, MCP dispatch) at the top of a command, because a
        *caller* turn is a process the agent invoked — not a daemon round trip, which is aua's
        own transport and would halve every gap it measured. Best-effort throughout: a ceiling
        is an optimisation, and no bookkeeping failure may cost the caller its command.
        """
        if self._caller_turn is not None:
            return
        store = self._caller_latency_store()
        if store is None:
            return
        with contextlib.suppress(Exception):
            self._caller_turn = store.open_turn()
            self._caller_profile_cache = False

    def close_caller_turn(self, fingerprint: str | None = None) -> None:
        """Stamp when this call returned, and the screen it handed back.

        The stamp is the far end of the next gap, so it has to be written even when the command
        failed — a caller thinks just as long about an error.

        *fingerprint* is passed explicitly by the adapter, from the payload it is about to emit.
        Falling back to this engine's own last observation is only right when the engine that
        answered is the engine that stamps: under the warm daemon the work happens in another
        process, so the CLI's engine has no observation and the fallback silently writes None —
        which is the whole "previous screen gone" feature quietly never arming itself. The
        adapter has the answer either way, so it is the one asked.
        """
        if self._caller_turn is None:
            return
        try:
            store = self._caller_latency_store()
            if store is None:
                return
            if fingerprint is None:
                cached = self._last_analyze_result
                fingerprint = cached.meta.fingerprint if cached is not None else None
            with contextlib.suppress(Exception):
                store.close_turn(fingerprint)
        finally:
            # A warm MCP engine serves many caller turns. Keeping this object made the next
            # open a no-op and every later report describe the first call forever.
            self._caller_turn = None

    def _caller_profile(self) -> Any:
        """The caller estimate this call should size its waits from.

        Falls back to the stored profile when no turn was opened: under the warm daemon the
        turn is opened in the CLI process while the wait runs here. That fallback reads a file,
        and `_bounded_wait_ms` runs once per wait — and once per step of a flow — so the answer
        is memoised for the life of this owner rather than re-read on the critical path.
        """
        turn = self._caller_turn
        if turn is not None:
            return turn.profile
        if self._caller_profile_cache is not False:
            return self._caller_profile_cache
        profile = None
        store = self._caller_latency_store()
        if store is not None:
            with contextlib.suppress(Exception):
                profile = store.profile()
        self._caller_profile_cache = profile
        return profile

    def caller_turn_report(self, current_fingerprint: str | None = None) -> dict[str, Any] | None:
        """The caller-facing summary attached to a response, or None with nothing to say.

        Returns None unless something was actually *measured* this turn — a gap since the last
        call, or a verdict on the previous screen. The first call of a session has neither, and
        on that call this block would be a header of nulls plus a ceiling nobody asked about,
        added to every response including ones that never wait (`screenshot`). Reporting the
        budget is worth a few tokens once there is a measurement to justify it; announcing it
        unprompted on a cold call is not.
        """
        turn = self._caller_turn
        if turn is None:
            return None
        report: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            report.update(turn.profile.as_response())
        gone = self._previous_screen_gone(current_fingerprint)
        if gone is not None:
            report["previous_screen_gone"] = gone
            if turn.previous_age_ms is not None:
                report["previous_screen_age_ms"] = turn.previous_age_ms
        if not report:
            return None
        with contextlib.suppress(Exception):
            # Only alongside a measurement: the ceiling is what the measurement bought, and on
            # its own it is a constant the caller can read from config.
            ceiling, mode = self._wait_ceiling()
            report["wait_ceiling_ms"] = ceiling
            report["wait_ceiling_mode"] = mode
        return report

    def _caller_turn_facts(self) -> Any | None:
        """What this caller was last handed, and how long ago — in either process.

        In-process the open turn already holds it. Under the warm daemon the turn belongs to
        the CLI (``open_caller_turn`` is deliberately never called here, because a daemon round
        trip is not a caller), so the record it stamped is read instead — read only, and without
        opening a turn, so the gap measurement stays the CLI's to make.
        """
        turn = self._caller_turn
        if turn is not None:
            return turn
        store = self._caller_latency_store()
        if store is None:
            return None
        with contextlib.suppress(Exception):
            return store.peek_turn()
        return None

    def _note_screen_moved(
        self, shown: AnalyzeResult | None, fresh: AnalyzeResult | None
    ) -> str | None:
        """Record whether *fresh* shows the caller's screen already replaced, and say why.

        Called from a **pre-action** resolution read, which is the only read that happens
        before the device is touched, so the caller's own action can never be reported as the
        world moving. *shown* must be the published screen as it was before that read, because
        a selector resolve refreshes the id cache on its way through.
        """
        verdict: tuple[str, str] | None = None
        with contextlib.suppress(Exception):
            verdict = self._screen_moved_verdict(shown, fresh)
        self._screen_moved = verdict
        return verdict[1] if verdict else None

    def _screen_moved_verdict(
        self, shown: AnalyzeResult | None, fresh: AnalyzeResult | None
    ) -> tuple[str, str] | None:
        """``(held fingerprint, reason)`` when the world moved by itself, else None.

        Six ways this must stay silent, because a warning a caller sees on every call is a
        warning it stops reading:

        * nothing was published for this caller to be holding (the first action of a session);
        * the published screen already carried ``stale_risk`` — a non-settled arrival predicts
          its own replacement, so the replacement is the caller's action landing late, not the
          world moving;
        * the live screen is the published one, byte for byte;
        * the case that decides this whole design: the fingerprint moved but **nothing the
          caller could act on** did. Measured on one live emulator screen, three consecutive
          `analyze` calls with nobody touching the device gave node counts 43, 43, 44 and two
          different hierarchy fingerprints, with the same nine actionable ids every time. A
          clock, a badge, a "typing…" line all do that. So the verdict is decided on the
          actionable set — which is also exactly the set the caller's next command can name;
        * the published screen is not the one that was stamped, so what the caller is holding
          cannot be established (a second agent drove the same device, a cache that never got
          written);
        * and the gap was not one this caller generated — `gap_ignored` already draws that line
          for the wait ceiling, and past ``IDLE_GAP_MS`` the screen being different means
          somebody walked away, not that an interstitial arrived.

        The order is load-bearing, not just tidy. The first three are pure comparisons of two
        payloads already in memory and answer the overwhelming majority of calls; the last two
        need this caller's stamped record, and identifying the caller can cost a `ps` (see
        :meth:`_caller_latency_store`). So the cheap gates run first and the hot path never
        pays for the bookkeeping.
        """
        if fresh is None or shown is None:
            return None
        held = shown.meta.fingerprint
        if not held or held == fresh.meta.fingerprint:
            return None
        if shown.meta.stale_risk:
            # Sixth silence: the tool itself told the caller this frame may be replaced when
            # content lands (a non-settled arrival). Its replacement is the *predicted*
            # consequence of the caller's own action — "nothing you sent caused that" would
            # be a false attribution stacked on a screen already flagged as not to be held.
            return None
        held_keys = _actionable_keys(shown.elements)
        live_keys = _actionable_keys(fresh.elements)
        gone = held_keys - live_keys
        arrived = live_keys - held_keys
        if not gone and not arrived:
            return None
        facts = self._caller_turn_facts()
        if facts is None or getattr(facts, "previous_fingerprint", None) != held:
            return None
        if getattr(getattr(facts, "profile", None), "gap_ignored", None):
            return None
        age_ms = getattr(facts, "previous_age_ms", None)
        held_for = f", held {age_ms / 1000.0:.1f}s" if isinstance(age_ms, int) else ""
        return (
            held,
            "the screen you were last shown was replaced before this call touched the device "
            f"(controls -{len(gone)} +{len(arrived)}{held_for}). Nothing you sent caused that "
            "— act on ids from THIS response, not the previous one.",
        )

    def _consume_screen_moved(self) -> str | None:
        """The pending verdict, if it is still about the screen this caller is holding."""
        verdict = self._screen_moved
        self._screen_moved = None
        if verdict is None:
            return None
        facts = self._caller_turn_facts()
        previous = getattr(facts, "previous_fingerprint", None)
        return verdict[1] if previous and previous == verdict[0] else None

    def _previous_screen_gone(self, current_fingerprint: str | None = None) -> bool | None:
        """Has the screen described by the caller's previous result been replaced?

        Answered from fingerprints already in hand — the one stamped when the last call returned
        and the one this call's observation computed — so it costs no device read. None means
        there is nothing to compare, which is honest rather than reassuring: a caller with no
        prior observation cannot be holding a stale one, and a call that read no screen has no
        evidence either way.
        """
        turn = self._caller_turn
        if turn is None:
            return None
        previous = getattr(turn, "previous_fingerprint", None)
        cached = self._last_analyze_result
        current = current_fingerprint or (cached.meta.fingerprint if cached is not None else None)
        if not previous or not current:
            return None
        return previous != current

    def _start_call(self) -> float:
        """Start the clock for a call the caller is waiting on, and return the stamp.

        `_acting` does this for every gesture. A wait needed it too and did not have it: with
        no stamp its response carried no `wall_ms` at all, so the calls most likely to BE the
        slow part of a run were the only ones that never said what they cost.

        Both clocks are read because they answer different questions. The monotonic one
        measures a duration and cannot jump; the epoch one names an instant and is comparable
        with another process's journal.
        """
        started = time.monotonic()
        self._call_started_at = started
        self._call_started_epoch_ms = int(time.time() * 1000)
        return started

    def _journal_call_answer(self, result: ActionResult, *, outcome: str | None = None) -> None:
        """Record what this call answered and what it cost in the session access log.

        One place, on the way out, so an action and a wait are measured the same way and the
        number in the log is the number the caller was handed (`wall_ms`) — not a second,
        smaller measurement of the gesture with the settle left out.

        No clock is read here and no device is touched: the measurement is the one the response
        already carries, so the whole cost is one small session write on a call that has just
        paid for a device round trip. A call that measured nothing gets no line, because a
        fabricated duration in a latency log is worse than a missing one.
        """
        # Consume the stamp for the same reason `_wall_ms` does: a leftover epoch would date
        # the next call's line by the age of this one.
        started_at_ms = self._call_started_epoch_ms
        self._call_started_epoch_ms = None
        if self._action_recording_suppression:
            return
        mem = self._memory
        if mem is None or self._device is None:
            return
        cost = result.wall_ms if result.wall_ms is not None else result.elapsed_ms
        if cost is None:
            return
        observation = result.observation
        try:
            mem.record_call_cost(
                self._device.serial,
                kind=result.action,
                elapsed_ms=cost,
                started_at_ms=started_at_ms,
                # A wait reports only two ends, and calling the second one "failed" would
                # make a screen that simply never arrived look like a broken call.
                outcome=outcome
                or result.await_outcome
                or (
                    "ok"
                    if result.ok
                    else "timeout"
                    if result.action.startswith("wait")
                    else "failed"
                ),
                screen=result.known_screen
                or (observation.meta.known_screen if observation is not None else None),
                detail=result.detail,
            )
        except Exception as exc:  # pragma: no cover - diagnostics never fail the call
            logger.debug("memory record_call_cost failed: %s", exc)

    def _wall_ms(self) -> int | None:
        """Milliseconds since this call started, consuming the stamp.

        Consume-once, because the engine outlives a single command: under the warm daemon a
        leftover stamp reported a 1.8s wait as 51s — the age of the previous action, not the
        duration of this one. A stamp that can only be read once cannot be misattributed; a
        call that never set one reports nothing rather than someone else's number.
        """
        started = getattr(self, "_call_started_at", None)
        if started is None:
            return None
        self._call_started_at = None
        return int((time.monotonic() - started) * 1000)

    # ------------------------------------------------------------- scroll internals

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

    # ----------------------------------------------------------------- device extras

    def shell(self, argv: list[str], *, timeout_ms: int = 30_000) -> ShellResult:
        """Run one bounded read-only target command through the leased device runtime."""

        if not argv:
            raise UsageError(
                "shell needs a command",
                hint="e.g. `aua shell pm path com.example.app`",
            )
        if not 100 <= int(timeout_ms) <= 120_000:
            raise UsageError("shell timeout must be between 100 and 120000 ms")
        platform = self.platform
        if not platform.supports("device.shell"):
            raise DeviceError(
                f"platform '{platform.name}' cannot run read-only target commands",
                code="unsupported_capability",
            )
        return self.device.run_read_only_shell(
            [str(part) for part in argv], timeout_s=int(timeout_ms) / 1000.0
        )

    # ----------------------------------------------------------------- app bundle installs

    #: Install modes, narrowest first. ``if-needed`` is the default because the common case is a
    #: run that just wants the build present, and re-pushing an APK that is already there costs
    #: tens of seconds on an emulator for no change in state.
    INSTALL_MODES = ("if-needed", "reinstall", "fresh")

    # ----------------------------------------------------------------- logcat / suite

    # -- per-app log preferences -------------------------------------------
    # `config.logs` is one setting for every app on the host. These two are the per-app, across
    # sessions half of it: what an agent learns about one app's loggers is worth keeping, and
    # re-learning it every session is exactly the cost the digest exists to avoid.

    # ----------------------------------------------------------------- annotate

    #: Frames kept per device per suffix once AUA is naming the files itself. Enough to look
    #: back over a short flow, small enough that an always-on default cannot fill a disk.
    MAX_RUN_FRAMES = 40

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
        # The wall clock starts before the device is touched, for the same reason the log
        # window does: a duration measured from after the gesture excludes the gesture.
        self._start_call()
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
        focused: ElementId | None = None
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

    _DISK_NOTE = (
        "read from the on-disk capture index, NOT from a live buffer: these are frames a "
        "previous process recorded, nothing is being sampled now, and pruned frames are "
        "missing. Start a warm daemon (`aua daemon start`) for live post-action capture."
    )

    def _invalidate_cache(self) -> None:
        path = self._cache_path()
        with contextlib.suppress(OSError):  # pragma: no cover
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------------------------
    # Methods implemented in the engine_* domain modules. Each is a module-level function
    # whose first parameter is the Engine; binding it here makes it a method, so calls,
    # monkeypatches, inspect.getsource and __doc__ all behave as if it were defined inline.
    # To add a method: write it in the domain module, then attach it below.
    # ------------------------------------------------------------------------------------

    # engine_analyze: Perceiving the screen for `aua analyze`: hierarchy, OCR and vision capture, the analyze pipeline and its semantic query path, screenshot/inspect/annotate, and perception-provider status.
    _effective_with_image = engine_analyze._effective_with_image
    _context = engine_analyze._context
    _capture_hierarchy = engine_analyze._capture_hierarchy
    _kick_hierarchy_prefetch = engine_analyze._kick_hierarchy_prefetch
    _screenshot = engine_analyze._screenshot
    _start_hierarchy_ocr = engine_analyze._start_hierarchy_ocr
    _finish_hierarchy_ocr = engine_analyze._finish_hierarchy_ocr
    _fuse_hierarchy_ocr = engine_analyze._fuse_hierarchy_ocr
    _capture_hierarchy_with_ocr = engine_analyze._capture_hierarchy_with_ocr
    _map_skips_ocr = engine_analyze._map_skips_ocr
    _run_vision = engine_analyze._run_vision
    _repair_lossy_text = engine_analyze._repair_lossy_text
    ask_screen = engine_analyze.ask_screen
    _resolve_pins = engine_analyze._resolve_pins
    _attach_visual_identity = staticmethod(engine_analyze._attach_visual_identity)
    analyze = engine_analyze.analyze
    _analyze_screen = engine_analyze._analyze_screen
    _gate_decide = engine_analyze._gate_decide
    _analyze_query = engine_analyze._analyze_query
    _finish_query = engine_analyze._finish_query
    _match_query = engine_analyze._match_query
    _map_grounding = engine_analyze._map_grounding
    inspect = engine_analyze.inspect
    screenshot = engine_analyze.screenshot
    provider_status = engine_analyze.provider_status
    _with_raw_image = engine_analyze._with_raw_image
    _prune_run_frames = engine_analyze._prune_run_frames
    _maybe_annotate = engine_analyze._maybe_annotate
    _default_annotate_path = engine_analyze._default_annotate_path
    _resolve = engine_analyze._resolve

    # engine_memory: Learning into per-app memory and reading it back: recording screens and actions (and their timings) into the app map, the runtime flag context, learned control costs and next-action hints, the `aua memory update` command, and the knowledge read-back commands orient and explore mine/plan (source-tree deeplink mining, exploration worklist).
    _version_for = engine_memory._version_for
    _sync_runtime_flag_context = engine_memory._sync_runtime_flag_context
    _record_screen_safe = engine_memory._record_screen_safe
    _join_memory_writers = engine_memory._join_memory_writers
    _record_action_safe = engine_memory._record_action_safe
    _next_actions = engine_memory._next_actions
    _price_elements = engine_memory._price_elements
    _screen_timings_safe = engine_memory._screen_timings_safe
    _slow_controls_safe = engine_memory._slow_controls_safe
    _record_action_timing_safe = engine_memory._record_action_timing_safe
    _learned_action_budget = engine_memory._learned_action_budget
    memory_update = engine_memory.memory_update
    explore_mine = engine_memory.explore_mine
    explore_plan = engine_memory.explore_plan
    orient = engine_memory.orient

    # engine_actions: Acting on elements by id: target and selector resolution, tap/long-press/double-tap, text input, clear and erase, mic audio injection, swipe/scroll/key gestures, keyboard, clipboard paste/copy, a11y actions, and the RouteStep record each action emits.
    _action_site = engine_actions._action_site
    _step = engine_actions._step
    resolve = engine_actions.resolve
    resolve_selector = engine_actions.resolve_selector
    _match_by_vision = engine_actions._match_by_vision
    _resolve_container_rid = engine_actions._resolve_container_rid
    _target = engine_actions._target
    _binding_label = staticmethod(engine_actions._binding_label)
    _key_may_be_visual = staticmethod(engine_actions._key_may_be_visual)
    _miss_observation = engine_actions._miss_observation
    _resolve_action_key = engine_actions._resolve_action_key
    _resolve_action_id = engine_actions._resolve_action_id
    target_report = engine_actions.target_report
    _acting_target = engine_actions._acting_target
    _tap_point = engine_actions._tap_point
    tap = engine_actions.tap
    tap_point = engine_actions.tap_point
    long_press = engine_actions.long_press
    mic_inject = engine_actions.mic_inject
    mic_speak = engine_actions.mic_speak
    double_tap = engine_actions.double_tap
    input_text = engine_actions.input_text
    _submission_status = staticmethod(engine_actions._submission_status)
    _semantic_send_recommendation = staticmethod(engine_actions._semantic_send_recommendation)
    _system_bar_top = engine_actions._system_bar_top
    _aim = engine_actions._aim
    _typed_text_landed = engine_actions._typed_text_landed
    clear = engine_actions.clear
    _dump = engine_actions._dump
    _scroll_box = engine_actions._scroll_box
    _swipe_path = engine_actions._swipe_path
    _settle_after_swipe = engine_actions._settle_after_swipe
    _probe = engine_actions._probe
    _swipe_once = engine_actions._swipe_once
    swipe = engine_actions.swipe
    scroll = engine_actions.scroll
    scroll_to = engine_actions.scroll_to
    key = engine_actions.key
    hide_keyboard = engine_actions.hide_keyboard
    _ime_shown = engine_actions._ime_shown
    paste = engine_actions.paste
    copy_text = engine_actions.copy_text
    a11y_scroll = engine_actions.a11y_scroll
    a11y_action = engine_actions.a11y_action
    erase = engine_actions.erase

    # engine_observation: The post-action `observation` every action returns: the shared _observe pipeline, the loading/readiness predicate and settle waits, arrival and stale-risk verdicts, the before/after change summary, and crash and app-log evidence.
    _compact_action_diff = staticmethod(engine_observation._compact_action_diff)
    _analyze_post_action = engine_observation._analyze_post_action
    _change_has_semantic_effect = staticmethod(engine_observation._change_has_semantic_effect)
    _unready_destination_risk = engine_observation._unready_destination_risk
    _content_bare = staticmethod(engine_observation._content_bare)
    _destination_confirmed = staticmethod(engine_observation._destination_confirmed)
    _post_action_change = engine_observation._post_action_change
    _await_rendered_destination = engine_observation._await_rendered_destination
    _readable_label = staticmethod(engine_observation._readable_label)
    _destination_rendered = staticmethod(engine_observation._destination_rendered)
    _arrival_report = staticmethod(engine_observation._arrival_report)
    _tap_settle_needs_confirmation = staticmethod(engine_observation._tap_settle_needs_confirmation)
    _observe = engine_observation._observe
    _frame_history_matters = staticmethod(engine_observation._frame_history_matters)
    _finalize_observed_action = engine_observation._finalize_observed_action
    _stale_observation_risk = staticmethod(engine_observation._stale_observation_risk)
    _spend_stable_delay = engine_observation._spend_stable_delay
    _note_empty_observation = engine_observation._note_empty_observation
    _await_post_action_ready = engine_observation._await_post_action_ready
    _observation_is_loading = engine_observation._observation_is_loading
    _app_left_foreground = staticmethod(engine_observation._app_left_foreground)
    _crash_evidence = engine_observation._crash_evidence
    _app_logs = engine_observation._app_logs
    _change_summary = engine_observation._change_summary

    # engine_waits: Waiting on and checking the screen: has/expect, wait/wait_stable/wait_changed/wait_after_change/await_predicate, the locale bridge for text matching, the caller-sized wait budget, hierarchy change detection, and the background jobs that carry a long wait.
    _job_checkpoint = engine_waits._job_checkpoint
    _current_job_cancel_event = engine_waits._current_job_cancel_event
    _job_sleep = engine_waits._job_sleep
    _job_requires_warm_transport = engine_waits._job_requires_warm_transport
    job_start = engine_waits.job_start
    job_status = engine_waits.job_status
    job_wait = engine_waits.job_wait
    job_cancel = engine_waits.job_cancel
    job_list = engine_waits.job_list
    wait_stable = engine_waits.wait_stable
    has = engine_waits.has
    _has_wait_result = engine_waits._has_wait_result
    _has_miss = engine_waits._has_miss
    _find_translated = engine_waits._find_translated
    _app_strings = engine_waits._app_strings
    _locale_candidates = engine_waits._locale_candidates
    _translated_hint = staticmethod(engine_waits._translated_hint)
    _text_miss_hint = engine_waits._text_miss_hint
    _locale_hint = staticmethod(engine_waits._locale_hint)
    _wait_for_any = engine_waits._wait_for_any
    _ocr_contains = engine_waits._ocr_contains
    _hand_back_what_is_on_screen = engine_waits._hand_back_what_is_on_screen
    _screen_already_answers = engine_waits._screen_already_answers
    _wait_ceiling = engine_waits._wait_ceiling
    _bounded_wait_ms = engine_waits._bounded_wait_ms
    _sleep_between_polls = engine_waits._sleep_between_polls
    _say_the_wait_was_shortened = engine_waits._say_the_wait_was_shortened
    _wait_ceiling_explanation = staticmethod(engine_waits._wait_ceiling_explanation)
    _hint_for_a_shortened_wait = staticmethod(engine_waits._hint_for_a_shortened_wait)
    _await_terms_on_observation = staticmethod(engine_waits._await_terms_on_observation)
    _await_observation_identity = staticmethod(engine_waits._await_observation_identity)
    _await_destination_changed = staticmethod(engine_waits._await_destination_changed)
    _arrival_predicate_suggestions = staticmethod(engine_waits._arrival_predicate_suggestions)
    _sample_action_destination = engine_waits._sample_action_destination
    await_predicate = engine_waits.await_predicate
    _unknown_map_selectors = engine_waits._unknown_map_selectors
    _await_result = engine_waits._await_result
    wait = engine_waits.wait
    _journal_wait_gave_up = engine_waits._journal_wait_gave_up
    hierarchy_fingerprint = engine_waits.hierarchy_fingerprint
    wait_changed = engine_waits.wait_changed
    wait_after_change = engine_waits.wait_after_change
    _wait_timeout_message = engine_waits._wait_timeout_message
    _node_state = engine_waits._node_state
    _check_predicates = engine_waits._check_predicates
    _expect_once = engine_waits._expect_once
    expect = engine_waits.expect

    # engine_navigation: Getting somewhere in the app: goto over the learned map with its planner fallback, navigate and reach, the goal-driven drive lanes, back_until with map-screen recognition, open_link deeplinks with chooser handling, and map_find route previews.
    drive_on_device = engine_navigation.drive_on_device
    _goal_in_the_apps_words = engine_navigation._goal_in_the_apps_words
    drive_on_host = engine_navigation.drive_on_host
    _mid_edge_path = engine_navigation._mid_edge_path
    _planner_view = engine_navigation._planner_view
    _drive_with_planner = engine_navigation._drive_with_planner
    _goto_assist_recover = engine_navigation._goto_assist_recover
    _assist_suggestion = engine_navigation._assist_suggestion
    goto = engine_navigation.goto
    reach = engine_navigation.reach
    navigate = engine_navigation.navigate
    map_find = engine_navigation.map_find
    back_until = engine_navigation.back_until
    _recognize_screen_read_only = engine_navigation._recognize_screen_read_only
    _await_known_screen = engine_navigation._await_known_screen
    _mapped_screen_state = engine_navigation._mapped_screen_state
    _mapped_screen_is_root = engine_navigation._mapped_screen_is_root
    _back_terminal_frame_is_weak = staticmethod(engine_navigation._back_terminal_frame_is_weak)
    _semantic_back_selector = staticmethod(engine_navigation._semantic_back_selector)
    _back_observation_identity = staticmethod(engine_navigation._back_observation_identity)
    _back_observed_package = staticmethod(engine_navigation._back_observed_package)
    _back_until_result = engine_navigation._back_until_result
    open_link = engine_navigation.open_link
    _flag_deeplink_that_did_not_land = engine_navigation._flag_deeplink_that_did_not_land
    _remember_pending_flag_context = engine_navigation._remember_pending_flag_context
    _is_chooser = engine_navigation._is_chooser
    _chooser_app_labels = engine_navigation._chooser_app_labels
    _dismiss_chooser = engine_navigation._dismiss_chooser
    _remember_deeplink_safe = engine_navigation._remember_deeplink_safe

    # engine_flows: Saved flows and step execution: flow_run/save/list/delete, the step executor and its on-device offload, nested-flow preflight and arrival evidence, demo recording of a person's journey, and suite_run for AC checklists.
    _flows_for = engine_flows._flows_for
    _source_for = engine_flows._source_for
    _analyze_route_step = engine_flows._analyze_route_step
    _run_flow_assertion = engine_flows._run_flow_assertion
    _run_flow_order_assertion = engine_flows._run_flow_order_assertion
    _device_runnable_step = engine_flows._device_runnable_step
    _device_step_payload = engine_flows._device_step_payload
    _device_runnable_run = engine_flows._device_runnable_run
    _pick_offload_start = engine_flows._pick_offload_start
    _offload_steps_to_device = engine_flows._offload_steps_to_device
    _offload_from = engine_flows._offload_from
    _recorder = engine_flows._recorder
    demo_record_start = engine_flows.demo_record_start
    demo_record_stop = engine_flows.demo_record_stop
    _run_steps = engine_flows._run_steps
    _flow_ref_key = staticmethod(engine_flows._flow_ref_key)
    _resolve_nested_flow_node = engine_flows._resolve_nested_flow_node
    _resolve_nested_flow = engine_flows._resolve_nested_flow
    _flow_graph_identity = staticmethod(engine_flows._flow_graph_identity)
    _preflight_nested_flow_graph = engine_flows._preflight_nested_flow_graph
    _resolved_flow_disclosure = engine_flows._resolved_flow_disclosure
    _validate_flow_arrival_screen = engine_flows._validate_flow_arrival_screen
    _flow_leading_launch_establishes_origin = staticmethod(engine_flows._flow_leading_launch_establishes_origin)
    _flow_runtime_state = engine_flows._flow_runtime_state
    _execute_flow_steps = engine_flows._execute_flow_steps
    _flow_arrival_evidence = engine_flows._flow_arrival_evidence
    _settle_for_next_step = engine_flows._settle_for_next_step
    flow_run = engine_flows.flow_run
    flow_save = engine_flows.flow_save
    flow_delete = engine_flows.flow_delete
    flow_list = engine_flows.flow_list
    suite_run = engine_flows.suite_run

    # engine_sessions: The session contract from session_start to session_finish: goal planning, phase progress and the recommended-call ranking, candidate flows, and the session review.
    _goal_session_plan = engine_sessions._goal_session_plan
    session_start = engine_sessions.session_start
    _phase_recommended_call = engine_sessions._phase_recommended_call
    session_mark_phase = engine_sessions.session_mark_phase
    _complete_contract_phase_from_observation = engine_sessions._complete_contract_phase_from_observation
    session_progress = engine_sessions.session_progress
    _session_state = engine_sessions._session_state
    session_review = engine_sessions.session_review
    _session_candidate = engine_sessions._session_candidate
    session_candidate_flow = engine_sessions.session_candidate_flow
    _session_finish_summary = staticmethod(engine_sessions._session_finish_summary)
    session_finish = engine_sessions.session_finish

    # engine_policy: The optional local policy model: model_control status/action/chat/agent-test, policy tap-candidate and selection helpers, the session policy side channel, and session_autopilot which lets the policy drive a bounded stretch.
    _configured_policy_mode = engine_policy._configured_policy_mode
    _session_policy_mode = engine_policy._session_policy_mode
    model_control_status = engine_policy.model_control_status
    model_control_action = engine_policy.model_control_action
    model_control_chat = engine_policy.model_control_chat
    _evaluate_policy_context = engine_policy._evaluate_policy_context
    model_control_agent_test = engine_policy.model_control_agent_test
    _policy_selector_arguments = staticmethod(engine_policy._policy_selector_arguments)
    _policy_target_terms = staticmethod(engine_policy._policy_target_terms)
    _policy_tap_candidates = engine_policy._policy_tap_candidates
    _policy_navigation_waypoints = staticmethod(engine_policy._policy_navigation_waypoints)
    _policy_waypoint_arrived = staticmethod(engine_policy._policy_waypoint_arrived)
    _restore_term_case = staticmethod(engine_policy._restore_term_case)
    _policy_selection_goal = staticmethod(engine_policy._policy_selection_goal)
    _policy_suggestion = staticmethod(engine_policy._policy_suggestion)
    _policy_handoff = staticmethod(engine_policy._policy_handoff)
    _policy_context_is_current = engine_policy._policy_context_is_current
    _session_policy_output = engine_policy._session_policy_output
    _autopilot_public_policy_output = staticmethod(engine_policy._autopilot_public_policy_output)
    _autopilot_provider_failure = staticmethod(engine_policy._autopilot_provider_failure)
    _execute_guarded_policy_call = engine_policy._execute_guarded_policy_call
    session_autopilot = engine_policy.session_autopilot

    # engine_apps: The app under test: launch/stop/clear and install with the launch observation that follows, feature flags and prefs, private databases, logcat and per-app log preferences (stored and effective), and the app-under-test process bookkeeping.
    _launch_observation_is_transitional = staticmethod(engine_apps._launch_observation_is_transitional)
    _await_meaningful_launch_observation = engine_apps._await_meaningful_launch_observation
    _await_foreground = staticmethod(engine_apps._await_foreground)
    _await_launch_hierarchy = engine_apps._await_launch_hierarchy
    _invalidate_launch_observation = engine_apps._invalidate_launch_observation
    _adopt_recovered_launch_observation = engine_apps._adopt_recovered_launch_observation
    _mark_transitional_launch_observation = engine_apps._mark_transitional_launch_observation
    _finish_launch_content_observation = engine_apps._finish_launch_content_observation
    flags_set = engine_apps.flags_set
    _foreground_activity = engine_apps._foreground_activity
    _wait_foreground = engine_apps._wait_foreground
    _launch_entry = engine_apps._launch_entry
    _restart_app = engine_apps._restart_app
    _verify_flags = engine_apps._verify_flags
    flags_apply = engine_apps.flags_apply
    prefs_write = engine_apps.prefs_write
    app = engine_apps.app
    app_status = engine_apps.app_status
    install_app = engine_apps.install_app
    database_list = engine_apps.database_list
    database_schema = engine_apps.database_schema
    database_query = engine_apps.database_query
    database_execute = engine_apps.database_execute
    database_backup = engine_apps.database_backup
    database_backups = engine_apps.database_backups
    database_restore = engine_apps.database_restore
    logcat_mark = engine_apps.logcat_mark
    logcat = engine_apps.logcat
    app_log_prefs = engine_apps.app_log_prefs
    app_log_prefs_set = engine_apps.app_log_prefs_set
    _app_for_log_prefs = engine_apps._app_for_log_prefs
    _app_log_store = engine_apps._app_log_store
    _log_tag_is_hidden_elsewhere = engine_apps._log_tag_is_hidden_elsewhere
    _app_log_prefs_view = engine_apps._app_log_prefs_view
    _could_be_app_under_test = staticmethod(engine_apps._could_be_app_under_test)
    _note_app_under_test = engine_apps._note_app_under_test
    _app_process_replaced = engine_apps._app_process_replaced
    _effective_app_logs = engine_apps._effective_app_logs

    # engine_environment: The conditions the app runs under: network and airplane state and network profiles, the mock proxy and its rules, clock, location, orientation, clipboard, media, and developer options.
    clipboard_set = engine_environment.clipboard_set
    clipboard_get = engine_environment.clipboard_get
    location_set = engine_environment.location_set
    orientation_set = engine_environment.orientation_set
    orientation_get = engine_environment.orientation_get
    airplane_set = engine_environment.airplane_set
    airplane_toggle = engine_environment.airplane_toggle
    network_status = engine_environment.network_status
    network_offline = engine_environment.network_offline
    network_restore = engine_environment.network_restore
    network_profile_list = engine_environment.network_profile_list
    network_profile_status = engine_environment.network_profile_status
    network_profile_apply = engine_environment.network_profile_apply
    network_profile_restore = engine_environment.network_profile_restore
    media_add = engine_environment.media_add
    clock_set = engine_environment.clock_set
    _clock_backup_path = engine_environment._clock_backup_path
    _dev_backup_path = engine_environment._dev_backup_path
    _proxy_port = engine_environment._proxy_port
    dev_show = engine_environment.dev_show
    dev_anim = engine_environment.dev_anim
    dev_crashes = engine_environment.dev_crashes
    dev_profile = engine_environment.dev_profile
    proxy_start = engine_environment.proxy_start
    _claim_or_reap_proxy = engine_environment._claim_or_reap_proxy
    proxy_stop = engine_environment.proxy_stop
    proxy_status = engine_environment.proxy_status
    _adopt_own_proxy = engine_environment._adopt_own_proxy
    proxy_survey = engine_environment.proxy_survey
    _refresh_proxy_ownership_pid = engine_environment._refresh_proxy_ownership_pid
    _proxy_health_warning = engine_environment._proxy_health_warning
    _proxy_serial = engine_environment._proxy_serial
    _arm_mock_rule = engine_environment._arm_mock_rule
    mock_map = engine_environment.mock_map
    mock_rewrite = engine_environment.mock_rewrite
    mock_list = engine_environment.mock_list
    mock_clear = engine_environment.mock_clear
    mock_rm = engine_environment.mock_rm
    mock_record = engine_environment.mock_record
    mock_replay = engine_environment.mock_replay

    # engine_capture: Pixel evidence over time: the rolling capture buffer and its status/last/export/sheet/explain views, the capture sidecar, the capture hint analyze attaches, and device screen recording.
    _capture_screenshot = engine_capture._capture_screenshot
    _capture_screenshot_fn = engine_capture._capture_screenshot_fn
    record_start = engine_capture.record_start
    record_stop = engine_capture.record_stop
    _capture_hint = engine_capture._capture_hint
    capture_start = engine_capture.capture_start
    capture_stop = engine_capture.capture_stop
    capture_on = engine_capture.capture_on
    capture_off = engine_capture.capture_off
    capture_idle_pause = engine_capture.capture_idle_pause
    capture_idle_resume = engine_capture.capture_idle_resume
    _capture_serial = engine_capture._capture_serial
    _capture_from_disk = engine_capture._capture_from_disk
    _disk_capture_payload = engine_capture._disk_capture_payload
    _capture_last_from_disk = engine_capture._capture_last_from_disk
    _disk_session_for = engine_capture._disk_session_for
    capture_status = engine_capture.capture_status
    capture_last = engine_capture.capture_last
    _region_for_rid = engine_capture._region_for_rid
    capture_export = engine_capture.capture_export
    capture_sheet = engine_capture.capture_sheet
    capture_explain = engine_capture.capture_explain
    _capture_explain_llm = engine_capture._capture_explain_llm
    capture_prune = engine_capture.capture_prune
    capture_sidecar_start = engine_capture.capture_sidecar_start
    capture_sidecar_stop = engine_capture.capture_sidecar_stop
