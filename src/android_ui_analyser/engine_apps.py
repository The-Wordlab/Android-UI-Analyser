"""The app under test: launch/stop/clear and install with the launch observation that follows, feature flags and prefs, private databases, logcat and per-app log preferences (stored and effective), and the app-under-test process bookkeeping.

Engine methods for apps. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_apps.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .device import Device
from .engine_support import _ResolvedFlagsResource, logger
from .errors import DeviceError, UsageError
from .memory import _LOG_PREF_MAX, AppMemoryStore, EffectiveAppLogs, RouteStep, _id_tail
from .platforms import AppBundle, InstalledApp
from .schema import ActionResult, AnalyzeResult, AppStatusResult
from .selectors import app_elements

if TYPE_CHECKING:
    from .engine import Engine


# Package-name fragments that mark a surface AUA is never *testing* — the system launcher and
# the system UI. Scoping an action's log window to one of these is not a smaller answer, it is a
# wrong one: measured on a real device, a Back to home attached 20 lines of `LauncherStateManager`
# animation state under a field that claims to be the app's own output.
_NOT_THE_APP_UNDER_TEST = ("launcher", "systemui", "com.android.systemui", ".home")


def _clean_tags(tags: Sequence[str] | None) -> list[str]:
    """Trim a caller's tag list, dropping blanks and repeats but keeping their order."""
    out: list[str] = []
    for tag in tags or ():
        name = str(tag).strip()
        if name and name not in out:
            out.append(name)
    return out


def _tag_hides(prefix: str, tag: str) -> bool:
    """Whether a filter written as *prefix* hides a log tag called *tag*.

    Case-insensitive prefix matching, because that is what `digest_app_logs` does. Comparing tag
    lists with `==` instead is how "stop ignoring this" ends up removing nothing.
    """
    return tag.casefold().startswith(prefix.casefold())


def _same_tag_family(one: str, other: str) -> bool:
    """Whether two tag entries are about the same tag — either one is a prefix of the other."""
    return _tag_hides(one, other) or _tag_hides(other, one)


_FLAGS_VERIFY_DEADLINE_S = 2.0  # how long a flag write gets to reach the app's prefs file


_FLAGS_ENTRY_TIMEOUT_S = 3.0  # how long a pinned entry Activity gets before the default one


_FLAGS_FOREGROUND_TIMEOUT_S = 6.0  # how long the relaunched app gets to reach the foreground


# Foreground ownership can lead accessibility-window attachment briefly on a cold launch. Retry
# only while the requested package demonstrably remains foreground, and never beyond this budget.
_LAUNCH_HIERARCHY_SETTLE_S = 2.0


_LAUNCH_CONTENT_SETTLE_S = 5.0


_LAUNCH_HIERARCHY_POLL_S = 0.05


_GENERIC_LAUNCH_SHELL_RIDS = frozenset({"action_bar_root", "actionbar_root", "content"})


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


class Restart(NamedTuple):
    """Whether the app was confirmed back up, through which entry, and why not."""

    ok: bool
    activity: str | None
    error: str | None


def _launch_observation_is_transitional(observation: AnalyzeResult) -> bool:
    """Whether a launch readback contains only framework shell nodes.

        Foreground ownership is not readiness.  Android can attach the Activity window before
        the app has published any text, control, scroll surface, or app-authored container.  A
        pixel-idle sample of that frame is still a loading frame, and advertising it as a fresh
        reusable observation sends the caller into either dead ids or an unexplained extra wait.

        Keep the test deliberately semantic and app-agnostic.  A known screen, any labelled or
        interactive app node, or any non-generic app resource id is meaningful.  A canvas with no
        accessible/vision content remains unproven, which is the honest result: a quiet root node
        alone cannot establish that the rendered experience is ready.
        """
    if observation.meta.known_screen:
        return False
    own = [
        element
        for element in app_elements(observation.elements)
        if element.window not in {"system", "ime", "overlay"}
    ]
    if not own:
        return True
    for element in own:
        if (element.text or "").strip() or (element.content_desc or "").strip():
            return False
        if (
            element.clickable
            or element.focused
            or element.checkable is True
            or element.scrollable is True
            or element.long_clickable is True
        ):
            return False
        rid = (_id_tail(element.resource_id) or "").casefold()
        if rid and rid not in _GENERIC_LAUNCH_SHELL_RIDS:
            return False
    return True


def _await_meaningful_launch_observation(
    self: Engine, initial: AnalyzeResult
) -> tuple[AnalyzeResult, int]:
    """Poll one short internal window for app content after a shell-only launch frame."""
    package = initial.screen.package
    if not package:
        return initial, 0
    started = time.monotonic()
    deadline = started + _LAUNCH_CONTENT_SETTLE_S
    last = initial
    while self._launch_observation_is_transitional(last):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_LAUNCH_HIERARCHY_POLL_S, remaining))
        with contextlib.suppress(Exception):
            candidate = self.analyze(
                source="hierarchy",
                with_ocr=False,
                no_cache=True,
                record=False,
            )
            # Never replace an app-owned hierarchy with a transition owned by SystemUI or
            # another app.  Package attachment races are handled by the existing typed
            # launch-observation recovery path.
            if candidate.screen.package == package:
                last = candidate
                if not self._launch_observation_is_transitional(last):
                    self._write_cache(last)
                    break
    return last, int((time.monotonic() - started) * 1000)


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


def _await_launch_hierarchy(self: Engine, package: str) -> AnalyzeResult:
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


def _invalidate_launch_observation(self: Engine) -> None:
    """Drop every cache layer that could still describe a rejected launch tree."""
    self._invalidate_cache()
    self._prefetch.invalidate()
    self._last_analyze_elements = None
    self._last_hierarchy_hash = None
    self._last_analyze_result = None


def _adopt_recovered_launch_observation(
    self: Engine, launched: ActionResult, fresh: AnalyzeResult
) -> None:
    """Replace all fields derived from a transient launch readback with *fresh*."""
    launched.observation = fresh
    launched.observation_present = True
    self._price_elements(fresh)
    launched.next_actions = (
        self._next_actions(fresh) if self.config.output.next_actions else None
    )
    nav = list(fresh.meta.known_routes or []) + list(fresh.meta.suggested_gotos or [])
    launched.routes = nav or None
    launched.known_screen = fresh.meta.known_screen
    launched.action_diff_summary = self._compact_action_diff(fresh.meta.element_diff)
    # The original before/after comparison was computed against the rejected tree. A fresh
    # hierarchy alone cannot reconstruct it, so absence is more truthful than mixed evidence.
    launched.change = None
    launched.stale_risk = fresh.meta.stale_risk
    launched.note = "No separate analyze needed; state is in observation."


def _mark_transitional_launch_observation(self: Engine, launched: ActionResult) -> None:
    """Keep package-recovery paths from certifying a same-app shell as arrival."""
    observation = launched.observation
    if observation is None or not self._launch_observation_is_transitional(observation):
        return
    risk = (
        "the app reached the foreground, but launch produced only framework shell nodes and "
        "no meaningful app content. This observation is transitional, not arrival evidence. "
        "Use `aua wait-and-analyze --after-change` or wait for an exact destination predicate."
    )
    observation.meta.stale_risk = risk
    launched.stale_risk = risk
    launched.next_actions = None
    launched.note = (
        "The app is foreground, but its launch readback contains only framework shell nodes, "
        "so it is not a settled/reusable destination. Run `aua wait-and-analyze "
        "--after-change` or wait for an exact destination predicate; do not act on ids from "
        "this frame."
    )


def _finish_launch_content_observation(self: Engine, launched: ActionResult) -> None:
    """Handle shell-only readbacks after package attribution/recovery has completed."""
    observation = launched.observation
    already_waited = bool((launched.settle or {}).get("content_ms"))
    if (
        observation is not None
        and self._launch_observation_is_transitional(observation)
        and not already_waited
    ):
        fresh, waited_ms = self._await_meaningful_launch_observation(observation)
        self._adopt_recovered_launch_observation(launched, fresh)
        launched.settle = {**(launched.settle or {}), "content_ms": waited_ms}
    self._mark_transitional_launch_observation(launched)


def flags_set(
    self: Engine,
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
    flags = self.platform.capability("feature_flags")

    pairs = (
        flags.parse_assignments(list(assignments))
        if not isinstance(assignments, dict)
        else dict(assignments)
    )
    templates = dict(self.config.flags.templates)
    uri = flags.build_uri(package, pairs, templates)
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
    payload = flags.dump_result(
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


def _foreground_activity(self: Engine, package: str) -> str | None:
    """The activity of *package* if it is in the foreground — the one to relaunch."""
    with contextlib.suppress(Exception):
        app = self.device.current_app() or {}
        if (app.get("package") or "") == package:
            return app.get("activity") or None
    return None


def _wait_foreground(self: Engine, package: str, timeout_s: float | None = None) -> bool:
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


def _launch_entry(self: Engine, package: str, activity: str | None) -> tuple[str | None, str | None]:
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


def _restart_app(self: Engine, package: str, activity: str | None) -> Restart:
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
        self._app_process_replaced(package)
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
    self: Engine,
    package: str,
    pairs: dict[str, str],
    *,
    prefs_file: str | None,
    deadline_s: float,
) -> Any:
    """Poll the app's prefs until every requested key is there, or time runs out."""
    flags = self.platform.capability("feature_flags")

    name = prefs_file or self.config.flags.prefs_files.get(package)
    deadline = time.monotonic() + deadline_s
    while True:
        prefs = flags.read_prefs(self.device, package, pairs, prefs_file=name)
        if not prefs.verified or not (prefs.ignored or prefs.mismatched):
            return prefs
        if time.monotonic() >= deadline:
            return prefs
        time.sleep(0.25)


def flags_apply(
    self: Engine,
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
    flags = self.platform.capability("feature_flags")

    if _snapshot is None:
        app, pairs = flags.load_flags_file(path)
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


def prefs_write(
    self: Engine,
    package: str,
    file: str,
    values: Mapping[str, Any],
    *,
    relaunch: bool = True,
) -> dict[str, Any]:
    """Set preferences in one of *package*'s own preference files, then read them back.

        The state a setup flow needs is often not reachable through the UI at all — which
        backend a build talks to, whether onboarding counts as seen — and a deeplink template
        only exists when the app declares one. Even where a deeplink does exist, it returns as
        soon as the intent is delivered while the app flushes the write on a background thread,
        so a `stop_app` straight afterwards kills the process first and the preference is lost
        with every step still reporting OK. This writes the app's preference store directly on
        a debuggable build instead.

        Three calls in a fixed order, and the order is the point: the capability force-stops
        the app and snapshots the file (a live process would overwrite it from its own
        in-memory copy), the snapshot is saved and the undo journalled, and only then is
        anything written. A crash between the record and the write leaves a redundant undo; a
        crash the other way would leave an app nobody can put back.
        """
    prefs = self.platform.capability("feature_flags")

    device = self.device
    snapshot = prefs.snapshot_prefs(device, package, file)
    key = f"app_prefs:{snapshot.package}:{snapshot.file}"
    backup: Path | None = None
    # Repeated writes in one session must still undo to the state before the *first* write.
    # The ledger is idempotent on key; overwriting its deterministic backup before replacing
    # the entry made teardown restore only the immediately preceding intermediate value.
    from . import device_ledger

    for entry in device_ledger.read_ledger(device.serial):
        candidate = Path(str(entry.args.get("backup_path") or ""))
        if entry.key == key and entry.op == "restore_app_prefs" and candidate.is_file():
            backup = candidate
            break
    if backup is None:
        backup = prefs.save_prefs_backup(self.config.cache.dir, device.serial, snapshot)
    self.record_device_change(
        key=key,
        kind="app_prefs",
        op="restore_app_prefs",
        args={
            "package": snapshot.package,
            "file": snapshot.file,
            "backup_path": str(backup),
        },
        detail=f"{snapshot.package} shared_prefs/{snapshot.file} rewritten by AUA",
    )
    return prefs.write_prefs(device, snapshot, dict(values), relaunch=relaunch)


def app(
    self: Engine,
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
        # `clear_app` returns a warning (rather than raising) when the wipe itself
        # succeeded but Android's post-wipe settle barrier could not be proven within its
        # window — see `Device.clear_app`. It is durable and non-retryable either way, so
        # the launch proceeds; the warning is folded into `detail` below so the caller can
        # still see it instead of it being silently dropped.
        clear_warning: str | None = None
        with self._acting():
            if clear_state:
                clear_warning = device.clear_app(package)
            self._app_process_replaced(package)
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
            if clear_warning:
                detail = f"{detail} — {clear_warning}"
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
            ActionResult(ok=True, action="app-launch", detail=detail),
            observe,
            with_image,
            finalize=False,
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
        self._finish_launch_content_observation(launched)
        if launch_note:
            # `_observe` owns `note` when it attaches a screen, so the ambiguity warning is
            # prepended afterwards rather than passed in — it must not be silently dropped.
            launched.note = f"{launch_note} {launched.note}" if launched.note else launch_note
        return self._finalize_observed_action(launched)
    if a in ("kill", "force-stop"):
        if not package:
            raise UsageError("app kill needs a package name")
        mem = self._memory
        if mem is not None and not self._join_memory_writers(timeout_s=5.0):
            raise UsageError("memory provenance is still being finalized")
        with self._acting():
            self._app_process_replaced(package)
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
            self._app_process_replaced(package)
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
            self._app_process_replaced(package)
            clear_warning = device.clear_app(package)
        if mem is not None:
            with self._mem_lock:
                mem.clear_context(device.serial, package)
        # See the `launch --clear` branch above: a warning here means the wipe succeeded
        # but quiescence could not be proven in time — non-fatal, so it rides on `detail`
        # rather than failing an otherwise-successful, non-retryable operation.
        detail = f"{package} — {clear_warning}" if clear_warning else package
        return ActionResult(ok=True, action="app-clear", detail=detail)
    if a in ("grant", "grant-permissions", "grant_permissions"):
        if not package:
            raise UsageError("app grant needs a package name")
        device.grant_permissions(package)
        return ActionResult(ok=True, action="app-grant", detail=package)
    raise UsageError(
        f"unknown app action '{action}'",
        hint="foreground|launch|stop|kill|clear|grant|current",
    )


def app_status(self: Engine, package: str) -> AppStatusResult:
    """Report package presence/version on the device selected by AUA's lease."""

    app_id = str(package or "").strip()
    if not app_id:
        raise UsageError("app status needs a package name")
    platform = self.platform
    if not platform.supports("app.status"):
        raise DeviceError(
            f"platform '{platform.name}' cannot query installed app status",
            code="unsupported_capability",
        )
    device = self.device
    status = platform.installed_app(device, app_id)
    return AppStatusResult(
        package=status.app_id,
        installed=status.installed,
        serial=device.serial,
        version_name=status.version_name,
        version_code=status.version_code,
    )


def install_app(
    self: Engine,
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
            hint=f"Or keep the data: `aua install {path} --reinstall`.",
        )
    started = time.perf_counter()
    if pushed:
        mem = self._memory
        if mem is not None and not self._join_memory_writers(timeout_s=5.0):
            raise UsageError("memory provenance is still being finalized")
        with self._acting():
            self._app_process_replaced(app_id)
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


def database_list(self: Engine, package: str) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

    return app_database.list_databases(self.device, package)


def database_schema(
    self: Engine,
    package: str,
    database: str,
    *,
    table: str | None = None,
    restart: bool = True,
) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

    return app_database.database_schema(
        self.device,
        package,
        database,
        table=table,
        restart=restart,
    )


def database_query(
    self: Engine,
    package: str,
    database: str,
    sql: str,
    *,
    parameters: dict[str, Any] | list[Any] | None = None,
    limit: int = 100,
    timeout_ms: int = 5000,
    restart: bool = True,
    live: bool = True,
) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

    return app_database.query_database(
        self.device,
        package,
        database,
        sql,
        parameters=parameters,
        limit=limit,
        timeout_ms=timeout_ms,
        restart=restart,
        live=live,
    )


def database_execute(
    self: Engine,
    package: str,
    database: str,
    sql: str,
    *,
    parameters: dict[str, Any] | list[Any] | None = None,
    timeout_ms: int = 5000,
    restart: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

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
    self: Engine,
    package: str,
    database: str,
    *,
    restart: bool = True,
) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

    return app_database.backup_database(
        self.device,
        self.config.cache.dir,
        package,
        database,
        restart=restart,
    )


def database_backups(self: Engine, package: str, database: str) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

    return app_database.list_backups(
        self.device,
        self.config.cache.dir,
        package,
        database,
    )


def database_restore(
    self: Engine,
    package: str,
    database: str,
    backup_id: str,
    *,
    restart: bool = True,
    confirmed: bool = False,
) -> dict[str, Any]:
    app_database = self.platform.capability("app_database")

    return app_database.restore_database(
        self.device,
        self.config.cache.dir,
        package,
        database,
        backup_id,
        restart=restart,
        confirmed=confirmed,
    )


def logcat_mark(self: Engine, name: str = "default", *, clear: bool = False) -> dict[str, Any]:
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
    self: Engine,
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


# `config.logs` is one setting for every app on the host. These two are the per-app, across
# sessions half of it: what an agent learns about one app's loggers is worth keeping, and
# re-learning it every session is exactly the cost the digest exists to avoid.
def app_log_prefs(self: Engine, *, app: str | None = None) -> dict[str, Any]:
    """What this host has been told to keep or drop from *app*'s action log windows."""
    package = self._app_for_log_prefs(app)
    store = self._app_log_store()
    return {
        "ok": True,
        "action": "app-log-prefs",
        **self._app_log_prefs_view(package, store.load_log_prefs(package), store),
    }


def app_log_prefs_set(
    self: Engine,
    *,
    app: str | None = None,
    ignore_tags: Sequence[str] | None = None,
    unignore_tags: Sequence[str] | None = None,
    only_tags: Sequence[str] | None = None,
    levels: str | None = None,
    limit: int | None = None,
    per_tag: int | None = None,
    scan_lines: int | None = None,
    enabled: bool | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """Persist one app's log-window preference locally and return the effective view.

        ``ignore_tags`` and ``unignore_tags`` are the two halves of the same list, and the
        second one has to reach the built-in deny list too: that list is a guess about apps in
        general, and the app in front of you is where the guess can be wrong. Un-ignoring a tag
        nobody was ignoring is reported in ``not_ignored`` rather than answered with a silent
        success — the failure it prevents is an agent that thinks it has widened the window and
        then spends the session reading an unchanged one.

        ``only_tags=[]`` clears the allow-list; ``None`` leaves it alone. Nothing here touches
        the device, so it works with no device attached as long as *app* is named.
        """
    from .memory import AppLogPrefs

    package = self._app_for_log_prefs(app)
    store = self._app_log_store()
    changes = {
        "ignore_tags": ignore_tags,
        "unignore_tags": unignore_tags,
        "only_tags": only_tags,
        "levels": levels,
        "limit": limit,
        "per_tag": per_tag,
        "scan_lines": scan_lines,
        "enabled": enabled,
    }
    if reset:
        named = sorted(name for name, value in changes.items() if value is not None)
        if named:
            raise UsageError(
                f"reset cannot be combined with {', '.join(named)}",
                hint="Reset first, then set what you want — or drop reset.",
            )
        existed = store.forget_log_prefs(package)
        return {
            "ok": True,
            "action": "app-log-prefs-set",
            "reset": True,
            "changed": existed,
            "not_ignored": [],
            "shadowed_by_only_tags": [],
            **self._app_log_prefs_view(package, None, store),
        }

    add = _clean_tags(ignore_tags)
    drop = _clean_tags(unignore_tags)
    contradictory = sorted(set(add) & set(drop))
    if contradictory:
        raise UsageError(
            f"cannot ignore and un-ignore the same tag: {', '.join(contradictory)}",
            hint="Pick one direction per call.",
        )
    if levels is not None:
        unknown = sorted({ch for ch in levels.upper() if ch not in "VDIWEF"})
        if not levels.strip() or unknown:
            raise UsageError(
                f"levels must be a set of V D I W E F, got {levels!r}",
                hint="It is a SET, not a floor — 'DWEF' is the default, 'DIWEF' is wider.",
            )
    for name, value in (("limit", limit), ("per_tag", per_tag), ("scan_lines", scan_lines)):
        if value is None:
            continue
        # Bounded, not just positive. This is attached to every action and outlives the
        # session that set it, so an unbounded number here is a permanent tax on whoever
        # drives this app next.
        ceiling = _LOG_PREF_MAX[name]
        if not 1 <= int(value) <= ceiling:
            raise UsageError(
                f"{name} must be between 1 and {ceiling}, got {value!r}",
                hint="It is folded into EVERY action for this app, in every later session.",
            )

    prefs = store.load_log_prefs(package) or AppLogPrefs(package=package)
    keep_ignoring = list(prefs.ignore_tags)
    keep_reporting = list(prefs.keep_tags)
    for tag in add:
        # Entries are prefixes, so a broader one absorbs the narrower ones it already covers
        # and a narrower one is not worth storing beside a prefix that already hides it.
        if not any(_tag_hides(held, tag) for held in keep_ignoring):
            keep_ignoring = [held for held in keep_ignoring if not _tag_hides(tag, held)]
            keep_ignoring.append(tag)
        # Ignoring beats an earlier exemption: the last explicit instruction wins.
        keep_reporting = [held for held in keep_reporting if not _same_tag_family(held, tag)]
    not_ignored: list[str] = []
    for tag in drop:
        # Prefix-aware, both ways: un-ignoring `NetworkError` has to clear a stored `Network`
        # that hides it, and un-ignoring `Network` has to clear the narrower entries under it.
        hidden_here = [held for held in keep_ignoring if _same_tag_family(held, tag)]
        keep_ignoring = [held for held in keep_ignoring if held not in hidden_here]
        if self._log_tag_is_hidden_elsewhere(tag, package):
            if not any(_tag_hides(held, tag) for held in keep_reporting):
                keep_reporting.append(tag)
        elif not hidden_here:
            not_ignored.append(tag)

    updates: dict[str, Any] = {
        "ignore_tags": keep_ignoring,
        "keep_tags": keep_reporting,
    }
    if only_tags is not None:
        updates["only_tags"] = _clean_tags(only_tags)
    scalars: tuple[tuple[str, Any], ...] = (
        ("levels", levels.upper() if levels is not None else None),
        ("limit", limit),
        ("per_tag", per_tag),
        ("scan_lines", scan_lines),
        ("enabled", enabled),
    )
    for name, value in scalars:
        if value is not None:
            updates[name] = value

    updated = prefs.model_copy(update=updates)
    # `changed` has to mean changed. Un-ignoring a tag nobody ignored writes the same
    # document back, and reporting that as a change is how an agent concludes it has widened
    # a window it has not touched.
    differs = updated.stored_fields() != prefs.stored_fields()
    if updated.is_empty():
        # Nothing left to remember. An empty document and no document must not be different
        # states, or "reset" would have two spellings with two behaviours.
        changed = store.forget_log_prefs(package)
        stored: Any = None
    elif differs:
        store.save_log_prefs(updated)
        changed = True
        stored = updated
    else:
        changed = False
        stored = updated
    view = self._app_log_prefs_view(package, stored, store)
    # An only-list drops every tag it does not name, so an exemption outside it is stored but
    # inert. Saying so is the difference between a preference that will work later and one the
    # agent believes is working now.
    active_only = view["effective"]["only_tags"]
    shadowed = [
        tag
        for tag in (*drop, *keep_reporting)
        if active_only and not any(_tag_hides(prefix, tag) for prefix in active_only)
    ]
    return {
        "ok": True,
        "action": "app-log-prefs-set",
        "reset": False,
        "changed": changed,
        "not_ignored": not_ignored,
        "shadowed_by_only_tags": _clean_tags(shadowed),
        **view,
    }


def _app_for_log_prefs(self: Engine, app: str | None) -> str:
    """The app a preference call is about — named explicitly, or the one in front."""
    if app and app.strip():
        return app.strip()
    package = self.current_package()
    if not package:
        raise UsageError(
            "could not determine which app these log preferences are for",
            hint="Pass the app id, or attach a device so the current app can be detected.",
        )
    return package


def _app_log_store(self: Engine) -> AppMemoryStore:
    """A store for the preference document alone — its own object, on purpose.

        Three things it must not do, each of which `self._mem` would. It must not claim a memory
        session (`_memory` reads the device on first access, and a preference call has to work
        with nothing attached). It must not become the memory subsystem's store: several call
        sites read `self._mem is None` as "memory is off", so filling it in on a
        `memory.enabled: false` run would switch parts of memory back on. And it must not open
        the sqlite backend — the preference is a file in both backends, so building the sqlite
        store would create a database, and run its one-shot legacy migration, as a side effect of
        one action's log digest.
        """
    if self._log_prefs_store is None:
        self._log_prefs_store = AppMemoryStore(
            self.config.memory.model_copy(update={"backend": "json"})
        )
    return self._log_prefs_store


def _log_tag_is_hidden_elsewhere(self: Engine, tag: str, app_id: str) -> bool:
    """Whether a filter outside this app's own ignore list would still hide *tag*.

        All three of the digest's other filters count — the built-in noise list, a host's
        `logs.deny_tags`, and the derived runtime-tag rule that drops ART logging under the app's
        own truncated process name. Un-ignoring a tag any of those hides needs a recorded
        exemption, and answering "it was not being ignored" for a tag that stays invisible is the
        one wrong answer that looks exactly like the right one.
        """
    from . import logcat as logcat_mod

    denied = (*logcat_mod.DEFAULT_DENY_TAG_PREFIXES, *self.config.logs.deny_tags)
    if any(_same_tag_family(prefix, tag) for prefix in denied if prefix.strip()):
        return True
    return logcat_mod._is_runtime_tag(tag, app_id)


def _app_log_prefs_view(
    self: Engine, package: str, prefs: Any, store: AppMemoryStore
) -> dict[str, Any]:
    """The stored document, what it resolves to, and what the built-ins already hide.

        All three, because two of them are indistinguishable on their own: an empty `stored`
        with no `builtin_ignore_tags` beside it reads as "nothing is being filtered", which is
        never true.
        """
    from . import logcat as logcat_mod
    from .memory import resolve_app_log_prefs

    effective = resolve_app_log_prefs(self.config.logs, prefs)
    return {
        "package": package,
        "stored": prefs.stored_fields() if prefs is not None else {},
        "effective": effective.as_dict(),
        "builtin_ignore_tags": list(logcat_mod.DEFAULT_DENY_TAG_PREFIXES),
        "path": str(store.log_prefs_path(package)),
    }


def _could_be_app_under_test(package: str | None) -> bool:
    """Whether *package* is plausibly the app being tested, rather than the shell around it."""
    if not package:
        return False
    folded = package.casefold()
    return not any(hint in folded for hint in _NOT_THE_APP_UNDER_TEST)


def _note_app_under_test(self: Engine, package: str | None) -> None:
    """Remember which app this run is driving, so a log window can be scoped to it.

        Needed because "the package in front right now" is the wrong answer twice over: after a
        Back to home it is the launcher, and during a cold launch there is no previous package at
        all — which is exactly the window whose logs matter most.
        """
    if self._could_be_app_under_test(package):
        self._app_under_test = package


def _app_process_replaced(self: Engine, app_id: str | None) -> None:
    """Tell the adapter an app's process is gone, so per-action log scoping stays truthful.

        Launch, stop, clear, reinstall and proxy-restart all replace the process. A platform
        that caches process identity to keep `_app_logs` to one round trip would otherwise scope
        the next action's window to a dead process and read back nothing — which is the one
        wrong answer that looks exactly like the right one.
        """
    # Every caller passes the app it is launching, stopping, clearing or reinstalling — which
    # is, by construction, the app this run is testing. Recording it here rather than at each
    # site means a cold launch (no previous package at all) still knows whose logs to read.
    self._note_app_under_test(app_id)
    with contextlib.suppress(Exception):
        self.platform.forget_app_process(app_id)


def _effective_app_logs(self: Engine, app_id: str) -> EffectiveAppLogs:
    """``config.logs`` with *app_id*'s stored preference layered on top.

        One small file read per action, uncached deliberately: the store caches nothing either,
        which is what lets a warm daemon pick up a preference the CLI just wrote without a
        restart. An unreadable store falls back to the config defaults rather than costing the
        action its whole log window.
        """
    from .memory import resolve_app_log_prefs

    prefs = None
    try:
        prefs = self._app_log_store().load_log_prefs(app_id)
    except Exception as exc:  # a broken store costs a filter, never the whole window
        logger.debug("per-app log preference unavailable: %s", exc)
    return resolve_app_log_prefs(
        self.config.logs, prefs, session_fields=self._session_log_fields
    )
