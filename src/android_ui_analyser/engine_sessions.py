"""The session contract from session_start to session_finish: goal planning, phase progress and the recommended-call ranking, candidate flows, and the session review.

Engine methods for sessions. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_sessions.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import re
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from .engine_support import _GENERIC_MANUAL_MATCH_TERMS, logger
from .errors import AuaError, DeviceError, UsageError
from .memory import (
    DEFAULT_CONTEXT_ID,
    AppMap,
    RouteStep,
    arrival_destination_terms,
    is_destructive_step,
)
from .schema import AnalyzeResult
from .selectors import match_selector

if TYPE_CHECKING:
    from .engine import Engine


_ANIMATION_GOAL_RE = re.compile(
    r"\b(?:animation|animations|animated|motion|transition|transitions|easing|tween|tweening)\b",
    re.IGNORECASE,
)


def _goal_session_plan(self: Engine, goal: str, observation: AnalyzeResult) -> Any:
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
        # `ref` rather than the storage name: with flows filed per app, a shared name only
        # loads when it is qualified, and a plan may only recommend a call that runs.
        storage_name = item.get("ref")
        if not isinstance(storage_name, str) or item.get("error"):
            continue
        try:
            source = Path(str(item["path"])).resolve()
            flow = store.load_file(source)
        except Exception:
            # Isolate each artifact: one renamed/corrupt flow must not hide every valid
            # recommendation that follows it alphabetically.
            continue
        if flow.app in (None, package) and flow.context_id in (None, context_id):
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
    self: Engine,
    goal: str,
    *,
    observation: AnalyzeResult | None = None,
    contract_file: str | None = None,
    contract_yaml: str | None = None,
    artifacts_dir: str | None = None,
    evidence: str = "failures",
    junit: bool = False,
    wait_for_lease_s: float = 0,
    start_emulator: bool = True,
    headed: bool = False,
    audio: bool = False,
    animations: bool = False,
    avd: str | None = None,
    needs: list[str] | None = None,
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
    normalized_needs = (
        [str(item).strip().lower() for item in needs if str(item).strip()]
        if needs is not None
        else list(self._lease_needs or [])
    )
    animations_requested = bool(
        animations or "animations" in normalized_needs or _ANIMATION_GOAL_RE.search(goal)
    )
    # ``animations`` is a reversible session environment requirement, not a hardware
    # capability. Do not reject otherwise compatible targets for lacking a probe key. Apply
    # this even when needs came from config/engine state instead of the current CLI argument.
    self._lease_needs = [item for item in normalized_needs if item != "animations"]
    self._lease_waited_ms = 0
    emulator_started = False
    if observation is None:
        prepared = self._prepare_session_target(
            wait_for_lease_s=wait_for_lease_s,
            start_emulator=start_emulator,
            headed=headed,
            audio=audio,
            avd=avd,
            animations=animations_requested,
            package=package,
            app_will_be_installed=bool(apk),
        )
        emulator_started = bool(prepared.get("emulator_started"))
        self._lease_waited_ms = int(prepared.get("lease_waited_ms") or 0)
    installed_bundle: dict[str, Any] | None = None
    animation_backup_path: Path | None = None
    animation_change_key: str | None = None
    animations_enabled = False
    try:
        if observation is None and animations_requested:
            serial = str(prepared["serial"])
            safe_serial = serial.replace(":", "_").replace("/", "_")
            animation_backup_path = (
                Path(self.config.cache.dir).expanduser()
                / "session-devopts"
                / f"{safe_serial}-{uuid.uuid4().hex}.json"
            )
            animation_change_key = f"session_animations:{animation_backup_path.name}"
            self.record_device_change(
                key=animation_change_key,
                kind="developer_settings",
                op="restore_developer_settings",
                args={"backup_path": str(animation_backup_path)},
                detail="animation scales enabled for animation-aware session",
                serial=serial,
            )
            devopts = self.platform.capability("developer_settings")
            animation_state = devopts.anim_on(self.device.shell, animation_backup_path)
            scales = (animation_state or {}).get("anim") or {}
            animations_enabled = bool(scales) and all(
                str(value) in {"1", "1.0"} for value in scales.values()
            )
            if not animations_enabled:
                raise DeviceError(
                    "could not enable Android animations for this session",
                    hint=(
                        "AUA read the animation scales back and at least one was not 1.0. "
                        "The saved pre-session values will be restored."
                    ),
                )
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
            # ``app launch`` marks its folded hierarchy unstable when it came from a
            # one-sample/timeout/unchanged settle path.  Reusing that explicitly unstable
            # frame here makes the goal planner answer ``manual_observation`` even though
            # the immediately following hierarchy is actionable.  Session bootstrap owns the
            # launch, so pay for that one bounded authoritative read now instead of handing
            # every agent a redundant analyze.  The note carries the verdict: the derived
            # ``next_actions`` list used to double as the signal and is off by default, so
            # its absence no longer distinguishes an unstable frame from a settled one.
            if (
                observation is not None
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
        if animation_backup_path is not None and animation_backup_path.is_file():
            try:
                self.platform.capability("developer_settings").anim_restore(
                    self.device.shell, animation_backup_path
                )
                if animation_change_key is not None:
                    self.forget_device_change(animation_change_key)
            except Exception:
                # Leave the ledger entry intact: the teardown watchdog can still restore it.
                pass
        if emulator_started:
            emulator_mod = self.platform.capability("virtual_devices")

            # Tear down only the boot this session performed (`prepared` carries the
            # instance/pid the platform recorded); a bare serial can name a foreign
            # device after a provisioning collision. Drop this command's shared use
            # fence before rollback takes the exclusive stop/ownership transaction.
            self.release_device_use()
            with contextlib.suppress(Exception):
                emulator_mod.stop_spawned_instance(
                    instance=str(prepared.get("instance") or ""),
                    pid=prepared.get("pid"),
                    cache_dir=self.config.cache.dir,
                    lease_registry_dir=self._lease_registry_dir,
                    owner=getattr(self, "_lease_owner_resolved", None),
                    requested_by="session-start-rollback",
                )
            self.close()
        raise
    plan = self._goal_session_plan(goal, observed)
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
    network_backup_preexisting = False
    network_profile_preexisting = False
    if self.platform.supports("network"):
        network = self.platform.capability("network")
        network_backup_preexisting = network.backup_path(
            self.config.cache.dir, serial
        ).is_file()
    if self.platform.supports("network_profiles"):
        network_profiles = self.platform.capability("network_profiles")
        network_profile_preexisting = network_profiles.profile_path(
            self.config.cache.dir, serial
        ).is_file()
    state = create_session_state(
        self.config.cache.dir,
        goal=goal,
        serial=serial,
        owner=session_owner,
        recommended_kind=plan.recommended_call.kind,
        recommended_cli=plan.recommended_call.cli,
        network_backup_preexisting=network_backup_preexisting,
        network_profile_preexisting=network_profile_preexisting,
        emulator_started=emulator_started,
        animations_enabled=animations_enabled,
        animation_backup_path=(
            str(animation_backup_path) if animation_backup_path is not None else None
        ),
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
            *(["animation_restore"] if animation_backup_path is not None else []),
            "network_restore",
            "network_profile_restore",
            *(["owned_emulator_handoff"] if emulator_started else []),
        ],
        cleanup_call={
            "cli": "aua session finish",
            "mcp": {
                "tool": "session_finish",
                "arguments": {"session_id": state.session_id},
            },
            "reason": (
                "Run this once when finished. It restores only session-owned reversible "
                "state, releases the device lease, and returns the efficiency review. "
                "An AUA-started emulator remains warm until its lease-gated idle timeout; "
                "do not restore the network separately first."
            ),
        },
        emulator_started=emulator_started,
        animations={
            "requested": animations_requested,
            "enabled": animations_enabled,
            "source": (
                "flag"
                if animations
                else "needs"
                if "animations" in normalized_needs
                else "goal"
                if _ANIMATION_GOAL_RE.search(goal)
                else "default"
            ),
        },
        lease_waited_ms=self._lease_waited_ms,
        artifacts_dir=state.artifact_dir,
        goal_progress=progress,
    )
    # A device setting another run left behind — a proxy, a moved clock — outlives its
    # process and outlives the app, so no amount of restarting clears it. `teardown status`
    # already knew; nothing consulted it on the path an agent actually takes, so a session
    # could take every observation through somebody else's proxy without ever being told.
    from .session import inherited_device_state_warning

    inherited = inherited_device_state_warning(
        self.teardown_status().get("devices") or [], state.serial, state.owner
    )
    if inherited:
        out["warnings"] = [*(out.get("warnings") or []), inherited]
    # Session bootstrap embeds an AnalyzeResult rather than an ActionResult, so the learned
    # per-control cost has to be attached here too — the bootstrap frame is the one a manual
    # handoff picks its first control from. The derived `next_actions` list follows the same
    # opt-in as every action response: filtering `observation.elements` on `clickable` is the
    # same answer, from the observation already in this payload.
    self._price_elements(observed)
    if self.config.output.next_actions:
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
            "cli": "aua session finish",
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
    # Goal planning can consult additional evidence and leave an older/internal observation
    # in this cache slot. The session response, its numeric IDs, and the policy candidates are
    # all explicitly bound to ``observed``; make that exact returned frame authoritative before
    # inference. A concurrent replacement during model latency is still detected below by
    # ``_policy_context_is_current``.
    self._last_analyze_result = observed
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


def _phase_recommended_call(
    self: Engine,
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
            "cli": "aua session finish",
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
            "cli": "aua analyze --source hierarchy",
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
            "cli": "aua analyze --source hierarchy --no-cache",
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
                    f"aua await-and-analyze {shlex.quote(loading_predicate)} "
                    "--timeout-ms 15000 --poll-ms 200 "
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
                "aua wait-and-analyze --changed --timeout-ms 15000 --interval 150 --observe"
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
        score = _match_score(target_goal, label, exactness=semantic_label)
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
            "cli": "aua scroll-and-analyze up --pages 1 --percent 70",
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
        "cli": "No call: inspect this result's observation.elements and choose deliberately",
        "mcp": None,
        "reason": (
            "No verified route, matching flow, or unambiguous goal-labelled control is "
            "available on this frame. The result already includes the reusable observation; "
            "filter observation.elements on clickable (plus checked/scrollable for toggles "
            "and scrollers) to see what can be acted on. Another capabilities/analyze call "
            "would add no evidence."
        ),
        "executes": False,
    }


def session_mark_phase(
    self: Engine,
    phase_id: str,
    evidence: str,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Acknowledge current-phase evidence without adding a dedicated device read."""
    from .session import (
        complete_current_ui_phase_from_observation,
        mark_phase_complete,
        phase_progress,
    )
    from .session_artifacts import SessionArtifactStore, observation_evidence_id

    state = self._session_state(session_id)
    evidence_value = " ".join(evidence.strip().split())
    evidence_prefix = f"session-{state.session_id}:observation:"
    if evidence_value.startswith(evidence_prefix):
        observation: AnalyzeResult | None = None
        cached = self._read_cache()
        if cached is not None:
            cached_id = observation_evidence_id(
                state.session_id,
                cached.model_dump(mode="json"),
            )
            if cached_id == evidence_value and cached.meta.stale_risk is None:
                observation = cached
        if observation is None and state.artifact_dir:
            raw = SessionArtifactStore(state.artifact_dir).observation_for_evidence(
                evidence_value
            )
            if raw is not None:
                try:
                    observation = AnalyzeResult.model_validate(raw)
                except (TypeError, ValueError):
                    observation = None
        if observation is None:
            raise UsageError(
                "the observation evidence_id is not a reusable frame from this session",
                hint=(
                    "Use observation_contract.evidence_id from a fresh result in this "
                    "session, or provide phase-specific observable facts instead."
                ),
            )
        updated = complete_current_ui_phase_from_observation(
            self.config.cache.dir,
            state,
            observation=observation,
        )
        if updated == state:
            raise UsageError(
                "the observation evidence_id does not satisfy the current goal phase",
                hint=(
                    "Use the checkpoint's exact visible/absent state, or provide at least "
                    "two phase-specific observable facts from that result."
                ),
            )
        return {"ok": True, "goal_progress": phase_progress(updated)}
    try:
        state = mark_phase_complete(
            self.config.cache.dir,
            state,
            phase_id=phase_id,
            evidence=evidence_value,
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return {"ok": True, "goal_progress": phase_progress(state)}


def _complete_contract_phase_from_observation(
    self: Engine,
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
    self: Engine,
    session_id: str | None = None,
    *,
    observation: AnalyzeResult | None = None,
    _avoid_deeplinks: bool = False,
    _include_policy: bool = True,
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
    if current is not None and _include_policy:
        out.update(
            self._session_policy_output(
                state,
                current,
                observation,
                recommended_call=call or current.recommended_call,
            )
        )
    return out


def _session_state(self: Engine, session_id: str | None = None) -> Any:
    from . import leases
    from .session import load_session_state

    resolved = session_id or getattr(self, "_session_id", None)
    state = load_session_state(self.config.cache.dir, session_id=resolved) if resolved else None
    if state is None:
        owner = getattr(self, "_lease_owner_resolved", None)
        cached_device = getattr(self, "_device", None)
        serial = (
            getattr(self, "_lease_serial", None)
            or self.config.device.serial
            or getattr(cached_device, "serial", None)
        )
        if serial is None and owner:
            held = leases.primary_held_by(self._lease_registry_dir, owner)
            serial = held[0] if len(held) == 1 else None
        if serial is not None:
            state = load_session_state(
                self.config.cache.dir,
                serial=serial,
                owner=owner,
            )
    if state is None:
        raise UsageError(
            "no active AUA goal session",
            hint='Start one with `aua session start --goal "<goal>"`.',
        )
    self._session_id = state.session_id
    return state


def session_review(self: Engine, session_id: str | None = None) -> dict[str, Any]:
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
    review = review_session_events(state, events)
    # The rest of this review counts calls and names avoidable ones; `call_log` is the
    # per-call timeline underneath those counts — what was called, when, what came back,
    # and what it cost — so "which call spent the eight seconds" is answerable without
    # re-running the journey under an external stopwatch.
    mem = self._memory
    if mem is not None and state.serial:
        try:
            lines = mem.call_log(state.serial, since_ms=state.started_ms)
        except Exception as exc:  # a broken log must be visible, not silently absent
            logger.warning("session call log unavailable: %s", exc)
        else:
            if lines:
                review["call_log"] = lines
    return review


def _session_candidate(self: Engine, state: Any, *, name: str) -> Any:
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
    self: Engine,
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


def _session_finish_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Project closure onto the verdict, recovery, cleanup, and accounting essentials."""

    keep = (
        "ok",
        "code",
        "session_id",
        "finished",
        "terminated",
        "verdict",
        "missing_checkpoints",
        "errors",
        "hint",
        "artifacts_dir",
    )
    summary = {key: result[key] for key in keep if key in result}
    summary["summary"] = True

    progress = result.get("goal_progress")
    if isinstance(progress, dict):
        compact_progress = {
            key: progress[key]
            for key in (
                "session_id",
                "completed",
                "total",
                "done",
                "terminated",
                "status",
                "next_call",
                "checkpoint",
                "blocking_phases",
            )
            if key in progress
        }
        current = progress.get("current")
        if isinstance(current, dict):
            compact_progress["current"] = {
                key: current[key]
                for key in ("id", "objective", "kind", "status")
                if key in current
            }
        else:
            compact_progress["current"] = current
        summary["goal_progress"] = compact_progress

    cleanup_summary: list[dict[str, Any]] = []
    for row in result.get("cleanup") or []:
        if not isinstance(row, dict):
            continue
        compact_row = {key: row[key] for key in ("action", "ok") if key in row}
        detail = row.get("result")
        if isinstance(detail, dict):
            selected = {
                key: detail[key]
                for key in (
                    "detail",
                    "serial",
                    "released",
                    "retained",
                    "leased",
                    "auto_stop",
                    "idle_stop_s",
                )
                if key in detail
            }
            if selected:
                compact_row["result"] = selected
        cleanup_summary.append(compact_row)
    summary["cleanup"] = cleanup_summary

    review = result.get("review")
    if isinstance(review, dict):
        summary["review"] = {
            key: review[key]
            for key in (
                "ok",
                "run_ok",
                "failures",
                "accounting",
                "duration_ms",
                "avoidable_calls",
                "estimated_calls_saved_next_run",
                "advice",
            )
            if key in review
        }
    observation = result.get("observation")
    if isinstance(observation, dict):
        screen = observation.get("screen") or {}
        meta = observation.get("meta") or {}
        summary["observation"] = {
            "screen": {
                key: screen[key]
                for key in ("package", "activity", "width", "height")
                if key in screen
            },
            "meta": {
                key: meta[key]
                for key in ("known_screen", "fingerprint", "device_serial")
                if key in meta
            },
            "element_count": len(observation.get("elements") or []),
        }
    contract = result.get("contract_verdict")
    if isinstance(contract, dict):
        summary["contract_verdict"] = {
            key: contract[key]
            for key in ("ok", "code", "status", "checkpoint_id", "failures")
            if key in contract
        }
    candidate = result.get("candidate_flow")
    if isinstance(candidate, dict):
        summary["candidate_flow"] = {
            key: candidate[key]
            for key in ("name", "verified", "hint", "error")
            if key in candidate
        }
    session_id = result.get("session_id")
    summary["full_review_call"] = {
        "cli": f"aua session review --session-id {shlex.quote(str(session_id))}",
        "mcp": {
            "tool": "session_review",
            "arguments": {"session_id": session_id},
        },
        "reason": "Fetch the full call timeline, patterns, and command counts only if needed.",
    }
    return summary


def session_finish(
    self: Engine,
    session_id: str | None = None,
    *,
    allow_incomplete: bool = False,
    summary: bool = False,
) -> dict[str, Any]:
    """Restore only reversible state created after this session started, then review it."""
    from .session import (
        complete_current_ui_phase_from_observation,
        finish_session_state,
        phase_progress,
    )

    state = self._session_state(session_id)
    contract_verdict: dict[str, Any] | None = None
    contract_observation: AnalyzeResult | None = None
    blocking_phases = [
        phase
        for phase in state.phases
        if phase.status != "completed"
        and not (
            phase.kind == "cleanup" and phase.satisfaction == "session_cleanup"
        )
    ]
    if blocking_phases and not allow_incomplete:
        fresh = self.analyze(source="hierarchy", with_ocr=False, no_cache=True)
        contract_observation = fresh
        if state.contract is not None:
            state, contract_verdict = self._complete_contract_phase_from_observation(
                state, fresh
            )
        else:
            state = complete_current_ui_phase_from_observation(
                self.config.cache.dir,
                state,
                observation=fresh,
            )
        incomplete = [
            {
                "id": phase.id,
                "objective": phase.objective,
                "status": phase.status,
            }
            for phase in state.phases
            if phase.status != "completed"
            and not (
                phase.kind == "cleanup" and phase.satisfaction == "session_cleanup"
            )
        ]
        if incomplete:
            progress = phase_progress(state)
            result = {
                "ok": False,
                "code": (
                    "contract_incomplete"
                    if state.contract is not None
                    else "session_incomplete"
                ),
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
                    "The session is still active. Follow goal_progress.next_call, then carry "
                    "goal_progress.checkpoint on the next AUA call and retry session finish. "
                    "Use --allow-incomplete only to abandon the unfinished goal explicitly."
                ),
            }
            return self._session_finish_summary(result) if summary else result
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

    if state.animation_backup_path:
        animation_path = Path(state.animation_backup_path)
        if animation_path.is_file():

            def restore_session_animations() -> dict[str, Any]:
                restored = self.platform.capability("developer_settings").anim_restore(
                    self.device.shell, animation_path
                )
                self.forget_device_change(
                    f"session_animations:{animation_path.name}", serial=state.serial
                )
                return {"ok": True, "action": "animation-restore", **restored}

            restore("animation_restore", restore_session_animations)

    if (
        not state.network_profile_preexisting
        and self.platform.supports("network_profiles")
        and self.platform.capability("network_profiles")
        .profile_path(self.config.cache.dir, state.serial)
        .is_file()
    ):
        restore("network_profile_restore", self.network_profile_restore)
    if (
        not state.network_backup_preexisting
        and self.platform.supports("network")
        and self.platform.capability("network")
        .backup_path(self.config.cache.dir, state.serial)
        .is_file()
    ):
        restore("network_restore", self.network_restore)

    lease_owner = getattr(self, "_lease_owner_resolved", None)
    # A completed session is also the ownership boundary. Release after every device cleanup
    # action, and drop the command fence first so the lease transition can take its exclusive
    # lock. Failed cleanup deliberately keeps the lease, allowing the same process to retry.
    # A healthy AUA-started emulator is handed to the warm pool rather than stopped here; its
    # detached, lease-gated idle watchdog owns eventual retirement.
    lease_serial = getattr(self, "_lease_serial", None)
    if not errors and lease_serial == state.serial and lease_owner:
        from . import leases

        self.release_device_use()
        released = leases.release(
            self._lease_registry_dir,
            state.serial,
            owner=lease_owner,
        )
        cleanup.append(
            {
                "action": "lease_release",
                "ok": released,
                "result": {"serial": state.serial, "released": released},
            }
        )
        if released:
            self._lease_serial = None
            self._leased_serial_resolved = None
            self._lease_owner_resolved = None
            self._lease_generation_resolved = None
            if state.emulator_started:
                idle_stop_s = float(
                    getattr(self.config.teardown, "emulator_idle_stop_s", 1200.0)
                )
                cleanup.append(
                    {
                        "action": "owned_emulator_handoff",
                        "ok": True,
                        "result": {
                            "ok": True,
                            "serial": state.serial,
                            "retained": True,
                            "leased": False,
                            "auto_stop": idle_stop_s > 0,
                            "idle_stop_s": idle_stop_s,
                        },
                    }
                )
        else:
            errors.append(
                {"action": "lease_release", "message": "session lease was not released"}
            )

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
        "verdict": (
            "passed"
            if not errors and bool(progress["done"])
            else "incomplete"
            if not errors
            else "cleanup_failed"
        ),
        "goal_progress": progress,
        "cleanup": cleanup,
        "errors": errors,
        "review": review,
        "hint": (
            (
                "session completed; session-owned state was restored, the lease was "
                "released, and any AUA-started emulator was handed to the warm pool"
                if progress["done"]
                else "session terminated, cleanup completed, and the lease was released; "
                "unfinished goal phases remain incomplete"
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
    return self._session_finish_summary(result) if summary else result
