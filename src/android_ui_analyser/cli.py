"""Typer CLI — a thin adapter over :class:`~android_ui_analyser.engine.Engine` (PRD §5).

Every command builds a fresh :class:`Config` via :func:`load_config` (honouring the
global options stashed on the Typer context), constructs an :class:`Engine` (the device
connects lazily), invokes the matching engine method, and prints ``result.render(fmt)``
to **stdout**. Logs go to **stderr**; any :class:`AuaError` is emitted as a structured
object to stderr with the mapped exit code. No perception logic lives here.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import click
import typer
from typer.core import TyperCommand, TyperGroup

from . import __version__
from .assertions import parse_selector_expression
from .config import (
    Config,
    default_config_yaml,
    find_project_config,
    load_config,
    user_config_path,
)
from .engine import (
    _AWAIT_PREDICATE_HELD,
    Engine,
    _parse_await_terms,
    _parse_point,
    _regex_literal_hint,
    _safe_adopted_change,
)
from .errors import (
    AuaError,
    ConfigError,
    DaemonBusyError,
    DeviceError,
    DeviceLeasedError,
    ExitCode,
    ExpectationFailed,
    SelectorAmbiguousError,
    SelectorNotFoundError,
    UsageError,
    emit_error,
)
from .memory import (
    DEFAULT_CONTEXT_ID,
    AppMap,
    AppMemoryStore,
    KnowledgeEvidence,
    context_view,
    find_result,
    render_map,
)
from .projection import Projection, render_action_tsv, trim_observation_payload
from .reconcile import ReconciliationStore, ResearchReport, audit_map, summarize_audit
from .schema import ActionResult, AnalyzeResult, OutputFormat, publish_ids

logger = logging.getLogger("android_ui_analyser")

T = TypeVar("T")

# Sentinel produced by an optional-value flag (``--annotate``/``--emit-skill``) given bare.
ANNOTATE_DEFAULT = "\x00aua_annotate_default"
_OPTIONAL_VALUE_OPTS = {
    "--annotate",
    "--emit-skill",
    "--emit-codex-metadata",
    "--with-image",
}

_LOG_LEVELS = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class AnnotateCommand(TyperCommand):
    """A Typer command whose ``--annotate`` / ``--emit-skill`` option takes an *optional* value.

    Typer (0.26) drops Click's ``flag_value``, so a bare ``--annotate`` would error
    asking for a value. We rebuild the Click option's optional-value state after Typer
    constructs it: ``--annotate`` → :data:`ANNOTATE_DEFAULT`; ``--annotate PATH`` → PATH.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, click.Option) and _OPTIONAL_VALUE_OPTS.intersection(param.opts):
                param.is_flag = False
                param.flag_value = ANNOTATE_DEFAULT
                param._flag_needs_value = True
                param.nargs = 1


class AnalyzeCommand(AnnotateCommand):
    """An explicit action-and-read command whose post-action analysis cannot be disabled."""

    def invoke(self, ctx: click.Context) -> Any:
        # The explicit name is a contract, not a default. Keep the existing callback signatures
        # for one implementation path, but do not let a contradictory legacy flag turn
        # ``tap-and-analyze`` back into the ambiguous action-only response it exists to prevent.
        if "observe" in ctx.params:
            ctx.params["observe"] = True
        if "no_observe" in ctx.params:
            ctx.params["no_observe"] = False
        return super().invoke(ctx)


def _annotate_arg(value: str | None) -> bool | str | None:
    """Translate the raw ``--annotate`` value into the engine's ``annotate`` arg."""
    if value is None:
        return None
    if value == ANNOTATE_DEFAULT:
        return True
    return value


# --------------------------------------------------------------------------- context


@dataclass
class GlobalOpts:
    """Global options parsed by the root callback and stashed on ``ctx.obj``."""

    platform: str | None = None
    serial: str | None = None
    config: str | None = None
    format: str | None = None
    profile: str | None = None
    timeout: int | None = None
    log_level: str = "warn"
    no_cache: bool = False
    #: `--no-app-logs` / `--app-logs LEVELS`. Deliberately NOT named `--log-level*`: that flag
    #: already exists and controls AUA's own stderr verbosity, which is a different thing from
    #: which of the app's log priorities get folded into an observation.
    no_app_logs: bool = False
    app_log_levels: str | None = None
    with_image: bool = False
    observe_fields: str | None = None
    observe_meta: str | None = None
    until: str | None = None
    #: `--answers TASK_ID="value"` pairs, applied before the command runs.
    answers: tuple[str, ...] = ()
    #: `--phase-done PHASE_ID="evidence"` checkpoints, applied before the next command.
    phases_done: tuple[str, ...] = ()
    #: Machine-readable error code intentionally exercised by this one invocation.
    expect_error: str | None = None
    until_timeout: int = 30000
    until_poll: int = 500
    #: Which of the wait-tuning flags the caller actually typed. A bound default is
    #: indistinguishable from an explicit value, and the difference decides whether a timeout
    #: with no `--until` was a mistake or just the default riding along.
    explicit_wait_flags: frozenset[str] = frozenset()
    owner: str | None = None
    needs: str | None = None
    no_lease: bool = False
    _cfg: Config | None = field(default=None, repr=False)

    def cli_overrides(self) -> dict[str, Any]:
        """Translate the global flags into a config-override tree (None = unset)."""
        overrides: dict[str, Any] = {}
        device: dict[str, Any] = {}
        if self.platform is not None:
            device["platform"] = self.platform
        if self.serial is not None:
            device["serial"] = self.serial
        if device:
            overrides["device"] = device
        if self.format is not None or self.with_image:
            out: dict[str, Any] = {}
            if self.format is not None:
                out["format"] = self.format
            if self.with_image:
                out["with_image"] = True
            overrides["output"] = out
        if self.no_lease:
            overrides["lease"] = {"enabled": False}
        if self.log_level is not None:
            overrides["log_level"] = self.log_level
        if self.timeout is not None:
            overrides["timeouts"] = {"action_ms": self.timeout}
        if self.no_cache:
            overrides["cache"] = {"enabled": False}
        if self.no_app_logs or self.app_log_levels is not None:
            logs: dict[str, Any] = {}
            if self.no_app_logs:
                logs["enabled"] = False
            if self.app_log_levels is not None:
                logs["levels"] = self.app_log_levels
            overrides["logs"] = logs
        return overrides

    def load(self) -> Config:
        """Build (and memoise) the merged config for this invocation."""
        if self._cfg is None:
            self._cfg = load_config(
                explicit_path=self.config,
                profile=self.profile,
                cli_overrides=self.cli_overrides(),
            )
        return self._cfg

    def fmt(self) -> OutputFormat:
        return self.load().output.format

    def engine(self) -> Engine:
        eng = Engine(self.load())
        # Hand the lease context over before anything touches the device: the claim happens
        # on first access to `Engine.device`, and needs to know who is asking.
        eng._lease_owner = self.owner
        eng._lease_needs = _split_needs(self.needs)
        return eng


def _split_needs(raw: str | None) -> list[str]:
    """``"root, play"`` → ``["root", "play"]``. Empty when unset, so nothing gets probed."""
    if not raw:
        return []
    return [p.strip().lower() for p in str(raw).replace(" ", ",").split(",") if p.strip()]


def _opts(ctx: typer.Context) -> GlobalOpts:
    if not isinstance(ctx.obj, GlobalOpts):  # pragma: no cover - defensive
        ctx.obj = GlobalOpts()
    return ctx.obj


def _platform_capability(ctx: typer.Context, capability: str) -> Any:
    """Resolve one optional platform service without connecting to a target."""

    return _opts(ctx).engine().platform.capability(capability)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(ExitCode.OK)


def _resolve_owner_context_without_device(engine: Engine) -> None:
    """Bind owner/session identity without claiming or connecting to a device."""
    from . import leases as _leases

    engine._lease_owner_resolved = _leases.resolve_owner(  # noqa: SLF001
        getattr(engine, "_lease_owner", None)
    )
    if engine.config.device.serial:
        engine._lease_serial = engine.config.device.serial  # noqa: SLF001


class RecommendedUsageError(UsageError):
    """A safe preflight refusal with one exact non-mutating recovery call."""

    def __init__(self, message: str, *, recommended_call: str, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.recommended_call = recommended_call

    def to_dict(self) -> dict[str, object]:
        out = super().to_dict()
        error = out.get("error")
        if isinstance(error, dict):
            error["recommended_call"] = self.recommended_call
        return out


_JOURNAL_RESPONSE_CAP = 8_000


def redacted_argv(argv: list[str]) -> list[str]:
    """The shape of what the caller typed, with every value replaced.

    Raw argv cannot be journalled as-is: a parse failure can leave a secret sitting in an
    untyped positional. But recording nothing meant a reader could not see which command
    an agent had actually attempted — the name survived only inside the prose of an error
    message, so "what did it send?" was unanswerable from the journal.

    Kept: option NAMES, and the first bare token (the subcommand being attempted, which is
    what a reader needs and is never a secret). Replaced: every option value and every
    later positional.
    """
    raw = list(argv or [])
    out: list[str] = []
    seen_command = False
    expect_value = False
    for token in raw:
        if expect_value:
            out.append("<redacted>")
            expect_value = False
            continue
        if token.startswith("-"):
            name, sep, _value = token.partition("=")
            if sep:
                out.append(f"{name}=<redacted>")
            else:
                out.append(name)
                # A bare flag may or may not take a value; assume it might, so a secret
                # passed as `--token SECRET` never lands here verbatim.
                expect_value = not name.startswith("--no-")
            continue
        if not seen_command:
            out.append(token)  # the attempted command name — the whole point of this row
            seen_command = True
            continue
        out.append("<redacted>")
    return out


def _journal_cli_recovery(
    ctx: click.Context,
    *,
    event: str,
    ok: bool,
    error: AuaError | None = None,
    recommended_call: str | None = None,
    response_text: str | None = None,
    argv: list[str] | None = None,
) -> None:
    """Best-effort host-side journal row for help and pre-device CLI recovery.

    Records the shape of the request (see :func:`redacted_argv`) and the text actually
    handed back to the caller, so a human reading the dashboard can judge whether what the
    tool returned was any use. Without the response body these rows were unreadable: a help
    call showed as a content-free marker, which is indistinguishable from help returning
    nothing at all. This never resolves a lease or connects to Android.
    """
    with contextlib.suppress(Exception):
        from . import journal as journal_mod
        from . import leases

        opts = ctx.obj if isinstance(ctx.obj, GlobalOpts) else None
        cfg = opts.load() if opts is not None else load_config()
        owner = leases.resolve_owner(opts.owner if opts is not None else None)
        command_path = str(getattr(ctx, "command_path", None) or "aua")
        error_value = error.to_dict().get("error") if error is not None else None
        result: dict[str, Any] = {
            "ok": ok,
            "action": "cli-recovery",
            "event": event,
        }
        exact_call = recommended_call or getattr(error, "recommended_call", None)
        if exact_call:
            result["recommended_call"] = exact_call
        if response_text:
            text = str(response_text)
            result["response_text"] = text[:_JOURNAL_RESPONSE_CAP]
            result["response_lines"] = text.count("\n") + 1
            if len(text) > _JOURNAL_RESPONSE_CAP:
                result["response_truncated"] = True
        journal_mod.record(
            cache_dir=cfg.cache.dir,
            serial=(opts.serial if opts is not None else None) or cfg.device.serial,
            source="cli",
            cmd=f"cli_{event}",
            args={"command": command_path, "argv": redacted_argv(argv or [])},
            ok=ok,
            result=result,
            error=error_value if isinstance(error_value, dict) else None,
            extra={
                "invocation_id": globals().get("_INVOCATION_ID") or uuid.uuid4().hex,
                "cli_event": event,
            },
            owner=owner,
        )


# --------------------------------------------------------------------------- error wrap


def _run(ctx: typer.Context, fn: Callable[[Engine, OutputFormat], T]) -> T:
    """Execute ``fn`` with a built engine+format, mapping AuaError → structured exit.

    Unknown exceptions become a generic structured error on stderr with exit 1.
    """
    opts = _opts(ctx)
    try:
        global _INVOCATION_ID, _CLI_OUTPUT_FORMAT, _CLI_OBSERVE_FIELDS_SPEC
        _INVOCATION_ID = uuid.uuid4().hex
        _CLI_JOURNAL_CONTEXTS.clear()
        cfg_fmt = opts.fmt()
        _CLI_OUTPUT_FORMAT = cfg_fmt
        # A global --until is logically part of the action, so validate it before constructing
        # the engine or applying any side effect.  Previously the parser lived only inside
        # `await_predicate`, which runs after the tap/input: a typo could therefore mutate the
        # device and then fail as a usage error.  A usage error must mean zero device actions.
        if opts.until:
            standalone_until = getattr(ctx.command, "name", None) == "wait-and-analyze"
            try:
                terms = _parse_await_terms(opts.until, require_positive=False)
            except UsageError as parse_error:
                _journal_cli_recovery(
                    ctx,
                    event="usage_error",
                    ok=False,
                    error=parse_error,
                )
                raise
            if not standalone_until and not any(not term.negated for term in terms):
                recommended_call = f"aua await-and-analyze {shlex.quote(opts.until)} --observe"
                error = RecommendedUsageError(
                    "an action-bound --until needs positive arrival evidence",
                    hint=(
                        "Use the exact standalone recovery call below for an absence-only "
                        "check, or add a positive text:/rid:/desc: term to the action. The "
                        "standalone call answers `await_outcome: absence-satisfied`, not "
                        "`satisfied` — it proves what you left is gone, never what arrived, so "
                        "read its `observation` to find a positive term for next time."
                    ),
                    recommended_call=recommended_call,
                )
                _journal_cli_recovery(
                    ctx,
                    event="usage_error",
                    ok=False,
                    error=error,
                    recommended_call=recommended_call,
                )
                raise error
        engine = opts.engine()
        # The far end of the gap this turn measures: the stamp aua wrote when it last returned.
        # Here rather than in the engine's own entry points because a *caller* turn is a process
        # the agent invoked — a daemon round trip is aua's own transport and counting it would
        # halve every gap measured.
        engine.open_caller_turn()
        global _OBSERVATION_VIEW, _UNTIL, _ENGINE, _ANNOTATION_WARNINGS
        global _EMITTED_FINGERPRINT
        _EMITTED_FINGERPRINT = None
        global _EXPECTED_ERROR_CODE
        spec = opts.observe_fields
        if spec is None:
            spec = getattr(engine.config.output, "observation_fields", None)
        _CLI_OBSERVE_FIELDS_SPEC = spec
        meta_spec = opts.observe_meta
        if meta_spec is None:
            meta_spec = getattr(engine.config.output, "observation_meta", None)
        _OBSERVATION_VIEW = Projection.for_observation(spec, meta=meta_spec, fmt=cfg_fmt)
        _ENGINE = engine
        _EXPECTED_ERROR_CODE = opts.expect_error
        _ANNOTATION_WARNINGS = []
        if not opts.until:
            # `--until-timeout` only bounds a `--until`, so on its own it does nothing at all.
            # A fresh agent passed `--until-timeout 3000` believing it had a "safety bound",
            # got no wait and no `await_outcome`, and then spent four extra commands proving
            # by hand what a predicate would have asserted for it.
            dangling = sorted(opts.explicit_wait_flags)
            if dangling:
                raise UsageError(
                    f"{' and '.join(dangling)} only bounds `--until`, and no --until was given",
                    hint="Name what you are waiting for — `--until 'rid:<target>'` or "
                    "`--until 'text:<label>'` — and the timeout applies to it. Without a "
                    "predicate nothing is waited for and `await_outcome` is not reported.",
                )
        _UNTIL = (opts.until, opts.until_timeout, opts.until_poll) if opts.until else None
        # Inline map/session annotations are opportunistic bookkeeping. A stale task id or
        # mistyped checkpoint must never suppress the UI action the caller actually requested.
        # They run before routing, so bind the same owner/explicit-serial context that routing
        # will use without claiming or connecting to a device. Otherwise a fresh CLI Engine
        # looks for an anonymous active session and rejects a valid --phase-done checkpoint.
        if opts.answers or opts.phases_done:
            _resolve_owner_context_without_device(engine)
        for kind, apply, values in (
            ("answer", _apply_answers, opts.answers),
            ("phase_done", _apply_phases_done, opts.phases_done),
        ):
            try:
                apply(engine, values)
            except AuaError as err:
                annotation_error = err.to_dict().get("error")
                warning = (
                    dict(annotation_error)
                    if isinstance(annotation_error, dict)
                    else {"message": str(err)}
                )
                warning["annotation"] = kind
                _ANNOTATION_WARNINGS.append(warning)
            except (OSError, ValueError) as err:
                _ANNOTATION_WARNINGS.append(
                    {"annotation": kind, "code": "annotation_failed", "message": str(err)}
                )
        # An inline annotation can read the local Device before the real command routes to a
        # warm daemon. Its read is finished now; keeping that process's device fence while
        # waiting on the daemon would deadlock the daemon on the same physical target.
        engine.release_device_use()
        try:
            return fn(engine, cfg_fmt)
        finally:
            # Stamped even on failure: a caller thinks about an error for just as long as it
            # thinks about a screen, and skipping the stamp would make the *next* gap measure
            # two turns instead of one. The fingerprint comes from the payload this process
            # emitted, because under the warm daemon the engine that observed is elsewhere.
            try:
                engine.close_caller_turn(_EMITTED_FINGERPRINT)
            finally:
                engine.release_device_use()
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive generic path
        generic = AuaError(str(exc), code="internal_error")
        generic.exit_code = ExitCode.INTERNAL
        emit_error(generic)
        raise typer.Exit(int(ExitCode.INTERNAL)) from exc


def _apply_answers(engine: Engine, answers: tuple[str, ...]) -> None:
    """Commit `--answers TASK_ID="value"` before the command the caller actually came to run.

    The map improves as a side effect of using it: whoever navigated to a screen is the only
    one who knows what it is, and they are already issuing another command. A bad answer is a
    loud failure, never a silent one — an unknown or ambiguous task id stops the command rather
    than quietly renaming the wrong screen.
    """
    if not answers:
        return
    from .memory import AppMemoryStore
    from .reconcile import ReconciliationStore

    package = engine.current_package()
    if not package:
        raise UsageError(
            "--answers needs to know which app it is about, and no package is in the foreground",
            hint="Run it alongside a command that touches the app, e.g. `aua analyze --answers …`.",
        )
    store = ReconciliationStore(AppMemoryStore(engine.config.memory))
    pairs: list[tuple[str, str]] = []
    for pair in answers:
        task_id, sep, value = pair.partition("=")
        if not sep or not value.strip():
            raise UsageError(
                f"--answers wants TASK_ID=value, got {pair!r}",
                hint='e.g. --answers research_23cf9="Dev Tools" — the id comes from `meta.ask.id`.',
            )
        pairs.append((task_id.strip(), value.strip().strip("\"'")))
    if len(pairs) > 1:
        # Several answers on one command are one correction, not several: each transaction
        # snapshots the whole map, so N answers used to cost N copies of it. Still loud — any
        # row the batch could not use stops the command exactly as a single bad answer does.
        try:
            # Preview first: this path promises that a bad answer stops the command, so it must
            # not half-apply a batch and then complain.
            result = store.answer_many(package, pairs, agent="inline", dry_run=True)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        refused_raw = result.get("skipped_answers")
        refused = refused_raw if isinstance(refused_raw, list) else []
        if refused:
            detail = "; ".join(
                f"{row.get('task_id')}: {row.get('reason')}"
                for row in refused
                if isinstance(row, dict)
            )
            raise UsageError(
                f"--answers could not use {len(refused)} of {len(pairs)} answers",
                hint=detail,
            )
        try:
            store.answer_many(package, pairs, agent="inline")
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        return
    for task_id, value in pairs:
        try:
            store.answer(package, task_id, value, agent="inline")
        except ValueError as exc:
            raise UsageError(str(exc), hint="`meta.ask.id` on the response that asked.") from exc


def _apply_phases_done(engine: Engine, phases_done: tuple[str, ...]) -> None:
    """Checkpoint proven phase evidence on the caller's next real AUA command."""
    for pair in phases_done:
        phase_id, sep, evidence = pair.partition("=")
        if not sep or not phase_id.strip() or not evidence.strip():
            raise UsageError(
                f"--phase-done wants PHASE_ID=evidence, got {pair!r}",
                hint=(
                    "Use the current goal_progress.checkpoint, for example "
                    '--phase-done phase_2="cached thread visible and Loading absent".'
                ),
            )
        engine.session_mark_phase(
            phase_id.strip(),
            evidence.strip().strip("\"'"),
        )


# --------------------------------------------------------------------------- selectors

# `rid` is accepted alongside `id`: it is how the resource-id is spelled by the `--rid`
# flag and the selector dict, so a runner reaches for `--by rid` and must not be refused
# on one surface while `wait --by rid` quietly text-searched on another.
_BY_KINDS = {"id": "rid", "rid": "rid", "text": "text", "desc": "desc"}

# Shared selector options — the same six flags on every action, so `--rid` means one thing
# everywhere. Typer copies an OptionInfo per command, so one instance is safe to reuse.
_SEL_BY = typer.Option(
    None, "--by", help="Read the positional as: id/rid (resource-id) | text | desc."
)
_SEL_RID = typer.Option(None, "--rid", help="Target this resource-id (bare tail accepted).")
_SEL_TEXT = typer.Option(None, "--text", help="Target this label (exact first, then substring).")
_SEL_DESC = typer.Option(None, "--desc", help="Target this content-desc.")
_SEL_KEY = typer.Option(
    None,
    "--key",
    metavar="STABLE_KEY",
    help="Target this stable_key (e.g. rid:continueBtn) - the one element name "
    "that survives the frame it was observed in.",
)
_SEL_INDEX = typer.Option(None, "--index", help="Take the nth (0-based) of several matches.")
_SEL_FIRST = typer.Option(
    False, "--first", help="Take the first of several matches instead of erroring."
)


def _has_target(
    positional: str | None,
    *,
    by: str,
    rid: str | None,
    text_sel: str | None,
    desc: str | None,
) -> tuple[str, str]:
    """Resolve a ``has`` target from either spelling, as ``(value, by)``.

    ``has`` predates the one-shot selectors and took only a positional, so
    ``has --rid foo`` was a usage error while ``tap --rid foo`` worked. A guard loop written
    the obvious way (`has --rid X` before `tap --rid X`) then failed on *every* iteration
    with an empty stdout, which reads as "not on screen" rather than "you held it wrong".
    """
    chosen = [
        (value, kind, flag)
        for value, kind, flag in (
            (rid, "id", "--rid"),
            (text_sel, "text", "--text"),
            (desc, "desc", "--desc"),
        )
        if value
    ]
    if len(chosen) > 1:
        raise UsageError(
            "pass only one of --rid/--text/--desc",
            hint="They are alternative ways to name the same target.",
        )
    if chosen:
        value, kind, flag = chosen[0]
        if positional is not None:
            raise UsageError(
                "pass either a positional or one of --rid/--text/--desc, not both",
                hint=f"Drop the positional: `aua has {flag} {value}`.",
            )
        return value, kind
    if positional is None:
        raise UsageError(
            "has needs something to look for",
            hint="`aua has 'Some label'` or `aua has --rid someId`.",
        )
    return positional, by


def _selector(
    *,
    ident: str | None = None,
    by: str | None = None,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
    key: str | None = None,
    index: int | None = None,
    first: bool = False,
) -> dict[str, Any] | None:
    """Build the engine selector, or ``None`` when the caller passed a plain frame ordinal.

    Two spellings resolve to the same thing: ``--by id <positional>`` (reads like the
    existing ``has``/``wait`` flag) and the one-shot ``--rid/--text/--desc <value>``.

    A bare positional that is not an integer is a **published stable id** and becomes a key
    selector: that is what the payload hands the caller, so pasting it back must just work
    rather than being reported as a malformed number.
    """
    bare = by is None and key is None and rid is None and text is None and desc is None
    if bare and _is_stable_id(ident):
        return {"key": ident}
    if by is not None:
        kind = _BY_KINDS.get(by.lower())
        if kind is None:
            raise UsageError(f"unknown --by '{by}'", hint="Choose one of: id (or rid), text, desc.")
        if not ident:
            raise UsageError(
                f"--by {by} needs the value as the positional argument",
                hint="e.g. `aua tap-and-analyze --by id homeTabBROWSE`",
            )
        return {kind: ident, "index": index, "first": first}
    if key:
        # A stable_key names one element across frames, so it needs neither the id cache nor
        # index/first to be meaningful; ambiguity is answered with the candidates instead.
        if rid or text or desc:
            raise UsageError(
                "--key already names the element, so it cannot be combined with a selector",
                hint="Use --key on its own, or drop it and use --rid/--text/--desc.",
            )
        return {"key": key}
    if rid or text or desc:
        return {"rid": rid, "text": text, "desc": desc, "index": index, "first": first}
    return None


def _exit_unless_ok(
    result: Any, exit_code: ExitCode, *, code: str, hint: str | None = None
) -> None:
    """Turn ``ok: false`` into a non-zero exit, echoing why on stderr.

    An agent branches on the exit status, so an action that did not achieve its goal must
    never exit 0 — the JSON stays on stdout either way.
    """
    ok = result.get("ok") if isinstance(result, dict) else getattr(result, "ok", True)
    if ok:
        return
    detail = result.get("detail") if isinstance(result, dict) else getattr(result, "detail", None)
    err = AuaError(str(detail or "the action did not achieve its goal"), hint=hint, code=code)
    err.exit_code = exit_code
    emit_error(err)
    raise typer.Exit(int(exit_code))


def _element_id(ident: str | None, selector: dict[str, Any] | None) -> int | None:
    """The positional as a frame-local ordinal, or ``None`` when it is a stable id.

    Ids published in a payload are stable keys (`rid:continue_btn`, `tx:9f0c1a2b3c#2`), and a
    caller must be able to paste one straight back in. Those are not ordinals, so they become a
    key selector instead and resolve by identity — see `Engine._resolve_action_key`. An integer
    is still accepted: it is what older scripts hold, and the ordinal path still validates it
    against the frame it came from.
    """
    if selector is not None or ident is None:
        return None
    try:
        return int(ident)
    except ValueError:
        return None


def _is_stable_id(ident: str | None) -> bool:
    """Whether the positional is a published stable id rather than a frame ordinal."""
    if not ident:
        return False
    try:
        int(ident)
    except ValueError:
        return True
    return False


def _require_target(verb: str, ident: str | None, selector: dict[str, Any] | None) -> int | None:
    """Element id or selector — raise usage (exit 2) before any device connect."""
    element_id = _element_id(ident, selector)
    if element_id is None and selector is None:
        raise UsageError(
            f"{verb} needs an element id or a selector",
            hint=f"e.g. `aua {verb} rid:continue_btn` (a published id) or "
            f"`aua {verb} --rid continue_btn` or `aua {verb} --text Continue`.",
        )
    return element_id


def _rehydrate(data: dict[str, Any]) -> Any:
    """Restore the result model behind a daemon response, keyed on the payload's shape.

    The daemon answers with a plain dict, which used to fall straight through to a raw dump —
    so the model's ``render`` never ran and ``--format`` was silently ignored whenever a
    daemon happened to be serving. ``--format compact`` returned the full verbose payload
    (~2x the bytes) on exactly the calls an agent makes most.

    Shape, not a method→model registry: a registry is one more place to forget when adding a
    command, and forgetting it reintroduces the same silent divergence.
    """
    if isinstance(data.get("elements"), list) and "screen" in data:
        model: Any = AnalyzeResult
    elif "action" in data and "ok" in data:
        model = ActionResult
    else:
        return data
    try:
        return model.model_validate(data)
    except Exception:  # pragma: no cover - an unparseable payload still has to reach stdout
        return data


# Session view for a folded post-action ``observation``, resolved once per invocation by
# ``_run`` from --observe-fields / config.output.observation_fields. Module-level because
# ``_emit`` has 80-odd call sites: threading a parameter through all of them to carry a
# session-wide default would be churn, not clarity.
_OBSERVATION_VIEW: Projection | None = None
# Global --until, and the engine to run it on. Same reasoning as _OBSERVATION_VIEW: these are
# session-wide and every action command would otherwise need four more parameters.
_UNTIL: tuple[str, int, int] | None = None
_ENGINE: Any = None
_INVOCATION_ID: str | None = None
_ANNOTATION_WARNINGS: list[dict[str, Any]] = []
_EXPECTED_ERROR_CODE: str | None = None
_CLI_OUTPUT_FORMAT: OutputFormat | None = None
_CLI_OBSERVE_FIELDS_SPEC: str | None = None
# Fingerprint of the screen this process actually handed back, captured on the way out and
# stamped when the turn closes. The next call compares against it to answer "is the screen you
# have been thinking about still up" without a second device read.
_EMITTED_FINGERPRINT: str | None = None


@dataclass(frozen=True)
class _CliJournalContext:
    cache_dir: str | Path
    serial: str | None
    invocation_id: str
    detail_id: str | None
    cmd: str
    args: dict[str, Any]
    client: dict[str, Any]


_CLI_JOURNAL_CONTEXTS: dict[int, _CliJournalContext] = {}


def _projection_request(view: Projection | None) -> dict[str, Any] | None:
    if view is None:
        return None
    values: dict[str, Any] = {
        "fields": list(view.fields),
        "nonempty": view.nonempty,
        "keep_actionable": view.keep_actionable,
        "no_system": view.no_system,
        "no_ime": view.no_ime,
        "no_wrappers": view.no_wrappers,
        "where_text": list(view.where_text),
        "where_rid": list(view.where_rid),
        "clickable": view.clickable_only,
        "regions": [list(region) for region in view.regions],
        "limit": view.limit,
        "meta": list(view.meta_keys) if view.meta_keys is not None else None,
        "no_meta": view.no_meta,
        "tsv": view.tsv,
    }
    return {
        key: value
        for key, value in values.items()
        if value is not None and value is not False and value != [] and value != ()
    }


def _cli_request_context() -> dict[str, Any]:
    context: dict[str, Any] = {}
    if _CLI_OUTPUT_FORMAT is not None:
        context["format"] = _CLI_OUTPUT_FORMAT.value
    if _UNTIL is not None:
        predicate, timeout_ms, poll_ms = _UNTIL
        context.update(
            {
                "until": predicate,
                "until_timeout_ms": timeout_ms,
                "until_poll_ms": poll_ms,
            }
        )
    if _CLI_OBSERVE_FIELDS_SPEC is not None:
        context["observe_fields"] = _CLI_OBSERVE_FIELDS_SPEC
    observation_projection = _projection_request(_OBSERVATION_VIEW)
    if observation_projection is not None:
        context["observation_projection"] = observation_projection
    if _EXPECTED_ERROR_CODE:
        context["expect_error"] = _EXPECTED_ERROR_CODE
    return context


def _remember_cli_journal(
    result: Any,
    *,
    engine: Engine,
    cmd: str,
    args: dict[str, Any],
    serial: str | None,
    detail_id: str | None = None,
) -> Any:
    """Associate a routed result with the journal row that produced it."""

    if _INVOCATION_ID:
        _CLI_JOURNAL_CONTEXTS[id(result)] = _CliJournalContext(
            cache_dir=engine.config.cache.dir,
            serial=serial,
            invocation_id=_INVOCATION_ID,
            detail_id=detail_id,
            cmd=cmd,
            args=dict(args),
            client=_cli_request_context(),
        )
    return result


def _take_cli_journal(result: Any) -> _CliJournalContext | None:
    return _CLI_JOURNAL_CONTEXTS.pop(id(result), None)


def _record_cli_emitted(
    context: _CliJournalContext | None,
    result: Any,
    *,
    client: dict[str, Any] | None = None,
) -> None:
    if context is None:
        return
    with contextlib.suppress(Exception):
        from . import journal as journal_mod

        journal_mod.record_emitted_response(
            cache_dir=context.cache_dir,
            serial=context.serial,
            invocation_id=context.invocation_id,
            detail_id=context.detail_id,
            cmd=context.cmd,
            args=context.args,
            result=result,
            request_context=client if client is not None else context.client,
        )


def _set_no_meta_observation_view(fmt: OutputFormat, enabled: bool) -> None:
    """Apply the familiar analyze ``--no-meta`` flag to a folded observation."""
    if not enabled:
        return
    global _OBSERVATION_VIEW
    if _OBSERVATION_VIEW is None:
        _OBSERVATION_VIEW = Projection.parse(fmt=fmt, no_meta=True)
    else:
        _OBSERVATION_VIEW = replace(_OBSERVATION_VIEW, no_meta=True, _explicit=True)


def _project_observation(result: Any, fmt: OutputFormat) -> dict[str, Any] | None:
    """The result as a dict with its ``observation`` trimmed, or None to render normally.

    Round-trips the model's own ``render`` so the envelope stays byte-identical to the
    unprojected path — only the nested observation is rewritten.
    """
    import json

    view = _OBSERVATION_VIEW
    if view is None or not hasattr(result, "render"):
        return None
    if getattr(result, "observation", None) is None:
        return None
    try:
        data = json.loads(result.render(fmt))
    except Exception:  # pragma: no cover - defensive; fall back to the plain render
        return None
    payload = data.get("observation") if isinstance(data, dict) else None
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return None
    data = trim_observation_payload(data, view, fmt=fmt)
    return data


def _action_dict(result: Any) -> dict[str, Any] | None:
    """An action envelope as a plain dict when it carries an observation with elements."""
    import json

    if not hasattr(result, "render") or getattr(result, "observation", None) is None:
        return None
    try:
        data = json.loads(result.render(OutputFormat.json))
    except Exception:  # pragma: no cover - defensive; fall back to the plain render
        return None
    if not isinstance(data, dict):
        return None
    payload = data.get("observation")
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        return None
    return data


def _predicate_needle(predicate: str) -> str:
    """The literal a caller was hoping to see, stripped of `!`, `text:`/`rid:` and commas."""
    try:
        return _parse_await_terms(predicate)[0].value
    except (AuaError, IndexError):  # preflight normally makes this unreachable
        first = predicate.split(",")[0].strip().lstrip("!")
        _, _, value = first.partition(":")
        return (value or first).strip().strip("'\"")


def _await_timeout_note(predicate: str, timeout_ms: int, result: Any) -> str:
    """Why a `--until` ran out, and what was on screen instead.

    A timed-out predicate and a broken app are indistinguishable from `await_outcome: timeout`
    alone. Run 8 (2026-08-10) asked for `text:No results` on a screen reading "No apps found" and
    `text:Sign` on one headed "Create your account": both actions had in fact landed, and both
    spent the full 30s default before reporting a timeout the caller then had to diagnose by eye.
    That is 60s of a 165s run. `nearest_elements` already answers the same question for a selector
    that matched nothing, so answer it here too.
    """
    note = (
        f"the action landed; `--until '{predicate}'` is what ran out after {timeout_ms}ms, so this "
        "is the predicate, not the app"
    )
    regex_hint = _regex_literal_hint(predicate)
    if regex_hint:
        note += ". " + regex_hint
    observation = getattr(result, "observation", None)
    elements = getattr(observation, "elements", None)
    if not elements:
        return note + ". Re-read the screen to find the exact label."
    from .selectors import element_digest, nearest_elements

    near = nearest_elements(elements, _predicate_needle(predicate), limit=3)
    if near:
        note += ". Closest on screen: " + " | ".join(element_digest(el) for el in near)
    return (
        note + ". Match the exact label, or wait for the element you will act on "
        "(`--until 'rid:<target>'`); the budget is `--until-timeout`, not `--timeout`."
    )


def _await_until(result: Any, *, journal_privacy_cmd: str | None = None) -> Any:
    """Honour a global ``--until``: wait for the predicate, then adopt *that* screen.

    The post-action settle can only ever wait ~1.1s (``_await_post_action_ready``), stretched
    to at most 1.6s by ``SettleProfiles``. A slower destination can therefore make the folded
    observation report "nothing changed" on a tap that has in fact landed. An agent cannot tell
    "no effect" from "not yet" from that, so it
    stopped trusting the observation and hand-rolled ``wait`` + ``analyze`` after every tap.

    A caller-supplied predicate is what resolves the ambiguity, so the budget comes from the
    predicate rather than the settle default. ``await_outcome`` says whether it was satisfied,
    the activity changed, the budget timed out, or an action reached a stable non-loading screen
    that does not contain the requested positive term (``settled-unmet``).
    """
    if _UNTIL is None or _ENGINE is None:
        return result
    # `observation_present` is the action contract's marker — set on both `_observe` branches
    # and absent everywhere else, so it distinguishes an action response from `devices`/`doctor`.
    if getattr(result, "observation_present", None) is None:
        return result
    if not getattr(result, "ok", False):
        return result  # a failed action produced no transition worth waiting on
    predicate, timeout_ms, poll_ms = _UNTIL
    try:
        awaited = _route(
            _ENGINE,
            "await_predicate",
            predicate=predicate,
            timeout_ms=timeout_ms,
            poll_ms=poll_ms,
            observe=True,
            adopt_action=True,
            _journal_privacy_cmd=journal_privacy_cmd,
        )
        # This wait is folded into the action response below; it is journaled as its own
        # internal row but is not itself emitted by the CLI.
        _take_cli_journal(awaited)
    except AuaError:
        raise
    except Exception:  # pragma: no cover - the action already happened; never lose its result
        return result
    if isinstance(awaited, dict):
        awaited = _rehydrate(awaited)
    previous_change = getattr(result, "change", None)
    for attr in (
        "await_outcome",
        "await_terms",
        "arrival_mismatch",
        "elapsed_ms",
        "observation",
        "observation_present",
        "known_screen",
        "stable_elements",
        "action_diff_summary",
        "change",
        "next_actions",
        "routes",
        "note",
    ):
        # These all describe the adopted screen. Assign even ``None`` so guidance from the
        # action's early readback cannot survive when the evidence-based re-read has none.
        with contextlib.suppress(Exception):
            value = getattr(awaited, attr, None)
            if attr == "change":
                value = _safe_adopted_change(previous_change, value)
            setattr(result, attr, value)
    if getattr(result, "await_outcome", None) == "timeout":
        with contextlib.suppress(Exception):
            result.note = _await_timeout_note(predicate, timeout_ms, result)
    # The action's own settle-derived caveat described a screen we have since re-read. An
    # absence-only predicate re-read it just as thoroughly; what it cannot prove is arrival,
    # which `await_outcome` reports and `stale_risk` was never about. (Action-bound `--until`
    # still requires a positive term, so this only matters if that ever changes.)
    if getattr(result, "await_outcome", None) in _AWAIT_PREDICATE_HELD:
        with contextlib.suppress(Exception):
            result.stale_risk = None
        detail = getattr(result, "detail", None)
        if isinstance(detail, str) and "stale_risk" in detail:
            cleaned = detail.replace("stale_risk", "").strip()
            with contextlib.suppress(Exception):
                result.detail = cleaned or None
    return result


def _emit(
    result: Any,
    fmt: OutputFormat,
    *,
    _journal_context: _CliJournalContext | None = None,
) -> None:
    """Render a pydantic result (``.render``) or a plain dict (daemon path) to stdout."""
    journal_context = _journal_context or _take_cli_journal(result)
    if isinstance(result, dict):
        result = _rehydrate(result)
    result = _await_until(
        result,
        journal_privacy_cmd=journal_context.cmd if journal_context is not None else None,
    )
    result = _caller_turn(result)
    projected = _project_observation(result, fmt)
    if _ANNOTATION_WARNINGS:
        if projected is None:
            import json

            if isinstance(result, dict):
                projected = dict(result)
            elif hasattr(result, "render"):
                with contextlib.suppress(Exception):
                    rendered = json.loads(result.render(OutputFormat.json))
                    if isinstance(rendered, dict):
                        projected = rendered
        if projected is not None:
            projected["annotation_warnings"] = list(_ANNOTATION_WARNINGS)
    if fmt is OutputFormat.tsv:
        payload = projected if projected is not None else _action_dict(result)
        if payload is not None:
            journal_payload = dict(payload)
            if projected is None:
                journal_payload = trim_observation_payload(
                    journal_payload,
                    _OBSERVATION_VIEW,
                    fmt=OutputFormat.json,
                )
            _record_cli_emitted(journal_context, journal_payload)
            typer.echo(render_action_tsv(payload, _OBSERVATION_VIEW))
            return
    if projected is not None:
        _record_cli_emitted(journal_context, projected)
        _echo_json(projected, fmt)
        return
    if hasattr(result, "render"):
        rendered = result.render(fmt)
        emitted: Any = result
        with contextlib.suppress(Exception):
            import json

            emitted = json.loads(rendered)
        _record_cli_emitted(journal_context, emitted)
        typer.echo(rendered)
        return
    _record_cli_emitted(journal_context, result)
    _echo_json(result, fmt)


def _caller_turn(result: Any) -> Any:
    """Attach the caller-latency facts measured by this CLI process.

    Here as well as in `decorate_result` because the warm daemon decorates its own responses,
    in a process that never saw the caller and so has no turn to report. The turn is opened and
    stamped out in this one, which makes this the only place that can answer for a routed call.
    """
    global _EMITTED_FINGERPRINT
    if _ENGINE is None:
        return result
    from .coaching import attach_caller_turn, emitted_fingerprint

    with contextlib.suppress(Exception):
        _EMITTED_FINGERPRINT = emitted_fingerprint(result) or _EMITTED_FINGERPRINT
    with contextlib.suppress(Exception):
        return attach_caller_turn(_ENGINE, result)
    return result


def _echo_json(data: Any, fmt: OutputFormat) -> None:
    import json

    indent = 2 if fmt is OutputFormat.pretty else None
    sep = None if indent else (",", ":")
    typer.echo(json.dumps(data, indent=indent, separators=sep, ensure_ascii=False))


def _same_caller(engine: Engine, previous: dict[str, Any]) -> bool:
    """Was *previous* run by whoever is running now, and recently enough to be "right after"?

    The journal is per-device, and running several agents against one host is a supported
    setup, so every one of them appends here. Measured 2026-08-10: an agent's very first
    `analyze` was told it was a "redundant analyze right after wait" — the `wait` belonged to
    a different process, minutes earlier. It reported the warning as misleading and guessed
    at "prior/shared AUA session state". A lint that accuses you of someone else's command
    teaches you to ignore it.

    The owner is the real discriminator; the age check covers entries written before the
    journal carried one, where a gap this large cannot be the "immediately after" the message
    claims. `pid` is no help — daemon-routed commands all carry the daemon's.
    """
    owner = getattr(engine, "_lease_owner_resolved", None)
    prev_owner = previous.get("owner")
    if owner and prev_owner:
        return bool(owner == prev_owner)
    prev_ms = previous.get("ts_ms")
    if not isinstance(prev_ms, (int, float)):
        return True
    return (time.time() * 1000.0 - float(prev_ms)) <= _SAME_TURN_MS


# One agent turn: the model reads the last response, decides, and issues the next command.
# Wide enough for a slow model to still be linted, far short of a gap between sessions.
_SAME_TURN_MS = 120_000


def _journal_serial_without_device(engine: Engine) -> str | None:
    """Return this caller's selected serial without connecting to the target.

    Soft lints run around daemon-routed commands.  Reading ``engine.device`` here creates a
    second local device user after the daemon has already produced the real response; that
    process can then wait behind the daemon's capture/command fence merely to print optional
    advice.  Routing has already resolved the lease into ``_lease_serial`` and the config, so
    either value is enough to choose the per-device journal without opening the device.
    """
    leased = getattr(engine, "_lease_serial", None)
    if isinstance(leased, str) and leased:
        return leased
    configured = getattr(getattr(engine.config, "device", None), "serial", None)
    return configured if isinstance(configured, str) and configured else None


def _warn_if_redundant_analyze(engine: Engine, args: dict[str, Any] | None = None) -> None:
    """Soft lint: `analyze` immediately after an observed action usually re-reads the same state."""
    if args is not None and args.get("cmd") != "analyze":
        return
    cfg = engine.config
    serial = _journal_serial_without_device(engine)
    try:
        from . import journal as journal_mod

        events = journal_mod.read_since(cfg.cache.dir, serial, limit=4)
    except Exception:  # pragma: no cover - best effort
        return
    if len(events) < 2:
        return
    latest, previous = events[-1], events[-2]
    if latest.get("cmd") != "analyze" or not previous.get("ok"):
        return
    if not _same_caller(engine, previous):
        return
    prev = previous.get("result")
    if not isinstance(prev, dict):
        return
    if not prev.get("observation"):
        return
    action = prev.get("action")
    if not isinstance(action, str):
        action = "session start" if previous.get("cmd") == "session_start" else None
    if action is None:
        return
    # If the user already asked for an intentionally different view (query/source), do not warn.
    latest_args = latest.get("args") or {}
    if latest_args.get("source") == "vision" or latest_args.get("query"):
        return
    if latest_args.get("with_ocr") is not None or latest_args.get("fields"):
        return
    logger.warning(
        "redundant analyze right after %s: that action already returned `observation` (id "
        "space is already in the action response). Prefer using the previous `observation` and "
        "running analyze only when you need a different view. If you are re-reading because the "
        "screen had not settled yet, do not sleep-then-analyze: re-run the action with "
        "`--until 'text:<label>,!text:Loading'`, which waits and returns the "
        "settled screen in the same call; for network-driven content use "
        "`aua wait-and-analyze --after-change`.",
        action,
    )


#: Journal `cmd` values that mean "the previous call was itself a wait". A global `--until`
#: records `await_predicate`; `wait-and-analyze` records its own internals.
_WAIT_COMMANDS = frozenset({"await_predicate", "wait_stable", "wait_changed", "wait_idle"})


def _warn_if_wait_could_have_been_until(engine: Engine, waited_for: str | None) -> None:
    """Soft lint: a settle-wait straight after an action is a `--until` the caller did not know.

    Measured on a fresh agent (2026-08-10): it typed, then ran `wait-and-analyze --after-change`
    to let the results land — 1851ms + 3762ms across two calls, where folding the same wait into
    `input-and-analyze --until 'text:No apps found'` took 2015ms in one. `--after-change` cannot
    do better: with no predicate it has to observe the screen go quiet, while `--until` returns
    the moment the thing arrives.

    It reached for `--after-change` because nothing had told it `--until` exists — the
    redundant-analyze lint only fires on `analyze`, and `wait-and-analyze --help` documents
    `--for`/`--for-stable`/`--changed` but not the global flag that replaces them here.
    """
    cfg = engine.config
    serial = _journal_serial_without_device(engine)
    try:
        from . import journal as journal_mod

        events = journal_mod.read_since(cfg.cache.dir, serial, limit=4)
    except Exception:  # pragma: no cover - best effort
        return
    # Unlike the redundant-analyze lint, this runs BEFORE its own command is journaled, so the
    # call being followed is the newest entry, not the one behind it.
    if not events:
        return
    previous = events[-1]
    if not previous.get("ok"):
        return
    if not _same_caller(engine, previous):
        return
    # A global `--until` is journaled as its own `await_predicate` entry, so the newest entry
    # after `tap-and-analyze --until X` is the await, not the tap. `await_outcome` never reaches
    # the journal at all — it is attached to the emitted result — so the *entry kind* is the only
    # usable signal that the caller already waited.
    if previous.get("cmd") in _WAIT_COMMANDS:
        logger.warning(
            "the call before this already waited%s — you were handed the settled screen. Act on "
            "that observation instead of re-reading it. When you do need to wait, name the "
            "element you are about to act on (`--until 'rid:<target>'`), which returns as soon as "
            "that element exists; a screen-wide predicate like `!text:Loading` waits for "
            "EVERYTHING on the page — measured 25.3s on a streaming feed against 2.3s for the "
            "element that was already there.",
            " with `--until`" if previous.get("cmd") == "await_predicate" else "",
        )
        return
    prev = previous.get("result")
    if not isinstance(prev, dict) or not prev.get("observation"):
        return
    action = prev.get("action")
    if not isinstance(action, str):
        return
    predicate = f"text:{waited_for}" if waited_for else "rid:<the element you will act on next>"
    logger.warning(
        "this wait follows `%s`, which already observed the screen. If you know what you are "
        "waiting for, pass it to the action instead: `%s ... --until '%s'` waits and returns the "
        "settled screen in ONE call, and reports `await_outcome` so you can tell arrived from "
        "timed-out. Name the element you are about to act on rather than the whole screen: a "
        "predicate-less settle wait, or a screen-wide one like `!text:Loading`, waits for every "
        "last thing on the page.",
        action,
        f"{action}-and-analyze" if not action.endswith("-and-analyze") else action,
        predicate,
    )


def _analyze_payload(result: Any) -> dict[str, Any] | None:
    """The full (untrimmed) dict form of an analyze result, whatever produced it.

    A projection must read fields the requested ``--format`` may have trimmed, and must
    work identically for the in-process pydantic result and the daemon's dict response.
    """
    if hasattr(result, "model_dump"):
        data = result.model_dump(mode="json")
    elif isinstance(result, dict):
        data = result
    else:  # pragma: no cover - defensive
        return None
    if not isinstance(data.get("elements"), list):
        return None
    # A projection is an output path like any other, so it publishes stable ids too. Without
    # this, `--fields` and `--format tsv` printed ordinals while plain `--format json` printed
    # identities — the same screen, two incompatible id spaces, chosen by a formatting flag.
    return publish_ids(data)


def _emit_analyze(result: Any, fmt: OutputFormat, view: Projection) -> None:
    """Emit an analyze result, through *view* when it asked for anything."""
    journal_context = _take_cli_journal(result)
    projection = _projection_request(view) if view.active else None
    if journal_context is not None and projection is not None:
        journal_context = replace(
            journal_context,
            client={**journal_context.client, "projection": projection},
        )
    result = _caller_turn(result)
    payload = _analyze_payload(result) if view.active else None
    if payload is None:
        _emit(result, fmt, _journal_context=journal_context)
        return
    if view.tsv:
        _record_cli_emitted(
            journal_context,
            view.apply(payload, fmt=OutputFormat.json),
        )
        typer.echo(view.render_tsv(payload))
        return
    emitted = view.apply(payload, fmt=fmt)
    _record_cli_emitted(journal_context, emitted)
    _echo_json(emitted, fmt)


# --------------------------------------------------------------------------- daemon route


def _warm(engine: Engine) -> None:
    """Force the lazy device connection so the engine's analyze-cache key (derived from
    the connected serial) matches what a prior ``analyze`` wrote. Action/inspect
    commands resolve cached element ids and need a device anyway, so this is free.
    """
    _ = engine.device


# Engine method name → daemon command name (they differ only for ``input``).
_DAEMON_CMD = {"input_text": "input"}
_HOST_ONLY_ROUTE_METHODS = frozenset({"flow_delete"})
_CAPTURE_READ_METHODS = frozenset(
    {"capture_status", "capture_last", "capture_export", "capture_explain"}
)

# Methods with NO durable representation outside the daemon process. For these an in-process
# answer is not a slower answer, it is a wrong one, so a daemon that cannot be made current has
# to produce an error rather than a silent downgrade.
#
# The capture *reads* used to be in here on the theory that their state lived only in the
# daemon. It does not: ``CaptureBuffer`` appends every kept frame to ``index.jsonl`` before the
# call that produced it returns, so ``capture_status``/``capture_last``/``capture_export``/
# ``capture_explain`` can be answered from disk by any process, labelled ``source:
# "disk-index"``. Listing them here blocked the restart that would have made the daemon current
# *and* blocked the fallback that could have answered anyway — the caller got neither.
#
# What is left genuinely has no file to read. A job's worker is a thread inside the daemon, and
# ``capture_on``/``capture_off``/``capture_prune`` mutate that process's sampler: run
# in-process they would start or stop a buffer inside a CLI that is about to exit, and report
# success for something that never happened.
_DAEMON_ONLY_METHODS = frozenset(
    {
        "capture_on",
        "capture_off",
        "capture_prune",
        "job_start",
        "job_status",
        "job_wait",
        "job_cancel",
        "job_list",
    }
)


def _daemon_error(err: dict[str, Any]) -> AuaError:
    """Reconstruct an :class:`AuaError` (with the right exit code) from a daemon error."""
    code = err.get("code", "error")
    message = err.get("message", "daemon error")
    hint = err.get("hint")
    mapping: dict[str, type[AuaError]] = {
        "usage": UsageError,
        "device": DeviceError,
        "config": ConfigError,
        "selector_not_found": SelectorNotFoundError,
        "selector_ambiguous": SelectorAmbiguousError,
        "expectation_failed": ExpectationFailed,
    }
    if code in {
        "mic_delivery_uncertain",
        "mic_delivered_release_failed",
        "mic_toggle_start_uncertain",
        "mic_toggle_stop_uncertain",
    }:
        from .mic import (
            MicDeliveredReleaseError,
            MicDeliveryUncertainError,
            MicToggleStartUncertainError,
            MicToggleStopUncertainError,
        )

        result = err.get("result")
        raw_followups = err.get("followup_errors")
        followups = (
            [dict(item) for item in raw_followups if isinstance(item, dict)]
            if isinstance(raw_followups, list)
            else None
        )
        error_type = {
            "mic_delivery_uncertain": MicDeliveryUncertainError,
            "mic_delivered_release_failed": MicDeliveredReleaseError,
            "mic_toggle_start_uncertain": MicToggleStartUncertainError,
            "mic_toggle_stop_uncertain": MicToggleStopUncertainError,
        }[code]
        return error_type(
            message,
            hint=hint,
            result=result if isinstance(result, dict) else None,
            followup_errors=followups,
        )
    device_error_codes = {
        "app_clear_unsettled",
        "launch_observation_mismatch",
        "mic_endpoint_missing",
        "mic_repeat_unsafe",
        "mic_guard_unavailable",
        "mic_injection_precondition",
        "mic_injection_rejected",
        "mic_endpoint_auth_failed",
        "mic_injection_timeout",
        "mic_emulator_unavailable",
        "mic_injection_failed",
        "mic_speech_failed",
        "mic_toggle_owner_unknown",
        "mic_toggle_owner_changed",
        "tap_delivery_uncertain",
        "single_tap_unsupported",
    }
    if code in device_error_codes:
        return DeviceError(message, hint=hint, code=code)
    if code in mapping:
        return mapping[code](message, hint=hint)
    if code.startswith("provider"):
        out = AuaError(message, hint=hint, code=code)
        out.exit_code = ExitCode.PROVIDER
        return out
    if code == "wait_timeout":
        out = AuaError(message, hint=hint, code=code)
        out.exit_code = ExitCode.DEVICE
        return out
    return AuaError(message, hint=hint, code=code)


def _capture_session_live(daemon_mod: Any, cfg: Any) -> bool:
    """True only when the daemon confirms it is recording. Any doubt answers False."""
    try:
        with daemon_mod.DaemonClient(daemon_mod.socket_path(cfg), timeout=2.0) as client:
            resp = client.call("capture_status")
    except Exception:  # pragma: no cover - a daemon too old to answer has no buffer to lose
        return False
    if not resp.get("ok"):
        return False
    return bool((resp.get("result") or {}).get("running"))


def _daemon_is_mid_job(daemon_mod: Any, cfg: Any) -> bool:
    """Whether the daemon at this config's socket still has a background job in flight.

    Answers False when it cannot tell. A daemon writes its pidfile before it accepts anything,
    so no pid means no job can be attributed to it; treating "unknown" as busy instead would
    make one unreadable file permanently reinstate the tax this check exists to bound.
    """
    with contextlib.suppress(Exception):
        pid, _ = daemon_mod.read_pidfile(daemon_mod.socket_path(cfg) + ".pid")
        if pid is None:
            return False
        from . import jobs as jobs_mod

        return bool(jobs_mod.worker_is_busy(cfg.cache.dir, pid))
    return False


def _restart_hint(daemon_mod: Any, cfg: Any) -> str:
    """A restart command that reaches the daemon actually in play.

    `socket_path` appends the serial when one is known, so a bare `aua daemon stop` signals
    `daemon.sock` while the daemon under discussion lives on `daemon.sock.<serial>`. The hint
    used to omit the serial, which made following it a no-op: `stop` reported success, `start`
    adopted the serial-less socket, and the next pinned call routed straight back to the same
    stale daemon. Resolve the serial into the text so the advice is executable as printed.
    """
    serial = None
    with contextlib.suppress(Exception):
        serial = daemon_mod.effective_serial(cfg)
    pin = f"--serial {serial} " if serial else ""
    return f"Restart it: `aua {pin}daemon stop && aua {pin}daemon start`."


# A failed restart costs `stop` (up to 5s) plus `start` (up to 5s) and changes nothing. Persist
# the retry deadline beside the socket: every CLI command is a new process, so a module dict
# forgot the cooldown before the next invocation could benefit from it.
_SKEW_RESTART_COOLDOWN_S = 30.0


def _skew_backoff_path(socket: str) -> Path:
    return Path(socket + ".restart-backoff")


def _skew_backoff_active(socket: str, now: float) -> bool:
    try:
        return float(_skew_backoff_path(socket).read_text(encoding="utf-8")) > now
    except (OSError, ValueError):
        return False


def _replace_skewed_daemon(daemon_mod: Any, cfg: Any, ver: str) -> bool:
    """Restart a daemon serving different code, so calls keep the warm path.

    Skew is not a reason to degrade silently. The in-process fallback pays a device connect
    on every call (~6x slower), and the warning saying so goes to stderr, where a caller
    reading stdout never sees it. Restarting pays one connect, once. In an editable install the
    source fingerprint moves on every edit, so skew is the resting state during development,
    not an exception: measured 2026-08-19, thirteen consecutive calls each warned and each paid
    the full attach because this function always refused.

    It refused because it also asked whether the capture ring buffer was recording, and
    `capture.enabled` defaults to true, so `serve` starts one on every warm daemon — the guard
    matched every daemon there is. It was protecting frames nobody could reach either: under
    skew every `capture_*` command is in `_DAEMON_ONLY_METHODS` and raises rather than
    answering, so those frames stay unreadable until the daemon is replaced. `_route` therefore
    lets that refusal stand for the reads themselves and restarts for everything else.

    A live background job is the one thing a restart genuinely destroys. Its worker is a thread
    inside the daemon, and the daemon can still answer pings while it runs — every other kind of
    in-flight call blocks the serialized accept loop, which makes the version ping unanswerable
    and stops skew from being detected at all. So that case is checked explicitly and yields to
    the slow path.

    Scope is per device throughout: `socket_path` appends the serial `_route` resolved from the
    lease, so a concurrent agent's daemon on another emulator is never signalled.
    """
    if _daemon_is_mid_job(daemon_mod, cfg):
        return False
    socket = ""
    with contextlib.suppress(Exception):
        socket = str(daemon_mod.socket_path(cfg))
    now = time.time()
    if socket and _skew_backoff_active(socket, now):
        return False
    with contextlib.suppress(Exception):
        daemon_mod.stop(cfg)
        daemon_mod.start(cfg, serial=cfg.device.serial)
        if daemon_mod.running_version(cfg) == daemon_mod._aua_version():
            with contextlib.suppress(OSError):
                _skew_backoff_path(socket).unlink()
            logger.info("replaced a daemon running aua %s with %s", ver, daemon_mod._aua_version())
            return True
    if socket:
        from .atomic import atomic_write_text

        with contextlib.suppress(Exception):
            atomic_write_text(_skew_backoff_path(socket), str(now + _SKEW_RESTART_COOLDOWN_S))
        logger.info(
            "a skew restart of %s did not take; not retrying for %.0fs",
            socket,
            _SKEW_RESTART_COOLDOWN_S,
        )
    return False


def _replace_policy_mismatched_daemon(daemon_mod: Any, cfg: Any, expected: str) -> bool:
    """Restart stale policy state only when doing so cannot discard a live capture."""
    if _capture_session_live(daemon_mod, cfg):
        return False
    with contextlib.suppress(Exception):
        daemon_mod.stop(cfg)
        daemon_mod.start(cfg, serial=cfg.device.serial)
        return daemon_mod.running_policy_fingerprint(cfg) == expected
    return False


def _replace_runtime_mismatched_daemon(daemon_mod: Any, cfg: Any, expected: str) -> bool:
    """Replace a per-device daemon that was born in another run's cache authority."""

    if _daemon_is_mid_job(daemon_mod, cfg):
        return False
    with contextlib.suppress(Exception):
        daemon_mod.stop(cfg)
        daemon_mod.start(cfg, serial=cfg.device.serial)
        return daemon_mod.running_runtime_fingerprint(cfg) == expected
    return False


def _route(engine: Engine, method: str, **kwargs: Any) -> Any:
    """Run an engine call through the daemon when one is live, else in-process.

    Best-effort: any failure connecting to / importing the daemon falls back to the
    in-process engine. A structured error returned by the daemon is raised as the
    matching :class:`AuaError` (it is the answer, so it must not be swallowed).
    """
    cfg = engine.config
    journal_privacy_cmd_raw = kwargs.pop("_journal_privacy_cmd", None)
    journal_privacy_cmd = (
        str(journal_privacy_cmd_raw) if journal_privacy_cmd_raw else None
    )
    # Resolve the process-bound lease *before* choosing a daemon socket.  A daemon is bound
    # to one physical device, so routing an unpinned call through the unsuffixed socket and
    # leasing only inside that daemon can make the lease registry say "emulator-5558" while
    # the already-connected Engine is still driving emulator-5554.  Besides crossing agent
    # boundaries, that also makes jobs/session state appear to vanish on the next invocation.
    # Lease discovery does not connect to uiautomator2; it is the cheap ownership decision
    # every transport must share.
    host_only = method in _HOST_ONLY_ROUTE_METHODS
    # Session bootstrap owns pool discovery, optional provisioning, and the first lease claim.
    # Run that boundary in this caller process so the lease is bound to the agent rather than a
    # detached daemon, then let ordinary calls use the selected per-device warm daemon.
    bootstrap_session = method == "session_start"
    if host_only:
        # Host-only flow metadata commands still belong to the calling agent's active goal
        # session. Resolve identity and the explicit serial without claiming/connecting to the
        # device, so two idempotent deletes are both accounted for instead of the second one
        # becoming an anonymous journal row.
        _resolve_owner_context_without_device(engine)
    elif not bootstrap_session:
        lease_serial = getattr(engine, "_lease_serial", None)
        lease_device = getattr(engine, "_lease_device", None)
        if not lease_serial and callable(lease_device):
            lease_serial = lease_device()
        if lease_serial:
            cfg.device.serial = lease_serial
    if getattr(cfg.daemon, "enabled", False) and not host_only and not bootstrap_session:
        try:
            from . import daemon as daemon_mod

            # Auto-start the warm daemon on first use so cold CLI calls don't pay
            # Python+u2 connect on every subsequent invocation.
            if getattr(cfg.perf, "auto_daemon", True) and not daemon_mod.is_running(cfg):
                with contextlib.suppress(Exception):
                    daemon_mod.start(cfg, serial=cfg.device.serial)

            ver = daemon_mod.running_version(cfg)
            # A daemon running OLDER code than this CLI can reject new args (e.g. a kwarg
            # added since it started) → confusing crashes. On a version mismatch, skip the
            # daemon and run in-process; a `None` version is a pre-report daemon (trusted).
            #
            # Compare the FULL identity (version + loaded-source fingerprint), not the bare
            # version: during development both sides are the same release, so a plain version
            # check never fires and an edited file keeps being served from the daemon's memory.
            # Must be symmetric — comparing a composite against a bare `__version__` would
            # make every call look skewed and silently disable the daemon.
            skew = isinstance(ver, str) and ver != daemon_mod._aua_version()
            # Try to make the daemon current BEFORE deciding anything else. The old order
            # refused the daemon-only methods first and only then attempted the replacement,
            # which made those methods the one class of call that could never heal itself: a
            # skewed daemon refused them forever, and the hint it printed
            # (`aua daemon stop && aua daemon start`, no serial) stopped a socket nobody used.
            # Everything else restarted and went warm on the very next line.
            preserve_live_capture = bool(
                skew and method in _CAPTURE_READ_METHODS and _capture_session_live(daemon_mod, cfg)
            )
            if (
                skew
                and not preserve_live_capture
                and _replace_skewed_daemon(daemon_mod, cfg, str(ver))
            ):
                ver = daemon_mod.running_version(cfg)
                skew = False
            elif preserve_live_capture:
                # Ask the process that owns the current session. Restarting before a read
                # destroys the only authoritative in-memory mark/window the caller requested.
                skew = False
            restart_hint = _restart_hint(daemon_mod, cfg)
            if skew and method in _DAEMON_ONLY_METHODS:
                # Reached only when the replacement could not be done — in practice because
                # another agent's background job is still running inside that daemon. There is
                # no durable form of this state to fall back on (see _DAEMON_ONLY_METHODS), so
                # an in-process answer would be wrong rather than merely slow.
                raise UsageError(
                    f"the running daemon is executing different code than this CLI, and "
                    f"`{method}` has no answer outside that process",
                    hint=restart_hint,
                )
            if skew:
                logger.warning(
                    "daemon runs aua %s but this CLI is %s; using in-process. Restart it: %s",
                    ver,
                    daemon_mod._aua_version(),
                    restart_hint,
                )
            expected_policy: str | None = None
            if ver is not False and not skew:
                expected_runtime = daemon_mod.runtime_config_fingerprint(cfg)
                live_runtime: str | None | bool = False
                with contextlib.suppress(Exception):
                    live_runtime = daemon_mod.running_runtime_fingerprint(cfg)
                runtime_mismatch = (
                    live_runtime is not False and live_runtime != expected_runtime
                )
                if runtime_mismatch and _replace_runtime_mismatched_daemon(
                    daemon_mod, cfg, expected_runtime
                ):
                    ver = daemon_mod.running_version(cfg)
                    runtime_mismatch = False
                if runtime_mismatch:
                    raise UsageError(
                        "the running daemon belongs to a different cache/lease runtime",
                        hint=_restart_hint(daemon_mod, cfg),
                        code="daemon_runtime_mismatch",
                    )
                expected_policy = daemon_mod.policy_config_fingerprint(cfg)
                live_policy = daemon_mod.running_policy_fingerprint(cfg)
                # ``False`` is an unresponsive/busy ping; the expected fingerprint is also
                # carried on the real request, so the daemon revalidates after it dequeues it.
                # ``None`` is a responsive pre-fingerprint daemon and is therefore stale.
                mismatch = live_policy is not False and live_policy != expected_policy
                if mismatch and _replace_policy_mismatched_daemon(daemon_mod, cfg, expected_policy):
                    ver = daemon_mod.running_version(cfg)
                    mismatch = False
                if mismatch:
                    raise UsageError(
                        "the running daemon uses a different local policy configuration",
                        hint="Restart it: `aua daemon stop && aua daemon start`.",
                        code="policy_config_mismatch",
                    )
            if ver is not False and not skew:
                from . import leases as _leases

                client_options: dict[str, Any] = {
                    "owner": _leases.resolve_owner(getattr(engine, "_lease_owner", None)),
                    "invocation_id": _INVOCATION_ID,
                }
                if _EXPECTED_ERROR_CODE:
                    client_options["expected_error_code"] = _EXPECTED_ERROR_CODE
                if journal_privacy_cmd:
                    client_options["journal_privacy_cmd"] = journal_privacy_cmd
                client_options["policy_fingerprint"] = expected_policy
                client_options["runtime_fingerprint"] = (
                    daemon_mod.runtime_config_fingerprint(cfg)
                )
                # The warm daemon owns the long-lived Engine/provider cache. Ask it to attach
                # goal progress and optional policy output before journaling/serialization so a
                # short-lived CLI process never reloads the local model for the same response.
                client_options["decorate_response"] = True
                client = daemon_mod.DaemonClient(
                    daemon_mod.socket_path(cfg),
                    **client_options,
                )
                cmd = _DAEMON_CMD.get(method, method)
                resp = client.call(cmd, **kwargs)
                if resp.get("ok"):
                    if resp.get("response_decorated") is True:
                        result = resp.get("result")
                    else:
                        from .coaching import decorate_result

                        # Compatibility fallback for an older daemon that ignored the request
                        # flag.
                        result = decorate_result(
                            engine,
                            cmd,
                            resp.get("result"),
                            args=kwargs,
                            invocation_id=_INVOCATION_ID,
                        )
                    daemon_serial = (
                        getattr(engine, "_lease_serial", None) or cfg.device.serial
                    )
                    detail_id = resp.get("journal_detail_id")
                    return _remember_cli_journal(
                        result,
                        engine=engine,
                        cmd=cmd,
                        args=kwargs,
                        serial=daemon_serial,
                        detail_id=detail_id if isinstance(detail_id, str) else None,
                    )
                raise _daemon_error(resp.get("error", {}))
        except AuaError:
            raise
        except Exception as exc:  # pragma: no cover - daemon optional / unreachable
            # A serialized daemon may be busy in a long Engine call and unable to answer its
            # two-second health ping.  Its live pidfile still proves that it owns the device;
            # falling back here would let a second Engine mutate the same screen concurrently.
            if "daemon_mod" in locals() and daemon_mod.process_running(cfg):
                raise DaemonBusyError(
                    "the live AUA daemon is still handling another request",
                    hint=(
                        "Wait for that call to finish and inspect its result. Do not start "
                        "another daemon or repeat a state-changing action."
                    ),
                ) from exc
            logger.debug("daemon route unavailable, running in-process: %s", exc)
    # In-process path — journal here (daemon path is journaled inside the daemon).
    from . import journal as journal_mod

    t0 = time.monotonic()
    serial = cfg.device.serial if host_only else None
    if not host_only and not bootstrap_session:
        with contextlib.suppress(Exception):
            serial = engine.device.serial
    try:
        if not host_only and not bootstrap_session:
            _warm(engine)
        result = getattr(engine, method)(**kwargs)
        if bootstrap_session and isinstance(result, dict):
            selected = result.get("serial")
            if isinstance(selected, str) and selected:
                serial = selected
        from .coaching import decorate_result

        result = decorate_result(
            engine,
            _DAEMON_CMD.get(method, method),
            result,
            args=kwargs,
            current_recorded=False,
            invocation_id=_INVOCATION_ID,
            duration_ms=(time.monotonic() - t0) * 1000.0,
        )
        detail_id = None
        with contextlib.suppress(Exception):
            detail_id = journal_mod.record(
                cache_dir=cfg.cache.dir,
                serial=serial,
                source="cli",
                cmd=_DAEMON_CMD.get(method, method),
                args=kwargs,
                ok=not (isinstance(result, dict) and result.get("ok") is False)
                and not (hasattr(result, "ok") and result.ok is False),
                duration_ms=(time.monotonic() - t0) * 1000.0,
                result=result,
                owner=getattr(engine, "_lease_owner_resolved", None),
                extra={
                    **({"invocation_id": _INVOCATION_ID} if _INVOCATION_ID else {}),
                    **(
                        {
                            "expected_error_code": _EXPECTED_ERROR_CODE,
                            "expected_error_matched": False,
                        }
                        if _EXPECTED_ERROR_CODE
                        else {}
                    ),
                },
                privacy_cmd=journal_privacy_cmd,
            )
        return _remember_cli_journal(
            result,
            engine=engine,
            cmd=_DAEMON_CMD.get(method, method),
            args=kwargs,
            serial=serial,
            detail_id=detail_id,
        )
    except AuaError as err:
        with contextlib.suppress(Exception):
            error_value = err.to_dict().get("error")
            journal_mod.record(
                cache_dir=cfg.cache.dir,
                serial=serial,
                source="cli",
                cmd=_DAEMON_CMD.get(method, method),
                args=kwargs,
                ok=False,
                duration_ms=(time.monotonic() - t0) * 1000.0,
                error=error_value if isinstance(error_value, dict) else None,
                owner=getattr(engine, "_lease_owner_resolved", None),
                extra={
                    **({"invocation_id": _INVOCATION_ID} if _INVOCATION_ID else {}),
                    **(
                        {
                            "expected_error_code": _EXPECTED_ERROR_CODE,
                            "expected_error_matched": err.code == _EXPECTED_ERROR_CODE,
                        }
                        if _EXPECTED_ERROR_CODE
                        else {}
                    ),
                },
                privacy_cmd=journal_privacy_cmd,
            )
        raise


# --------------------------------------------------------------------------- app


_GUIDE_POINTER = (
    "Run `aua guide` for the full agent operating manual (session protocol, escalation "
    "ladder, memory, schema, exit codes); `aua guide --emit-skill` regenerates the shared "
    "Claude/Codex skill from the same source."
)

#: Lines per page of `--help` / `guide`. Sized so one page survives a tool-output limit whole.
HELP_PAGE_LINES = 55


def _page_arg(argv: list[str] | None = None) -> int:
    """``--page N`` / ``--page=N`` read straight from argv.

    Click's ``--help`` is eager: it renders and exits during parsing, before a normally-declared
    option would bind. Reading argv is what makes ``aua --help --page 2`` work at all.
    """
    args = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(args):
        if arg == "--page" and i + 1 < len(args):
            raw = args[i + 1]
        elif arg.startswith("--page="):
            raw = arg.split("=", 1)[1]
        else:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    return 1


def paginate(text: str, page: int, *, per_page: int = HELP_PAGE_LINES, more: str = "") -> str:
    """One page of *text*, ending with the command that returns the next one.

    Long help reaches an agent through a tool that truncates, and a silent cut is
    indistinguishable from "that is all there is". Measured: a fresh agent read `aua --help`
    (172 lines), had it cut, concluded typing was undocumented, and only learned the syntax by
    failing a call. Splitting the text is not the fix on its own — the footer is, because it is
    the only thing that tells the reader something was withheld.
    """
    lines = text.splitlines()
    total = max(1, -(-len(lines) // per_page))
    page = min(max(1, page), total)
    if total == 1:
        return text
    chunk = lines[(page - 1) * per_page : page * per_page]
    footer = f"— page {page} of {total} —"
    footer += f"  next: {more.format(page=page + 1)}" if page < total and more else "  (end)"
    return "\n".join([*chunk, "", footer])


class UnknownCommand(AuaError):
    """A name that is not a command, answered with the one that was meant plus how to drive."""

    exit_code = ExitCode.USAGE
    code = "unknown_command"

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        from .guide import COMMAND_SYNONYMS

        self.meant = COMMAND_SYNONYMS.get(name.lower())
        # The real vocabulary. Without it the only way to learn a command name was
        # `aua --help`, whose first page carried none — so a caller that guessed wrong
        # had nothing new to go on and guessed again.
        self.available = sorted(available or [])
        message = f"`aua {name}` is not a command."
        if self.meant:
            message += f" Use `aua {self.meant}`."
        elif self.available:
            message += f" Available: {', '.join(self.available)}."
        self.recommended_call = f"aua {self.meant} --help" if self.meant else "aua guide --brief"
        super().__init__(message, hint=_GUIDE_POINTER)

    def to_dict(self) -> dict[str, object]:
        from .guide import ORIENTATION

        out = super().to_dict()
        err = out["error"]
        if isinstance(err, dict):
            if self.meant:
                err["did_you_mean"] = self.meant
            if self.available:
                err["available_commands"] = self.available
            err["recommended_call"] = self.recommended_call
            err["how_to_drive"] = [f"{cmd}  # {why}" for cmd, why in ORIENTATION]
        return out


class GuidingGroup(TyperGroup):
    """Answer an unknown command with what was meant and how to drive, not a spelling guess.

    Click's fallback is string distance over command names, which is worse than nothing here: it
    sent `tree` to `target` and `state` to `paste`, and offered no suggestion at all for `ui`,
    `dump` or `elements`. An agent that guessed has not read the guide, so the failure is the only
    place it will read anything — it carries the orientation block rather than a usage page.
    """

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Typer renders help through rich straight to stdout rather than into `formatter`, so the
        # only way to page it is to catch what it wrote.
        import contextlib
        import io
        import textwrap

        from .guide import ORIENTATION

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            super().format_help(ctx, formatter)
        rendered = buffer.getvalue() or formatter.getvalue()
        # Paging alone would make page 1 fifty-five lines of global options and not one command.
        # Whatever a truncating reader gets, it must be the loop.
        names = sorted(self.list_commands(ctx))
        head = [
            "The loop — everything else is a variation on these:",
            *(f"  {cmd}  # {why}" for cmd, why in ORIENTATION),
            "",
            # The rendered help puts ~55 lines of global options ahead of the command
            # table, so on a paged terminal page 1 carried not one command name and
            # `aua --help | grep <name>` found nothing. Agents answered that by guessing
            # names, failing, and re-reading page 1 — measured 110 help calls in one run.
            "All commands (`aua <command> --help` for one):",
            *textwrap.wrap(", ".join(names), width=76, initial_indent="  ",
                           subsequent_indent="  "),
            "",
        ]
        body = "\n".join([*head, rendered.rstrip("\n")])
        shown = paginate(body, _page_arg(), more="aua --help --page {page}")
        click.echo(shown)
        _journal_cli_recovery(
            ctx,
            event="help",
            ok=True,
            recommended_call="aua guide --brief",
            response_text=shown,
            argv=["--help"] + ([f"--page={_page_arg()}"] if _page_arg() > 1 else []),
        )

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            name = next((a for a in args if not a.startswith("-")), "")
            if not name or name in self.commands:
                raise
            import json

            err = UnknownCommand(name, list(self.list_commands(ctx)))
            _journal_cli_recovery(
                ctx,
                event="usage_error",
                ok=False,
                error=err,
                recommended_call=err.recommended_call,
                response_text=json.dumps(err.to_dict(), ensure_ascii=False, indent=2),
                argv=list(args),
            )
            emit_error(err)
            raise typer.Exit(int(err.exit_code)) from None

    def invoke(self, ctx: click.Context) -> Any:
        """Answer a missing argument with the values it accepts.

        Click prints `Missing argument 'ACTION'` over a usage line that names the argument
        again and nothing else. The accepted values are right there in the parameter's help and
        never reach the caller, so the only way on is a second call to `--help`. Measured
        2026-08-10: two separate agents ran `aua app`, and both spent that extra command.
        """
        try:
            return super().invoke(ctx)
        except click.MissingParameter as exc:
            param = exc.param
            choices = (getattr(param, "help", "") or "").strip().rstrip(".")
            if not choices or not param:
                raise
            metavar = getattr(param, "metavar", None) or param.name.upper()
            # The group's own `info_name` is `aua`; the command that is missing the argument
            # is one level down, and naming the wrong one would put a broken example in the
            # hint — the exact failure this is here to stop.
            path = getattr(exc.ctx, "command_path", None) or ctx.info_name
            recommended_call = f"{path} {choices.split('|')[0].strip()}"
            import json

            err = RecommendedUsageError(
                f"`{path} {metavar}` needs a value",
                hint=f"One of: {choices}. Pass it as the first argument, e.g. "
                f"`{recommended_call}`.",
                recommended_call=recommended_call,
            )
            _journal_cli_recovery(
                exc.ctx or ctx,
                event="usage_error",
                ok=False,
                error=err,
                recommended_call=recommended_call,
                response_text=json.dumps(err.to_dict(), ensure_ascii=False, indent=2),
            )
            emit_error(err)
            raise typer.Exit(int(err.exit_code)) from None


app = typer.Typer(
    name="aua",
    cls=GuidingGroup,
    help=(
        "android-ui-analyser — structured Android UI perception + action for agents.\n\n"
        + _GUIDE_POINTER
    ),
    epilog=_GUIDE_POINTER,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.command("capabilities")
def capabilities_cmd(
    ctx: typer.Context,
    goal: str | None = typer.Option(None, "--goal", help="Rank capabilities for this goal."),
) -> None:
    """Discover the canonical CLI/MCP capability catalogue without needing a device."""
    from .capabilities import capabilities_for_goal, capability_manifest

    payload = capabilities_for_goal(goal) if goal else capability_manifest()
    _echo_json({"goal": goal, "capabilities": payload}, _opts(ctx).fmt())


@app.callback()
def main(
    ctx: typer.Context,
    platform: str | None = typer.Option(
        None,
        "--platform",
        help="Platform strategy (default: android; installed plugins may add others).",
    ),
    serial: str | None = typer.Option(
        None,
        "--serial",
        help="Initial/explicit target override; ordinary commands use the agent's one sticky "
        "lease. Switching requires `lease acquire <serial> --replace`.",
    ),
    config: str | None = typer.Option(None, "--config", help="Explicit config file path."),
    format: str | None = typer.Option(
        None,
        "--format",
        help="Output format: json|pretty|compact|tsv|delta|msgpack (tsv/delta/msgpack: analyze).",
    ),
    profile: str | None = typer.Option(None, "--profile", help="Named config profile to overlay."),
    # Declared so Click accepts it; the value is read from argv in `_page_arg`, because `--help`
    # is eager and renders before a normal option would ever bind.
    page: int = typer.Option(
        1, "--page", hidden=True, help="Which page of `--help` / `guide` output to print."
    ),
    timeout: int | None = typer.Option(None, "--timeout", help="Per-operation timeout in ms."),
    log_level: str = typer.Option(
        "warn", "--log-level", help="error|warn|info|debug (logs → stderr)."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the cached analyze result."),
    no_app_logs: bool = typer.Option(
        False,
        "--no-app-logs",
        help="Do not fold what the app logged during an action into its observation.",
    ),
    app_log_levels: str | None = typer.Option(
        None,
        "--app-logs",
        metavar="LEVELS",
        help="Log priorities folded into an action's observation (default DWEF). "
        "A set, not a floor: `I` is noisier than `D` on Android. `F` is always included.",
    ),
    with_image: bool = typer.Option(
        False,
        "--with-image",
        help="Session default: save raw screenshots on analyze/actions "
        "(override per-command with --with-image PATH or omit).",
    ),
    observe_fields: str | None = typer.Option(
        None,
        "--observe-fields",
        metavar="NAMES|all",
        help="Columns kept in an action's post-action observation ('all' = full dump). "
        "Defaults to a compact view so you never need --no-observe.",
    ),
    observe_meta: str | None = typer.Option(
        None,
        "--observe-meta",
        metavar="PRESET|KEYS|all",
        help="`meta` keys kept in an action's post-action observation: 'changed' (default), "
        "'all', or a comma-separated list. Independent of --observe-fields: asking for every "
        "column is not asking for every hint. Research tasks, deeplink suggestions and capture "
        "hints are not in 'changed' — run `analyze` when you want them.",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        metavar="PREDICATE",
        help="After the action, wait until this holds before observing "
        "(`rid:resultsPanel`, `text:Results,!text:Loading`). At least one term must be positive; "
        "standalone wait-and-analyze may use an absence-only predicate. Escape a literal "
        "comma as `\\,`. Sets await_outcome.",
    ),
    answers: list[str] | None = typer.Option(
        None,
        "--answers",
        metavar='TASK_ID="value"',
        help="Answer a map question `meta.ask` raised on an earlier call, e.g. "
        '--answers research_23cf9="Dev Tools". Repeatable. Applies before the command runs.',
    ),
    phase_done: list[str] | None = typer.Option(
        None,
        "--phase-done",
        metavar='PHASE_ID="evidence"',
        help=(
            "Acknowledge the current goal phase using evidence from the previous result while "
            "running this next command; repeatable and adds no extra device call."
        ),
    ),
    expect_error: str | None = typer.Option(
        None,
        "--expect-error",
        metavar="CODE",
        help=(
            "Declare the exact machine-readable error code this invocation intentionally "
            "probes; session review counts it as expected only when it matches."
        ),
    ),
    until_timeout: int = typer.Option(
        30000, "--until-timeout", metavar="MS", help="Give up on --until after this long."
    ),
    until_poll: int = typer.Option(
        500, "--until-poll", metavar="MS", help="How often --until re-checks."
    ),
    owner: str | None = typer.Option(
        None,
        "--owner",
        metavar="AGENT",
        help="Agent identity for the device lease (else $AUA_OWNER, else derived).",
    ),
    needs: str | None = typer.Option(
        None,
        "--needs",
        metavar="root,play,proxy",
        help="Capabilities the device must have; refuses rather than handing back a device "
        "that cannot do them.",
    ),
    no_lease: bool = typer.Option(
        False, "--no-lease", help="Skip device leasing entirely (single-agent / scripts)."
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """Parse global options, configure stderr logging, stash opts on the context."""
    level = _LOG_LEVELS.get((log_level or "warn").lower(), logging.WARNING)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if format is not None and format not in {f.value for f in OutputFormat}:
        # Surface as a usage error (exit 2) before any command runs.
        err = UsageError(
            f"invalid --format '{format}'",
            hint="Choose one of: json, pretty, compact, tsv, delta, msgpack.",
        )
        emit_error(err)
        raise typer.Exit(int(err.exit_code))
    ctx.obj = GlobalOpts(
        platform=platform,
        serial=serial,
        config=config,
        format=format,
        profile=profile,
        timeout=timeout,
        log_level=log_level,
        no_cache=no_cache,
        no_app_logs=no_app_logs,
        app_log_levels=app_log_levels,
        with_image=with_image,
        observe_fields=observe_fields,
        observe_meta=observe_meta,
        answers=tuple(answers or ()),
        phases_done=tuple(phase_done or ()),
        expect_error=expect_error,
        until=until,
        until_timeout=until_timeout,
        until_poll=until_poll,
        explicit_wait_flags=frozenset(
            flag
            for name, flag in (("until_timeout", "--until-timeout"), ("until_poll", "--until-poll"))
            if getattr(ctx.get_parameter_source(name), "name", None) == "COMMANDLINE"
        ),
        owner=owner,
        needs=needs,
        no_lease=no_lease,
    )


# --------------------------------------------------------------------------- perception


@app.command()
def ask(
    ctx: typer.Context,
    question: str = typer.Argument(
        ..., help="Question for the configured vision model about the current screen."
    ),
) -> None:
    """Ask about the current screenshot fused with AUA's element graph."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "ask_screen", question=question), fmt)

    _run(ctx, go)


@app.command(cls=AnnotateCommand)
def analyze(
    ctx: typer.Context,
    source: str = typer.Option(
        "auto", "--source", help="auto|hierarchy|vision (force perception path)."
    ),
    with_ocr: bool | None = typer.Option(
        None, "--with-ocr/--no-ocr", help="Include OCR text boxes."
    ),
    annotate: str | None = typer.Option(
        None,
        "--annotate",
        metavar="[PATH]",
        help="Also write an annotated screenshot; bare flag uses a default path.",
        show_default=False,
    ),
    query: str | None = typer.Option(
        None, "--query", help="Return the single best-matching element."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
    deep: bool = typer.Option(False, "--deep", help="Raise the escalation ceiling for this call."),
    cheap: bool = typer.Option(
        False, "--cheap", help="Lower the escalation ceiling for this call."
    ),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help="Pin a tier: text|selector|hierarchy|vision|grounding|auto.",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass / do not write the analyze cache."
    ),
    fields: str | None = typer.Option(
        None,
        "--fields",
        metavar="CSV",
        help="Project elements to these keys: id,text,rid,desc,bounds,center,type,"
        "clickable,enabled,checked,selected,scrollable,long_clickable,password.",
    ),
    nonempty: bool = typer.Option(
        False, "--nonempty", help="Drop elements with no text, resource_id or content_desc."
    ),
    no_system: bool = typer.Option(
        False, "--no-system", help="Drop status-bar / system chrome (systemui ids, battery…)."
    ),
    no_ime: bool = typer.Option(
        False, "--no-ime", help="Drop soft-keyboard (IME) chrome so chat trees stay readable."
    ),
    no_wrappers: bool = typer.Option(
        False,
        "--no-wrappers",
        help="Drop pure layout containers (id'd but unlabeled, inert, wrapping something). "
        "Leaves and addressable containers stay.",
    ),
    show_all: bool = typer.Option(
        False, "--all", help="Keep every element (undoes tsv's implicit --nonempty --no-system)."
    ),
    where_text: list[str] | None = typer.Option(
        None, "--where-text", metavar="SUBSTR", help="Keep elements whose text contains this."
    ),
    where_rid: list[str] | None = typer.Option(
        None, "--where-rid", metavar="SUBSTR", help="Keep elements whose resource_id contains this."
    ),
    clickable: bool = typer.Option(False, "--clickable", help="Keep only clickable elements."),
    region: list[str] | None = typer.Option(
        None,
        "--region",
        metavar="x1,y1,x2,y2",
        help="Keep elements intersecting this box (e.g. 0,0,1080,300 = the header).",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Keep at most N elements."),
    meta: str | None = typer.Option(
        None, "--meta", metavar="CSV", help="Return only these meta keys."
    ),
    no_meta: bool = typer.Option(False, "--no-meta", help="Omit meta entirely."),
) -> None:
    """Emit Set-of-Marks JSON (§8) for the current screen."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        view = Projection.parse(
            fmt=fmt,
            fields=fields,
            nonempty=nonempty,
            no_system=no_system,
            no_ime=no_ime,
            no_wrappers=no_wrappers,
            show_all=show_all,
            where_text=where_text,
            where_rid=where_rid,
            clickable=clickable,
            region=region,
            limit=limit,
            meta=meta,
            no_meta=no_meta,
        )
        nc = no_cache or _opts(ctx).no_cache
        result = _route(
            engine,
            "analyze",
            source=source,
            with_ocr=with_ocr,
            query=query,
            annotate=_annotate_arg(annotate),
            with_image=_annotate_arg(with_image),
            strategy=strategy,
            cheap=cheap,
            deep=deep,
            no_cache=nc,
        )
        _warn_if_redundant_analyze(
            engine,
            {
                "cmd": "analyze",
                "source": source,
                "query": query,
                "with_ocr": with_ocr,
                "fields": fields,
            },
        )
        _emit_analyze(result, fmt, view)

    _run(ctx, go)


@app.command()
def screenshot(
    ctx: typer.Context,
    path: str | None = typer.Argument(None, help="Output PNG path (default under run dir)."),
    out: str | None = typer.Option(None, "--out", "-o", help="Output PNG path (same as the arg)."),
    annotate: bool = typer.Option(False, "--annotate", help="Overlay Set-of-Marks numbers."),
    region: str | None = typer.Option(
        None,
        "--region",
        metavar="x1,y1,x2,y2",
        help="Crop to this box before saving (e.g. 0,0,1080,300 = the header).",
    ),
    scale: float | None = typer.Option(
        None, "--scale", help="Downscale by this factor (0.5 = half width)."
    ),
    max_width: int | None = typer.Option(
        None, "--max-width", help="Downscale so the width is at most this many pixels."
    ),
) -> None:
    """Save a screenshot (PNG); crop/downscale it, or ``--annotate`` the last analyze marks.

    ``--region``/``--scale``/``--max-width`` exist to keep an agent's context cheap: reading
    a 1080x2400 PNG to check one header icon costs an order of magnitude more image tokens
    than reading the strip it lives in. The written path is the last line of output.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        def emit_captured(captured: Any) -> None:
            journal_context = _take_cli_journal(captured)
            ok = captured.get("ok") if isinstance(captured, dict) else captured.ok
            action = (
                captured.get("action") if isinstance(captured, dict) else captured.action
            )
            detail = (
                captured.get("detail") if isinstance(captured, dict) else captured.detail
            )
            # Routing decoration belongs to the daemon/session journal, but screenshot's public
            # response has always been the three-field ActionResult. Rebuild that response so
            # fixing the transport does not silently change the CLI contract.
            emitted = ActionResult(
                ok=bool(ok),
                action=str(action or "screenshot"),
                detail=str(detail) if detail is not None else None,
            )
            _emit(emitted, fmt, _journal_context=journal_context)

        target = out or path
        narrowed = region is not None or scale is not None or max_width is not None
        if not narrowed:
            emit_captured(_route(engine, "screenshot", path=target, annotate=annotate))
            return
        if annotate:
            raise UsageError(
                "--annotate cannot be combined with --region/--scale/--max-width",
                hint="Marks are placed in full-screen coordinates; crop a plain screenshot.",
            )
        from . import imaging
        from .providers.base import ScreenImage

        box = imaging.parse_region(region) if region else None
        captured = _route(engine, "screenshot", path=target, annotate=False)
        captured_path = (
            captured.get("detail")
            if isinstance(captured, dict)
            else getattr(captured, "detail", None)
        )
        if not isinstance(captured_path, str) or not captured_path:
            raise DeviceError("screenshot did not return a readable output path")
        raw = Path(captured_path).read_bytes()
        view = imaging.crop_and_scale(
            ScreenImage(raw, path=captured_path),
            region=box,
            scale=scale,
            max_width=max_width,
        )
        view.save(captured_path)
        emit_captured(captured)

    _run(ctx, go)


@app.command()
def inspect(
    ctx: typer.Context,
    element_id: int = typer.Argument(..., metavar="ID", help="Element id from the last analyze."),
) -> None:
    """Print full attributes for one element from the last analyze."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _warm(engine)  # align the cache key with the serial a prior analyze wrote
        el = engine.inspect(element_id)
        typer.echo(el.model_dump_json(indent=2 if fmt is OutputFormat.pretty else None))

    _run(ctx, go)


# --------------------------------------------------------------------------- quick check


@app.command()
def has(
    ctx: typer.Context,
    text: str | None = typer.Argument(None, help="Text to look for on screen."),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
    ocr_fallback: bool = typer.Option(
        True,
        "--ocr-fallback/--no-ocr-fallback",
        help="OCR the screenshot on a hierarchy miss.",
    ),
    source: str = typer.Option("auto", "--source", help="hierarchy|vision|auto."),
    timeout: int = typer.Option(
        0, "--timeout", "--timeout-ms", help="Poll until present or timeout ms (0 = instant)."
    ),
    by: str = typer.Option(
        "text", "--by", help="Match by: text (default) | id (resource-id) | desc."
    ),
    rid: str | None = _SEL_RID,
    text_sel: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
) -> None:
    """Is this on screen right now? Exit 0 if present, 1 if not.

    ``--by id`` checks a resource-id (a bare tail works) — verifies containers the element
    list prunes, i.e. Maestro-style ``assertVisible: id:``.

    Takes the same one-shot selectors as the action commands, so a check reads like the act
    it guards: `aua has --rid buttonSettings` then `aua tap-and-analyze --rid buttonSettings`.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        target, target_by = _has_target(text, by=by, rid=rid, text_sel=text_sel, desc=desc)
        result = _route(
            engine,
            "has",
            text=target,
            match=match,
            ignore_case=ignore_case,
            ocr_fallback=ocr_fallback,
            source=source,
            timeout_ms=timeout,
            by=target_by,
        )
        _emit(result, fmt)
        found = result.get("found") if isinstance(result, dict) else getattr(result, "found", False)
        if not found:
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- actions


@app.command(name="tap-and-analyze", cls=AnalyzeCommand)
def tap(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id from the last analyze (or the selector value)."
    ),
    id_opt: int | None = typer.Option(
        None,
        "--id",
        min=0,
        help="Element id from the last analyze; accepted as an explicit alias for positional ID.",
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    point: str | None = typer.Option(
        None,
        "--point",
        metavar="X,Y",
        help="Tap this exact coordinate instead of an element (for a canvas or cell-less grid).",
    ),
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="Also return the screen after the tap (skips a follow-up analyze).",
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Tap an element — by id from the last analyze, or by a one-shot selector.

    `aua tap-and-analyze 9` (or `--id 9`) · `aua tap-and-analyze --rid notificationsButton` · `aua tap-and-analyze --text "Create an app"` ·
    `aua tap-and-analyze --by id homeTabBROWSE`. A selector resolves on the live screen in this one
    call; matching nothing exits 6 and matching several exits 7 with the candidates — it
    never silently taps nothing.

    `aua tap-and-analyze --point 412,733` addresses a coordinate directly, for a canvas or a grid whose
    cells publish no node of their own. It is recorded like any other action, so a journey
    across such a surface can still be captured as a flow.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        effective_ident = ident
        if id_opt is not None:
            if ident is not None or rid or text or desc or by:
                raise UsageError(
                    "--id is one complete target; do not combine it with another selector",
                    hint="Use either `tap-and-analyze --id 4` or one --rid/--text/--desc target.",
                )
            effective_ident = str(id_opt)
        if point is not None:
            if effective_ident or rid or text or desc:
                raise UsageError(
                    "--point taps a coordinate, so it cannot be combined with a selector",
                    hint="drop the selector, or drop --point and address the element",
                )
            xy = _parse_point(point)
            if xy is None:
                raise UsageError(
                    f"--point wants two non-negative numbers like 412,733 — got {point!r}"
                )
            _emit(
                _route(
                    engine,
                    "tap_point",
                    x=xy[0],
                    y=xy[1],
                    observe=observe,
                    with_image=_annotate_arg(with_image),
                ),
                fmt,
            )
            return
        selector = _selector(
            ident=effective_ident,
            by=by,
            rid=rid,
            text=text,
            desc=desc,
            key=key,
            index=index,
            first=first,
        )
        _emit(
            _route(
                engine,
                "tap",
                element_id=_require_target("tap", effective_ident, selector),
                selector=selector,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="await-and-analyze", cls=AnalyzeCommand)
def await_cmd(
    ctx: typer.Context,
    predicate: str = typer.Argument(
        ...,
        metavar="PREDICATE",
        help="Comma-separated terms, all of which must hold: `text:Done`, `rid:resultCard`, "
        "`desc:Play`; prefix a term with `!` to require absence.",
    ),
    timeout: int = typer.Option(60000, "--timeout-ms", help="Give up after this long."),
    poll: int = typer.Option(500, "--poll-ms", help="How often to re-check the predicate."),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
    observe: bool = typer.Option(
        False, "--observe/--no-observe", help="Also return the screen when the wait ends."
    ),
    no_meta: bool = typer.Option(
        False,
        "--no-meta",
        help="Omit meta from the returned observation (same meaning as analyze).",
    ),
) -> None:
    """Wait for a condition, distinguishing success, navigation, and timeout.

    `await_outcome` is `satisfied`, `screen-changed` (the foreground activity moved while
    waiting — the surface is gone, so more waiting cannot help) or `timeout`. That distinction
    is the point: a coordinator twice had to ask a lane "is that a hang or a slow backend?"
    because nothing in the output could tell them apart. Per-term results come back either way,
    so "spinner gone but results absent" reads differently from "spinner still spinning".

        aua await-and-analyze 'rid:resultCard,!text:Generating' --timeout-ms 240000

    Deliberately not network idle: this app never is (analytics post continuously, chat
    streams), so idleness would be a flaky proxy for readiness.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _set_no_meta_observation_view(fmt, no_meta)
        result = _route(
            engine,
            "await_predicate",
            predicate=predicate,
            timeout_ms=timeout,
            poll_ms=poll,
            match=match,
            ignore_case=ignore_case,
            observe=observe,
        )
        _emit(result, fmt)
        # A wait that did not get what it was told to wait for must not exit 0.
        _exit_unless_ok(
            result,
            ExitCode.NOT_FOUND,
            code="await_unsatisfied",
            hint="`await_outcome` says which: `screen-changed` means the surface moved on, "
            "`timeout` means it never arrived.",
        )

    _run(ctx, go)


@app.command(name="target")
def target_cmd(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[VALUE]", help="Selector value; pair with --by, or use --rid/--text/--desc."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
) -> None:
    """What does this label actually address? Reads the screen; touches nothing.

    A visible title is often *not* the node that acts. A design-system tile puts the click on
    an inner Box and renders the caption outside those bounds, so the caption reports
    `clickable:false` and its `enabled` describes a caption rather than a control — the same
    pair of values whether the real control is enabled or disabled. A lane read that as a
    broken product and filed a critical failure against a working one.

    Prints the node named, the node that acts, their relation, the state that belongs to the
    control, and the point a tap would use.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        if not selector:
            raise UsageError(
                "target needs a selector",
                hint="`aua target --text 'Beat Painter'` or `aua target --by id tileHit`",
            )
        _emit(
            _route(
                engine,
                "target_report",
                rid=selector.get("rid"),
                text=selector.get("text"),
                desc=selector.get("desc"),
                index=selector.get("index"),
                first=bool(selector.get("first")),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="click-and-analyze", cls=AnalyzeCommand)
def click_cmd(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to tap (alias of tap)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-tap screen."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Alias of ``tap``."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            _route(
                engine,
                "tap",
                element_id=_require_target("tap", ident, selector),
                selector=selector,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="long-press-and-analyze", cls=AnalyzeCommand)
def long_press(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to long-press (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    ms: int = typer.Option(600, "--ms", help="Press duration in milliseconds."),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-action screen."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Long-press an element (id or selector, same as ``tap``)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            _route(
                engine,
                "long_press",
                element_id=_element_id(ident, selector),
                selector=selector,
                ms=ms,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="double-tap-and-analyze", cls=AnalyzeCommand)
def double_tap(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to double-tap (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-action screen."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Double-tap an element (id or selector, same as ``tap``)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            _route(
                engine,
                "double_tap",
                element_id=_element_id(ident, selector),
                selector=selector,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="input-and-analyze", cls=AnalyzeCommand)
def input_cmd(
    ctx: typer.Context,
    first_arg: str | None = typer.Argument(
        None, metavar="[ID] TEXT", help="Element id then text — or just the text with --rid/--by."
    ),
    second_arg: str | None = typer.Argument(None, metavar="", help="", show_default=False),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    # Accepted only to answer it properly: `--text` is a selector on every other command, so
    # click's nearest-name hint here was "--index", which sends people further from the fix.
    text_opt: str | None = typer.Option(None, "--text", hidden=True),
    submit: bool = typer.Option(False, "--submit", help="Send the IME action after typing."),
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="Also return the screen after typing (skips a follow-up analyze).",
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Focus an element and type text (fast path built-in); ``--submit`` sends the IME action.

    Prefers accessibility ``set_text``, then clipboard paste (clipboard restored), then IME
    keys — no need for a manual ``clipboard set`` + ``paste`` workaround.

    `aua input-and-analyze 9 "hello"` · `aua input-and-analyze --rid promptField "hello" --submit` ·
    `aua input-and-analyze --by id promptField "hello"`. With ``--rid``/``--desc`` the single positional
    is the text; with ``--by`` (or a plain id) the first positional addresses the field and
    the second is the text. ``--text`` is not a selector here — it would read as the value.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        if text_opt is not None:
            raise UsageError(
                "input-and-analyze takes the text to type positionally, not with --text",
                hint=f'e.g. `aua input-and-analyze --rid <resourceId> "{text_opt}"`; on other '
                "commands --text selects an element by its label, so input cannot reuse it",
            )
        selector = _selector(
            ident=first_arg, by=by, rid=rid, desc=desc, key=key, index=index, first=first
        )
        # --rid/--desc address the field, so the lone positional is the text to type;
        # --by consumes the first positional as the selector value.
        typed = first_arg if (selector is not None and by is None) else second_arg
        if selector is not None and by is None and second_arg is not None:
            raise UsageError(
                "with --rid/--desc, pass only the text to type",
                hint='e.g. `aua input-and-analyze --rid promptField "hello"`',
            )
        if typed is None and first_arg is not None:
            # The caller DID pass text; it was consumed as the target. "needs the text to type"
            # reads as false from their side, so name what actually happened to their argument.
            raise UsageError(
                f'"{first_arg}" was read as the element to type INTO, not the text to type — '
                "without --rid/--desc the first positional addresses the field",
                hint=f'`aua input-and-analyze --rid <resourceId> "{first_arg}"` addresses the '
                f'field by resource-id, or `aua input-and-analyze <id> "{first_arg}"` gives both.',
            )
        if typed is None:
            raise UsageError(
                "input-and-analyze needs the text to type",
                hint='e.g. `aua input-and-analyze 9 "hello"` or '
                '`aua input-and-analyze --rid promptField "hello"`',
            )
        _emit(
            _route(
                engine,
                "input_text",
                element_id=_element_id(first_arg, selector),
                selector=selector,
                text=typed,
                submit=submit,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="clear-and-analyze", cls=AnalyzeCommand)
def clear(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to clear (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-action screen."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Clear the text of an element (id or selector, same as ``tap``)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            _route(
                engine,
                "clear",
                element_id=_element_id(ident, selector),
                selector=selector,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="swipe-and-analyze", cls=AnalyzeCommand)
def swipe(
    ctx: typer.Context,
    direction_arg: str | None = typer.Argument(
        None, metavar="[DIRECTION]", help="up|down|left|right (or use --direction / --coords)."
    ),
    direction_opt: str | None = typer.Option(
        None, "--direction", "-d", help="Same as the positional direction."
    ),
    from_id: int | None = typer.Option(None, "--from", help="Anchor the swipe at this element."),
    from_rid: str | None = typer.Option(
        None, "--from-rid", help="Anchor the swipe at this resource-id (no analyze needed)."
    ),
    percent: int = typer.Option(
        70, "--percent", help="Swipe distance as a % of the scrolled container."
    ),
    coords: tuple[int, int, int, int] | None = typer.Option(
        None,
        "--coords",
        help="Explicit x1 y1 x2 y2 (overrides direction).",
    ),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-swipe screen."
    ),
    verify: bool = typer.Option(
        False,
        "--verify/--no-verify",
        help="Also probe whether the list moved (slower; default is cheap browse swipe).",
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Swipe in a direction, from an element, or by explicit coordinates.

    `aua swipe-and-analyze up` · `aua swipe-and-analyze --direction up` · `aua swipe-and-analyze --from-rid notificationList up`.
    No anchor needed: the gesture is aimed at the scrollable container on screen rather than
    the middle of the display, and ``detail`` reports ``moved``/``no-change`` so a swipe that
    did nothing cannot look like a swipe that worked. For list scrolling prefer
    ``aua scroll-and-analyze``, which turns that verdict into an exit code.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        coord_tuple = tuple(coords) if coords is not None else None
        _emit(
            _route(
                engine,
                "swipe",
                direction=direction_arg or direction_opt,
                from_id=from_id,
                selector=_selector(rid=from_rid),
                percent=percent,
                coords=coord_tuple,
                observe=observe,
                verify=verify,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="scroll-and-analyze", cls=AnalyzeCommand)
def scroll(
    ctx: typer.Context,
    direction_arg: str | None = typer.Argument(
        None, metavar="[DIRECTION]", help="up|down|left|right (default: up, i.e. further down)."
    ),
    direction_opt: str | None = typer.Option(
        None, "--direction", "-d", help="Same as the positional direction."
    ),
    pages: int = typer.Option(1, "--pages", help="Scroll this many screenfuls."),
    to_end: bool = typer.Option(False, "--to-end", help="Scroll until nothing moves any more."),
    to_start: bool = typer.Option(False, "--to-start", help="Scroll back to the top/start."),
    from_id: int | None = typer.Option(
        None, "--from", help="Scroll the container at this element."
    ),
    in_rid: str | None = typer.Option(
        None, "--in-rid", help="Scroll the container at this resource-id."
    ),
    percent: int = typer.Option(70, "--percent", help="Travel per step as a % of the container."),
    max_steps: int = typer.Option(25, "--max-steps", help="Safety cap for --to-end/--to-start."),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the screen after scrolling."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Scroll a container and say what happened — verified, with a real exit code.

    `aua scroll-and-analyze up` · `aua scroll-and-analyze --pages 3` · `aua scroll-and-analyze --to-end` · `aua scroll-and-analyze --to-start`.
    ``detail`` starts with the outcome: ``moved`` · ``reached-end`` · ``already-at-end``.
    A scroll that moved nothing exits 6 (except with ``--to-end/--to-start``, where already
    being at the end is success), so "nothing left to scroll" is never confused with
    "my swipe missed the list".
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "scroll",
            direction=direction_arg or direction_opt,
            pages=pages,
            to_end=to_end,
            to_start=to_start,
            from_id=from_id,
            selector=_selector(rid=in_rid),
            percent=percent,
            max_steps=max_steps,
            observe=observe,
            with_image=_annotate_arg(with_image),
        )
        _emit(result, fmt)
        _exit_unless_ok(
            result,
            ExitCode.NOT_FOUND,
            code="scroll_no_movement",
            hint="`already-at-end` means there is nothing more in that direction; "
            "`scrollable=false` means no scrollable container was found on this screen.",
        )

    _run(ctx, go)


@app.command(name="scroll-to-and-analyze", cls=AnalyzeCommand)
def scroll_to(
    ctx: typer.Context,
    text: str | None = typer.Argument(
        None, metavar="[TEXT]", help="Text or resource-id to scroll to (or use --rid)."
    ),
    rid: str | None = typer.Option(
        None, "--rid", help="Scroll to this resource-id (same as `--by id <TEXT>`)."
    ),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="Also return the screen after scrolling (skips a follow-up analyze).",
    ),
    by: str = typer.Option("text", "--by", help="Match by: text (default) | id | desc."),
    direction: str = typer.Option("up", "--direction", "-d", help="Scroll this way while looking."),
    max_swipes: int = typer.Option(10, "--max-swipes", help="Give up after this many steps."),
    percent: int = typer.Option(70, "--percent", help="Travel per step as a % of the container."),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Scroll until something is on screen, verifying every step actually moved.

    `aua scroll-to-and-analyze "Red Square Tap"` · `aua scroll-to-and-analyze --rid listRow_7`.
    ``detail`` starts with the outcome: ``already-visible`` · ``moved`` (with ``dy``) ·
    ``already-at-end`` (nothing scrolled, so it is not on this screen) ·
    ``target-not-found`` (scrolled the whole way, never saw it). The last two exit 6, so a
    miss is no longer reported as a silent success.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        query = rid or text
        if not query:
            raise UsageError(
                "scroll-to needs the text to look for (or --rid)",
                hint='e.g. `aua scroll-to-and-analyze "Red Square Tap"`',
            )
        result = _route(
            engine,
            "scroll_to",
            query=query,
            match=match,
            ignore_case=ignore_case,
            observe=observe,
            by="id" if rid else by,
            direction=direction,
            max_swipes=max_swipes,
            percent=percent,
            with_image=_annotate_arg(with_image),
        )
        _emit(result, fmt)
        _exit_unless_ok(
            result,
            ExitCode.NOT_FOUND,
            code="selector_not_found",
            hint="`already-at-end` means the screen never scrolled (it is not there); "
            "`target-not-found` means it scrolled the whole way without finding it.",
        )

    _run(ctx, go)


@app.command(name="expect-and-analyze", cls=AnalyzeCommand)
def expect(
    ctx: typer.Context,
    rid: str | None = typer.Option(None, "--rid", help="Assert about this resource-id."),
    text: str | None = typer.Option(None, "--text", help="Assert about this label."),
    desc: str | None = typer.Option(None, "--desc", help="Assert about this content-desc."),
    exists: bool = typer.Option(False, "--exists", help="It must be on screen (the default)."),
    absent: bool = typer.Option(False, "--absent", help="It must NOT be on screen."),
    text_is: str | None = typer.Option(None, "--text-is", help="Its label must equal this."),
    text_contains: str | None = typer.Option(
        None, "--text-contains", help="Its label must contain this."
    ),
    checked: bool | None = typer.Option(
        None, "--checked/--unchecked", help="Toggle/checkbox state."
    ),
    enabled: bool | None = typer.Option(None, "--enabled/--disabled", help="Enabled state."),
    selected: bool | None = typer.Option(
        None, "--selected/--unselected", help="Selected state (tabs)."
    ),
    focused: bool | None = typer.Option(None, "--focused/--unfocused", help="Focus state."),
    count: int | None = typer.Option(
        None, "--count", min=0, help="Exactly this many elements must match."
    ),
    within: str | None = typer.Option(
        None,
        "--within",
        help="Keep matches descended from this selector, e.g. 'rid:card'.",
    ),
    same_parent_as: str | None = typer.Option(
        None,
        "--same-parent-as",
        help="Require the same canonical parent as this selector.",
    ),
    contains: list[str] | None = typer.Option(
        None,
        "--contains",
        help="Require a matching descendant; repeat for contains_all.",
    ),
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    timeout: int = typer.Option(
        0,
        "--timeout",
        "--timeout-ms",
        help="Poll until it holds, up to this many ms (capped by perf.max_wait_ms).",
    ),
) -> None:
    """Assert one thing about the screen. Exit 0 = pass, 8 = the assertion failed.

    `aua expect-and-analyze --rid notificationsButton --exists` ·
    `aua expect-and-analyze --text "Loading" --absent --timeout 5000` ·
    `aua expect-and-analyze --rid itemDetailLikeCount --text-is "7"` ·
    `aua expect-and-analyze --rid settingsPushToggleSwitch --checked`

    One acceptance criterion per call, so a criteria list becomes a script instead of a pile
    of eyeballed screenshots. Exit 8 is a *test* failure and stays distinct from 3 (device)
    and 6 (a selector that matches nothing). On failure stderr says what was sought, what
    was actually there, and the nearest candidates. ``--timeout`` polls — it is what
    replaces a ``sleep`` guess.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "expect",
            rid=rid,
            text=text,
            desc=desc,
            exists=exists,
            absent=absent,
            text_is=text_is,
            text_contains=text_contains,
            checked=checked,
            enabled=enabled,
            selected=selected,
            focused=focused,
            count=count,
            within=(parse_selector_expression(within, field="--within") if within else None),
            same_parent_as=(
                parse_selector_expression(same_parent_as, field="--same-parent-as")
                if same_parent_as
                else None
            ),
            contains_all=[
                parse_selector_expression(value, field="--contains") for value in (contains or ())
            ]
            or None,
            index=index,
            first=first,
            timeout_ms=timeout,
        )
        _emit(result, fmt)
        _exit_unless_ok(result, ExitCode.ASSERTION, code="expectation_failed")

    _run(ctx, go)


@app.command(name="key-and-analyze", cls=AnalyzeCommand)
def key(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="back|home|enter|recents|KEYCODE_*."),
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="Also return the screen after the key (skips a follow-up analyze).",
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Press a hardware/navigation key."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "key",
                name=name,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="back-until-and-analyze", cls=AnalyzeCommand)
def back_until_cmd(
    ctx: typer.Context,
    predicate: str = typer.Argument(
        ...,
        help=(
            "Mapped known_screen name or destination evidence, e.g. 'home', 'rid:bottomNav', "
            "or 'text:Grammar,text:Mathematics'."
        ),
    ),
    max_steps: int = typer.Option(
        4,
        "--max-steps",
        min=1,
        max=12,
        help="Maximum semantic Back/navigation-up steps before returning unmet evidence.",
    ),
    max_back: int | None = typer.Option(
        None,
        "--max-back",
        min=1,
        max=12,
        help="Deprecated alias for --max-steps; use --max-steps in new calls.",
    ),
    step_timeout_ms: int = typer.Option(
        1_200,
        "--step-timeout",
        min=0,
        help="Milliseconds to wait for the destination after each Back press.",
    ),
    poll_ms: int = typer.Option(
        200,
        "--poll",
        min=10,
        help="Milliseconds between semantic destination checks.",
    ),
    back_id: int | None = typer.Option(
        None,
        "--back-id",
        min=0,
        help="Fresh frame-local id for an unlabeled Back control; used only for the first step.",
    ),
    back_rid: str | None = typer.Option(
        None,
        "--back-rid",
        help="Stable resource id for the app-owned Back control, re-resolved on each frame.",
    ),
    back_text: str | None = typer.Option(
        None,
        "--back-text",
        help="Exact visible text for the app-owned Back control, re-resolved on each frame.",
    ),
    back_desc: str | None = typer.Option(
        None,
        "--back-desc",
        help="Exact content description for Back, re-resolved on each frame.",
    ),
) -> None:
    """Return from nested screens, stopping on a mapped screen or semantic evidence."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        effective_max_steps = max_steps
        if max_back is not None:
            source = getattr(ctx.get_parameter_source("max_steps"), "name", None)
            if source == "COMMANDLINE":
                raise UsageError(
                    "pass only one of --max-steps or deprecated --max-back",
                    hint="Use `--max-steps N`; it is the canonical spelling.",
                )
            typer.echo("warning: --max-back is deprecated; use --max-steps.", err=True)
            effective_max_steps = max_back
        if _UNTIL is not None:
            raise UsageError(
                "back-until owns its destination predicate; do not combine it with global --until",
                hint="Pass the destination as the command argument, for example "
                "`aua back-until-and-analyze 'rid:bottomNav'`.",
            )
        supplied = {"rid": back_rid, "text": back_text, "desc": back_desc}
        selector = {key: value for key, value in supplied.items() if value}
        if len(selector) > 1 or (back_id is not None and selector):
            raise UsageError(
                "choose only one of --back-id, --back-rid, --back-text, or --back-desc"
            )
        result = _route(
            engine,
            "back_until",
            predicate=predicate,
            back_id=back_id,
            back_selector=selector or None,
            max_steps=effective_max_steps,
            step_timeout_ms=step_timeout_ms,
            poll_ms=poll_ms,
        )
        _emit(result, fmt)
        ok = result.get("ok") if isinstance(result, dict) else getattr(result, "ok", None)
        if ok is False:
            raise typer.Exit(1)

    _run(ctx, go)


@app.command(name="hide-keyboard-and-analyze", cls=AnalyzeCommand)
def hide_keyboard(
    ctx: typer.Context,
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="Also return the screen after dismissing the keyboard.",
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Dismiss the soft keyboard (prefer this over ``key back`` when the IME is up)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "hide_keyboard",
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="open-and-analyze", cls=AnalyzeCommand)
def open(  # noqa: A001 - matches the user-facing verb `aua open-and-analyze`
    ctx: typer.Context,
    uri: str = typer.Argument(..., help="Deeplink URI, e.g. 'myapp://set-flags?flag=value'."),
    app_pkg: str | None = typer.Option(
        None,
        "--package",
        "--app",
        help="Pin the VIEW intent to this package (default: foreground app). Skips 'Open with…'.",
    ),
    prefer: str | None = typer.Option(
        None,
        "--prefer",
        help="If a chooser still appears, auto-pick the row matching this package/label.",
    ),
    pin_package: bool = typer.Option(
        True,
        "--package-pin/--no-package-pin",
        help="Pin to foreground/known package by default; --no-package-pin exercises the chooser.",
    ),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the screen after opening."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the raw screenshot; bare flag uses a timestamped default path.",
        show_default=False,
    ),
) -> None:
    """Open a deeplink — jump straight to a screen or trigger an app action (latency shortcut).

    Pins the target package by default (foreground app, or ``--package``) so emulators
    with both prod + dev installs never hit Android's "Open with…" chooser. If a chooser
    still appears, aua errors with the competing app names rather than leaving you there.
    Use ``--no-package-pin`` only when you intentionally want to test the chooser.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "open_link",
                uri=uri,
                package=app_pkg,
                prefer=prefer,
                pin_package=pin_package,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@app.command()
def resolve(
    ctx: typer.Context,
    target: str = typer.Argument(
        ...,
        help="Previous-frame element id (integer) or a stable_key (e.g. rid:continue_btn).",
    ),
) -> None:
    """Remap a prior id or ``stable_key`` onto the current screen (cross-frame binding).

    Integer ids are rewritten every analyze; ``stable_key`` survives. Use this after a
    state-changing action when you still hold an old id.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.resolve(target)
        _emit(result, fmt)

    _run(ctx, go)


@app.command(name="wait-and-analyze", cls=AnalyzeCommand)
def wait(
    ctx: typer.Context,
    for_: str | None = typer.Option(None, "--for", help="Text/resource-id to wait for."),
    idle: bool = typer.Option(False, "--idle", help="Wait for the UI to go idle."),
    for_stable: bool = typer.Option(
        False, "--for-stable", help="Wait until the screen stops visually changing."
    ),
    changed: bool = typer.Option(
        False,
        "--changed",
        help="Wait until the hierarchy fingerprint changes (any UI tree change).",
    ),
    after_change: bool = typer.Option(
        False,
        "--after-change",
        help="Wait for the screen to CHANGE and then settle. Use for network-driven content "
        "(AI replies, image generation): plain --for-stable can return before anything starts.",
    ),
    interval: int = typer.Option(
        120, "--interval", help="--for-stable/--changed: poll interval ms."
    ),
    settle: int = typer.Option(
        200, "--settle", help="--for-stable: ms of no (non-animated) change to settle."
    ),
    timeout: int | None = typer.Option(
        None,
        "--timeout",
        "--timeout-ms",
        help="Timeout in ms (default 5000; 30000 for --for-stable; 15000 for --changed). "
        "Capped by perf.max_wait_ms — a clamped wait says so and you call again.",
    ),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
    observe: bool = typer.Option(
        False,
        "--observe",
        help="Also return the (settled) screen with fresh ids — act on it without a re-analyze.",
    ),
    by: str = typer.Option(
        "text", "--by", help="--for match by: text (default) | id/rid (resource-id) | desc."
    ),
    absent: bool = typer.Option(
        False, "--absent", help="With --for: wait until it DISAPPEARS (loading spinners, dialogs)."
    ),
    no_meta: bool = typer.Option(
        False,
        "--no-meta",
        help="Omit meta from the returned observation (same meaning as analyze).",
    ),
) -> None:
    """Wait for text to appear (or with ``--absent`` disappear), for idle, or for settle.

    FIRST: if you are waiting on your own last action, you do not need this command. Pass the
    global ``--until`` to the action — `aua tap-and-analyze --rid send --until 'text:Sent'` —
    and it waits and returns the settled screen in one call, reporting ``await_outcome`` so
    "arrived" is distinguishable from "timed out". Reach for ``wait-and-analyze`` when there is
    no action to attach to, or when you need ``--for-stable``/``--after-change`` semantics.

    ``--for-stable`` polls cheap screenshots (a perceptual-hash "settled" check — no OCR,
    no hierarchy parse; works on opaque screens) and returns once the screen stops changing
    for ``--settle`` ms. ``--changed`` waits for any hierarchy-tree change (host-polled
    stand-in for a11y event push). ``--observe`` folds in the post-wait screen so you can act
    on what you waited for in one fewer call.

    For anything network-driven — an AI reply, image generation, a slow load — use
    ``--after-change``, NOT ``--for-stable`` on its own. Immediately after you tap send the
    screen has not started changing yet, so it is already "stable" and ``--for-stable``
    returns at once (measured: 1.2s, before the reply existed) leaving you to discover the
    emptiness and wait again. ``--after-change`` waits for the first change, then for the
    screen to settle (measured on the same reply: 22s, and the answer was there).
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _set_no_meta_observation_view(fmt, no_meta)
        _warn_if_wait_could_have_been_until(engine, for_)
        global _UNTIL
        if not (for_ or idle or for_stable or after_change or changed) and _UNTIL:
            # `--until` is the global "then wait for this" that every action takes, and on a
            # wait command it IS the wait — the same concept in a richer predicate language
            # (`!text:`, `rid:`, comma-separated terms) than `--for`. Refusing it with "wait
            # needs --for <text> or --idle" reads as "--until is not a thing here", which is
            # false: it parsed, and it names exactly what the caller is waiting for.
            predicate, timeout_ms, poll_ms = _UNTIL
            _UNTIL = None  # this IS the wait; the post-action pass must not repeat it
            _emit(
                _route(
                    engine,
                    "await_predicate",
                    predicate=predicate,
                    timeout_ms=timeout_ms,
                    poll_ms=poll_ms,
                    observe=observe,
                ),
                fmt,
            )
            return
        if after_change:
            # One bounded engine contract owns all three phases: first change, visual settle,
            # then a quiet confirmation window that catches a result arriving after a stable
            # loading shell. Keeping it in the engine also makes daemon and in-process calls
            # behave identically.
            _emit(
                _route(
                    engine,
                    "wait_after_change",
                    interval_ms=interval,
                    settle_ms=settle if settle != 200 else 1200,
                    timeout_ms=timeout if timeout is not None else 60000,
                    observe=observe,
                ),
                fmt,
            )
            return
        if for_stable:
            eff = timeout if timeout is not None else 30000
            _emit(
                _route(
                    engine,
                    "wait_stable",
                    interval_ms=interval,
                    settle_ms=settle,
                    timeout_ms=eff,
                    observe=observe,
                ),
                fmt,
            )
            return
        if changed:
            eff = timeout if timeout is not None else 15000
            result = _route(
                engine,
                "wait_changed",
                timeout_ms=eff,
                interval_ms=interval,
                observe=observe,
            )
            _emit(result, fmt)
            return
        eff = timeout if timeout is not None else 5000
        result = _route(
            engine,
            "wait",
            for_=for_,
            idle=idle,
            timeout_ms=eff,
            match=match,
            ignore_case=ignore_case,
            observe=observe,
            by=by,
            absent=absent,
        )
        _emit(result, fmt)
        # Misses used to exit 0 with ok:false — silent and costly to debug. Non-zero now,
        # with detail naming match mode / fields / closest candidates.
        if not idle:
            _exit_unless_ok(
                result,
                ExitCode.DEVICE,
                code="wait_timeout",
                hint="Check --match (contains vs regex), --by, and the closest candidates in detail.",
            )

    _run(ctx, go)


# --------------------------------------------------------------------------- navigate


@app.command()
def goto(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Target screen/goal (fuzzy match against memory)."),
    plan: bool = typer.Option(False, "--plan", help="Print the route only; do not act."),
    max_steps: int = typer.Option(8, "--max-steps", help="Max hops before handing off."),
    allow_destructive: bool = typer.Option(
        False,
        "--allow-destructive",
        help="Replay steps whose label matches memory.destructive_labels (delete/sign out/…).",
    ),
    allow_unsafe: bool = typer.Option(
        False,
        "--allow-unsafe",
        help="After reviewing the risk preview, permit deeplinks, external packages, "
        "settings/data/environment mutation, lifecycle changes, and other non-navigation steps.",
    ),
    assist: bool = typer.Option(
        False,
        "--assist",
        help="On divergence, let the opt-in planner LLM try to recover (needs planner.enabled).",
    ),
    from_here: bool = typer.Option(
        False,
        "--from-here",
        help="Resume mid-edge: you already opened part of the journey; skip steps that "
        "already match the current screen (same idea as mid-auth resume).",
    ),
) -> None:
    """Navigate to a known screen using app memory — drives and verifies each hop (§6b).

    Resolves the goal against the learned map, then replays each edge's recorded steps
    along the shortest route from the current screen, confirming ``known_screen`` after
    every hop. Stops and returns the remaining route/steps + current screen if anything
    diverges. ``--plan`` prints every step and its risk without acting. Destructive steps are
    refused without ``--allow-destructive``; deeplinks, external/settings/data/lifecycle and
    other non-navigation effects are refused without ``--allow-unsafe``. A refusal happens
    before the first route action. ``--assist`` lets a fast model recover a divergence.
    ``--from-here`` resumes mid-edge when you already navigated part of the way yourself.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "goto",
            goal=goal,
            plan=plan,
            max_steps=max_steps,
            allow_destructive=allow_destructive,
            allow_unsafe=allow_unsafe,
            assist=assist,
            from_here=from_here,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


@app.command()
def navigate(
    ctx: typer.Context,
    goal: str = typer.Argument(
        ..., help="Natural-language destination, e.g. 'open the image generator'."
    ),
    until: str | None = typer.Option(
        None, "--until", help="Stop when this text appears (a deterministic arrival check)."
    ),
    max_steps: int = typer.Option(12, "--max-steps", help="Max planner actions."),
    allow_destructive: bool = typer.Option(
        False, "--allow-destructive", help="Permit destructive taps (delete/sign out/…)."
    ),
    save_flow: str | None = typer.Option(
        None, "--save-flow", help="Also save the taken path as a reusable flow with this name."
    ),
) -> None:
    """Drive to a goal with the opt-in planner LLM, recording the path for a free replay.

    Needs `planner.enabled` (this command is the explicit opt-in). The planner chooses
    each action; because they run through the normal actions, the journey is learned into
    memory — next time, `aua goto <that screen>` replays it deterministically. `--save-flow`
    also writes the path as a YAML flow.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "navigate",
            goal=goal,
            until=until,
            max_steps=max_steps,
            allow_destructive=allow_destructive,
            save_flow=save_flow,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- goal sessions


session_app = typer.Typer(
    help="Goal-aware bootstrap, efficiency review, and reversible cleanup.",
    no_args_is_help=True,
)
app.add_typer(session_app, name="session")


@session_app.command("start")
def session_start_cmd(
    ctx: typer.Context,
    goal: str = typer.Option(..., "--goal", help="The end-to-end Android verification goal."),
    contract: Path | None = typer.Option(
        None,
        "--contract",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Authored YAML checkpoints that only fresh UI assertions can complete.",
    ),
    contract_yaml: str | None = typer.Option(
        None,
        "--contract-yaml",
        help="Inline authored checkpoint YAML (MCP-equivalent input).",
    ),
    artifacts_dir: Path | None = typer.Option(
        None,
        "--artifacts-dir",
        help="Portable cross-command session evidence bundle directory.",
    ),
    evidence: str = typer.Option(
        "failures",
        "--evidence",
        help="Observation/screenshot capture: none|failures|all (needs --artifacts-dir).",
    ),
    junit: bool = typer.Option(False, "--junit", help="Write JUnit XML in the artifact bundle."),
    wait_for_lease: float = typer.Option(
        0,
        "--wait-for-lease",
        min=0,
        help="Wait up to this many seconds for the requested/eligible device; never steal.",
    ),
    start_emulator: bool = typer.Option(
        True,
        "--start-emulator/--no-start-emulator",
        help="Automatically boot a compatible AVD when no matching unleased device is online.",
    ),
    headed: bool = typer.Option(
        False,
        "--headed",
        help=(
            "Require a visible emulator for this session; when AUA boots one, "
            "show its window instead of using headless mode."
        ),
    ),
    audio: bool = typer.Option(
        False,
        "--audio/--no-audio",
        help=(
            "When AUA boots one, keep host audio enabled so microphone "
            "injection is available."
        ),
    ),
    avd: str | None = typer.Option(None, "--avd", help="AVD name when several are configured."),
    needs: str | None = typer.Option(
        None,
        "--needs",
        help="Required target capabilities: root, play, proxy. AUA selects or boots a match.",
    ),
    package: str | None = typer.Option(
        None,
        "--app",
        "--package",
        help="Launch and observe this package before planning (aliases are equivalent).",
    ),
    activity: str | None = typer.Option(
        None,
        "--activity",
        help="With --app/--package, pin a launcher Activity.",
    ),
    apk: str | None = typer.Option(
        None,
        "--apk",
        help="Install this .apk before launching (skipped when that version is already there). "
        "The bundle names the package, so --app is optional with it.",
    ),
    reinstall: bool = typer.Option(
        False, "--reinstall", help="With --apk, push the bundle even if that version is present."
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="With --apk, uninstall any existing copy first (wipes app data). Needs --yes.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Required for --fresh."),
) -> None:
    """Observe once and return the safest exact CLI and MCP next call.

    With ``--apk <path>`` this is the entire bootstrap in one call: choose or boot a compatible
    target, lease it, install the build, launch it, observe, and return the goal plan.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        if (reinstall or fresh) and not apk:
            raise UsageError(
                "--reinstall/--fresh only apply to --apk",
                hint="Add `--apk <path>`, or drop the flag.",
            )
        result = _route(
            engine,
            "session_start",
            goal=goal,
            contract_file=str(contract.resolve()) if contract is not None else None,
            contract_yaml=contract_yaml,
            artifacts_dir=str(artifacts_dir.expanduser().resolve()) if artifacts_dir else None,
            evidence=evidence,
            junit=junit,
            wait_for_lease_s=wait_for_lease,
            start_emulator=start_emulator,
            headed=headed,
            audio=audio,
            avd=avd,
            needs=(
                _split_needs(needs)
                if needs is not None
                else list(getattr(engine, "_lease_needs", None) or [])
            ),
            package=package,
            activity=activity,
            apk=apk,
            reinstall=reinstall,
            fresh=fresh,
            confirmed=yes,
        )
        if isinstance(result, dict) and _OBSERVATION_VIEW is not None:
            result = trim_observation_payload(result, _OBSERVATION_VIEW, fmt=fmt)
        _emit(result, fmt)

    _run(ctx, go)


@session_app.command("review")
def session_review_cmd(
    ctx: typer.Context,
    session_id: str | None = typer.Option(None, "--session-id", help="Review a prior session."),
) -> None:
    """Report exact top-level/task/lifecycle calls, folded events, failures, and savings."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "session_review", session_id=session_id), fmt)

    _run(ctx, go)


@session_app.command("progress")
def session_progress_cmd(
    ctx: typer.Context,
    session_id: str | None = typer.Option(None, "--session-id", help="Inspect a prior session."),
) -> None:
    """Return ordered goal phases and the current phase's exact next call."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "session_progress", session_id=session_id), fmt)

    _run(ctx, go)


@session_app.command("autopilot")
def session_autopilot_cmd(
    ctx: typer.Context,
    session_id: str | None = typer.Option(None, "--session-id", help="Continue a prior session."),
    max_steps: int = typer.Option(
        6,
        "--max-steps",
        min=1,
        max=20,
        help="Maximum guard-approved local actions before handing control back.",
    ),
    max_duration_ms: int = typer.Option(
        30_000,
        "--max-duration-ms",
        min=1_000,
        max=300_000,
        help="Wall-time budget for the complete local policy stretch.",
    ),
) -> None:
    """Let the warm local policy execute a bounded safe navigation stretch."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "session_autopilot",
            session_id=session_id,
            max_steps=max_steps,
            max_duration_ms=max_duration_ms,
        )
        if isinstance(result, dict) and _OBSERVATION_VIEW is not None:
            result = trim_observation_payload(result, _OBSERVATION_VIEW, fmt=fmt)
        _emit(result, fmt)

    _run(ctx, go)


@session_app.command("finish")
def session_finish_cmd(
    ctx: typer.Context,
    session_id: str | None = typer.Option(None, "--session-id", help="Finish a prior session."),
    allow_incomplete: bool = typer.Option(
        False,
        "--allow-incomplete",
        help="Explicitly terminate even though authored checkpoints are incomplete.",
    ),
) -> None:
    """Restore only session-owned reversible state and return the final review."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "session_finish",
            session_id=session_id,
            allow_incomplete=allow_incomplete,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and not result.get("ok", False):
            raise typer.Exit(1)

    _run(ctx, go)


@session_app.command("candidate-flow")
def session_candidate_flow_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name for the candidate and eventual saved flow."),
    session_id: str | None = typer.Option(None, "--session-id"),
    reset_flow: str | None = typer.Option(
        None,
        "--reset-flow",
        help="Saved flow name or YAML file that deterministically restores state before replay.",
    ),
    replay: bool = typer.Option(False, "--replay", help="Reset and replay the candidate."),
    save: bool = typer.Option(
        False,
        "--save",
        help="Reset, replay successfully, then save without overwriting an existing flow.",
    ),
) -> None:
    """Preview a proven session path; replay with explicit reset before promotion."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        reset_value = reset_flow
        if reset_flow:
            reset_path = Path(reset_flow).expanduser()
            if reset_path.is_file():
                reset_value = str(reset_path.resolve())
        result = _route(
            engine,
            "session_candidate_flow",
            name=name,
            session_id=session_id,
            reset_flow=reset_value,
            replay=replay,
            save=save,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- durable jobs


job_app = typer.Typer(
    help="Run long read-only waits with reconnectable status and cancellation.",
    no_args_is_help=True,
)
app.add_typer(job_app, name="job")


@job_app.command("start")
def job_start_cmd(
    ctx: typer.Context,
    operation: str = typer.Argument(
        ...,
        help="await|wait-stable|wait-changed|wait-after-change",
    ),
    predicate: str | None = typer.Option(
        None,
        "--predicate",
        help="Semantic predicate for await, e.g. 'rid:result,!text:Loading'.",
    ),
    timeout_ms: int | None = typer.Option(None, "--timeout-ms", help="Complete job budget."),
    poll_ms: int = typer.Option(120, "--poll-ms", help="Milliseconds between device reads."),
    settle_ms: int = typer.Option(1_200, "--settle-ms", help="Visual settle duration."),
    confirmation_ms: int = typer.Option(
        1_800,
        "--confirmation-ms",
        help="Quiet confirmation after a late screen change.",
    ),
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="Attach a fresh analyzed screen to a successful terminal result.",
    ),
) -> None:
    """Start one durable wait and immediately return its job id."""
    normalized = operation.casefold().replace("_", "-")
    defaults = {
        "await": 60_000,
        "wait-stable": 30_000,
        "wait-changed": 15_000,
        "wait-after-change": 60_000,
    }

    def go(engine: Engine, fmt: OutputFormat) -> None:
        effective_timeout = timeout_ms if timeout_ms is not None else defaults.get(normalized, 0)
        args: dict[str, Any] = {
            "operation": normalized,
            "timeout_ms": effective_timeout,
            "poll_ms": poll_ms,
            "observe": observe,
        }
        if predicate is not None:
            args["predicate"] = predicate
        if normalized in {"wait-stable", "wait-after-change"}:
            args["settle_ms"] = settle_ms
        if normalized == "wait-after-change":
            args["confirmation_ms"] = confirmation_ms
        _emit(_route(engine, "job_start", **args), fmt)

    _run(ctx, go)


@job_app.command("status")
def job_status_cmd(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    recent_output: bool = typer.Option(
        False,
        "--recent-output",
        help="Include persisted lifecycle events and the latest result/error.",
    ),
) -> None:
    """Reconnect to a job without restarting its wait."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(engine, "job_status", job_id=job_id, recent_output=recent_output),
            fmt,
        )

    _run(ctx, go)


@job_app.command("wait")
def job_wait_cmd(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    timeout_ms: int = typer.Option(
        5_000,
        "--timeout-ms",
        min=0,
        max=10_000,
        help="Bound this reconnect call; the underlying job keeps running.",
    ),
) -> None:
    """Wait briefly for completion; return running state when the call budget expires."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "job_wait", job_id=job_id, timeout_ms=timeout_ms), fmt)

    _run(ctx, go)


@job_app.command("cancel")
def job_cancel_cmd(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    wait_ms: int = typer.Option(
        1_000,
        "--wait-ms",
        min=0,
        max=10_000,
        help="Briefly wait for a terminal cancellation acknowledgement.",
    ),
) -> None:
    """Cancel and briefly wait for the worker's safe device-read boundary."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "job_cancel", job_id=job_id, wait_ms=wait_ms), fmt)

    _run(ctx, go)


@job_app.command("list")
def job_list_cmd(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", min=1, max=100),
) -> None:
    """List recent jobs visible to the current owner."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "job_list", limit=limit), fmt)

    _run(ctx, go)


@app.command("reach")
def reach_cmd(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Natural-language destination."),
    until: str | None = typer.Option(
        None,
        "--until",
        help="Semantic arrival evidence, e.g. 'rid:result,!text:Loading'.",
    ),
    timeout_ms: int = typer.Option(30_000, "--timeout-ms", help="Arrival evidence timeout."),
    poll_ms: int = typer.Option(300, "--poll-ms", help="Arrival evidence polling interval."),
    allow_unsafe: bool = typer.Option(
        False, "--allow-unsafe", help="Permit disclosed unsafe effects."
    ),
    allow_destructive: bool = typer.Option(
        False, "--allow-destructive", help="Permit disclosed destructive effects."
    ),
    assist: bool = typer.Option(False, "--assist", help="Permit configured planner recovery."),
) -> None:
    """Reach a goal through verified goto, then a matching safe flow, with arrival proof."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        if until:
            _parse_await_terms(until, require_positive=True)
        result = _route(
            engine,
            "reach",
            goal=goal,
            until=until,
            timeout_ms=timeout_ms,
            interval_ms=poll_ms,
            allow_unsafe=allow_unsafe,
            allow_destructive=allow_destructive,
            assist=assist,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and not result.get("ok", False):
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- device/session


@app.command()
def devices(ctx: typer.Context) -> None:
    """List attached devices (serial, model, platform version, locale, state)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        infos = engine.list_devices()
        payload = [d.model_dump(mode="json") for d in infos]
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        typer.echo(json.dumps(payload, indent=indent, separators=sep, ensure_ascii=False))

    _run(ctx, go)


@app.command()
def fanout(
    ctx: typer.Context,
    command: list[str] = typer.Argument(..., help="Subcommand + args, e.g. analyze or has Sign."),
    serials: str | None = typer.Option(
        None,
        "--serials",
        help="Comma-separated serials (default: every online device from `aua devices`).",
    ),
    parallel: bool = typer.Option(True, "--parallel/--serial", help="Run devices concurrently."),
) -> None:
    """Run one aua subcommand on many devices and gather JSON results (phase 5).

    Each serial gets its own warm-daemon socket (``daemon.sock.<serial>``). Example::

        aua fanout --serials emulator-5554,emulator-5556 analyze
        aua fanout has "Sign in"
    """
    import json
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from . import leases as lease_mod

    opts = _opts(ctx)
    base_owner = lease_mod.resolve_owner(opts.owner)
    base_held = set(lease_mod.primary_held_by(opts.load().lease.registry_dir, base_owner))
    targets = [s.strip() for s in (serials or "").split(",") if s.strip()]
    if not targets:
        targets = [
            d.serial
            for d in opts.engine().list_devices()
            if getattr(d, "state", "device") == "device"
        ]
    if not targets:
        raise UsageError(
            "no devices for fanout",
            hint="Attach emulators or pass --serials emulator-5554,emulator-5556.",
        )

    def one(ser: str) -> dict[str, Any]:
        scoped_owner = (
            str(base_owner) if ser in base_held else f"{base_owner}:fanout:{ser}"
        )
        cmd = [
            "aua",
            "--owner",
            scoped_owner,
            "--serial",
            ser,
            "--format",
            "compact",
            *command,
        ]
        if opts.platform:
            cmd[1:1] = ["--platform", opts.platform]
        if opts.config:
            cmd[1:1] = ["--config", opts.config]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {"serial": ser, "ok": False, "error": "timeout"}
        out = proc.stdout.strip()
        payload: Any
        try:
            payload = json.loads(out) if out else None
        except json.JSONDecodeError:
            payload = out
        return {
            "serial": ser,
            "lease_owner": scoped_owner,
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "result": payload,
            "stderr": proc.stderr.strip() or None,
        }

    results: list[dict[str, Any]] = []
    if parallel and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
            futs = {pool.submit(one, s): s for s in targets}
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda r: targets.index(r["serial"]))
    else:
        results = [one(s) for s in targets]
    typer.echo(json.dumps({"ok": all(r["ok"] for r in results), "devices": results}, indent=2))
    if not all(r["ok"] for r in results):
        raise typer.Exit(1)


emulator_app = typer.Typer(
    help="Boot / list / stop Android AVDs (headless by default for unattended agent runs).",
    no_args_is_help=True,
)
app.add_typer(emulator_app, name="emulator")


def _emulator_emit(payload: dict[str, Any], ctx: typer.Context) -> None:
    import json

    opts = _opts(ctx)
    fmt = (opts.format or "json").lower()
    indent = 2 if fmt == "pretty" else None
    sep = None if indent else (",", ":")
    typer.echo(json.dumps(payload, indent=indent, separators=sep, ensure_ascii=False))


_EMULATOR_PROGRESS_INTERVAL_S = 10.0


@contextlib.contextmanager
def _emulator_start_progress(
    *,
    avd: str | None,
    headless: bool,
    wait_s: float,
    action: str = "emulator-start",
) -> Iterator[None]:
    """Keep a long emulator bootstrap visibly alive without polluting result stdout.

    Agent shells commonly yield control while a command is still running. With no output,
    that non-terminal yield looks enough like a completed empty command that a caller may start
    a duplicate and then broadly clean up both. Progress is therefore unconditional, sent to
    stderr, and says exactly how to continue. The final machine result remains the one JSON
    document on stdout.
    """
    import json

    started = time.monotonic()
    stopped = threading.Event()
    interval = max(0.01, float(_EMULATOR_PROGRESS_INTERVAL_S))

    def emit(stage: str, message: str) -> None:
        payload = {
            "action": action,
            "stage": stage,
            "avd": avd or "auto",
            "mode": "headless" if headless else "windowed",
            "elapsed_s": round(max(0.0, time.monotonic() - started), 1),
            "message": message,
        }
        with contextlib.suppress(OSError):
            typer.echo(
                "AUA_PROGRESS " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                err=True,
            )

    emit(
        "starting",
        f"command is running and may take up to {int(wait_s)}s; if your runner yields, "
        "poll the same process and do not start another emulator",
    )

    def heartbeat() -> None:
        while not stopped.wait(interval):
            emit(
                "waiting",
                "command is still running; keep polling the same process and do not retry",
            )

    thread = threading.Thread(
        target=heartbeat,
        name="aua-emulator-start-progress",
        daemon=True,
    )
    thread.start()
    outcome = "failed"
    try:
        yield
        outcome = "completed"
    except KeyboardInterrupt:
        outcome = "cancelled"
        raise
    finally:
        stopped.set()
        thread.join(timeout=min(0.2, interval))
        if outcome == "completed":
            emit("completed", "startup completed; the final result follows on stdout")
        elif outcome == "cancelled":
            emit("cancelled", "startup was interrupted; no final result will be emitted")
        else:
            emit("failed", "startup failed; the structured error follows on stderr")


@emulator_app.command("list")
def emulator_list_cmd(ctx: typer.Context) -> None:
    """List configured AVDs (marks Play Store vs rootable Google APIs)."""
    emulator_mod = _platform_capability(ctx, "virtual_devices")

    try:
        _emulator_emit(emulator_mod.list_avds(), ctx)
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@emulator_app.command("recommend-proxy")
def emulator_recommend_proxy_cmd(
    ctx: typer.Context,
    name: str = typer.Option("aua_proxy", "--name", help="Suggested AVD name for ensure-proxy."),
    api: int = typer.Option(
        30,
        "--api",
        help="API level for the google_apis system image (30 = small/fast default).",
    ),
) -> None:
    """Suggest a small rootable Google APIs AVD for HTTPS proxy / system CA install.

    Google Play images refuse `adb root`, so mitm system-CA install fails. This prints
    the package + commands; does not download or create anything.
    """
    emulator_mod = _platform_capability(ctx, "virtual_devices")

    try:
        _emulator_emit(emulator_mod.recommend_proxy_avd(api=api, name=name), ctx)
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@emulator_app.command("ensure-proxy")
def emulator_ensure_proxy_cmd(
    ctx: typer.Context,
    name: str = typer.Option("aua_proxy", "--name", help="AVD name to create or reuse."),
    api: int = typer.Option(
        30,
        "--api",
        help="API level (google_apis, not Play Store). Lower = smaller/faster.",
    ),
    force: bool = typer.Option(False, "--force", help="Recreate even if the AVD already exists."),
    start_after: bool = typer.Option(
        False,
        "--start/--no-start",
        help="Boot the AVD headless after create/reuse.",
    ),
    wait: int = typer.Option(180, "--wait", help="Seconds to wait for adb when --start is set."),
) -> None:
    """Install a small google_apis system image (if needed) and create a rootable AVD.

    Needed for `aua proxy` HTTPS capture when the app only trusts system CAs. Prefer this
    over Google Play AVDs — those block `adb root`. Downloads can take several minutes.
    """
    opts = _opts(ctx)
    cfg = opts.load()
    emulator_mod = _platform_capability(ctx, "virtual_devices")
    try:
        payload = emulator_mod.ensure_proxy_avd(name=name, api=api, force=force)
        if start_after:
            with _emulator_start_progress(avd=name, headless=True, wait_s=float(wait)):
                boot = emulator_mod.start(
                    name,
                    headless=True,
                    wait_s=float(wait),
                    cache_dir=cfg.cache.dir,
                    lease_registry_dir=cfg.lease.registry_dir,
                )
            payload["started"] = boot
            payload["hint"] = (
                f"Booted {boot.get('serial')}. Start goal work with "
                "`aua session start --goal <goal> --needs root,proxy`; AUA selects and leases "
                "automatically. Then run `aua proxy start` without a serial."
            )
        _emulator_emit(payload, ctx)
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@emulator_app.command("status")
def emulator_status_cmd(ctx: typer.Context) -> None:
    """SDK / AVD tooling + currently running emulator-* serials."""
    opts = _opts(ctx)
    cfg = opts.load()
    emulator_mod = _platform_capability(ctx, "virtual_devices")
    try:
        _emulator_emit(emulator_mod.status(cache_dir=cfg.cache.dir), ctx)
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@emulator_app.command("start")
def emulator_start_cmd(
    ctx: typer.Context,
    avd: str | None = typer.Option(
        None, "--avd", help="AVD name (`aua emulator list`). Default: the only configured AVD."
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--windowed",
        help="Headless (-no-window) so agents can verify without stealing the desktop UI.",
    ),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        help="Emulator -gpu mode (default: host on Mac/Windows, swiftshader on headless Linux CI). "
        "Never use swiftshader on a laptop unless you must — it pegs the CPU.",
    ),
    audio: bool = typer.Option(
        False,
        "--audio/--no-audio",
        help="Give a headless emulator a real audio device. Off by default; a scenario that "
        "checks sound needs it (playback state is readable via dumpsys either way).",
    ),
    animations: bool = typer.Option(
        False,
        "--animations/--no-animations",
        help="Keep system animations. Off by default: a tap settles in 272ms instead of 357ms "
        "and the spread narrows from 225ms to 69ms, and every wait is sized by the worst case.",
    ),
    wait: int = typer.Option(
        120, "--wait", help="Seconds to wait for adb state=device after boot."
    ),
    idle_stop: int | None = typer.Option(
        None,
        "--idle-stop",
        help="Auto-stop this AVD after N seconds idle AND unleased (0=never, and that choice "
        "is remembered). Defaults to `teardown.emulator_idle_stop_s` (1200 = 20 min). Safety net "
        "when agents forget `aua emulator stop --mine`, or are killed before they can.",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Safe for concurrent agents: allocate a free -port, pass -read-only, tag an owner. "
        "Standalone start does not retarget a lease. Select it only when unleased, or explicitly "
        "switch with lease acquire --replace; ordinary commands then stay unpinned. Stop with "
        "--serial or AUA_OWNER=… --mine.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Emulator console port (even, 5554–5682). Implies a fixed serial emulator-{port}. "
        "Auto-allocated with --parallel.",
    ),
    read_only: bool | None = typer.Option(
        None,
        "--read-only/--no-read-only",
        help="Pass -read-only so the same AVD can run multiple times (default on with --parallel).",
    ),
    owner: str | None = typer.Option(
        None,
        "--owner",
        help="Owner tag for parallel cleanup (default: $AUA_OWNER, or auto id with --parallel). "
        "Then `AUA_OWNER=… aua emulator stop --mine` only kills yours.",
    ),
    apk: str | None = typer.Option(
        None,
        "--apk",
        help="Also install this .apk on the emulator once it boots (see `aua install`). "
        "Skipped when the same version is already there.",
    ),
    launch: bool = typer.Option(
        False,
        "--launch",
        help="With --apk, also open the app and return the screen it landed on — boot, install, "
        "launch, and observe in one call.",
    ),
    reinstall: bool = typer.Option(
        False, "--reinstall", help="With --apk, push the bundle even if that version is present."
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="With --apk, uninstall any existing copy first (wipes app data). Needs --yes.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Required for --fresh."),
) -> None:
    """Boot an AVD and wait until adb sees it (headless by default).

    With ``--apk … --launch`` this is the whole bootstrap in one call: boot, install the build if
    it is not already there, open it, and return the first screen with usable element ids.
    Startup progress and ten-second heartbeats go to stderr; the final JSON stays on stdout.
    If a runner yields while this command is active, poll that process instead of retrying it.
    A confirmed unowned proxy black hole inherited from the AVD is cleared before app launch;
    reachable foreign proxies and AUA-owned proxies are preserved.

    Standalone start provisions an emulator. Agents should use ``aua session start`` for goal
    work; it owns pool selection, compatible provisioning, and process-bound leasing.
    """
    opts = _opts(ctx)
    cfg = opts.load()
    emulator_mod = _platform_capability(ctx, "virtual_devices")
    if (launch or reinstall or fresh) and not apk:
        err = UsageError(
            "--launch/--reinstall/--fresh only apply to --apk",
            hint="Add `--apk <path>`, or drop the flag.",
        )
        emit_error(err)
        raise typer.Exit(int(err.exit_code))
    try:
        with _emulator_start_progress(avd=avd, headless=headless, wait_s=float(wait)):
            booted = emulator_mod.start(
                avd,
                headless=headless,
                animations=animations,
                audio=audio,
                wait_s=float(wait),
                cache_dir=cfg.cache.dir,
                lease_registry_dir=cfg.lease.registry_dir,
                gpu=gpu,
                # No `if headless` gate: a windowed AVD nobody has touched for the timeout is just
                # as forgotten as a headless one, and it was the case with no safety net at all.
                idle_timeout_s=float(
                    idle_stop
                    if idle_stop is not None
                    else getattr(cfg.teardown, "emulator_idle_stop_s", 1200.0)
                ),
                parallel=parallel,
                port=port,
                read_only=read_only,
                owner=owner,
            )
        if apk:
            booted["install"] = _install_on_booted_emulator(
                ctx,
                serial=str(booted["serial"]),
                bundle=apk,
                launch=launch,
                reinstall=reinstall,
                fresh=fresh,
                confirmed=yes,
            )
        _emulator_emit(booted, ctx)
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


def _install_on_booted_emulator(
    ctx: typer.Context,
    *,
    serial: str,
    bundle: str,
    launch: bool,
    reinstall: bool,
    fresh: bool,
    confirmed: bool,
) -> dict[str, Any]:
    """Install (and optionally launch) a bundle on the emulator this command just booted.

    A second ``GlobalOpts`` pinned to the new serial rather than the ambient one: the boot may
    have allocated a console port nobody has named yet, and an Engine that reconnected by
    "the only attached device" would pick a different emulator on any host running two.
    """

    opts = _opts(ctx)
    pinned = replace(opts, serial=serial, _cfg=None)
    engine = pinned.engine()
    result = engine.install_app(
        bundle,
        mode="fresh" if fresh else "reinstall" if reinstall else "if-needed",
        confirmed=confirmed,
        launch=launch,
        observe=launch,
    )
    payload = result.model_dump(mode="json", exclude_none=True)
    if _OBSERVATION_VIEW is not None:
        payload = trim_observation_payload(payload, _OBSERVATION_VIEW, fmt=pinned.fmt())
    return payload


@emulator_app.command("stop")
def emulator_stop_cmd(
    ctx: typer.Context,
    serial: str | None = typer.Option(
        None, "--serial", help="Emulator serial to kill (e.g. emulator-5554)."
    ),
    avd: str | None = typer.Option(None, "--avd", help="Stop the AVD AUA started by name."),
    owner: str | None = typer.Option(
        None,
        "--owner",
        help="Stop only instances tagged with this owner (or set $AUA_OWNER).",
    ),
    mine: bool = typer.Option(
        False,
        "--mine",
        help="Stop emulators aua recorded as started. With $AUA_OWNER/--owner, only yours.",
    ),
    all_devices: bool = typer.Option(
        False, "--all", help="Kill EVERY running emulator (needed when no --serial/--avd/--mine)."
    ),
) -> None:
    """Stop a running emulator (`adb emu kill`).

    Needs an explicit target: an emulator may hold a signed-in session or seeded data, and
    killing it cannot be undone. Prefer `--serial` or `--mine` (scoped by `$AUA_OWNER` when
    parallel agents share a host).
    """
    opts = _opts(ctx)
    cfg = opts.load()
    emulator_mod = _platform_capability(ctx, "virtual_devices")
    # A *global* `--serial` — written before the subcommand, the position every other command
    # wants it in — used to be dropped on the floor here. `emulator stop` declares its own
    # `--serial`, so `hoist_global_options` rightly leaves the subcommand's flag alone, and this
    # function only ever read the local one. `aua --serial X emulator stop --owner Y` then fell
    # into the owner branch, reported `ok:true` with an empty `stopped` list, and left qemu
    # running. Honour the global flag here too; a teardown that silently does nothing is how
    # orphaned instances (and one coordinator killing a live worker) happened.
    if serial and opts.serial and serial != opts.serial:
        err = UsageError(
            f"conflicting serials: global --serial {opts.serial} vs "
            f"`emulator stop --serial` {serial}",
            hint="Pass one target. Both positions work; naming two different devices cannot.",
        )
        emit_error(err)
        raise typer.Exit(int(err.exit_code))
    serial = serial or opts.serial
    from . import leases

    try:
        _emulator_emit(
            emulator_mod.stop(
                serial=serial,
                avd=avd,
                all_devices=all_devices,
                mine=mine,
                owner=owner,
                cache_dir=cfg.cache.dir,
                # A target another live agent leases is skipped, never killed; the caller's
                # own lease (matched by process identity) still authorises its own stop.
                lease_registry_dir=cfg.lease.registry_dir,
                lease_owner=leases.resolve_owner(owner),
            ),
            ctx,
        )
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


_INSTALL_MODE_HELP = (
    "Only when the app is ALREADY installed do these differ. Default: leave it alone unless the "
    "bundle's version differs. --reinstall: push anyway, keeping app data. --fresh: uninstall "
    "first (wipes data; the only mode that survives a signing-key change) — needs --yes."
)


@app.command(name="install")
def install_cmd(
    ctx: typer.Context,
    bundle: str = typer.Argument(
        ...,
        metavar="APK",
        help="Path to the .apk to install (e.g. app/build/outputs/apk/debug/app-debug.apk).",
    ),
    launch: bool = typer.Option(
        False,
        "--launch",
        help="Open the app afterwards and return the screen it landed on — install, launch, and "
        "observe in one call instead of three.",
    ),
    reinstall: bool = typer.Option(
        False, "--reinstall", help=f"Reinstall mode. {_INSTALL_MODE_HELP}"
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Uninstall first, then install. Needs --yes."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Required for --fresh: confirms removing the installed app and ALL its data "
        "(feature-flag overrides, login session, local config).",
    ),
    grant: bool = typer.Option(
        False,
        "--grant",
        help="Auto-grant declared runtime permissions. Off by default — pre-granting silently "
        "skips the permission dialog a scenario may be there to verify.",
    ),
    package: str | None = typer.Option(
        None,
        "--package",
        help="Assert the bundle's package id; the install fails if the APK declares another.",
    ),
    activity: str | None = typer.Option(
        None, "--activity", help="With --launch, pin the entry Activity on multi-launcher builds."
    ),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="With --launch, fold in the resulting screen."
    ),
    install_timeout: int = typer.Option(
        300,
        "--install-timeout",
        help="Seconds to allow the install itself; a cold emulator is slow.",
    ),
) -> None:
    """Put a build on the device — replaces a hand-rolled `adb install`.

    Idempotent by default: an app already installed at the bundle's version is left alone, so a
    repeated run costs one package query instead of a multi-second push. Version is compared, not
    assumed — a rebuilt bundle with *any* different version installs, including an older one, so
    pointing this at a previous build deliberately downgrades rather than refusing.

    It also reports two things raw adb does not: it fails if the package manager never registered
    the package (adb can print Success anyway), and it says so when the target is a ``-read-only``
    emulator, where the install is real but disappears when the AVD stops.

    With ``--launch`` the whole bootstrap is one call. ``aua emulator start --apk`` goes further
    and boots the AVD too.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        if fresh and reinstall:
            raise UsageError(
                "pass --reinstall or --fresh, not both",
                hint="--reinstall keeps app data; --fresh wipes it.",
            )
        _emit(
            _route(
                engine,
                "install_app",
                bundle=bundle,
                package=package,
                mode="fresh" if fresh else "reinstall" if reinstall else "if-needed",
                confirmed=yes,
                grant_permissions=grant,
                launch=launch,
                activity=activity,
                observe=observe,
                timeout_ms=int(install_timeout) * 1000,
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="app")
def app_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(
        ...,
        metavar="ACTION",
        help="exists|status|foreground|launch-and-analyze|launch|restart-and-analyze|restart|"
        "stop|kill|clear|grant|current.",
    ),
    package: str | None = typer.Argument(
        None, metavar="[PKG]", help="Package for exists/status/launch/stop/kill/clear/grant."
    ),
    activity: str | None = typer.Option(
        None,
        "--activity",
        help="launch: pin the entry Activity (e.g. .LaunchActivity) on multi-launcher builds.",
    ),
    clear_state: bool = typer.Option(
        False,
        "--clear",
        help="launch: wipe app data first (Maestro launchApp clearState). Requires --yes.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "--yes-wipe-flags",
        help="Required for `clear` / `launch --clear`: confirms wiping app data "
        "(feature-flag overrides, login session, local config, …).",
    ),
    observe: bool = typer.Option(
        True,
        "--observe/--no-observe",
        help="launch: also return the screen the app opened on (skips a follow-up analyze).",
    ),
) -> None:
    """Inspect or control the foreground app.

    Some dev builds have several launcher activities (a Dev Tools menu next to the real
    entry), so a bare `launch` opens the wrong one nondeterministically — pass
    ``--activity`` to pin it.

    ``clear`` / ``launch --clear`` run ``pm clear`` — a **full wipe** of app data. Many apps
    keep feature-flag overrides and the login session in app data, so a wipe resets your test
    preconditions (re-auth required). Always pass ``--yes`` / ``--yes-wipe-flags`` to confirm;
    re-apply flags afterwards (e.g. via deeplink) before asserting experiment tabs.
    ``grant`` auto-grants declared runtime permissions so agents skip permission sheets.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        a = action.lower().replace("_", "-")
        if a in ("exists", "status"):
            if not package:
                raise UsageError(
                    f"app {a} needs a package",
                    hint=f"e.g. `aua app {a} com.example.app`",
                )
            result = _route(engine, "app_status", package=package)
            _emit(result, fmt)
            installed = (
                bool(result.get("installed"))
                if isinstance(result, dict)
                else bool(getattr(result, "installed", False))
            )
            if a == "exists" and not installed:
                raise typer.Exit(1)
            return
        explicit_analyze = a in ("launch-and-analyze", "restart-and-analyze")
        if a in ("launch-and-analyze", "restart-and-analyze"):
            a = a.removesuffix("-and-analyze")
        if a == "restart":
            # Replaces the two-step `adb shell am force-stop` + `adb shell am start -n pkg/act`
            # that every test setup was hand-rolling. Data is preserved on purpose — a reset that
            # silently wiped feature-flag overrides and the session would change the very
            # preconditions it is being used to establish; that is what `clear --yes` is for.
            if not package:
                raise UsageError(
                    "app restart needs a package",
                    hint="e.g. `aua app restart-and-analyze com.example.app "
                    "--activity .LaunchActivity`",
                )
            _route(
                engine,
                "app",
                action="stop",
                package=package,
                confirmed=yes,
                observe=False,
            )
            _emit(
                _route(
                    engine,
                    "app",
                    action="launch",
                    package=package,
                    activity=activity,
                    clear_state=False,
                    confirmed=yes,
                    observe=True if explicit_analyze else observe,
                ),
                fmt,
            )
            return
        wiping = a in ("clear", "clear-state", "clear_state") or (a == "launch" and clear_state)
        if wiping and not yes:
            raise UsageError(
                f"app {action}{' --clear' if a == 'launch' else ''} wipes ALL app data "
                f"(feature flags, login session, local config) — pass --yes to confirm",
                hint="Example: `aua app clear com.example.app --yes`. "
                "Then re-apply flag overrides / re-login before asserting experiment UI.",
            )
        _emit(
            _route(
                engine,
                "app",
                action=a,
                package=package,
                activity=activity,
                clear_state=clear_state,
                confirmed=yes,
                observe=True if explicit_analyze else observe,
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="shell")
def shell_cmd(
    ctx: typer.Context,
    command: list[str] = typer.Argument(
        ...,
        metavar="COMMAND...",
        help="Read-only command argv. Each output stream is capped at 256 KiB; use `--` before "
        "command flags, e.g. `aua shell -- logcat -d`.",
    ),
    shell_timeout: int = typer.Option(
        30,
        "--shell-timeout",
        min=1,
        max=120,
        help="Maximum seconds to wait (1-120).",
    ),
) -> None:
    """Run one bounded read-only command on AUA's leased target.

    Android's remote shell parses the command string, so AUA shell-quotes every argv item before
    handing it to adb; operators, substitutions, and newlines inside an argument stay literal. A
    conservative allow-list permits package/status and diagnostic reads such as ``pm path``,
    ``getprop``, ``dumpsys package``, ``settings get``, ``pidof`` and ``logcat -d``. Unknown or
    mutating verbs are refused. Stdout and stderr are each capped at 256 KiB and report truncation.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "shell",
            argv=list(command),
            timeout_ms=int(shell_timeout) * 1000,
        )
        _emit(result, fmt)
        ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result.ok)
        if not ok:
            raise typer.Exit(1)

    _run(ctx, go)


database_app = typer.Typer(
    help="Inspect, mutate, back up, and restore debuggable app SQLite databases.",
    no_args_is_help=True,
)
app.add_typer(database_app, name="db")

_DATABASE_RESTART = typer.Option(
    True,
    "--restart/--no-restart",
    help="Relaunch the package after the coherent stop-and-snapshot operation.",
)


def _database_sql(sql: str | None, file: str | None) -> str:
    if sql is not None and file is not None:
        raise UsageError("pass SQL as an argument or with --file, not both")
    if file is not None:
        path = Path(file).expanduser()
        try:
            value = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"could not read SQL file {path}: {exc}") from exc
    else:
        value = sql or ""
    if not value.strip():
        raise UsageError("a SQL statement is required (argument or --file PATH)")
    return value


def _database_parameters(raw: str | None) -> dict[str, Any] | list[Any] | None:
    if raw is None:
        return None
    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UsageError(f"--params must be valid JSON: {exc}") from exc
    if not isinstance(parsed, (dict, list)):
        raise UsageError("--params must be a JSON object or array")
    return parsed


@database_app.command("list")
def database_list_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
) -> None:
    """List private database files and WAL/SHM sidecar sizes."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "database_list", package=package), fmt)

    _run(ctx, go)


@database_app.command("schema")
def database_schema_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
    database: str = typer.Argument(..., help="Database basename from `aua db list`."),
    table: str | None = typer.Option(None, "--table", help="Inspect one table or view."),
    restart: bool = _DATABASE_RESTART,
) -> None:
    """Return tables/views with columns, indexes, foreign keys, and CREATE SQL."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "database_schema",
                package=package,
                database=database,
                table=table,
                restart=restart,
            ),
            fmt,
        )

    _run(ctx, go)


@database_app.command("query")
def database_query_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
    database: str = typer.Argument(..., help="Database basename from `aua db list`."),
    sql: str | None = typer.Argument(None, help="One read-only SQLite statement."),
    file: str | None = typer.Option(None, "--file", help="Read SQL from a UTF-8 file."),
    params: str | None = typer.Option(
        None,
        "--params",
        help='JSON object/array bound as SQLite parameters, e.g. {"id":"abc"}.',
    ),
    limit: int = typer.Option(100, "--limit", help="Maximum rows returned (1-1000)."),
    timeout_ms: int = typer.Option(
        5000,
        "--sql-timeout",
        help="Abort host-side SQLite work after this many milliseconds.",
    ),
    restart: bool = _DATABASE_RESTART,
    live: bool = typer.Option(
        True,
        "--live/--coherent",
        help="Keep the app running and preserve navigation/UI state (default). Pass "
        "--coherent to stop the app for a transactionally coherent snapshot. Live copies "
        "may be point-in-time-ish if the app is mid-write; --restart applies only to "
        "--coherent.",
    ),
) -> None:
    """Run one read-only query without disturbing the current app screen.

    The default live snapshot preserves in-app navigation/UI state. Pass ``--coherent`` when a
    transactionally coherent snapshot is required; that mode force-stops the app and relaunches
    it by default (or leaves it stopped with ``--no-restart``), so its result warns that the
    previous UI state is gone.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "database_query",
                package=package,
                database=database,
                sql=_database_sql(sql, file),
                parameters=_database_parameters(params),
                limit=limit,
                timeout_ms=timeout_ms,
                restart=restart,
                live=live,
            ),
            fmt,
        )

    _run(ctx, go)


@database_app.command("execute")
def database_execute_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
    database: str = typer.Argument(..., help="Database basename from `aua db list`."),
    sql: str | None = typer.Argument(None, help="INSERT/UPDATE/DELETE/REPLACE/WITH SQL."),
    file: str | None = typer.Option(
        None, "--file", help="Read one or more statements from a file."
    ),
    params: str | None = typer.Option(
        None,
        "--params",
        help="JSON object/array; supported when executing one statement.",
    ),
    timeout_ms: int = typer.Option(
        5000,
        "--sql-timeout",
        help="Abort host-side SQLite work after this many milliseconds.",
    ),
    restart: bool = _DATABASE_RESTART,
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Required: confirms direct app-data mutation after an automatic backup.",
    ),
) -> None:
    """Execute guarded data mutations, verify integrity, and keep a restore point."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "database_execute",
                package=package,
                database=database,
                sql=_database_sql(sql, file),
                parameters=_database_parameters(params),
                timeout_ms=timeout_ms,
                restart=restart,
                confirmed=yes,
            ),
            fmt,
        )

    _run(ctx, go)


@database_app.command("backup")
def database_backup_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
    database: str = typer.Argument(..., help="Database basename from `aua db list`."),
    restart: bool = _DATABASE_RESTART,
) -> None:
    """Create a private host restore point containing the database and sidecars."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "database_backup",
                package=package,
                database=database,
                restart=restart,
            ),
            fmt,
        )

    _run(ctx, go)


@database_app.command("backups")
def database_backups_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
    database: str = typer.Argument(..., help="Database basename from `aua db list`."),
) -> None:
    """List restore points for this device, package, and database."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "database_backups",
                package=package,
                database=database,
            ),
            fmt,
        )

    _run(ctx, go)


@database_app.command("restore")
def database_restore_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="Debuggable app package id."),
    database: str = typer.Argument(..., help="Database basename from `aua db list`."),
    backup_id: str = typer.Argument(..., help="Restore point from `aua db backups`."),
    restart: bool = _DATABASE_RESTART,
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Required: confirms replacing the current database with this restore point.",
    ),
) -> None:
    """Restore a backup after first preserving the current database as another backup."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "database_restore",
                package=package,
                database=database,
                backup_id=backup_id,
                restart=restart,
                confirmed=yes,
            ),
            fmt,
        )

    _run(ctx, go)


clipboard_app = typer.Typer(help="Clipboard — Maestro setClipboard / copyTextFrom / pasteText.")
app.add_typer(clipboard_app, name="clipboard")


@clipboard_app.command("set")
def clipboard_set(
    ctx: typer.Context, text: str = typer.Argument(..., help="Text to copy.")
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.clipboard_set(text), fmt)

    _run(ctx, go)


@clipboard_app.command("get")
def clipboard_get(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.clipboard_get(), fmt)

    _run(ctx, go)


@app.command(name="paste-and-analyze", cls=AnalyzeCommand)
def paste(
    ctx: typer.Context,
    observe: bool = typer.Option(True, "--observe/--no-observe"),
    with_image: str | None = typer.Option(
        None, "--with-image", metavar="[PATH]", show_default=False
    ),
) -> None:
    """Paste the clipboard into the focused field (Maestro pasteText)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.paste(observe=observe, with_image=_annotate_arg(with_image)), fmt)

    _run(ctx, go)


@app.command(cls=AnnotateCommand)
def copy(
    ctx: typer.Context,
    ident: str | None = typer.Argument(None, metavar="[ID]"),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
) -> None:
    """Copy an element's text/content-desc to the clipboard (Maestro copyTextFrom)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            engine.copy_text(element_id=_element_id(ident, selector), selector=selector),
            fmt,
        )

    _run(ctx, go)


@app.command(name="erase-and-analyze", cls=AnalyzeCommand)
def erase(
    ctx: typer.Context,
    ident: str | None = typer.Argument(None, metavar="[ID]"),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    chars: int | None = typer.Option(
        None, "--chars", help="Delete this many characters; omit to clear the whole field."
    ),
    observe: bool = typer.Option(True, "--observe/--no-observe"),
    with_image: str | None = typer.Option(
        None, "--with-image", metavar="[PATH]", show_default=False
    ),
) -> None:
    """Erase text in a field (Maestro eraseText)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            engine.erase(
                _element_id(ident, selector),
                selector=selector,
                chars=chars,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


location_app = typer.Typer(help="GPS mock location (Maestro setLocation).")
app.add_typer(location_app, name="location")


@location_app.command("set")
def location_set(
    ctx: typer.Context,
    coords: str = typer.Argument(..., help="LAT,LON (e.g. 37.42,-122.08)."),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        parts = [p.strip() for p in coords.split(",") if p.strip()]
        if len(parts) != 2:
            raise UsageError("location needs LAT,LON", hint="e.g. `aua location set 37.42,-122.08`")
        lat, lon = float(parts[0]), float(parts[1])
        _emit(engine.location_set(lat, lon), fmt)

    _run(ctx, go)


orientation_app = typer.Typer(help="Screen orientation (Maestro setOrientation).")
app.add_typer(orientation_app, name="orientation")


@orientation_app.command("set")
def orientation_set_cmd(
    ctx: typer.Context,
    mode: str = typer.Argument(..., help="portrait|landscape|left|right|natural."),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.orientation_set(mode), fmt)

    _run(ctx, go)


@orientation_app.command("get")
def orientation_get_cmd(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.orientation_get(), fmt)

    _run(ctx, go)


airplane_app = typer.Typer(help="Airplane mode (Maestro setAirplaneMode).")
app.add_typer(airplane_app, name="airplane")


@airplane_app.command("on")
def airplane_on(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.airplane_set(True), fmt)

    _run(ctx, go)


@airplane_app.command("off")
def airplane_off(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.airplane_set(False), fmt)

    _run(ctx, go)


@airplane_app.command("toggle")
def airplane_toggle_cmd(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.airplane_toggle(), fmt)

    _run(ctx, go)


network_app = typer.Typer(
    help="Inspect connectivity or enter a verified, reversible offline state."
)
app.add_typer(network_app, name="network")


@network_app.command("status")
def network_status_cmd(
    ctx: typer.Context,
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help=(
            "Read Android's current radio controls and active default network. Status is "
            "already a read-back operation; --verify is accepted for command-family parity."
        ),
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        # Reading status is itself the verification. Keep the explicit flag as an intuitive,
        # command-family-compatible spelling without asking older daemon/Engine adapters to
        # implement a second behavior.
        _ = verify
        _emit(_route(engine, "network_status"), fmt)

    _run(ctx, go)


@network_app.command("offline")
def network_offline_cmd(
    ctx: typer.Context,
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Read back radio controls and the active default network (recommended).",
    ),
    timeout_ms: int = typer.Option(
        10_000,
        "--timeout",
        min=0,
        help="Milliseconds to wait for Android to detach active transports.",
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(engine, "network_offline", verify=verify, timeout_ms=timeout_ms)
        _emit(result, fmt)
        _exit_unless_ok(
            result,
            ExitCode.DEVICE,
            code="network_verification_failed",
            hint="Inspect `aua network status`, then retry or run `aua network restore`.",
        )

    _run(ctx, go)


@network_app.command("restore")
def network_restore_cmd(
    ctx: typer.Context,
    timeout_ms: int = typer.Option(
        15_000,
        "--timeout",
        min=0,
        help="Milliseconds to wait for the saved controls and connectivity to return.",
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(engine, "network_restore", timeout_ms=timeout_ms)
        _emit(result, fmt)
        _exit_unless_ok(
            result,
            ExitCode.DEVICE,
            code="network_restore_failed",
            hint="The restore point was retained; inspect `aua network status` and retry.",
        )

    _run(ctx, go)


network_profile_app = typer.Typer(help="Apply one reversible network condition at a time.")
network_app.add_typer(network_profile_app, name="profile")


@network_profile_app.command("list")
def network_profile_list_cmd(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.network_profile_list(), fmt)

    _run(ctx, go)


@network_profile_app.command("status")
def network_profile_status_cmd(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.network_profile_status(), fmt)

    _run(ctx, go)


@network_profile_app.command("apply")
def network_profile_apply_cmd(
    ctx: typer.Context,
    profile: str = typer.Argument(
        ...,
        help="wifi-only | cellular-only | slow | lossy",
    ),
    loss_percent: float = typer.Option(
        10.0,
        "--loss-percent",
        min=0.1,
        max=100.0,
        help="Packet loss for the lossy profile (rootable emulator required).",
    ),
    timeout_ms: int = typer.Option(
        15_000,
        "--timeout",
        min=0,
        help="Milliseconds to wait for the requested condition to verify.",
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.network_profile_apply(
            profile,
            loss_percent=loss_percent,
            timeout_ms=timeout_ms,
        )
        _emit(result, fmt)
        _exit_unless_ok(
            result,
            ExitCode.DEVICE,
            code="network_profile_verification_failed",
            hint="Inspect `aua network profile status`, then restore before retrying.",
        )

    _run(ctx, go)


@network_profile_app.command("restore")
def network_profile_restore_cmd(
    ctx: typer.Context,
    timeout_ms: int = typer.Option(
        20_000,
        "--timeout",
        min=0,
        help="Milliseconds to wait for the original conditions to return.",
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.network_profile_restore(timeout_ms=timeout_ms)
        _emit(result, fmt)
        _exit_unless_ok(
            result,
            ExitCode.DEVICE,
            code="network_profile_restore_failed",
            hint="The restore point was retained; inspect profile status and retry.",
        )

    _run(ctx, go)


media_app = typer.Typer(help="Push media into the device gallery (Maestro addMedia).")
app.add_typer(media_app, name="media")


@media_app.command("add")
def media_add_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Local image/video file to push."),
    remote_dir: str = typer.Option(
        "/sdcard/DCIM/Camera", "--dir", help="Remote folder under which to store the file."
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.media_add(path, remote_dir=remote_dir), fmt)

    _run(ctx, go)


mic_app = typer.Typer(
    help="Inject PCM audio into an Android Emulator microphone.",
    no_args_is_help=True,
)
app.add_typer(mic_app, name="mic")


def _mic_control_target(
    *,
    ident: str | None,
    by: str | None,
    rid: str | None,
    text: str | None,
    desc: str | None,
    index: int | None,
    first: bool,
) -> tuple[int | None, dict[str, Any] | None]:
    """Build one optional control target with the same ambiguity rules as MCP."""

    direct_selectors = [value for value in (rid, text, desc) if value is not None]
    if len(direct_selectors) > 1 or (direct_selectors and (ident is not None or by is not None)):
        raise UsageError(
            "microphone control accepts only one id/rid/text/desc target",
            hint="Pass one fresh id or one stable selector; omit all four for audio-only input.",
        )
    selector = _selector(
        ident=ident,
        by=by,
        rid=rid,
        text=text,
        desc=desc,
        index=index,
        first=first,
    )
    if selector is None and (index is not None or first):
        raise UsageError(
            "--index/--first needs a microphone control selector",
            hint="Pass one of --rid/--text/--desc (or --by), or remove the modifier.",
        )
    return _element_id(ident, selector), selector


@mic_app.command("inject", cls=AnalyzeCommand)
def mic_inject_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., metavar="PCM-WAV", help="Host PCM WAV file to inject."),
    ident: str | None = typer.Argument(
        None,
        metavar="[CONTROL-ID]",
        help="Optional hold/toggle control id (or a --by selector value).",
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    control_mode: str = typer.Option(
        "hold",
        "--control-mode",
        metavar="hold|toggle",
        help="Target behavior: push-to-talk hold (default), or one tap to start and one to stop.",
    ),
    pre_roll_ms: int = typer.Option(
        250, "--pre-roll-ms", min=0, help="Wait this long after control start before audio."
    ),
    post_roll_ms: int = typer.Option(
        250, "--post-roll-ms", min=0, help="Wait this long after audio drains before control stop."
    ),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-action screen."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the post-action screenshot; bare flag uses a timestamped path.",
        show_default=False,
    ),
) -> None:
    """Inject U8/S16 PCM WAV audio with an optional hold or toggle control."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        element_id, selector = _mic_control_target(
            ident=ident,
            by=by,
            rid=rid,
            text=text,
            desc=desc,
            index=index,
            first=first,
        )
        # Resolve in the caller's cwd before daemon routing; a warm daemon can have a different
        # cwd and must never reinterpret a relative path as a different audio file.
        mic = engine.platform.capability("microphone")

        normalized_mode = mic.validate_control_mode(
            control_mode, has_target=element_id is not None or selector is not None
        )
        resolved_path = str(mic.inspect_pcm_wav(path).path)
        _emit(
            _route(
                engine,
                "mic_inject",
                wav_path=resolved_path,
                element_id=element_id,
                selector=selector,
                control_mode=normalized_mode,
                pre_roll_ms=pre_roll_ms,
                post_roll_ms=post_roll_ms,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


@mic_app.command("speak", cls=AnalyzeCommand)
def mic_speak_cmd(
    ctx: typer.Context,
    speech: str = typer.Argument(..., metavar="TEXT", help="Text for macOS say to synthesize."),
    ident: str | None = typer.Argument(
        None,
        metavar="[CONTROL-ID]",
        help="Optional hold/toggle control id (or a --by selector value).",
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    control_mode: str = typer.Option(
        "hold",
        "--control-mode",
        metavar="hold|toggle",
        help="Target behavior: push-to-talk hold (default), or one tap to start and one to stop.",
    ),
    voice: str | None = typer.Option(None, "--voice", help="Installed macOS say voice name."),
    rate: int | None = typer.Option(None, "--rate", min=1, help="Speech rate in words per minute."),
    pre_roll_ms: int = typer.Option(
        250, "--pre-roll-ms", min=0, help="Wait this long after control start before speech."
    ),
    post_roll_ms: int = typer.Option(
        250, "--post-roll-ms", min=0, help="Wait this long after speech drains before control stop."
    ),
    observe: bool = typer.Option(
        True, "--observe/--no-observe", help="Also return the post-action screen."
    ),
    with_image: str | None = typer.Option(
        None,
        "--with-image",
        metavar="[PATH]",
        help="Also save the post-action screenshot; bare flag uses a timestamped path.",
        show_default=False,
    ),
) -> None:
    """Synthesize speech, then inject it with an optional hold or toggle control."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        element_id, selector = _mic_control_target(
            ident=ident,
            by=by,
            rid=rid,
            text=text,
            desc=desc,
            index=index,
            first=first,
        )
        mic = engine.platform.capability("microphone")

        normalized_mode = mic.validate_control_mode(
            control_mode, has_target=element_id is not None or selector is not None
        )
        _emit(
            _route(
                engine,
                "mic_speak",
                text=speech,
                element_id=element_id,
                selector=selector,
                control_mode=normalized_mode,
                voice=voice,
                rate=rate,
                pre_roll_ms=pre_roll_ms,
                post_roll_ms=post_roll_ms,
                observe=observe,
                with_image=_annotate_arg(with_image),
            ),
            fmt,
        )

    _run(ctx, go)


record_app = typer.Typer(help="Screen recording (Maestro startRecording / stopRecording).")
app.add_typer(record_app, name="record")


@record_app.command("start")
def record_start_cmd(
    ctx: typer.Context,
    remote: str = typer.Option(
        "/sdcard/aua_recording.mp4", "--remote", help="Path on the device while recording."
    ),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "record_start", path=remote), fmt)

    _run(ctx, go)


@record_app.command("stop")
def record_stop_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Local path to save the MP4."),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        # Resolve in the caller's cwd. A warm daemon may have been launched from a
        # different directory and must not save the recording there.
        local_path = str(Path(path).expanduser().resolve())
        _emit(_route(engine, "record_stop", local_path=local_path), fmt)

    _run(ctx, go)


clock_app = typer.Typer(help="Device clock (Maestro travel).")
app.add_typer(clock_app, name="clock")


@clock_app.command("set")
def clock_set_cmd(
    ctx: typer.Context,
    ms: int | None = typer.Option(None, "--ms", help="Unix timestamp in milliseconds."),
    restore: bool = typer.Option(
        False,
        "--restore",
        help="Restore the wall clock saved by the last `clock set` (undo time travel).",
    ),
) -> None:
    """Set the device clock (emulator/rooted). Invalidates auth tokens — always restore.

    One-shot tests only (e.g. >30-day notification expiry). After asserting, run
    ``aua clock restore`` (or ``aua clock set --restore``) so the app can talk to APIs again.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.clock_set(timestamp_ms=ms, restore=restore), fmt)

    _run(ctx, go)


@clock_app.command("restore")
def clock_restore_cmd(ctx: typer.Context) -> None:
    """Restore the device wall clock saved by the last ``clock set``."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.clock_set(restore=True), fmt)

    _run(ctx, go)


@app.command()
def orient(ctx: typer.Context) -> None:
    """What the tool already knows about the foreground app (playbook, deeplinks, recipes).

    This is the orientation blob ``daemon start`` prints. It is worth reading once per
    session and is pure noise afterwards, so it lives here too — start the daemon with
    ``--quiet`` and call this when you actually want it.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "orient"), fmt)

    _run(ctx, go)


dashboard_app = typer.Typer(
    help="Persistent browser dashboard: start, status, open, QR, stop, or foreground run.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(dashboard_app, name="dashboard")


def _dashboard_access(
    cfg: Any, *, port: int | None, lan: bool | None, hostname: str | None, auth: bool | None
) -> tuple[int | None, bool, str | None, bool]:
    """Merge the access flags over ``dashboard`` config; an explicit flag always wins.

    The flags are tri-state on purpose: ``None`` means "not typed", which is the only way
    to tell ``--auth`` from a default that happens to agree with it.
    """
    defaults = cfg.dashboard
    name = defaults.name if hostname is None else hostname
    name = (name or "").strip() or None
    if lan is False and hostname is None:
        # --local means loopback. A name nobody typed must not drag the bind back onto the
        # network; a name the caller *did* type alongside --local is a contradiction, and
        # is left intact so start_service can say so.
        name = None
    return (
        defaults.port if port is None else port,
        (defaults.lan or bool(name)) if lan is None else lan,
        name,
        defaults.auth if auth is None else auth,
    )


def _dashboard_start(
    ctx: typer.Context,
    *,
    serial: str | None,
    grid: bool,
    port: int | None,
    lan: bool | None,
    hostname: str | None,
    auth: bool | None,
    open_browser: bool,
    poll_ms: int,
) -> None:
    from . import dashboard as dash

    opts = _opts(ctx)
    cfg = opts.load()
    if serial:
        cfg.device.serial = serial
    port, lan, hostname, auth = _dashboard_access(
        cfg, port=port, lan=lan, hostname=hostname, auth=auth
    )
    try:
        result = dash.start_service(
            cfg,
            serial=serial,
            port=port,
            poll_ms=poll_ms,
            grid=grid,
            lan=lan,
            hostname=hostname,
            auth=auth,
            explicit_config=opts.config,
            profile=opts.profile,
            platform=opts.platform,
        )
        if open_browser:
            import webbrowser

            result["opened"] = bool(webbrowser.open(str(result["access_url"])))
        _echo_json(result, opts.fmt())
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@dashboard_app.callback(invoke_without_command=True)
def dashboard_cmd(
    ctx: typer.Context,
    serial: str | None = typer.Option(
        None, "--serial", help="Pin the detail view to one device serial."
    ),
    grid: bool = typer.Option(
        True,
        "--grid/--detail",
        help="Start with the live device grid (default); --detail focuses the first device.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Dedicated exact dashboard port; never shifts if occupied (default 48765, or 80 with --name).",
    ),
    lan: bool | None = typer.Option(None, "--lan/--local", help="--lan serves every interface, --local serves loopback only. Default comes from `dashboard.lan`."),
    hostname: str | None = typer.Option(None, "--name", help="mDNS hostname to publish; `aua` serves http://aua.local/ and implies --lan. Pass `--name ""` to publish nothing. Default comes from `dashboard.name`."),
    auth: bool | None = typer.Option(None, "--auth/--no-auth", help="--auth requires an access token on network access; --no-auth serves the LAN with no token, so anything that can reach the port gets full device control. Default comes from `dashboard.auth`."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the detached dashboard in your default browser."
    ),
    poll_ms: int = typer.Option(500, "--poll-ms", help="Browser live-frame poll interval."),
) -> None:
    """Start the persistent dashboard (compatibility alias for ``dashboard start``)."""
    if ctx.invoked_subcommand is not None:
        return
    _dashboard_start(
        ctx,
        serial=serial,
        grid=grid,
        port=port,
        lan=lan,
        hostname=hostname,
        auth=auth,
        open_browser=open_browser,
        poll_ms=poll_ms,
    )


@dashboard_app.command("start")
def dashboard_start_cmd(
    ctx: typer.Context,
    serial: str | None = typer.Option(None, "--serial", help="Pin detail to one device."),
    grid: bool = typer.Option(True, "--grid/--detail", help="Grid or one-device detail view."),
    port: int | None = typer.Option(
        None, "--port", help="Dedicated exact dashboard port (default 48765, or 80 with --name)."
    ),
    lan: bool | None = typer.Option(None, "--lan/--local", help="--lan serves every interface, --local serves loopback only. Default comes from `dashboard.lan`."),
    hostname: str | None = typer.Option(None, "--name", help="mDNS hostname to publish; `aua` serves http://aua.local/ and implies --lan. Pass `--name ""` to publish nothing. Default comes from `dashboard.name`."),
    auth: bool | None = typer.Option(None, "--auth/--no-auth", help="--auth requires an access token on network access; --no-auth serves the LAN with no token, so anything that can reach the port gets full device control. Default comes from `dashboard.auth`."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the dashboard after ensuring it is running."
    ),
    poll_ms: int = typer.Option(500, "--poll-ms", help="Browser live-frame poll interval."),
) -> None:
    """Start once in the background; repeated calls reuse the same dashboard."""
    _dashboard_start(
        ctx,
        serial=serial,
        grid=grid,
        port=port,
        lan=lan,
        hostname=hostname,
        auth=auth,
        open_browser=open_browser,
        poll_ms=poll_ms,
    )


@dashboard_app.command("status")
def dashboard_status_cmd(
    ctx: typer.Context,
    port: int | None = typer.Option(
        None, "--port", help="Dedicated dashboard port (default: the running dashboard's)."
    ),
) -> None:
    """Show process, network scope, and authenticated phone URLs."""
    from . import dashboard as dash

    opts = _opts(ctx)
    _echo_json(dash.service_status(opts.load(), port=port), opts.fmt())


@dashboard_app.command("open")
def dashboard_open_cmd(
    ctx: typer.Context,
    port: int | None = typer.Option(
        None, "--port", help="Dedicated dashboard port (default: the running dashboard's)."
    ),
) -> None:
    """Open the already-running dashboard in the default browser."""
    from . import dashboard as dash

    opts = _opts(ctx)
    try:
        _echo_json(dash.open_service(opts.load(), port=port), opts.fmt())
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@dashboard_app.command("qr")
def dashboard_qr_cmd(
    ctx: typer.Context,
    port: int | None = typer.Option(
        None, "--port", help="Dedicated dashboard port (default: the running dashboard's)."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        help="PNG destination (default: the private AUA cache).",
    ),
    open_image: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the generated QR image after writing it.",
    ),
) -> None:
    """Generate the authenticated phone-access QR for a running LAN dashboard."""
    import webbrowser

    from . import dashboard as dash

    opts = _opts(ctx)
    try:
        result = dash.create_access_qr(opts.load(), port=port, output=output)
        if open_image:
            result["opened"] = bool(webbrowser.open(Path(result["path"]).as_uri()))
        _echo_json(result, opts.fmt())
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@dashboard_app.command("stop")
def dashboard_stop_cmd(
    ctx: typer.Context,
    port: int | None = typer.Option(
        None, "--port", help="Dedicated dashboard port (default: the running dashboard's)."
    ),
) -> None:
    """Stop the owned background dashboard without touching another port owner."""
    from . import dashboard as dash

    opts = _opts(ctx)
    try:
        _echo_json(dash.stop_service(opts.load(), port=port), opts.fmt())
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@dashboard_app.command("run")
def dashboard_run_cmd(
    ctx: typer.Context,
    serial: str | None = typer.Option(None, "--serial", help="Pin detail to one device."),
    grid: bool = typer.Option(True, "--grid/--detail", help="Grid or one-device detail view."),
    port: int | None = typer.Option(
        None, "--port", help="Exact foreground dashboard port (default 48765, or 80 with --name)."
    ),
    lan: bool | None = typer.Option(None, "--lan/--local", help="--lan serves every interface, --local serves loopback only. Default comes from `dashboard.lan`."),
    hostname: str | None = typer.Option(None, "--name", help="mDNS hostname to publish; `aua` serves http://aua.local/ and implies --lan. Pass `--name ""` to publish nothing. Default comes from `dashboard.name`."),
    auth: bool | None = typer.Option(None, "--auth/--no-auth", help="--auth requires an access token on network access; --no-auth serves the LAN with no token, so anything that can reach the port gets full device control. Default comes from `dashboard.auth`."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the page in your default browser."
    ),
    poll_ms: int = typer.Option(500, "--poll-ms", help="Browser live-frame poll interval."),
) -> None:
    """Run in the foreground for debugging; Ctrl-C stops it."""
    import secrets

    from . import dashboard as dash

    opts = _opts(ctx)
    cfg = opts.load()
    port, lan, hostname, auth = _dashboard_access(
        cfg, port=port, lan=lan, hostname=hostname, auth=auth
    )
    try:
        if hostname:
            hostname = dash.bonjour.normalise_hostname(hostname)
        if port is None:
            port = 80 if hostname else dash.DEFAULT_DASHBOARD_PORT
            if port != dash.DEFAULT_DASHBOARD_PORT and not dash._can_bind(
                "0.0.0.0" if lan else "127.0.0.1", port
            ):
                port = dash.DEFAULT_DASHBOARD_PORT
        require_auth = bool(lan and auth)
        token = secrets.token_urlsafe(32) if require_auth else None
        if lan:
            urls = dash._service_urls(port=port, lan=True, access_token=token)
            _echo_json({"ok": True, "action": "dashboard-run", **urls}, opts.fmt())
        dash.run(
            serial=serial,
            port=port,
            config=cfg,
            open_browser=open_browser,
            poll_ms=poll_ms,
            grid=grid,
            block=True,
            bind_host="0.0.0.0" if lan else "127.0.0.1",
            exact_port=True,
            require_auth=require_auth,
            access_token=token,
            hostname=hostname,
        )
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err


@app.command()
def daemon(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="start|stop|status|reap."),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="On start, skip the app orientation blob (get it later with `aua orient`).",
    ),
    stop_all: bool = typer.Option(
        False,
        "--all",
        help=(
            "On stop, stop every live daemon, including the per-serial ones an unpinned "
            "`daemon stop` cannot reach."
        ),
    ),
) -> None:
    """Manage the optional warm-state daemon (§10)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        a = action.lower()
        try:
            from . import daemon as daemon_mod
        except Exception as exc:
            raise UsageError(
                "daemon support is not available in this build",
                hint="The optional daemon module could not be imported.",
            ) from exc
        cfg = engine.config
        if a == "start":
            started = daemon_mod.start(cfg) or {}
            detail = daemon_mod.status(cfg)
            # Report the daemon's OBSERVED state, not a flat ok=True: a child that dies at
            # startup (a bad config slice is enough) left agents believing they had warm
            # state while every later command silently paid the full device attach cost.
            running = bool(detail.get("running")) or daemon_mod.is_running(cfg)
            out: dict[str, Any] = {
                "ok": running,
                "action": "daemon-start",
                "detail": detail,
            }
            if not running:
                out["error"] = {
                    "code": "daemon_not_started",
                    "message": f"the daemon did not come up ({started.get('status') or 'unknown'})",
                    "hint": (
                        "Commands still work, they just pay full device attach cost each "
                        f"time. The child's traceback is in {cfg.cache.dir}/daemon.log."
                    ),
                }
            # Best-effort: surface what we already know about the foreground app, so an
            # agent that starts the daemon first immediately sees the map + top gotos.
            if daemon_mod.is_running(cfg) and not quiet:
                try:
                    out["orientation"] = _route(engine, "orient")
                except Exception:  # noqa: BLE001 - orientation is purely advisory
                    logger.debug("daemon-start orientation unavailable")
            elif quiet:
                out["hint"] = "orientation suppressed; run `aua orient` for the app playbook"
        elif a == "stop":
            if stop_all:
                # The only command that ends every daemon whatever it is named. Kept explicit
                # rather than made the default for a bare `stop`: agents run concurrently, one
                # warm daemon per emulator, and stopping a colleague's device by omission
                # would be worse than the trap this replaces.
                out = {"action": "daemon-stop", **daemon_mod.stop_all(cfg)}
            else:
                result = daemon_mod.stop(cfg)
                # Emit what `stop` actually returned. It reports `status` and
                # `others_still_running` precisely so a caller can learn it signalled a socket
                # nobody uses; hard-coding `{"ok": True}` over the top of that is what closed
                # the loop behind the skew refusal, whose hint said to run exactly this.
                survivors = result.get("others_still_running") or []
                stopped = result.get("status") == "stopped"
                pinned = bool(daemon_mod.effective_serial(cfg))
                out = {
                    "ok": result.get("status") in {"stopped", "not_running"}
                    and (pinned or not survivors),
                    "action": "daemon-stop",
                    **result,
                }
                if not stopped and survivors:
                    commands = [
                        daemon_mod.describe_socket(sock)["stop_command"] for sock in survivors
                    ]
                    out["error"] = {
                        "code": "daemon_not_stopped",
                        "message": (
                            f"nothing was listening on {result.get('socket')}, and "
                            f"{len(survivors)} daemon(s) are still serving other sockets"
                        ),
                        "hint": (
                            "A daemon started for a device lives on `daemon.sock.<serial>`, "
                            "which an unpinned stop cannot reach. Run "
                            + " ; ".join(f"`{c}`" for c in commands)
                            + " — or `aua daemon stop --all` for every one of them."
                        ),
                    }
        elif a == "status":
            detail = daemon_mod.status(cfg)
            # One key, one meaning, per payload. The top-level `running` used to be widened to
            # "any daemon on this host" while `detail.running` stayed scoped to this socket, so
            # a single answer read `running: true` over `running: false` — both true of their
            # own socket, and useless together. The concern that widening addressed was real (a
            # bare status said "not running" while routed commands were using
            # `daemon.sock.<serial>`), so it is answered under its own name instead:
            # `any_running` plus `daemons`, which names every live socket and its serial.
            #
            # `detail.running` keeps its scope because `daemon start` reads it to decide
            # whether the child it just spawned came up.
            out = {
                "ok": True,
                "action": "daemon-status",
                "running": bool(detail.get("running")),
                "socket": detail.get("socket"),
                "any_running": bool(detail.get("daemons")),
                "daemons": detail.get("daemons") or [],
                "detail": detail,
            }
        elif a == "reap":
            out = daemon_mod.reap(cfg)
        else:
            raise UsageError(f"unknown daemon action '{action}'", hint="start|stop|status|reap")
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        typer.echo(json.dumps(out, indent=indent, separators=sep, ensure_ascii=False, default=str))
        if out.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- config


config_app = typer.Typer(
    name="config", help="Inspect and initialise configuration.", no_args_is_help=True
)
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file."),
) -> None:
    """Write a commented default config to the user config path."""
    path = user_config_path()
    try:
        if path.exists() and not force:
            typer.echo(f"config already exists at {path} (use --force to overwrite)")
            raise typer.Exit(ExitCode.OK)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default_config_yaml(), encoding="utf-8")
    except typer.Exit:
        raise
    except OSError as exc:
        err = ConfigError(f"could not write config to {path}: {exc}")
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from exc
    typer.echo(str(path))


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    effective: bool = typer.Option(
        False,
        "--effective",
        help="Print the merged config after precedence (default shows it too).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of YAML."),
) -> None:
    """Print the merged, masked config (secrets never shown). YAML by default; ``--json``
    (or ``--format compact``) emits JSON. ``--effective`` is the default behaviour."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        import yaml

        data = engine.config.masked_dict()
        if as_json or fmt is OutputFormat.compact:
            sep = (",", ":") if fmt is OutputFormat.compact else None
            indent = None if fmt is OutputFormat.compact else 2
            typer.echo(json.dumps(data, indent=indent, separators=sep, ensure_ascii=False))
        else:
            typer.echo(yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip())

    _run(ctx, go)


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Print the resolved config file path."""
    opts = _opts(ctx)
    if opts.config:
        typer.echo(str(opts.config))
        return
    project = find_project_config()
    typer.echo(str(project) if project is not None else str(user_config_path()))


# --------------------------------------------------------------------------- policy


policy_app = typer.Typer(
    name="policy", help="Inspect the optional guarded local policy.", no_args_is_help=True
)
app.add_typer(policy_app, name="policy")


@policy_app.command("status")
def policy_status_cmd(ctx: typer.Context) -> None:
    """Report local config, dependency, artifact, hash, and warm-daemon readiness."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        from .daemon import policy_runtime_status

        _echo_json(policy_runtime_status(engine.config), fmt)

    _run(ctx, go)


# --------------------------------------------------------------------------- doctor


@app.command(name="lease")
def lease_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(
        "list",
        metavar="ACTION",
        help="list|acquire|renew|release|transfer|accept|cancel-transfer",
    ),
    serial_arg: str | None = typer.Argument(
        None, metavar="[SERIAL_OR_TOKEN]", help="Device to act on, or a transfer token to accept."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="On release: drop a lease this agent does not own (wedged device escape hatch).",
    ),
    replace_lease: bool = typer.Option(
        False,
        "--replace",
        help="On acquire: acknowledge cleaning and releasing this agent's previous device.",
    ),
) -> None:
    """Who is driving which emulator — and claim one for this agent.

    Parallel agents otherwise all land on "the only/first device" and silently drive each
    other's screens. A lease is claimed automatically by any command, so this is for
    reserving one up front (with `--needs`), inspecting who holds what, or handing one back
    early.

        aua lease list
        aua lease acquire --needs root,proxy
        aua lease release emulator-5554
        aua lease transfer emulator-5554     # returns a one-time token
        aua lease accept <token>              # run from the receiving agent
        aua lease cancel-transfer emulator-5554

    A lease ages out once its owning process is gone. It does NOT age out while that process
    is still alive — and an owner is stable for the life of the calling process, so a
    long-running orchestrator that touched a device keeps holding it (a warm daemon serving
    that device keeps the entry fresh). The sole exception is an explicitly pending transfer:
    its reservation survives a source crash for at most five minutes so the receiver can accept
    it. When a device is wedged, break it with
    `aua lease release <serial> --force`; `aua lease list` names the holder.

    `--owner` (or `$AUA_OWNER`) names the agent; otherwise it is derived and stable for the
    life of the calling process.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        from . import leases as lease_mod
        from .teardown import cleanup_complete as _lease_cleanup_complete

        cache = engine.config.lease.registry_dir
        owner = lease_mod.resolve_owner(_opts(ctx).owner)
        verb = (action or "list").strip().lower()

        if replace_lease and verb != "acquire":
            raise UsageError("--replace is only valid with `lease acquire`")
        if force and verb != "release":
            raise UsageError("--force is only valid with `lease release`")

        if verb == "list":
            live = {e["serial"]: e for e in lease_mod.list_leases(cache)}
            rows = []
            for d in engine.list_devices():
                held = live.get(d.serial)
                handoff = lease_mod.pending_handoff(held) if held else None
                rows.append(
                    {
                        "serial": d.serial,
                        "model": d.model,
                        "owner": held.get("owner") if held else None,
                        "idle_s": round(lease_mod.idle_seconds(held), 1) if held else None,
                        "app": held.get("app") if held else None,
                        "needs": held.get("needs") if held else None,
                        "role": held.get("role", "primary") if held else None,
                        "mine": lease_mod.entry_owned_by(held, owner),
                        "handoff_pending": bool(handoff),
                        "handoff_expires_in_s": (
                            max(0, int(float(handoff.get("expires") or 0) - time.time()))
                            if handoff
                            else None
                        ),
                    }
                )
            _echo_json({"owner": owner, "devices": rows}, fmt)
            return

        if verb == "acquire":
            with lease_mod.owner_transaction(cache, owner):
                # Claiming is device resolution, so it shares the exact path every command uses.
                # The positional target is more specific than an ambient config/env pin.
                if serial_arg:
                    engine.config.device.serial = serial_arg
                previous = lease_mod.held_by(cache, owner)
                engine._lease_allow_replacement = replace_lease
                try:
                    serial = engine._lease_device()
                finally:
                    engine._lease_allow_replacement = False
                if not serial:
                    raise DeviceError("no device was available to lease")
                replaced = [old_serial for old_serial in previous if old_serial != serial]
                resets: list[dict[str, Any]] = []
                if replaced:
                    # The new target is reserved first. Old targets remain leased until every
                    # durable device mutation is reset, so no other agent can inherit dirty state.
                    with contextlib.ExitStack() as cleanup_locks:
                        for locked_serial in sorted({serial, *replaced}):
                            cleanup_locks.enter_context(
                                lease_mod.device_transaction(cache, locked_serial)
                            )
                        if not lease_mod.renew(cache, serial, owner=owner):
                            raise DeviceError(
                                f"the replacement reservation on {serial} changed before cleanup",
                                hint=(
                                    "The original lease was kept. Run `aua lease list`, then retry "
                                    "the replacement against a free target."
                                ),
                            )
                        for old_serial in replaced:
                            reset = engine.teardown_run(serial=old_serial, force=True)
                            resets.append({"serial": old_serial, "result": reset})
                        failed = [
                            item
                            for item in resets
                            if not _lease_cleanup_complete(item["result"])
                        ]
                        if failed:
                            if serial not in previous:
                                lease_mod.release(cache, serial, owner=owner)
                            raise DeviceError(
                                "the replacement device was reserved, but the previous device "
                                "could not be cleaned; the original lease was kept",
                                hint=(
                                    "Run `aua teardown status`, fix the reported cleanup failure, "
                                    "then retry the same `lease acquire … --replace` command."
                                ),
                            )
                        unreleased = [
                            old_serial
                            for old_serial in replaced
                            if not lease_mod.release(cache, old_serial, owner=owner)
                        ]
                        if unreleased:
                            raise DeviceError(
                                "cleaned but could not release previous lease(s): "
                                f"{', '.join(unreleased)}",
                                hint="Run `aua lease list`, then retry the replacement.",
                            )
                        if not lease_mod.promote_replacement(cache, serial, owner=owner):
                            raise DeviceError(
                                f"{serial} is reserved but could not be promoted",
                                hint=(
                                    f"Resume with `aua lease acquire {serial} --replace`; the old "
                                    "device is already clean and no ordinary call will misroute."
                                ),
                            )
                payload = {
                    "ok": True,
                    "action": "lease-acquire",
                    "serial": serial,
                    "owner": owner,
                    "replaced": replaced,
                    "resets": resets,
                    "warning": (
                        f"cleaned and released previous lease(s): {', '.join(replaced)}"
                        if replaced
                        else None
                    ),
                }
            _echo_json(payload, fmt)
            return

        if verb == "transfer":
            target = serial_arg or engine.config.device.serial
            if not target:
                raise UsageError(
                    "`lease transfer` needs a serial",
                    hint="e.g. `aua lease transfer emulator-5554`.",
                )
            handoff = lease_mod.create_handoff(cache, target, owner=owner)
            token = str(handoff["token"])
            _echo_json(
                {
                    "ok": True,
                    "action": "lease-transfer",
                    **handoff,
                    "accept_call": f"aua lease accept {token}",
                    "detail": (
                        "Pass the one-time token to the receiving agent. The device stays reserved "
                        "but frozen for the current owner until accept, cancel, or expiry."
                    ),
                },
                fmt,
            )
            return

        if verb == "cancel-transfer":
            target = serial_arg or engine.config.device.serial
            if not target:
                raise UsageError(
                    "`lease cancel-transfer` needs a serial",
                    hint="e.g. `aua lease cancel-transfer emulator-5554`.",
                )
            cancelled = lease_mod.cancel_handoff(cache, target, owner=owner)
            if not cancelled:
                raise UsageError(
                    f"{target} has no pending handoff owned by this agent",
                    hint="Run `aua lease list` to confirm the current holder.",
                )
            _echo_json(
                {
                    "ok": True,
                    "action": "lease-cancel-transfer",
                    "serial": target,
                    "owner": owner,
                },
                fmt,
            )
            return

        if verb == "accept":
            if not serial_arg:
                raise UsageError(
                    "`lease accept` needs the one-time token",
                    hint="The current holder gets it from `aua lease transfer <serial>`.",
                )
            accepted = lease_mod.accept_handoff(cache, serial_arg, owner=owner)
            _echo_json(
                {
                    "ok": True,
                    "action": "lease-accept",
                    **accepted,
                    "detail": (
                        "The lease now follows this agent process. Goal-session ownership was not "
                        "transferred."
                    ),
                },
                fmt,
            )
            return

        target = serial_arg or engine.config.device.serial
        if not target:
            raise UsageError(
                f"`lease {verb}` needs a serial",
                hint="e.g. `aua lease release emulator-5554`, or pass --serial.",
            )
        if verb == "renew":
            ok = lease_mod.renew(cache, target, owner=owner)
            _echo_json({"ok": ok, "action": "lease-renew", "serial": target, "owner": owner}, fmt)
            if not ok:
                raise DeviceLeasedError(
                    f"{target} is not leased by {owner}",
                    hint="`aua lease list` shows the holder.",
                )
            return
        if verb == "release":
            # Handing the device back means handing it back *clean*. Releasing the lease and
            # then resetting it races the next acquirer; reset while it is still quarantined by
            # this lease, and keep the lease if cleanup fails.
            with lease_mod.device_transaction(cache, target):
                owned = force or lease_mod.renew(cache, target, owner=owner)
                if not owned:
                    raise DeviceLeasedError(
                        f"{target} is held by another agent",
                        hint=(
                            "An agent may only release its own lease. If a transfer is pending, "
                            f"cancel it first with `aua lease cancel-transfer {target}`. Use "
                            f"`aua lease release {target} --force` only for a known wedged run."
                        ),
                    )
                reset = engine.teardown_run(serial=target, force=True)
                if not _lease_cleanup_complete(reset):
                    raise DeviceError(
                        f"could not clean {target}; its lease was kept",
                        hint=(
                            "Run `aua teardown status`, fix the reported undo, then retry release."
                        ),
                    )
                # `--force` skips ownership only after operator-authorized cleanup.
                ok = lease_mod.release(cache, target, owner=None if force else owner)
            _echo_json(
                {
                    "ok": ok,
                    "action": "lease-release",
                    "serial": target,
                    "forced": force,
                    "reset": reset.get("reports"),
                },
                fmt,
            )
            if not ok:
                raise DeviceError(
                    f"{target} was cleaned but its lease file could not be removed",
                    hint="Run `aua lease list`, then retry release.",
                )
            return
        raise UsageError(
            f"unknown lease action {verb!r}",
            hint=(
                "Use list, acquire, renew, release, transfer, accept or cancel-transfer."
            ),
        )

    _run(ctx, go)


@app.command(name="teardown")
def teardown_cmd(
    ctx: typer.Context,
    action: str = typer.Argument("status", metavar="ACTION", help="status|run"),
    serial_arg: str | None = typer.Option(
        None, "--serial-target", help="Only this device (defaults to every dirty device)."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Undo even while a holder looks live — for a device you know is yours.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Say what would be undone without touching any device."
    ),
) -> None:
    """Show or undo persistent device changes AUA is still holding on your behalf.

    A proxy, an offline network, a moved clock and disabled animations all outlive the agent
    that set them: they are device-global settings, and an agent is a process that can be
    killed. Every such change is journalled with its undo, so it can be reversed by a later
    command, by a detached watchdog, or by hand here.

        aua teardown status                    # what is still pending, and on which device
        aua teardown run                       # undo everything no live agent still holds
        aua teardown run --serial-target emulator-5554 --force

    Changes on a device a live agent holds are reported, never undone — pulling a proxy out
    from under a running test is worse than leaving it set. `--force` overrides that for a
    device you know is yours.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        verb = (action or "status").strip().lower()
        if verb == "status":
            _emit(engine.teardown_status(), fmt)
            return
        if verb == "run":
            result = engine.teardown_run(serial=serial_arg, force=force, dry_run=dry_run)
            _emit(result, fmt)
            return
        raise UsageError(f"unknown teardown action {verb!r}", hint="Use `status` or `run`.")

    _run(ctx, go)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check environment + provider availability (never prints secret values)."""
    opts = _opts(ctx)
    # doctor never fails on unavailable subsystems: a config error still surfaces, but
    # an unreachable device / missing provider deps must yield exit 0.
    try:
        engine = opts.engine()
    except AuaError as err:
        emit_error(err)
        raise typer.Exit(int(err.exit_code)) from err

    report = _build_doctor_report(engine)
    # Default to a readable report; emit machine JSON only when explicitly requested.
    explicit = (opts.format or "").lower()
    if explicit in {"json", "compact"}:
        import json

        sep = (",", ":") if explicit == "compact" else None
        indent = None if explicit == "compact" else 2
        typer.echo(json.dumps(report, indent=indent, separators=sep, ensure_ascii=False))
    else:
        typer.echo(_render_doctor_pretty(report))


def _build_doctor_report(engine: Engine) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    try:
        checks.update(engine.platform.doctor_checks())
    except AuaError as exc:
        checks["platform"] = {"ok": False, "detail": exc.message}
    except Exception as exc:  # pragma: no cover - defensive
        checks["platform"] = {"ok": False, "detail": str(exc)}

    try:
        infos = engine.list_devices()
        checks["devices"] = {
            "ok": len(infos) > 0,
            "count": len(infos),
            "detail": [d.model_dump(mode="json") for d in infos] if infos else "no devices",
        }
        if not infos:
            if engine.platform.supports("virtual_devices"):
                checks["devices"]["hint"] = (
                    "No target attached — boot a virtual device with "
                    "`aua emulator start` (see `aua emulator list`)."
                )
            else:
                checks["devices"]["hint"] = (
                    f"No {engine.platform.name} target is available; connect one or configure "
                    "the selected platform plugin."
                )
    except AuaError as exc:
        checks["devices"] = {"ok": False, "detail": exc.message}
    except Exception as exc:  # pragma: no cover - defensive
        checks["devices"] = {"ok": False, "detail": str(exc)}

    # A device pointed at a proxy nobody owns is exactly what doctor is for: an arriving agent
    # runs `aua doctor`, not `aua proxy status --serial X` for a serial it has not thought about
    # yet. Read-only by contract (`proxy_survey` never heals), so sweeping serials the caller
    # did not name costs two adb reads each and mutates nothing.
    try:
        checks["proxy"] = engine.proxy_survey()
    except AuaError as exc:
        checks["proxy"] = {"ok": True, "detail": f"proxy state unavailable: {exc.message}"}
    except Exception as exc:  # pragma: no cover - defensive
        checks["proxy"] = {"ok": True, "detail": f"proxy state unavailable: {exc}"}

    checks["skills"] = _installed_skill_checks()

    # The optional local policy is configured in a different file from the one that installs its
    # runtime, so the two drift apart silently: on the host this was written for, `policy.mode` was
    # `advisory` and no MLX runtime existed in the environment `aua` runs from. Doctor said nothing.
    # Host-only and read-only: no model is loaded and no device is touched.
    try:
        from . import policy_doctor

        checks["policy"] = policy_doctor.policy_check(engine.config, factory=engine.factory)
    except Exception as exc:  # pragma: no cover - defensive
        checks["policy"] = {"ok": False, "detail": f"policy readiness unavailable: {exc}"}

    try:
        providers = engine.provider_status()
    except Exception as exc:  # pragma: no cover - defensive
        providers = {}
        checks["providers_error"] = str(exc)

    return {"checks": checks, "providers": providers}


# Where `install.sh` puts the user-level skill. Checked rather than rewritten: doctor
# reports, it does not mutate the user's Claude Code config.
_CLAUDE_USER_SKILL = Path.home() / ".claude" / "skills" / "android-ui-analyser" / "SKILL.md"
# Backward-compatible test/extension seam for the original single-skill check.
_USER_SKILL = _CLAUDE_USER_SKILL
_CODEX_USER_SKILL = (
    Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    / "skills"
    / "android-ui-analyser"
    / "SKILL.md"
)


def _installed_skill_check(path: Path | None = None) -> dict[str, Any]:
    """Is the *installed* SKILL.md still what ``guide.py`` renders?

    The pre-commit hook keeps the two in-repo copies in sync, but nothing syncs the copy
    agents actually load — ``~/.claude/skills/…`` only changes when someone re-runs
    ``install.sh``. Observed: a guide change was committed, both repo copies updated, and
    live agents kept reading a day-old skill for hours. Nothing anywhere reported it.

    The failure is silent by construction, which is why it belongs in ``doctor``: an agent
    following stale instructions does not error, it just uses flags that no longer exist and
    misses ones that do.
    """
    target = path or _USER_SKILL
    try:
        from . import guide as guide_mod

        expected = guide_mod.render_skill()
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "detail": f"could not render the guide: {exc}"}

    if not target.is_file():
        return {
            "ok": True,  # not every install wants the user-level skill; absence is not breakage
            "detail": f"no user-level skill at {target} (plugin/project copies still apply)",
        }
    try:
        installed = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "detail": f"unreadable: {exc}"}

    if installed == expected:
        return {"ok": True, "detail": f"{target} matches guide.py"}
    return {
        "ok": False,
        "detail": f"{target} is stale — agents are reading older instructions than this build",
        "hint": f"aua guide --emit-skill {target}",
    }


def _installed_skill_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "claude": _installed_skill_check(_CLAUDE_USER_SKILL),
        "codex": _installed_skill_check(_CODEX_USER_SKILL),
    }
    checks["ok"] = all(bool(value.get("ok")) for value in checks.values())
    return checks


def _render_doctor_pretty(report: dict[str, Any]) -> str:
    def mark(ok: bool) -> str:
        return "OK  " if ok else "FAIL"

    lines: list[str] = ["aua doctor", "=========="]
    checks = report.get("checks", {})

    platform = checks.get("platform", {})
    if platform:
        lines.append(
            f"[{mark(platform.get('ok', False))}] platform      {platform.get('detail', '')}"
        )
    adb = checks.get("adb")
    if adb:
        lines.append(f"[{mark(adb.get('ok', False))}] adb           {adb.get('detail', '')}")
    u2 = checks.get("uiautomator2")
    if u2:
        lines.append(f"[{mark(u2.get('ok', False))}] uiautomator2  {u2.get('detail', '')}")
    dev = checks.get("devices", {})
    dev_detail = dev.get("detail", "")
    if isinstance(dev_detail, list):
        dev_detail = ", ".join(d.get("serial", "?") for d in dev_detail) or "(none)"
    lines.append(f"[{mark(dev.get('ok', False))}] devices       {dev_detail}")
    if dev.get("hint"):
        lines.append(f"               hint: {dev['hint']}")
    emu = checks.get("emulator", {})
    if emu:
        detail = emu.get("detail", "")
        if isinstance(detail, dict):
            avds = detail.get("avds") or []
            running = detail.get("running") or []
            detail = (
                f"bin={detail.get('binary') or 'missing'}  avds={len(avds)}  running={len(running)}"
            )
        lines.append(f"[{mark(emu.get('ok', False))}] emulator      {detail}")

    proxy = checks.get("proxy", {})
    if proxy:
        detail = proxy.get("detail", "")
        lines.append(f"[{mark(proxy.get('ok', False))}] proxy         {detail}")
        if proxy.get("hint"):
            lines.append(f"               hint: {proxy['hint']}")

    skills = checks.get("skills", {})
    if not skills and checks.get("skill"):
        skills = {"claude": checks["skill"]}
    for name in ("claude", "codex"):
        skill = skills.get(name, {}) if isinstance(skills, dict) else {}
        if skill:
            lines.append(
                f"[{mark(skill.get('ok', False))}] skill:{name:<6} {skill.get('detail', '')}"
            )
            if skill.get("hint"):
                lines.append(f"               hint: {skill['hint']}")

    policy = checks.get("policy", {})
    if policy:
        lines.append(f"[{mark(policy.get('ok', False))}] policy        {policy.get('detail', '')}")
        if policy.get("cost"):
            lines.append(f"               {policy['cost']}")
        for entry in policy.get("providers", []) or []:
            # Named one per line even when unavailable: a configured provider that cannot run is
            # the finding, and summarising it away is how it read as "no providers at all".
            lines.append(
                f"               [{mark(bool(entry.get('ok', entry.get('runnable'))))}] "
                f"{str(entry.get('name', '?')):<14} {entry.get('reason', '')}"
            )
            if entry.get("remedy"):
                lines.append(f"                      fix: {entry['remedy']}")
        if policy.get("hint"):
            lines.append(f"               hint: {policy['hint']}")

    lines.append("")
    lines.append("Providers:")
    providers = report.get("providers", {})
    for kind in ("ocr", "detection", "grounding", "planner"):
        items = providers.get(kind, [])
        lines.append(f"  {kind}:")
        if not items:
            lines.append("    (none registered)")
            continue
        for item in items:
            chain = " *" if item.get("in_chain") else "  "
            lines.append(
                f"    [{mark(item.get('available', False))}]{chain} "
                f"{item.get('name', '?'):<14} {item.get('reason', '')}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- memory / map


def _resolve_package(engine: Engine, app_pkg: str | None) -> str:
    """Use ``--app`` if given, else detect the foreground package (needs a device)."""
    if app_pkg:
        return app_pkg
    pkg = engine.current_package()
    if not pkg:
        raise UsageError(
            "could not determine the foreground app",
            hint="Pass --app <package>, or attach a device so the current app can be detected.",
        )
    return pkg


def _active_map_context(
    engine: Engine,
    opts: GlobalOpts,
    store: AppMemoryStore,
    package: str,
    *,
    explicit_package: bool,
) -> str:
    """Resolve a context without making offline ``map --app`` require a device."""
    configured_serial = opts.serial or opts.load().device.serial
    if configured_serial:
        return store.load_session(configured_serial).active_context_id
    if not explicit_package:
        return store.load_session(engine.device.serial).active_context_id
    latest = store.latest_session(package)
    return latest.active_context_id if latest else DEFAULT_CONTEXT_ID


@app.command(name="map")
def map_cmd(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package to map (default: current)."),
    brief: bool = typer.Option(False, "--brief", help="Skeleton only (screens + routes)."),
    screen: str | None = typer.Option(None, "--screen", help="Drill into one screen."),
    depth: int | None = typer.Option(
        None, "--depth", help="Compatibility option; ignored by the flat logical outline."
    ),
    find: str | None = typer.Option(None, "--find", help="Just the route to a target goal."),
    audit: bool = typer.Option(
        False, "--audit", help="Find ambiguous names/routes and research questions."
    ),
    audit_summary: bool = typer.Option(
        False,
        "--summary",
        help="With --audit, emit token-cheap issue/research counts instead of every issue.",
    ),
    context: str | None = typer.Option(
        None, "--context", help="Show/audit one feature-flag context."
    ),
    all_contexts: bool = typer.Option(False, "--all-contexts", help="Show every recorded context."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of the text tree."),
) -> None:
    """Print the app's known layout from memory (screens, key elements, routes)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(engine, app_pkg)
        app_map = store.load(pkg) or AppMap(package=pkg)
        selected_context = context or _active_map_context(
            engine,
            opts,
            store,
            pkg,
            explicit_package=app_pkg is not None,
        )
        if context and context not in app_map.contexts:
            raise UsageError(
                f"unknown map context: {context}",
                hint="Use `aua map --all-contexts --json` to list recorded contexts.",
            )
        compact = fmt is OutputFormat.compact
        if audit_summary and not audit:
            raise UsageError("--summary requires --audit", hint="Run `aua map --audit --summary`.")
        if audit:
            result = audit_map(app_map, context_id=None if all_contexts else selected_context)
            audit_payload = result.model_dump(mode="json")
            tasks = ReconciliationStore(store).plan(
                pkg, context_id=None if all_contexts else selected_context
            )
            audit_payload["research_tasks"] = [task.model_dump(mode="json") for task in tasks]
            if audit_summary:
                summary = summarize_audit(result, research_tasks=tasks)
                if as_json or compact or fmt is OutputFormat.json:
                    typer.echo(
                        json.dumps(
                            summary,
                            indent=None if compact else 2,
                            separators=(",", ":") if compact else None,
                            ensure_ascii=False,
                        )
                    )
                else:
                    counts = summary["issues"]
                    severities = counts["by_severity"]
                    typer.echo(f"# Map audit summary: {pkg} [{selected_context}]")
                    typer.echo(
                        f"Issues: {counts['total']} "
                        f"(error {severities['error']}, warning {severities['warning']}, "
                        f"info {severities['info']})"
                    )
                    typer.echo("Types:")
                    if counts["by_type"]:
                        for kind, count in counts["by_type"].items():
                            typer.echo(f"- {kind}: {count}")
                    else:
                        typer.echo("- none")
                    research = summary["research_tasks"]
                    typer.echo(
                        f"Research tasks: {research['open']} open / {research['total']} total"
                    )
                    typer.echo("Full evidence: `aua map --audit --json`.")
                return
            if as_json or compact or fmt is OutputFormat.json:
                typer.echo(
                    json.dumps(
                        audit_payload,
                        indent=None if compact else 2,
                        separators=(",", ":") if compact else None,
                        ensure_ascii=False,
                    )
                )
            else:
                typer.echo(f"# Map audit: {pkg} [{selected_context}]")
                if not result.issues:
                    typer.echo("No structural issues found.")
                for issue in result.issues:
                    typer.echo(f"- [{issue.severity}] {issue.type}: {issue.message}")
                    for question in issue.questions:
                        typer.echo(f"    ? {question}")
                if tasks:
                    typer.echo("\nResearch tasks saved:")
                    for task in tasks:
                        typer.echo(f"- {task.id} ({task.issue_type})")
            return
        if as_json or compact:
            if find:
                payload: Any = find_result(
                    app_map,
                    find,
                    None if all_contexts else selected_context,
                )
            elif screen:
                rec = app_map.screens.get(screen)
                payload = rec.model_dump(mode="json") if rec else {}
            else:
                payload = (
                    app_map if all_contexts else context_view(app_map, selected_context)
                ).model_dump(mode="json")
            sep = (",", ":") if compact else None
            indent = None if compact else 2
            typer.echo(json.dumps(payload, indent=indent, separators=sep, ensure_ascii=False))
            return
        detail = "brief" if brief else "default"
        typer.echo(
            render_map(
                app_map,
                detail=detail,
                find=find,
                screen=screen,
                depth=depth,
                context_id=selected_context,
                all_contexts=all_contexts,
            )
        )

    _run(ctx, go)


memory_app = typer.Typer(
    name="memory", help="Inspect / manage the persistent app map (§6b).", no_args_is_help=True
)
app.add_typer(memory_app, name="memory")


@app.command()
def remember(
    ctx: typer.Context,
    about: str | None = typer.Option(None, "--about", help="One-line description of the app."),
    note: str | None = typer.Option(None, "--note", help="A quirk/fact to remember."),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe NAME (needs --note)."),
    deeplink: str | None = typer.Option(
        None, "--deeplink", help="A useful deeplink URI (needs/uses --note)."
    ),
    launch_activity: str | None = typer.Option(
        None,
        "--launch-activity",
        help="Pin the MAIN/LAUNCHER Activity bare `app launch` should cold-start.",
    ),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Teach the app playbook: a description, a quirk note, a login/etc. recipe, or a deeplink.

    The agent should record what it learns so the NEXT run starts informed, e.g.
    `aua remember --recipe login_full --note "tap 'Use demo account'"` or
    `aua remember --deeplink "myapp://set-flags?flag=value" --note "set flags, then restart"`.

    `--launch-activity .ui.MainActivity` fixes a multi-launcher build: dev flavours often ship a
    Dev Tools entry beside the product one, and without a pin every cold start is a guess.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(engine, app_pkg)
        did: list[str] = []
        if about:
            store.set_description(pkg, about)
            did.append("description")
        if recipe:
            if not note:
                raise UsageError(
                    "--recipe needs --note", hint='e.g. --recipe login_full --note "tap X"'
                )
            store.remember_recipe(pkg, recipe, note)
            did.append(f"recipe:{recipe}")
        if deeplink:
            store.remember_deeplink(pkg, deeplink, note=note)
            did.append("deeplink")
        if launch_activity:
            # source="user" so later autodetection can refresh the alternatives but never
            # overwrite the entry someone deliberately taught.
            entry = store.remember_launch_entry(pkg, launch_activity, source="user")
            did.append(f"launch:{entry.activity}" if entry is not None else "launch")
        if note and not recipe and not deeplink:
            store.remember_note(pkg, note)
            did.append("note")
        if not did:
            raise UsageError(
                "remember needs something to store",
                hint=(
                    "pass --about / --note / --recipe NAME --note / --deeplink URI / "
                    "--launch-activity ACTIVITY"
                ),
            )
        typer.echo(json.dumps({"ok": True, "action": "remember", "package": pkg, "saved": did}))

    _run(ctx, go)


@app.command()
def about(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Print the app playbook — description, deeplinks, recipes, and quirks the tool learned."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(engine, app_pkg)
        app_map = store.load(pkg)
        if app_map is None:
            typer.echo(f"nothing recorded for {pkg} yet")
            return
        selected_context = _active_map_context(
            engine,
            opts,
            store,
            pkg,
            explicit_package=app_pkg is not None,
        )
        from .memory import _playbook_lines, launch_payload, playbook_view

        current = playbook_view(app_map, context_id=selected_context)
        if fmt in (OutputFormat.json, OutputFormat.compact):
            play = {
                "package": pkg,
                "context_id": selected_context,
                "description": current["description"],
                "recipes": {r.name: r.note for r in current["recipes"]},
                "deeplinks": [
                    {"uri": link.uri, "note": link.note} for link in current["deeplinks"]
                ],
                "notes": current["notes"],
                "counts": current["counts"],
            }
            play.update(launch_payload(app_map))
            typer.echo(json.dumps(play, indent=None if fmt is OutputFormat.compact else 2))
        else:
            lines = _playbook_lines(app_map, context_id=selected_context)
            typer.echo("\n".join(lines) if lines else f"no playbook for {pkg} yet")

    _run(ctx, go)


knowledge_app = typer.Typer(
    name="knowledge",
    help="Inspect and add provenance-bearing app knowledge.",
    no_args_is_help=True,
)
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("list")
def knowledge_list(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
    status: str | None = typer.Option(
        None, "--status", help="Filter accepted/proposed/stale/rejected."
    ),
) -> None:
    """List durable knowledge with source, scope, and status."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        app_map = AppMemoryStore(opts.load().memory).load(pkg) or AppMap(package=pkg)
        items = [
            item.model_dump(mode="json")
            for item in app_map.knowledge
            if status is None or item.status == status
        ]
        typer.echo(json.dumps({"package": pkg, "knowledge": items}, indent=2, ensure_ascii=False))

    _run(ctx, go)


@knowledge_app.command("show")
def knowledge_show(
    ctx: typer.Context,
    knowledge_id: str = typer.Argument(..., help="Knowledge item id."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Show one knowledge item including its evidence."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        app_map = AppMemoryStore(opts.load().memory).load(pkg) or AppMap(package=pkg)
        item = next((known for known in app_map.knowledge if known.id == knowledge_id), None)
        if item is None:
            raise UsageError(f"unknown knowledge item: {knowledge_id}")
        typer.echo(json.dumps(item.model_dump(mode="json"), indent=2, ensure_ascii=False))

    _run(ctx, go)


@knowledge_app.command("add")
def knowledge_add(
    ctx: typer.Context,
    text: str = typer.Option(..., "--text", help="Fact or experience to retain."),
    kind: str = typer.Option("claim", "--kind", help="description|note|recipe|deeplink|claim"),
    name: str | None = typer.Option(None, "--name", help="Recipe name or deeplink URI."),
    context: str | None = typer.Option(None, "--context", help="Feature-flag context scope."),
    source: str = typer.Option("agent", "--source", help="user|agent|runtime|source"),
    agent: str | None = typer.Option(None, "--agent", help="Agent/provider name."),
    session: str | None = typer.Option(None, "--session", help="External agent session id."),
    evidence: list[str] | None = typer.Option(
        None, "--evidence", help="Evidence reference; repeat for more."
    ),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Add agent feedback or source/runtime research to the app knowledge base."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        allowed_kinds = {"description", "note", "recipe", "deeplink", "claim"}
        allowed_sources = {"user", "agent", "runtime", "source"}
        if kind not in allowed_kinds or source not in allowed_sources:
            raise UsageError("invalid knowledge kind or source")
        item = AppMemoryStore(opts.load().memory).remember_knowledge(
            pkg,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            name=name,
            context_id=context,
            source=source,  # type: ignore[arg-type]
            agent=agent,
            session=session,
            evidence=[KnowledgeEvidence(kind="agent", ref=ref) for ref in (evidence or [])],
        )
        typer.echo(json.dumps(item.model_dump(mode="json") if item else {}, indent=2))

    _run(ctx, go)


@knowledge_app.command("stale")
def knowledge_stale(
    ctx: typer.Context,
    knowledge_id: str = typer.Argument(..., help="Knowledge item id."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Mark one learned fact stale without deleting its evidence."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        store = AppMemoryStore(opts.load().memory)
        app_map = store.load(pkg) or AppMap(package=pkg)
        item = next((known for known in app_map.knowledge if known.id == knowledge_id), None)
        if item is None:
            raise UsageError(f"unknown knowledge item: {knowledge_id}")
        item.status = "stale"
        store.save(app_map)
        typer.echo(json.dumps({"ok": True, "id": knowledge_id, "status": "stale"}))

    _run(ctx, go)


reconcile_app = typer.Typer(
    name="reconcile",
    help="Exchange map research with an external agent and apply validated corrections.",
    no_args_is_help=True,
)
app.add_typer(reconcile_app, name="reconcile")


def _read_json_document(path: str) -> dict[str, Any]:
    import json
    import sys

    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise UsageError("expected a JSON object")
    return parsed


@reconcile_app.command("plan")
def reconcile_plan(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
    context: str | None = typer.Option(None, "--context", help="Audit one flag context."),
) -> None:
    """Emit canonical research tasks for an external coding/runtime agent."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        store = AppMemoryStore(opts.load().memory)
        selected_context = context or _active_map_context(
            engine,
            opts,
            store,
            pkg,
            explicit_package=app_pkg is not None,
        )
        tasks = ReconciliationStore(store).plan(pkg, context_id=selected_context)
        typer.echo(
            json.dumps(
                {"package": pkg, "tasks": [task.model_dump(mode="json") for task in tasks]},
                indent=2,
                ensure_ascii=False,
            )
        )

    _run(ctx, go)


@reconcile_app.command("submit")
def reconcile_submit(
    ctx: typer.Context,
    report: str = typer.Argument("-", help="Research report JSON file, or - for stdin."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Submit an agent verdict; `apply` is committed automatically and transactionally."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        try:
            parsed = ResearchReport.model_validate(_read_json_document(report))
            result = ReconciliationStore(AppMemoryStore(opts.load().memory)).submit(pkg, parsed)
        except (ValueError, OSError) as exc:
            raise UsageError(str(exc)) from exc
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))

    _run(ctx, go)


@reconcile_app.command("answers")
def reconcile_answers(
    ctx: typer.Context,
    document: str = typer.Argument("-", help="Answers JSON file, or - for stdin."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would happen and write nothing."
    ),
) -> None:
    """Answer many naming questions offline, in one transaction with one rollback id.

    Takes `{"answers": {"<task-id>": "<name>", ...}}`, or a list of
    `{"task_id": ..., "value": ...}` when the same id appears twice or the order matters — the
    caller's order decides who wins a contested name. Needs no device: pass `--app`.

    `--answers` remains the way to answer the one question about the screen you are standing on.
    This is for draining a backlog an agent has already researched, e.g. from `reconcile plan`.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        raw = _read_json_document(document)
        payload = raw.get("answers", raw) if isinstance(raw, dict) else raw
        pairs: list[tuple[str, str]] = []
        if isinstance(payload, dict):
            pairs = [(str(k), str(v)) for k, v in payload.items()]
        elif isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict) or "task_id" not in row or "value" not in row:
                    raise UsageError(
                        "each answer needs task_id and value",
                        hint='e.g. {"answers": [{"task_id": "research_ab12", "value": "Dev Tools"}]}',
                    )
                pairs.append((str(row["task_id"]), str(row["value"])))
        else:
            raise UsageError(
                "answers must be an object of task_id: name, or a list of {task_id, value}"
            )
        try:
            result = ReconciliationStore(AppMemoryStore(opts.load().memory)).answer_many(
                pkg, pairs, agent="bulk", dry_run=dry_run
            )
        except (ValueError, OSError) as exc:
            raise UsageError(str(exc)) from exc
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))

    _run(ctx, go)


@reconcile_app.command("status")
def reconcile_status(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Show open tasks, queued reports, and correction events."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        result = ReconciliationStore(AppMemoryStore(opts.load().memory)).status(pkg)
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))

    _run(ctx, go)


@reconcile_app.command("apply")
def reconcile_apply(
    ctx: typer.Context,
    task_id: str = typer.Argument(..., help="Queued review task id to apply."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Apply a queued review report after a human/agent decision."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        store = AppMemoryStore(opts.load().memory)
        app_map = store.load(pkg) or AppMap(package=pkg)
        raw = next(
            (item for item in app_map.pending_reports if item.get("task_id") == task_id),
            None,
        )
        if raw is None:
            raise UsageError(f"no queued report for task: {task_id}")
        report = ResearchReport.model_validate({**raw, "verdict": "apply"})
        event = ReconciliationStore(store).apply(pkg, report)
        typer.echo(json.dumps(event.model_dump(mode="json"), indent=2, ensure_ascii=False))

    _run(ctx, go)


@reconcile_app.command("rollback")
def reconcile_rollback(
    ctx: typer.Context,
    rollback_id: str = typer.Argument(..., help="Correction event/rollback id."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Restore the exact map snapshot from before a correction event."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        pkg = _resolve_package(engine, app_pkg)
        try:
            event = ReconciliationStore(AppMemoryStore(opts.load().memory)).rollback(
                pkg, rollback_id
            )
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        typer.echo(json.dumps(event.model_dump(mode="json"), indent=2, ensure_ascii=False))

    _run(ctx, go)


@memory_app.command("show")
def memory_show(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
    screen: str | None = typer.Option(None, "--screen", help="Show one screen's full detail."),
) -> None:
    """Inspect the recorded map (whole app, or one ``--screen``)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(engine, app_pkg)
        app_map = store.load(pkg)
        if app_map is None:
            typer.echo(f"no memory recorded for {pkg} yet (run `aua analyze` while navigating)")
            return
        if fmt in (OutputFormat.json, OutputFormat.compact):
            sep = (",", ":") if fmt is OutputFormat.compact else None
            indent = None if fmt is OutputFormat.compact else 2
            data = (
                app_map.screens[screen].model_dump(mode="json")
                if screen and screen in app_map.screens
                else app_map.model_dump(mode="json")
            )
            typer.echo(json.dumps(data, indent=indent, separators=sep, ensure_ascii=False))
        else:
            typer.echo(render_map(app_map, detail="default", screen=screen))

    _run(ctx, go)


@memory_app.command("path")
def memory_path(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Print where this app's memory lives on disk."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(engine, app_pkg)
        typer.echo(str(store.app_dir(pkg)))

    _run(ctx, go)


@memory_app.command("update")
def memory_update_cmd(
    ctx: typer.Context,
    screen: str | None = typer.Option(
        None, "--screen", help="Name (or rename) the current screen."
    ),
) -> None:
    """Force-record the current screen now (recording is automatic by default)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "memory_update", screen_name=screen), fmt)

    _run(ctx, go)


@memory_app.command("forget")
def memory_forget(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package to forget (required)."),
    screen: str | None = typer.Option(None, "--screen", help="Forget just this one screen."),
) -> None:
    """Clear an app's memory (or one ``--screen``). Requires ``--app`` for safety."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        if not app_pkg:
            raise UsageError(
                "memory forget requires --app <package>",
                hint="Scope the deletion explicitly, e.g. `aua memory forget --app com.x`.",
            )
        store = AppMemoryStore(_opts(ctx).load().memory)
        result = store.forget(app_pkg, screen)
        typer.echo(
            json.dumps({"ok": True, "action": "memory-forget", **result}, ensure_ascii=False)
        )

    _run(ctx, go)


# --------------------------------------------------------------------------- explore


explore_app = typer.Typer(
    name="explore",
    help="Discover app knowledge — mine deeplink shortcuts from source, plan a crawl (§6b).",
    no_args_is_help=True,
)
app.add_typer(explore_app, name="explore")


@explore_app.command("mine")
def explore_mine_cmd(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Path to the app's source tree (repo root)."),
    app_pkg: str | None = typer.Option(
        None, "--app", help="Package to attribute the deeplinks to (default: current)."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save found deeplinks to the app playbook."
    ),
) -> None:
    """Scan an app's source for deeplinks (shortcuts) and record them in its playbook.

    Deeplinks let you jump straight to a screen — `aua open-and-analyze "myapp://catalog/books"`
    instead of tapping through the app's menus. Run once per app; `aua about` then lists them.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "explore_mine", source=source, package=app_pkg, save=save), fmt)

    _run(ctx, go)


@explore_app.command("plan")
def explore_plan_cmd(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
    max_tasks: int = typer.Option(12, "--max-tasks", help="Cap on returned tasks."),
) -> None:
    """Get a risk-classified worklist: map debt, safe dead ends, then deeplinks.

    Run the tasks with normal `aua` commands — results auto-record into the map/playbook,
    so re-running the plan shows what's left. Listed tasks are not authorization for any
    destructive or external effect.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "explore_plan", package=app_pkg, max_tasks=max_tasks), fmt)

    _run(ctx, go)


# --------------------------------------------------------------------------- flows


flow_app = typer.Typer(
    name="flow",
    help="Author, save, and replay whole journeys in one call (Maestro-style flows, §6b).",
    no_args_is_help=True,
)
app.add_typer(flow_app, name="flow")


def _parse_params(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise UsageError(f"bad --param '{pair}'", hint="use --param NAME=value")
        k, v = pair.split("=", 1)
        params[k.strip()] = v
    return params


@flow_app.command("run")
def flow_run_cmd(
    ctx: typer.Context,
    name: str | None = typer.Argument(
        None, help="Saved flow name, or <package>:<name> when apps share it (`aua flow list`)."
    ),
    param: list[str] = typer.Option(  # noqa: B008 - typer option factory
        [], "--param", "-p", help="Substitute ${NAME} placeholders: --param NAME=value."
    ),
    file: str | None = typer.Option(None, "--file", help="Run a flow YAML file directly."),
    yaml: str | None = typer.Option(
        None, "--yaml", help="Run an inline flow YAML body (useful for MCP-equivalent scripts)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved steps; do not act."),
    from_step: int = typer.Option(0, "--from-step", help="Resume from this step index."),
    allow_destructive: bool = typer.Option(
        True,
        "--allow-destructive/--no-allow-destructive",
        help="Authored flows may take destructive steps by default.",
    ),
    assist: bool = typer.Option(
        False,
        "--assist",
        help="On divergence, let the opt-in planner LLM clear the blocker (needs planner.enabled).",
    ),
    artifacts_dir: str | None = typer.Option(
        None,
        "--artifacts-dir",
        help="Write flow.yaml, result.json, report.md, evidence, and optional JUnit here.",
    ),
    evidence: str = typer.Option(
        "failures", "--evidence", help="Screenshot evidence: none, failures, or all."
    ),
    junit: bool = typer.Option(False, "--junit", help="Also write junit.xml (needs artifacts)."),
) -> None:
    """Replay a whole journey in one call; on divergence returns a resumable step index."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        # Make the path absolute *here*, in the process the user invoked. `_route` may hand
        # this call to the warm daemon, whose working directory is wherever it happened to
        # be started — so a relative `--file` was looked up against a directory the caller
        # has never seen, and reported as missing even though `cwd` plainly contained it.
        # An absolute path always worked, which is exactly the shape of this bug.
        resolved = str(Path(file).expanduser().resolve()) if file else None
        resolved_artifacts = (
            str(Path(artifacts_dir).expanduser().resolve()) if artifacts_dir else None
        )
        result = _route(
            engine,
            "flow_run",
            name=name,
            file=resolved,
            yaml=yaml,
            params=_parse_params(param),
            dry_run=dry_run,
            from_step=from_step,
            allow_destructive=allow_destructive,
            assist=assist,
            artifacts_dir=resolved_artifacts,
            evidence=evidence,
            junit=junit,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


@flow_app.command("save")
def flow_save_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name for the saved flow."),
    last: int = typer.Option(
        12,
        "--last",
        min=1,
        help="How many recent actions to materialize.",
    ),
    save: bool = typer.Option(
        False,
        "--save",
        help="Write the previewed flow. Without this flag, flow save writes nothing.",
    ),
    force: bool = typer.Option(False, "--force", help="With --save, overwrite an existing flow."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Deprecated compatibility alias for the default non-writing preview.",
    ),
) -> None:
    """Preview scope, selector resilience, and captured arrival proof; --save writes."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "flow_save",
            name=name,
            last=last,
            save=save,
            force=force,
            dry_run=dry_run,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


@flow_app.command("list")
def flow_list_cmd(
    ctx: typer.Context,
    app_filter: str | None = typer.Option(
        None, "--app", help="Only flows this package can run (its own plus app-agnostic ones)."
    ),
) -> None:
    """List saved flows (name, app, steps, params); --app narrows to one package."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        typer.echo(json.dumps(engine.flow_list(app=app_filter), indent=2, ensure_ascii=False))

    _run(ctx, go)


@flow_app.command("show")
def flow_show_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Flow name, or <package>:<name> when apps share it."),
) -> None:
    """Print a saved flow's YAML (edit it in place under memory.dir/flows/<package>/)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        from .flows import FlowStore

        store = FlowStore(_opts(ctx).load().memory)
        typer.echo(store.resolve(name).read_text(encoding="utf-8"))

    _run(ctx, go)


@flow_app.command("delete")
def flow_delete_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Flow to delete, or <package>:<name> when apps share it."),
) -> None:
    """Idempotently delete a saved flow (an already-absent flow is success)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "flow_delete", name=name), fmt)

    _run(ctx, go)


# --------------------------------------------------------------------------- logcat


logcat_app = typer.Typer(
    help="Mark + dump device logcat (filter by mark / grep / tag).",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(logcat_app, name="logcat")


@logcat_app.callback(invoke_without_command=True)
def logcat_cmd(
    ctx: typer.Context,
    grep: str | None = typer.Option(None, "--grep", help="Regex filter on log lines."),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Mark name, last-action, duration (30s), or unix-ms. Default: last-action or 30s.",
    ),
    tag: str | None = typer.Option(None, "--tag", help="Exact log tag filter."),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON."),
    lines: int | None = typer.Option(
        None, "--lines", "-n", help="Keep only the last N matching lines."
    ),
) -> None:
    """Dump recent logcat (default since last-action mark, else last 30s)."""
    if ctx.invoked_subcommand is not None:
        return

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        result = engine.logcat(grep=grep, since=since, tag=tag, lines=lines)
        if as_json:
            indent = 2 if fmt is OutputFormat.pretty else None
            sep = None if indent else (",", ":")
            typer.echo(json.dumps(result, indent=indent, separators=sep, ensure_ascii=False))
        else:
            for line in result.get("lines") or []:
                typer.echo(line)

    _run(ctx, go)


@logcat_app.command("mark")
def logcat_mark_cmd(
    ctx: typer.Context,
    name: str = typer.Argument("default", help="Mark name (default: default)."),
    clear: bool = typer.Option(
        False, "--clear", help="Also clear the device logcat buffer (`logcat -c`)."
    ),
) -> None:
    """Store a device-clock mark for later ``aua logcat --since NAME``.

    Reports the device timestamp plus the measured host↔device skew, because logcat lines
    are stamped by the device and the two clocks drift (seconds, on emulators).
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        result = engine.logcat_mark(name, clear=clear)
        typer.echo(json.dumps(result, ensure_ascii=False))

    _run(ctx, go)


# --------------------------------------------------------------------------- session capture


capture_app = typer.Typer(
    help="Rolling screencap buffer (always-on with daemon) — recover fast loading flashes.",
    no_args_is_help=True,
)
app.add_typer(capture_app, name="capture")


@capture_app.command("status")
def capture_status_cmd(ctx: typer.Context) -> None:
    """Show whether the buffer is running, fps mode, frame count, disk use."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "capture_status"), fmt)

    _run(ctx, go)


@capture_app.command("last")
def capture_last_cmd(
    ctx: typer.Context,
    seconds: float | None = typer.Option(
        None, "--seconds", help="Only frames from the last N seconds."
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="last-action — frames since the last tap/input/swipe mark.",
    ),
    region: str | None = typer.Option(
        None,
        "--region",
        help="Filter diff summary to a grid region (center, upper, left, …).",
    ),
    where_rid: str | None = typer.Option(
        None,
        "--where-rid",
        help="Infer --region from a resource-id's last-known center cell.",
    ),
) -> None:
    """Emit timeline JSON + frame paths + cheap local diff summary."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "capture_last",
                seconds=seconds,
                since=since,
                region=region,
                where_rid=where_rid,
            ),
            fmt,
        )

    _run(ctx, go)


@capture_app.command("export")
def capture_export_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Output .gif (or .mp4 with imageio)."),
    seconds: float | None = typer.Option(None, "--seconds"),
    since: str | None = typer.Option(None, "--since", help="last-action"),
    fmt: str = typer.Option("gif", "--format", help="gif|mp4"),
    fps: float = typer.Option(8.0, "--fps"),
) -> None:
    """Assemble recent kept frames into a GIF (or MP4)."""

    def go(engine: Engine, fmt_out: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "capture_export",
                path=path,
                seconds=seconds,
                since=since,
                fmt=fmt,
                fps=fps,
            ),
            fmt_out,
        )

    _run(ctx, go)


@capture_app.command("explain")
def capture_explain_cmd(
    ctx: typer.Context,
    seconds: float | None = typer.Option(None, "--seconds"),
    since: str | None = typer.Option(None, "--since", help="last-action"),
    llm: bool = typer.Option(False, "--llm", help="Also ask the opt-in planner LLM to narrate."),
) -> None:
    """Narrate the recent capture window (local diff summary; optional --llm)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(engine, "capture_explain", seconds=seconds, since=since, llm=llm),
            fmt,
        )

    _run(ctx, go)


@capture_app.command("on")
def capture_on_cmd(ctx: typer.Context) -> None:
    """Resume (or start) the rolling capture buffer."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "capture_on"), fmt)

    _run(ctx, go)


@capture_app.command("off")
def capture_off_cmd(ctx: typer.Context) -> None:
    """Pause capture without stopping the daemon."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "capture_off"), fmt)

    _run(ctx, go)


@capture_app.command("prune")
def capture_prune_cmd(ctx: typer.Context) -> None:
    """Force TTL / max-size prune of old frames."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "capture_prune"), fmt)

    _run(ctx, go)


@capture_app.command("sidecar")
def capture_sidecar_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="start|stop"),
) -> None:
    """Host capture process that survives without the full warm daemon."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        a = action.lower()
        if a == "start":
            _emit(engine.capture_sidecar_start(), fmt)
        elif a == "stop":
            _emit(engine.capture_sidecar_stop(), fmt)
        else:
            raise UsageError("use `aua capture sidecar start` or `stop`")

    _run(ctx, go)


# --------------------------------------------------------------------------- developer options / a11y / flags / proxy


dev_app = typer.Typer(
    help="Developer options — anim scales, crash dialogs, AC profiles (always restore).",
    no_args_is_help=True,
)
app.add_typer(dev_app, name="dev")


@dev_app.command("show")
def dev_show_cmd(ctx: typer.Context) -> None:
    """Print current anim scales, crash/ANR flags, don't-keep-activities."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.dev_show(), fmt)

    _run(ctx, go)


@dev_app.command("anim")
def dev_anim_cmd(
    ctx: typer.Context,
    mode: str = typer.Argument(..., help="off|restore"),
) -> None:
    """Turn animations off (saving prior scales) or restore them."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.dev_anim(mode), fmt)

    _run(ctx, go)


@dev_app.command("crashes")
def dev_crashes_cmd(
    ctx: typer.Context,
    mode: str = typer.Argument(..., help="on|off"),
) -> None:
    """Show or hide crash/ANR dialogs (``show_crash_dialog`` + background ANR)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        m = mode.lower()
        if m not in ("on", "off"):
            raise UsageError("use `aua dev crashes on` or `aua dev crashes off`")
        _emit(engine.dev_crashes(m == "on"), fmt)

    _run(ctx, go)


@dev_app.command("profile")
def dev_profile_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="ac|default"),
) -> None:
    """Apply a named profile: ``ac`` (anim off + crashes on) or ``default`` (restore)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.dev_profile(name), fmt)

    _run(ctx, go)


a11y_app = typer.Typer(
    help="Accessibility actions — scroll / expand / dismiss via the a11y tree.",
    no_args_is_help=True,
)
app.add_typer(a11y_app, name="a11y")


@a11y_app.command("scroll-and-analyze", cls=AnalyzeCommand)
@a11y_app.command("scroll", hidden=True)
def a11y_scroll_cmd(
    ctx: typer.Context,
    ident: str | None = typer.Argument(None, help="Element id from the last analyze."),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    forward: bool = typer.Option(False, "--forward", help="Scroll forward (default)."),
    backward: bool = typer.Option(False, "--backward", help="Scroll backward."),
    no_observe: bool = typer.Option(False, "--no-observe", help="Skip post-action analyze."),
) -> None:
    """Accessibility scroll on a scrollable node (prefer over coordinate swipe)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        if forward and backward:
            raise UsageError("pass only one of --forward / --backward")
        direction = "backward" if backward else "forward"
        sel = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            engine.a11y_scroll(
                _element_id(ident, sel),
                selector=sel,
                direction=direction,
                observe=not no_observe,
            ),
            fmt,
        )

    _run(ctx, go)


@a11y_app.command("action-and-analyze", cls=AnalyzeCommand)
@a11y_app.command("action", hidden=True)
def a11y_action_cmd(
    ctx: typer.Context,
    ident_or_action: str = typer.Argument(
        ...,
        metavar="ID_OR_ACTION",
        help="Element id followed by action, or just ACTION when a --rid/--text/--desc selector is used.",
    ),
    action: str | None = typer.Argument(
        None,
        metavar="[ACTION]",
        help="CLICK|LONG_CLICK|SCROLL_FORWARD|SCROLL_BACKWARD|EXPAND|COLLAPSE|DISMISS.",
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    key: str | None = _SEL_KEY,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    no_observe: bool = typer.Option(False, "--no-observe", help="Skip post-action analyze."),
) -> None:
    """Perform a named accessibility action on an element."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        ident: str | None = ident_or_action
        action_name = action
        selector_given = any(value is not None for value in (by, rid, text, desc))
        if action_name is None:
            if not selector_given:
                raise UsageError(
                    "a11y action needs both an element id and an action, or a selector plus an action",
                    hint=(
                        "Use `aua a11y action-and-analyze 7 CLICK`, or "
                        "`aua a11y action-and-analyze --rid expandButton CLICK`."
                    ),
                )
            action_name = ident_or_action
            ident = None
        sel = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, key=key, index=index, first=first
        )
        _emit(
            engine.a11y_action(
                _element_id(ident, sel),
                selector=sel,
                action=action_name,
                observe=not no_observe,
            ),
            fmt,
        )

    _run(ctx, go)


flags_app = typer.Typer(
    help="Feature flags via a configured package deeplink template (`flags.templates`).",
    no_args_is_help=True,
)
app.add_typer(flags_app, name="flags")

_FLAGS_RESTART = typer.Option(
    True,
    "--restart/--no-restart",
    help="Force-stop + relaunch after writing (flags read at cold start need it).",
)
_FLAGS_ACTIVITY = typer.Option(
    None, "--activity", help="Pin the relaunch entry Activity (default: the one in front)."
)
_FLAGS_VERIFY = typer.Option(
    True,
    "--verify/--no-verify",
    help="Read the app's shared_prefs back and report which keys actually landed.",
)
_FLAGS_PREFS_FILE = typer.Option(
    None, "--prefs-file", help="shared_prefs XML to verify against (default: search all)."
)
_FLAGS_NOT_APPLIED_HINT = (
    "The app only honours keys it knows, so a dropped key is a broken precondition for "
    "whatever runs next: check the spelling against the app's flag registry. "
    "`--no-verify` returns to fire-and-forget."
)
_FLAGS_NO_RESTART_HINT = (
    "The flags are set but the app is not running, so the next command has nothing to "
    "read. Pin the entry point with `--activity <exported launcher activity>`."
)


def _exit_unless_flags_ok(result: dict[str, Any]) -> None:
    """Non-zero when the flags did not land, or when the app did not come back."""
    lost = result.get("ignored") or result.get("mismatched")
    if result.get("restart_error") and not lost:
        _exit_unless_ok(
            result, ExitCode.ASSERTION, code="app_not_restarted", hint=_FLAGS_NO_RESTART_HINT
        )
    _exit_unless_ok(
        result, ExitCode.ASSERTION, code="flags_not_applied", hint=_FLAGS_NOT_APPLIED_HINT
    )


@flags_app.command("set-and-analyze", cls=AnalyzeCommand)
@flags_app.command("set", hidden=True)
def flags_set_cmd(
    ctx: typer.Context,
    package: str = typer.Argument(..., help="App package id."),
    assignments: list[str] = typer.Argument(..., help="KEY=VAL pairs."),
    no_observe: bool = typer.Option(False, "--no-observe"),
    restart: bool = _FLAGS_RESTART,
    activity: str | None = _FLAGS_ACTIVITY,
    verify: bool = _FLAGS_VERIFY,
    prefs_file: str | None = _FLAGS_PREFS_FILE,
) -> None:
    """Set flags through the package's deeplink, verify them, and restart the app.

    Both defaults exist because the alternative lies to you: an app that reads a flag at
    cold start ignores an override written into the live process, and a deeplink for a key
    the app does not know is dropped silently. Verified keys land in ``applied``, dropped
    ones in ``ignored`` — a non-empty ``ignored`` exits 8.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.flags_set(
            package,
            assignments,
            observe=not no_observe,
            restart=restart,
            activity=activity,
            verify=verify,
            prefs_file=prefs_file,
        )
        _emit(result, fmt)
        _exit_unless_flags_ok(result)

    _run(ctx, go)


@flags_app.command("apply-and-analyze", cls=AnalyzeCommand)
@flags_app.command("apply", hidden=True)
def flags_apply_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="YAML file: optional app: + flags: mapping."),
    package: str | None = typer.Option(None, "--package", "-p", help="Override package."),
    no_observe: bool = typer.Option(False, "--no-observe"),
    restart: bool = _FLAGS_RESTART,
    activity: str | None = _FLAGS_ACTIVITY,
    verify: bool = _FLAGS_VERIFY,
    prefs_file: str | None = _FLAGS_PREFS_FILE,
) -> None:
    """Batch-apply flags from a YAML file (verifies + restarts like `flags set`)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.flags_apply(
            path,
            package=package,
            observe=not no_observe,
            restart=restart,
            activity=activity,
            verify=verify,
            prefs_file=prefs_file,
        )
        _emit(result, fmt)
        _exit_unless_flags_ok(result)

    _run(ctx, go)


proxy_app = typer.Typer(
    help="Headless mitmproxy — device http_proxy + adb reverse (optional proxy extra).",
    no_args_is_help=True,
)
app.add_typer(proxy_app, name="proxy")


@proxy_app.command("start")
def proxy_start_cmd(
    ctx: typer.Context,
    port: int = typer.Option(
        0,
        "--port",
        help="mitmdump listen port (0 = pick a free random high port; never defaults to 8080).",
    ),
    install_ca: bool = typer.Option(
        True,
        "--install-ca/--no-install-ca",
        help="Install mitm CA as a system trust anchor (needs rootable emulator).",
    ),
) -> None:
    """Start mitmdump, adb-reverse, set device HTTP proxy, and (by default) install the CA."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.proxy_start(port=port or None, install_ca=install_ca), fmt)

    _run(ctx, go)


@proxy_app.command("stop")
def proxy_stop_cmd(ctx: typer.Context) -> None:
    """Clear the device proxy and stop mitmdump."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.proxy_stop(), fmt)

    _run(ctx, go)


@proxy_app.command("ca-install")
def proxy_ca_install_cmd(ctx: typer.Context) -> None:
    """Install the mitm CA into the system trust store (Android 14+ zygote overlay)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        pm = engine.platform.capability("proxy")

        _emit(pm.install_system_ca(engine.device.serial), fmt)

    _run(ctx, go)


@proxy_app.command("status")
def proxy_status_cmd(
    ctx: typer.Context,
    heal: bool = typer.Option(
        True,
        "--heal/--no-heal",
        help=(
            "Re-establish a dropped `adb reverse` tunnel automatically when the process and "
            "device setting both already check out."
        ),
    ),
) -> None:
    """Is interception ACTUALLY working — process alive, tunnel reachable, setting armed?

    Checks the three pieces that all have to hold together, not just whichever one is easiest
    to see: the mitmdump process being alive and listening does not mean the device can reach
    it, and the device's `http_proxy` setting pointing at the right port does not mean anything
    is on the other end. A dropped `adb reverse` tunnel leaves every one of those individually
    "fine" while every app's network call silently fails — this is the check that catches it.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.proxy_status(heal=heal)
        _emit(result, fmt)
        # `_emit` sets no exit code. Without this tail a black-holed device — every app request
        # failing — exited 0, so no script or agent could branch on it.
        if result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


helper_app = typer.Typer(
    help=(
        "Optional on-device helper APK (AccessibilityService): pushed screen-change events "
        "and in-process hierarchy reads. Off by default; polling stays the fallback."
    ),
    no_args_is_help=True,
)
app.add_typer(helper_app, name="helper")


demo_app = typer.Typer(
    help=(
        "Record a journey a PERSON performs on the device, as flow steps. "
        "Needs the on-device helper. Nothing else may drive the device while it runs."
    ),
    no_args_is_help=True,
)
app.add_typer(demo_app, name="demo")


@demo_app.command("start")
def demo_start_cmd(ctx: typer.Context) -> None:
    """Arm the recorder, then hand the device to a human and get out of the way.

    Unlike every other authoring path, nothing is driven from here: the person demonstrates
    the journey and the device reports what it saw, so there is no agent turn per step and no
    selector to guess. Run `aua demo stop` when they are done.

    Do not run other aua commands until then. They connect uiautomator2, which suppresses the
    accessibility service the recorder is listening through.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.demo_record_start(), fmt)

    _run(ctx, go)


@demo_app.command("stop")
def demo_stop_cmd(
    ctx: typer.Context,
    save: str | None = typer.Option(
        None, "--save", help="Write the journey as a named flow (refuses an incomplete one)."
    ),
    force: bool = typer.Option(
        False, "--force", help="With --save, write the draft even though it has holes."
    ),
) -> None:
    """Stop recording and return the journey as steps, with any holes named.

    Two sources are combined. Accessibility reports a tap only when the view announces one,
    and plenty do not; the kernel touch stream sees every press there was. A press with no
    announcement is matched against what was on screen just before it, which recovers the
    button by name — `recovered_from_touches` counts those.

    What is left over is reported in `gaps`: the screen changed, nothing announced it, and no
    touch explains it. `--save` refuses a draft that has any, because a saved flow which skips
    a step fails later somewhere else and reads as a product bug.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.demo_record_stop(save=save, force=force), fmt)

    _run(ctx, go)


@helper_app.command("status")
def helper_status_cmd(ctx: typer.Context) -> None:
    """Report whether the helper is installed, enabled, and actually bound."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        state = agent.status(engine.device.serial)
        state["ok"] = True
        state["action"] = "helper-status"
        _emit(state, fmt)

    _run(ctx, go)


@helper_app.command("install")
def helper_install_cmd(
    ctx: typer.Context,
    reinstall: bool = typer.Option(False, "--reinstall", help="Reinstall even if present."),
    force: bool = typer.Option(
        False, "--force", help="Install even on a target that cannot run it (rarely wanted)."
    ),
) -> None:
    """Push the bundled helper APK to the target (does not switch it on)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        result = agent.install(engine.device.serial, reinstall=reinstall, force=force)
        _emit({"ok": True, "action": "helper-install", **result}, fmt)

    _run(ctx, go)


@helper_app.command("enable")
def helper_enable_cmd(ctx: typer.Context) -> None:
    """Install if needed and switch the service on. Requires a rootable target."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        result = agent.enable(engine.device.serial)
        _emit({"ok": True, "action": "helper-enable", **result}, fmt)

    _run(ctx, go)


@helper_app.command("disable")
def helper_disable_cmd(ctx: typer.Context) -> None:
    """Switch the service off, leaving any other accessibility service alone."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        result = agent.disable(engine.device.serial)
        _emit({"ok": True, "action": "helper-disable", **result}, fmt)

    _run(ctx, go)


@helper_app.command("remove")
def helper_remove_cmd(ctx: typer.Context) -> None:
    """Switch off and uninstall the helper."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        result = agent.remove(engine.device.serial)
        _emit({"ok": True, "action": "helper-remove", **result}, fmt)

    _run(ctx, go)


@helper_app.command("tree")
def helper_tree_cmd(
    ctx: typer.Context,
    all_windows: bool = typer.Option(
        False, "--all-windows", help="Every window, not just the active one."
    ),
) -> None:
    """Read the hierarchy through the helper — no adb dump, no UiAutomation slot."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        channel = agent.open_channel(engine.device.serial)
        try:
            result = channel.request("ui.tree", {"all_windows": all_windows})
        finally:
            channel.close()
        _emit({"ok": True, "action": "helper-tree", **result}, fmt)

    _run(ctx, go)


@helper_app.command("watch")
def helper_watch_cmd(
    ctx: typer.Context,
    timeout_ms: int = typer.Option(5000, "--timeout-ms", help="How long to listen."),
) -> None:
    """Stream pushed screen-change events until the timeout, then report them."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        agent = engine.platform.capability("device_agent")
        channel = agent.open_channel(engine.device.serial)
        try:
            channel.subscribe()
            collected = list(channel.events(timeout=max(0.0, timeout_ms / 1000.0)))
        finally:
            channel.close()
        _emit(
            {"ok": True, "action": "helper-watch", "count": len(collected), "events": collected},
            fmt,
        )

    _run(ctx, go)


mock_app = typer.Typer(
    help="HTTP mock map / record / replay (YAML cassettes under memory.dir/cassettes/).",
    no_args_is_help=True,
)
app.add_typer(mock_app, name="mock")


@mock_app.command("map")
def mock_map_cmd(
    ctx: typer.Context,
    method: str = typer.Argument(..., help="HTTP method (GET|POST|…)."),
    path: str = typer.Argument(..., help="Path to match (prefix or exact)."),
    status: int = typer.Option(200, "--status", help="Response status."),
    body: str | None = typer.Option(None, "--body", help="JSON or raw body string."),
    host: str | None = typer.Option(None, "--host", help="Restrict the rule to one host."),
    times: int = typer.Option(0, "--times", help="Fire at most N times (0 = unlimited)."),
) -> None:
    """Add a live mock rule (reloaded by the mitmproxy addon)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_map(method, path, status=status, body=body, host=host, times=times), fmt)

    _run(ctx, go)


@mock_app.command("rewrite")
def mock_rewrite_cmd(
    ctx: typer.Context,
    method: str = typer.Argument(..., help="HTTP method (GET|POST|…), or `*` for any."),
    path: str = typer.Argument(..., help="Path to match (prefix or exact)."),
    host: str | None = typer.Option(None, "--host", help="Restrict to one host."),
    query: str | None = typer.Option(None, "--query", help="Substring the query must contain."),
    request_body: str | None = typer.Option(
        None, "--request-body", help="Substring the request body must contain."
    ),
    status: int | None = typer.Option(None, "--status", help="Replace the response status."),
    header: list[str] = typer.Option(
        [], "--header", help="Set a response header, `Name: value` (repeatable)."
    ),
    body: str | None = typer.Option(None, "--body", help="Replace the whole response body."),
    set_json: list[str] = typer.Option(
        [], "--set", help="Set a JSON field, `a.b[0]=<json>` (repeatable)."
    ),
    delete_json: list[str] = typer.Option(
        [], "--delete", help="Delete a JSON field by path (repeatable)."
    ),
    replace: list[str] = typer.Option(
        [], "--replace", help="Literal body substitution, `old=>new` (repeatable)."
    ),
    times: int = typer.Option(0, "--times", help="Fire at most N times (0 = unlimited)."),
) -> None:
    """Patch the REAL response instead of stubbing it (reloaded by the mitmproxy addon).

    `mock map` answers from the rule and the server never sees the request. This lets the
    request through and edits what comes back — the status an app sees, a header, a JSON
    field — which is how you reproduce a server-side condition you cannot trigger on demand.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        # Parsed in here, not in the command body: `_run` is what turns a UsageError into
        # the one-line JSON envelope every other refusal uses. Raised outside it, a
        # malformed `--set` printed a forty-line Python traceback and exited 1 instead of 2.
        import json

        headers: dict[str, str] = {}
        for item in header:
            name, sep, value = item.partition(":")
            if not sep or not name.strip():
                raise UsageError(
                    f"--header {item!r} is not `Name: value`",
                    hint="Example: --header 'Retry-After: 30'.",
                )
            headers[name.strip()] = value.strip()

        sets: dict[str, Any] = {}
        for item in set_json:
            field, sep, raw = item.partition("=")
            if not sep or not field.strip():
                raise UsageError(
                    f"--set {item!r} is not `path=value`",
                    hint='Example: --set \'items[0].title="hi"\' or --set enabled=false.',
                )
            try:
                sets[field.strip()] = json.loads(raw)
            except json.JSONDecodeError:
                sets[field.strip()] = raw  # a bare string is the common case

        pairs: list[tuple[str, str]] = []
        for item in replace:
            old, sep, new = item.partition("=>")
            if not sep:
                raise UsageError(
                    f"--replace {item!r} is not `old=>new`",
                    hint="Example: --replace 'premium=>free'.",
                )
            pairs.append((old, new))

        _emit(
            engine.mock_rewrite(
                method,
                path,
                host=host,
                query=query,
                request_body=request_body,
                status=status,
                headers=headers or None,
                body=body,
                set_json=sets or None,
                delete_json=[d.strip() for d in delete_json] or None,
                replace=pairs or None,
                times=times,
            ),
            fmt,
        )

    _run(ctx, go)


@mock_app.command("record")
def mock_record_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="start|stop"),
    name: str | None = typer.Argument(None, help="Cassette name (required for start)."),
) -> None:
    """Record traffic into a named YAML cassette."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_record(action, name), fmt)

    _run(ctx, go)


@mock_app.command("replay")
def mock_replay_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Cassette name (or path to .yaml)."),
) -> None:
    """Load a cassette as live mock rules."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_replay(name), fmt)

    _run(ctx, go)


@mock_app.command("list")
def mock_list_cmd(ctx: typer.Context) -> None:
    """Show the live mock mode, owner, and every armed rule with a stable id.

    Mock rules are a host-side sidecar that outlives the command that armed it — a stale rule
    from an unrelated earlier session is otherwise invisible until it silently stubs a request
    it should not. Run this before `mock map`/`mock replay` if you did not just start clean.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_list(), fmt)

    _run(ctx, go)


@mock_app.command("clear")
def mock_clear_cmd(ctx: typer.Context) -> None:
    """Remove every mock rule and disarm mode (map/record) in one call."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_clear(), fmt)

    _run(ctx, go)


@mock_app.command("rm")
def mock_rm_cmd(
    ctx: typer.Context,
    rule_id: str = typer.Argument(..., help="Rule id from `aua mock list`."),
) -> None:
    """Remove one mock rule by id."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_rm(rule_id), fmt)

    _run(ctx, go)


# --------------------------------------------------------------------------- suite


suite_app = typer.Typer(
    help="Run a YAML acceptance-criteria checklist (has / expect / wait_for).",
    no_args_is_help=True,
)
app.add_typer(suite_app, name="suite")


@suite_app.command("run")
def suite_run_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Suite YAML path, or `-` to read stdin."),
    continue_on_fail: bool = typer.Option(
        False,
        "--continue",
        help="Keep going after a failed check (default: stop on first fail).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON summary."),
) -> None:
    """Run each check via has/expect/wait. Exit 0 if all pass, 8 if any fail."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        from . import suite as suite_mod

        text = sys.stdin.read() if path == "-" else None
        payload = _route(
            engine,
            "suite_run",
            path=path,
            continue_on_fail=continue_on_fail,
            text=text,
        )
        result = suite_mod.SuiteResult(
            ok=bool(payload["ok"]),
            name=str(payload["name"]),
            results=[suite_mod.CheckResult(**r) for r in payload.get("results") or []],
            passed=int(payload.get("passed") or 0),
            failed=int(payload.get("failed") or 0),
            stopped_early=bool(payload.get("stopped_early")),
        )

        if as_json:
            indent = 2 if fmt is OutputFormat.pretty else None
            sep = None if indent else (",", ":")
            typer.echo(json.dumps(payload, indent=indent, separators=sep, ensure_ascii=False))
        else:
            typer.echo(suite_mod.render_summary(result))
        if not result.ok:
            raise typer.Exit(int(ExitCode.ASSERTION))

    _run(ctx, go)


# --------------------------------------------------------------------------- guide


@app.command(cls=AnnotateCommand, name="guide")
def guide_cmd(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Emit the manual as structured JSON."),
    brief: bool = typer.Option(False, "--brief", help="Print the short session-protocol form."),
    emit_skill: str | None = typer.Option(
        None,
        "--emit-skill",
        metavar="[PATH]",
        help="Regenerate the Claude Code SKILL.md from this manual (default skill path).",
        show_default=False,
    ),
    emit_codex_metadata: str | None = typer.Option(
        None,
        "--emit-codex-metadata",
        metavar="[PATH]",
        help="Write deterministic Codex agents/openai.yaml metadata.",
        show_default=False,
    ),
) -> None:
    """Print the agent operating manual (the single source for the SKILL.md), §17b."""
    from . import guide as guide_mod

    opts = _opts(ctx)
    if emit_codex_metadata is not None:
        path = (
            Path("agents/openai.yaml")
            if emit_codex_metadata == ANNOTATE_DEFAULT
            else Path(emit_codex_metadata)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(guide_mod.render_codex_agent_metadata(), encoding="utf-8")
        typer.echo(str(path))
        return
    if emit_skill is not None:
        skill_path = None if emit_skill == ANNOTATE_DEFAULT else emit_skill
        target = guide_mod.emit_skill(skill_path)
        typer.echo(str(target))
        return
    if as_json:
        import json

        fmt = opts.fmt()
        indent = None if fmt is OutputFormat.compact else 2
        sep = (",", ":") if fmt is OutputFormat.compact else None
        typer.echo(
            json.dumps(guide_mod.render_json(), indent=indent, separators=sep, ensure_ascii=False)
        )
        return
    typer.echo(guide_mod.render_brief() if brief else guide_mod.render_markdown())


# Aliases for discoverability: `aua skill` / `aua agent` behave like `aua guide`.
app.command(cls=AnnotateCommand, name="skill", hidden=True)(guide_cmd)
app.command(cls=AnnotateCommand, name="agent", hidden=True)(guide_cmd)


# --------------------------------------------------------------------------- mcp


@app.command()
def mcp(ctx: typer.Context) -> None:
    """Run the MCP server over stdio (exposes the engine as MCP tools, §11)."""
    from . import mcp_server

    mcp_server.run_stdio(_opts(ctx).load())


# ------------------------------------------------------------------- entry point

# Options declared on the top-level callback rather than on a command. Only the ones Click
# cannot be asked about are written out; the rest are read off the callback itself, because a
# hand-kept copy drifts. It drifted: `--owner`, `--needs` and `--page` were all absent, and a
# missing value-taking global makes `_first_subcommand` land on the option's *value*, so
# `aua --owner X analyze --fields id,text` rewrote `--fields` for a command that has one.
_STATIC_GLOBAL_OPTS: dict[str, bool] = {  # name -> takes a value
    "--platform": True,
    "--serial": True,
    "--config": True,
    "--format": True,
    "--profile": True,
    "--timeout": True,
    "--log-level": True,
    "--no-cache": False,
    "--with-image": False,
    # These read as per-command options — `tap --rid x --until "text:Chats"` is how anyone
    # writes it, and an agent that tried exactly that got a parse error, went to `tap --help`,
    # then fell back to the `--no-observe` + `analyze` pair the flags exist to replace. A
    # global that only parses in the unnatural position is a global nobody uses.
    "--observe-fields": True,
    "--until": True,
    "--until-timeout": True,
    "--until-poll": True,
}


@lru_cache(maxsize=1)
def _global_opts() -> dict[str, bool]:
    try:
        import typer.main

        found = dict(_STATIC_GLOBAL_OPTS)
        for param in typer.main.get_command(app).params:
            takes_value = not getattr(param, "is_flag", False)
            for name in [*getattr(param, "opts", []), *getattr(param, "secondary_opts", [])]:
                if name.startswith("--"):
                    found[name] = takes_value
        return found
    except Exception:  # introspection must never break the CLI
        return dict(_STATIC_GLOBAL_OPTS)


def _defines_option(argv: list[str], name: str) -> bool:
    """Does the subcommand *argv* addresses declare *name* itself?

    Several commands legitimately reuse a global name for their own purpose - `emulator
    stop --serial` names the emulator to kill, and an export command's `--format` means
    gif|mp4. Those must keep winning, so the option is only moved when the target command
    has no opinion about it.
    """
    try:
        import click
        import typer.main

        cmd: Any = typer.main.get_command(app)
        for token in argv:
            if token.startswith("-"):
                break
            if not isinstance(cmd, click.Group):
                break
            nxt = cmd.get_command(click.Context(cmd), token)
            if nxt is None:
                break
            cmd = nxt
        params = getattr(cmd, "params", [])
        return any(name in getattr(p, "opts", []) for p in params)
    except Exception:  # introspection must never break the CLI
        return True


def _first_subcommand(head: list[str]) -> int | None:
    """Index of the subcommand token, skipping the *values* of value-taking globals.

    A bare "first token that does not start with `-`" scan lands on the value instead: in
    ``--serial emulator-5558 tap ...`` it picks ``emulator-5558`` at index 1, so every later
    option looks like it sits after the subcommand and hoisting quietly stops. That is the
    common shape — an agent passes ``--serial`` on every call — so `aua --serial X analyze
    --format json` never got the fix that `aua analyze --format json` did.

    Only globals are legal before the subcommand, so consuming their values here is safe.
    """
    i = 0
    while i < len(head):
        tok = head[i]
        if not tok.startswith("-"):
            return i
        base, eq, _ = tok.partition("=")
        # `--opt=value` carries its value inline; `--opt value` eats the next token.
        i += 1 if eq or not _global_opts().get(base, False) else 2
    return None


def alias_fields_on_actions(argv: list[str]) -> list[str]:
    """`--fields` on an action means the global `--observe-fields`; accept it as that.

    `analyze --fields id,text` is how every caller learns to project element columns, so it is
    what they then type on `tap-and-analyze`. That is not a different concept — both name the
    columns of the element rows being returned — so a second spelling is fragmentation, not
    precision, and Click answered it with "No such option '--fields'. (Did you mean one of:
    '--first', '--rid'?)". Measured 2026-08-10: an agent took that at face value and piped the
    whole JSON response through `jq` for the rest of the run.

    This is an alias, unlike the guessed command names, because the two spellings do the same
    thing to the same data. `--text` on `input` was refused precisely because it did not.

    Only rewritten when the target command has no `--fields` of its own, so `analyze` keeps hers.
    """
    if not any(a == "--fields" or a.startswith("--fields=") for a in argv):
        return argv
    start = _first_subcommand(argv)
    if start is None or _defines_option(argv[start:], "--fields"):
        return argv
    out: list[str] = []
    for arg in argv:
        if arg == "--fields":
            out.append("--observe-fields")
        elif arg.startswith("--fields="):
            out.append("--observe-fields=" + arg.split("=", 1)[1])
        else:
            out.append(arg)
    return out


def hoist_global_options(argv: list[str]) -> list[str]:
    """Move top-level options that were written after the subcommand to the front.

    `aua analyze --format json` is the single most repeated mistake in this project -
    across sessions, models and a documented warning. Click is right that `--format` binds
    to the group, but being right costs an agent a failed call, a wasted turn, and a detour
    into `--help`; the intent was never ambiguous. So accept both spellings.

    Only moves an option the target command does not define itself, and stops at `--` so a
    literal argument is never touched.
    """
    if not argv:
        return argv
    end = argv.index("--") if "--" in argv else len(argv)
    head, tail = argv[:end], argv[end:]

    # Everything before the first subcommand token is already global.
    first_cmd = _first_subcommand(head)
    if first_cmd is None:
        return argv

    hoisted: list[str] = []
    kept: list[str] = []
    i = 0
    while i < len(head):
        tok = head[i]
        base, eq, inline = tok.partition("=")
        globals_ = _global_opts()
        if i > first_cmd and base in globals_ and not _defines_option(head[first_cmd:], base):
            takes_value = globals_[base]
            if eq:
                hoisted.append(tok)
            elif takes_value and i + 1 < len(head) and not head[i + 1].startswith("-"):
                hoisted.extend([base, head[i + 1]])
                i += 1
            else:
                hoisted.append(base)
        else:
            kept.append(tok)
        i += 1
    return hoisted + kept + tail


# --------------------------------------------------------------- removed short aliases

#: Action verbs that used to also exist under their bare name. Each bare alias performed the
#: action but returned a *weaker* response than its ``-and-analyze`` twin, so a caller reaching
#: for the obvious short name silently got less, then spent a second round-trip on ``analyze``
#: to recover what the first call could have returned.
#:
#: Measured on one downstream suite (2026-08-08, 2322 invocations): ``tap-and-analyze`` was used
#: **zero** times while bare ``tap`` followed by a separate ``analyze`` happened 255 times - 36%
#: of every tap. Of 665 bare-``tap`` results, 509 carried no observation at all.
#:
#: They were hidden from ``--help`` in 100a392 and that did not work, because **a hidden alias
#: still answers**. Nothing ever corrects the caller, so the habit survives in agent memory, in
#: downstream docs, and in prompts. Removing them costs one failed call and fixes the caller for
#: the rest of the session - which is the whole point of failing loudly instead of quietly
#: returning less. The equivalent short names were already removed from the MCP surface.
_REMOVED_ACTION_ALIASES = (
    "tap",
    "click",
    "long-press",
    "double-tap",
    "input",
    "clear",
    "swipe",
    "scroll",
    "scroll-to",
    "expect",
    "key",
    "hide-keyboard",
    "open",
    "wait",
    "paste",
    "erase",
    "await",
)


class RemovedCommand(AuaError):
    """A short action alias that no longer exists, answered with its replacement."""

    exit_code = ExitCode.USAGE
    code = "removed_command"


def _register_removed_alias(
    old: str,
    *,
    target: typer.Typer | None = None,
    replacement: str | None = None,
    prefix: str = "",
) -> None:
    """Register ``old`` as a command that fails, naming the replacement."""
    target = app if target is None else target
    replacement = replacement or f"{old}-and-analyze"
    spoken = f"{prefix}{old}"

    def _removed(ctx: typer.Context) -> None:
        err = RemovedCommand(
            f"`aua {spoken}` was removed. Use `aua {replacement}` instead.",
            hint=(
                f"`{replacement}` performs the same action and returns the resulting screen "
                f"in the same response, so a follow-up `analyze` is not needed. Every option "
                f"you passed to `{spoken}` is accepted unchanged."
            ),
        )
        emit_error(err)
        raise typer.Exit(int(err.exit_code))

    _removed.__name__ = "removed_" + old.replace("-", "_")
    # `add_help_option=False` is the point of this call, not a detail. With Click's built-in
    # `--help`, a removed command renders a plausible, empty options page and exits 0 — so the
    # careful caller who checks help BEFORE guessing is told the command exists and takes nothing,
    # while the caller who just guesses gets the correct error. Three of four lanes in the
    # 2026-08-08 probe hit exactly that. Turning help off routes `--help` into the callback below,
    # which names the replacement and exits 2 like every other invocation.
    target.command(
        name=old,
        hidden=True,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )(_removed)


for _removed_alias in _REMOVED_ACTION_ALIASES:
    _register_removed_alias(_removed_alias)


def run() -> None:
    """Console-script entry point: tolerate misplaced global options, then dispatch."""
    sys.argv[1:] = hoist_global_options(alias_fields_on_actions(sys.argv[1:]))
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
