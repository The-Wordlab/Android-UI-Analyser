"""Typer CLI — a thin adapter over :class:`~android_ui_analyser.engine.Engine` (PRD §5).

Every command builds a fresh :class:`Config` via :func:`load_config` (honouring the
global options stashed on the Typer context), constructs an :class:`Engine` (the device
connects lazily), invokes the matching engine method, and prints ``result.render(fmt)``
to **stdout**. Logs go to **stderr**; any :class:`AuaError` is emitted as a structured
object to stderr with the mapped exit code. No perception logic lives here.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import click
import typer
from typer.core import TyperCommand

from . import __version__
from .config import (
    Config,
    default_config_yaml,
    find_project_config,
    load_config,
    user_config_path,
)
from .engine import Engine
from .errors import (
    AuaError,
    ConfigError,
    DeviceError,
    ExitCode,
    ExpectationFailed,
    SelectorAmbiguousError,
    SelectorNotFoundError,
    UsageError,
    emit_error,
)
from .memory import AppMap, AppMemoryStore, find_result, render_map
from .projection import Projection
from .schema import ActionResult, OutputFormat

logger = logging.getLogger("android_ui_analyser")

T = TypeVar("T")

# Sentinel produced by an optional-value flag (``--annotate``/``--emit-skill``) given bare.
ANNOTATE_DEFAULT = "\x00aua_annotate_default"
_OPTIONAL_VALUE_OPTS = {"--annotate", "--emit-skill", "--with-image"}

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

    serial: str | None = None
    config: str | None = None
    format: str | None = None
    profile: str | None = None
    timeout: int | None = None
    log_level: str = "warn"
    no_cache: bool = False
    with_image: bool = False
    _cfg: Config | None = field(default=None, repr=False)

    def cli_overrides(self) -> dict[str, Any]:
        """Translate the global flags into a config-override tree (None = unset)."""
        overrides: dict[str, Any] = {}
        if self.serial is not None:
            overrides["device"] = {"serial": self.serial}
        if self.format is not None or self.with_image:
            out: dict[str, Any] = {}
            if self.format is not None:
                out["format"] = self.format
            if self.with_image:
                out["with_image"] = True
            overrides["output"] = out
        if self.log_level is not None:
            overrides["log_level"] = self.log_level
        if self.timeout is not None:
            overrides["timeouts"] = {"action_ms": self.timeout}
        if self.no_cache:
            overrides["cache"] = {"enabled": False}
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
        return Engine(self.load())


def _opts(ctx: typer.Context) -> GlobalOpts:
    if not isinstance(ctx.obj, GlobalOpts):  # pragma: no cover - defensive
        ctx.obj = GlobalOpts()
    return ctx.obj


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(ExitCode.OK)


# --------------------------------------------------------------------------- error wrap


def _run(ctx: typer.Context, fn: Callable[[Engine, OutputFormat], T]) -> T:
    """Execute ``fn`` with a built engine+format, mapping AuaError → structured exit.

    Unknown exceptions become a generic structured error on stderr with exit 1.
    """
    opts = _opts(ctx)
    try:
        cfg_fmt = opts.fmt()
        engine = opts.engine()
        return fn(engine, cfg_fmt)
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


# --------------------------------------------------------------------------- selectors

_BY_KINDS = {"id": "rid", "text": "text", "desc": "desc"}

# Shared selector options — the same six flags on every action, so `--rid` means one thing
# everywhere. Typer copies an OptionInfo per command, so one instance is safe to reuse.
_SEL_BY = typer.Option(
    None, "--by", help="Read the positional as: id (resource-id) | text | desc."
)
_SEL_RID = typer.Option(None, "--rid", help="Target this resource-id (bare tail accepted).")
_SEL_TEXT = typer.Option(None, "--text", help="Target this label (exact first, then substring).")
_SEL_DESC = typer.Option(None, "--desc", help="Target this content-desc.")
_SEL_INDEX = typer.Option(None, "--index", help="Take the nth (0-based) of several matches.")
_SEL_FIRST = typer.Option(
    False, "--first", help="Take the first of several matches instead of erroring."
)


def _selector(
    *,
    ident: str | None = None,
    by: str | None = None,
    rid: str | None = None,
    text: str | None = None,
    desc: str | None = None,
    index: int | None = None,
    first: bool = False,
) -> dict[str, Any] | None:
    """Build the engine selector, or ``None`` when the caller passed a plain element id.

    Two spellings resolve to the same thing: ``--by id <positional>`` (reads like the
    existing ``has``/``wait`` flag) and the one-shot ``--rid/--text/--desc <value>``.
    """
    if by is not None:
        kind = _BY_KINDS.get(by.lower())
        if kind is None:
            raise UsageError(f"unknown --by '{by}'", hint="Choose one of: id, text, desc.")
        if not ident:
            raise UsageError(
                f"--by {by} needs the value as the positional argument",
                hint="e.g. `aua tap --by id homeTabBROWSE`",
            )
        return {kind: ident, "index": index, "first": first}
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
    """The positional as an element id — only meaningful when no selector is in play."""
    if selector is not None or ident is None:
        return None
    try:
        return int(ident)
    except ValueError as exc:
        raise UsageError(
            f"'{ident}' is not an element id",
            hint="Ids are integers from the last analyze. To address by name use "
            "`--rid <resource-id>`, `--text <label>`, or `--by id <resource-id>`.",
        ) from exc


def _require_target(
    verb: str, ident: str | None, selector: dict[str, Any] | None
) -> int | None:
    """Element id or selector — raise usage (exit 2) before any device connect."""
    element_id = _element_id(ident, selector)
    if element_id is None and selector is None:
        raise UsageError(
            f"{verb} needs an element id or a selector",
            hint=f"e.g. `aua {verb} 4` or `aua {verb} --rid continue_btn` "
            f"or `aua {verb} --text Continue`.",
        )
    return element_id


def _emit(result: Any, fmt: OutputFormat) -> None:
    """Render a pydantic result (``.render``) or a plain dict (daemon path) to stdout."""
    if hasattr(result, "render"):
        typer.echo(result.render(fmt))
        return
    _echo_json(result, fmt)


def _echo_json(data: Any, fmt: OutputFormat) -> None:
    import json

    indent = 2 if fmt is OutputFormat.pretty else None
    sep = None if indent else (",", ":")
    typer.echo(json.dumps(data, indent=indent, separators=sep, ensure_ascii=False))


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
    return data if isinstance(data.get("elements"), list) else None


def _emit_analyze(result: Any, fmt: OutputFormat, view: Projection) -> None:
    """Emit an analyze result, through *view* when it asked for anything."""
    payload = _analyze_payload(result) if view.active else None
    if payload is None:
        _emit(result, fmt)
        return
    if view.tsv:
        typer.echo(view.render_tsv(payload))
        return
    _echo_json(view.apply(payload, fmt=fmt), fmt)


# --------------------------------------------------------------------------- daemon route


def _warm(engine: Engine) -> None:
    """Force the lazy device connection so the engine's analyze-cache key (derived from
    the connected serial) matches what a prior ``analyze`` wrote. Action/inspect
    commands resolve cached element ids and need a device anyway, so this is free.
    """
    _ = engine.device


# Engine method name → daemon command name (they differ only for ``input``).
_DAEMON_CMD = {"input_text": "input"}

# Methods whose STATE lives only in the daemon process. For these, an in-process fallback
# cannot produce a correct answer — a process with no capture buffer reports "not running"
# while the daemon is happily writing frames — so a stale daemon must be an error, not a
# silent downgrade.
_DAEMON_ONLY_METHODS = frozenset(
    {"capture_status", "capture_last", "capture_on", "capture_off", "capture_prune"}
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


def _route(engine: Engine, method: str, **kwargs: Any) -> Any:
    """Run an engine call through the daemon when one is live, else in-process.

    Best-effort: any failure connecting to / importing the daemon falls back to the
    in-process engine. A structured error returned by the daemon is raised as the
    matching :class:`AuaError` (it is the answer, so it must not be swallowed).
    """
    cfg = engine.config
    if getattr(cfg.daemon, "enabled", False):
        try:
            from . import daemon as daemon_mod

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
            if skew and method in _DAEMON_ONLY_METHODS:
                # An in-process answer here is not a slower answer, it is a WRONG one: the
                # buffer lives in the daemon, so this process would report "not running"
                # while frames are being written. Say so instead of guessing.
                raise UsageError(
                    f"the running daemon has older code than this CLI, and `{method}` can "
                    "only be answered by the daemon that holds the buffer",
                    hint="Restart it: `aua daemon stop && aua daemon start`.",
                )
            if skew:
                logger.warning(
                    "daemon runs aua %s but this CLI is %s; using in-process. "
                    "Restart it: `aua daemon stop && aua daemon start`.",
                    ver,
                    daemon_mod._aua_version(),
                )
            if ver is not False and not skew:
                client = daemon_mod.DaemonClient(daemon_mod.socket_path(cfg))
                cmd = _DAEMON_CMD.get(method, method)
                resp = client.call(cmd, **kwargs)
                if resp.get("ok"):
                    return resp.get("result")
                raise _daemon_error(resp.get("error", {}))
        except AuaError:
            raise
        except Exception as exc:  # pragma: no cover - daemon optional / unreachable
            logger.debug("daemon route unavailable, running in-process: %s", exc)
    _warm(engine)
    return getattr(engine, method)(**kwargs)

# --------------------------------------------------------------------------- app


_GUIDE_POINTER = (
    "Run `aua guide` for the full agent operating manual (session protocol, escalation "
    "ladder, memory, schema, exit codes); `aua guide --emit-skill` regenerates the Claude "
    "Code skill from the same source."
)

app = typer.Typer(
    name="aua",
    help=(
        "android-ui-analyser — structured Android UI perception + action for agents.\n\n"
        + _GUIDE_POINTER
    ),
    epilog=_GUIDE_POINTER,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    serial: str | None = typer.Option(
        None, "--serial", help="Target device serial (default: only/first)."
    ),
    config: str | None = typer.Option(None, "--config", help="Explicit config file path."),
    format: str | None = typer.Option(
        None, "--format", help="Output format: json|pretty|compact|tsv (tsv: analyze only)."
    ),
    profile: str | None = typer.Option(None, "--profile", help="Named config profile to overlay."),
    timeout: int | None = typer.Option(None, "--timeout", help="Per-operation timeout in ms."),
    log_level: str = typer.Option(
        "warn", "--log-level", help="error|warn|info|debug (logs → stderr)."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the cached analyze result."),
    with_image: bool = typer.Option(
        False,
        "--with-image",
        help="Session default: save raw screenshots on analyze/actions "
        "(override per-command with --with-image PATH or omit).",
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
            f"invalid --format '{format}'", hint="Choose one of: json, pretty, compact, tsv."
        )
        emit_error(err)
        raise typer.Exit(int(err.exit_code))
    ctx.obj = GlobalOpts(
        serial=serial,
        config=config,
        format=format,
        profile=profile,
        timeout=timeout,
        log_level=log_level,
        no_cache=no_cache,
        with_image=with_image,
    )


# --------------------------------------------------------------------------- perception


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
        target = out or path
        narrowed = region is not None or scale is not None or max_width is not None
        if not narrowed:
            _emit(engine.screenshot(target, annotate=annotate), fmt)
            return
        if annotate:
            raise UsageError(
                "--annotate cannot be combined with --region/--scale/--max-width",
                hint="Marks are placed in full-screen coordinates; crop a plain screenshot.",
            )
        from . import imaging

        box = imaging.parse_region(region) if region else None
        view = imaging.crop_and_scale(
            engine.device.screenshot(), region=box, scale=scale, max_width=max_width
        )
        saved = view.save(
            target or imaging.capture_path(engine.config.cache.dir, engine.device.serial)
        )
        _emit(ActionResult(ok=True, action="screenshot", detail=saved), fmt)

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
    text: str = typer.Argument(..., help="Text to look for on screen."),
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
) -> None:
    """Is this on screen right now? Exit 0 if present, 1 if not.

    ``--by id`` checks a resource-id (a bare tail works) — verifies containers the element
    list prunes, i.e. Maestro-style ``assertVisible: id:``.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.has(
            text,
            match=match,
            ignore_case=ignore_case,
            ocr_fallback=ocr_fallback,
            source=source,
            timeout_ms=timeout,
            by=by,
        )
        _emit(result, fmt)
        if not result.found:
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- actions


@app.command(cls=AnnotateCommand)
def tap(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id from the last analyze (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
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

    `aua tap 9` · `aua tap --rid notificationsButton` · `aua tap --text "Create an app"` ·
    `aua tap --by id homeTabBROWSE`. A selector resolves on the live screen in this one
    call; matching nothing exits 6 and matching several exits 7 with the candidates — it
    never silently taps nothing.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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


@app.command(name="click", cls=AnnotateCommand)
def click_cmd(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to tap (alias of tap)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
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
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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


@app.command(name="long-press", cls=AnnotateCommand)
def long_press(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to long-press (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
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
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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


@app.command(name="double-tap", cls=AnnotateCommand)
def double_tap(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to double-tap (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
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
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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


@app.command(name="input", cls=AnnotateCommand)
def input_cmd(
    ctx: typer.Context,
    first_arg: str | None = typer.Argument(
        None, metavar="[ID] TEXT", help="Element id then text — or just the text with --rid/--by."
    ),
    second_arg: str | None = typer.Argument(None, metavar="", help="", show_default=False),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    desc: str | None = _SEL_DESC,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
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
    """Focus an element and type text; ``--submit`` sends the IME action.

    `aua input 9 "hello"` · `aua input --rid promptField "hello" --submit` ·
    `aua input --by id promptField "hello"`. With ``--rid``/``--desc`` the single positional
    is the text; with ``--by`` (or a plain id) the first positional addresses the field and
    the second is the text. ``--text`` is not a selector here — it would read as the value.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=first_arg, by=by, rid=rid, desc=desc, index=index, first=first
        )
        # --rid/--desc address the field, so the lone positional is the text to type;
        # --by consumes the first positional as the selector value.
        typed = first_arg if (selector is not None and by is None) else second_arg
        if selector is not None and by is None and second_arg is not None:
            raise UsageError(
                "with --rid/--desc, pass only the text to type",
                hint='e.g. `aua input --rid promptField "hello"`',
            )
        if typed is None:
            raise UsageError(
                "input needs the text to type",
                hint='e.g. `aua input 9 "hello"` or `aua input --rid promptField "hello"`',
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


@app.command(cls=AnnotateCommand)
def clear(
    ctx: typer.Context,
    ident: str | None = typer.Argument(
        None, metavar="[ID]", help="Element id to clear (or the selector value)."
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
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
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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


@app.command(cls=AnnotateCommand)
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
        True, "--verify/--no-verify", help="Report whether the screen actually moved."
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

    `aua swipe up` · `aua swipe --direction up` · `aua swipe --from-rid notificationList up`.
    No anchor needed: the gesture is aimed at the scrollable container on screen rather than
    the middle of the display, and ``detail`` reports ``moved``/``no-change`` so a swipe that
    did nothing cannot look like a swipe that worked. For list scrolling prefer
    ``aua scroll``, which turns that verdict into an exit code.
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


@app.command(cls=AnnotateCommand)
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
    from_id: int | None = typer.Option(None, "--from", help="Scroll the container at this element."),
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

    `aua scroll up` · `aua scroll --pages 3` · `aua scroll --to-end` · `aua scroll --to-start`.
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


@app.command(name="scroll-to", cls=AnnotateCommand)
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

    `aua scroll-to "Red Square Tap"` · `aua scroll-to --rid listRow_7`.
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
                hint='e.g. `aua scroll-to "Red Square Tap"`',
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


@app.command()
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
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    timeout: int = typer.Option(
        0, "--timeout", "--timeout-ms", help="Poll until it holds, up to this many ms."
    ),
) -> None:
    """Assert one thing about the screen. Exit 0 = pass, 8 = the assertion failed.

    `aua expect --rid notificationsButton --exists` ·
    `aua expect --text "Loading" --absent --timeout 5000` ·
    `aua expect --rid itemDetailLikeCount --text-is "7"` ·
    `aua expect --rid settingsPushToggleSwitch --checked`

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
            index=index,
            first=first,
            timeout_ms=timeout,
        )
        _emit(result, fmt)
        _exit_unless_ok(result, ExitCode.ASSERTION, code="expectation_failed")

    _run(ctx, go)


@app.command(cls=AnnotateCommand)
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


@app.command(name="hide-keyboard", cls=AnnotateCommand)
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


@app.command(cls=AnnotateCommand)
def open(  # noqa: A001 - matches the user-facing verb `aua open`
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


@app.command()
def wait(
    ctx: typer.Context,
    for_: str | None = typer.Option(None, "--for", help="Text/resource-id to wait for."),
    idle: bool = typer.Option(False, "--idle", help="Wait for the UI to go idle."),
    for_stable: bool = typer.Option(
        False, "--for-stable", help="Wait until the screen stops visually changing."
    ),
    interval: int = typer.Option(200, "--interval", help="--for-stable: ms between screenshots."),
    settle: int = typer.Option(600, "--settle", help="--for-stable: ms of no change to settle."),
    timeout: int | None = typer.Option(
        None, "--timeout", "--timeout-ms", help="Timeout in ms (default 5000; 30000 for --for-stable)."
    ),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
    observe: bool = typer.Option(
        False,
        "--observe",
        help="Also return the (settled) screen with fresh ids — act on it without a re-analyze.",
    ),
    by: str = typer.Option("text", "--by", help="--for match by: text (default) | id | desc."),
    absent: bool = typer.Option(
        False, "--absent", help="With --for: wait until it DISAPPEARS (loading spinners, dialogs)."
    ),
) -> None:
    """Wait for text to appear (or with ``--absent`` disappear), for idle, or for settle.

    ``--for-stable`` polls cheap screenshots (a perceptual-hash "settled" check — no OCR,
    no hierarchy parse; works on opaque screens) and returns once the screen stops changing
    for ``--settle`` ms. Ideal for waiting on image generation / loading. ``--observe``
    folds in the post-wait screen so you can act on what you waited for in one fewer call.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
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
    assist: bool = typer.Option(
        False,
        "--assist",
        help="On divergence, let the opt-in planner LLM try to recover (needs planner.enabled).",
    ),
) -> None:
    """Navigate to a known screen using app memory — drives and verifies each hop (§6b).

    Resolves the goal against the learned map, then replays each edge's recorded steps
    along the shortest route from the current screen, confirming ``known_screen`` after
    every hop. Stops and returns the remaining route/steps + current screen if anything
    diverges. ``--plan`` prints the annotated route only; destructive steps are refused
    without ``--allow-destructive``. ``--assist`` lets a fast model recover a divergence.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "goto",
            goal=goal,
            plan=plan,
            max_steps=max_steps,
            allow_destructive=allow_destructive,
            assist=assist,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


@app.command()
def navigate(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Natural-language destination, e.g. 'open the image generator'."),
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


# --------------------------------------------------------------------------- device/session


@app.command()
def devices(ctx: typer.Context) -> None:
    """List attached devices (serial, model, android version, state)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        infos = engine.list_devices()
        payload = [d.model_dump(mode="json") for d in infos]
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        typer.echo(json.dumps(payload, indent=indent, separators=sep, ensure_ascii=False))

    _run(ctx, go)


@app.command(name="app")
def app_cmd(
    ctx: typer.Context,
    action: str = typer.Argument(
        ...,
        metavar="ACTION",
        help="foreground|launch|stop|kill|clear|grant|current.",
    ),
    package: str | None = typer.Argument(
        None, metavar="[PKG]", help="Package for launch/stop/kill/clear/grant."
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
        a = action.lower()
        wiping = a in ("clear", "clear-state", "clear_state") or (
            a == "launch" and clear_state
        )
        if wiping and not yes:
            raise UsageError(
                f"app {action}{' --clear' if a == 'launch' else ''} wipes ALL app data "
                f"(feature flags, login session, local config) — pass --yes to confirm",
                hint="Example: `aua app clear com.example.app --yes`. "
                "Then re-apply flag overrides / re-login before asserting experiment UI.",
            )
        _emit(
            engine.app(
                action,
                package=package,
                activity=activity,
                clear_state=clear_state,
                confirmed=yes,
            ),
            fmt,
        )

    _run(ctx, go)


clipboard_app = typer.Typer(help="Clipboard — Maestro setClipboard / copyTextFrom / pasteText.")
app.add_typer(clipboard_app, name="clipboard")


@clipboard_app.command("set")
def clipboard_set(ctx: typer.Context, text: str = typer.Argument(..., help="Text to copy.")) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.clipboard_set(text), fmt)

    _run(ctx, go)


@clipboard_app.command("get")
def clipboard_get(ctx: typer.Context) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.clipboard_get(), fmt)

    _run(ctx, go)


@app.command(cls=AnnotateCommand)
def paste(
    ctx: typer.Context,
    observe: bool = typer.Option(True, "--observe/--no-observe"),
    with_image: str | None = typer.Option(None, "--with-image", metavar="[PATH]", show_default=False),
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
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
) -> None:
    """Copy an element's text/content-desc to the clipboard (Maestro copyTextFrom)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
        )
        _emit(
            engine.copy_text(element_id=_element_id(ident, selector), selector=selector),
            fmt,
        )

    _run(ctx, go)


@app.command(cls=AnnotateCommand)
def erase(
    ctx: typer.Context,
    ident: str | None = typer.Argument(None, metavar="[ID]"),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    chars: int | None = typer.Option(
        None, "--chars", help="Delete this many characters; omit to clear the whole field."
    ),
    observe: bool = typer.Option(True, "--observe/--no-observe"),
    with_image: str | None = typer.Option(None, "--with-image", metavar="[PATH]", show_default=False),
) -> None:
    """Erase text in a field (Maestro eraseText)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        selector = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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
            raise UsageError(
                "location needs LAT,LON", hint="e.g. `aua location set 37.42,-122.08`"
            )
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
        _emit(engine.record_start(remote), fmt)

    _run(ctx, go)


@record_app.command("stop")
def record_stop_cmd(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Local path to save the MP4."),
) -> None:
    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.record_stop(path), fmt)

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


@app.command()
def daemon(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="start|stop|status."),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="On start, skip the app orientation blob (get it later with `aua orient`).",
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
            daemon_mod.start(cfg)
            out: dict[str, Any] = {
                "ok": True,
                "action": "daemon-start",
                "detail": daemon_mod.status(cfg),
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
            daemon_mod.stop(cfg)
            out = {"ok": True, "action": "daemon-stop"}
        elif a == "status":
            out = {
                "ok": True,
                "action": "daemon-status",
                "running": daemon_mod.is_running(cfg),
                "detail": daemon_mod.status(cfg),
            }
        else:
            raise UsageError(f"unknown daemon action '{action}'", hint="start|stop|status")
        indent = 2 if fmt is OutputFormat.pretty else None
        sep = None if indent else (",", ":")
        typer.echo(json.dumps(out, indent=indent, separators=sep, ensure_ascii=False, default=str))

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


# --------------------------------------------------------------------------- doctor


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

    adb = shutil.which("adb")
    checks["adb"] = {"ok": adb is not None, "detail": adb or "adb not found on PATH"}

    try:
        import importlib.util

        spec = importlib.util.find_spec("uiautomator2")
        checks["uiautomator2"] = {
            "ok": spec is not None,
            "detail": "importable" if spec is not None else "not installed",
        }
    except Exception as exc:  # pragma: no cover - defensive
        checks["uiautomator2"] = {"ok": False, "detail": f"error: {exc}"}

    try:
        infos = engine.list_devices()
        checks["devices"] = {
            "ok": len(infos) > 0,
            "count": len(infos),
            "detail": [d.model_dump(mode="json") for d in infos] if infos else "no devices",
        }
    except AuaError as exc:
        checks["devices"] = {"ok": False, "detail": exc.message}
    except Exception as exc:  # pragma: no cover - defensive
        checks["devices"] = {"ok": False, "detail": str(exc)}

    try:
        providers = engine.provider_status()
    except Exception as exc:  # pragma: no cover - defensive
        providers = {}
        checks["providers_error"] = str(exc)

    return {"checks": checks, "providers": providers}


def _render_doctor_pretty(report: dict[str, Any]) -> str:
    def mark(ok: bool) -> str:
        return "OK  " if ok else "FAIL"

    lines: list[str] = ["aua doctor", "=========="]
    checks = report.get("checks", {})

    adb = checks.get("adb", {})
    lines.append(f"[{mark(adb.get('ok', False))}] adb           {adb.get('detail', '')}")
    u2 = checks.get("uiautomator2", {})
    lines.append(f"[{mark(u2.get('ok', False))}] uiautomator2  {u2.get('detail', '')}")
    dev = checks.get("devices", {})
    dev_detail = dev.get("detail", "")
    if isinstance(dev_detail, list):
        dev_detail = ", ".join(d.get("serial", "?") for d in dev_detail) or "(none)"
    lines.append(f"[{mark(dev.get('ok', False))}] devices       {dev_detail}")

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


def _resolve_package(opts: GlobalOpts, app_pkg: str | None) -> str:
    """Use ``--app`` if given, else detect the foreground package (needs a device)."""
    if app_pkg:
        return app_pkg
    pkg = opts.engine().current_package()
    if not pkg:
        raise UsageError(
            "could not determine the foreground app",
            hint="Pass --app <package>, or attach a device so the current app can be detected.",
        )
    return pkg


@app.command(name="map")
def map_cmd(
    ctx: typer.Context,
    app_pkg: str | None = typer.Option(None, "--app", help="Package to map (default: current)."),
    brief: bool = typer.Option(False, "--brief", help="Skeleton only (screens + routes)."),
    screen: str | None = typer.Option(None, "--screen", help="Drill into one screen."),
    depth: int | None = typer.Option(None, "--depth", help="Limit the route-tree depth."),
    find: str | None = typer.Option(None, "--find", help="Just the route to a target goal."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of the text tree."),
) -> None:
    """Print the app's known layout from memory (screens, key elements, routes)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(opts, app_pkg)
        app_map = store.load(pkg) or AppMap(package=pkg)
        compact = fmt is OutputFormat.compact
        if as_json or compact:
            if find:
                payload: Any = find_result(app_map, find)
            elif screen:
                rec = app_map.screens.get(screen)
                payload = rec.model_dump(mode="json") if rec else {}
            else:
                payload = app_map.model_dump(mode="json")
            sep = (",", ":") if compact else None
            indent = None if compact else 2
            typer.echo(json.dumps(payload, indent=indent, separators=sep, ensure_ascii=False))
            return
        detail = "brief" if brief else "default"
        typer.echo(render_map(app_map, detail=detail, find=find, screen=screen, depth=depth))

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
    deeplink: str | None = typer.Option(None, "--deeplink", help="A useful deeplink URI (needs/uses --note)."),
    app_pkg: str | None = typer.Option(None, "--app", help="Package (default: current)."),
) -> None:
    """Teach the app playbook: a description, a quirk note, a login/etc. recipe, or a deeplink.

    The agent should record what it learns so the NEXT run starts informed, e.g.
    `aua remember --recipe login_full --note "tap 'Login with test user'"` or
    `aua remember --deeplink "myapp://set-flags?flag=value" --note "set flags, then restart"`.
    """

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        opts = _opts(ctx)
        store = AppMemoryStore(opts.load().memory)
        pkg = _resolve_package(opts, app_pkg)
        did: list[str] = []
        if about:
            store.set_description(pkg, about)
            did.append("description")
        if recipe:
            if not note:
                raise UsageError("--recipe needs --note", hint='e.g. --recipe login_full --note "tap X"')
            store.remember_recipe(pkg, recipe, note)
            did.append(f"recipe:{recipe}")
        if deeplink:
            store.remember_deeplink(pkg, deeplink, note=note)
            did.append("deeplink")
        if note and not recipe and not deeplink:
            store.remember_note(pkg, note)
            did.append("note")
        if not did:
            raise UsageError(
                "remember needs something to store",
                hint="pass --about / --note / --recipe NAME --note / --deeplink URI",
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
        pkg = _resolve_package(opts, app_pkg)
        app_map = store.load(pkg)
        if app_map is None:
            typer.echo(f"nothing recorded for {pkg} yet")
            return
        if fmt in (OutputFormat.json, OutputFormat.compact):
            play = {
                "package": pkg,
                "description": app_map.description,
                "recipes": {r.name: r.note for r in app_map.recipes},
                "deeplinks": [{"uri": d.uri, "note": d.note} for d in app_map.deeplinks],
                "notes": list(app_map.notes),
            }
            typer.echo(json.dumps(play, indent=None if fmt is OutputFormat.compact else 2))
        else:
            from .memory import _playbook_lines

            lines = _playbook_lines(app_map)
            typer.echo("\n".join(lines) if lines else f"no playbook for {pkg} yet")

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
        pkg = _resolve_package(opts, app_pkg)
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
        pkg = _resolve_package(opts, app_pkg)
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

    Deeplinks let you jump straight to a screen — `aua open "myapp://tools/summarize"`
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
    """Get a prioritized exploration worklist for THIS app (probe deeplinks, expand screens).

    Run the tasks with normal `aua` commands — results auto-record into the map/playbook,
    so re-running the plan shows what's left. The way to have an agent index an app.
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
    name: str | None = typer.Argument(None, help="Saved flow name (see `aua flow list`)."),
    param: list[str] = typer.Option(  # noqa: B008 - typer option factory
        [], "--param", "-p", help="Substitute ${NAME} placeholders: --param NAME=value."
    ),
    file: str | None = typer.Option(None, "--file", help="Run a flow YAML file directly."),
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
) -> None:
    """Replay a whole journey in one call; on divergence returns a resumable step index."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = _route(
            engine,
            "flow_run",
            name=name,
            file=file,
            params=_parse_params(param),
            dry_run=dry_run,
            from_step=from_step,
            allow_destructive=allow_destructive,
            assist=assist,
        )
        _emit(result, fmt)
        if isinstance(result, dict) and result.get("ok") is False:
            raise typer.Exit(1)

    _run(ctx, go)


@flow_app.command("save")
def flow_save_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name for the saved flow."),
    last: int = typer.Option(12, "--last", help="How many recent actions to materialize."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing flow."),
) -> None:
    """Materialize the session's recent actions into an editable flow YAML."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "flow_save", name=name, last=last, force=force), fmt)

    _run(ctx, go)


@flow_app.command("list")
def flow_list_cmd(ctx: typer.Context) -> None:
    """List saved flows (name, app, steps, params)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        from .flows import FlowStore

        flows = FlowStore(_opts(ctx).load().memory).list()
        typer.echo(json.dumps({"flows": flows}, indent=2, ensure_ascii=False))

    _run(ctx, go)


@flow_app.command("show")
def flow_show_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Flow name."),
) -> None:
    """Print a saved flow's YAML (edit it in place under memory.dir/flows/)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        from .flows import FlowStore

        store = FlowStore(_opts(ctx).load().memory)
        typer.echo(store.path(name).read_text(encoding="utf-8") if store.path(name).is_file() else "")
        if not store.path(name).is_file():
            raise UsageError(f"no flow named '{name}'", hint="see `aua flow list`")

    _run(ctx, go)


@flow_app.command("delete")
def flow_delete_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Flow name to delete."),
) -> None:
    """Delete a saved flow."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        import json

        from .flows import FlowStore

        deleted = FlowStore(_opts(ctx).load().memory).delete(name)
        typer.echo(json.dumps({"ok": deleted, "action": "flow-delete", "flow": name}))
        if not deleted:
            raise typer.Exit(1)

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
    lines: int | None = typer.Option(None, "--lines", "-n", help="Keep only the last N matching lines."),
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
) -> None:
    """Emit timeline JSON + frame paths + cheap local diff summary."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "capture_last", seconds=seconds, since=since), fmt)

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


@a11y_app.command("scroll")
def a11y_scroll_cmd(
    ctx: typer.Context,
    ident: str | None = typer.Argument(None, help="Element id from the last analyze."),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
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
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
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


@a11y_app.command("action")
def a11y_action_cmd(
    ctx: typer.Context,
    ident: str | None = typer.Argument(None, help="Element id from the last analyze."),
    action: str = typer.Argument(
        ...,
        help="CLICK|LONG_CLICK|SCROLL_FORWARD|SCROLL_BACKWARD|EXPAND|COLLAPSE|DISMISS",
    ),
    by: str | None = _SEL_BY,
    rid: str | None = _SEL_RID,
    text: str | None = _SEL_TEXT,
    desc: str | None = _SEL_DESC,
    index: int | None = _SEL_INDEX,
    first: bool = _SEL_FIRST,
    no_observe: bool = typer.Option(False, "--no-observe", help="Skip post-action analyze."),
) -> None:
    """Perform a named accessibility action on an element."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        sel = _selector(
            ident=ident, by=by, rid=rid, text=text, desc=desc, index=index, first=first
        )
        _emit(
            engine.a11y_action(
                _element_id(ident, sel),
                selector=sel,
                action=action,
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


@flags_app.command("set")
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


@flags_app.command("apply")
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
    port: int = typer.Option(8080, "--port", help="mitmdump listen port."),
) -> None:
    """Start mitmdump, adb-reverse, and set the device HTTP proxy."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.proxy_start(port=port), fmt)

    _run(ctx, go)


@proxy_app.command("stop")
def proxy_stop_cmd(ctx: typer.Context) -> None:
    """Clear the device proxy and stop mitmdump."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.proxy_stop(), fmt)

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
) -> None:
    """Add a live mock rule (reloaded by the mitmproxy addon)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.mock_map(method, path, status=status, body=body), fmt)

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
) -> None:
    """Print the agent operating manual (the single source for the SKILL.md), §17b."""
    from . import guide as guide_mod

    opts = _opts(ctx)
    if emit_skill is not None:
        path = None if emit_skill == ANNOTATE_DEFAULT else emit_skill
        target = guide_mod.emit_skill(path)
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

    mcp_server.run_stdio()


if __name__ == "__main__":  # pragma: no cover
    app()
