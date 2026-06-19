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
from .errors import AuaError, ConfigError, DeviceError, ExitCode, UsageError, emit_error
from .schema import OutputFormat

logger = logging.getLogger("android_ui_analyser")

T = TypeVar("T")

# Sentinel produced by the optional-value ``--annotate`` flag when given with no value.
ANNOTATE_DEFAULT = "\x00aua_annotate_default"

_LOG_LEVELS = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class AnnotateCommand(TyperCommand):
    """A Typer command whose ``--annotate`` option takes an *optional* value.

    Typer (0.26) drops Click's ``flag_value``, so a bare ``--annotate`` would error
    asking for a value. We rebuild the Click option's optional-value state after Typer
    constructs it: ``--annotate`` → :data:`ANNOTATE_DEFAULT`; ``--annotate PATH`` → PATH.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, click.Option) and "--annotate" in param.opts:
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
    _cfg: Config | None = field(default=None, repr=False)

    def cli_overrides(self) -> dict[str, Any]:
        """Translate the global flags into a config-override tree (None = unset)."""
        overrides: dict[str, Any] = {}
        if self.serial is not None:
            overrides["device"] = {"serial": self.serial}
        if self.format is not None:
            overrides["output"] = {"format": self.format}
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
        generic.exit_code = ExitCode(1)
        emit_error(generic)
        raise typer.Exit(1) from exc


def _emit(result: Any, fmt: OutputFormat) -> None:
    """Render a pydantic result (``.render``) or a plain dict (daemon path) to stdout."""
    if hasattr(result, "render"):
        typer.echo(result.render(fmt))
        return
    import json

    indent = 2 if fmt is OutputFormat.pretty else None
    sep = None if indent else (",", ":")
    typer.echo(json.dumps(result, indent=indent, separators=sep, ensure_ascii=False))


# --------------------------------------------------------------------------- daemon route


def _warm(engine: Engine) -> None:
    """Force the lazy device connection so the engine's analyze-cache key (derived from
    the connected serial) matches what a prior ``analyze`` wrote. Action/inspect
    commands resolve cached element ids and need a device anyway, so this is free.
    """
    _ = engine.device


# Engine method name → daemon command name (they differ only for ``input``).
_DAEMON_CMD = {"input_text": "input"}


def _daemon_error(err: dict[str, Any]) -> AuaError:
    """Reconstruct an :class:`AuaError` (with the right exit code) from a daemon error."""
    code = err.get("code", "error")
    message = err.get("message", "daemon error")
    hint = err.get("hint")
    mapping: dict[str, type[AuaError]] = {
        "usage": UsageError,
        "device": DeviceError,
        "config": ConfigError,
    }
    if code in mapping:
        return mapping[code](message, hint=hint)
    if code.startswith("provider"):
        out = AuaError(message, hint=hint, code=code)
        out.exit_code = ExitCode.PROVIDER
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

            if daemon_mod.is_running(cfg):
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


app = typer.Typer(
    name="aua",
    help="android-ui-analyser — structured Android UI perception + action for agents.",
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
    format: str | None = typer.Option(None, "--format", help="Output format: json|pretty|compact."),
    profile: str | None = typer.Option(None, "--profile", help="Named config profile to overlay."),
    timeout: int | None = typer.Option(None, "--timeout", help="Per-operation timeout in ms."),
    log_level: str = typer.Option(
        "warn", "--log-level", help="error|warn|info|debug (logs → stderr)."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the cached analyze result."),
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
            f"invalid --format '{format}'", hint="Choose one of: json, pretty, compact."
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
) -> None:
    """Emit Set-of-Marks JSON (§8) for the current screen."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        nc = no_cache or _opts(ctx).no_cache
        result = _route(
            engine,
            "analyze",
            source=source,
            with_ocr=with_ocr,
            query=query,
            annotate=_annotate_arg(annotate),
            strategy=strategy,
            cheap=cheap,
            deep=deep,
            no_cache=nc,
        )
        _emit(result, fmt)

    _run(ctx, go)


@app.command()
def screenshot(
    ctx: typer.Context,
    path: str | None = typer.Argument(None, help="Output PNG path (default under run dir)."),
    annotate: bool = typer.Option(False, "--annotate", help="Overlay Set-of-Marks numbers."),
) -> None:
    """Save a raw screenshot (PNG); ``--annotate`` overlays the last analyze marks."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.screenshot(path, annotate=annotate), fmt)

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
        0, "--timeout", help="Poll until present or timeout ms (0 = instant)."
    ),
) -> None:
    """Is this text on screen right now? Exit 0 if present, 1 if not."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        result = engine.has(
            text,
            match=match,
            ignore_case=ignore_case,
            ocr_fallback=ocr_fallback,
            source=source,
            timeout_ms=timeout,
        )
        _emit(result, fmt)
        if not result.found:
            raise typer.Exit(1)

    _run(ctx, go)


# --------------------------------------------------------------------------- actions


@app.command()
def tap(
    ctx: typer.Context,
    element_id: int = typer.Argument(..., metavar="ID", help="Element id to tap."),
) -> None:
    """Tap an element (by id from the last analyze)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "tap", element_id=element_id), fmt)

    _run(ctx, go)


@app.command(name="click")
def click_cmd(
    ctx: typer.Context,
    element_id: int = typer.Argument(..., metavar="ID", help="Element id to tap (alias of tap)."),
) -> None:
    """Alias of ``tap``."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "tap", element_id=element_id), fmt)

    _run(ctx, go)


@app.command(name="long-press")
def long_press(
    ctx: typer.Context,
    element_id: int = typer.Argument(..., metavar="ID", help="Element id to long-press."),
    ms: int = typer.Option(600, "--ms", help="Press duration in milliseconds."),
) -> None:
    """Long-press an element."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "long_press", element_id=element_id, ms=ms), fmt)

    _run(ctx, go)


@app.command(name="input")
def input_cmd(
    ctx: typer.Context,
    element_id: int = typer.Argument(..., metavar="ID", help="Element id to type into."),
    text: str = typer.Argument(..., help="Text to type."),
    submit: bool = typer.Option(False, "--submit", help="Send the IME action after typing."),
) -> None:
    """Focus an element and type text; ``--submit`` sends the IME action."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "input_text", element_id=element_id, text=text, submit=submit), fmt)

    _run(ctx, go)


@app.command()
def clear(
    ctx: typer.Context,
    element_id: int = typer.Argument(..., metavar="ID", help="Element id to clear."),
) -> None:
    """Clear the text of an element."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "clear", element_id=element_id), fmt)

    _run(ctx, go)


@app.command()
def swipe(
    ctx: typer.Context,
    direction: str | None = typer.Argument(None, help="up|down|left|right (or use --coords)."),
    from_id: int | None = typer.Option(None, "--from", help="Anchor the swipe at this element."),
    percent: int = typer.Option(50, "--percent", help="Swipe distance as a % of the screen."),
    coords: tuple[int, int, int, int] | None = typer.Option(
        None,
        "--coords",
        help="Explicit x1 y1 x2 y2 (overrides direction).",
    ),
) -> None:
    """Swipe in a direction (optionally from an element) or by explicit coordinates."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        coord_tuple = tuple(coords) if coords is not None else None
        _emit(
            _route(
                engine,
                "swipe",
                direction=direction,
                from_id=from_id,
                percent=percent,
                coords=coord_tuple,
            ),
            fmt,
        )

    _run(ctx, go)


@app.command(name="scroll-to")
def scroll_to(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text or resource-id to scroll to."),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
) -> None:
    """Scroll the container until an element appears (or the swipe limit is hit)."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "scroll_to", query=text, match=match, ignore_case=ignore_case), fmt)

    _run(ctx, go)


@app.command()
def key(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="back|home|enter|recents|KEYCODE_*."),
) -> None:
    """Press a hardware/navigation key."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(_route(engine, "key", name=name), fmt)

    _run(ctx, go)


@app.command()
def wait(
    ctx: typer.Context,
    for_: str | None = typer.Option(None, "--for", help="Text/resource-id to wait for."),
    idle: bool = typer.Option(False, "--idle", help="Wait for the UI to go idle."),
    timeout: int = typer.Option(5000, "--timeout", help="Timeout in milliseconds."),
    match: str = typer.Option("contains", "--match", help="exact|contains|regex."),
    ignore_case: bool = typer.Option(False, "--ignore-case", help="Case-insensitive match."),
) -> None:
    """Wait for text to appear, or for the UI to become idle."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(
            _route(
                engine,
                "wait",
                for_=for_,
                idle=idle,
                timeout_ms=timeout,
                match=match,
                ignore_case=ignore_case,
            ),
            fmt,
        )

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
    action: str = typer.Argument(..., metavar="ACTION", help="foreground|launch|stop|current."),
    package: str | None = typer.Argument(None, metavar="[PKG]", help="Package for launch/stop."),
) -> None:
    """Inspect or control the foreground app."""

    def go(engine: Engine, fmt: OutputFormat) -> None:
        _emit(engine.app(action, package=package), fmt)

    _run(ctx, go)


@app.command()
def daemon(
    ctx: typer.Context,
    action: str = typer.Argument(..., help="start|stop|status."),
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
    for kind in ("ocr", "detection", "grounding"):
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


# --------------------------------------------------------------------------- mcp


@app.command()
def mcp(ctx: typer.Context) -> None:
    """Run the MCP server over stdio (exposes the engine as MCP tools, §11)."""
    from . import mcp_server

    mcp_server.run_stdio()


if __name__ == "__main__":  # pragma: no cover
    app()
