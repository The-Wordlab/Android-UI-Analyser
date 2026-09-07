"""Host-only accounting for CLI paths that do not pass through the engine router.

A root Click invocation owns one id. Routed exchanges keep their existing detailed rows;
help, parsing failures and direct callbacks get a fallback row only when none was written.
This module never claims a lease or opens a device.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import click


@dataclass
class Invocation:
    argv: list[str]
    invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    context: click.Context | None = None
    records: int = 0
    result: Any = None
    help: bool = False
    error: dict[str, Any] | None = None
    exit_code: int = 0


_current: ContextVar[Invocation | None] = ContextVar("aua_cli_invocation", default=None)


def current() -> Invocation | None:
    return _current.get()


def note_record(invocation_id: str | None) -> None:
    state = current()
    if state is not None and state.invocation_id == invocation_id:
        state.records += 1


def emitted(result: Any) -> None:
    state = current()
    if state is not None:
        state.result = result


def context(ctx: click.Context) -> None:
    state = current()
    if state is not None:
        state.context = ctx


def _contexts(ctx: click.Context | None) -> list[click.Context]:
    out = []
    while ctx is not None:
        out.append(ctx)
        ctx = ctx.parent
    return list(reversed(out))


def _safe_argv(state: Invocation) -> list[str]:
    """Retain registered syntax, never guessed names that may themselves contain secrets."""
    from .guide import COMMAND_SYNONYMS

    allowed = {"--help", "-h", "--version"}
    allowed.update(COMMAND_SYNONYMS)
    for ctx in _contexts(state.context):
        if ctx.command.name:
            allowed.add(ctx.command.name)
        for parameter in ctx.command.params:
            if isinstance(parameter, click.Option):
                allowed.update(parameter.opts)
                allowed.update(parameter.secondary_opts)
    out = []
    for token in state.argv:
        name, separator, _value = token.partition("=")
        if name in allowed:
            out.append(f"{name}=<redacted>" if separator else name)
        else:
            out.append("<redacted>")
    return out


def scrub_request_values(state: Invocation, value: Any) -> Any:
    """Remove private argv values if a parser error/help response echoes them elsewhere."""
    literals = set()
    for raw, safe in zip(state.argv, _safe_argv(state), strict=True):
        if raw == safe:
            continue
        literals.add(raw)
        _name, separator, option_value = raw.partition("=")
        if separator and option_value:
            literals.add(option_value)

    def scrub(nested: Any) -> Any:
        if isinstance(nested, str):
            for literal in sorted(literals, key=len, reverse=True):
                if literal:
                    nested = nested.replace(literal, "<redacted>")
            return nested
        if isinstance(nested, dict):
            return {key: scrub(item) for key, item in nested.items()}
        if isinstance(nested, list):
            return [scrub(item) for item in nested]
        return nested

    return scrub(value)


def _fallback(state: Invocation) -> None:
    if state.records:
        return
    from . import journal, leases
    from .cli import GlobalOpts
    from .config import load_config

    ctx = state.context
    opts = ctx.obj if ctx is not None and isinstance(ctx.obj, GlobalOpts) else None
    if opts is None:
        # Eager --help exits before the root callback constructs GlobalOpts. Honor only
        # the host-side routing/config options here; never run the callback or its engine.
        parsed: dict[str, Any] = {}
        supported = {"--config", "--profile", "--owner", "--serial", "--platform"}
        args = iter(state.argv)
        for argument in args:
            if argument == "--":
                break
            name, separator, value = argument.partition("=")
            if name in supported:
                value = value if separator else next(args, "")
                if value and not value.startswith("--"):
                    parsed[name.removeprefix("--")] = value
        opts = GlobalOpts(**parsed)
    cfg = opts.load() if opts else load_config()
    owner = leases.resolve_owner(opts.owner if opts else None)
    serial = (opts.serial if opts else None) or cfg.device.serial
    # Attribute host-only calls to a unique existing sticky lease without acquiring it.
    if serial is None and cfg.lease.enabled:
        owned = [
            entry
            for entry in leases.list_leases(cfg.lease.registry_dir, platform=cfg.device.platform)
            if leases.entry_owned_by(entry, owner)
        ]
        if len(owned) == 1:
            serial = owned[0].get("target_id") or owned[0].get("serial")
    if serial:
        from .session import active_session_metadata

        metadata = active_session_metadata(cfg.cache.dir, serial, owner, platform=cfg.device.platform)
    else:
        metadata = {}
    names = [item.command.name for item in _contexts(ctx)[1:] if item.command.name]
    path = " ".join(["aua", *names])
    command = "_".join(path.split()[1:]).replace("-", "_") or "cli"
    command = command.removesuffix("_and_analyze")
    if state.help:
        command = "cli_help"
    elif state.error and state.error.get("code") == "cli_usage_error":
        command = "cli_usage_error"
    result = state.result
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    # Direct callbacks have no structured request/privacy class. Even a successful response
    # can be clipboard content, private preferences or configuration, so retain only protocol
    # outcomes/timings. Routed exchanges still own their normal redacted full response.
    result = {
        key: value
        for key, value in (result.items() if isinstance(result, dict) else ())
        if key in {"ok", "verified", "duration_ms", "wall_ms", "elapsed_ms"}
        and isinstance(value, (bool, int, float))
    }
    result.setdefault("ok", state.exit_code == 0)
    error = (
        {"code": state.error.get("code"), "message": "CLI invocation failed."}
        if state.error
        else None
    )
    expected_error = opts.expect_error if opts else None
    journal.record(
        cache_dir=cfg.cache.dir,
        serial=serial,
        platform=cfg.device.platform,
        source="cli",
        cmd=command,
        args={"command": path, "argv": _safe_argv(state)},
        result=result,
        error=error,
        ok=state.exit_code == 0 and not (isinstance(result, dict) and result.get("ok") is False),
        duration_ms=(time.monotonic() - state.started) * 1000,
        owner=owner,
        extra={
            **metadata,
            "invocation_id": state.invocation_id,
            "caller_fallback": True,
            **(
                {
                    "expected_error_code": expected_error,
                    "expected_error_matched": bool(error and error.get("code") == expected_error),
                }
                if expected_error
                else {}
            ),
        },
    )


class CommandAccounting:
    """Click mixin shared by ordinary commands, optional-value commands and groups."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        context(ctx)
        try:
            return super().parse_args(ctx, args)  # type: ignore[misc]
        except click.UsageError:
            state = current()
            if state is not None:
                state.error = {
                    "code": "cli_usage_error",
                    "message": "Command-line parsing failed; no command was executed.",
                }
            raise

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        state = current()
        if state is not None:
            state.help = True
        super().format_help(ctx, formatter)  # type: ignore[misc]

    def main(self, *args: Any, **kwargs: Any) -> Any:
        if current() is not None:
            return super().main(*args, **kwargs)  # type: ignore[misc]
        import sys

        argv = kwargs.get("args", args[0] if args else None)
        state = Invocation(list(sys.argv[1:] if argv is None else argv))
        token = _current.set(state)
        try:
            result = super().main(*args, **kwargs)  # type: ignore[misc]
            if isinstance(result, int) and not isinstance(result, bool):
                state.exit_code = result
            return result
        except (SystemExit, click.exceptions.Exit) as exc:
            state.exit_code = int(getattr(exc, "exit_code", getattr(exc, "code", 1)) or 0)
            raise
        except BaseException:
            state.exit_code = 1
            raise
        finally:
            with contextlib.suppress(Exception):
                _fallback(state)
            _current.reset(token)
