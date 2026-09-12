"""Host-only, explicit run context for callers of the existing CLI.

One file binds one caller process and one goal. Commands still run through the installed
CLI/engine exactly once. No observations, credentials or process environments are saved.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import leases
from .agent_results import from_cli
from .config import _ENV_ALIASES, Config
from .errors import UsageError
from .schema import OutputFormat
from .session import load_session_state


class RunContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    caller: dict[str, Any]
    cwd: Path
    config_path: Path
    config_sha256: str = ""
    cache_dir: Path
    platform: str = Field(min_length=1)
    target_id: str | None = None
    session_id: str | None = None
    needs: list[str] = Field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json", exclude={"caller", "config_path", "config_sha256", "schema_version"}
        )


def _write_private(path: Path, value: Any, *, create: bool = False) -> None:
    """Atomic private JSON; initialization never overwrites an existing run."""
    if create:
        with open(path, "x", encoding="utf-8", opener=lambda p, f: os.open(p, f, 0o600)) as out:
            json.dump(value, out, ensure_ascii=False)
        return
    fd, temporary = tempfile.mkstemp(prefix=".run-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def create_run(
    path: Path, config: Config, *, owner: str | None = None, needs: list[str] | None = None
) -> RunContext:
    path = path.expanduser().absolute()
    if path.exists():
        raise UsageError("run context already exists", code="run_context_exists")
    if not config.lease.enabled:
        raise UsageError("a shared run requires device leasing", code="run_context_invalid")
    run_id = uuid.uuid4().hex
    resolved_owner = leases.resolve_owner(owner or f"aua-run-{run_id[:12]}")
    caller = leases.owner_caller(resolved_owner)
    if not caller:
        raise UsageError("cannot identify the run's caller process", code="run_context_owner")
    # The lease registry remains shared. Only observation/session/daemon storage is isolated.
    directory = path.parent / f".{path.name}.{run_id}.d"
    directory.mkdir(mode=0o700, parents=True)
    frozen = config.model_copy(deep=True)
    frozen.cache.dir = str(directory / "cache")
    frozen.output.format = OutputFormat.json
    config_path = directory / "config.json"
    context = RunContext(
        run_id=run_id,
        owner=str(resolved_owner),
        caller=caller,
        cwd=Path.cwd(),
        config_path=config_path,
        cache_dir=Path(frozen.cache.dir),
        platform=frozen.device.platform,
        target_id=frozen.device.serial,
        needs=list(needs or []),
    )
    try:
        _write_private(config_path, frozen.model_dump(mode="json"), create=True)
        context.config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        _write_private(path, context.model_dump(mode="json"), create=True)
    except Exception:
        config_path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return context


def load_run(path: Path) -> RunContext:
    try:
        context = RunContext.model_validate_json(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as exc:
        raise UsageError(
            "run context is missing or invalid; no command was executed",
            code="run_context_invalid",
            hint="Initialize a new run file with `aua run init PATH`.",
        ) from exc
    if leases.owner_caller(leases.resolve_owner(context.owner)) != context.caller:
        raise UsageError(
            "this run belongs to a different caller process; no command was executed",
            code="run_context_owner",
            hint="Use this file from its original agent, or initialize a separate run.",
        )
    try:
        saved = context.config_path.read_bytes()
        if hashlib.sha256(saved).hexdigest() != context.config_sha256:
            raise ValueError("saved configuration changed")
        Config.model_validate_json(saved)
    except (OSError, ValueError) as exc:
        raise UsageError(
            "the run's saved configuration is missing or changed", code="run_context_invalid"
        ) from exc
    return context


@contextlib.contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    # Reuse AUA's portable host file locking. This is independent of device/lease locks.
    with (
        leases._thread_lock(f"agent-run|{path}"),
        open(
            path.with_suffix(path.suffix + ".lock"),
            "a+",
            encoding="utf-8",
            opener=lambda p, f: os.open(p, f, 0o600),
        ) as handle,
    ):
        backend = leases._acquire_file_lock(handle, exclusive=True)
        try:
            yield
        finally:
            leases._release_file_lock(handle, backend)


def _command_context(arguments: list[str]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Read the installed Click schema without invoking command callbacks or a device."""
    import click
    from typer.main import get_command

    from .cli import alias_fields_on_actions, app, hoist_global_options
    from .cli_invocations import current

    invocation = current()
    previous = invocation.context if invocation is not None else None
    try:
        command = get_command(app)
        args = hoist_global_options(alias_fields_on_actions(arguments))
        ctx = command.make_context("aua", args, resilient_parsing=True)
        root = dict(ctx.params)
        names: list[str] = []
        while isinstance(command, click.Group):
            remaining = [*ctx._protected_args, *ctx.args]
            if not remaining:
                break
            name = remaining[0]
            child = command.get_command(ctx, name)
            if child is None:
                break
            names.append(name)
            ctx = child.make_context(name, remaining[1:], parent=ctx, resilient_parsing=True)
            command = child
        return root, names, dict(ctx.params)
    finally:
        if invocation is not None:
            invocation.context = previous


def _validate_call(context: RunContext, arguments: list[str]) -> str:
    root, names, params = _command_context(arguments)
    if not names:
        raise UsageError("run exec needs an AUA command after --", code="run_context_usage")
    for key in ("config", "profile", "owner", "serial", "platform"):
        if root.get(key) is not None:
            raise UsageError(
                f"--{key} cannot replace saved run context; no command was executed",
                code="run_context_mismatch",
                hint="Set run-wide options before `aua run init`; start a new file to change scope.",
            )
    if root.get("no_lease") or root.get("format") not in (None, "json"):
        raise UsageError("run exec requires leasing and JSON output", code="run_context_mismatch")
    from .cli import _split_needs

    for value in (root.get("needs"), params.get("needs")):
        if value is not None and _split_needs(value) != context.needs:
            raise UsageError(
                "device requirements cannot replace saved run context", code="run_context_mismatch"
            )
    if names[0] in {"run", "mcp", "daemon", "lease"}:
        raise UsageError("this command cannot run inside a goal wrapper", code="run_context_usage")
    if names[:2] in [
        ["emulator", "start"],
        ["virtual-target", "start"],
        ["virtual-target", "provision"],
    ]:
        raise UsageError("use session start to select the run's target", code="run_context_usage")
    command = "_".join(names).replace("-", "_")
    if context.session_id is None:
        if command != "session_start":
            raise UsageError(
                "this run has no goal session; no command was executed",
                code="run_session_missing",
                hint='Run `aua run exec PATH -- session start --goal "<goal>"` first.',
            )
        return command
    if command == "session_start":
        raise UsageError(
            "one run file holds one goal; initialize another file", code="run_context_mismatch"
        )
    if names[0] == "session" and params.get("session_id") not in (None, context.session_id):
        raise UsageError("session id does not match this run", code="run_context_mismatch")
    if params.get("serial") not in (None, context.target_id):
        raise UsageError("target does not match this run", code="run_context_mismatch")
    if names[0] in {"emulator", "virtual-target"}:
        if names[-1] not in {"status", "list", "stop"}:
            raise UsageError(
                "target lifecycle changes require a separate run", code="run_context_usage"
            )
        if params.get("target_id") not in (None, context.target_id) or params.get("owner") not in (
            None,
            context.owner,
        ):
            raise UsageError("target or owner does not match this run", code="run_context_mismatch")
        if any(
            params.get(key)
            for key in ("all_targets", "all_devices", "mine", "definition_id", "avd")
        ):
            raise UsageError(
                "run cleanup must name only its saved target", code="run_context_mismatch"
            )
        if command == "virtual_target_stop" and params.get("target_id") != context.target_id:
            raise UsageError(
                "pass the run's exact --target-id to stop it", code="run_context_mismatch"
            )
    state = load_session_state(
        context.cache_dir, session_id=context.session_id, platform=context.platform
    )
    if state is None or (state.session_id, state.serial, state.owner) != (
        context.session_id,
        context.target_id,
        context.owner,
    ):
        raise UsageError("saved goal session is missing or mismatched", code="run_session_missing")
    if state.finished_ms is None:
        active = load_session_state(
            context.cache_dir,
            serial=context.target_id,
            owner=context.owner,
            platform=context.platform,
        )
        if active is None or active.session_id != context.session_id:
            raise UsageError(
                "the active goal changed; no command was executed", code="run_context_mismatch"
            )
    elif not (
        command
        in {
            "session_review",
            "session_progress",
            "emulator_status",
            "emulator_stop",
            "virtual_target_status",
            "virtual_target_stop",
        }
        or names[0] == "capture"
    ):
        raise UsageError("this goal has ended; initialize a new run", code="run_session_finished")
    return command


def _invoke_cli(
    arguments: list[str], *, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "android_ui_analyser.cli", *arguments],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _response_mismatch(context: RunContext, command: str, result: dict[str, Any]) -> bool:
    """Validate identities actually returned, without treating a capture id as a goal."""
    observation = result.get("observation")
    meta = observation.get("meta") if isinstance(observation, dict) else None
    if isinstance(meta, dict) and (
        (context.target_id and meta.get("device_serial") not in (None, context.target_id))
        or meta.get("platform") not in (None, context.platform)
    ):
        return True
    native = result.get("result")
    if command.startswith("session_") and isinstance(native, dict):
        expected = {
            "owner": context.owner,
            "serial": context.target_id,
            "platform": context.platform,
        }
        if context.session_id:
            expected["session_id"] = context.session_id
        return any(
            value is not None and native.get(key) not in (None, value)
            for key, value in expected.items()
        )
    return False


def _refuse_response(result: dict[str, Any], code: str, message: str) -> None:
    result["ok"] = False
    if isinstance(result.get("error"), dict):
        result["error"]["run_context_error"] = {"code": code, "message": message}
    else:
        result["error"] = {"code": code, "message": message}
    result["observation_contract"].update(reusable=False, analyze_needed=True, reason=message)


def execute_run(path: Path, arguments: list[str]) -> tuple[dict[str, Any], int]:
    path = path.expanduser().absolute()
    # Refuse a missing/wrong-owner file before creating even its host lock.
    load_run(path)
    with _run_lock(path):
        context = load_run(path)
        command = _validate_call(context, arguments)
        # Config discovery and ambient AUA overrides cannot redirect the saved run. Secrets
        # referenced by config (e.g. PROVIDER_API_KEY) stay in the current environment only.
        controls = {
            "AUA_CONFIG",
            "AUA_PROFILE",
            "AUA_OWNER",
            "AUA_WORKER_SCOPE",
            "AUA_DAEMON_SOCKET",
        }
        env = {
            key: value
            for key, value in os.environ.items()
            if not (
                key.startswith("AUA_") and ("__" in key or key in _ENV_ALIASES or key in controls)
            )
        }
        env.update(
            AUA_CACHE__DIR=str(context.cache_dir),
            AUA_OWNER=context.owner,
            AUA_WORKER_SCOPE=context.run_id,
        )
        prefix = [
            "--config",
            str(context.config_path),
            "--format",
            "json",
            "--owner",
            context.owner,
            "--log-level",
            Config.model_validate_json(context.config_path.read_bytes()).log_level,
        ]
        if context.target_id:
            prefix += ["--serial", context.target_id]
        if context.needs:
            prefix += ["--needs", ",".join(context.needs)]
        if not context.config_path.is_file():
            raise UsageError("the run's saved configuration is missing", code="run_context_invalid")
        completed = _invoke_cli([*prefix, *arguments], env=env, cwd=context.cwd)
        # The child CLI owns the actual invocation journal, including its failures. The
        # outer process is transport, not a second device call in session accounting.
        if completed.returncode >= 0:
            from .cli_invocations import current

            invocation = current()
            if invocation is not None:
                invocation.records += 1
        result = from_cli(
            completed.stdout,
            completed.stderr,
            completed.returncode,
            command=command,
            context=context.public(),
        )
        if _response_mismatch(context, command, result):
            _refuse_response(
                result,
                "run_context_mismatch",
                "returned evidence belongs to another run or target; no action was replayed",
            )
        if command == "session_start" and result["ok"]:
            native = result["result"]
            if not isinstance(native, dict) or not all(
                native.get(key) for key in ("session_id", "serial", "owner")
            ):
                _refuse_response(
                    result,
                    "run_context_mismatch",
                    "session start did not return complete run identity; no action was replayed",
                )
            elif native["owner"] != context.owner or (
                context.target_id and native["serial"] != context.target_id
            ):
                _refuse_response(
                    result,
                    "run_context_mismatch",
                    "session start returned another owner or target; no action was replayed",
                )
            else:
                context.session_id = str(native["session_id"])
                context.target_id = str(native["serial"])
                result["context"] = context.public()
                state = load_session_state(
                    context.cache_dir, session_id=context.session_id, platform=context.platform
                )
                if (
                    _response_mismatch(context, command, result)
                    or state is None
                    or (state.session_id, state.serial, state.owner)
                    != (context.session_id, context.target_id, context.owner)
                ):
                    _refuse_response(
                        result,
                        "run_context_mismatch",
                        "session start's returned identity does not match its saved goal; no action was replayed",
                    )
                else:
                    try:
                        _write_private(path, context.model_dump(mode="json"))
                    except OSError:
                        _refuse_response(
                            result,
                            "run_context_persist_failed",
                            "session started but saving its context failed; keep the returned identity and do not repeat session start",
                        )
        return result, completed.returncode if completed.returncode > 0 else (
            0 if result["ok"] else 1
        )
