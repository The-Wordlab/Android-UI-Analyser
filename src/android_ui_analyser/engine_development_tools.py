"""Optional developer tools through the selected platform, shared by CLI and MCP.

The adapter owns native command syntax. This layer owns target leases, evidence and the
write-ahead cleanup record, exactly as it does for AUA's other device-facing operations.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .errors import AuaError, DevelopmentToolError, DeviceError, UsageError
from .projection import Projection, trim_observation_payload
from .schema import ActionResult

if TYPE_CHECKING:
    from .engine import Engine


def development_tools_status(self: Engine) -> dict[str, Any]:
    return self.platform.capability("development_tools").status()


def development_tools_setup(self: Engine) -> dict[str, Any]:
    """Explicit optional installation; never runs as a side effect of discovery."""
    return self.platform.capability("development_tools").setup()


def development_tools_catalogue(
    self: Engine, *, operation: str | None = None,
) -> dict[str, Any]:
    catalogue = self.platform.capability("development_tools").catalogue()
    if operation:
        catalogue = [item for item in catalogue if item["name"] == operation]
        if not catalogue:
            raise UsageError(f"unknown developer-tool operation: {operation}")
    return {"platform": self.platform.name, "operations": catalogue}


def _profile_id(self: Engine, target_id: str, family: str = "trace") -> str:
    owner = self._ledger_identity()["owner"]
    digest = hashlib.sha256(f"{self.platform.name}:{target_id}:{owner}:{family}".encode()).hexdigest()
    return f"aua-{digest[:24]}"


def development_tool_spec(
    self: Engine, operation: str, *, arguments: list[str] | None = None,
    options: dict[str, Any] | None = None, project_dir: str | None = None,
    timeout_s: float = 120, confirmed: bool = False, profile_kind: str | None = None,
) -> dict[str, Any]:
    """Validate the entire request before a transport can lease or warm a target."""
    spec = self.platform.capability("development_tools").validate(
        operation, list(arguments or []), dict(options or {}), project_dir, timeout_s,
    )
    if profile_kind is not None:
        permitted = spec.get("profile_kinds") or (
            ["heap"] if spec.get("profile_action") == "capture"
            else ["trace"] if spec.get("profile_action") else []
        )
        if profile_kind not in permitted:
            raise UsageError(f"{operation} does not accept profile_kind={profile_kind!r}")
    if spec.get("needs_confirmation") and not confirmed:
        raise UsageError(
            f"{operation} requires explicit confirmation",
            hint="Review its catalogue entry, then pass --yes (MCP: confirmed=true).",
            code="development_tool_confirmation_required",
        )
    return spec


def development_tool(
    self: Engine,
    operation: str,
    *,
    arguments: list[str] | None = None,
    options: dict[str, Any] | None = None,
    project_dir: str | None = None,
    timeout_s: float = 120,
    confirmed: bool = False,
    profile_kind: str | None = None,
) -> dict[str, Any]:
    """Run a catalogued operation on the host or AUA's exact leased target.

    Native target and session overrides are forbidden by adapter validation. A failed or
    timed-out mutation is never automatically retried.
    """
    service = self.platform.capability("development_tools")
    arguments = list(arguments or [])
    options = dict(options or {})
    spec = development_tool_spec(
        self, operation, arguments=arguments, options=options, project_dir=project_dir,
        timeout_s=timeout_s, confirmed=confirmed, profile_kind=profile_kind,
    )
    route = spec.get("route")
    if route == "virtual.start":
        if len(arguments) != 1:
            raise UsageError("virtual target start requires one explicit definition id")
        result = self.virtual_target_start(
            arguments[0], headless=bool(options.get("headless", False)),
            wait_s=timeout_s, options={"cold": True} if options.get("cold") else {},
        )
        return {**result, "operation": operation, "implementation": "aua-owned-lifecycle"}
    if route == "virtual.stop":
        if len(arguments) != 1:
            raise UsageError("virtual target stop requires one explicit target id")
        result = self.virtual_target_stop(target_id=arguments[0])
        return {**result, "operation": operation, "implementation": "aua-owned-lifecycle"}
    if route == "virtual.delete":
        if len(arguments) != 1:
            raise UsageError("virtual target delete requires one explicit definition id")
        result = self.virtual_target_delete(arguments[0], confirmed=confirmed)
        return {**result, "operation": operation, "implementation": "aua-owned-lifecycle"}
    if spec["scope"] == "host":
        return service.execute(operation, arguments, options, project_dir, timeout_s)

    self.begin_device_use()
    device = self.platform.runtime_capability("device.development_tools", self.device)
    profile_action = spec.get("profile_action")
    session_kind = profile_kind or ("heap" if profile_action == "capture" else "trace")
    session_id = _profile_id(
        self, device.serial, session_kind,
    ) if profile_action else None
    cleanup_key = f"development-tool:{session_id}"
    if session_id:
        options["session"] = session_id
    if profile_action in {"begin", "continue", "capture"}:
        if not self.config.teardown.enabled:
            raise UsageError("profiling requires device teardown to keep its cleanup durable")
        self.record_device_change(
            key=cleanup_key, kind="development_tool_profile", op="restore_development_tool",
            args={"session_id": session_id}, serial=device.serial,
            detail=f"{operation}: restore the exact owned diagnostic session",
        )

    observe = bool(spec.get("observe"))
    if observe:
        self._last_action_kind = str(spec.get("action_kind") or operation)
        self._last_action_site = None
    context = self._acting(label=operation) if observe else contextlib.nullcontext()
    with context:
        if spec.get("ui_automation"):
            if not self._quiesce_background_device_work():
                raise DeviceError(
                    "background device work did not stop for the developer tool",
                    code="development_tool_handover_refused",
                    hint="The native command was not executed; wait for background work to finish.",
                )
            if not self.config.teardown.enabled:
                raise UsageError("developer UI tools require durable device teardown")
            ui_state = device.prepare_development_ui()
            ui_key = "development-tool-ui"
            self.record_device_change(
                key=ui_key, kind="development_tool_ui", op="restore_development_ui",
                args={"state": ui_state}, serial=device.serial,
                detail=f"{operation}: release the temporary developer UI helper",
            )
            with self._device_stood_down() as quiet:
                if not quiet:
                    self.forget_device_change(ui_key, serial=device.serial)
                    raise DeviceError(
                        "capture did not settle for the developer tool",
                        code="development_tool_handover_refused",
                        hint="The native command was not executed; wait for capture to settle.",
                    )
                # Keep the captured runtime: asking self.device while stood down would
                # reacquire the very UI automation connection being handed over.
                def restore() -> None:
                    restored = device.restore_development_ui(ui_state)
                    if not isinstance(restored, dict) or restored.get("ok") is not True:
                        raise DeviceError("developer UI helper cleanup was unsuccessful")
                    self.forget_device_change(ui_key, serial=device.serial)

                try:
                    result = device.execute_development_tool(
                        operation, arguments, options, project_dir, timeout_s,
                    )
                except BaseException as native_error:
                    try:
                        restore()
                    except Exception as cleanup_error:
                        # Keep the original delivery outcome, and the durable undo.
                        if isinstance(native_error, DevelopmentToolError):
                            native_error.result.update(
                                cleanup_pending=True, cleanup_error=str(cleanup_error),
                            )
                            uncertain = native_error.result.get("outcome_unknown", True)
                            native_error.result.setdefault("operation_succeeded", None if uncertain else False)
                            native_error.result.setdefault("action_succeeded", None if uncertain and observe else False)
                        elif isinstance(native_error, Exception):
                            original = (
                                native_error.to_dict()["error"] if isinstance(native_error, AuaError)
                                else {"type": type(native_error).__name__, "message": str(native_error)}
                            )
                            raise DevelopmentToolError(
                                "developer operation failed and its UI helper cleanup also failed",
                                result={"ok": False, "operation": operation,
                                        "operation_succeeded": None,
                                        "action_succeeded": None if observe else False,
                                        "outcome_unknown": spec.get("effect") != "read",
                                        "native_error": original, "cleanup_pending": True,
                                        "cleanup_error": str(cleanup_error)},
                                hint="Do not repeat the operation. Its durable cleanup remains pending.",
                            ) from native_error
                        else:
                            native_error.add_note(f"Developer UI cleanup remains pending: {cleanup_error}")
                    raise
                try:
                    restore()
                except Exception as cleanup_error:
                    operation_succeeded = (
                        True if result.get("ok") is True else
                        None if result.get("outcome_unknown") or type(result.get("ok")) is not bool else False
                    )
                    raise DevelopmentToolError(
                        "developer operation returned but its UI helper cleanup failed",
                        code="development_tool_failed",
                        result={**result,
                                "operation_succeeded": operation_succeeded,
                                "action_succeeded": operation_succeeded if observe else False,
                                "cleanup_pending": True,
                                "cleanup_error": str(cleanup_error)},
                        hint="Do not repeat the operation. Its durable cleanup remains pending.",
                    ) from cleanup_error
        else:
            result = device.execute_development_tool(
                operation, arguments, options, project_dir, timeout_s,
            )
    if result.get("ok") is False:
        raise DeviceError(
            f"developer-tool operation {operation} failed", code="development_tool_failed",
            hint=str(result.get("stderr") or result.get("stdout") or "Inspect target before retrying."),
        )
    if session_id:
        result["session_id"] = session_id
        result["profile_kind"] = session_kind
        finished = profile_action == "capture" or (
            profile_action == "end" and not options.get("keep-session")
        ) or (
            profile_action in {"begin", "continue"}
            and options.get("duration") and options.get("wait")
            and not options.get("keep-session")
        )
        if finished:
            self.forget_device_change(cleanup_key, serial=device.serial)
        if profile_action != "read":
            result["cleanup_pending"] = not finished
        next_operation = spec.get("next_operation")
        if not finished and isinstance(next_operation, str):
            result["recommended_call"] = {
                "cli": f"aua devtools run {next_operation}",
                "mcp": {"tool": "development_tool", "arguments": {"operation": next_operation}},
                "reason": "Continue the same owned diagnostic session through AUA.",
                "executes": False,
            }
    if observe:
        self._version_cache.clear()
        self._last_analyze_result = None
        self._last_hierarchy_hash = None
        self._prefetch.invalidate()
        result["action_succeeded"] = True
        try:
            # Use the same settling, arrival evidence and stable-ID publication as
            # ordinary AUA actions. The adapter describes semantic action meaning.
            observed = self._observe(
                ActionResult(ok=True, action=str(spec.get("action_kind") or operation)), True,
            )
            payload = json.loads(observed.render())
            view = Projection.for_observation(
                self.config.output.observation_fields,
                meta=self.config.output.observation_meta,
            )
            if view is not None:
                payload = trim_observation_payload(payload, view)
            # Native command output and timing remain intact; the AUA action envelope
            # supplies observation, arrival caveats and the reusable-screen contract.
            payload.pop("ok", None)
            result.update(payload)
            if observed.observation is None:
                result["observation_present"] = False
                result["observation_error"] = {
                    "code": "observation_unavailable",
                    "message": "The command succeeded but no post-action frame was available.",
                }
                result["next_action"] = "Inspect the current screen; do not repeat the action."
        except AuaError as exc:
            # Preserve successful action delivery even when evidence collection fails.
            result["observation_present"] = False
            result["observation_error"] = exc.to_dict()["error"]
            result["next_action"] = "Inspect the current screen; do not repeat the action."
    return result
