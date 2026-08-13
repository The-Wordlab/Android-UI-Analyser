"""Durable, cancellable background waits for daemon and MCP transports.

Jobs are deliberately limited to read-only wait primitives.  A long wait runs on the
transport's warm :class:`Engine`, while status/cancel calls remain responsive.  The manager
also exposes an active-job guard so no second UI operation can race the worker.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .atomic import atomic_write_text
from .errors import AuaError, JobCancelledError, UsageError

if TYPE_CHECKING:
    from .engine import Engine

JobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]

SUPPORTED_OPERATIONS = frozenset({"await", "wait-stable", "wait-changed", "wait-after-change"})
_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


class JobState(BaseModel):
    """Persisted transport-independent state for one background wait."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    serial: str
    owner: str | None = None
    session_id: str | None = None
    status: JobStatus = "queued"
    created_ms: int
    started_ms: int | None = None
    finished_ms: int | None = None
    worker_pid: int | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


def _event(state: JobState, status: JobStatus, detail: str) -> None:
    """Append a small persisted lifecycle breadcrumb for reconnecting callers."""
    state.events.append(
        {"at_ms": int(time.time() * 1000), "status": status, "detail": detail}
    )
    # Lifecycle histories are diagnostics, not an unbounded log.
    state.events = state.events[-20:]


def _job_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / "jobs"


def _job_path(cache_dir: str | Path, job_id: str) -> Path:
    safe = "".join(char for char in job_id if char.isalnum() or char in "-_.")
    return _job_dir(cache_dir) / f"{safe or 'invalid'}.json"


def _write(cache_dir: str | Path, state: JobState) -> None:
    atomic_write_text(_job_path(cache_dir, state.job_id), state.model_dump_json(indent=2))


def _read(cache_dir: str | Path, job_id: str) -> JobState | None:
    try:
        payload = json.loads(_job_path(cache_dir, job_id).read_text(encoding="utf-8"))
        return JobState.model_validate(payload)
    except (OSError, ValueError, TypeError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _execute(engine: Engine, operation: str, args: dict[str, Any]) -> Any:
    """Execute the small public job vocabulary on one already-owned Engine."""
    if operation == "await":
        return _dump(
            engine.await_predicate(
                str(args["predicate"]),
                timeout_ms=int(args.get("timeout_ms", 60_000)),
                poll_ms=int(args.get("poll_ms", 500)),
                match=str(args.get("match", "contains")),
                ignore_case=bool(args.get("ignore_case", False)),
                observe=bool(args.get("observe", True)),
            )
        )
    if operation == "wait-stable":
        return _dump(
            engine.wait_stable(
                timeout_ms=int(args.get("timeout_ms", 30_000)),
                interval_ms=int(args.get("poll_ms", 120)),
                settle_ms=int(args.get("settle_ms", 200)),
                observe=bool(args.get("observe", True)),
            )
        )
    if operation == "wait-changed":
        return _dump(
            engine.wait_changed(
                timeout_ms=int(args.get("timeout_ms", 15_000)),
                interval_ms=int(args.get("poll_ms", 120)),
                observe=bool(args.get("observe", True)),
            )
        )
    if operation == "wait-after-change":
        return _dump(
            engine.wait_after_change(
                timeout_ms=int(args.get("timeout_ms", 60_000)),
                interval_ms=int(args.get("poll_ms", 120)),
                settle_ms=int(args.get("settle_ms", 1_200)),
                confirmation_ms=int(args.get("confirmation_ms", 1_800)),
                observe=bool(args.get("observe", True)),
            )
        )
    raise UsageError(
        f"unsupported job operation {operation!r}",
        hint="Use await, wait-stable, wait-changed, or wait-after-change.",
    )


class JobManager:
    """One-job-at-a-time worker bound to a warm Engine."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.cache_dir = engine.config.cache.dir
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_id: str | None = None

    def _owner(self) -> str | None:
        return getattr(self.engine, "_lease_owner_resolved", None) or getattr(
            self.engine, "_lease_owner", None
        )

    def _serial(self) -> str:
        device = getattr(self.engine, "_device", None)
        serial = getattr(device, "serial", None) or getattr(
            self.engine.config.device, "serial", None
        )
        if serial:
            return str(serial)
        return str(self.engine.device.serial)

    def _session_id(self) -> str | None:
        from .session import load_session_state

        state = load_session_state(
            self.cache_dir,
            serial=self._serial(),
            owner=self._owner(),
        )
        return state.session_id if state is not None else None

    def _visible(self, state: JobState) -> bool:
        owner = self._owner()
        return state.owner == owner

    def _load(self, job_id: str) -> JobState:
        state = _read(self.cache_dir, job_id)
        if state is None or not self._visible(state):
            raise UsageError(f"job {job_id!r} was not found for this owner")
        if state.status in {"queued", "running", "cancel_requested"} and not _pid_alive(
            state.worker_pid
        ):
            state.status = "interrupted"
            state.finished_ms = int(time.time() * 1000)
            state.error = {
                "code": "job_interrupted",
                "message": "the worker process exited before this job completed",
            }
            _event(state, "interrupted", "worker process exited")
            _write(self.cache_dir, state)
        return state

    def active(self) -> JobState | None:
        with self._lock:
            if not self._active_id:
                return None
            state = _read(self.cache_dir, self._active_id)
            if state is None or state.status in _TERMINAL:
                self._active_id = None
                return None
            return state

    def start(self, operation: str, args: dict[str, Any]) -> dict[str, Any]:
        operation = operation.strip().casefold().replace("_", "-")
        if operation not in SUPPORTED_OPERATIONS:
            raise UsageError(
                f"unsupported job operation {operation!r}",
                hint="Use await, wait-stable, wait-changed, or wait-after-change.",
            )
        if operation == "await":
            predicate = str(args.get("predicate") or "").strip()
            if not predicate:
                raise UsageError("an await job needs a predicate")
            from .engine import _parse_await_terms

            _parse_await_terms(predicate)
        timeout_ms = int(args.get("timeout_ms", 0))
        if timeout_ms <= 0:
            raise UsageError("a job timeout must be greater than zero")
        with self._lock:
            active = self.active()
            if active is not None:
                raise UsageError(
                    f"job {active.job_id} is already {active.status}",
                    hint=f"Use `aua job status {active.job_id}` or cancel it first.",
                    code="job_busy",
                )
            state = JobState(
                job_id=uuid.uuid4().hex,
                operation=operation,
                args=dict(args),
                serial=self._serial(),
                owner=self._owner(),
                session_id=self._session_id(),
                created_ms=int(time.time() * 1000),
                worker_pid=os.getpid(),
            )
            _event(state, "queued", "durable wait queued")
            self._cancel = threading.Event()
            self._active_id = state.job_id
            _write(self.cache_dir, state)
            self._thread = threading.Thread(
                target=self._run,
                args=(state.job_id,),
                name=f"aua-job-{state.job_id[:8]}",
                daemon=True,
            )
            self._thread.start()
            return self._public(state)

    def _run(self, job_id: str) -> None:
        with self._lock:
            state = self._load(job_id)
            state.status = "running"
            state.started_ms = int(time.time() * 1000)
            _event(state, "running", "worker started")
            _write(self.cache_dir, state)
        self.engine._job_cancel_event = self._cancel
        try:
            result = _execute(self.engine, state.operation, state.args)
            from .coaching import decorate_result

            result = decorate_result(
                self.engine,
                f"job:{state.operation}",
                result,
                args=state.args,
                current_recorded=False,
            )
            if isinstance(result, dict):
                from .projection import Projection, trim_observation_payload
                from .schema import OutputFormat

                view = Projection.for_observation(
                    getattr(self.engine.config.output, "observation_fields", None),
                    fmt=OutputFormat.json,
                )
                result = trim_observation_payload(result, view, fmt=OutputFormat.json)
            with self._lock:
                state = self._load(job_id)
                if self._cancel.is_set() or state.status == "cancel_requested":
                    state.status = "cancelled"
                    state.error = {
                        "code": "job_cancelled",
                        "message": "job cancelled after its current device read completed",
                    }
                    _event(state, "cancelled", "cancel observed at a safe device-read boundary")
                else:
                    state.status = "succeeded"
                    state.result = _dump(result)
                    _event(state, "succeeded", "wait completed")
        except JobCancelledError as exc:
            with self._lock:
                state = self._load(job_id)
                state.status = "cancelled"
                error_value = exc.to_dict().get("error")
                state.error = error_value if isinstance(error_value, dict) else None
                _event(state, "cancelled", "worker acknowledged cancellation")
        except AuaError as exc:
            with self._lock:
                state = self._load(job_id)
                state.status = "failed"
                error_value = exc.to_dict().get("error")
                state.error = error_value if isinstance(error_value, dict) else None
                _event(state, "failed", "wait failed")
        except Exception as exc:  # noqa: BLE001 - persisted structured terminal state
            with self._lock:
                state = self._load(job_id)
                state.status = "failed"
                state.error = {"code": "internal_error", "message": str(exc)}
                _event(state, "failed", "worker raised an internal error")
        finally:
            with self._lock:
                state.finished_ms = int(time.time() * 1000)
                _write(self.cache_dir, state)
                if self._active_id == job_id:
                    self._active_id = None
            self.engine._job_cancel_event = None

    def status(self, job_id: str, *, recent_output: bool = False) -> dict[str, Any]:
        return self._public(self._load(job_id), recent_output=recent_output)

    def wait(self, job_id: str, *, timeout_ms: int = 5_000) -> dict[str, Any]:
        if timeout_ms < 0 or timeout_ms > 10_000:
            raise UsageError("job wait timeout must be between 0 and 10000 ms")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            state = self._load(job_id)
            if state.status in _TERMINAL or time.monotonic() >= deadline:
                return self._public(state)
            time.sleep(0.05)

    def cancel(self, job_id: str, *, wait_ms: int = 1_000) -> dict[str, Any]:
        if wait_ms < 0 or wait_ms > 10_000:
            raise UsageError("job cancel wait must be between 0 and 10000 ms")
        with self._lock:
            state = self._load(job_id)
            if state.status in _TERMINAL:
                return self._public(state, recent_output=True)
            if state.worker_pid != os.getpid() or self._active_id != job_id:
                raise UsageError(
                    "this process no longer owns the running job",
                    hint="Reconnect to the original AUA daemon, or inspect it until it becomes interrupted.",
                )
            state.status = "cancel_requested"
            _event(state, "cancel_requested", "caller requested cancellation")
            _write(self.cache_dir, state)
            self._cancel.set()
            thread = self._thread
        # Do not hold the manager lock while the worker needs it to persist its terminal state.
        if thread is not None and thread.is_alive() and wait_ms:
            thread.join(timeout=wait_ms / 1000.0)
        return self._public(self._load(job_id), recent_output=True)

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Cancel and briefly join the owned worker before its warm Engine is closed."""
        active = self.active()
        if active is None:
            return
        with self._lock:
            self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout_s))

    def list(self, *, limit: int = 20) -> dict[str, Any]:
        states: list[JobState] = []
        paths = sorted(
            _job_dir(self.cache_dir).glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for path in paths:
            try:
                state = JobState.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if self._visible(state):
                states.append(self._load(state.job_id))
            if len(states) >= max(1, min(limit, 100)):
                break
        return {"ok": True, "jobs": [self._public(state) for state in states]}

    def _public(self, state: JobState, *, recent_output: bool = False) -> dict[str, Any]:
        now_ms = state.finished_ms or int(time.time() * 1000)
        started_ms = state.started_ms or state.created_ms
        elapsed_ms = max(0, now_ms - started_ms)
        timeout_ms = int(state.args.get("timeout_ms", 0) or 0)
        progress = None
        if timeout_ms > 0:
            progress = min(100 if state.status in _TERMINAL else 99, int(elapsed_ms * 100 / timeout_ms))
        cli_status = f"aua job status {shlex.quote(state.job_id)}"
        payload = state.model_dump(mode="json", exclude={"args"})
        payload.update(
            {
                # This reports that the lifecycle query succeeded. `run_ok` separately says
                # whether the wait itself succeeded, just like session_review avoids poisoning
                # its own journal entry when the reviewed run failed.
                "ok": True,
                "run_ok": (
                    state.result.get("ok")
                    if state.status == "succeeded" and isinstance(state.result, dict)
                    else state.status == "succeeded"
                    if state.status in _TERMINAL
                    else None
                ),
                "terminal": state.status in _TERMINAL,
                "elapsed_ms": elapsed_ms,
                "timeout_ms": timeout_ms,
                "progress_percent": progress,
                "recommended_call": (
                    None
                    if state.status in _TERMINAL
                    else {
                        "cli": cli_status,
                        "mcp": {"tool": "job_status", "arguments": {"job_id": state.job_id}},
                        "note": "Reconnect with this id; do not restart the wait.",
                    }
                ),
            }
        )
        if not recent_output:
            payload.pop("events", None)
        else:
            payload["recent_output"] = (
                state.result
                if state.result is not None
                else state.error
                if state.error is not None
                else state.events[-5:]
            )
        return payload


def manager_for(engine: Engine) -> JobManager:
    """Return the process-local manager associated with a warm Engine."""
    manager = getattr(engine, "_aua_job_manager", None)
    if isinstance(manager, JobManager):
        return manager
    manager = JobManager(engine)
    engine._aua_job_manager = manager
    return manager


def reject_if_active(engine: Engine, command: str) -> None:
    """Serialize ordinary engine calls behind the one active background job."""
    active = manager_for(engine).active()
    if active is None:
        return
    raise UsageError(
        f"job {active.job_id} is {active.status}; {command!r} cannot share its Engine",
        hint=f"Use `aua job status {active.job_id}` or `aua job cancel {active.job_id}`.",
        code="job_busy",
    )
