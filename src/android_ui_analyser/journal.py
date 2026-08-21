"""Append-only JSONL journal of aua commands — feeds the dashboard + usage analysis.

Every daemon dispatch and every in-process CLI ``_route`` writes one line under
``cache.dir/journal/<serial_or_host>.jsonl``. Dashboard tails it for live agent I/O.
The compact journal stays small for polling; a separate bounded detail journal keeps the full
request and response for on-demand dashboard expansion. Secrets, typed input, SQL, parameters,
microphone speech, and audio paths remain redacted in both.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_REDACT_KEYS = frozenset(
    {
        "password",
        "passwd",
        "token",
        "api_key",
        "apikey",
        "secret",
        "authorization",
        "cookie",
    }
)
_MAX_STR = 400
_MAX_RESULT = 1200
_MAX_FILE_BYTES = 8 * 1024 * 1024  # rotate soft-cap per serial file
_MAX_DETAIL_FILE_BYTES = 32 * 1024 * 1024
_INPUT_COMMANDS = frozenset({"input", "input_text", "input_and_analyze"})
_PRIVATE_RESPONSE_COMMANDS = _INPUT_COMMANDS | frozenset(
    {
        "mic_speak",
        "mic_speak_and_analyze",
        "mic_inject",
        "mic_inject_and_analyze",
    }
)
_PRIVATE_RESPONSE_PROTOCOL_STRINGS = frozenset(
    {"action", "code", "await_outcome", "stop_reason"}
)


def _effective_privacy_cmd(cmd: str, privacy_cmd: str | None) -> str:
    """Return the strongest privacy class; metadata can never downgrade a real command."""

    if cmd in _PRIVATE_RESPONSE_COMMANDS:
        return cmd
    return privacy_cmd or cmd


def journal_dir(cache_dir: str | Path) -> Path:
    d = Path(cache_dir).expanduser() / "journal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serial_key(serial: str | None) -> str:
    safe = "host"
    if serial:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in serial)
    return safe


def journal_path(cache_dir: str | Path, serial: str | None) -> Path:
    return journal_dir(cache_dir) / f"{_serial_key(serial)}.jsonl"


def journal_detail_path(cache_dir: str | Path, serial: str | None) -> Path:
    return journal_dir(cache_dir) / f"{_serial_key(serial)}.details.jsonl"


def clear(cache_dir: str | Path, serial: str | None, *, include_host: bool = True) -> list[Path]:
    """Remove compact and detailed journal files visible in one dashboard scope."""

    roots = [journal_path(cache_dir, serial), journal_detail_path(cache_dir, serial)]
    if include_host and serial is not None:
        roots.extend((journal_path(cache_dir, None), journal_detail_path(cache_dir, None)))
    deleted: list[Path] = []
    for root in dict.fromkeys(roots):
        for path in (root, root.with_suffix(root.suffix + ".1")):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted.append(path)
    return deleted


def detail_revision(cache_dir: str | Path, serial: str | None) -> str:
    """Return a cheap change token for detail rows visible to one dashboard scope."""

    current_paths = [journal_detail_path(cache_dir, serial)]
    host = journal_detail_path(cache_dir, None)
    if host not in current_paths:
        current_paths.append(host)
    parts: list[str] = []
    for current in current_paths:
        for path in (current, current.with_suffix(current.suffix + ".1")):
            with contextlib.suppress(OSError):
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _truncate(val: Any, limit: int = _MAX_STR) -> Any:
    if isinstance(val, str) and len(val) > limit:
        return val[:limit] + f"…(+{len(val) - limit})"
    if isinstance(val, dict):
        return {k: _truncate(v, limit) for k, v in list(val.items())[:40]}
    if isinstance(val, list):
        return [_truncate(v, limit) for v in val[:30]]
    return val


def redact_args(args: dict[str, Any] | None, *, cmd: str | None = None) -> dict[str, Any]:
    source = args or {}
    sensitive_literals = frozenset(_detail_sensitive_literals(source, cmd=cmd))
    safe = _detail_value(
        source,
        cmd=cmd,
        request=True,
        sensitive_literals=sensitive_literals,
    )
    if not isinstance(safe, dict):
        return {}
    # Preserve the compact feed's tighter top-level prose preview in addition
    # to its general 400-character/collection bounds.
    for key, value in safe.items():
        if key in {"text", "body", "value"} and isinstance(value, str) and len(value) > 80:
            safe[key] = value[:80] + "…"
    return _truncate(safe)


def _redacted_key(key: object) -> bool:
    lowered = str(key).lower()
    return lowered in _REDACT_KEYS or any(
        part in lowered for part in ("password", "secret", "token")
    )


def _detail_sensitive_literals(
    value: Any,
    *,
    cmd: str | None,
    key: object | None = None,
    depth: int = 0,
) -> set[str]:
    """Collect values that must also be scrubbed if a response echoes them elsewhere."""

    def string_literals(nested_value: Any) -> set[str]:
        if isinstance(nested_value, (str, Path, int, float, bool)):
            return {str(nested_value)} if str(nested_value) else set()
        if isinstance(nested_value, dict):
            return {
                literal
                for child in nested_value.values()
                for literal in string_literals(child)
            }
        if isinstance(nested_value, (list, tuple)):
            return {
                literal for child in nested_value for literal in string_literals(child)
            }
        return set()

    found: set[str] = set()
    lowered = str(key).lower() if key is not None else ""
    mic_speech = (
        depth == 1
        and cmd in {"mic_speak", "mic_speak_and_analyze"}
        and lowered in {"text", "speech"}
    )
    mic_path = depth == 1 and cmd in {
        "mic_inject",
        "mic_inject_and_analyze",
    } and lowered in {
        "path",
        "wav_path",
    }
    typed_input = depth == 1 and cmd in _INPUT_COMMANDS and lowered in {
        "text",
        "value",
    }
    if (
        _redacted_key(lowered)
        or lowered in {"params", "parameters", "sql"}
        or mic_speech
        or mic_path
        or typed_input
    ):
        found.update(string_literals(value))
        return found
    if isinstance(value, dict):
        for nested_key, nested in value.items():
            found.update(
                _detail_sensitive_literals(
                    nested,
                    cmd=cmd,
                    key=nested_key,
                    depth=depth + 1,
                )
            )
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(_detail_sensitive_literals(nested, cmd=cmd, depth=depth + 1))
    return found


def _detail_value(
    value: Any,
    *,
    cmd: str | None,
    request: bool,
    key: object | None = None,
    sensitive_literals: frozenset[str] = frozenset(),
    depth: int = 0,
) -> Any:
    """Return an untruncated JSON-safe value while retaining journal redaction guarantees."""

    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            value = value.model_dump(mode="json")
    lowered = str(key).lower() if key is not None else ""
    if lowered == "sql" and isinstance(value, str):
        return f"<redacted SQL: {len(value)} chars>"
    if lowered in {"params", "parameters"} or _redacted_key(lowered):
        return "<redacted>"
    if request:
        if depth == 1 and cmd in _INPUT_COMMANDS and lowered in {"text", "value"}:
            return f"<redacted input: {len(str(value))} chars>"
        if (
            depth == 1
            and cmd in {"mic_speak", "mic_speak_and_analyze"}
            and lowered in {"text", "speech"}
        ):
            return f"<redacted speech: {len(str(value))} chars>"
        if depth == 1 and cmd in {"mic_inject", "mic_inject_and_analyze"} and lowered in {
            "path",
            "wav_path",
        }:
            return "<redacted audio path>"
    if isinstance(value, dict):
        return {
            str(nested_key): _detail_value(
                nested,
                cmd=cmd,
                request=request,
                key=nested_key,
                sensitive_literals=sensitive_literals,
                depth=depth + 1,
            )
            for nested_key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _detail_value(
                nested,
                cmd=cmd,
                request=request,
                sensitive_literals=sensitive_literals,
                depth=depth + 1,
            )
            for nested in value
        ]
    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and sensitive_literals:
        for literal in sorted(sensitive_literals, key=len, reverse=True):
            if literal:
                value = value.replace(literal, "<redacted>")
        return value
    if isinstance(value, (int, float, bool)) and str(value) in sensitive_literals:
        return "<redacted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _private_response_value(
    value: Any,
    *,
    cmd: str,
    key: object | None = None,
) -> Any:
    """Hide post-input/audio text even when the UI transforms or splits it.

    Exact literal replacement cannot prove privacy after a UI divides input across nodes,
    changes case, or reports only a basename. For these commands, retain response structure
    and non-text values while exposing only fixed AUA protocol strings.
    """

    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(nested_key): _private_response_value(
                nested,
                cmd=cmd,
                key=nested_key,
            )
            for nested_key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_private_response_value(nested, cmd=cmd) for nested in value]
    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if str(key).lower() in _PRIVATE_RESPONSE_PROTOCOL_STRINGS:
            return value
        private_kind = "input" if cmd in _INPUT_COMMANDS else "microphone"
        return f"<redacted post-{private_kind} text>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return "<redacted private response value>"


def _detail_record(
    *,
    detail_id: str,
    ts_ms: int,
    serial: str | None,
    source: str,
    cmd: str,
    args: dict[str, Any] | None,
    ok: bool,
    result: Any,
    error: dict[str, Any] | None,
    privacy_cmd: str | None = None,
) -> dict[str, Any]:
    effective_cmd = _effective_privacy_cmd(cmd, privacy_cmd)
    derived_private = cmd not in _PRIVATE_RESPONSE_COMMANDS and (
        privacy_cmd in _PRIVATE_RESPONSE_COMMANDS
    )
    literals = frozenset(_detail_sensitive_literals(args or {}, cmd=effective_cmd))
    private_response = effective_cmd in _PRIVATE_RESPONSE_COMMANDS
    response: dict[str, Any] = {"ok": ok}
    if result is not None:
        detail_result = (
            _private_response_value(result, cmd=effective_cmd) if private_response else result
        )
        response["result"] = _detail_value(
            detail_result,
            cmd=effective_cmd,
            request=False,
            sensitive_literals=literals,
        )
    if error is not None:
        detail_error = (
            _private_response_value(error, cmd=effective_cmd) if private_response else error
        )
        response["error"] = _detail_value(
            detail_error,
            cmd=effective_cmd,
            request=False,
            sensitive_literals=literals,
        )
    return {
        "detail_id": detail_id,
        "ts_ms": ts_ms,
        "serial": serial,
        "source": source,
        "request": {
            "cmd": cmd,
            "args": (
                _private_response_value(args or {}, cmd=effective_cmd)
                if derived_private
                else _detail_value(
                    args or {},
                    cmd=effective_cmd,
                    request=True,
                    sensitive_literals=literals,
                )
            ),
        },
        "response": response,
    }


def _rotate(path: Path, max_bytes: int) -> None:
    if not path.is_file() or path.stat().st_size <= max_bytes:
        return
    rotated = path.with_suffix(path.suffix + ".1")
    with contextlib.suppress(OSError):
        rotated.unlink(missing_ok=True)
    path.rename(rotated)


def _append_private(path: Path, line: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as file:
        file.write(line)


def summarize_result(result: Any) -> Any:
    """Keep journal lines small but useful for the dashboard."""
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        with contextlib.suppress(Exception):
            result = result.model_dump(mode="json")
    if isinstance(result, dict):
        slim: dict[str, Any] = {}
        for key in (
            "ok",
            "action",
            "detail",
            "code",
            "hint",
            "serial",
            "found",
            "matched",
            "path",
            "package",
            "activity",
            "known_screen",
            "tier_used",
            "duration_ms",
            "via",
            "error",
            "message",
            "session_id",
            "goal_hash",
            "recommended_call",
            "goal_progress",
            "advice",
            "stale_risk",
            "verified",
            "finished",
            "errors",
            "cleanup",
            "await_outcome",
            "await_terms",
            "arrival_mismatch",
            "elapsed_ms",
            "stop_reason",
            "steps_run",
        ):
            if key in result:
                slim[key] = _truncate(result[key], 200)
        meta = result.get("meta")
        if isinstance(meta, dict):
            slim["meta"] = {
                k: meta.get(k)
                for k in (
                    "known_screen",
                    "tier_used",
                    "duration_ms",
                    "via",
                    "package",
                    "capture_hint",
                    "path",
                )
                if k in meta
            }
        screen = result.get("screen")
        if isinstance(screen, dict):
            slim["screen"] = {
                k: screen.get(k) for k in ("package", "activity", "width", "height") if k in screen
            }
        if "elements" in result and isinstance(result["elements"], list):
            slim["elements_count"] = len(result["elements"])
        if "observation" in result and isinstance(result["observation"], dict):
            obs = result["observation"]
            slim["observation"] = {
                "elements_count": len(obs.get("elements") or []),
                "meta": (obs.get("meta") or {}).get("known_screen")
                if isinstance(obs.get("meta"), dict)
                else None,
            }
        raw = json.dumps(slim, ensure_ascii=False, default=str)
        if len(raw) > _MAX_RESULT:
            return _truncate(slim, 120)
        return slim
    return _truncate(result, 200)


def summarize_error(
    error: dict[str, Any],
    *,
    cmd: str | None = None,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bound nested action observations carried by structured error results."""

    safe = dict(error)
    if cmd in {"mic_inject", "mic_inject_and_analyze"}:
        for key in ("path", "wav_path"):
            raw_path = (args or {}).get(key)
            if not isinstance(raw_path, (str, Path)):
                continue
            for field in ("message", "hint"):
                value = safe.get(field)
                if isinstance(value, str):
                    safe[field] = value.replace(str(raw_path), "<redacted audio path>")
    if "result" in safe:
        safe["result"] = summarize_result(safe["result"])
    return _truncate(safe, 300)


def record(
    *,
    cache_dir: str | Path,
    serial: str | None,
    source: str,
    cmd: str,
    args: dict[str, Any] | None = None,
    ok: bool = True,
    duration_ms: float | None = None,
    result: Any = None,
    error: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    owner: str | None = None,
    privacy_cmd: str | None = None,
) -> str | None:
    """Append one journal event (best-effort; never raises into the caller).

    ``owner`` is the lease holder that ran the command. The journal is per-device and every
    agent driving that device appends to it, so without a name a reader cannot tell its own
    last command from someone else's — and `pid` cannot stand in, because everything routed
    through the warm daemon carries the daemon's pid.
    """
    try:
        path = journal_path(cache_dir, serial)
        now = time.time()
        ts_ms = int(now * 1000)
        detail_id = uuid.uuid4().hex
        effective_cmd = _effective_privacy_cmd(cmd, privacy_cmd)
        derived_private = cmd not in _PRIVATE_RESPONSE_COMMANDS and (
            privacy_cmd in _PRIVATE_RESPONSE_COMMANDS
        )
        compact_args = redact_args(args, cmd=effective_cmd)
        if derived_private:
            compact_args = _private_response_value(compact_args, cmd=effective_cmd)
        event: dict[str, Any] = {
            "ts": now,
            "ts_ms": ts_ms,
            "source": source,
            "cmd": cmd,
            "args": compact_args,
            "ok": ok,
            "serial": serial,
            "pid": os.getpid(),
        }
        if owner:
            event["owner"] = owner
        correlation = dict(extra or {})
        with contextlib.suppress(Exception):
            from .session import active_session_metadata

            for key, value in active_session_metadata(cache_dir, serial, owner).items():
                correlation.setdefault(key, value)
        if isinstance(result, dict):
            for key in ("session_id", "goal_hash"):
                if result.get(key):
                    correlation.setdefault(key, result[key])
        for key in ("session_id", "goal_hash", "invocation_id"):
            if correlation.get(key):
                event[key] = correlation[key]
        if duration_ms is not None:
            event["duration_ms"] = round(float(duration_ms), 1)
        sensitive_literals = frozenset(
            _detail_sensitive_literals(args or {}, cmd=effective_cmd)
        )
        if error:
            compact_error: Any = summarize_error(error, cmd=effective_cmd, args=args)
            if effective_cmd in _PRIVATE_RESPONSE_COMMANDS:
                compact_error = _private_response_value(compact_error, cmd=effective_cmd)
            event["error"] = _detail_value(
                compact_error,
                cmd=effective_cmd,
                request=False,
                sensitive_literals=sensitive_literals,
            )
        if result is not None:
            compact_result = summarize_result(result)
            if effective_cmd in _PRIVATE_RESPONSE_COMMANDS:
                compact_result = _private_response_value(compact_result, cmd=effective_cmd)
            event["result"] = _detail_value(
                compact_result,
                cmd=effective_cmd,
                request=False,
                sensitive_literals=sensitive_literals,
            )
        if correlation:
            event["extra"] = _truncate(correlation, 200)
        detail = _detail_record(
            detail_id=detail_id,
            ts_ms=ts_ms,
            serial=serial,
            source=source,
            cmd=cmd,
            args=args,
            ok=ok,
            result=result,
            error=error,
            privacy_cmd=privacy_cmd,
        )
        with _lock:
            detail_path = journal_detail_path(cache_dir, serial)
            detail_written = False
            try:
                _rotate(detail_path, _MAX_DETAIL_FILE_BYTES)
                _append_private(
                    detail_path,
                    json.dumps(detail, ensure_ascii=False, default=str) + "\n",
                )
                event["detail_id"] = detail_id
                detail_written = True
            except OSError as exc:
                logger.debug("journal detail write failed: %s", exc)
            with contextlib.suppress(OSError):
                _rotate(path, _MAX_FILE_BYTES)
            _append_private(path, json.dumps(event, ensure_ascii=False, default=str) + "\n")
        # Keep headless idle-watchdog from auto-stopping mid-session.
        with contextlib.suppress(Exception):
            from .emulator import touch_activity

            touch_activity(cache_dir, serial)
        return detail_id if detail_written else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("journal write failed: %s", exc)
        return None


def read_since(
    cache_dir: str | Path,
    serial: str | None,
    *,
    since_ms: int | None = None,
    limit: int = 200,
    include_dashboard: bool = False,
) -> list[dict[str, Any]]:
    path = journal_path(cache_dir, serial)
    # Also merge host journal if serial-specific is empty of recent events.
    paths = [path]
    host = journal_path(cache_dir, None)
    if host != path:
        paths.append(host)
    events: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[-(limit * 2) :]:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if not include_dashboard and row.get("source") == "dashboard":
                    continue
                if since_ms is not None and int(row.get("ts_ms") or 0) < since_ms:
                    continue
                # keep host events without a serial; drop other devices
                if serial and row.get("serial") not in (None, serial, ""):
                    continue
                events.append(row)
    events.sort(key=lambda e: int(e.get("ts_ms") or 0))
    return events[-limit:]


def read_detail(
    cache_dir: str | Path,
    serial: str | None,
    detail_id: str,
) -> dict[str, Any] | None:
    """Read one full redacted exchange without adding it to the dashboard polling feed."""

    if not detail_id:
        return None
    current_paths = [journal_detail_path(cache_dir, serial)]
    host = journal_detail_path(cache_dir, None)
    if host not in current_paths:
        current_paths.append(host)
    paths: list[Path] = []
    for current in current_paths:
        paths.extend((current, current.with_suffix(current.suffix + ".1")))
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("detail_id") != detail_id:
                    continue
                if serial and row.get("serial") not in (None, serial, ""):
                    continue
                return row
    return None


def record_emitted_response(
    *,
    cache_dir: str | Path,
    serial: str | None,
    invocation_id: str,
    detail_id: str | None,
    cmd: str,
    args: dict[str, Any] | None,
    result: Any,
    request_context: dict[str, Any] | None = None,
) -> bool:
    """Append the final CLI-visible response to an existing exchange detail.

    CLI output may adopt a ``--until`` observation and project element fields after the
    engine or daemon has journaled its transport response. The compact event remains the
    streaming index; appending the same detail id makes on-demand reads return the final
    agent-visible payload without duplicating a polling row.
    """

    if not invocation_id or not cmd:
        return False
    try:
        if not detail_id:
            for event in reversed(
                read_since(cache_dir, serial, limit=200, include_dashboard=True)
            ):
                if event.get("invocation_id") != invocation_id or event.get("cmd") != cmd:
                    continue
                candidate = event.get("detail_id")
                if isinstance(candidate, str) and candidate:
                    detail_id = candidate
                    break
        if detail_id is None:
            return False
        result_value = result
        if hasattr(result_value, "model_dump"):
            with contextlib.suppress(Exception):
                result_value = result_value.model_dump(mode="json")
        ok = not (isinstance(result_value, dict) and result_value.get("ok") is False)
        revised = _detail_record(
            detail_id=detail_id,
            ts_ms=int(time.time() * 1000),
            serial=serial,
            source="cli",
            cmd=cmd,
            args=args,
            ok=ok,
            result=result_value,
            error=None,
        )
        if request_context:
            stored_context = dict(request_context)
            if cmd in _PRIVATE_RESPONSE_COMMANDS and "until" in stored_context:
                stored_context["until"] = _private_response_value(
                    stored_context["until"],
                    cmd=cmd,
                    key="until",
                )
            literals = frozenset(_detail_sensitive_literals(args or {}, cmd=cmd))
            revised["request"]["client"] = _detail_value(
                stored_context,
                cmd=cmd,
                request=True,
                sensitive_literals=literals,
            )
        with _lock:
            path = journal_detail_path(cache_dir, serial)
            _rotate(path, _MAX_DETAIL_FILE_BYTES)
            _append_private(
                path,
                json.dumps(revised, ensure_ascii=False, default=str) + "\n",
            )
        return True
    except Exception as exc:  # noqa: BLE001 — journaling never changes command output
        logger.debug("journal emitted-response write failed: %s", exc)
        return False


def failure_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Cheap usage rollup for the dashboard (failing cmds, slow calls)."""
    by_cmd: dict[str, dict[str, int]] = {}
    slow: list[dict[str, Any]] = []
    fails: list[dict[str, Any]] = []
    for e in events:
        cmd = str(e.get("cmd") or "?")
        slot = by_cmd.setdefault(cmd, {"ok": 0, "fail": 0})
        if e.get("ok"):
            slot["ok"] += 1
        else:
            slot["fail"] += 1
            fails.append(
                {
                    "ts_ms": e.get("ts_ms"),
                    "cmd": cmd,
                    "error": (e.get("error") or {}).get("message")
                    if isinstance(e.get("error"), dict)
                    else e.get("error"),
                }
            )
        dur = e.get("duration_ms")
        if isinstance(dur, (int, float)) and dur >= 1500:
            slow.append({"ts_ms": e.get("ts_ms"), "cmd": cmd, "duration_ms": dur})
    return {
        "by_cmd": by_cmd,
        "failures": fails[-30:],
        "slow": slow[-30:],
        "total": len(events),
        "fail_count": sum(1 for e in events if not e.get("ok")),
    }
