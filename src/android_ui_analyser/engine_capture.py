"""Pixel evidence over time: the rolling capture buffer and its status/last/export/sheet/explain views, the capture sidecar, the capture hint analyze attaches, and device screen recording.

Engine methods for capture. Each function's first parameter ``self`` is the
:class:`~android_ui_analyser.engine.Engine`; ``Engine`` binds these functions as methods in its
class body, so ``engine.<name>(...)`` runs ``engine_capture.<name>(engine, ...)``. Static helpers are
plain functions bound with ``staticmethod``. Add a new method for this domain here, then attach
it in ``Engine``.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .engine_support import logger
from .errors import DeviceError, UsageError
from .memory import _id_tail
from .providers.base import PlannerProvider
from .schema import ActionResult

if TYPE_CHECKING:
    from .engine import Engine


def _region_from_point(cx: int, cy: int, width: int, height: int) -> str:
    """Map a screen point onto the same 3×3 names used by ``diff_summary``."""
    gx = 0 if cx < width / 3 else (2 if cx >= 2 * width / 3 else 1)
    gy = 0 if cy < height / 3 else (2 if cy >= 2 * height / 3 else 1)
    names = (
        ("upper-left", "upper", "upper-right"),
        ("left", "center", "right"),
        ("lower-left", "lower", "lower-right"),
    )
    return names[gy][gx]


class DeviceStoodDownError(DeviceError):
    """The device is mid-handover and must not be touched until it is picked up again.

    Only the rolling capture buffer ever sees this. It samples on its own thread, so it is
    the one caller that can ask for a frame in the window where the on-device helper holds
    the UiAutomation slot — and satisfying that ask would reconnect uiautomator2 and take the
    slot straight back. Failing fast is the point: a background sampler must never resurrect
    a connection the foreground deliberately tore down.
    """


def _capture_screenshot(self: Engine) -> Any:
    """Grab a frame for the rolling capture buffer, or refuse if the device is on loan.

        The buffer must not hold a device. It used to be handed ``device.screenshot``, a
        method bound to the uiautomator2 client, and that binding outlived every teardown the
        engine performed: closing the client and dropping the engine's reference left the
        sampling thread still holding a live handle, so its next tick reconnected the server
        AUA had just stepped away from.

        Going through the engine makes ``self._device = None`` mean what it says, and makes a
        handover a hard edge rather than a request the sampler is free to ignore.
        """

    if self._stood_down:
        raise DeviceStoodDownError(
            "the device is handed to the on-device helper",
            hint="This is normal during a flow offload; the buffer samples again after.",
        )
    # Never ``self.device``. That property connects, and connecting costs ~2.1s — so a
    # tick firing the instant a handover released the buffer would start a reconnect the
    # *next* handover's two-second settle wait then expires inside. Sampling waits for the
    # engine to pick the device up in its own time; a skipped frame is not worth a refused
    # offload. ``_tick`` treats a frame without pixels as a no-op, so this costs nothing.
    device = self._device
    if device is None:
        return None
    with self.device_use_context(device.serial):
        platform = self.platform.adapter_capability("ui.screenshot")
        return platform.capture_screenshot(device)


def _capture_screenshot_fn(self: Engine) -> Any:
    """The callable handed to :class:`CaptureBuffer`. Bound to the engine, never a device."""

    return self._capture_screenshot


def record_start(self: Engine, path: str | None = None) -> ActionResult:
    runtime = self.platform.runtime_capability("device.recording", self.device)
    remote_path = runtime.recording_destination(path)
    active = runtime.active_recording()
    if active is not None:
        raise DeviceError(
            f"a screen recording is already in progress at {active}",
            hint="Run `aua record stop <path>` before starting another.",
        )
    pending = self._pending_device_change("screen_recording", serial=runtime.target_id)
    if pending is not None:
        raise DeviceError(
            "an earlier screen recording still has pending target-side cleanup",
            code="recording_cleanup_pending",
            hint=(
                "Run `aua teardown run --serial <target> --force` with the original platform "
                "options before starting another recording."
            ),
        )
    self.record_device_change(
        key="screen_recording",
        kind="screen_recording",
        op="discard_recording",
        args={"remote_path": remote_path},
        detail=f"screen recording started at {remote_path}",
    )
    remote = runtime.start_recording(remote_path)
    return ActionResult(ok=True, action="record-start", detail=remote)


def record_stop(self: Engine, local_path: str) -> ActionResult:
    runtime = self.platform.runtime_capability("device.recording", self.device)
    saved = runtime.stop_recording(local_path)
    self.forget_device_change("screen_recording")
    return ActionResult(ok=True, action="record-stop", detail=saved)


def _capture_hint(self: Engine) -> str | None:
    buf = self._capture
    if buf is None or not self.config.capture.hint:
        return None
    if not buf.hint_ready():
        return None
    return "recent pixel change after last action — `aua capture last --since last-action`"


def capture_start(self: Engine, *, connect_if_needed: bool = True) -> dict[str, Any]:
    """Start the rolling capture buffer (daemon-warm sessions).

        ``connect_if_needed=False`` is the daemon auto-start seam. A per-device daemon already
        knows its target from config, so initializing the host-side buffer must not eagerly
        attach uiautomator2 before the accept loop can answer its first request. The sampler
        already waits for ``self._device`` instead of connecting by itself; this keeps buffer
        creation subject to the same rule.
        """
    from .capture import CaptureBuffer, CaptureCfgView

    with self._capture_lock:
        if not self.config.capture.enabled and self._capture is None:
            # Explicit start still allowed even if config default is off.
            pass
        cfg = self.config.capture
        serial: str | None
        if self._device is not None:
            serial = str(self._device.serial)
        else:
            configured = getattr(self.config.device, "serial", None)
            serial = str(configured) if configured else None
        if serial is None:
            if not connect_if_needed:
                raise UsageError(
                    "capture auto-start needs a device-bound daemon",
                    hint=(
                        "Start the daemon with --serial, or run `aua capture start` after "
                        "selecting a device."
                    ),
                )
            serial = str(self.device.serial)
        root = Path(self.config.cache.dir).expanduser() / "captures"
        view = CaptureCfgView(
            enabled=True,
            idle_fps=cfg.idle_fps,
            burst_fps=cfg.burst_fps,
            burst_ms=cfg.burst_ms,
            extend_burst_on_change=cfg.extend_burst_on_change,
            ttl_s=cfg.ttl_s,
            max_mb=cfg.max_mb,
            jpeg_quality=cfg.jpeg_quality,
            hint=cfg.hint,
        )
        if self._capture is not None:
            self._capture.resume()
            if not self._capture.running:
                self._capture.start()
            return self._capture.status()
        # Default is the u2 path: it is ~2.2x faster than `adb exec-out screencap -p` (the
        # device encodes JPEG instead of a full-res PNG) and every frame is re-encoded to JPEG
        # on write anyway, so the lossless capture buys nothing here. The opt-in flag stays for
        # callers that need pixel-exact frames.
        # Deliberately not ``device.screenshot``: a bound method keeps the uiautomator2
        # client alive past every teardown, which is what let a sampling tick reconnect the
        # server mid-handover. The engine picks the source per frame instead.
        shot = self._capture_screenshot_fn()
        buf = CaptureBuffer(
            root=root,
            serial=serial,
            cfg=view,
            screenshot=shot,
            platform=self.platform.name,
        )
        buf.start()
        self._capture = buf
        return buf.status()


def capture_stop(self: Engine) -> dict[str, Any]:
    with self._capture_lock:
        buf = self._capture
        if buf is None:
            return {"ok": True, "action": "capture-stop", "running": False}
        buf.stop()
        self._capture = None
        return {
            "ok": True,
            "action": "capture-stop",
            "running": False,
            "session_id": buf.session_id,
        }


def capture_on(self: Engine) -> dict[str, Any]:
    with self._capture_lock:
        if self._capture is None:
            return self.capture_start()
        self._capture.resume()
        return self._capture.status()


def capture_off(self: Engine) -> dict[str, Any]:
    with self._capture_lock:
        if self._capture is None:
            return {
                "ok": True,
                "action": "capture-status",
                "running": False,
                "paused": True,
            }
        self._capture.pause()
        return self._capture.status()


def capture_idle_pause(self: Engine) -> bool:
    """Stop sampling because the client went quiet; frames already kept stay readable."""
    buf = self._capture
    if buf is None or not buf.running or buf.paused:
        return False
    buf.pause("idle")
    return True


def capture_idle_resume(self: Engine) -> bool:
    """Resume a buffer that idle-paused. An explicit ``capture off`` stays off."""
    buf = self._capture
    if buf is None or not buf.paused:
        return False
    buf.resume(only_if_idle=True)
    return not buf.paused


def _capture_serial(self: Engine) -> str | None:
    """Which device's frames to look for, without connecting to one.

        A disk read must stay host-only: it is the answer of last resort precisely when the
        process holding the device is gone or unusable, so paying a device attach to find the
        directory name would defeat it.
        """
    serial = getattr(self.config.device, "serial", None)
    if serial:
        return str(serial)
    if self._device is not None:
        with contextlib.suppress(Exception):
            return str(self._device.serial)
    root = Path(self.config.cache.dir).expanduser() / "captures"
    with contextlib.suppress(OSError):
        from .capture import target_for_capture_root

        targets = [
            ref
            for entry in root.iterdir()
            if entry.is_dir()
            and (ref := target_for_capture_root(entry)) is not None
            and ref.platform == self.platform.name
        ]
        if len(targets) == 1:
            # Explicit metadata makes this host-only fallback safe even when another platform
            # has the same target id or the filesystem key had to be escaped.
            return targets[0].target_id
    return None


def _capture_from_disk(self: Engine) -> Any:
    """The newest capture session for this device, recovered from ``index.jsonl``.

        Returns ``None`` when nothing was ever recorded for this serial.
        """
    serial = self._capture_serial()
    if not serial:
        return None
    from .capture import read_session_from_disk

    root = Path(self.config.cache.dir).expanduser() / "captures"
    try:
        return read_session_from_disk(root, serial, platform=self.platform.name)
    except Exception:  # noqa: BLE001 - a corrupt cache must not break the caller's command
        logger.debug("capture disk index unreadable", exc_info=True)
        return None


def _disk_capture_payload(self: Engine, found: Any) -> dict[str, Any]:
    """The provenance every disk-sourced capture answer carries, in words."""
    return {
        "source": "disk-index",
        "live": False,
        "session_id": found.session_id,
        "dir": str(found.dir),
        "indexed": found.indexed,
        "available": found.available,
        "newest_frame_age_ms": found.newest_frame_age_ms,
        "note": self._DISK_NOTE,
    }


def _capture_last_from_disk(
    self: Engine,
    *,
    seconds: float | None,
    since: str | None,
    region: str | None,
    where_rid: str | None,
) -> dict[str, Any]:
    """``capture_last`` answered from ``index.jsonl``, for both callers that need it.

        Reached either when this process holds no buffer at all, or when it holds a live one
        that cannot answer a ``--since last-action`` because a restart just superseded the
        session the mark is in. The provenance keys are the same in both cases: whatever is
        returned here is not the live buffer and must never read as though it were.
        """
    found, disk_since = self._disk_session_for(since)
    entries = list(found.entries)
    if disk_since is not None:
        entries = [e for e in entries if e.t_ms >= disk_since]
    elif seconds is not None:
        cutoff = int(time.time() * 1000) - int(seconds * 1000)
        entries = [e for e in entries if e.t_ms >= cutoff]
    if not entries:
        raise UsageError(
            "the capture index has no available frame in the requested window",
            hint="Capture the action again with a live warm daemon; old/pruned pixels are not evidence.",
        )
    from .capture import change_duration_ms, diff_summary

    return {
        "ok": True,
        "action": "capture-last",
        **self._disk_capture_payload(found),
        "frames": [e.__dict__ for e in entries],
        "count": len(entries),
        "summary": diff_summary(entries, region=region),
        "change_duration_ms": change_duration_ms(entries),
        "region": region,
        "where_rid": where_rid,
    }


def _disk_session_for(self: Engine, since: str | None) -> tuple[Any, int | None]:
    """Pick the session that can answer, and the window within it.

        Only the newest session may answer. Walking backward to an older mark turned a frame
        from a previous daemon (and sometimes a previous action) into current post-action proof.
        Missing or pruned evidence is an error; an old screenshot is not a degraded success.
        """
    from .capture import read_sessions_from_disk

    root = Path(self.config.cache.dir).expanduser() / "captures"
    serial = self._capture_serial()
    sessions: list[Any] = []
    if serial:
        with contextlib.suppress(Exception):
            sessions = read_sessions_from_disk(
                root,
                serial,
                platform=self.platform.name,
            )
    if not sessions:
        raise UsageError(
            "capture buffer is not running, and nothing was ever recorded for this device",
            hint=(
                "`aua capture on` if the daemon is warm, otherwise `aua daemon start` "
                "(capture.enabled) first — then re-run."
            ),
        )
    session = sessions[0]
    ttl_ms = max(0, int(getattr(self.config.capture, "ttl_s", 180))) * 1000
    age_ms = session.newest_frame_age_ms
    if age_ms is None or (ttl_ms and age_ms > ttl_ms):
        raise UsageError(
            "the newest capture session has no current frame evidence",
            hint="Start or resume the warm capture buffer and repeat the action.",
        )
    if not session.entries:
        raise UsageError(
            "the newest capture session has no available frames",
            hint="Its JPEGs were pruned; repeat the action with a live capture buffer.",
        )
    if not since:
        return session, None
    if session.last_action_ms is None:
        raise UsageError(
            "no last-action mark in the current capture session",
            hint="Perform a tap/input/swipe with the live buffer running, then retry.",
        )
    if not any(entry.t_ms >= session.last_action_ms for entry in session.entries):
        raise UsageError(
            "the current action has no available post-action frame",
            hint="The screen did not change or its pixels were pruned; capture the action again.",
        )
    return session, session.last_action_ms


def capture_status(self: Engine) -> dict[str, Any]:
    if self._capture is None:
        found = self._capture_from_disk()
        base: dict[str, Any] = {
            "ok": True,
            "action": "capture-status",
            # NEVER true from disk alone. A crashed daemon leaves a session directory that
            # looks live, and "running" is what a caller keys its next move off.
            "running": False,
            "paused": False,
            # Do NOT assert the daemon is down: this same answer comes back from
            # INSIDE a warm daemon whose buffer is simply off, and telling the caller
            # to start a daemon they already started sends them down a blind alley.
            "hint": (
                "no capture buffer in this process — `aua capture on` if the daemon is "
                "warm, otherwise `aua daemon start` (capture.enabled) first."
            ),
        }
        if found is None:
            return base
        return {
            **base,
            **self._disk_capture_payload(found),
            "frames": found.available,
            "last_action_ms": found.last_action_ms,
            "age_span_ms": (
                [found.entries[0].t_ms, found.entries[-1].t_ms] if found.entries else None
            ),
        }
    return self._capture.status()


def capture_last(
    self: Engine,
    *,
    seconds: float | None = None,
    since: str | None = None,
    region: str | None = None,
    where_rid: str | None = None,
) -> dict[str, Any]:
    if self._capture is None:
        # The frames are durable, so "no buffer here" is not "no frames anywhere". This
        # used to raise, which is how a caller under daemon skew ended up with nothing at
        # all: the routing layer refused the warm call, and the in-process fallback
        # refused too. Answer from the index instead, labelled as what it is.
        return self._capture_last_from_disk(
            seconds=seconds, since=since, region=region, where_rid=where_rid
        )
    since_ms: int | None = None
    if since:
        since_ms = self._capture.last_action_ms()
        if since_ms is None:
            raise UsageError(
                "no last-action mark in the live capture session",
                hint="Perform the action again; an older session cannot prove its result.",
            )
    resolved_region = region
    if where_rid and not resolved_region:
        resolved_region = self._region_for_rid(where_rid)
    result = self._capture.last(
        seconds=seconds,
        since_ms=since_ms,
        region=resolved_region,
        where_rid=where_rid,
    )
    if since_ms is not None and not result.get("count"):
        raise UsageError(
            "the current action has no live post-action frame",
            hint="The screen did not change; capture the action again if pixel evidence is required.",
        )
    return result


def _region_for_rid(self: Engine, rid: str) -> str | None:
    """Best-effort grid cell for a resource-id from the last analyze cache."""
    cached = self._read_cache()
    if cached is None:
        return None
    want = rid.strip()
    for el in cached.elements:
        if not el.resource_id:
            continue
        if (
            el.resource_id == want
            or el.resource_id.endswith("/" + want)
            or _id_tail(el.resource_id) == want
        ):
            w = cached.screen.width or 0
            h = cached.screen.height or 0
            if w <= 0 or h <= 0:
                with contextlib.suppress(Exception):
                    w, h = self.device.window_size()
            if w > 0 and h > 0:
                cx, cy = el.center
                return _region_from_point(cx, cy, w, h)
    return None


def capture_export(
    self: Engine,
    path: str,
    *,
    seconds: float | None = None,
    since: str | None = None,
    fmt: str = "gif",
    fps: float = 8.0,
) -> dict[str, Any]:
    if self._capture is None:
        # Same durability argument as ``capture_last``: the JPEGs an export assembles are
        # already files on disk, so a process without a buffer can still stitch them.
        found, disk_since = self._disk_session_for(since)
        entries = list(found.entries)
        if disk_since is not None:
            entries = [e for e in entries if e.t_ms >= disk_since]
        elif seconds is not None:
            cutoff = int(time.time() * 1000) - int(seconds * 1000)
            entries = [e for e in entries if e.t_ms >= cutoff]
        if not entries:
            raise UsageError(
                "the capture index has no available frame in the requested window",
                hint="Capture the action again with a live warm daemon.",
            )
        from .capture import export_animation

        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = export_animation(entries, out, fmt=fmt, fps=fps)
        except (ValueError, ImportError) as exc:
            raise UsageError(str(exc)) from exc
        return {
            "ok": True,
            "action": "capture-export",
            **self._disk_capture_payload(found),
            "path": written,
            "frames": len(entries),
            "format": fmt,
        }
    since_ms = None
    if since and since.lower().strip() in ("last-action", "last_action", "action"):
        since_ms = self._capture.last_action_ms()
        if since_ms is None:
            raise UsageError(
                "no last-action mark in the live capture session",
                hint="Perform the action again; an older session cannot prove its result.",
            )
    try:
        return self._capture.export(path, seconds=seconds, since_ms=since_ms, fmt=fmt, fps=fps)
    except (ValueError, ImportError) as exc:
        raise UsageError(str(exc)) from exc


def capture_sheet(
    self: Engine,
    path: str,
    *,
    seconds: float | None = None,
    since: str | None = None,
    max_frames: int = 6,
    columns: int = 3,
    timestamps: bool = True,
) -> dict[str, Any]:
    """Export a bounded visual timeline without requiring ffmpeg."""

    payload = self.capture_last(seconds=seconds, since=since)
    from .capture import FrameEntry, export_contact_sheet

    entries = [FrameEntry(**item) for item in payload.get("frames") or []]
    try:
        written, selected = export_contact_sheet(
            entries,
            Path(path),
            max_frames=max_frames,
            columns=columns,
            timestamps=timestamps,
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    provenance: dict[str, Any] = {
        key: payload[key]
        for key in (
            "source",
            "live",
            "session_id",
            "dir",
            "indexed",
            "available",
            "newest_frame_age_ms",
            "note",
        )
        if key in payload
    }
    return {
        "ok": True,
        "action": "capture-sheet",
        **provenance,
        "path": written,
        "frames": len(selected),
        "source_frames": len(entries),
        "timestamps": timestamps,
        "columns": min(columns, len(selected)),
        "selected_timestamps_ms": [entry.t_ms for entry in selected],
    }


def capture_explain(
    self: Engine,
    *,
    seconds: float | None = None,
    since: str | None = None,
    llm: bool = False,
) -> dict[str, Any]:
    """Narrate the recent capture window (local summary; optional LLM)."""
    if self._capture is None:
        # ``local_narration`` reads the payload, not the buffer, so the disk answer
        # narrates exactly as well as the live one — minus any pruned frames, which the
        # provenance keys carried through from ``capture_last`` already declare.
        payload = self.capture_last(seconds=seconds, since=since)
        out = {**payload, "action": "capture-explain"}
        from .capture import local_narration

        out["narration"] = local_narration(payload)
        if llm:
            out["llm"] = self._capture_explain_llm(out)
        return out
    since_ms = None
    if since and since.lower().strip() in ("last-action", "last_action", "action"):
        since_ms = self._capture.last_action_ms()
        if since_ms is None:
            raise UsageError(
                "no last-action mark in the live capture session",
                hint="Perform the action again; an older session cannot prove its result.",
            )
    out = self._capture.explain_local(seconds=seconds, since_ms=since_ms)
    if since_ms is not None and not out.get("count"):
        raise UsageError(
            "the current action has no live post-action frame",
            hint="The screen did not change; capture the action again if pixel evidence is required.",
        )
    if llm:
        out["llm"] = self._capture_explain_llm(out)
    return out


def _capture_explain_llm(self: Engine, payload: dict[str, Any]) -> str | None:
    """Best-effort narration via the planner chain (opt-in)."""
    try:
        if not self.factory.is_enabled("planner"):
            return "(llm skipped: planner disabled — enable planner in config or use local narration)"
        names = self.factory.chain_names("planner")
        objective = (
            "Summarize this Android UI transition for a QA agent in 2-4 sentences.\n"
            f"{payload.get('narration')}\n"
            f"Diff lines: {payload.get('summary')}"
        )
        for name in names:
            try:
                prov = self.factory.create("planner", name)
                # `create` is typed to the generic `Provider`, so `decide` was reached
                # unchecked inside a bare `except Exception: continue`. An entry that is not
                # actually a planner is now skipped by name rather than by swallowed
                # AttributeError — the same shape as a call to a method that does not exist.
                if not isinstance(prov, PlannerProvider):
                    continue
                if not prov.is_available().ok:
                    continue
                decision = prov.decide(objective, [])
                if decision is None:
                    continue
                text = getattr(decision, "reason", None) or getattr(decision, "thought", None)
                return str(text or decision)[:2000]
            except Exception:
                continue
    except Exception as exc:
        return f"(llm error: {exc})"
    return None


def capture_prune(self: Engine) -> dict[str, Any]:
    if self._capture is None:
        return {"ok": True, "action": "capture-prune", "removed": 0, "running": False}
    return self._capture.prune()


def capture_sidecar_start(self: Engine) -> dict[str, Any]:
    """Start a host-side capture sidecar (survives without the full daemon)."""
    from . import capture_sidecar as cs

    if not self.config.capture.sidecar:
        raise UsageError(
            "capture sidecar is disabled",
            hint="Set capture.sidecar: true in config.",
        )
    return cs.start(
        serial=self.device.serial,
        cache_dir=Path(self.config.cache.dir).expanduser(),
        cfg=self.config.capture,
        platform=self.platform.name,
        platform_options=self.config.platform_options(self.platform.name),
    )


def capture_sidecar_stop(self: Engine) -> dict[str, Any]:
    from . import capture_sidecar as cs

    serial = self._capture_serial()
    if not serial:
        raise UsageError(
            "capture sidecar stop needs a target id",
            hint="Pass --serial, or stop it from the cache containing one platform capture.",
        )
    return cs.stop(
        Path(self.config.cache.dir).expanduser(),
        serial=serial,
        platform=self.platform.name,
    )
