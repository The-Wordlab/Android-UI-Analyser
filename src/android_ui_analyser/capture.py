"""Rolling session capture buffer — adaptive screencap ring with dedupe + diff summary.

Always-on while the warm daemon holds an Engine. Frames live under
``cache.dir/captures/<serial>/<session_id>/``. Identical pixels are dropped; after a
marked action the sampler bursts briefly so sub-second loading flashes survive for
``aua capture last``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ScreenshotFn = Callable[[], Any]  # returns ScreenImage-like with .png_bytes / .pil()

_BASE_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


@dataclass
class CaptureCfgView:
    """Plain settings the buffer reads (avoids importing Config here)."""

    enabled: bool = True
    idle_fps: float = 2.0
    burst_fps: float = 10.0
    burst_ms: int = 1500
    extend_burst_on_change: bool = True  # keep bursting while pixels keep changing
    ttl_s: int = 180
    max_mb: int = 200
    jpeg_quality: int = 70
    hint: bool = True


@dataclass
class FrameEntry:
    t_ms: int
    path: str
    hash: str
    bytes: int
    w: int
    h: int
    action: str | None = None


def _set_event() -> threading.Event:
    """A ``threading.Event`` that starts set — "nothing in flight" is the resting state."""

    event = threading.Event()
    event.set()
    return event


@dataclass
class CaptureBuffer:
    """Host-side ring of JPEG frames + timeline marks."""

    root: Path
    serial: str
    cfg: CaptureCfgView
    screenshot: ScreenshotFn
    session_id: str = field(
        default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    # Set whenever the sampling thread is *not* inside a frame grab. This is what makes
    # ``pause(settle_s=...)`` a handshake rather than a hint — see :meth:`pause`.
    _idle: threading.Event = field(default_factory=_set_event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _paused: bool = False
    _pause_reason: str | None = None
    _last_hash: str | None = None
    _entries: list[FrameEntry] = field(default_factory=list)
    _burst_until: float = 0.0
    _last_action_ms: int | None = None
    _kept_since_action: int = 0
    _pending_action: str | None = None
    _seq: int = 0
    _latest_img: Any = None
    _latest_at: float = 0.0

    @property
    def dir(self) -> Path:
        safe = str(self.serial).replace(":", "_").replace("/", "_")
        return self.root / safe / self.session_id

    @property
    def serial_root(self) -> Path:
        """All sessions for this device, live and dead."""
        safe = str(self.serial).replace(":", "_").replace("/", "_")
        return self.root / safe

    @property
    def index_path(self) -> Path:
        return self.dir / "index.jsonl"

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        # Sessions from earlier runs have no owner to prune them, so the aggregate only
        # stays bounded if each new session clears up after the dead ones.
        with contextlib.suppress(Exception):
            self.sweep_sessions()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused = False
        self._pause_reason = None
        self._thread = threading.Thread(target=self._loop, name="aua-capture", daemon=True)
        self._thread.start()
        logger.info("capture started session=%s dir=%s", self.session_id, self.dir)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        # A session that captured nothing (or deduped everything) must not leave a directory
        # behind for the next sweep to find.
        with contextlib.suppress(OSError):
            index_has_evidence = self.index_path.is_file() and self.index_path.stat().st_size > 0
            if (
                self.dir.is_dir()
                and not any((self.dir / "frames").glob("*.jpg"))
                and not index_has_evidence
            ):
                shutil.rmtree(self.dir, ignore_errors=True)

    def pause(self, reason: str = "manual", *, settle_s: float = 0.0) -> bool:
        """Stop sampling. With *settle_s*, wait for any frame already in flight to finish.

        Returns whether the sampling thread is genuinely idle — always True without
        *settle_s*, because then it is not being asked.

        The waiting form exists for the on-device helper handover, and it is the difference
        between that working and not. Setting the flag only tells the loop to stop *next*
        time round; a thread already past the check goes on to grab one more frame, and that
        frame reconnects uiautomator2, which takes back the UiAutomation slot the helper was
        just handed and tears its accessibility service down mid-run. One stale frame is
        enough, and it showed up as the run stopping at a different step every time.

        The flag is set under the same lock the loop uses to decide, so the two cannot
        interleave: either the loop sees the pause and parks, or it has already claimed the
        tick and this waits for it.
        """

        with self._lock:
            self._paused = True
            self._pause_reason = reason
        if settle_s <= 0:
            return True
        return self._idle.wait(settle_s)

    def resume(self, *, only_if_idle: bool = False) -> None:
        """Resume sampling. With *only_if_idle*, an explicit ``capture off`` stays off."""
        with self._lock:
            if only_if_idle and self._pause_reason != "idle":
                return
            self._paused = False
            self._pause_reason = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def paused(self) -> bool:
        return self._paused

    def mark(self, action: str) -> None:
        """Durably stamp *action* now, then label the next changed frame too."""
        now = time.time()
        with self._lock:
            self._pending_action = action
            self._burst_until = now + max(0, self.cfg.burst_ms) / 1000.0
            self._last_action_ms = int(now * 1000)
            self._kept_since_action = 0
            # An unchanged screen produces no new JPEG. Persisting the action only on the next
            # changed frame therefore lost the mark entirely, and two quick actions collapsed
            # into the last label. The marker is its own append-only record for exactly that
            # reason; frame rows remain backwards compatible.
            self.dir.mkdir(parents=True, exist_ok=True)
            self._append_record(
                {"kind": "action", "t_ms": self._last_action_ms, "action": action}
            )

    def hint_ready(self) -> bool:
        """True when a post-action burst kept at least one non-deduped frame."""
        with self._lock:
            return bool(self.cfg.hint and self._kept_since_action > 0)

    def last_action_ms(self) -> int | None:
        with self._lock:
            return self._last_action_ms

    def status(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
            burst = time.time() < self._burst_until
            last_action = self._last_action_ms
            kept_since = self._kept_since_action
            paused = self._paused
            pause_reason = self._pause_reason
        disk = self._disk_bytes()
        age_span: list[int] | None = None
        if entries:
            age_span = [entries[0].t_ms, entries[-1].t_ms]
        return {
            "ok": True,
            "action": "capture-status",
            "running": self.running,
            "paused": paused,
            "pause_reason": pause_reason,
            "mode": "burst" if burst else "idle",
            "session_id": self.session_id,
            "dir": str(self.dir),
            "frames": len(entries),
            "age_span_ms": age_span,
            "disk_bytes": disk,
            "total_disk_bytes": self.total_disk_bytes(),
            "last_action_ms": last_action,
            "kept_since_action": kept_since,
            "idle_fps": self.cfg.idle_fps,
            "burst_fps": self.cfg.burst_fps,
        }

    def last(
        self,
        *,
        seconds: float | None = None,
        since_ms: int | None = None,
        region: str | None = None,
        where_rid: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
        if since_ms is not None:
            entries = [e for e in entries if e.t_ms >= since_ms]
        elif seconds is not None:
            cutoff = int(time.time() * 1000) - int(seconds * 1000)
            entries = [e for e in entries if e.t_ms >= cutoff]
        summary = diff_summary(entries, region=region)
        duration = change_duration_ms(entries)
        return {
            "ok": True,
            "action": "capture-last",
            "session_id": self.session_id,
            "dir": str(self.dir),
            "frames": [e.__dict__ for e in entries],
            "count": len(entries),
            "summary": summary,
            "change_duration_ms": duration,
            "region": region,
            "where_rid": where_rid,
        }

    def export(
        self,
        path: str | Path,
        *,
        seconds: float | None = None,
        since_ms: int | None = None,
        fmt: str = "gif",
        fps: float = 8.0,
    ) -> dict[str, Any]:
        """Assemble kept frames into a GIF (or MJPEG-style multipage PDF fallback)."""
        payload = self.last(seconds=seconds, since_ms=since_ms)
        entries = [FrameEntry(**f) if isinstance(f, dict) else f for f in payload["frames"]]
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        written = export_animation(entries, out, fmt=fmt, fps=fps)
        return {
            "ok": True,
            "action": "capture-export",
            "path": written,
            "frames": len(entries),
            "format": fmt,
        }

    def explain_local(
        self,
        *,
        seconds: float | None = None,
        since_ms: int | None = None,
    ) -> dict[str, Any]:
        """Cheap text narration from marks + diff summary (no LLM)."""
        payload = self.last(seconds=seconds, since_ms=since_ms)
        narration = local_narration(payload)
        return {**payload, "action": "capture-explain", "narration": narration}

    def latest_frame(self) -> Any | None:
        """Most recent ScreenImage from the sampler (may be identical to a prior hash)."""
        with self._lock:
            return self._latest_img

    def latest_age_ms(self) -> float | None:
        with self._lock:
            if self._latest_img is None or self._latest_at <= 0:
                return None
            return (time.monotonic() - self._latest_at) * 1000.0

    def prune(self) -> dict[str, Any]:
        removed = self._prune()
        return {"ok": True, "action": "capture-prune", "removed": removed, **self.status()}

    # -- sampler -----------------------------------------------------------

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            # Claiming the tick under the lock is what lets ``pause(settle_s=...)`` be a
            # handshake: a pause either lands before the claim and parks the loop, or lands
            # after it and waits on ``_idle`` for this frame to finish.
            with self._lock:
                parked = self._paused
                if not parked:
                    self._idle.clear()
            if parked:
                self._idle.set()
                self._stop.wait(0.2)
                continue
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — never kill the daemon for a bad frame
                failures += 1
                logger.debug("capture tick failed (%d in a row)", failures, exc_info=True)
            else:
                failures = 0
            finally:
                self._idle.set()
            if failures:
                # A detached device fails every tick. Backing off keeps a dead daemon from
                # spending the device's CPU (and the log's disk) at full sampling rate.
                self._stop.wait(min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** (failures - 1))))
                continue
            with self._lock:
                bursting = time.time() < self._burst_until
            fps = self.cfg.burst_fps if bursting else self.cfg.idle_fps
            fps = max(0.5, float(fps or 1.0))
            delay = max(0.0, (1.0 / fps) - (time.perf_counter() - t0))
            self._stop.wait(delay)

    def _tick(self) -> None:
        img = self.screenshot()
        png = getattr(img, "png_bytes", None)
        if not png:
            return
        now_mono = time.monotonic()
        with self._lock:
            self._latest_img = img
            self._latest_at = now_mono
        gray, w, h = _downscale_gray(img)
        digest = frame_hash(gray)
        now_ms = int(time.time() * 1000)
        with self._lock:
            if digest == self._last_hash:
                return
            action = self._pending_action
            self._pending_action = None
            self._seq += 1
            name = f"{self._seq:06d}.jpg"
            path = self.dir / "frames" / name
            size = _write_jpeg(img, path, quality=self.cfg.jpeg_quality)
            entry = FrameEntry(
                t_ms=now_ms,
                path=str(path),
                hash=digest,
                bytes=size,
                w=w,
                h=h,
                action=action,
            )
            self._entries.append(entry)
            self._last_hash = digest
            if (
                self._last_action_ms is not None
                and now_ms >= self._last_action_ms
                and (time.time() < self._burst_until or action)
            ):
                self._kept_since_action += 1
            # Animation-aware: while frames keep changing during a burst, keep it open.
            if self.cfg.extend_burst_on_change and (
                action or time.time() < self._burst_until
            ):
                self._burst_until = time.time() + max(0, self.cfg.burst_ms) / 1000.0
            self._append_index(entry)
        self._prune()

    def _append_index(self, entry: FrameEntry) -> None:
        self._append_record(entry.__dict__)

    def _append_record(self, record: dict[str, Any]) -> None:
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _disk_bytes(self) -> int:
        total = 0
        frames = self.dir / "frames"
        if not frames.is_dir():
            return 0
        for p in frames.glob("*.jpg"):
            with contextlib.suppress(OSError):
                total += p.stat().st_size
        return total

    def _dir_bytes(self, session: Path) -> int:
        total = 0
        for p in (session / "frames").glob("*.jpg"):
            with contextlib.suppress(OSError):
                total += p.stat().st_size
        return total

    def total_disk_bytes(self) -> int:
        """Every session for this device — what the tool is ACTUALLY consuming.

        ``_disk_bytes`` covers the live session only, which under-reports badly once dead
        sessions accumulate: it read 114 kB while 11 MB sat on disk.
        """
        if not self.serial_root.is_dir():
            return 0
        return sum(self._dir_bytes(d) for d in self.serial_root.iterdir() if d.is_dir())

    def sweep_sessions(self) -> dict[str, int]:
        """Prune DEAD sessions left by earlier runs. Returns what it removed.

        The per-session TTL/size caps are enforced by the thread that owns the session, so
        when a daemon stops, whatever was still inside the TTL window is orphaned: no process
        owns those files any more, and nothing ever prunes them. Every restart then mints
        another session directory, so the aggregate grew without bound — 9 sessions and 11 MB
        in one afternoon, the oldest 95 minutes past a 3-minute TTL.

        Bound it here instead: drop a dead session once its newest frame is older than the
        TTL, then, oldest-first, drop whole dead sessions until the total fits ``max_mb``.
        The live session is never touched — ``_prune`` owns it.
        """
        removed_sessions = 0
        removed_bytes = 0
        if not self.serial_root.is_dir():
            return {"sessions": 0, "bytes": 0}

        cutoff = time.time() - float(self.cfg.ttl_s)
        dead: list[tuple[float, Path, int]] = []
        for session in sorted(self.serial_root.iterdir()):
            if not session.is_dir() or session.name == self.session_id:
                continue
            frames = sorted((session / "frames").glob("*.jpg"))
            newest = 0.0
            for p in frames:
                with contextlib.suppress(OSError):
                    newest = max(newest, p.stat().st_mtime)
            size = self._dir_bytes(session)
            # An action marker is useful even without a changed frame: it lets a reader say
            # "no post-action evidence" rather than reuse an older screenshot. Keep a recent
            # marker-only index until the same TTL as pixels.
            with contextlib.suppress(OSError):
                newest = max(newest, (session / "index.jsonl").stat().st_mtime)
            if newest <= 0 or newest < cutoff:
                removed_bytes += size
                removed_sessions += 1
                shutil.rmtree(session, ignore_errors=True)
            else:
                dead.append((newest, session, size))

        max_bytes = max(1, int(self.cfg.max_mb)) * 1024 * 1024
        total = self._dir_bytes(self.dir) + sum(size for _, _, size in dead)
        dead.sort()  # oldest first
        for _, session, size in dead:
            if total <= max_bytes:
                break
            shutil.rmtree(session, ignore_errors=True)
            total -= size
            removed_bytes += size
            removed_sessions += 1
        if removed_sessions:
            logger.info(
                "capture swept %d dead session(s), %d bytes", removed_sessions, removed_bytes
            )
        return {"sessions": removed_sessions, "bytes": removed_bytes}

    def _prune(self) -> int:
        """Drop frames older than TTL or over max_mb. Returns count removed."""
        cutoff = int(time.time() * 1000) - int(self.cfg.ttl_s) * 1000
        max_bytes = max(1, int(self.cfg.max_mb)) * 1024 * 1024
        with self._lock:
            entries = list(self._entries)
        removed = 0
        keep: list[FrameEntry] = []
        for e in entries:
            if e.t_ms < cutoff:
                _unlink(e.path)
                removed += 1
            else:
                keep.append(e)
        total = sum(e.bytes for e in keep)
        while keep and total > max_bytes:
            old = keep.pop(0)
            _unlink(old.path)
            total -= old.bytes
            removed += 1
        with self._lock:
            self._entries = keep
        return removed


# --------------------------------------------------------------------------- pure helpers


def frame_hash(gray_small: Any) -> str:
    """Stable hash of a small grayscale array."""
    import numpy as np

    arr = np.asarray(gray_small, dtype=np.uint8)
    return hashlib.sha1(arr.tobytes(), usedforsecurity=False).hexdigest()[:16]


def _downscale_gray(img: Any, size: int = 64) -> tuple[Any, int, int]:
    import numpy as np
    from PIL import Image

    if hasattr(img, "pil"):
        pil = img.pil()
    elif isinstance(img, Image.Image):
        pil = img
    else:
        raw = img.png_bytes if hasattr(img, "png_bytes") else img
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = pil.size
    small = pil.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.uint8), w, h


def _write_jpeg(img: Any, path: Path, *, quality: int = 70) -> int:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = img.pil() if hasattr(img, "pil") else Image.open(io.BytesIO(img.png_bytes)).convert("RGB")
    pil.save(path, format="JPEG", quality=quality, optimize=True)
    return path.stat().st_size


def diff_summary(
    entries: list[FrameEntry],
    *,
    grid: int = 3,
    threshold: float = 8.0,
    region: str | None = None,
) -> list[str]:
    """Cheap local summary of where consecutive kept frames differ."""
    if len(entries) < 2:
        return [] if not entries else [f"t={entries[0].t_ms}: single frame (no diff)"]

    lines: list[str] = []
    t0 = entries[0].t_ms
    i = 0
    want = (region or "").lower().strip() or None
    while i < len(entries) - 1:
        j = i + 1
        regions: set[str] = set()
        end = entries[j]
        while j < len(entries):
            end = entries[j]
            cells = _changed_cells(
                entries[i].path if j == i + 1 else entries[j - 1].path,
                end.path,
                grid=grid,
                threshold=threshold,
            )
            if not cells:
                break
            regions.update(cells)
            j += 1
            if j - i > 30:
                break
        if want:
            regions = {r for r in regions if want in r}
        if regions:
            rel0 = entries[i].t_ms - t0
            rel1 = end.t_ms - t0
            label = "+".join(sorted(regions))
            action = end.action or entries[i].action
            suffix = f" after {action}" if action else ""
            hint = " (loading/transition?)" if "center" in regions or len(regions) >= 2 else ""
            lines.append(f"t+{rel0}–t+{rel1}ms: {label} changed{hint}{suffix}")
            i = max(j, i + 1)
        else:
            i += 1
    if not lines and len(entries) >= 2:
        if want:
            lines.append(
                f"t+0–t+{entries[-1].t_ms - t0}ms: no '{want}' cell change "
                f"across {len(entries)} kept frames"
            )
        else:
            lines.append(
                f"t+0–t+{entries[-1].t_ms - t0}ms: {len(entries)} frames kept (subtle/no grid change)"
            )
    return lines


def change_duration_ms(entries: list[FrameEntry]) -> int | None:
    """Wall ms from first → last kept frame in the window (loading-flash duration)."""
    if len(entries) < 2:
        return None
    return max(0, entries[-1].t_ms - entries[0].t_ms)


def local_narration(payload: dict[str, Any]) -> str:
    """Turn marks + summary into a short agent-readable paragraph."""
    frames = payload.get("frames") or []
    summary = payload.get("summary") or []
    duration = payload.get("change_duration_ms")
    # `str(...)` inside the comprehension, not a cast afterwards: the filter already drops the
    # falsy entries, but only at runtime — the list still typed as possibly-None keys, which
    # `dict.fromkeys` refuses.
    marks = [str(f["action"]) for f in frames if isinstance(f, dict) and f.get("action")]
    parts: list[str] = []
    if marks:
        parts.append("Actions: " + " → ".join(dict.fromkeys(marks)) + ".")
    if duration is not None:
        parts.append(f"Visible change spanned ~{duration}ms across {len(frames)} kept frames.")
    elif frames:
        parts.append(f"{len(frames)} kept frame(s).")
    if summary:
        parts.append("Diff: " + "; ".join(summary) + ".")
    else:
        parts.append("No coarse-grid pixel change detected between kept frames.")
    return " ".join(parts)


def export_animation(
    entries: list[FrameEntry],
    path: Path,
    *,
    fmt: str = "gif",
    fps: float = 8.0,
) -> str:
    """Write a GIF (Pillow) from frame paths. ``fmt=mp4`` requires imageio+ffmpeg."""
    from PIL import Image

    if not entries:
        raise ValueError("no frames to export")
    fmt_l = (fmt or "gif").lower().lstrip(".")
    images = []
    for e in entries:
        try:
            images.append(Image.open(e.path).convert("RGB"))
        except OSError:
            continue
    if not images:
        raise ValueError("could not open any frame images")
    duration_ms = max(20, int(1000 / max(0.5, fps)))
    if fmt_l == "gif":
        out = path if path.suffix.lower() == ".gif" else path.with_suffix(".gif")
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        return str(out)
    if fmt_l in ("mp4", "video"):
        out = path if path.suffix.lower() == ".mp4" else path.with_suffix(".mp4")
        try:
            import imageio.v2 as imageio
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "mp4 export needs imageio (and typically ffmpeg). "
                "Use fmt=gif, or `pip install imageio imageio-ffmpeg`."
            ) from exc
        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7)
        try:
            for im in images:
                writer.append_data(np.asarray(im))
        finally:
            writer.close()
        return str(out)
    raise ValueError(f"unsupported export format {fmt!r} (use gif or mp4)")


def _changed_cells(path_a: str, path_b: str, *, grid: int, threshold: float) -> list[str]:
    import numpy as np
    from PIL import Image

    try:
        a = Image.open(path_a).convert("L").resize((grid * 16, grid * 16), Image.Resampling.BILINEAR)
        b = Image.open(path_b).convert("L").resize((grid * 16, grid * 16), Image.Resampling.BILINEAR)
    except OSError:
        return []
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    names = (
        ["upper-left", "upper", "upper-right"],
        ["left", "center", "right"],
        ["lower-left", "lower", "lower-right"],
    )
    cell_h = aa.shape[0] // grid
    cell_w = aa.shape[1] // grid
    out: list[str] = []
    for gy in range(grid):
        for gx in range(grid):
            block_a = aa[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            block_b = bb[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            mad = float(np.mean(np.abs(block_a - block_b)))
            if mad >= threshold:
                out.append(names[gy][gx] if grid == 3 else f"r{gy}c{gx}")
    return out


# --------------------------------------------------------------------------- disk reader


@dataclass
class DiskSession:
    """One capture session recovered from disk, by a process that never owned the buffer.

    ``CaptureBuffer`` is a *disk* ring, not an in-memory one: every kept frame is appended to
    ``index.jsonl`` — timestamp, path, hash, size, dimensions and the action mark — before the
    call that produced it returns. Only ``_entries`` and ``_last_action_ms`` are process-local,
    and both are reconstructed here. That matters because the daemon holding the buffer is
    exactly the process a version-skew restart replaces, and a capture read used to be refused
    outright on the grounds that its frames would die with it. They do not.

    ``indexed`` and ``available`` are reported separately on purpose. ``_prune`` deletes JPEGs
    from a live session while the index only ever grows, so a session can hold 48 records and
    one surviving image. Returning entries that point at deleted files would break every reader
    that opens them; dropping them silently would make a gutted session look whole.
    """

    session_id: str
    dir: Path
    entries: list[FrameEntry]
    indexed: int
    available: int
    last_action_ms: int | None
    newest_frame_ms: int | None

    @property
    def newest_frame_age_ms(self) -> int | None:
        """How stale this is. A crashed daemon leaves a session dir that looks live."""
        if self.newest_frame_ms is None:
            return None
        return max(0, int(time.time() * 1000) - self.newest_frame_ms)


def _serial_dir_name(serial: str) -> str:
    """Match :attr:`CaptureBuffer.dir`'s sanitisation, so the reader looks in the right place."""
    return str(serial).replace(":", "_").replace("/", "_")


def _read_index(session: Path) -> DiskSession | None:
    """Parse one session's append-only index. A truncated final line is tolerated.

    The index is appended to by a live writer, so the last record can be half-written at the
    moment it is read. Skipping an unparseable line loses at most the newest frame; refusing
    the whole file would lose the session.
    """
    index = session / "index.jsonl"
    try:
        raw = index.read_text(encoding="utf-8")
    except OSError:
        return None
    indexed = 0
    entries: list[FrameEntry] = []
    action_marks: list[int] = []
    newest_indexed_frame_ms: int | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("kind") == "action":
            mark_ms = record.get("t_ms")
            if isinstance(mark_ms, int) and record.get("action"):
                action_marks.append(mark_ms)
            continue
        indexed += 1
        try:
            entry = FrameEntry(**record)
        except TypeError:
            continue
        newest_indexed_frame_ms = max(newest_indexed_frame_ms or entry.t_ms, entry.t_ms)
        if entry.action:
            action_marks.append(entry.t_ms)  # legacy frame-attached marks
        if Path(entry.path).exists():
            entries.append(entry)
    if not indexed and not action_marks:
        return None
    entries.sort(key=lambda e: e.t_ms)
    return DiskSession(
        session_id=session.name,
        dir=session,
        entries=entries,
        indexed=indexed,
        available=len(entries),
        last_action_ms=max(action_marks) if action_marks else None,
        # Staleness is about what was recorded, not which JPEGs survived pruning. Otherwise a
        # gutted index claims it has no age and an old surviving file can look current.
        newest_frame_ms=newest_indexed_frame_ms,
    )


def read_sessions_from_disk(root: Path | str, serial: str) -> list[DiskSession]:
    """Every readable session for *serial*, newest index first.

    Ordered by index mtime rather than directory name because two writers — the daemon's
    always-on buffer and the capture sidecar — can both own sessions for one device, and a
    tie broken by name would silently pick the wrong one.
    """
    serial_root = Path(root) / _serial_dir_name(serial)
    if not serial_root.is_dir():
        return []
    dated: list[tuple[float, DiskSession]] = []
    try:
        candidates = sorted(serial_root.iterdir())
    except OSError:
        return []
    for session in candidates:
        if not session.is_dir():
            continue
        found = _read_index(session)
        if found is None:
            continue
        mtime = 0.0
        with contextlib.suppress(OSError):
            mtime = (session / "index.jsonl").stat().st_mtime
        dated.append((mtime, found))
    dated.sort(key=lambda pair: (pair[0], pair[1].session_id), reverse=True)
    return [found for _, found in dated]


def read_session_from_disk(root: Path | str, serial: str) -> DiskSession | None:
    """The newest readable session for *serial*, or ``None`` if nothing was ever recorded."""
    sessions = read_sessions_from_disk(root, serial)
    return sessions[0] if sessions else None


def _unlink(path: str) -> None:
    with contextlib.suppress(OSError):
        Path(path).unlink(missing_ok=True)


__all__ = [
    "CaptureBuffer",
    "CaptureCfgView",
    "DiskSession",
    "FrameEntry",
    "change_duration_ms",
    "diff_summary",
    "export_animation",
    "frame_hash",
    "local_narration",
    "read_session_from_disk",
    "read_sessions_from_disk",
]
