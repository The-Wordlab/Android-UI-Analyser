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
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ScreenshotFn = Callable[[], Any]  # returns ScreenImage-like with .png_bytes / .pil()


@dataclass
class CaptureCfgView:
    """Plain settings the buffer reads (avoids importing Config here)."""

    enabled: bool = True
    idle_fps: float = 2.0
    burst_fps: float = 10.0
    burst_ms: int = 1500
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
    _thread: threading.Thread | None = field(default=None, repr=False)
    _paused: bool = False
    _last_hash: str | None = None
    _entries: list[FrameEntry] = field(default_factory=list)
    _burst_until: float = 0.0
    _last_action_ms: int | None = None
    _kept_since_action: int = 0
    _pending_action: str | None = None
    _seq: int = 0

    @property
    def dir(self) -> Path:
        safe = str(self.serial).replace(":", "_").replace("/", "_")
        return self.root / safe / self.session_id

    @property
    def index_path(self) -> Path:
        return self.dir / "index.jsonl"

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused = False
        self._thread = threading.Thread(target=self._loop, name="aua-capture", daemon=True)
        self._thread.start()
        logger.info("capture started session=%s dir=%s", self.session_id, self.dir)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def paused(self) -> bool:
        return self._paused

    def mark(self, action: str) -> None:
        """Stamp the next kept frame with *action* and enter burst mode."""
        now = time.time()
        with self._lock:
            self._pending_action = action
            self._burst_until = now + max(0, self.cfg.burst_ms) / 1000.0
            self._last_action_ms = int(now * 1000)
            self._kept_since_action = 0

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
        disk = self._disk_bytes()
        age_span: list[int] | None = None
        if entries:
            age_span = [entries[0].t_ms, entries[-1].t_ms]
        return {
            "ok": True,
            "action": "capture-status",
            "running": self.running,
            "paused": paused,
            "mode": "burst" if burst else "idle",
            "session_id": self.session_id,
            "dir": str(self.dir),
            "frames": len(entries),
            "age_span_ms": age_span,
            "disk_bytes": disk,
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
    ) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
        if since_ms is not None:
            entries = [e for e in entries if e.t_ms >= since_ms]
        elif seconds is not None:
            cutoff = int(time.time() * 1000) - int(seconds * 1000)
            entries = [e for e in entries if e.t_ms >= cutoff]
        summary = diff_summary(entries)
        return {
            "ok": True,
            "action": "capture-last",
            "session_id": self.session_id,
            "dir": str(self.dir),
            "frames": [e.__dict__ for e in entries],
            "count": len(entries),
            "summary": summary,
        }

    def prune(self) -> dict[str, Any]:
        removed = self._prune()
        return {"ok": True, "action": "capture-prune", "removed": removed, **self.status()}

    # -- sampler -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._paused:
                self._stop.wait(0.2)
                continue
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — never kill the daemon for a bad frame
                logger.debug("capture tick failed", exc_info=True)
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
            self._append_index(entry)
        self._prune()

    def _append_index(self, entry: FrameEntry) -> None:
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")

    def _disk_bytes(self) -> int:
        total = 0
        frames = self.dir / "frames"
        if not frames.is_dir():
            return 0
        for p in frames.glob("*.jpg"):
            with contextlib.suppress(OSError):
                total += p.stat().st_size
        return total

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


def diff_summary(entries: list[FrameEntry], *, grid: int = 3, threshold: float = 8.0) -> list[str]:
    """Cheap local summary of where consecutive kept frames differ."""
    if len(entries) < 2:
        return [] if not entries else [f"t={entries[0].t_ms}: single frame (no diff)"]

    lines: list[str] = []
    t0 = entries[0].t_ms
    i = 0
    while i < len(entries) - 1:
        start = entries[i]
        j = i + 1
        regions: set[str] = set()
        end = entries[j]
        while j < len(entries):
            end = entries[j]
            cells = _changed_cells(start.path if j == i + 1 else entries[j - 1].path, end.path, grid=grid, threshold=threshold)
            if not cells:
                break
            regions.update(cells)
            j += 1
            if j - i > 30:
                break
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
        lines.append(
            f"t+0–t+{entries[-1].t_ms - t0}ms: {len(entries)} frames kept (subtle/no grid change)"
        )
    return lines


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


def _unlink(path: str) -> None:
    with contextlib.suppress(OSError):
        Path(path).unlink(missing_ok=True)


__all__ = [
    "CaptureBuffer",
    "CaptureCfgView",
    "FrameEntry",
    "diff_summary",
    "frame_hash",
]
