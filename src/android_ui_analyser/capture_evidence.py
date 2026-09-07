"""Host-only, bounded action evidence, independent of the rolling frame TTL.

References name an exact target and capture window. The first reader seals the recorded
frame list atomically, so a later GIF uses exactly the window shown by a contact sheet.
Hard links preserve existing JPEGs when the rolling buffer prunes them; no device is read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .atomic import atomic_create_text, atomic_write_text
from .errors import UsageError

RETENTION_SECONDS = 3600
MAX_REFERENCES = 128
_REF = re.compile(r"cap:([0-9a-f]{24}):([0-9a-f]{32})\Z")


class EvidenceStore:
    """One target's retained windows. Writers append; the first seal is immutable."""

    def __init__(self, capture_root: Path, platform: str, target: str) -> None:
        self.platform = platform
        self.target = target
        identity = json.dumps([platform, target], separators=(",", ":"))
        self.scope = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.root = capture_root.parent / "capture-evidence" / self.scope

    def _dir(self, ref: str) -> Path:
        match = _REF.fullmatch(ref)
        if match is None or match[1] != self.scope:
            raise UsageError(
                "capture evidence reference does not belong to this target",
                code="capture_evidence_not_found",
                hint="Use the exact capture_evidence.ref returned by the action on this target.",
            )
        return self.root / match[2]

    def _metadata(self, ref: str) -> dict[str, Any]:
        try:
            value = json.loads((self._dir(ref) / "window.json").read_text())
        except (OSError, ValueError) as exc:
            raise UsageError(
                "capture evidence is unavailable or was pruned",
                code="capture_evidence_not_found",
                hint="This reference never falls back to a newer action. Saved exports remain usable.",
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("ref") != ref
            or value.get("platform") != self.platform
            or value.get("target_id") != self.target
        ):
            raise UsageError("invalid capture evidence metadata", code="capture_evidence_not_found")
        if int(time.time() * 1000) >= int(value.get("expires_ms", 0)):
            raise UsageError(
                "capture evidence retention has expired", code="capture_evidence_expired"
            )
        return value

    def begin(
        self,
        action: str,
        capture_session_id: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        now = int(time.time() * 1000)
        ref = f"cap:{self.scope}:{uuid.uuid4().hex}"
        value = {
            "ref": ref,
            "platform": self.platform,
            "target_id": self.target,
            "capture_session_id": capture_session_id,
            "session_id": session_id,
            "owner": owner,
            "action": action,
            "start_ms": now,
            "created_ns": time.time_ns(),
            "expires_ms": now + RETENTION_SECONDS * 1000,
            "retention_seconds": RETENTION_SECONDS,
        }
        atomic_write_text(self._dir(ref) / "window.json", json.dumps(value))
        return value

    def append(self, ref: str, entry: dict[str, Any]) -> bool:
        directory = self._dir(ref)
        if (directory / "sealed.json").exists():
            return False
        metadata = self._metadata(ref)
        if int(entry["t_ms"]) < int(metadata["start_ms"]):
            return True  # the sample was already in flight when this window started
        source = Path(entry["path"])
        destination = directory / "frames" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except FileExistsError:
            return True
        except OSError:
            shutil.copyfile(source, destination)
        with (directory / "frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**entry, "path": str(destination)}) + "\n")
        return True

    def _entries(self, directory: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with contextlib.suppress(OSError):
            for line in (directory / "frames.jsonl").read_text().splitlines():
                with contextlib.suppress(ValueError):
                    value = json.loads(line)
                    if isinstance(value, dict):
                        entries.append(value)
        return entries

    def seal(self, ref: str) -> dict[str, Any]:
        self._metadata(ref)
        directory = self._dir(ref)
        path = directory / "sealed.json"
        if not path.exists():
            value = {"end_ms": int(time.time() * 1000), "frames": self._entries(directory)}
            with contextlib.suppress(FileExistsError):
                atomic_create_text(path, json.dumps(value))
        return self.describe(ref)

    def describe(self, ref: str) -> dict[str, Any]:
        value = self._metadata(ref)
        sealed: dict[str, Any] | None = None
        with contextlib.suppress(OSError, ValueError):
            sealed = json.loads((self._dir(ref) / "sealed.json").read_text())
        return {
            **value,
            "state": "sealed" if sealed is not None else "recording",
            "end_ms": sealed.get("end_ms") if sealed is not None else None,
            "frames": len(sealed["frames"])
            if sealed is not None
            else len(self._entries(self._dir(ref))),
            "note": "Retained recorded frames; first export fixes the window for every later export.",
        }

    def read(self, ref: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._metadata(ref)
        directory = self._dir(ref)
        # A caller arriving before the sampler must not permanently freeze an empty window.
        if not (directory / "sealed.json").exists() and not self._entries(directory):
            raise UsageError(
                "this capture window has no recorded changed frame yet",
                code="capture_evidence_empty",
                hint="Keep this reference; no additional device capture was requested.",
            )
        metadata = self.seal(ref)
        value = json.loads((directory / "sealed.json").read_text())
        entries = value["frames"]
        if not entries:
            raise UsageError(
                "this capture window contains no recorded changed frame",
                code="capture_evidence_empty",
                hint="No later action is substituted. Inspect the action observation for its final state.",
            )
        for entry in entries:
            path = Path(entry["path"])
            if path.parent != directory / "frames" or not path.is_file():
                raise UsageError(
                    "capture evidence is incomplete; a retained frame is missing",
                    code="capture_evidence_incomplete",
                )
        return metadata, entries

    def disk_bytes(self) -> int:
        total = 0
        for path in self.root.glob("*/frames/*.jpg"):
            with contextlib.suppress(OSError):
                total += path.stat().st_size
        return total

    def prune(self, max_bytes: int) -> None:
        """Drop whole windows, oldest first; never return silently truncated evidence."""
        windows: list[tuple[int, Path, int]] = []
        now = int(time.time() * 1000)
        for path in self.root.glob("*/window.json"):
            try:
                value = json.loads(path.read_text())
                expired = now >= int(value["expires_ms"])
                start = int(value.get("created_ns", value["start_ms"]))
            except (OSError, ValueError, KeyError, TypeError):
                expired, start = True, 0
            directory = path.parent
            if expired:
                shutil.rmtree(directory, ignore_errors=True)
                continue
            size = sum(p.stat().st_size for p in directory.glob("frames/*.jpg") if p.is_file())
            windows.append((start, directory, size))
        windows.sort()
        total = sum(size for _, _, size in windows)
        while windows and (total > max_bytes or len(windows) > MAX_REFERENCES):
            _, directory, size = windows.pop(0)
            shutil.rmtree(directory, ignore_errors=True)
            total -= size
