"""Bounded recall of host-approved evidence archives, never a current UI read."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from experiments.aua_controller.hosted_projection import hosted_model_view

MAX_RECORD_BYTES = 100_000
_REFERENCE = re.compile(r"(?:(?P<namespace>[A-Za-z0-9][A-Za-z0-9_.-]{0,127})/)?(?P<record>E[0-9]{4})")


def _error(code: str) -> dict:
    return {"ok": False, "historical_only": True, "error": {"code": code}}


def _invalid_constant(value: str):
    raise ValueError("non-JSON numeric constant")


def read_recorded_evidence(reference: str, allowed_roots: dict[str, Path]) -> dict:
    """Read E#### or an explicitly mapped namespace/E#### from local evidence.

    The caller owns the root mapping and any additional application redaction.
    Source hashes identify the bytes read; they do not establish a QA verdict or
    freshness. The historical_record wrapper deliberately cannot resolve as a
    current observation through SessionState's known-wrapper resolver.
    """
    match = _REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
    if match is None or not isinstance(allowed_roots, dict):
        return _error("invalid_evidence_reference")
    namespace = match.group("namespace") or ""
    if namespace not in allowed_roots:
        return _error("unknown_evidence_namespace")
    try:
        root = Path(allowed_roots[namespace]).resolve(strict=True)
        source = (root / "evidence" / (match.group("record") + ".json")).resolve(strict=True)
        if not source.is_relative_to(root):
            return _error("evidence_unavailable")
        # A special file must not block a bounded archive read. Refuse a final
        # symlink replacement after resolution; ordinary in-root links resolve.
        descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return _error("evidence_unavailable")
            if info.st_size > MAX_RECORD_BYTES:
                return _error("evidence_too_large")
            data = stream.read(MAX_RECORD_BYTES + 1)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _error("evidence_unavailable")
    if len(data) > MAX_RECORD_BYTES:
        return _error("evidence_too_large")
    try:
        record = json.loads(data.decode("utf-8"), parse_constant=_invalid_constant)
        if not isinstance(record, dict):
            return _error("invalid_evidence_record")
        projected = hosted_model_view(record)
    except (ValueError, UnicodeError, RecursionError):
        return _error("invalid_evidence_record")
    return {"ok": True, "historical_only": True, "source_evidence_ref": reference,
            "source_sha256": hashlib.sha256(data).hexdigest(), "historical_record": projected}
