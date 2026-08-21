"""Persistent evidence bundles for multi-command goal sessions.

Unlike a flow, a goal session is driven by several short-lived CLI/MCP processes.  This store is
therefore deliberately stateless: every operation re-opens the manifest under an inter-process
lock, deduplicates by invocation id, and writes replacements atomically.  Device capture remains
an injected platform-adapter callback.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text

_EVIDENCE_MODES = frozenset({"none", "failures", "all"})


def validate_session_evidence_mode(value: str) -> str:
    normalized = str(value or "failures").strip().lower()
    if normalized not in _EVIDENCE_MODES:
        raise ValueError(
            f"evidence must be one of {', '.join(sorted(_EVIDENCE_MODES))}, got {value!r}"
        )
    return normalized


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")[:96] or "frame"


def _redact(value: Any, *, key: str | None = None) -> Any:
    lowered = (key or "").lower()
    if any(token in lowered for token in ("password", "secret", "token", "authorization")):
        return "<redacted>"
    if lowered in {"params", "parameters", "sql", "speech", "wav_path"}:
        return "<redacted>"
    if lowered == "app_logs" and isinstance(value, dict):
        # An action's folded log window is live guidance for the agent, not evidence worth
        # archiving. These bundles get published as review evidence, and raw device log lines are
        # the likeliest place for a bearer token, an install id, or an unreleased flag name to
        # ride along. So the record keeps the fact that the app spoke — which is what a reader
        # of the archive needs — and withholds what it said.
        summary: dict[str, Any] = {"withheld": "app_logs not archived"}
        for field in ("count", "total_count", "omitted", "truncated", "levels", "since"):
            if field in value:
                summary[field] = value[field]
        return summary
    if isinstance(value, dict):
        out = {str(k): _redact(v, key=str(k)) for k, v in value.items()}
        if value.get("kind") == "input" and "text" in out:
            out["text"] = "<redacted>"
        return out
    if isinstance(value, list):
        return [_redact(item, key=key) for item in value]
    return value


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            value = value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {}


def _observation(value: Any) -> dict[str, Any] | None:
    data = _payload(value)
    nested = data.get("observation")
    if isinstance(nested, dict):
        return nested
    if isinstance(data.get("screen"), dict) and isinstance(data.get("elements"), list):
        return data
    return None


def observation_evidence_id(session_id: str, observation: dict[str, Any]) -> str:
    raw_meta = observation.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    fingerprint = str(meta.get("fingerprint") or "")
    if not fingerprint:
        canonical = json.dumps(
            {
                "screen": observation.get("screen"),
                "elements": observation.get("elements"),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        import hashlib

        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    return f"session-{session_id}:observation:{_safe(fingerprint)[:24]}"


@contextlib.contextmanager
def _locked(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".lock").open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - best effort off Unix
            pass
        yield
    finally:
        with contextlib.suppress(Exception):
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class SessionArtifactStore:
    """Append and finalize one portable session bundle."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @classmethod
    def create(
        cls,
        requested_dir: str | Path,
        *,
        session_id: str,
        goal: str,
        evidence: str,
        junit: bool,
        contract_yaml: str | None,
    ) -> SessionArtifactStore:
        mode = validate_session_evidence_mode(evidence)
        requested = Path(requested_dir).expanduser()
        requested.mkdir(parents=True, exist_ok=True)
        run_id = f"session-{uuid.uuid4()}"
        root = requested / run_id if any(requested.iterdir()) else requested
        root.mkdir(parents=True, exist_ok=True)
        store = cls(root)
        created = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "session_id": session_id,
            "goal": goal,
            "evidence": mode,
            "junit": bool(junit),
            "started_at": created,
            "finished_at": None,
            "verdict": "active",
            "entries": [],
            "invocations": [],
            "capture_errors": [],
        }
        store._write_json("manifest.json", manifest)
        store._write_json("goal.json", {"goal": goal, "session_id": session_id})
        if contract_yaml is not None:
            atomic_write_text(root / "contract.yaml", contract_yaml)
        atomic_write_text(root / "calls.jsonl", "")
        return store

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def _write_json(self, relative: str, value: Any) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path,
            json.dumps(_redact(value), indent=2, sort_keys=True, ensure_ascii=False, default=str)
            + "\n",
        )
        return str(path)

    def record(
        self,
        *,
        command: str,
        result: Any,
        invocation_id: str,
        duration_ms: float | None,
        args: dict[str, Any] | None = None,
        screenshot: Callable[[Path], str] | None = None,
        diagnostics: Callable[[], str | None] | None = None,
    ) -> dict[str, Any] | None:
        """Record one caller-visible result and return its observation contract."""

        data = _payload(result)
        observation = _observation(data)
        with _locked(self.root):
            manifest = self._read_manifest()
            if invocation_id in manifest.get("invocations", []):
                existing = data.get("observation_contract")
                return dict(existing) if isinstance(existing, dict) else None
            sequence = len(manifest.get("entries", [])) + 1
            entry: dict[str, Any] = {
                "sequence": sequence,
                "invocation_id": invocation_id,
                "command": command,
                "ok": bool(data.get("ok", True)),
                "duration_ms": round(float(duration_ms or 0), 1),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            contract: dict[str, Any] | None = None
            if observation is not None:
                evidence_id = observation_evidence_id(str(manifest["session_id"]), observation)
                fingerprint = evidence_id.rsplit(":", 1)[-1]
                stale_reason = data.get("stale_risk")
                if stale_reason is None and isinstance(observation.get("meta"), dict):
                    stale_reason = observation["meta"].get("stale_risk")
                contract = {
                    "fingerprint": str(
                        (observation.get("meta") or {}).get("fingerprint") or fingerprint
                    ),
                    "evidence_id": evidence_id,
                    "produced_by": command,
                    "reusable": stale_reason is None,
                    "analyze_needed": stale_reason is not None,
                    "reason": str(stale_reason or "fresh settled observation"),
                }
                data["observation_contract"] = contract
                entry["evidence_id"] = evidence_id
                entry["observation_contract"] = contract
                mode = str(manifest.get("evidence") or "failures")
                failed = not bool(data.get("ok", True))
                if mode == "all" or (mode == "failures" and failed):
                    relative = f"evidence/{sequence:03d}-{fingerprint}-observation.json"
                    entry["observation"] = self._write_json(relative, observation)
                    if screenshot is not None:
                        screenshot_path = self.root / f"evidence/{sequence:03d}-{fingerprint}.png"
                        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            entry["screenshot"] = screenshot(screenshot_path)
                        except Exception as exc:  # noqa: BLE001 - evidence never changes truth
                            manifest.setdefault("capture_errors", []).append(
                                f"{screenshot_path.name}: {type(exc).__name__}: {exc}"
                            )
                if failed and diagnostics is not None and mode in {"failures", "all"}:
                    try:
                        raw = diagnostics()
                        if raw is not None:
                            path = self.root / "failure-diagnostics.txt"
                            atomic_write_text(path, raw)
                            entry["diagnostics"] = str(path)
                    except Exception as exc:  # noqa: BLE001
                        manifest.setdefault("capture_errors", []).append(
                            f"failure-diagnostics.txt: {type(exc).__name__}: {exc}"
                        )
            else:
                data["observation_contract"] = {
                    "fingerprint": None,
                    "evidence_id": None,
                    "produced_by": command,
                    "reusable": False,
                    "analyze_needed": True,
                    "reason": "this result did not contain an observation",
                }
            call = {
                "sequence": sequence,
                "invocation_id": invocation_id,
                "command": command,
                "args": {
                    key: (
                        "<redacted>"
                        if command in {"input", "input_text"} and key == "text"
                        else value
                    )
                    for key, value in (args or {}).items()
                },
                "duration_ms": round(float(duration_ms or 0), 1),
                "result": data,
            }
            with (self.root / "calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_redact(call), ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            manifest.setdefault("entries", []).append(entry)
            manifest.setdefault("invocations", []).append(invocation_id)
            self._write_json("manifest.json", manifest)
            return contract

    def _write_junit(self, result: dict[str, Any], checkpoints: list[dict[str, Any]]) -> str:
        failures = sum(item.get("status") != "completed" for item in checkpoints)
        suite = ET.Element(
            "testsuite",
            {
                "name": "aua.session",
                "tests": str(len(checkpoints)),
                "failures": str(failures),
                "time": f"{float(result.get('duration_ms') or 0) / 1000:.3f}",
            },
        )
        for item in checkpoints:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "name": str(item.get("id") or "checkpoint"),
                    "classname": "aua.session.contract",
                    "time": "0.000",
                },
            )
            if item.get("status") != "completed":
                node = ET.SubElement(case, "failure", {"type": "contract_incomplete"})
                node.text = str(item.get("detail") or item.get("objective") or "incomplete")
        tree = ET.ElementTree(suite)
        ET.indent(tree, space="  ")
        path = self.root / "junit.xml"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return str(path)

    def finalize(
        self,
        result: dict[str, Any],
        *,
        verdict: str,
        checkpoints: list[dict[str, Any]],
        candidate_yaml: str | None = None,
    ) -> dict[str, Any]:
        with _locked(self.root):
            manifest = self._read_manifest()
            finished = datetime.now(UTC)
            manifest["finished_at"] = finished.isoformat()
            with contextlib.suppress(ValueError, TypeError):
                started = datetime.fromisoformat(str(manifest["started_at"]))
                manifest["duration_ms"] = round(
                    max(0.0, (finished - started).total_seconds() * 1000),
                    1,
                )
                result.setdefault("duration_ms", manifest["duration_ms"])
            manifest["verdict"] = verdict
            manifest["checkpoints"] = checkpoints
            self._write_json("manifest.json", manifest)
            if candidate_yaml is not None:
                atomic_write_text(self.root / "candidate-flow.yaml", candidate_yaml)
            result_path = self._write_json("result.json", result)
            lines = [
                "# AUA session report",
                "",
                f"- Session: `{manifest['session_id']}`",
                f"- Verdict: **{verdict.upper()}**",
                f"- Calls: `{len(manifest.get('entries', []))}`",
                "",
                "## Checkpoints",
                "",
            ]
            for checkpoint in checkpoints:
                marker = "PASS" if checkpoint.get("status") == "completed" else "FAIL"
                raw_proof = checkpoint.get("proof")
                proof: dict[str, Any] = raw_proof if isinstance(raw_proof, dict) else {}
                evidence = checkpoint.get("evidence_id") or proof.get("evidence_id") or ""
                lines.append(
                    f"- {marker} `{checkpoint.get('id')}` {checkpoint.get('objective', '')} "
                    f"`{evidence}`".rstrip()
                )
            report_path = self.root / "report.md"
            atomic_write_text(report_path, "\n".join(lines) + "\n")
            artifacts: dict[str, Any] = {
                "dir": str(self.root),
                "manifest": str(self.root / "manifest.json"),
                "calls": str(self.root / "calls.jsonl"),
                "report": str(report_path),
                "result": result_path,
            }
            if bool(manifest.get("junit")):
                artifacts["junit"] = self._write_junit(result, checkpoints)
            if candidate_yaml is not None:
                artifacts["candidate_flow"] = str(self.root / "candidate-flow.yaml")
            result["artifacts"] = artifacts
            self._write_json("result.json", result)
            return result


def session_artifact_store(root: str | Path | None) -> SessionArtifactStore | None:
    return SessionArtifactStore(root) if root else None
