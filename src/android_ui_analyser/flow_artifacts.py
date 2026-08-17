"""Portable evidence bundles for one ``flow run`` execution.

The executor owns UI semantics; this module only records its already-structured result and asks
for screenshots through an injected callback.  That keeps artifact formatting independent from
the selected platform runtime and gives CLI, daemon, and MCP exactly the same files.
"""

from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .schema import AnalyzeResult


_EVIDENCE_MODES = frozenset({"none", "failures", "all"})


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized[:80] or "checkpoint"


def _redact_inputs(value: Any) -> Any:
    """Keep resolved input values out of portable artifacts."""

    if isinstance(value, list):
        return [_redact_inputs(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: _redact_inputs(item) for key, item in value.items()}
    if value.get("kind") == "input" and "text" in out:
        out["text"] = "<redacted>"
    return out


def _observation_payload(result: AnalyzeResult) -> dict[str, Any]:
    return {
        "known_screen": result.meta.known_screen,
        "screen": result.screen.model_dump(mode="json"),
        "elements": [element.compact() for element in result.elements],
    }


class FlowArtifactWriter:
    """Collect screenshots, observations, JSON, Markdown, and optional JUnit XML."""

    def __init__(
        self,
        requested_dir: str | Path,
        *,
        flow_name: str,
        evidence: str,
        junit: bool,
        screenshot: Callable[[Path], str],
        diagnostics: Callable[[], str | None] | None = None,
    ) -> None:
        if evidence not in _EVIDENCE_MODES:
            raise ValueError(f"unknown evidence mode {evidence!r}")
        self.run_id = f"flow-{uuid.uuid4()}"
        requested = Path(requested_dir).expanduser()
        requested.mkdir(parents=True, exist_ok=True)
        # Reusing an output directory must not overwrite an earlier test run.  An empty path is
        # used directly for the ergonomic first run; later runs get an immutable run directory.
        self.root = requested / self.run_id if any(requested.iterdir()) else requested
        self.root.mkdir(parents=True, exist_ok=True)
        self.flow_name = flow_name
        self.evidence = evidence
        self.junit = junit
        self._screenshot = screenshot
        self._diagnostics = diagnostics
        self.started_at = datetime.now(UTC)
        self._step_sequence = 0
        self._checkpoint_sequence = 0
        self._entries: list[dict[str, Any]] = []
        self._capture_errors: list[str] = []

    def _write_json(self, relative: str, value: Any) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_redact_inputs(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(path)

    def _capture(self, relative: str) -> str | None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._screenshot(path)
        except Exception as exc:  # noqa: BLE001 - evidence failure must not change test truth
            self._capture_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            return None

    def capture_checkpoint(self, name: str) -> str | None:
        self._checkpoint_sequence += 1
        relative = f"screenshots/{self._checkpoint_sequence:03d}-{_safe_name(name)}.png"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._screenshot(path)
        except Exception as exc:  # noqa: BLE001 - caller turns this into a structured step failure
            self._capture_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            raise

    def record_step(
        self,
        row: dict[str, Any],
        *,
        kind: str,
        observation: AnalyzeResult,
    ) -> None:
        self._step_sequence += 1
        sequence = self._step_sequence
        evidence_id = f"{self.run_id}:step:{sequence}"
        row["evidence_id"] = evidence_id
        entry: dict[str, Any] = {
            "evidence_id": evidence_id,
            "sequence": sequence,
            "step": row.get("step"),
            "duration_ms": row.get("duration_ms"),
        }
        if self.evidence == "all" and kind not in {"repeat", "retry", "flow", "screenshot"}:
            screenshot = self._capture(f"steps/{sequence:03d}-after.png")
            observation_path = self._write_json(
                f"steps/{sequence:03d}-observation.json", _observation_payload(observation)
            )
            entry["observation"] = observation_path
            row.setdefault("artifacts", []).append(observation_path)
            if screenshot:
                entry["screenshot"] = screenshot
                row.setdefault("artifacts", []).append(screenshot)
        self._entries.append(entry)

    def record_failure(self, result: dict[str, Any], observation: AnalyzeResult) -> None:
        evidence_id = f"{self.run_id}:failure"
        result["failure_evidence_id"] = evidence_id
        entry: dict[str, Any] = {
            "evidence_id": evidence_id,
            "step": result.get("failed_step", {}).get("display"),
            "code": result.get("code"),
        }
        if self.evidence in {"failures", "all"}:
            screenshot = self._capture("failure.png")
            observation_path = self._write_json(
                "failure-observation.json", _observation_payload(observation)
            )
            entry["observation"] = observation_path
            if screenshot:
                entry["screenshot"] = screenshot
            if self._diagnostics is None:
                entry["diagnostics_status"] = "unsupported"
            else:
                try:
                    diagnostics = self._diagnostics()
                    if diagnostics is None:
                        entry["diagnostics_status"] = "unsupported"
                    else:
                        diagnostics_path = self.root / "failure-diagnostics.txt"
                        diagnostics_path.write_text(diagnostics, encoding="utf-8")
                        entry["diagnostics"] = str(diagnostics_path)
                except Exception as exc:  # noqa: BLE001 - diagnostics never change test truth
                    self._capture_errors.append(
                        f"failure-diagnostics.txt: {type(exc).__name__}: {exc}"
                    )
        self._entries.append(entry)

    def record_preflight_failure(self, result: dict[str, Any]) -> None:
        """Record a refusal that intentionally happened before any device observation."""

        evidence_id = f"{self.run_id}:failure"
        result["failure_evidence_id"] = evidence_id
        self._entries.append(
            {
                "evidence_id": evidence_id,
                "step": result.get("failed_step", {}).get("display"),
                "code": result.get("code"),
                "evidence_status": "not_collected_preflight",
            }
        )

    def _write_junit(self, result: dict[str, Any], duration_ms: int) -> str:
        rows = list(result.get("steps_run") or [])
        failure = result.get("failed_step") if result.get("ok") is False else None
        if result.get("ok") is False and not failure:
            failure = {"display": str(result.get("code") or "flow failure")}
        suite = ET.Element(
            "testsuite",
            {
                "name": self.flow_name,
                "tests": str(len(rows) + (1 if failure else 0)),
                "failures": "1" if failure else "0",
                "time": f"{duration_ms / 1000:.3f}",
            },
        )
        for sequence, row in enumerate(rows, start=1):
            ET.SubElement(
                suite,
                "testcase",
                {
                    "name": str(row.get("step") or f"step {sequence}"),
                    "classname": f"aua.flow.{self.flow_name}",
                    "time": f"{float(row.get('duration_ms') or 0) / 1000:.3f}",
                },
            )
        if failure:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "name": str(failure.get("display") or "flow failure"),
                    "classname": f"aua.flow.{self.flow_name}",
                    "time": "0.000",
                },
            )
            node = ET.SubElement(
                case,
                "failure",
                {"type": str(result.get("code") or "flow_failure")},
            )
            node.text = str(result.get("failure_detail") or result.get("hint") or "flow failed")
        tree = ET.ElementTree(suite)
        path = self.root / "junit.xml"
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return str(path)

    def _write_report(self, result: dict[str, Any], duration_ms: int) -> str:
        status = "PASS" if result.get("ok") else "FAIL"
        lines = [
            f"# AUA flow report: {self.flow_name}",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Result: **{status}**",
            f"- Duration: `{duration_ms} ms`",
            f"- Steps completed: `{len(result.get('steps_run') or [])}`",
        ]
        if result.get("code"):
            lines.append(f"- Failure code: `{result['code']}`")
        lines.extend(["", "## Steps", ""])
        for row in result.get("steps_run") or []:
            lines.append(
                f"- PASS `{row.get('evidence_id', '')}` {row.get('step')} "
                f"({row.get('duration_ms', 0)} ms)"
            )
        if result.get("failed_step"):
            lines.append(
                f"- FAIL `{result.get('failure_evidence_id', '')}` "
                f"{result['failed_step'].get('display')} — {result.get('code')}"
            )
        path = self.root / "report.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def finalize(
        self,
        result: dict[str, Any],
        *,
        canonical_flow_yaml: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        flow_path = self.root / "flow.yaml"
        flow_path.write_text(canonical_flow_yaml, encoding="utf-8")
        result["run_id"] = self.run_id
        report_path = self._write_report(result, duration_ms)
        junit_path = self._write_junit(result, duration_ms) if self.junit else None
        finished_at = datetime.now(UTC)
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "flow": self.flow_name,
            "ok": bool(result.get("ok")),
            "code": result.get("code"),
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
            "evidence": self.evidence,
            "entries": self._entries,
            "capture_errors": self._capture_errors,
        }
        manifest_path = self._write_json("manifest.json", manifest)
        artifacts: dict[str, Any] = {
            "dir": str(self.root),
            "manifest": manifest_path,
            "flow": str(flow_path),
            "report": report_path,
            "result": str(self.root / "result.json"),
        }
        if junit_path:
            artifacts["junit"] = junit_path
        result["artifacts"] = artifacts
        self._write_json("result.json", result)
        return result


def validate_evidence_mode(value: str) -> str:
    normalized = str(value or "failures").strip().lower()
    if normalized not in _EVIDENCE_MODES:
        choices = ", ".join(sorted(_EVIDENCE_MODES))
        raise ValueError(f"evidence must be one of {choices}, got {value!r}")
    return normalized
