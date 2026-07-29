"""YAML acceptance-criteria checklist runner (``aua suite run``).

A suite is a short list of ``has`` / ``expect`` / ``wait_for`` checks — the same
primitives an agent would call one-by-one, batched so an AC list becomes one exit
code (0 = all pass, 8 = any fail).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .errors import UsageError

if TYPE_CHECKING:
    from .engine import Engine


@dataclass
class SuiteCheck:
    kind: str  # has | expect | wait_for
    raw: dict[str, Any] = field(default_factory=dict)
    # has
    text: str | None = None
    # expect
    rid: str | None = None
    expect_text: str | None = None
    desc: str | None = None
    exists: bool | None = None
    absent: bool | None = None
    match: str | None = None
    # wait_for
    wait_for: str | None = None
    timeout_ms: int | None = None


@dataclass
class Suite:
    name: str
    checks: list[SuiteCheck]
    app: str | None = None


@dataclass
class CheckResult:
    index: int
    ok: bool
    kind: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    ok: bool
    name: str
    results: list[CheckResult]
    passed: int
    failed: int
    stopped_early: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "stopped_early": self.stopped_early,
            "results": [r.as_dict() for r in self.results],
        }


def parse_suite(text: str, *, source: str = "<stdin>") -> Suite:
    """Parse a suite YAML document."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise UsageError(f"invalid suite YAML in {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError(f"suite YAML must be a mapping ({source})")
    name = str(data.get("name") or Path(source).stem or "suite")
    app = data.get("app")
    if app is not None:
        app = str(app)
    raw_checks = data.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        raise UsageError(f"suite needs a non-empty checks: list ({source})")
    checks: list[SuiteCheck] = []
    for i, item in enumerate(raw_checks):
        checks.append(_parse_check(item, index=i, source=source))
    return Suite(name=name, checks=checks, app=app)


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    if not p.is_file():
        raise UsageError(f"suite file not found: {p}")
    return parse_suite(p.read_text(encoding="utf-8"), source=str(p))


def _parse_check(item: Any, *, index: int, source: str) -> SuiteCheck:
    if not isinstance(item, dict) or len(item) < 1:
        raise UsageError(f"check[{index}] must be a mapping ({source})")
    if "has" in item:
        text = item["has"]
        if not isinstance(text, str) or not text:
            raise UsageError(f"check[{index}].has must be a non-empty string ({source})")
        return SuiteCheck(kind="has", raw=dict(item), text=text)
    if "wait_for" in item:
        text = item["wait_for"]
        if not isinstance(text, str) or not text:
            raise UsageError(f"check[{index}].wait_for must be a non-empty string ({source})")
        timeout = item.get("timeout_ms")
        if timeout is not None:
            timeout = int(timeout)
        return SuiteCheck(kind="wait_for", raw=dict(item), wait_for=text, timeout_ms=timeout)
    if "expect" in item:
        body = item["expect"]
        if not isinstance(body, dict):
            raise UsageError(f"check[{index}].expect must be a mapping ({source})")
        rid = body.get("rid")
        text = body.get("text")
        desc = body.get("desc")
        n = sum(1 for v in (rid, text, desc) if v)
        if n != 1:
            raise UsageError(
                f"check[{index}].expect needs exactly one of rid/text/desc ({source})"
            )
        exists = body.get("exists")
        absent = body.get("absent")
        if exists is not None:
            exists = bool(exists)
        if absent is not None:
            absent = bool(absent)
        return SuiteCheck(
            kind="expect",
            raw=dict(item),
            rid=str(rid) if rid is not None else None,
            expect_text=str(text) if text is not None else None,
            desc=str(desc) if desc is not None else None,
            exists=exists,
            absent=absent,
            match=str(body["match"]) if body.get("match") is not None else None,
            timeout_ms=int(body["timeout_ms"]) if body.get("timeout_ms") is not None else None,
        )
    raise UsageError(
        f"check[{index}] needs has: / expect: / wait_for: ({source})",
        hint='e.g. `- has: "Notifications"` or `- expect: {rid: foo, exists: true}`',
    )


def run_check(engine: Engine, check: SuiteCheck) -> tuple[bool, str]:
    """Run one check; returns ``(ok, detail)``."""
    if check.kind == "has":
        assert check.text is not None
        has_result = engine.has(check.text)
        detail = f"has:{check.text!r} found={has_result.found}"
        return bool(has_result.found), detail

    if check.kind == "wait_for":
        assert check.wait_for is not None
        timeout = check.timeout_ms if check.timeout_ms is not None else 5000
        wait_result = engine.wait(for_=check.wait_for, timeout_ms=timeout, observe=False)
        detail = wait_result.detail or f"wait_for:{check.wait_for!r}"
        return bool(wait_result.ok), f"wait_for:{detail} ok={wait_result.ok}"

    # expect
    kwargs: dict[str, Any] = {
        "rid": check.rid,
        "text": check.expect_text,
        "desc": check.desc,
        "observe": False,
    }
    if check.exists is not None:
        kwargs["exists"] = check.exists
    if check.absent is not None:
        kwargs["absent"] = check.absent
    if check.timeout_ms is not None:
        kwargs["timeout_ms"] = check.timeout_ms
    # Optional match:contains on a text selector → also require text_contains.
    if (
        check.match
        and check.match.lower() == "contains"
        and check.expect_text
        and check.exists is not False
        and not check.absent
    ):
        kwargs["text_contains"] = check.expect_text
    expect_result = engine.expect(**kwargs)
    detail = expect_result.detail or "expect"
    return bool(expect_result.ok), detail


def run_suite(
    engine: Engine,
    suite: Suite,
    *,
    continue_on_fail: bool = False,
) -> SuiteResult:
    """Execute every check; stop on first failure unless *continue_on_fail*."""
    if suite.app:
        engine.app("launch", package=suite.app)

    results: list[CheckResult] = []
    stopped_early = False
    for i, check in enumerate(suite.checks):
        ok, detail = run_check(engine, check)
        results.append(CheckResult(index=i, ok=ok, kind=check.kind, detail=detail))
        if not ok and not continue_on_fail:
            stopped_early = True
            break

    passed = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    return SuiteResult(
        ok=failed == 0 and len(results) == len(suite.checks),
        name=suite.name,
        results=results,
        passed=passed,
        failed=failed,
        stopped_early=stopped_early,
    )


def render_summary(result: SuiteResult) -> str:
    lines = [
        f"suite {result.name}: {'PASS' if result.ok else 'FAIL'} "
        f"({result.passed} passed, {result.failed} failed"
        + (", stopped early" if result.stopped_early else "")
        + ")"
    ]
    for r in result.results:
        mark = "ok" if r.ok else "FAIL"
        lines.append(f"  [{r.index}] {mark}  {r.kind}: {r.detail}")
    return "\n".join(lines)
