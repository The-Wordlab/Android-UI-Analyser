"""logcat mark + filtered dump."""

from __future__ import annotations

import json
import time
from pathlib import Path

from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.logcat import (
    filter_logcat,
    line_unix_ms,
    load_marks,
    marks_path,
    resolve_since_ms,
    set_mark,
)
from android_ui_analyser.memory import RouteStep
from conftest import FakeDevice, make_config, make_engine

runner = CliRunner()

# Fixed year so threadtime → unix_ms is deterministic in tests.
_REF_YEAR = 2026


def _line(mon: int, day: int, h: int, m: int, s: int, ms: int, tag: str, msg: str) -> str:
    return (
        f"{mon:02d}-{day:02d} {h:02d}:{m:02d}:{s:02d}.{ms:03d}  1234  5678 I {tag}: {msg}"
    )


def test_line_unix_ms_threadtime() -> None:
    line = _line(1, 15, 10, 30, 45, 123, "Foo", "hello")
    ts = line_unix_ms(line, ref_year=_REF_YEAR)
    assert ts is not None
    # 2026-01-15 10:30:45.123 UTC
    assert ts == int(__import__("datetime").datetime(2026, 1, 15, 10, 30, 45, 123000, tzinfo=__import__("datetime").UTC).timestamp() * 1000)


def test_mark_persist_and_since(tmp_path: Path) -> None:
    serial = "fake-1"
    entry = set_mark(tmp_path, serial, "before")
    assert entry["name"] == "before"
    assert "unix_ms" in entry and "iso" in entry
    marks = load_marks(marks_path(tmp_path, serial))
    assert "before" in marks
    since_ms, label = resolve_since_ms(marks, "before")
    assert label == "before"
    assert since_ms == entry["unix_ms"]


def test_resolve_default_last_action_or_30s(tmp_path: Path) -> None:
    marks: dict = {}
    since_ms, label = resolve_since_ms(marks, None)
    assert label == "30s"
    assert since_ms <= int(time.time() * 1000)

    set_mark(tmp_path, "s", "last-action")
    marks = load_marks(marks_path(tmp_path, "s"))
    since_ms2, label2 = resolve_since_ms(marks, None)
    assert label2 == "last-action"
    assert since_ms2 == marks["last-action"]["unix_ms"]


def test_filter_grep_tag_since_lines() -> None:
    lines = [
        _line(1, 15, 10, 0, 0, 0, "Alpha", "one"),
        _line(1, 15, 10, 0, 10, 0, "Beta", "two error"),
        _line(1, 15, 10, 0, 20, 0, "Alpha", "three error"),
        _line(1, 15, 10, 0, 30, 0, "Gamma", "four"),
    ]
    raw = "\n".join(lines)
    mid = line_unix_ms(lines[1], ref_year=_REF_YEAR)
    assert mid is not None

    got = filter_logcat(raw, since_ms=mid, grep="error", ref_year=_REF_YEAR)
    assert len(got) == 2
    assert "Beta" in got[0] and "Alpha" in got[1]

    tagged = filter_logcat(raw, tag="Alpha", ref_year=_REF_YEAR)
    assert len(tagged) == 2

    last = filter_logcat(raw, lines=1, ref_year=_REF_YEAR)
    assert last == [lines[-1]]


def test_engine_logcat_mark_and_dump() -> None:
    cfg = make_config()
    dev = FakeDevice()
    t0 = line_unix_ms(_line(1, 15, 10, 0, 0, 0, "A", "early"), ref_year=_REF_YEAR)
    t1 = line_unix_ms(_line(1, 15, 10, 1, 0, 0, "B", "late boom"), ref_year=_REF_YEAR)
    assert t0 and t1
    # Build lines around "now" so default windows still work; use absolute marks instead.
    dev._logcat_lines = [
        _line(1, 15, 10, 0, 0, 0, "A", "early"),
        _line(1, 15, 10, 1, 0, 0, "B", "late boom"),
        _line(1, 15, 10, 1, 5, 0, "B", "late ok"),
    ]
    eng = Engine(cfg, device=dev)

    marked = eng.logcat_mark("checkpoint")
    assert marked["ok"] and marked["name"] == "checkpoint"

    # Force mark into the past relative to our canned timestamps.
    path = marks_path(cfg.cache.dir, dev.serial)
    marks = load_marks(path)
    marks["checkpoint"] = {"unix_ms": t0 + 1, "iso": "x"}
    path.write_text(json.dumps(marks), encoding="utf-8")

    # Monkeypatch filter to use ref_year via wrapping device output — filter_logcat
    # defaults to current year; for January dates in the future year it's fine with 2026.
    dump = eng.logcat(since="checkpoint", grep="late")
    assert dump["ok"]
    assert dump["since"] == "checkpoint"
    assert dump["grep"] == "late"
    assert len(dump["lines"]) == 2
    assert all("late" in ln for ln in dump["lines"])


def test_engine_auto_marks_last_action() -> None:
    eng = make_engine()
    eng._record_action_safe(RouteStep(kind="tap", label="X"))
    marks = load_marks(marks_path(eng.config.cache.dir, eng.device.serial))
    assert "last-action" in marks


def test_logcat_clear_on_mark() -> None:
    dev = FakeDevice()
    dev._logcat_lines = ["noise"]
    eng = Engine(make_config(), device=dev)
    eng.logcat_mark("cleared", clear=True)
    assert ("logcat", (None, False)) in dev.calls
    assert dev._logcat_lines == []


def test_cli_logcat_mark_and_dump(monkeypatch) -> None:
    cfg = make_config()
    dev = FakeDevice()
    line = _line(7, 29, 12, 0, 0, 0, "Ui", "hello world")
    ts = line_unix_ms(line, ref_year=_REF_YEAR)
    assert ts
    # Use epoch-prefixed lines so filtering doesn't depend on calendar year.
    epoch_line = f"{ts // 1000}.{ts % 1000:03d}  1  1 I Ui: hello world"
    later = f"{ts // 1000 + 60}.000  1  1 I Ui: after mark"
    dev._logcat_lines = [epoch_line, later]
    monkeypatch.setattr(engine_mod, "connect", lambda serial=None: dev)

    r = runner.invoke(app, ["logcat", "mark", "t0"])
    assert r.exit_code == 0, r.stderr
    body = json.loads(r.stdout)
    assert body["name"] == "t0"

    # Point mark just after first line.
    p = marks_path(cfg.cache.dir, dev.serial)
    marks = load_marks(p)
    marks["t0"]["unix_ms"] = ts + 1
    p.write_text(json.dumps(marks), encoding="utf-8")

    r2 = runner.invoke(app, ["logcat", "--since", "t0", "--json"])
    assert r2.exit_code == 0, r2.stderr
    out = json.loads(r2.stdout)
    assert out["ok"]
    assert out["since"] == "t0"
    assert len(out["lines"]) == 1
    assert "after mark" in out["lines"][0]

    r3 = runner.invoke(app, ["logcat", "--since", "t0", "--grep", "after"])
    assert r3.exit_code == 0, r3.stderr
    assert "after mark" in r3.stdout
