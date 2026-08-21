"""logcat mark + filtered dump."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from android_ui_analyser import engine as engine_mod
from android_ui_analyser.cli import app
from android_ui_analyser.engine import Engine
from android_ui_analyser.logcat import (
    extract_crash_evidence,
    filter_logcat,
    line_unix_ms,
    load_marks,
    marks_path,
    resolve_since_ms,
    set_mark,
)
from conftest import FakeDevice, make_config, make_engine

runner = CliRunner()

# Fixed year so threadtime → unix_ms is deterministic in tests.
_REF_YEAR = 2026


def _line(
    mon: int,
    day: int,
    h: int,
    m: int,
    s: int,
    ms: int,
    tag: str,
    msg: str,
    *,
    priority: str = "I",
    pid: int = 1234,
    tid: int = 5678,
) -> str:
    return (
        f"{mon:02d}-{day:02d} {h:02d}:{m:02d}:{s:02d}.{ms:03d}  "
        f"{pid}  {tid} {priority} {tag}: {msg}"
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


def test_crash_evidence_keeps_the_exception_stack_and_drops_unrelated_errors() -> None:
    raw = "\n".join(
        [
            _line(8, 20, 12, 0, 0, 0, "Other", "unrelated error", priority="E", pid=9000),
            _line(
                8,
                20,
                12,
                0,
                0,
                1,
                "AndroidRuntime",
                "FATAL EXCEPTION: main",
                priority="E",
            ),
            _line(
                8,
                20,
                12,
                0,
                0,
                2,
                "AndroidRuntime",
                "Process: com.example.app, PID: 1234",
                priority="E",
            ),
            _line(
                8,
                20,
                12,
                0,
                0,
                3,
                "AndroidRuntime",
                "java.lang.IllegalStateException: broken state",
                priority="E",
            ),
            _line(
                8,
                20,
                12,
                0,
                0,
                4,
                "AndroidRuntime",
                "at com.example.app.MainActivity.onClick(MainActivity.kt:42)",
                priority="E",
            ),
            _line(8, 20, 12, 0, 0, 5, "Noise", "ordinary info", pid=9001),
        ]
    )

    evidence = extract_crash_evidence(raw, app_id="com.example.app")

    assert evidence["kind"] == "fatal"
    assert evidence["matched_app"] is True
    assert evidence["count"] == 4
    assert "FATAL EXCEPTION" in evidence["lines"][0]
    assert "IllegalStateException" in "\n".join(evidence["lines"])
    assert "unrelated error" not in "\n".join(evidence["lines"])


def test_crash_evidence_falls_back_to_error_priority_lines() -> None:
    raw = "\n".join(
        [
            _line(8, 20, 12, 0, 0, 0, "Example", "request failed", priority="E"),
            _line(8, 20, 12, 0, 0, 1, "Example", "ordinary retry", priority="I"),
        ]
    )

    evidence = extract_crash_evidence(raw, app_id="com.example.app")

    assert evidence["kind"] == "error"
    assert evidence["count"] == 1
    assert "request failed" in evidence["lines"][0]


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
    with eng._acting():
        pass
    marks = load_marks(marks_path(eng.config.cache.dir, eng.device.serial))
    assert "last-action" in marks


def test_logcat_clear_on_mark() -> None:
    dev = FakeDevice()
    dev._logcat_lines = ["noise"]
    eng = Engine(make_config(), device=dev)
    eng.logcat_mark("cleared", clear=True)
    assert ("logcat", (None, False, None)) in dev.calls
    assert dev._logcat_lines == []


# --------------------------------------------------------------------- device-clock windows

# `-v threadtime` is stamped in device-LOCAL time: 0 is a UTC emulator, 120 Europe/Madrid
# in summer. The measured host↔device skew on a plain AVD was +9.4s.
_TZ_OFFSET = 120
_SKEWS = (9_400, -9_400)


@pytest.mark.parametrize("skew_ms", _SKEWS)
@pytest.mark.parametrize("utc_offset", [0, _TZ_OFFSET])
def test_mark_window_captures_the_line_logged_right_after_it(
    skew_ms: int, utc_offset: int
) -> None:
    """mark → app logs → dump. Host-clock windows drop the line or stop filtering at all.

    A boundary derived from the host lands ``skew_ms`` away from the clock that stamped the
    log, so the fresh line falls outside the window and the dump comes back empty — which
    reads exactly like "the app never logged anything".
    """
    dev = FakeDevice(clock_skew_ms=skew_ms, utc_offset=utc_offset)
    eng = Engine(make_config(), device=dev)
    stale = dev.log_now("Analytics", "event_from_previous_screen", offset_ms=-13_000)
    eng.logcat_mark("before_tap")
    fresh = dev.log_now("Analytics", "event_message_sent")

    got = eng.logcat(since="before_tap")["lines"]
    assert fresh in got, "the line logged right after the mark was silently dropped"
    assert stale not in got, "a line from before the mark leaked into the window"


def test_mark_reports_device_clock_and_skew() -> None:
    dev = FakeDevice(clock_skew_ms=7_000, utc_offset=_TZ_OFFSET)
    eng = Engine(make_config(), device=dev)
    marked = eng.logcat_mark("m")
    assert marked["clock"] == "device"
    assert 6_000 <= marked["skew_ms"] <= 8_000
    assert marked["host_unix_ms"] - marked["unix_ms"] == marked["skew_ms"]


def test_duration_window_counts_back_from_device_now() -> None:
    dev = FakeDevice(clock_skew_ms=9_400, utc_offset=_TZ_OFFSET)
    eng = Engine(make_config(), device=dev)
    inside = dev.log_now("A", "inside window", offset_ms=-5_000)
    outside = dev.log_now("A", "outside window", offset_ms=-40_000)
    dump = eng.logcat(since="30s")
    assert inside in dump["lines"]
    assert outside not in dump["lines"]


def test_unreadable_device_clock_falls_back_to_host_and_says_so() -> None:
    dev = FakeDevice()
    dev.get_clock_ms = lambda: None  # type: ignore[method-assign]
    eng = Engine(make_config(), device=dev)
    marked = eng.logcat_mark("m")
    assert marked["clock"] == "host"
    assert "skew_ms" not in marked
    assert eng.logcat(since="m")["clock"] == "host"


def test_legacy_host_time_mark_is_converted() -> None:
    """A marks file written before windows moved to the device clock must not mis-window."""
    dev = FakeDevice(clock_skew_ms=9_400, utc_offset=_TZ_OFFSET)
    eng = Engine(make_config(), device=dev)
    host_ms = int(time.time() * 1000)
    path = marks_path(eng.config.cache.dir, dev.serial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"old": {"unix_ms": host_ms, "iso": "x"}}), encoding="utf-8")
    fresh = dev.log_now("A", "after the legacy mark")
    assert fresh in eng.logcat(since="old")["lines"]


def test_dump_delegates_the_time_window_to_the_device() -> None:
    dev = FakeDevice(clock_skew_ms=9_400, utc_offset=_TZ_OFFSET)
    eng = Engine(make_config(), device=dev)
    eng.logcat_mark("m")
    dump = eng.logcat(since="m")
    passed = [args[0] for name, args in dev.calls if name == "logcat" and args[1]]
    assert passed and passed[-1] == dump["since_unix_ms"]


def test_real_device_logcat_uses_native_T_filter(monkeypatch) -> None:
    """The device compares against the clock that stamped the lines; nothing host-side can."""
    from android_ui_analyser import device as device_mod

    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return type("P", (), {"stdout": "07-29 17:00:00.000  1  1 I A: x\n"})()

    monkeypatch.setattr(device_mod.subprocess, "run", fake_run)
    real = object.__new__(device_mod.Uiautomator2Device)
    real.serial = "emulator-5554"
    real.logcat(since_ms=1_785_337_446_619)
    assert seen[-1][-2:] == ["-T", "1785337446.619000000"]


def test_real_device_falls_back_to_device_tz_post_filter(monkeypatch) -> None:
    """When `-T` is unsupported the post-filter must still compare on the device's clock."""
    import subprocess as sp

    from android_ui_analyser import device as device_mod

    inside = "07-29 17:00:10.000  1  1 I A: inside"
    outside = "07-29 16:59:00.000  1  1 I A: outside"

    def fake_run(cmd, **kwargs):
        if "-T" in cmd:
            raise sp.CalledProcessError(1, cmd)
        return type("P", (), {"stdout": f"{outside}\n{inside}\n"})()

    monkeypatch.setattr(device_mod.subprocess, "run", fake_run)
    real = object.__new__(device_mod.Uiautomator2Device)
    real.serial = "emulator-5554"
    monkeypatch.setattr(type(real), "utc_offset_minutes", lambda self: 120)
    # 17:00:00 device-local at UTC+2 == 15:00:00Z
    boundary = int(
        datetime(2026, 7, 29, 15, 0, 0, tzinfo=UTC).timestamp() * 1000
    )
    got = real.logcat(since_ms=boundary)
    assert inside in got and outside not in got


_ACTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.EditText" text="Go"
        resource-id="com.x:id/field" clickable="true" enabled="true" focused="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


# A real gesture is an adb round-trip: the app logs partway through, and the call returns
# some milliseconds later. Both instants are needed to tell the two orderings apart.
_INTERACTION_MS = 15


class LoggingDevice(FakeDevice):
    """A device whose interactions LOG, the way a real app's do.

    Without this the mark's position relative to the action is unobservable — a fake that
    never logs makes a window opened *after* the tap look identical to one opened before.
    """

    def _responds(self, event: str) -> None:
        self.log_now("AnalyticsLog", f"[ACTION] {event}")
        self.advance_clock(_INTERACTION_MS)

    def click(self, x: int, y: int) -> None:
        super().click(x, y)
        self._responds("element_tapped")

    def send_text(self, text: str, *, clear: bool = True) -> None:
        super().send_text(text, clear=clear)
        self._responds("text_entered")

    def press(self, key: str) -> None:
        super().press(key)
        self._responds("key_pressed")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        super().swipe(x1, y1, x2, y2, duration_ms)
        self._responds("list_scrolled")


@pytest.mark.parametrize(
    ("action", "event"),
    [
        ("tap", "element_tapped"),
        ("input", "text_entered"),
        ("key", "key_pressed"),
        ("swipe", "list_scrolled"),
    ],
)
def test_last_action_window_covers_the_actions_own_log_output(action: str, event: str) -> None:
    """`--since last-action` must mean "since just BEFORE the action" — that is the point.

    Stamped after the interaction returns, the window opens past everything the app logged
    in response, so the caller's `mark -> act -> what did the app do` comes back empty.
    """
    dev = LoggingDevice(
        hierarchy_xml=_ACTION_XML, package="com.x", clock_skew_ms=9_400, utc_offset=_TZ_OFFSET
    )
    eng = Engine(make_config(), device=dev)
    target = eng.analyze(source="hierarchy").elements[0].id
    stale = dev.log_now("AnalyticsLog", "[ACTION] event_from_previous_screen", offset_ms=-13_000)

    runners = {
        "tap": lambda: eng.tap(target, observe=False),
        "input": lambda: eng.input_text(target, "hi", observe=False),
        "key": lambda: eng.key("back", observe=False),
        "swipe": lambda: eng.swipe("up", observe=False, verify=False),
    }
    runners[action]()

    got = eng.logcat(since="last-action")["lines"]
    assert any(event in line for line in got), (
        f"{action} logged {event!r} but --since last-action excluded it: {got}"
    )
    assert stale not in got, "the window must still start at THIS action, not an earlier one"


def test_every_state_changing_action_restamps_last_action() -> None:
    """`--since last-action` is only meaningful if the LAST action wrote it."""
    from android_ui_analyser import logcat as logcat_mod

    dev = FakeDevice(hierarchy_xml=_ACTION_XML, package="com.x", clock_skew_ms=4_000)
    eng = Engine(make_config(), device=dev)
    path = marks_path(eng.config.cache.dir, dev.serial)
    target = eng.analyze(source="hierarchy").elements[0].id

    actions = {
        "tap": lambda: eng.tap(target, observe=False),
        "long_press": lambda: eng.long_press(target, observe=False),
        "double_tap": lambda: eng.double_tap(target, observe=False),
        "input": lambda: eng.input_text(target, "hi", observe=False),
        "clear": lambda: eng.clear(target, observe=False),
        "swipe": lambda: eng.swipe("up", observe=False, verify=False),
        "scroll": lambda: eng.scroll("down", observe=False),
        "scroll_to": lambda: eng.scroll_to("Go", observe=False),
        "key": lambda: eng.key("back", observe=False),
        "hide_keyboard": lambda: eng.hide_keyboard(observe=False),
        "open": lambda: eng.open_link("app://x", observe=False),
        "erase": lambda: eng.erase(chars=1, observe=False),
        "paste": lambda: eng.paste(observe=False),
        "orientation": lambda: eng.orientation_set("landscape"),
        "app_launch": lambda: eng.app("launch", package="com.x"),
    }
    for name, run in actions.items():
        path.unlink(missing_ok=True)
        eng.analyze(source="hierarchy")
        run()
        marks = load_marks(path)
        assert "last-action" in marks, f"{name} did not re-stamp last-action"
        expected = logcat_mod.DeviceClock(skew_ms=4_000, measured=True).now_ms()
        assert abs(int(marks["last-action"]["unix_ms"]) - expected) < 2_000, (
            f"{name} stamped the wrong clock"
        )


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
