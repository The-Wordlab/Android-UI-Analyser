"""The journal summary is how we find what to fix next, so it must count what really happened."""

from __future__ import annotations

from android_ui_analyser import journal
from android_ui_analyser.journal_stats import summarize


def _call(tmp_path, target, cmd, *, ok=True, ms=100.0, code=None):
    journal.record(
        cache_dir=tmp_path,
        serial=target,
        platform="web",
        source="daemon",
        cmd=cmd,
        ok=ok,
        duration_ms=ms,
        error={"code": code, "message": "x"} if code else None,
    )


def test_web_calls_are_split_by_attachment_and_failures_are_counted(tmp_path):
    for _ in range(5):
        _call(tmp_path, "browser:abc:0", "tap", ms=300.0)
    _call(tmp_path, "browser:abc:0", "scroll", ok=False, code="scroll_no_movement")
    _call(tmp_path, "existing-chrome", "select_option", ok=False, code="platform_capability_unsupported")

    summary = summarize(tmp_path, platform="web")

    dedicated = summary["surfaces"]["web:dedicated"]
    assert dedicated["calls"] == 6
    assert dedicated["failed"] == 1
    assert dedicated["top_errors"] == [("scroll_no_movement", 1)]
    assert dedicated["slowest_commands_ms"] == [{"cmd": "tap", "median": 300, "calls": 5}]
    assert summary["surfaces"]["web:extension"]["top_failing_commands"] == [("select_option", 1)]
    assert summary["total"]["calls"] == 7
