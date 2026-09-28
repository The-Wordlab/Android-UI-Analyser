"""Console and cross-origin network evidence travels with the screen in every format."""

import json
import time

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import OutputFormat
from test_web_platform import FakeConnection, _adapter, _config


class Connection(FakeConnection):
    def __init__(self):
        super().__init__()
        self.events = []
        self.broken = False

    def diagnostics(self, *, limit, kinds, since_ms):
        if self.broken:
            raise RuntimeError("fixture driver failure")
        events = [event for event in self.events if event["timestamp_ms"] >= since_ms]
        return {
            "events": events[-limit:],
            "total_count": len(events),
            "truncated": len(events) > limit,
        }

    def emit(self, kind, message, **fields):
        self.events.append(
            {"timestamp_ms": int(time.time() * 1000), "kind": kind, "message": message, **fields}
        )


@pytest.fixture
def browser(tmp_path):
    connection = Connection()
    engine = Engine(_config(tmp_path), platform=_adapter(tmp_path, connection))
    try:
        yield engine, connection
    finally:
        engine.close()


@pytest.mark.parametrize("fmt", [OutputFormat.json, OutputFormat.compact, OutputFormat.delta])
def test_unchanged_screen_still_reports_logs_and_cross_origin_failures(browser, fmt):
    engine, connection = browser
    engine.analyze(source="hierarchy", with_ocr=False)
    connection.emit("console", "Hello from console.log", level="log")
    connection.emit(
        "response",
        "Unauthorized",
        url="https://api.fixture.test/data",
        status=401,
        method="GET",
        level="warning",
    )
    observation = engine.analyze(source="hierarchy", with_ocr=False)
    assert observation.meta.unchanged
    data = json.loads(observation.render(fmt))["meta"]["browser_diagnostics"]
    assert data["events"] == connection.events
    assert data["count"] == 2 and not data["truncated"]
    assert data["scope"] == "recent"


def test_noisy_window_keeps_errors_and_reports_omitted_count(browser):
    engine, connection = browser
    engine.config.logs.limit = 3
    connection.emit("page_error", "Unhandled failure", level="error")
    for i in range(10):
        connection.emit("console", f"progress {i}", level="log")
    data = engine.analyze(source="hierarchy", with_ocr=False).meta.browser_diagnostics
    assert data["count"] == 3 and data["total_count"] == 11
    assert data["omitted_count"] == 8 and data["truncated"]
    assert [event["message"] for event in data["events"]] == [
        "Unhandled failure",
        "progress 8",
        "progress 9",
    ]


def test_diagnostics_respect_opt_out_and_do_not_include_bodies(browser):
    engine, connection = browser
    connection.emit(
        "request", "GET /data", method="GET", body="private", headers={"authorization": "private"}
    )
    data = engine.analyze(source="hierarchy", with_ocr=False).meta.browser_diagnostics
    assert "body" not in data["events"][0] and "headers" not in data["events"][0]
    engine.config.logs.enabled = False
    observation = engine.analyze(source="hierarchy", with_ocr=False)
    assert "browser_diagnostics" not in observation.as_dict("compact")["meta"]


def test_failed_diagnostics_do_not_fail_the_screen_or_claim_silence(browser):
    engine, connection = browser
    connection.broken = True
    observation = engine.analyze(source="hierarchy", with_ocr=False)
    assert observation.elements
    assert observation.meta.browser_diagnostics == {"status": "unavailable"}


def test_internal_reads_do_not_discard_the_final_responses_events(browser):
    engine, connection = browser
    connection.emit("console", "Action accepted", level="log")
    engine.analyze(source="hierarchy", with_ocr=False, record_ids=False)
    assert engine.analyze(source="hierarchy", with_ocr=False).meta.browser_diagnostics["count"] == 1


def test_recent_window_does_not_report_old_session_events(browser):
    engine, connection = browser
    connection.events.append({"timestamp_ms": 1, "kind": "console", "message": "Old event"})
    data = engine.analyze(source="hierarchy", with_ocr=False).meta.browser_diagnostics
    assert data["status"] == "ok" and data["count"] == 0


def test_raw_browser_diagnostics_are_not_archived():
    from android_ui_analyser.session_artifacts import _redact

    result = _redact(
        {
            "meta": {
                "browser_diagnostics": {
                    "events": [{"message": "private console content"}],
                    "count": 1,
                    "total_count": 2,
                    "omitted_count": 1,
                    "truncated": True,
                }
            }
        }
    )
    assert result["meta"]["browser_diagnostics"] == {
        "withheld": "browser_diagnostics not archived",
        "count": 1,
        "total_count": 2,
        "omitted_count": 1,
        "truncated": True,
    }
