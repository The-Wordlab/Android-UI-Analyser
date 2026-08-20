"""Tests for the append-only command journal."""

from __future__ import annotations

import json
from pathlib import Path

from android_ui_analyser import journal


def test_database_sql_and_parameters_are_never_journaled() -> None:
    redacted = journal.redact_args(
        {
            "sql": "UPDATE users SET token = 'private-value'",
            "parameters": {"token": "another-private-value"},
        }
    )
    assert redacted == {
        "sql": "<redacted SQL: 40 chars>",
        "parameters": "<redacted>",
    }


def test_record_and_read_since(tmp_path: Path) -> None:
    import time

    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="test",
        cmd="has",
        args={"text": "hello", "password": "secret"},
        ok=True,
        duration_ms=12.5,
        result={"found": True},
    )
    time.sleep(0.02)
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="test",
        cmd="tap",
        args={"id": 4},
        ok=False,
        duration_ms=2000,
        error={"message": "element gone"},
    )

    events = journal.read_since(tmp_path, "emulator-5554", limit=10)
    assert len(events) == 2
    assert events[0]["cmd"] == "has"
    assert events[0]["ok"] is True
    assert events[0]["args"]["password"] == "<redacted>"
    assert events[1]["cmd"] == "tap"
    assert events[1]["ok"] is False

    since = int(events[0]["ts_ms"]) + 1
    newer = journal.read_since(tmp_path, "emulator-5554", since_ms=since, limit=10)
    assert len(newer) == 1
    assert newer[0]["cmd"] == "tap"

    stats = journal.failure_stats(events)
    assert stats["fail_count"] == 1
    assert stats["by_cmd"]["has"]["ok"] == 1
    assert stats["by_cmd"]["tap"]["fail"] == 1
    assert len(stats["slow"]) == 1
    assert stats["slow"][0]["cmd"] == "tap"


def test_read_since_hides_dashboard_by_default(tmp_path: Path) -> None:
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="cli",
        cmd="tap",
        args={"id": 1},
        ok=True,
    )
    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="dashboard",
        cmd="dashboard_start",
        args={},
        ok=True,
    )
    visible = journal.read_since(tmp_path, "emulator-5554")
    assert len(visible) == 1
    assert visible[0]["source"] == "cli"
    all_ev = journal.read_since(tmp_path, "emulator-5554", include_dashboard=True)
    assert len(all_ev) == 2


def test_full_detail_is_on_demand_complete_and_recursively_redacted(tmp_path: Path) -> None:
    long_value = "request-value-" * 50
    secret = "nested-private-password"
    sql = "SELECT private_value FROM records WHERE token = :token"
    parameter = "private-bind-value"
    numeric_parameter = 123456
    elements = [
        {"id": index, "text": f"element {index}", "bounds": [0, index, 10, index + 1]}
        for index in range(45)
    ]
    args = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "long_arg": long_value,
        "many": list(range(35)),
        "credentials": {"password": secret},
        "sql": sql,
        "parameters": {"token": parameter, "pin": numeric_parameter},
    }
    result = {
        "ok": True,
        "custom_response_field": long_value,
        "echoed_secret": secret,
        "echoed_sql": sql,
        "echoed_parameter": parameter,
        "echoed_numeric_parameter": numeric_parameter,
        "observation": {
            "elements": elements,
            "meta": {"known_screen": "detail-screen"},
        },
    }

    journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="mcp",
        cmd="custom_action",
        args=args,
        result=result,
    )

    event = journal.read_since(tmp_path, "emulator-5554", limit=1)[0]
    serialized_event = json.dumps(event)
    assert event["args"]["long_arg"].endswith("…(+300)")
    assert len(event["args"]["many"]) == 30
    assert event["args"]["credentials"]["password"] == "<redacted>"
    assert "custom_response_field" not in event["result"]
    assert event["result"]["observation"] == {
        "elements_count": 45,
        "meta": "detail-screen",
    }
    assert "request" not in event
    assert "response" not in event
    assert secret not in serialized_event
    assert sql not in serialized_event
    assert parameter not in serialized_event
    assert str(numeric_parameter) not in serialized_event

    detail = journal.read_detail(tmp_path, "emulator-5554", event["detail_id"])
    assert detail is not None
    request = detail["request"]
    response = detail["response"]
    assert list(request["args"])[4] == "fifth"
    assert request["args"]["long_arg"] == long_value
    assert request["args"]["many"] == list(range(35))
    assert request["args"]["credentials"]["password"] == "<redacted>"
    assert request["args"]["sql"] == f"<redacted SQL: {len(sql)} chars>"
    assert request["args"]["parameters"] == "<redacted>"
    assert response["result"]["custom_response_field"] == long_value
    assert response["result"]["observation"]["elements"] == elements
    serialized_response = json.dumps(response)
    assert secret not in serialized_response
    assert sql not in serialized_response
    assert parameter not in serialized_response
    assert str(numeric_parameter) not in serialized_response


def test_input_text_never_enters_compact_or_full_journal(tmp_path: Path) -> None:
    private_input = "correct horse battery staple"
    args = {"selector": {"text": "Prompt"}, "text": private_input}
    result = {
        "ok": True,
        "action": "input",
        "detail": "correct horse",
        "observation": {
            "elements": [
                {"id": 1, "text": "correct horse"},
                {"id": 2, "text": "battery staple"},
            ],
        },
    }
    detail_id = journal.record(
        cache_dir=tmp_path,
        serial="emulator-5554",
        source="mcp",
        cmd="input",
        args=args,
        result=result,
        extra={"invocation_id": "private-input-test"},
        privacy_cmd="tap",
    )
    assert detail_id is not None
    assert journal.record_emitted_response(
        cache_dir=tmp_path,
        serial="emulator-5554",
        invocation_id="private-input-test",
        detail_id=detail_id,
        cmd="input",
        args=args,
        result=result,
    )

    event = journal.read_since(tmp_path, "emulator-5554", limit=1)[0]
    detail = journal.read_detail(tmp_path, "emulator-5554", event["detail_id"])
    assert detail is not None
    assert event["args"]["text"] == f"<redacted input: {len(private_input)} chars>"
    assert detail["request"]["args"]["selector"] == {"text": "Prompt"}
    assert detail["request"]["args"]["text"] == (
        f"<redacted input: {len(private_input)} chars>"
    )
    assert detail["response"]["result"]["action"] == "input"
    assert detail["response"]["result"]["observation"]["elements"] == [
        {"id": 1, "text": "<redacted post-input text>"},
        {"id": 2, "text": "<redacted post-input text>"},
    ]
    assert private_input not in json.dumps(event)
    assert private_input not in json.dumps(detail)
    assert "correct horse" not in json.dumps(detail)
    assert "battery staple" not in json.dumps(detail)
