"""Review reads emitted responses once, while retaining compact caller accounting."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser import cli_invocations, journal
from android_ui_analyser.observation_contract import result_has_reusable_observation


def _record(tmp_path: Path) -> str:
    detail_id = journal.record(
        cache_dir=tmp_path,
        serial="example-target",
        source="cli",
        cmd="analyze",
        ok=True,
        duration_ms=42,
        args={"source": "auto"},
        extra={"invocation_id": "example-call"},
        result={
            "observation": {
                "elements": [{"id": "el:next", "text": "Next"}],
                "meta": {"fingerprint": "example-frame"},
            }
        },
    )
    assert detail_id is not None
    return detail_id


def test_review_uses_latest_projected_result_and_preserves_original_timing(tmp_path: Path) -> None:
    detail_id = _record(tmp_path)
    for elements in ([{"id": "el:intermediate"}], []):
        assert journal.record_emitted_response(
            cache_dir=tmp_path,
            serial="example-target",
            invocation_id="example-call",
            detail_id=detail_id,
            cmd="analyze",
            args={"source": "auto"},
            result={"observation": {"elements": elements}},
            request_context={"projection": {"where_text": ["Absent"]}},
        )
    compact = journal.read_since(tmp_path, "example-target")
    assert compact[0]["result"]["observation"]["elements_count"] == 1

    reviewed = journal.review_events(tmp_path, "example-target", compact)

    assert len(reviewed) == 1
    assert reviewed[0]["duration_ms"] == 42
    assert reviewed[0]["result"]["observation"]["elements"] == []
    assert reviewed[0]["client"] == {"projection": {"where_text": ["Absent"]}}
    assert not result_has_reusable_observation(reviewed[0]["result"])


def test_review_with_missing_retained_details_keeps_the_compact_event(tmp_path: Path) -> None:
    _record(tmp_path)
    compact = journal.read_since(tmp_path, "example-target")
    journal.journal_detail_path(tmp_path, "example-target").unlink()

    assert journal.review_events(tmp_path, "example-target", compact) == compact


def test_reverse_detail_read_handles_long_utf8_frames_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "details.jsonl"
    lines = ["first", "é" * 70_000, "last"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert list(journal._reverse_lines(path)) == list(reversed(lines))


def test_compact_journal_preserves_readiness_caveats_even_without_details() -> None:
    result = {
        "ok": True,
        "settled_unmet": True,
        "observation_empty": True,
        "observation_contract": {"reusable": False, "readiness": "unmet"},
        "observation": {"elements": [{"id": "el:next"}]},
    }

    compact = journal.summarize_result(result)

    assert compact["settled_unmet"] is True
    assert compact["observation_empty"] is True
    assert compact["observation_contract"]["reusable"] is False
    assert not result_has_reusable_observation(compact)


def test_cli_root_is_marked_only_after_a_successful_journal_append(tmp_path: Path, monkeypatch) -> None:
    state = cli_invocations.Invocation([], invocation_id="example-call")
    token = cli_invocations._current.set(state)
    try:
        _record(tmp_path)
        assert state.records == 1

        def cannot_write(*_args, **_kwargs):
            raise OSError("example full disk")

        monkeypatch.setattr(journal, "_append_private", cannot_write)
        assert journal.record(
            cache_dir=tmp_path,
            serial="example-target",
            source="cli",
            cmd="analyze",
            extra={"invocation_id": "example-call"},
        ) is None
        assert state.records == 1
    finally:
        cli_invocations._current.reset(token)
