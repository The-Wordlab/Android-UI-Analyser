"""`net:` and `log:` — waiting on evidence the screen cannot show.

A UI predicate answers "is it drawn yet". Sometimes that is unanswerable: a streamed LaTeX
answer reaches the accessibility tree as U+FFFD, so no `text:` term can confirm it arrived,
and `wait --for-stable` is useless on a surface that streams (the guide rejects network idle
for the same reason — this app is never idle).

`net:` waits on the actual HTTP exchange; mitmproxy's response hook fires at stream
completion, which is the moment wanted. `log:` needs no proxy at all but is only as good as
what the app logs. Terms are ANDed, so `net:POST /v1/chat,text:x =` means "the backend
replied *and* the screen shows it".
"""

from __future__ import annotations

import json

from android_ui_analyser import proxy_mock

# --------------------------------------------------------------------------- spec matching


def _entry(**kw):
    base = {"method": "POST", "path": "/v1/chat/completions", "status": 200, "ts": 100.0}
    base.update(kw)
    return base


def test_path_substring_matches_without_the_full_route():
    assert proxy_mock.flow_matches(_entry(), "/v1/chat")
    assert proxy_mock.flow_matches(_entry(), "completions")


def test_method_is_honoured_when_given():
    assert proxy_mock.flow_matches(_entry(), "POST /v1/chat")
    assert not proxy_mock.flow_matches(_entry(), "GET /v1/chat")


def test_status_suffix_is_honoured():
    assert proxy_mock.flow_matches(_entry(), "POST /v1/chat=200")
    assert not proxy_mock.flow_matches(_entry(), "POST /v1/chat=500")
    assert proxy_mock.flow_matches(_entry(status=500), "POST /v1/chat=500")


def test_a_bare_word_is_not_read_as_a_method():
    """`completions` is a path fragment, not an HTTP verb — it must not be parsed as one."""
    assert proxy_mock.flow_matches(_entry(), "completions")
    assert not proxy_mock.flow_matches(_entry(), "PATCH /v1/chat")


def test_non_matching_path_is_rejected():
    assert not proxy_mock.flow_matches(_entry(), "/feed")


# --------------------------------------------------------------------------- the flow log


def test_only_exchanges_after_the_baseline_count(tmp_path):
    """Without a baseline the previous turn's response satisfies this turn's wait instantly."""
    log = proxy_mock.flow_log_path(tmp_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                _entry(ts=100.0, path="/v1/chat/old"),
                _entry(ts=200.0, path="/v1/chat/new"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fresh = proxy_mock.read_flows_since(tmp_path, 150.0)
    assert [f["path"] for f in fresh] == ["/v1/chat/new"]


def test_a_half_written_trailing_line_is_survivable(tmp_path):
    """The proxy appends while we poll, so the last line can be torn mid-write."""
    log = proxy_mock.flow_log_path(tmp_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(_entry(ts=200.0)) + '\n{"ts": 201.0, "meth', encoding="utf-8")
    flows = proxy_mock.read_flows_since(tmp_path, 0.0)
    assert len(flows) == 1


def test_missing_flow_log_is_not_an_error(tmp_path):
    """`net:` with no proxy running must read as "not yet", never as a crash."""
    assert proxy_mock.read_flows_since(tmp_path, 0.0) == []


def test_flow_log_is_separate_from_the_cassette_record(tmp_path):
    """Waiting must not require `record` mode, and must not disturb cassette replay."""
    assert proxy_mock.flow_log_path(tmp_path) != proxy_mock.record_path(tmp_path)
    assert proxy_mock.flow_log_path(tmp_path).suffix == ".jsonl"


# --------------------------------------------------------------------------- the addon


def test_addon_logs_every_exchange_regardless_of_mode():
    """The flow log is always-on; only the cassette record is gated on `record` mode."""
    src = proxy_mock.ADDON_SCRIPT
    body = src.split("def response(", 1)[1]
    flow_write = body.index("_FLOW_LOG_PATH")
    mode_gate = body.index('mode != "record"')
    assert flow_write < mode_gate, "the flow log must be written before the record-mode gate"


def test_addon_flow_entry_carries_a_timestamp():
    """Without `ts` there is no baseline, and every wait matches stale traffic."""
    assert '"ts": time.time()' in proxy_mock.ADDON_SCRIPT


def test_addon_does_not_log_bodies():
    """A streamed chat turn would otherwise write megabytes per response."""
    response_block = proxy_mock.ADDON_SCRIPT.split("if str(_FLOW_LOG_PATH):", 1)[1]
    first_write = response_block.split("mode = os.environ", 1)[0]
    assert "body" not in first_write
