"""Proxy mock cassette / rule helpers (no real mitmproxy)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import proxy_mock as pm
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from conftest import FakeDevice, make_config


def test_map_rule_and_cassette_roundtrip(tmp_path: Path) -> None:
    rule = pm.map_rule("GET", "/notifications", status=200, body='{"items":[]}')
    assert rule["request"]["method"] == "GET"
    assert rule["response"]["status"] == 200
    assert rule["response"]["body"] == {"items": []}

    path = tmp_path / "empty_inbox.yaml"
    pm.save_cassette(path, "empty_inbox", [rule])
    loaded = pm.load_cassette(path)
    assert len(loaded) == 1
    assert loaded[0]["request"]["path"] == "/notifications"


def test_engine_mock_map_replay(tmp_path: Path) -> None:
    device = FakeDevice()
    cfg = make_config(cache={"dir": str(tmp_path / "cache")}, memory={"dir": str(tmp_path / "mem")})
    engine = Engine(cfg, device=device)

    out = engine.mock_map("GET", "/x", status=204, body=None)
    assert out["count"] == 1

    cass = pm.cassette_dir(cfg.memory.dir) / "empty.yaml"
    pm.save_cassette(cass, "empty", [pm.map_rule("GET", "/notifications", body={"items": []})])
    replay = engine.mock_replay("empty")
    assert replay["entries"] == 1
    rules = pm.load_rules(pm.rules_path(Path(cfg.cache.dir)))
    assert rules[0]["request"]["path"] == "/notifications"


def test_proxy_start_missing_mitm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = FakeDevice()
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)

    def boom(**kwargs: Any) -> tuple[int, int]:
        raise UsageError("mitmproxy is not installed", hint="install [proxy]")

    monkeypatch.setattr(pm, "start_mitm", boom)
    monkeypatch.setattr(pm, "install_system_ca", lambda *_a, **_k: {"ok": True})
    with pytest.raises(UsageError, match="mitmproxy"):
        engine.proxy_start(install_ca=False)


def test_pick_listen_port_avoids_8080_and_persists(tmp_path: Path) -> None:
    port = pm.pick_listen_port()
    assert port != 8080
    assert 1024 < port < 65536
    pm.save_listen_port(tmp_path, port)
    assert pm.load_listen_port(tmp_path) == port
    pm.clear_listen_port(tmp_path)
    assert pm.load_listen_port(tmp_path) is None


def test_proxy_start_uses_random_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = FakeDevice()
    cache = tmp_path / "cache"
    engine = Engine(make_config(cache={"dir": str(cache)}), device=device)

    def fake_start(*, cache_dir: Path, port: int | None = None, mode: str = "map") -> tuple[int, int]:
        listen = pm.pick_listen_port(preferred=port)
        pm.save_listen_port(cache_dir, listen)
        (cache_dir / "mitmproxy.pid").write_text("12345", encoding="utf-8")
        return 12345, listen

    monkeypatch.setattr(pm, "start_mitm", fake_start)
    monkeypatch.setattr(pm, "install_system_ca", lambda *_a, **_k: {"ok": True, "hash": "abc"})
    out = engine.proxy_start(install_ca=True)
    assert out["port"] != 8080
    assert pm.load_listen_port(cache) == out["port"]
    # Record must reuse the same port, not fall back to 8080.
    seen: list[int | None] = []

    def fake_start2(*, cache_dir: Path, port: int | None = None, mode: str = "map") -> tuple[int, int]:
        seen.append(port)
        listen = port or 54321
        pm.save_listen_port(cache_dir, listen)
        return 99, listen

    monkeypatch.setattr(pm, "start_mitm", fake_start2)
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)
    rec = engine.mock_record("start", "hub")
    assert seen == [out["port"]]
    assert rec["port"] == out["port"]


def test_tls_failures_in_log(tmp_path: Path) -> None:
    log = tmp_path / "mitmdump.log"
    log.write_text(
        "info\nClient TLS handshake failed. The client does not trust the proxy's certificate "
        "for api.staging.example.com\nok\n",
        encoding="utf-8",
    )
    hits = pm.tls_failures_in_log(tmp_path)
    assert hits and "does not trust" in hits[0]


def test_mock_record_survives_the_start_mode_flip_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the reported bug directly: real flows the addon durably appended to the
    on-disk cassette record during a recording window were reported as ``entries: 0`` by
    ``mock record stop``, even though nothing was lost in memory.

    The addon's ``AuaMock.response()`` appends one JSON line per completed exchange straight
    to disk (``_append`` opens the file, writes, and closes it on every flow) — there is no
    in-memory buffer to lose across a process restart. The actual mechanism: ``mock record
    start`` primes the cassette file with the literal text ``"[]"`` (no trailing newline), and
    ``mock record stop`` parses the *whole file* as one JSON document. The very first appended
    line glues onto that ``]`` with no separator, so the file is no longer valid JSON from that
    point on and the whole parse raises — silently caught, reported as zero entries — even
    though every line the addon wrote is intact JSONL on disk.
    """
    device = FakeDevice()
    cache = tmp_path / "cache"
    cfg = make_config(cache={"dir": str(cache)}, memory={"dir": str(tmp_path / "mem")})
    engine = Engine(cfg, device=device)

    def fake_start(*, cache_dir: Path, port: int | None = None, mode: str = "map") -> tuple[int, int]:
        return 4242, port or 49099

    monkeypatch.setattr(pm, "start_mitm", fake_start)
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)

    cache.mkdir(parents=True, exist_ok=True)  # normally created by an earlier `proxy start`
    engine.mock_record("start", "login_flow")

    # Simulate the addon appending two real, fully-decrypted flows during the window exactly
    # as `AuaMock.response()` does in `record` mode: open the record file in append mode and
    # write one JSON object per line, body included (`_snippet(flow.response)` is captured
    # into `response.body` unconditionally in record mode — independent of `capture_bodies`,
    # which only gates the separate always-on flow-bodies log).
    widget_body = '{"id": 42, "name": "left-handed smoke shifter", "in_stock": true}'
    rec = pm.record_path(cache)
    with rec.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "request": {"method": "POST", "path": "/api/v1/widgets", "host": "api.example.com"},
                    "response": {"status": 200, "body": widget_body},
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "request": {"method": "POST", "path": "/a1", "host": "events.example.com"},
                    "response": {"status": 200, "body": "ok"},
                }
            )
            + "\n"
        )

    out = engine.mock_record("stop", "login_flow")

    assert out["entries"] == 2, (
        "flows the addon durably appended to disk during the window must be counted, not "
        "discarded because the cassette file was primed with a bare `[]` the JSONL reader "
        "was never used to parse back"
    )
    assert out["ok"] is True

    # Not just a count: the actual response body an agent would need to install a matching
    # stub must survive the round trip onto the saved cassette, untouched.
    saved = pm.load_cassette(Path(out["path"]))
    widget_entries = [e for e in saved if e["request"]["path"] == "/api/v1/widgets"]
    assert len(widget_entries) == 1
    assert widget_entries[0]["response"]["body"] == widget_body, (
        "recording exists to let an agent see and replay a real response body — a durable "
        "entry with the body silently dropped is the same failure by another name"
    )


def test_diagnose_empty_recording_ignores_stale_tls_failures_before_the_window(
    tmp_path: Path,
) -> None:
    log = tmp_path / "mitmdump.log"
    log.write_text(
        "[10:00:00.000][127.0.0.1:1] Client TLS handshake failed. The client does not trust "
        "the proxy's certificate for update.googleapis.com (OpenSSL Error(...))\n",
        encoding="utf-8",
    )
    offset = log.stat().st_size  # the recording window starts strictly after this line

    diag = pm.diagnose_empty_recording(tmp_path, since_ts=time.time(), log_offset=offset)

    assert diag["diagnosis"] != "tls_failed"
    assert diag["tls_failures_app"] == []


def test_diagnose_empty_recording_does_not_blame_the_ca_when_flows_decrypted_in_window(
    tmp_path: Path,
) -> None:
    log = tmp_path / "mitmdump.log"
    log.write_text(
        "[10:00:00.000][127.0.0.1:1] Client TLS handshake failed. The client does not trust "
        "the proxy's certificate for update.googleapis.com (OpenSSL Error(...))\n",
        encoding="utf-8",
    )
    offset = log.stat().st_size
    since = time.time()
    flow_log = pm.flow_log_path(tmp_path)
    flow_log.parent.mkdir(parents=True, exist_ok=True)
    with flow_log.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "n": 1,
                    "ts": since + 1,
                    "method": "POST",
                    "path": "/api/v1/widgets",
                    "host": "api.example.com",
                    "status": 200,
                    "action": None,
                    "rule": None,
                }
            )
            + "\n"
        )

    diag = pm.diagnose_empty_recording(tmp_path, since_ts=since, log_offset=offset)

    assert diag["diagnosis"] != "tls_failed", (
        "the CA must never be blamed when flows for the app under test demonstrably "
        "decrypted during the same window"
    )
    assert diag["decrypted_flows_app"] == 1


def test_diagnose_empty_recording_flags_real_ca_distrust_for_non_system_hosts(
    tmp_path: Path,
) -> None:
    log = tmp_path / "mitmdump.log"
    log.write_text(
        "[10:00:00.000][127.0.0.1:1] Client TLS handshake failed. The client does not trust "
        "the proxy's certificate for api.example.com (OpenSSL Error(...))\n",
        encoding="utf-8",
    )

    diag = pm.diagnose_empty_recording(tmp_path, since_ts=0.0, log_offset=0)

    assert diag["diagnosis"] == "tls_failed"
    assert diag["tls_failures_app"]


def test_diagnose_empty_recording_treats_system_only_failures_as_no_app_traffic(
    tmp_path: Path,
) -> None:
    log = tmp_path / "mitmdump.log"
    log.write_text(
        "[10:00:00.000][127.0.0.1:1] Client TLS handshake failed. The client does not trust "
        "the proxy's certificate for android.googleapis.com (OpenSSL Error(...))\n",
        encoding="utf-8",
    )

    diag = pm.diagnose_empty_recording(tmp_path, since_ts=0.0, log_offset=0)

    assert diag["diagnosis"] != "tls_failed", (
        "OS/Google-services traffic legitimately never trusts the overlay and must not be "
        "read as evidence that the app under test does not trust it"
    )


def test_mock_record_stop_hint_does_not_blame_ca_when_window_decrypted_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice()
    cache = tmp_path / "cache"
    cfg = make_config(cache={"dir": str(cache)}, memory={"dir": str(tmp_path / "mem")})
    engine = Engine(cfg, device=device)

    def fake_start(*, cache_dir: Path, port: int | None = None, mode: str = "map") -> tuple[int, int]:
        return 4242, port or 49099

    monkeypatch.setattr(pm, "start_mitm", fake_start)
    monkeypatch.setattr(pm, "stop_mitm", lambda _c: True)
    cache.mkdir(parents=True, exist_ok=True)  # normally created by an earlier `proxy start`

    # A stale, pre-window TLS failure against a Google system host, left over from an
    # unrelated earlier run — must not influence this window's diagnosis.
    (cache / "mitmdump.log").write_text(
        "[10:00:00.000][127.0.0.1:1] Client TLS handshake failed. The client does not trust "
        "the proxy's certificate for update.googleapis.com (OpenSSL Error(...))\n",
        encoding="utf-8",
    )

    engine.mock_record("start", "login_flow")

    # The app under test made a real, decrypted call during the window (recorded on the
    # always-on flow log), but nothing landed in the cassette this time.
    since = time.time()
    with pm.flow_log_path(cache).open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "n": 1,
                    "ts": since,
                    "method": "GET",
                    "path": "/api/v1/widgets",
                    "host": "api.example.com",
                    "status": 200,
                    "action": None,
                    "rule": None,
                }
            )
            + "\n"
        )

    out = engine.mock_record("stop", "login_flow")

    assert out["code"] != "proxy_tls", (
        "flows for the app under test decrypted fine in this window — the CA is not the "
        "problem and must not be reported as such"
    )


def test_mitmdump_bin_prefers_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "mitmdump"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(pm.sys, "executable", str(tmp_path / "python"))
    assert pm.mitmdump_bin() == str(fake)
