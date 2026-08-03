"""Proxy mock cassette / rule helpers (no real mitmproxy)."""

from __future__ import annotations

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


def test_mitmdump_bin_prefers_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "mitmdump"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(pm.sys, "executable", str(tmp_path / "python"))
    assert pm.mitmdump_bin() == str(fake)
