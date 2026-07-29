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

    def boom(**kwargs: Any) -> int:
        raise UsageError("mitmproxy is not installed", hint="install [proxy]")

    monkeypatch.setattr(pm, "start_mitm", boom)
    with pytest.raises(UsageError, match="mitmproxy"):
        engine.proxy_start()
