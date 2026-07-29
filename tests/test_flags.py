"""Flags deeplink helper + engine apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flags import build_uri, load_flags_file, parse_assignments
from conftest import FakeDevice, make_config

PKG = "com.example.app"
TEMPLATES = {PKG: "myapp://set-flags?{query}"}


def test_build_uri_from_template() -> None:
    uri = build_uri(PKG, {"hub_experiment": "a", "x": "1"}, TEMPLATES)
    assert uri.startswith("myapp://set-flags?")
    assert "hub_experiment=a" in uri
    assert "x=1" in uri


def test_build_uri_unknown_package() -> None:
    with pytest.raises(UsageError, match="no flags deeplink"):
        build_uri("com.unknown.app", {"k": "v"}, TEMPLATES)


def test_parse_and_load(tmp_path: Path) -> None:
    assert parse_assignments(["a=1", "b=two"]) == {"a": "1", "b": "two"}
    path = tmp_path / "flags.yaml"
    path.write_text(
        f"app: {PKG}\nflags:\n  hub: a\n  other: b\n",
        encoding="utf-8",
    )
    app, flags = load_flags_file(path)
    assert app == PKG
    assert flags == {"hub": "a", "other": "b"}


def test_engine_flags_set(tmp_path: Path) -> None:
    device = FakeDevice()
    cfg = make_config(cache={"dir": str(tmp_path)}, flags={"templates": TEMPLATES})
    engine = Engine(cfg, device=device)
    result = engine.flags_set(PKG, ["hub_experiment=a"], observe=False)
    assert result.ok
    assert any(c[0] == "open_link" for c in device.calls)
    uri = device.calls[-1][1][0]
    assert "myapp://set-flags?" in uri
