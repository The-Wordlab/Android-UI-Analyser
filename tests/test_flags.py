"""Flags deeplink helper + engine apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.flags import build_uri, load_flags_file, parse_assignments
from conftest import FakeDevice, make_config


def test_build_uri_luzia() -> None:
    uri = build_uri("co.thewordlab.luzia.dev", {"apps_hub_experiment": "a", "x": "1"})
    assert uri.startswith("luzia-test://set-flags?")
    assert "apps_hub_experiment=a" in uri
    assert "x=1" in uri


def test_build_uri_unknown_package() -> None:
    with pytest.raises(UsageError, match="no flags deeplink"):
        build_uri("com.unknown.app", {"k": "v"})


def test_parse_and_load(tmp_path: Path) -> None:
    assert parse_assignments(["a=1", "b=two"]) == {"a": "1", "b": "two"}
    path = tmp_path / "flags.yaml"
    path.write_text(
        "app: co.thewordlab.luzia.dev\nflags:\n  hub: a\n  other: b\n",
        encoding="utf-8",
    )
    app, flags = load_flags_file(path)
    assert app == "co.thewordlab.luzia.dev"
    assert flags == {"hub": "a", "other": "b"}


def test_engine_flags_set(tmp_path: Path) -> None:
    device = FakeDevice()
    engine = Engine(make_config(cache={"dir": str(tmp_path)}), device=device)
    result = engine.flags_set(
        "co.thewordlab.luzia.dev",
        ["apps_hub_experiment=a"],
        observe=False,
    )
    assert result.ok
    assert any(c[0] == "open_link" for c in device.calls)
    uri = device.calls[-1][1][0]
    assert "luzia-test://set-flags?" in uri
