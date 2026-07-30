"""Tests for native phases 2–6 host slices."""

from __future__ import annotations

import base64
from pathlib import Path

from android_ui_analyser.binary_dump import pack_analyze_dict, unpack_b64, unpack_frame
from android_ui_analyser.config import Config, DaemonCfg
from android_ui_analyser.daemon import socket_path
from android_ui_analyser.schema import (
    AnalyzeResult,
    Element,
    Meta,
    OutputFormat,
    PathKind,
    Screen,
    ScreenSource,
    Source,
    Tier,
)


def _result(*, unchanged: bool = False) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=100, height=200, package="app", source=ScreenSource.hierarchy),
        elements=[
            Element(
                id=1,
                type="TextView",
                text="Hi",
                bounds=(0, 0, 10, 10),
                center=(5, 5),
                clickable=True,
                enabled=True,
                source=Source.hierarchy,
            )
        ],
        meta=Meta(
            duration_ms=12,
            tier_used=Tier.hierarchy,
            path=PathKind.hierarchy,
            unchanged=unchanged,
            fingerprint="abc",
            via="hierarchy-unchanged" if unchanged else "hierarchy",
            element_diff={"added": [], "removed": [], "changed": []},
        ),
    )


def test_delta_format_omits_elements_when_unchanged() -> None:
    data = _result(unchanged=True).as_dict(OutputFormat.delta)
    assert data["elements"] == []
    assert data["meta"]["unchanged"] is True
    assert data["meta"]["fingerprint"] == "abc"


def test_delta_format_keeps_elements_when_changed() -> None:
    data = _result(unchanged=False).as_dict(OutputFormat.delta)
    assert len(data["elements"]) == 1


def test_msgpack_roundtrip() -> None:
    result = _result()
    b64 = result.render(OutputFormat.msgpack)
    decoded = unpack_b64(b64)
    assert decoded["screen"]["package"] == "app"
    assert decoded["elements"][0]["text"] == "Hi"
    raw = base64.b64decode(b64)
    assert unpack_frame(raw)["meta"]["fingerprint"] == "abc"


def test_pack_analyze_dict_magic() -> None:
    blob = pack_analyze_dict({"hello": 1})
    assert blob.startswith(b"AUA1")


def test_per_serial_socket_path(tmp_path: Path) -> None:
    cfg = Config(daemon=DaemonCfg(socket=str(tmp_path / "daemon.sock")))
    assert socket_path(cfg) == str(tmp_path / "daemon.sock")
    assert socket_path(cfg, serial="emulator-5554").endswith("daemon.sock.emulator-5554")
    assert socket_path(cfg, serial="bad/serial:1").endswith("daemon.sock.bad_serial_1")


def test_ocr_default_favours_accuracy_and_yolo_uses_the_gpu() -> None:
    """Assert the CODE defaults, not whatever this machine's config files resolve to.

    `load_config()` layers in the real user + project configs, so this passed or failed by
    developer machine: a checked-out `.android-ui-analyser.yaml` pinning a value made it fail
    with nothing wrong in the code.

    `apple_vision` deliberately defaults to `accurate`: OCR only runs when the accessibility
    tree could not read the screen, so it has no fallback behind it, and `fast` truncates —
    the repo's own smoke image reads "Hello" as "Hel".
    """
    cfg = Config()
    assert cfg.models["apple_vision"]["recognition_level"] == "accurate"
    assert cfg.models["yolo"]["device"] == "mps"
    assert cfg.perf.differential is True
    assert cfg.perf.skip_unchanged_analyze is True


def test_hierarchy_fbs_schema_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "schemas" / "hierarchy.fbs").read_text()
    assert "table AnalyzeResult" in text
    assert "root_type AnalyzeResult" in text


def test_push_ws_accept_key() -> None:
    from android_ui_analyser.push import _ws_accept

    # RFC6455 example key
    assert _ws_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
