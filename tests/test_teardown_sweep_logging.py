"""A command must not claim successful cleanup when a foreign platform undo failed."""

import logging

import pytest

from android_ui_analyser import device_ledger, teardown
from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.identity import TargetRef
from conftest import FakeDevice, make_config


@pytest.mark.parametrize("platform,failed,undone,warning", [
    ("ios", ["offline"], [], False),
    ("ios", [], [{"kind": "location"}], False),
    ("android", ["offline"], [], True),
    ("android", [], [{"kind": "location"}], True),
    ("android", [], [], False),
])
def test_cleanup_logs_are_truthful_and_platform_scoped(monkeypatch, caplog, platform, failed, undone, warning):
    config = make_config(teardown={"enabled": True, "sweep_on_command": True})
    engine = Engine(config, device=FakeDevice())
    monkeypatch.setattr(engine, "_reclaim_owned_virtual_targets", lambda: None)
    monkeypatch.setattr(device_ledger, "pending_targets", lambda: [TargetRef(platform, "fixture")])
    monkeypatch.setattr(teardown, "sweep", lambda **kw: [
        {"platform": platform, "serial": "fixture", "failed": failed, "undone": undone},
    ])
    with caplog.at_level(logging.WARNING):
        engine._sweep_abandoned_devices(skip=None)
    assert bool(caplog.records) is warning
    if failed:
        assert "reset abandoned" not in caplog.text
    if warning and failed:
        assert "cleanup pending" in caplog.text
