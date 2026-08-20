from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import load_session_state
from conftest import FakeDevice, make_config

CONTRACT = """\
version: 1
checkpoints:
  - id: result_ready
    description: Confirm the result
    assertions:
      - assert: {rid: result, exists: true}
cleanup:
  description: Restore the fixture home
  assertions:
    - assert: {rid: home, exists: true}
"""


def _frame(serial: str, rid: str, fingerprint: str) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="dev.aua.fixture",
            activity=".MainActivity",
            source="hierarchy",
        ),
        elements=[
            Element(
                id=1,
                type="android.widget.TextView",
                resource_id=f"dev.aua.fixture:id/{rid}",
                bounds=(0, 0, 500, 100),
                center=(250, 50),
                source=Source.hierarchy,
            )
        ],
        meta=Meta(
            duration_ms=2,
            tier_used="hierarchy",
            path="hierarchy",
            device_serial=serial,
            fingerprint=fingerprint,
        ),
    )


def _engine(tmp_path: Path) -> Engine:
    config = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    return Engine(config, device=FakeDevice(serial="contract-device"))


def test_contract_finish_refuses_cleanup_until_fresh_assertions_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path)
    result = _frame(engine.device.serial, "result", "result-frame")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: result)

    started = engine.session_start("Verify public fixture", contract_yaml=CONTRACT)
    assert started["contract_verdict"]["ok"] is True
    assert started["goal_progress"]["current"]["id"] == "cleanup"

    refused = engine.session_finish(started["session_id"])
    assert refused["ok"] is False
    assert refused["code"] == "contract_incomplete"
    assert refused["terminated"] is False
    assert refused["missing_checkpoints"][0]["id"] == "cleanup"
    persisted = load_session_state(engine.config.cache.dir, session_id=started["session_id"])
    assert persisted is not None and persisted.finished_ms is None

    home = _frame(engine.device.serial, "home", "home-frame")
    progressed = engine.session_progress(started["session_id"], observation=home)
    assert progressed["contract_verdict"]["ok"] is True
    assert progressed["goal_progress"]["done"] is True

    finished = engine.session_finish(started["session_id"])
    assert finished["ok"] is True
    assert finished["finished"] is True
    assert finished["terminated"] is True


def test_contract_manual_phase_completion_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path)
    initial = _frame(engine.device.serial, "other", "other-frame")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: initial)
    engine.session_start("Verify public fixture", contract_yaml=CONTRACT)

    with pytest.raises(UsageError, match="manual evidence cannot complete it"):
        engine.session_mark_phase("result_ready", "I saw it")


def test_allow_incomplete_explicitly_terminates_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path)
    initial = _frame(engine.device.serial, "other", "other-frame")
    monkeypatch.setattr(engine, "analyze", lambda **_kwargs: initial)
    started = engine.session_start("Verify public fixture", contract_yaml=CONTRACT)

    finished = engine.session_finish(started["session_id"], allow_incomplete=True)

    assert finished["ok"] is True
    assert finished["terminated"] is True
    assert finished["finished"] is False
