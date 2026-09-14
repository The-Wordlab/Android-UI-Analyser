from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import load_session_state
from conftest import FakeDevice, make_config

CONTRACT = """\
version: 1
checkpoints:
  - id: destination
    description: Reach the destination
    assertions:
      - assert: {rid: destination, exists: true}
      - assert: {text: Loading, absent: true}
cleanup:
  description: Return to home
  assertions:
    - assert: {rid: home, exists: true}
"""


def _observation(serial: str, rid: str = "start") -> AnalyzeResult:
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
            fingerprint=f"{rid}-frame",
        ),
    )


class _Channel:
    def __init__(self, agent: _Agent) -> None:
        self.agent = agent

    def request(self, method: str, params: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.agent.requests.append({"method": method, "params": params, "timeout": timeout})
        return self.agent.reply

    def close(self) -> None:
        self.agent.closed += 1


class _Agent:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.closed = 0

    def is_enabled(self, _serial: str) -> bool:
        return True

    def status(self, _serial: str) -> dict[str, Any]:
        return {"enabled": True, "bound": True, "installed": True}

    def release_uiautomation(self, _serial: str) -> None:
        return None

    def uiautomation_held(self, _serial: str) -> bool:
        return False

    def is_bound(self, _serial: str) -> bool:
        return True

    def open_channel(self, _serial: str, *, timeout: float) -> _Channel:
        return _Channel(self)


def _engine(tmp_path: Path, agent: _Agent) -> Engine:
    config = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
    )
    engine = Engine(config, device=FakeDevice(serial="helper-session-device"))
    capability = engine.platform.capability
    engine.platform.capability = (  # type: ignore[method-assign]
        lambda name: agent if name == "device_agent" else capability(name)
    )
    return engine


def _checkpoint_result(checkpoint_id: str, order: int, frame: int, *checks: str) -> dict[str, Any]:
    definitions = {
        "destination:1": ("rid", "destination", "present"),
        "destination:2": ("text", "Loading", "absent"),
        "cleanup:1": ("rid", "home", "present"),
    }
    return {
        "id": checkpoint_id,
        "order": order,
        "passed": True,
        "evidence_frame": frame,
        "evidence_signature": f"signature-{frame}",
        "package": "dev.aua.fixture",
        "checks": [
            {
                "id": check_id,
                "selector": definitions[check_id][0],
                "value": definitions[check_id][1],
                "required": definitions[check_id][2],
                "passed": True,
            }
            for check_id in checks
        ],
    }


def test_helper_mode_runs_and_finishes_an_ordered_contract_in_one_engine_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _Agent(
        {
            "ok": True,
            "verified": True,
            "ran_on": "device",
            "stop_reason": "verified",
            "checkpoints": [
                _checkpoint_result("destination", 0, 1, "destination:1", "destination:2"),
                _checkpoint_result("cleanup", 1, 2, "cleanup:1"),
            ],
            "metrics": {"requests": 2, "reported_usd": 0.001},
        }
    )
    engine = _engine(tmp_path, agent)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "ephemeral-test-key")

    result = engine.session_start(
        "Reach the destination and return home",
        observation=_observation(engine.device.serial),
        contract_yaml=CONTRACT,
        helper=True,
    )

    assert result["ok"] is True
    assert result["mode"] == "helper"
    assert result["verdict"] == "passed"
    assert result["finished"] is True
    assert result["terminated"] is True
    assert [request["method"] for request in agent.requests] == ["model.run"]
    sent = agent.requests[0]["params"]
    assert "checks" not in sent
    assert [checkpoint["id"] for checkpoint in sent["checkpoints"]] == [
        "destination",
        "cleanup",
    ]
    assert sent["api_key"] == "ephemeral-test-key"
    state = load_session_state(engine.config.cache.dir, session_id=result["session_id"])
    assert state is not None and state.finished_ms is not None
    assert [phase.proof.source for phase in state.phases if phase.proof] == [
        "helper_contract_checks",
        "helper_contract_checks",
    ]


def test_helper_mode_rebases_the_suffix_after_bootstrap_proves_the_first_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _Agent(
        {
            "ok": True,
            "verified": True,
            "ran_on": "device",
            "stop_reason": "verified",
            "checkpoints": [_checkpoint_result("cleanup", 0, 2, "cleanup:1")],
            "metrics": {"requests": 1, "reported_usd": 0.0005},
        }
    )
    engine = _engine(tmp_path, agent)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "ephemeral-test-key")

    result = engine.session_start(
        "Reach the destination and return home",
        observation=_observation(engine.device.serial, "destination"),
        contract_yaml=CONTRACT,
        helper=True,
    )

    assert result["ok"] is True
    assert agent.requests[0]["params"]["checkpoints"][0]["id"] == "cleanup"
    assert agent.requests[0]["params"]["checkpoints"][0]["order"] == 0


def test_helper_mode_returns_fail_without_a_host_agent_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _Agent(
        {
            "ok": False,
            "verified": False,
            "ran_on": "device",
            "stop_reason": "step_limit",
            "checkpoints": [
                {
                    "id": "destination",
                    "order": 0,
                    "passed": False,
                    "evidence_frame": None,
                    "evidence_signature": None,
                    "package": None,
                    "checks": [],
                }
            ],
            "metrics": {"requests": 4, "reported_usd": 0.003},
        }
    )
    engine = _engine(tmp_path, agent)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "ephemeral-test-key")

    result = engine.session_start(
        "Reach the destination and return home",
        observation=_observation(engine.device.serial),
        contract_yaml=CONTRACT,
        helper=True,
    )

    assert result["ok"] is False
    assert result["code"] == "helper_contract_failed"
    assert result["verdict"] == "failed"
    assert result["terminated"] is True
    assert result["goal_progress"]["status"] == "terminated_incomplete"
    assert [request["method"] for request in agent.requests] == ["model.run"]


def test_helper_mode_rejects_a_mismatched_device_proof_and_still_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoofed = _checkpoint_result(
        "destination", 0, 1, "destination:1", "destination:2"
    )
    spoofed["checks"][0]["value"] = "a-different-control"
    agent = _Agent(
        {
            "ok": True,
            "verified": True,
            "ran_on": "device",
            "stop_reason": "verified",
            "checkpoints": [spoofed],
            "metrics": {"requests": 1, "reported_usd": 0.0005},
        }
    )
    engine = _engine(tmp_path, agent)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "ephemeral-test-key")

    result = engine.session_start(
        "Reach the destination and return home",
        observation=_observation(engine.device.serial),
        contract_yaml=CONTRACT,
        helper=True,
    )

    assert result["ok"] is False
    assert result["terminated"] is True
    assert result["goal_progress"]["status"] == "terminated_incomplete"


def test_helper_mode_requires_a_contract_before_touching_a_target(tmp_path: Path) -> None:
    agent = _Agent({})
    engine = _engine(tmp_path, agent)

    with pytest.raises(UsageError) as raised:
        engine.session_start("Inspect the fixture", helper=True)

    assert raised.value.code == "helper_unsupported_contract"
    assert agent.requests == []


def test_helper_mode_rejects_a_rich_contract_instead_of_weakening_it(tmp_path: Path) -> None:
    agent = _Agent({})
    engine = _engine(tmp_path, agent)
    contract = """\
checkpoints:
  - id: enabled
    description: Prove the control is enabled
    assertions:
      - assert: {rid: continue, enabled: true}
"""

    with pytest.raises(UsageError) as raised:
        engine.session_start("Prove enabled state", contract_yaml=contract, helper=True)

    assert raised.value.code == "helper_unsupported_contract"
    assert "unsupported predicates: enabled" in raised.value.message
    assert agent.requests == []
