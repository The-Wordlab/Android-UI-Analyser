from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.model_control import ModelControlStore, model_context_window
from android_ui_analyser.providers.base import Availability, ChainSpec
from android_ui_analyser.providers.registry import ProviderFactory


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.cache.dir = str(tmp_path)
    config.policy.mode = "advisory"
    config.policy.enabled = False
    config.policy.chain = ["functiongemma", "gemma4"]
    config.policy.strategy = "selective_hybrid"
    return config


def test_dashboard_override_enables_and_kills_passive_policy_immediately(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    control = engine.factory.model_control

    assert control.intercept_override() is None
    assert engine._session_policy_mode() == "off"

    control.update(intercept_enabled=True)
    assert control.intercept_override() is True
    assert engine._session_policy_mode() == "advisory"

    engine._policy_mode_override = "advisory"
    control.update(intercept_enabled=False)
    assert control.intercept_override() is False
    assert engine._session_policy_mode() == "off"


def test_provider_switch_removes_model_from_the_real_policy_chain(tmp_path: Path) -> None:
    config = _config(tmp_path)
    factory = ProviderFactory(config)

    assert factory.build_chain("policy").names() == ["functiongemma", "gemma4"]

    factory.model_control.update(provider="functiongemma", provider_enabled=False)
    assert factory.build_chain("policy").names() == ["gemma4"]

    factory.model_control.update(provider="gemma4", provider_enabled=False)
    assert factory.build_chain("policy").names() == []


def test_model_events_keep_exact_local_exchange_shape_and_shared_operation_id(
    tmp_path: Path,
) -> None:
    control = ModelControlStore(_config(tmp_path))
    control.record(
        {
            "id": "op-1",
            "provider": "functiongemma",
            "phase": "running",
            "input": [{"role": "user", "content": "choose safely"}],
            "input_tokens": 12,
        }
    )
    control.record(
        {
            "id": "op-1",
            "provider": "functiongemma",
            "phase": "complete",
            "output": "candidate 2",
            "output_tokens": 4,
            "duration_ms": 31.5,
        }
    )

    events = control.events()
    assert [event["phase"] for event in events] == ["running", "complete"]
    assert {event["id"] for event in events} == {"op-1"}
    assert events[0]["input"][0]["content"] == "choose safely"
    assert events[1]["output"] == "candidate 2"


def test_context_window_reads_text_only_and_nested_multimodal_configs(tmp_path: Path) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    small.mkdir()
    large.mkdir()
    (small / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    (large / "config.json").write_text(
        json.dumps({"text_config": {"max_position_embeddings": 131072}})
    )

    assert model_context_window({"model_path": str(small)}) == 32768
    assert model_context_window({"model_path": str(large)}) == 131072


def test_agent_sample_uses_the_real_policy_context_and_returns_raw_exchange(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    control = ModelControlStore(config)

    class FakeProvider:
        name = "functiongemma"
        last_error = None
        select_calls = 0

        def is_available(self) -> Availability:
            return Availability(True, "ready")

        def supports_candidate_count(self, count: int) -> bool:
            return 2 <= count <= 4

        def supports_mode(self, mode: str) -> bool:
            return mode == "advisory"

        def supports_handoff(self) -> bool:
            return True

        def select(self, context: Any) -> int:
            self.select_calls += 1
            assert context.goal == "Open Settings"
            assert sorted(candidate.candidate_id for candidate in context.candidates) == [0, 1, 2]
            selected_id = next(
                candidate.candidate_id
                for candidate in context.candidates
                if candidate.purpose == "Open Settings"
            )
            operation_id = "agent-sample"
            control.record(
                {
                    "id": operation_id,
                    "provider": self.name,
                    "source": "agent",
                    "phase": "running",
                    "input": [{"role": "developer", "content": "choose one"}],
                }
            )
            control.record(
                {
                    "id": operation_id,
                    "provider": self.name,
                    "source": "agent",
                    "phase": "complete",
                    "timestamp_ms": int(time.time() * 1000),
                    "output": (
                        "<start_function_call>call:select_candidate"
                        f"{{candidate_id:{selected_id}}}"
                    ),
                    "selected_id": selected_id,
                    "duration_ms": 12.0,
                }
            )
            return selected_id

    class FakeFactory:
        model_control = control

        def __init__(self) -> None:
            self.provider = FakeProvider()

        def create(self, kind: str, name: str) -> FakeProvider:
            assert (kind, name) == ("policy", "functiongemma")
            return self.provider

        def build_chain(self, kind: str) -> ChainSpec:
            assert kind == "policy"
            return ChainSpec(kind=kind, providers=[self.provider])  # type: ignore[list-item]

    factory = FakeFactory()
    engine = Engine(config, factory=factory)  # type: ignore[arg-type]
    result = engine.model_control_agent_test(
        "agent_chain",
        {
            "goal": "Open Settings",
            "phase": "Choose next control",
            "candidates": [
                {"id": 0, "label": "Search", "purpose": "Open Search"},
                {"id": 1, "label": "Settings", "purpose": "Open Settings"},
                {"id": 2, "label": "Profile", "purpose": "Open Profile"},
            ],
            "allow_handoff": True,
        },
    )

    assert result["status"] == "selected", result["decision"]
    assert result["providers"] == ["functiongemma"]
    assert result["decision"]["selection_strategy"] == "selective_hybrid"
    assert result["decision"]["selection_trace"][0]["status"] == "selected"
    assert result["selected_id"] == 1
    assert result["selected_candidate"]["purpose"] == "Open Settings"
    assert "call:select_candidate" in result["exchange"]["output"]
    assert len(result["exchanges"]) == config.policy.primary_reviews
    assert factory.provider.select_calls == config.policy.primary_reviews
    assert result["compiled_context"]["request"]
    assert result["tool_schema"][0]["function"]["name"] == "select_candidate"
