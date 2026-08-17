"""Host-only integration tests for optional guarded goal-session policy advice."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from android_ui_analyser import journal
from android_ui_analyser.coaching import decorate_result
from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.base import Availability, ChainSpec
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import mark_phase_complete
from conftest import FakeDevice, make_config


class _Selector:
    name = "fixture_policy"

    def __init__(
        self,
        choose: Callable[[Any], int | None] | None = None,
    ) -> None:
        self.choose = choose or (lambda context: context.candidates[0].candidate_id)
        self.availability_calls = 0
        self.select_calls = 0
        self.contexts: list[Any] = []

    def is_available(self) -> Availability:
        self.availability_calls += 1
        return Availability(True, "fixture ready")

    def select(self, context: Any) -> int | None:
        self.select_calls += 1
        self.contexts.append(context)
        return self.choose(context)


class _Factory:
    def __init__(self, selector: _Selector | None, *, refuse: bool = False) -> None:
        self.selector = selector
        self.refuse = refuse
        self.build_calls = 0

    def build_chain(self, kind: str) -> ChainSpec:
        self.build_calls += 1
        if self.refuse:
            raise AssertionError("disabled policy must not resolve a provider")
        assert kind == "policy"
        return ChainSpec(kind=kind, providers=[self.selector] if self.selector else [])


def _element(
    element_id: int,
    text: str,
    *,
    rid: str | None,
    type_name: str = "android.widget.Button",
    **values: Any,
) -> Element:
    top = 200 + element_id * 50
    return Element(
        id=element_id,
        type=type_name,
        text=text,
        resource_id=rid,
        bounds=(20, top, 900, top + 44),
        center=(460, top + 22),
        clickable=True,
        source=Source.hierarchy,
        **values,
    )


def _observation(
    serial: str,
    elements: list[Element],
    *,
    fingerprint: str | None = "frame-policy-1",
    stale_risk: str | None = None,
) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(
            width=1080,
            height=2400,
            package="com.example.catalog",
            source="hierarchy",
        ),
        elements=elements,
        meta=Meta(
            duration_ms=8,
            tier_used="hierarchy",
            path="hierarchy",
            known_screen="catalog_home",
            device_serial=serial,
            fingerprint=fingerprint,
            stale_risk=stale_risk,
        ),
    )


def _engine(
    tmp_path: Path,
    mode: str,
    selector: _Selector | None,
    *,
    refuse_factory: bool = False,
    serial: str = "policy-host-only",
) -> tuple[Engine, _Factory]:
    config = make_config(
        cache={"dir": str(tmp_path / "cache")},
        memory={"enabled": False, "dir": str(tmp_path / "memory")},
        policy={
            "enabled": mode != "off",
            "mode": mode,
            "chain": ["fixture_policy"],
            "max_candidates": 4,
        },
    )
    factory = _Factory(selector, refuse=refuse_factory)
    return Engine(config, device=FakeDevice(serial=serial), factory=factory), factory


def _alternatives(serial: str) -> AnalyzeResult:
    return _observation(
        serial,
        [
            _element(1, "Grammar", rid="com.example.catalog:id/openGrammar"),
            _element(2, "Mathematics", rid="com.example.catalog:id/openMathematics"),
        ],
    )


def _choose_mathematics(context: Any) -> int:
    return next(
        candidate.candidate_id
        for candidate in context.candidates
        if "Mathematics" in candidate.purpose
    )


def test_policy_off_does_not_resolve_a_factory_or_change_bootstrap(tmp_path: Path) -> None:
    engine, factory = _engine(tmp_path, "off", None, refuse_factory=True)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    assert "policy" not in result
    assert "policy_suggestion" not in result
    assert result["recommended_call"]["kind"] == "manual_action"
    assert factory.build_calls == 0
    assert engine.device.calls == []


def test_shadow_selects_from_opaque_current_candidates_without_exposing_a_call(
    tmp_path: Path,
) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "shadow", selector)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    assert result["policy"] == {
        "mode": "shadow",
        "status": "selected",
        "provider": "fixture_policy",
        "model_used": True,
        "candidate_count": 2,
        "eligible_candidate_ids": result["policy"]["eligible_candidate_ids"],
        "selected_candidate_id": result["policy"]["selected_candidate_id"],
    }
    assert "recommended_call" not in result["policy"]
    assert "policy_suggestion" not in result
    assert result["recommended_call"]["kind"] == "manual_action"
    context = selector.contexts[0]
    assert context.session_id == result["session_id"]
    assert context.phase == "phase_1"
    assert context.observation_fingerprint == "frame-policy-1"
    assert {candidate.candidate_id for candidate in context.candidates} == {0, 1}
    assert all(candidate.session_id == result["session_id"] for candidate in context.candidates)
    assert all(candidate.phase == "phase_1" for candidate in context.candidates)
    assert all(
        candidate.observation_fingerprint == "frame-policy-1" for candidate in context.candidates
    )
    assert engine.device.calls == []


def test_advisory_is_separate_and_never_replaces_or_executes_deterministic_call(
    tmp_path: Path,
) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "advisory", selector)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    deterministic = result["recommended_call"]
    suggestion = result["policy_suggestion"]
    assert deterministic["kind"] == "manual_action"
    assert suggestion["kind"] == "policy_advisory"
    assert suggestion["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"rid": "openMathematics"},
    }
    assert suggestion["cli"] == "aua tap-and-analyze --rid openMathematics"
    assert result["policy"]["status"] == "selected"
    assert "recommended_call" not in result["policy"]
    assert engine.device.calls == []


def test_candidate_ids_and_order_do_not_depend_on_element_order(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "off", None, refuse_factory=True)
    observation = _alternatives(engine.device.serial)
    started = engine.session_start("Open Grammar or Mathematics", observation=observation)
    state = engine._session_state(started["session_id"])  # noqa: SLF001 - integration contract
    phase = next(item for item in state.phases if item.status != "completed")

    first = engine._policy_tap_candidates(state, phase, observation)  # noqa: SLF001
    reversed_observation = observation.model_copy(
        update={"elements": list(reversed(observation.elements))}
    )
    second = engine._policy_tap_candidates(  # noqa: SLF001
        state,
        phase,
        reversed_observation,
    )

    assert [(candidate.candidate_id, candidate.call) for candidate in first] == [
        (candidate.candidate_id, candidate.call) for candidate in second
    ]
    assert {candidate.candidate_id for candidate in first} == set(range(len(first)))


def test_compiler_withholds_destructive_secret_dynamic_ambiguous_and_mutating_controls(
    tmp_path: Path,
) -> None:
    selector = _Selector()
    engine, _factory = _engine(tmp_path, "advisory", selector)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Grammar", rid="com.example.catalog:id/openGrammar"),
            _element(2, "Delete Grammar", rid="com.example.catalog:id/deleteGrammar"),
            _element(3, "person@example.com", rid="com.example.catalog:id/grammarProfile"),
            _element(4, "Grammar 12 hours ago", rid="com.example.catalog:id/recentGrammar"),
            _element(5, "Grammar", rid=None),
            _element(6, "Grammar", rid=None),
            _element(
                7,
                "Grammar",
                rid="com.example.catalog:id/grammarToggle",
                checkable=True,
                checked=False,
            ),
            _element(
                8,
                "Grammar",
                rid="com.example.catalog:id/grammarInput",
                type_name="android.widget.EditText",
            ),
        ],
    )

    result = engine.session_start("Open Grammar", observation=observation)

    assert result["policy"]["status"] == "deterministic"
    assert result["policy"]["candidate_count"] == 1
    assert "policy_suggestion" not in result
    assert selector.availability_calls == 0
    assert selector.select_calls == 0
    state = engine._session_state(result["session_id"])  # noqa: SLF001 - integration contract
    phase = next(item for item in state.phases if item.status != "completed")
    candidates = engine._policy_tap_candidates(state, phase, observation)  # noqa: SLF001
    assert [candidate.call for candidate in candidates] == [
        {"tool": "tap_and_analyze", "arguments": {"rid": "openGrammar"}}
    ]
    assert all("person@example.com" not in candidate.purpose for candidate in candidates)
    assert all("hours ago" not in candidate.purpose for candidate in candidates)


def test_zero_and_one_candidate_paths_do_not_touch_the_model(tmp_path: Path) -> None:
    zero_selector = _Selector()
    zero_engine, zero_factory = _engine(
        tmp_path / "zero",
        "advisory",
        zero_selector,
        serial="policy-zero",
    )
    zero = zero_engine.session_start(
        "Open Grammar",
        observation=_observation(
            zero_engine.device.serial,
            [_element(1, "Unrelated", rid="com.example.catalog:id/unrelated")],
        ),
    )
    assert zero["policy"]["status"] == "no_candidate"
    assert "policy_suggestion" not in zero
    assert zero_selector.availability_calls == zero_selector.select_calls == 0
    assert zero_factory.build_calls == 0

    one_selector = _Selector()
    one_engine, one_factory = _engine(
        tmp_path / "one",
        "advisory",
        one_selector,
        serial="policy-one",
    )
    one = one_engine.session_start(
        "Open Grammar",
        observation=_observation(
            one_engine.device.serial,
            [_element(1, "Grammar", rid="com.example.catalog:id/openGrammar")],
        ),
    )
    assert one["policy"]["status"] == "deterministic"
    assert one["policy"]["model_used"] is False
    assert "policy_suggestion" not in one
    assert one_selector.availability_calls == one_selector.select_calls == 0
    assert one_factory.build_calls == 0


def test_unfingerprinted_or_stale_observation_fails_closed_without_model_use(
    tmp_path: Path,
) -> None:
    selector = _Selector(_choose_mathematics)
    engine, factory = _engine(tmp_path, "advisory", selector)
    stale = _alternatives(engine.device.serial)
    stale.meta.fingerprint = None
    stale.meta.stale_risk = "the prior action outcome is unknown"

    result = engine.session_start("Open Grammar or Mathematics", observation=stale)

    assert result["policy"]["status"] == "skipped_unbound_observation"
    assert "policy_suggestion" not in result
    assert factory.build_calls == 0
    assert selector.availability_calls == selector.select_calls == 0


def test_session_progress_evaluates_the_new_fresh_frame_after_bootstrap_skip(
    tmp_path: Path,
) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    unbound = _alternatives(engine.device.serial)
    unbound.meta.fingerprint = None
    started = engine.session_start("Open Grammar or Mathematics", observation=unbound)

    progressed = engine.session_progress(
        started["session_id"],
        observation=_alternatives(engine.device.serial),
    )

    assert started["policy"]["status"] == "skipped_unbound_observation"
    assert progressed["policy"]["status"] == "selected"
    assert progressed["policy_suggestion"]["mcp"]["arguments"] == {"rid": "openMathematics"}
    assert progressed["goal_progress"]["next_call"] == started["goal_progress"]["next_call"]
    assert selector.select_calls == 1


def test_action_decoration_preserves_policy_advisory_in_goal_progress(tmp_path: Path) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    observation = _alternatives(engine.device.serial)
    started = engine.session_start("Open Grammar or Mathematics", observation=observation)

    decorated = decorate_result(
        engine,
        "tap",
        {"ok": True, "observation": observation.model_dump(mode="json")},
        current_recorded=False,
    )

    progress = decorated["goal_progress"]
    assert progress["policy"]["status"] == "selected"
    assert progress["policy_suggestion"]["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"rid": "openMathematics"},
    }
    assert started["recommended_call"] == progress["next_call"]


def test_engine_session_results_are_not_refreshed_or_generated_twice(tmp_path: Path) -> None:
    calls: list[str] = []
    engine = SimpleNamespace(
        config=SimpleNamespace(
            cache=SimpleNamespace(dir=str(tmp_path / "cache")),
            device=SimpleNamespace(serial="policy-host-only"),
        ),
        _lease_serial="policy-host-only",
        _lease_owner_resolved=None,
        _session_state=lambda *_args, **_kwargs: calls.append("session_state"),
        session_progress=lambda *_args, **_kwargs: calls.append("session_progress"),
    )

    for command in ("session_start", "session_progress", "session_finish"):
        result = {
            "goal_progress": {"done": False, "current": {"id": "phase_1"}},
            "policy": {"status": "selected", "model_used": True},
            "policy_suggestion": {"mcp": {"tool": "tap_and_analyze", "arguments": {}}},
        }

        decorated = decorate_result(
            engine,
            command,
            result,
            current_recorded=False,
        )

        assert decorated["goal_progress"]["policy"] == result["policy"]
        assert decorated["goal_progress"]["policy_suggestion"] == result["policy_suggestion"]

    assert calls == []


def test_provider_failure_is_structured_and_never_breaks_session_bootstrap(
    tmp_path: Path,
) -> None:
    def fail(_context: Any) -> int:
        raise RuntimeError("raw provider detail must not escape")

    selector = _Selector(fail)
    engine, _factory = _engine(tmp_path, "advisory", selector)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    assert result["recommended_call"]["kind"] == "manual_action"
    assert result["policy"]["status"] == "provider_error"
    assert result["policy"]["model_used"] is True
    assert result["policy"]["error"] == "policy provider failed closed"
    assert "policy_suggestion" not in result
    assert "raw provider detail" not in repr(result["policy"])


def test_phase_change_during_inference_withholds_advisory_suggestion(tmp_path: Path) -> None:
    engine_ref: list[Engine] = []

    def complete_phase(context: Any) -> int:
        engine = engine_ref[0]
        state = engine._session_state(context.session_id)  # noqa: SLF001 - integration contract
        mark_phase_complete(
            engine.config.cache.dir,
            state,
            phase_id=context.phase,
            evidence="Grammar or Mathematics navigation completed",
        )
        return context.candidates[0].candidate_id

    selector = _Selector(complete_phase)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine_ref.append(engine)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    assert result["policy"]["status"] == "rejected_stale_context"
    assert "selected_candidate_id" not in result["policy"]
    assert "policy_suggestion" not in result
    assert result["recommended_call"]["kind"] == "manual_action"


def test_newer_fingerprint_during_inference_withholds_advisory_suggestion(tmp_path: Path) -> None:
    engine_ref: list[Engine] = []

    def replace_frame(context: Any) -> int:
        engine = engine_ref[0]
        engine._last_analyze_result = _observation(  # noqa: SLF001 - simulate concurrent frame
            engine.device.serial,
            [_element(9, "Grammar", rid="com.example.catalog:id/openGrammar")],
            fingerprint="frame-policy-newer",
        )
        return context.candidates[0].candidate_id

    selector = _Selector(replace_frame)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine_ref.append(engine)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    assert result["policy"]["status"] == "rejected_stale_context"
    assert result["policy"]["error"] == "a newer observation replaced the policy input frame"
    assert "policy_suggestion" not in result


def test_policy_inference_does_not_create_an_extra_top_level_journal_call(tmp_path: Path) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "shadow", selector)
    started = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )
    journal.record(
        cache_dir=engine.config.cache.dir,
        serial=engine.device.serial,
        source="cli",
        owner=None,
        cmd="session_start",
        ok=True,
        result=started,
        extra={"session_id": started["session_id"], "invocation_id": "inv-policy-start"},
    )

    review = engine.session_review(started["session_id"])
    events = journal.read_since(engine.config.cache.dir, engine.device.serial)

    assert [event["cmd"] for event in events] == ["session_start"]
    assert review["accounting"]["journal_events"] == 1
    assert review["accounting"]["top_level_calls"] == 1
