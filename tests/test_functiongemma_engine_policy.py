"""Host-only integration tests for optional guarded goal-session policy advice."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from android_ui_analyser import journal
from android_ui_analyser.coaching import decorate_result
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.providers.base import Availability, ChainSpec
from android_ui_analyser.schema import ActionResult, AnalyzeResult, Element, Meta, Screen, Source
from android_ui_analyser.session import mark_phase_complete
from conftest import FakeDevice, make_config


class _Selector:
    name = "fixture_policy"

    def __init__(
        self,
        choose: Callable[[Any], int | None] | None = None,
        *,
        handoff: bool = False,
    ) -> None:
        self.choose = choose or (lambda context: context.candidates[0].candidate_id)
        self.handoff = handoff
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

    def supports_handoff(self) -> bool:
        return self.handoff

    def supports_mode(self, mode: str) -> bool:
        return mode in {"shadow", "advisory"}


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


class _HybridFactory:
    def __init__(self, *selectors: _Selector) -> None:
        self.selectors = list(selectors)
        self.build_calls = 0

    def build_chain(self, kind: str) -> ChainSpec:
        self.build_calls += 1
        assert kind == "policy"
        return ChainSpec(kind=kind, providers=self.selectors)


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
        "compiler": {
            "schema_version": 1,
            "target_term_count": 2,
            "stages": {
                "elements": 2,
                "enabled_clickable": 2,
                "safe_control": 2,
                "stable_selector": 2,
                "frame_selector": 0,
                "non_destructive": 2,
                "target_matched": 2,
                "offered": 2,
            },
            "recommended_call_offered": True,
        },
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


def test_safe_visible_scope_offers_guarded_distractors_for_semantic_selection(
    tmp_path: Path,
) -> None:
    selector = _Selector(
        lambda context: next(
            candidate.candidate_id
            for candidate in context.candidates
            if "Grammar" in candidate.purpose
        )
    )
    engine, _factory = _engine(tmp_path, "shadow", selector)
    engine.config.policy.candidate_scope = "safe_visible"
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Grammar", rid="com.example.catalog:id/openGrammar"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
            _element(3, "Settings", rid="com.example.catalog:id/openSettings"),
        ],
    )

    result = engine.session_start("Open Grammar", observation=observation)

    assert result["policy"]["status"] == "selected"
    assert result["policy"]["candidate_count"] == 3
    assert result["policy"]["compiler"]["stages"]["target_matched"] == 1
    assert result["policy"]["compiler"]["stages"]["offered"] == 3
    assert selector.select_calls == 1


def test_selective_hybrid_engine_uses_reviewer_only_for_unrelated_primary(
    tmp_path: Path,
) -> None:
    primary = _Selector(
        lambda context: next(
            candidate.candidate_id
            for candidate in context.candidates
            if "Ideas" in candidate.purpose
        )
    )
    primary.name = "fast_fixture"
    reviewer = _Selector(
        lambda context: next(
            candidate.candidate_id
            for candidate in context.candidates
            if "Settings" in candidate.purpose
        )
    )
    reviewer.name = "reviewer_fixture"
    engine, _factory = _engine(tmp_path, "advisory", primary)
    engine.factory = _HybridFactory(primary, reviewer)
    engine.config.policy.strategy = "selective_hybrid"
    engine.config.policy.candidate_scope = "safe_visible"
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Settings", rid="com.example.catalog:id/openSettings"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
    )

    result = engine.session_start("Open Settings", observation=observation)

    assert result["policy"]["selection_strategy"] == "selective_hybrid"
    assert result["policy"]["provider"] == "reviewer_fixture"
    assert result["policy"]["selection_trace"][0]["status"] == "review_required"
    assert result["policy_suggestion"]["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"rid": "openSettings"},
    }
    assert primary.select_calls == 2
    assert reviewer.select_calls == 3


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
    assert result["policy"]["compiler"] == {
        "schema_version": 1,
        "target_term_count": 1,
        "stages": {
            "elements": 8,
            "enabled_clickable": 8,
            "safe_control": 6,
            "stable_selector": 2,
            "frame_selector": 0,
            "non_destructive": 1,
            "target_matched": 1,
            "offered": 1,
        },
        "recommended_call_offered": True,
    }
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
    assert zero["policy_handoff"] == {
        "kind": "policy_handoff",
        "reason_code": "no_guard_approved_candidate",
        "reason": (
            "The optional local policy found no supplied guard-approved action that "
            "directly advances the active goal. It has executed nothing; return control "
            "to the parent agent for a fresh observation, broader recovery, or a clear "
            "target-absent result."
        ),
        "model_used": False,
        "executes": False,
    }
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


def test_authenticated_model_handoff_is_non_executing_and_separate_from_advice(
    tmp_path: Path,
) -> None:
    selector = _Selector(lambda _context: -1, handoff=True)
    engine, _factory = _engine(tmp_path, "advisory", selector)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    assert selector.contexts[0].allow_handoff is True
    assert result["policy"]["status"] == "handoff"
    assert result["policy"]["model_used"] is True
    assert result["policy"]["handoff_reason"] == "no_supplied_candidate_advances_goal"
    assert result["policy_handoff"]["kind"] == "policy_handoff"
    assert result["policy_handoff"]["model_used"] is True
    assert result["policy_handoff"]["executes"] is False
    assert "policy_suggestion" not in result
    assert result["recommended_call"]["kind"] == "manual_action"
    assert engine.device.calls == []


def test_compound_goal_alternatives_do_not_contaminate_requested_target(
    tmp_path: Path,
) -> None:
    selector = _Selector()
    engine, factory = _engine(tmp_path, "advisory", selector)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Grammar tools Language lessons", rid="com.example.catalog:id/grammar"),
            _element(
                2,
                "Mathematics Manage applications and notices",
                rid="com.example.catalog:id/mathematics",
            ),
            _element(
                3,
                "History archive Review saved lessons",
                rid="com.example.catalog:id/historyArchive",
            ),
            _element(
                4,
                "Physics laboratory Connected experiments",
                rid="com.example.catalog:id/physics",
            ),
        ],
    )

    result = engine.session_start(
        (
            "Open History archive from these Example destinations: Grammar tools, "
            "Mathematics, History archive, Physics laboratory."
        ),
        observation=observation,
    )

    assert result["policy"]["status"] == "deterministic"
    assert result["policy"]["candidate_count"] == 1
    assert result["policy"]["model_used"] is False
    assert "policy_suggestion" not in result
    assert result["recommended_call"]["mcp"] == {
        "tool": "tap_and_analyze",
        "arguments": {"rid": "historyArchive"},
    }
    assert factory.build_calls == 0
    assert selector.availability_calls == selector.select_calls == 0
    state = engine._session_state(result["session_id"])  # noqa: SLF001
    phase = next(item for item in state.phases if item.status != "completed")
    candidates = engine._policy_tap_candidates(state, phase, observation)  # noqa: SLF001
    assert [candidate.call for candidate in candidates] == [
        {"tool": "tap_and_analyze", "arguments": {"rid": "historyArchive"}}
    ]


def test_contract_proof_words_do_not_offer_an_unrelated_search_control(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "off", None, refuse_factory=True)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Clear text", rid="com.example.catalog:id/openSearchClearButton"),
            _element(2, "Example archive", rid="com.example.catalog:id/exampleArchive"),
            _element(3, "Example lessons", rid="com.example.catalog:id/exampleLessons"),
        ],
    )
    diagnostics: dict[str, Any] = {}

    candidates = engine._policy_tap_candidates(  # noqa: SLF001
        SimpleNamespace(session_id="proof-session"),
        SimpleNamespace(
            id="archive_checkpoint",
            objective="Prove the real Example archive destination, not the search result",
        ),
        observation,
        diagnostics=diagnostics,
    )

    assert diagnostics["target_term_count"] == 2
    assert {candidate.call["arguments"]["rid"] for candidate in candidates} == {
        "exampleArchive",
        "exampleLessons",
    }
    assert all("Clear text" not in candidate.purpose for candidate in candidates)


def test_navigation_object_precedes_its_later_proof_clause() -> None:
    assert Engine._policy_target_terms(  # noqa: SLF001 - compiler contract
        "Tap SOUND and vibration and prove the resulting page by showing Media volume"
    ) == ["sound", "vibration"]


def test_unique_clickable_row_can_use_a_fingerprint_bound_id_when_title_duplicates_text(
    tmp_path: Path,
) -> None:
    engine, _factory = _engine(tmp_path, "off", None, refuse_factory=True)
    observation = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Example & details",
                bounds=(20, 200, 900, 244),
                center=(460, 222),
                clickable=False,
                source=Source.hierarchy,
            ),
            _element(2, "Example & details", rid=None),
            _element(3, "Example locking detail", rid="com.example.catalog:id/lockingDetail"),
            _element(4, "Example click details", rid="com.example.catalog:id/clickDetails"),
        ],
    )
    diagnostics: dict[str, Any] = {}

    candidates = engine._policy_tap_candidates(  # noqa: SLF001
        SimpleNamespace(session_id="frame-session"),
        SimpleNamespace(
            id="details_checkpoint",
            objective="Prove the Example and details destination reached from Example details",
        ),
        observation,
        diagnostics=diagnostics,
    )

    assert diagnostics["stages"]["frame_selector"] == 1
    frame_candidate = next(
        candidate for candidate in candidates if "Example & details" in candidate.purpose
    )
    assert frame_candidate.call == {"tool": "tap_and_analyze", "arguments": {"id": 2}}
    assert engine._policy_suggestion(frame_candidate) == {  # noqa: SLF001
        "kind": "policy_advisory",
        "candidate_id": frame_candidate.candidate_id,
        "cli": "aua tap-and-analyze 2",
        "mcp": {"tool": "tap_and_analyze", "arguments": {"id": 2}},
        "reason": (
            "The optional local policy selected this guard-approved current-frame call. "
            "AUA has not executed it and has not replaced the deterministic recommendation."
        ),
        "executes": True,
    }


def test_ambiguous_target_candidates_receive_only_the_requested_target_goal(
    tmp_path: Path,
) -> None:
    selector = _Selector(
        lambda context: next(
            candidate.candidate_id
            for candidate in context.candidates
            if "History archive" in candidate.purpose
        )
    )
    engine, _factory = _engine(tmp_path, "shadow", selector)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "History archive", rid="com.example.catalog:id/historyArchive"),
            _element(2, "History lessons", rid="com.example.catalog:id/historyLessons"),
            _element(3, "Grammar tools", rid="com.example.catalog:id/grammar"),
            _element(4, "Physics laboratory", rid="com.example.catalog:id/physics"),
        ],
    )

    result = engine.session_start(
        (
            "Open History from these Example destinations: History archive, "
            "History lessons, Grammar tools, Physics laboratory."
        ),
        observation=observation,
    )

    assert result["policy"]["status"] == "selected"
    assert result["policy"]["candidate_count"] == 2
    # Spelled as the objective spells it: the model matches this against cased
    # candidate labels, and case-folding breaks that binding for rarer words.
    assert selector.contexts[0].goal == "History"
    assert {
        candidate.call["arguments"]["rid"] for candidate in selector.contexts[0].candidates
    } == {
        "historyArchive",
        "historyLessons",
    }


def test_ambiguous_target_preserves_only_safe_candidate_backed_qualifiers(
    tmp_path: Path,
) -> None:
    selector = _Selector(lambda context: context.candidates[0].candidate_id)
    engine, _factory = _engine(tmp_path, "shadow", selector)
    observation = _observation(
        engine.device.serial,
        [
            _element(
                1,
                "History archive Saved records",
                rid="com.example.catalog:id/historyArchive",
            ),
            _element(
                2,
                "History lessons Study plans",
                rid="com.example.catalog:id/historyLessons",
            ),
            _element(
                3,
                "History reports Activity summaries",
                rid="com.example.catalog:id/historyReports",
            ),
            _element(
                4,
                "History settings Display options",
                rid="com.example.catalog:id/historySettings",
            ),
        ],
    )

    result = engine.session_start(
        (
            "Open History using the row whose summary mentions saved records while ignoring "
            "private-code-847291."
        ),
        observation=observation,
    )

    assert result["policy"]["status"] == "selected"
    assert result["policy"]["candidate_count"] == 4
    assert selector.contexts[0].goal == (
        "Requested destination: History. Matching evidence: saved records."
    )
    assert "847291" not in selector.contexts[0].goal


def test_unfingerprinted_or_stale_observation_fails_closed_without_model_use(
    tmp_path: Path,
) -> None:
    selector = _Selector(_choose_mathematics)
    engine, factory = _engine(tmp_path, "advisory", selector)
    stale = _alternatives(engine.device.serial)
    stale.meta.fingerprint = None
    stale.meta.stale_risk = "the prior action outcome is unknown"

    result = engine.session_start("Open Grammar or Mathematics", observation=stale)

    assert result["recommended_call"]["kind"] == "refresh_observation"
    assert result["recommended_call"]["mcp"] == {
        "tool": "analyze_screen",
        "arguments": {"source": "hierarchy", "no_cache": True},
    }
    assert result["policy"]["status"] == "skipped_deterministic"
    assert "policy_suggestion" not in result
    assert factory.build_calls == 0
    assert selector.availability_calls == selector.select_calls == 0


def _recovery_call(
    engine: Engine,
    observation: AnalyzeResult,
    *,
    goal: str = "Open Example archive",
) -> dict[str, Any]:
    return engine._phase_recommended_call(  # noqa: SLF001 - trusted compiler seam under test
        SimpleNamespace(session_id="recovery-session", serial=engine.device.serial),
        SimpleNamespace(
            id="phase-recovery",
            objective=goal,
            kind="verify",
            constraints=[],
            recommended_call=None,
        ),
        observation,
    )


def test_named_loading_recommends_one_bounded_negative_await(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "off", None)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Example archive", rid="com.example.catalog:id/exampleArchive"),
            Element(
                id=2,
                type="android.widget.TextView",
                text="Loading Example records",
                bounds=(20, 400, 900, 460),
                center=(460, 430),
                source=Source.hierarchy,
            ),
        ],
    )

    call = _recovery_call(engine, observation)

    assert call["kind"] == "await_loading"
    assert call["mcp"] == {
        "tool": "await_and_analyze",
        "arguments": {
            "predicate": "!text:Loading",
            "timeout_ms": 15000,
            "poll_ms": 200,
            "ignore_case": True,
        },
    }
    assert "--observe" in call["cli"]


def test_unlabelled_progress_recommends_one_changed_frame_wait(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "off", None)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Example archive", rid="com.example.catalog:id/exampleArchive"),
            Element(
                id=2,
                type="android.widget.ProgressBar",
                bounds=(20, 400, 900, 460),
                center=(460, 430),
                source=Source.hierarchy,
            ),
        ],
    )

    call = _recovery_call(engine, observation)

    assert call["kind"] == "wait_for_change"
    assert call["mcp"] == {
        "tool": "wait_changed_and_analyze",
        "arguments": {"timeout_ms": 15000, "interval_ms": 150},
    }
    assert "--changed" in call["cli"] and "--observe" in call["cli"]


def test_missing_target_on_app_scrollable_recommends_one_folded_scroll(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "off", None)
    observation = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="androidx.recyclerview.widget.RecyclerView",
                bounds=(0, 100, 1080, 2200),
                center=(540, 1150),
                scrollable=True,
                window="app",
                source=Source.hierarchy,
            )
        ],
    )

    call = _recovery_call(engine, observation)

    assert call["kind"] == "scroll_action"
    assert call["mcp"] == {
        "tool": "scroll_and_analyze",
        "arguments": {"direction": "up", "percent": 70},
    }
    assert "scroll-and-analyze up" in call["cli"]


def test_system_scrollable_does_not_authorize_an_app_scroll(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "off", None)
    observation = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.ScrollView",
                bounds=(0, 100, 1080, 2200),
                center=(540, 1150),
                scrollable=True,
                window="system",
                source=Source.hierarchy,
            )
        ],
    )

    call = _recovery_call(engine, observation)

    assert call["kind"] == "manual_observation"
    assert call["executes"] is False


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
            "policy_handoff": {
                "kind": "policy_handoff",
                "executes": False,
                "reason": "No guarded candidate advances the active goal.",
            },
        }

        decorated = decorate_result(
            engine,
            command,
            result,
            current_recorded=False,
        )

        assert decorated["goal_progress"]["policy"] == result["policy"]
        assert decorated["goal_progress"]["policy_suggestion"] == result["policy_suggestion"]
        assert decorated["goal_progress"]["policy_handoff"] == result["policy_handoff"]

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


def test_session_start_anchors_the_exact_returned_frame_before_policy(tmp_path: Path) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine._last_analyze_result = _observation(  # noqa: SLF001 - internal planning residue
        engine.device.serial,
        [_element(9, "Older frame", rid="com.example.catalog:id/older")],
        fingerprint="frame-policy-older",
    )
    current = _alternatives(engine.device.serial)

    result = engine.session_start(
        "Open Grammar or Mathematics",
        observation=current,
    )

    assert result["policy"]["status"] == "selected"
    assert result["policy_suggestion"]["mcp"]["tool"] == "tap_and_analyze"
    assert engine._last_analyze_result is current  # noqa: SLF001 - frame binding contract


def test_navigation_waypoints_keep_order_and_stop_before_input_or_proof() -> None:
    assert Engine._policy_navigation_waypoints(  # noqa: SLF001 - local-loop compiler contract
        "From Example home, open Catalog, then open Image workshop, enter the prompt, "
        "generate a result, and prove the result card."
    ) == ["Catalog", "Image workshop"]


def test_waypoint_arrival_requires_a_passive_current_screen_title() -> None:
    clickable = _observation(
        "policy-host-only",
        [_element(1, "Internet", rid="com.example.catalog:id/openInternet")],
    )
    passive = _observation(
        "policy-host-only",
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Internet",
                bounds=(20, 80, 900, 140),
                center=(460, 110),
                source=Source.hierarchy,
            )
        ],
    )

    assert Engine._policy_waypoint_arrived("Internet", clickable) is False  # noqa: SLF001
    assert Engine._policy_waypoint_arrived("Internet", passive) is True  # noqa: SLF001
    assert (
        Engine._policy_waypoint_arrived(  # noqa: SLF001
            "Internet",
            passive.model_copy(
                update={
                    "elements": [
                        passive.elements[0].model_copy(update={"text": "Network & internet"})
                    ]
                }
            ),
        )
        is False
    )


def test_session_autopilot_executes_selected_calls_without_parent_round_trips(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def choose(context: Any) -> int:
        target = "Settings" if "Settings" in context.goal else "Archive"
        return next(
            candidate.candidate_id
            for candidate in context.candidates
            if target in candidate.purpose
        )

    selector = _Selector(choose)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine.config.policy.candidate_scope = "safe_visible"
    home = _observation(
        engine.device.serial,
        [
            _element(1, "Settings", rid="com.example.catalog:id/openSettings"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
        fingerprint="frame-home",
    )
    settings = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Settings",
                bounds=(20, 100, 900, 160),
                center=(460, 130),
                source=Source.hierarchy,
            ),
            _element(2, "Archive", rid="com.example.catalog:id/openArchive"),
            _element(3, "Profile", rid="com.example.catalog:id/openProfile"),
        ],
        fingerprint="frame-settings",
    )
    destination = _observation(
        engine.device.serial,
        [
            Element(
                id=1,
                type="android.widget.TextView",
                text="Archive",
                bounds=(20, 100, 900, 160),
                center=(460, 130),
                source=Source.hierarchy,
            ),
            Element(
                id=2,
                type="android.widget.TextView",
                text="Saved records",
                bounds=(20, 200, 900, 260),
                center=(460, 230),
                source=Source.hierarchy,
            ),
        ],
        fingerprint="frame-archive",
    )
    returned = {
        "openSettings": settings,
        "openArchive": destination,
    }
    executed: list[str] = []

    def execute_tap(
        _element_id: int | None = None,
        *,
        selector: dict[str, Any] | None = None,
        observe: bool = True,
        **_kwargs: Any,
    ) -> ActionResult:
        assert observe is True
        rid = str((selector or {}).get("rid"))
        executed.append(rid)
        observed = returned[rid]
        engine._last_analyze_result = observed  # noqa: SLF001 - fake folded action result
        return ActionResult(
            ok=True,
            action="tap",
            observation=observed,
            observation_present=True,
        )

    monkeypatch.setattr(engine, "tap", execute_tap)
    started = engine.session_start(
        "From Example home, open Settings; then open Archive and prove the destination.",
        observation=home,
    )

    result = engine.session_autopilot(
        started["session_id"],
        max_steps=4,
        observation=home,
    )

    assert executed == ["openSettings", "openArchive"], result
    assert result["autopilot"]["steps_executed"] == 2
    assert result["autopilot"]["completed_waypoints"] == ["Settings", "Archive"]
    assert result["autopilot"]["terminal_reason"] in {"goal_complete", "navigation_complete"}
    assert all(item["executed"] is True for item in result["autopilot"]["trace"])
    assert all(item["call"]["tool"] == "tap_and_analyze" for item in result["autopilot"]["trace"])
    assert engine.device.calls == []


def test_session_autopilot_stops_after_one_unchanged_frame_without_replay(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    selector = _Selector(_choose_mathematics)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    observation = _alternatives(engine.device.serial)
    executed = 0

    def unchanged_tap(*_args: Any, **_kwargs: Any) -> ActionResult:
        nonlocal executed
        executed += 1
        engine._last_analyze_result = observation  # noqa: SLF001 - fake unchanged action
        return ActionResult(
            ok=True,
            action="tap",
            observation=observation,
            observation_present=True,
        )

    monkeypatch.setattr(engine, "tap", unchanged_tap)
    started = engine.session_start(
        "Open Grammar or Mathematics",
        observation=observation,
    )

    result = engine.session_autopilot(
        started["session_id"],
        max_steps=6,
        observation=observation,
    )

    assert executed == 1
    assert result["ok"] is True
    assert result["autopilot"]["terminal_reason"] == "no_progress"
    assert result["autopilot"]["steps_executed"] == 1


def test_session_autopilot_requires_explicit_advisory_mode(tmp_path: Path) -> None:
    engine, _factory = _engine(tmp_path, "shadow", _Selector())
    started = engine.session_start(
        "Open Grammar or Mathematics",
        observation=_alternatives(engine.device.serial),
    )

    import pytest

    with pytest.raises(UsageError, match="requires policy advisory mode"):
        engine.session_autopilot(
            started["session_id"],
            observation=_alternatives(engine.device.serial),
        )


def test_session_autopilot_rejects_a_shadow_only_provider_before_execution(tmp_path: Path) -> None:
    class ShadowOnlySelector(_Selector):
        def supports_mode(self, mode: str) -> bool:
            return mode == "shadow"

    engine, _factory = _engine(tmp_path, "advisory", ShadowOnlySelector())
    observation = _alternatives(engine.device.serial)
    started = engine.session_start("Open Grammar or Mathematics", observation=observation)

    import pytest

    with pytest.raises(UsageError, match="authenticated for autopilot execution"):
        engine.session_autopilot(started["session_id"], observation=observation)

    assert engine.device.calls == []


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


def test_autopilot_refuses_up_front_when_the_model_cannot_actually_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Being authorised to act is not the same as being able to.

    `supports_mode` reads the adapter's manifest, which says nothing about whether the runtime can
    load it. With the optional MLX extras absent, that check passed, autopilot started, and every
    step found the provider unavailable and handed off: 32 of 41 handoffs in one recorded session
    were nothing but "optional dependency missing". From the outside that is indistinguishable from
    a slow, useless model, so the failure has to arrive once, before any work, carrying the reason.
    """

    class _Unavailable(_Selector):
        name = "fixture_policy"

        def is_available(self) -> Availability:
            self.availability_calls += 1
            return Availability(False, "optional dependency missing; install ...[functiongemma]")

    selector = _Unavailable()
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine.config.policy.candidate_scope = "safe_visible"
    monkeypatch.setattr(
        engine,
        "analyze",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("autopilot must refuse before it observes a device")
        ),
    )

    with pytest.raises(UsageError) as caught:
        engine.session_autopilot(max_steps=6, max_duration_ms=30_000)

    assert caught.value.code == "policy_autopilot_unavailable"
    # The provider's own reason has to survive into the message, because that reason is the fix.
    assert "optional dependency missing" in str(caught.value)
    assert "functiongemma" in str(caught.value)
    # And it must never have started stepping.
    assert selector.select_calls == 0


def test_autopilot_runs_on_its_own_opt_in_without_taxing_every_analyze(
    tmp_path: Path,
) -> None:
    """Typing the command is the opt-in; `policy.enabled` governs the passive advice.

    Running the chain costs real time — the operator who hit this measured roughly twenty seconds
    per analyze with the reviewer in play — so `policy.enabled` exists to keep ordinary navigation
    from paying it. Gating autopilot on the same flag left no way to have one without the other:
    switching it on taxed every unrelated analyze, and switching it off made autopilot refuse with
    "set policy.enabled=true". The two are now separate, and the override must not leak past the
    run, or disabling the policy would stop working the moment autopilot was used once.
    """

    selector = _Selector()
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine.config.policy.enabled = False
    engine.config.policy.mode = "advisory"

    # Passive advice stays off, which is what keeps an ordinary analyze cheap.
    assert engine._session_policy_mode() == "off"
    # The command's own gate reads the configured mode, not the resource switch.
    assert engine._configured_policy_mode() == "advisory"

    observed: list[str] = []
    engine._policy_mode_override = "advisory"
    observed.append(engine._session_policy_mode())
    engine._policy_mode_override = None
    assert observed == ["advisory"]
    # Cleared again, so nothing that runs after the command pays for it.
    assert engine._session_policy_mode() == "off"

    # And a configured `mode: off` still refuses outright — that flag remains a hard no.
    engine.config.policy.mode = "off"
    with pytest.raises(UsageError) as caught:
        engine.session_autopilot(max_steps=4, max_duration_ms=10_000)
    assert caught.value.code == "policy_autopilot_disabled"
    # The remedy lives in the hint, and it must not send anyone back to the flag that costs
    # twenty seconds an analyze.
    assert "policy.enabled` is not required" in str(caught.value.hint)
    assert "policy.mode=advisory" in str(caught.value.hint)
