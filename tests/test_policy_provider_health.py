"""A provider whose output is mostly unusable is a broken provider, not a fallback.

Observed live: the configured chain was a local primary with a small local fallback, and about
four in five of the fallback's responses never parsed into a selectable candidate ID. Every one
of those was absorbed silently — the guard rejected them, which is correct, and then the chain
asked again, which costs real seconds per call and reported the run as working.

So the invalid rate is now measured per provider, reported in the decision trace and in
`aua policy status`, and once a provider's recent output is majority-unusable it is refused
instead of consulted. Autopilot refuses once, up front, with the measured rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import policy_health
from android_ui_analyser.errors import UsageError
from android_ui_analyser.policy import (
    PolicyCandidate,
    PolicyContext,
    evaluate_policy,
    evaluate_selective_policy,
)
from android_ui_analyser.providers.base import Availability
from test_functiongemma_engine_policy import _element, _engine, _observation, _Selector


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    policy_health.registry().reset()
    yield
    policy_health.registry().reset()


def _candidate(candidate_id: int, purpose: str) -> PolicyCandidate:
    selector = {"rid": f"com.example.catalog:id/open{purpose}"}
    return PolicyCandidate(
        candidate_id=candidate_id,
        call={"tool": "tap_and_analyze", "arguments": dict(selector)},
        purpose=f"Open {purpose}",
        proof=f"The {purpose} screen is shown",
        model_arguments=dict(selector),
        session_id="session-health",
        phase="phase_1",
        observation_fingerprint="frame-health-1",
        package="com.example.catalog",
    )


def _context() -> PolicyContext:
    return PolicyContext(
        goal="Open Archive",
        phase="phase_1",
        session_id="session-health",
        candidates=(_candidate(1, "Archive"), _candidate(2, "Ideas")),
        observation={"fresh": True},
        observation_fingerprint="frame-health-1",
        package="com.example.catalog",
    )


def test_the_invalid_rate_is_measured_and_a_thin_sample_is_not_condemned() -> None:
    registry = policy_health.registry()

    registry.record("gemma_fixture", attempts=2, invalid=2)

    report = registry.report("gemma_fixture")
    assert report["attempts"] == 2
    assert report["invalid"] == 2
    assert report["invalid_rate"] == 1.0
    # Two bad answers are noise, not a verdict.
    assert registry.unusable_reason("gemma_fixture") is None


def test_a_majority_invalid_provider_is_refused_with_the_measured_rate() -> None:
    registry = policy_health.registry()

    registry.record("gemma_fixture", attempts=10, invalid=8)

    reason = registry.unusable_reason("gemma_fixture")
    assert reason is not None
    assert "8 of 10" in reason
    assert "80%" in reason


def test_a_provider_that_mostly_works_is_never_refused() -> None:
    registry = policy_health.registry()

    registry.record("good_fixture", attempts=20, invalid=3)

    assert registry.unusable_reason("good_fixture") is None
    assert registry.report("good_fixture")["invalid_rate"] == pytest.approx(0.15)


def test_only_the_recent_window_counts_so_a_recovered_provider_is_usable() -> None:
    registry = policy_health.registry()

    registry.record("gemma_fixture", attempts=40, invalid=40)
    assert registry.unusable_reason("gemma_fixture") is not None

    registry.record("gemma_fixture", attempts=40, invalid=0)
    # The window has rolled over entirely; nothing invalid remains in it.
    assert registry.report("gemma_fixture")["invalid_rate"] == 0.0


def test_evaluate_policy_counts_unparsable_and_off_list_output() -> None:
    class _Garbage:
        name = "gemma_fixture"

        def is_available(self) -> Availability:
            return Availability(True, "ready")

        def supports_mode(self, mode: str) -> bool:
            return mode in {"shadow", "advisory"}

        def select(self, _context: PolicyContext) -> Any:
            return "candidate_1"  # the shape a truncated tool call arrives in

    for _ in range(6):
        decision = evaluate_policy(_context(), _Garbage(), mode="advisory")
        assert decision.status in {"invalid_selection", "provider_unusable"}

    report = policy_health.registry().report("gemma_fixture")
    assert report["invalid"] >= 5
    assert policy_health.registry().unusable_reason("gemma_fixture") is not None


def test_shadow_observations_do_not_condemn_a_provider_for_advisory_use() -> None:
    class _Garbage:
        name = "shadow_fixture"

        def is_available(self) -> Availability:
            return Availability(True, "ready")

        def supports_mode(self, mode: str) -> bool:
            return mode in {"shadow", "advisory"}

        def select(self, _context: PolicyContext) -> Any:
            return "not-an-id"

    for _ in range(10):
        assert evaluate_policy(_context(), _Garbage(), mode="shadow").status == "invalid_selection"

    assert policy_health.registry().report("shadow_fixture")["attempts"] == 0
    assert policy_health.registry().unusable_reason("shadow_fixture") is None


def test_a_condemned_provider_is_not_consulted_again() -> None:
    calls = 0

    class _Garbage:
        name = "gemma_fixture"

        def is_available(self) -> Availability:
            return Availability(True, "ready")

        def supports_mode(self, mode: str) -> bool:
            return mode in {"shadow", "advisory"}

        def select(self, _context: PolicyContext) -> Any:
            nonlocal calls
            calls += 1
            return None

    provider = _Garbage()
    for _ in range(8):
        evaluate_policy(_context(), provider, mode="advisory")
    spent = calls

    decision = evaluate_policy(_context(), provider, mode="advisory")

    assert decision.status == "provider_unusable"
    assert decision.selected_candidate is None
    assert calls == spent  # not one more second of inference
    assert "unusable output in 6 of 6" in str(decision.error)


def test_the_hybrid_chain_skips_a_condemned_fallback_and_says_so() -> None:
    policy_health.registry().record("gemma_fixture", attempts=10, invalid=9)
    primary = _Selector(lambda context: None)
    primary.name = "primary_fixture"
    fallback = _Selector(lambda context: context.candidates[0].candidate_id)
    fallback.name = "gemma_fixture"

    decision = evaluate_selective_policy(
        _context(),
        [primary, fallback],
        mode="advisory",
        primary_reviews=1,
        reviewer_reviews=1,
    )

    assert decision.status == "handoff"
    assert fallback.select_calls == 0
    statuses = {str(item.get("status")): item for item in decision.selection_trace}
    assert "provider_unusable" in statuses
    assert "9 of 10" in str(statuses["provider_unusable"].get("reason"))


def test_autopilot_refuses_once_when_every_provider_is_condemned(tmp_path: Path) -> None:
    selector = _Selector()
    selector.name = "gemma_fixture"
    policy_health.registry().record("gemma_fixture", attempts=10, invalid=8)
    engine, _factory = _engine(tmp_path, "advisory", selector)
    observation = _observation(
        engine.device.serial,
        [
            _element(1, "Archive", rid="com.example.catalog:id/openArchive"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
    )

    with pytest.raises(UsageError) as caught:
        engine.session_autopilot(observation=observation)

    assert caught.value.code == "policy_autopilot_unusable"
    assert "8 of 10" in str(caught.value)
    assert selector.select_calls == 0


def test_autopilot_blames_the_model_when_the_model_is_what_failed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Unparsable output is not "nothing on screen advances the goal".

    The run that prompted this reported a handoff whose stated reason was the *screen* — no
    visible guard-approved tap advances the navigation — while the actual cause was a provider
    whose answers did not parse. That reading sends whoever is watching to look at the app.
    """

    selector = _Selector(lambda _context: "candidate_1")
    selector.name = "gemma_fixture"
    engine, _factory = _engine(tmp_path, "advisory", selector)
    engine.config.policy.candidate_scope = "safe_visible"
    home = _observation(
        engine.device.serial,
        [
            _element(1, "Archive", rid="com.example.catalog:id/openArchive"),
            _element(2, "Ideas", rid="com.example.catalog:id/openIdeas"),
        ],
        fingerprint="frame-home",
    )

    def forbidden_tap(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("nothing may be executed on unparsable output")

    monkeypatch.setattr(engine, "tap", forbidden_tap)
    started = engine.session_start(
        "From Example home, open Archive and prove the header.",
        observation=home,
    )

    result = engine.session_autopilot(started["session_id"], max_steps=3, observation=home)

    autopilot = result["autopilot"]
    assert result["ok"] is False  # asked to drive, could not: not a success
    assert autopilot["steps_executed"] == 0
    assert autopilot["terminal_reason"] == "provider_output_unusable"
    assert "gemma_fixture" in autopilot["detail"]
    assert "integer candidate ID" in autopilot["detail"]
    # And the rate that made it a verdict rather than an anecdote.
    assert "invalid" in autopilot["detail"]


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))
