"""``--until`` makes an action's readback wait on evidence instead of a blind timer.

``_await_post_action_ready`` waits at most ~1.1s, stretched to 1.6s by ``SettleProfiles``.
Real transitions in the app under test took 2.4s, 18s, 32s and 62s, so the folded observation
reported "nothing changed" for taps that had landed — 38 times across a 5-scenario run. The
agent could not tell "no effect" from "not yet", so it stopped trusting the observation and
hand-rolled ``wait`` + ``analyze`` after every action.

A caller-supplied predicate resolves that ambiguity: the budget comes from the predicate and
``await_outcome`` names what ended the wait, including a stable wrong-arrival correction.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.memory import AppMemoryStore, RouteStep
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult, AnalyzeResult, Element, Meta, Screen, Source
from conftest import FakeDevice, make_config


def _tapped(**kw) -> ActionResult:
    base = {
        "ok": True,
        "action": "tap",
        "id": 7,
        "observation_present": True,
        "detail": "stale_risk settle=1100ms via=unchanged",
    }
    base.update(kw)
    return ActionResult(**base)


def _awaited(outcome: str, **kw) -> ActionResult:  # type: ignore[no-untyped-def]
    base = {
        "ok": outcome == "satisfied",
        "action": "await",
        "await_outcome": outcome,
        "await_terms": [{"term": "rid:resultsPanel", "present": outcome == "satisfied"}],
        "elapsed_ms": 2381,
        "observation_present": True,
        "known_screen": "home",
    }
    base.update(kw)
    return ActionResult(**base)


def _run(monkeypatch, result, until, awaited=None):
    from android_ui_analyser import cli

    calls: list[dict] = []

    def fake_route(_engine, method, **kwargs):
        calls.append({"method": method, **kwargs})
        return awaited

    monkeypatch.setattr(cli, "_route", fake_route)
    monkeypatch.setattr(cli, "_ENGINE", object())
    monkeypatch.setattr(cli, "_UNTIL", until)
    return cli._await_until(result), calls


def test_until_adopts_the_awaited_outcome_and_budget(monkeypatch) -> None:
    out, calls = _run(
        monkeypatch, _tapped(), ("rid:resultsPanel", 45000, 500), _awaited("satisfied")
    )

    assert calls[0]["method"] == "await_predicate"
    assert calls[0]["predicate"] == "rid:resultsPanel"
    assert calls[0]["adopt_action"] is True
    # The predicate's budget, not the 1.6s settle ceiling — that is the entire fix.
    assert calls[0]["timeout_ms"] == 45000
    assert out.await_outcome == "satisfied"
    assert out.elapsed_ms == 2381
    assert out.action == "tap", "the action's own identity must survive the merge"
    assert out.id == 7


def test_satisfied_clears_the_settle_derived_stale_caveat(monkeypatch) -> None:
    """``stale_risk`` describes a screen we have since re-read on evidence."""
    out, _ = _run(
        monkeypatch,
        _tapped(stale_risk="the early read may be stale"),
        ("rid:resultsPanel", 30000, 500),
        _awaited("satisfied"),
    )
    assert out.detail is not None
    assert "stale_risk" not in out.detail
    assert out.stale_risk is None


def test_until_adopts_guidance_from_the_awaited_screen(monkeypatch) -> None:
    """Top-level guidance must describe the observation adopted after ``--until``."""
    out, _ = _run(
        monkeypatch,
        _tapped(
            known_screen="search",
            next_actions=[{"id": 3, "label": "Old search field"}],
            routes=["old route"],
        ),
        ("rid:resultsPanel", 30000, 500),
        _awaited(
            "satisfied",
            known_screen="home",
            next_actions=[{"id": 8, "label": "Continue"}],
            routes=["tap Continue -> details"],
        ),
    )

    assert out.known_screen == "home"
    assert out.next_actions == [{"id": 8, "label": "Continue"}]
    assert out.routes == ["tap Continue -> details"]


def test_until_preserves_structured_settled_arrival_mismatch(monkeypatch) -> None:
    mismatch = {
        "code": "arrival_mismatch",
        "original_predicate": "text:Old title,!text:Loading",
        "suggested_positive_predicates": ["rid:recentItem"],
        "recommended_call": ("aua await-and-analyze 'rid:recentItem,!text:Loading' --observe"),
        "recommended_mcp_call": {
            "tool": "await_and_analyze",
            "arguments": {"predicate": "rid:recentItem,!text:Loading"},
        },
        "action_repeated": False,
    }
    out, _ = _run(
        monkeypatch,
        _tapped(),
        ("text:Old title,!text:Loading", 30_000, 500),
        _awaited(
            "settled-unmet",
            arrival_mismatch=mismatch,
            note="The action ran once; reuse this destination and do not repeat it.",
        ),
    )

    assert out.action == "tap"
    assert out.await_outcome == "settled-unmet"
    assert out.arrival_mismatch == mismatch
    assert "do not repeat" in (out.note or "")


def test_until_replaces_the_early_change_claim_with_the_adopted_screen(monkeypatch) -> None:
    """The envelope and nested diff must not describe two different observation moments."""
    early = {
        "changed": False,
        "detail": "nothing changed: same activity, same node count",
    }
    adopted = {
        "changed": True,
        "text_added": ["Fictional result"],
        "text_removed": ["Loading"],
    }
    out, _ = _run(
        monkeypatch,
        _tapped(change=early, action_diff_summary={"added": 0, "removed": 0, "changed": 0}),
        ("text:Fictional result", 30000, 500),
        _awaited(
            "satisfied",
            change=adopted,
            action_diff_summary={"added": 1, "removed": 1, "changed": 0},
        ),
    )

    assert out.change == adopted
    assert out.action_diff_summary == {"added": 1, "removed": 1, "changed": 0}


def test_until_never_replaces_a_valid_delta_with_a_no_baseline_false(monkeypatch) -> None:
    early = {
        "changed": True,
        "node_count_before": 2,
        "node_count_after": 3,
        "text_added": ["Intermediate"],
    }
    no_baseline = {
        "changed": False,
        "node_count_before": None,
        "node_count_after": 3,
        "detail": "no pre-action snapshot — deltas unavailable",
    }

    out, _ = _run(
        monkeypatch,
        _tapped(change=early),
        ("text:Destination", 30000, 500),
        _awaited("satisfied", change=no_baseline),
    )

    assert out.change == early


def _screen_with(label: str) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package="com.example.app", source="hierarchy"),
        elements=[
            Element(
                id=0,
                type="TextView",
                text=label,
                bounds=(20, 200, 800, 280),
                center=(410, 240),
            )
        ],
        meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy"),
    )


def test_adopted_observation_reuses_the_actions_original_baseline(tmp_path, monkeypatch) -> None:
    cfg = make_config(
        memory={"enabled": False, "dir": str(tmp_path / "home")},
        daemon={"enabled": False},
    )
    eng = Engine(cfg, device=FakeDevice(package="com.example.app"))
    observations = iter([_screen_with("Intermediate"), _screen_with("Destination")])
    monkeypatch.setattr(eng, "_analyze_post_action", lambda *_a, **_k: next(observations))
    monkeypatch.setattr(eng, "_read_activity", lambda: "com.example.app/.MainActivity")
    eng._pre_action_state = {
        "count": 1,
        "focused": None,
        "labels": ["Before"],
        "package": "com.example.app",
        "activity": "com.example.app/.MainActivity",
        "known_screen": None,
    }

    early = eng._observe(ActionResult(ok=True, action="tap"), True, settle=False)
    adopted = eng._await_result(
        "satisfied",
        [{"term": "text:Destination", "present": True, "satisfied": True}],
        time.monotonic(),
        1,
        ("com.example.app", ".MainActivity"),
        ("com.example.app", ".MainActivity"),
        True,
        adopt_action=True,
    )

    assert early.change and early.change["text_added"] == ["Intermediate"]
    assert adopted.change and adopted.change["changed"] is True
    assert adopted.change["text_added"] == ["Destination"]
    assert adopted.change["text_removed"] == ["Before"]


def test_regex_looking_until_timeout_explains_literal_matching(monkeypatch) -> None:
    out, _ = _run(
        monkeypatch,
        _tapped(),
        ("text:^Destination.*$", 5000, 500),
        _awaited("timeout"),
    )

    assert "looks regex-like" in (out.note or "")
    assert "literal contains" in (out.note or "")
    assert "aua await-and-analyze" in (out.note or "")
    assert "--match regex" in (out.note or "")


def test_timeout_keeps_the_caveat_and_reports_which_term_failed(monkeypatch) -> None:
    """A timeout is not a failed tap — the outcome must stay distinguishable."""
    out, _ = _run(monkeypatch, _tapped(), ("rid:resultsPanel", 5000, 500), _awaited("timeout"))
    assert out.await_outcome == "timeout"
    assert out.await_terms and out.await_terms[0]["present"] is False
    assert "stale_risk" in (out.detail or "")


def test_no_until_leaves_the_result_untouched(monkeypatch) -> None:
    original = _tapped()
    out, calls = _run(monkeypatch, original, None, None)
    assert out is original
    assert calls == []


def test_adopt_action_rejects_negative_only_before_reading_the_device(tmp_path) -> None:
    dev = FakeDevice(package="com.example.app")
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

    with pytest.raises(UsageError, match="positive arrival"):
        eng.await_predicate("!text:Loading", adopt_action=True)

    assert dev.calls == []
    assert dev.hierarchy_calls == 0


def test_adopt_action_rejects_malformed_predicate_before_reading_the_device(tmp_path) -> None:
    """Relaxing standalone absence must not make malformed action evidence reach Android."""
    dev = FakeDevice(package="com.example.app")
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

    with pytest.raises(UsageError, match="unknown field"):
        eng.await_predicate("!nosuchfield:Loading", adopt_action=True)

    assert dev.calls == []
    assert dev.hierarchy_calls == 0


def test_failed_action_is_not_waited_on(monkeypatch) -> None:
    """Waiting after an action that never happened would just burn the whole budget."""
    out, calls = _run(
        monkeypatch, _tapped(ok=False), ("rid:resultsPanel", 30000, 500), _awaited("timeout")
    )
    assert calls == []
    assert out.await_outcome is None


def test_non_action_responses_are_ignored(monkeypatch) -> None:
    """``observation_present`` is the action-contract marker; `devices`/`doctor` lack it."""
    plain = ActionResult(ok=True, action="devices")
    out, calls = _run(monkeypatch, plain, ("rid:resultsPanel", 30000, 500), _awaited("satisfied"))
    assert calls == []
    assert out.await_outcome is None


def test_satisfied_until_records_the_awaited_destination_not_the_early_readback(
    tmp_path,
) -> None:
    """The predicate's final screen owns the pending action in memory.

    The action's folded observation is deliberately passive because it may be a loading shell.
    When ``--until`` succeeds, however, the screen satisfying the caller's evidence is safe to
    run through normal recording. This is the engine half of the CLI's ``adopt_action`` contract.
    """
    from test_memory import APPS, HOME, _elements

    package = "com.example.app"
    serial = "awaited-recording"
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    store = AppMemoryStore(cfg.memory)
    source = store.record_screen(package=package, elements=_elements(HOME), name_hint="home")
    target = store.record_screen(package=package, elements=_elements(APPS), name_hint="catalog")
    store.save_session(
        serial,
        store.load_session(serial).model_copy(
            update={"package": package, "current_screen": source.name}
        ),
    )
    store.observe_action(serial, RouteStep(kind="tap", label="Catalog", resource_id="nav_catalog"))

    dev = FakeDevice(
        hierarchy_xml=APPS,
        package=package,
        serial=serial,
        text_index={"Apps": (0, 0, 100, 60)},
    )
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))

    out = eng.await_predicate(
        "text:Apps",
        timeout_ms=1000,
        poll_ms=1,
        observe=True,
        adopt_action=True,
    )

    assert out.await_outcome == "satisfied"
    routes = AppMemoryStore(cfg.memory).load(package).routes
    assert any(
        route.from_screen == source.name
        and route.to_screen == target.name
        and route.steps[0].resource_id == "nav_catalog"
        for route in routes
    )


def _ocr_observation(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        elements=[
            Element(
                id=90,
                type="Text",
                text=text,
                bounds=(20, 200, 800, 280),
                center=(410, 240),
                source=Source.ocr,
            )
        ]
    )


def test_negated_text_is_not_satisfied_by_a_hierarchy_only_miss(tmp_path) -> None:
    dev = FakeDevice(package="com.example.app", text_index={})
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    rich_calls: list[dict[str, object]] = []

    def rich(**kwargs):  # type: ignore[no-untyped-def]
        rich_calls.append(kwargs)
        return _ocr_observation("Loading illustration")

    eng.analyze = rich  # type: ignore[method-assign]

    out = eng.await_predicate("!text:Loading", timeout_ms=5, poll_ms=1)

    assert out.await_outcome == "timeout"
    assert rich_calls and all(call["with_ocr"] is True for call in rich_calls)


def test_positive_text_gets_one_rich_verification_before_timeout(tmp_path) -> None:
    dev = FakeDevice(package="com.example.app", text_index={})
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    eng.analyze = lambda **_kwargs: _ocr_observation("Rendered result")  # type: ignore[method-assign]

    out = eng.await_predicate("text:Rendered result", timeout_ms=0, poll_ms=1)

    assert out.await_outcome == "satisfied"
    assert out.await_terms and out.await_terms[0]["present"] is True


def test_internal_hierarchy_only_wait_skips_visual_verification_and_escalation(
    tmp_path, monkeypatch
) -> None:
    dev = FakeDevice(package="com.example.app", text_index={})
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    eng = Engine(cfg, device=dev, factory=ProviderFactory(cfg))
    analyzed: list[dict[str, object]] = []

    def hierarchy(**kwargs):  # type: ignore[no-untyped-def]
        analyzed.append(kwargs)
        return AnalyzeResult(
            screen=Screen(
                width=1080,
                height=2400,
                package="com.example.app",
                source="hierarchy",
            ),
            elements=[],
            meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy"),
        )

    monkeypatch.setattr(eng, "analyze", hierarchy)
    monkeypatch.setattr(
        eng,
        "_analyze_post_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not escalate intermediate navigation")
        ),
    )

    out = eng.await_predicate(
        "text:Destination",
        timeout_ms=0,
        poll_ms=10,
        observe=True,
        rich_ui=False,
        hierarchy_only=True,
    )

    assert out.await_outcome == "timeout"
    assert out.observation is not None
    assert not any(name == "current_app" for name, _args in dev.calls)
    assert analyzed == [
        {
            "source": "hierarchy",
            "with_ocr": False,
            "record": False,
            "with_image": None,
        }
    ]
