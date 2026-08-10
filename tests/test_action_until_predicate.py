"""``--until`` makes an action's readback wait on evidence instead of a blind timer.

``_await_post_action_ready`` waits at most ~1.1s, stretched to 1.6s by ``SettleProfiles``.
Real transitions in the app under test took 2.4s, 18s, 32s and 62s, so the folded observation
reported "nothing changed" for taps that had landed — 38 times across a 5-scenario run. The
agent could not tell "no effect" from "not yet", so it stopped trusting the observation and
hand-rolled ``wait`` + ``analyze`` after every action.

A caller-supplied predicate resolves that ambiguity: the budget comes from the predicate and
``await_outcome`` names which of three things ended the wait.
"""

from __future__ import annotations

from types import SimpleNamespace

from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMemoryStore, RouteStep
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult, Element, Source
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
        "await_terms": [{"term": "rid:introCard", "present": outcome == "satisfied"}],
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
    out, calls = _run(monkeypatch, _tapped(), ("rid:introCard", 45000, 500), _awaited("satisfied"))

    assert calls[0]["method"] == "await_predicate"
    assert calls[0]["predicate"] == "rid:introCard"
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
        ("rid:introCard", 30000, 500),
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
        ("rid:introCard", 30000, 500),
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


def test_timeout_keeps_the_caveat_and_reports_which_term_failed(monkeypatch) -> None:
    """A timeout is not a failed tap — the outcome must stay distinguishable."""
    out, _ = _run(monkeypatch, _tapped(), ("rid:introCard", 5000, 500), _awaited("timeout"))
    assert out.await_outcome == "timeout"
    assert out.await_terms and out.await_terms[0]["present"] is False
    assert "stale_risk" in (out.detail or "")


def test_no_until_leaves_the_result_untouched(monkeypatch) -> None:
    original = _tapped()
    out, calls = _run(monkeypatch, original, None, None)
    assert out is original
    assert calls == []


def test_failed_action_is_not_waited_on(monkeypatch) -> None:
    """Waiting after an action that never happened would just burn the whole budget."""
    out, calls = _run(
        monkeypatch, _tapped(ok=False), ("rid:introCard", 30000, 500), _awaited("timeout")
    )
    assert calls == []
    assert out.await_outcome is None


def test_non_action_responses_are_ignored(monkeypatch) -> None:
    """``observation_present`` is the action-contract marker; `devices`/`doctor` lack it."""
    plain = ActionResult(ok=True, action="devices")
    out, calls = _run(monkeypatch, plain, ("rid:introCard", 30000, 500), _awaited("satisfied"))
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
