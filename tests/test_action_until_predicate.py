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

from android_ui_analyser.schema import ActionResult


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


def _awaited(outcome: str) -> ActionResult:
    return ActionResult(
        ok=outcome == "satisfied",
        action="await",
        await_outcome=outcome,
        await_terms=[{"term": "rid:introCard", "present": outcome == "satisfied"}],
        elapsed_ms=2381,
        observation_present=True,
        known_screen="home",
    )


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
    # The predicate's budget, not the 1.6s settle ceiling — that is the entire fix.
    assert calls[0]["timeout_ms"] == 45000
    assert out.await_outcome == "satisfied"
    assert out.elapsed_ms == 2381
    assert out.action == "tap", "the action's own identity must survive the merge"
    assert out.id == 7


def test_satisfied_clears_the_settle_derived_stale_caveat(monkeypatch) -> None:
    """``stale_risk`` describes a screen we have since re-read on evidence."""
    out, _ = _run(monkeypatch, _tapped(), ("rid:introCard", 30000, 500), _awaited("satisfied"))
    assert out.detail is not None
    assert "stale_risk" not in out.detail


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
