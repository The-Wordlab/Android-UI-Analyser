"""Action-bound arrival predicates fail fast only on a proved settled destination.

A folded ``--until`` runs after exactly one action.  If that action has already reached a stable,
non-loading screen in the same Activity, waiting thirty more seconds for text from the screen we
left cannot help.  This suite pins the conservative boundary: action waits get a structured
correction, while standalone streaming waits keep their package/activity-only semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.schema import ActionResult, AnalyzeResult, Element, Meta, Screen
from conftest import FakeDevice, make_config

PKG = "com.example.app"


def _observation(*, title: str = "Recent items", loading: bool = False) -> AnalyzeResult:
    elements = [
        Element(
            id=0,
            type="TextView",
            text=title,
            resource_id=f"{PKG}:id/recentTitle",
            bounds=(20, 100, 700, 180),
            center=(360, 140),
        ),
        Element(
            id=1,
            type="Button",
            text="Cached example",
            resource_id=f"{PKG}:id/catalogItemCard",
            bounds=(20, 220, 900, 360),
            center=(460, 290),
            clickable=True,
        ),
    ]
    if loading:
        elements.append(
            Element(
                id=2,
                type="android.widget.ProgressBar",
                bounds=(480, 400, 600, 520),
                center=(540, 460),
            )
        )
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=PKG, source="hierarchy"),
        elements=elements,
        meta=Meta(duration_ms=1, tier_used="hierarchy", path="hierarchy"),
    )


def _engine(tmp_path: Path, device: FakeDevice, output: dict | None = None) -> Engine:
    cfg = make_config(
        memory={"enabled": False, "dir": str(tmp_path / "home")},
        daemon={"enabled": False},
        **({"output": output} if output else {}),
    )
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


def _seed_action_baseline(engine: Engine) -> None:
    engine._action_observation_baseline = {
        "count": 2,
        "focused": None,
        "labels": ["Product detail", "Open"],
        "rids": ["detailTitle", "openButton"],
        "package": PKG,
        "activity": f"{PKG}/.MainActivity",
        "known_screen": None,
    }


def test_same_activity_wrong_predicate_returns_bounded_arrival_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(package=PKG, activity=".MainActivity", text_index={})
    engine = _engine(tmp_path, device)
    _seed_action_baseline(engine)
    destination = _observation()
    samples = 0

    def sample() -> AnalyzeResult:
        nonlocal samples
        samples += 1
        return destination.model_copy(deep=True)

    monkeypatch.setattr(engine, "_sample_action_destination", sample)
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_args, **_kwargs: destination.model_copy(deep=True),
    )

    result = engine.await_predicate(
        "text:Product detail,!text:Loading",
        timeout_ms=600_000,
        poll_ms=1,
        rich_ui=False,
        observe=True,
        adopt_action=True,
    )

    assert result.ok is False
    assert result.await_outcome == "settled-unmet"
    assert samples == 2, "two equal fresh frames prove stability; the long budget is not spent"
    selector_polls = [call for call in device.calls if call[0] == "find_text"]
    assert len(selector_polls) == 4, "two terms over two bounded predicate polls"
    assert not any(call[0] in {"click", "press", "swipe"} for call in device.calls)
    assert result.observation is not None
    assert any(element.text == "Recent items" for element in result.observation.elements)
    assert result.arrival_mismatch == {
        "code": "arrival_mismatch",
        "original_predicate": "text:Product detail,!text:Loading",
        "unmet_positive_terms": ["text:Product detail"],
        "suggested_positive_predicates": [
            "rid:catalogItemCard",
            "rid:recentTitle",
            "text:Cached example",
        ],
        "stable_checks": 2,
        "screen_changed": True,
        "loading": False,
        "action_repeated": False,
        "recommended_call": ("aua await-and-analyze 'rid:catalogItemCard,!text:Loading' --observe"),
        "recommended_mcp_call": {
            "tool": "await_and_analyze",
            "arguments": {"predicate": "rid:catalogItemCard,!text:Loading"},
        },
    }
    assert "do not repeat the action" in (result.note or "")


def test_correct_action_predicate_succeeds_without_mismatch_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(
        package=PKG,
        activity=".MainActivity",
        resource_index={"catalogItemCard": (20, 220, 900, 360)},
        text_index={},
    )
    engine = _engine(tmp_path, device)
    _seed_action_baseline(engine)
    destination = _observation()
    monkeypatch.setattr(
        engine,
        "_sample_action_destination",
        lambda: (_ for _ in ()).throw(AssertionError("a satisfied predicate must win first")),
    )
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_args, **_kwargs: destination.model_copy(deep=True),
    )

    result = engine.await_predicate(
        "rid:catalogItemCard",
        timeout_ms=30_000,
        poll_ms=1,
        rich_ui=False,
        observe=True,
        adopt_action=True,
    )

    assert result.ok is True
    assert result.await_outcome == "satisfied"
    assert result.arrival_mismatch is None
    assert len([call for call in device.calls if call[0] == "find_text"]) == 1


def test_streaming_standalone_wait_keeps_package_activity_only_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(package=PKG, activity=".StreamingActivity", text_index={})
    engine = _engine(tmp_path, device)
    _seed_action_baseline(engine)  # Even stale action state cannot opt a standalone wait in.
    monkeypatch.setattr(
        engine,
        "_sample_action_destination",
        lambda: (_ for _ in ()).throw(
            AssertionError("standalone waits must never inspect tree stability")
        ),
    )

    result = engine.await_predicate(
        "text:Finished",
        # Wide enough that "did it poll repeatedly" cannot depend on how fast the host is. At
        # 45ms a 3-core CI runner fitted exactly two polls into the window and failed
        # `> 2` — the wait behaved correctly, the assertion was really measuring the machine.
        timeout_ms=300,
        poll_ms=1,
        rich_ui=False,
        observe=False,
        adopt_action=False,
    )

    assert result.await_outcome == "timeout"
    assert result.arrival_mismatch is None
    assert len([call for call in device.calls if call[0] == "find_text"]) > 2


def test_visible_loading_state_never_becomes_arrival_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(package=PKG, activity=".MainActivity", text_index={})
    engine = _engine(tmp_path, device)
    _seed_action_baseline(engine)
    loading = _observation(loading=True)
    monkeypatch.setattr(
        engine,
        "_sample_action_destination",
        lambda: loading.model_copy(deep=True),
    )
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_args, **_kwargs: loading.model_copy(deep=True),
    )

    result = engine.await_predicate(
        "text:Product detail",
        timeout_ms=45,
        poll_ms=1,
        rich_ui=False,
        observe=True,
        adopt_action=True,
    )

    assert result.await_outcome == "timeout"
    assert result.arrival_mismatch is None


def test_changing_action_destination_is_not_declared_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FakeDevice(package=PKG, activity=".MainActivity", text_index={})
    engine = _engine(tmp_path, device)
    _seed_action_baseline(engine)
    samples = 0

    def streaming() -> AnalyzeResult:
        nonlocal samples
        samples += 1
        return _observation(title=f"Streaming frame {samples % 2}")

    monkeypatch.setattr(engine, "_sample_action_destination", streaming)
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_args, **_kwargs: _observation(title="Streaming frame final"),
    )

    result = engine.await_predicate(
        "text:Product detail",
        # Same reason as the standalone-wait test above: `samples > 2` asks how many polls fit
        # in the window, and at 45ms a 3-core CI runner fitted two. The sibling below keeps 45ms
        # because it only asserts the wait timed out, which happens on any machine.
        timeout_ms=300,
        poll_ms=1,
        rich_ui=False,
        observe=True,
        adopt_action=True,
    )

    assert samples > 2
    assert result.await_outcome == "timeout"
    assert result.arrival_mismatch is None


@pytest.mark.parametrize(
    ("ready", "advertised"),
    [
        ({"changed": True, "timeout": False, "via": "pixels", "ms": 80}, True),
        ({"changed": True, "timeout": False, "via": "hierarchy-fast", "ms": 80}, False),
        ({"changed": True, "timeout": True, "via": "timeout", "ms": 1100}, False),
    ],
)
def test_launch_only_advertises_derived_ids_after_a_stable_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ready: dict[str, object],
    advertised: bool,
) -> None:
    """The derived list is opt-in now, so it is turned on here: withholding is the invariant.

    A one-sample/timeout/unchanged settle path proves the app is foreground and nothing more.
    Publishing a pre-filtered list of its ids reads as "these are the controls you can act on",
    and the frame they came from may be gone before the next call reaches it.
    """
    device = FakeDevice(package=PKG, activity=".MainActivity")
    engine = _engine(tmp_path, device, output={"next_actions": True})
    destination = _observation()
    engine._pre_action_state = {
        "count": 1,
        "focused": None,
        "labels": ["Launcher"],
        "rids": ["appIcon"],
        "package": "com.example.launcher",
        "activity": "com.example.launcher/.LauncherActivity",
        "known_screen": None,
    }
    monkeypatch.setattr(engine, "_await_post_action_ready", lambda **_kwargs: dict(ready))
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_args, **_kwargs: destination.model_copy(deep=True),
    )

    result = engine._observe(ActionResult(ok=True, action="app-launch"), True)

    assert bool(result.next_actions) is advertised
    if not advertised:
        assert "its ids may not survive until your next call" in (result.note or "")
        assert "`aua analyze`" in (result.note or "")
