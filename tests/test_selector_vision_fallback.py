"""A label selector must find text that only vision can see.

The bug this guards: web content publishes almost nothing to the accessibility tree, so
`aua tap --text Continue` raised SelectorNotFoundError on an OAuth consent page where
"Continue" was plainly on screen. Vision element ids do not survive between CLI
invocations, so the only way through was a raw coordinate tap - the one thing this tool
exists to remove. Automating a Google sign-in was impossible for that reason alone.

The fallback is deliberately narrow: labels only. A resource-id is a property of the tree
and pixels cannot supply one, and a content-desc is likewise unobservable, so both stay
strict rather than silently matching something that merely looks similar.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import SelectorNotFoundError
from android_ui_analyser.schema import AnalyzeResult, Element, Meta, Screen


def _el(eid: int, text: str, *, rid: str | None = None) -> Element:
    return Element(
        id=eid,
        type="Button",
        text=text,
        resource_id=rid,
        bounds=[0, eid * 100, 200, eid * 100 + 80],
        center=[100, eid * 100 + 40],
    )


def _result(*elements: Element, tier: str = "hierarchy") -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=720, height=1280, package="com.example", source=tier),
        elements=list(elements),
        meta=Meta(duration_ms=10, tier_used=tier, path=tier),
    )


class _FakeEngine:
    """Only the collaborators resolve_selector touches, so no device is needed."""

    def __init__(self, tree: AnalyzeResult, seen: AnalyzeResult | None = None):
        self._tree = tree
        self._seen = seen
        self.vision_calls = 0

    def analyze(self, *, source="auto", record=True, **_kw):
        if source == "vision":
            self.vision_calls += 1
            if self._seen is None:
                raise AssertionError("vision was consulted when it should not have been")
            return self._seen
        return self._tree

    def _read_cache(self):
        return None

    def _resolve_container_rid(self, _rid):
        return None


def _bind(engine: _FakeEngine):
    """Give the fake the two real methods under test."""
    from android_ui_analyser.engine import Engine

    engine.resolve_selector = Engine.resolve_selector.__get__(engine)
    engine._match_by_vision = Engine._match_by_vision.__get__(engine)
    return engine


def test_text_only_visible_to_vision_is_resolved():
    """An empty WebView in the tree; "Continue" is only in the pixels."""
    engine = _bind(
        _FakeEngine(
            tree=_result(_el(1, "", rid="webView")),
            seen=_result(_el(7, "Continue"), _el(8, "Cancel")),
        )
    )

    el = engine.resolve_selector(text="Continue")

    assert el.text == "Continue"
    assert tuple(el.center) == (100, 740), "must return the coordinates vision reported"
    assert engine.vision_calls == 1


def test_hierarchy_match_never_pays_for_vision():
    """The fallback is a failure path. A normal tap must not get slower."""
    engine = _bind(_FakeEngine(tree=_result(_el(1, "Continue")), seen=None))

    assert engine.resolve_selector(text="Continue").id == 1
    assert engine.vision_calls == 0


def test_resource_id_does_not_fall_back():
    """Pixels cannot supply a resource-id; guessing one would be a wrong answer."""
    engine = _bind(_FakeEngine(tree=_result(_el(1, "Continue")), seen=None))

    with pytest.raises(SelectorNotFoundError):
        engine.resolve_selector(rid="continueButton")
    assert engine.vision_calls == 0


def test_desc_does_not_fall_back():
    """A content-desc is unobservable, so --desc stays strict."""
    engine = _bind(_FakeEngine(tree=_result(_el(1, "Continue")), seen=None))

    with pytest.raises(SelectorNotFoundError):
        engine.resolve_selector(desc="Continue")
    assert engine.vision_calls == 0


def test_miss_names_what_is_visible():
    """On a genuine miss the hint must describe the screen, not an empty WebView."""
    engine = _bind(
        _FakeEngine(
            tree=_result(_el(1, "", rid="webView")),
            seen=_result(_el(7, "Choose an account")),
        )
    )

    with pytest.raises(SelectorNotFoundError) as exc:
        engine.resolve_selector(text="Choose an accont")  # typo on purpose

    assert "Choose an account" in str(exc.value.hint), "vision text belongs in the hint"


def test_fallback_can_be_switched_off():
    engine = _bind(_FakeEngine(tree=_result(_el(1, "", rid="webView")), seen=None))

    with pytest.raises(SelectorNotFoundError):
        engine.resolve_selector(text="Continue", vision_fallback=False)
    assert engine.vision_calls == 0
