"""Quality-gate tests (PRD §13.1 AC3).

Fixture-level: the gate must keep ``normal_views`` on the hierarchy and send both
``compose_no_semantics`` and ``empty_canvas`` to vision. Plus a unit test per rule with
tiny synthetic element lists so each heuristic is exercised in isolation.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.config import GateCfg
from android_ui_analyser.gate import GateDecision, decide
from android_ui_analyser.hierarchy import parse_hierarchy
from android_ui_analyser.schema import Element, Source

FIXTURES = Path(__file__).parent / "fixtures"
SCREEN = (1080, 2400)


def _els(name: str) -> list[Element]:
    return parse_hierarchy((FIXTURES / f"{name}.xml").read_text(encoding="utf-8"), SCREEN)


def _el(
    id: int,
    *,
    type: str = "View",
    text: str | None = None,
    content_desc: str | None = None,
    clickable: bool = False,
    bounds: tuple[int, int, int, int] = (0, 0, 100, 100),
) -> Element:
    return Element(
        id=id,
        type=type,
        text=text,
        content_desc=content_desc,
        bounds=bounds,
        center=((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2),
        clickable=clickable,
        source=Source.hierarchy,
    )


# --------------------------------------------------------------------------- fixtures (AC3)


def test_normal_views_uses_hierarchy() -> None:
    d = decide(
        _els("normal_views"), package="com.example.shop", activity=".LoginActivity", cfg=GateCfg()
    )
    assert isinstance(d, GateDecision)
    assert d.use_vision is False
    assert d.reason


def test_empty_canvas_uses_vision() -> None:
    d = decide(
        _els("empty_canvas"), package="com.example.game", activity=".GameActivity", cfg=GateCfg()
    )
    assert d.use_vision is True


def test_compose_no_semantics_uses_vision() -> None:
    d = decide(
        _els("compose_no_semantics"),
        package="com.example.compose",
        activity=".MainActivity",
        cfg=GateCfg(),
    )
    assert d.use_vision is True


# --------------------------------------------------------------------------- rule 1: count


def test_rule_min_elements() -> None:
    cfg = GateCfg(min_elements=3)
    few = [_el(0, text="a"), _el(1, text="b")]
    d = decide(few, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is True
    assert "min_elements" in d.reason


def test_rule_min_elements_boundary_passes() -> None:
    cfg = GateCfg(min_elements=3)
    three = [_el(0, text="a"), _el(1, text="b"), _el(2, text="c")]
    d = decide(three, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is False


# --------------------------------------------------------------------------- rule 2: semantics


def test_rule_no_semantics() -> None:
    cfg = GateCfg(min_elements=1)
    none_labeled = [_el(0), _el(1), _el(2)]
    d = decide(none_labeled, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is True
    assert "text or content-desc" in d.reason


def test_rule_semantics_via_content_desc_passes() -> None:
    cfg = GateCfg(min_elements=1)
    labeled = [_el(0), _el(1, content_desc="Menu"), _el(2)]
    d = decide(labeled, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is False


# --------------------------------------------------------------------------- rule 3: packages


def test_rule_vision_package_substring() -> None:
    cfg = GateCfg(min_elements=1, vision_packages=["io.flutter"])
    els = [_el(0, text="a"), _el(1, text="b")]
    d = decide(els, package="io.flutter.embedding.android.FlutterActivity", activity=".X", cfg=cfg)
    assert d.use_vision is True
    assert "io.flutter" in d.reason


def test_rule_vision_package_glob_against_activity() -> None:
    cfg = GateCfg(min_elements=1, vision_packages=["*.WebView"])
    els = [_el(0, text="a"), _el(1, text="b")]
    d = decide(els, package="com.normal.app", activity="org.chromium.WebView", cfg=cfg)
    assert d.use_vision is True


def test_rule_vision_package_glob_against_element_class() -> None:
    cfg = GateCfg(min_elements=1, vision_packages=["*.WebView"])
    els = [_el(0, text="a"), _el(1, type="WebView", text="page")]
    d = decide(els, package="com.normal.app", activity=".Main", cfg=cfg)
    assert d.use_vision is True
    assert "WebView" in d.reason


def test_rule_vision_package_no_match_passes() -> None:
    cfg = GateCfg(min_elements=1, vision_packages=["io.flutter", "com.unity3d"])
    els = [_el(0, text="a"), _el(1, text="b")]
    d = decide(els, package="com.example.regular", activity=".Main", cfg=cfg)
    assert d.use_vision is False


# --------------------------------------------------------------------------- rule 4: labeled ratio


def test_rule_labeled_ratio_triggers() -> None:
    cfg = GateCfg(min_elements=1, min_labeled_ratio=0.5)
    # 4 clickables, only 1 labeled -> ratio 0.25 < 0.5
    els = [
        _el(0, text="title"),  # not clickable, provides semantics so rule 2 passes
        _el(1, clickable=True, text="ok"),
        _el(2, clickable=True),
        _el(3, clickable=True),
        _el(4, clickable=True),
    ]
    d = decide(els, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is True
    assert "ratio" in d.reason


def test_rule_labeled_ratio_passes_when_well_labeled() -> None:
    cfg = GateCfg(min_elements=1, min_labeled_ratio=0.5)
    els = [
        _el(0, clickable=True, text="ok"),
        _el(1, clickable=True, content_desc="cancel"),
        _el(2, clickable=True),
    ]
    d = decide(els, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is False


def test_rule_labeled_ratio_no_clickables_does_not_divide_by_zero() -> None:
    cfg = GateCfg(min_elements=1, min_labeled_ratio=0.99)
    # all labeled (rule 2 passes), zero clickable -> rule 4 must NOT fire
    els = [_el(0, text="a"), _el(1, text="b")]
    d = decide(els, package="com.x", activity=".A", cfg=cfg)
    assert d.use_vision is False


def test_decision_unpacks_as_tuple() -> None:
    use_vision, reason = decide(
        [_el(0, text="a")], package="com.x", activity=".A", cfg=GateCfg(min_elements=1)
    )
    assert use_vision is False
    assert isinstance(reason, str)
