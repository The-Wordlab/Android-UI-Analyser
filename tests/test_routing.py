"""AC12 — adaptive routing / escalation ladder (PRD §6a), with call-spies.

Asserts: an exact-literal lookup resolves at the hierarchy tier and vision is NEVER
invoked; a semantic query that the hierarchy can answer does not call grounding; under
a ``vision`` ceiling a hierarchy+vision miss does NOT call the paid grounding provider;
and an explicit ``--deep`` opt-in does escalate to grounding. Plus the pure routing
helpers.
"""

from __future__ import annotations

from android_ui_analyser.providers.base import (
    Availability,
    DetBox,
    DetectionProvider,
    GroundingProvider,
    OcrProvider,
    Point,
    ScreenImage,
    TextBox,
)
from android_ui_analyser.providers.registry import (
    register_detection,
    register_grounding,
    register_ocr,
)
from android_ui_analyser.routing import (
    Intent,
    QueryKind,
    allows,
    classify_query,
    entry_tier,
    next_tier,
    resolve_ceiling,
    salient_tokens,
)
from android_ui_analyser.schema import Tier, tier_rank
from conftest import FakeDevice, make_config, make_engine

# --- call spies (module-level counters; the factory builds fresh instances each call) ---
ocr_calls: list[int] = []
det_calls: list[int] = []
gnd_calls: list[int] = []


@register_ocr("spy_ocr")
class _SpyOcr(OcrProvider):
    def is_available(self) -> Availability:
        return Availability(True, "ok")

    def recognize(self, image: ScreenImage) -> list[TextBox]:
        ocr_calls.append(1)
        return [TextBox(text="noise", bounds=(0, 0, 5, 5))]


@register_detection("spy_det")
class _SpyDet(DetectionProvider):
    def is_available(self) -> Availability:
        return Availability(True, "ok")

    def detect(self, image: ScreenImage) -> list[DetBox]:
        det_calls.append(1)
        return [DetBox(bounds=(10, 10, 60, 60), label="thing")]


@register_grounding("spy_gnd")
class _SpyGnd(GroundingProvider):
    def is_available(self) -> Availability:
        return Availability(True, "ok")

    def locate(self, image: ScreenImage, instruction: str) -> Point:
        gnd_calls.append(1)
        return Point(x=50, y=50, confidence=0.9)


def _cfg(**over):
    base = {
        "ocr": {"enabled": True, "chain": ["spy_ocr"]},
        "detection": {"enabled": True, "chain": ["spy_det"]},
        "grounding": {"enabled": True, "chain": ["spy_gnd"]},
        "routing": {"max_tier": "vision"},
        "cache": {"enabled": False},
    }
    base.update(over)
    return make_config(**base)


def _reset() -> None:
    ocr_calls.clear()
    det_calls.clear()
    gnd_calls.clear()


SUBMIT_XML = (
    '<hierarchy rotation="0">'
    '<node class="android.widget.Button" text="Submit" resource-id="x:id/submit" '
    'clickable="true" enabled="true" bounds="[100,200][300,260]"/>'
    '<node class="android.widget.TextView" text="Welcome" enabled="true" bounds="[0,0][200,40]"/>'
    '<node class="android.widget.TextView" text="Help" enabled="true" bounds="[0,60][120,100]"/>'
    "</hierarchy>"
)
EMPTY_XML = '<hierarchy rotation="0"></hierarchy>'


# --------------------------------------------------------------------------- AC12


def test_literal_has_resolves_hierarchy_vision_never_invoked() -> None:
    _reset()
    eng = make_engine(config=_cfg(), device=FakeDevice(text_index={"Continue": (10, 20, 200, 80)}))
    r = eng.has("Continue")
    assert r.found is True and r.source == "hierarchy"
    assert ocr_calls == []  # the OCR fallback (vision) was never invoked


def test_absent_literal_has_is_false() -> None:
    _reset()
    eng = make_engine(config=_cfg(), device=FakeDevice(text_index={}))
    # disable ocr fallback so this is a pure T0 check
    assert eng.has("Nope", ocr_fallback=False).found is False
    assert ocr_calls == []


def test_semantic_query_satisfied_by_hierarchy_no_grounding() -> None:
    _reset()
    eng = make_engine(config=_cfg(), device=FakeDevice(hierarchy_xml=SUBMIT_XML))
    res = eng.analyze(query="the Submit button")
    assert len(res.elements) == 1
    assert res.elements[0].text == "Submit"
    assert res.meta.tier_used.value in ("selector", "hierarchy")
    assert gnd_calls == []  # grounding (paid) never called
    assert det_calls == []  # didn't even need vision


def test_semantic_miss_under_vision_ceiling_does_not_call_paid_grounding() -> None:
    _reset()
    eng = make_engine(
        config=_cfg(routing={"max_tier": "vision"}), device=FakeDevice(hierarchy_xml=EMPTY_XML)
    )
    res = eng.analyze(query="the fuzzy purple widget")
    assert det_calls or ocr_calls  # it DID escalate to local vision
    assert gnd_calls == []  # but NOT to the paid grounding provider
    assert tier_rank(res.meta.tier_used) <= tier_rank(Tier.vision)


def test_deep_opt_in_escalates_to_grounding() -> None:
    _reset()
    eng = make_engine(
        config=_cfg(routing={"max_tier": "vision"}), device=FakeDevice(hierarchy_xml=EMPTY_XML)
    )
    res = eng.analyze(query="the fuzzy purple widget", deep=True)
    assert gnd_calls  # --deep raises the ceiling so grounding runs
    assert res.meta.tier_used.value == "grounding"


# --------------------------------------------------------------------- pure helpers


def test_classify_query() -> None:
    assert classify_query("com.app:id/login") is QueryKind.resource_id
    assert classify_query("the gear icon in the top right") is QueryKind.visual
    assert classify_query("Submit") is QueryKind.literal
    assert (
        classify_query("a longer descriptive sentence with no visual cue tokens")
        is QueryKind.general
    )


def test_salient_tokens_strip_stopwords() -> None:
    toks = salient_tokens("the Submit button")
    assert "submit" in toks and "button" not in toks and "the" not in toks


def test_entry_tier_by_intent() -> None:
    assert entry_tier(Intent.has) is Tier.text
    assert entry_tier(Intent.locate) is Tier.selector
    assert entry_tier(Intent.analyze) is Tier.hierarchy
    assert entry_tier(Intent.query, query="Login") is Tier.selector


def test_resolve_ceiling() -> None:
    assert resolve_ceiling("vision") is Tier.vision
    assert resolve_ceiling("vision", deep=True) is Tier.grounding
    assert resolve_ceiling("vision", cheap=True) is Tier.hierarchy
    assert resolve_ceiling("text", cheap=True) is Tier.text  # floored at cheapest


def test_next_tier_and_allows() -> None:
    assert next_tier(Tier.hierarchy, Tier.vision) is Tier.vision
    assert next_tier(Tier.vision, Tier.vision) is None
    assert allows(Tier.vision, Tier.grounding) is True
    assert allows(Tier.grounding, Tier.vision) is False
