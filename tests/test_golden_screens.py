"""Golden-screen fixtures → projection / expect stability."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import OutputFormat
from conftest import FakeDevice, make_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_golden_projection_tsv_nonempty() -> None:
    xml = (FIXTURES / "normal_views.xml").read_text(encoding="utf-8")
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=xml)
    engine = Engine(cfg, device=device)
    result = engine.analyze(source="hierarchy")
    proj = Projection.parse(fmt=OutputFormat.tsv, fields="id,text,rid", nonempty=True)
    view = proj.apply(result.as_dict(OutputFormat.json))
    assert view["elements"]
    assert all("id" in el for el in view["elements"])


def test_golden_expect_login_visible() -> None:
    xml = (FIXTURES / "normal_views.xml").read_text(encoding="utf-8")
    cfg = make_config()
    device = FakeDevice(hierarchy_xml=xml)
    engine = Engine(cfg, device=device)
    engine.analyze(source="hierarchy")
    out = engine.expect(text="Log in", exists=True)
    assert out.ok
