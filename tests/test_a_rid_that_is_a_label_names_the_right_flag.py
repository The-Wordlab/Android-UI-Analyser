"""`--rid 'Sample settings'` is a label passed as a resource-id, and the miss should say so.

`--rid` matches resource-ids only, but on a screen whose controls have no resource-id, the string
an agent can see is the content-desc or the text. Reported 2026-09-30 from a web prototype: the
first try was `--rid` with the button's desc, and the answer listed the nearest elements, leaving
the caller to spot that the "typo" was an exact match in the wrong column.

When the value is exactly an element's desc or text, the diagnosis is certain, so the hint names
the flag that matches it instead of a nearest-element list. `--rid` still never falls back on its
own: a selector that quietly searches another field is a guess.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import SelectorNotFoundError
from conftest import FakeDevice, make_config

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.ImageButton" content-desc="Sample settings"
        clickable="true" enabled="true" bounds="[900,40][1040,160]"/>
  <node index="1" class="android.widget.Button" text="Create item"
        resource-id="catalogCreateItem" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


def _miss(rid: str) -> str:
    engine = Engine(make_config(), device=FakeDevice(hierarchy_xml=_XML))
    with pytest.raises(SelectorNotFoundError) as excinfo:
        engine.resolve_selector(rid=rid)
    return excinfo.value.hint or ""


def test_a_desc_passed_as_rid_is_pointed_at_desc() -> None:
    hint = _miss("Sample settings")
    assert "--desc 'Sample settings'" in hint, hint
    assert "nearest:" not in hint, hint


def test_a_text_passed_as_rid_is_pointed_at_text() -> None:
    hint = _miss("Create item")
    assert "--text 'Create item'" in hint, hint


def test_the_match_ignores_case_and_quotes_the_screen_spelling() -> None:
    hint = _miss("sample settings")
    assert "--desc 'Sample settings'" in hint, hint


def test_a_real_typo_still_gets_the_nearest_elements() -> None:
    hint = _miss("catalogCreateItemm")
    assert "nearest:" in hint, hint
    assert "--desc" not in hint and "--text" not in hint, hint
