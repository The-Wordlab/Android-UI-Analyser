"""`--rid 49` is the id/rid conflation, and the error should say so.

Every TSV row carries both an `id` (this analyze's ordinal) and a `rid` (the app's resource-id),
so reaching for `--rid 49` after reading `id=49` off a row is the obvious mistake, not a careless
one. Measured 2026-08-10: a fresh agent did exactly that, and the old answer — "nearest:
action_bar_root | System UI notification:" — sent it hunting for a spelling mistake that did not
exist, costing a re-analyze and a second attempt.

A digit-only resource-id is never a real one, so the miss is diagnosable with certainty. The
nearest-element list stays for every other miss, where a typo genuinely is the likely cause.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import SelectorNotFoundError
from conftest import FakeDevice, make_config

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.Button" text="Create item"
        resource-id="catalogSearchCreateItem" clickable="true" enabled="true"
        bounds="[40,200][1040,320]"/>
</hierarchy>"""


def _engine() -> Engine:
    return Engine(make_config(), device=FakeDevice(hierarchy_xml=_XML))


def _miss(rid: str) -> SelectorNotFoundError:
    with pytest.raises(SelectorNotFoundError) as excinfo:
        _engine().resolve_selector(rid=rid)
    return excinfo.value


def test_a_numeric_rid_is_told_it_is_an_element_id() -> None:
    err = _miss("49")
    assert "element id" in (err.hint or ""), err.hint
    assert "`aua tap-and-analyze 49`" in (err.hint or ""), err.hint


def test_the_numeric_hint_does_not_bury_the_advice_in_nearest_elements() -> None:
    assert "nearest:" not in (_miss("49").hint or "")


def test_a_real_typo_still_gets_the_nearest_elements() -> None:
    """The diagnosis is only certain for digits; everything else is still probably a typo."""
    hint = _miss("catalogSearchCreateItemm").hint or ""
    assert "nearest:" in hint, hint
    assert "element id" not in hint, hint
