"""Rich wait --for timeout diagnostics."""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.schema import MatchMode
from conftest import FakeDevice, make_config

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.widget.TextView" text="Hello there" bounds="[0,0][1080,80]"/>
  <node class="android.widget.Button" text="Continue" clickable="true"
        bounds="[40,200][1040,280]"/>
</hierarchy>"""


def test_wait_timeout_detail_names_mode_and_candidates() -> None:
    eng = Engine(make_config(), device=FakeDevice(hierarchy_xml=_XML))
    res = eng.wait(for_="(Hi|Hello)", match="contains", timeout_ms=50)
    assert res.ok is False
    assert "match=contains" in (res.detail or "")
    assert "regex" in (res.detail or "").lower()
    assert "closest" in (res.detail or "").lower()


def test_wait_timeout_message_helper() -> None:
    eng = Engine(make_config(), device=FakeDevice(hierarchy_xml=_XML))
    msg = eng._wait_timeout_message(
        "(Hi|Hello)", mode=MatchMode.contains, by="text", ignore_case=False, absent=False
    )
    assert "match=contains" in msg
    assert "--match regex" in msg
