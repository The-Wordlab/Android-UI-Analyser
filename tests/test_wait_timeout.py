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


def test_wait_timeout_detail_names_mode_without_a_new_read_after_expiry(monkeypatch) -> None:
    eng = Engine(make_config(), device=FakeDevice(hierarchy_xml=_XML))

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("an exhausted wait must not start timeout-diagnostic analysis")

    monkeypatch.setattr(eng, "analyze", unexpected_analysis)
    res = eng.wait(for_="(Hi|Hello)", match="contains", timeout_ms=50)
    assert res.ok is False
    assert "match=contains" in (res.detail or "")
    assert "regex" in (res.detail or "").lower()
    assert "budget was exhausted" in (res.note or "")


def test_wait_timeout_message_helper() -> None:
    eng = Engine(make_config(), device=FakeDevice(hierarchy_xml=_XML))
    msg = eng._wait_timeout_message(
        "(Hi|Hello)", mode=MatchMode.contains, by="text", ignore_case=False, absent=False
    )
    assert "match=contains" in msg
    assert "--match regex" in msg
    assert "closest" in msg
