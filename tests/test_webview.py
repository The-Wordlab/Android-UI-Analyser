"""WebView enrichment before OCR."""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.webview import enrich_from_hierarchy, parse_dom_html, should_try_webview

FIXTURES = Path(__file__).parent / "fixtures"

RICH_INNER = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.webkit.WebView" bounds="[0,0][1080,1920]">
    <node index="0" class="android.view.View" text="Welcome" bounds="[40,100][1040,160]"/>
    <node index="1" class="android.widget.Button" text="Sign in" clickable="true"
          bounds="[40,200][1040,280]"/>
    <node index="2" class="android.view.View" content-desc="Help link"
          clickable="true" bounds="[40,300][400,360]"/>
  </node>
</hierarchy>"""


def test_enrich_pulls_webview_children() -> None:
    els = enrich_from_hierarchy(RICH_INNER, screen_size=(1080, 1920))
    assert len(els) >= 3
    assert all(e.source.value == "webview" for e in els)
    assert any(e.text == "Sign in" for e in els)


def test_should_try_on_hollow_fixture() -> None:
    xml = (FIXTURES / "webview_hollow.xml").read_text(encoding="utf-8")
    from android_ui_analyser.hierarchy import parse_hierarchy

    els = parse_hierarchy(xml, (1080, 1920))
    assert should_try_webview(els, xml)


def test_parse_dom_html_links() -> None:
    html = '<html><body><a href="/x">Go</a><button>OK</button><p>noise</p></body></html>'
    els = parse_dom_html(html, frame=(0, 0, 400, 800))
    labels = {e.text for e in els}
    assert "Go" in labels
    assert "OK" in labels
