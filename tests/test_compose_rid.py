"""Resource-id containers stay addressable (Compose opacity fix)."""

from __future__ import annotations

from android_ui_analyser.hierarchy import parse_hierarchy

XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node class="android.view.View" clickable="true" bounds="[0,100][1080,800]">
    <node class="android.view.View" resource-id="com.app:id/containerChatDetail"
          bounds="[0,100][1080,700]"/>
    <node class="android.widget.TextView" text="Hello" bounds="[40,200][400,260]"/>
  </node>
</hierarchy>"""


def test_resource_id_container_not_absorbed() -> None:
    els = parse_hierarchy(XML, (1080, 1920))
    rids = {e.resource_id for e in els if e.resource_id}
    assert "com.app:id/containerChatDetail" in rids
