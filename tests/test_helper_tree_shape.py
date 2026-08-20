"""A helper tree must normalize to the same elements the adb XML dump would produce.

The point of translating rather than writing a second normalizer: element semantics — what
counts as interesting, how a label is chosen, which nodes are system chrome — are subtle and
already settled in one place. Two implementations of that would drift apart quietly, and the
symptom would be an agent seeing different elements depending on which path fetched them.
"""

from __future__ import annotations

from android_ui_analyser import device_agent
from android_ui_analyser.config import Config
from android_ui_analyser.platforms.android import AndroidPlatform

SCREEN = (720, 1280)

# Shaped like a real `ui.tree` reply: a scrollable container holding a clickable row whose
# label lives on a child TextView, which is the arrangement that makes labelling non-obvious.
HELPER_TREE = {
    "roots": [
        {
            "class": "android.widget.FrameLayout",
            "package": "dev.example.app",
            "text": "",
            "desc": "",
            "rid": "",
            "bounds": "[0,0][720,1280]",
            "clickable": False,
            "enabled": True,
            "visible": True,
            "children": [
                {
                    "class": "androidx.recyclerview.widget.RecyclerView",
                    "package": "dev.example.app",
                    "text": "",
                    "desc": "",
                    "rid": "dev.example.app:id/list",
                    "bounds": "[0,100][720,1200]",
                    "scrollable": True,
                    "enabled": True,
                    "visible": True,
                    "children": [
                        {
                            "class": "android.widget.LinearLayout",
                            "package": "dev.example.app",
                            "text": "",
                            "desc": "",
                            "rid": "dev.example.app:id/row",
                            "bounds": "[0,100][720,220]",
                            "clickable": True,
                            "enabled": True,
                            "visible": True,
                            "children": [
                                {
                                    "class": "android.widget.TextView",
                                    "package": "dev.example.app",
                                    "text": "Network & internet",
                                    "desc": "",
                                    "rid": "dev.example.app:id/title",
                                    "bounds": "[24,130][400,190]",
                                    "clickable": False,
                                    "enabled": True,
                                    "visible": True,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ],
    "truncated": False,
}


def _normalize(xml: str):
    return AndroidPlatform(Config()).normalize_tree(xml, SCREEN)


def test_a_helper_tree_normalizes_into_real_elements() -> None:
    result = _normalize(device_agent.tree_to_xml(HELPER_TREE))

    assert result.app_id == "dev.example.app"
    labels = [e.text or e.content_desc for e in result.elements]
    assert any(label and "Network & internet" in label for label in labels), (
        f"the row label did not survive translation: {labels}"
    )


def test_the_attribute_names_the_normalizer_reads_are_all_translated() -> None:
    """`rid`/`desc`/`long_clickable` are the three that differ; a miss is silent."""

    xml = device_agent.tree_to_xml(HELPER_TREE)
    for xml_name in ("resource-id", "content-desc", "bounds", "class", "package", "text"):
        assert f"{xml_name}=" in xml, f"{xml_name} never made it into the XML"
    assert "rid=" not in xml and "desc=" not in xml.replace("content-desc=", ""), (
        "an accessibility-side name leaked through untranslated"
    )


def test_booleans_become_the_strings_the_normalizer_tests_for() -> None:
    xml = device_agent.tree_to_xml(HELPER_TREE)
    assert 'scrollable="true"' in xml
    assert 'clickable="false"' in xml, "False must be rendered, not omitted"


def test_a_label_with_xml_metacharacters_survives() -> None:
    """`&` is ordinary in UI copy ("Network & internet") and fatal in unescaped XML."""

    tree = {"roots": [{"class": "android.widget.TextView", "package": "p",
                       "text": 'A & B <c> "d"', "bounds": "[0,0][10,10]",
                       "enabled": True, "visible": True}]}
    result = _normalize(device_agent.tree_to_xml(tree))
    assert any((e.text or "") == 'A & B <c> "d"' for e in result.elements)


def test_an_empty_tree_is_not_a_crash() -> None:
    assert _normalize(device_agent.tree_to_xml({"roots": []})).elements == []
