"""The CLI and MCP must return the same shape of folded observation.

They had already drifted. The observation-cost work measured that 37 taps produced 73 separate
`analyze` calls and made the CLI's folded observation compact — but MCP applied no projection at
all, so every action returned every field of every element. Then the rename stripped MCP's `observe`
switch, leaving that surface with a full dump on every action and no way to ask for less.

Both now go through `trim_observation_payload`. One derived list is left to filter — the opt-in
`next_actions` (`output.next_actions`), and `stable_elements` is gone — so the trim must both
filter it when it is there and stay a no-op when it is not. Unfiltered, an unprojected copy
re-adds every system-bar node the view just dropped, and worse, names an id that is absent from
the observation the caller was handed: the same "a value you cannot distinguish from its absence"
failure this engine keeps producing.
"""

from __future__ import annotations

from android_ui_analyser.projection import Projection, trim_observation_payload
from android_ui_analyser.schema import OutputFormat

_SPEC = "id,text,desc,rid,clickable,enabled,checked"


def _payload() -> dict:
    return {
        "ok": True,
        "action": "tap",
        "observation": {
            "schema_version": 1,
            "screen": {"width": 1080, "height": 2400, "package": "com.example.app"},
            "elements": [
                {"id": 0, "resource_id": "com.android.systemui:id/status_bar", "window": "system"},
                {"id": 2, "text": "Send", "clickable": True, "window": "app"},
                {"id": 3, "text": "Cancel", "clickable": True, "window": "app"},
            ],
        },
        "next_actions": [{"id": 0}, {"id": 2, "label": "Send"}, {"id": 3, "label": "Cancel"}],
    }


def _trimmed() -> dict:
    view = Projection.for_observation(_SPEC, fmt=OutputFormat.json)
    return trim_observation_payload(_payload(), view, fmt=OutputFormat.json)


def test_the_observation_is_trimmed() -> None:
    ids = {e["id"] for e in _trimmed()["observation"]["elements"]}
    assert 0 not in ids, "system chrome is where the cost win comes from"
    assert {2, 3} <= ids, "and the app's own controls must survive"


def test_derived_lists_cannot_name_a_dropped_element() -> None:
    out = _trimmed()
    kept = {e["id"] for e in out["observation"]["elements"]}
    for key in ("next_actions",):
        named = {r["id"] for r in out[key]}
        assert named <= kept, f"{key} names ids absent from the observation: {named - kept}"


def test_the_trim_is_a_no_op_when_the_opt_in_list_is_absent() -> None:
    """`next_actions` is off by default, which is now the *normal* payload shape here."""
    payload = _payload()
    payload.pop("next_actions")
    view = Projection.for_observation(_SPEC, fmt=OutputFormat.json)
    out = trim_observation_payload(payload, view, fmt=OutputFormat.json)
    assert "next_actions" not in out
    assert {e["id"] for e in out["observation"]["elements"]} == {2, 3}, (
        "an absent derived list must not stop the observation itself from being trimmed"
    )


def test_all_means_all_and_leaves_the_payload_untouched() -> None:
    view = Projection.for_observation("all", fmt=OutputFormat.json)
    assert view is None, "'all' must mean 'do not touch'"
    out = trim_observation_payload(_payload(), view, fmt=OutputFormat.json)
    assert len(out["observation"]["elements"]) == 3
    assert len(out["next_actions"]) == 3


def test_a_payload_without_an_observation_is_returned_unchanged() -> None:
    view = Projection.for_observation(_SPEC, fmt=OutputFormat.json)
    bare = {"ok": True, "action": "screenshot", "detail": "/tmp/x.png"}
    assert trim_observation_payload(dict(bare), view, fmt=OutputFormat.json) == bare


def test_mcp_advertises_the_width_dial_it_kept() -> None:
    """`observe` contradicted `tap_and_analyze`; a width control does not, and MCP needs one."""
    from android_ui_analyser import mcp_server

    props = mcp_server._OBSERVE_FIELDS_PROP
    assert props["type"] == "string" and "all" in props["description"]
