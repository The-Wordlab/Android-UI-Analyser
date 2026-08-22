"""The warm daemon is a surface too, and it was publishing frame ordinals.

Every response the daemon returns is built by `model_dump(mode="json")` — 53 call sites, all
funnelled through `_result_ok`. That is the internal form, so the daemon handed back element
ids as reading-order integers while the *same payload* carried stable ids in the parts the
engine builds itself (`id`, `next_actions`, `meta.element_diff`). One response, two id spaces,
and the numeric half is the half a caller cannot resolve — it is validated against a shared
per-device cache file whose last writer is whoever ran most recently.

That is what the dashboard displays, because the dashboard reads the daemon directly, so the
inconsistency was visible on screen: `"id": 20` in `elements` next to
`"id": "rid:switchWidget#1"` in `next_actions`, naming the same control.

`acting.id` had the same problem for the same reason — it is the id of the node that actually
received the action, so reporting an ordinal there means the one field explaining "I aimed at
the label but pressed its container" cannot be looked up in the observation beside it.
"""

from __future__ import annotations

from typing import Any

import pytest

from android_ui_analyser.schema import (
    ActionResult,
    AnalyzeResult,
    Element,
    Meta,
    Screen,
)

PACKAGE = "com.example.fiction"


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[
            Element(
                id=20,
                type="LinearLayout",
                text="Bluetooth tethering",
                bounds=[0, 992, 1080, 1227],
                center=[540, 1109],
                clickable=True,
                enabled=True,
                window="app",
            ),
            Element(
                id=21,
                type="Switch",
                resource_id=f"{PACKAGE}:id/switchWidget",
                bounds=[859, 1046, 996, 1172],
                center=[927, 1109],
                clickable=True,
                checkable=True,
                checked=False,
                parent=20,
                window="app",
            ),
        ],
        meta=Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy"),
    )


def _daemon_response(result: Any) -> dict[str, Any]:
    """What a client actually receives from the daemon for *result*."""
    from android_ui_analyser import daemon as daemon_mod

    return daemon_mod._result_ok(result.model_dump(mode="json"))


def test_the_daemon_publishes_stable_ids_in_the_observation() -> None:
    payload = _daemon_response(_observation())

    ids = [e["id"] for e in payload["result"]["elements"]]
    assert all(isinstance(i, str) for i in ids), f"the daemon published ordinals: {ids}"


def test_the_daemon_publishes_stable_ids_inside_an_action() -> None:
    """The nested observation is the part the dashboard draws boxes from."""
    action = ActionResult(ok=True, action="tap", id="tx:abc", observation=_observation())

    payload = _daemon_response(action)

    ids = [e["id"] for e in payload["result"]["observation"]["elements"]]
    assert all(isinstance(i, str) for i in ids), f"the daemon published ordinals: {ids}"


def test_one_response_never_mixes_id_kinds() -> None:
    """The defect the dashboard made visible: two id spaces in a single payload."""
    action = ActionResult(
        ok=True,
        action="tap",
        id="rid:switchWidget#1",
        observation=_observation(),
    )

    payload = _daemon_response(action)["result"]
    element_ids = {type(e["id"]) for e in payload["observation"]["elements"]}

    assert element_ids == {type(payload["id"])}, (
        "the acted id and the observation's ids must be the same kind of name, or the caller "
        "cannot look up the thing the response says it touched"
    )


def test_publishing_is_idempotent() -> None:
    """A payload that crosses two boundaries must not be rewritten twice into nonsense."""
    from android_ui_analyser.schema import publish_ids

    once = _daemon_response(_observation())["result"]
    twice = publish_ids(dict(once))

    assert [e["id"] for e in twice["elements"]] == [e["id"] for e in once["elements"]]


@pytest.mark.parametrize("field", ["parent"])
def test_element_relations_are_published_too(field: str) -> None:
    """A `parent` still holding an ordinal points at a different element than its own id."""
    payload = _daemon_response(_observation())["result"]

    parents = [e.get(field) for e in payload["elements"] if e.get(field) is not None]
    assert parents, "the fixture must exercise a parent pointer"
    assert all(isinstance(p, str) for p in parents), f"{field} kept an ordinal: {parents}"


def test_a_changed_diff_row_does_not_crash_publishing() -> None:
    """`element_diff.changed` holds dicts, not ids, and one was used as a dict key.

    `added`/`removed` are flat id lists, so remapping them by lookup is right; `changed` is a
    list of `{"id": ..., "text": {"from": ..., "to": ...}}` rows, and `by_ordinal.get(row)`
    on one of those raises `TypeError: unhashable type: 'dict'`. It only fires when something
    actually changed between two frames, which is why it surfaced as an intermittent
    `internal_error` on real taps and never in a test — the fixtures had empty diffs.
    """
    from android_ui_analyser.schema import publish_ids

    payload = {
        "elements": [
            {"id": 20, "stable_key": "tx:aaaa", "bounds": [0, 0, 1, 1]},
            {"id": 21, "stable_key": "rid:bbbb", "bounds": [0, 0, 1, 1]},
        ],
        "meta": {
            "element_diff": {
                "added": [21],
                "removed": [99],
                "changed": [{"id": 20, "text": {"from": "a", "to": "b"}}],
            }
        },
    }

    published = publish_ids(payload)
    diff = published["meta"]["element_diff"]

    assert diff["added"] == ["rid:bbbb"]
    assert diff["removed"] == [99], "an id that left the screen has no key here to map to"
    assert diff["changed"] == [{"id": "tx:aaaa", "text": {"from": "a", "to": "b"}}]
