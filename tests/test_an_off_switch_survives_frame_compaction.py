"""An off switch is a reading, not a missing flag.

`compact_element` drops false flags because for `clickable`, `scrollable` and `focused` a
false really is the default and carries nothing. `checked` is not like that: a switch that
is off is the answer to "is it off", and dropping it makes an off switch indistinguishable
from an element that is not a switch at all. The core product already settles this — see
`tests/test_element_state.py` and the "an off switch IS the reading" assertion in
`tests/test_an_action_can_be_asked_for_next_actions.py` — so the controller's model-facing
projection must not disagree with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.compaction import compact_frame  # noqa: E402
from experiments.aua_controller.judgement import evidence_frame  # noqa: E402


def frame() -> dict[str, object]:
    return {
        "ok": True,
        "observation": {
            "screen": {"package": "com.example.demo", "activity": ".NotificationSettings"},
            "meta": {"fingerprint": "f1"},
            "elements": [
                {"id": "rid:sw_promos", "text": "Promotional messages", "checkable": True,
                 "checked": False, "clickable": True, "bounds": [0, 100, 720, 200]},
                {"id": "rid:sw_security", "text": "Security alerts", "checkable": True,
                 "checked": True, "clickable": True, "bounds": [0, 200, 720, 300]},
                {"id": "t:1", "text": "Notifications", "clickable": False, "focused": False,
                 "scrollable": False, "checked": None, "bounds": [0, 0, 720, 100]},
            ],
        },
    }


def elements_by_label(projected: dict) -> dict[str, dict]:
    return {element["text"]: element for element in projected["observation"]["elements"]}


def test_an_off_switch_keeps_its_reading_through_compaction() -> None:
    by_label = elements_by_label(compact_frame(frame()))
    assert by_label["Promotional messages"]["checked"] is False
    assert by_label["Security alerts"]["checked"] is True


def test_an_off_switch_reaches_the_judge() -> None:
    # The judge path adds an id strip on top of compaction; the reading must survive both,
    # or no model in this harness can ever verify "the X switch is off".
    by_label = elements_by_label(evidence_frame(frame()))
    assert by_label["Promotional messages"]["checked"] is False


def test_flags_whose_false_really_is_the_default_are_still_dropped() -> None:
    # The size trim this projection exists for is unchanged: only `checked` is exempt.
    plain = elements_by_label(compact_frame(frame()))["Notifications"]
    assert "clickable" not in plain
    assert "focused" not in plain
    assert "scrollable" not in plain
    # A None checked means the element is no switch at all, so it carries nothing either.
    assert "checked" not in plain


def raw_dump_frame() -> dict[str, object]:
    """A raw hierarchy read: every node carries ``checkable: false, checked: false``."""
    return {
        "ok": True,
        "observation": {
            "screen": {"package": "com.example.demo", "source": "hierarchy"},
            "meta": {"fingerprint": "f-raw"},
            "elements": [
                {"id": "el:clock", "text": "11:28", "resource_id": "com.android.systemui:id/clock",
                 "clickable": False, "checkable": False, "checked": False, "bounds": [0, 0, 100, 40]},
                {"id": "el:login", "text": "Log in", "clickable": True, "checkable": False,
                 "checked": False, "bounds": [0, 100, 720, 200]},
                {"id": "el:dark", "text": "Dark mode", "clickable": True, "checkable": True,
                 "checked": False, "bounds": [0, 200, 720, 300]},
            ],
        },
    }


def test_a_raw_dumps_checked_false_on_a_non_switch_is_dropped() -> None:
    """Seen on the first frame of every row: 22 status-bar nodes each carried ``checked: false``
    and became "switches" in the navigator's menu. Only a checkable node has a reading."""
    by_label = elements_by_label(compact_frame(raw_dump_frame()))
    assert "11:28" not in by_label, "the status-bar clock is not part of the app's screen at all"
    assert "checked" not in by_label["Log in"], "a plain button is not a switch"
    assert by_label["Dark mode"]["checked"] is False, "a real off switch keeps its reading"


def test_the_navigator_offers_no_status_bar_switches() -> None:
    from experiments.aua_controller.typesafe_navigator import candidates
    options = candidates(compact_frame(raw_dump_frame(), keep_ids=True)["observation"])
    assert set(options) == {"el:login", "el:dark"}
    assert options["el:login"] == "Log in" and options["el:dark"].endswith("[switch is OFF]")
