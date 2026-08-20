"""Turning what the device announced into steps — including what it could not announce.

The device's recorder is a faithful log of what the accessibility framework *reported*, and
that is not the same as what the user did: a view only emits a click event if it actually
calls performClick, and plenty do not. Tapping a row in one stock list produced 35 content
changes and no click at all.

That makes silence ambiguous, and ambiguity is the whole problem — a draft flow that quietly
omits a step looks exactly like a draft flow that is complete. So the device reports the
shadow a missed action leaves (the screen changed with nothing announced before it), and this
module's job is to carry that admission through to the person reading the draft rather than
quietly dropping it.
"""

from __future__ import annotations

from typing import Any

from android_ui_analyser.recordings import steps_from_recording


def _click(rid: str | None = None, label: str | None = None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": "tap", "ts": 1, "package": "com.example.placeholder"}
    if rid:
        row["resource_id"] = rid
    if label:
        row["label"] = label
    row.update(extra)
    return row


def _gap(ms: int = 2) -> dict[str, Any]:
    return {
        "kind": "gap",
        "reason": "screen_changed_with_no_announced_action",
        "ts": ms,
        "package": "com.example.placeholder",
    }


def test_an_announced_tap_becomes_a_step_with_its_selector() -> None:
    draft = steps_from_recording([_click(rid="rowSettings", label="Settings")])

    assert len(draft.steps) == 1
    step = draft.steps[0]
    assert step.kind == "tap"
    assert step.resource_id == "rowSettings"
    assert step.label == "Settings"


def test_a_hole_in_the_recording_is_reported_where_it_happened() -> None:
    """The draft must say where it stopped being able to see, not just how much it saw."""

    draft = steps_from_recording([_click(rid="rowOne"), _gap(), _click(rid="rowTwo")])

    assert len(draft.steps) == 2
    assert len(draft.gaps) == 1
    assert draft.gaps[0].after_step == 1, (
        "the gap was not anchored to the step it followed, so a reader cannot tell where "
        "the recording went blind"
    )
    assert draft.complete is False


def test_a_recording_with_nothing_missing_says_so() -> None:
    draft = steps_from_recording([_click(rid="rowOne"), _click(rid="rowTwo")])

    assert draft.gaps == []
    assert draft.complete is True


def test_a_missed_action_before_the_first_step_is_still_reported() -> None:
    """The dangerous case, and the one an earlier version threw away.

    Arming noise was suppressed by *position* — any gap before the first recorded step was
    dropped as "the app coming to the foreground". Start recording on an app that is already
    open, tap a row Android does not announce, and that is precisely the shape of a real
    missed action: nothing announced, nothing recorded yet, screen changes. The draft came
    back `complete` while missing its opening step, which is the exact failure this whole
    mechanism exists to prevent.

    Arming noise is suppressed on the device instead, by time since `record.start` — which is
    what actually distinguishes "the recorder just attached" from "the user did something".
    """

    draft = steps_from_recording([_gap(), _click(rid="rowOne")])

    assert len(draft.gaps) == 1
    assert draft.gaps[0].after_step == 0, "a gap before any step must anchor at 0"
    assert draft.complete is False
    assert len(draft.steps) == 1


def test_a_typed_value_never_survives_the_recording() -> None:
    """What someone typed is exactly what must not end up in a shareable file."""

    draft = steps_from_recording(
        [{"kind": "input", "resource_id": "fieldEmail", "content_desc": "a@example.test"}]
    )

    step = draft.steps[0]
    assert step.text is None or step.text.startswith("${"), (
        f"a typed value reached the draft: {step.text!r}"
    )
    assert step.content_desc is None, (
        "content_desc survived on an input step — a widget that mirrors what was typed into "
        "its description would leak the value the label suppression exists to protect"
    )


def test_typing_into_one_field_is_one_step() -> None:
    """Every keystroke fires an event; a draft with forty input steps is unusable."""

    rows = [{"kind": "input", "resource_id": "fieldEmail"} for _ in range(5)]
    draft = steps_from_recording(rows)

    assert len(draft.steps) == 1


def test_typing_into_two_unlabelled_fields_stays_two_steps() -> None:
    """The collapse must key on identity, and 'no identity' is not a shared identity.

    Two adjacent text fields with neither a resource id nor a description both key to
    "nothing". Collapsing on that merges two genuinely different fields into one step and one
    parameter, and the draft then silently types the same value into the wrong place.
    """

    draft = steps_from_recording([{"kind": "input"}, {"kind": "input"}])

    assert len(draft.steps) == 2, (
        "two unidentifiable input fields collapsed into one step"
    )


def test_typing_into_two_named_fields_stays_two_steps() -> None:
    draft = steps_from_recording(
        [{"kind": "input", "resource_id": "fieldEmail"}, {"kind": "input", "resource_id": "fieldPin"}]
    )

    assert len(draft.steps) == 2


def test_a_kind_the_host_cannot_replay_is_named_rather_than_dropped() -> None:
    """A scroll's container and extent were never captured, so the draft must not pretend."""

    draft = steps_from_recording([_click(rid="rowOne"), {"kind": "scroll", "arg": "up"}])

    assert any("scroll" in blocker for blocker in draft.blockers), (
        f"a lossy recorded kind was advertised as replayable: {draft.blockers}"
    )
    assert draft.complete is False


def test_an_unrecognised_row_is_ignored_rather_than_crashing() -> None:
    """The device may learn to report things this version has never heard of."""

    draft = steps_from_recording(
        [{"kind": "something-new"}, _click(rid="rowOne"), {}, "not a row"]
    )

    assert len(draft.steps) == 1


def test_an_empty_recording_is_not_an_error() -> None:
    draft = steps_from_recording([])

    assert draft.steps == []
    assert draft.complete is True


def test_a_burst_of_screen_changes_is_one_hole_not_several() -> None:
    """One navigation reports several window changes, and each is not a separate missed tap.

    The device debounces this, but the host must not depend on that alone: the two run
    independently, an older helper does not debounce at all, and a gap count that inflates
    with the device's animation timing is not a number anyone can act on. Reporting three
    holes where there is one sends a human hunting for two taps that were never missed.
    """

    draft = steps_from_recording([_click(rid="rowOne"), _gap(), _gap(), _gap()])

    assert len(draft.gaps) == 1
    assert draft.gaps[0].after_step == 1


# -- recovering the taps Android never announced ----------------------------
#
# The kernel touch stream says exactly where a finger landed, including for the presses no
# view bothered to announce. On its own that is only a coordinate; what turns it into a
# selector is the snapshot of what was pressable at that moment, which the device keeps as the
# screen changes. These are the rules for putting the two together.


def _touch(x: int, y: int, ms: int) -> Any:
    from android_ui_analyser.device_agent import Touch

    return Touch(x=x, y=y, down_ms=ms, up_ms=ms + 60, travel_px=0)


def _snapshot(ms: int, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ts": ms, "nodes": nodes}


_ROW = {"label": "Network settings Mobile, hotspot", "bounds": [0, 600, 1080, 800]}
_OTHER = {"resource_id": "rowSecurity", "bounds": [0, 810, 1080, 1010]}


def test_an_unannounced_tap_is_recovered_as_a_named_step() -> None:
    """The whole point: a press Android stayed silent about still gets a real selector."""

    draft = steps_from_recording(
        [_gap(5_200)],
        touches=[_touch(540, 700, 5_000)],
        snapshots=[_snapshot(4_800, [_ROW, _OTHER])],
    )

    assert len(draft.steps) == 1
    step = draft.steps[0]
    assert step.kind == "tap"
    assert step.label == "Network settings Mobile, hotspot"
    assert draft.recovered == 1


def test_recovering_a_tap_closes_the_gap_it_explains() -> None:
    """A hole with a known cause is no longer a hole, and must stop being reported as one."""

    draft = steps_from_recording(
        [_gap(5_200)],
        touches=[_touch(540, 700, 5_000)],
        snapshots=[_snapshot(4_800, [_ROW])],
    )

    assert draft.gaps == []
    assert draft.complete is True


def test_a_tap_the_framework_did_announce_is_not_recorded_twice() -> None:
    """Both sources see the same press; only one step may come out of it."""

    rows = [{"kind": "tap", "resource_id": "rowSecurity", "ts": 5_000}]
    draft = steps_from_recording(
        rows,
        touches=[_touch(540, 900, 5_050)],
        snapshots=[_snapshot(4_800, [_ROW, _OTHER])],
    )

    assert len(draft.steps) == 1
    assert draft.steps[0].resource_id == "rowSecurity"
    assert draft.recovered == 0


def test_a_tap_on_nothing_pressable_does_not_invent_a_selector() -> None:
    """Padding between rows is a real thing to tap, and naming it would be a lie.

    Falling back to coordinates keeps the step honest — it records where the finger went and
    nothing more — while `recorded_step_blockers` still refuses to call the draft replayable.
    """

    draft = steps_from_recording(
        [_gap(5_200)],
        touches=[_touch(540, 805, 5_000)],  # in the gutter between the two rows
        snapshots=[_snapshot(4_800, [_ROW, _OTHER])],
    )

    assert len(draft.steps) == 1
    assert draft.steps[0].kind == "tap-point"
    assert draft.steps[0].arg == "540,805"


def test_the_smallest_pressable_thing_under_the_finger_wins() -> None:
    """Rows nest inside containers; naming the container would tap the wrong thing."""

    outer = {"resource_id": "listContainer", "bounds": [0, 0, 1080, 2000]}
    inner = {"resource_id": "rowInner", "bounds": [0, 600, 1080, 800]}
    draft = steps_from_recording(
        [_gap()],
        touches=[_touch(540, 700, 5_000)],
        snapshots=[_snapshot(4_800, [outer, inner])],
    )

    assert draft.steps[0].resource_id == "rowInner"


def test_the_snapshot_used_is_the_one_from_before_the_screen_moved() -> None:
    """A tap navigates, so the newest snapshot describes where it went, not what was hit."""

    before = _snapshot(4_800, [{"resource_id": "rowBefore", "bounds": [0, 600, 1080, 800]}])
    after = _snapshot(5_400, [{"resource_id": "rowAfter", "bounds": [0, 600, 1080, 800]}])
    draft = steps_from_recording(
        [_gap(5_200)],
        touches=[_touch(540, 700, 5_000)],
        snapshots=[before, after],
    )

    assert draft.steps[0].resource_id == "rowBefore"


def test_recovered_taps_land_in_the_order_they_happened() -> None:
    rows = [{"kind": "tap", "resource_id": "rowSecurity", "ts": 6_000}]
    draft = steps_from_recording(
        rows,
        touches=[_touch(540, 700, 5_000)],
        snapshots=[_snapshot(4_800, [_ROW, _OTHER])],
    )

    assert [s.label or s.resource_id for s in draft.steps] == [
        "Network settings Mobile, hotspot",
        "rowSecurity",
    ]


def test_a_gap_with_no_touch_behind_it_is_still_reported() -> None:
    """Recovery must not become a way of quietly declaring every recording complete."""

    draft = steps_from_recording(
        [_click(rid="rowOne"), _gap()], touches=[], snapshots=[_snapshot(4_800, [_ROW])]
    )

    assert len(draft.gaps) == 1
    assert draft.complete is False


def test_a_drag_is_not_mistaken_for_a_tap() -> None:
    """A scroll is a finger too, and replaying it as a tap would press whatever it started on."""

    from android_ui_analyser.device_agent import Touch

    swipe = Touch(x=540, y=700, down_ms=5_000, up_ms=5_300, travel_px=600)
    draft = steps_from_recording(
        [_gap(5_200)], touches=[swipe], snapshots=[_snapshot(4_800, [_ROW])]
    )

    assert draft.steps == []
    assert len(draft.gaps) == 1, "the gap was closed by something that was not a tap"


def test_a_control_with_no_accessible_name_is_named_as_a_finding() -> None:
    """A coordinate-only step is evidence about the app, and should read that way.

    Half the pressable controls on one real screen had no text, no content description and no
    resource id — the back arrow among them. Nothing can name those, and inventing a label
    would replay as a press on whatever happened to be nearby. But "step 11 is coordinates"
    tells the reader nothing actionable, whereas "this control has no accessible name, here is
    where it is" tells them what to fix to make the flow durable — and is a genuine
    accessibility defect in its own right.
    """

    blank = {"bounds": [21, 147, 147, 273]}   # pressable, but nothing to call it
    draft = steps_from_recording(
        [],
        touches=[_touch(86, 243, 5_000)],
        snapshots=[_snapshot(4_800, [blank])],
    )

    assert draft.steps[0].kind == "tap-point"
    assert len(draft.unnamed_controls) == 1
    found = draft.unnamed_controls[0]
    assert found.step == 1
    assert found.bounds == (21, 147, 147, 273), "the reader cannot find the control"


def test_a_tap_that_landed_on_nothing_is_not_reported_as_an_unnamed_control() -> None:
    """Padding is not a control, and telling someone to go label it wastes their time."""

    draft = steps_from_recording(
        [],
        touches=[_touch(500, 1500, 5_000)],
        snapshots=[_snapshot(4_800, [{"bounds": [0, 0, 100, 100]}])],
    )

    assert draft.steps[0].kind == "tap-point"
    assert draft.unnamed_controls == []


def test_a_screen_that_moved_with_nobody_touching_it_is_not_a_missed_tap() -> None:
    """With the touch stream running, "no finger" is positive evidence, not absence of it.

    A gap means "the screen changed and I cannot explain it". That only implies a missed
    action if there *was* an action. Once every press is being recorded, a change with no
    press anywhere near it is the app moving on its own — a toast landing, a request
    returning — and reporting it sends someone hunting for a tap that never happened. One
    showed up on the first real journey, two and a half seconds after a copy.
    """

    draft = steps_from_recording(
        [_click(rid="rowOne"), _gap(20_000)],
        touches=[_touch(540, 700, 5_000)],
        snapshots=[],
        touch_capture=True,
    )

    assert draft.gaps == []
    assert draft.app_initiated_changes == 1
    assert draft.complete is True


def test_without_the_touch_stream_every_unexplained_change_is_still_reported() -> None:
    """The suppression above is only sound because the finger record is complete.

    On a target that cannot give up its touch stream there is no evidence either way, and
    silence has to go back to meaning "something may have been missed".
    """

    draft = steps_from_recording(
        [_click(rid="rowOne"), _gap(20_000)], touches=[], snapshots=[], touch_capture=False
    )

    assert len(draft.gaps) == 1
    assert draft.complete is False


def test_a_press_read_as_a_drag_still_leaves_the_hole_reported() -> None:
    """This is the one way an action can go missing while the touch stream is running.

    Every press that becomes a step is, by definition, not missing. A press classified as a
    drag does not become one — so if the screen changed shortly after, something the person
    did is unaccounted for and the hole has to keep being reported. Suppressing it on elapsed
    time alone would have hidden exactly this.
    """

    from android_ui_analyser.device_agent import Touch

    smudged = Touch(x=540, y=700, down_ms=5_000, up_ms=5_300, travel_px=600)
    draft = steps_from_recording(
        [_click(rid="rowOne"), _gap(6_000)],
        touches=[smudged],
        snapshots=[],
        touch_capture=True,
    )

    assert len(draft.gaps) == 1
    assert draft.app_initiated_changes == 0
    assert draft.complete is False
