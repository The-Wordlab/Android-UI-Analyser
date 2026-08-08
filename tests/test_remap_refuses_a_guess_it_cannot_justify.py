"""``remap_ids`` must refuse to re-point an id when it cannot tell which candidate is right.

Found in the field on 2026-08-07 by a downstream UI suite: a numeric element id taken from one
``analyze`` and used in a later ``tap-and-analyze`` resolved to the **system nav bar's Home
button**, twice, silently backgrounding the app under test. ``ok: true`` both times.

The mechanism is here rather than in `Engine._resolve`, which already refuses an unknown id cleanly.
A numeric target is first run through ``remap_ids``, which re-points the old id at whatever it
believes is the same element on the *current* screen. It seeded ``best_iou = -1.0`` and then accepted
the best candidate unconditionally — so **any** IoU beat the seed, including ``0.0``. A candidate
sharing the previous element's ``stable_key`` but with no spatial overlap at all was accepted as
"the same element".

That matters because two of the three key kinds carry little or no position:
``rid:<tail>`` carries none whatsoever, and ``cd:``/``tx:`` carry only a coarse quadrant. (``geo:``
hashes the exact bounds, so geo-matched candidates necessarily overlap perfectly — it is not the
exposed case, and an earlier write-up of this bug wrongly blamed it.)

The nav bar is the worst possible landing spot: it removes the app from the foreground, so the fault
destroys the state the next step meant to observe and then presents as a product crash. A tooling
bug that manufactures product findings is worth a hard refusal.

So: when several candidates share a key and none of them overlaps where the element used to be,
there is no evidence for picking any of them. Refuse, and let `Engine._resolve_id` raise — the
caller's recovery is one ``analyze`` away, whereas a silent mis-tap costs the whole journey.

What is deliberately *not* changed: the single-candidate case still remaps with no overlap test. A
control that legitimately moved — a scrolled list row, a re-laid-out button — has exactly one
same-key candidate and zero overlap, and remapping it is the entire point of the remapper. "A moved
or renamed control is not a failure when its meaning and outcome remain intact."
"""

from __future__ import annotations

from android_ui_analyser.identity import remap_ids, stable_key
from android_ui_analyser.schema import Element


def _el(id_: int, bounds: tuple[int, int, int, int], **kw) -> Element:
    cx, cy = (bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2
    return Element(id=id_, type=kw.pop("type", "android.widget.Button"),
                   bounds=bounds, center=(cx, cy), **kw)


def test_two_same_key_candidates_neither_overlapping_are_refused() -> None:
    """The reported shape: an ambiguous key, no positional evidence, so no answer.

    Both candidates share the previous element's ``rid:`` key — which carries no position at all —
    and neither overlaps where it used to be. Before the fix the first one in list order won purely
    because ``0.0 > -1.0``.
    """
    previous = [_el(1, (40, 200, 400, 280), resource_id="com.x:id/submit")]
    current = [
        _el(7, (40, 900, 400, 980), resource_id="com.x:id/submit"),
        _el(8, (40, 1180, 400, 1260), resource_id="com.x:id/submit"),
    ]

    mapping = remap_ids(previous, current)

    assert 1 not in mapping, (
        "two same-key candidates, neither overlapping the original bounds, is not evidence for "
        f"either one — the remapper must decline rather than pick. got {mapping!r}"
    )


def test_the_overlapping_candidate_still_wins() -> None:
    """The floor must not break the case the disambiguator exists for."""
    previous = [_el(1, (40, 200, 400, 280), resource_id="com.x:id/submit")]
    current = [
        _el(7, (40, 210, 400, 290), resource_id="com.x:id/submit"),  # same control, nudged
        _el(8, (40, 1180, 400, 1260), resource_id="com.x:id/submit"),  # far away
    ]

    assert remap_ids(previous, current).get(1) == 7


def test_a_single_candidate_still_remaps_across_a_move() -> None:
    """Deliberately preserved: one candidate and no overlap is a moved control, not an ambiguity.

    This is the behaviour the suite exists to protect — a scrolled row must still resolve.
    """
    previous = [_el(1, (40, 200, 400, 280), resource_id="com.x:id/row")]
    current = [_el(9, (40, 900, 400, 980), resource_id="com.x:id/row")]

    assert remap_ids(previous, current).get(1) == 9


def test_the_keys_in_this_test_really_do_collide() -> None:
    """Guard the guard: if these stopped sharing a key the tests above would pass vacuously."""
    a = _el(1, (40, 200, 400, 280), resource_id="com.x:id/submit")
    b = _el(8, (40, 1180, 400, 1260), resource_id="com.x:id/submit")
    assert stable_key(a) == stable_key(b), "precondition: same rid tail, so the keys must match"
    assert stable_key(a).startswith("rid:"), "and it must be the position-free key kind"
