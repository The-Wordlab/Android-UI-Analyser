"""The folded post-action ``observation`` must be the cheap path, on every surface.

An action returns the new screen so the caller does not need a second `analyze`. That only
pays off if the returned screen is small: measured on one real settings screen the folded
observation was 919 tokens, of which 299 were `meta` keys — 16 of them empty, and three of
them (`research_tasks`, `suggested_deeplinks`, `capture_hint`) belonging to a deliberate
`analyze`, not to every tap. An agent that finds the automatic observation expensive routes
around it, which is the failure this file exists to prevent.

Two rules, and they pull against each other:

* Omit a key whose value carries nothing — a flag at its default, a null, an empty list.
* Never omit a flag whose *off* state is the payload. A checkable node's ``checked: false``
  is the whole reading of a switch, so it survives the trim even though ``False`` looks
  droppable. :func:`schema.drop_default_flags` owns that rule and this file pins it.

``analyze`` is deliberately untouched: a caller who wants everything asks for it, either
with a plain `analyze` or with ``observe_fields=all`` / ``observe_meta=all``.
"""

from __future__ import annotations

import json

import pytest

from android_ui_analyser.projection import (
    OBSERVATION_META_PRESETS,
    Projection,
    trim_observation_payload,
)
from android_ui_analyser.schema import (
    FLAG_DEFAULTS,
    AnalyzeResult,
    Element,
    OutputFormat,
    drop_default_flags,
)

# A screen with the three shapes that make this hard: status-bar chrome, a pure wrapper
# layout, and a pair of switches whose whole meaning is `checked`. Fictional app, as this
# repository requires.
PACKAGE = "com.example.fixture"


def _hierarchy(**kw: object) -> dict[str, object]:
    """A hierarchy element with every a11y flag reported, i.e. tri-state resolved to bool."""
    base: dict[str, object] = {
        "text": None,
        "resource_id": None,
        "content_desc": None,
        "clickable": False,
        "enabled": True,
        "focused": False,
        "checkable": False,
        "checked": False,
        "selected": False,
        "scrollable": False,
        "long_clickable": False,
        "password": False,
        "source": "hierarchy",
        "confidence": None,
        "window": "app",
        "parent": None,
    }
    base.update(kw)
    bounds = base["bounds"]
    assert isinstance(bounds, list)
    base.setdefault("center", [(bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2])
    base["center"] = [(bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2]
    return base


def _payload() -> dict[str, object]:
    elements = [
        # A pure wrapper: app resource-id, nothing to read, nothing to act on, wraps others.
        _hierarchy(
            id=0,
            type="LinearLayout",
            resource_id=f"{PACKAGE}:id/action_bar_root",
            bounds=[0, 0, 1080, 2400],
            stable_key="rid:action_bar_root",
        ),
        # Status-bar chrome — a different package, and a labelled one in the top band.
        _hierarchy(
            id=1,
            type="TextView",
            text="6:19",
            content_desc="6:19 PM",
            resource_id="com.android.systemui:id/clock",
            bounds=[11, 2, 126, 60],
            window="system",
            stable_key="rid:clock",
        ),
        _hierarchy(
            id=2,
            type="View",
            content_desc="Battery 100 percent.",
            bounds=[935, 14, 998, 48],
            window="system",
            stable_key="cd:aaaaaaaaaa",
        ),
        # Real content.
        _hierarchy(
            id=3,
            type="View",
            text="Alpha Setting",
            clickable=True,
            bounds=[0, 523, 1080, 673],
            stable_key="tx:1111111111",
        ),
        # The switch that is OFF. No label at all — its `checked: false` is the payload.
        _hierarchy(
            id=4,
            type="View",
            clickable=True,
            checkable=True,
            checked=False,
            bounds=[891, 523, 1038, 673],
            parent=3,
            stable_key="px:View:2222222222222222",
        ),
        _hierarchy(
            id=5,
            type="View",
            text="Beta Setting",
            clickable=True,
            bounds=[0, 673, 1080, 823],
            stable_key="tx:3333333333",
        ),
        # The switch that is ON.
        _hierarchy(
            id=6,
            type="View",
            clickable=True,
            checkable=True,
            checked=True,
            bounds=[891, 673, 1038, 823],
            parent=5,
            stable_key="px:View:4444444444444444",
        ),
        # A vision element: the tri-state flags are genuinely unknown here.
        {
            "id": 7,
            "type": "Text",
            "text": "9",
            "resource_id": None,
            "content_desc": None,
            "bounds": [502, 269, 589, 366],
            "center": [545, 317],
            "clickable": False,
            "enabled": True,
            "focused": False,
            "checkable": None,
            "checked": None,
            "selected": None,
            "scrollable": None,
            "long_clickable": None,
            "password": None,
            "source": "ocr",
            "confidence": 0.5,
            "stable_key": "tx:5555555555",
            "window": None,
            "parent": None,
        },
    ]
    return {
        "schema_version": 1,
        "screen": {
            "width": 1080,
            "height": 2400,
            "package": PACKAGE,
            "activity": None,
            "source": "mixed",
        },
        "elements": elements,
        "meta": {
            "duration_ms": 584,
            "tier_used": "hierarchy",
            "path": "hierarchy",
            "providers_used": ["apple_vision"],
            "known_screen": None,
            "known_routes": [],
            "suggested_gotos": [],
            "suggested_deeplinks": ["open demo://one/aaaa", "open demo://two/bbbb"],
            "research_tasks": ["research research_aaaa: does this screen still exist?"],
            "flows": [],
            "ask": None,
            "map_hint": None,
            "slow_controls": [],
            "capture_hint": "recent pixel change after last action",
            "lossy_text": False,
            "lossy_hint": None,
            "ocr_repaired": 0,
            "annotated_image": None,
            "raw_image": "/tmp/frame.png",
            "device_serial": "emulator-5554",
            "device_locale": "en-US",
            "observation_contract": None,
            "element_diff": None,
            "unchanged": False,
            "stale_risk": None,
            "fingerprint": "a" * 40,
            "via": "hierarchy",
            "caller": None,
            "goal_progress": None,
        },
    }


@pytest.fixture
def payload() -> dict[str, object]:
    # Round-tripping through the model proves the fixture is a payload the tool could emit.
    return AnalyzeResult.model_validate(_payload()).as_dict(OutputFormat.json)


def _observation(payload: dict[str, object], **kw: object) -> dict[str, object]:
    view = Projection.for_observation(
        kw.pop("fields", "id,text,desc,rid,clickable,enabled,checked,selected"),
        meta=kw.pop("meta", "changed"),
    )
    assert view is not None
    return view.apply(payload)


def _key(ordinal: int) -> str:
    """The stable id the payload publishes for the fixture element with this ordinal."""
    from android_ui_analyser.identity import stable_key

    raw = next(e for e in _payload()["elements"] if e["id"] == ordinal)  # type: ignore[index]
    return str(raw.get("stable_key") or stable_key(raw))


def _ids(observation: dict[str, object]) -> list[str]:
    elements = observation["elements"]
    assert isinstance(elements, list)
    return [str(e["id"]) for e in elements]


def _row(observation: dict[str, object], ordinal: int) -> dict[str, object]:
    """The row for a fixture ordinal, looked up by its published stable id."""
    elements = observation["elements"]
    assert isinstance(elements, list)
    return next(e for e in elements if e["id"] == _key(ordinal))


# --------------------------------------------------------------- the flag-default rule


def test_a_flag_at_its_default_is_omitted() -> None:
    """`clickable: false` and `enabled: true` say nothing, so they cost nothing."""
    trimmed = drop_default_flags(
        {"id": 9, "clickable": False, "enabled": True, "focused": False, "selected": False}
    )
    assert trimmed == {"id": 9}


def test_a_flag_that_is_on_survives() -> None:
    trimmed = drop_default_flags({"id": 9, "clickable": True, "enabled": False, "focused": True})
    assert trimmed == {"id": 9, "clickable": True, "enabled": False, "focused": True}


def test_checked_false_survives_on_a_checkable_node() -> None:
    """The off state of a switch is the reading — dropping it makes the control unreadable."""
    trimmed = drop_default_flags({"id": 9, "checkable": True, "checked": False})
    assert trimmed["checked"] is False


def test_checked_false_is_dropped_when_the_node_is_not_checkable() -> None:
    trimmed = drop_default_flags({"id": 9, "checkable": False, "checked": False})
    assert "checked" not in trimmed


def test_nulls_are_omitted() -> None:
    trimmed = drop_default_flags({"id": 9, "text": None, "checkable": None, "scrollable": None})
    assert trimmed == {"id": 9}


def test_the_rule_matches_element_compact() -> None:
    """One rule, two call sites. `Element.compact` and `drop_default_flags` must not drift."""
    elements = _payload()["elements"]
    assert isinstance(elements, list)
    for raw in elements:
        element = Element.model_validate(raw)
        by_model = element.compact()
        by_dict = drop_default_flags(element.model_dump(mode="json"))
        for key in (*FLAG_DEFAULTS, "checked"):
            assert by_model.get(key) == by_dict.get(key), f"{key} drifted on id={element.id}"


# ------------------------------------------------------------------ the observation view


def test_the_observation_drops_the_status_bar(payload: dict[str, object]) -> None:
    assert _ids(_observation(payload)) == [_key(i) for i in (3, 4, 5, 6, 7)]


def test_the_observation_drops_pure_wrapper_layouts(payload: dict[str, object]) -> None:
    assert _key(0) not in _ids(_observation(payload))


def test_the_observation_keeps_both_switches_with_their_state(
    payload: dict[str, object],
) -> None:
    """The regression that matters: an unlabelled switch must keep `checked`, on or off."""
    observation = _observation(payload)
    assert _row(observation, 4)["checked"] is False
    assert _row(observation, 6)["checked"] is True


def test_the_observation_omits_default_flags(payload: dict[str, object]) -> None:
    row = _row(_observation(payload), 3)
    assert row == {"id": _key(3), "text": "Alpha Setting", "clickable": True}


def test_a_vision_row_keeps_its_unknown_flags_absent(payload: dict[str, object]) -> None:
    """Absent on an OCR row means unknown, and `source` is how a reader tells which."""
    row = _row(_observation(payload), 7)
    assert "checked" not in row and "selected" not in row


# ------------------------------------------------------------------------ the meta budget


def test_the_observation_meta_keeps_what_changed(payload: dict[str, object]) -> None:
    meta = _observation(payload)["meta"]
    assert isinstance(meta, dict)
    for key in ("fingerprint", "device_serial", "raw_image"):
        assert key in meta, f"{key} must survive: it is how a caller reads the new screen"


def test_the_meta_preset_registers_the_arrival_state() -> None:
    """A non-settled arrival must survive the observation trim, or the verdict is CLI-only.

    Not asserted on the healthy payload fixture above, deliberately: `arrival_state` is
    None-stripped when an action settles — its absence IS the healthy answer, exactly like
    `stale_risk` and `screen_moved`.
    """
    from android_ui_analyser.projection import OBSERVATION_META_PRESETS

    assert "arrival_state" in OBSERVATION_META_PRESETS["changed"]


def test_the_observation_meta_drops_the_analyze_only_keys(payload: dict[str, object]) -> None:
    meta = _observation(payload)["meta"]
    assert isinstance(meta, dict)
    for key in ("research_tasks", "suggested_deeplinks", "capture_hint", "device_locale"):
        assert key not in meta, f"{key} belongs to a deliberate analyze, not to every action"


def test_the_observation_meta_drops_pure_telemetry(payload: dict[str, object]) -> None:
    """How the read was obtained and how long it took change nothing a caller can do.

    `tier_used` and `via` were also the same word twice on the overwhelming majority of reads
    — both said "hierarchy" — so between them they spent two keys to say nothing once.
    """
    meta = _observation(payload)["meta"]
    assert isinstance(meta, dict)
    for key in ("tier_used", "via", "duration_ms", "path"):
        assert key not in meta, f"{key} is telemetry; an agent cannot act on it"


def test_the_fingerprint_survives_because_something_reads_it(
    payload: dict[str, object],
) -> None:
    """Not decoration: `coaching.emitted_fingerprint` reads this back out of the payload.

    Under the warm daemon the answering engine is a different process, so the payload is the
    only place that value can come from — dropping it silently breaks caller-turn tracking
    rather than just saving bytes.
    """
    meta = _observation(payload)["meta"]
    assert isinstance(meta, dict)
    assert "fingerprint" in meta


def test_the_observation_meta_drops_empty_keys(payload: dict[str, object]) -> None:
    meta = _observation(payload)["meta"]
    assert isinstance(meta, dict)
    assert not [k for k, v in meta.items() if v is None or v == [] or v is False]


def test_meta_all_restores_the_full_meta(payload: dict[str, object]) -> None:
    meta = _observation(payload, meta="all")["meta"]
    assert isinstance(meta, dict)
    assert "research_tasks" in meta and "capture_hint" in meta


def test_meta_accepts_an_explicit_key_list(payload: dict[str, object]) -> None:
    meta = _observation(payload, meta="fingerprint,tier_used")["meta"]
    assert set(meta) == {"fingerprint", "tier_used"}  # type: ignore[arg-type]


def test_the_changed_preset_is_documented_as_a_preset() -> None:
    assert "changed" in OBSERVATION_META_PRESETS
    assert "fingerprint" in OBSERVATION_META_PRESETS["changed"]


# ------------------------------------------------------- fields=all still trims the meta


def test_fields_all_keeps_every_column_but_still_trims_meta(
    payload: dict[str, object],
) -> None:
    """The two dials are independent: asking for all columns must not re-add analyze-only meta."""
    observation = _observation(payload, fields="all")
    meta = observation["meta"]
    assert isinstance(meta, dict)
    assert "research_tasks" not in meta
    assert _row(observation, 3)["bounds"] == [0, 523, 1080, 673]


def test_both_dials_at_all_returns_the_untouched_payload(payload: dict[str, object]) -> None:
    view = Projection.for_observation("all", meta="all")
    assert view is None, "'all'/'all' means 'do not touch', so the full dump is emitted verbatim"


# ------------------------------------------------------------------- the measured budget


def test_the_observation_is_much_cheaper_than_the_full_dump(
    payload: dict[str, object],
) -> None:
    """The whole point. A number, so a future change that regresses cost fails here."""
    full = len(json.dumps(payload, separators=(",", ":")))
    slim = len(json.dumps(_observation(payload), separators=(",", ":")))
    assert slim < full / 2, f"observation {slim}B vs full {full}B — the cheap path is not cheap"


# --------------------------------------------------------- the shared trim on both surfaces


def test_trim_observation_payload_filters_the_derived_lists(
    payload: dict[str, object],
) -> None:
    """A derived list must not name an id the caller was not given."""
    action = {
        "ok": True,
        "action": "tap",
        "observation": payload,
        # `next_actions` names ids the same way the observation does.
        "next_actions": [{"id": _key(1)}, {"id": _key(3)}],
    }
    view = Projection.for_observation("id,text,clickable", meta="changed")
    trimmed = trim_observation_payload(action, view)
    assert [r["id"] for r in trimmed["next_actions"]] == [_key(3)]
