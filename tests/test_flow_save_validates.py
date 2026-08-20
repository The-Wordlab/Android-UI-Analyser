"""`flow save` wrote flows that could never run, and reported success doing it.

Found by a sweep, not by inspection: of the flows captured in one pass, six could not execute.
The capture step is the one that makes the *next* run cheaper, so a dead artefact costs twice —
once when it silently fails, and again because 61 routes then had to be hand-authored.

Two distinct ways a saved flow was dead:

1. **Unloadable.** A recorded `input` carrying no field selector renders as
   `input: {text: ${PARAM_1}}`, which `parse_flow_yaml` rejects with "input needs an `id:` or
   `label:` field selector". So `flow save` wrote a file that `flow run` cannot even read.
2. **Volatile selector.** One step addressed a document-picker row by a content-desc holding the
   file's *size* and a *wall-clock time* ("report.pdf, 1.4 MB, 09:42"). It matched on the visit
   that recorded it and could not match again. Another baked in a uuid.

A note on what is deliberately *not* fatal: a declared param with an empty default is how
`flow save` is designed to work — typed values are never recorded, so every input becomes a
parameter the agent fills in afterwards. Refusing those would remove the feature, so an empty
default warns and an *undeclared* `${...}` — which nothing can ever supply — refuses.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.errors import UsageError
from android_ui_analyser.flows import (
    Flow,
    check_saveable,
    parse_flow_yaml,
    render_flow_yaml,
    steps_from_recent,
)
from android_ui_analyser.memory import RouteStep, is_destructive_step


def test_an_input_without_a_selector_is_refused():
    """The exact shape `flow save` produced: a param'd input with nothing to type into."""
    steps, params = steps_from_recent([RouteStep(kind="input", text="whatever")])
    with pytest.raises(UsageError, match="cannot be loaded back"):
        check_saveable(Flow(name="dead", steps=steps, params=params))


def test_an_input_with_a_selector_is_allowed_through():
    """The same capture is fine once the field is identified - the guard is not a blanket ban."""
    steps, params = steps_from_recent(
        [RouteStep(kind="input", text="whatever", resource_id="com.example:id/query")]
    )
    check_saveable(Flow(name="live", steps=steps, params=params))


def test_an_undeclared_parameter_is_refused():
    """Nothing can supply it, so the flow raises before the device is ever touched."""
    flow = Flow(name="unbound", steps=[RouteStep(kind="tap", label="${PARAM_1}")], params={})
    with pytest.raises(UsageError, match="unbound parameter"):
        check_saveable(flow)


def test_a_declared_but_empty_parameter_only_warns():
    warnings = check_saveable(
        Flow(
            name="fillme", steps=[RouteStep(kind="tap", label="${PARAM_1}")], params={"PARAM_1": ""}
        )
    )
    assert any("PARAM_1" in w for w in warnings), warnings


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("report.pdf, 1.4 MB, 09:42", "a wall-clock time"),  # the picker row, as recorded
        ("report.pdf, 1.4 MB", "a file size"),
        ("Saved 2026-08-04", "a date"),
        ("row 9f8b2c1d-4e5a-4b7c-8d9e-1a2b3c4d5e6f", "a uuid"),
        ("item a3f9c2e8b7d10456", "a backend-looking id"),
    ],
)
def test_a_volatile_selector_warns(label, expected):
    warnings = check_saveable(Flow(name="volatile", steps=[RouteStep(kind="tap", label=label)]))
    assert any(expected in w for w in warnings), f"{label!r} -> {warnings}"


def test_a_stable_selector_does_not_warn():
    """A label with a digit in it is not automatically volatile - false alarms train people out."""
    flow = Flow(
        name="stable",
        steps=[RouteStep(kind="tap", label="Top 4"), RouteStep(kind="tap", label="Match 3")],
    )
    assert check_saveable(flow) == []


def test_a_volatile_selector_is_still_written():
    """Warn, do not refuse: the runner may know the row is stable for its own run."""
    warnings = check_saveable(
        Flow(name="volatile", steps=[RouteStep(kind="tap", label="clip.m4a, 2.1 MB")])
    )
    assert warnings  # returning rather than raising is the assertion


def test_id_selector_keeps_supplemental_destructive_label_after_round_trip() -> None:
    flow = Flow(
        name="guarded",
        steps=[
            RouteStep(
                kind="tap",
                resource_id="genericButton",
                label="Delete",
                by="id",
            )
        ],
    )
    loaded = parse_flow_yaml(render_flow_yaml(flow)).steps[0]
    assert loaded.by == "id"
    assert loaded.resource_id == "genericButton"
    assert loaded.label == "Delete"
    assert is_destructive_step(loaded, ["delete"])


def test_id_selector_keeps_supplemental_destructive_description_after_round_trip() -> None:
    flow = Flow(
        name="guarded_desc",
        steps=[
            RouteStep(
                kind="tap",
                resource_id="genericButton",
                content_desc="Delete account",
                by="id",
            )
        ],
    )
    loaded = parse_flow_yaml(render_flow_yaml(flow)).steps[0]
    assert loaded.by == "id"
    assert loaded.resource_id == "genericButton"
    assert loaded.content_desc == "Delete account"
    assert is_destructive_step(loaded, ["delete"])


@pytest.mark.parametrize(
    ("kind", "default"),
    [
        ("key", "unsupported"),
        ("tap-point", "-1,2"),
        ("swipe", "diagonal"),
        ("scroll", "forward"),
        ("a11y-scroll", "sideways"),
        ("dev-profile", "battery"),
        ("network-profile", "airplane"),
    ],
)
def test_invalid_finite_default_is_refused_recursively(kind: str, default: str) -> None:
    leaf = RouteStep(
        kind=kind,
        arg="${VALUE}",
        resource_id="catalogList" if kind == "a11y-scroll" else None,
    )
    flow = Flow(
        name=f"invalid_{kind}",
        params={"VALUE": default},
        steps=[RouteStep(kind="repeat", repeat=1, substeps=[leaf])],
    )

    with pytest.raises(UsageError):
        check_saveable(flow)


def test_save_validation_materializes_chained_defaults() -> None:
    flow = Flow(
        name="default_chain",
        params={"PROFILE": "${BASE_PROFILE}", "BASE_PROFILE": "wifi_only"},
        steps=[RouteStep(kind="network-profile", arg="${PROFILE}")],
    )

    assert check_saveable(flow) == []


def test_required_empty_finite_parameter_remains_saveable_metadata() -> None:
    flow = Flow(
        name="required_direction",
        params={"DIRECTION": ""},
        steps=[RouteStep(kind="swipe", arg="${DIRECTION}")],
    )

    warnings = check_saveable(flow)

    assert any("DIRECTION" in warning for warning in warnings)


@pytest.mark.parametrize(
    "arrival",
    ["!text:Loading", "unknown:state", "text:", "text:Ready\\"],
)
def test_invalid_or_absence_only_arrival_is_refused_before_save(arrival: str) -> None:
    flow = Flow(
        name="invalid_arrival",
        arrival=arrival,
        steps=[RouteStep(kind="key", arg="back")],
    )

    with pytest.raises(UsageError):
        check_saveable(flow)


def test_combined_positive_and_negative_arrival_is_saveable() -> None:
    flow = Flow(
        name="proved_arrival",
        arrival="rid:cachedContent,!text:Loading",
        steps=[RouteStep(kind="key", arg="back")],
    )

    assert check_saveable(flow) == []
