"""The dashboard is a client of the warm daemon, so it owes the same trim every client owes.

The daemon deliberately answers with everything — that is what `--observe-fields all` depends
on — and each client applies the default observation view on the way out: the CLI at three call
sites, MCP at its single return boundary. The dashboard applied it nowhere. Every inspection and
every tap therefore shipped the browser each element with all 21 columns plus the full 29-key
`meta` (`research_tasks`, `capture_hint`, `device_locale`, `providers_used`, …), where a CLI
caller reading the same screen sees a handful of columns and the `changed` meta preset. A human
watching both surfaces asked the obvious question: "why are we back to full size elements".

The trim belongs on what is **served**, never on what is **stored**. The stored record is this
panel's resolution table for a click: `_inspection_selector` reads `stable_key` and the exact
`bounds` off it, and every id the overlay published has to stay resolvable there. A display
budget that could make a drawn box unclickable would be a worse bug than the one it fixed, so
these tests pin both halves — the served payload shrinks, and the click path does not notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.projection import OBSERVATION_META_PRESETS

SERIAL = "emulator-5554"

# What the panel is allowed to show per element: the shared default observation columns
# (`config.output.observation_fields`) plus the two the overlay itself is made of.
_ALLOWED_COLUMNS = frozenset(
    {"id", "text", "desc", "clickable", "enabled", "checked", "selected", "bounds", "stable_key"}
)

# Columns the daemon always sends and no dashboard reader ever consumes.
_NOISE_COLUMNS = ("window", "parent", "source", "confidence", "center", "type", "resource_id")

# `meta` keys that are pure provenance or research payload: measured as the bulk of the block.
_NOISE_META = (
    "research_tasks",
    "suggested_deeplinks",
    "capture_hint",
    "device_locale",
    "providers_used",
    "tier_used",
    "via",
    "path",
    "duration_ms",
    "annotated_image",
    "ocr_repaired",
    "slow_controls",
)


@pytest.fixture(autouse=True)
def _device_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin discovery, so the host's `adb` cannot decide whether this test passes."""
    from android_ui_analyser import dashboard as dash

    monkeypatch.setattr(dash, "discover_online_serials", lambda *_a, **_k: ([SERIAL], None))


def _state(tmp_path: Path) -> Any:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    config = Config()
    config.cache.dir = str(tmp_path)
    config.memory.dir = str(tmp_path)
    return dash._DashboardState(
        serials=[SERIAL],
        focus=SERIAL,
        mode="detail",
        cache_dir=tmp_path,
        ensures={},
        poll_ms=500,
        config=config,
    )


def _element(**overrides: Any) -> dict[str, Any]:
    """A node carrying every column the daemon emits, so a trim is visible as absence."""
    base: dict[str, Any] = {
        "id": 0,
        "type": "android.widget.FrameLayout",
        "text": None,
        "resource_id": None,
        "content_desc": None,
        "bounds": [0, 0, 1080, 100],
        "center": [540, 50],
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
        "confidence": 1.0,
        "window": "com.example.fiction",
        "stable_key": None,
        "parent": None,
    }
    base.update(overrides)
    return base


def _frame() -> dict[str, Any]:
    """One screen of a fictional app as the daemon answers it: everything, untrimmed."""
    return {
        "screen": {"width": 1080, "height": 2400, "rotation": 0, "density": None},
        "elements": [
            _element(  # status-bar chrome — not the app, and not clickable
                id=1,
                type="android.widget.ImageView",
                resource_id="com.android.systemui:id/battery",
                content_desc="Battery 87%",
                bounds=[960, 0, 1040, 84],
                window="com.android.systemui",
                stable_key="rid:battery",
            ),
            _element(  # soft-keyboard key
                id=2,
                type="android.inputmethodservice.Keyboard",
                text="Q",
                bounds=[0, 1800, 96, 1900],
                window="ime",
                stable_key="text:Q",
            ),
            _element(  # pure layout container around id 4
                id=3,
                resource_id="com.example.fiction:id/greetingContainer",
                bounds=[0, 200, 1080, 700],
                stable_key="rid:greetingContainer",
            ),
            _element(
                id=4,
                type="android.widget.TextView",
                text="Fictional greeting panel",
                resource_id="com.example.fiction:id/greetingPanel",
                bounds=[32, 296, 1048, 465],
                center=[540, 380],
                clickable=True,
                stable_key="rid:greetingPanel",
            ),
            _element(  # an unnamed switch: no label, but the row a caller acts on next
                id=5,
                type="android.widget.Switch",
                bounds=[820, 900, 1000, 980],
                center=[910, 940],
                clickable=True,
                checkable=True,
                checked=False,
                stable_key="rid:fictionToggle",
            ),
        ],
        "meta": {key: f"value-for-{key}" for key in _all_meta_keys()},
    }


def _all_meta_keys() -> tuple[str, ...]:
    from android_ui_analyser.schema import Meta

    return tuple(Meta.model_fields)


def _analyze(state: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    def fake_call(_serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        assert cmd == "analyze"
        Path(args["with_image"]).write_bytes(b"frame")
        return _frame()

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    return state.inspection_operation("analyze", {"serial": SERIAL})


def _tap(state: Any, monkeypatch: pytest.MonkeyPatch, element_id: Any) -> dict[str, Any]:
    sent: list[dict[str, Any]] = []

    def fake_call(_serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        sent.append({"cmd": cmd, **args})
        Path(args["with_image"]).write_bytes(b"post-action")
        return {
            "ok": True,
            "action": "tap",
            "observation": _frame(),
            "next_actions": [
                {"id": 1, "label": "Battery 87%"},
                {"id": 4, "label": "Fictional greeting panel"},
            ],
        }

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    served = state.inspection_operation(
        "tap",
        {
            "serial": SERIAL,
            "inspection_id": state._inspections[SERIAL]["inspection_id"],
            "element_id": element_id,
        },
    )
    served["_sent"] = sent
    return served


def test_an_inspection_serves_the_same_observation_view_every_client_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Projected columns and the `changed` meta preset — not the daemon's whole answer."""
    state = _state(tmp_path)
    served = _analyze(state, monkeypatch)
    shown = served["result"]

    for element in shown["elements"]:
        extra = set(element) - _ALLOWED_COLUMNS
        assert not extra, f"element {element.get('id')} still carries {sorted(extra)}"
    for column in _NOISE_COLUMNS:
        assert all(column not in e for e in shown["elements"]), f"{column} survived the trim"

    preset = set(OBSERVATION_META_PRESETS["changed"])
    assert set(shown["meta"]) <= preset, sorted(set(shown["meta"]) - preset)
    for key in _NOISE_META:
        assert key not in shown["meta"], f"meta.{key} is provenance, not an affordance"

    ids = [element["id"] for element in shown["elements"]]
    assert ids == [4, 5], "status bar, keyboard and pure wrappers are not app rows"
    switch = next(element for element in shown["elements"] if element["id"] == 5)
    assert switch["checked"] is False, "an off switch must stay readable as off"


def test_the_overlay_can_still_draw_and_name_every_element_it_is_served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`bounds` places the box, `stable_key` names it — the panel is those two columns."""
    state = _state(tmp_path)
    served = _analyze(state, monkeypatch)

    screen = served["view"]["screen"]
    assert screen["width"] and screen["height"], "no screen size, no percentage-positioned box"
    assert served["view"]["elements"], "an overlay with no rows is a blank frame"
    for element in served["view"]["elements"]:
        bounds = element.get("bounds")
        assert isinstance(bounds, list) and len(bounds) == 4, f"element {element['id']} undrawable"
        assert element.get("stable_key"), f"element {element['id']} has no name for its badge"

    # The panel locates an element inside the served `result` JSON by matching its id line and
    # then its `stable_key` text, so the key has to survive in the displayed copy too.
    assert all(element.get("stable_key") for element in served["result"]["elements"])


def test_a_click_on_a_trimmed_frame_still_reaches_the_control_it_drew(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The critical one: trimming the display must not disarm the tap path.

    Green before this change and after — that is the point of it. It fails the moment the
    trim moves onto the stored record, because the default observation columns carry neither
    `stable_key` nor `bounds` and `_inspection_selector` needs both.
    """
    state = _state(tmp_path)
    served = _analyze(state, monkeypatch)

    for drawn in served["view"]["elements"]:
        result = _tap(state, monkeypatch, drawn["id"])
        answered = next(e for e in _frame()["elements"] if e["id"] == drawn["id"])
        assert len(result["_sent"]) == 1, "the click must reach the device exactly once"
        assert result["_sent"][0]["selector"] == {
            "key": answered["stable_key"],
            "bounds": answered["bounds"],
        }, "the stored frame is a click's resolution table; a display view may not thin it"
        served = result


def test_the_stored_frame_keeps_what_the_daemon_answered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Serve a view, store the evidence: every published id stays resolvable afterwards."""
    state = _state(tmp_path)
    _analyze(state, monkeypatch)

    stored = state._inspections[SERIAL]["view"]
    ids = {element["id"] for element in stored["elements"]}
    assert ids == {1, 2, 3, 4, 5}, "a row the display filtered is still a row that was published"
    panel = next(element for element in stored["elements"] if element["id"] == 4)
    assert panel["window"] == "com.example.fiction", "the store is the daemon's answer, untrimmed"
    assert panel["bounds"] == [32, 296, 1048, 465]
    assert panel["stable_key"] == "rid:greetingPanel"


def test_a_tap_serves_a_trimmed_observation_and_a_consistent_next_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The folded post-action observation is the payload the CLI and MCP trim; so does this."""
    state = _state(tmp_path)
    served = _analyze(state, monkeypatch)
    result = _tap(state, monkeypatch, served["view"]["elements"][0]["id"])

    observation = result["result"]["observation"]
    assert [element["id"] for element in observation["elements"]] == [4, 5]
    assert set(observation["meta"]) <= set(OBSERVATION_META_PRESETS["changed"])
    for element in observation["elements"]:
        assert not set(element) - _ALLOWED_COLUMNS

    assert [row["id"] for row in result["result"]["next_actions"]] == [4], (
        "next_actions is derived from the same tree; leaving it whole names ids the "
        "observation no longer contains"
    )
    assert result["view"] is observation, "the overlay draws the payload the panel displays"


def test_the_served_payload_is_materially_smaller_than_the_daemon_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point is a number: the browser stops paying for what it cannot use."""
    state = _state(tmp_path)
    served = _analyze(state, monkeypatch)

    raw = len(json.dumps(state._inspections[SERIAL]["result"]))
    shown = len(json.dumps(served["result"]))
    assert shown * 2 < raw, f"served {shown} bytes of a {raw}-byte answer — barely a trim"
