"""Whatever a surface prints as ``id``, that same surface must accept back.

Ids are published as stable ids (``rid:greetingPanel``), and every path that hands one to a
human or an agent has to take it back unchanged. Two did not, and both failed in the worst
way available — not with "unknown id" but by turning the id into something that could never
match anything:

* the dashboard's browser code sent ``Number(elementId)``, which is ``NaN`` for a string id,
  and ``NaN`` serialises to ``null`` — so a click arrived as a missing field and was reported
  as "needs a non-negative AUA element id" about an id the page had just drawn;
* the server behind it required ``isinstance(element_id, int)``, so even a correctly sent
  string id would have been refused.

A round trip is the only thing worth asserting here: publish, read one id out of what was
published, send exactly that back, and require it to resolve to the element it named.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.schema import (
    AnalyzeResult,
    Element,
    Meta,
    OutputFormat,
    Screen,
)

PACKAGE = "com.example.fiction"
DASHBOARD_JS = Path(__file__).resolve().parent.parent / "src/android_ui_analyser/dashboard.py"


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        schema_version=1,
        screen=Screen(width=1080, height=2400, package=PACKAGE, source="hierarchy"),
        elements=[
            Element(
                id=30,
                type="Button",
                text="Fictional greeting panel",
                resource_id=f"{PACKAGE}:id/greetingPanel",
                bounds=[32, 296, 1048, 465],
                center=[540, 380],
                clickable=True,
                enabled=True,
                window="app",
            )
        ],
        meta=Meta(duration_ms=10, tier_used="hierarchy", path="hierarchy"),
    )


def _published_frame() -> dict[str, Any]:
    """What the dashboard actually stores: the payload as a caller receives it."""
    return _observation().as_dict(OutputFormat.json)


@pytest.fixture(autouse=True)
def _device_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep discovery off the host: a real `adb` here would decide these tests."""
    from android_ui_analyser import dashboard as dash

    monkeypatch.setattr(
        dash, "discover_online_serials", lambda *_a, **_k: (["emulator-5554"], None)
    )


def _state(tmp_path: Path) -> Any:
    from android_ui_analyser import dashboard as dash
    from android_ui_analyser.config import Config

    config = Config()
    config.cache.dir = str(tmp_path)
    config.memory.dir = str(tmp_path)
    return dash._DashboardState(
        serials=["emulator-5554"],
        focus="emulator-5554",
        mode="detail",
        cache_dir=tmp_path,
        ensures={},
        poll_ms=500,
        config=config,
    )


# ------------------------------------------------------------------ what gets published


def test_the_published_frame_names_elements_by_stable_id() -> None:
    """The premise of everything below: the id in a payload is not a number."""
    frame = _published_frame()

    assert [e["id"] for e in frame["elements"]] == ["rid:greetingPanel"]


# ---------------------------------------------------------------- the surfaces agents use


def test_mcp_publishes_stable_ids() -> None:
    """MCP is the surface agents actually drive, and it dumps the model directly.

    `AnalyzeResult.as_dict` was the only place that rewrote ids, so every caller reaching for
    the more obvious `model_dump(mode="json")` — MCP, and the CLI's projection path — kept
    handing out frame ordinals. Publishing has to happen where the payload leaves, not only on
    the one method that happened to be written first.
    """
    from android_ui_analyser.mcp_server import _dump
    from android_ui_analyser.schema import ActionResult

    observation = _observation()
    payload = _dump(ActionResult(ok=True, action="tap", id=30, observation=observation))

    assert [e["id"] for e in payload["observation"]["elements"]] == ["rid:greetingPanel"]


def test_the_projected_cli_view_publishes_stable_ids() -> None:
    """`--fields`/`--format tsv` read the untrimmed dump, so they bypassed `as_dict` too."""
    from android_ui_analyser import cli
    from android_ui_analyser.projection import Projection

    payload = cli._analyze_payload(_observation())
    assert payload is not None
    view = Projection.parse(fmt=OutputFormat.json, fields="id,text,clickable")

    rows = view.apply(payload, fmt=OutputFormat.json)["elements"]
    assert [r["id"] for r in rows] == ["rid:greetingPanel"]


def test_the_internal_dump_still_carries_ordinals() -> None:
    """`model_dump` is the internal form and must stay one.

    The analyze cache is written and read through it, and a numeric action resolves against
    that file — publishing there turned the cache into a store the resolver could not read,
    which broke 127 tests at once. Publishing belongs at the boundary, not in the model.
    """
    raw = _observation().model_dump(mode="json")

    assert [e["id"] for e in raw["elements"]] == [30]


# ------------------------------------------------------------------------- the dashboard


def test_a_dashboard_click_resolves_the_id_the_overlay_drew(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The round trip: take an id out of the stored frame, send it back, expect a tap."""
    state = _state(tmp_path)
    frame = _published_frame()
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    state._store_inspection("emulator-5554", "source-id", source, frame, frame)

    published_id = frame["elements"][0]["id"]
    calls: list[dict[str, Any]] = []

    def fake_call(_serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append({"cmd": cmd, **args})
        Path(args["with_image"]).write_bytes(b"post-action")
        return {"ok": True, "action": "tap", "observation": frame}

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    state.inspection_operation(
        "tap",
        {
            "serial": "emulator-5554",
            "inspection_id": "source-id",
            "element_id": published_id,
        },
    )

    assert len(calls) == 1, "the click was refused, so no tap reached the device"
    assert calls[0]["selector"] == {
        "key": "rid:greetingPanel",
        "bounds": [32, 296, 1048, 465],
    }


def test_a_dashboard_click_on_an_absent_id_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accepting string ids must not turn a genuinely wrong id into a silent tap."""
    from android_ui_analyser.errors import UsageError

    state = _state(tmp_path)
    frame = _published_frame()
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    state._store_inspection("emulator-5554", "source-id", source, frame, frame)
    monkeypatch.setattr(
        state,
        "_inspection_daemon_call",
        lambda *_a, **_k: pytest.fail("no tap may be sent for an id that is not in the frame"),
    )

    with pytest.raises(UsageError):
        state.inspection_operation(
            "tap",
            {
                "serial": "emulator-5554",
                "inspection_id": "source-id",
                "element_id": "rid:notOnThisScreen",
            },
        )


def test_the_browser_never_coerces_a_published_id_to_a_number() -> None:
    """`Number("rid:x")` is NaN, and NaN serialises to null — a click with no id at all.

    Asserted against the source because the failure is in browser code no Python test
    exercises: by the time it is visible, it is visible to a person clicking.
    """
    js = DASHBOARD_JS.read_text(encoding="utf-8")

    offenders = re.findall(r"Number\(\s*element(?:_id|Id)?\.?\w*\s*\)", js)
    assert not offenders, f"a published id must not be coerced to a number: {offenders}"
