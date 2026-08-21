"""A dashboard click must address the element it drew a box around.

The overlay's integer ids are frame-local ordinals. Resolving one goes through the numeric
id cache, which is a single file per device shared with every other caller of that device —
so the dashboard was validating a human's click against whichever screen last wrote that
file. On 2026-08-21 that turned a click on the app's own intro card into
``element id 30 is stale for tap: binding 'rid:launcher_widget' changed``: id 30 in the
file was the launcher's At a Glance widget, an element the clicker never saw.

The dashboard made that certain by asking for ``no_cache`` on its own analyze, to avoid
overwriting an agent's ids. There is nothing to protect: the id cache describes the device,
both callers are looking at the same device, and an action remaps by stable identity anyway.
What it actually bought was an id space the dashboard published but could not resolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.errors import UsageError


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


_FRAME = {
    "screen": {"width": 1080, "height": 2400},
    "elements": [
        {
            "id": 30,
            "text": "Fictional greeting panel",
            "resource_id": "com.example.fiction:id/greetingPanel",
            "bounds": [32, 296, 1048, 465],
            "clickable": True,
            "stable_key": "rid:greetingPanel",
        }
    ],
    "meta": {},
}


def _stored(state: Any, tmp_path: Path, frame: dict[str, Any]) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    state._store_inspection("emulator-5554", "source-id", source, frame, frame)


def test_the_dashboard_tap_sends_the_elements_own_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _state(tmp_path)
    _stored(state, tmp_path, _FRAME)
    calls: list[dict[str, Any]] = []

    def fake_call(_serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append({"cmd": cmd, **args})
        Path(args["with_image"]).write_bytes(b"post-action")
        return {"ok": True, "action": "tap", "observation": _FRAME}

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    state.inspection_operation(
        "tap", {"serial": "emulator-5554", "inspection_id": "source-id", "element_id": 30}
    )

    assert len(calls) == 1
    sent = calls[0]
    assert sent["selector"] == {"key": "rid:greetingPanel", "bounds": [32, 296, 1048, 465]}
    assert "element_id" not in sent, (
        "a frame-local ordinal resolves through the shared id cache, which is exactly "
        "the file the dashboard does not own"
    )


def test_a_frame_element_with_no_identity_is_refused_not_sent_as_a_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Falling back to the number is the bug; a clear refusal is recoverable, a mis-tap is not."""
    state = _state(tmp_path)
    keyless = {
        "screen": _FRAME["screen"],
        "elements": [{"id": 30, "bounds": [32, 296, 1048, 465], "clickable": True}],
        "meta": {},
    }
    _stored(state, tmp_path, keyless)
    calls: list[str] = []
    monkeypatch.setattr(
        state, "_inspection_daemon_call", lambda *a, **k: calls.append("sent") or {}
    )

    with pytest.raises(UsageError) as caught:
        state.inspection_operation(
            "tap", {"serial": "emulator-5554", "inspection_id": "source-id", "element_id": 30}
        )

    assert caught.value.code == "element_identity_missing"
    assert calls == [], "nothing may reach the device on an unaddressable element"


def test_the_dashboard_records_the_ids_it_shows_a_human(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from android_ui_analyser import dashboard as dash

    state = _state(tmp_path)
    monkeypatch.setattr(dash, "list_online_serials", lambda *a, **k: ["emulator-5554"])
    calls: list[dict[str, Any]] = []

    def fake_call(_serial: str, cmd: str, **args: Any) -> dict[str, Any]:
        calls.append({"cmd": cmd, **args})
        Path(args["with_image"]).write_bytes(b"frame")
        return _FRAME

    monkeypatch.setattr(state, "_inspection_daemon_call", fake_call)
    state.inspection_operation("analyze", {"serial": "emulator-5554"})

    assert calls[0]["cmd"] == "analyze"
    assert calls[0].get("no_cache") is not True, (
        "an analyze that publishes numbered elements to a human must record those numbers; "
        "hiding them leaves the overlay and `aua tap <id>` describing different screens"
    )
