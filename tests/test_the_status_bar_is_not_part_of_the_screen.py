"""Android's status bar is not part of the app's screen, so no model is shown it.

A raw hierarchy read returns the status-bar window beside the app's: clock, battery, signal
icons, notification icons -- 22 nodes on a real first frame, each tagged
``window: "system"`` with a ``com.android.systemui`` id. The action path never returns them.
Sent on, they cost tokens on every request and put "Battery 100 percent" beside the app's own
controls in the navigator's view of the screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aua_controller.compaction import compact_frame  # noqa: E402
from experiments.aua_controller.typesafe_navigator import screen_for_model  # noqa: E402


def frame() -> dict[str, object]:
    return {
        "ok": True,
        "observation": {
            "screen": {"package": "com.example.demo", "source": "hierarchy"},
            "meta": {"fingerprint": "f-raw"},
            "elements": [
                {"id": "el:clock", "text": "11:28", "resource_id": "com.android.systemui:id/clock",
                 "window": "system", "bounds": [0, 0, 100, 40]},
                {"id": "el:batt", "desc": "Battery 100 percent.", "resource_id": "com.android.systemui:id/battery",
                 "window": "system", "bounds": [600, 0, 720, 40]},
                {"id": "el:title", "text": "Welcome back", "window": "app", "bounds": [0, 100, 720, 200]},
                {"id": "el:login", "text": "Log in", "clickable": True, "window": "app", "bounds": [0, 200, 720, 300]},
                {"id": "el:dialog", "text": "Allow notifications?", "clickable": True, "window": "system",
                 "resource_id": "com.android.permissioncontroller:id/permission_message",
                 "bounds": [0, 400, 720, 500]},
            ],
        },
    }


def test_status_bar_nodes_are_dropped_by_compaction() -> None:
    labels = [e.get("text") or e.get("desc") for e in compact_frame(frame())["observation"]["elements"]]
    assert labels == ["Welcome back", "Log in", "Allow notifications?"]


def test_a_system_dialog_is_not_the_status_bar() -> None:
    # Permission prompts also live in a system window; they are exactly what a run must see.
    labels = [e.get("text") for e in compact_frame(frame())["observation"]["elements"]]
    assert "Allow notifications?" in labels


def test_the_navigator_never_sees_the_status_bar() -> None:
    shown = screen_for_model(compact_frame(frame(), keep_ids=True))
    assert "systemui" not in str(shown) and "Battery" not in str(shown)
    assert [e.get("text") for e in shown["elements"]][:2] == ["Welcome back", "Log in"]


def test_unnamed_status_bar_icons_are_dropped_too() -> None:
    """Notification icons carry no resource id, only a description, and slipped past the
    package test on a real first frame: two "Android System notification:" images and one blank
    icon, all in the system window, none pressable, none with text."""
    frame_ = frame()
    frame_["observation"]["elements"] += [
        {"id": "el:n1", "desc": "Android System notification:", "window": "system", "clickable": False,
         "bounds": [126, 0, 170, 48]},
        {"id": "el:n2", "window": "system", "clickable": False, "bounds": [648, 11, 664, 37]},
        {"id": "el:toast", "text": "Copied to clipboard", "window": "system", "clickable": False,
         "bounds": [100, 1100, 620, 1160]},
    ]
    labels = [e.get("text") or e.get("desc") for e in compact_frame(frame_)["observation"]["elements"]]
    assert "Android System notification:" not in labels and None not in labels
    assert "Copied to clipboard" in labels, "a system toast has text a run may need to read"
