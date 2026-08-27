"""Goal discovery advertises AUA's complete evidence path before agents improvise."""

from android_ui_analyser.capabilities import capabilities_for_goal, capability_manifest


def test_animation_and_backend_goal_surfaces_session_capture_and_logs() -> None:
    found = {
        item["id"]: item
        for item in capabilities_for_goal(
            "Verify the expand transition animation and that the backend response echoes visibility"
        )
    }

    assert {"session", "animations", "capture", "logcat"} <= found.keys()
    assert "capture sheet" in found["capture"]["cli"]
    assert found["capture"]["mcp"] == "capture_sheet"


def test_emulator_discovery_keeps_goal_work_on_session_bootstrap() -> None:
    emulator = next(item for item in capability_manifest() if item["id"] == "emulator")

    assert emulator["cli"].startswith("aua session start --goal")
    assert emulator["mcp"] == "session_start"
    assert emulator["cleanup"] == "aua session finish"
