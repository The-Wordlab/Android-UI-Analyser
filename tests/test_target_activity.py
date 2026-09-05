from __future__ import annotations

from android_ui_analyser import target_activity


def test_target_activity_is_scoped_by_platform(tmp_path) -> None:
    target_activity.touch(tmp_path, "shared", platform="android", at=10)
    target_activity.touch(tmp_path, "shared", platform="ios", at=20)

    assert target_activity.read(tmp_path, "shared", platform="android")["last_activity"] == 10
    assert target_activity.read(tmp_path, "shared", platform="ios")["last_activity"] == 20
    assert target_activity.activity_path(
        tmp_path, "shared", platform="android"
    ) != target_activity.activity_path(tmp_path, "shared", platform="ios")
