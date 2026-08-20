"""Two answers that must never be quietly wrong: "your element is gone" and "your app is fine".

An empty ``elements`` list and an ``ok: true`` action both read as good news. When a filter
shape silently matches nothing, or the app under test has crashed out of the foreground, that
good news is false — and a caller acts on it instead of recovering.
"""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from android_ui_analyser.projection import Projection
from android_ui_analyser.schema import (
    ActionResult,
    AnalyzeResult,
    Element,
    Meta,
    PathKind,
    Screen,
    ScreenSource,
    Tier,
)
from conftest import FakeDevice, make_config


def _element(rid: str | None = None) -> Element:
    return Element(id=0, type="View", bounds=[0, 0, 10, 10], center=[5, 5], resource_id=rid)


def _obs(*rids: str | None) -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, source=ScreenSource.hierarchy),
        elements=[_element(r) for r in rids],
        meta=Meta(duration_ms=1, tier_used=Tier.hierarchy, path=PathKind.hierarchy),
    )


# --------------------------------------------------------------- --where-rid OR filtering


def test_where_rid_accepts_a_comma_separated_or_list() -> None:
    # `--where-rid a,b,c` used to be ONE substring that matched nothing, and the caller got a
    # quiet empty list that is indistinguishable from "verified absent".
    proj = Projection.parse(where_rid=["navTabHome,navTabBrowse,navTabSettings"])
    assert proj.where_rid == ("navtabhome", "navtabbrowse", "navtabsettings")


def test_where_rid_tolerates_spaces_and_empty_parts() -> None:
    proj = Projection.parse(where_rid=["alpha, beta ,,"])
    assert proj.where_rid == ("alpha", "beta")


def test_where_rid_still_accepts_the_repeated_flag_form() -> None:
    proj = Projection.parse(where_rid=["alpha", "beta"])
    assert proj.where_rid == ("alpha", "beta")


def test_where_text_is_not_split_because_real_copy_contains_commas() -> None:
    # Splitting this would silently WIDEN the filter: "Hello, world" would start matching any
    # screen containing just "world".
    proj = Projection.parse(where_text=["Hello, world"])
    assert proj.where_text == ("hello, world",)


# --------------------------------------------------------------- app left the foreground


def test_crash_to_launcher_is_reported() -> None:
    left = Engine._app_left_foreground(
        "com.example.app/.MainActivity",
        "com.google.android.apps.nexuslauncher/.NexusLauncherActivity",
        _obs("launcher"),
    )
    assert left == {
        "from": "com.example.app",
        "to": "com.google.android.apps.nexuslauncher",
        "crash_dialog": False,
    }


def test_the_system_crash_dialog_is_reported_even_off_a_launcher() -> None:
    left = Engine._app_left_foreground(
        "com.example.app/.MainActivity",
        "android/com.android.server.am.AppErrorDialog",
        _obs("android:id/aerr_close"),
    )
    assert left is not None
    assert left["crash_dialog"] is True


def test_an_ordinary_app_to_app_handoff_is_not_reported() -> None:
    # A share sheet or a browser is a legitimate destination; flagging it would cry wolf on
    # every deliberate hand-off.
    assert (
        Engine._app_left_foreground(
            "com.example.app/.MainActivity",
            "com.example.other/.ShareActivity",
            _obs("share_list"),
        )
        is None
    )


def test_staying_in_the_same_package_is_not_reported() -> None:
    assert (
        Engine._app_left_foreground(
            "com.example.app/.MainActivity",
            "com.example.app/.DetailActivity",
            _obs("detail"),
        )
        is None
    )


def test_unknown_activities_are_not_guessed_at() -> None:
    assert Engine._app_left_foreground(None, "com.x/.A", _obs()) is None
    assert Engine._app_left_foreground("com.x/.A", None, _obs()) is None
    # No "/" means we cannot name a package, so we must not invent one.
    assert Engine._app_left_foreground("com.x", "nexuslauncher", _obs()) is None


def test_detected_crash_attaches_the_last_action_error_logs(monkeypatch) -> None:
    app_id = "com.example.app"
    launcher = "com.example.launcher"
    device = FakeDevice(package=launcher, activity=".LauncherActivity")
    engine = Engine(
        make_config(memory={"enabled": False}, lease={"enabled": False}),
        device=device,
    )
    engine._pre_action_state = {
        "count": 1,
        "focused": None,
        "labels": ["Crash now"],
        "rids": ["crashNow"],
        "package": app_id,
        "activity": f"{app_id}/.MainActivity",
        "known_screen": None,
    }
    device.log_now(
        "AndroidRuntime",
        "java.lang.IllegalStateException: stale failure from the previous action",
        offset_ms=-1_000,
    )
    engine.logcat_mark("last-action")
    device.log_now("AndroidRuntime", "FATAL EXCEPTION: main")
    device.log_now("AndroidRuntime", f"Process: {app_id}, PID: 1234")
    device.log_now("AndroidRuntime", "java.lang.IllegalStateException: broken state")
    device.log_now(
        "AndroidRuntime",
        f"at {app_id}.MainActivity.onClick(MainActivity.kt:42)",
    )
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_args, **_kwargs: _obs("launcher"),
    )

    result = engine._observe(ActionResult(ok=True, action="tap"), True, settle=False)

    assert result.change is not None and result.change["app_left_foreground"]["from"] == app_id
    assert result.crash_evidence is not None
    assert result.crash_evidence["available"] is True
    assert result.crash_evidence["since"] == "last-action"
    assert result.crash_evidence["kind"] == "fatal"
    evidence_lines = "\n".join(result.crash_evidence["lines"])
    assert "IllegalStateException: broken state" in evidence_lines
    assert "stale failure" not in evidence_lines
    assert "attached in `crash_evidence`" in (result.note or "")
    assert "Check `aua logcat" not in (result.note or "")
