"""Adding to an ignore list is easy to get right; removing from it is where these things rot.

An agent that silenced a tag last week has to be able to un-silence it this week, including a
tag it never silenced itself — the built-in noise list is a guess about *apps in general*, and
in one app that guess is wrong. So "stop ignoring" has to reach the built-ins too, and asking
to un-ignore something that was never ignored has to say so rather than quietly succeed.
"""

from __future__ import annotations

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config

APP = "com.example.notes"


def _engine() -> tuple[Engine, FakeDevice]:
    device = FakeDevice(package=APP, activity=".Main")
    engine = Engine(
        make_config(memory={"enabled": False}, lease={"enabled": False}),
        device=device,
    )
    return engine, device


def _lines(engine: Engine) -> list[str]:
    engine._app_logs_reported_ms = None  # a second real action would open a fresh window
    digest = engine._app_logs(APP)
    return [line.split(": ", 1)[1] for line in (digest or {}).get("lines", [])]


def test_a_tag_ignored_yesterday_can_be_reported_again_today() -> None:
    engine, device = _engine()
    engine.logcat_mark("last-action")
    device.log_now("ChattyThing", "per-frame noise", priority="D")

    engine.app_log_prefs_set(app=APP, ignore_tags=["ChattyThing"])
    assert _lines(engine) == []

    engine.app_log_prefs_set(app=APP, unignore_tags=["ChattyThing"])
    assert _lines(engine) == ["per-frame noise"]


def test_a_built_in_ignored_tag_can_be_un_ignored_for_one_app() -> None:
    engine, device = _engine()
    engine.logcat_mark("last-action")
    device.log_now("OkHttp", "--> GET /v1/feed", priority="D")
    assert _lines(engine) == [], "the built-in list hides it by default"

    engine.app_log_prefs_set(app=APP, unignore_tags=["OkHttp"])

    assert _lines(engine) == ["--> GET /v1/feed"]


def test_un_ignoring_a_tag_nobody_ignored_says_so_instead_of_claiming_success() -> None:
    engine, _device = _engine()

    result = engine.app_log_prefs_set(app=APP, unignore_tags=["NeverIgnoredTag"])

    assert result["not_ignored"] == ["NeverIgnoredTag"]
    assert result["stored"].get("keep_tags", []) == [], "no exemption is needed for a kept tag"
    assert result["changed"] is False, "nothing changed, so nothing may claim it did"


def test_a_reset_forgets_the_preference_and_goes_back_to_the_defaults() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChattyThing"], limit=40)

    result = engine.app_log_prefs_set(app=APP, reset=True)

    assert result["reset"] is True
    assert result["stored"] == {}
    assert result["effective"]["limit"] == 20
    engine.logcat_mark("last-action")
    device.log_now("ChattyThing", "per-frame noise", priority="D")
    assert _lines(engine) == ["per-frame noise"]


def test_the_effective_view_separates_what_this_app_asked_for_from_the_built_ins() -> None:
    engine, _device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChattyThing"], unignore_tags=["OkHttp"])

    view = engine.app_log_prefs(app=APP)

    assert view["stored"]["ignore_tags"] == ["ChattyThing"]
    assert view["stored"]["keep_tags"] == ["OkHttp"]
    assert "OkHttp" in view["builtin_ignore_tags"]
    assert view["effective"]["ignore_tags"] == ["ChattyThing"]
    assert view["effective"]["keep_tags"] == ["OkHttp"]


def test_an_app_nobody_configured_reports_the_defaults_rather_than_an_empty_answer() -> None:
    engine, _device = _engine()

    view = engine.app_log_prefs(app=APP)

    assert view["stored"] == {}
    assert view["effective"]["limit"] == 20
    assert view["effective"]["levels"] == "DWEF"
    assert view["builtin_ignore_tags"], "silence and 'the defaults apply' must look different"
