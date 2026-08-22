"""What one app wants from its logs is not what the next app wants.

``config.logs`` is one setting for every app on the host: widen it for the app whose own
breadcrumbs are being truncated and you also pay for it on every other app in every other
project. So a stored per-app preference overrides those defaults **for that app id only** —
tags to ignore, tags to stop ignoring, an allow-list of the only tags wanted, the line count,
the per-tag cap and the priority set.

Two invariants survive the new path, because both are what make the digest trustworthy:
a global "off" still turns everything off (cost control belongs to whoever pays), and a
``F`` line is never hidden however narrow the filter (a narrow filter must not hide a crash).
"""

from __future__ import annotations

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.logcat import digest_app_logs
from conftest import FakeDevice, make_config

APP = "com.example.notes"
OTHER_APP = "com.example.other"


def _line(priority: str, tag: str, message: str, *, ms: int = 1) -> str:
    return f"08-21 18:14:44.{ms:03d}  5928  6079 {priority} {tag}: {message}"


def _engine(**cfg: object) -> tuple[Engine, FakeDevice]:
    device = FakeDevice(package=APP, activity=".Main")
    engine = Engine(
        make_config(memory={"enabled": False}, lease={"enabled": False}, **cfg),
        device=device,
    )
    return engine, device


def _digest(engine: Engine, app_id: str = APP) -> dict[str, object] | None:
    # `_app_logs` reports one window once, so a wait cannot re-report the previous action's
    # lines. A test asking the same window twice — before and after a preference change — has
    # to clear that dedupe, which is exactly what a second real action would do.
    engine._app_logs_reported_ms = None
    return engine._app_logs(app_id)


# ------------------------------------------------------------------ the filter itself


def test_an_only_list_keeps_just_the_tags_that_were_asked_for() -> None:
    raw = "\n".join(
        [
            _line("D", "Checkout", "wanted"),
            _line("D", "SomethingElse", "not wanted"),
        ]
    )

    digest = digest_app_logs(raw, app_id=APP, allow_tag_prefixes=("Checkout",))

    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["wanted"]
    assert digest["only"] == ["Checkout"], "a narrowed window must say it was narrowed"


def test_a_fatal_line_survives_an_only_list_that_does_not_name_it() -> None:
    raw = "\n".join(
        [
            _line("D", "Checkout", "wanted"),
            _line("F", "SomethingElse", "the process is going down"),
        ]
    )

    digest = digest_app_logs(raw, app_id=APP, allow_tag_prefixes=("Checkout",))

    assert "the process is going down" in "\n".join(digest["lines"])


def test_an_only_list_can_name_a_tag_the_built_in_deny_list_hides() -> None:
    # Asking for a library by name is the one case where the built-in noise list is wrong.
    raw = _line("D", "OkHttp", "--> GET /v1/feed")

    digest = digest_app_logs(raw, app_id=APP, allow_tag_prefixes=("OkHttp",))

    assert digest["count"] == 1


def test_a_kept_tag_overrules_the_deny_list_without_narrowing_anything_else() -> None:
    raw = "\n".join([_line("D", "OkHttp", "--> GET /v1/feed"), _line("D", "MyOwnTag", "kept")])

    digest = digest_app_logs(raw, app_id=APP, keep_tag_prefixes=("OkHttp",))

    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["--> GET /v1/feed", "kept"]
    assert "only" not in digest, "nothing was narrowed, so nothing should claim it was"


def test_an_unnarrowed_digest_says_nothing_about_an_allow_list() -> None:
    digest = digest_app_logs(_line("D", "MyOwnTag", "kept"), app_id=APP)

    assert "only" not in digest


# ------------------------------------------------------------------ resolution against config


def test_a_stored_ignore_tag_stops_reaching_the_observation() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChattyThing"])
    engine.logcat_mark("last-action")
    device.log_now("ChattyThing", "per-frame noise", priority="D")
    device.log_now("MyOwnTag", "the answer", priority="D")

    digest = _digest(engine)

    assert digest is not None
    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["the answer"]


def test_a_stored_line_count_returns_more_than_the_default_twenty() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, limit=40, per_tag=50)
    engine.logcat_mark("last-action")
    for index in range(45):
        device.log_now("MyOwnTag", f"line {index}", priority="D")

    digest = _digest(engine)

    assert digest is not None
    assert digest["count"] == 40, "the agent asked for 40 lines and must get 40"


def test_a_stored_priority_set_widens_only_that_app_s_window() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, levels="DIWEF")
    engine.logcat_mark("last-action")
    device.log_now("MyOwnTag", "an info breadcrumb", priority="I")

    digest = _digest(engine)

    assert digest is not None and digest["count"] == 1
    assert _digest(engine, OTHER_APP) is None, "the default set still drops I for every other app"


def test_a_preference_applies_to_its_own_app_and_not_the_next_one() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=OTHER_APP, ignore_tags=["MyOwnTag"])
    engine.logcat_mark("last-action")
    device.log_now("MyOwnTag", "the answer", priority="D")

    digest = _digest(engine)

    assert digest is not None and digest["count"] == 1


def test_a_stored_only_list_narrows_the_folded_observation() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, only_tags=["Checkout"])
    engine.logcat_mark("last-action")
    device.log_now("MyOwnTag", "not asked for", priority="D")
    device.log_now("Checkout", "refused: quota exceeded", priority="E")

    digest = _digest(engine)

    assert digest is not None
    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["refused: quota exceeded"]


def test_the_global_switch_still_turns_every_app_off() -> None:
    engine, device = _engine(logs={"enabled": False})
    engine.app_log_prefs_set(app=APP, limit=40)
    engine.logcat_mark("last-action")
    device.log_now("MyOwnTag", "the answer", priority="D")

    assert _digest(engine) is None, "whoever turned it off pays the bill and keeps the decision"


def test_one_app_can_be_silenced_while_the_rest_stay_on() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, enabled=False)
    engine.logcat_mark("last-action")
    device.log_now("MyOwnTag", "the answer", priority="D")

    assert _digest(engine) is None
    assert _digest(engine, OTHER_APP) is not None


def test_an_explicit_preference_applies_even_though_learning_is_off() -> None:
    # Every engine in this file runs with `memory.enabled: false`, which means "record nothing
    # you discover" — not "discard what I explicitly told you". This test says so out loud.
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChattyThing"])
    engine.logcat_mark("last-action")
    device.log_now("ChattyThing", "per-frame noise", priority="D")

    assert _digest(engine) is None


def test_a_stored_preference_is_read_back_with_the_defaults_it_did_not_override() -> None:
    engine, _device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChattyThing"], limit=40)

    view = engine.app_log_prefs(app=APP)

    assert view["package"] == APP
    assert view["stored"]["ignore_tags"] == ["ChattyThing"]
    assert view["effective"]["limit"] == 40
    assert view["effective"]["per_tag"] == 5, "an unset field must still report the default"
    assert view["effective"]["levels"] == "DWEF"
    # "nothing ignored" and "the built-ins are ignored" must not look the same.
    assert "OkHttp" in view["builtin_ignore_tags"]


def test_a_contradictory_preference_is_refused_rather_than_guessed() -> None:
    engine, _device = _engine()

    with pytest.raises(Exception) as excinfo:
        engine.app_log_prefs_set(app=APP, ignore_tags=["Checkout"], unignore_tags=["Checkout"])

    assert "Checkout" in str(excinfo.value)
