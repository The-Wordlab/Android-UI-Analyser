"""A preference an agent cannot trust is worse than no preference at all.

Every test here is a lie the first draft told: it accepted an instruction, reported success, and
then filtered the window by something else. The class matters more than the individual bugs —
this preference is *persisted*, so a wrong answer is not one confusing call, it is every action
on that app in every later session, with no obvious way to notice.

So: tag entries are prefixes and are compared as prefixes; a filter can never hide `F`; the
narrow instruction beats the broad one and the immediate instruction beats the remembered one;
and anything that cannot take effect yet says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from android_ui_analyser.cli import app as cli_app
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import AuaError
from android_ui_analyser.logcat import digest_app_logs
from android_ui_analyser.mcp_server import _dispatch as mcp_dispatch
from android_ui_analyser.memory import AppLogPrefs, AppMemoryStore
from android_ui_analyser.platforms.android import AndroidPlatform
from conftest import FakeDevice, make_config

APP = "com.example.notes"
runner = CliRunner()


def _engine(**cfg: Any) -> tuple[Engine, FakeDevice]:
    device = FakeDevice(package=APP, activity=".Main")
    sections: dict[str, Any] = {"memory": {"enabled": False}, "lease": {"enabled": False}}
    for section, values in cfg.items():
        sections[section] = {**sections.get(section, {}), **values}
    engine = Engine(make_config(**sections), device=device)
    return engine, device


def _lines(engine: Engine, app_id: str = APP) -> list[str]:
    engine._app_logs_reported_ms = None  # a second real action would open a fresh window
    digest = engine._app_logs(app_id)
    return [line.split(": ", 1)[1] for line in (digest or {}).get("lines", [])]


def _line(priority: str, tag: str, message: str, *, ms: int = 1) -> str:
    return f"08-21 18:14:44.{ms:03d}  5928  6079 {priority} {tag}: {message}"


# ------------------------------------------------------------------ tags are prefixes


def test_un_ignoring_a_tag_in_a_different_case_actually_un_ignores_it() -> None:
    engine, device = _engine()
    engine.logcat_mark("last-action")
    device.log_now("ChatSync", "syncing", priority="D")
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChatSync"])
    assert _lines(engine) == []

    result = engine.app_log_prefs_set(app=APP, unignore_tags=["chatsync"])

    assert result["not_ignored"] == [], "the filter is case-insensitive, so the answer must be too"
    assert result["changed"] is True
    assert _lines(engine) == ["syncing"]


def test_un_ignoring_a_tag_a_stored_prefix_hides_clears_that_prefix() -> None:
    engine, device = _engine()
    engine.logcat_mark("last-action")
    device.log_now("NetworkError", "timeout talking to /v1/sync", priority="E")
    engine.app_log_prefs_set(app=APP, ignore_tags=["Network"])
    assert _lines(engine) == []

    result = engine.app_log_prefs_set(app=APP, unignore_tags=["NetworkError"])

    assert result["not_ignored"] == []
    assert _lines(engine) == ["timeout talking to /v1/sync"]


def test_a_broader_ignore_absorbs_the_narrower_ones_it_already_covers() -> None:
    engine, _device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChatSyncWorker", "ChatSyncQueue"])

    result = engine.app_log_prefs_set(app=APP, ignore_tags=["ChatSync"])

    assert result["stored"]["ignore_tags"] == ["ChatSync"], "three prefixes for one tag family"


def test_an_ignore_a_stored_prefix_already_covers_is_not_stored_twice() -> None:
    engine, _device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["ChatSync"])

    result = engine.app_log_prefs_set(app=APP, ignore_tags=["ChatSyncWorker"])

    assert result["stored"]["ignore_tags"] == ["ChatSync"]
    assert result["changed"] is False


def test_un_ignoring_a_runtime_tag_records_the_exemption_it_needs() -> None:
    # ART logs GC and JIT under the app's own process name truncated to the tag field, and the
    # digest drops that by deriving it. Un-ignoring it has to beat that rule too, or the answer
    # is "it was not being ignored" about a tag that stays invisible.
    engine, device = _engine()
    runtime_tag = APP[-12:]
    engine.logcat_mark("last-action")
    device.log_now(runtime_tag, "Background concurrent copying GC freed", priority="D")
    assert _lines(engine) == []

    result = engine.app_log_prefs_set(app=APP, unignore_tags=[runtime_tag])

    assert result["not_ignored"] == []
    assert result["stored"]["keep_tags"] == [runtime_tag]
    assert _lines(engine) == ["Background concurrent copying GC freed"]


# ------------------------------------------------------------------ narrow beats broad


def test_an_ignored_tag_stays_ignored_even_when_an_only_list_names_it_too() -> None:
    # "Show me only Payment*" and "never show me PaymentDebug" are both instructions, and the
    # specific one wins — otherwise the banned tag comes back and spends the whole budget.
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, only_tags=["Payment"], ignore_tags=["PaymentDebug"])
    engine.logcat_mark("last-action")
    for index in range(6):
        device.log_now("PaymentDebug", f"chatty {index}", priority="D")
    device.log_now("Payment", "declined", priority="E")

    assert _lines(engine) == ["declined"]


def test_an_exemption_that_an_only_list_shadows_is_reported_as_shadowed() -> None:
    engine, _device = _engine()
    engine.app_log_prefs_set(app=APP, only_tags=["Payment"])

    result = engine.app_log_prefs_set(app=APP, unignore_tags=["OkHttp"])

    assert result["changed"] is True, "it is stored, and applies once the only-list is cleared"
    assert result["shadowed_by_only_tags"] == ["OkHttp"], "but it does nothing right now"


def test_narrowing_to_one_tag_does_not_then_cap_that_tag_at_five_lines() -> None:
    # The per-tag cap exists so one logger cannot spend a budget nobody spent on purpose. An
    # only-list IS that purpose, and answering it with 5 lines makes `limit` a lie.
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, only_tags=["Payment"], limit=30)
    engine.logcat_mark("last-action")
    for index in range(12):
        device.log_now("Payment", f"step {index}", priority="D")

    assert len(_lines(engine)) == 12


# ------------------------------------------------------------------ F is never droppable


def test_no_tag_filter_can_hide_a_fatal_line() -> None:
    raw = "\n".join(
        [
            _line("F", "PaymentDebug", "FATAL EXCEPTION: main"),
            _line("D", "PaymentDebug", "ordinary noise"),
        ]
    )

    digest = digest_app_logs(raw, app_id=APP, drop_tag_prefixes=("PaymentDebug",))

    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["FATAL EXCEPTION: main"]


def test_a_stored_ignore_cannot_hide_a_fatal_line_either() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, ignore_tags=["PaymentDebug"])
    engine.logcat_mark("last-action")
    device.log_now("PaymentDebug", "FATAL EXCEPTION: main", priority="F")

    assert _lines(engine) == ["FATAL EXCEPTION: main"]


def test_a_blank_tag_entry_does_not_delete_the_whole_window() -> None:
    # `"anything".startswith("")` is true, so one empty string in a config list or a hand-edited
    # preference would silently filter out every line the app wrote.
    raw = _line("D", "MyOwnTag", "the answer")

    assert digest_app_logs(raw, app_id=APP, deny_tag_prefixes=("",))["count"] == 1
    assert digest_app_logs(raw, app_id=APP, drop_tag_prefixes=(" ",))["count"] == 1
    assert digest_app_logs(raw, app_id=APP, allow_tag_prefixes=("",))["count"] == 1


# ------------------------------------------------------------------ immediate beats remembered


def test_a_session_configure_beats_a_stored_preference() -> None:
    engine, device = _engine()
    engine.app_log_prefs_set(app=APP, levels="WEF", limit=20)
    mcp_dispatch(
        engine,
        "configure",
        {"app_log_levels": "DIWEF", "app_log_limit": 40, "app_log_per_tag": 40},
    )
    engine.logcat_mark("last-action")
    for index in range(25):
        device.log_now("MyOwnTag", f"info {index}", priority="I")

    lines = _lines(engine)

    assert len(lines) == 25, "the agent asked for this session, so this session wins"


def test_a_typed_app_logs_flag_beats_a_stored_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[set[str]] = []
    monkeypatch.setattr(
        AndroidPlatform,
        "connect",
        lambda self, target_id=None: FakeDevice(package=APP),
    )
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")
    monkeypatch.setattr(
        Engine,
        "app_log_prefs",
        lambda self, **kw: seen.append(set(self._session_log_fields)) or {"ok": True},
    )

    result = runner.invoke(
        cli_app, ["--no-lease", "--app-logs", "DIWEF", "logcat", "prefs", "show", "--app", APP]
    )

    assert result.exit_code == 0, result.stderr
    assert seen == [{"levels"}], "a flag typed on this invocation must outrank what is stored"


# ------------------------------------------------------------------ bounded, and device-free


@pytest.mark.parametrize("value", [0, -1, 5_000])
def test_a_line_count_outside_the_bounds_is_refused(value: int) -> None:
    engine, _device = _engine()

    with pytest.raises(AuaError):
        engine.app_log_prefs_set(app=APP, limit=value)


def test_a_hand_edited_preference_cannot_disable_the_per_tag_cap(tmp_path: Path) -> None:
    cfg = make_config(memory={"dir": str(tmp_path / "home")})
    store = AppMemoryStore(cfg.memory)
    store.save_log_prefs(AppLogPrefs(package=APP))
    store.log_prefs_path(APP).write_text(
        json.dumps({"package": APP, "per_tag": 0, "limit": 99_999, "levels": "zz"}),
        encoding="utf-8",
    )

    again = store.load_log_prefs(APP)

    assert again is not None
    assert again.per_tag == 1, "a zero cap is no cap at all"
    assert again.limit == 500
    assert again.levels is None, "a typo must fall back to the default, not silence the window"


def test_reading_a_preference_neither_claims_a_memory_session_nor_opens_sqlite(
    tmp_path: Path,
) -> None:
    # Two side effects that would otherwise land on the per-action hot path: `self._mem` is what
    # several call sites read as "memory is on", and building the sqlite store creates the
    # database and runs its legacy migration.
    database = tmp_path / "memory.db"
    engine, device = _engine(
        memory={"enabled": False, "backend": "sqlite", "sqlite_path": str(database)}
    )
    engine.logcat_mark("last-action")
    device.log_now("MyOwnTag", "the answer", priority="D")

    assert _lines(engine) == ["the answer"]
    assert engine._mem is None, "the log digest must not switch the memory subsystem on"
    assert not database.exists(), "a file-based preference must not create a database"


def test_a_preference_call_needs_no_device_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # The MCP door never needed one; the CLI door demanded a lease and refused with
    # "no device found" for what is a read of one local JSON file.
    def no_device(serial: str | None = None) -> Any:
        raise AssertionError("a preference call must not connect to a device")

    monkeypatch.setattr(AndroidPlatform, "connect", lambda self, target_id=None: no_device(target_id))
    monkeypatch.setenv("AUA_DAEMON__ENABLED", "false")

    result = runner.invoke(cli_app, ["logcat", "prefs", "show", "--app", APP])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["package"] == APP
