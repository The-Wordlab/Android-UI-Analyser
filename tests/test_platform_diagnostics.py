"""Normalized diagnostics are sufficient for shared Engine behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UnsupportedPlatformCapabilityError
from android_ui_analyser.logcat import clock_path, marks_path
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.diagnostics import (
    AppExitEvidence,
    CrashEvidence,
    DiagnosticEvent,
    DiagnosticLevel,
    DiagnosticWindow,
    UnknownDiagnosticMark,
)
from android_ui_analyser.platforms.identity import TargetRef
from android_ui_analyser.platforms.runtime import TargetRuntime
from android_ui_analyser.schema import (
    AnalyzeResult,
    AppContext,
    Element,
    Meta,
    PathKind,
    Screen,
    ScreenSource,
    Tier,
)
from conftest import FakeDevice, make_config


class _NeutralPlatform(PlatformAdapter):
    name = "sample-os"
    capabilities = frozenset({"device.logs"})

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[tuple[str, Any]] = []
        self.marks: dict[str, int] = {}
        self.events: tuple[DiagnosticEvent, ...] = (
            DiagnosticEvent(
                message="render complete",
                level=DiagnosticLevel.INFO,
                source="renderer",
                display_text="sample diagnostic | renderer | render complete",
            ),
        )

    def connect(self, target_id: str | None = None) -> TargetRuntime:
        raise AssertionError("runtime is injected")

    def list_targets(self) -> list[Any]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        del raw_tree, screen_size, ignored_app_ids
        return NormalizedTree([])

    def mark_diagnostics(
        self,
        runtime: TargetRuntime,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, object]:
        del clear, refresh_clock
        self.calls.append(("mark", (runtime, name)))
        self.marks[name] = 42
        return {"name": name, "unix_ms": 42, "iso": "sample", "clock": "target"}

    def diagnostic_window(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        self.calls.append(("window", (runtime, lines, since, app_id)))
        if isinstance(since, str):
            if since not in self.marks:
                raise UnknownDiagnosticMark(since, self.marks)
            since_ms = self.marks[since]
            since_label = since
        else:
            since_ms = int(since or 42)
            since_label = str(since) if since is not None else "current-window"
        return DiagnosticWindow(
            events=self.events,
            target=TargetRef(self.name, runtime.target_id),
            since=since_label,
            since_unix_ms=since_ms,
            clock="target",
        )

    def diagnostic_logs(
        self,
        runtime: TargetRuntime,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        return self.diagnostic_window(
            runtime,
            lines=lines,
            since=since_ms,
            app_id=app_id,
        ).text

    def clear_diagnostics(self, runtime: TargetRuntime) -> None:
        self.calls.append(("clear", runtime))

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        del target_id, app_id
        return [event.text for event in self.events][-limit:]


class _NoDiagnosticsPlatform(_NeutralPlatform):
    name = "no-diagnostics"
    capabilities = frozenset()


class _ExitPlatform(_NoDiagnosticsPlatform):
    name = "exit-platform"

    def app_exit_evidence(
        self,
        before: object,
        after: object,
        elements: Sequence[Element],
    ) -> AppExitEvidence | None:
        del before, after, elements
        return AppExitEvidence("sample.app", "sample.shell", crash_dialog=True)


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=100, height=200, source=ScreenSource.hierarchy),
        elements=[],
        meta=Meta(duration_ms=1, tier_used=Tier.hierarchy, path=PathKind.hierarchy),
    )


def test_normalized_window_filters_and_digests_without_native_grammar() -> None:
    hidden = DiagnosticEvent(
        message="framework chatter",
        level=DiagnosticLevel.DEBUG,
        source="framework",
        display_text="opaque native rendering one",
        hidden_by_default=True,
    )
    fatal = DiagnosticEvent(
        message="process terminated",
        level=DiagnosticLevel.FATAL,
        source="framework",
        display_text="opaque native rendering two",
        hidden_by_default=True,
    )
    window = DiagnosticWindow(events=(hidden, fatal))

    assert window.select_lines(source="framework", grep="native rendering") == [
        hidden.text,
        fatal.text,
    ]
    digest = window.digest(levels="D", drop_source_prefixes=("framework",))
    assert digest["lines"] == [fatal.text], "fatal evidence must survive every preference"


def test_android_legacy_and_other_platform_marks_cannot_share_a_path(tmp_path) -> None:
    android = TargetRef("android", "same:target")
    other = TargetRef("sample-os", "same:target")

    assert marks_path(tmp_path, android) == marks_path(tmp_path, "same:target")
    assert clock_path(tmp_path, android) == clock_path(tmp_path, "same:target")
    assert marks_path(tmp_path, other) != marks_path(tmp_path, android)
    assert clock_path(tmp_path, other) != clock_path(tmp_path, android)


def test_public_log_aliases_consume_the_selected_platform_window() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    runtime = FakeDevice()
    platform = _NeutralPlatform(cfg)
    engine = Engine(cfg, device=runtime, platform=platform)

    mark = engine.logcat_mark("checkpoint")
    dump = engine.logcat(since="checkpoint", grep="render complete")

    assert mark["action"] == "logcat-mark"
    assert dump["lines"] == ["sample diagnostic | renderer | render complete"]
    assert not any(name == "logcat" for name, _args in runtime.calls)


def test_android_adapter_parses_logcat_into_normalized_evidence() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    runtime = FakeDevice(package="com.example.app")
    platform = AndroidPlatform(cfg)
    platform.mark_diagnostics(runtime, "before")
    fatal = runtime.log_now(
        "AndroidRuntime",
        "FATAL EXCEPTION: main",
        priority="F",
    )
    runtime.log_now(
        "AndroidRuntime",
        "Process: com.example.app, PID: 1234",
        priority="E",
    )

    window = platform.diagnostic_window(
        runtime,
        since="before",
        app_id="com.example.app",
    )

    assert window.target == TargetRef("android", runtime.serial)
    assert window.events[0].level is DiagnosticLevel.FATAL
    assert window.events[0].source == "AndroidRuntime"
    assert window.events[0].message == "FATAL EXCEPTION: main"
    assert window.events[0].text == fatal
    assert window.crash_evidence.kind == "fatal"
    assert window.crash_evidence.matched_app is True


def test_log_wait_uses_normalized_window_and_never_the_runtime_logcat() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    runtime = FakeDevice()
    platform = _NeutralPlatform(cfg)

    result = Engine(cfg, device=runtime, platform=platform).await_predicate(
        "log:render complete",
        timeout_ms=10,
        poll_ms=1,
        observe=False,
    )

    assert result.await_outcome == "satisfied"
    assert any(name == "window" for name, _value in platform.calls)
    assert not any(name == "logcat" for name, _args in runtime.calls)


def test_explicit_log_wait_on_an_unsupported_platform_is_typed() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    runtime = FakeDevice()

    with pytest.raises(UnsupportedPlatformCapabilityError):
        Engine(cfg, device=runtime, platform=_NoDiagnosticsPlatform(cfg)).await_predicate(
            "log:anything",
            timeout_ms=1,
            poll_ms=1,
            observe=False,
        )


def test_explicit_log_dump_on_an_unsupported_platform_is_typed() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    runtime = FakeDevice()

    with pytest.raises(UnsupportedPlatformCapabilityError):
        Engine(cfg, device=runtime, platform=_NoDiagnosticsPlatform(cfg)).logcat()


def test_engine_serializes_adapter_owned_app_exit_evidence() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    engine = Engine(cfg, device=FakeDevice(), platform=_ExitPlatform(cfg))

    assert engine._app_left_foreground("opaque-before", "opaque-after", _observation()) == {
        "from": "sample.app",
        "to": "sample.shell",
        "crash_dialog": True,
    }


def test_change_summary_keeps_app_context_neutral_until_the_output_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContextPlatform(_NoDiagnosticsPlatform):
        name = "context-platform"

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.contexts: tuple[object, object] | None = None

        def app_exit_evidence(
            self,
            before: object,
            after: object,
            elements: Sequence[Element],
        ) -> AppExitEvidence | None:
            del elements
            self.contexts = (before, after)
            return None

    before = AppContext(app_id="sample.app", surface_id="first")
    after = AppContext(app_id="sample.app", surface_id="second")
    platform = ContextPlatform(make_config(memory={"enabled": False}, lease={"enabled": False}))
    engine = Engine(platform.config, device=FakeDevice(), platform=platform)
    monkeypatch.setattr(engine, "_read_app_context", lambda: after)

    change = engine._change_summary(
        {
            "count": 0,
            "focused": None,
            "labels": [],
            "rids": [],
            "app_context": before,
            "activity": "legacy/value",
        },
        _observation(),
    )

    assert platform.contexts == (before, after)
    assert change["activity_before"] == "sample.app/first"
    assert change["activity_after"] == "sample.app/second"


def test_crash_evidence_is_a_normalized_value_object() -> None:
    event = DiagnosticEvent("fatal detail", level=DiagnosticLevel.FATAL)
    evidence = CrashEvidence(
        kind="fatal",
        events=(event,),
        total_count=1,
        matched_app=True,
    )

    assert evidence.as_dict() == {
        "kind": "fatal",
        "lines": ["fatal detail"],
        "count": 1,
        "total_count": 1,
        "truncated": False,
        "matched_app": True,
    }
