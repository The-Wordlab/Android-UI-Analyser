"""An action should say what the app *said*, not only what the screen shows.

The screen is the app's conclusion; its log is the reasoning. Between them sits the class of
failure that costs an agent its whole budget: the tap landed, the screen looks plausible, and
the app quietly logged the refusal. AUA already opens a device-clock window before every
action (``last-action``) — this is that window, reduced to something affordable enough to
attach to every folded observation.

Affordability is the whole design, and it is measured, not assumed. On one real app:

* an idle two-second window logged **0** lines — the feature costs nothing when nothing happens
* an ordinary tap logged 11 lines, all framework noise → **0** after filtering
* a cold app launch logged 210 lines (~30 KB); every one of the 113 ``I`` lines came from a
  third-party HTTP/attribution/advertising SDK or the ART runtime, and none carried app logic

Hence the default priority set ``DWEF``: ``I`` and ``V`` are dropped because they were pure
noise, and ``D`` is kept because that is where an app writes its own breadcrumbs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.logcat import DEFAULT_LEVELS, digest_app_logs
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.platforms.diagnostics import (
    DiagnosticEvent,
    DiagnosticLevel,
    DiagnosticWindow,
    UnknownDiagnosticMark,
)
from android_ui_analyser.platforms.identity import TargetRef
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

APP = "com.example.notes"


def _line(priority: str, tag: str, message: str, *, ms: int = 1) -> str:
    return f"08-21 18:14:44.{ms:03d}  5928  6079 {priority} {tag}: {message}"


def _obs() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(
            width=1080, height=2400, source=ScreenSource.hierarchy, package=APP, activity=f"{APP}/.Main"
        ),
        elements=[Element(id=0, type="View", bounds=[0, 0, 10, 10], center=[5, 5])],
        meta=Meta(duration_ms=1, tier_used=Tier.hierarchy, path=PathKind.hierarchy),
    )


# ------------------------------------------------------------------ the priority set


def test_the_default_priority_set_drops_verbose_and_info() -> None:
    # Measured: 113 of 113 `I` lines in a real launch window were third-party SDK / ART
    # chatter. Keeping them would triple the cost of every launch for no signal.
    assert DEFAULT_LEVELS == "DWEF"
    raw = "\n".join(
        [
            _line("V", "Anything", "trace"),
            _line("I", "MyOwnTag", "informational"),
            _line("D", "MyOwnTag", "breadcrumb"),
            _line("W", "MyOwnTag", "warned"),
            _line("E", "MyOwnTag", "failed"),
        ]
    )

    digest = digest_app_logs(raw, app_id=APP)

    assert [line.split(": ", 1)[1] for line in digest["lines"]] == [
        "breadcrumb",
        "warned",
        "failed",
    ]


def test_debug_is_kept_because_that_is_where_an_app_logs_its_own_breadcrumbs() -> None:
    raw = _line("D", "MyOwnTag", "resolved someInventedFlag=true")

    digest = digest_app_logs(raw, app_id=APP)

    assert digest["count"] == 1
    assert "someInventedFlag" in digest["lines"][0]


def test_the_level_set_is_adjustable_by_the_caller() -> None:
    raw = "\n".join([_line("D", "MyOwnTag", "breadcrumb"), _line("E", "MyOwnTag", "failed")])

    assert digest_app_logs(raw, app_id=APP, levels="WEF")["count"] == 1
    assert digest_app_logs(raw, app_id=APP, levels="DIWEF")["count"] == 2
    # Reported back, so a reader can tell a quiet window from a narrow filter.
    assert digest_app_logs(raw, app_id=APP, levels="WEF")["levels"] == "WEF"


def test_fatal_is_never_droppable_however_narrow_the_level_set() -> None:
    # A caller narrowing the filter must not be able to hide the one line that explains a
    # crash. `F` is added back whatever was asked for.
    raw = _line("F", "libc", "Fatal signal 11 (SIGSEGV)")

    digest = digest_app_logs(raw, app_id=APP, levels="D")

    assert digest["count"] == 1
    assert "F" in digest["levels"]


# ------------------------------------------------------------------ noise removal


def test_known_third_party_sdk_tags_are_dropped() -> None:
    raw = "\n".join(
        [
            _line("I", "OkHttp", "--> GET /v1/feed"),
            _line("D", "AppsFlyer_6.17.6", "sending event"),
            _line("D", "MyOwnTag", "kept"),
        ]
    )

    digest = digest_app_logs(raw, app_id=APP, levels="DIWEF")

    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["kept"]


def test_a_versioned_or_namespaced_sdk_tag_is_matched_by_prefix() -> None:
    # `AppsFlyer_6.17.6` and `TRuntime.CctTransportBackend` are the real shapes; one deny
    # entry has to cover every version an app might ship.
    raw = "\n".join(
        [
            _line("D", "AppsFlyer_9.99.9", "future version"),
            _line("I", "TRuntime.CctTransportBackend", "upload"),
        ]
    )

    assert digest_app_logs(raw, app_id=APP, levels="DIWEF")["count"] == 0


def test_the_art_runtime_tag_is_dropped_without_naming_any_app() -> None:
    # ART logs under the process name truncated to fit logcat's tag field, so for
    # `com.example.notes` the tag is a *suffix* of the package. Derived, never listed —
    # a hardcoded tag list would have to name real apps.
    raw = "\n".join(
        [
            _line("W", "example.notes", "Method void x.y() failed lock verification"),
            _line("I", "example.notes", "Background concurrent mark compact GC freed 18MB"),
            _line("E", "MyOwnTag", "kept"),
        ]
    )

    digest = digest_app_logs(raw, app_id=APP, levels="DIWEF")

    assert [line.split(": ", 1)[1] for line in digest["lines"]] == ["kept"]


def test_an_app_tag_that_merely_looks_short_is_not_mistaken_for_the_runtime_tag() -> None:
    raw = _line("E", "es", "a two-letter tag is not the runtime tag")

    assert digest_app_logs(raw, app_id=APP, levels="DIWEF")["count"] == 1


def test_the_deny_list_only_names_publicly_known_framework_and_sdk_tags() -> None:
    # This repo is public, and a deny list is exactly the kind of file that quietly accumulates
    # the tags of whichever app was being debugged that week. Every entry has to be a tag
    # anyone can find in Android or in a published library, so an app's own logger is never
    # listed — it survives the filter instead, which is also what makes the digest useful.
    from android_ui_analyser.logcat import DEFAULT_DENY_TAG_PREFIXES

    publicly_documented = {
        "AdvertisingIdClient",
        "AppsFlyer",
        "ApplicationLoaders",
        "Choreographer",
        "Chucker",
        "CompatChangeReporter",
        "DesktopExperienceFlags",
        "FirebaseSessions",
        "HWUI",
        "ImeTracker",
        "InsetsController",
        "LeakCanary",
        "OkHttp",
        "StrictMode",
        "TRuntime",
        "VRI[",
        "ViewRootImpl",
        "WindowOnBackDispatcher",
        "ashmem",
        "com.facebook.",
        "libEGL",
        "nativeloader",
    }
    assert set(DEFAULT_DENY_TAG_PREFIXES) == publicly_documented, (
        "a new deny entry must be a public framework/SDK tag, never an app's own logger"
    )


def test_buffer_separator_lines_are_not_reported_as_app_output() -> None:
    raw = "\n".join(["--------- beginning of main", _line("E", "MyOwnTag", "kept")])

    assert digest_app_logs(raw, app_id=APP)["count"] == 1


# ------------------------------------------------------------------ bounding the cost


def test_one_chatty_tag_cannot_spend_the_whole_line_budget() -> None:
    # The real failure: 44 near-identical config lines at launch would fill a 20-line cap
    # and push out the single error line that mattered.
    raw = "\n".join(
        [_line("D", "Chatty", f"line {i}", ms=i) for i in range(40)]
        + [_line("E", "TheOneThatMatters", "refused: quota exceeded", ms=900)]
    )

    digest = digest_app_logs(raw, app_id=APP, limit=20, per_tag=5)

    assert sum("Chatty:" in line for line in digest["lines"]) == 5
    assert any("quota exceeded" in line for line in digest["lines"])
    assert digest["omitted"] == 35


def test_the_line_cap_is_reported_rather_than_silently_applied() -> None:
    raw = "\n".join(_line("E", f"Tag{i}", f"line {i}", ms=i) for i in range(50))

    digest = digest_app_logs(raw, app_id=APP, limit=20, per_tag=5)

    assert digest["count"] == 20
    assert digest["total_count"] == 50
    assert digest["truncated"] is True
    # Head and tail, so the first and last thing the app said both survive the cap.
    assert "line 0" in digest["lines"][0]
    assert "line 49" in digest["lines"][-1]


def test_a_quiet_window_reports_nothing_at_all() -> None:
    # Measured at 0 lines for an idle window and for an ordinary tap. "Nothing to say" must
    # cost nothing, or the feature taxes every step of every flow.
    digest = digest_app_logs("", app_id=APP)

    assert digest["count"] == 0
    assert digest["lines"] == []


# ------------------------------------------------------------------ the platform boundary


def test_android_scopes_the_action_log_window_to_the_app_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = AndroidPlatform(Config())
    runtime = FakeDevice(package=APP)

    platform.diagnostic_logs(runtime, lines=100, since_ms=1_700_000_000_000, app_id=APP)

    assert ("logcat", (1_700_000_000_000, True, "5928")) in runtime.calls or any(
        name == "logcat" and len(args) > 2 and args[2] for name, args in runtime.calls
    ), "the app id must reach the runtime as a process filter, not be dropped"


class _FakeLogPlatform(PlatformAdapter):
    """A non-Android platform that can serve an app-scoped log window."""

    name = "fake-logs"
    capabilities = frozenset({"ui.tree", "device.logs"})

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[dict[str, object]] = []
        self.marks: dict[str, int] = {}

    def connect(self, target_id: str | None = None) -> object:
        raise AssertionError("the engine already holds its runtime")

    def list_targets(self) -> list[object]:
        return []

    def normalize_tree(
        self, raw_tree: str, screen_size: tuple[int, int], *, ignored_app_ids: Sequence[str] = ()
    ) -> NormalizedTree:
        return NormalizedTree([])

    def diagnostic_logs(
        self,
        runtime: object,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        return self.diagnostic_window(
            runtime,
            lines=lines,
            since=since_ms if since_ms is not None else 0,
            app_id=app_id,
        ).text

    def diagnostic_window(
        self,
        runtime: object,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        if isinstance(since, str):
            if since not in self.marks:
                raise UnknownDiagnosticMark(since, self.marks)
            since_ms = self.marks[since]
            since_label = since
        else:
            since_ms = int(since or 0)
            since_label = str(since) if since is not None else "30s"
        self.calls.append({"lines": lines, "since_ms": since_ms, "app_id": app_id})
        rendered = _line("E", "Checkout", "refused: quota exceeded")
        return DiagnosticWindow(
            events=(
                DiagnosticEvent(
                    message="refused: quota exceeded",
                    level=DiagnosticLevel.ERROR,
                    source="Checkout",
                    display_text=rendered,
                    app_id=app_id,
                ),
            ),
            target=TargetRef(self.name, "fake-target"),
            since=since_label,
            since_unix_ms=since_ms,
            clock="target",
        )

    def mark_diagnostics(
        self,
        runtime: object,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, object]:
        del runtime, clear, refresh_clock
        unix_ms = int(time.time() * 1000)
        self.marks[name] = unix_ms
        return {"name": name, "unix_ms": unix_ms, "iso": "fake", "clock": "target"}

    def clear_diagnostics(self, runtime: object) -> None:
        del runtime

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        del target_id, limit, app_id
        return []


class _NoLogPlatform(_FakeLogPlatform):
    name = "fake-no-logs"
    capabilities = frozenset({"ui.tree"})


def test_the_core_reads_the_window_through_the_selected_adapter() -> None:
    # No Android dependency anywhere in the path: a non-Android platform serving the window is
    # enough for the engine to produce the digest.
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    device = FakeDevice(package=APP)
    platform = _FakeLogPlatform(cfg)
    engine = Engine(cfg, device=device, platform=platform)
    engine.logcat_mark("last-action")

    digest = engine._app_logs(APP)

    assert digest is not None
    assert "quota exceeded" in "\n".join(digest["lines"])
    assert platform.calls[0]["app_id"] == APP
    assert isinstance(platform.calls[0]["since_ms"], int)
    assert not any(name == "logcat" for name, _args in device.calls), (
        "the engine must go through the adapter, never reach an Android runtime directly"
    )


def test_a_platform_that_cannot_serve_logs_costs_the_action_nothing() -> None:
    cfg = make_config(memory={"enabled": False}, lease={"enabled": False})
    device = FakeDevice(package=APP)
    engine = Engine(cfg, device=device, platform=_NoLogPlatform(cfg))

    assert engine._app_logs(APP) is None


# ------------------------------------------------------------------ the folded observation


def _engine(**cfg: object) -> tuple[Engine, FakeDevice]:
    device = FakeDevice(package=APP, activity=".Main")
    engine = Engine(
        make_config(memory={"enabled": False}, lease={"enabled": False}, **cfg),
        device=device,
    )
    engine._pre_action_state = {
        "count": 1,
        "focused": None,
        "labels": ["Save"],
        "rids": ["save"],
        "package": APP,
        "activity": f"{APP}/.Main",
        "known_screen": None,
    }
    return engine, device


def test_a_folded_action_observation_carries_what_the_app_logged(monkeypatch) -> None:
    engine, device = _engine()
    device.log_now("Stale", "from before the action", priority="E", offset_ms=-5_000)
    engine.logcat_mark("last-action")
    device.log_now("Checkout", "refused: quota exceeded", priority="E")
    monkeypatch.setattr(engine, "_analyze_post_action", lambda *_a, **_k: _obs())

    result = engine._observe(ActionResult(ok=True, action="tap"), True, settle=False)

    assert result.app_logs is not None
    assert result.app_logs["since"] == "last-action"
    joined = "\n".join(result.app_logs["lines"])
    assert "quota exceeded" in joined
    assert "from before the action" not in joined, "the window must start at the action"


def test_the_app_log_window_can_be_turned_off(monkeypatch) -> None:
    engine, device = _engine(logs={"enabled": False})
    engine.logcat_mark("last-action")
    device.log_now("Checkout", "refused", priority="E")
    monkeypatch.setattr(engine, "_analyze_post_action", lambda *_a, **_k: _obs())

    result = engine._observe(ActionResult(ok=True, action="tap"), True, settle=False)

    assert result.app_logs is None


def test_a_quiet_action_attaches_no_log_field_at_all(monkeypatch) -> None:
    engine, _device = _engine()
    engine.logcat_mark("last-action")
    monkeypatch.setattr(engine, "_analyze_post_action", lambda *_a, **_k: _obs())

    result = engine._observe(ActionResult(ok=True, action="tap"), True, settle=False)

    assert result.app_logs is None, "an empty window must not cost a field on every action"


# ------------------------------------------------------------------ drift guard


def test_every_process_replacing_call_tells_the_adapter_about_it() -> None:
    """A new launch/stop/clear/install site must not silently skip the invalidation.

    The failure this guards is invisible in review and invisible at runtime: scoping an action's
    log window to a process that has been replaced returns an empty window, and an empty window
    is the same shape as "the app logged nothing". So the rule is enforced against the source
    rather than trusted to memory — this is the same reasoning as the undo-registration guard.
    """
    import ast
    from pathlib import Path

    src_dir = Path(__file__).resolve().parents[1] / "src" / "android_ui_analyser"
    # engine.py and its engine_*.py domain modules together are "the engine"
    trees = [ast.parse(p.read_text(encoding="utf-8")) for p in sorted(src_dir.glob("engine*.py"))]
    replacing = {"stop_app", "clear_app", "launch_app", "install_app_bundle", "uninstall_app"}

    offenders: list[str] = []
    for node in (n for tree in trees for n in ast.walk(tree)):
        if not isinstance(node, ast.FunctionDef):
            continue
        called = {
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        if called & replacing and "_app_process_replaced" not in called:
            offenders.append(f"{node.name} (line {node.lineno})")

    assert not offenders, (
        "these engine methods replace an app process without telling the adapter, so the next "
        "action would scope its log window to a dead pid and read back a misleading empty "
        f"window — call self._app_process_replaced(<app id>): {offenders}"
    )


def test_an_apps_own_printing_is_never_denied() -> None:
    # `System.out` / `System.err` is how an app prints. A `System` prefix would have been the
    # single biggest win by line count on a real launch window, and taking it would have deleted
    # real app output to remove framework chatter.
    raw = "\n".join(
        [
            _line("W", "System.err", "at com.example.app.Checkout.submit(Checkout.kt:42)"),
            _line("D", "System.out", "printed by the app"),
        ]
    )

    assert digest_app_logs(raw, app_id=APP)["count"] == 2


def test_the_log_window_is_not_archived_into_a_publishable_evidence_bundle() -> None:
    """Session artifacts are review evidence that gets published; log lines must not ride along.

    `calls.jsonl` stores the whole result payload, so adding a field to an action result silently
    adds it to every archived bundle too. Raw device log lines are the likeliest place for a
    bearer token, an install id, or an unreleased flag name to appear, so the archive keeps the
    fact that the app spoke and withholds what it said.
    """
    from android_ui_analyser.session_artifacts import _redact

    archived = _redact(
        {
            "ok": True,
            "app_logs": {
                "app_id": APP,
                "levels": "DWEF",
                "count": 2,
                "total_count": 9,
                "omitted": 7,
                "truncated": False,
                "since": "last-action",
                "lines": [_line("E", "Checkout", "Bearer abc123 rejected")],
            },
        }
    )

    assert "Bearer abc123" not in json.dumps(archived)
    assert archived["app_logs"]["withheld"] == "app_logs not archived"
    # Still says the app spoke, and how much was withheld — the archive stays honest.
    assert archived["app_logs"]["count"] == 2
    assert archived["app_logs"]["total_count"] == 9


def test_a_second_observation_of_the_same_window_does_not_report_it_twice(monkeypatch) -> None:
    """A wait stamps no new mark, so it must not re-serve the previous action's lines.

    Without this the same log block reappears on the next observation and reads as though the app
    had just said all of it again — which is how an agent concludes a failure is still happening
    after it has stopped.
    """
    engine, device = _engine()
    engine.logcat_mark("last-action")
    device.log_now("Checkout", "refused: quota exceeded", priority="E")
    monkeypatch.setattr(engine, "_analyze_post_action", lambda *_a, **_k: _obs())

    first = engine._observe(ActionResult(ok=True, action="tap"), True, settle=False)
    second = engine._observe(ActionResult(ok=True, action="await"), True, settle=False)

    assert first.app_logs is not None
    assert second.app_logs is None


def test_the_launcher_is_never_reported_as_the_app_under_test(monkeypatch) -> None:
    # Measured on a real device: a Back to home attached 20 lines of launcher animation state
    # under a field claiming to be the app's own output.
    launcher = "com.google.android.apps.nexuslauncher"
    device = FakeDevice(package=launcher, activity=".NexusLauncherActivity")
    engine = Engine(
        make_config(memory={"enabled": False}, lease={"enabled": False}), device=device
    )
    engine._pre_action_state = {
        "count": 1,
        "focused": None,
        "labels": [],
        "rids": [],
        "package": launcher,
        "activity": f"{launcher}/.NexusLauncherActivity",
        "known_screen": None,
    }
    engine.logcat_mark("last-action")
    device.log_now("LauncherStateManager", "goToState: AllApps -> Normal", priority="D")
    monkeypatch.setattr(
        engine,
        "_analyze_post_action",
        lambda *_a, **_k: AnalyzeResult(
            screen=Screen(
                width=1080,
                height=2400,
                source=ScreenSource.hierarchy,
                package=launcher,
                activity=f"{launcher}/.NexusLauncherActivity",
            ),
            elements=[],
            meta=Meta(duration_ms=1, tier_used=Tier.hierarchy, path=PathKind.hierarchy),
        ),
    )

    result = engine._observe(ActionResult(ok=True, action="key"), True, settle=False)

    assert result.app_logs is None
