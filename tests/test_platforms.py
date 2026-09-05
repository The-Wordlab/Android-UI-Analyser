"""Platform factory/strategy seam; Android remains the only built-in implementation."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from android_ui_analyser.cli import GlobalOpts, hoist_global_options
from android_ui_analyser.config import Config, load_config
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import (
    ConfigError,
    InvalidPlatformCapabilityError,
    UnsupportedPlatformCapabilityError,
)
from android_ui_analyser.platforms import (
    CAPABILITY_METHODS,
    DisplayGeometry,
    NormalizedTree,
    PlatformAdapter,
    PlatformFactory,
    register_platform,
    registered_platforms,
)
from android_ui_analyser.platforms.android import AndroidPlatform
from android_ui_analyser.platforms.diagnostics import (
    CrashEvidence,
    DiagnosticEvent,
    DiagnosticLevel,
    DiagnosticWindow,
)
from android_ui_analyser.platforms.identity import TargetRef
from android_ui_analyser.schema import DeviceInfo, Element
from conftest import FakeDevice


@register_platform("test-native")
class _RegisteredPlatform(PlatformAdapter):
    capabilities = frozenset({"ui.tree"})

    def connect(self, target_id: str | None = None) -> Device:
        raise AssertionError("factory selection must not connect eagerly")

    def list_targets(self) -> list[DeviceInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
        geometry: DisplayGeometry | None = None,
    ) -> NormalizedTree:
        del geometry
        return NormalizedTree([])


class _InjectedPlatform(_RegisteredPlatform):
    name = "injected"
    capabilities = frozenset({"ui.tree", "ui.screenshot"})

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.calls: list[tuple[str, object]] = []

    def dump_tree(self, runtime: Device, *, compact: bool = False) -> str:
        self.calls.append(("dump", compact))
        return "native-tree"

    def capture_screenshot(self, runtime: Device):
        return runtime.screenshot()

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
        geometry: DisplayGeometry | None = None,
    ) -> NormalizedTree:
        del geometry
        self.calls.append(("normalize", (raw_tree, screen_size, tuple(ignored_app_ids))))
        element = Element(
            id=0,
            type="Button",
            text="From plugin",
            bounds=(10, 20, 110, 70),
            center=(60, 45),
            clickable=True,
        )
        return NormalizedTree([element], app_id="example.native")


class _FakeDatabaseService:
    def __init__(self) -> None:
        self.calls: list[tuple[Device, str]] = []

    def list_databases(self, runtime: Device, app_id: str) -> dict[str, object]:
        self.calls.append((runtime, app_id))
        return {"ok": True, "databases": ["plugin.db"]}

    backup_database = database_schema = execute_database = list_backups = list_databases
    query_database = restore_database = list_databases


class _CapabilityPlatform(_InjectedPlatform):
    name = "capability-test"
    capabilities = frozenset({"ui.tree", "app_database"})

    def __init__(self, config: Config, database: _FakeDatabaseService) -> None:
        super().__init__(config)
        self.database = database

    def load_capability(self, capability: str) -> object | None:
        return self.database if capability == "app_database" else None


class _LogPlatform(_InjectedPlatform):
    name = "log-test"
    capabilities = frozenset({"ui.tree", "device.logs"})

    def diagnostic_logs(
        self,
        runtime: Device,
        *,
        lines: int = 400,
        since_ms: int | None = None,
        app_id: str | None = None,
    ) -> str:
        self.calls.append(("diagnostic_logs", (runtime, lines, since_ms, app_id)))
        return (
            "08-20 12:00:00.001  1234  1234 E AndroidRuntime: FATAL EXCEPTION: main\n"
            "08-20 12:00:00.002  1234  1234 E AndroidRuntime: "
            "Process: example.native, PID: 1234\n"
            "08-20 12:00:00.003  1234  1234 E AndroidRuntime: "
            "java.lang.IllegalStateException: broken\n"
        )

    def diagnostic_window(
        self,
        runtime: Device,
        *,
        lines: int = 400,
        since: str | int | None = None,
        app_id: str | None = None,
    ) -> DiagnosticWindow:
        self.calls.append(("diagnostic_window", (runtime, lines, since, app_id)))
        records = (
            DiagnosticEvent(
                message="FATAL EXCEPTION: main",
                level=DiagnosticLevel.ERROR,
                source="native-runtime",
                display_text="native fatal: FATAL EXCEPTION: main",
                app_id=app_id,
            ),
            DiagnosticEvent(
                message="java.lang.IllegalStateException: broken",
                level=DiagnosticLevel.ERROR,
                source="native-runtime",
                display_text="native fatal: java.lang.IllegalStateException: broken",
                app_id=app_id,
            ),
        )
        return DiagnosticWindow(
            events=records,
            target=TargetRef(self.name, runtime.target_id),
            since="last-action",
            since_unix_ms=123456,
            clock="target",
            crash_evidence=CrashEvidence(
                kind="fatal",
                events=records,
                total_count=len(records),
                matched_app=True,
            ),
        )

    def mark_diagnostics(
        self,
        runtime: Device,
        name: str = "default",
        *,
        clear: bool = False,
        refresh_clock: bool = False,
    ) -> dict[str, object]:
        del runtime, clear, refresh_clock
        return {"name": name, "unix_ms": 123456, "iso": "fake", "clock": "target"}

    def clear_diagnostics(self, runtime: Device) -> None:
        del runtime

    def recent_logs(
        self, target_id: str, *, limit: int = 80, app_id: str | None = None
    ) -> list[str]:
        del target_id, limit, app_id
        return []


def test_android_is_the_only_builtin_platform() -> None:
    builtins = registered_platforms()
    assert builtins["android"] is AndroidPlatform
    assert AndroidPlatform(Config()).supports("ui.tree")


def test_factory_selects_and_memoizes_registered_strategy() -> None:
    cfg = Config.model_validate({"device": {"platform": "TEST-NATIVE"}})
    factory = PlatformFactory(cfg)

    first = factory.create()

    assert isinstance(first, _RegisteredPlatform)
    assert factory.create() is first


def test_missing_optional_capability_is_a_typed_platform_refusal() -> None:
    platform = _RegisteredPlatform(Config())

    with pytest.raises(UnsupportedPlatformCapabilityError) as exc:
        platform.capability("virtual-devices")

    assert exc.value.code == "platform_capability_unsupported"
    assert "test-native" in exc.value.message
    assert "virtual_targets" in exc.value.message


def test_android_capabilities_are_lazy_and_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = AndroidPlatform(Config())
    loaded: list[str] = []

    class Service:
        create_virtual_target = delete_virtual_target = list_virtual_targets = (
            provision_virtual_target
        ) = reclaim_virtual_targets = select_virtual_target = start_virtual_target = (
            stop_virtual_targets
        ) = stop_virtual_target_instance = virtual_target_status = lambda *args, **kwargs: None

    sentinel = Service()
    monkeypatch.setattr(platform, "prepare_host", lambda: None)
    monkeypatch.setattr(
        "android_ui_analyser.platforms.android.importlib.import_module",
        lambda name: loaded.append(name) or sentinel,
    )

    assert platform.capability("virtual_devices") is sentinel
    assert platform.capability("virtual-devices") is sentinel
    assert loaded == ["android_ui_analyser.emulator"]


def test_android_recent_logs_uses_the_app_process_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = AndroidPlatform(Config())
    calls: list[list[str]] = []

    def run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(args)
        if "pidof" in args:
            return SimpleNamespace(stdout="4312 4313\n")
        return SimpleNamespace(
            stdout="08-21 11:50:00.000  4312  4312 I Notes: app-scoped line\n"
        )

    monkeypatch.setattr(platform, "prepare_host", lambda: None)
    monkeypatch.setattr("android_ui_analyser.platforms.android.subprocess.run", run)

    lines = platform.recent_logs(
        "emulator-5554", limit=50, app_id="com.example.notes"
    )

    assert lines == ["08-21 11:50:00.000  4312  4312 I Notes: app-scoped line"]
    assert calls[0] == [
        "adb",
        "-s",
        "emulator-5554",
        "shell",
        "pidof",
        "com.example.notes",
    ]
    assert calls[1] == [
        "adb",
        "-s",
        "emulator-5554",
        "logcat",
        "-d",
        "-v",
        "threadtime",
        "--pid",
        "4312",
        "-t",
        "50",
    ]


def test_android_services_satisfy_every_common_capability_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = AndroidPlatform(Config())
    monkeypatch.setattr(platform, "prepare_host", lambda: None)

    assert set(CAPABILITY_METHODS) <= platform.capabilities
    for capability in CAPABILITY_METHODS:
        assert platform.capability(capability) is not None


def test_incomplete_plugin_capability_is_rejected_at_the_gate() -> None:
    class Broken(_RegisteredPlatform):
        name = "broken"
        capabilities = frozenset({"ui.tree", "app_database"})

        def load_capability(self, capability: str) -> object | None:
            return object() if capability == "app_database" else None

    with pytest.raises(InvalidPlatformCapabilityError) as exc:
        Broken(Config()).capability("app_database")

    assert exc.value.code == "platform_capability_invalid"
    assert "list_databases" in exc.value.message


def test_factory_reports_installed_platforms_for_unknown_name() -> None:
    factory = PlatformFactory(Config.model_validate({"device": {"platform": "missing"}}))

    with pytest.raises(ConfigError, match="unknown platform 'missing'") as exc:
        factory.create()

    assert "android" in (exc.value.hint or "")


def test_engine_uses_injected_strategy_for_tree_capture_and_normalization() -> None:
    cfg = Config.model_validate(
        {
            "memory": {"enabled": False},
            "perf": {"prefetch": False},
            "lease": {"enabled": False},
        }
    )
    runtime = FakeDevice(package="example.native")
    platform = _InjectedPlatform(cfg)
    engine = Engine(cfg, device=runtime, platform=platform)

    result = engine.analyze(source="hierarchy", record=False)

    assert [element.text for element in result.elements] == ["From plugin"]
    assert platform.calls == [
        ("dump", True),
        (
            "normalize",
            ("native-tree", (1080, 2400), tuple(cfg.memory.ignore_packages)),
        ),
    ]


def test_locale_metadata_is_optional_on_a_non_android_runtime() -> None:
    cfg = Config.model_validate({"memory": {"enabled": False}, "lease": {"enabled": False}})

    class LocaleNeutralRuntime(FakeDevice):
        device_locale = Device.device_locale

    runtime = LocaleNeutralRuntime(package="example.native")
    platform = _InjectedPlatform(cfg)

    result = Engine(cfg, device=runtime, platform=platform).analyze(
        source="hierarchy", record=False
    )

    assert result.meta.device_locale is None
    assert not any(name == "shell" for name, _args in runtime.calls)


def test_flow_artifacts_do_not_fall_back_to_android_logs_on_another_platform(tmp_path) -> None:
    cfg = Config.model_validate(
        {
            "memory": {"enabled": False},
            "cache": {"dir": str(tmp_path / "cache")},
            "perf": {"prefetch": False},
            "lease": {"enabled": False},
        }
    )
    runtime = FakeDevice(package="example.native")
    platform = _InjectedPlatform(cfg)

    result = Engine(cfg, device=runtime, platform=platform).flow_run(
        yaml="steps:\n  - assert: {text: Missing, exists: true}\n",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert result["ok"] is False
    assert not any(call[0] == "logcat" for call in runtime.calls)
    assert not (tmp_path / "artifacts" / "failure-diagnostics.txt").exists()


def test_android_platform_provides_bounded_failure_diagnostics() -> None:
    runtime = FakeDevice()
    runtime.log_now(tag="First", msg="older")
    runtime.log_now(tag="Second", msg="newer")

    logs = AndroidPlatform(Config()).diagnostic_logs(runtime, lines=1, since_ms=123456)

    assert "newer" in logs
    assert "older" not in logs
    assert ("logcat", (123456, True, None)) in runtime.calls


def test_crash_evidence_uses_the_selected_platform_log_capability() -> None:
    cfg = Config.model_validate(
        {
            "memory": {"enabled": False},
            "lease": {"enabled": False},
        }
    )
    runtime = FakeDevice(package="example.native")
    platform = _LogPlatform(cfg)

    evidence = Engine(cfg, device=runtime, platform=platform)._crash_evidence("example.native")

    assert evidence["available"] is True
    assert evidence["kind"] == "fatal"
    assert "IllegalStateException" in "\n".join(evidence["lines"])
    diagnostic_call = next(call for call in platform.calls if call[0] == "diagnostic_window")
    assert diagnostic_call[1][0] is runtime
    assert diagnostic_call[1][2] is None, "the adapter owns resolving its diagnostic cursor"
    assert not any(name == "logcat" for name, _args in runtime.calls), (
        "the engine must use the selected adapter, not reach an Android runtime directly"
    )


def test_crash_evidence_reports_an_unsupported_log_capability() -> None:
    cfg = Config.model_validate({"memory": {"enabled": False}, "lease": {"enabled": False}})
    runtime = FakeDevice(package="example.native")

    evidence = Engine(
        cfg,
        device=runtime,
        platform=_InjectedPlatform(cfg),
    )._crash_evidence("example.native")

    assert evidence == {
        "available": False,
        "source": "device.logs",
        "app_id": "example.native",
        "code": "platform_capability_unsupported",
        "detail": "platform 'injected' does not support capability 'device.logs'",
    }
    assert not any(name == "logcat" for name, _args in runtime.calls)


def test_engine_optional_action_uses_selected_platform_capability() -> None:
    cfg = Config.model_validate({"memory": {"enabled": False}, "lease": {"enabled": False}})
    runtime = FakeDevice(package="example.native")
    service = _FakeDatabaseService()
    platform = _CapabilityPlatform(cfg, service)
    engine = Engine(cfg, device=runtime, platform=platform)

    result = engine.database_list("example.native")

    assert result == {"ok": True, "databases": ["plugin.db"]}
    assert service.calls == [(runtime, "example.native")]


def test_android_strategy_normalizes_xml_and_ignores_system_overlay_package() -> None:
    raw = """<hierarchy>
      <node class="android.widget.TextView" text="App" package="com.example.app"
            bounds="[0,0][100,50]" />
      <node class="android.widget.TextView" text="Clock" package="com.android.systemui"
            bounds="[0,50][100,100]" />
      <node class="android.widget.TextView" text="Shade" package="com.android.systemui"
            bounds="[0,100][100,150]" />
    </hierarchy>"""

    normalized = AndroidPlatform(Config()).normalize_tree(
        raw,
        (100, 150),
        ignored_app_ids=("com.android.systemui",),
    )

    assert normalized.app_id == "com.example.app"
    assert [element.text for element in normalized.elements] == ["App", "Clock", "Shade"]


def test_platform_selection_flows_through_env_and_combined_cli_device_overrides(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cfg = load_config(cwd=tmp_path, env={"AUA_PLATFORM": "TEST-NATIVE"})

    overrides = GlobalOpts(platform="test-native", serial="target-1").cli_overrides()

    assert cfg.device.platform == "test-native"
    assert overrides["device"] == {"platform": "test-native", "serial": "target-1"}
    assert hoist_global_options(["devices", "--platform", "test-native"]) == [
        "--platform",
        "test-native",
        "devices",
    ]
