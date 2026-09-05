"""Cross-platform coordination state never aliases targets with the same native id."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from android_ui_analyser import capture_sidecar, daemon, device_ledger, journal, leases, teardown
from android_ui_analyser.capture import CaptureBuffer, CaptureCfgView
from android_ui_analyser.config import Config
from android_ui_analyser.engine import Engine
from android_ui_analyser.platforms.identity import AppRef, TargetRef, safe_component
from android_ui_analyser.session import active_session_metadata, create_session_state


class _Target:
    def __init__(self, target_id: str) -> None:
        self.serial = target_id
        self.calls: list[tuple[str, Any]] = []

    def instance_token(self) -> None:
        return None

    def set_http_proxy(self, value: str | None) -> None:
        self.calls.append(("set_http_proxy", value))


class _Adapter:
    def __init__(self, name: str, target: _Target) -> None:
        self.name = name
        self.target = target

    def connect(self, target_id: str | None = None) -> _Target:
        assert target_id == self.target.serial
        return self.target

    def validate_runtime(self, runtime: _Target) -> _Target:
        return runtime

    def runtime_capability(self, name: str, runtime: _Target) -> _Target:
        return runtime

    def capability(self, name: str) -> Any:
        raise AssertionError(f"unexpected capability {name}")


def _record(ref: TargetRef, cache_dir: Path) -> None:
    device_ledger.record(
        ref,
        key="http_proxy",
        kind="http_proxy",
        op="set_http_proxy",
        args={"host_port": None},
        cache_dir=cache_dir,
        leased=True,
    )


def test_target_ref_preserves_android_paths_and_namespaces_plugins(tmp_path: Path) -> None:
    android = TargetRef("android", "shared/id")
    ios = TargetRef("ios", "shared/id")

    assert android.storage_key == "shared_id"
    assert not android.storage_key.startswith("android--")
    assert ios.storage_key == "@ios@shared%2Fid"
    assert android.storage_key != ios.storage_key

    assert leases.acquire(tmp_path, android, owner="same-worker")
    assert leases.acquire(tmp_path, ios, owner="same-worker")

    assert leases.holder(tmp_path, android) == "same-worker"
    assert leases.holder(tmp_path, ios) == "same-worker"
    assert leases.read_lease(tmp_path, android)["platform"] == "android"  # type: ignore[index]
    assert leases.read_lease(tmp_path, ios)["platform"] == "ios"  # type: ignore[index]
    assert teardown.watchdog_pid_path(android) != teardown.watchdog_pid_path(ios)

    android_app = AppRef("android", "com.example.notes")
    ios_app = AppRef("ios", "com.example.notes")
    assert android_app.package == android_app.app_id
    assert android_app.storage_key != ios_app.storage_key


def test_plugin_storage_keys_cannot_alias_platforms_or_legacy_android(tmp_path: Path) -> None:
    refs = [
        TargetRef("alpha--beta", "gamma"),
        TargetRef("alpha", "beta--gamma"),
        TargetRef("android", "alpha--beta--gamma"),
        TargetRef("alpha", "a/b"),
        TargetRef("alpha", "a%2Fb"),
        TargetRef("alpha", "../target"),
    ]
    assert len({ref.storage_key for ref in refs}) == len(refs)
    for index, ref in enumerate(refs):
        assert leases.acquire(tmp_path, ref, owner=f"worker-{index}")
        _record(ref, tmp_path)
    for index, ref in enumerate(refs):
        assert leases.holder(tmp_path, ref) == f"worker-{index}"
        assert device_ledger.read_ledger(ref)[0].target_id == ref.target_id
    assert len({AppRef(ref.platform, ref.target_id).storage_key for ref in refs}) == len(refs)


def test_platform_path_components_never_traverse_or_alias_escaped_names() -> None:
    names = [".", "..", "a/b", "a%2Fb", "a@b", "a--b", "a_b", "Mixed", "mixed"]
    components = [safe_component(name) for name in names]
    assert len({value.casefold() for value in components}) == len(names)
    assert all(Path(value).name == value and value not in {".", ".."} for value in components)


def test_legacy_lease_without_platform_is_android_only(tmp_path: Path) -> None:
    path = leases._lease_path(tmp_path, "same-id")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "serial": "same-id",
                "owner": "legacy-worker",
                "acquired": 100.0,
                "last_activity": 100.0,
                "ttl_s": 10_000_000_000,
            }
        ),
        encoding="utf-8",
    )

    assert leases.holder(tmp_path, "same-id") == "legacy-worker"
    assert leases.holder(tmp_path, TargetRef("ios", "same-id")) is None


def test_legacy_ledger_without_platform_is_android(tmp_path: Path) -> None:
    path = device_ledger.ledger_path("legacy-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "serial": "legacy-1",
                "entries": [
                    {
                        "key": "http_proxy",
                        "kind": "http_proxy",
                        "op": "set_http_proxy",
                        "args": {"host_port": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert device_ledger.pending_targets() == [TargetRef("android", "legacy-1")]
    entry = device_ledger.read_ledger("legacy-1")[0]
    assert entry.platform == "android"
    assert entry.target_id == "legacy-1"
    assert device_ledger.read_ledger(TargetRef("ios", "legacy-1")) == []


def test_sweep_replays_each_target_through_its_recorded_platform(tmp_path: Path) -> None:
    android_ref = TargetRef("android", "same-id")
    ios_ref = TargetRef("ios", "same-id")
    _record(android_ref, tmp_path)
    _record(ios_ref, tmp_path)

    android_target = _Target("same-id")
    ios_target = _Target("same-id")
    android = _Adapter("android", android_target)
    ios = _Adapter("ios", ios_target)
    created: list[str] = []

    def adapter_for(name: str) -> _Adapter:
        created.append(name)
        assert name == "android"
        return android

    reports = teardown.sweep(
        platform=ios,
        platform_factory=adapter_for,
        cache_dir=tmp_path,
        grace_s=0,
    )

    assert {report["platform"] for report in reports} == {"android", "ios"}
    assert created == ["android"]
    assert android_target.calls == [("set_http_proxy", None)]
    assert ios_target.calls == [("set_http_proxy", None)]
    assert device_ledger.read_ledger(android_ref) == []
    assert device_ledger.read_ledger(ios_ref) == []


def test_sweep_never_uses_the_selected_adapter_for_another_platform(tmp_path: Path) -> None:
    android_ref = TargetRef("android", "same-id")
    _record(android_ref, tmp_path)
    ios_target = _Target("same-id")

    reports = teardown.sweep(
        platform=_Adapter("ios", ios_target),
        cache_dir=tmp_path,
        grace_s=0,
    )

    assert reports[0]["platform"] == "android"
    assert "no adapter available" in reports[0]["skipped"]
    assert ios_target.calls == []
    assert device_ledger.read_ledger(android_ref)


def test_direct_reap_refuses_an_adapter_from_another_platform(tmp_path: Path) -> None:
    ref = TargetRef("ios", "same-id")
    _record(ref, tmp_path)
    android_target = _Target("same-id")

    report = teardown.reap(
        ref,
        platform=_Adapter("android", android_target),
        cache_dir=tmp_path,
        grace_s=0,
    )

    assert "does not match" in report["skipped"]
    assert android_target.calls == []
    assert device_ledger.read_ledger(ref)


def test_capture_journal_and_detached_sockets_are_platform_scoped(tmp_path: Path) -> None:
    android_capture = CaptureBuffer(
        root=tmp_path / "captures",
        serial="same-id",
        cfg=CaptureCfgView(),
        screenshot=lambda: None,
    )
    ios_capture = CaptureBuffer(
        root=tmp_path / "captures",
        serial="same-id",
        cfg=CaptureCfgView(),
        screenshot=lambda: None,
        platform="ios",
    )
    assert android_capture.serial_root != ios_capture.serial_root

    journal.record(
        cache_dir=tmp_path,
        serial="same-id",
        platform="android",
        source="test",
        cmd="analyze",
    )
    journal.record(
        cache_dir=tmp_path,
        serial="same-id",
        platform="ios",
        source="test",
        cmd="analyze",
    )
    assert journal.journal_path(tmp_path, "same-id") != journal.journal_path(
        tmp_path, "same-id", platform="ios"
    )
    assert {row["platform"] for row in journal.read_since(tmp_path, "same-id")} == {
        "android"
    }
    assert {
        row["platform"]
        for row in journal.read_since(tmp_path, "same-id", platform="ios")
    } == {"ios"}

    config = SimpleNamespace(
        daemon=SimpleNamespace(socket=str(tmp_path / "daemon.sock")),
        device=SimpleNamespace(platform="android", serial=None),
    )
    android_socket = daemon.socket_path(config, "same-id")
    ios_socket = daemon.socket_path(config, "same-id", platform="ios")
    assert android_socket != ios_socket
    assert capture_sidecar.socket_path(tmp_path) != capture_sidecar.socket_path(
        tmp_path, serial="same-id", platform="ios"
    )


def test_engine_cache_and_auto_named_images_are_platform_scoped(tmp_path: Path) -> None:
    config = Config.model_validate({"cache": {"dir": str(tmp_path)}})

    def engine_for(platform: str) -> Engine:
        engine = Engine.__new__(Engine)
        engine.config = config
        engine._platform = SimpleNamespace(name=platform)  # type: ignore[assignment]
        engine._device = None
        return engine

    android = engine_for("android")
    ios = engine_for("ios")

    assert android._cache_path("same:id") == tmp_path / "analyze_same_id.json"
    assert android._cache_path("same:id") != ios._cache_path("same:id")
    assert android._default_annotate_path("same:id") != ios._default_annotate_path("same:id")


def test_active_session_pointers_are_platform_scoped(tmp_path: Path) -> None:
    common = {
        "cache_dir": tmp_path,
        "goal": "Inspect the fictional app",
        "serial": "same-id",
        "owner": "same-worker",
        "recommended_kind": "analyze",
        "recommended_cli": "aua analyze",
        "network_backup_preexisting": False,
        "network_profile_preexisting": False,
    }
    android = create_session_state(**common, platform="android")
    ios = create_session_state(
        **common,
        platform="ios",
        virtual_target_started=True,
        virtual_target_definition_id="iphone-15",
        virtual_target_instance_token="sim-boot-1",
    )

    android_meta = active_session_metadata(
        tmp_path, "same-id", "same-worker", platform="android"
    )
    ios_meta = active_session_metadata(
        tmp_path, "same-id", "same-worker", platform="ios"
    )

    assert android_meta["session_id"] == android.session_id
    assert ios_meta["session_id"] == ios.session_id
    assert android_meta != ios_meta
    assert ios.virtual_target_started is True
    assert ios.virtual_target_definition_id == "iphone-15"
    assert ios.virtual_target_instance_token == "sim-boot-1"
