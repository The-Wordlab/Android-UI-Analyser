from __future__ import annotations

from android_ui_analyser.config import Config
from android_ui_analyser.flows import Flow, FlowStore
from android_ui_analyser.memory import AppMap, AppMemoryStore, RouteStep, SessionState


def _config(tmp_path, *, backend: str = "json") -> Config:
    return Config.model_validate(
        {
            "memory": {
                "dir": str(tmp_path / "memory-home"),
                "backend": backend,
                "sqlite_path": str(tmp_path / "memory-home" / "memory.db"),
            }
        }
    )


def test_identical_app_and_target_ids_do_not_share_file_memory(tmp_path) -> None:
    config = _config(tmp_path)
    android = AppMemoryStore(config.memory, platform="android")
    other = AppMemoryStore(config.memory, platform="example-os")
    app_id = "example.app"
    target_id = "shared-target"

    android.save(AppMap(package=app_id, label="Android map"))
    other.save(AppMap(package=app_id, label="Other map"))
    android.save_session(target_id, SessionState(package="android.app"))
    other.save_session(target_id, SessionState(package="other.app"))

    assert android.load(app_id).label == "Android map"  # type: ignore[union-attr]
    assert other.load(app_id).label == "Other map"  # type: ignore[union-attr]
    assert android.load_session(target_id).package == "android.app"
    assert other.load_session(target_id).package == "other.app"
    assert android.index_path(app_id) != other.index_path(app_id)
    assert android.session_path(target_id) != other.session_path(target_id)


def test_injected_adapter_namespace_can_override_config_selection(tmp_path) -> None:
    config = _config(tmp_path)

    store = AppMemoryStore.from_config(config, platform="external-test")

    assert store.platform == "external-test"
    assert "platforms/external-test" in str(store.memory_root())


def test_identical_flow_names_do_not_cross_platform_libraries(tmp_path) -> None:
    config = _config(tmp_path)
    android = FlowStore(config.memory, platform="android")
    other = FlowStore(config.memory, platform="example-os")

    android_path = android.save(
        Flow(name="sign_in", app="example.app", steps=[RouteStep(kind="key", arg="back")])
    )
    other_path = other.save(
        Flow(name="sign_in", app="example.app", steps=[RouteStep(kind="key", arg="home")])
    )

    assert android_path != other_path
    assert android.load("example.app:sign_in").steps[0].arg == "back"
    assert other.load("example.app:sign_in").steps[0].arg == "home"


def test_android_memory_paths_remain_backward_compatible(tmp_path) -> None:
    config = _config(tmp_path)
    store = AppMemoryStore(config.memory)

    assert store.index_path("example.app") == (
        tmp_path / "memory-home" / "memory" / "example.app" / "index.json"
    )
    assert store.session_path("emulator:5554") == (
        tmp_path / "memory-home" / "state" / "session_emulator_5554.json"
    )


def test_identical_app_and_target_ids_do_not_share_sqlite_memory(tmp_path) -> None:
    config = _config(tmp_path, backend="sqlite")
    android = AppMemoryStore(config.memory, platform="android")
    other = AppMemoryStore(config.memory, platform="example-os")

    android.save(AppMap(package="example.app", label="Android map"))
    other.save(AppMap(package="example.app", label="Other map"))
    android.save_session("shared-target", SessionState(package="android.app"))
    other.save_session("shared-target", SessionState(package="other.app"))

    assert android.load("example.app").label == "Android map"  # type: ignore[union-attr]
    assert other.load("example.app").label == "Other map"  # type: ignore[union-attr]
    assert android.load_session("shared-target").package == "android.app"
    assert other.load_session("shared-target").package == "other.app"
