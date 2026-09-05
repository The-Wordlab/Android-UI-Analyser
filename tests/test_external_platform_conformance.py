"""A separately imported plugin can use AUA without inheriting Android's runtime surface."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path

import pytest

from android_ui_analyser import leases
from android_ui_analyser.config import Config
from android_ui_analyser.errors import IncompatiblePlatformPluginError
from android_ui_analyser.platforms import registry
from android_ui_analyser.platforms.conformance import (
    AttachedTargetCase,
    run_attached_target_conformance,
)
from android_ui_analyser.platforms.identity import TargetRef

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PLUGIN_MODULE = "external_platform_plugin"
PLUGIN_NAME = "strict-external"


def _build_fixture_wheel(destination: Path) -> Path:
    """Build a tiny valid wheel so metadata discovery is tested outside pytest monkeypatches."""

    distribution = "aua_strict_external_fixture"
    dist_info = f"{distribution}-1.0.dist-info"
    files = {
        "external_platform_plugin.py": (FIXTURE_DIR / "external_platform_plugin.py").read_bytes(),
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\n"
            b"Name: aua-strict-external-fixture\n"
            b"Version: 1.0\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: aua-conformance\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            b"[aua.platforms]\n"
            b"strict-external = external_platform_plugin:StrictExternalPlatform\n"
        ),
        f"{dist_info}/top_level.txt": b"external_platform_plugin\n",
    }
    record_path = f"{dist_info}/RECORD"
    records = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append(f"{name},sha256={digest},{len(data)}")
    files[record_path] = ("\n".join([*records, f"{record_path},,"]) + "\n").encode()
    wheel = destination / f"{distribution}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


def _entry(name: str, attribute: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(
        name=name,
        value=f"{PLUGIN_MODULE}:{attribute}",
        group=registry.ENTRY_POINT_GROUP,
    )


@pytest.fixture
def external_plugins(monkeypatch: pytest.MonkeyPatch):
    """Expose real importable entry points without installing a test distribution."""

    monkeypatch.syspath_prepend(str(FIXTURE_DIR))
    sys.modules.pop(PLUGIN_MODULE, None)
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    monkeypatch.setattr(
        registry,
        "_ENTRY_POINTS",
        {
            PLUGIN_NAME: [_entry(PLUGIN_NAME, "StrictExternalPlatform")],
            "strict-external-future": [
                _entry("strict-external-future", "FutureStrictExternalPlatform")
            ],
            "unselected-broken": [
                metadata.EntryPoint(
                    name="unselected-broken",
                    value="module_that_must_not_be_imported:Platform",
                    group=registry.ENTRY_POINT_GROUP,
                )
            ],
        },
    )
    yield
    sys.modules.pop(PLUGIN_MODULE, None)


def _config(tmp_path: Path, *, platform: str = PLUGIN_NAME) -> Config:
    return Config.model_validate(
        {
            "device": {"platform": platform},
            "platforms": {
                platform: {"endpoint": "memory://fixture/"},
                "unselected-broken": {"explode": True},
            },
            "cache": {"dir": str(tmp_path / "cache")},
            "memory": {"enabled": False},
            "lease": {"enabled": False},
            "teardown": {"enabled": False},
            "ocr": {"enabled": False},
            "perf": {
                "prefetch": False,
                "predictive_prefetch": False,
                "auto_daemon": False,
            },
        }
    )


def test_real_entry_point_passes_the_external_attached_target_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_plugins: None,
) -> None:
    # An external selection must not import even AUA's built-in Android strategy. Native host
    # commands are forbidden as a second line of proof while the Engine path runs.
    monkeypatch.setattr(
        registry,
        "_load_builtin",
        lambda name: pytest.fail(f"external selection tried to load built-in {name!r}"),
    )
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(f"native process invoked: {args!r} {kwargs!r}"),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(f"native process invoked: {args!r} {kwargs!r}"),
    )

    adapter = registry.PlatformFactory(_config(tmp_path)).create()
    report = run_attached_target_conformance(
        adapter,
        AttachedTargetCase(
            target_id="shared-target",
            element_text="Continue",
            expected_bounds=(300, 20, 360, 60),
            expected_app_id="org.example.conformance",
            require_non_identity_geometry=True,
            input_element_text="Continue",
            key_name="fixture-back",
            expected_scrollable_bounds=(20, 10, 380, 190),
        ),
    )

    assert type(adapter).__module__ == PLUGIN_MODULE
    assert dict(adapter.options) == {"endpoint": "memory://fixture"}
    assert report.platform == PLUGIN_NAME
    assert report.geometry == (0.0, 2.0, -2.0, 0.0, 400.0, 0.0)
    assert report.screenshot_size == (400, 200)
    assert report.tap_point == (330, 40)
    runtime = getattr(type(adapter), "last_runtime", None)
    assert runtime is not None and runtime.closed
    assert "target_id" in type(runtime).__dict__
    assert "serial" not in type(runtime).__dict__
    assert runtime.serial == runtime.target_id == "shared-target"
    assert ("click", (20.0, 35.0)) in runtime.events
    assert ("send_text", ("aua conformance", True)) in runtime.events
    assert ("press", "fixture-back") in runtime.events
    assert any(name == "swipe" for name, _detail in runtime.events)
    assert "engine-verified-swipe" in report.checks
    assert sum(name == "screenshot" for name, _detail in runtime.events) >= 2, (
        "the profile capture and Engine pre-action capture must both use the plugin"
    )
    assert not any(
        hasattr(runtime, name)
        for name in ("adb", "adb_reverse", "dumpsys", "logcat", "run_as", "shell")
    )


def test_external_fixture_imports_plugin_contract_types_only_from_the_stable_facade() -> None:
    tree = ast.parse((FIXTURE_DIR / "external_platform_plugin.py").read_text(encoding="utf-8"))
    aua_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("android_ui_analyser")
    }

    assert aua_imports == {"android_ui_analyser.platforms"}


def test_installed_wheel_passes_in_a_fresh_process_without_android_imports(
    tmp_path: Path,
) -> None:
    wheel = _build_fixture_wheel(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    # A pure-Python wheel installs by unpacking its records into site-packages. Do that directly
    # so this contract test does not depend on pip being present in uv-managed environments.
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import importlib.abc
import json
import subprocess
import sys

sys.path[:0] = [{str(site)!r}, {str(source_root)!r}]
blocked = (
    "android_ui_analyser.device",
    "android_ui_analyser.platforms.android",
    "adbutils",
    "uiautomator2",
)

class BlockAndroid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == blocked or any(fullname == item or fullname.startswith(item + ".") for item in blocked):
            raise ImportError("blocked Android dependency: " + fullname)
        return None

sys.meta_path.insert(0, BlockAndroid())
from android_ui_analyser.config import Config
from android_ui_analyser.platforms import AttachedTargetCase, PlatformFactory, run_attached_target_conformance

def forbidden(*args, **kwargs):
    raise AssertionError("native subprocess was invoked")

subprocess.run = forbidden
subprocess.Popen = forbidden
config = Config.model_validate({{
    "device": {{"platform": "strict-external"}},
    "platforms": {{"strict-external": {{"endpoint": "memory://installed/"}}}},
    "cache": {{"dir": {str(tmp_path / 'child-cache')!r}}},
    "memory": {{"enabled": False}},
    "lease": {{"enabled": False}},
    "teardown": {{"enabled": False}},
    "ocr": {{"enabled": False}},
    "perf": {{"prefetch": False, "predictive_prefetch": False, "auto_daemon": False}},
}})
adapter = PlatformFactory(config).create()
report = run_attached_target_conformance(adapter, AttachedTargetCase(
    target_id="shared-target",
    element_text="Continue",
    expected_bounds=(300, 20, 360, 60),
    expected_app_id="org.example.conformance",
    require_non_identity_geometry=True,
    input_element_text="Continue",
    key_name="fixture-back",
    expected_scrollable_bounds=(20, 10, 380, 190),
))
print(json.dumps({{"platform": report.platform, "checks": report.checks}}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(site), str(source_root)))
    fresh = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert fresh.returncode == 0, fresh.stderr
    payload = json.loads(fresh.stdout)
    assert payload["platform"] == PLUGIN_NAME
    assert "engine-verified-swipe" in payload["checks"]


def test_platform_discovery_and_selection_are_lazy_for_real_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_plugins: None,
) -> None:
    loaded_builtins: list[str] = []
    monkeypatch.setattr(registry, "_load_builtin", loaded_builtins.append)

    assert {PLUGIN_NAME, "strict-external-future", "unselected-broken"} <= set(
        registry.available_platforms()
    )
    assert PLUGIN_MODULE not in sys.modules
    assert "module_that_must_not_be_imported" not in sys.modules

    adapter = registry.PlatformFactory(_config(tmp_path)).create()

    assert adapter.name == PLUGIN_NAME
    assert PLUGIN_MODULE in sys.modules
    assert "module_that_must_not_be_imported" not in sys.modules
    assert loaded_builtins == []


def test_real_entry_point_with_an_incompatible_api_is_rejected_before_init(
    tmp_path: Path,
    external_plugins: None,
) -> None:
    config = _config(tmp_path, platform="strict-external-future")

    with pytest.raises(IncompatiblePlatformPluginError) as exc:
        registry.PlatformFactory(config).create()

    assert exc.value.code == "platform_api_incompatible"
    assert "expected 1" in exc.value.message


def test_two_platforms_with_the_same_target_id_have_independent_state(
    tmp_path: Path,
) -> None:
    android = TargetRef("android", "shared-target")
    external = TargetRef(PLUGIN_NAME, "shared-target")

    assert leases.acquire(tmp_path, android, owner="conformance-worker")
    assert leases.acquire(tmp_path, external, owner="conformance-worker")
    assert leases.holder(tmp_path, android) == "conformance-worker"
    assert leases.holder(tmp_path, external) == "conformance-worker"
    assert leases._lease_path(tmp_path, android) != leases._lease_path(tmp_path, external)
